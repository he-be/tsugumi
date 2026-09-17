import CryptoKit
import Foundation

/// An `HTTPTransport` that answers from a directory of recorded responses and
/// records what it does not have (`docs/qwen38/21` §4 E-1).
///
/// The tool-loop check fetches Serper and live pages, so a conversation read
/// different pages on each run. Recording at the HTTP layer rather than per
/// tool call keeps the pages the same when the tools change how they are
/// called (a `fetch_page` that reads on from an offset, a smaller page
/// limit): the same URL gets the same body, and the executor clips it.
///
/// The key is the method, the URL and the body, with a JSON body re-written
/// with sorted keys (`JSONSerialization` does not keep a dictionary's order
/// from one process to the next). Request headers are not part of the key
/// and are not stored: they carry the API keys. Transport errors are
/// recorded too, as the text the executor would have shown.
public final class RecordedHTTPTransport: HTTPTransport, @unchecked Sendable {
    public struct Counts: Equatable, Sendable {
        public var replayed = 0
        public var recorded = 0
    }

    struct Entry: Codable {
        var key: String
        var url: String?
        var status: Int?
        var headers: [String: String]?
        var body: String?
        var error: String?
    }

    struct ReplayedError: Error, CustomStringConvertible {
        let description: String
    }

    let directory: URL
    let live: any HTTPTransport
    private let lock = NSLock()
    private var counts = Counts()

    public init(directory: URL, live: any HTTPTransport = URLSessionTransport()) throws {
        self.directory = directory
        self.live = live
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    /// Replayed and recorded responses so far; `take` resets them.
    public func takeCounts() -> Counts {
        lock.lock(); defer { lock.unlock() }
        defer { counts = Counts() }
        return counts
    }

    public func perform(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let key = Self.key(for: request)
        let file = directory.appendingPathComponent(Self.fileName(for: key))
        if let data = try? Data(contentsOf: file),
           let entry = try? JSONDecoder().decode(Entry.self, from: data), entry.key == key {
            count { $0.replayed += 1 }
            return try Self.response(from: entry, request: request)
        }
        var entry = Entry(key: key)
        let outcome: Result<(Data, HTTPURLResponse), Error>
        do {
            let (body, response) = try await live.perform(request)
            entry.url = response.url?.absoluteString
            entry.status = response.statusCode
            entry.headers = response.allHeaderFields.reduce(into: [:]) { headers, field in
                if let name = field.key as? String, let value = field.value as? String { headers[name] = value }
            }
            entry.body = body.base64EncodedString()
            outcome = .success((body, response))
        } catch {
            entry.error = "\(error)"
            outcome = .failure(error)
        }
        let encoded = try JSONEncoder().encode(entry)
        try encoded.write(to: file, options: .atomic)
        count { $0.recorded += 1 }
        return try outcome.get()
    }

    /// Whether `request` would be answered from the directory, without going live.
    public func hasRecording(for request: URLRequest) -> Bool {
        let key = Self.key(for: request)
        let file = directory.appendingPathComponent(Self.fileName(for: key))
        guard let data = try? Data(contentsOf: file),
              let entry = try? JSONDecoder().decode(Entry.self, from: data) else { return false }
        return entry.key == key
    }

    private func count(_ change: (inout Counts) -> Void) {
        lock.lock(); defer { lock.unlock() }
        change(&counts)
    }

    static func key(for request: URLRequest) -> String {
        let method = request.httpMethod ?? "GET"
        let url = request.url?.absoluteString ?? ""
        guard let body = request.httpBody, !body.isEmpty else { return "\(method) \(url)" }
        let canonical: Data
        if let object = try? JSONSerialization.jsonObject(with: body),
           let sorted = try? JSONSerialization.data(withJSONObject: object,
                                                    options: [.sortedKeys, .withoutEscapingSlashes]) {
            canonical = sorted
        } else {
            canonical = body
        }
        return "\(method) \(url)\n" + (String(data: canonical, encoding: .utf8) ?? canonical.base64EncodedString())
    }

    static func fileName(for key: String) -> String {
        SHA256.hash(data: Data(key.utf8)).map { String(format: "%02x", $0) }.joined() + ".json"
    }

    static func response(from entry: Entry, request: URLRequest) throws -> (Data, HTTPURLResponse) {
        if let error = entry.error { throw ReplayedError(description: error) }
        guard let status = entry.status,
              let url = entry.url.flatMap(URL.init(string:)) ?? request.url,
              let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1",
                                             headerFields: entry.headers) else {
            throw WebToolError.badResponse("recorded response for \(entry.key) is incomplete")
        }
        return (entry.body.flatMap { Data(base64Encoded: $0) } ?? Data(), response)
    }
}
