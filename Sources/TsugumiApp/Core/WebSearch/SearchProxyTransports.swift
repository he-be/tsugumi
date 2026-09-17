import Foundation

/// The hosts of the metered search APIs (Serper, Brave). Pages are not metered.
public enum MeteredSearchHosts {
    public static let all: Set<String> = ["google.serper.dev", "api.search.brave.com"]

    public static func contains(_ request: URLRequest) -> Bool {
        request.url?.host.map(all.contains) ?? false
    }
}

/// Caps the searches that reach a metered API (`docs/qwen38/32` §5).
///
/// Sits in front of a `RecordedHTTPTransport`: a search the directory already
/// has is replayed and not counted; one it does not have is counted, and past
/// the budget it fails here, before the store, so the refusal is not recorded
/// as the query's answer.
public final class SearchBudgetTransport: HTTPTransport, @unchecked Sendable {
    public let budget: Int
    let store: RecordedHTTPTransport
    private let lock = NSLock()
    private var spent = 0

    public init(budget: Int, store: RecordedHTTPTransport) {
        self.budget = budget
        self.store = store
    }

    /// Searches that went to the API so far.
    public var used: Int {
        lock.lock(); defer { lock.unlock() }
        return spent
    }

    public func perform(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        if MeteredSearchHosts.contains(request), !store.hasRecording(for: request) {
            guard spend() else {
                throw WebToolError.transport("search budget of \(budget) spent (\(request.url?.host ?? ""))")
            }
        }
        return try await store.perform(request)
    }

    private func spend() -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard spent < budget else { return false }
        spent += 1
        return true
    }
}

/// Answers every search of a conversation with the first one it got (`docs/qwen38/32` §5).
///
/// The model words its query a little differently on each run, so a store
/// keyed by the request would go live again for each wording. Here the first
/// successful search a pin gets is kept in `DIR/pinned/<pin>.json` and answers
/// every later search under that pin, whatever the query. `order` (1-based)
/// re-ranks the results of that answer, so one recorded search serves several
/// orderings; results it does not name keep their order after the named ones.
public final class PinnedSearchTransport: HTTPTransport, @unchecked Sendable {
    public struct Counts: Equatable, Sendable {
        public var pinned = 0
        public var fetched = 0
    }

    struct Pinned: Codable {
        var url: String
        var status: Int
        var headers: [String: String]
        var body: String
    }

    let directory: URL
    let inner: any HTTPTransport
    private let lock = NSLock()
    private var pin: String?
    private var order: [Int]?
    private var counts = Counts()

    public init(directory: URL, inner: any HTTPTransport) throws {
        self.directory = directory.appendingPathComponent("pinned", isDirectory: true)
        self.inner = inner
        try FileManager.default.createDirectory(at: self.directory, withIntermediateDirectories: true)
    }

    /// The pin (and ordering) the next searches belong to; nil passes searches through.
    public func start(pin: String?, order: [Int]?) {
        lock.lock(); defer { lock.unlock() }
        self.pin = pin
        self.order = order
    }

    public func takeCounts() -> Counts {
        lock.lock(); defer { lock.unlock() }
        defer { counts = Counts() }
        return counts
    }

    private func current() -> (pin: String?, order: [Int]?) {
        lock.lock(); defer { lock.unlock() }
        return (pin, order)
    }

    public func perform(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (pin, order) = current()
        guard let pin, MeteredSearchHosts.contains(request) else { return try await inner.perform(request) }
        let file = directory.appendingPathComponent(Self.fileName(pin))
        if let data = try? Data(contentsOf: file), let pinned = try? JSONDecoder().decode(Pinned.self, from: data),
           let url = URL(string: pinned.url),
           let response = HTTPURLResponse(url: url, statusCode: pinned.status, httpVersion: "HTTP/1.1",
                                          headerFields: pinned.headers) {
            count { $0.pinned += 1 }
            return (Self.reordered(Data(base64Encoded: pinned.body) ?? Data(), order: order), response)
        }
        let (body, response) = try await inner.perform(request)
        count { $0.fetched += 1 }
        if (200..<300).contains(response.statusCode) {
            let headers = response.allHeaderFields.reduce(into: [String: String]()) { headers, field in
                if let name = field.key as? String, let value = field.value as? String { headers[name] = value }
            }
            let pinned = Pinned(url: response.url?.absoluteString ?? request.url?.absoluteString ?? "",
                                status: response.statusCode, headers: headers, body: body.base64EncodedString())
            try JSONEncoder().encode(pinned).write(to: file, options: .atomic)
        }
        return (Self.reordered(body, order: order), response)
    }

    private func count(_ change: (inout Counts) -> Void) {
        lock.lock(); defer { lock.unlock() }
        change(&counts)
    }

    static func fileName(_ pin: String) -> String {
        pin.map { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" ? String($0) : "_" }.joined() + ".json"
    }

    /// Serper's `organic` or Brave's `web.results`, re-ranked by `order`.
    static func reordered(_ body: Data, order: [Int]?) -> Data {
        guard let order, !order.isEmpty,
              var object = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else { return body }
        func rank(_ items: [Any]) -> [Any] {
            let named = order.compactMap { $0 >= 1 && $0 <= items.count ? $0 - 1 : nil }
            let rest = items.indices.filter { !named.contains($0) }
            return (named + rest).map { items[$0] }
        }
        if let organic = object["organic"] as? [Any] {
            object["organic"] = rank(organic)
        } else if var web = object["web"] as? [String: Any], let results = web["results"] as? [Any] {
            web["results"] = rank(results)
            object["web"] = web
        } else {
            return body
        }
        return (try? JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes])) ?? body
    }
}
