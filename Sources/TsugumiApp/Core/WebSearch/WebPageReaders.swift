import Foundation

/// What a page read hands the model.
public struct WebPageText: Equatable, Sendable {
    public var reader: String
    public var title: String
    public var text: String

    public init(reader: String, title: String, text: String) {
        self.reader = reader
        self.title = title
        self.text = text
    }
}

public protocol WebPageReader: Sendable {
    var name: String { get }
    func read(_ url: URL) async throws -> WebPageText
}

/// Jina Reader (`https://r.jina.ai/<url>`): a hosted readability pass that
/// also renders JavaScript pages. Works without a key at a lower rate.
public struct JinaPageReader: WebPageReader {
    public let name = "Jina Reader"
    let apiKey: String
    let transport: any HTTPTransport

    public init(apiKey: String, transport: any HTTPTransport) {
        self.apiKey = apiKey
        self.transport = transport
    }

    public func read(_ url: URL) async throws -> WebPageText {
        guard let endpoint = URL(string: "https://r.jina.ai/" + url.absoluteString) else {
            throw WebToolError.badResponse("cannot form the reader URL")
        }
        var request = URLRequest(url: endpoint)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("text", forHTTPHeaderField: "X-Return-Format")
        request.setValue("15", forHTTPHeaderField: "X-Timeout")
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await transport.perform(request)
        guard (200..<300).contains(response.statusCode) else {
            throw WebToolError.status(response.statusCode, name)
        }
        return try Self.parse(data)
    }

    static func parse(_ data: Data) throws -> WebPageText {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let payload = object["data"] as? [String: Any],
              let content = payload["content"] as? String else {
            throw WebToolError.badResponse("Jina Reader body has no data.content")
        }
        return WebPageText(reader: "Jina Reader",
                           title: payload["title"] as? String ?? "",
                           text: HTMLTextExtractor.normalize(content))
    }
}

/// The app's own fetch: one GET with a browser-like agent, then
/// `HTMLTextExtractor`. Free and private, but blind to pages that render
/// their text with JavaScript.
public struct DirectPageReader: WebPageReader {
    public let name = "direct fetch"
    let transport: any HTTPTransport
    /// Bodies past this are cut before extraction; a search result is not
    /// a download.
    let maximumBytes: Int

    public init(transport: any HTTPTransport, maximumBytes: Int = 3_000_000) {
        self.transport = transport
        self.maximumBytes = maximumBytes
    }

    public func read(_ url: URL) async throws -> WebPageText {
        var request = URLRequest(url: url)
        request.setValue(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 Tsugumi/1.0",
            forHTTPHeaderField: "User-Agent")
        request.setValue("text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
                         forHTTPHeaderField: "Accept")
        let (data, response) = try await transport.perform(request)
        guard (200..<300).contains(response.statusCode) else {
            throw WebToolError.status(response.statusCode, url.host ?? "the site")
        }
        let contentType = response.value(forHTTPHeaderField: "Content-Type")?.lowercased() ?? ""
        let body = data.prefix(maximumBytes)
        if contentType.contains("text/plain") || contentType.contains("markdown") {
            let text = HTMLTextExtractor.decode(body, contentType: contentType)
            return WebPageText(reader: name, title: "", text: HTMLTextExtractor.normalize(text))
        }
        guard contentType.isEmpty || contentType.contains("html") || contentType.contains("xml") else {
            throw WebToolError.unsupportedContent(contentType)
        }
        let html = HTMLTextExtractor.decode(body, contentType: contentType)
        let extract = HTMLTextExtractor.extract(html: html)
        return WebPageText(reader: name, title: extract.title, text: extract.text)
    }
}
