import Foundation

/// One result of a web search, as the model reads it.
public struct WebSearchHit: Equatable, Sendable {
    public var title: String
    public var url: String
    public var snippet: String
    public var date: String?

    public init(title: String, url: String, snippet: String, date: String? = nil) {
        self.title = title
        self.url = url
        self.snippet = snippet
        self.date = date
    }
}

public struct WebSearchResponse: Equatable, Sendable {
    public var provider: String
    public var hits: [WebSearchHit]
    /// A direct answer or knowledge panel the engine attached, when it did.
    public var highlights: [String]

    public init(provider: String, hits: [WebSearchHit], highlights: [String] = []) {
        self.provider = provider
        self.hits = hits
        self.highlights = highlights
    }
}

public protocol WebSearchProvider: Sendable {
    var name: String { get }
    func search(_ query: String, count: Int) async throws -> WebSearchResponse
}

/// Serper: Google results over `POST https://google.serper.dev/search`.
public struct SerperSearchProvider: WebSearchProvider {
    public let name = "Serper"
    let apiKey: String
    let country: String
    let language: String
    let transport: any HTTPTransport

    public init(apiKey: String, country: String, language: String, transport: any HTTPTransport) {
        self.apiKey = apiKey
        self.country = country
        self.language = language
        self.transport = transport
    }

    public func search(_ query: String, count: Int) async throws -> WebSearchResponse {
        guard !apiKey.isEmpty else { throw WebToolError.missingKey(name) }
        var request = URLRequest(url: URL(string: "https://google.serper.dev/search")!)
        request.httpMethod = "POST"
        request.setValue(apiKey, forHTTPHeaderField: "X-API-KEY")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = ["q": query, "gl": country, "hl": language, "num": count]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await transport.perform(request)
        guard (200..<300).contains(response.statusCode) else {
            throw WebToolError.status(response.statusCode, name)
        }
        return try Self.parse(data, count: count)
    }

    static func parse(_ data: Data, count: Int) throws -> WebSearchResponse {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw WebToolError.badResponse("Serper body is not a JSON object")
        }
        var highlights: [String] = []
        if let box = object["answerBox"] as? [String: Any] {
            let answer = (box["answer"] as? String) ?? (box["snippet"] as? String) ?? ""
            let title = box["title"] as? String ?? ""
            let joined = [title, answer].filter { !$0.isEmpty }.joined(separator: ": ")
            if !joined.isEmpty { highlights.append(joined) }
        }
        if let graph = object["knowledgeGraph"] as? [String: Any] {
            let title = graph["title"] as? String ?? ""
            let type = graph["type"] as? String ?? ""
            let description = graph["description"] as? String ?? ""
            let joined = [title, type, description].filter { !$0.isEmpty }.joined(separator: " — ")
            if !joined.isEmpty { highlights.append(joined) }
        }
        let organic = object["organic"] as? [[String: Any]] ?? []
        let hits = organic.prefix(count).compactMap { item -> WebSearchHit? in
            guard let link = item["link"] as? String, !link.isEmpty else { return nil }
            return WebSearchHit(title: item["title"] as? String ?? "",
                                url: link,
                                snippet: item["snippet"] as? String ?? "",
                                date: item["date"] as? String)
        }
        return WebSearchResponse(provider: "Serper", hits: hits, highlights: highlights)
    }
}

/// Brave Search: `GET https://api.search.brave.com/res/v1/web/search`.
public struct BraveSearchProvider: WebSearchProvider {
    public let name = "Brave"
    let apiKey: String
    let country: String
    let language: String
    let transport: any HTTPTransport

    public init(apiKey: String, country: String, language: String, transport: any HTTPTransport) {
        self.apiKey = apiKey
        self.country = country
        self.language = language
        self.transport = transport
    }

    public func search(_ query: String, count: Int) async throws -> WebSearchResponse {
        guard !apiKey.isEmpty else { throw WebToolError.missingKey(name) }
        var components = URLComponents(string: "https://api.search.brave.com/res/v1/web/search")!
        components.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "country", value: country.uppercased()),
            URLQueryItem(name: "search_lang", value: language),
            URLQueryItem(name: "count", value: String(count)),
        ]
        var request = URLRequest(url: components.url!)
        request.setValue(apiKey, forHTTPHeaderField: "X-Subscription-Token")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, response) = try await transport.perform(request)
        guard (200..<300).contains(response.statusCode) else {
            throw WebToolError.status(response.statusCode, name)
        }
        return try Self.parse(data, count: count)
    }

    static func parse(_ data: Data, count: Int) throws -> WebSearchResponse {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw WebToolError.badResponse("Brave body is not a JSON object")
        }
        let results = (object["web"] as? [String: Any])?["results"] as? [[String: Any]] ?? []
        let hits = results.prefix(count).compactMap { item -> WebSearchHit? in
            guard let url = item["url"] as? String, !url.isEmpty else { return nil }
            return WebSearchHit(title: item["title"] as? String ?? "",
                                url: url,
                                snippet: item["description"] as? String ?? "",
                                date: (item["age"] as? String) ?? (item["page_age"] as? String))
        }
        return WebSearchResponse(provider: "Brave", hits: hits)
    }
}
