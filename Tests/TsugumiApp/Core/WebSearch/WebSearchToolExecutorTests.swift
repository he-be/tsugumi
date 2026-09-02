import Foundation
import Synchronization
import Testing
@testable import TsugumiAppCore

/// Answers each request from a table keyed by host, recording what was asked.
final class StubTransport: HTTPTransport, @unchecked Sendable {
    struct Reply {
        var status: Int
        var body: Data
        var contentType: String? = "application/json"
    }

    private let replies: Mutex<[String: [Reply]]>
    private let seen = Mutex<[URLRequest]>([])

    init(_ replies: [String: [Reply]]) {
        self.replies = Mutex(replies)
    }

    var requests: [URLRequest] { seen.withLock { $0 } }

    func perform(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        seen.withLock { $0.append(request) }
        let host = request.url?.host ?? ""
        let reply: Reply? = replies.withLock { table in
            guard var queue = table[host], !queue.isEmpty else { return nil }
            let first = queue.removeFirst()
            table[host] = queue.isEmpty ? [first] : queue
            return first
        }
        guard let reply else { throw WebToolError.transport("no stub for \(host)") }
        var headers: [String: String] = [:]
        if let contentType = reply.contentType { headers["Content-Type"] = contentType }
        let response = HTTPURLResponse(url: request.url!, statusCode: reply.status,
                                       httpVersion: nil, headerFields: headers)!
        return (reply.body, response)
    }
}

@Suite struct WebSearchToolExecutorTests {
    static let serperBody = Data("""
    {"answerBox":{"title":"東京の人口","answer":"約1,400万人"},
     "organic":[
      {"title":"東京都の人口","link":"https://example.jp/tokyo","snippet":"東京都の推計人口は…","date":"2026-08-01","position":1},
      {"title":"統計局","link":"https://stat.example.jp/","snippet":"人口推計"}
     ]}
    """.utf8)

    static let braveBody = Data("""
    {"web":{"results":[
      {"title":"Brave 一件目","url":"https://brave.example.jp/a","description":"説明 A","age":"2 days ago"},
      {"title":"Brave 二件目","url":"https://brave.example.jp/b","description":"説明 B"}
    ]}}
    """.utf8)

    func executor(transport: StubTransport,
                  serper: String = "serper-key",
                  brave: String = "brave-key",
                  preferJina: Bool = true) -> WebSearchToolExecutor {
        var configuration = WebSearchConfiguration()
        configuration.serperAPIKey = serper
        configuration.braveAPIKey = brave
        configuration.preferJinaReader = preferJina
        configuration.pageCharacterLimit = 500
        return WebSearchToolExecutor(configuration: configuration, transport: transport,
                                     today: Self.fixedDay)
    }

    /// 2026-09-02 in Japan (the day the fixtures were written).
    static var fixedDay: Date {
        var components = DateComponents()
        components.year = 2026; components.month = 9; components.day = 2; components.hour = 12
        components.timeZone = TimeZone(identifier: "Asia/Tokyo")
        return Calendar(identifier: .gregorian).date(from: components)!
    }

    @Test func urlsInThePromptAreFetchedBeforeTheFirstRound() async throws {
        // Long enough not to count as a thin page (which would try the direct reader too).
        let body = "事前登録フォーム。FAQ が 11 個。" + String(repeating: "画面写真はイメージです。", count: 20)
        let jina = Data(#"{"data":{"title":"ホリエモンAIローカルPC","content":"\#(body)"}}"#.utf8)
        let transport = StubTransport([
            "r.jina.ai": [.init(status: 200, body: jina), .init(status: 200, body: jina)],
        ])
        let executor = executor(transport: transport)
        let lookups = await executor.lookups(
            prompt: "https://localpc.horiemon.ai/ 何これ？ https://example.jp/a）も見て。 https://localpc.horiemon.ai/",
            callIDPrefix: "lookup-")
        // Two distinct URLs, once each, the closing bracket not part of the second.
        #expect(lookups.map(\.call) == [
            AppToolCall(id: "lookup-1", name: "fetch_page", argumentsJSON: #"{"url":"https://localpc.horiemon.ai/"}"#),
            AppToolCall(id: "lookup-2", name: "fetch_page", argumentsJSON: #"{"url":"https://example.jp/a"}"#),
        ])
        #expect(lookups.map(\.subject) == ["https://localpc.horiemon.ai/", "https://example.jp/a"])
        #expect(lookups[0].result.content.hasPrefix("タイトル: ホリエモンAIローカルPC\nURL: https://localpc.horiemon.ai/\n取得日 2026年9月2日\n\n事前登録フォーム。"))
        #expect(!lookups[0].result.isError)
        #expect(transport.requests.map { $0.url?.absoluteString } == [
            "https://r.jina.ai/https://localpc.horiemon.ai/", "https://r.jina.ai/https://example.jp/a"])
        // No URL, nothing to seed; a private address is refused as a result, not silently.
        #expect(await executor.lookups(prompt: "東京の天気", callIDPrefix: "lookup-").isEmpty)
        let refused = await executor.lookups(prompt: "http://192.168.0.1/admin を見て", callIDPrefix: "lookup-")
        #expect(refused.count == 1 && refused[0].result.isError)
        #expect(WebSearchToolExecutor.urls(in: "見て https://a.jp/x. そして「https://b.jp/y」。") == ["https://a.jp/x", "https://b.jp/y"])
        #expect(WebSearchToolExecutor.urls(in: "ftp://a.jp/x や a.jp").isEmpty)
    }

    @Test func serperResultsRenderAsANumberedListWithHighlights() async throws {
        let transport = StubTransport([
            "google.serper.dev": [.init(status: 200, body: Self.serperBody)],
        ])
        let result = await executor(transport: transport).execute(
            AppToolCall(id: "c1", name: "web_search", argumentsJSON: #"{"query":"東京 人口"}"#))
        #expect(!result.isError)
        #expect(result.summary == "Serper · 2 hits")
        #expect(result.content.hasPrefix("検索: 東京 人口 (Serper, 2 件、取得日 2026年9月2日)"))
        #expect(result.content.contains("★ 東京の人口: 約1,400万人"))
        #expect(result.content.contains("[1] 東京都の人口\n    https://example.jp/tokyo\n    2026-08-01 東京都の推計人口は…"))
        #expect(result.content.contains("[2] 統計局"))

        let request = try #require(transport.requests.first)
        #expect(request.value(forHTTPHeaderField: "X-API-KEY") == "serper-key")
        let body = try JSONSerialization.jsonObject(with: try #require(request.httpBody)) as? [String: Any]
        #expect(body?["gl"] as? String == "jp")
        #expect(body?["hl"] as? String == "ja")
        #expect(body?["q"] as? String == "東京 人口")
    }

    @Test func braveAnswersWhenSerperFails() async throws {
        let transport = StubTransport([
            "google.serper.dev": [.init(status: 429, body: Data())],
            "api.search.brave.com": [.init(status: 200, body: Self.braveBody)],
        ])
        let result = await executor(transport: transport).execute(
            AppToolCall(id: "c1", name: "web_search", argumentsJSON: #"{"query":"q"}"#))
        #expect(!result.isError)
        #expect(result.summary == "Brave · 2 hits")
        #expect(result.content.contains("https://brave.example.jp/a"))
        #expect(result.content.contains("2 days ago 説明 A"))
        let brave = try #require(transport.requests.last)
        #expect(brave.value(forHTTPHeaderField: "X-Subscription-Token") == "brave-key")
        #expect(brave.url?.query?.contains("country=JP") == true)
        #expect(brave.url?.query?.contains("search_lang=ja") == true)
    }

    @Test func everyProviderFailingIsAnErrorTheModelCanRead() async {
        let transport = StubTransport([
            "google.serper.dev": [.init(status: 500, body: Data())],
            "api.search.brave.com": [.init(status: 401, body: Data())],
        ])
        let result = await executor(transport: transport).execute(
            AppToolCall(id: "c1", name: "web_search", argumentsJSON: #"{"query":"q"}"#))
        #expect(result.isError)
        #expect(result.content.contains("Serper answered HTTP 500"))
        #expect(result.content.contains("Brave answered HTTP 401"))
    }

    @Test func noKeyMeansNoProvider() async {
        let transport = StubTransport([:])
        let result = await executor(transport: transport, serper: "", brave: "").execute(
            AppToolCall(id: "c1", name: "web_search", argumentsJSON: #"{"query":"q"}"#))
        #expect(result.isError)
        #expect(transport.requests.isEmpty)
    }

    @Test func malformedArgumentsAreAnError() async {
        let result = await executor(transport: StubTransport([:])).execute(
            AppToolCall(id: "c1", name: "web_search", argumentsJSON: "{not json"))
        #expect(result.isError)
        #expect(result.summary == "missing query")
    }

    @Test func fetchGoesThroughJinaFirstAndClips() async throws {
        let long = String(repeating: "本文の一行。\n", count: 200)
        let jina = Data(#"{"data":{"title":"記事","content":"\#(long.replacingOccurrences(of: "\n", with: "\\n"))"}}"#.utf8)
        let transport = StubTransport([
            "r.jina.ai": [.init(status: 200, body: jina)],
        ])
        let result = await executor(transport: transport).execute(
            AppToolCall(id: "c2", name: "fetch_page", argumentsJSON: #"{"url":"https://example.jp/tokyo"}"#))
        #expect(!result.isError)
        #expect(result.content.hasPrefix("タイトル: 記事\nURL: https://example.jp/tokyo\n取得日 2026年9月2日\n\n本文の一行。"))
        #expect(result.content.hasSuffix("…(本文はここで打ち切り)"))
        #expect(result.content.count < 700)
        #expect(result.summary.hasPrefix("Jina Reader · "))
        #expect(result.summary.hasSuffix("(clipped)"))
        let request = try #require(transport.requests.first)
        #expect(request.url?.absoluteString == "https://r.jina.ai/https://example.jp/tokyo")
        #expect(request.value(forHTTPHeaderField: "Authorization") == nil)
    }

    @Test func aThinJinaPageFallsBackToTheDirectFetch() async throws {
        let html = "<html><head><title>直</title></head><body><article>"
            + String(repeating: "<p>直接取得した本文。</p>", count: 60) + "</article></body></html>"
        let transport = StubTransport([
            "r.jina.ai": [.init(status: 200, body: Data(#"{"data":{"title":"","content":"少し"}}"#.utf8))],
            "example.jp": [.init(status: 200, body: Data(html.utf8), contentType: "text/html; charset=utf-8")],
        ])
        let result = await executor(transport: transport).execute(
            AppToolCall(id: "c2", name: "fetch_page", argumentsJSON: #"{"url":"https://example.jp/p"}"#))
        #expect(!result.isError)
        #expect(result.content.contains("タイトル: 直"))
        #expect(result.content.contains("直接取得した本文。"))
        #expect(result.summary.hasPrefix("direct fetch · "))
        #expect(transport.requests.count == 2)
    }

    @Test func directFirstWhenConfiguredAndJinaIsTheFallback() async throws {
        let transport = StubTransport([
            "example.jp": [.init(status: 403, body: Data(), contentType: "text/html")],
            "r.jina.ai": [.init(status: 200, body: Data(#"{"data":{"title":"J","content":"\#(String(repeating: "本文。", count: 100))"}}"#.utf8))],
        ])
        let result = await executor(transport: transport, preferJina: false).execute(
            AppToolCall(id: "c2", name: "fetch_page", argumentsJSON: #"{"url":"https://example.jp/p"}"#))
        #expect(!result.isError)
        #expect(result.summary.hasPrefix("Jina Reader · "))
        #expect(transport.requests.first?.url?.host == "example.jp")
    }

    @Test func fetchRefusesPrivateAddresses() async {
        let transport = StubTransport([:])
        for url in ["http://127.0.0.1:8080/v1/models", "http://localhost/",
                    "http://192.168.1.5/", "file:///etc/passwd", "ftp://example.jp/"] {
            let result = await executor(transport: transport).execute(
                AppToolCall(id: "c3", name: "fetch_page", argumentsJSON: #"{"url":"\#(url)"}"#))
            #expect(result.isError, "\(url)")
        }
        #expect(transport.requests.isEmpty)
        #expect(URL(string: "https://example.jp/a")!.isPublicWebAddress)
    }

    @Test func fetchRefusesNonTextContent() async {
        let transport = StubTransport([
            "r.jina.ai": [.init(status: 500, body: Data())],
            "example.jp": [.init(status: 200, body: Data([0x25, 0x50, 0x44, 0x46]), contentType: "application/pdf")],
        ])
        let result = await executor(transport: transport).execute(
            AppToolCall(id: "c2", name: "fetch_page", argumentsJSON: #"{"url":"https://example.jp/x.pdf"}"#))
        #expect(result.isError)
        #expect(result.content.contains("application/pdf"))
    }

    @Test func definitionsDeclareBothToolsWithObjectSchemas() throws {
        let definitions = executor(transport: StubTransport([:])).definitions
        #expect(definitions.map(\.name) == ["web_search", "fetch_page"])
        for definition in definitions {
            let schema = try JSONSerialization.jsonObject(
                with: Data(definition.parametersJSON.utf8)) as? [String: Any]
            #expect(schema?["type"] as? String == "object")
            #expect((schema?["required"] as? [String])?.isEmpty == false)
        }
    }

    @Test func systemPromptNamesTheDateAndTheRoundBudget() {
        let date = Calendar(identifier: .gregorian).date(
            from: DateComponents(timeZone: TimeZone(identifier: "Asia/Tokyo"),
                                 year: 2026, month: 9, day: 2))!
        let text = WebSearchPrompt.system(date: date, maxRounds: 6)
        #expect(text.contains("2026年9月2日"))
        #expect(text.contains("6 回まで"))
        #expect(text.contains("添えられた本文を読んで答えます"))
    }

    @Test func configurationRoundTripsAndAppliesTheEnvironment() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("web-search-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("web-search.json")
        var configuration = WebSearchConfiguration()
        configuration.serperAPIKey = "stored"
        configuration.maxToolRounds = 3
        try WebSearchConfigurationStore.save(configuration, to: file)
        let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
        #expect((attributes[.posixPermissions] as? Int) == 0o600)
        let loaded = WebSearchConfigurationStore.load(from: file)
        #expect(loaded == configuration)
        let resolved = loaded.resolved(environment: [
            WebSearchConfiguration.serperEnvironmentKey: "from-env",
        ])
        #expect(resolved.serperAPIKey == "from-env")
        #expect(resolved.maxToolRounds == 3)
        #expect(WebSearchConfigurationStore.load(from: directory.appendingPathComponent("missing.json"))
                == WebSearchConfiguration())
        #expect(!WebSearchConfiguration().canSearch)
        #expect(loaded.canSearch)
    }
}
