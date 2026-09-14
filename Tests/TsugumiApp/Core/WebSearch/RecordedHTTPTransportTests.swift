import Foundation
import Testing
@testable import TsugumiAppCore

@Suite struct RecordedHTTPTransportTests {
    func store() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("recorded-http-\(UUID().uuidString)", isDirectory: true)
        return directory
    }

    func executor(_ transport: any HTTPTransport, pageCharacters: Int = 1_000) -> WebSearchToolExecutor {
        var configuration = WebSearchConfiguration()
        configuration.serperAPIKey = "serper-key"
        configuration.preferJinaReader = false
        configuration.pageCharacterLimit = pageCharacters
        return WebSearchToolExecutor(configuration: configuration, transport: transport,
                                     today: WebSearchToolExecutorTests.fixedDay)
    }

    static let calls = [
        AppToolCall(id: "c1", name: "web_search", argumentsJSON: #"{"query":"東京 人口"}"#),
        AppToolCall(id: "c2", name: "fetch_page", argumentsJSON: #"{"url":"https://example.jp/tokyo"}"#),
        AppToolCall(id: "c3", name: "fetch_page", argumentsJSON: #"{"url":"https://down.example.jp/"}"#),
    ]

    /// The second run reads the same results without the network, and a smaller page limit clips the same body.
    @Test func aRecordedRunReplaysWithoutTheNetwork() async throws {
        let directory = try store()
        defer { try? FileManager.default.removeItem(at: directory) }
        let page = String(repeating: "東京都の推計人口は約1,400万人です。", count: 80)
        let live = StubTransport([
            "google.serper.dev": [.init(status: 200, body: WebSearchToolExecutorTests.serperBody)],
            "example.jp": [.init(status: 200, body: Data(page.utf8), contentType: "text/plain; charset=utf-8")],
        ])
        let recording = try RecordedHTTPTransport(directory: directory, live: live)
        var first: [AppToolResult] = []
        for call in Self.calls { first.append(await executor(recording).execute(call)) }
        #expect(recording.takeCounts() == .init(replayed: 0, recorded: 4))  // search, page, down.example.jp direct and via Jina
        #expect(!first[1].isError && first[2].isError)

        let offline = StubTransport([:])
        let replaying = try RecordedHTTPTransport(directory: directory, live: offline)
        var second: [AppToolResult] = []
        for call in Self.calls { second.append(await executor(replaying).execute(call)) }
        #expect(offline.requests.isEmpty)
        #expect(replaying.takeCounts() == .init(replayed: 4, recorded: 0))
        #expect(second.map(\.content) == first.map(\.content))
        #expect(second.map(\.isError) == first.map(\.isError))

        let short = await executor(replaying, pageCharacters: 500).execute(Self.calls[1])
        #expect(first[1].content.contains("…(本文はここで打ち切り。") && short.content.contains("…(本文はここで打ち切り。"))
        #expect(short.content.count < first[1].content.count)
        #expect(offline.requests.isEmpty)
    }

    /// A JSON body keys on its content, not on the order a dictionary happened to serialize in; no header is kept.
    @Test func theKeyIgnoresJSONKeyOrderAndHeaders() throws {
        var a = URLRequest(url: URL(string: "https://google.serper.dev/search")!)
        a.httpMethod = "POST"
        a.httpBody = Data(#"{"q":"東京","gl":"jp","hl":"ja"}"#.utf8)
        a.setValue("secret-1", forHTTPHeaderField: "X-API-KEY")
        var b = a
        b.httpBody = Data(#"{"hl":"ja","gl":"jp","q":"東京"}"#.utf8)
        b.setValue("secret-2", forHTTPHeaderField: "X-API-KEY")
        #expect(RecordedHTTPTransport.key(for: a) == RecordedHTTPTransport.key(for: b))
        #expect(!RecordedHTTPTransport.key(for: a).contains("secret"))
        var c = a
        c.httpBody = Data(#"{"q":"大阪","gl":"jp","hl":"ja"}"#.utf8)
        #expect(RecordedHTTPTransport.key(for: a) != RecordedHTTPTransport.key(for: c))
        let get = URLRequest(url: URL(string: "https://r.jina.ai/https://example.jp/")!)
        #expect(RecordedHTTPTransport.key(for: get) == "GET https://r.jina.ai/https://example.jp/")
    }
}
