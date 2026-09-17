import Foundation
import Testing
@testable import TsugumiAppCore

@Suite struct SearchProxyTransportsTests {
    func directory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("search-proxy-\(UUID().uuidString)", isDirectory: true)
    }

    func executor(_ transport: any HTTPTransport, brave: Bool = false) -> WebSearchToolExecutor {
        var configuration = WebSearchConfiguration()
        configuration.serperAPIKey = "serper-key"
        if brave { configuration.braveAPIKey = "brave-key" }
        configuration.preferJinaReader = false
        return WebSearchToolExecutor(configuration: configuration, transport: transport,
                                     today: WebSearchToolExecutorTests.fixedDay)
    }

    func search(_ query: String) -> AppToolCall {
        AppToolCall(id: UUID().uuidString, name: "web_search", argumentsJSON: #"{"query":"\#(query)"}"#)
    }

    func metered(_ stub: StubTransport) -> Int {
        stub.requests.filter(MeteredSearchHosts.contains).count
    }

    /// Past the budget a new query reaches neither Serper nor Brave, and the refusal is not recorded: a later run
    /// with budget still gets the query live. A recorded query is replayed without spending.
    @Test func theBudgetStopsNewSearchesOnly() async throws {
        let dir = directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let live = StubTransport([
            "google.serper.dev": [.init(status: 200, body: WebSearchToolExecutorTests.serperBody)],
            "api.search.brave.com": [.init(status: 200, body: WebSearchToolExecutorTests.braveBody)],
        ])
        let budget = SearchBudgetTransport(budget: 1, store: try RecordedHTTPTransport(directory: dir, live: live))
        let tools = executor(budget, brave: true)

        #expect(!(await tools.execute(search("東京 人口"))).isError)
        #expect(!(await tools.execute(search("東京 人口"))).isError)  // recorded: not counted
        let refused = await tools.execute(search("大阪 人口"))
        #expect(refused.isError)
        #expect(metered(live) == 1)
        #expect(budget.used == 1)

        let again = SearchBudgetTransport(budget: 1, store: try RecordedHTTPTransport(directory: dir, live: live))
        #expect(!(await executor(again).execute(search("大阪 人口"))).isError)
        #expect(metered(live) == 2)
    }

    /// Every query under a pin reads the first answer, re-ranked by the conversation's order; another pin fetches.
    @Test func aPinAnswersEveryQueryWithItsFirstResult() async throws {
        let dir = directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let live = StubTransport([
            "google.serper.dev": [.init(status: 200, body: WebSearchToolExecutorTests.serperBody)],
        ])
        let store = try RecordedHTTPTransport(directory: dir, live: live)
        let pinned = try PinnedSearchTransport(directory: dir, inner: SearchBudgetTransport(budget: 2, store: store))
        let tools = executor(pinned)

        pinned.start(pin: "tokyo", order: nil)
        let first = await tools.execute(search("東京 人口"))
        let reworded = await tools.execute(search("東京都 人口 2026"))
        #expect(reworded.content.replacingOccurrences(of: "東京都 人口 2026", with: "東京 人口") == first.content)
        #expect(pinned.takeCounts() == .init(pinned: 1, fetched: 1))

        pinned.start(pin: "tokyo", order: [2])
        let swapped = await tools.execute(search("何でも"))
        let tokyo = try #require(swapped.content.range(of: "https://example.jp/tokyo"))
        let stat = try #require(swapped.content.range(of: "https://stat.example.jp/"))
        #expect(stat.lowerBound < tokyo.lowerBound)

        pinned.start(pin: "osaka", order: nil)
        _ = await tools.execute(search("大阪 人口"))
        #expect(pinned.takeCounts() == .init(pinned: 1, fetched: 1))
        #expect(metered(live) == 2)

        // A new process reads the pins from the directory without the network.
        let offline = StubTransport([:])
        let later = try PinnedSearchTransport(
            directory: dir,
            inner: SearchBudgetTransport(budget: 0, store: try RecordedHTTPTransport(directory: dir, live: offline)))
        later.start(pin: "tokyo", order: nil)
        #expect((await executor(later).execute(search("別の言い方"))).content.contains("https://example.jp/tokyo"))
        #expect(offline.requests.isEmpty)
    }

    /// A failed search is not pinned, so the pin still fetches once the API answers.
    @Test func aFailureIsNotPinned() async throws {
        let dir = directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let live = StubTransport([
            "google.serper.dev": [.init(status: 500, body: Data()),
                                  .init(status: 200, body: WebSearchToolExecutorTests.serperBody)],
        ])
        let pinned = try PinnedSearchTransport(directory: dir, inner: live)
        pinned.start(pin: "tokyo", order: nil)
        #expect((await executor(pinned).execute(search("東京 人口"))).isError)
        #expect(!(await executor(pinned).execute(search("東京 人口"))).isError)
        #expect(pinned.takeCounts() == .init(pinned: 0, fetched: 2))
    }
}
