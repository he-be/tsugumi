import Foundation
import Testing
@testable import TsugumiAppCore

/// The fixture index is built by `Scripts/wiki/build_jawiki_index.py` from
/// `Fixtures/wikipedia-fixture.jsonl`, so these tests also prove the Python
/// writer and the Swift reader agree on the layout and the tokenizer.
private func fixtureIndex() throws -> LocalWikipediaIndex {
    let url = try #require(Bundle.module.url(forResource: "wikipedia-fixture", withExtension: "sqlite",
                                             subdirectory: "Fixtures"))
    return try LocalWikipediaIndex(path: url.path)
}

@Suite struct WikipediaTokenizerTests {
    @Test func matchesTheBuildScriptOnTheCheckSample() {
        // The value the script writes into meta.tokenizer_check for the same sample.
        let expected = "東京 京タ タワ ワー ーは は2 20 02 26 6年 年9 9月 月1 1日 日に iphone 16 pro で撮 撮っ った a 1 アイ イウ 々 ー a b m 4"
        #expect(WikipediaTokenizer.tokenize(WikipediaTokenizer.checkSample).joined(separator: " ") == expected)
    }

    @Test func bigramsCJKAndDigitsWordsTheRest() {
        #expect(WikipediaTokenizer.tokenize("米国") == ["米国"])
        #expect(WikipediaTokenizer.tokenize("東京タワー") == ["東京", "京タ", "タワ", "ワー"])
        #expect(WikipediaTokenizer.tokenize("Apple の M4") == ["apple", "の", "m", "4"])
        #expect(WikipediaTokenizer.tokenize("2026年") == ["20", "02", "26", "6年"])
        #expect(WikipediaTokenizer.tokenize("・・・") == [])
        #expect(WikipediaTokenizer.tokenize("東京・大阪") == ["東京", "大阪"])
        #expect(WikipediaTokenizer.tokenize("ｶﾀｶﾅ") == ["カタ", "タカ", "カナ"])
    }

    @Test func matchExpressionIsOnePhrasePerTerm() {
        #expect(WikipediaTokenizer.matchExpression("東京タワー 高さ", mode: .allTerms)
                == "\"東京 京タ タワ ワー\" AND \"高さ\"")
        #expect(WikipediaTokenizer.matchExpression("東京 駅", mode: .anyTerm) == "\"東京\" OR \"駅\"")
        #expect(WikipediaTokenizer.matchExpression("東京の駅", mode: .anyToken) == "\"東京\" OR \"京の\" OR \"の駅\"")
        #expect(WikipediaTokenizer.matchExpression("・", mode: .allTerms) == nil)
    }

    @Test func titleNormalizationFoldsCaseWidthAndUnderscores() {
        #expect(WikipediaTokenizer.normalizeTitle("iPhone_16") == "iphone 16")
        #expect(WikipediaTokenizer.normalizeTitle("  Ａ  Ｂ ") == "a b")
    }
}

@Suite struct LocalWikipediaIndexTests {
    @Test func opensAndReadsTheSummary() throws {
        let index = try fixtureIndex()
        #expect(index.summary.articles == 5)
        #expect(index.summary.dumpDate == "20260830")
        #expect(index.summary.dumpDateJapanese == "2026年8月30日")
    }

    @Test func refusesFilesThatAreNotAnIndex() throws {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("not-an-index-\(UUID().uuidString).sqlite")
        try Data("hello".utf8).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        #expect(throws: LocalWikipediaIndex.OpenError.self) { try LocalWikipediaIndex(path: file.path) }
        let missing = LocalWikipediaIndex.probe(path: "/nonexistent/dir/wikipedia-ja.sqlite")
        guard case .failure = missing else { Issue.record("a missing file opened"); return }
    }

    @Test func searchRanksWellLinkedArticlesFirstAndSnipsTheOpening() throws {
        let index = try fixtureIndex()
        let hits = index.search("東京", limit: 5)
        #expect(hits.map(\.title) == ["東京駅", "東京タワー"])
        #expect(hits[0].snippet.hasPrefix("東京駅（とうきょうえき）"))
        #expect(index.search("東京タワー", limit: 5).first?.title == "東京タワー")
    }

    @Test func twoCharacterWordsAndAliasesAreFound() throws {
        let index = try fixtureIndex()
        #expect(index.search("米国", limit: 3).first?.title == "アメリカ合衆国")
        #expect(index.search("iphone", limit: 3).first?.title == "iPhone 16")
        #expect(index.search("存在しない語", limit: 3).isEmpty)
    }

    @Test func orFallbackWhenNoArticleHasEveryTerm() throws {
        let index = try fixtureIndex()
        let hits = index.search("東京タワー ワシントン", limit: 5)
        #expect(Set(hits.map(\.title)) == ["東京タワー", "アメリカ合衆国"])
        // No article has the phrase; its bigrams still find the station.
        #expect(index.search("東京の駅", limit: 5).first?.title == "東京駅")
    }

    @Test func pageLookupFollowsRedirectsAndInflatesTheBody() throws {
        let index = try fixtureIndex()
        let page = try #require(index.page(title: "米国"))
        #expect(page.title == "アメリカ合衆国")
        #expect(page.text.contains("首都はワシントン"))
        #expect(index.page(title: "iphone_16")?.title == "iPhone 16")
        #expect(index.page(title: "日本電波塔")?.title == "東京タワー")
        #expect(index.page(title: "無い記事") == nil)
        #expect(index.page(id: 6)?.text.hasSuffix("おわり") == true)
    }
}

@Suite struct WikipediaToolExecutorTests {
    func executor(limit: Int = 500) throws -> WikipediaToolExecutor {
        WikipediaToolExecutor(index: try fixtureIndex(), maxResults: 5, pageCharacterLimit: limit)
    }

    @Test func declaresTwoOfflineToolsAndItsFacts() throws {
        let executor = try executor()
        #expect(executor.definitions.map(\.name) == ["wikipedia_search", "wikipedia_page"])
        #expect(executor.definitions[0].description.contains("2026年8月30日"))
        #expect(executor.promptFacts == AppToolPromptFacts(web: false, wikipediaDate: "2026年8月30日"))
    }

    @Test func searchRendersNumberedTitlesWithSnippets() async throws {
        let executor = try executor()
        let call = AppToolCall(id: "1", name: "wikipedia_search", argumentsJSON: #"{"query":"東京"}"#)
        let result = await executor.execute(call)
        #expect(!result.isError)
        #expect(result.content.hasPrefix("Wikipedia 検索: 東京 (2 件)\n[1] 東京駅\n    東京駅（"))
        #expect(result.content.contains("[2] 東京タワー"))
        #expect(result.summary == "Wikipedia · 2 hits")
        #expect(executor.subject(of: call) == "東京")
        let empty = await executor.execute(AppToolCall(id: "2", name: "wikipedia_search",
                                                       argumentsJSON: #"{"query":"存在しない語"}"#))
        #expect(!empty.isError)
        #expect(empty.content.contains("該当する記事はありません"))
    }

    @Test func pageClipsAndOffersTheContinuation() async throws {
        let executor = try executor(limit: 500)
        let first = await executor.execute(AppToolCall(id: "1", name: "wikipedia_page",
                                                       argumentsJSON: #"{"title":"長い記事"}"#))
        #expect(first.content.hasPrefix("Wikipedia 記事: 長い記事\n\n長い記事の導入部。"))
        #expect(first.content.contains("続きは from=500"))
        #expect(first.summary.hasSuffix("(clipped)"))
        let rest = await executor.execute(AppToolCall(id: "2", name: "wikipedia_page",
                                                      argumentsJSON: #"{"title":"長い記事","from":500}"#))
        #expect(rest.content.contains("(500 文字目から)"))
        #expect(rest.content.hasSuffix("おわり"))
        #expect(!rest.summary.hasSuffix("(clipped)"))
    }

    @Test func missingPageSuggestsNearTitles() async throws {
        let executor = try executor()
        let result = await executor.execute(AppToolCall(id: "1", name: "wikipedia_page",
                                                        argumentsJSON: #"{"title":"東京の駅"}"#))
        #expect(result.isError)
        #expect(result.content.contains("近い題名: 東京駅"))
        let bad = await executor.execute(AppToolCall(id: "2", name: "wikipedia_page", argumentsJSON: "{}"))
        #expect(bad.isError)
    }
}

@Suite struct CompositeToolExecutorTests {
    struct Fixed: AppToolExecutor {
        var name: String
        var web: Bool
        var definitions: [AppToolDefinition] { [AppToolDefinition(name: name, description: "", parametersJSON: "{}")] }
        var promptFacts: AppToolPromptFacts { AppToolPromptFacts(web: web, wikipediaDate: web ? nil : "d") }
        func execute(_ call: AppToolCall) async -> AppToolResult { AppToolResult(content: name) }
        func subject(of call: AppToolCall) -> String { name + ":" + call.argumentsJSON }
    }

    @Test func dispatchesByNameAndMergesFacts() async {
        let composite = CompositeToolExecutor([Fixed(name: "wikipedia_search", web: false),
                                               Fixed(name: "web_search", web: true)])
        #expect(composite.definitions.map(\.name) == ["wikipedia_search", "web_search"])
        #expect(composite.promptFacts == AppToolPromptFacts(web: true, wikipediaDate: "d"))
        let call = AppToolCall(id: "1", name: "web_search", argumentsJSON: "{}")
        #expect(await composite.execute(call).content == "web_search")
        #expect(composite.subject(of: call) == "web_search:{}")
        let unknown = await composite.execute(AppToolCall(id: "2", name: "nope", argumentsJSON: "{}"))
        #expect(unknown.isError)
    }
}

@Suite struct SearchToolPromptTests {
    @Test func webOnlyNamesOnlyTheWebTools() {
        let text = WebSearchPrompt.system(maxRounds: 6, mode: .auto)
        #expect(text.contains("web_search と fetch_page (いまのインターネット)"))
        #expect(!text.contains("wikipedia_search"))
        #expect(!text.contains("{"))
        #expect(text.contains("「参照:」"))
    }

    @Test func wikipediaOnlySaysOfflineAndDated() {
        let text = WebSearchPrompt.system(maxRounds: 6, mode: .always,
                                          tools: AppToolPromptFacts(web: false, wikipediaDate: "2026年8月30日"))
        #expect(text.contains("2026年8月30日 時点"))
        #expect(text.contains("インターネットには接続しません"))
        #expect(!text.contains("web_search"))
        #expect(text.contains("刻々と変わることは Wikipedia にはありません"))
        #expect(text.hasSuffix("この会話では、まず必ず検索してから答えます。"))
        #expect(!text.contains("{"))
    }

    @Test func bothToolsGetTheChoiceRule() {
        let text = WebSearchPrompt.system(maxRounds: 6, mode: .auto,
                                          tools: AppToolPromptFacts(web: true, wikipediaDate: "2026年8月30日"))
        #expect(text.contains("使い分け:"))
        #expect(text.contains("wikipedia_search と wikipedia_page"))
        #expect(text.contains("web_search と fetch_page"))
        #expect(text.contains("(Wikipedia の記事名やURL)"))
        #expect(!text.contains("{"))
    }
}

@Suite struct ToolExecutorFactoryTests {
    @Test func wikipediaAloneNeedsNoKeyAndABadPathIsAnError() throws {
        var configuration = WebSearchConfiguration()
        #expect(!configuration.canUseTools)
        #expect(throws: AppInferenceError.self) { try AppModel.makeToolExecutor(configuration: configuration) }
        let url = try #require(Bundle.module.url(forResource: "wikipedia-fixture", withExtension: "sqlite",
                                                 subdirectory: "Fixtures"))
        configuration.wikipediaIndexPath = url.path
        #expect(configuration.canUseTools && !configuration.canSearch)
        let alone = try AppModel.makeToolExecutor(configuration: configuration)
        #expect(alone.definitions.map(\.name) == ["wikipedia_search", "wikipedia_page"])
        configuration.serperAPIKey = "k"
        let both = try AppModel.makeToolExecutor(configuration: configuration)
        #expect(both.definitions.map(\.name) == ["wikipedia_search", "wikipedia_page", "web_search", "fetch_page"])
        #expect(both.promptFacts == AppToolPromptFacts(web: true, wikipediaDate: "2026年8月30日"))
        configuration.wikipediaIndexPath = "/nonexistent/wikipedia-ja.sqlite"
        #expect(throws: AppInferenceError.self) { try AppModel.makeToolExecutor(configuration: configuration) }
    }

    @Test func environmentOverridesThePath() {
        var configuration = WebSearchConfiguration()
        configuration.wikipediaIndexPath = "~/a.sqlite"
        #expect(configuration.wikipediaIndexURL?.path.hasSuffix("/a.sqlite") == true)
        #expect(configuration.wikipediaIndexURL?.path.hasPrefix("~") == false)
        let resolved = configuration.resolved(environment: [WebSearchConfiguration.wikipediaEnvironmentKey: "/env.sqlite"])
        #expect(resolved.wikipediaIndexPath == "/env.sqlite")
    }
}

/// Against a real index, when `TSUGUMI_WIKIPEDIA_INDEX` names one — a
/// check that the file the script built opens in Swift and answers.
@Suite struct RealWikipediaIndexTests {
    @Test func opensSearchesAndReadsAPage() throws {
        guard let path = ProcessInfo.processInfo.environment[WebSearchConfiguration.wikipediaEnvironmentKey],
              !path.isEmpty else { return }
        let index = try LocalWikipediaIndex(path: path)
        #expect(index.summary.articles > 0)
        let hits = index.search("東京タワー", limit: 5)
        #expect(!hits.isEmpty)
        let page = try #require(index.page(title: hits[0].title))
        #expect(!page.text.isEmpty)
        #expect(index.search("米国", limit: 3).isEmpty == false)
    }
}
