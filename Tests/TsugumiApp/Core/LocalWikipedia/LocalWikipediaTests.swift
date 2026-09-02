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

@Suite struct WikipediaMentionFinderTests {
    @Test func cutsWordsAndDropsURLs() {
        let words = WikipediaMentionFinder.words(in: "淀城の遺構が https://example.jp/a/b?c=淀城 見つかった")
        #expect(words.contains("遺構"))
        #expect(!words.contains { $0.contains("example") || $0.contains("http") })
        #expect(WikipediaMentionFinder.words(in: "").isEmpty)
        #expect(WikipediaMentionFinder.candidates(in: []).isEmpty)
    }

    @Test func candidatesAreEveryWindowLongestFirst() {
        let candidates = WikipediaMentionFinder.candidates(in: ["城崎", "シー", "ワールド", "に", "つい", "て"])
        #expect(candidates.first?.text == "城崎シーワールドに")
        #expect(candidates.first?.words == 0..<4)
        #expect(candidates.contains { $0.text == "城崎シーワールド" && $0.wordCount == 3 })
        #expect(candidates.contains { $0.text == "シーワールド" && $0.words == 1..<3 })
        // One-character windows are not names; "に" and "て" are skipped.
        #expect(!candidates.contains { $0.text == "に" })
        #expect(candidates.last?.text == "つい")
        // Latin words keep their space, CJK neighbours do not.
        let latin = WikipediaMentionFinder.candidates(in: ["M6", "Mac", "mini", "について"])
        #expect(latin.contains { $0.text == "Mac mini" && $0.isLatin })
        #expect(latin.contains { $0.text == "Mac miniについて" })
        // Numbers alone are not looked up.
        #expect(!WikipediaMentionFinder.candidates(in: ["2026", "年"]).contains { $0.text == "2026" })
    }

    @Test func keepsNamesByLinkProbabilityAndShape() {
        // The numbers of the chat history (docs/LOCAL_WIKIPEDIA.md §5).
        #expect(WikipediaMentionFinder.keeps(linkProbability: 5.5, wordCount: 1, isLatin: false))   // 米国
        #expect(WikipediaMentionFinder.keeps(linkProbability: 1.28, wordCount: 2, isLatin: false))  // 淀城
        #expect(WikipediaMentionFinder.keeps(linkProbability: 0.17, wordCount: 2, isLatin: false))  // 熊本地震
        #expect(WikipediaMentionFinder.keeps(linkProbability: 0.99, wordCount: 1, isLatin: true))   // IBM
        #expect(!WikipediaMentionFinder.keeps(linkProbability: 0.33, wordCount: 1, isLatin: false)) // 遺構
        #expect(!WikipediaMentionFinder.keeps(linkProbability: 0.57, wordCount: 1, isLatin: false)) // クーデター
        #expect(!WikipediaMentionFinder.keeps(linkProbability: 0.04, wordCount: 1, isLatin: false)) // ニュース
        #expect(!WikipediaMentionFinder.keeps(linkProbability: 0.06, wordCount: 2, isLatin: false)) // もう少し
        #expect(!WikipediaMentionFinder.keeps(linkProbability: 0.12, wordCount: 1, isLatin: true))  // granite
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
        #expect(index.search("東京タワー", limit: 5).first?.isExactTitle == true)
        #expect(hits[0].isExactTitle == false)
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

    @Test func mentionsNameTheArticlesInAPrompt() throws {
        let index = try fixtureIndex()
        let found = index.mentions(in: "東京タワーと米国の話をして", limit: 3)
        #expect(found.map(\.title) == ["アメリカ合衆国", "東京タワー"])
        #expect(found[0].mention == "米国")
        #expect(found[0].incomingLinks == 90000 && found[0].documentFrequency == 1)
        #expect(found[1].mention == "東京タワー")
        #expect(found[1].opening.hasPrefix("東京タワー（とうきょうタワー）"))
        // The longer span claims its words: 東京 alone is not tried after 東京タワー.
        #expect(!found.contains { $0.title == "東京駅" })
        // Latin titles join with a space; the limit trims the tail.
        #expect(index.mentions(in: "iPhone 16 が欲しい", limit: 3).map(\.title) == ["iPhone 16"])
        #expect(index.mentions(in: "東京タワーと米国", limit: 1).count == 1)
        #expect(index.mentions(in: "存在しない語ばかり", limit: 3).isEmpty)
        #expect(WikipediaTokenizer.phraseExpression("Mac mini") == "\"mac mini\"")
        #expect(WikipediaTokenizer.phraseExpression("・") == nil)
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
        #expect(result.content.hasPrefix("Wikipedia 検索: 東京 (2 件、2026年8月30日 時点の複製)\n[1] 東京駅\n    東京駅（"))
        #expect(result.content.contains("[2] 東京タワー"))
        // One term: the top hit's text follows the list (the "go" fold).
        #expect(result.content.contains("\n\n[1] 東京駅 の本文:\n東京駅（"))
        #expect(result.content.hasSuffix("他の記事を読むには wikipedia_page に題名を渡します。"))
        #expect(result.summary.hasPrefix("Wikipedia · 2 hits + 東京駅 "))
        #expect(executor.subject(of: call) == "東京")
        let empty = await executor.execute(AppToolCall(id: "2", name: "wikipedia_search",
                                                       argumentsJSON: #"{"query":"存在しない語"}"#))
        #expect(!empty.isError)
        #expect(empty.content.contains("該当する記事はありません"))
    }

    @Test func goFoldsTheArticleForTitleMatchesAndOneTermQueries() async throws {
        let executor = try executor(limit: 500)
        // A title match among several terms: the fold is on and clipped.
        let exact = await executor.execute(AppToolCall(id: "1", name: "wikipedia_search",
                                                       argumentsJSON: #"{"query":"長い記事"}"#))
        #expect(exact.content.contains("[1] 長い記事 の本文:\n長い記事の導入部。"))
        #expect(exact.content.contains("続きは wikipedia_page の from=500"))
        #expect(exact.summary.hasSuffix("(clipped)"))
        // A redirect name is a title match too.
        let redirect = await executor.execute(AppToolCall(id: "2", name: "wikipedia_search",
                                                          argumentsJSON: #"{"query":"米国"}"#))
        #expect(redirect.content.contains("[1] アメリカ合衆国 の本文:\n"))
        // Several terms and no title match: only the list, as before.
        let described = await executor.execute(AppToolCall(id: "3", name: "wikipedia_search",
                                                           argumentsJSON: #"{"query":"東京タワー ワシントン"}"#))
        #expect(!described.content.contains("の本文:"))
        #expect(described.content.hasSuffix("本文を読むには wikipedia_page に題名を渡します。"))
        #expect(described.summary == "Wikipedia · 2 hits")
        #expect(!WikipediaToolExecutor.shouldGo([], query: "x"))
    }

    @Test func pageClipsAndOffersTheContinuation() async throws {
        let executor = try executor(limit: 500)
        let first = await executor.execute(AppToolCall(id: "1", name: "wikipedia_page",
                                                       argumentsJSON: #"{"title":"長い記事"}"#))
        #expect(first.content.hasPrefix("Wikipedia 記事: 長い記事 (2026年8月30日 時点)\n\n長い記事の導入部。"))
        #expect(first.content.contains("続きは wikipedia_page の from=500"))
        #expect(first.summary.hasSuffix("(clipped)"))
        let rest = await executor.execute(AppToolCall(id: "2", name: "wikipedia_page",
                                                      argumentsJSON: #"{"title":"長い記事","from":500}"#))
        #expect(rest.content.contains("(500 文字目から)"))
        #expect(rest.content.hasSuffix("おわり"))
        #expect(!rest.summary.hasSuffix("(clipped)"))
    }

    @Test func lookupRendersTheOpeningsAsAReference() async throws {
        let executor = try executor()
        let lookup = try #require(await executor.lookups(prompt: "東京タワーと米国の話", callIDPrefix: "lookup-").first)
        #expect(lookup.call == AppToolCall(id: "lookup-1", name: "wikipedia_lookup",
                                          argumentsJSON: #"{"titles":["アメリカ合衆国","東京タワー"]}"#))
        #expect(lookup.subject == "アメリカ合衆国 / 東京タワー")
        #expect(lookup.result.summary == "Wikipedia · 2 件")
        #expect(lookup.result.content.hasPrefix("参考: 質問に含まれる語を Wikipedia (2026年8月30日 時点の複製) で引いた記事の導入部です。"))
        #expect(lookup.result.content.contains("\n■ アメリカ合衆国 (質問中の「米国」)\nアメリカ合衆国（"))
        #expect(lookup.result.content.contains("\n■ 東京タワー\n東京タワー（"))
        #expect(await executor.lookups(prompt: "なにもない", callIDPrefix: "lookup-").isEmpty)
        #expect(WikipediaToolExecutor.clip("abcdef", to: 3) == "abc…")
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

    struct Seeding: AppToolExecutor {
        var name: String
        var definitions: [AppToolDefinition] { [] }
        func execute(_ call: AppToolCall) async -> AppToolResult { AppToolResult(content: "") }
        func lookups(prompt: String, callIDPrefix: String) async -> [AppToolLookup] {
            [AppToolLookup(call: AppToolCall(id: callIDPrefix + "1", name: name, argumentsJSON: "{}"),
                           result: AppToolResult(content: name), subject: name)]
        }
    }

    @Test func dispatchesByNameAndMergesFacts() async {
        let composite = CompositeToolExecutor([Fixed(name: "wikipedia_search", web: false),
                                               Fixed(name: "web_search", web: true)])
        #expect(composite.definitions.map(\.name) == ["wikipedia_search", "web_search"])
        #expect(composite.promptFacts == AppToolPromptFacts(web: true, wikipediaDate: "d"))
        let call = AppToolCall(id: "1", name: "web_search", argumentsJSON: "{}")
        #expect(await composite.execute(call).content == "web_search")
        // Each executor numbers its lookups under its own prefix.
        let seeded = CompositeToolExecutor([Seeding(name: "a"), Seeding(name: "b")])
        #expect(await seeded.lookups(prompt: "p", callIDPrefix: "lookup-").map(\.call.id) == ["lookup-1-1", "lookup-2-1"])
        #expect(composite.subject(of: call) == "web_search:{}")
        let unknown = await composite.execute(AppToolCall(id: "2", name: "nope", argumentsJSON: "{}"))
        #expect(unknown.isError)
    }
}

@Suite struct SearchToolPromptTests {
    @Test func webOnlyNamesOnlyTheWebTools() {
        let text = WebSearchPrompt.system(maxRounds: 6)
        #expect(text.contains("web_search と fetch_page (いまのインターネット)"))
        #expect(!text.contains("wikipedia_search"))
        #expect(!text.contains("{"))
        #expect(text.contains("「参照:」"))
    }

    @Test func wikipediaOnlySaysOfflineAndDated() {
        let text = WebSearchPrompt.system(maxRounds: 6,
                                          tools: AppToolPromptFacts(web: false, wikipediaDate: "2026年8月30日"))
        #expect(text.contains("2026年8月30日 時点"))
        #expect(text.contains("インターネットには接続しません"))
        #expect(!text.contains("web_search"))
        #expect(text.contains("刻々と変わることは Wikipedia にはありません"))
        #expect(text.contains("wikipedia_lookup"))
        #expect(!text.contains("{"))
    }

    @Test func bothToolsGetTheChoiceRule() {
        let text = WebSearchPrompt.system(maxRounds: 6,
                                          tools: AppToolPromptFacts(web: true, wikipediaDate: "2026年8月30日"))
        #expect(text.contains("使い分け:"))
        #expect(text.contains("wikipedia_search と wikipedia_page"))
        #expect(text.contains("web_search と fetch_page"))
        #expect(text.contains("(Wikipedia の記事名やURL)"))
        #expect(!text.contains("{"))
    }
}

@Suite struct ToolExecutorFactoryTests {
    @Test func offlineDeclaresWikipediaOrNothingAndOnlineNeedsAKey() throws {
        var configuration = WebSearchConfiguration()
        // Nothing set: offline is a plain turn, online is an error to act on.
        #expect(try AppModel.makeToolExecutor(configuration: configuration, mode: .offline) == nil)
        #expect(throws: AppInferenceError.self) { try AppModel.makeToolExecutor(configuration: configuration, mode: .online) }
        let url = try #require(Bundle.module.url(forResource: "wikipedia-fixture", withExtension: "sqlite",
                                                 subdirectory: "Fixtures"))
        configuration.wikipediaIndexPath = url.path
        #expect(configuration.canUseTools && !configuration.canSearch)
        let alone = try #require(try AppModel.makeToolExecutor(configuration: configuration, mode: .offline))
        #expect(alone.definitions.map(\.name) == ["wikipedia_search", "wikipedia_page"])
        // An index alone does not make online: the web tools need a key.
        #expect(throws: AppInferenceError.self) { try AppModel.makeToolExecutor(configuration: configuration, mode: .online) }
        configuration.serperAPIKey = "k"
        // Offline keeps the key out of it.
        #expect(try AppModel.makeToolExecutor(configuration: configuration, mode: .offline)?.definitions.map(\.name)
                == ["wikipedia_search", "wikipedia_page"])
        let both = try #require(try AppModel.makeToolExecutor(configuration: configuration, mode: .online))
        #expect(both.definitions.map(\.name) == ["wikipedia_search", "wikipedia_page", "web_search", "fetch_page"])
        #expect(both.promptFacts == AppToolPromptFacts(web: true, wikipediaDate: "2026年8月30日"))
        configuration.wikipediaIndexPath = "/nonexistent/wikipedia-ja.sqlite"
        #expect(throws: AppInferenceError.self) { try AppModel.makeToolExecutor(configuration: configuration, mode: .offline) }
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
    @Test func opensSearchesAndReadsAPage() async throws {
        guard let path = ProcessInfo.processInfo.environment[WebSearchConfiguration.wikipediaEnvironmentKey],
              !path.isEmpty else { return }
        let index = try LocalWikipediaIndex(path: path)
        #expect(index.summary.articles > 0)
        let hits = index.search("東京タワー", limit: 5)
        #expect(!hits.isEmpty)
        let page = try #require(index.page(title: hits[0].title))
        #expect(!page.text.isEmpty)
        #expect(index.search("米国", limit: 3).isEmpty == false)
        // The two shapes the chat history showed: a title the model got
        // slightly wrong (one term → the best match's text is folded in),
        // and a title it got right (a redirect/title match).
        let executor = WikipediaToolExecutor(index: index, maxResults: 8, pageCharacterLimit: 3_000)
        let near = await executor.execute(AppToolCall(id: "1", name: "wikipedia_search",
                                                      argumentsJSON: #"{"query":"城崎シーワールド"}"#))
        #expect(near.content.contains("[1] 城崎マリンワールド の本文:"))
        let exact = await executor.execute(AppToolCall(id: "2", name: "wikipedia_search",
                                                       argumentsJSON: #"{"query":"熊本地震"}"#))
        #expect(exact.content.contains("[1] 熊本地震 の本文:"))
        #expect(exact.content.contains("時点の複製"))
        // The lookup on three prompts of the history: the name, not the nouns
        // around it; two names in one prompt; nothing out of a URL.
        #expect(index.mentions(in: "淀城の遺構が桂川の工事中に見つかったって本当？", limit: 3).map(\.title) == ["淀城"])
        let rail = index.mentions(in: "えきねっと、e5489、ex予約など、鉄道のWEBは利用者を苦しめるのが要件に入ってるのか？", limit: 3)
        #expect(Set(rail.map(\.title)).isSuperset(of: ["えきねっと", "エクスプレス予約"]))
        #expect(!rail.contains { $0.title == "鉄道" })
        let commit = index.mentions(in: "https://github.com/torvalds/linux/commit/a5148bc2fa27092862ac4b9e7b5c8340d60cff34 を解説して", limit: 3)
        #expect(commit.isEmpty)
    }
}
