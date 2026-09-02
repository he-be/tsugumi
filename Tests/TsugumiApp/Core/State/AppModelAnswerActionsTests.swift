import Foundation
import Testing
@testable import TsugumiAppCore

/// The actions under the last answer: regenerate (with or without a
/// directive), switch between the answers it produced, open a follow-up
/// turn, search and answer again — and the persona that leads every
/// system prompt.
@Suite struct AppModelAnswerActionsTests {
    let searchCall = AppToolCall(id: "call-1", name: "web_search",
                                 argumentsJSON: #"{"query":"淀城 遺構"}"#)

    @MainActor
    @Test func regenerateKeepsTheQuestionAndSetsTheOldAnswerAside() async throws {
        let client = ScriptedToolClient([.answer("A"), .answer("B")])
        let model = readyModel(client: client)
        model.promptText = "淀城の遺構は本当?"
        model.run()
        await waitForIdle(model)
        #expect(model.canRegenerate)
        #expect(model.answerCount == 1)

        model.regenerate(.again)
        await waitForIdle(model)

        #expect(client.requests.count == 2)
        let second = try #require(client.requests.last)
        #expect(second.prompt == "淀城の遺構は本当?")
        #expect(second.history.isEmpty)
        #expect(model.outputPromptText == "淀城の遺構は本当?")
        #expect(model.outputResponsePlainText == "B")
        #expect(model.outputDirective == nil)
        #expect(model.outputVariants.map(\.text) == ["A"])
        #expect(model.selectedVariantIndex == 1)
        #expect(model.answerCount == 2)
    }

    @MainActor
    @Test func aDirectiveIsAppendedToTheQuestionAndFoldedAsSent() async throws {
        let client = ScriptedToolClient([.answer("A"), .answer("B"), .answer("C")])
        let model = readyModel(client: client)
        model.promptText = "q"
        model.run()
        await waitForIdle(model)

        model.regenerate(.concise)
        await waitForIdle(model)
        let regenerated = try #require(client.requests.last)
        #expect(regenerated.prompt == "q\n\n" + AppAnswerDirective.concise.instruction)
        #expect(model.outputPromptText == "q")
        #expect(model.outputDirective == .concise)
        #expect(model.outputVariants.first?.directive == nil)

        model.promptText = "next"
        model.run()
        await waitForIdle(model)
        let next = try #require(client.requests.last)
        #expect(next.history.map(\.text) == ["q\n\n" + AppAnswerDirective.concise.instruction, "B"])
        #expect(model.outputVariants.isEmpty)
        #expect(model.outputDirective == nil)
        #expect(model.selectedVariantIndex == 0)
    }

    @MainActor
    @Test func selectingAVariantSwapsTheDisplayAndTheFoldFollowsIt() async throws {
        let client = ScriptedToolClient([.answer("A"), .answer("B"), .answer("C"), .answer("D")])
        let model = readyModel(client: client)
        model.promptText = "q"
        model.run()
        await waitForIdle(model)
        model.regenerate(.again)
        await waitForIdle(model)
        model.regenerate(.blunt)
        await waitForIdle(model)
        #expect(model.outputResponsePlainText == "C")
        #expect(model.outputVariants.map(\.text) == ["A", "B"])
        #expect(model.selectedVariantIndex == 2)

        model.selectVariant(0)
        #expect(model.outputResponsePlainText == "A")
        #expect(model.outputDirective == nil)
        #expect(model.outputVariants.map(\.text) == ["B", "C"])
        #expect(model.selectedVariantIndex == 0)

        model.selectVariant(2)
        #expect(model.outputResponsePlainText == "C")
        #expect(model.outputDirective == .blunt)
        #expect(model.outputVariants.map(\.text) == ["A", "B"])

        model.selectVariant(1)
        #expect(model.outputResponsePlainText == "B")
        // A regeneration from the middle appends at the end.
        model.regenerate(.again)
        await waitForIdle(model)
        #expect(model.outputResponsePlainText == "D")
        #expect(model.outputVariants.map(\.text) == ["A", "B", "C"])
        #expect(model.selectedVariantIndex == 3)

        model.selectVariant(1)
        model.promptText = "next"
        model.run()
        await waitForIdle(model)
        #expect(model.conversationTurns.map(\.text) == ["q", "B"])
    }

    @MainActor
    @Test func oppositeViewOpensANewTurnAndLeavesTheDraft() async throws {
        let client = ScriptedToolClient([.answer("A"), .answer("B")])
        let model = readyModel(client: client)
        model.promptText = "q"
        model.run()
        await waitForIdle(model)
        model.promptText = "書きかけ"

        model.askFollowUp(.opposite)
        #expect(model.isRunning)
        #expect(model.promptText == "書きかけ")
        await waitForIdle(model)

        let request = try #require(client.requests.last)
        #expect(request.prompt == AppFollowUp.opposite.prompt)
        #expect(request.history.map(\.text) == ["q", "A"])
        #expect(model.outputResponsePlainText == "B")
        #expect(model.outputVariants.isEmpty)
    }

    @MainActor
    @Test func searchAgainGoesOnlineAndRequiresATool() async throws {
        let client = ScriptedToolClient([
            .answer("知りません"),
            .calls([searchCall]),
            .answer("見つかりました。\n参照: https://example.jp/yodo"),
        ])
        let executor = ScriptedToolExecutor(results: [
            "call-1": AppToolResult(content: "検索: 淀城 遺構 (Serper, 1 件)", summary: "Serper · 1 hits"),
        ])
        let model = readyModel(client: client, executor: executor)
        model.networkMode = .offline
        model.promptText = "淀城の遺構は本当?"
        model.run()
        await waitForIdle(model)
        #expect(model.outputGrounding.isEmpty)
        #expect(model.canSearchAgain)

        model.regenerate(.searched)
        await waitForIdle(model)

        #expect(model.error == nil)
        let first = client.requests[1]
        #expect(first.tools.map(\.name) == ["web_search", "fetch_page"])
        #expect(first.toolChoice == .function(name: "web_search"))
        #expect(first.prompt.hasSuffix(AppAnswerDirective.searched.instruction))
        #expect(client.requests[2].toolChoice == .function(name: "fetch_page"))
        #expect(model.outputGrounding == AppAnswerGrounding(webSearches: 1))
        #expect(!model.outputLacksCitation)
        #expect(!model.canSearchAgain)
        #expect(model.outputVariants.map(\.text) == ["知りません"])
        // The switch itself is untouched.
        #expect(model.networkMode == .offline)
    }

    @MainActor
    @Test func searchAgainNeedsAKeyAndAWebStepFreeAnswer() async throws {
        let client = ScriptedToolClient([.answer("A")])
        let model = readyModel(client: client)
        model.webSearchConfiguration = WebSearchConfiguration()
        model.promptText = "q"
        model.run()
        await waitForIdle(model)
        #expect(model.canRegenerate)
        #expect(!model.canSearchAgain)
        model.regenerate(.searched)
        #expect(!model.isRunning)
        #expect(client.requests.count == 1)
    }

    @MainActor
    @Test func nothingToRegenerateBeforeAnAnswer() {
        let model = readyModel(client: ScriptedToolClient([]))
        #expect(!model.canRegenerate)
        #expect(!model.canAskFollowUp)
        model.regenerate(.again)
        #expect(!model.isRunning)
    }

    @MainActor
    @Test func thePersonaLeadsTheSystemPromptWithAndWithoutTools() throws {
        let model = readyModel(client: ScriptedToolClient([]))
        model.promptText = "q"
        model.persona = AppPersona()
        #expect(try model.makeRequest().systemPrompt == nil)

        model.persona = AppPersona(identity: "あなたは Tsugumi。", aboutUser: "京都在住", answerStyle: "短く")
        let offline = try model.makeRequest()
        #expect(offline.tools.isEmpty)
        #expect(offline.systemPrompt == "あなたは Tsugumi。\n\n# 質問している人\n京都在住\n\n# 答え方\n短く")

        model.networkMode = .online
        let online = try model.makeRequest()
        #expect(online.systemPrompt?.hasPrefix("あなたは Tsugumi。\n\n# 質問している人\n京都在住\n\n# 答え方\n短く\n\n") == true)
        #expect(online.systemPrompt?.contains("web_search") == true)
    }

    @MainActor
    @Test func thePersonaIsSavedAndReloaded() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("persona-\(UUID().uuidString)", isDirectory: true)
        let url = directory.appendingPathComponent("persona.json")
        defer { try? FileManager.default.removeItem(at: directory) }

        let fresh = AppModel(client: ScriptedToolClient([]), personaURL: url)
        #expect(fresh.persona == .defaults)
        fresh.persona.aboutUser = "キャンプが趣味"
        fresh.persona.identity = ""
        fresh.savePersona()

        let reloaded = AppModel(client: ScriptedToolClient([]), personaURL: url)
        #expect(reloaded.persona == AppPersona(identity: "", aboutUser: "キャンプが趣味"))
        #expect(reloaded.persona.promptSection == "# 質問している人\nキャンプが趣味")
    }

    @MainActor
    @Test func contextUsageComesFromTheLastRound() async {
        let model = readyModel(client: ScriptedToolClient([.answer("A")]))
        #expect(model.contextUsedTokens == nil)
        model.promptText = "q"
        model.run()
        await waitForIdle(model)
        // The scripted client reports 1 prompt token and 1 generated token.
        #expect(model.contextUsedTokens == 2)
    }

    @MainActor
    @Test func variantsAndTheDirectiveSurviveARestart() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("chats-\(UUID().uuidString)", isDirectory: true)
        let store = AppChatStore(fileURL: directory.appendingPathComponent("chats.json"))
        defer { try? FileManager.default.removeItem(at: directory) }

        let model = readyModel(client: ScriptedToolClient([.answer("A"), .answer("B")]), chatStore: store)
        model.promptText = "q"
        model.run()
        await waitForIdle(model)
        model.regenerate(.concise)
        await waitForIdle(model)

        let restored = AppModel(client: ScriptedToolClient([]), chatStore: store)
        #expect(restored.outputResponsePlainText == "B")
        #expect(restored.outputDirective == .concise)
        #expect(restored.outputVariants.map(\.text) == ["A"])
        #expect(restored.selectedVariantIndex == 1)
        #expect(restored.answerCount == 2)
    }

    @Test func groundingCountsFinishedStepsByFamily() {
        let trace = [
            AppToolTraceEntry(id: "1", name: "wikipedia_lookup", subject: "淀城", status: .done),
            AppToolTraceEntry(id: "2", name: "web_search", subject: "q", status: .done),
            AppToolTraceEntry(id: "3", name: "fetch_page", subject: "u", status: .failed),
            AppToolTraceEntry(id: "4", name: "wikipedia_page", subject: "淀城", status: .done),
        ]
        #expect(AppAnswerGrounding.of(trace) == AppAnswerGrounding(webSearches: 1, pagesRead: 0, wikipediaSteps: 2))
        #expect(AppAnswerGrounding.of([]).isEmpty)
        let read = trace + [AppToolTraceEntry(id: "5", name: "fetch_page", subject: "u2", status: .done)]
        #expect(AppAnswerGrounding.of(read).pagesRead == 1)
        #expect(AppAnswerGrounding.of(read).webSteps == 2)
    }

    @Test func aCitationIsAReferenceLineOrAURL() {
        #expect(AppAnswerGrounding.citesSources("晴れです。\n\n参照: 天気予報"))
        #expect(AppAnswerGrounding.citesSources("晴れです。\n参照：https://example.jp"))
        #expect(AppAnswerGrounding.citesSources("詳しくは https://example.jp/w を。"))
        #expect(!AppAnswerGrounding.citesSources("プランA: 豚バラとレンコンの照り炒め"))
    }

    @MainActor
    @Test func anAnswerThatSearchedButNamesNoSourceLacksACitation() async throws {
        let client = ScriptedToolClient([.calls([searchCall]), .answer("プランA: 照り炒め")])
        let executor = ScriptedToolExecutor(results: [
            "call-1": AppToolResult(content: "検索: 淀城 遺構 (Serper, 1 件)", summary: "Serper · 1 hits"),
        ])
        let model = readyModel(client: client, executor: executor)
        model.networkMode = .online
        model.promptText = "q"
        model.run()
        await waitForIdle(model)
        #expect(model.outputGrounding.webSearches == 1)
        #expect(model.outputLacksCitation)
    }

    @Test func searchAgainAsksToReadPagesAndCiteThem() {
        let line = AppAnswerDirective.searched.instruction
        #expect(line.contains("fetch_page"))
        #expect(line.contains("参照:"))
    }

    @Test func directivesAppendOneLineExceptAgain() {
        #expect(AppAnswerDirective.again.apply(to: "q") == "q")
        #expect(AppAnswerDirective.blunt.apply(to: "q") == "q\n\n" + AppAnswerDirective.blunt.instruction)
        for directive in AppAnswerDirective.allCases where directive != .again {
            #expect(!directive.instruction.isEmpty)
        }
    }

    @MainActor
    private func readyModel(client: any AppInferenceClient,
                            executor: ScriptedToolExecutor = ScriptedToolExecutor(results: [:]),
                            chatStore: AppChatStore? = nil) -> AppModel {
        let model = AppModel(client: client, chatStore: chatStore,
                             toolExecutorProvider: { _, mode in mode == .online ? executor : nil })
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        var configuration = WebSearchConfiguration()
        configuration.serperAPIKey = "test"
        model.webSearchConfiguration = configuration
        model.persona = AppPersona()
        return model
    }

    @MainActor
    private func waitForIdle(_ model: AppModel) async {
        for _ in 0..<400 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }
}
