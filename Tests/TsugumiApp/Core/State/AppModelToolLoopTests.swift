import Foundation
import Testing
@testable import TsugumiAppCore

@Suite struct AppModelToolLoopTests {
    let searchCall = AppToolCall(id: "call-1", name: "web_search",
                                 argumentsJSON: #"{"query":"東京 天気"}"#)
    let fetchCall = AppToolCall(id: "call-2", name: "fetch_page",
                                argumentsJSON: #"{"url":"https://example.jp/w"}"#)

    @MainActor
    @Test func offModeDeclaresNothing() throws {
        let model = readyModel(client: ScriptedToolClient([]), executor: ScriptedToolExecutor(results: [:]))
        model.promptText = "hi"
        let request = try model.makeRequest()
        #expect(request.tools.isEmpty)
        #expect(request.systemPrompt == nil)
        #expect(request.toolChoice == .auto)
    }

    @MainActor
    @Test func onlineDeclaresTheToolsAndForcesTheSearch() throws {
        let model = readyModel(client: ScriptedToolClient([]), executor: ScriptedToolExecutor(results: [:]))
        model.promptText = "hi"
        model.networkMode = .online
        let online = try model.makeRequest()
        #expect(online.tools.map(\.name) == ["web_search", "fetch_page"])
        #expect(online.toolChoice == .function(name: "web_search"))
        #expect(online.systemPrompt?.contains("web_search") == true)
        // Offline the provider has nothing to declare here (no index in this test).
        model.networkMode = .offline
        #expect(try model.makeRequest().tools.isEmpty)
    }

    @MainActor
    @Test func runRefusesWithoutAKey() {
        let model = readyModel(client: ScriptedToolClient([]), executor: ScriptedToolExecutor(results: [:]))
        model.webSearchConfiguration = WebSearchConfiguration()
        model.networkMode = .online
        model.promptText = "hi"
        model.run()
        #expect(!model.isRunning)
        #expect(model.error?.userMessage.contains("API") == true)
    }

    @MainActor
    @Test func twoRoundsOfToolsThenTheAnswerFoldIntoHistory() async throws {
        let client = ScriptedToolClient([
            .calls([searchCall], reasoning: "調べよう"),
            .calls([fetchCall]),
            .answer("晴れです。\n参照: https://example.jp/w"),
        ])
        let executor = ScriptedToolExecutor(results: [
            "call-1": AppToolResult(content: "検索: 東京 天気 (Serper, 1 件)\n[1] 天気\n    https://example.jp/w",
                                    summary: "Serper · 1 hits"),
            "call-2": AppToolResult(content: "URL: https://example.jp/w\n\n晴れ", summary: "Jina Reader · 2 chars"),
        ])
        let model = readyModel(client: client, executor: executor)
        model.networkMode = .online
        model.promptText = "東京の天気は?"
        model.run()
        await waitForIdle(model)

        #expect(model.error == nil)
        #expect(client.requests.count == 3)
        // Round 2 resends the call the model made and its result.
        let second = client.requests[1]
        #expect(second.prompt == "東京の天気は?")
        #expect(second.continuation.count == 2)
        #expect(second.continuation[0].role == .assistant)
        #expect(second.continuation[0].toolCalls == [searchCall])
        #expect(second.continuation[0].reasoningText == "調べよう")
        #expect(second.continuation[1].role == .tool)
        #expect(second.continuation[1].toolCallID == "call-1")
        #expect(second.continuation[1].toolName == "web_search")
        #expect(second.continuation[1].text.contains("https://example.jp/w"))
        // Searched, nothing read yet: the second round must fetch. All the
        // tools stay declared so the prompt prefix (and its cache) holds.
        #expect(second.toolChoice == .function(name: "fetch_page"))
        #expect(second.tools.map(\.name) == ["web_search", "fetch_page"])
        // Round 3 has both rounds and, a page read, the model's choice.
        #expect(client.requests[2].continuation.count == 4)
        #expect(client.requests[2].toolChoice == .auto)
        #expect(client.requests[2].tools.count == 2)
        #expect(client.requests[0].toolChoice == .function(name: "web_search"))
        #expect(executor.calls == [searchCall, fetchCall])

        // The live turn shows the answer and the trace.
        #expect(model.outputResponsePlainText.contains("晴れです"))
        #expect(model.outputToolTrace.map(\.subject) == ["東京 天気", "https://example.jp/w"])
        #expect(model.outputToolTrace.allSatisfy { $0.status == .done })
        #expect(model.outputToolTrace[0].summary == "Serper · 1 hits")
        #expect(model.outputContinuationTurns.count == 4)

        // The next run folds question, rounds, answer into history in order
        // (the new turn itself stays live until the run after it).
        model.promptText = "ありがとう"
        model.networkMode = .offline
        model.run()
        await waitForIdle(model)
        let roles = model.conversationTurns.map(\.role)
        #expect(roles == [.user, .assistant, .tool, .assistant, .tool, .assistant])
        #expect(model.conversationTurns[5].text.contains("晴れです"))
        #expect(model.conversationTurns[5].toolCalls.isEmpty)
        #expect(model.outputContinuationTurns.isEmpty)
        #expect(model.outputToolTrace.isEmpty)
        let last = try #require(client.requests.last)
        #expect(last.history.count == 6)
        #expect(last.tools.isEmpty)
        #expect(!model.outputConversationPlainText.contains("Answer:\n\nAnswer"))
    }

    @MainActor
    @Test func theAppsOwnLookupSeedsTheFirstRound() async throws {
        let seeds = [
            AppToolLookup(
                call: AppToolCall(id: "x", name: "fetch_page", argumentsJSON: #"{"url":"https://example.jp/yodo"}"#),
                result: AppToolResult(content: "URL: https://example.jp/yodo\n\n淀城の遺構が…", summary: "Jina Reader · 9 chars"),
                subject: "https://example.jp/yodo"),
            AppToolLookup(
                call: AppToolCall(id: "y", name: "wikipedia_lookup", argumentsJSON: #"{"titles":["淀城"]}"#),
                result: AppToolResult(content: "参考: …\n\n■ 淀城\n淀城は…", summary: "Wikipedia · 1 件"),
                subject: "淀城"),
        ]
        let client = ScriptedToolClient([.answer("本当です。\n参照: 淀城"), .answer("はい")])
        let executor = ScriptedToolExecutor(results: [:], seeds: seeds)
        let model = readyModel(client: client, executor: executor)
        model.networkMode = .online
        model.promptText = "淀城の遺構が見つかったって本当？"
        model.run()
        await waitForIdle(model)

        #expect(model.error == nil)
        #expect(executor.lookups == ["淀城の遺構が見つかったって本当？"])
        #expect(executor.calls.isEmpty)
        // The one generation already carries the lookup as a finished round.
        #expect(client.requests.count == 1)
        let first = try #require(client.requests.first)
        // One assistant turn carrying both calls, then a result per call.
        try #require(first.continuation.count == 3)
        #expect(first.continuation[0].role == .assistant)
        #expect(first.continuation[0].toolCalls.map(\.name) == ["fetch_page", "wikipedia_lookup"])
        let ids = first.continuation[0].toolCalls.map(\.id)
        #expect(ids.allSatisfy { $0.hasPrefix("lookup-") } && ids[0] != ids[1])
        #expect(first.continuation[1].role == .tool && first.continuation[1].toolCallID == ids[0])
        #expect(first.continuation[1].text.contains("淀城の遺構が"))
        #expect(first.continuation[2].role == .tool && first.continuation[2].toolCallID == ids[1])
        #expect(first.continuation[2].text.contains("■ 淀城"))
        // The seed is not a round: the model's first call is still its own choice.
        #expect(first.toolChoice == .auto)
        #expect(model.outputToolTrace.map(\.subject) == ["https://example.jp/yodo", "淀城"])
        #expect(model.outputToolTrace.allSatisfy { $0.status == .done })
        #expect(model.outputToolTrace.map(\.summary) == ["Jina Reader · 9 chars", "Wikipedia · 1 件"])
        #expect(model.outputResponsePlainText.contains("本当です"))

        // The lookup folds into history with the rest of the turn.
        model.promptText = "次"
        model.run()
        await waitForIdle(model)
        try #require(client.requests.count == 2)
        let history = client.requests[1].history
        #expect(history.map(\.role) == [.user, .assistant, .tool, .tool, .assistant])
        #expect(history.dropFirst().first?.toolCalls.map(\.name) == ["fetch_page", "wikipedia_lookup"])
    }

    @MainActor
    @Test func nothingToLookUpStartsTheFirstRoundBare() async throws {
        let client = ScriptedToolClient([.answer("ok")])
        let executor = ScriptedToolExecutor(results: [:])
        let model = readyModel(client: client, executor: executor)
        model.networkMode = .online
        model.promptText = "hi"
        model.run()
        await waitForIdle(model)
        #expect(executor.lookups == ["hi"])
        #expect(client.requests.count == 1)
        #expect(client.requests[0].continuation.isEmpty)
        #expect(model.outputToolTrace.isEmpty)
    }

    @MainActor
    @Test func exhaustedRoundsWithdrawTheTools() async throws {
        let calls = (1...3).map {
            AppToolCall(id: "c\($0)", name: "web_search", argumentsJSON: #"{"query":"q\#($0)"}"#)
        }
        let client = ScriptedToolClient([
            .calls([calls[0]]), .calls([calls[1]]), .calls([calls[2]]), .answer("done"),
        ])
        let executor = ScriptedToolExecutor(results: Dictionary(
            uniqueKeysWithValues: calls.map { ($0.id, AppToolResult(content: "r", summary: "s")) }))
        let model = readyModel(client: client, executor: executor)
        model.webSearchConfiguration.maxToolRounds = 2
        model.networkMode = .online
        model.promptText = "q"
        model.run()
        await waitForIdle(model)

        #expect(client.requests.count == 4)
        #expect(client.requests[0].toolChoice == .function(name: "web_search"))
        #expect(client.requests[1].toolChoice == .function(name: "fetch_page"))
        // Two rounds used: the third request withdraws the tools.
        #expect(client.requests[2].tools.isEmpty)
        #expect(client.requests[2].toolChoice == .none)
        // The script still emitted a call (a model can), and the loop ran it
        // rather than dropping it on the floor; the fourth request answers.
        #expect(client.requests[3].tools.isEmpty)
        #expect(model.outputResponsePlainText == "done")
        #expect(model.error == nil)
    }

    @MainActor
    @Test func failedToolResultReachesTheModelAndTheTrace() async throws {
        let client = ScriptedToolClient([.calls([searchCall]), .answer("分かりません")])
        let executor = ScriptedToolExecutor(results: [
            "call-1": AppToolResult(content: "error: every search provider failed", isError: true,
                                    summary: "Serper: HTTP 429"),
        ])
        let model = readyModel(client: client, executor: executor)
        model.networkMode = .online
        model.promptText = "q"
        model.run()
        await waitForIdle(model)
        #expect(client.requests[1].continuation[1].text.hasPrefix("error:"))
        #expect(model.outputToolTrace[0].status == .failed)
        #expect(model.outputToolTrace[0].summary == "Serper: HTTP 429")
        #expect(model.outputResponsePlainText == "分かりません")
    }

    @MainActor
    @Test func cancellingDuringAToolStopsTheLoop() async throws {
        let client = ScriptedToolClient([.calls([searchCall]), .answer("never")])
        let executor = ScriptedToolExecutor(
            results: ["call-1": AppToolResult(content: "r", summary: "s")],
            delayNanos: 300_000_000)
        let model = readyModel(client: client, executor: executor)
        model.networkMode = .online
        model.promptText = "q"
        model.run()
        // The app's own lookup is a tools phase too, before any generation;
        // the one to cancel in is the model's, after its first round.
        for _ in 0..<200 where !(model.phase == .tools && client.requests.count == 1) {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        #expect(model.phase == .tools)
        #expect(model.presentation.label == AppLocalization.string("Searching the web"))
        model.cancel()
        #expect(!model.isRunning)
        #expect(model.error == .cancelled)
        #expect(model.outputToolTrace[0].status == .failed)
        #expect(model.outputToolTrace[0].summary == "cancelled")
        try? await Task.sleep(nanoseconds: 400_000_000)
        // The scripted answer round never started.
        #expect(client.requests.count == 1)
        #expect(!model.isRunning)
    }

    @MainActor
    @Test func ornithIgnoresTheMode() throws {
        let model = readyModel(client: ScriptedToolClient([]), executor: ScriptedToolExecutor(results: [:]))
        model.selectModel(.ornith)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.networkMode = .online
        model.promptText = "q"
        #expect(!model.toolsAvailable)
        #expect(model.effectiveNetworkMode == .offline)
        #expect(try model.makeRequest().tools.isEmpty)
    }

    @MainActor
    @Test func aPageTheAppPrefetchedSatisfiesTheOnlinePolicy() async throws {
        let seed = AppToolLookup(
            call: AppToolCall(id: "x", name: "fetch_page", argumentsJSON: #"{"url":"https://example.jp/p"}"#),
            result: AppToolResult(content: "URL: https://example.jp/p\n\n本文", summary: "direct · 2 chars"),
            subject: "https://example.jp/p")
        let client = ScriptedToolClient([.answer("読みました。\n参照: https://example.jp/p")])
        let executor = ScriptedToolExecutor(results: [:], seeds: [seed])
        let model = readyModel(client: client, executor: executor)
        model.networkMode = .online
        model.promptText = "https://example.jp/p 何これ"
        model.run()
        await waitForIdle(model)
        let request = try #require(client.requests.first)
        #expect(request.toolChoice == .auto)
        #expect(request.tools.count == 2)
    }

    @MainActor
    @Test func aFailedFetchCountsAsTriedSoTheBudgetIsNotEaten() async throws {
        let client = ScriptedToolClient([
            .calls([searchCall]),
            .calls([fetchCall]),
            .answer("取れませんでした"),
        ])
        let executor = ScriptedToolExecutor(results: [
            "call-1": AppToolResult(content: "検索: 東京 天気 (Serper, 1 件)", summary: "Serper · 1 hits"),
            "call-2": AppToolResult(content: "error: 503", isError: true, summary: "503"),
        ])
        let model = readyModel(client: client, executor: executor)
        model.networkMode = .online
        model.promptText = "東京の天気は?"
        model.run()
        await waitForIdle(model)
        #expect(client.requests.map(\.toolChoice) == [
            .function(name: "web_search"), .function(name: "fetch_page"), .auto])
        #expect(client.requests[2].tools.count == 2)
    }

    @MainActor
    @Test func aStructuredOutputFailureRunsTheRoundOnceMore() async throws {
        let client = ScriptedToolClient([
            .calls([searchCall]),
            .calls([fetchCall]),
            .fail("structured_output_failure kind=decoder_consume cause=malformed rendered_prompt_tokens=4582"),
            .answer("読みました。\n参照: https://example.jp/w"),
        ])
        let executor = ScriptedToolExecutor(results: [
            "call-1": AppToolResult(content: "検索: 東京 天気 (Serper, 1 件)", summary: "Serper · 1 hits"),
            "call-2": AppToolResult(content: "URL: https://example.jp/w\n\n晴れ", summary: "direct · 2 chars"),
        ])
        let model = readyModel(client: client, executor: executor)
        model.networkMode = .online
        model.promptText = "東京の天気は?"
        model.run()
        await waitForIdle(model)
        #expect(model.error == nil)
        #expect(client.requests.count == 4)
        #expect(client.requests[2] == client.requests[3])
        #expect(model.outputResponsePlainText.contains("読みました"))
        #expect(model.outputToolTrace.count == 2)
    }

    @MainActor
    @Test func aSecondStructuredOutputFailureFailsTheTurn() async throws {
        let client = ScriptedToolClient([
            .fail("structured_output_failure kind=decoder_consume cause=malformed"),
            .fail("structured_output_failure kind=decoder_finish cause=malformed"),
            .answer("never"),
        ])
        let model = readyModel(client: client, executor: ScriptedToolExecutor(results: [:]))
        model.networkMode = .online
        model.promptText = "q"
        model.run()
        await waitForIdle(model)
        #expect(client.requests.count == 2)
        #expect(model.error?.userMessage.contains("structured_output_failure") == true)
        #expect(model.outputResponsePlainText.isEmpty)
    }

    @MainActor
    @Test func otherFailuresAreNotRetried() async throws {
        let client = ScriptedToolClient([.fail("decode service failed"), .answer("never")])
        let model = readyModel(client: client, executor: ScriptedToolExecutor(results: [:]))
        model.networkMode = .online
        model.promptText = "q"
        model.run()
        await waitForIdle(model)
        #expect(client.requests.count == 1)
        #expect(model.error != nil)
    }

    @MainActor
    @Test func offlineLeavesTheChoiceToTheModel() async throws {
        let wiki = AppToolCall(id: "w1", name: "wikipedia_search", argumentsJSON: #"{"query":"淀城"}"#)
        let client = ScriptedToolClient([.calls([wiki]), .answer("淀城です")])
        let executor = ScriptedToolExecutor(results: ["w1": AppToolResult(content: "hit", summary: "1")])
        let model = AppModel(client: client, toolExecutorProvider: { _, _ in executor })
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.networkMode = .offline
        model.promptText = "淀城"
        model.run()
        await waitForIdle(model)
        #expect(client.requests.map(\.toolChoice) == [.auto, .auto])
    }

    @MainActor
    private func readyModel(client: any AppInferenceClient,
                            executor: ScriptedToolExecutor) -> AppModel {
        let model = AppModel(client: client, toolExecutorProvider: { _, mode in mode == .online ? executor : nil })
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        var configuration = WebSearchConfiguration()
        configuration.serperAPIKey = "test"
        model.webSearchConfiguration = configuration
        return model
    }

    @MainActor
    private func waitForIdle(_ model: AppModel) async {
        for _ in 0..<400 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }
}

@Suite struct AppModelFirstRoundThinkingTests {
    @Test func noToolsAndThinkingOffAreUntouched() {
        #expect(AppModel.firstRoundThinking(tools: false, thinking: true, budget: 512) == (true, -1))
        #expect(AppModel.firstRoundThinking(tools: true, thinking: false, budget: 512) == (false, -1))
    }

    @Test func toolsBoundTheFirstRound() {
        #expect(AppModel.firstRoundThinking(tools: true, thinking: true, budget: 512) == (true, 512))
        #expect(AppModel.firstRoundThinking(tools: true, thinking: true, budget: 0) == (false, -1))
        #expect(AppModel.firstRoundThinking(tools: true, thinking: true, budget: -1) == (true, -1))
    }

    @MainActor
    @Test func onlyTheFirstRoundCarriesTheBudget() async throws {
        let call = AppToolCall(id: "c1", name: "web_search", argumentsJSON: #"{"query":"q"}"#)
        let client = ScriptedToolClient([.calls([call]), .answer("ok")])
        let executor = ScriptedToolExecutor(results: ["c1": AppToolResult(content: "r", summary: "s")])
        let model = AppModel(client: client, toolExecutorProvider: { _, mode in mode == .online ? executor : nil })
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.webSearchConfiguration.serperAPIKey = "k"
        model.webSearchConfiguration.preSearchThinkingBudget = 300
        model.thinkingEnabled = true
        model.networkMode = .online
        model.promptText = "q"
        model.run()
        for _ in 0..<400 where model.isRunning { try? await Task.sleep(nanoseconds: 5_000_000) }
        #expect(client.requests.count == 2)
        #expect(client.requests[0].enableThinking)
        #expect(client.requests[0].reasoningBudgetTokens == 300)
        #expect(client.requests[1].enableThinking)
        #expect(client.requests[1].reasoningBudgetTokens == -1)
    }
}
