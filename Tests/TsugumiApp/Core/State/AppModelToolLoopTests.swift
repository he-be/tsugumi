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
    @Test func autoDeclaresTheToolsAndAlwaysForcesTheFirstCall() throws {
        let model = readyModel(client: ScriptedToolClient([]), executor: ScriptedToolExecutor(results: [:]))
        model.promptText = "hi"
        model.webSearchMode = .auto
        let auto = try model.makeRequest()
        #expect(auto.tools.map(\.name) == ["web_search", "fetch_page"])
        #expect(auto.toolChoice == .auto)
        #expect(auto.systemPrompt?.contains("web_search") == true)
        model.webSearchMode = .always
        #expect(try model.makeRequest().toolChoice == .required)
    }

    @MainActor
    @Test func alwaysSkipsThinkingOnTheForcedRoundOnly() async throws {
        let client = ScriptedToolClient([.calls([searchCall]), .answer("ok")])
        let executor = ScriptedToolExecutor(results: [
            "call-1": AppToolResult(content: "r", summary: "s"),
        ])
        let model = readyModel(client: client, executor: executor)
        model.thinkingEnabled = true
        model.webSearchMode = .always
        model.promptText = "q"
        #expect(try !model.makeRequest().enableThinking)
        model.run()
        await waitForIdle(model)
        #expect(client.requests.count == 2)
        #expect(!client.requests[0].enableThinking)
        #expect(client.requests[1].enableThinking)
        model.webSearchMode = .auto
        model.promptText = "q2"
        #expect(try model.makeRequest().enableThinking)
    }

    @MainActor
    @Test func runRefusesWithoutAKey() {
        let model = readyModel(client: ScriptedToolClient([]), executor: ScriptedToolExecutor(results: [:]))
        model.webSearchConfiguration = WebSearchConfiguration()
        model.webSearchMode = .auto
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
        model.webSearchMode = .auto
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
        #expect(second.toolChoice == .auto)
        #expect(second.tools.count == 2)
        // Round 3 has both rounds.
        #expect(client.requests[2].continuation.count == 4)
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
        model.webSearchMode = .off
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
        model.webSearchMode = .always
        model.promptText = "q"
        model.run()
        await waitForIdle(model)

        #expect(client.requests.count == 4)
        #expect(client.requests[0].toolChoice == .required)
        #expect(client.requests[1].toolChoice == .auto)
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
        model.webSearchMode = .auto
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
        model.webSearchMode = .auto
        model.promptText = "q"
        model.run()
        for _ in 0..<200 where model.phase != .tools {
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
        model.webSearchMode = .always
        model.promptText = "q"
        #expect(!model.webSearchAvailable)
        #expect(model.effectiveWebSearchMode == .off)
        #expect(try model.makeRequest().tools.isEmpty)
    }

    @MainActor
    private func readyModel(client: any AppInferenceClient,
                            executor: ScriptedToolExecutor) -> AppModel {
        let model = AppModel(client: client, toolExecutorProvider: { _ in executor })
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
    @Test func offAndThinkingOffAreUntouched() {
        #expect(AppModel.firstRoundThinking(mode: .off, thinking: true, budget: 512) == (true, -1))
        #expect(AppModel.firstRoundThinking(mode: .auto, thinking: false, budget: 512) == (false, -1))
    }

    @Test func autoBoundsTheFirstRoundAndAlwaysClosesIt() {
        #expect(AppModel.firstRoundThinking(mode: .auto, thinking: true, budget: 512) == (true, 512))
        #expect(AppModel.firstRoundThinking(mode: .auto, thinking: true, budget: 0) == (false, -1))
        #expect(AppModel.firstRoundThinking(mode: .auto, thinking: true, budget: -1) == (true, -1))
        #expect(AppModel.firstRoundThinking(mode: .always, thinking: true, budget: 512) == (false, -1))
    }

    @MainActor
    @Test func onlyTheFirstRoundCarriesTheBudget() async throws {
        let call = AppToolCall(id: "c1", name: "web_search", argumentsJSON: #"{"query":"q"}"#)
        let client = ScriptedToolClient([.calls([call]), .answer("ok")])
        let executor = ScriptedToolExecutor(results: ["c1": AppToolResult(content: "r", summary: "s")])
        let model = AppModel(client: client, toolExecutorProvider: { _ in executor })
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.webSearchConfiguration.serperAPIKey = "k"
        model.webSearchConfiguration.preSearchThinkingBudget = 300
        model.thinkingEnabled = true
        model.webSearchMode = .auto
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
