import Foundation
import Testing
@testable import TsugumiAppCore

@Suite struct AppModelTests {
    @MainActor
    @Test func defaultsUseSampledRequest() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"

        let request = try model.makeRequest()
        #expect(request.temperature == 1.0)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.maxNewTokens == 32_768)
        #expect(request.repetitionPenalty == 1)
        #expect(!request.isPureGreedy)
        #expect(request.runtimeOptions.expertCacheSlots == 32)
        #expect(request.runtimeOptions.expertCachePolicy == .lfu)
        #expect(request.runtimeOptions.rdadvisePolicy == .off)
        #expect(request.runtimeOptions.prefillEnabled)
    }

    @MainActor
    @Test func runDisabledWhenPromptEmpty() {
        let model = AppModel()
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.promptText = "   "
        #expect(!model.canRun)
    }

    @MainActor
    @Test func runDisabledUntilModelReady() {
        let model = AppModel()
        model.promptText = "go"
        #expect(!model.canRun)
    }

    @MainActor
    @Test func disablingTopKNeutralizesBothTruncationControls() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.topKEnabled = false
        model.topPEnabled = true

        let request = try model.makeRequest()
        #expect(request.topK == nil)
        #expect(request.topP == nil)
    }

    @MainActor
    @Test func prefillToggleSurvivesRequestCreation() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"

        model.runtimeOptions.prefillEnabled = false
        #expect(try !model.makeRequest().runtimeOptions.prefillEnabled)

        model.runtimeOptions.prefillEnabled = true
        #expect(try model.makeRequest().runtimeOptions.prefillEnabled)
    }

    @MainActor
    @Test func adaptiveRDAdvicePolicySurvivesRequestCreation() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.runtimeOptions.rdadvisePolicy = .adaptive

        let request = try model.makeRequest()
        #expect(request.runtimeOptions.rdadvisePolicy == .adaptive)
    }

    @MainActor
    @Test func loadAffectingRuntimeChangeMarksReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(!model.hasStaleLoadedRuntime)
        model.runtimeOptions.rdadvisePolicy = .bounded
        #expect(model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func contextChangeMarksReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(!model.hasStaleLoadedRuntime)
        model.maxContextTokens = AppContextLengthOption.eightK.tokens
        #expect(model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func appResponseLimitUsesSelectedContext() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.maxContextTokens = AppContextLengthOption.sixtyFourK.tokens

        #expect(try model.makeRequest().maxNewTokens == AppContextLengthOption.sixtyFourK.tokens)
    }

    @MainActor
    @Test func prefillChangeMarksReadySessionStale() {
        // The family sessions bind prefill (and the MTP loop that rides on
        // it) at load, so flipping it requires a reload.
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        model.runtimeOptions.prefillEnabled = false

        #expect(model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func newlineShortcutDoesNotMarkReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        model.setNewlineShortcut(.shiftReturn)

        #expect(model.newlineShortcut == .shiftReturn)
        #expect(!model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func mockRunUpdatesOutputAndDiagnostics() async throws {
        let client = MockInferenceClient(response: "alpha beta", tokenDelayNanos: 1)
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.promptText = "go"
        model.maxNewTokensOverride = 4
        model.run()

        for _ in 0..<200 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }

        #expect(!model.isRunning)
        #expect(model.outputText.contains("alpha beta"))
        #expect(model.diagnostics != nil)
        #expect(model.error == nil)
    }

    @MainActor
    @Test func runSnapshotsPromptIntoOutputTranscript() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.promptText = "original prompt"
        model.maxNewTokensOverride = 1
        model.run()

        #expect(model.outputPromptText == "original prompt")
        #expect(model.promptText == "original prompt")
        #expect(model.hasOutputTranscript)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText == "You:\noriginal prompt")

        model.promptText = "edited prompt"
        await waitForIdle(model)

        #expect(model.outputPromptText == "original prompt")
        #expect(model.outputResponsePlainText == "answer")
        #expect(model.outputConversationPlainText
            == "You:\noriginal prompt\n\nAnswer:\nanswer")
        #expect(!model.outputConversationPlainText.contains("edited prompt"))
    }

    @MainActor
    @Test func clearAfterSendingPreservesTranscriptAndNextDraft() async throws {
        let client = MockInferenceClient(
            response: "answer",
            tokenDelayNanos: 20_000_000)
        let model = readyModel(client: client)
        model.setSentPromptBehavior(.clear)
        model.promptText = "original prompt"
        model.maxNewTokensOverride = 1

        model.run()

        #expect(model.isRunning)
        #expect(model.outputPromptText == "original prompt")
        #expect(model.promptText.isEmpty)

        model.promptText = "next draft"
        await waitForIdle(model)

        #expect(model.promptText == "next draft")
        #expect(model.outputConversationPlainText.hasPrefix(
            "You:\noriginal prompt\n\nAnswer:\n"))
    }

    @MainActor
    @Test func failedValidationDoesNotClearPrompt() {
        let model = readyModel(client: MockInferenceClient(response: "answer"))
        model.setSentPromptBehavior(.clear)
        model.promptText = "keep invalid prompt"
        model.maxNewTokensOverride = 0

        model.run()

        #expect(!model.isRunning)
        #expect(model.promptText == "keep invalid prompt")
        #expect(model.outputPromptText.isEmpty)
        #expect(model.error != nil)
    }

    @MainActor
    @Test func staleReadySessionDisablesGenerationUntilReload() throws {
        let client = MockLifecycleInferenceClient()
        let directory = try makeCompleteModelInstall("stale-runtime")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory, client: client)
        model.promptText = "go"
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(model.canRun)
        model.runtimeOptions.rdadvisePolicy = .bounded
        #expect(model.hasStaleLoadedRuntime)
        #expect(!model.canRun)
        #expect(model.canReloadModel)
        #expect(client.ensureLoadedCallCount() == 0)
    }

    @MainActor
    @Test func cancelAfterPartialOutputCanBeCleared() async throws {
        let client = MockInferenceClient(response: "one two three four five", tokenDelayNanos: 20_000_000)
        client.prefillSteps = 0
        let model = readyModel(client: client)
        model.promptText = "stop after token"
        model.maxNewTokensOverride = 10
        model.run()

        for _ in 0..<200 where model.liveTokenCount == 0 {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }

        #expect(model.liveTokenCount > 0)
        model.cancel()
        #expect(model.isCancellationPending)
        await waitForIdle(model)

        #expect(!model.isRunning)
        #expect(!model.isCancellationPending)
        #expect(model.error == .cancelled)
        #expect(model.hasOutputTranscript)
        #expect(!model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText.hasPrefix(
            "You:\nstop after token\n\nAnswer:\n"))

        model.clearOutput()
        #expect(!model.hasOutputTranscript)
        #expect(model.outputPromptText.isEmpty)
        #expect(model.outputText.isEmpty)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText.isEmpty)
        #expect(model.error == nil)
    }

    @MainActor
    @Test func cancelDuringPrefillKeepsPromptSnapshotUntilClear() async throws {
        let client = MockInferenceClient(response: "unused", tokenDelayNanos: 1_000_000)
        client.prefillSteps = 20
        let model = readyModel(client: client)
        model.promptText = "prefill prompt"
        model.run()

        for _ in 0..<200 where model.livePrefillDone == 0 {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }

        #expect(model.outputPromptText == "prefill prompt")
        model.cancel()
        await waitForIdle(model)

        #expect(!model.isRunning)
        #expect(model.outputPromptText == "prefill prompt")
        #expect(model.outputText.isEmpty)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText == "You:\nprefill prompt")
        #expect(model.hasOutputTranscript)

        model.clearOutput()
        #expect(!model.hasOutputTranscript)
    }

    @MainActor
    @Test func failedEventThenThrownErrorKeepsFirstTerminalState() async throws {
        let client = MockInferenceClient(tokenDelayNanos: 1, failureMessage: "synthetic failure")
        let model = readyModel(client: client)
        model.promptText = "fail"

        model.run()
        await waitForIdle(model)

        #expect(model.error?.userMessage == "synthetic failure")
        #expect(model.diagnostics?.stopReason == .failed)
    }

    @MainActor
    @Test func changingModelPathInvalidatesLoadedStateAndDiagnostics() {
        let model = AppModel(client: MockInferenceClient(),
                             installer: MockModelInstallerClient())
        let testDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("app-model-path-\(UUID().uuidString)", isDirectory: true)
        let oldURL = testDirectory.appendingPathComponent("old.moepack")
        let newURL = testDirectory.appendingPathComponent("new.moepack")
        model.modelPathText = oldURL.path
        model.loadState = .ready(modelDirectory: oldURL, loadSeconds: 1)
        model.diagnostics = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 1,
            tokensPerSecond: 1,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
        model.error = .unknown("old error")

        model.setModelURL(newURL)

        #expect(model.modelPathText == newURL.standardizedFileURL.path)
        #expect(model.loadState == .notLoaded)
        #expect(model.loadedRuntimeKey == nil)
        #expect(model.diagnostics == nil)
        #expect(model.error == nil)
        #expect(model.presentation.label == AppLocalization.string("Model required"))
        #expect(!model.canRun)
    }

    @MainActor
    @Test func secondRunFoldsCompletedTurnIntoHistory() async throws {
        let client = MockInferenceClient(response: "first answer", tokenDelayNanos: 1)
        client.reasoning = "because of the premise"
        let model = readyModel(client: client)
        model.promptText = "first prompt"
        model.run()
        await waitForIdle(model)

        #expect(model.conversationTurns.isEmpty)
        let firstAnswer = model.outputResponsePlainText
        #expect(firstAnswer.contains("first answer"))

        client.response = "second answer"
        client.reasoning = nil
        model.promptText = "second prompt"
        model.run()

        #expect(model.conversationTurns.count == 2)
        #expect(model.conversationTurns[0]
            == AppChatTurn(role: .user, text: "first prompt"))
        #expect(model.conversationTurns[1]
            == AppChatTurn(role: .assistant, text: firstAnswer,
                           reasoningText: "because of the premise"))
        let foldedTurns = model.conversationTurns
        #expect(model.outputPromptText == "second prompt")
        await waitForIdle(model)
        let request = client.capturedRequests.last
        #expect(request?.history == foldedTurns)
        #expect(request?.prompt == "second prompt")
        #expect(model.outputResponsePlainText.contains("second answer"))
    }

    @MainActor
    @Test func failedTurnIsNotResentAsHistory() async throws {
        let client = MockInferenceClient(response: "unused", tokenDelayNanos: 1,
                                         failureMessage: "engine exploded")
        let model = readyModel(client: client)
        model.promptText = "doomed prompt"
        model.run()
        await waitForIdle(model)
        #expect(model.error != nil)

        client.failureMessage = nil
        client.response = "recovered answer"
        model.promptText = "retry prompt"
        model.run()

        #expect(model.conversationTurns.isEmpty)
        await waitForIdle(model)
        #expect(client.capturedRequests.last?.history.isEmpty == true)
        #expect(model.outputResponsePlainText.contains("recovered answer"))
    }

    @MainActor
    @Test func conversationPlainTextThreadsHistoryInOrder() async throws {
        let client = MockInferenceClient(response: "alpha", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.promptText = "one"
        model.run()
        await waitForIdle(model)

        client.response = "beta"
        model.promptText = "two"
        model.run()
        await waitForIdle(model)

        let text = model.outputConversationPlainText
        #expect(text.hasPrefix("You:\none\n\nAnswer:\nalpha"))
        let secondTurn = text.range(of: "You:\ntwo\n\nAnswer:\nbeta")
        #expect(secondTurn != nil)
    }

    @MainActor
    @Test func clearOutputDropsConversationHistory() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.promptText = "one"
        model.run()
        await waitForIdle(model)
        model.promptText = "two"
        model.run()
        await waitForIdle(model)
        #expect(!model.conversationTurns.isEmpty)

        model.clearOutput()

        #expect(model.conversationTurns.isEmpty)
        #expect(!model.hasOutputTranscript)
        model.promptText = "three"
        #expect(try model.makeRequest().history.isEmpty)
    }

    @MainActor
    private func readyModel(client: MockInferenceClient) -> AppModel {
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        return model
    }

    @MainActor
    private func waitForIdle(_ model: AppModel) async {
        for _ in 0..<200 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }
}
