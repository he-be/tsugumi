import Foundation
import Testing
@testable import TsugumiAppCore

@Suite struct AppChatSessionsTests {
    @MainActor
    @Test func newChatCreatesAndSelectsFreshChatKeepingTheOld() async throws {
        let client = MockInferenceClient(response: "alpha", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.promptText = "first"
        model.run()
        await waitForIdle(model)
        let firstID = model.selectedChatID

        model.newChat()

        #expect(model.chats.count == 2)
        #expect(model.selectedChatID != firstID)
        #expect(!model.hasOutputTranscript)
        #expect(model.promptText.isEmpty)
        #expect(model.conversationTurns.isEmpty)

        model.selectChat(firstID)
        #expect(model.outputResponsePlainText.contains("alpha"))
    }

    @MainActor
    @Test func newChatReusesAnExistingEmptyChat() {
        let model = readyModel(client: MockInferenceClient())
        let onlyID = model.selectedChatID

        model.newChat()

        #expect(model.chats.count == 1)
        #expect(model.selectedChatID == onlyID)
    }

    @MainActor
    @Test func switchingChatsKeepsEachDraftAndTranscript() async throws {
        let client = MockInferenceClient(response: "alpha", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.promptText = "first"
        model.run()
        await waitForIdle(model)
        let firstID = model.selectedChatID
        model.promptText = "draft A"

        model.newChat()
        let secondID = model.selectedChatID
        model.promptText = "draft B"

        model.selectChat(firstID)
        #expect(model.promptText == "draft A")
        #expect(model.outputPromptText == "first")
        #expect(model.outputResponsePlainText.contains("alpha"))

        model.selectChat(secondID)
        #expect(model.promptText == "draft B")
        #expect(!model.hasOutputTranscript)
    }

    @MainActor
    @Test func generationStreamsIntoItsOwnChatWhileAnotherIsSelected() async throws {
        let client = MockInferenceClient(
            response: "one two three four five",
            tokenDelayNanos: 20_000_000)
        client.prefillSteps = 0
        let model = readyModel(client: client)
        model.promptText = "slow prompt"
        model.run()
        let generatingID = model.selectedChatID

        model.newChat()
        #expect(model.selectedChatID != generatingID)
        #expect(model.generatingChatID == generatingID)
        #expect(model.isSelectedChatReadOnly)
        #expect(!model.isSelectedChatGenerating)
        model.promptText = "queued question"
        #expect(!model.canRun)
        #expect(model.outputText.isEmpty)

        let generatingChat = try #require(
            model.chats.first(where: { $0.id == generatingID }))
        for _ in 0..<200 where generatingChat.outputText.isEmpty {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        #expect(!generatingChat.outputText.isEmpty)
        #expect(model.outputText.isEmpty)

        await waitForIdle(model)
        #expect(!model.isSelectedChatReadOnly)
        #expect(model.canRun)
        model.selectChat(generatingID)
        #expect(model.outputResponsePlainText.contains("one two three four five"))
    }

    @MainActor
    @Test func historyStaysPerChat() async throws {
        let client = MockInferenceClient(response: "alpha", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.promptText = "chat A question"
        model.run()
        await waitForIdle(model)
        let firstID = model.selectedChatID

        model.newChat()
        client.response = "beta"
        model.promptText = "chat B question"
        model.run()
        await waitForIdle(model)
        #expect(client.capturedRequests.last?.history.isEmpty == true)

        model.selectChat(firstID)
        client.response = "gamma"
        model.promptText = "chat A follow-up"
        model.run()
        await waitForIdle(model)

        let request = try #require(client.capturedRequests.last)
        #expect(request.history.count == 2)
        #expect(request.history[0].text == "chat A question")
        #expect(!request.history.contains(where: { $0.text.contains("chat B") }))
    }

    @MainActor
    @Test func deletingSelectedChatFallsBackToNeighborAndNeverToZeroChats() async throws {
        let client = MockInferenceClient(response: "alpha", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.promptText = "first"
        model.run()
        await waitForIdle(model)
        let firstID = model.selectedChatID
        model.newChat()
        let secondID = model.selectedChatID

        model.deleteChat(secondID)
        #expect(model.chats.count == 1)
        #expect(model.selectedChatID == firstID)

        model.deleteChat(firstID)
        #expect(model.chats.count == 1)
        #expect(model.selectedChatID != firstID)
        #expect(!model.hasOutputTranscript)
        #expect(model.outputResponsePlainText.isEmpty)
    }

    @MainActor
    @Test func generatingChatCannotBeDeleted() async throws {
        let client = MockInferenceClient(
            response: "one two three",
            tokenDelayNanos: 20_000_000)
        let model = readyModel(client: client)
        model.promptText = "busy"
        model.run()
        let generatingID = model.selectedChatID

        #expect(!model.canDeleteChat(generatingID))
        model.deleteChat(generatingID)
        #expect(model.chats.contains(where: { $0.id == generatingID }))

        model.cancel()
        await waitForIdle(model)
        #expect(model.canDeleteChat(generatingID))
    }

    @MainActor
    @Test func serviceTranscriptStaysWithItsOwningChatAcrossSwitches() {
        let client = ChatReportingInferenceClient()
        let model = readyModel(client: client)
        client.generationTranscriptMailbox.append("owned output")
        let ownerID = model.selectedChatID
        #expect(model.outputResponsePlainText == "owned output")

        model.newChat()
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.selectedChatTranscriptMailbox == nil)
        #expect(model.outputConversationPlainText.isEmpty)

        model.selectChat(ownerID)
        #expect(model.outputResponsePlainText == "owned output")
        #expect(model.selectedChatTranscriptMailbox != nil)
    }

    @MainActor
    private func readyModel(client: any AppInferenceClient) -> AppModel {
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(
            modelDirectory: FileManager.default.temporaryDirectory,
            loadSeconds: 1)
        return model
    }

    @MainActor
    private func waitForIdle(_ model: AppModel) async {
        for _ in 0..<200 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }
}

private final class ChatReportingInferenceClient: AppInferenceClient,
    AppInferenceTranscriptReporting, @unchecked Sendable {
    let generationTranscriptMailbox = GenerationTranscriptMailbox()

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in continuation.finish() }
    }

    func cancel() {}
}
