import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppChatStoreTests {
    @MainActor
    @Test func chatsSurviveARestartWithSelectionAndDrafts() async throws {
        let store = makeStore()
        defer { removeStore(store) }
        let client = MockInferenceClient(response: "alpha", tokenDelayNanos: 1)
        let first = readyModel(client: client, chatStore: store)
        first.promptText = "question one"
        first.run()
        await waitForIdle(first)
        let firstChatID = first.selectedChatID
        first.newChat()
        client.response = "beta"
        first.promptText = "question two"
        first.run()
        await waitForIdle(first)
        first.selectChat(firstChatID)

        let second = readyModel(
            client: MockInferenceClient(), chatStore: store)

        #expect(second.chats.count == 2)
        #expect(second.selectedChatID == second.chats[0].id)
        #expect(second.outputPromptText == "question one")
        #expect(second.outputResponsePlainText.contains("alpha"))
        #expect(second.chats[1].outputPromptText == "question two")
        #expect(second.chats[1].outputText.contains("beta"))
    }

    @MainActor
    @Test func restoredHistoryFoldsAndResendsOnTheNextRun() async throws {
        let store = makeStore()
        defer { removeStore(store) }
        let client = MockInferenceClient(response: "first answer", tokenDelayNanos: 1)
        let first = readyModel(client: client, chatStore: store)
        first.promptText = "first prompt"
        first.run()
        await waitForIdle(first)
        let firstAnswer = first.outputResponsePlainText

        let secondClient = MockInferenceClient(
            response: "second answer", tokenDelayNanos: 1)
        let second = readyModel(client: secondClient, chatStore: store)
        second.promptText = "second prompt"
        second.run()
        await waitForIdle(second)

        let request = try #require(secondClient.capturedRequests.last)
        #expect(request.history == [
            AppChatTurn(role: .user, text: "first prompt"),
            AppChatTurn(role: .assistant, text: firstAnswer),
        ])
        #expect(request.prompt == "second prompt")
    }

    @MainActor
    @Test func draftEditsPersistAfterTheDebounce() async throws {
        let store = makeStore()
        defer { removeStore(store) }
        let saved = AppModel.chatSaveDebounceNanos
        AppModel.chatSaveDebounceNanos = 1_000_000
        defer { AppModel.chatSaveDebounceNanos = saved }

        let first = readyModel(client: MockInferenceClient(), chatStore: store)
        first.promptText = "unsent draft"
        for _ in 0..<200 where store.load() == nil {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }

        let persisted = try #require(store.load())
        #expect(persisted.chats.first?.promptText == "unsent draft")
    }

    @MainActor
    @Test func restoredChatIsNotMaskedByTheEmptyTranscriptMailbox() async throws {
        let store = makeStore()
        defer { removeStore(store) }
        let client = MockInferenceClient(response: "kept answer", tokenDelayNanos: 1)
        let first = readyModel(client: client, chatStore: store)
        first.promptText = "keep me"
        first.run()
        await waitForIdle(first)

        let second = readyModel(
            client: ReportingChatClient(), chatStore: store)

        #expect(second.selectedChatTranscriptMailbox == nil)
        #expect(second.outputResponsePlainText.contains("kept answer"))
    }

    @MainActor
    @Test func vanishedImagePathsAreDroppedOnLoad() throws {
        let store = makeStore()
        defer { removeStore(store) }
        try store.save(PersistedChats(chats: [PersistedChat(
            attachedImagePaths: ["/nonexistent/pending.png"],
            turns: [
                AppChatTurn(role: .user, text: "look",
                            imagePaths: ["/nonexistent/gone.png"]),
                AppChatTurn(role: .assistant, text: "seen"),
            ])]))

        let model = readyModel(client: MockInferenceClient(), chatStore: store)

        #expect(model.attachedImagePaths.isEmpty)
        #expect(model.conversationTurns[0].imagePaths.isEmpty)
        #expect(model.conversationTurns[1].text == "seen")
    }

    @MainActor
    @Test func corruptStoreFallsBackToAFreshChat() throws {
        let store = makeStore()
        defer { removeStore(store) }
        try FileManager.default.createDirectory(
            at: store.fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try Data("not json".utf8).write(to: store.fileURL)

        let model = readyModel(client: MockInferenceClient(), chatStore: store)

        #expect(model.chats.count == 1)
        #expect(!model.hasOutputTranscript)
        #expect(!FileManager.default.fileExists(atPath: store.fileURL.path))
    }

    @MainActor
    @Test func deletingAChatPersistsImmediately() async throws {
        let store = makeStore()
        defer { removeStore(store) }
        let client = MockInferenceClient(response: "alpha", tokenDelayNanos: 1)
        let model = readyModel(client: client, chatStore: store)
        model.promptText = "one"
        model.run()
        await waitForIdle(model)
        model.newChat()
        model.deleteChat(model.selectedChatID)

        let persisted = try #require(store.load())
        #expect(persisted.chats.count == 1)
        #expect(persisted.chats[0].outputPromptText == "one")
    }

    private func makeStore() -> AppChatStore {
        AppChatStore(fileURL: FileManager.default.temporaryDirectory
            .appendingPathComponent("chat-store-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("chats.json", isDirectory: false))
    }

    private func removeStore(_ store: AppChatStore) {
        try? FileManager.default.removeItem(
            at: store.fileURL.deletingLastPathComponent())
    }

    @MainActor
    private func readyModel(client: any AppInferenceClient,
                            chatStore: AppChatStore) -> AppModel {
        let model = AppModel(client: client, chatStore: chatStore)
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

private final class ReportingChatClient: AppInferenceClient,
    AppInferenceTranscriptReporting, @unchecked Sendable {
    let generationTranscriptMailbox = GenerationTranscriptMailbox()

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in continuation.finish() }
    }

    func cancel() {}
}
