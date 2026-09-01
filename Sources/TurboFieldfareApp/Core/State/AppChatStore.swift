import Foundation

/// On-disk persistence for the chat list. One JSON file for the whole app —
/// chats are not tied to a model (a conversation started under one
/// checkpoint can be continued under the other), so this deliberately does
/// not follow the per-model settings files.
public struct AppChatStore: Sendable {
    public let fileURL: URL

    public init(fileURL: URL) {
        self.fileURL = fileURL
    }

    /// `~/Library/Application Support/TurboFieldfare/chats.json`.
    public static var defaultStore: AppChatStore {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return AppChatStore(fileURL: base
            .appendingPathComponent("TurboFieldfare", isDirectory: true)
            .appendingPathComponent("chats.json", isDirectory: false))
    }

    /// A corrupt or incompatible file is discarded rather than surfaced:
    /// losing saved chats beats an app that cannot start.
    func load(fileManager: FileManager = .default) -> PersistedChats? {
        guard fileManager.fileExists(atPath: fileURL.path) else { return nil }
        do {
            let data = try Data(contentsOf: fileURL)
            let persisted = try JSONDecoder().decode(PersistedChats.self, from: data)
            guard persisted.version == PersistedChats.currentVersion else {
                throw InvalidChats()
            }
            return persisted
        } catch {
            try? fileManager.removeItem(at: fileURL)
            return nil
        }
    }

    func save(_ chats: PersistedChats,
              fileManager: FileManager = .default) throws {
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(chats)
        data.append(0x0A)
        try data.write(to: fileURL, options: .atomic)
    }

    private struct InvalidChats: Error {}
}

struct PersistedChats: Codable, Equatable {
    static let currentVersion = 1

    var version: Int = currentVersion
    /// Selection travels as an index: session identity is per-process.
    var selectedChatIndex: Int = 0
    var chats: [PersistedChat] = []
}

struct PersistedChat: Codable, Equatable {
    var promptText: String = ""
    var attachedImagePaths: [String] = []
    var turns: [AppChatTurn] = []
    var outputPromptText: String = ""
    var outputImagePaths: [String] = []
    var outputText: String = ""
    var outputReasoningText: String = ""
}

extension PersistedChat {
    @MainActor
    init(_ session: AppChatSession) {
        self.init(
            promptText: session.promptText,
            attachedImagePaths: session.attachedImagePaths,
            turns: session.conversationTurns,
            outputPromptText: session.outputPromptText,
            outputImagePaths: session.outputImagePaths,
            outputText: session.outputText,
            outputReasoningText: session.outputReasoningText)
    }

    /// Rebuilds a session, dropping image paths whose files are gone — a
    /// vanished history image would otherwise fail request validation on
    /// every future run of the chat, with no way to edit it out.
    @MainActor
    func makeSession(fileManager: FileManager = .default) -> AppChatSession {
        let session = AppChatSession()
        session.promptText = promptText
        session.attachedImagePaths = attachedImagePaths
            .filter { fileManager.fileExists(atPath: $0) }
        session.conversationTurns = turns.map { turn in
            var turn = turn
            turn.imagePaths = turn.imagePaths
                .filter { fileManager.fileExists(atPath: $0) }
            return turn
        }
        session.outputPromptText = outputPromptText
        session.outputImagePaths = outputImagePaths
            .filter { fileManager.fileExists(atPath: $0) }
        session.outputText = outputText
        session.outputReasoningText = outputReasoningText
        return session
    }
}
