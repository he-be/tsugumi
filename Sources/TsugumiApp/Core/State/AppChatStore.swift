import Foundation

/// On-disk persistence for the chat list. One JSON file for the whole app —
/// chats are not tied to a model (a conversation started under one
/// checkpoint can be continued under the other), so this deliberately does
/// not follow the per-model settings files.
public struct AppChatStore: Sendable {
    public let fileURL: URL
    /// Where the same file lived while the app was called TurboFieldfare.
    /// Read once, when the current path has nothing: a rename of the
    /// application must not read as "all your chats are gone". Nothing writes
    /// here, so the first save moves the chats to `fileURL` for good.
    let legacyFileURL: URL?

    public init(fileURL: URL, legacyFileURL: URL? = nil) {
        self.fileURL = fileURL
        self.legacyFileURL = legacyFileURL
    }

    /// `~/Library/Application Support/Tsugumi/chats.json`.
    public static var defaultStore: AppChatStore {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return AppChatStore(
            fileURL: base
                .appendingPathComponent("Tsugumi", isDirectory: true)
                .appendingPathComponent("chats.json", isDirectory: false),
            legacyFileURL: base
                .appendingPathComponent("TurboFieldfare", isDirectory: true)
                .appendingPathComponent("chats.json", isDirectory: false))
    }

    /// A corrupt or incompatible file is discarded rather than surfaced:
    /// losing saved chats beats an app that cannot start.
    func load(fileManager: FileManager = .default) -> PersistedChats? {
        if let persisted = load(from: fileURL, fileManager: fileManager) {
            return persisted
        }
        guard let legacyFileURL,
              !fileManager.fileExists(atPath: fileURL.path) else { return nil }
        return load(from: legacyFileURL, fileManager: fileManager)
    }

    private func load(from url: URL, fileManager: FileManager) -> PersistedChats? {
        guard fileManager.fileExists(atPath: url.path) else { return nil }
        do {
            let data = try Data(contentsOf: url)
            let persisted = try JSONDecoder().decode(PersistedChats.self, from: data)
            guard persisted.version == PersistedChats.currentVersion else {
                throw InvalidChats()
            }
            return persisted
        } catch {
            try? fileManager.removeItem(at: url)
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
    var outputContinuationTurns: [AppChatTurn] = []
    var outputToolTrace: [AppToolTraceEntry] = []
    var outputDirective: AppAnswerDirective?
    var outputNetworkMode: AppNetworkMode?
    var outputVariants: [AppAnswerVariant] = []
    var selectedVariantIndex: Int = 0

    init(promptText: String = "",
         attachedImagePaths: [String] = [],
         turns: [AppChatTurn] = [],
         outputPromptText: String = "",
         outputImagePaths: [String] = [],
         outputText: String = "",
         outputReasoningText: String = "",
         outputContinuationTurns: [AppChatTurn] = [],
         outputToolTrace: [AppToolTraceEntry] = [],
         outputDirective: AppAnswerDirective? = nil,
         outputNetworkMode: AppNetworkMode? = nil,
         outputVariants: [AppAnswerVariant] = [],
         selectedVariantIndex: Int = 0) {
        self.promptText = promptText
        self.attachedImagePaths = attachedImagePaths
        self.turns = turns
        self.outputPromptText = outputPromptText
        self.outputImagePaths = outputImagePaths
        self.outputText = outputText
        self.outputReasoningText = outputReasoningText
        self.outputContinuationTurns = outputContinuationTurns
        self.outputToolTrace = outputToolTrace
        self.outputDirective = outputDirective
        self.outputNetworkMode = outputNetworkMode
        self.outputVariants = outputVariants
        self.selectedVariantIndex = selectedVariantIndex
    }

    private enum CodingKeys: String, CodingKey {
        case promptText, attachedImagePaths, turns, outputPromptText, outputImagePaths
        case outputText, outputReasoningText, outputContinuationTurns, outputToolTrace
        case outputDirective, outputNetworkMode, outputVariants, selectedVariantIndex
    }

    /// Fields added after the first file version decode as their defaults,
    /// so a chats file from before the tool loop still restores.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        promptText = try container.decodeIfPresent(String.self, forKey: .promptText) ?? ""
        attachedImagePaths = try container.decodeIfPresent(
            [String].self, forKey: .attachedImagePaths) ?? []
        turns = try container.decodeIfPresent([AppChatTurn].self, forKey: .turns) ?? []
        outputPromptText = try container.decodeIfPresent(
            String.self, forKey: .outputPromptText) ?? ""
        outputImagePaths = try container.decodeIfPresent(
            [String].self, forKey: .outputImagePaths) ?? []
        outputText = try container.decodeIfPresent(String.self, forKey: .outputText) ?? ""
        outputReasoningText = try container.decodeIfPresent(
            String.self, forKey: .outputReasoningText) ?? ""
        outputContinuationTurns = try container.decodeIfPresent(
            [AppChatTurn].self, forKey: .outputContinuationTurns) ?? []
        outputToolTrace = try container.decodeIfPresent(
            [AppToolTraceEntry].self, forKey: .outputToolTrace) ?? []
        outputDirective = try container.decodeIfPresent(
            AppAnswerDirective.self, forKey: .outputDirective)
        outputNetworkMode = try container.decodeIfPresent(
            AppNetworkMode.self, forKey: .outputNetworkMode)
        outputVariants = try container.decodeIfPresent(
            [AppAnswerVariant].self, forKey: .outputVariants) ?? []
        selectedVariantIndex = min(
            max(try container.decodeIfPresent(Int.self, forKey: .selectedVariantIndex) ?? 0, 0),
            outputVariants.count)
    }
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
            outputReasoningText: session.outputReasoningText,
            outputContinuationTurns: session.outputContinuationTurns,
            outputToolTrace: session.outputToolTrace,
            outputDirective: session.outputDirective,
            outputNetworkMode: session.outputNetworkMode,
            outputVariants: session.outputVariants,
            selectedVariantIndex: session.selectedVariantIndex)
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
        session.outputContinuationTurns = outputContinuationTurns
        // A step that was still running when the app quit never finished.
        session.outputToolTrace = outputToolTrace.map { entry in
            var entry = entry
            if entry.status == .running {
                entry.status = .failed
                entry.summary = "interrupted"
            }
            return entry
        }
        session.outputDirective = outputDirective
        session.outputNetworkMode = outputNetworkMode
        session.outputVariants = outputVariants
        session.selectedVariantIndex = selectedVariantIndex
        return session
    }
}
