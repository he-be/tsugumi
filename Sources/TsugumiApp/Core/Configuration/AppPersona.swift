import Foundation

/// The instruction layer the user writes once and every turn carries: who
/// the assistant is, who is asking, and how answers should read. The form
/// follows ChatGPT's custom instructions (about you / how to respond) with
/// the identity split out, because a small local model is the one that
/// needs telling what it is — without it Gemma introduces itself as
/// Google's and argues with the date.
///
/// Empty fields are left out of the prompt; with all three empty no system
/// prompt is added at all, so a turn renders exactly as before.
public struct AppPersona: Codable, Equatable, Sendable {
    /// What the assistant is ("Tsugumi, a local assistant on this Mac").
    public var identity: String
    /// Who is asking: where they live, what they do, what they like.
    public var aboutUser: String
    /// How to answer: length, tone, language, what to skip.
    public var answerStyle: String

    public init(identity: String = "", aboutUser: String = "", answerStyle: String = "") {
        self.identity = identity
        self.aboutUser = aboutUser
        self.answerStyle = answerStyle
    }

    /// The identity a fresh install starts with. It names the app, says the
    /// model runs locally, and leaves the rest to the user.
    public static let defaultIdentity =
        "あなたは Tsugumi。この Mac の中だけで動くローカル AI アシスタントです。"

    public static var defaults: AppPersona {
        AppPersona(identity: defaultIdentity)
    }

    public var isEmpty: Bool {
        promptSection == nil
    }

    /// The prompt text, or nil when nothing was written. Each field is one
    /// short heading and the user's text as written.
    public var promptSection: String? {
        var parts: [String] = []
        let identity = identity.trimmingCharacters(in: .whitespacesAndNewlines)
        let aboutUser = aboutUser.trimmingCharacters(in: .whitespacesAndNewlines)
        let answerStyle = answerStyle.trimmingCharacters(in: .whitespacesAndNewlines)
        if !identity.isEmpty { parts.append(identity) }
        if !aboutUser.isEmpty { parts.append("# 質問している人\n" + aboutUser) }
        if !answerStyle.isEmpty { parts.append("# 答え方\n" + answerStyle) }
        return parts.isEmpty ? nil : parts.joined(separator: "\n\n")
    }

    private enum CodingKeys: String, CodingKey {
        case identity, aboutUser, answerStyle
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        identity = try container.decodeIfPresent(String.self, forKey: .identity) ?? ""
        aboutUser = try container.decodeIfPresent(String.self, forKey: .aboutUser) ?? ""
        answerStyle = try container.decodeIfPresent(String.self, forKey: .answerStyle) ?? ""
    }
}

/// `~/Library/Application Support/Tsugumi/persona.json`: one file for the
/// app, like the chats — the person asking is the same whichever model
/// answers.
public enum AppPersonaStore {
    public static var defaultFileURL: URL {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return base
            .appendingPathComponent("Tsugumi", isDirectory: true)
            .appendingPathComponent("persona.json", isDirectory: false)
    }

    /// No file yet is the defaults (the identity line prefilled); a file the
    /// user emptied stays empty. An unreadable file is the defaults too.
    public static func load(from fileURL: URL = defaultFileURL,
                            fileManager: FileManager = .default) -> AppPersona {
        guard fileManager.fileExists(atPath: fileURL.path),
              let data = try? Data(contentsOf: fileURL),
              let decoded = try? JSONDecoder().decode(AppPersona.self, from: data)
        else { return .defaults }
        return decoded
    }

    public static func save(_ persona: AppPersona,
                            to fileURL: URL = defaultFileURL,
                            fileManager: FileManager = .default) throws {
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(persona)
        data.append(0x0A)
        try data.write(to: fileURL, options: .atomic)
    }
}
