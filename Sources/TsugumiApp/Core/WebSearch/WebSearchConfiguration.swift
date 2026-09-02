import Foundation

/// Keys and limits for the web search tools. Stored in one file for the
/// whole app (`~/Library/Application Support/Tsugumi/web-search.json`,
/// owner-readable only); the environment can override each key, which is
/// how a script or a test supplies one without touching the file.
public struct WebSearchConfiguration: Codable, Equatable, Sendable {
    public static let currentVersion = 1

    public var version: Int = currentVersion
    /// Serper (Google results, `gl=jp` / `hl=ja`). The first provider tried.
    public var serperAPIKey: String = ""
    /// Brave Search. The fallback when Serper is missing or fails.
    public var braveAPIKey: String = ""
    /// Jina Reader. Optional — the reader works without a key at a lower
    /// rate limit.
    public var jinaAPIKey: String = ""
    /// Whether page reads go to Jina Reader first and fall back to the
    /// app's own fetch, or the other way round.
    public var preferJinaReader: Bool = true
    public var maxSearchResults: Int = 8
    /// Characters of page text one `fetch_page` may hand the model.
    public var pageCharacterLimit: Int = 6_000
    /// Rounds of tool calls one user turn may spend before the model has to
    /// answer with what it has.
    public var maxToolRounds: Int = 6
    /// With thinking on, how many tokens the thought channel may spend on a
    /// round that has no search results yet (the one that decides the first
    /// search). 0 closes the channel for that round; -1 leaves it unbounded.
    /// Gemma 4 spends this round arguing with the date otherwise
    /// (`WebSearchPrompt`); results-bearing rounds are never bounded.
    public var preSearchThinkingBudget: Int = 512
    /// Path of the local Japanese Wikipedia index
    /// (`Scripts/wiki/build_jawiki_index.py`). Empty declares no Wikipedia
    /// tools. With a Wikipedia index the tools work without any API key.
    public var wikipediaIndexPath: String = ""
    public var country: String = "jp"
    public var language: String = "ja"

    public init() {}

    private enum CodingKeys: String, CodingKey {
        case version, serperAPIKey, braveAPIKey, jinaAPIKey, preferJinaReader
        case maxSearchResults, pageCharacterLimit, maxToolRounds, preSearchThinkingBudget
        case wikipediaIndexPath, country, language
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decodeIfPresent(Int.self, forKey: .version) ?? Self.currentVersion
        serperAPIKey = try container.decodeIfPresent(String.self, forKey: .serperAPIKey) ?? ""
        braveAPIKey = try container.decodeIfPresent(String.self, forKey: .braveAPIKey) ?? ""
        jinaAPIKey = try container.decodeIfPresent(String.self, forKey: .jinaAPIKey) ?? ""
        preferJinaReader = try container.decodeIfPresent(Bool.self, forKey: .preferJinaReader) ?? true
        maxSearchResults = try container.decodeIfPresent(Int.self, forKey: .maxSearchResults) ?? 8
        pageCharacterLimit = try container.decodeIfPresent(Int.self, forKey: .pageCharacterLimit) ?? 6_000
        maxToolRounds = try container.decodeIfPresent(Int.self, forKey: .maxToolRounds) ?? 6
        preSearchThinkingBudget = try container.decodeIfPresent(
            Int.self, forKey: .preSearchThinkingBudget) ?? 512
        wikipediaIndexPath = try container.decodeIfPresent(String.self, forKey: .wikipediaIndexPath) ?? ""
        country = try container.decodeIfPresent(String.self, forKey: .country) ?? "jp"
        language = try container.decodeIfPresent(String.self, forKey: .language) ?? "ja"
    }

    /// Environment variables that override the stored keys.
    public static let serperEnvironmentKey = "TSUGUMI_SERPER_API_KEY"
    public static let braveEnvironmentKey = "TSUGUMI_BRAVE_API_KEY"
    public static let jinaEnvironmentKey = "TSUGUMI_JINA_API_KEY"
    public static let wikipediaEnvironmentKey = "TSUGUMI_WIKIPEDIA_INDEX"

    /// The configuration with the environment applied on top.
    public func resolved(environment: [String: String] = ProcessInfo.processInfo.environment)
        -> WebSearchConfiguration {
        var copy = self
        if let key = environment[Self.serperEnvironmentKey], !key.isEmpty { copy.serperAPIKey = key }
        if let key = environment[Self.braveEnvironmentKey], !key.isEmpty { copy.braveAPIKey = key }
        if let key = environment[Self.jinaEnvironmentKey], !key.isEmpty { copy.jinaAPIKey = key }
        if let path = environment[Self.wikipediaEnvironmentKey], !path.isEmpty { copy.wikipediaIndexPath = path }
        copy.maxSearchResults = min(max(copy.maxSearchResults, 1), 10)
        copy.pageCharacterLimit = min(max(copy.pageCharacterLimit, 500), 40_000)
        copy.maxToolRounds = min(max(copy.maxToolRounds, 1), 12)
        copy.preSearchThinkingBudget = min(max(copy.preSearchThinkingBudget, -1), 8_192)
        return copy
    }

    /// Whether any search provider has a key. Without one the tools cannot
    /// be declared — the model would call into nothing.
    public var canSearch: Bool {
        !serperAPIKey.trimmingCharacters(in: .whitespaces).isEmpty
            || !braveAPIKey.trimmingCharacters(in: .whitespaces).isEmpty
    }

    /// Whether a local Wikipedia index is named. Whether the file opens is
    /// checked when the executor is built.
    public var hasWikipediaIndex: Bool {
        !wikipediaIndexPath.trimmingCharacters(in: .whitespaces).isEmpty
    }

    /// Whether any tool can be declared: a web key or a Wikipedia index.
    public var canUseTools: Bool { canSearch || hasWikipediaIndex }

    /// The index path with `~` expanded, or nil when none is set.
    public var wikipediaIndexURL: URL? {
        let trimmed = wikipediaIndexPath.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return nil }
        return URL(fileURLWithPath: (trimmed as NSString).expandingTildeInPath)
    }
}

public enum WebSearchConfigurationStore {
    /// `~/Library/Application Support/Tsugumi/web-search.json`.
    public static var defaultFileURL: URL {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return base
            .appendingPathComponent("Tsugumi", isDirectory: true)
            .appendingPathComponent("web-search.json", isDirectory: false)
    }

    /// A missing or unreadable file is the default configuration; the keys
    /// are the user's to re-enter, not something to fail the app over.
    public static func load(from fileURL: URL = defaultFileURL,
                            fileManager: FileManager = .default) -> WebSearchConfiguration {
        guard fileManager.fileExists(atPath: fileURL.path),
              let data = try? Data(contentsOf: fileURL),
              let decoded = try? JSONDecoder().decode(WebSearchConfiguration.self, from: data)
        else { return WebSearchConfiguration() }
        return decoded
    }

    /// Written owner-readable only: the file holds API keys.
    public static func save(_ configuration: WebSearchConfiguration,
                            to fileURL: URL = defaultFileURL,
                            fileManager: FileManager = .default) throws {
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(configuration)
        data.append(0x0A)
        try data.write(to: fileURL, options: .atomic)
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: fileURL.path)
    }
}
