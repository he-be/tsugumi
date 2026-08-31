import Foundation

struct MacAppSettings: Codable, Equatable, Sendable {
    static let currentVersion = 2

    var version: Int = currentVersion
    var contextTokens: Int = AppContextLengthOption.thirtyTwoK.tokens
    var expertCacheSlots: Int = 32
    var temperature: Double = 1.0
    var topKEnabled: Bool = true
    var topK: Int = 64
    var topPEnabled: Bool = true
    var topP: Double = 0.95
    var prefillEnabled: Bool = true
    var mtpEnabled: Bool = true
    var thinkingEnabled: Bool = false
    var newlineShortcut: AppNewlineShortcut = .return
    var showPromptExamples: Bool = true
    var sentPromptBehavior: AppSentPromptBehavior = .keep

    private enum CodingKeys: String, CodingKey {
        case version
        case contextTokens
        case expertCacheSlots
        case temperature
        case topKEnabled
        case topK
        case topPEnabled
        case topP
        case prefillEnabled
        case mtpEnabled
        case thinkingEnabled
        case newlineShortcut
        case showPromptExamples
        case sentPromptBehavior
    }

    init(version: Int = currentVersion,
         contextTokens: Int = AppContextLengthOption.thirtyTwoK.tokens,
         expertCacheSlots: Int = 32,
         temperature: Double = 1.0,
         topKEnabled: Bool = true,
         topK: Int = 64,
         topPEnabled: Bool = true,
         topP: Double = 0.95,
         prefillEnabled: Bool = true,
         mtpEnabled: Bool = true,
         thinkingEnabled: Bool = false,
         newlineShortcut: AppNewlineShortcut = .return,
         showPromptExamples: Bool = true,
         sentPromptBehavior: AppSentPromptBehavior = .keep) {
        self.version = version
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.temperature = temperature
        self.topKEnabled = topKEnabled
        self.topK = topK
        self.topPEnabled = topPEnabled
        self.topP = topP
        self.prefillEnabled = prefillEnabled
        self.mtpEnabled = mtpEnabled
        self.thinkingEnabled = thinkingEnabled
        self.newlineShortcut = newlineShortcut
        self.showPromptExamples = showPromptExamples
        self.sentPromptBehavior = sentPromptBehavior
    }

    /// The adopted operating point for one checkpoint: the official sampler,
    /// MTP on, 32 slots, chunked prefill, 32K context, and the thought
    /// channel as each family recommends it.
    static func defaults(for kind: AppModelKind) -> MacAppSettings {
        MacAppSettings(
            temperature: kind.officialTemperature,
            topK: kind.officialTopK,
            topP: kind.officialTopP,
            thinkingEnabled: kind.thinkingDefault)
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decode(Int.self, forKey: .version)
        contextTokens = try container.decode(Int.self, forKey: .contextTokens)
        expertCacheSlots = try container.decode(Int.self, forKey: .expertCacheSlots)
        temperature = try container.decode(Double.self, forKey: .temperature)
        topKEnabled = try container.decode(Bool.self, forKey: .topKEnabled)
        topK = try container.decode(Int.self, forKey: .topK)
        topPEnabled = try container.decode(Bool.self, forKey: .topPEnabled)
        topP = try container.decode(Double.self, forKey: .topP)
        prefillEnabled = try container.decode(Bool.self, forKey: .prefillEnabled)
        mtpEnabled = try container.decodeIfPresent(Bool.self, forKey: .mtpEnabled) ?? true
        thinkingEnabled = try container.decodeIfPresent(
            Bool.self, forKey: .thinkingEnabled) ?? false
        newlineShortcut = try container.decodeIfPresent(
            AppNewlineShortcut.self,
            forKey: .newlineShortcut) ?? .return
        showPromptExamples = try container.decodeIfPresent(
            Bool.self,
            forKey: .showPromptExamples) ?? true
        sentPromptBehavior = try container.decodeIfPresent(
            AppSentPromptBehavior.self,
            forKey: .sentPromptBehavior) ?? .keep
    }

    func isValid() -> Bool {
        version == Self.currentVersion
            && AppContextLengthOption.allCases.contains { $0.tokens == contextTokens }
            && AppRuntimeOptions.allowedSlotCounts.contains(expertCacheSlots)
            && temperature.isFinite && (0...2).contains(temperature)
            && (1...256).contains(topK)
            && topP.isFinite && (0.01...1).contains(topP)
    }
}

enum MacAppSettingsFileStore {
    /// One file per installed model, beside the model directory: the two
    /// checkpoints keep different samplers and thinking defaults, so a shared
    /// file would let one model's settings leak into the other.
    static func fileURL(forModelDirectory modelDirectory: URL) -> URL {
        let directory = modelDirectory.standardizedFileURL
        let name = "mac-app-settings-\(directory.lastPathComponent).json"
        return directory
            .deletingLastPathComponent()
            .appendingPathComponent(name, isDirectory: false)
    }

    static func loadOrCreate(forModelDirectory modelDirectory: URL,
                             defaults: MacAppSettings = MacAppSettings(),
                             fileManager: FileManager = .default) -> MacAppSettings {
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        if fileManager.fileExists(atPath: fileURL.path) {
            do {
                let data = try Data(contentsOf: fileURL)
                let settings = try JSONDecoder().decode(MacAppSettings.self, from: data)
                guard settings.isValid() else { throw InvalidSettings() }
                return settings
            } catch {
                try? fileManager.removeItem(at: fileURL)
            }
        }

        let settings = defaults
        try? save(settings, forModelDirectory: modelDirectory, fileManager: fileManager)
        return settings
    }

    static func save(_ settings: MacAppSettings,
                     forModelDirectory modelDirectory: URL,
                     fileManager: FileManager = .default) throws {
        guard settings.isValid() else { throw InvalidSettings() }
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(settings)
        data.append(0x0A)
        try data.write(to: fileURL, options: .atomic)
    }

    private struct InvalidSettings: Error {}
}
