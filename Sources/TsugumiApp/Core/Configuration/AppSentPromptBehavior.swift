public enum AppSentPromptBehavior: String, CaseIterable, Codable, Sendable, Identifiable {
    case keep
    case clear

    public var id: String { rawValue }

    public var settingsLabel: String {
        switch self {
        case .keep: return AppLocalization.string("Keep Prompt")
        case .clear: return AppLocalization.string("Clear Prompt")
        }
    }
}
