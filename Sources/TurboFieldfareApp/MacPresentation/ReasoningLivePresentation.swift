import Foundation

/// The live thought channel renders only the newest slice of the growing
/// reasoning text. Laying out the whole text once per token is O(n²) over a
/// long think phase and saturates the main thread — the transcript then
/// crawls behind the decode service and the Stop button stops responding
/// while the queued token events drain.
public enum ReasoningLivePresentation {
    public static let liveTailCharacterCount = 1_500

    public static func liveTail(
        of text: String,
        cap: Int = liveTailCharacterCount
    ) -> String {
        guard cap > 0 else { return "" }
        guard text.count > cap else { return text }
        return "…" + String(text.suffix(cap))
    }
}
