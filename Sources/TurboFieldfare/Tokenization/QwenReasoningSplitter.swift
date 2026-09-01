import Foundation

/// Which turn a piece of Ornith's output belongs to, for a run that declared
/// no tools.
///
/// Ornith writes its reasoning inline, between `<think>` and `</think>`, and
/// the template *opens* that block in the generation prompt — so a run with
/// thinking on starts inside it and the closing marker is the only signal.
/// Tracking the marker IDs rather than the text means a token that merely
/// spells `<think>` in the answer cannot move the boundary.
///
/// `QwenStructuredAssistantDecoder`'s smaller sibling, and deliberately not a
/// mode of it: with no tools declared, a `<tool_call>` the model writes unasked
/// is **text**, and that is what a run which declared nothing should show. The
/// decoder would refuse it as an unknown tool. What the two share is the
/// channel rule and the event type, so a caller routes either the same way
/// (`docs/qwen35moe/26-PHASE8-SERVER.md` §3).
public struct QwenReasoningSplitter: Sendable {
    private let thinkStart: Int32
    private let thinkEnd: Int32
    public private(set) var isInsideReasoning: Bool

    public init(tokenizer: QwenTokenizer, startsInsideReasoning: Bool) {
        self.thinkStart = tokenizer.thinkStartID
        self.thinkEnd = tokenizer.thinkEndID
        self.isInsideReasoning = startsInsideReasoning
    }

    /// One generated token and the text the detokenizer released for it.
    public mutating func consume(tokenID: Int32,
                                 delta: String) -> [StructuredAssistantEvent] {
        let (channel, marker) = route(tokenID)
        var text = delta
        // The marker's own spelling has to come back out because these markers
        // are *not* special tokens (`tokenizer.json` declares them
        // `special: false`), so the detokenizer emits them as text, preceded by
        // whatever bytes it was holding back from before them. Dropping the
        // whole delta would drop those bytes too — a codepoint that straddles
        // the token in front of `</think>` would vanish from the answer.
        if let marker, text.hasSuffix(marker) {
            text = String(text.dropLast(marker.count))
        }
        guard !text.isEmpty else { return [] }
        return [channel == .reasoning ? .reasoning(text) : .content(text)]
    }

    /// The detokenizer's flush, which belongs to whichever channel is open.
    public func consumeTail(_ text: String) -> [StructuredAssistantEvent] {
        guard !text.isEmpty else { return [] }
        return [isInsideReasoning ? .reasoning(text) : .content(text)]
    }

    private mutating func route(_ id: Int32) -> (channel: Channel, marker: String?) {
        if id == thinkStart {
            isInsideReasoning = true
            return (.reasoning, "<think>")
        }
        if id == thinkEnd {
            // The held-back bytes belong to the reasoning that ends here.
            isInsideReasoning = false
            return (.reasoning, "</think>")
        }
        return (isInsideReasoning ? .reasoning : .content, nil)
    }

    private enum Channel { case reasoning, content }
}
