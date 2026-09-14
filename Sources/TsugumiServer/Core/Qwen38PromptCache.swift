import Foundation

/// SPEC §7 for Qwen3.8 (`docs/qwen38/15` §2 G, `18`): where the next request starts from.
///
/// `QwenPromptCache`'s rule with two differences the requirement table asks for. The live state is a candidate only
/// while it has a name (`livePosition`): a request that threw leaves the checkpoints it took in the prompt valid but
/// the live state unnamed, so a regenerate after a stop still starts one token short instead of from zero. And the
/// checkpoint positions are the session's, passed in, so the rule holds no copies.
struct Qwen38PromptCache: Equatable, Sendable {
    /// The sequence the live state and every checkpoint are prefixes of.
    private(set) var tokens: [Int32] = []
    /// How many of `tokens` the live state holds; nil when it holds something nothing can name.
    private(set) var livePosition: Int?

    enum Decision: Equatable {
        /// Continue from the live state, which holds this many of the prompt's tokens.
        case live(Int)
        /// Restore the checkpoint at this position.
        case restore(Int)
        /// Start from position 0.
        case miss
    }

    /// The newest position at or before the divergence that the live state or a checkpoint can be put back to, always
    /// leaving at least one prompt token to draw from; and how far the prompt agrees with `tokens`.
    func decide(_ prompt: [Int32], checkpoints: [Int], cachePrompt: Bool = true) -> (Decision, agreed: Int) {
        guard cachePrompt, !tokens.isEmpty else { return (.miss, 0) }
        var agreed = 0
        let limit = min(tokens.count, prompt.count)
        while agreed < limit, tokens[agreed] == prompt[agreed] { agreed += 1 }
        let usable = { (p: Int) in p > 0 && p <= agreed && p < prompt.count }
        let checkpoint = checkpoints.filter(usable).max()
        if let live = livePosition, usable(live), live >= (checkpoint ?? 0) { return (.live(live), agreed) }
        if let checkpoint { return (.restore(checkpoint), agreed) }
        return (.miss, agreed)
    }

    /// After a run: the live state holds `kvPosition` of prompt + generated (from the runner, not re-derived).
    @discardableResult
    mutating func publish(prompt: [Int32], generated: [Int32], kvPosition: Int, cachePrompt: Bool = true) -> Bool {
        guard cachePrompt else { invalidate(); return true }
        let full = prompt + generated
        guard kvPosition >= 0, kvPosition <= full.count else { invalidate(); return false }
        tokens = Array(full.prefix(kvPosition))
        livePosition = kvPosition
        return true
    }

    /// After a run that threw (or was cancelled): the prompt is still what the checkpoints taken in it are prefixes
    /// of, the live state is not.
    mutating func publishInterrupted(prompt: [Int32]) {
        tokens = prompt
        livePosition = nil
    }

    mutating func invalidate() {
        tokens = []
        livePosition = nil
    }
}
