import Foundation

/// SPEC §7 for the Ornith (Qwen 3.5-MoE) family: the decision of whether the
/// next request may start where the last one stopped
/// (`docs/qwen35moe/41-PROMPT-CACHE.md`).
///
/// `ServerPromptCache`'s sibling rather than a use of it, and much smaller,
/// because this family can hold exactly one shape. Gemma's cache keeps token
/// sequences and rewinds the K/V to the longest common prefix; thirty of these
/// forty layers are recurrent, and a recurrent state has no rewind at all —
/// `RecurrentStateManager`'s own header says dropping a token from the middle
/// is not an operation the recurrence has. So the only reusable prefix is
/// **all of it**, and the only entry is the one living in the runner right now.
/// There is nothing to evict, nothing to copy, and no second conversation.
///
/// It is a value type with no runner in it for the same reason
/// `QwenGenerationPlan` is: the rule is a function of two token sequences and a
/// position, and that is exactly the part worth checking without weights.
struct QwenPromptCache: Equatable, Sendable {
    /// The token ids the runner's state has consumed. Empty means position zero.
    private(set) var tokens: [Int32] = []

    enum Match: Equatable {
        /// Continue: the state holds this many of the prompt's leading tokens.
        case hit(cached: Int)
        /// Start over. `divergedAt` is how far the prompt agreed with the state
        /// before it stopped agreeing — nil when there was no state to agree
        /// with, or when the request opted out.
        case miss(divergedAt: Int?)
    }

    /// What can be reused of `promptIDs`.
    ///
    /// **All or nothing.** A prompt that agrees with only part of the state
    /// cannot be served from it: the rows past the agreement are already folded
    /// into the recurrence.
    ///
    /// The last token is never reused either. The state stops one token short
    /// of what the model has seen (the loop feeds token *t* to draw *t+1*), and
    /// prefill needs at least one row to draw from — so a prompt identical to
    /// what the state holds is a miss. That is the same corner CACHE-3 puts the
    /// Gemma path in, reached for a different reason.
    func match(_ promptIDs: [Int32], cachePrompt: Bool = true) -> Match {
        guard cachePrompt else { return .miss(divergedAt: nil) }
        guard !tokens.isEmpty else { return .miss(divergedAt: nil) }
        guard promptIDs.count > tokens.count else {
            return .miss(divergedAt: min(tokens.count, promptIDs.count))
        }
        for index in 0..<tokens.count where tokens[index] != promptIDs[index] {
            return .miss(divergedAt: index)
        }
        return .hit(cached: tokens.count)
    }

    /// Record what the state holds after a run, or refuse to.
    ///
    /// `kvPosition` comes from the runner and is **not** re-derived here. It is
    /// usually one short of `prompt + generated`, but not always: when the
    /// speculative loop accepts a draft and the run ends on the token that
    /// draft predicted, the row for that token is already in the state, and the
    /// recurrent half of it cannot be taken back out
    /// (`41-PROMPT-CACHE.md` §3-2).
    ///
    /// Returns false when the position is not a prefix of what this request
    /// could have produced — a disagreement between the loop and the K/V cursor
    /// that must reset the runner rather than be recorded.
    @discardableResult
    mutating func publish(promptIDs: [Int32],
                          generated: [Int32],
                          kvPosition: Int,
                          cachePrompt: Bool = true) -> Bool {
        // CACHE-5: a request that opted out neither reads nor writes.
        guard cachePrompt else { tokens = []; return true }
        let full = promptIDs + generated
        guard kvPosition >= 0, kvPosition <= full.count else {
            tokens = []
            return false
        }
        tokens = Array(full.prefix(kvPosition))
        return true
    }

    mutating func invalidate() { tokens = [] }
}
