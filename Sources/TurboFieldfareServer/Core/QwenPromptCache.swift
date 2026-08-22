import Foundation

/// SPEC §7 for the Ornith (Qwen 3.5-MoE) family: the decision of whether the
/// next request may start where the last one stopped
/// (`docs/qwen35moe/41-PROMPT-CACHE.md`).
///
/// `ServerPromptCache`'s sibling rather than a use of it. Gemma's cache rewinds
/// the K/V to the longest common prefix; thirty of these forty layers are
/// recurrent and a recurrent state has no rewind at all, so getting back to an
/// earlier point needs **a snapshot taken there** — CACHE-8's checkpoints, which
/// is what `KVCacheManager.maximumSafeRewind`'s header was waiting for.
///
/// So the reusable prefix is not "all of it or nothing". It is **the newest
/// position at or before the divergence that the runner can be put back to**:
/// one of the recorded checkpoints, or the live state itself when the prompt is
/// a strict extension. Only when there is no checkpoint behind the divergence
/// does the whole prompt get re-processed (CACHE-2, and the reference's own
/// `do_reset` branch — `server-context.cpp:3284` names SWA and hybrid/recurrent
/// memory as the reason it exists).
///
/// It is a value type with no runner in it for the same reason
/// `QwenGenerationPlan` is: the rule is a function of two token sequences and a
/// position, and that is exactly the part worth checking without weights.
struct QwenPromptCache: Equatable, Sendable {
    /// The token ids the runner's state has consumed. Empty means position zero.
    private(set) var tokens: [Int32] = []

    /// CACHE-8. Positions the runner holds a restorable copy of, ascending.
    ///
    /// The live state's own position (`tokens.count`) is **not** in here: it is
    /// always a candidate and is never a stored copy. This list only ever names
    /// positions the session has actually captured, so a hit it reports is a
    /// hit the runner can serve.
    private(set) var checkpoints: [Int] = []

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
    /// CACHE-1 decides *where* the two sequences stop agreeing; CACHE-8 decides
    /// *which* of the positions at or before that the runner can actually be
    /// put back to. The answer is the newest of them.
    ///
    /// The reuse point always leaves at least one row for prefill to draw from
    /// (`< promptIDs.count`). The state stops one token short of what the model
    /// has seen — the loop feeds token *t* to draw *t+1* — so a prompt identical
    /// to what the state holds cannot be served from the live state; it can
    /// still be served from a checkpoint behind it, which is this family's form
    /// of CACHE-3's `n_past--`.
    func match(_ promptIDs: [Int32], cachePrompt: Bool = true) -> Match {
        guard cachePrompt else { return .miss(divergedAt: nil) }
        guard !tokens.isEmpty else { return .miss(divergedAt: nil) }
        var agreed = 0
        let limit = min(tokens.count, promptIDs.count)
        while agreed < limit, tokens[agreed] == promptIDs[agreed] { agreed += 1 }
        var best = 0
        for candidate in checkpoints + [tokens.count]
        where candidate <= agreed && candidate < promptIDs.count && candidate > best {
            best = candidate
        }
        guard best > 0 else { return .miss(divergedAt: agreed) }
        return .hit(cached: best)
    }

    /// CACHE-8. The session says a restorable copy exists at `position`.
    ///
    /// Only the session can say this — it is the one holding the copy — so the
    /// cache never claims a checkpoint the runner does not have.
    mutating func recordCheckpoint(at position: Int) {
        guard position > 0, !checkpoints.contains(position) else { return }
        checkpoints.append(position)
        checkpoints.sort()
    }

    /// Drop the checkpoints the session no longer holds a copy of (the budget
    /// in FLAG-8 is finite, so the oldest are evicted).
    mutating func keepCheckpoints(_ positions: [Int]) {
        checkpoints = positions.filter { $0 > 0 }.sorted()
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
        guard cachePrompt else { tokens = []; checkpoints = []; return true }
        let full = promptIDs + generated
        guard kvPosition >= 0, kvPosition <= full.count else {
            tokens = []
            checkpoints = []
            return false
        }
        tokens = Array(full.prefix(kvPosition))
        // A checkpoint past what the state holds names a position the runner is
        // not at any more.
        checkpoints.removeAll { $0 > tokens.count }
        return true
    }

    mutating func invalidate() {
        tokens = []
        checkpoints = []
    }
}
