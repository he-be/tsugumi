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

    /// The prompt with `tokens`' split wherever the two differ only in how the same bytes are cut into tokens.
    ///
    /// The model generates splits the tokenizer would not (`前回` + `答` where re-encoding the answer gives `前` +
    /// `回答`), so the re-rendered history leaves `tokens` inside the answer. KV could be cut back to that point, the
    /// recurrent state cannot: without this the request falls back to the checkpoint before the answer and pays the
    /// whole answer again (`docs/qwen38/19`). `piece` is a token's byte-level string, nil for an added token — those
    /// are never re-split, a different one is a different sequence. A span is matched within `window` tokens a side;
    /// past the first span that is not a re-split, the prompt is left as rendered.
    func aligned(_ prompt: [Int32], window: Int = 32, piece: (Int32) -> String?) -> (tokens: [Int32], spans: Int) {
        let held = tokens
        var i = 0, j = 0, spans = 0
        var out: [Int32] = []
        out.reserveCapacity(prompt.count)
        scan: while i < held.count, j < prompt.count {
            if held[i] == prompt[j] {
                out.append(held[i]); i += 1; j += 1
                continue
            }
            // Grow whichever side covers fewer bytes until both cover the same bytes, or they stop agreeing.
            var a = i, b = j
            var left: [Unicode.Scalar] = [], right: [Unicode.Scalar] = []
            repeat {
                if left.count <= right.count {
                    guard a < held.count, a - i < window, let p = piece(held[a]) else { break scan }
                    left.append(contentsOf: p.unicodeScalars); a += 1
                } else {
                    guard b < prompt.count, b - j < window, let p = piece(prompt[b]) else { break scan }
                    right.append(contentsOf: p.unicodeScalars); b += 1
                }
                let shared = min(left.count, right.count)
                guard left[..<shared] == right[..<shared] else { break scan }
            } while left.count != right.count
            out.append(contentsOf: held[i..<a]); i = a; j = b; spans += 1
        }
        guard spans > 0 else { return (prompt, 0) }
        return (out + prompt[j...], spans)
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
