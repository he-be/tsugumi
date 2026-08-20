import Foundation

/// SPEC §8 RSN-4: a source of tokens the decode loop emits **without drawing
/// them**.
///
/// A constraint (`GenerationConstraint`, GEN-7) narrows what the model may draw
/// and then draws; this narrows nothing and does not draw at all. The loop asks
/// once per position "is there a token I must emit here?", and a non-nil answer
/// is emitted verbatim — no logits are read, no mask is built, no redraw
/// happens. That is why forcing works even behind the fused greedy head, which
/// a constraint cannot use (it never writes the logits a mask needs).
///
/// Everything else about a forced token is a sampled token: it reaches the
/// constraint through `ConstraintGate.accept` in generation order, it is pushed
/// through the detokenizer, it is reported by `onProgress`, and it is appended
/// to the history that becomes `RawDecodeResult.kvBackedTokenIDs` — the token
/// stream the prompt cache publishes. SPEC §7 INV-1 is exactly the requirement
/// that the KV and that list agree, so a forced token that skipped any of those
/// steps would break the next turn's prefix.
public protocol ForcedTokenSource: AnyObject, Sendable {
    /// The token this position must emit instead of drawing one, or nil to
    /// sample as usual. Called once per generated token, before the draw.
    func nextForcedToken() -> Int32?

    /// Advance by one token the loop kept, in generation order — forced tokens
    /// included, exactly as `GenerationConstraint.accept` sees them.
    ///
    /// - Parameters:
    ///   - generationIndex: the token's index in this completion, so a budget
    ///     can measure itself against `maxNewTokens`.
    ///   - completesCharacter: whether the detokenizer emitted text for this
    ///     token. False means a byte-fallback run is still open — the token sits
    ///     inside a multi-byte character — and a marker forced in right here
    ///     would cut that character in half.
    func accept(tokenID: Int32, generationIndex: Int, completesCharacter: Bool)
}

/// SPEC §8 **RSN-4**: when the thought budget runs out, force the closing tag
/// into the stream so the model leaves the thought channel and writes an
/// answer.
///
/// A port of the reference implementation's budget sampler
/// (`common/reasoning-budget.cpp`, pin `34af94cd9`), minus the parts that only
/// exist because that one lives inside a sampler chain:
///
/// | reference | here |
/// | --- | --- |
/// | `IDLE → COUNTING → WAITING_UTF8 → FORCING → DONE`, re-arming on a fresh start tag | the same five states, the same re-arm |
/// | start / end **token sequences** matched with Aho-Corasick | single start / end **token ids** — this template's thought block is delimited by one token on each side (`<\|channel>` / `<channel\|>`), so a matcher over sequences would have nothing to match |
/// | forcing masks every other logit to `-inf` and lets the chain draw | the loop emits the token without a draw (`ForcedTokenSource`) — the outcome is the same token and it costs no sampling |
/// | `common_utf8_is_complete` on the token's piece | the detokenizer's own verdict, passed in as `completesCharacter` |
///
/// The forced sequence is "message + end tag" in the reference; here it is
/// whatever the caller passes, which the server fills with the closing tag its
/// **template** writes (`GFTokenizer.channelEndID`) and no message. The state
/// it leaves behind is then byte for byte the state a thinking-*off* prompt
/// creates on purpose — that template renders `<|channel>thought\n<channel|>`
/// itself and the model answers from there — which is the reason to trust that
/// an answer follows.
///
/// The budget has two halves and both are checked here (RSN-4: "予算 =
/// `reasoning_budget_tokens`、および `max_tokens` の残り"): `budget` counts
/// tokens inside the block, and `deadline` is the last generation index at
/// which the forced sequence may still start and leave the answer room. The
/// arithmetic that turns `max_tokens` into that index is the server's
/// (`ServerReasoningPlan`); this type only counts.
///
/// Not thread-safe, like `GrammarTokenConstraint`: one decode loop drives it in
/// order.
public final class ReasoningBudgetForcer: ForcedTokenSource, @unchecked Sendable {
    public enum State: Equatable, Sendable {
        /// Outside the thought block, watching for the start tag.
        case idle
        /// Inside the block, counting down.
        case counting
        /// Out of budget, but the last token left a multi-byte character
        /// half-written; the tag goes in as soon as it closes.
        case waitingUTF8
        /// Emitting the forced sequence, one token per position.
        case forcing
        /// Passthrough. Re-arms if a fresh start tag shows up (a model may open
        /// more than one thought block in an answer).
        case done
    }

    /// No bound at all — `-1` in the wire and flag spellings alike.
    public static let unlimited = -1

    private let startTokenID: Int32
    private let endTokenID: Int32
    private let forcedTokenIDs: [Int32]
    private let budget: Int
    private let deadline: Int

    private(set) public var state: State = .idle
    private var remaining = 0
    private var forcePosition = 0

    /// - Parameters:
    ///   - startTokenID: the token that opens the thought block.
    ///   - endTokenID: the token that closes it naturally.
    ///   - forcedTokenIDs: what to force when the budget is gone. Must end with
    ///     something that closes the block, or the model never leaves it.
    ///   - budget: tokens the block may spend, or `unlimited`.
    ///   - deadline: the last generation index at which the forced sequence may
    ///     start; `Int.max` for no deadline.
    public init(startTokenID: Int32,
                endTokenID: Int32,
                forcedTokenIDs: [Int32],
                budget: Int,
                deadline: Int) {
        self.startTokenID = startTokenID
        self.endTokenID = endTokenID
        self.forcedTokenIDs = forcedTokenIDs
        self.budget = budget
        self.deadline = deadline
    }

    public func nextForcedToken() -> Int32? {
        // P4 (赤): まだ強制しない。
        nil
    }

    public func accept(tokenID: Int32, generationIndex: Int, completesCharacter: Bool) {
        // P4 (赤): 状態機械は未実装。
    }
}
