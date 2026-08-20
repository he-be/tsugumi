import Foundation

/// A stateful constraint on what the next token may be (SPEC §6 GEN-7).
///
/// The mechanism only. This protocol says nothing about *why* a token is
/// allowed — a GBNF grammar, a JSON-Schema-derived grammar, or a test stub all
/// answer the same two questions, and the sampler never learns what a token
/// piece is.
///
/// GEN-7 applies a constraint by **rejection sampling**, exactly as the
/// reference implementation does in `common_sampler_sample`
/// (`common/sampling.cpp`, pin `34af94cd9`): draw one token the normal way, ask
/// `allows(tokenID:)`, and keep it if the answer is yes. Only when the answer is
/// no does the whole vocabulary get masked and the draw repeated. So the common
/// path costs exactly one `allows` call per generated token and no extra GPU
/// work; `fillAllowedMask` — the expensive one — runs only on a rejection.
///
/// The implementation is a class because it *advances*: `accept(tokenID:)` is
/// called once for every token the decode loop keeps, in order, and every other
/// member answers for the state that leaves behind.
public protocol GenerationConstraint: AnyObject, Sendable {
    /// Whether the sequence may legally stop at the current state.
    ///
    /// This is the reference's `allow_eog` (`llama_grammar_apply_impl`,
    /// `src/llama-grammar.cpp`): while it is false, every end-of-generation
    /// token — the tokenizer's stop ids and the caller's `extraStopTokens` — is
    /// masked, so generation cannot end in the middle of the constrained
    /// structure. While it is true they are left alone and behave exactly as
    /// they do in an unconstrained run.
    ///
    /// The constraint does not need to know which ids those are: the decode
    /// loop knows them, and `ConstraintGate` applies the rule.
    var mayEndHere: Bool { get }

    /// The cheap single-token probe, called once per generated token on the
    /// common path. Must agree with `fillAllowedMask` for the same state.
    func allows(tokenID: Int32) -> Bool

    /// Fill the whole-vocabulary mask: `allowed[i] == true` iff token id `i` is
    /// acceptable next. **Every element must be assigned** — the caller does not
    /// pre-fill the buffer.
    ///
    /// A caller-supplied buffer rather than a returned id collection, because
    /// the rejected set is normally almost the entire vocabulary (262 144 ids at
    /// the pinned model): returning it would allocate a quarter-million-element
    /// collection per rejection, while the mask is one flat byte array the
    /// sampler already needs to walk linearly. The buffer is owned and reused by
    /// `Sampler`, so a rejection allocates nothing.
    ///
    /// Runs only on the rejection path.
    func fillAllowedMask(_ allowed: UnsafeMutableBufferPointer<Bool>) throws

    /// Advance the state by one token the decode loop kept. Called in
    /// generation order, including for the token that ends generation.
    func accept(tokenID: Int32) throws
}

/// Failures of the constraint mechanism (SPEC §6 GEN-7, §12 DEV-14). All of
/// these are 500 `server_error` conditions on the server side: generation is
/// never silently truncated and an unconstrained token is never emitted in
/// their place.
public enum GenerationConstraintError: Error, CustomStringConvertible, Equatable {
    /// The mask left no token at all. GEN-7 calls this an error, not a stop.
    case noAllowedToken(position: Int)
    /// The masked redraw still produced a token the constraint rejects. Only
    /// reachable if the masked distribution and the probe disagree.
    case maskedDrawRejected(position: Int, tokenID: Int32)
    /// The producer cannot supply logits for this draw (the fused greedy head
    /// answers with a GPU argmax and never writes the logits buffer), so the
    /// constraint could not be applied. Refusing is the only honest answer:
    /// the alternative is unconstrained text under a constrained request.
    case logitsUnavailable(String)

    public var description: String {
        switch self {
        case .noAllowedToken(let position):
            return "the constraint allows no token at generation position \(position)"
        case .maskedDrawRejected(let position, let tokenID):
            return "the constraint-masked draw at generation position \(position) "
                + "returned rejected token \(tokenID)"
        case .logitsUnavailable(let reason):
            return reason
        }
    }
}

/// The constraint plus the end-of-generation rule, as the sampler sees it.
///
/// Kept separate from `GenerationConstraint` so that the `mayEndHere` masking
/// is applied in exactly one place and applies identically to the single-token
/// probe and to the whole-vocabulary mask — the reference does both inside
/// `llama_grammar_apply_impl`, and the two drifting apart is precisely how a
/// stop token would leak through the fast path.
struct ConstraintGate {
    let constraint: any GenerationConstraint
    /// The tokenizer's stop ids unioned with the caller's `extraStopTokens`.
    let endOfGenerationTokenIDs: Set<Int32>

    init(constraint: any GenerationConstraint, endOfGenerationTokenIDs: Set<Int32>) {
        self.constraint = constraint
        self.endOfGenerationTokenIDs = endOfGenerationTokenIDs
    }

    func allows(_ tokenID: Int32) -> Bool {
        if !constraint.mayEndHere, endOfGenerationTokenIDs.contains(tokenID) {
            return false
        }
        return constraint.allows(tokenID: tokenID)
    }

    func fillAllowedMask(_ allowed: UnsafeMutableBufferPointer<Bool>) throws {
        try constraint.fillAllowedMask(allowed)
        guard !constraint.mayEndHere else { return }
        for id in endOfGenerationTokenIDs where id >= 0 && Int(id) < allowed.count {
            allowed[Int(id)] = false
        }
    }

    func accept(_ tokenID: Int32) throws {
        try constraint.accept(tokenID: tokenID)
    }
}
