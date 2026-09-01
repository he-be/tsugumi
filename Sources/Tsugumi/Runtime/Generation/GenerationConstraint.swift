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
/// called once for every token the decode loop keeps, in order — bar the
/// end-of-generation token that ends it, which no constraint rules on — and
/// every other member answers for the state that leaves behind.
public protocol GenerationConstraint: AnyObject, Sendable {
    /// Whether the sequence may legally stop at the current state.
    ///
    /// This is the reference's `allow_eog` (`llama_grammar_apply_impl`,
    /// `src/llama-grammar.cpp`): while it is false, every end-of-generation
    /// token — the tokenizer's stop ids and the caller's `extraStopTokens` — is
    /// masked, so generation cannot end in the middle of the constrained
    /// structure. While it is true they are allowed, and generation can end.
    ///
    /// This answer is the *only* thing consulted about those ids: they are never
    /// passed to `allows`, never left in the mask by the constraint's own
    /// verdict, and the one that ends generation is never passed to `accept`. So
    /// a constraint does not special-case stop tokens — it does not need to know
    /// which ids they are. The decode loop knows them, and `ConstraintGate`
    /// applies the whole rule in one place.
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

    /// Advance the state by one token the decode loop kept, in generation
    /// order. The token that ends generation is included unless it is an
    /// end-of-generation id the constraint never ruled on — see `mayEndHere`.
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

    /// An end-of-generation id is decided by `mayEndHere` and by nothing else —
    /// it is never put to the constraint. The reference keeps eog ids out of the
    /// candidate set entirely (`llama_grammar_apply_impl` masks them when
    /// `!allow_eog` and otherwise leaves them alone; either way they are not
    /// among the candidates the rules reject). Asking the constraint instead is
    /// how a completed constraint ends up unable to stop: a grammar that has
    /// nothing left to match rejects `<end_of_turn>` along with everything else.
    func allows(_ tokenID: Int32) -> Bool {
        if endOfGenerationTokenIDs.contains(tokenID) {
            return constraint.mayEndHere
        }
        return constraint.allows(tokenID: tokenID)
    }

    /// The same rule, applied after the constraint has filled the buffer, so the
    /// mask and the probe cannot drift apart. The constraint deliberately does
    /// not special-case stop tokens; this is the one place that does.
    func fillAllowedMask(_ allowed: UnsafeMutableBufferPointer<Bool>) throws {
        try constraint.fillAllowedMask(allowed)
        let mayEnd = constraint.mayEndHere
        for id in endOfGenerationTokenIDs where id >= 0 && Int(id) < allowed.count {
            allowed[Int(id)] = mayEnd
        }
    }

    /// The token that ends generation is not forwarded: a constraint that may
    /// end here has no state left for it to advance, and a real one throws on
    /// it. `llama_grammar_accept_impl` returns early for exactly this case.
    ///
    /// An eog id while `mayEndHere` is false cannot occur — the mask forbade it,
    /// and the reference calls the same situation fatal — so it is passed
    /// through and the constraint decides what to do with it.
    func accept(_ tokenID: Int32) throws {
        if constraint.mayEndHere, endOfGenerationTokenIDs.contains(tokenID) { return }
        try constraint.accept(tokenID: tokenID)
    }
}
