import Foundation
import Metal

/// Streaming callbacks from `runRawCompletion`. `.prefill` reports monotonic
/// producer-defined prompt progress; scalar replay reports per token, while a
/// prefill-capable producer may report per internal chunk. `.token` fires per
/// decoded non-stop token; `.tail` carries the detokenizer flush remainder at a
/// stop boundary.
public enum RawDecodeProgress: Sendable {
    case prefill(done: Int, total: Int)
    case token(index: Int, id: Int32, delta: String)
    case tail(String)
}

public enum RawCompletionStart: Sendable, Equatable {
    case reset
    case resume(cachedPromptTokens: Int)
}

public struct RawDecodeResult: Sendable {
    public let prefillTokens: Int
    public let cachedPromptTokens: Int
    public let computedPrefillTokens: Int
    public let prefillSeconds: Double
    public let newTokens: Int
    public let decodeSeconds: Double
    /// Wall time from the start of prefill to the first sampled token. Zero when
    /// nothing was generated.
    public let timeToFirstTokenSeconds: Double
    public let reason: StopReason
    public let kvPosition: Int
    public let kvBackedTokenIDs: [Int32]
    public let uncommittedBoundaryTokenIDs: [Int32]
}

/// Preallocated per-generation buffers (two 512 KiB vocab buffers plus a token
/// slot) and sampler. A warm session reuses them for every token, avoiding
/// per-token Metal buffer allocation.
///
/// `@unchecked Sendable`: the buffers and sampler are exclusively owned by one
/// generation at a time — the single-in-flight guard upstream is the contract.
public struct RawCompletionScratch: @unchecked Sendable {
    let logits: MTLBuffer
    let probs: MTLBuffer
    let outToken: MTLBuffer
    let sampler: Sampler

    public init(context: MetalContext, vocab: Int) throws {
        guard let logits = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                     options: .storageModeShared),
              let probs = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                    options: .storageModeShared),
              let outToken = context.device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                                       options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        self.logits = logits
        self.probs = probs
        self.outToken = outToken
        self.sampler = try Sampler(context: context, vocab: vocab)
    }
}

/// A producer that may answer with a GPU argmax instead of writing the logits
/// buffer. `RealForwardRunner` is the production conformer; the constraint hook
/// has to be able to ask the question without owning that concrete type, both
/// because the answer is a policy decision (GEN-7 cannot mask logits that were
/// never written) and because it has to be checkable without a model.
protocol FusedGreedyReporting: LogitProducer {
    var usesFusedGreedyHead: Bool { get }
}

extension RealForwardRunner: FusedGreedyReporting {}

/// The refusal both fused-greedy shortcuts share: the producer answered with a
/// GPU argmax and never wrote the logits a constraint has to mask.
private let unconstrainableGreedyToken = GenerationConstraintError.logitsUnavailable(
    "the producer answered with a GPU argmax instead of logits, so the constraint "
    + "could not be applied; construct the runner with forceLogitsHead: true")

extension GenerationConfig {
    /// A pure-greedy config can use the fused head's GPU argmax
    /// (`RealForwardRunner.lastGreedyToken`) instead of sampling from the
    /// logits buffer. Anything else needs real logits.
    public var isPureGreedy: Bool {
        temperature == 0 && repetitionPenalty == 1
    }

}

/// Raw-completion prefill + decode loop shared by the CLI and the Mac app.
/// Consumes pre-encoded `promptIds` (BOS + verbatim encode upstream — no chat
/// template). Stop handling, detokenizer flush ordering, and history append
/// ordering are shared by both front ends.
///
/// When the producer runs the fused lm_head (`RealForwardRunner` default) the
/// logits buffer is never written; the loop then requires a pure-greedy config
/// and reads `lastGreedyToken`. Callers with sampling configs must construct
/// the runner with `forceLogitsHead: true`.
///
/// `constraint` is the GEN-7 hook. Nil is the path everything took before it
/// existed, down to the sampler call, so an unconstrained caller draws the same
/// tokens it always did. Non-nil applies rejection sampling per token and rules
/// out the fused-greedy shortcut, which never writes the logits a mask needs.
///
/// `forcer` is the RSN-4 hook, and it is the other shape: it does not narrow a
/// draw, it replaces one. A token it names is emitted without reading logits at
/// all, and then travels the rest of this loop exactly as a sampled token does
/// — through the constraint, the detokenizer, `onProgress`, and the history
/// that becomes `kvBackedTokenIDs` (SPEC §7 INV-1).
public func runRawCompletion(producer: any LogitProducer,
                             tokenizer: GFTokenizer,
                             promptIds: [Int32],
                             config: GenerationConfig,
                             constraint: (any GenerationConstraint)? = nil,
                             forcer: (any ForcedTokenSource)? = nil,
                             context: MetalContext,
                             scratch: RawCompletionScratch,
                             prefillConfig: PrefillRuntimeConfig = .defaultChunked,
                             vision: VisionPrefillInput? = nil,
                             start: RawCompletionStart = .reset,
                             shouldStop: () -> Bool = { false },
                             onProgress: (RawDecodeProgress) -> Void) async throws -> RawDecodeResult {
    // GEN-7: one gate for the whole generation. It carries the end-of-generation
    // ids the constraint itself does not know about, so `mayEndHere` can mask
    // them, and it is consulted per draw, so it always reflects the state the
    // last `accept` left behind.
    let gate = constraint.map {
        ConstraintGate(constraint: $0,
                       endOfGenerationTokenIDs: tokenizer.stopTokenIDs.union(config.extraStopTokens))
    }
    if gate != nil, (producer as? any FusedGreedyReporting)?.usesFusedGreedyHead == true {
        throw GenerationConstraintError.logitsUnavailable(
            "a constrained request needs real logits, and this producer runs the "
            + "fused greedy head; construct the runner with forceLogitsHead: true")
    }
    let prepared = try await prepareGeneration(producer: producer,
                                               promptIds: promptIds,
                                               config: config,
                                               scratch: scratch,
                                               prefillConfig: prefillConfig,
                                               vision: vision,
                                               start: start,
                                               onProgress: onProgress)
    let fusedRunner = prepared.fusedRunner
    let fusedGreedy = prepared.fusedGreedy
    var position = prepared.position
    var history = prepared.history
    let prefillStart = prepared.prefillStart
    let prefillSeed = prepared.prefillSeed
    var detok = GFDetokenizer(tokenizer: tokenizer,
                              barrierTokenIDs: tokenizer.structuralMarkerIDs)

    let decodeStart = prepared.decodeStart
    let prefillSeconds = prepared.prefillSeconds
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var uncommittedBoundaryTokenIDs: [Int32] = []
    var timeToFirstToken: Double = 0

    while true {
        try Task.checkCancellation()

        let tokenID: Int32
        // RSN-4. Forcing is not sampling: the token is emitted without reading
        // a logit, so this comes before every draw and short-circuits all three
        // of them — including the fused-greedy shortcuts, which a constraint
        // cannot use but which a forced token has no quarrel with (it needs no
        // logits either).
        if let forced = forcer?.nextForcedToken() {
            tokenID = forced
        } else if generated == 0, let seed = prefillSeed {
            switch seed {
            case .greedyToken(let token):
                guard gate == nil else { throw unconstrainableGreedyToken }
                tokenID = Int32(bitPattern: token)
            case .logitsWritten:
                tokenID = try sampleOnce(scratch: scratch, context: context,
                                         history: history, config: config,
                                         position: generated, constraint: gate).id
            }
        } else if fusedGreedy {
            guard gate == nil else { throw unconstrainableGreedyToken }
            tokenID = Int32(bitPattern: fusedRunner!.lastGreedyToken)
        } else {
            tokenID = try sampleOnce(scratch: scratch, context: context,
                                     history: history, config: config,
                                     position: generated, constraint: gate).id
        }
        // Every token the loop keeps goes to the gate, in order, including the
        // one that ends generation. The gate decides what reaches the
        // constraint: an end-of-generation id it allowed on `mayEndHere` alone
        // is not something the constraint has state for.
        try gate?.accept(tokenID)
        generated += 1
        if generated == 1 {
            timeToFirstToken = Date().timeIntervalSince(prefillStart)
        }
        uncommittedBoundaryTokenIDs = [tokenID]

        if tokenizer.stopTokenIDs.contains(tokenID) || config.extraStopTokens.contains(tokenID) {
            if tokenID == tokenizer.endOfTurnID {
                reason = .endOfTurn
            } else if tokenID == tokenizer.toolResponseID {
                reason = .toolCalls
            } else {
                reason = .eos
            }
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            break
        }

        let delta = detok.push(tokenID)
        // RSN-4. Fed after the detokenizer so `completesCharacter` can be the
        // detokenizer's own verdict: an empty delta means a byte-fallback run
        // is still open, i.e. this token sits inside a multi-byte character and
        // a marker forced in right here would cut it in half. The reference
        // asks `common_utf8_is_complete` about the token's piece for the same
        // reason. The state this leaves behind is the state the *next*
        // position is judged by, which is the same ordering the constraint
        // gets.
        forcer?.accept(tokenID: tokenID,
                       generationIndex: generated - 1,
                       completesCharacter: !delta.isEmpty)
        let visible = stopMatcher.push(delta)
        onProgress(.token(index: generated - 1, id: tokenID, delta: visible))

        let hitStopString = stopMatcher.isStopped || shouldStop()
        let hitMax = generated >= config.maxNewTokens
        if hitStopString || hitMax {
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            reason = hitStopString ? .stopString : .maxTokens
            break
        }

        history.append(tokenID)
        try await producer.produce(token: tokenID, position: position, into: scratch.logits)
        position += 1
        uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
    }

    return RawDecodeResult(prefillTokens: promptIds.count,
                           cachedPromptTokens: prepared.cachedPromptTokens,
                           computedPrefillTokens: prepared.computedPrefillTokens,
                           prefillSeconds: prefillSeconds,
                           newTokens: generated,
                           decodeSeconds: Date().timeIntervalSince(decodeStart),
                           timeToFirstTokenSeconds: timeToFirstToken,
                           reason: reason,
                           kvPosition: position,
                           kvBackedTokenIDs: history,
                           uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs)
}
