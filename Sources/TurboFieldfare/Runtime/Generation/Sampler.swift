import Foundation
import Metal

/// Generation knobs threaded from the caller through the `Generator` into the
/// sampler. Pure value type; one per `generate(...)` call.
///
/// Canonical home is here (the sampler is the primary consumer); `Generator`
/// reuses the same type rather than redeclaring it.
public struct GenerationConfig: Sendable, Equatable {
    public var maxNewTokens: Int = 256
    public var temperature: Float = 1.0
    public var topK: Int? = nil            // nil = no truncation
    public var topP: Float? = nil          // nil = no nucleus truncation
    public var repetitionPenalty: Float = 1.0
    public var seed: UInt64? = nil         // nil = nondeterministic
    public var stopStrings: [String] = []
    public var extraStopTokens: Set<Int32> = []

    public init(maxNewTokens: Int = 256,
                temperature: Float = 1.0,
                topK: Int? = nil,
                topP: Float? = nil,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stopStrings: [String] = [],
                extraStopTokens: Set<Int32> = []) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stopStrings = stopStrings
        self.extraStopTokens = extraStopTokens
    }

    public func validate() throws {
        guard maxNewTokens > 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "maxNewTokens must be greater than zero")
        }
        guard temperature.isFinite, temperature >= 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "temperature must be finite and nonnegative")
        }
        if let topK, !(1...256).contains(topK) {
            throw GeneratorError.invalidGenerationConfig(
                "topK must be between 1 and 256")
        }
        if let topP, (!topP.isFinite || topP <= 0 || topP > 1) {
            throw GeneratorError.invalidGenerationConfig(
                "topP must be greater than zero and at most one")
        }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw GeneratorError.invalidGenerationConfig(
                "topP below one requires topK; full-vocabulary nucleus sampling is not implemented")
        }
    }

}

/// Which path a `sample(...)` call took.
enum SamplePath: Sendable, Equatable {
    case greedyGPU
    case gpuSampled
    case hostPenalty
    /// GEN-7: the first draw was rejected by the constraint and the token
    /// reported came from the masked redraw.
    case constraintResampled
}

/// One draw with the GEN-7 constraint hook: draw normally, probe the drawn
/// token, and only on a rejection mask the vocabulary and draw again.
///
/// `constraint` nil is the unconstrained path and delegates to the plain
/// `sampleOnce` unchanged, so a caller that passes no constraint draws the same
/// token, from the same seed, as it did before this hook existed.
///
/// The rejection path costs, on top of the draw that was thrown away: one
/// `fillAllowedMask` over the vocabulary (the constraint's own cost — for a
/// grammar this is the expensive part), one host pass that rebuilds `probs` as
/// the masked distribution (~V byte reads, V FP16 writes, and one `tanh`+`exp`
/// per *allowed* id), and one extra command buffer holding the draw kernel
/// alone. At the pinned vocabulary that is well under a millisecond of host
/// work plus one round trip — paid per rejected token, never per token.
func sampleOnce(scratch: RawCompletionScratch, context: MetalContext,
                history: [Int32], config: GenerationConfig, position: Int,
                constraint gate: ConstraintGate?) throws -> (id: Int32, path: SamplePath) {
    let drawn = try sampleOnce(scratch: scratch, context: context,
                              history: history, config: config, position: position)
    let path = scratch.sampler.lastPath
    guard let gate, !gate.allows(drawn) else { return (drawn, path) }

    let allowed = try scratch.sampler.writeMaskedProbs(logits: scratch.logits,
                                                       probs: scratch.probs,
                                                       gate: gate)
    guard allowed > 0 else {
        throw GenerationConstraintError.noAllowedToken(position: position)
    }
    guard let cb = context.queue.makeCommandBuffer() else { throw MetalError.noDevice }
    // Only the draw: `probs` already holds the masked distribution, and the
    // repetition penalty was applied in place to `logits` by the first draw —
    // running the front end again would penalize the same history twice.
    scratch.sampler.encodeDraw(commandBuffer: cb, probs: scratch.probs,
                               config: config, position: position,
                               outToken: scratch.outToken)
    cb.commit(); cb.waitUntilCompleted()
    try checkCommandBufferError(cb.error)
    let redrawn = Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
    // The probe already recorded this step from the first draw; recording the
    // masked distribution too would double-count the position.
    guard gate.allows(redrawn) else {
        throw GenerationConstraintError.maskedDrawRejected(position: position,
                                                           tokenID: redrawn)
    }
    scratch.sampler.noteConstraintResample()
    return (redrawn, .constraintResampled)
}

/// Turns `GenerationConfig` + a logits buffer into one token id, staying
/// GPU-resident wherever the kernels allow.
///
/// The built `sample` kernel already does temperature / top-k / top-p / seeded
/// draw / greedy argmax on GPU reading softmaxed probs, so this type's job is:
/// (1) run the softcap+softmax front-end (`logit_softcap_softmax`), (2) apply
/// repetition penalty — the one policy that needs `history` random access — as
/// a single in-place CPU pass over the (shared) logits before the front-end,
/// and (3) derive a per-position seed so a fixed `seed` is reproducible across
/// token positions.
///
/// The chosen id lands in a 1-element UInt32 buffer. The generation loop reads
/// that value after the command buffer completes.
///
/// Truncation follows mlx-lm's sampler order: Top-P is computed from the
/// model's full probability distribution, Top-K caps that surviving set, and
/// temperature is applied only to the final categorical draw.
final class Sampler {
    private let softcap: LogitSoftcapSoftmax
    private let sampleKernel: Sample
    private let topK64Kernel: SampleTopK64
    let vocab: Int
    private let logitSoftcap: Float
    /// Path of the most recent `sample(...)`, so a caller that wraps the draw
    /// (the GEN-7 rejection path) can report it without re-deriving it.
    private(set) var lastPath: SamplePath = .greedyGPU
    /// Reused across rejections so a masked draw allocates nothing. Stays empty
    /// for a run with no constraint.
    private var allowedMask: [Bool] = []

    init(context: MetalContext, vocab: Int = 262_144,
                logitSoftcap: Float = 30.0) throws {
        self.softcap = try LogitSoftcapSoftmax(context: context)
        self.sampleKernel = try Sample(context: context)
        self.topK64Kernel = try SampleTopK64(context: context, vocab: vocab)
        self.vocab = vocab
        self.logitSoftcap = logitSoftcap
    }

    /// Encode the sampler onto `commandBuffer`. `logits` is FP16 [vocab],
    /// post-lm_head and pre-softcap, in a `.storageModeShared` buffer (the
    /// repetition-penalty path edits it in place). `probs` is a preallocated
    /// FP16 [vocab] scratch. `outToken` holds one UInt32. `position` indexes the
    /// per-position seed advance. Returns the path taken.
    @discardableResult
    func sample(commandBuffer: MTLCommandBuffer,
                       logits: MTLBuffer,
                       probs: MTLBuffer,
                       history: [Int32],
                       config: GenerationConfig,
                       position: Int,
                       outToken: MTLBuffer) -> SamplePath {
        let v = UInt32(vocab)

        let appliedPenalty = config.repetitionPenalty != 1.0 && !history.isEmpty
        if appliedPenalty {
            applyRepetitionPenaltyInPlace(logits: logits,
                                          history: history,
                                          penalty: config.repetitionPenalty)
        }

        softcap.encode(commandBuffer: commandBuffer,
                       logits: logits, probs: probs, v: v, softcap: logitSoftcap)

        let drawn = encodeDraw(commandBuffer: commandBuffer, probs: probs,
                               config: config, position: position, outToken: outToken)

        let path: SamplePath = appliedPenalty ? .hostPenalty : drawn
        lastPath = path
        return path
    }

    /// Encode the draw kernel alone, from a `probs` buffer somebody else has
    /// already prepared. Split out of `sample` so the GEN-7 rejection path can
    /// redraw from a masked distribution without re-running the front end.
    @discardableResult
    func encodeDraw(commandBuffer: MTLCommandBuffer,
                    probs: MTLBuffer,
                    config: GenerationConfig,
                    position: Int,
                    outToken: MTLBuffer) -> SamplePath {
        let isGreedy = config.temperature == 0
        let seed = Self.seedFor(config: config, position: position)
        if config.temperature > 0,
           config.topK == 64 {
            topK64Kernel.encode(commandBuffer: commandBuffer,
                                probs: probs,
                                outToken: outToken,
                                temperature: config.temperature,
                                topP: config.topP ?? 1.0,
                                seed: seed)
        } else {
            sampleKernel.encode(commandBuffer: commandBuffer,
                                probs: probs, outToken: outToken, v: UInt32(vocab),
                                temperature: isGreedy ? 0.0 : config.temperature,
                                topK: UInt32(config.topK ?? 0),
                                topP: config.topP ?? 1.0,
                                seed: seed,
                                position: UInt32(position))
        }
        return isGreedy ? .greedyGPU : .gpuSampled
    }

    func noteConstraintResample() { lastPath = .constraintResampled }

    // MARK: - Constraint mask (GEN-7)

    /// Rebuild `probs` as the constraint-masked distribution and return how many
    /// ids survived. Returns 0 when the constraint allows nothing; the caller
    /// turns that into `GenerationConstraintError.noAllowedToken`.
    ///
    /// **Why the host, and why `probs` rather than `logits`.** The reference
    /// masks by writing `-INFINITY` into the candidate logits
    /// (`llama_grammar_apply_impl`) and letting the softmax turn that into an
    /// exact zero. That does not carry over here: this runtime's front end is
    /// the *fused* `logit_softcap_softmax`, and softcap is `30*tanh(z/30)`,
    /// which maps `-inf` — and equally the most negative finite Float16 — to
    /// exactly `-softcap`. A masked id would therefore keep the weight
    /// `exp(-softcap - m)` relative to the surviving maximum `m`, which is
    /// negligible when the model is confident and catastrophic when it is not
    /// (with the whole vocabulary at `-softcap` and the allowed ids only a few
    /// units above it, the rejected mass dominates the draw). No sentinel
    /// value fixes that, because `tanh` has already floored it. So the mask is
    /// applied where it is exact: the masked softmax is computed here, over the
    /// allowed ids only, and rejected ids are written as a literal zero, which
    /// the draw kernel skips (`!(p > 0)`) and the greedy argmax can never
    /// prefer. Normalizing over the allowed set is also what keeps top-p
    /// meaning what it means in the reference — the nucleus is taken from the
    /// *masked* distribution.
    ///
    /// Cost: one pass to find the masked maximum, one to sum, one to write, and
    /// `tanh`/`exp` only for allowed ids. `expf` here against the kernel's
    /// `fast::exp` there is a last-bit difference in FP16 probabilities.
    func writeMaskedProbs(logits: MTLBuffer, probs: MTLBuffer,
                          gate: ConstraintGate) throws -> Int {
        if allowedMask.count != vocab {
            allowedMask = [Bool](repeating: false, count: vocab)
        }
        var allowedCount = 0
        try allowedMask.withUnsafeMutableBufferPointer { mask in
            try gate.fillAllowedMask(mask)
            allowedCount = maskedSoftmax(logits: logits, probs: probs, allowed: mask)
        }
        return allowedCount
    }

    private func maskedSoftmax(logits: MTLBuffer, probs: MTLBuffer,
                               allowed: UnsafeMutableBufferPointer<Bool>) -> Int {
        let src = logits.contents().bindMemory(to: Float16.self, capacity: vocab)
        let dst = probs.contents().bindMemory(to: Float16.self, capacity: vocab)
        var m = -Float.infinity
        var count = 0
        for i in 0..<vocab where allowed[i] {
            count += 1
            let z = softcapped(Float(src[i]))
            if z > m { m = z }
        }
        guard count > 0 else { return 0 }
        var d: Float = 0
        for i in 0..<vocab where allowed[i] {
            d += expf(softcapped(Float(src[i])) - m)
        }
        // `d >= 1`: the id that set `m` contributes exp(0).
        let invD = 1 / d
        for i in 0..<vocab {
            dst[i] = allowed[i]
                ? Float16(expf(softcapped(Float(src[i])) - m) * invD)
                : 0
        }
        return count
    }

    /// The same cap the front-end kernel applies, on the host.
    private func softcapped(_ z: Float) -> Float {
        logitSoftcap > 0 ? logitSoftcap * tanhf(z / logitSoftcap) : z
    }

    // MARK: - Repetition penalty (host, in place)

    /// HF convention: for each token id seen in `history`, a positive logit is
    /// divided by `penalty`, a negative logit multiplied. Edits the shared
    /// `logits` buffer in place — no full-buffer copy, only the unique history
    /// entries are touched (counted for the audit).
    ///
    /// The penalty must act on the POST-softcap logit (HF applies it to the
    /// model's output logits, and Gemma's output includes the 30*tanh(z/30)
    /// cap). Real Gemma 4 raw logits reach the hundreds, deep in tanh
    /// saturation, where dividing the raw value by 1.1 moves the capped logit
    /// by ~nothing — the penalty silently no-ops on exactly the
    /// high-confidence tokens that form repetition loops. So: softcap the raw
    /// value, penalize, and invert through atanh so the downstream
    /// softcap+softmax kernel reproduces the penalized capped logit.
    private func applyRepetitionPenaltyInPlace(logits: MTLBuffer,
                                               history: [Int32],
                                               penalty: Float) {
        let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocab)
        var seen = Set<Int32>()
        seen.reserveCapacity(history.count)
        for id in history {
            guard id >= 0 && Int(id) < vocab, seen.insert(id).inserted else { continue }
            let i = Int(id)
            let z = Float(ptr[i])
            let penalized: Float
            if logitSoftcap > 0 {
                let capped = logitSoftcap * tanhf(z / logitSoftcap)
                let cappedPenalized = capped > 0 ? capped / penalty : capped * penalty
                // A saturated negative logit times the penalty can leave the
                // softcap's open interval; clamp inside it so atanh stays
                // finite.
                let limit = logitSoftcap * 0.9999
                let clamped = max(min(cappedPenalized, limit), -limit)
                penalized = logitSoftcap * atanhf(clamped / logitSoftcap)
            } else {
                penalized = z > 0 ? z / penalty : z * penalty
            }
            ptr[i] = Float16(penalized)
        }
    }

    // MARK: - Seed

    /// Deterministic per-position seed when `config.seed != nil` so a fixed seed
    /// reproduces across token positions; clock-derived (non-zero) otherwise.
    /// xorshift64 in the kernel has a fixed point at 0, so we never emit 0.
    static func seedFor(config: GenerationConfig, position: Int) -> UInt64 {
        if let s = config.seed {
            let mixed = Self.splitmix64(s &+ UInt64(bitPattern: Int64(position)))
            return mixed == 0 ? 0x9E3779B97F4A7C15 : mixed
        }
        var t = timespec()
        clock_gettime(CLOCK_MONOTONIC, &t)
        let raw = UInt64(bitPattern: Int64(t.tv_nsec)) &* 0x9E3779B97F4A7C15
            &+ UInt64(bitPattern: Int64(t.tv_sec))
        return raw == 0 ? 0x9E3779B97F4A7C15 : raw
    }

    private static func splitmix64(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
