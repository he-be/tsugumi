import Foundation
import Metal

/// The truncated, temperature-reweighted categorical one draw is taken from —
/// small enough to hold by value (`top_k` caps it at 20 for the official Ornith
/// settings, `docs/qwen35moe/42-SAMPLING.md` §0 S1).
///
/// Speculative sampling needs the *probabilities*, not only the draw: the
/// accept test is `u <= p(d)/q(d)` and the rejection path samples from the
/// normalized residual `(p - q)+` (§2-2). That is why this family samples on
/// the host while Gemma draws inside the `sample` kernel — the kernel answers
/// with a token id and throws the distribution away.
public struct QwenCategorical: Sendable, Equatable {
    /// Token ids, descending by full-vocabulary probability mass.
    public let ids: [Int32]
    /// Weights over `ids`, normalized to sum to one **after** truncation and
    /// the temperature reweight.
    public let weights: [Float]

    public init(ids: [Int32], weights: [Float]) {
        precondition(ids.count == weights.count, "shape mismatch")
        self.ids = ids
        self.weights = weights
    }

    public var isEmpty: Bool { ids.isEmpty }

    /// This distribution's probability for `id`, or zero when truncation
    /// dropped it. Zero is a real answer here: it is what makes `p(d)/q(d)`
    /// reject a draft the target would never have produced.
    public func probability(of id: Int32) -> Float {
        guard let index = ids.firstIndex(of: id) else { return 0 }
        return weights[index]
    }

    /// Inverse-CDF draw for `u` in [0, 1). Mirrors the `sample` kernel's walk,
    /// including its fallback: a CDF that never crosses `u` because of FP
    /// rounding returns the first id (the argmax), never a sentinel.
    public func draw(u: Float) -> Int32 {
        guard let first = ids.first else { return 0 }
        var run: Float = 0
        for index in 0..<ids.count {
            run += weights[index]
            if u <= run { return ids[index] }
        }
        return first
    }

    /// `(self - other)+`, renormalized — the distribution a rejected draft is
    /// redrawn from (§2-2 step 4). Empty only if the two are identical, which
    /// the caller treats as "accept anything", never as a stop.
    public func residual(minus other: QwenCategorical) -> QwenCategorical {
        var keptIDs: [Int32] = []
        var keptWeights: [Float] = []
        var total: Float = 0
        for (index, id) in ids.enumerated() {
            let diff = weights[index] - other.probability(of: id)
            if diff > 0 {
                keptIDs.append(id)
                keptWeights.append(diff)
                total += diff
            }
        }
        guard total > 0 else { return QwenCategorical(ids: [], weights: []) }
        return QwenCategorical(ids: keptIDs, weights: keptWeights.map { $0 / total })
    }
}

/// The host half of this family's sampler: full-vocabulary softmax on the GPU,
/// truncation and draw here.
///
/// **The truncation is a mirror, not a new rule.** Every step below is the
/// `sample` kernel's (`Metal/Sampling/logit.metal`), in its order and with its
/// tie-breaks, because two samplers in one runtime that disagree about what
/// `top_p = 0.95` means is exactly the kind of silent divergence
/// `42-SAMPLING.md` exists to stop:
///
///   1. `top_p` truncates against the **full-vocabulary** normalized mass,
///      taking the shortest descending prefix whose cumulative mass reaches it
///      (the element that crosses the threshold is kept);
///   2. `top_k` caps the survivors;
///   3. temperature reweights the survivors as `p^(1/T)`;
///   4. inverse-CDF draw with xorshift64*.
///
/// Steps 1 and 2 commute for the *set* they produce — both keep a prefix of the
/// descending order — so scanning for the top `k` alone is enough, and the
/// nucleus is then read off that prefix's cumulative mass. That is what makes
/// the whole thing one pass over the vocabulary instead of a sort.
///
/// **Softcap is off.** Qwen 3.5-MoE has no logit softcap
/// (`docs/qwen35moe/03-DESIGN.md` §2); passing Gemma's 30 here would flatten
/// every confident logit into the same value.
public final class QwenSampler {
    /// Scored vocabulary rows — the tokenizer's 248,077, not the padded size.
    public let vocab: Int
    private let softmax: LogitSoftcapSoftmax
    private let probsBuffer: MTLBuffer
    /// Reused mask storage for the constrained path, so a rejection allocates
    /// nothing after the first.
    private var allowedFlags: [Bool] = []
    /// The logits each probs row was softmaxed from, by row. `encodeProbabilities`
    /// is the only writer of `probsBuffer`, so this cannot name a row the probs
    /// did not come from. It is what a masked renormalization falls back to when
    /// the FP16 row has no allowed mass left to divide.
    private var rowLogits: [Int: (buffer: MTLBuffer, offset: Int)] = [:]

    public init(context: MetalContext, vocab: Int, maxRows: Int) throws {
        self.vocab = vocab
        self.softmax = try LogitSoftcapSoftmax(context: context)
        guard let probs = context.device.makeBuffer(
                  length: maxRows * vocab * MemoryLayout<Float16>.size,
                  options: .storageModeShared) else { throw MetalError.noDevice }
        probs.label = "qwen.sampler.probs"
        self.probsBuffer = probs
    }

    /// Full-vocabulary softmax for `rows` rows of logits, appended to a command
    /// buffer the caller is already building — the head's own, so the softmax
    /// costs no extra round trip.
    /// `destinationRow` names where the first row lands, so a caller that keeps
    /// several distributions alive at once — the speculative loop holds the
    /// draft's `q` across its verify pass — does not overwrite one with another.
    public func encodeProbabilities(commandBuffer: MTLCommandBuffer,
                             logits: MTLBuffer,
                             logitsOffset: Int = 0,
                             rows: Int,
                             destinationRow: Int = 0) {
        let rowBytes = vocab * MemoryLayout<Float16>.size
        for row in 0..<rows {
            rowLogits[destinationRow + row] = (logits, logitsOffset + row * rowBytes)
            softmax.encode(commandBuffer: commandBuffer,
                           logits: logits, logitsOffset: logitsOffset + row * rowBytes,
                           probs: probsBuffer,
                           probsOffset: (destinationRow + row) * rowBytes,
                           v: UInt32(vocab),
                           softcap: 0)
        }
    }

    /// The categorical for one row of the last `encodeProbabilities`.
    ///
    /// `gate` nil is the unconstrained path. With a gate, the distribution is
    /// renormalized over the allowed ids **before** the nucleus is taken, which
    /// is the rule `Sampler.writeMaskedProbs` already spells out for Gemma: a
    /// nucleus read off the unmasked mass would mean something different from
    /// what the reference means by `top_p` under a grammar.
    /// The unconstrained categorical. `ConstraintGate` is internal, so the
    /// gated form below is the module's; this is what a check (or any caller
    /// outside the module) can reach.
    public func categorical(row: Int, config: GenerationConfig) throws -> QwenCategorical {
        try categorical(row: row, config: config, gate: nil)
    }

    /// `reuseMask` skips the whole-vocabulary `fillAllowedMask` and reuses the
    /// one the last gated call left behind. Valid **only** while the gate has
    /// not accepted a token since — which is exactly the window the speculative
    /// loop needs, where `q` (the draft's distribution) and `p` (the target's
    /// at the same position) are asked in that order with nothing emitted in
    /// between (`docs/qwen35moe/42-SAMPLING.md` §2-3).
    ///
    /// `position` is the generation position, and is only ever used to name the
    /// place in `GenerationConstraintError`. It is not the row: the three rows
    /// (`target`, `speculative`, `draft`) are three distributions at one
    /// position, and reporting a row number as a position sent one real
    /// investigation to the wrong end of the loop.
    func categorical(row: Int,
                     config: GenerationConfig,
                     gate: ConstraintGate?,
                     reuseMask: Bool = false,
                     position: Int = 0) throws -> QwenCategorical {
        let base = probsBuffer.contents()
            .advanced(by: row * vocab * MemoryLayout<Float16>.size)
            .bindMemory(to: Float16.self, capacity: vocab)

        // Greedy is not a special case of the draw here — it is the absence of
        // one. Kept so evaluation (S4) runs the same code path as production
        // with one parameter changed.
        if config.temperature <= 0 {
            let best = try argmax(base, row: row, gate: gate, position: position)
            return QwenCategorical(ids: [best], weights: [1])
        }

        let k = min(config.topK ?? vocab, vocab)
        guard k > 0 else { return QwenCategorical(ids: [], weights: []) }

        var top: [(id: Int32, p: Float)] = []
        top.reserveCapacity(k + 1)
        var allowedMass: Float = 0

        if let gate {
            if !reuseMask || allowedFlags.count != vocab { try fillAllowed(gate) }
            var allowedCount = 0
            allowedFlags.withUnsafeBufferPointer { flags in
                for index in 0..<vocab where flags[index] {
                    allowedCount += 1
                    let p = Float(base[index])
                    guard p > 0 else { continue }
                    allowedMass += p
                    insert(&top, id: Int32(index), p: p, k: k)
                }
            }
            // GEN-7's error is "the mask allows nothing", and that is this
            // count — not the mass. The two used to be conflated.
            guard allowedCount > 0 else {
                throw GenerationConstraintError.noAllowedToken(position: position)
            }
            if allowedMass <= 0 {
                // The mask is not empty; every allowed id simply sits under
                // FP16's floor (2^-24) in this row's softmax, so there is
                // nothing left to renormalize. Go back to the logits, the way
                // the other family's `Sampler.maskedSoftmax` always does.
                return try maskedCategoricalFromLogits(row: row, config: config, k: k,
                                                       position: position)
            }
        } else {
            for index in 0..<vocab {
                let p = Float(base[index])
                guard p > 0 else { continue }
                insert(&top, id: Int32(index), p: p, k: k)
            }
            allowedMass = 1
        }
        guard !top.isEmpty else { return QwenCategorical(ids: [], weights: []) }
        return nucleus(top, mass: allowedMass, config: config)
    }

    /// Top-p against the (masked) full-vocabulary mass, then temperature, then
    /// normalize — the tail both renormalizations share.
    private func nucleus(_ top: [(id: Int32, p: Float)],
                         mass: Float,
                         config: GenerationConfig) -> QwenCategorical {
        // Top-p against the (masked) full-vocabulary mass, inclusive of the
        // element that crosses it.
        var kept = top.count
        if let topP = config.topP, topP > 0, topP < 1 {
            var cumulative: Float = 0
            for index in 0..<top.count {
                cumulative += top[index].p / mass
                if cumulative >= topP { kept = index + 1; break }
            }
        }

        // Temperature, then normalize. `p^(1/T)` is scale-invariant in the
        // mask's normalizer only up to the exponent, so the division by
        // `mass` happens first — the same order the masked softmax uses.
        let invT = 1 / config.temperature
        var weights = [Float](repeating: 0, count: kept)
        var total: Float = 0
        for index in 0..<kept {
            let p = top[index].p / mass
            let w = config.temperature == 1 ? p : powf(p, invT)
            weights[index] = w
            total += w
        }
        guard total > 0 else {
            return QwenCategorical(ids: [top[0].id], weights: [1])
        }
        return QwenCategorical(ids: (0..<kept).map { top[$0].id },
                               weights: weights.map { $0 / total })
    }

    /// The masked categorical rebuilt from this row's logits, for the one case
    /// the FP16 probabilities cannot answer.
    ///
    /// Mirrors `Sampler.maskedSoftmax`: the shift is the maximum over the
    /// **allowed** ids, so the id that set it contributes `exp(0)` and the
    /// denominator is at least 1. Nothing can underflow, and the ordering is
    /// the logits' own rather than a tie at zero.
    ///
    /// Reached only when the row left no allowed mass at all, so it changes no
    /// draw that had any: every currently-drawable position keeps its numbers.
    private func maskedCategoricalFromLogits(row: Int,
                                             config: GenerationConfig,
                                             k: Int,
                                             position: Int) throws -> QwenCategorical {
        guard let source = rowLogits[row] else {
            // No `encodeProbabilities` wrote this row, so there is no logit to
            // rank the allowed ids by. Refusing beats inventing an order.
            throw GenerationConstraintError.noAllowedToken(position: position)
        }
        let logits = source.buffer.contents()
            .advanced(by: source.offset)
            .bindMemory(to: Float16.self, capacity: vocab)

        var top: [(id: Int32, p: Float)] = []
        top.reserveCapacity(k + 1)
        var mass: Float = 0
        allowedFlags.withUnsafeBufferPointer { flags in
            var maxLogit = -Float.infinity
            for index in 0..<vocab where flags[index] {
                let z = Float(logits[index])
                if z > maxLogit { maxLogit = z }
            }
            for index in 0..<vocab where flags[index] {
                let p = expf(Float(logits[index]) - maxLogit)
                mass += p
                insert(&top, id: Int32(index), p: p, k: k)
            }
        }
        guard !top.isEmpty, mass > 0 else { return QwenCategorical(ids: [], weights: []) }
        return nucleus(top, mass: mass, config: config)
    }

    /// One uniform in [0, 1), from the same generator the `sample` kernel uses,
    /// so a seeded run is reproducible and a check can predict the draw.
    ///
    /// `stream` separates the uses that share a position: the draw itself, the
    /// speculative accept test, and the residual redraw are three different
    /// questions asked at the same index (§2-2).
    public static func uniform(config: GenerationConfig, position: Int, stream: UInt64) -> Float {
        var state = Sampler.seedFor(config: config, position: position) ^ (stream &* 0x9E37_79B9_7F4A_7C15)
        if state == 0 { state = 0x9E37_79B9_7F4A_7C15 }
        // The kernel takes one step before drawing, for the same reason: a
        // small seed's first xorshift output is a few bits short of entropy.
        _ = Self.xorshift64(&state)
        let bits = UInt32(truncatingIfNeeded: Self.xorshift64(&state) >> 40)
        return Float(bits) * (1.0 / 16_777_216.0)
    }

    private static func xorshift64(_ state: inout UInt64) -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state &* 2_685_821_657_736_338_717
    }

    /// Keep `top` sorted descending by mass, at most `k` long, ties to the
    /// lower id — the kernel's repeated-argmax with its `simd_min` tie-break.
    private func insert(_ top: inout [(id: Int32, p: Float)], id: Int32, p: Float, k: Int) {
        if top.count == k, let last = top.last,
           p < last.p || (p == last.p && id > last.id) { return }
        var index = top.count
        while index > 0, p > top[index - 1].p || (p == top[index - 1].p && id < top[index - 1].id) {
            index -= 1
        }
        top.insert((id, p), at: index)
        if top.count > k { top.removeLast() }
    }

    private func argmax(_ probs: UnsafeMutablePointer<Float16>,
                        row: Int,
                        gate: ConstraintGate?,
                        position: Int) throws -> Int32 {
        var best: Float = -.infinity
        var bestID: Int32 = -1
        if let gate {
            try fillAllowed(gate)
            allowedFlags.withUnsafeBufferPointer { flags in
                for index in 0..<vocab where flags[index] {
                    let p = Float(probs[index])
                    if p > best { best = p; bestID = Int32(index) }
                }
            }
            guard bestID >= 0 else {
                throw GenerationConstraintError.noAllowedToken(position: position)
            }
            if best <= 0 {
                // Every allowed id underflowed FP16, so `best` is a tie at zero
                // and `bestID` is just the lowest allowed id — an order the
                // model never expressed. Rank them by logit instead, the same
                // fallback the sampled path takes.
                return try maskedArgmaxFromLogits(row: row, position: position)
            }
        } else {
            for index in 0..<vocab {
                let p = Float(probs[index])
                if p > best { best = p; bestID = Int32(index) }
            }
            guard bestID >= 0 else {
                throw GenerationConstraintError.noAllowedToken(position: position)
            }
        }
        return bestID
    }

    /// The allowed id with the largest logit, ties to the lower id — the
    /// tie-break `insert` and the `sample` kernel's `simd_min` both use.
    private func maskedArgmaxFromLogits(row: Int, position: Int) throws -> Int32 {
        guard let source = rowLogits[row] else {
            throw GenerationConstraintError.noAllowedToken(position: position)
        }
        let logits = source.buffer.contents()
            .advanced(by: source.offset)
            .bindMemory(to: Float16.self, capacity: vocab)
        var best: Float = -.infinity
        var bestID: Int32 = -1
        allowedFlags.withUnsafeBufferPointer { flags in
            for index in 0..<vocab where flags[index] {
                let z = Float(logits[index])
                if z > best { best = z; bestID = Int32(index) }
            }
        }
        guard bestID >= 0 else {
            throw GenerationConstraintError.noAllowedToken(position: position)
        }
        return bestID
    }

    private func fillAllowed(_ gate: ConstraintGate) throws {
        if allowedFlags.count != vocab {
            allowedFlags = [Bool](repeating: false, count: vocab)
        }
        try allowedFlags.withUnsafeMutableBufferPointer { flags in
            try gate.fillAllowedMask(flags)
        }
    }
}
