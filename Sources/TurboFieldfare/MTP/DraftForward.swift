import Foundation
import Metal

/// Optional stage capture for the M2 fixture comparison — the drafter's
/// equivalent of `VisionTowerProbes`.
public final class DraftProbes {
    public enum Stage: Int, CaseIterable, Sendable {
        case hPre = 0
        case layer0 = 1
        case layer1 = 2
        case layer2 = 3
        case layer3 = 4
        case hNorm = 5
    }

    public let buffers: [Stage: MTLBuffer]

    public init(device: MTLDevice, hiddenSize: Int, numLayers: Int) throws {
        var buffers: [Stage: MTLBuffer] = [:]
        for stage in Stage.allCases {
            let isLayer = stage.rawValue >= 1 && stage.rawValue <= numLayers
            guard stage == .hPre || stage == .hNorm || isLayer else { continue }
            guard let buffer = device.makeBuffer(length: hiddenSize * MemoryLayout<Float16>.size,
                                                 options: .storageModeShared) else {
                throw MetalError.noDevice
            }
            buffers[stage] = buffer
        }
        self.buffers = buffers
    }

    public func buffer(_ stage: Stage) -> MTLBuffer? { buffers[stage] }
}

/// One MTP drafter forward step (q_len = 1), assembled from the decode-path
/// kernels (`docs/mtp/02-RUNTIME-FIT.md` §2):
///
///   in   = concat(target_embed(tok) × √2816, target_last_hidden)   # [5632]
///   h    = pre_projection(in)                                        # [1024]
///   h    = 4 × (shared-KV attention + MLP + sandwich residual × layer_scalar)
///   hn   = rmsnorm(h, norm)
///   out  = post_projection(hn)                                       # [2816]
///   tok  = argmax(embed_tokens @ hn)                                 # greedy draft
///
/// The drafter has no K/V of its own: its sliding layers attend to the
/// target's last sliding layer's cache, its full layer to the last full one
/// (`sharedSlidingKVLayer` / `sharedFullKVLayer`). The K/V buffers are
/// therefore inputs, in the decode attention layout `[position, kvHeads, dim]`.
///
/// The layer tail is the *plain* Gemma sandwich (HF `Gemma4TextDecoderLayer`
/// with `enable_moe_block=False`): `h1 = h + rmsnorm(attn)`,
/// `out = (h1 + rmsnorm(mlp)) × layer_scalar` — one norm fewer than the
/// target's MoE tail, which is why this uses the vision tower's
/// norm-residual join instead of `FusedLayerTail`.
public final class DraftForward {
    /// Deliberate faults for the M2 detection-power cases: each one must make
    /// the fixture comparison fail by a wide margin, or the comparison is not
    /// evidence (`PLAN_VISION.md` §6-3 pattern).
    public enum Fault: String, CaseIterable, Sendable {
        case none
        /// RoPE applied at `position + 1`.
        case ropeOffByOne
        /// q_norm skipped (identity).
        case qNormDropped
        /// `1/sqrt(head_dim)` attention scale instead of Gemma's 1.0.
        case attentionScaleClassic
    }

    private let weights: DraftWeights
    private let config: DraftConfig
    private let attention: Attention
    private let rms: RMSNorm
    private let rope: RoPE
    private let int4: DequantInt4GEMV
    private let mlp: SharedExpertInt4
    private let normResidualAdd: VisionNormResidualAdd
    private let scalePSO: MTLComputePipelineState
    private let lmHead: LMHeadChainInt4

    // Scratch, all FP16.
    private let inputConcat: MTLBuffer    // 2 * backboneHidden
    private let hidden: MTLBuffer         // drafter hiddenSize
    private let normed: MTLBuffer         // hiddenSize
    private let qScratch: MTLBuffer       // numHeads * fullHeadDim (worst layer)
    private let attnOut: MTLBuffer        // hiddenSize
    private let mlpScratchA: MTLBuffer    // intermediateSize
    private let mlpScratchB: MTLBuffer    // intermediateSize
    private let mlpOut: MTLBuffer         // hiddenSize

    public init(context: MetalContext, weights: DraftWeights) throws {
        self.weights = weights
        self.config = weights.config
        self.attention = try Attention(context: context)
        self.rms = try RMSNorm(context: context)
        self.rope = try RoPE(context: context)
        self.int4 = try DequantInt4GEMV(context: context)
        self.mlp = try SharedExpertInt4(context: context)
        self.normResidualAdd = try VisionNormResidualAdd(context: context)
        self.scalePSO = try context.pipeline("scale_inplace_fp16")
        self.lmHead = try LMHeadChainInt4(context: context)

        let c = config
        let device = context.device
        func buffer(_ count: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: count * MemoryLayout<Float16>.size,
                                            options: .storageModePrivate) else {
                throw MetalError.noDevice
            }
            return b
        }
        self.inputConcat = try buffer(2 * c.backboneHiddenSize)
        self.hidden = try buffer(c.hiddenSize)
        self.normed = try buffer(c.hiddenSize)
        self.qScratch = try buffer(c.numHeads * c.fullHeadDim)
        self.attnOut = try buffer(c.hiddenSize)
        self.mlpScratchA = try buffer(c.intermediateSize)
        self.mlpScratchB = try buffer(c.intermediateSize)
        self.mlpOut = try buffer(c.hiddenSize)
    }

    /// One drafter step.
    ///
    /// - Parameters:
    ///   - targetEmbed: `[backboneHidden]` FP16 — the *target's* embedding of
    ///     the bonus token, already multiplied by √backboneHidden. The drafter
    ///     has no input embedding of its own (01 §5 Q6); production gets this
    ///     from the decode path's embed lookup, the M2 harness from a fixture.
    ///   - lastHidden: `[backboneHidden]` FP16 — the target's post-norm final
    ///     hidden for the bonus position.
    ///   - slidingK/V, fullK/V: decode-layout `[position+1, kvHeads, headDim]`
    ///     FP16 views of the target's shared-KV layers.
    ///   - position: absolute RoPE position of the bonus token; the reference
    ///     holds it constant across every draft step of a round.
    ///   - outLastHidden: `[backboneHidden]` FP16 — `post_projection` output.
    ///   - outToken: `UInt32` — greedy argmax over the tied embedding table.
    ///     Skipped when nil.
    ///   - outLogits: `[vocabSize]` FP16, optional — full logits for the
    ///     fixture comparison.
    public func encode(commandBuffer cb: MTLCommandBuffer,
                       targetEmbed: MTLBuffer, targetEmbedOffset: Int = 0,
                       lastHidden: MTLBuffer, lastHiddenOffset: Int = 0,
                       slidingK: MTLBuffer, slidingKOffset: Int = 0,
                       slidingV: MTLBuffer, slidingVOffset: Int = 0,
                       fullK: MTLBuffer, fullKOffset: Int = 0,
                       fullV: MTLBuffer, fullVOffset: Int = 0,
                       position: UInt32,
                       fault: Fault = .none,
                       outLastHidden: MTLBuffer,
                       outToken: MTLBuffer?,
                       outLogits: MTLBuffer? = nil,
                       probes: DraftProbes? = nil) throws {
        let c = config
        let backbone = c.backboneHiddenSize
        let hiddenSize = c.hiddenSize
        let eps = Float(c.rmsNormEps)
        let embedScale = Float(backbone).squareRoot()

        // in = concat(target_embed, last_hidden). The caller's scale (√2816)
        // is already folded into `targetEmbed` where production applies it
        // (the fused embed lookup), so the reference and the runtime agree on
        // *when* the multiplication happens.
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw MetalError.noDevice
        }
        let half = MemoryLayout<Float16>.size
        blit.copy(from: targetEmbed, sourceOffset: targetEmbedOffset * half,
                  to: inputConcat, destinationOffset: 0,
                  size: backbone * half)
        blit.copy(from: lastHidden, sourceOffset: lastHiddenOffset * half,
                  to: inputConcat, destinationOffset: backbone * half,
                  size: backbone * half)
        blit.endEncoding()

        let pre = try weights.preProjection
        int4.encode(commandBuffer: cb,
                    weights: pre.buffer, weightsOffset: Int(pre.offset),
                    scales: pre.buffer, scalesOffset: Int(pre.scaleOffset),
                    biases: pre.buffer, biasesOffset: Int(pre.biasOffset),
                    x: inputConcat,
                    y: hidden,
                    m: UInt32(hiddenSize), n: UInt32(2 * backbone))
        if let probe = probes?.buffer(.hPre) {
            copy(cb, from: hidden, to: probe, count: hiddenSize)
        }

        let seqLen = position + 1
        for L in 0..<c.numLayers {
            try encodeLayer(cb, layer: L, seqLen: seqLen, position: position,
                            slidingK: slidingK, slidingKOffset: slidingKOffset,
                            slidingV: slidingV, slidingVOffset: slidingVOffset,
                            fullK: fullK, fullKOffset: fullKOffset,
                            fullV: fullV, fullVOffset: fullVOffset,
                            fault: fault, eps: eps)
            if let probe = DraftProbes.Stage(rawValue: L + 1).flatMap({ probes?.buffer($0) }) {
                copy(cb, from: hidden, to: probe, count: hiddenSize)
            }
        }

        rms.encodeBF16W(commandBuffer: cb,
                        x: hidden,
                        weight: try weights.norm.buffer,
                        weightOffset: Int(try weights.norm.offset),
                        out: normed,
                        d: UInt32(hiddenSize),
                        eps: eps)
        if let probe = probes?.buffer(.hNorm) {
            copy(cb, from: normed, to: probe, count: hiddenSize)
        }

        let post = try weights.postProjection
        int4.encode(commandBuffer: cb,
                    weights: post.buffer, weightsOffset: Int(post.offset),
                    scales: post.buffer, scalesOffset: Int(post.scaleOffset),
                    biases: post.buffer, biasesOffset: Int(post.biasOffset),
                    x: normed,
                    y: outLastHidden,
                    m: UInt32(backbone), n: UInt32(hiddenSize))

        if let outToken {
            // The chain applies the final RMSNorm itself, so it is fed the
            // pre-norm residual — the same contract the decode path uses.
            lmHead.encodeGreedyDecode(commandBuffer: cb,
                                      hidden: hidden,
                                      normWeight: try weights.norm.buffer,
                                      normOffset: Int(try weights.norm.offset),
                                      weights: try weights.embedTokens.buffer,
                                      weightsOffset: Int(try weights.embedTokens.offset),
                                      scales: try weights.embedTokens.buffer,
                                      scalesOffset: Int(try weights.embedTokens.scaleOffset),
                                      biases: try weights.embedTokens.buffer,
                                      biasesOffset: Int(try weights.embedTokens.biasOffset),
                                      outToken: outToken,
                                      d: UInt32(hiddenSize),
                                      vocab: UInt32(c.vocabSize),
                                      rmsEps: eps)
        }
        if let outLogits {
            let embed = try weights.embedTokens
            int4.encode(commandBuffer: cb,
                        weights: embed.buffer, weightsOffset: Int(embed.offset),
                        scales: embed.buffer, scalesOffset: Int(embed.scaleOffset),
                        biases: embed.buffer, biasesOffset: Int(embed.biasOffset),
                        x: normed,
                        y: outLogits,
                        m: UInt32(c.vocabSize), n: UInt32(hiddenSize))
        }
        _ = embedScale // documented above; the scale is the caller's contract
    }

    private func encodeLayer(_ cb: MTLCommandBuffer,
                             layer L: Int,
                             seqLen: UInt32,
                             position: UInt32,
                             slidingK: MTLBuffer, slidingKOffset: Int,
                             slidingV: MTLBuffer, slidingVOffset: Int,
                             fullK: MTLBuffer, fullKOffset: Int,
                             fullV: MTLBuffer, fullVOffset: Int,
                             fault: Fault,
                             eps: Float) throws {
        let c = config
        let hiddenSize = c.hiddenSize
        let isFull = c.fullAttentionLayerMask[L] == 1
        let headDimL = c.headDim(forLayer: L)
        let numKVL = isFull ? c.numFullKVHeads : c.numKVHeads
        let qDim = c.numHeads * headDimL

        rms.encodeBF16W(commandBuffer: cb,
                        x: hidden,
                        weight: try weights.inputNorm(layer: L).buffer,
                        weightOffset: Int(try weights.inputNorm(layer: L).offset),
                        out: normed,
                        d: UInt32(hiddenSize),
                        eps: eps)

        let q = try weights.qProj(layer: L)
        int4.encode(commandBuffer: cb,
                    weights: q.buffer, weightsOffset: Int(q.offset),
                    scales: q.buffer, scalesOffset: Int(q.scaleOffset),
                    biases: q.buffer, biasesOffset: Int(q.biasOffset),
                    x: normed,
                    y: qScratch,
                    m: UInt32(qDim), n: UInt32(hiddenSize))

        if fault != .qNormDropped {
            let qNorm = try weights.qNorm(layer: L)
            rms.encodeBF16WPerHead(commandBuffer: cb,
                                   x: qScratch,
                                   weight: qNorm.buffer,
                                   weightOffset: Int(qNorm.offset),
                                   out: qScratch,
                                   headDim: UInt32(headDimL),
                                   numHeads: c.numHeads,
                                   eps: eps)
        }

        let ropePosition = fault == .ropeOffByOne ? position + 1 : position
        if isFull {
            rope.encodeProportionalNeox(commandBuffer: cb,
                                        data: qScratch,
                                        position: ropePosition,
                                        headDim: UInt32(headDimL),
                                        numHeads: UInt32(c.numHeads),
                                        rotatedPairs: UInt32(
                                            Double(headDimL) * c.partialRotaryFactor / 2.0),
                                        theta: Float(c.fullRopeTheta))
        } else {
            rope.encodeDefaultNeox(commandBuffer: cb,
                                   data: qScratch,
                                   position: ropePosition,
                                   headDim: UInt32(headDimL),
                                   numHeads: UInt32(c.numHeads),
                                   theta: Float(c.ropeTheta))
        }

        let scale: Float = fault == .attentionScaleClassic
            ? 1.0 / Float(headDimL).squareRoot()
            : 1.0
        if isFull {
            attention.encodeFull(commandBuffer: cb,
                                 q: qScratch,
                                 k: fullK, kOffset: fullKOffset,
                                 v: fullV, vOffset: fullVOffset,
                                 out: attnOut,
                                 headDim: UInt32(headDimL),
                                 numQHeads: UInt32(c.numHeads),
                                 numKVHeads: UInt32(numKVL),
                                 seqLen: seqLen,
                                 scale: scale)
        } else {
            attention.encodeSWA(commandBuffer: cb,
                                q: qScratch,
                                k: slidingK, kOffset: slidingKOffset,
                                v: slidingV, vOffset: slidingVOffset,
                                out: attnOut,
                                headDim: UInt32(headDimL),
                                numQHeads: UInt32(c.numHeads),
                                numKVHeads: UInt32(numKVL),
                                seqLen: seqLen,
                                window: UInt32(c.slidingWindow),
                                scale: scale)
        }

        let o = try weights.oProj(layer: L)
        int4.encode(commandBuffer: cb,
                    weights: o.buffer, weightsOffset: Int(o.offset),
                    scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                    biases: o.buffer, biasesOffset: Int(o.biasOffset),
                    x: attnOut,
                    y: normed,
                    m: UInt32(hiddenSize), n: UInt32(qDim))

        // h1 = hidden + rmsnorm(attn, post_attention_norm)
        let postAttn = try weights.postAttentionNorm(layer: L)
        normResidualAdd.encode(commandBuffer: cb,
                               hidden: hidden,
                               x: normed,
                               weight: postAttn.buffer,
                               weightOffset: Int(postAttn.offset),
                               t: 1, d: hiddenSize, eps: eps)

        // mlp on rmsnorm(h1, pre_feedforward_norm)
        rms.encodeBF16W(commandBuffer: cb,
                        x: hidden,
                        weight: try weights.preFeedForwardNorm(layer: L).buffer,
                        weightOffset: Int(try weights.preFeedForwardNorm(layer: L).offset),
                        out: normed,
                        d: UInt32(hiddenSize),
                        eps: eps)
        try mlp.encode(commandBuffer: cb,
                       x: normed,
                       gate: try projection(try weights.gateProj(layer: L)),
                       up: try projection(try weights.upProj(layer: L)),
                       down: try projection(try weights.downProj(layer: L)),
                       y: mlpOut,
                       scratchGate: mlpScratchA,
                       scratchUp: mlpScratchB,
                       scratchAct: mlpScratchA)

        // out = (h1 + rmsnorm(mlp, post_feedforward_norm)) * layer_scalar
        let postFFN = try weights.postFeedForwardNorm(layer: L)
        normResidualAdd.encode(commandBuffer: cb,
                               hidden: hidden,
                               x: mlpOut,
                               weight: postFFN.buffer,
                               weightOffset: Int(postFFN.offset),
                               t: 1, d: hiddenSize, eps: eps)
        scaleInPlace(cb, x: hidden, scale: try weights.layerScalar(layer: L))
    }

    private func projection(_ view: TensorView) -> SharedExpertProjection {
        SharedExpertProjection(weights: view.buffer,
                               scales: view.buffer,
                               biases: view.buffer,
                               weightsOffset: Int(view.offset),
                               scalesOffset: Int(view.scaleOffset),
                               biasesOffset: Int(view.biasOffset),
                               rows: UInt32(view.shape.0),
                               cols: UInt32(view.shape.1))
    }

    private func scaleInPlace(_ cb: MTLCommandBuffer, x: MTLBuffer, scale: Float) {
        guard let encoder = cb.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(scalePSO)
        encoder.setBuffer(x, offset: 0, index: 0)
        var scaleVar = scale
        var count = UInt32(config.hiddenSize)
        encoder.setBytes(&scaleVar, length: MemoryLayout<Float>.size, index: 1)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
        let width = min(scalePSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: config.hiddenSize, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func copy(_ cb: MTLCommandBuffer, from: MTLBuffer, to: MTLBuffer, count: Int) {
        guard let blit = cb.makeBlitCommandEncoder() else { return }
        blit.copy(from: from, sourceOffset: 0, to: to, destinationOffset: 0,
                  size: count * MemoryLayout<Float16>.size)
        blit.endEncoding()
    }
}
