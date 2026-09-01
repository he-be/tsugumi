import Foundation
import Metal

/// The Gemma 4 vision tower, end to end: preprocessed patches in, soft tokens
/// out (`PLAN_VISION.md` §5 V4).
///
/// The tower runs **once per image, before the prefill chunk loop** (§4-4), so
/// none of this is on the decode path and none of it touches the KV cache. The
/// working set is allocated once for the largest supported image and reused —
/// about 84 MB at the 280-soft-token maximum — while the soft tokens themselves
/// are a fresh per-image buffer, since a prompt may carry several images and
/// each one's output has to survive until the scatter.
///
/// Everything here is arranged as it is upstream (`modeling_gemma4.py`):
///
///     h  = patch_embedder(pixels)                                  §2-2
///     for each of 27 layers:                                       §2-3
///         h += post_attention_layernorm(attn(input_layernorm(h)))
///         h += post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
///     pooled = standardize(avgpool_3x3(h) * sqrt(hidden))          §2-5
///     soft   = projector(rmsnorm_no_scale(pooled))                 §2-6
///
/// Attention is full — every patch sees every patch (`is_causal = False`,
/// `scaling = 1.0`) — and there is no padding to mask, because images are
/// processed one at a time.
package final class VisionTower {
    package typealias Fault = VisionTowerFault

    package let config: VisionConfig
    package let textHiddenSize: Int
    /// Largest patch count the scratch was sized for.
    package let maxPatchCount: Int

    private let device: MTLDevice
    private let weights: VisionWeights

    private let qmm: VisionBF16QMM
    private let patchEmbed: VisionPatchEmbed
    private let qkNormRoPE: VisionQKNormRoPE2D
    private let attention: VisionAttentionFull
    private let mlpActivation: VisionMLPActivation
    private let normResidualAdd: VisionNormResidualAdd
    private let poolStandardize: VisionPoolStandardize
    private let rmsNorm: PrefillRMSNorm
    private let perHeadNorm: PrefillPerHeadNorm

    /// Scratch, sized for `maxPatchCount`. Named for what they hold at the
    /// point of use; several are reused once their previous contents are dead,
    /// which is noted at each reuse in `encode`.
    private let scaled: MTLBuffer      // [P, patchDim]
    private let hidden: MTLBuffer      // [P, hidden] — the residual stream
    private let normed: MTLBuffer      // [P, hidden]
    private let qBuffer: MTLBuffer     // [P, hidden]
    private let kBuffer: MTLBuffer     // [P, hidden]
    private let vBuffer: MTLBuffer     // [P, hidden]
    private let attnOut: MTLBuffer     // [P, hidden]
    private let gate: MTLBuffer        // [P, intermediate]
    private let up: MTLBuffer          // [P, intermediate]
    private let pooled: MTLBuffer      // [S, hidden]

    package init(context: MetalContext,
                 weights: VisionWeights,
                 maxPatchCount: Int? = nil) throws {
        let config = weights.config
        let patchBudget = maxPatchCount
            ?? config.maxSoftTokens * config.poolingKernelSize * config.poolingKernelSize
        precondition(patchBudget > 0, "patch budget must be positive")
        precondition(config.headDim <= VisionQKNormRoPE2D.maxHeadDim,
                     "head dim \(config.headDim) exceeds the vision kernels' "
                     + "\(VisionQKNormRoPE2D.maxHeadDim)")

        self.config = config
        self.textHiddenSize = weights.textHiddenSize
        self.maxPatchCount = patchBudget
        self.device = context.device
        self.weights = weights

        self.qmm = try VisionBF16QMM(context: context)
        self.patchEmbed = try VisionPatchEmbed(context: context)
        self.qkNormRoPE = try VisionQKNormRoPE2D(context: context)
        self.attention = try VisionAttentionFull(context: context)
        self.mlpActivation = try VisionMLPActivation(context: context)
        self.normResidualAdd = try VisionNormResidualAdd(context: context)
        self.poolStandardize = try VisionPoolStandardize(context: context)
        self.rmsNorm = try PrefillRMSNorm(context: context)
        self.perHeadNorm = try PrefillPerHeadNorm(context: context)

        let hiddenSize = config.hiddenSize
        let patchDim = config.patchSize * config.patchSize * 3
        let softBudget = patchBudget / (config.poolingKernelSize * config.poolingKernelSize)
        func scratch(_ elements: Int, _ label: String) throws -> MTLBuffer {
            guard let buffer = context.device.makeBuffer(
                length: elements * MemoryLayout<Float16>.size,
                options: .storageModePrivate) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = "vision.\(label)"
            return buffer
        }
        self.scaled = try scratch(patchBudget * patchDim, "scaled")
        self.hidden = try scratch(patchBudget * hiddenSize, "hidden")
        self.normed = try scratch(patchBudget * hiddenSize, "normed")
        self.qBuffer = try scratch(patchBudget * hiddenSize, "q")
        self.kBuffer = try scratch(patchBudget * hiddenSize, "k")
        self.vBuffer = try scratch(patchBudget * hiddenSize, "v")
        self.attnOut = try scratch(patchBudget * hiddenSize, "attn")
        self.gate = try scratch(patchBudget * config.intermediateSize, "gate")
        self.up = try scratch(patchBudget * config.intermediateSize, "up")
        self.pooled = try scratch(max(softBudget, 1) * hiddenSize, "pooled")
    }

    /// Bytes of reusable scratch this tower holds, for the memory budget
    /// (`PLAN_VISION.md` §3-3). Excludes the per-image soft-token buffers.
    package var scratchBytes: Int {
        [scaled, hidden, normed, qBuffer, kBuffer, vBuffer, attnOut, gate, up, pooled]
            .reduce(0) { $0 + $1.length }
    }

    /// Number of patches this tower can take: the resize budget, not the image.
    package func accepts(_ geometry: VisionImageGeometry) -> Bool {
        geometry.patchCount <= maxPatchCount
    }

    /// Encode one image's tower pass into `commandBuffer`.
    ///
    /// `pixels` holds `[patchCount, 768]` FP16 patch rows in the layout
    /// `VisionImagePreprocessor` produces. The returned buffer is freshly
    /// allocated and owned by the caller; the scratch is not, so two images
    /// cannot be in flight on this tower at once.
    @discardableResult
    package func encode(commandBuffer: MTLCommandBuffer,
                        pixels: MTLBuffer,
                        pixelsOffset: Int = 0,
                        geometry: VisionImageGeometry,
                        fault: Fault = .none,
                        probes: VisionTowerProbes? = nil) throws -> VisionTowerOutput {
        guard accepts(geometry) else {
            throw VisionError.patchBudgetExceeded(width: geometry.targetWidth,
                                                  height: geometry.targetHeight,
                                                  patches: geometry.patchCount,
                                                  budget: maxPatchCount)
        }

        let hiddenSize = config.hiddenSize
        let patchCount = geometry.patchCount
        let patchDim = config.patchSize * config.patchSize * 3
        let softTokenCount = geometry.softTokenCount
        let eps = Float(config.rmsNormEps)
        let theta = Float(config.ropeTheta)

        // The one place the fault is applied: every kernel that needs to know
        // where a patch sits reads the grid from here, so transposing it here
        // transposes the position embedding, the 2D RoPE and the pooling
        // together — which is what a grid plumbed through wrongly would do.
        let gridWide = fault == .gridTransposed ? geometry.patchesHigh : geometry.patchesWide
        let gridHigh = fault == .gridTransposed ? geometry.patchesWide : geometry.patchesHigh

        // Under `.layerWeightsPinned` every layer reads layer 0's weights, so a
        // per-layer accessor that ignored its argument would look identical.
        func weightLayer(_ layer: Int) -> Int { fault == .layerWeightsPinned ? 0 : layer }

        guard let outputBuffer = device.makeBuffer(
            length: softTokenCount * textHiddenSize * MemoryLayout<Float16>.size,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        outputBuffer.label = "vision.softTokens"

        // -- Patch embedder (§2-2)
        patchEmbed.encodePrescale(commandBuffer: commandBuffer,
                                  x: pixels, xOffset: pixelsOffset,
                                  out: scaled,
                                  count: patchCount * patchDim)
        let projection = try weights.patchProjection
        qmm.encode(commandBuffer: commandBuffer,
                   weights: projection.buffer, weightsOffset: Int(projection.offset),
                   x: scaled, y: hidden,
                   t: patchCount, n: hiddenSize, k: patchDim)
        let table = try weights.positionTable
        patchEmbed.encodePositionAdd(commandBuffer: commandBuffer,
                                     h: hidden,
                                     table: table.buffer, tableOffset: Int(table.offset),
                                     patchCount: patchCount, d: hiddenSize,
                                     patchesWide: gridWide,
                                     tableLength: config.positionEmbeddingSize)
        probes?.capture(.patchEmbed, from: hidden,
                        bytes: patchCount * hiddenSize * MemoryLayout<Float16>.size,
                        commandBuffer: commandBuffer)

        // -- Encoder layers (§2-3)
        for layer in 0..<config.numLayers {
            let w = weightLayer(layer)

            let inputNorm = try weights.inputNorm(layer: w)
            rmsNorm.encodeBF16W(commandBuffer: commandBuffer,
                                x: hidden,
                                weight: inputNorm.buffer, weightOffset: Int(inputNorm.offset),
                                out: normed,
                                t: UInt32(patchCount), d: UInt32(hiddenSize), eps: eps)

            let qkv: [(TensorView, MTLBuffer)] = [
                (try weights.qProj(layer: w), qBuffer),
                (try weights.kProj(layer: w), kBuffer),
                (try weights.vProj(layer: w), vBuffer),
            ]
            for (projView, destination) in qkv {
                qmm.encode(commandBuffer: commandBuffer,
                           weights: projView.buffer, weightsOffset: Int(projView.offset),
                           x: normed, y: destination,
                           t: patchCount, n: hiddenSize, k: hiddenSize)
            }

            let qNorm = try weights.qNorm(layer: w)
            let kNorm = try weights.kNorm(layer: w)
            qkNormRoPE.encode(commandBuffer: commandBuffer,
                              q: qBuffer, k: kBuffer, v: vBuffer,
                              qWeight: qNorm.buffer, qWeightOffset: Int(qNorm.offset),
                              kWeight: kNorm.buffer, kWeightOffset: Int(kNorm.offset),
                              patchCount: patchCount,
                              headDim: config.headDim,
                              numHeads: config.numHeads,
                              patchesWide: gridWide,
                              theta: theta, eps: eps)

            attention.encode(commandBuffer: commandBuffer,
                             q: qBuffer, k: kBuffer, v: vBuffer, out: attnOut,
                             patchCount: patchCount,
                             headDim: config.headDim,
                             numHeads: config.numHeads,
                             scale: 1.0)

            // `normed` is dead once Q/K/V are projected, so the output
            // projection lands there rather than in a tenth buffer.
            let oProj = try weights.oProj(layer: w)
            qmm.encode(commandBuffer: commandBuffer,
                       weights: oProj.buffer, weightsOffset: Int(oProj.offset),
                       x: attnOut, y: normed,
                       t: patchCount, n: hiddenSize, k: hiddenSize)
            let postAttention = try weights.postAttentionNorm(layer: w)
            normResidualAdd.encode(commandBuffer: commandBuffer,
                                   hidden: hidden, x: normed,
                                   weight: postAttention.buffer,
                                   weightOffset: Int(postAttention.offset),
                                   t: patchCount, d: hiddenSize, eps: eps)

            let preFFN = try weights.preFeedForwardNorm(layer: w)
            rmsNorm.encodeBF16W(commandBuffer: commandBuffer,
                                x: hidden,
                                weight: preFFN.buffer, weightOffset: Int(preFFN.offset),
                                out: normed,
                                t: UInt32(patchCount), d: UInt32(hiddenSize), eps: eps)
            let gateProj = try weights.gateProj(layer: w)
            qmm.encode(commandBuffer: commandBuffer,
                       weights: gateProj.buffer, weightsOffset: Int(gateProj.offset),
                       x: normed, y: gate,
                       t: patchCount, n: config.intermediateSize, k: hiddenSize)
            let upProj = try weights.upProj(layer: w)
            qmm.encode(commandBuffer: commandBuffer,
                       weights: upProj.buffer, weightsOffset: Int(upProj.offset),
                       x: normed, y: up,
                       t: patchCount, n: config.intermediateSize, k: hiddenSize)
            // Elementwise, one thread per element, so writing back over `gate`
            // is safe and saves a third [P, 4304] buffer.
            mlpActivation.encode(commandBuffer: commandBuffer,
                                 gate: gate, up: up, out: gate,
                                 count: patchCount * config.intermediateSize)
            let downProj = try weights.downProj(layer: w)
            qmm.encode(commandBuffer: commandBuffer,
                       weights: downProj.buffer, weightsOffset: Int(downProj.offset),
                       x: gate, y: normed,
                       t: patchCount, n: hiddenSize, k: config.intermediateSize)
            let postFFN = try weights.postFeedForwardNorm(layer: w)
            normResidualAdd.encode(commandBuffer: commandBuffer,
                                   hidden: hidden, x: normed,
                                   weight: postFFN.buffer, weightOffset: Int(postFFN.offset),
                                   t: patchCount, d: hiddenSize, eps: eps)

            probes?.capture(.layer(layer), from: hidden,
                            bytes: patchCount * hiddenSize * MemoryLayout<Float16>.size,
                            commandBuffer: commandBuffer)
        }

        // -- Pooling and standardization (§2-5)
        let stdScale = try weights.stdScale
        let stdBias = try weights.stdBias
        poolStandardize.encode(commandBuffer: commandBuffer,
                               h: hidden, out: pooled,
                               stdScale: stdScale.buffer, stdScaleOffset: Int(stdScale.offset),
                               stdBias: stdBias.buffer, stdBiasOffset: Int(stdBias.offset),
                               d: hiddenSize,
                               patchesWide: gridWide, patchesHigh: gridHigh,
                               kernelSize: config.poolingKernelSize,
                               rootHidden: Float(hiddenSize).squareRoot(),
                               standardize: config.standardize && fault != .standardizeSkipped)
        probes?.capture(.pooled, from: pooled,
                        bytes: softTokenCount * hiddenSize * MemoryLayout<Float16>.size,
                        commandBuffer: commandBuffer)

        // -- Projector (§2-6). The pre-projection norm has no learnable scale,
        // and — the trap this whole path exists to avoid — the result is *not*
        // multiplied by sqrt(text hidden): that scaling belongs to the text
        // embedding lookup alone.
        perHeadNorm.encodeNoScale(commandBuffer: commandBuffer,
                                  x: pooled,
                                  out: normed,
                                  queryCount: UInt32(softTokenCount),
                                  headDim: UInt32(hiddenSize),
                                  numHeads: 1,
                                  tokenStrideElements: UInt32(hiddenSize),
                                  eps: eps)
        let projector = try weights.projector
        qmm.encode(commandBuffer: commandBuffer,
                   weights: projector.buffer, weightsOffset: Int(projector.offset),
                   x: normed, y: outputBuffer,
                   t: softTokenCount, n: textHiddenSize, k: hiddenSize)

        return VisionTowerOutput(softTokens: outputBuffer,
                                 softTokenCount: softTokenCount,
                                 hiddenSize: textHiddenSize)
    }
}

/// `[softTokenCount, hiddenSize]` FP16 soft tokens for one image.
package struct VisionTowerOutput {
    package let softTokens: MTLBuffer
    package let softTokenCount: Int
    package let hiddenSize: Int
}

/// A deliberate deviation from the upstream algorithm.
///
/// These exist so a check can be shown to fail (`PLAN_VISION.md` §6-3): a
/// comparison that has never been seen to reject anything is not evidence that
/// the tower is right. They are never reachable from a normal run — the only
/// caller that passes anything but `.none` is the validation harness.
package enum VisionTowerFault: String, Sendable, CaseIterable {
    /// The upstream algorithm.
    case none
    /// Patch grid width and height exchanged, which moves every patch off the
    /// grid diagonal in the position embedding, the 2D RoPE and the pooling.
    case gridTransposed
    /// `std_scale` / `std_bias` not applied, so the tower's output keeps the
    /// pooler's scale.
    case standardizeSkipped
    /// Every layer reads layer 0's weights.
    case layerWeightsPinned
}

/// Intermediate captures for the layer-by-layer fixture comparison.
///
/// The tower keeps one working set, so an intermediate is gone as soon as the
/// next layer overwrites it. These buffers are blit copies taken inside the
/// same command buffer, which is what makes it possible to say *where* a
/// divergence starts instead of only that the soft tokens differ.
package final class VisionTowerProbes {
    package enum Stage: Hashable, Sendable {
        case patchEmbed
        case layer(Int)
        case pooled
    }

    private var buffers: [Stage: MTLBuffer] = [:]

    package init(device: MTLDevice,
                 hiddenSize: Int,
                 patchCount: Int,
                 softTokenCount: Int,
                 layers: [Int]) throws {
        func make(_ elements: Int, _ label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: elements * MemoryLayout<Float16>.size,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = "vision.probe.\(label)"
            return buffer
        }
        buffers[.patchEmbed] = try make(patchCount * hiddenSize, "patchEmbed")
        for layer in layers {
            buffers[.layer(layer)] = try make(patchCount * hiddenSize, "layer\(layer)")
        }
        buffers[.pooled] = try make(softTokenCount * hiddenSize, "pooled")
    }

    package func buffer(_ stage: Stage) -> MTLBuffer? { buffers[stage] }

    fileprivate func capture(_ stage: Stage,
                             from source: MTLBuffer,
                             bytes: Int,
                             commandBuffer: MTLCommandBuffer) {
        guard let destination = buffers[stage], destination.length >= bytes else { return }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(from: source, sourceOffset: 0,
                  to: destination, destinationOffset: 0, size: bytes)
        blit.endEncoding()
    }
}
