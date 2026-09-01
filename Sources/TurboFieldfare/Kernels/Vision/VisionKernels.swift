import Foundation
import Metal

/// Host side of `Metal/Vision/vision.metal`.
///
/// The tower runs once per image, before the prefill chunk loop, so nothing
/// here is on the decode path and nothing here touches the KV cache. Each type
/// wraps one kernel and validates the shape assumptions the kernel makes rather
/// than letting a wrong dispatch produce plausible-looking numbers.

/// `Y[T, N] = X[T, K] * W[N, K]^T` with BF16 weights.
///
/// Every projection in the tower goes through this: patch embedding (K = 768),
/// Q/K/V/O (1152), the MLP (1152 -> 4304 -> 1152) and the multimodal projector
/// (1152 -> 2816). Unlike the INT4 prefill QMM there is no group constraint, so
/// K = 4304 — which is not a multiple of the 32-wide K tile — is a valid shape.
package final class VisionBF16QMM {
    package static let tileM = 64
    package static let tileN = 64
    package static let tileK = 32
    package static let threadsPerGroup = 128

    private let pso: MTLComputePipelineState

    package init(context: MetalContext) throws {
        self.pso = try context.pipeline("vision_bf16_qmm_f16")
        precondition(pso.maxTotalThreadsPerThreadgroup >= Self.threadsPerGroup,
                     "vision_bf16_qmm_f16 needs \(Self.threadsPerGroup) threads per group")
    }

    package func encode(commandBuffer: MTLCommandBuffer,
                        weights: MTLBuffer, weightsOffset: Int = 0,
                        x: MTLBuffer, xOffset: Int = 0,
                        y: MTLBuffer, yOffset: Int = 0,
                        t: Int, n: Int, k: Int) {
        precondition(t > 0 && n > 0 && k > 0, "QMM shape must be positive")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(x, offset: xOffset, index: 1)
        enc.setBuffer(y, offset: yOffset, index: 2)
        var tVar = UInt32(t)
        var nVar = UInt32(n)
        var kVar = UInt32(k)
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: (n + Self.tileN - 1) / Self.tileN,
                    height: (t + Self.tileM - 1) / Self.tileM,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadsPerGroup, height: 1, depth: 1))
        enc.endEncoding()
    }
}

/// The two elementwise halves of the patch embedder: the [0,1] -> [-1,1]
/// rescale that precedes its projection, and the position-table sum that
/// follows it. The projection itself is `VisionBF16QMM`.
package final class VisionPatchEmbed {
    private let psoPrescale: MTLComputePipelineState
    private let psoPositionAdd: MTLComputePipelineState

    package init(context: MetalContext) throws {
        self.psoPrescale = try context.pipeline("vision_patch_prescale_block")
        self.psoPositionAdd = try context.pipeline("vision_patch_pos_add_block")
    }

    package func encodePrescale(commandBuffer: MTLCommandBuffer,
                                x: MTLBuffer, xOffset: Int = 0,
                                out: MTLBuffer, outOffset: Int = 0,
                                count: Int) {
        precondition(count > 0, "count must be positive")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoPrescale)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var countVar = UInt32(count)
        enc.setBytes(&countVar, length: MemoryLayout<UInt32>.size, index: 2)
        let threads = min(psoPrescale.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `h[t] += table[0, t % pw] + table[1, t / pw]`.
    ///
    /// The grid is the position table's row count, so a patch grid that would
    /// index past it is a bug in the geometry, not something to clamp silently.
    package func encodePositionAdd(commandBuffer: MTLCommandBuffer,
                                   h: MTLBuffer, hOffset: Int = 0,
                                   table: MTLBuffer, tableOffset: Int = 0,
                                   patchCount: Int, d: Int, patchesWide: Int,
                                   tableLength: Int) {
        precondition(patchesWide > 0 && patchCount % patchesWide == 0,
                     "patch count must be a whole number of rows")
        precondition(patchesWide <= tableLength &&
                     patchCount / patchesWide <= tableLength,
                     "patch grid \(patchesWide)x\(patchCount / patchesWide) exceeds the "
                     + "position table (\(tableLength))")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoPositionAdd)
        enc.setBuffer(h, offset: hOffset, index: 0)
        enc.setBuffer(table, offset: tableOffset, index: 1)
        var pVar = UInt32(patchCount)
        var dVar = UInt32(d)
        var pwVar = UInt32(patchesWide)
        var lenVar = UInt32(tableLength)
        enc.setBytes(&pVar, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&pwVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&lenVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.dispatchThreads(MTLSize(width: d, height: patchCount, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
        enc.endEncoding()
    }
}

/// Q/K per-head RMSNorm, scale-less V RMSNorm, and the 2D RoPE, in one pass
/// over the three projections.
package final class VisionQKNormRoPE2D {
    /// The kernel gives each (patch, head) pair one simdgroup and three
    /// elements per lane, so it covers head dimensions up to 96. Gemma 4's
    /// tower is 72.
    package static let maxHeadDim = 96

    private let pso: MTLComputePipelineState

    package init(context: MetalContext) throws {
        self.pso = try context.pipeline("vision_qk_norm_rope2d_block")
    }

    package func encode(commandBuffer: MTLCommandBuffer,
                        q: MTLBuffer, qOffset: Int = 0,
                        k: MTLBuffer, kOffset: Int = 0,
                        v: MTLBuffer, vOffset: Int = 0,
                        qWeight: MTLBuffer, qWeightOffset: Int = 0,
                        kWeight: MTLBuffer, kWeightOffset: Int = 0,
                        patchCount: Int, headDim: Int, numHeads: Int,
                        patchesWide: Int, theta: Float, eps: Float) {
        precondition(headDim % 4 == 0,
                     "head dim must split into two halves of even width")
        precondition(headDim <= Self.maxHeadDim,
                     "head dim \(headDim) exceeds the kernel's \(Self.maxHeadDim)")
        precondition(patchesWide > 0 && patchCount % patchesWide == 0,
                     "patch count must be a whole number of rows")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(qWeight, offset: qWeightOffset, index: 3)
        enc.setBuffer(kWeight, offset: kWeightOffset, index: 4)
        var pVar = UInt32(patchCount)
        var hdVar = UInt32(headDim)
        var headsVar = UInt32(numHeads)
        var pwVar = UInt32(patchesWide)
        var thetaVar = theta
        var epsVar = eps
        enc.setBytes(&pVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&hdVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&headsVar, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&pwVar, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&thetaVar, length: MemoryLayout<Float>.size, index: 9)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 10)
        enc.dispatchThreadgroups(
            MTLSize(width: numHeads, height: patchCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }
}

struct VisionAttentionParams: Sendable, Equatable {
    var tokenCount: UInt32
    var headDim: UInt32
    var numHeads: UInt32
    var qTokenStrideElements: UInt32
    var kvTokenStrideElements: UInt32
    var oTokenStrideElements: UInt32
    var scale: Float
}

/// Full attention over one image's patches: every patch sees every patch.
package final class VisionAttentionFull {
    /// Which kernel served a call, so a benchmark can assert it measured the
    /// path it meant to.
    package enum Path: String, Sendable {
        /// Eight-lane segments, one head row per segment. Head dim 72 only.
        case segment8 = "segment8"
        /// The whole simdgroup on one query, as the text prefill kernels do.
        /// Covers any head dim up to 96 and is the fallback.
        case qBlock = "qblock"
    }

    package static let threadsPerGroup = 256
    /// Both specialisations put 64 queries in a threadgroup: 8 simdgroups of
    /// 8 for `qBlock`, and 4 segments x 2 queries x 8 simdgroups for
    /// `segment8`.
    package static let queriesPerGroup = 64
    /// The head dimension the segmented kernel is compiled for (8 lanes x 9).
    package static let segmentedHeadDim = 72

    /// `TF_VISION_ATTN=qblock` forces the fallback, so the two can be measured
    /// against each other without a rebuild.
    private static let forcedPath = ProcessInfo.processInfo.environment["TF_VISION_ATTN"]

    private let pso: MTLComputePipelineState
    private let segmentedPSO: MTLComputePipelineState?

    package init(context: MetalContext) throws {
        self.pso = try context.pipeline("vision_attention_full_qblock_d96")
        self.segmentedPSO = Self.forcedPath == "qblock"
            ? nil
            : try context.pipeline("vision_attention_full_seg_d72")
    }

    package func path(headDim: Int) -> Path {
        segmentedPSO != nil && headDim == Self.segmentedHeadDim ? .segment8 : .qBlock
    }

    @discardableResult
    package func encode(commandBuffer: MTLCommandBuffer,
                        q: MTLBuffer, qOffset: Int = 0,
                        k: MTLBuffer, kOffset: Int = 0,
                        v: MTLBuffer, vOffset: Int = 0,
                        out: MTLBuffer, outOffset: Int = 0,
                        patchCount: Int, headDim: Int, numHeads: Int,
                        scale: Float = 1.0,
                        forcePath: Path? = nil) -> Path {
        precondition(headDim <= VisionQKNormRoPE2D.maxHeadDim,
                     "head dim \(headDim) exceeds the kernel's "
                     + "\(VisionQKNormRoPE2D.maxHeadDim)")
        let selected = forcePath ?? path(headDim: headDim)
        precondition(selected == .qBlock
                     || (segmentedPSO != nil && headDim == Self.segmentedHeadDim),
                     "the segmented kernel is compiled for head dim "
                     + "\(Self.segmentedHeadDim) only")
        let stride = UInt32(numHeads * headDim)
        var params = VisionAttentionParams(tokenCount: UInt32(patchCount),
                                           headDim: UInt32(headDim),
                                           numHeads: UInt32(numHeads),
                                           qTokenStrideElements: stride,
                                           kvTokenStrideElements: stride,
                                           oTokenStrideElements: stride,
                                           scale: scale)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return selected }
        enc.setComputePipelineState(selected == .segment8 ? segmentedPSO! : pso)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        enc.setBytes(&params, length: MemoryLayout<VisionAttentionParams>.stride, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: (patchCount + Self.queriesPerGroup - 1) / Self.queriesPerGroup,
                    height: numHeads,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadsPerGroup, height: 1, depth: 1))
        enc.endEncoding()
        return selected
    }
}

/// `hidden += rmsnorm(x) * weight`: the post-attention and post-MLP joins.
package final class VisionNormResidualAdd {
    private let pso: MTLComputePipelineState

    package init(context: MetalContext) throws {
        self.pso = try context.pipeline("vision_norm_residual_add_block")
    }

    package func encode(commandBuffer: MTLCommandBuffer,
                        hidden: MTLBuffer, hiddenOffset: Int = 0,
                        x: MTLBuffer, xOffset: Int = 0,
                        weight: MTLBuffer, weightOffset: Int = 0,
                        t: Int, d: Int, eps: Float) {
        precondition(t > 0 && d > 0, "shape must be positive")
        precondition(hidden !== x || hiddenOffset != xOffset,
                     "the residual and the branch output must not be the same rows")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 0)
        enc.setBuffer(x, offset: xOffset, index: 1)
        enc.setBuffer(weight, offset: weightOffset, index: 2)
        var tVar = UInt32(t)
        var dVar = UInt32(d)
        var epsVar = eps
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 5)
        let threads = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreadgroups(MTLSize(width: t, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}

/// `out = gelu_tanh(gate) * up` over the tower MLP's intermediate block.
package final class VisionMLPActivation {
    private let pso: MTLComputePipelineState

    package init(context: MetalContext) throws {
        self.pso = try context.pipeline("vision_mlp_act_block")
    }

    package func encode(commandBuffer: MTLCommandBuffer,
                        gate: MTLBuffer, gateOffset: Int = 0,
                        up: MTLBuffer, upOffset: Int = 0,
                        out: MTLBuffer, outOffset: Int = 0,
                        count: Int) {
        precondition(count > 0, "count must be positive")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(gate, offset: gateOffset, index: 0)
        enc.setBuffer(up, offset: upOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var countVar = UInt32(count)
        enc.setBytes(&countVar, length: MemoryLayout<UInt32>.size, index: 3)
        let threads = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}

/// 3x3 average pool, `* sqrt(hidden)`, and the tower's output standardization.
package final class VisionPoolStandardize {
    private let pso: MTLComputePipelineState

    package init(context: MetalContext) throws {
        self.pso = try context.pipeline("vision_pool_std_block")
    }

    package func encode(commandBuffer: MTLCommandBuffer,
                        h: MTLBuffer, hOffset: Int = 0,
                        out: MTLBuffer, outOffset: Int = 0,
                        stdScale: MTLBuffer, stdScaleOffset: Int = 0,
                        stdBias: MTLBuffer, stdBiasOffset: Int = 0,
                        d: Int, patchesWide: Int, patchesHigh: Int,
                        kernelSize: Int, rootHidden: Float,
                        standardize: Bool) {
        precondition(kernelSize > 0, "pooling kernel must be positive")
        precondition(patchesWide % kernelSize == 0 && patchesHigh % kernelSize == 0,
                     "patch grid \(patchesWide)x\(patchesHigh) is not a whole number of "
                     + "\(kernelSize)x\(kernelSize) cells")
        let pooledW = patchesWide / kernelSize
        let pooledH = patchesHigh / kernelSize
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(h, offset: hOffset, index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        enc.setBuffer(stdScale, offset: stdScaleOffset, index: 2)
        enc.setBuffer(stdBias, offset: stdBiasOffset, index: 3)
        var dVar = UInt32(d)
        var pwVar = UInt32(patchesWide)
        var pooledWVar = UInt32(pooledW)
        var pooledHVar = UInt32(pooledH)
        var kVar = UInt32(kernelSize)
        var rootVar = rootHidden
        var stdVar: UInt32 = standardize ? 1 : 0
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&pwVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&pooledWVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&pooledHVar, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&rootVar, length: MemoryLayout<Float>.size, index: 9)
        enc.setBytes(&stdVar, length: MemoryLayout<UInt32>.size, index: 10)
        enc.dispatchThreads(MTLSize(width: d, height: pooledW * pooledH, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
        enc.endEncoding()
    }
}

/// Writes one image's soft tokens over the image rows of a prefill chunk's
/// embedding block (`PLAN_VISION.md` §4-5-c).
package final class VisionSoftTokenScatter {
    private let pso: MTLComputePipelineState

    package init(context: MetalContext) throws {
        self.pso = try context.pipeline("vision_scatter_soft_tokens_block")
    }

    /// - Parameters:
    ///   - hiddenRow: first row *within this chunk* to overwrite.
    ///   - softRow: first soft token to take, for an image the chunk holds only
    ///     part of.
    package func encode(commandBuffer: MTLCommandBuffer,
                        hidden: MTLBuffer, hiddenOffset: Int = 0,
                        soft: MTLBuffer, softOffset: Int = 0,
                        d: Int,
                        hiddenRow: Int,
                        softRow: Int,
                        rowCount: Int,
                        hiddenStrideElements: Int,
                        scale: Float = 1.0) {
        precondition(d > 0, "hidden size must be positive")
        precondition(rowCount > 0, "row count must be positive")
        precondition(hiddenRow >= 0 && softRow >= 0, "row offsets must be non-negative")
        precondition(hiddenStrideElements >= d, "hidden stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 0)
        enc.setBuffer(soft, offset: softOffset, index: 1)
        var dVar = UInt32(d)
        var hiddenRowVar = UInt32(hiddenRow)
        var softRowVar = UInt32(softRow)
        var rowCountVar = UInt32(rowCount)
        var strideVar = UInt32(hiddenStrideElements)
        var scaleVar = scale
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&hiddenRowVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&softRowVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&rowCountVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&strideVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&scaleVar, length: MemoryLayout<Float>.size, index: 7)
        enc.dispatchThreads(MTLSize(width: d, height: rowCount, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
        enc.endEncoding()
    }
}
