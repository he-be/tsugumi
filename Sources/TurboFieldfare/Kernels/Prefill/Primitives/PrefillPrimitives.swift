import Foundation
import Metal

final class PrefillEmbedLookupInt4 {
    private let affineGroupSize: Int
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.affineGroupSize = context.affineGroupSize
        self.pso = try context.pipeline("prefill_embed_lookup_int4_block")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       table: MTLBuffer, tableOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       tokens: MTLBuffer, tokensOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       t: UInt32,
                       d: UInt32,
                       outScale: Float) {
        precondition(d % UInt32(affineGroupSize) == 0,
                     "D must be a multiple of \(affineGroupSize)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(table, offset: tableOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(tokens, offset: tokensOffset, index: 3)
        enc.setBuffer(out, offset: outOffset, index: 4)
        var tVar = t
        var dVar = d
        var scaleVar = outScale
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&scaleVar, length: MemoryLayout<Float>.size, index: 7)
        enc.dispatchThreads(MTLSize(width: Int(d), height: Int(t), depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
}

final class PrefillRMSNorm {
    private let psoBF16W: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoBF16W = try context.pipeline("prefill_rmsnorm_bf16w_block")
    }

    func encodeBF16W(commandBuffer: MTLCommandBuffer,
                            x: MTLBuffer, xOffset: Int = 0,
                            weight: MTLBuffer, weightOffset: Int = 0,
                            out: MTLBuffer, outOffset: Int = 0,
                            t: UInt32,
                            d: UInt32,
                            eps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoBF16W)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var tVar = t
        var dVar = d
        var epsVar = eps
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 5)
        let threads = min(Int(psoBF16W.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: Int(t), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}

package final class PrefillInt4QMM {
    /// Which kernel served a call. `encode` returns it so an A/B test can
    /// assert the path it meant to measure actually ran.
    package enum Path: String, Sendable {
        /// One thread per (token, row), scalar K reduction.
        case scalarBlock = "scalar-block"
        /// 64x64 output tile per threadgroup, 8x8 `simdgroup_matrix` products.
        case simdgroupMatrix = "simdgroup-matrix"
    }

    package static let tileM = 64
    package static let tileN = 64
    package static let tileK = 32
    package static let threadsPerGroup = 128

    /// `TF_PREFILL_QMM=scalar` forces the pre-existing scalar kernel, so the
    /// two paths can be measured against each other without a rebuild. It is
    /// also the way back to FP32 weight arithmetic: the tiled path stages its
    /// dequantized weights as FP16, which costs one rounding per weight.
    /// Anything else, including unset, takes the tiled path.
    private static let forcedPath = ProcessInfo.processInfo.environment["TF_PREFILL_QMM"]

    private let affineGroupSize: Int
    private let pso: MTLComputePipelineState
    private let simdgroupPSO: MTLComputePipelineState?

    package init(context: MetalContext) throws {
        self.affineGroupSize = context.affineGroupSize
        self.pso = try context.pipeline("prefill_dequant_int4_qmm_f16_block")
        if Self.forcedPath == "scalar" {
            self.simdgroupPSO = nil
        } else {
            // The tile shape fixes the threadgroup at 128 threads; a build
            // where register pressure caps it lower cannot run this kernel.
            let candidate = try? context.pipeline("prefill_int4_qmm_simdgroup_f16")
            self.simdgroupPSO = (candidate?.maxTotalThreadsPerThreadgroup ?? 0) >= Self.threadsPerGroup
                ? candidate
                : nil
        }
    }

    /// The tiled kernel walks K in 32-element steps and reads one scale/bias
    /// pair per 8 weights, so it needs K aligned to the tile and to the group.
    package func usesSimdgroupPath(k: Int) -> Bool {
        simdgroupPSO != nil && k % Self.tileK == 0 && k % affineGroupSize == 0
    }

    @discardableResult
    package func encode(commandBuffer: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       t: Int,
                       n: Int,
                       k: Int) -> Path {
        precondition(k % affineGroupSize == 0,
                     "K must be a multiple of \(affineGroupSize)")
        let tiled = usesSimdgroupPath(k: k)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            return tiled ? .simdgroupMatrix : .scalarBlock
        }
        enc.setComputePipelineState(tiled ? simdgroupPSO! : pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(x, offset: xOffset, index: 3)
        enc.setBuffer(y, offset: yOffset, index: 4)
        var tVar = UInt32(t)
        var nVar = UInt32(n)
        var kVar = UInt32(k)
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 7)
        if tiled {
            enc.dispatchThreadgroups(
                MTLSize(width: (n + Self.tileN - 1) / Self.tileN,
                        height: (t + Self.tileM - 1) / Self.tileM,
                        depth: 1),
                threadsPerThreadgroup: MTLSize(width: Self.threadsPerGroup, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(
                MTLSize(width: (n + 7) / 8, height: (t + 7) / 8, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        }
        enc.endEncoding()
        return tiled ? .simdgroupMatrix : .scalarBlock
    }
}
