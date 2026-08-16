import Foundation
import Metal

@frozen
public struct PrefillTokenExpertPair: Equatable, Sendable {
    public var token: UInt32
    public var expert: UInt32
    public var rank: UInt32
    public var weightBitsAndReserved: UInt32

    public init(token: UInt32, expert: UInt32, rank: UInt32, weight: Float16) {
        self.token = token
        self.expert = expert
        self.rank = rank
        self.weightBitsAndReserved = UInt32(weight.bitPattern)
    }

    public init(token: UInt32, expert: UInt32, rank: UInt32, weightBitsAndReserved: UInt32) {
        self.token = token
        self.expert = expert
        self.rank = rank
        self.weightBitsAndReserved = weightBitsAndReserved
    }

    public var weight: Float16 {
        Float16(bitPattern: UInt16(truncatingIfNeeded: weightBitsAndReserved))
    }
}

final class PrefillRouter {
    private let pso: MTLComputePipelineState

    private let affineGroupSize: Int
    /// 8 for affine INT8 router weights, 16 for BF16. Only the matching kernel
    /// is compiled.
    let routerWeightBits: Int

    init(context: MetalContext, routerWeightBits: Int = 8) throws {
        self.affineGroupSize = context.affineGroupSize
        self.routerWeightBits = routerWeightBits
        switch routerWeightBits {
        case 8:  self.pso = try context.pipeline("prefill_router_gemma4_block")
        case 16: self.pso = try context.pipeline("prefill_router_gemma4_bf16_block")
        default: throw RouterError.unsupportedWeightBits(routerWeightBits)
        }
    }

    func encodeGemma4Block(commandBuffer: MTLCommandBuffer,
                                  weights: MTLBuffer,
                                  weightsOffset: Int = 0,
                                  scales: MTLBuffer,
                                  scalesOffset: Int = 0,
                                  biases: MTLBuffer,
                                  biasesOffset: Int = 0,
                                  hidden: MTLBuffer,
                                  hiddenOffset: Int = 0,
                                  effectiveScale: MTLBuffer,
                                  effectiveScaleOffset: Int = 0,
                                  perExpertScale: MTLBuffer,
                                  perExpertScaleOffset: Int = 0,
                                  outIndices: MTLBuffer,
                                  outIndicesOffset: Int = 0,
                                  outWeights: MTLBuffer,
                                  outWeightsOffset: Int = 0,
                                  queryCount: UInt32,
                                  numExperts: UInt32,
                                  d: UInt32,
                                  topK: UInt32,
                                  hiddenStrideElements: UInt32) {
        precondition(routerWeightBits == 8,
                     "INT8 router encode on a \(routerWeightBits)-bit router")
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(numExperts <= 256, "numExperts > 256 is not supported")
        precondition(topK > 0 && topK <= 64, "topK must be in 1...64")
        precondition(d % UInt32(affineGroupSize) == 0,
                     "D must be a multiple of \(affineGroupSize)")
        precondition(hiddenStrideElements >= d, "hidden stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 3)
        enc.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 4)
        enc.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 5)
        enc.setBuffer(outIndices, offset: outIndicesOffset, index: 6)
        enc.setBuffer(outWeights, offset: outWeightsOffset, index: 7)
        var tVar = queryCount
        var neVar = numExperts
        var dVar = d
        var topKVar = topK
        var strideVar = hiddenStrideElements
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&neVar, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&topKVar, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&strideVar, length: MemoryLayout<UInt32>.size, index: 12)
        let tgWidth = min(max(Int(numExperts), 32), pso.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(MTLSize(width: Int(queryCount), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// BF16 (unquantized) router weights. Same contract as `encodeGemma4Block`
    /// minus the scale/bias buffers.
    func encodeGemma4BF16Block(commandBuffer: MTLCommandBuffer,
                               weights: MTLBuffer,
                               weightsOffset: Int = 0,
                               hidden: MTLBuffer,
                               hiddenOffset: Int = 0,
                               effectiveScale: MTLBuffer,
                               effectiveScaleOffset: Int = 0,
                               perExpertScale: MTLBuffer,
                               perExpertScaleOffset: Int = 0,
                               outIndices: MTLBuffer,
                               outIndicesOffset: Int = 0,
                               outWeights: MTLBuffer,
                               outWeightsOffset: Int = 0,
                               queryCount: UInt32,
                               numExperts: UInt32,
                               d: UInt32,
                               topK: UInt32,
                               hiddenStrideElements: UInt32) {
        precondition(routerWeightBits == 16,
                     "BF16 router encode on a \(routerWeightBits)-bit router")
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(numExperts <= 256, "numExperts > 256 is not supported")
        precondition(topK > 0 && topK <= 64, "topK must be in 1...64")
        precondition(hiddenStrideElements >= d, "hidden stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 1)
        enc.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 2)
        enc.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 3)
        enc.setBuffer(outIndices, offset: outIndicesOffset, index: 4)
        enc.setBuffer(outWeights, offset: outWeightsOffset, index: 5)
        var tVar = queryCount
        var neVar = numExperts
        var dVar = d
        var topKVar = topK
        var strideVar = hiddenStrideElements
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&neVar, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&topKVar, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&strideVar, length: MemoryLayout<UInt32>.size, index: 10)
        let tgWidth = min(max(Int(numExperts), 32), pso.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(MTLSize(width: Int(queryCount), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        enc.endEncoding()
    }

    static func makeTokenExpertPairs(indices: [UInt32],
                                            weights: [Float16],
                                            queryCount: Int,
                                            topK: Int) -> [PrefillTokenExpertPair] {
        precondition(queryCount >= 0, "queryCount must be non-negative")
        precondition(topK >= 0, "topK must be non-negative")
        precondition(indices.count == queryCount * topK, "indices count mismatch")
        precondition(weights.count == queryCount * topK, "weights count mismatch")
        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(indices.count)
        for token in 0..<queryCount {
            for rank in 0..<topK {
                let i = token * topK + rank
                pairs.append(PrefillTokenExpertPair(token: UInt32(token),
                                                    expert: indices[i],
                                                    rank: UInt32(rank),
                                                    weight: weights[i]))
            }
        }
        return pairs
    }
}
