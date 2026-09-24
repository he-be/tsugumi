import Foundation
import Metal

final class PrefillQKVEpilogue {
    private let perHeadNorm: PrefillPerHeadNorm
    private let rope: PrefillRoPE
    /// Every norm and rotation in one dispatch (docs/m6-prefill/39), the same
    /// output as the separate passes. A candidate for now:
    /// `TF_PREFILL_QKV_EPILOGUE=fused`.
    private let fusedPSO: MTLComputePipelineState
    static let fusedRequested =
        ProcessInfo.processInfo.environment["TF_PREFILL_QKV_EPILOGUE"] == "fused"

    init(context: MetalContext) throws {
        self.perHeadNorm = try PrefillPerHeadNorm(context: context)
        self.rope = try PrefillRoPE(context: context)
        self.fusedPSO = try context.pipeline("prefill_qkv_norm_rope_block")
    }

    private func encodeFused(commandBuffer: MTLCommandBuffer,
                             q: MTLBuffer, qOffset: Int, k: MTLBuffer, kOffset: Int,
                             v: MTLBuffer, vOffset: Int,
                             qWeight: MTLBuffer, qWeightOffset: Int,
                             kWeight: MTLBuffer, kWeightOffset: Int,
                             startPosition: UInt32, queryCount: UInt32, headDim: UInt32,
                             numQHeads: UInt32, numKVHeads: UInt32,
                             qTokenStrideElements: UInt32, kvTokenStrideElements: UInt32,
                             theta: Float, rotatedPairs: UInt32, eps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(fusedPSO)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(qWeight, offset: qWeightOffset, index: 3)
        enc.setBuffer(kWeight, offset: kWeightOffset, index: 4)
        var values: [UInt32] = [queryCount, headDim, numQHeads, numKVHeads,
                                qTokenStrideElements, kvTokenStrideElements]
        for (i, index) in (5...10).enumerated() {
            enc.setBytes(&values[i], length: MemoryLayout<UInt32>.size, index: index)
        }
        var epsVar = eps
        var start = startPosition
        var thetaVar = theta
        var pairs = rotatedPairs
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 11)
        enc.setBytes(&start, length: MemoryLayout<UInt32>.size, index: 12)
        enc.setBytes(&thetaVar, length: MemoryLayout<Float>.size, index: 13)
        enc.setBytes(&pairs, length: MemoryLayout<UInt32>.size, index: 14)
        // 256 threads, as the per-head norm runs, so the sums reduce alike.
        enc.dispatchThreadgroups(
            MTLSize(width: Int(numQHeads + 2 * numKVHeads), height: Int(queryCount), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       q: MTLBuffer,
                       qOffset: Int = 0,
                       k: MTLBuffer,
                       kOffset: Int = 0,
                       v: MTLBuffer,
                       vOffset: Int = 0,
                       qWeight: MTLBuffer,
                       qWeightOffset: Int = 0,
                       kWeight: MTLBuffer,
                       kWeightOffset: Int = 0,
                       startPosition: UInt32,
                       queryCount: UInt32,
                       headDim: UInt32,
                       numQHeads: UInt32,
                       numKVHeads: UInt32,
                       qTokenStrideElements: UInt32,
                       kvTokenStrideElements: UInt32,
                       theta: Float,
                       rotatedPairs: UInt32,
                       eps: Float,
                       fused: Bool = PrefillQKVEpilogue.fusedRequested) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim > 0 && headDim % 2 == 0, "headDim must be positive and even")
        precondition(numQHeads > 0, "numQHeads must be positive")
        precondition(numKVHeads > 0, "numKVHeads must be positive")
        precondition(rotatedPairs > 0, "rotatedPairs must be positive")
        precondition(rotatedPairs * 2 <= headDim, "rotatedPairs must fit inside one head")
        precondition(qTokenStrideElements >= numQHeads * headDim,
                     "Q token stride is too small")
        precondition(kvTokenStrideElements >= numKVHeads * headDim,
                     "KV token stride is too small")

        if fused && headDim <= 512
            && fusedPSO.maxTotalThreadsPerThreadgroup >= 256 {
            encodeFused(commandBuffer: commandBuffer, q: q, qOffset: qOffset, k: k, kOffset: kOffset,
                        v: v, vOffset: vOffset, qWeight: qWeight, qWeightOffset: qWeightOffset,
                        kWeight: kWeight, kWeightOffset: kWeightOffset,
                        startPosition: startPosition, queryCount: queryCount, headDim: headDim,
                        numQHeads: numQHeads, numKVHeads: numKVHeads,
                        qTokenStrideElements: qTokenStrideElements,
                        kvTokenStrideElements: kvTokenStrideElements,
                        theta: theta, rotatedPairs: rotatedPairs, eps: eps)
            return
        }

        perHeadNorm.encodeBF16W(commandBuffer: commandBuffer,
                                x: q,
                                xOffset: qOffset,
                                weight: qWeight,
                                weightOffset: qWeightOffset,
                                out: q,
                                outOffset: qOffset,
                                queryCount: queryCount,
                                headDim: headDim,
                                numHeads: numQHeads,
                                tokenStrideElements: qTokenStrideElements,
                                eps: eps)
        perHeadNorm.encodeBF16W(commandBuffer: commandBuffer,
                                x: k,
                                xOffset: kOffset,
                                weight: kWeight,
                                weightOffset: kWeightOffset,
                                out: k,
                                outOffset: kOffset,
                                queryCount: queryCount,
                                headDim: headDim,
                                numHeads: numKVHeads,
                                tokenStrideElements: kvTokenStrideElements,
                                eps: eps)
        perHeadNorm.encodeNoScale(commandBuffer: commandBuffer,
                                  x: v,
                                  xOffset: vOffset,
                                  out: v,
                                  outOffset: vOffset,
                                  queryCount: queryCount,
                                  headDim: headDim,
                                  numHeads: numKVHeads,
                                  tokenStrideElements: kvTokenStrideElements,
                                  eps: eps)

        if rotatedPairs * 2 == headDim {
            rope.encodeDefaultNeox(commandBuffer: commandBuffer,
                                   data: q,
                                   dataOffset: qOffset,
                                   startPosition: startPosition,
                                   queryCount: queryCount,
                                   headDim: headDim,
                                   numHeads: numQHeads,
                                   tokenStrideElements: qTokenStrideElements,
                                   theta: theta)
            rope.encodeDefaultNeox(commandBuffer: commandBuffer,
                                   data: k,
                                   dataOffset: kOffset,
                                   startPosition: startPosition,
                                   queryCount: queryCount,
                                   headDim: headDim,
                                   numHeads: numKVHeads,
                                   tokenStrideElements: kvTokenStrideElements,
                                   theta: theta)
        } else {
            rope.encodeProportionalNeox(commandBuffer: commandBuffer,
                                        data: q,
                                        dataOffset: qOffset,
                                        startPosition: startPosition,
                                        queryCount: queryCount,
                                        headDim: headDim,
                                        numHeads: numQHeads,
                                        rotatedPairs: rotatedPairs,
                                        tokenStrideElements: qTokenStrideElements,
                                        theta: theta)
            rope.encodeProportionalNeox(commandBuffer: commandBuffer,
                                        data: k,
                                        dataOffset: kOffset,
                                        startPosition: startPosition,
                                        queryCount: queryCount,
                                        headDim: headDim,
                                        numHeads: numKVHeads,
                                        rotatedPairs: rotatedPairs,
                                        tokenStrideElements: kvTokenStrideElements,
                                        theta: theta)
        }
    }
}
