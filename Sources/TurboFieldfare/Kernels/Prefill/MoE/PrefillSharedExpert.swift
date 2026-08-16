import Foundation
import Metal

final class PrefillSharedExpert {
    private let shared: SharedExpertRuntime
    /// Batched path, INT4 only. `SharedExpertRuntime` runs one GEMV per token,
    /// which reloads the whole gate/up/down weight set for every row of the
    /// chunk; the chunk's rows differ only in the activation, so one QMM over
    /// all of them reads the weights once instead.
    private let qmm: PrefillInt4QMM?
    private let geluMulPSO: MTLComputePipelineState

    init(context: MetalContext, weightBits: Int = 8) throws {
        self.shared = try SharedExpertRuntime(context: context, weightBits: weightBits)
        self.qmm = weightBits == 4 ? try PrefillInt4QMM(context: context) : nil
        self.geluMulPSO = try context.pipeline("gelu_mul_fp16")
    }

    func encodeBlock(commandBuffer cb: MTLCommandBuffer,
                            x: MTLBuffer,
                            xOffset: Int = 0,
                            y: MTLBuffer,
                            yOffset: Int = 0,
                            gate: SharedExpertInt8Proj,
                            up: SharedExpertInt8Proj,
                            down: SharedExpertInt8Proj,
                            scratchGate: MTLBuffer,
                            scratchGateOffset: Int = 0,
                            scratchUp: MTLBuffer,
                            scratchUpOffset: Int = 0,
                            scratchAct: MTLBuffer,
                            scratchActOffset: Int = 0,
                            queryCount: Int,
                            d: Int,
                            intermediate: Int,
                            xStrideElements: Int,
                            yStrideElements: Int) throws {
        precondition(queryCount >= 0, "queryCount must be non-negative")
        precondition(d > 0, "d must be positive")
        precondition(intermediate > 0, "intermediate must be positive")
        precondition(xStrideElements >= d, "x stride is too small")
        precondition(yStrideElements >= d, "y stride is too small")
        guard gate.rows == UInt32(intermediate), gate.cols == UInt32(d),
              up.rows == UInt32(intermediate), up.cols == UInt32(d),
              down.rows == UInt32(d), down.cols == UInt32(intermediate) else {
            throw SharedExpertInt8Error.dimensionMismatch(
                "expected gate/up=(\(intermediate),\(d)) down=(\(d),\(intermediate))")
        }
        guard queryCount > 0 else { return }

        let halfBytes = MemoryLayout<Float16>.stride
        if let qmm,
           xStrideElements == d,
           yStrideElements == d,
           batchScratchFits(queryCount: queryCount,
                            intermediate: intermediate,
                            scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                            scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                            scratchAct: scratchAct, scratchActOffset: scratchActOffset) {
            encodeBatched(commandBuffer: cb,
                          qmm: qmm,
                          x: x, xOffset: xOffset,
                          y: y, yOffset: yOffset,
                          gate: gate, up: up, down: down,
                          scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                          scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                          scratchAct: scratchAct, scratchActOffset: scratchActOffset,
                          queryCount: queryCount,
                          d: d,
                          intermediate: intermediate)
            return
        }

        for row in 0..<queryCount {
            try shared.encode(commandBuffer: cb,
                              x: x,
                              xOffset: xOffset + row * xStrideElements * halfBytes,
                              gate: gate,
                              up: up,
                              down: down,
                              y: y,
                              yOffset: yOffset + row * yStrideElements * halfBytes,
                              scratchGate: scratchGate,
                              scratchGateOffset: scratchGateOffset,
                              scratchUp: scratchUp,
                              scratchUpOffset: scratchUpOffset,
                              scratchAct: scratchAct,
                              scratchActOffset: scratchActOffset)
        }
    }

    /// The batched path writes `queryCount * intermediate` elements into each
    /// intermediate buffer; a layout sized for a single row falls back.
    private func batchScratchFits(queryCount: Int,
                                  intermediate: Int,
                                  scratchGate: MTLBuffer, scratchGateOffset: Int,
                                  scratchUp: MTLBuffer, scratchUpOffset: Int,
                                  scratchAct: MTLBuffer, scratchActOffset: Int) -> Bool {
        let required = queryCount * intermediate * MemoryLayout<Float16>.stride
        return scratchGateOffset + required <= scratchGate.length
            && scratchUpOffset + required <= scratchUp.length
            && scratchActOffset + required <= scratchAct.length
    }

    private func encodeBatched(commandBuffer cb: MTLCommandBuffer,
                               qmm: PrefillInt4QMM,
                               x: MTLBuffer, xOffset: Int,
                               y: MTLBuffer, yOffset: Int,
                               gate: SharedExpertInt8Proj,
                               up: SharedExpertInt8Proj,
                               down: SharedExpertInt8Proj,
                               scratchGate: MTLBuffer, scratchGateOffset: Int,
                               scratchUp: MTLBuffer, scratchUpOffset: Int,
                               scratchAct: MTLBuffer, scratchActOffset: Int,
                               queryCount: Int,
                               d: Int,
                               intermediate: Int) {
        qmm.encode(commandBuffer: cb,
                   weights: gate.weights, weightsOffset: gate.weightsOffset,
                   scales: gate.scales, scalesOffset: gate.scalesOffset,
                   biases: gate.biases, biasesOffset: gate.biasesOffset,
                   x: x, xOffset: xOffset,
                   y: scratchGate, yOffset: scratchGateOffset,
                   t: queryCount, n: intermediate, k: d)
        qmm.encode(commandBuffer: cb,
                   weights: up.weights, weightsOffset: up.weightsOffset,
                   scales: up.scales, scalesOffset: up.scalesOffset,
                   biases: up.biases, biasesOffset: up.biasesOffset,
                   x: x, xOffset: xOffset,
                   y: scratchUp, yOffset: scratchUpOffset,
                   t: queryCount, n: intermediate, k: d)

        // `gelu_mul_fp16` is elementwise, so the [t, F] block runs as one
        // dispatch of t * F elements.
        if let encoder = cb.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(geluMulPSO)
            encoder.setBuffer(scratchGate, offset: scratchGateOffset, index: 0)
            encoder.setBuffer(scratchUp, offset: scratchUpOffset, index: 1)
            encoder.setBuffer(scratchAct, offset: scratchActOffset, index: 2)
            let elements = queryCount * intermediate
            var count = UInt32(elements)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
            let width = min(geluMulPSO.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(MTLSize(width: elements, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            encoder.endEncoding()
        }

        qmm.encode(commandBuffer: cb,
                   weights: down.weights, weightsOffset: down.weightsOffset,
                   scales: down.scales, scalesOffset: down.scalesOffset,
                   biases: down.biases, biasesOffset: down.biasesOffset,
                   x: scratchAct, xOffset: scratchActOffset,
                   y: y, yOffset: yOffset,
                   t: queryCount, n: d, k: intermediate)
    }
}
