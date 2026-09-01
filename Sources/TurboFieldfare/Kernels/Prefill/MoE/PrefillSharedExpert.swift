import Foundation
import Metal

final class PrefillSharedExpert {
    private let shared: SharedExpertRuntime
    /// Batched path, INT4 only. `SharedExpertRuntime` runs one GEMV per token,
    /// which reloads the whole gate/up/down weight set for every row of the
    /// chunk; the chunk's rows differ only in the activation, so one QMM over
    /// all of them reads the weights once instead.
    private let qmm: PrefillInt4QMM?
    /// INT4 only. The QMM computes a 64-row output tile per threadgroup, which
    /// a prompt chunk fills and a k-token speculative block does not: at k=4 it
    /// is 16x the arithmetic for the same answer. This path keeps the bytes and
    /// drops the waste (docs/mtp/16-M4.5-PLAN.md §4 c).
    private let rowsGEMV: DequantInt4GEMV?
    private let geluMulPSO: MTLComputePipelineState

    init(context: MetalContext, weightBits: Int = 8) throws {
        self.shared = try SharedExpertRuntime(context: context, weightBits: weightBits)
        self.qmm = weightBits == 4 ? try PrefillInt4QMM(context: context) : nil
        self.rowsGEMV = weightBits == 4 ? try DequantInt4GEMV(context: context) : nil
        self.geluMulPSO = try context.pipeline("gelu_mul_fp16")
    }

    /// `rowsPath` is true only for a speculative verify block: that path is
    /// decode's arithmetic rather than the QMM's, so a prompt chunk that
    /// happens to be narrow keeps the numbers it had before
    /// (`docs/mtp/16-M4.5-PLAN.md` §4 c).
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
                            yStrideElements: Int,
                            rowsPath: Bool = false) throws {
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
        if rowsPath, let rowsGEMV,
           queryCount <= DequantInt4GEMV.maxRows,
           xStrideElements == d,
           yStrideElements == d,
           batchScratchFits(queryCount: queryCount,
                            intermediate: intermediate,
                            scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                            scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                            scratchAct: scratchAct, scratchActOffset: scratchActOffset) {
            try encodeRows(commandBuffer: cb,
                       rowsGEMV: rowsGEMV,
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

    /// The batched path's three GEMMs as three k-row GEMVs: same weights read
    /// once each, same elementwise stage between them, one row of arithmetic
    /// per row of the block instead of a 64-row tile's worth.
    ///
    /// `encodeRowsWide` rather than `encodeRows`: at 80 expert-cache slots the
    /// shared expert is GPU-bound and its k-scaling was arithmetic the block
    /// need not repeat per row (`docs/mtp/20-M4.8-RESULTS.md` §2). Same bits.
    private func encodeRows(commandBuffer cb: MTLCommandBuffer,
                            rowsGEMV: DequantInt4GEMV,
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
                            intermediate: Int) throws {
        try rowsGEMV.encodeRowsWide(commandBuffer: cb,
                            weights: gate.weights, weightsOffset: gate.weightsOffset,
                            scales: gate.scales, scalesOffset: gate.scalesOffset,
                            biases: gate.biases, biasesOffset: gate.biasesOffset,
                            x: x, xOffset: xOffset, xStrideElements: d,
                            y: scratchGate, yOffset: scratchGateOffset,
                            yStrideElements: intermediate,
                            t: queryCount, m: UInt32(intermediate), n: UInt32(d))
        try rowsGEMV.encodeRowsWide(commandBuffer: cb,
                            weights: up.weights, weightsOffset: up.weightsOffset,
                            scales: up.scales, scalesOffset: up.scalesOffset,
                            biases: up.biases, biasesOffset: up.biasesOffset,
                            x: x, xOffset: xOffset, xStrideElements: d,
                            y: scratchUp, yOffset: scratchUpOffset,
                            yStrideElements: intermediate,
                            t: queryCount, m: UInt32(intermediate), n: UInt32(d))
        encodeGeluMul(commandBuffer: cb,
                      scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                      scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                      scratchAct: scratchAct, scratchActOffset: scratchActOffset,
                      elements: queryCount * intermediate)
        try rowsGEMV.encodeRowsWide(commandBuffer: cb,
                            weights: down.weights, weightsOffset: down.weightsOffset,
                            scales: down.scales, scalesOffset: down.scalesOffset,
                            biases: down.biases, biasesOffset: down.biasesOffset,
                            x: scratchAct, xOffset: scratchActOffset,
                            xStrideElements: intermediate,
                            y: y, yOffset: yOffset, yStrideElements: d,
                            t: queryCount, m: UInt32(d), n: UInt32(intermediate))
    }

    /// `gelu_mul_fp16` is elementwise, so a `[t, F]` block runs as one dispatch
    /// of `t * F` elements.
    private func encodeGeluMul(commandBuffer cb: MTLCommandBuffer,
                               scratchGate: MTLBuffer, scratchGateOffset: Int,
                               scratchUp: MTLBuffer, scratchUpOffset: Int,
                               scratchAct: MTLBuffer, scratchActOffset: Int,
                               elements: Int) {
        guard elements > 0, let encoder = cb.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(geluMulPSO)
        encoder.setBuffer(scratchGate, offset: scratchGateOffset, index: 0)
        encoder.setBuffer(scratchUp, offset: scratchUpOffset, index: 1)
        encoder.setBuffer(scratchAct, offset: scratchActOffset, index: 2)
        var count = UInt32(elements)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(geluMulPSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: elements, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
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

        encodeGeluMul(commandBuffer: cb,
                      scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                      scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                      scratchAct: scratchAct, scratchActOffset: scratchActOffset,
                      elements: queryCount * intermediate)

        qmm.encode(commandBuffer: cb,
                   weights: down.weights, weightsOffset: down.weightsOffset,
                   scales: down.scales, scalesOffset: down.scalesOffset,
                   biases: down.biases, biasesOffset: down.biasesOffset,
                   x: scratchAct, xOffset: scratchActOffset,
                   y: y, yOffset: yOffset,
                   t: queryCount, n: d, k: intermediate)
    }
}
