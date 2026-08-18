import Foundation
import Metal

final class PrefillFinalRowHeadInt4 {
    private let rms: RMSNorm
    private let blockRMS: PrefillRMSNorm
    private let int4: DequantInt4GEMV
    private let normed: MTLBuffer
    private let device: MTLDevice
    /// Block staging for `encodeLogitsRows`, allocated on first use.
    private var blockNormed: MTLBuffer?
    private let maxD: Int

    private let affineGroupSize: Int

    init(context: MetalContext, maxD: Int = 2816) throws {
        self.affineGroupSize = context.affineGroupSize
        precondition(maxD > 0, "maxD must be positive")
        self.rms = try RMSNorm(context: context)
        self.blockRMS = try PrefillRMSNorm(context: context)
        self.int4 = try DequantInt4GEMV(context: context)
        self.device = context.device
        self.maxD = maxD
        guard let normed = context.device.makeBuffer(length: maxD * MemoryLayout<Float16>.size,
                                                     options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        self.normed = normed
    }

    func encodeLogits(commandBuffer: MTLCommandBuffer,
                             hiddenBlock: MTLBuffer,
                             row: Int,
                             rowStrideElements: Int,
                             normWeight: MTLBuffer,
                             normWeightOffset: Int = 0,
                             weights: MTLBuffer,
                             weightsOffset: Int = 0,
                             scales: MTLBuffer,
                             scalesOffset: Int = 0,
                             biases: MTLBuffer,
                             biasesOffset: Int = 0,
                             logits: MTLBuffer,
                             logitsOffset: Int = 0,
                             d: UInt32,
                             vocab: UInt32,
                             rmsEps: Float) {
        precondition(row >= 0, "row must be non-negative")
        precondition(rowStrideElements >= Int(d), "row stride must cover d")
        precondition(Int(d) <= maxD, "d=\(d) exceeds maxD=\(maxD)")
        precondition(d % UInt32(affineGroupSize) == 0,
                     "d must be a multiple of \(affineGroupSize)")
        let hiddenOffset = (row * rowStrideElements) * MemoryLayout<Float16>.size
        rms.encodeBF16W(commandBuffer: commandBuffer,
                        x: hiddenBlock,
                        xOffset: hiddenOffset,
                        weight: normWeight,
                        weightOffset: normWeightOffset,
                        out: normed,
                        d: d,
                        eps: rmsEps)
        int4.encode(commandBuffer: commandBuffer,
                    weights: weights,
                    weightsOffset: weightsOffset,
                    scales: scales,
                    scalesOffset: scalesOffset,
                    biases: biases,
                    biasesOffset: biasesOffset,
                    x: normed,
                    y: logits,
                    yOffset: logitsOffset,
                    m: vocab,
                    n: d)
    }

    /// `rowCount` consecutive rows of the block, each scored into its own
    /// `vocab`-wide logits row.
    ///
    /// `encodeLogits` in a loop reads the lm-head table once per row; this
    /// reads it once for the block (docs/mtp/16-M4.5-PLAN.md §4 b). Per output
    /// element the arithmetic is `encodeLogits`', so one row is bit-identical
    /// to it. The rows must be contiguous with stride `d`, which is how a
    /// prefill chunk stages them.
    func encodeLogitsRows(commandBuffer: MTLCommandBuffer,
                          hiddenBlock: MTLBuffer,
                          firstRow: Int,
                          rowCount: Int,
                          rowStrideElements: Int,
                          normWeight: MTLBuffer,
                          normWeightOffset: Int = 0,
                          weights: MTLBuffer,
                          weightsOffset: Int = 0,
                          scales: MTLBuffer,
                          scalesOffset: Int = 0,
                          biases: MTLBuffer,
                          biasesOffset: Int = 0,
                          logits: MTLBuffer,
                          logitsOffset: Int = 0,
                          d: UInt32,
                          vocab: UInt32,
                          rmsEps: Float) throws {
        precondition(firstRow >= 0, "firstRow must be non-negative")
        precondition(rowCount >= 1 && rowCount <= DequantInt4GEMV.maxRows,
                     "rowCount=\(rowCount) is outside 1...\(DequantInt4GEMV.maxRows)")
        precondition(rowStrideElements == Int(d),
                     "the block head needs contiguous rows, stride \(rowStrideElements) != d \(d)")
        precondition(Int(d) <= maxD, "d=\(d) exceeds maxD=\(maxD)")
        precondition(d % UInt32(affineGroupSize) == 0,
                     "d must be a multiple of \(affineGroupSize)")
        let normed = try blockNormedBuffer()
        blockRMS.encodeBF16W(commandBuffer: commandBuffer,
                             x: hiddenBlock,
                             xOffset: firstRow * rowStrideElements * MemoryLayout<Float16>.size,
                             weight: normWeight,
                             weightOffset: normWeightOffset,
                             out: normed,
                             t: UInt32(rowCount),
                             d: d,
                             eps: rmsEps)
        int4.encodeRows(commandBuffer: commandBuffer,
                        weights: weights, weightsOffset: weightsOffset,
                        scales: scales, scalesOffset: scalesOffset,
                        biases: biases, biasesOffset: biasesOffset,
                        x: normed, xStrideElements: Int(d),
                        y: logits, yOffset: logitsOffset,
                        yStrideElements: Int(vocab),
                        t: rowCount, m: vocab, n: d)
    }

    private func blockNormedBuffer() throws -> MTLBuffer {
        if let blockNormed { return blockNormed }
        guard let buffer = device.makeBuffer(
                length: DequantInt4GEMV.maxRows * maxD * MemoryLayout<Float16>.size,
                options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        buffer.label = "prefill.finalRowHead.blockNormed"
        blockNormed = buffer
        return buffer
    }
}
