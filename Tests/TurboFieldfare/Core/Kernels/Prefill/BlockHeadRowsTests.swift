import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// The two block heads against the per-row heads they replace.
///
/// A verify block scores every one of its rows, and doing that a row at a time
/// reads the 461 MB lm-head table once per row (docs/mtp/16-M4.5-PLAN.md §4 b).
/// The block heads read it once. What they must not change is the answer: per
/// output element the reduction is the per-row head's, so the comparison here
/// is exact, not approximate.
@Suite struct BlockHeadRowsTests {
    private static func packRows(_ rows: [Quantization.Int4AffineRow])
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16])
    {
        let rowBytes = rows[0].packed.count
        let groups = rows[0].scales.count
        var packed = [UInt8](repeating: 0, count: rows.count * rowBytes)
        var scales = [UInt16](repeating: 0, count: rows.count * groups)
        var biases = [UInt16](repeating: 0, count: rows.count * groups)
        for row in 0..<rows.count {
            for i in 0..<rowBytes { packed[row * rowBytes + i] = rows[row].packed[i] }
            for g in 0..<groups {
                scales[row * groups + g] = rows[row].scales[g]
                biases[row * groups + g] = rows[row].biases[g]
            }
        }
        return (packed, scales, biases)
    }

    private struct Fixture {
        let hidden: MTLBuffer
        let norm: MTLBuffer
        let weights: MTLBuffer
        let scales: MTLBuffer
        let biases: MTLBuffer
    }

    /// `d` spans more than one 128-byte vectorized block and leaves a scalar
    /// tail at either affine group size.
    private static func fixture(ctx: MetalContext, rows: Int, d: Int, vocab: Int,
                                seed: UInt64) throws -> Fixture {
        var rng = SeedTree(seed).key("block-head-d\(d)-v\(vocab)")
        var hidden = [Float16](repeating: 0, count: rows * d)
        for i in 0..<(rows * d) { hidden[i] = Float16(rng.uniform(-1.0, 1.0)) }
        let normBits = (0..<d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let weightRows = (0..<vocab).map { _ -> Quantization.Int4AffineRow in
            Quantization.quantizeInt4Affine((0..<d).map { _ in rng.uniform(-0.5, 0.5) })
        }
        let (packed, scales, biases) = Self.packRows(weightRows)
        guard let hiddenBuf = Fp16Buffer.make(ctx.device, halves: hidden),
              let normBuf = ctx.device.makeBuffer(
                bytes: normBits, length: normBits.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
              let wBuf = ctx.device.makeBuffer(
                bytes: packed, length: packed.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(
                bytes: scales, length: scales.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
              let bBuf = ctx.device.makeBuffer(
                bytes: biases, length: biases.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        return Fixture(hidden: hiddenBuf, norm: normBuf,
                       weights: wBuf, scales: sBuf, biases: bBuf)
    }

    @Test(arguments: [1, 2, 4, 8] as [Int])
    func greedyBlockHeadMatchesPerRowHead(rows: Int) throws {
        let d = 576
        let vocab = 1024
        let ctx = try MetalContext()
        let fixture = try Self.fixture(ctx: ctx, rows: rows, d: d, vocab: vocab, seed: 0xD101)
        let head = try LMHeadChainInt4(context: ctx, maxD: d, maxVocab: vocab)

        guard let blockTokens = ctx.device.makeBuffer(
                length: rows * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let rowTokens = ctx.device.makeBuffer(
                length: rows * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let commandBuffer = ctx.queue.makeCommandBuffer() else {
            Issue.record("allocation failed"); return
        }

        try head.encodeGreedyDecodeBlock(
            commandBuffer: commandBuffer,
            hidden: fixture.hidden,
            normWeight: fixture.norm,
            weights: fixture.weights,
            scales: fixture.scales,
            biases: fixture.biases,
            outTokens: blockTokens,
            rows: rows,
            d: UInt32(d),
            vocab: UInt32(vocab))
        for row in 0..<rows {
            head.encodeGreedyDecode(
                commandBuffer: commandBuffer,
                hidden: fixture.hidden,
                hiddenOffset: row * d * MemoryLayout<Float16>.stride,
                normWeight: fixture.norm,
                weights: fixture.weights,
                scales: fixture.scales,
                biases: fixture.biases,
                outToken: rowTokens,
                outTokenOffset: row * MemoryLayout<UInt32>.stride,
                d: UInt32(d),
                vocab: UInt32(vocab))
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let block = UnsafeBufferPointer(
            start: blockTokens.contents().assumingMemoryBound(to: UInt32.self), count: rows)
        let perRow = UnsafeBufferPointer(
            start: rowTokens.contents().assumingMemoryBound(to: UInt32.self), count: rows)
        for row in 0..<rows {
            #expect(block[row] == perRow[row],
                    "rows=\(rows) row=\(row): block=\(block[row]) perRow=\(perRow[row])")
        }
    }

    @Test(arguments: [1, 2, 4, 8] as [Int])
    func logitsBlockHeadMatchesPerRowHead(rows: Int) throws {
        let d = 576
        let vocab = 640
        let eps: Float = 1e-6
        let ctx = try MetalContext()
        let fixture = try Self.fixture(ctx: ctx, rows: rows, d: d, vocab: vocab, seed: 0xD202)
        let head = try PrefillFinalRowHeadInt4(context: ctx, maxD: d)

        guard let blockLogits = Fp16Buffer.make(ctx.device, count: rows * vocab),
              let rowLogits = Fp16Buffer.make(ctx.device, count: rows * vocab),
              let commandBuffer = ctx.queue.makeCommandBuffer() else {
            Issue.record("allocation failed"); return
        }

        try head.encodeLogitsRows(
            commandBuffer: commandBuffer,
            hiddenBlock: fixture.hidden,
            firstRow: 0,
            rowCount: rows,
            rowStrideElements: d,
            normWeight: fixture.norm,
            weights: fixture.weights,
            scales: fixture.scales,
            biases: fixture.biases,
            logits: blockLogits,
            d: UInt32(d),
            vocab: UInt32(vocab),
            rmsEps: eps)
        for row in 0..<rows {
            head.encodeLogits(
                commandBuffer: commandBuffer,
                hiddenBlock: fixture.hidden,
                row: row,
                rowStrideElements: d,
                normWeight: fixture.norm,
                weights: fixture.weights,
                scales: fixture.scales,
                biases: fixture.biases,
                logits: rowLogits,
                logitsOffset: row * vocab * MemoryLayout<Float16>.stride,
                d: UInt32(d),
                vocab: UInt32(vocab),
                rmsEps: eps)
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let block = Fp16Buffer.readHalf(blockLogits, count: rows * vocab)
        let perRow = Fp16Buffer.readHalf(rowLogits, count: rows * vocab)
        var mismatches = 0
        for i in 0..<(rows * vocab) where block[i] != perRow[i] { mismatches += 1 }
        #expect(mismatches == 0, "rows=\(rows): \(mismatches) of \(rows * vocab) logits differ")
    }
}
