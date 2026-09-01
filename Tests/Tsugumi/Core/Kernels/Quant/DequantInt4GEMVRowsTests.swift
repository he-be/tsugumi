import Testing
import Foundation
import Metal
@testable import Tsugumi
import TsugumiValidationSupport

/// The k-row INT4 GEMV against the one-row kernel it generalizes.
///
/// The claim being checked is not "close enough": the rows kernel keeps the
/// one-row kernel's reduction for every output element, in the same order, so a
/// speculative verify block scores its rows with decode's arithmetic rather
/// than a tiled kernel's (docs/mtp/16-M4.5-PLAN.md §4 c). Anything short of an
/// exact match means the two paths have drifted and the block is no longer a
/// generalization of decode.
@Suite struct DequantInt4GEMVRowsTests {

    private static func packWeights(_ rows: [Quantization.Int4AffineRow])
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16])
    {
        let m = rows.count
        let n2 = rows[0].packed.count
        let s = rows[0].scales.count
        var packed = [UInt8](repeating: 0, count: m * n2)
        var scales = [UInt16](repeating: 0, count: m * s)
        var biases = [UInt16](repeating: 0, count: m * s)
        for row in 0..<m {
            for i in 0..<n2 { packed[row * n2 + i] = rows[row].packed[i] }
            for i in 0..<s {
                scales[row * s + i] = rows[row].scales[i]
                biases[row * s + i] = rows[row].biases[i]
            }
        }
        return (packed, scales, biases)
    }

    /// Runs `t` activations through `encodeRows` once and through `encode` once
    /// per row, and requires the two to agree bit for bit.
    ///
    /// `weightByteOffset` 2 is the resident layout's alignment (BF16 scale/bias
    /// regions leave the weight offset 2-aligned but not 4-aligned), which is
    /// the case the vectorized `ushort` load exists for.
    private static func runAndCompare(m: Int, n: Int, t: Int, seed: UInt64,
                                      weightByteOffset: Int = 0,
                                      xPadElements: Int = 0,
                                      yPadElements: Int = 0) throws {
        precondition(n % Quantization.groupSize == 0)
        var rng = SeedTree(seed).key("int4-gemv-rows-m\(m)-n\(n)-t\(t)")

        var weightRows: [Quantization.Int4AffineRow] = []
        weightRows.reserveCapacity(m)
        for _ in 0..<m {
            let raw = (0..<n).map { _ in rng.uniform(-0.5, 0.5) }
            weightRows.append(Quantization.quantizeInt4Affine(raw))
        }
        let (packed, scales, biases) = packWeights(weightRows)

        // Strides deliberately exceed the row length: the prefill scratch binds
        // activation and output rows inside wider blocks.
        let xStride = n + xPadElements
        let yStride = m + yPadElements
        var xFp16 = [Float16](repeating: 0, count: t * xStride)
        for row in 0..<t {
            for i in 0..<n { xFp16[row * xStride + i] = Float16(rng.uniform(-1.0, 1.0)) }
        }

        let ctx = try MetalContext()
        let kernel = try DequantInt4GEMV(context: ctx)

        var paddedPacked = [UInt8](repeating: 0, count: packed.count + weightByteOffset)
        for i in 0..<packed.count { paddedPacked[weightByteOffset + i] = packed[i] }

        guard let wBuf = ctx.device.makeBuffer(
                bytes: paddedPacked, length: paddedPacked.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(
                bytes: scales, length: scales.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
              let bBuf = ctx.device.makeBuffer(
                bytes: biases, length: biases.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let rowsOut = Fp16Buffer.make(ctx.device, count: t * yStride),
              let loopOut = Fp16Buffer.make(ctx.device, count: t * yStride) else {
            Issue.record("Failed to allocate buffers"); return
        }

        guard let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to make command buffer"); return
        }
        kernel.encodeRows(commandBuffer: cmd,
                          weights: wBuf, weightsOffset: weightByteOffset,
                          scales: sBuf, biases: bBuf,
                          x: xBuf, xStrideElements: xStride,
                          y: rowsOut, yStrideElements: yStride,
                          t: t, m: UInt32(m), n: UInt32(n))
        let halfBytes = MemoryLayout<Float16>.stride
        for row in 0..<t {
            kernel.encode(commandBuffer: cmd,
                          weights: wBuf, weightsOffset: weightByteOffset,
                          scales: sBuf, biases: bBuf,
                          x: xBuf, xOffset: row * xStride * halfBytes,
                          y: loopOut, yOffset: row * yStride * halfBytes,
                          m: UInt32(m), n: UInt32(n))
        }
        cmd.commit()
        cmd.waitUntilCompleted()

        let rows = Fp16Buffer.read(rowsOut, count: t * yStride)
        let loop = Fp16Buffer.read(loopOut, count: t * yStride)
        for row in 0..<t {
            for i in 0..<m {
                let index = row * yStride + i
                let detail = "M=\(m) N=\(n) t=\(t) row=\(row) out=\(i): "
                    + "rows=\(rows[index]) loop=\(loop[index])"
                #expect(rows[index] == loop[index], "\(detail)")
            }
        }
    }

    @Test(arguments: [1, 2, 4, 8] as [Int])
    func rowsMatchRepeatedGEMV(t: Int) throws {
        try Self.runAndCompare(m: 128, n: 2816, t: t, seed: 0xF1)
    }

    /// Production hidden size against a non-multiple-of-8 output count, at the
    /// 2-aligned weight offset the resident layout produces.
    @Test func rowsMatchAtLiveOffsetAndRaggedM() throws {
        try Self.runAndCompare(m: 129, n: 2816, t: 4, seed: 0xF2, weightByteOffset: 2)
    }

    /// Activation and output rows bound inside wider blocks, as the prefill
    /// scratch binds them.
    @Test func rowsMatchWithPaddedStrides() throws {
        try Self.runAndCompare(m: 64, n: 704, t: 3, seed: 0xF3,
                               xPadElements: 32, yPadElements: 16)
    }

    /// The scalar tail: N smaller than one 128-byte vectorized block.
    @Test func rowsMatchOnScalarTail() throws {
        try Self.runAndCompare(m: 64, n: Quantization.groupSize, t: 5, seed: 0xF4)
    }
}
