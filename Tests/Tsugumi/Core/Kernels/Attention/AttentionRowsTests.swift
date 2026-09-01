import Testing
import Foundation
import Metal
@testable import Tsugumi
import TsugumiValidationSupport

/// `Attention.encodeRows` is the speculative block's attention: `rows` query
/// rows walking one KV history, row `r` seeing one position more than row
/// `r-1` (docs/mtp/24-M5.5-RESULTS.md §7-1). The row-at-a-time decode calls it
/// replaces are the contract it has to keep, so every case here checks both
/// ends — against `AttentionRef` per row, and against `encodeSWA`/`encodeFull`
/// called once per row, which is what the block did before.
@Suite struct AttentionRowsTests {

    /// The chunks are cut over the union of the rows' ranges: the earliest
    /// window start (row 0) to the last row's history end.
    @Test func rowsGeometry_cutsChunksOverTheUnionRange() {
        let windowed = Attention.rowsGeometry(rows: 4,
                                              startPosition: 1000,
                                              window: 512,
                                              numQHeads: 16)
        // Row 0 sees [1001-512, 1001); the last row ends at 1004.
        #expect(windowed.kvStart == 489)
        #expect(windowed.effectiveLength == 515)
        #expect(windowed.numChunks == 16)
        #expect(windowed.chunkLength == 33)
        #expect(windowed.partialThreadgroups == 256)
        #expect(windowed.partialThreadsPerGroup == 128)

        let full = Attention.rowsGeometry(rows: 4,
                                          startPosition: 100,
                                          window: 0,
                                          numQHeads: 16)
        #expect(full.kvStart == 0)
        #expect(full.effectiveLength == 104)

        // A block shorter than the split factor keeps one position per chunk.
        let tiny = Attention.rowsGeometry(rows: 2,
                                          startPosition: 3,
                                          window: 0,
                                          numQHeads: 4)
        #expect(tiny.numChunks == 5)
        #expect(tiny.chunkLength == 1)
        #expect(tiny.partialThreadgroups == 20)
    }

    private struct Fixture {
        let headDim: Int
        let numQHeads: Int
        let numKVHeads: Int
        let rows: Int
        let startPosition: Int
        /// 0 = full attention.
        let window: Int
        var kvLength: Int { startPosition + rows }
        var qStride: Int { numQHeads * headDim }
        var kvStride: Int { numKVHeads * headDim }
    }

    /// Runs `encodeRows` and, on the same inputs, the row-at-a-time decode
    /// calls, then checks both against the FP32 reference row by row.
    private static func check(_ fixture: Fixture,
                              seed: UInt64,
                              ringCapacity: Int = 0,
                              tolerance: Float = Tolerance.fp16ChainedReduction) throws {
        var rng = SeedTree(seed).key("attn-rows-d\(fixture.headDim)-r\(fixture.rows)")
        let qCount = fixture.rows * fixture.qStride
        let kvCount = fixture.kvLength * fixture.kvStride
        let qFp16 = (0..<qCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let kFp16 = (0..<kvCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let vFp16 = (0..<kvCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }

        // The KV cache the kernel reads is ring-addressed when a capacity is
        // set; the reference always sees the linear history.
        let storedLength = ringCapacity > 0 ? ringCapacity : fixture.kvLength
        var kStore = [Float16](repeating: 0, count: storedLength * fixture.kvStride)
        var vStore = kStore
        for p in 0..<fixture.kvLength {
            let dst = (ringCapacity > 0 ? p % ringCapacity : p) * fixture.kvStride
            let src = p * fixture.kvStride
            kStore.replaceSubrange(dst..<(dst + fixture.kvStride),
                                   with: kFp16[src..<(src + fixture.kvStride)])
            vStore.replaceSubrange(dst..<(dst + fixture.kvStride),
                                   with: vFp16[src..<(src + fixture.kvStride)])
        }

        let ctx = try MetalContext()
        let kernel = try Attention(context: ctx)
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qFp16),
              let kBuf = Fp16Buffer.make(ctx.device, halves: kStore),
              let vBuf = Fp16Buffer.make(ctx.device, halves: vStore),
              let rowsOut = Fp16Buffer.make(ctx.device, count: qCount),
              let perRowOut = Fp16Buffer.make(ctx.device, count: qCount) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeRows(commandBuffer: cb,
                          q: qBuf, k: kBuf, v: vBuf, out: rowsOut,
                          headDim: UInt32(fixture.headDim),
                          numQHeads: UInt32(fixture.numQHeads),
                          numKVHeads: UInt32(fixture.numKVHeads),
                          rows: fixture.rows,
                          startPosition: fixture.startPosition,
                          window: UInt32(fixture.window),
                          scale: 1.0,
                          ringCapacity: UInt32(ringCapacity))
        let rowStride = fixture.qStride * MemoryLayout<Float16>.stride
        for row in 0..<fixture.rows {
            let seqLen = UInt32(fixture.startPosition + row + 1)
            if fixture.window == 0 {
                kernel.encodeFull(commandBuffer: cb,
                                  q: qBuf, qOffset: row * rowStride,
                                  k: kBuf, v: vBuf,
                                  out: perRowOut, outOffset: row * rowStride,
                                  headDim: UInt32(fixture.headDim),
                                  numQHeads: UInt32(fixture.numQHeads),
                                  numKVHeads: UInt32(fixture.numKVHeads),
                                  seqLen: seqLen,
                                  scale: 1.0)
            } else {
                kernel.encodeSWA(commandBuffer: cb,
                                 q: qBuf, qOffset: row * rowStride,
                                 k: kBuf, v: vBuf,
                                 out: perRowOut, outOffset: row * rowStride,
                                 headDim: UInt32(fixture.headDim),
                                 numQHeads: UInt32(fixture.numQHeads),
                                 numKVHeads: UInt32(fixture.numKVHeads),
                                 seqLen: seqLen,
                                 window: UInt32(fixture.window),
                                 scale: 1.0,
                                 ringCapacity: UInt32(ringCapacity))
            }
        }
        cb.commit()
        cb.waitUntilCompleted()

        let rowsValues = Fp16Buffer.read(rowsOut, count: qCount)
        let perRowValues = Fp16Buffer.read(perRowOut, count: qCount)
        let kRef = kFp16.map { Float($0) }
        let vRef = vFp16.map { Float($0) }

        for row in 0..<fixture.rows {
            let seqLen = fixture.startPosition + row + 1
            let qRow = (0..<fixture.qStride).map { Float(qFp16[row * fixture.qStride + $0]) }
            let ref = AttentionRef.apply(
                q: qRow,
                k: Array(kRef[0..<(seqLen * fixture.kvStride)]),
                v: Array(vRef[0..<(seqLen * fixture.kvStride)]),
                headDim: fixture.headDim,
                numQHeads: fixture.numQHeads,
                numKVHeads: fixture.numKVHeads,
                seqLen: seqLen,
                window: fixture.window == 0 ? nil : fixture.window,
                scale: 1.0)
            let slice = Array(rowsValues[(row * fixture.qStride)..<((row + 1) * fixture.qStride)])
            let perRow = Array(perRowValues[(row * fixture.qStride)..<((row + 1) * fixture.qStride)])
            let relRef = RelError.compute(actual: slice, reference: ref)
            #expect(relRef < tolerance,
                    "rows kernel vs reference: row=\(row) rel=\(relRef)")
            let relRow = RelError.compute(actual: slice, reference: perRow.map { Float($0) })
            #expect(relRow < tolerance,
                    "rows kernel vs row-at-a-time decode calls: row=\(row) rel=\(relRow)")
        }
    }

    @Test func rowsSWA_matchesReferenceAndRowAtATimeCalls() throws {
        // Window shorter than the history: every row has a different window
        // start as well as a different end.
        try Self.check(Fixture(headDim: 64, numQHeads: 4, numKVHeads: 2,
                               rows: 4, startPosition: 40, window: 16),
                       seed: 0x5601)
    }

    @Test func rowsSWA_matchesAtProductionShape() throws {
        try Self.check(Fixture(headDim: 256, numQHeads: 16, numKVHeads: 8,
                               rows: 4, startPosition: 557, window: 1024),
                       seed: 0x5602)
    }

    @Test func rowsFull_matchesReferenceAndRowAtATimeCalls() throws {
        try Self.check(Fixture(headDim: 512, numQHeads: 16, numKVHeads: 2,
                               rows: 4, startPosition: 300, window: 0),
                       seed: 0x5603)
    }

    /// k = 8 is `SpeculativeVerifier.maxTokens`, the widest block, and the
    /// point where one SIMD group per row fills the 256-thread group.
    @Test func rowsSWA_widestBlock() throws {
        try Self.check(Fixture(headDim: 256, numQHeads: 16, numKVHeads: 8,
                               rows: 8, startPosition: 200, window: 128),
                       seed: 0x5604)
    }

    /// A single row is what a `TF_MTP_DRAFTS=0` block runs; the rows kernel has
    /// to hold at rows == 1 even though the runtime keeps that case on the
    /// decode call.
    @Test func rowsSWA_singleRow() throws {
        try Self.check(Fixture(headDim: 256, numQHeads: 16, numKVHeads: 8,
                               rows: 1, startPosition: 130, window: 64),
                       seed: 0x5605)
    }

    /// The block's KV lives in the FP16 ring once the history passes its
    /// capacity, and the block straddles a wrap here (capacity 24, rows at
    /// positions 22...25).
    @Test func rowsSWA_ringCapacityAcrossWrap() throws {
        try Self.check(Fixture(headDim: 64, numQHeads: 4, numKVHeads: 2,
                               rows: 4, startPosition: 22, window: 16),
                       seed: 0x5606,
                       ringCapacity: 24)
    }
}
