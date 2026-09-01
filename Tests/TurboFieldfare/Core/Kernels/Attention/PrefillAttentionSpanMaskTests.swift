import Testing
import Accelerate
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// The bidirectional image span in prefill attention (`PLAN_VISION.md` §4-5-b).
///
/// A query inside an image span sees the whole span, its own future included;
/// every other query stays causal. Both attention kernels have to implement it —
/// the query-blocked one that production dispatches and the tiled one it falls
/// back to — because a fallback that quietly reverted to causal would produce a
/// fluent answer about an image the model half-saw.
@Suite struct PrefillAttentionSpanMaskTests {
    private struct Fixture {
        var q: [Float]
        var k: [Float]
        var v: [Float]
        var qStride: Int
        var kvStride: Int
        var oStride: Int
        var headDim: Int
        var qHeads: Int
        var kvHeads: Int
        var start: Int
        var chunk: Int
        var kvValid: Int
        var window: Int
        /// Local query rows `[spanFirst, spanEnd)` form one image.
        var spanFirst: Int
        var spanCount: Int

        var spanEnds: [UInt32] {
            var ends = [UInt32](repeating: 0, count: chunk)
            for row in spanFirst..<(spanFirst + spanCount) {
                ends[row] = UInt32(start + spanFirst + spanCount)
            }
            return ends
        }
    }

    /// headDim 8 takes the tiled fallback, 256 the sliding-window query block,
    /// 512 the full-attention query block. All three carry the mask.
    @Test(arguments: [
        (label: "tiled-d8", headDim: 8, qHeads: 4, kvHeads: 2),
        (label: "qblock-d256", headDim: 256, qHeads: 16, kvHeads: 8),
        (label: "qblock-d512", headDim: 512, qHeads: 16, kvHeads: 2),
    ])
    func spanQueriesSeeTheWholeSpan(
        _ shape: (label: String, headDim: Int, qHeads: Int, kvHeads: Int)
    ) throws {
        let fixture = Self.makeFixture(start: 40,
                                       chunk: 48,
                                       window: 1_024,
                                       spanFirst: 9,
                                       spanCount: 30,
                                       seed: 0xB100,
                                       headDim: shape.headDim,
                                       qHeads: shape.qHeads,
                                       kvHeads: shape.kvHeads)
        let actual = try Self.runKernel(fixture, spanMask: true)
        let reference = Self.reference(fixture, spanMask: true)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(maxAbs <= 2e-2, "\(shape.label) maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= 2e-2, "\(shape.label) rel=\(rel) maxAbs=\(maxAbs)")

        // The check has to be able to fail: compared against the causal
        // reference — what the kernel produced before this change — the same
        // output must be far outside the tolerance it just met. Only the span's
        // rows may move, and they must move a lot.
        let causal = Self.reference(fixture, spanMask: false)
        let causalDiff = RelError.maxAbsDiff(actual, causal)
        #expect(causalDiff > 2e-1,
                "\(shape.label) span mask changed nothing (maxAbs vs causal \(causalDiff))")
    }

    /// Rows outside the span are untouched by the mask. Measured against the
    /// kernel's own unmasked output rather than the CPU reference, so this is
    /// exact rather than within a tolerance.
    @Test func rowsOutsideTheSpanAreBitIdenticalToTheCausalRun() throws {
        let fixture = Self.makeFixture(start: 40,
                                       chunk: 48,
                                       window: 1_024,
                                       spanFirst: 9,
                                       spanCount: 30,
                                       seed: 0xB210,
                                       headDim: 256,
                                       qHeads: 16,
                                       kvHeads: 8)
        let masked = try Self.runKernel(fixture, spanMask: true)
        let causal = try Self.runKernel(fixture, spanMask: false)
        let perRow = fixture.qHeads * fixture.headDim
        var movedRows: [Int] = []
        for row in 0..<fixture.chunk {
            let range = (row * perRow)..<((row + 1) * perRow)
            if Array(masked[range]) != Array(causal[range]) { movedRows.append(row) }
        }
        // The span's first row is causal-equivalent only by accident, so the
        // expectation is "no row outside the span moved", not "every row inside
        // it did".
        let span = fixture.spanFirst..<(fixture.spanFirst + fixture.spanCount)
        #expect(movedRows.allSatisfy { span.contains($0) },
                "rows outside \(span) moved: \(movedRows)")
        #expect(movedRows.count >= fixture.spanCount - 1,
                "only \(movedRows.count) of \(fixture.spanCount) span rows moved")
    }

    /// A chunk with the mask enabled but no image in it (every `span_end` zero)
    /// has to be the causal path exactly. This is the shape every text chunk of
    /// a prompt that carries an image somewhere else would take if the runner
    /// bound the buffer unconditionally.
    @Test func allZeroSpanEndsMatchTheCausalRunBitForBit() throws {
        var fixture = Self.makeFixture(start: 40,
                                       chunk: 48,
                                       window: 1_024,
                                       spanFirst: 0,
                                       spanCount: 0,
                                       seed: 0xB220,
                                       headDim: 256,
                                       qHeads: 16,
                                       kvHeads: 8)
        fixture.spanCount = 0
        let masked = try Self.runKernel(fixture, spanMask: true)
        let causal = try Self.runKernel(fixture, spanMask: false)
        #expect(masked == causal)
    }

    // MARK: - Harness

    private static func makeFixture(start: Int,
                                    chunk: Int,
                                    window: Int,
                                    spanFirst: Int,
                                    spanCount: Int,
                                    seed: UInt64,
                                    headDim: Int,
                                    qHeads: Int,
                                    kvHeads: Int) -> Fixture {
        let qStride = qHeads * headDim + 3
        let kvStride = kvHeads * headDim + 5
        let oStride = qHeads * headDim + 7
        let kvValid = start + chunk
        var rng = SeedTree(seed).key("span-mask-\(headDim)-\(chunk)")
        var q = [Float](repeating: 0, count: chunk * qStride)
        var k = [Float](repeating: 0, count: kvValid * kvStride)
        var v = [Float](repeating: 0, count: kvValid * kvStride)
        for t in 0..<chunk {
            for h in 0..<qHeads {
                for d in 0..<headDim {
                    q[t * qStride + h * headDim + d] = rng.uniform(-0.35, 0.35)
                }
            }
        }
        // V carries a ramp in the key position on top of the noise. Without it
        // the two visibility sets average similar random rows and the causal
        // and bidirectional outputs differ by less than the tolerance — the
        // comparison would pass whether or not the mask did anything.
        for pos in 0..<kvValid {
            for h in 0..<kvHeads {
                for d in 0..<headDim {
                    k[pos * kvStride + h * headDim + d] = rng.uniform(-0.35, 0.35)
                    v[pos * kvStride + h * headDim + d] =
                        Float(pos - start) * 0.05 + rng.uniform(-0.05, 0.05)
                }
            }
        }
        return Fixture(q: q, k: k, v: v,
                       qStride: qStride, kvStride: kvStride, oStride: oStride,
                       headDim: headDim, qHeads: qHeads, kvHeads: kvHeads,
                       start: start, chunk: chunk, kvValid: kvValid, window: window,
                       spanFirst: spanFirst, spanCount: spanCount)
    }

    private static func runKernel(_ fixture: Fixture, spanMask: Bool) throws -> [Float] {
        let ctx = try MetalContext()
        let prefill = try PrefillAttention(context: ctx)
        let outCount = fixture.chunk * fixture.oStride
        guard let qBuf = Fp16Buffer.make(ctx.device, values: fixture.q),
              let kBuf = Fp16Buffer.make(ctx.device, values: fixture.k),
              let vBuf = Fp16Buffer.make(ctx.device, values: fixture.v),
              let outBuf = Fp16Buffer.make(ctx.device, count: outCount) else {
            Issue.record("alloc failed")
            return []
        }
        var ends = fixture.spanEnds
        let spanBuffer = spanMask
            ? ctx.device.makeBuffer(bytes: &ends,
                                    length: ends.count * MemoryLayout<UInt32>.stride,
                                    options: .storageModeShared)
            : nil
        if spanMask && spanBuffer == nil {
            Issue.record("span buffer alloc failed")
            return []
        }

        let params = PrefillAttentionParams(
            startPosition: UInt32(fixture.start),
            queryCount: UInt32(fixture.chunk),
            headDim: UInt32(fixture.headDim),
            numQHeads: UInt32(fixture.qHeads),
            numKVHeads: UInt32(fixture.kvHeads),
            kvValidCount: UInt32(fixture.kvValid),
            slidingWindow: UInt32(fixture.window),
            kvTokenStrideElements: UInt32(fixture.kvStride),
            qTokenStrideElements: UInt32(fixture.qStride),
            oTokenStrideElements: UInt32(fixture.oStride),
            scale: 1.0,
            spanMaskEnabled: spanMask)

        let cb = ctx.queue.makeCommandBuffer()!
        prefill.encodeCausal(commandBuffer: cb,
                             q: qBuf, k: kBuf, v: vBuf, out: outBuf,
                             params: params,
                             spanEnd: spanBuffer)
        cb.commit()
        cb.waitUntilCompleted()

        let out = Fp16Buffer.read(outBuf, count: outCount)
        var compact = [Float](repeating: 0,
                              count: fixture.chunk * fixture.qHeads * fixture.headDim)
        for t in 0..<fixture.chunk {
            for h in 0..<fixture.qHeads {
                for d in 0..<fixture.headDim {
                    compact[(t * fixture.qHeads + h) * fixture.headDim + d] =
                        out[t * fixture.oStride + h * fixture.headDim + d]
                }
            }
        }
        return compact
    }

    private static func reference(_ fixture: Fixture, spanMask: Bool) -> [Float] {
        var out = [Float](repeating: 0,
                          count: fixture.chunk * fixture.qHeads * fixture.headDim)
        let qPerKV = fixture.qHeads / fixture.kvHeads
        let spanEnds = fixture.spanEnds
        for t in 0..<fixture.chunk {
            let absQ = fixture.start + t
            let first = fixture.window == 0 ? 0 : max(0, absQ + 1 - fixture.window)
            var last = min(fixture.kvValid, absQ + 1)
            if spanMask, spanEnds[t] != 0 {
                last = min(fixture.kvValid, max(last, Int(spanEnds[t])))
            }
            for qh in 0..<fixture.qHeads {
                let kvh = qh / qPerKV
                let qBase = t * fixture.qStride + qh * fixture.headDim
                let kvBase = kvh * fixture.headDim
                let keyCount = last - first
                guard keyCount > 0 else { continue }

                var weights = [Float](repeating: 0, count: keyCount)
                fixture.q.withUnsafeBufferPointer { pq in
                    fixture.k.withUnsafeBufferPointer { pk in
                        weights.withUnsafeMutableBufferPointer { pw in
                            for i in 0..<keyCount {
                                let kBase = (first + i) * fixture.kvStride + kvBase
                                var dot: Float = 0
                                vDSP_dotpr(pq.baseAddress! + qBase, 1,
                                           pk.baseAddress! + kBase, 1,
                                           &dot, vDSP_Length(fixture.headDim))
                                pw[i] = dot
                            }
                        }
                    }
                }

                var maxScore: Float = 0
                vDSP_maxv(weights, 1, &maxScore, vDSP_Length(keyCount))
                var negMax = -maxScore
                vDSP_vsadd(weights, 1, &negMax, &weights, 1, vDSP_Length(keyCount))
                weights = vForce.exp(weights)
                var denom: Float = 0
                vDSP_sve(weights, 1, &denom, vDSP_Length(keyCount))

                let outBase = (t * fixture.qHeads + qh) * fixture.headDim
                weights.withUnsafeBufferPointer { pw in
                    fixture.v.withUnsafeBufferPointer { pv in
                        let vColumn = pv.baseAddress! + first * fixture.kvStride + kvBase
                        for d in 0..<fixture.headDim {
                            var acc: Float = 0
                            vDSP_dotpr(pw.baseAddress!, 1,
                                       vColumn + d, vDSP_Stride(fixture.kvStride),
                                       &acc, vDSP_Length(keyCount))
                            out[outBase + d] = denom > 0 ? acc / denom : 0
                        }
                    }
                }
            }
        }
        return out
    }
}
