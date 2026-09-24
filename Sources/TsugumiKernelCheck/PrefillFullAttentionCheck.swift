import Foundation
import Metal
import Tsugumi
import TsugumiValidationSupport

// MARK: - Case 14 — prefill full attention, headDim 512 (qblock and MPP)

// The full-attention layers' prefill attention at the production head shape
// (16 query heads over 2 KV heads, headDim 512, every earlier key visible),
// against a CPU softmax over the FP16 inputs. The query-blocked kernel (the
// default without tensor ops, and `TF_PREFILL_ATTN_FULL=qblock`) runs
// everywhere; the `matmul2d` kernels `attention_prefill_full_mpp_d512_r64`
// (the default where the library has tensor ops, docs/m6-prefill/36) and
// `_r32` only exist in an MSL 4.0 build and are SKIPped elsewhere with the
// skip printed. The unfused candidate (`TF_PREFILL_ATTN_FULL=unfused`:
// `attention_prefill_full_scores_d512` → `_softmax_d512` → `_pv_d512`) is
// checked the way `PrefillAttention` dispatches it, over head groups of 2 and
// of all 16.
//
// Shapes: a first chunk that ends mid-block and a second chunk over the first
// one's keys. A causal mask shifted by one key scored against the same output
// shows the comparison can fail.

private let fullHeads = 16
private let fullKVHeads = 2
private let fullHeadDim = 512

/// `PrefillAttentionParams` in prefill.metal.
private struct FullParams {
    var startPosition: UInt32
    var queryCount: UInt32
    var headDim: UInt32
    var numQHeads: UInt32
    var numKVHeads: UInt32
    var kvValidCount: UInt32
    var slidingWindow: UInt32
    var kvTokenStrideElements: UInt32
    var qTokenStrideElements: UInt32
    var oTokenStrideElements: UInt32
    var scale: Float
    var hasSpanMask: UInt32
}

/// Softmax over keys `0 ... abs - lag` for each row (`lag` 0 is the causal mask).
private func fullReference(q: [Float16], k: [Float16], v: [Float16],
                           start: Int, count: Int, lag: Int = 0) -> [Float] {
    let qStride = fullHeads * fullHeadDim, kvStride = fullKVHeads * fullHeadDim
    var out = [Float](repeating: 0, count: count * qStride)
    for t in 0..<count {
        let abs = start + t
        let last = Swift.max(0, abs - lag)
        for h in 0..<fullHeads {
            let kvh = h / (fullHeads / fullKVHeads)
            var scores: [Double] = []
            for key in 0...last {
                var s = 0.0
                for d in 0..<fullHeadDim {
                    s += Double(q[abs * qStride + h * fullHeadDim + d])
                        * Double(k[key * kvStride + kvh * fullHeadDim + d])
                }
                scores.append(s)
            }
            let m = scores.max()!
            let w = scores.map { Foundation.exp($0 - m) }
            let sum = w.reduce(0, +)
            for d in 0..<fullHeadDim {
                var acc = 0.0
                for key in 0...last {
                    acc += w[key] * Double(v[key * kvStride + kvh * fullHeadDim + d])
                }
                out[t * qStride + h * fullHeadDim + d] = Float(acc / sum)
            }
        }
    }
    return out
}

func checkPrefillFullAttention(groupSize: Int, seed: UInt64) throws -> [CaseResult] {
    var rng = SeedTree(seed).key("prefill-full-attention")
    let total = 220
    let qStride = fullHeads * fullHeadDim, kvStride = fullKVHeads * fullHeadDim
    let q = (0..<(total * qStride)).map { _ in Float16(rng.uniform(-0.25, 0.25)) }
    let k = (0..<(total * kvStride)).map { _ in Float16(rng.uniform(-0.25, 0.25)) }
    let v = (0..<(total * kvStride)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }

    let context = try makeContext(groupSize: groupSize)
    let device = context.device
    guard let dummy = device.makeBuffer(length: 16, options: .storageModeShared) else {
        fatalError("prefill full attention allocation failed")
    }

    /// One dispatch of `function` over rows `start ..< start + count`. Key
    /// slots past the chunk's end hold NaN, so a read past `kvValidCount`
    /// that is not masked shows up.
    func run(_ function: String, threads: Int, rowsPerGroup: Int,
             start: Int, count: Int) throws -> [Float] {
        let slots = total + 64
        var kPhys = [Float16](repeating: .nan, count: slots * kvStride)
        var vPhys = [Float16](repeating: .nan, count: slots * kvStride)
        for i in 0..<((start + count) * kvStride) {
            kPhys[i] = k[i]
            vPhys[i] = v[i]
        }
        let qRows = Array(q[(start * qStride)..<((start + count) * qStride)])
        let pso = try context.pipeline(function)
        guard let qBuffer = Fp16Buffer.make(device, halves: qRows),
              let kBuffer = Fp16Buffer.make(device, halves: kPhys),
              let vBuffer = Fp16Buffer.make(device, halves: vPhys),
              let oBuffer = Fp16Buffer.make(device, count: count * qStride),
              let cmd = context.queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            fatalError("prefill full attention allocation failed")
        }
        var p = FullParams(startPosition: UInt32(start), queryCount: UInt32(count),
                           headDim: UInt32(fullHeadDim), numQHeads: UInt32(fullHeads),
                           numKVHeads: UInt32(fullKVHeads), kvValidCount: UInt32(start + count),
                           slidingWindow: 0, kvTokenStrideElements: UInt32(kvStride),
                           qTokenStrideElements: UInt32(qStride), oTokenStrideElements: UInt32(qStride),
                           scale: 1.0, hasSpanMask: 0)
        enc.setComputePipelineState(pso)
        enc.setBuffer(qBuffer, offset: 0, index: 0)
        enc.setBuffer(kBuffer, offset: 0, index: 1)
        enc.setBuffer(vBuffer, offset: 0, index: 2)
        enc.setBuffer(oBuffer, offset: 0, index: 3)
        enc.setBytes(&p, length: MemoryLayout<FullParams>.stride, index: 4)
        enc.setBuffer(dummy, offset: 0, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: (count + rowsPerGroup - 1) / rowsPerGroup,
                    height: fullHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
        waitAndCheck(cmd, "\(function) start=\(start) count=\(count)")
        return Fp16Buffer.read(oBuffer, count: count * qStride)
    }

    struct UnfusedParams { var headBase: UInt32; var scoreStride: UInt32; var headRows: UInt32 }

    /// The three unfused kernels over head groups of `group`, as
    /// `PrefillAttention.encodeFullUnfused` dispatches them.
    func runUnfused(group: Int, start: Int, count: Int) throws -> [Float] {
        let slots = total + 64
        var kPhys = [Float16](repeating: .nan, count: slots * kvStride)
        var vPhys = [Float16](repeating: .nan, count: slots * kvStride)
        for i in 0..<((start + count) * kvStride) {
            kPhys[i] = k[i]
            vPhys[i] = v[i]
        }
        let qRows = Array(q[(start * qStride)..<((start + count) * qStride)])
        let scores = try context.pipeline("attention_prefill_full_scores_d512")
        let softmax = try context.pipeline("attention_prefill_full_softmax_d512")
        let pv = try context.pipeline("attention_prefill_full_pv_d512")
        let stride = (start + count + 63) / 64 * 64
        let sBytes = group * count * stride * 4, pBytes = group * count * stride * 2
        guard let qBuffer = Fp16Buffer.make(device, halves: qRows),
              let kBuffer = Fp16Buffer.make(device, halves: kPhys),
              let vBuffer = Fp16Buffer.make(device, halves: vPhys),
              let oBuffer = Fp16Buffer.make(device, count: count * qStride),
              let scratch = device.makeBuffer(length: sBytes + pBytes + group * count * 4,
                                              options: .storageModePrivate),
              let cmd = context.queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            fatalError("prefill full attention allocation failed")
        }
        var p = FullParams(startPosition: UInt32(start), queryCount: UInt32(count),
                           headDim: UInt32(fullHeadDim), numQHeads: UInt32(fullHeads),
                           numKVHeads: UInt32(fullKVHeads), kvValidCount: UInt32(start + count),
                           slidingWindow: 0, kvTokenStrideElements: UInt32(kvStride),
                           qTokenStrideElements: UInt32(qStride), oTokenStrideElements: UInt32(qStride),
                           scale: 1.0, hasSpanMask: 0)
        let rowTiles = (count + 63) / 64
        let threads = MTLSize(width: 128, height: 1, depth: 1)
        for headBase in Swift.stride(from: 0, to: fullHeads, by: group) {
            var u = UnfusedParams(headBase: UInt32(headBase), scoreStride: UInt32(stride),
                                  headRows: UInt32(count))
            enc.setComputePipelineState(scores)
            enc.setBuffer(qBuffer, offset: 0, index: 0)
            enc.setBuffer(kBuffer, offset: 0, index: 1)
            enc.setBuffer(scratch, offset: 0, index: 2)
            enc.setBytes(&p, length: MemoryLayout<FullParams>.stride, index: 4)
            enc.setBytes(&u, length: MemoryLayout<UnfusedParams>.stride, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: stride / 64, height: rowTiles, depth: group),
                                     threadsPerThreadgroup: threads)
            enc.setComputePipelineState(softmax)
            enc.setBuffer(scratch, offset: sBytes, index: 3)
            enc.setBuffer(scratch, offset: sBytes + pBytes, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: (count + 3) / 4, height: group, depth: 1),
                                     threadsPerThreadgroup: threads)
            enc.setComputePipelineState(pv)
            enc.setBuffer(oBuffer, offset: 0, index: 0)
            enc.setBuffer(vBuffer, offset: 0, index: 1)
            enc.dispatchThreadgroups(MTLSize(width: fullHeadDim / 64, height: rowTiles, depth: group),
                                     threadsPerThreadgroup: threads)
        }
        enc.endEncoding()
        waitAndCheck(cmd, "unfused group=\(group) start=\(start) count=\(count)")
        return Fp16Buffer.read(oBuffer, count: count * qStride)
    }

    let candidates: [(label: String, function: String, threads: Int, rows: Int)] = [
        ("mpp32", "attention_prefill_full_mpp_d512_r32", 128, 32),
        ("mpp64", "attention_prefill_full_mpp_d512_r64", 256, 64),
    ]
    var cases: [CaseResult] = []
    for (start, count) in [(0, 150), (150, 70)] {
        let name = "prefill-full-attn start=\(start) t=\(count)"
        let expected = fullReference(q: q, k: k, v: v, start: start, count: count)
        // 2 queries per simdgroup, 8 simdgroups (`PrefillAttention.qBlockSpecialisations`).
        let qblock = try run("attention_prefill_causal_qblock_d512", threads: 256,
                             rowsPerGroup: 16, start: start, count: count)
        cases.append(result(name + " qblock", groupSize: groupSize,
                            rel: relativeError(actual: qblock, reference: expected),
                            tolerance: Double(Tolerance.fp16ChainedReduction)))
        for candidate in candidates {
            guard (try? context.pipeline(candidate.function)) != nil else {
                print("  SKIP  \(name) \(candidate.label)  no MPP pipeline on this device")
                continue
            }
            let mpp = try run(candidate.function, threads: candidate.threads,
                              rowsPerGroup: candidate.rows, start: start, count: count)
            cases.append(result(name + " " + candidate.label, groupSize: groupSize,
                                rel: relativeError(actual: mpp, reference: expected),
                                tolerance: Double(Tolerance.fp16ChainedReduction)))
            if start == 0 && candidate.label == "mpp64" {
                let wrong = fullReference(q: q, k: k, v: v, start: start, count: count, lag: 1)
                cases.append(detectionResult(name + " " + candidate.label + " vs lag 1",
                                             groupSize: groupSize,
                                             rel: relativeError(actual: mpp, reference: wrong),
                                             floor: 10 * Double(Tolerance.fp16ChainedReduction)))
            }
        }
        guard (try? context.pipeline("attention_prefill_full_scores_d512")) != nil else {
            print("  SKIP  \(name) unfused  no MPP pipeline on this device")
            continue
        }
        for group in [2, 16] {
            let unfused = try runUnfused(group: group, start: start, count: count)
            cases.append(result(name + " unfused group=\(group)", groupSize: groupSize,
                                rel: relativeError(actual: unfused, reference: expected),
                                tolerance: Double(Tolerance.fp16ChainedReduction)))
            if start == 0 && group == 2 {
                let wrong = fullReference(q: q, k: k, v: v, start: start, count: count, lag: 1)
                cases.append(detectionResult(name + " unfused vs lag 1", groupSize: groupSize,
                                             rel: relativeError(actual: unfused, reference: wrong),
                                             floor: 10 * Double(Tolerance.fp16ChainedReduction)))
            }
        }
    }
    return cases
}
