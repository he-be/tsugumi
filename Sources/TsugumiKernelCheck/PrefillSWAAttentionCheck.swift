import Foundation
import Metal
import Tsugumi
import TsugumiValidationSupport

// MARK: - Case 13 — prefill sliding-window attention, headDim 256 (sgmm and MPP)

// The sliding-window layers' prefill attention at the production head shape
// (16 query heads over 8 KV heads, headDim 256), against a CPU softmax over
// the FP16 inputs. `attention_prefill_causal_sgmm_gqa_d256` (the default) runs
// as the control; `attention_prefill_swa_mpp_d256` (`matmul2d`,
// `TF_PREFILL_ATTN=mpp`, docs/m6-prefill/25) only exists in an MSL 4.0 build
// and is SKIPped elsewhere with the skip printed.
//
// Shapes: a window narrower than the chunk, a chunk that ends mid-block, a
// second chunk over the first one's keys, and the same second chunk read
// through a ring (FC_PREFILL_KV_RING_CAP) that has already wrapped. A wrong
// window scored against the same output shows the comparison can fail.

private let swaHeads = 16
private let swaKVHeads = 8
private let swaHeadDim = 256

/// `PrefillAttentionParams` in prefill.metal.
private struct SWAParams {
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

/// Softmax over keys `max(0, abs + 1 - window) ..< abs + 1` for each row.
private func swaReference(q: [Float16], k: [Float16], v: [Float16],
                          start: Int, count: Int, window: Int) -> [Float] {
    let qStride = swaHeads * swaHeadDim, kvStride = swaKVHeads * swaHeadDim
    var out = [Float](repeating: 0, count: count * qStride)
    for t in 0..<count {
        let abs = start + t
        let first = Swift.max(0, abs + 1 - window)
        for h in 0..<swaHeads {
            let kvh = h / (swaHeads / swaKVHeads)
            var scores: [Double] = []
            for key in first...abs {
                var s = 0.0
                for d in 0..<swaHeadDim {
                    s += Double(q[abs * qStride + h * swaHeadDim + d])
                        * Double(k[key * kvStride + kvh * swaHeadDim + d])
                }
                scores.append(s)
            }
            let m = scores.max()!
            let w = scores.map { Foundation.exp($0 - m) }
            let sum = w.reduce(0, +)
            for d in 0..<swaHeadDim {
                var acc = 0.0
                for (i, key) in (first...abs).enumerated() {
                    acc += w[i] * Double(v[key * kvStride + kvh * swaHeadDim + d])
                }
                out[t * qStride + h * swaHeadDim + d] = Float(acc / sum)
            }
        }
    }
    return out
}

func checkPrefillSWAAttention(groupSize: Int, seed: UInt64) throws -> [CaseResult] {
    var rng = SeedTree(seed).key("prefill-swa-attention")
    let window = 100
    let total = 220
    let qStride = swaHeads * swaHeadDim, kvStride = swaKVHeads * swaHeadDim
    let q = (0..<(total * qStride)).map { _ in Float16(rng.uniform(-0.25, 0.25)) }
    let k = (0..<(total * kvStride)).map { _ in Float16(rng.uniform(-0.25, 0.25)) }
    let v = (0..<(total * kvStride)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }

    let context = try makeContext(groupSize: groupSize)
    let device = context.device
    guard let dummy = device.makeBuffer(length: 16, options: .storageModeShared) else {
        fatalError("prefill SWA attention allocation failed")
    }

    /// One dispatch of `function` over rows `start ..< start + count`, with K/V
    /// stored at `logical % ring` when `ring` is set.
    func run(_ function: String, threads: Int, rowsPerGroup: Int, headsPerGroup: Int,
             start: Int, count: Int, ring: Int) throws -> [Float] {
        let slots = ring > 0 ? ring : total
        var kPhys = [Float16](repeating: .nan, count: slots * kvStride)
        var vPhys = [Float16](repeating: .nan, count: slots * kvStride)
        // Every logical key up to the chunk's end, in order, so a wrapped slot
        // holds the newest key that maps to it. Slots never written stay NaN.
        for key in 0..<(start + count) {
            let slot = ring > 0 ? key % ring : key
            for i in 0..<kvStride {
                kPhys[slot * kvStride + i] = k[key * kvStride + i]
                vPhys[slot * kvStride + i] = v[key * kvStride + i]
            }
        }
        let qRows = Array(q[(start * qStride)..<((start + count) * qStride)])
        let constants = ring > 0
            ? [MetalFunctionConstant(index: 76, value: .uint32(UInt32(ring)))] : []
        let pso = try context.pipeline(function, constants: constants)
        guard let qBuffer = Fp16Buffer.make(device, halves: qRows),
              let kBuffer = Fp16Buffer.make(device, halves: kPhys),
              let vBuffer = Fp16Buffer.make(device, halves: vPhys),
              let oBuffer = Fp16Buffer.make(device, count: count * qStride),
              let cmd = context.queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            fatalError("prefill SWA attention allocation failed")
        }
        var p = SWAParams(startPosition: UInt32(start), queryCount: UInt32(count),
                          headDim: UInt32(swaHeadDim), numQHeads: UInt32(swaHeads),
                          numKVHeads: UInt32(swaKVHeads), kvValidCount: UInt32(start + count),
                          slidingWindow: UInt32(window), kvTokenStrideElements: UInt32(kvStride),
                          qTokenStrideElements: UInt32(qStride), oTokenStrideElements: UInt32(qStride),
                          scale: 1.0, hasSpanMask: 0)
        enc.setComputePipelineState(pso)
        enc.setBuffer(qBuffer, offset: 0, index: 0)
        enc.setBuffer(kBuffer, offset: 0, index: 1)
        enc.setBuffer(vBuffer, offset: 0, index: 2)
        enc.setBuffer(oBuffer, offset: 0, index: 3)
        enc.setBytes(&p, length: MemoryLayout<SWAParams>.stride, index: 4)
        enc.setBuffer(dummy, offset: 0, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: (count + rowsPerGroup - 1) / rowsPerGroup,
                    height: swaHeads / headsPerGroup, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
        waitAndCheck(cmd, "\(function) start=\(start) count=\(count) ring=\(ring)")
        return Fp16Buffer.read(oBuffer, count: count * qStride)
    }

    let hasMPP = (try? context.pipeline("attention_prefill_swa_mpp_d256")) != nil
    var cases: [CaseResult] = []
    // (start, count, ring): one chunk ending mid-block, a second chunk over the
    // first one's keys, and that second chunk through a wrapped 192-slot ring.
    for (start, count, ring) in [(0, 150, 0), (150, 70, 0), (150, 70, 192)] {
        let name = "prefill-swa-attn start=\(start) t=\(count) window=\(window) ring=\(ring)"
        let expected = swaReference(q: q, k: k, v: v, start: start, count: count, window: window)
        let sgmm = try run("attention_prefill_causal_sgmm_gqa_d256", threads: 256,
                           rowsPerGroup: 32, headsPerGroup: 2,
                           start: start, count: count, ring: ring)
        cases.append(result(name + " sgmm-gqa", groupSize: groupSize,
                            rel: relativeError(actual: sgmm, reference: expected),
                            tolerance: Double(Tolerance.fp16ChainedReduction)))
        guard hasMPP else {
            print("  SKIP  \(name) mpp  no MPP pipeline on this device")
            continue
        }
        let mpp = try run("attention_prefill_swa_mpp_d256", threads: 128,
                          rowsPerGroup: 64, headsPerGroup: 1,
                          start: start, count: count, ring: ring)
        cases.append(result(name + " mpp", groupSize: groupSize,
                            rel: relativeError(actual: mpp, reference: expected),
                            tolerance: Double(Tolerance.fp16ChainedReduction)))
        if ring == 0 && start == 0 {
            let wrong = swaReference(q: q, k: k, v: v, start: start, count: count,
                                     window: window / 2)
            cases.append(detectionResult(name + " mpp vs window/2", groupSize: groupSize,
                                         rel: relativeError(actual: mpp, reference: wrong),
                                         floor: 10 * Double(Tolerance.fp16ChainedReduction)))
        }
    }
    return cases
}
