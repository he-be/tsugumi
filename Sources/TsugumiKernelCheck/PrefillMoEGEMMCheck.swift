import Accelerate
import Foundation
import Metal
import Tsugumi
import TsugumiValidationSupport

// MARK: - Case 12 — prefill routed MoE, tiled GEMM (simdgroup and MPP)

// The prompt-chunk routed path (`encodeStreamedTiled`): gate and up as one
// GEMM over gathered hidden rows, the activation pass, then down scattered into
// route_partials. It runs here with both tile kernels at the production shapes
// (D = 2816, F = 704): `prefill_moe_gemm_int4` (simdgroup) and
// `prefill_moe_gemm_int4_mpp` (`matmul2d`, docs/m6-prefill/04-GATES.md G3).
// The MPP kernel only exists in an MSL 4.0 build, so it is SKIPped elsewhere
// with the skip printed.
//
// Routes are chosen so the block list has the shapes production produces: one
// expert with more than 64 rows (a full block plus a partial one), several
// narrow experts, and scattered tokens in every block.

private let moeGEMMD = 2816
private let moeGEMMF = 704
private let moeGEMMTokens = 72
private let moeGEMMTopK = 4
private let moeGEMMExperts = 16
private let moeGEMMTileM = 64

/// `PrefillGroupedRoutedMoEStreamedParamsMSL`.
private struct MoEGEMMParams {
    var pairStart: UInt32 = 0
    var pairCount: UInt32 = 0
    var d: UInt32 = 0
    var f: UInt32 = 0
    var topK: UInt32 = 0
    var hiddenStrideElements: UInt32 = 0
    var liveExpertCount: UInt32 = 0
    var localExperts: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                       UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) =
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    var gateWOff: UInt32 = 0
    var gateSOff: UInt32 = 0
    var gateBOff: UInt32 = 0
    var upWOff: UInt32 = 0
    var upSOff: UInt32 = 0
    var upBOff: UInt32 = 0
    var downWOff: UInt32 = 0
    var downSOff: UInt32 = 0
    var downBOff: UInt32 = 0
}

/// `PrefillRoutedGEMMBlockMSL`.
private struct MoEGEMMBlock {
    var localSlot: UInt32
    var pairStart: UInt32
    var rowCount: UInt32
    var localRow: UInt32
}

/// `PrefillTokenExpertPairMSL`.
private struct MoEGEMMPair {
    var token: UInt32
    var expert: UInt32
    var rank: UInt32
    var weightBits: UInt32
}

private struct MoEGEMMExpert {
    let gate: [Quantization.Int4AffineRow]   // F rows of D
    let up: [Quantization.Int4AffineRow]     // F rows of D
    let down: [Quantization.Int4AffineRow]   // D rows of F
}

private func gelu(_ x: Float) -> Float {
    let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1 + tanh(inner))
}

/// Row-major dequantized matrix, `rows x cols`.
private func dequantized(_ rows: [Quantization.Int4AffineRow], cols: Int,
                         groupSize: Int) -> [Float] {
    rows.flatMap { Quantization.dequantizeInt4Affine($0, n: cols, groupSize: groupSize) }
}

private func dot(_ matrix: UnsafePointer<Float>, row: Int, cols: Int,
                 _ x: UnsafePointer<Float>) -> Float {
    var out: Float = 0
    vDSP_dotpr(matrix + row * cols, 1, x, 1, &out, vDSP_Length(cols))
    return out
}

/// Returns one case per available kernel plus, when both ran, their agreement.
func checkPrefillRoutedMoEGEMM(groupSize: Int, seed: UInt64) throws -> [CaseResult] {
    let d = moeGEMMD, f = moeGEMMF, tokens = moeGEMMTokens, topK = moeGEMMTopK
    var rng = SeedTree(seed).key("prefill-moe-gemm-g\(groupSize)")

    // Weights are drawn at a quarter of `quantizedRows`' range so that
    // gate, up and the down sum stay well inside FP16.
    func rows(_ count: Int, _ cols: Int) -> [Quantization.Int4AffineRow] {
        (0..<count).map { _ in
            let raw = (0..<cols).map { _ in rng.uniform(-0.125, 0.125) }
            return affineScheme == .sym
                ? Quantization.quantizeInt4Symmetric(raw, groupSize: groupSize)
                : Quantization.quantizeInt4Affine(raw, groupSize: groupSize)
        }
    }
    let experts = (0..<moeGEMMExperts).map { _ in
        MoEGEMMExpert(gate: rows(f, d), up: rows(f, d), down: rows(d, f))
    }
    let hidden = (0..<(tokens * d)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }

    // Rank 0 always goes to expert 0 (72 rows: a full block and a partial one);
    // the other ranks spread over experts 1...15.
    var byExpert = [[MoEGEMMPair]](repeating: [], count: moeGEMMExperts)
    for token in 0..<tokens {
        let picks = [0, 1 + token % 5, 6 + token % 4, 10 + token % 6]
        for (rank, expert) in picks.enumerated() {
            byExpert[expert].append(MoEGEMMPair(token: UInt32(token), expert: UInt32(expert),
                                                rank: UInt32(rank), weightBits: 0))
        }
    }
    var pairs: [MoEGEMMPair] = []
    var blocks: [MoEGEMMBlock] = []
    for expert in 0..<moeGEMMExperts {
        var start = 0
        while start < byExpert[expert].count {
            let count = min(moeGEMMTileM, byExpert[expert].count - start)
            blocks.append(MoEGEMMBlock(localSlot: UInt32(expert),
                                       pairStart: UInt32(pairs.count + start),
                                       rowCount: UInt32(count),
                                       localRow: UInt32(pairs.count + start)))
            start += count
        }
        pairs.append(contentsOf: byExpert[expert])
    }
    let rowTotal = pairs.count
    let routesByExpert = byExpert

    // CPU reference, rounded where the GPU path stores FP16: gate and up, the
    // activation, and the output.
    var expected = [Float](repeating: 0, count: tokens * topK * d)
    expected.withUnsafeMutableBufferPointer { outBuffer in
        nonisolated(unsafe) let out = outBuffer.baseAddress!
        DispatchQueue.concurrentPerform(iterations: moeGEMMExperts) { expert in
            let gate = dequantized(experts[expert].gate, cols: d, groupSize: groupSize)
            let up = dequantized(experts[expert].up, cols: d, groupSize: groupSize)
            let down = dequantized(experts[expert].down, cols: f, groupSize: groupSize)
            for pair in routesByExpert[expert] {
                let base = Int(pair.token) * d
                let x = (0..<d).map { Float(hidden[base + $0]) }
                var act = [Float](repeating: 0, count: f)
                x.withUnsafeBufferPointer { px in
                    gate.withUnsafeBufferPointer { pg in
                        up.withUnsafeBufferPointer { pu in
                            for row in 0..<f {
                                let g = Float(Float16(dot(pg.baseAddress!, row: row, cols: d, px.baseAddress!)))
                                let u = Float(Float16(dot(pu.baseAddress!, row: row, cols: d, px.baseAddress!)))
                                act[row] = Float(Float16(gelu(g) * u))
                            }
                        }
                    }
                }
                let dst = (Int(pair.token) * topK + Int(pair.rank)) * d
                act.withUnsafeBufferPointer { pa in
                    down.withUnsafeBufferPointer { pd in
                        for row in 0..<d {
                            out[dst + row] = Float(Float16(
                                dot(pd.baseAddress!, row: row, cols: f, pa.baseAddress!)))
                        }
                    }
                }
            }
        }
    }

    // Expert blobs in the packer's order: gate (W, S, B), up, down.
    var params = MoEGEMMParams()
    var blobs: [[UInt8]] = []
    for expert in experts {
        var bytes: [UInt8] = []
        func appendHalves(_ values: [UInt16]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        let g = packRows(expert.gate), u = packRows(expert.up), dn = packRows(expert.down)
        params.gateWOff = UInt32(bytes.count); bytes.append(contentsOf: g.packed)
        params.gateSOff = UInt32(bytes.count); appendHalves(g.scales)
        params.gateBOff = UInt32(bytes.count); appendHalves(g.biases)
        params.upWOff = UInt32(bytes.count); bytes.append(contentsOf: u.packed)
        params.upSOff = UInt32(bytes.count); appendHalves(u.scales)
        params.upBOff = UInt32(bytes.count); appendHalves(u.biases)
        params.downWOff = UInt32(bytes.count); bytes.append(contentsOf: dn.packed)
        params.downSOff = UInt32(bytes.count); appendHalves(dn.scales)
        params.downBOff = UInt32(bytes.count); appendHalves(dn.biases)
        blobs.append(bytes)
    }
    params.pairCount = UInt32(rowTotal)
    params.d = UInt32(d)
    params.f = UInt32(f)
    params.topK = UInt32(topK)
    params.hiddenStrideElements = UInt32(d)
    params.liveExpertCount = UInt32(moeGEMMExperts)

    let context = try makeContext(groupSize: groupSize)
    let device = context.device
    let blobBuffers = blobs.compactMap {
        device.makeBuffer(bytes: $0, length: $0.count, options: .storageModeShared)
    }
    guard blobBuffers.count == moeGEMMExperts,
          let function = try context.library.makeFunction(name: "prefill_moe_gemm_int4") else {
        fatalError("prefill routed GEMM fixture failed")
    }
    let argumentEncoder = function.makeArgumentEncoder(bufferIndex: 9)
    guard let argumentBuffer = device.makeBuffer(length: argumentEncoder.encodedLength,
                                                 options: .storageModeShared),
          let pairBuffer = device.makeBuffer(bytes: pairs,
                                             length: pairs.count * MemoryLayout<MoEGEMMPair>.stride,
                                             options: .storageModeShared),
          let hiddenBuffer = Fp16Buffer.make(device, halves: hidden) else {
        fatalError("prefill routed GEMM allocation failed")
    }
    argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
    for (index, blob) in blobBuffers.enumerated() {
        argumentEncoder.setBuffer(blob, offset: 0, index: index)
    }
    let geluMul = try context.pipeline("prefill_moe_gate_up_gelu_mul")

    func run(_ pso: MTLComputePipelineState, label: String) -> [Float] {
        guard let act = Fp16Buffer.make(device, count: 3 * rowTotal * f),
              let partials = Fp16Buffer.make(device, halves: [Float16](
                repeating: Float16(-7.0), count: tokens * topK * d)),
              let cmd = context.queue.makeCommandBuffer() else {
            fatalError("prefill routed GEMM allocation failed")
        }
        func gemm(mode: UInt32, nTiles: Int, depth: Int) {
            guard let enc = cmd.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(pso)
            enc.setBuffer(hiddenBuffer, offset: 0, index: 0)
            enc.setBuffer(pairBuffer, offset: 0, index: 1)
            var blockCopy = blocks
            enc.setBytes(&blockCopy, length: blockCopy.count * MemoryLayout<MoEGEMMBlock>.stride,
                         index: 2)
            enc.setBuffer(partials, offset: 0, index: 5)
            enc.setBuffer(act, offset: 0, index: 7)
            enc.setBuffer(argumentBuffer, offset: 0, index: 9)
            var p = params
            enc.setBytes(&p, length: MemoryLayout<MoEGEMMParams>.stride, index: 10)
            var m = mode
            enc.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 11)
            for blob in blobBuffers { enc.useResource(blob, usage: .read) }
            enc.dispatchThreadgroups(MTLSize(width: nTiles, height: blocks.count, depth: depth),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.endEncoding()
        }
        gemm(mode: 0, nTiles: (f + 63) / 64, depth: 2)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(geluMul)
            enc.setBuffer(act, offset: 0, index: 0)
            var r = UInt32(rowTotal), fv = UInt32(f)
            enc.setBytes(&r, length: 4, index: 1)
            enc.setBytes(&fv, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: rowTotal * f, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        gemm(mode: 1, nTiles: (d + 63) / 64, depth: 1)
        waitAndCheck(cmd, label)
        return Fp16Buffer.read(partials, count: tokens * topK * d)
    }

    let name = "prefill-moe-gemm t=\(tokens) topK=\(topK) d=\(d) f=\(f)"
    let detail = "rows=\(rowTotal) blocks=\(blocks.count)"
    var cases: [CaseResult] = []
    let simdgroup = run(try context.pipeline("prefill_moe_gemm_int4"), label: name + " simdgroup")
    cases.append(result(name + " simdgroup", groupSize: groupSize,
                        rel: relativeError(actual: simdgroup, reference: expected),
                        tolerance: Double(Tolerance.fp16ChainedReduction), detail: detail))
    guard let mppPSO = try? context.pipeline("prefill_moe_gemm_int4_mpp") else {
        print("  SKIP  \(name) mpp g=\(groupSize)  no MPP pipeline on this device")
        return cases
    }
    let mpp = run(mppPSO, label: name + " mpp")
    cases.append(result(name + " mpp", groupSize: groupSize,
                        rel: relativeError(actual: mpp, reference: expected),
                        tolerance: Double(Tolerance.fp16ChainedReduction), detail: detail))
    // How many outputs the reduction order actually moved, and by how much.
    var differing = 0
    var maxAbs: Float = 0
    for (a, b) in zip(mpp, simdgroup) where a != b {
        differing += 1
        maxAbs = max(maxAbs, abs(a - b))
    }
    cases.append(result(name + " mpp vs simdgroup", groupSize: groupSize,
                        rel: relativeError(actual: mpp, reference: simdgroup),
                        tolerance: Double(Tolerance.fp16ChainedReduction),
                        detail: "differing=\(differing)/\(mpp.count) maxAbs=\(maxAbs)"))

    // The prefetch candidate (docs/m6-prefill/28) moves only when loads are
    // issued, so it has to match the mpp run bit for bit.
    guard let pfPSO = try? context.pipeline("prefill_moe_gemm_int4_mpp_pf") else {
        print("  SKIP  \(name) mpp-pf g=\(groupSize)  no pipeline on this device")
        return cases
    }
    let pf = run(pfPSO, label: name + " mpp-pf")
    cases.append(result(name + " mpp-pf", groupSize: groupSize,
                        rel: relativeError(actual: pf, reference: expected),
                        tolerance: Double(Tolerance.fp16ChainedReduction), detail: detail))
    let pfDiffering = zip(pf, mpp).filter { $0 != $1 }.count
    cases.append(result(name + " mpp-pf vs mpp", groupSize: groupSize,
                        rel: pfDiffering == 0 ? 0 : 1,
                        tolerance: 0,
                        detail: "differing=\(pfDiffering)/\(pf.count)"))

    // Symmetric builds also carry the half-precision weight staging
    // (`_hd`, docs/m6-prefill/28), which claims the same rounding.
    if let hdPSO = try? context.pipeline("prefill_moe_gemm_int4_mpp_pf_hd") {
        let hd = run(hdPSO, label: name + " mpp-pf-hd")
        let hdDiffering = zip(hd, mpp).filter { $0 != $1 }.count
        cases.append(result(name + " mpp-pf-hd vs mpp", groupSize: groupSize,
                            rel: hdDiffering == 0 ? 0 : 1,
                            tolerance: 0,
                            detail: "differing=\(hdDiffering)/\(hd.count)"))
    }
    return cases
}
