import Darwin
import Foundation
import Metal
import MetalPerformanceShaders
import Tsugumi

// MARK: - `--qfn-layer-bench <gguf> <down sidecar> <route .bin> resident|cold [iterations] [arms]`
//
// docs/qfn-prefill/12 F1: one layer's GEMMs at T = 2048 -- the dense projections (Q8_0 attention / GDN in-out,
// F16 hyper-connection down/up), the shared expert, and the routed experts of a real route dump (`Q38_DUMP_ROUTE`) --
// through three arms, with the layer's weights read the way the runner reads them:
//   A  today's code: GGMLDenseGEMV (dequantize to float32 + MPS sgemm) and the per-pair IQ2_XXS / Q2_K kernels.
//      The reference for the precision check.
//   C  fp16: Q8_0 dequantized to half + MPS fp16 matrix multiplication (F16 weights read as they are); routed as
//      Q2GemmBench's arm C (half dequant of 4 experts at a time + MPS fp16).
//   D  the Metal 4 tensor ops (bench/qfn/qfn_nax.metal, the ds4-metal port): 64 x 64 tiles over the quantized rows.
//      Needs MSL 4.0 and the tensor API (the M6); skipped where it does not compile.
//   M  C's dense groups with D's routed tiles (the faster arm per group on the M6's first run).
// The small float projections (router, ssm_alpha / beta, the hyper-connection inject, the shared expert's gate) stay
// on GGMLDenseGEMV in every arm (group "fp32"). The attention / GDN core, norms and element-wise work are not here.
// Inputs are random; each GEMM reads its own random x, so the groups do not depend on each other.
//
// Weights: the layer's dense tensors and its selected experts (gate/up from the GGUF, down from the 212 B sidecar
// as the runner), read with `preadRanges` on 4 threads (the runner's `Q38_READ_THREADS` default) into the page cache,
// used through per-tensor / per-expert no-copy views.
//   resident: read once, then `iterations` rounds of every arm (after one warm-up) with the read repeated each
//             round (a read of cached pages); prints medians and the precision of C and D against A.
//   cold:     the single arm in `arms`, one process per run (bench/qfn/f1_layer.sh): a warm-up round, then the layer's
//             ranges are dropped from the page cache (`msync(MS_INVALIDATE)` over a fresh read-only mapping; reading
//             15 GB of other files did not push them out on the M6) and one round is timed with the read from cold.
// GPU ms per group are separate command buffers, committed and waited in turn.

private let lD = 2560, lF = 640, lActStride = 768, lExperts = 512, lTopK = 10

private struct DenseItem {
    let name: String, group: String
    let tensor: GGUFFile.Tensor
    var m: Int { tensor.rowCount }
    var k: Int { tensor.rowWidth }
}

func runQFNLayerBench(ggufPath: String, sidecarPath: String, routePath: String, mode: String,
                      iterations: Int, arms armList: String) throws {
    let file = try GGUFFile(url: URL(fileURLWithPath: (ggufPath as NSString).expandingTildeInPath))
    let side = try GGUFFile(url: URL(fileURLWithPath: (sidecarPath as NSString).expandingTildeInPath))
    let context = try MetalContext()
    let device = context.device, queue = context.queue
    let cold = mode == "cold"
    precondition(cold || mode == "resident", "mode is resident or cold")
    var arms = armList.map { String($0) }
    let needC = arms.contains("C") || arms.contains("M"), needD = arms.contains("D") || arms.contains("M")
    precondition(!cold || arms.count == 1, "cold runs one arm")

    // Route.
    let il = Int(routePath.split(separator: "-").last!.dropFirst().split(separator: ".").first!)!
    let route = try Data(contentsOf: URL(fileURLWithPath: routePath))
    let T0 = Int(route.withUnsafeBytes { $0.load(as: Int32.self) })
    let T = min(T0, 2048), P = T * lTopK, P0 = T0 * lTopK
    var experts = [Int32](repeating: 0, count: P), weights = [Float](repeating: 0, count: P)
    route.withUnsafeBytes { raw in
        experts.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[4..<(4 + 4 * P)])) }
        weights.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[(4 + 4 * P0)..<(4 + 4 * P0 + 4 * P)])) }
    }
    var slotOf = [Int](repeating: -1, count: lExperts)
    var slots: [Int] = []
    var pairSlot = [UInt32](repeating: 0, count: P)
    for p in 0..<P {
        let ex = Int(experts[p])
        if slotOf[ex] < 0 { slotOf[ex] = slots.count; slots.append(ex) }
        pairSlot[p] = UInt32(slotOf[ex])
    }
    let S = slots.count
    var count = [Int](repeating: 0, count: S)
    for p in 0..<P { count[Int(pairSlot[p])] += 1 }
    var start = [Int](repeating: 0, count: S + 1)
    for s in 0..<S { start[s + 1] = start[s] + count[s] }
    var fill = start
    var order = [UInt32](repeating: 0, count: P), at = [UInt32](repeating: 0, count: P)
    for p in 0..<P {
        let s = Int(pairSlot[p])
        order[fill[s]] = UInt32(p); at[p] = UInt32(fill[s]); fill[s] += 1
    }
    let listCap = count.max()!
    var lists = [Int32](repeating: -1, count: S * listCap)
    for s in 0..<S { for i in 0..<count[s] { lists[s * listCap + i] = Int32(order[start[s] + i]) } }
    let tiles = min(8, (listCap + 63) / 64)

    // Tensors of the layer.
    let pre = "blk.\(il)."
    let small: Set<String> = ["ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "ssm_alpha.weight", "ssm_beta.weight",
                              "hc_attn_inject.weight", "hc_ffn_inject.weight"]
    var items: [DenseItem] = []
    for t in file.tensors.values where t.name.hasPrefix(pre) && !t.name.contains("_exps") {
        let short = String(t.name.dropFirst(pre.count))
        if small.contains(short) {
            items.append(DenseItem(name: short, group: "fp32", tensor: t))
        } else if t.dims.count == 2 && (t.type == .q8_0 || t.type == .f16) && t.rowCount >= 64 {
            items.append(DenseItem(name: short, group: short.hasPrefix("hc_") ? "hc" : short.contains("shexp") ? "shexp" : "proj",
                                   tensor: t))
        }
    }
    items.sort { $0.name < $1.name }
    let isLinear = items.contains { $0.name == "attn_qkv.weight" }
    let gateT = try file.tensor(pre + "ffn_gate_exps.weight"), upT = try file.tensor(pre + "ffn_up_exps.weight")
    let downT = try side.tensor(pre + "ffn_down_exps.weight")
    let gateRow = gateT.bytesPerRow, downRow = downT.byteCount / (lD * lExperts)
    precondition(downRow == 212, "the down sidecar's rows are 212 B")

    // Byte ranges, adjacent experts merged (the runner's preadRanges runs).
    var denseRanges: [(offset: Int, byteCount: Int)] = items.map { ($0.tensor.offset, $0.tensor.byteCount) }
    denseRanges.sort { $0.offset < $1.offset }
    func expertRanges(_ t: GGUFFile.Tensor, _ per: Int) -> [(offset: Int, byteCount: Int)] {
        var r: [(offset: Int, byteCount: Int)] = []
        for ex in slots.sorted() {
            let o = t.offset + ex * per
            if let last = r.last, last.offset + last.byteCount == o { r[r.count - 1].byteCount += per } else { r.append((o, per)) }
        }
        return r
    }
    let guRanges = expertRanges(gateT, gateRow * lF) + expertRanges(upT, gateRow * lF)
    let downRanges = expertRanges(downT, downRow * lD)
    let mb = { (r: [(offset: Int, byteCount: Int)]) in Double(r.reduce(0) { $0 + $1.byteCount }) / 1e6 }
    func coldMB() -> (Double, Double) {
        let d = denseRanges.reduce(0) { $0 + file.nonResidentBytes(offset: $1.offset, byteCount: $1.byteCount) }
        let e = guRanges.reduce(0) { $0 + file.nonResidentBytes(offset: $1.offset, byteCount: $1.byteCount) }
            + downRanges.reduce(0) { $0 + side.nonResidentBytes(offset: $1.offset, byteCount: $1.byteCount) }
        return (Double(d) / 1e6, Double(e) / 1e6)
    }
    func readAll() -> Double {
        let t0 = CFAbsoluteTimeGetCurrent()
        GGUFFile.preadRanges(denseRanges.map { (file, $0.offset, $0.byteCount) } + guRanges.map { (file, $0.offset, $0.byteCount) }
                             + downRanges.map { (side, $0.offset, $0.byteCount) }, threads: 4)
        return CFAbsoluteTimeGetCurrent() - t0
    }
    /// Drops the layer's ranges from the page cache: `msync(MS_INVALIDATE)` over a fresh read-only mapping.
    func invalidate(_ path: String, _ ranges: [(offset: Int, byteCount: Int)]) {
        let fd = open((path as NSString).expandingTildeInPath, O_RDONLY)
        precondition(fd >= 0, "open \(path)")
        defer { close(fd) }
        let page = Int(getpagesize())
        for r in ranges {
            let a = r.offset / page * page, n = r.byteCount + (r.offset - a)
            let p = mmap(nil, n, PROT_READ, MAP_SHARED, fd, off_t(a))!
            precondition(p != MAP_FAILED && msync(p, n, MS_INVALIDATE) == 0, "msync")
            munmap(p, n)
        }
    }
    let sortedCounts = count.sorted()
    print(String(format: "layer %d (%@) T=%d: %d experts, pairs per expert median %d / max %d, %d tiles per launch",
                 il, isLinear ? "gdn" : "attention", T, S, sortedCounts[S / 2], listCap, tiles))
    print(String(format: "  bytes MB: dense %.1f (%d tensors), experts %.1f (gate/up %.1f, down sidecar %.1f)",
                 mb(denseRanges), items.count, mb(guRanges) + mb(downRanges), mb(guRanges), mb(downRanges)))
    print("  groups: " + ["proj", "hc", "shexp", "fp32"].map { g in
        "\(g) [" + items.filter { $0.group == g }.map { "\($0.name.replacingOccurrences(of: ".weight", with: "")) \($0.k)->\($0.m)" }
            .joined(separator: ", ") + "]"
    }.joined(separator: "  "))
    var (cd, ce) = coldMB()
    print(String(format: "  not in page cache before the views MB: dense %.1f experts %.1f", cd, ce))

    // Views: one per dense tensor, three per selected expert (the runner's expert views). Rebuilt in a cold run: views the
    // GPU has used keep their pages, which `MS_INVALIDATE` then leaves in the cache.
    var denseView: [String: (MTLBuffer, Int)] = [:]
    var expertViews: [MTLBuffer] = []
    var blobs = [UInt64](repeating: 0, count: 3 * lExperts)
    var partOff = [UInt32](repeating: 0, count: 3 * lExperts)
    func makeViews() {
        for it in items { denseView[it.name] = file.noCopyBuffer(device: device, tensor: it.tensor)! }
        for (s, ex) in slots.enumerated() {
            for (k, (f, t, per)) in [(file, gateT, gateRow * lF), (file, upT, gateRow * lF), (side, downT, downRow * lD)].enumerated() {
                let (v, o) = f.noCopyBuffer(device: device, offset: t.offset + ex * per, byteCount: per)!
                blobs[k * lExperts + s] = v.gpuAddress
                partOff[3 * s + k] = UInt32(o)
                expertViews.append(v)
            }
        }
    }
    makeViews()
    (cd, ce) = coldMB()
    print(String(format: "  not in page cache after the views MB: dense %.1f experts %.1f", cd, ce))

    func shared<E>(_ a: [E]) -> MTLBuffer {
        a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)! }
    }
    func buf(_ bytes: Int) -> MTLBuffer { device.makeBuffer(length: max(bytes, 4), options: .storageModeShared)! }
    var rng = SystemRandomNumberGenerator()
    func randomFloats(_ n: Int, _ r: ClosedRange<Float>) -> MTLBuffer {
        let b = buf(n * 4)
        let p = b.contents().bindMemory(to: Float.self, capacity: n)
        for i in 0..<n { p[i] = Float.random(in: r, using: &rng) }
        return b
    }
    let blobBuf = shared(blobs), partOffBuf = shared(partOff), pairSlotBuf = shared(pairSlot)
    func rebuildViews() {
        denseView = [:]; expertViews = []
        makeViews()
        blobBuf.contents().copyMemory(from: blobs, byteCount: blobs.count * 8)
        partOffBuf.contents().copyMemory(from: partOff, byteCount: partOff.count * 4)
    }
    let orderBuf = shared(order), atBuf = shared(at), wBuf = shared(weights)
    let listsBuf = shared(lists), countsBuf = shared(count.map { Int32($0) })
    let xR = randomFloats(T * lD, -1...1), residual = randomFloats(T * lD, -0.1...0.1)
    // Dense inputs: one random x per GEMM (float for A and the fp32 group, half made in the arm for C and D).
    var xDense: [String: MTLBuffer] = [:], xDenseH: [String: MTLBuffer] = [:]
    for it in items {
        xDense[it.name] = randomFloats(T * it.k, -1...1)
        if it.group != "fp32" { xDenseH[it.name] = buf(T * it.k * 2) }
    }

    // Pipelines.
    let lib = try MetalContext.moduleLibrary(device: device, module: "moe_ggml")
    func pso(_ l: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
        try device.makeComputePipelineState(function: l.makeFunction(name: name)!)
    }
    let gemv = try GGMLDenseGEMV(device: device)
    let ggmlSource = try moeGGMLSource()
    let naxSource = try String(contentsOfFile: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("../../bench/qfn/qfn_nax.metal").standardized.path, encoding: .utf8)
    var naxLib: MTLLibrary?
    if needD {
        if #available(macOS 26.0, *) {
            let opts = MTLCompileOptions()
            opts.languageVersion = .version4_0
            do { naxLib = try device.makeLibrary(source: ggmlSource + "\n" + naxSource, options: opts) } catch {
                print("  arm D: the tensor-op kernels did not compile here, skipped (\(String(describing: error).prefix(300)))")
            }
        } else {
            print("  arm D: needs macOS 26 (MSL 4.0), skipped")
        }
        if naxLib == nil { arms.removeAll { $0 == "D" || $0 == "M" } }
    }
    var hlib: MTLLibrary?
    if needC { hlib = try device.makeLibrary(source: ggmlSource + "\n" + q2HalfKernelSource(ggmlSource), options: nil) }
    let halfLib = try device.makeLibrary(source: halfOnlySource, options: nil)

    // Outputs.
    var yDense: [String: [String: MTLBuffer]] = [:]   // arm -> tensor -> y (C: half)
    for arm in arms {
        var m: [String: MTLBuffer] = [:]
        for it in items where arm == "A" || it.group != "fp32" { m[it.name] = buf(T * it.m * (arm == "C" || arm == "M" ? 2 : 4)) }
        yDense[arm] = m
    }
    var yRouted: [String: MTLBuffer] = [:]
    for arm in arms { yRouted[arm] = buf(T * lD * 4) }
    let fp32Y = buf(T * 512 * 4)   // C / D write the fp32 group here (A's outputs are compared, not these)

    // --- Arm encoders, one command buffer per group. ---
    func run(_ cb: MTLCommandBuffer, _ p: MTLComputePipelineState, grid: MTLSize, tg: MTLSize, threadgroups: Bool = false,
             shmem: Int = 0, _ set: (MTLComputeCommandEncoder) -> Void) {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        set(enc)
        if shmem > 0 { enc.setThreadgroupMemoryLength(shmem, index: 0) }
        if threadgroups { enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg) } else { enc.dispatchThreads(grid, threadsPerThreadgroup: tg) }
        enc.endEncoding()
    }
    func encodeFP32(_ cb: MTLCommandBuffer, arm: String) {
        for it in items where it.group == "fp32" {
            let (v, o) = denseView[it.name]!
            gemv.encode(commandBuffer: cb, type: it.tensor.type, weights: v, weightsOffset: o, x: xDense[it.name]!,
                        y: arm == "A" ? yDense["A"]![it.name]! : fp32Y, m: it.m, n: it.k, tokens: T)
        }
    }
    // A
    let psoPhase1 = try pso(lib, "moe_iq2xxs_phase1_gate_up_act"), psoPhase2 = try pso(lib, "moe_q2k_phase2_down_reduce")
    let acts = arms.contains("A") ? buf(P * lActStride * 4) : nil
    func encodeA(_ cb: MTLCommandBuffer, group: String) {
        if group == "routed" {
            var offsets = (UInt32(gateRow), UInt32(downRow))
            var dV = UInt32(lD), fV = UInt32(lF), kV = UInt32(lTopK), sV = UInt32(lActStride), tV = UInt32(T)
            let y = yRouted["A"]!
            var enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoPhase1)
            enc.setBuffer(blobBuf, offset: 0, index: 0); enc.useResources(expertViews, usage: .read)
            enc.setBytes(&offsets, length: 8, index: 1); enc.setBuffer(xR, offset: 0, index: 2); enc.setBuffer(acts!, offset: 0, index: 3)
            enc.setBytes(&dV, length: 4, index: 4); enc.setBytes(&fV, length: 4, index: 5); enc.setBytes(&kV, length: 4, index: 6)
            enc.setBytes(&sV, length: 4, index: 7); enc.setBuffer(partOffBuf, offset: 0, index: 8); enc.setBytes(&tV, length: 4, index: 9)
            enc.setBuffer(pairSlotBuf, offset: 0, index: 10)
            enc.setThreadgroupMemoryLength(256 * 8 + 128, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: (P * lF + 7) / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.endEncoding()
            enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoPhase2)
            enc.setBuffer(blobBuf, offset: 0, index: 0); enc.useResources(expertViews, usage: .read)
            enc.setBytes(&offsets, length: 8, index: 1); enc.setBuffer(acts!, offset: 0, index: 2); enc.setBuffer(wBuf, offset: 0, index: 3)
            enc.setBuffer(residual, offset: 0, index: 4); enc.setBuffer(y, offset: 0, index: 5)
            enc.setBytes(&dV, length: 4, index: 6); enc.setBytes(&sV, length: 4, index: 7); enc.setBytes(&kV, length: 4, index: 8)
            enc.setBuffer(partOffBuf, offset: 0, index: 9); enc.setBytes(&tV, length: 4, index: 10); enc.setBuffer(pairSlotBuf, offset: 0, index: 11)
            enc.dispatchThreadgroups(MTLSize(width: (T * lD + 7) / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.endEncoding()
            return
        }
        if group == "fp32" { encodeFP32(cb, arm: "A"); return }
        for it in items where it.group == group {
            let (v, o) = denseView[it.name]!
            gemv.encode(commandBuffer: cb, type: it.tensor.type, weights: v, weightsOffset: o, x: xDense[it.name]!,
                        y: yDense["A"]![it.name]!, m: it.m, n: it.k, tokens: T)
        }
    }
    // C and D share the float -> half pass of each GEMM's input.
    let psoToHalf = try pso(halfLib, "qfn_f32_to_f16")
    func toHalf(_ cb: MTLCommandBuffer, _ src: MTLBuffer, _ dst: MTLBuffer, _ n: Int) {
        var n4 = UInt32(n / 4)
        run(cb, psoToHalf, grid: MTLSize(width: n / 4, height: 1, depth: 1), tg: MTLSize(width: 256, height: 1, depth: 1)) { enc in
            enc.setBuffer(src, offset: 0, index: 0); enc.setBuffer(dst, offset: 0, index: 1); enc.setBytes(&n4, length: 4, index: 2)
        }
    }
    // C
    let G = 4
    var cState: (deqGU: MTLComputePipelineState, deqDown: MTLComputePipelineState, gather: MTLComputePipelineState,
                 silu: MTLComputePipelineState, scatter: MTLComputePipelineState, q8h: MTLComputePipelineState,
                 xgh: MTLBuffer, guh: MTLBuffer, dnh: MTLBuffer, wGUh: MTLBuffer, wDownh: MTLBuffer, wDense: MTLBuffer)?
    var mulsH: [Int: (MPSMatrixMultiplication, MPSMatrixMultiplication)] = [:]
    var denseMul: [String: MPSMatrixMultiplication] = [:]
    if let hlib {
        let maxW = items.filter { $0.group != "fp32" && $0.tensor.type == .q8_0 }.map { $0.m * $0.k }.max() ?? 1
        cState = (try pso(hlib, "moe_iq2xxs_dequant_gate_up_f16"), try pso(hlib, "moe_q2k_dequant_down_f16"),
                  try pso(hlib, "qfn_gather_pair_rows_h"), try pso(hlib, "qfn_silu_mul_halves_h"), try pso(hlib, "qfn_scatter_weighted_h"),
                  try pso(halfLib, "qfn_q8_0_dequant_f16"),
                  buf(P * lD * 2), buf(P * 2 * lF * 2), buf(P * lD * 2), buf(G * 2 * lF * lD * 2), buf(G * lD * lF * 2), buf(maxW * 2))
        for n in Set(count) {
            mulsH[n] = (MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                                resultRows: n, resultColumns: 2 * lF, interiorColumns: lD, alpha: 1, beta: 0),
                        MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                                resultRows: n, resultColumns: lD, interiorColumns: lF, alpha: 1, beta: 0))
        }
        for it in items where it.group != "fp32" {
            denseMul[it.name] = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                                        resultRows: T, resultColumns: it.m, interiorColumns: it.k, alpha: 1, beta: 0)
        }
    }
    func hmatrix(_ b: MTLBuffer, _ offset: Int, _ rows: Int, _ cols: Int, rowBytes: Int? = nil) -> MPSMatrix {
        MPSMatrix(buffer: b, offset: offset,
                  descriptor: MPSMatrixDescriptor(rows: rows, columns: cols, rowBytes: rowBytes ?? cols * 2, dataType: .float16))
    }
    func encodeC(_ cb: MTLCommandBuffer, group: String, out: String) {
        let c = cState!
        if group == "fp32" { encodeFP32(cb, arm: "C"); return }
        if group == "routed" {
            run(cb, c.gather, grid: MTLSize(width: (lD + 31) / 32, height: P, depth: 1), tg: MTLSize(width: 16, height: 64, depth: 1)) { enc in
                enc.setBuffer(xR, offset: 0, index: 0); enc.setBuffer(orderBuf, offset: 0, index: 1); enc.setBuffer(c.xgh, offset: 0, index: 2)
                var wV = UInt32(lD), kV = UInt32(lTopK)
                enc.setBytes(&wV, length: 4, index: 3); enc.setBytes(&kV, length: 4, index: 4)
            }
            for g0 in stride(from: 0, to: S, by: G) {
                let n = min(G, S - g0)
                var firstSlot = UInt32(g0), dV = UInt32(lD), fV = UInt32(lF)
                run(cb, c.deqGU, grid: MTLSize(width: lD / 32, height: 2 * lF, depth: n), tg: MTLSize(width: 8, height: 32, depth: 1)) { enc in
                    enc.setBuffer(blobBuf, offset: 0, index: 0); enc.useResources(expertViews, usage: .read)
                    enc.setBuffer(partOffBuf, offset: 0, index: 1); enc.setBuffer(c.wGUh, offset: 0, index: 2)
                    enc.setBytes(&dV, length: 4, index: 3); enc.setBytes(&fV, length: 4, index: 4); enc.setBytes(&firstSlot, length: 4, index: 5)
                }
                var sV = UInt32(lActStride), cV = UInt32(lF), rbV = UInt32(downRow)
                run(cb, c.deqDown, grid: MTLSize(width: lF / 16, height: lD, depth: n), tg: MTLSize(width: 4, height: 64, depth: 1)) { enc in
                    enc.setBuffer(blobBuf, offset: 0, index: 0); enc.useResources(expertViews, usage: .read)
                    enc.setBuffer(partOffBuf, offset: 0, index: 1); enc.setBuffer(c.wDownh, offset: 0, index: 2)
                    enc.setBytes(&dV, length: 4, index: 3); enc.setBytes(&sV, length: 4, index: 4); enc.setBytes(&cV, length: 4, index: 5)
                    enc.setBytes(&firstSlot, length: 4, index: 6); enc.setBytes(&rbV, length: 4, index: 7)
                }
                for s in g0..<(g0 + n) {
                    mulsH[count[s]]!.0.encode(commandBuffer: cb, leftMatrix: hmatrix(c.xgh, start[s] * lD * 2, count[s], lD),
                                              rightMatrix: hmatrix(c.wGUh, (s - g0) * 2 * lF * lD * 2, 2 * lF, lD),
                                              resultMatrix: hmatrix(c.guh, start[s] * 2 * lF * 2, count[s], 2 * lF))
                }
                var firstV = UInt32(start[g0])
                run(cb, c.silu, grid: MTLSize(width: lF, height: start[g0 + n] - start[g0], depth: 1), tg: MTLSize(width: 64, height: 16, depth: 1)) { enc in
                    enc.setBuffer(c.guh, offset: 0, index: 0); enc.setBytes(&fV, length: 4, index: 1); enc.setBytes(&firstV, length: 4, index: 2)
                }
                for s in g0..<(g0 + n) {
                    mulsH[count[s]]!.1.encode(commandBuffer: cb, leftMatrix: hmatrix(c.guh, start[s] * 2 * lF * 2, count[s], lF, rowBytes: 2 * lF * 2),
                                              rightMatrix: hmatrix(c.wDownh, (s - g0) * lD * lF * 2, lD, lF),
                                              resultMatrix: hmatrix(c.dnh, start[s] * lD * 2, count[s], lD))
                }
            }
            run(cb, c.scatter, grid: MTLSize(width: (lD + 31) / 32, height: T, depth: 1), tg: MTLSize(width: 16, height: 64, depth: 1)) { enc in
                enc.setBuffer(c.dnh, offset: 0, index: 0); enc.setBuffer(atBuf, offset: 0, index: 1); enc.setBuffer(wBuf, offset: 0, index: 2)
                enc.setBuffer(residual, offset: 0, index: 3); enc.setBuffer(yRouted[out]!, offset: 0, index: 4)
                var dV = UInt32(lD), kV = UInt32(lTopK)
                enc.setBytes(&dV, length: 4, index: 5); enc.setBytes(&kV, length: 4, index: 6)
            }
            return
        }
        for it in items where it.group == group {
            let (v, o) = denseView[it.name]!
            toHalf(cb, xDense[it.name]!, xDenseH[it.name]!, T * it.k)
            var right = hmatrix(v, o, it.m, it.k)
            if it.tensor.type == .q8_0 {
                var kV = UInt32(it.k)
                run(cb, c.q8h, grid: MTLSize(width: it.k / 32, height: it.m, depth: 1), tg: MTLSize(width: min(32, it.k / 32), height: 8, depth: 1)) { enc in
                    enc.setBuffer(v, offset: o, index: 0); enc.setBuffer(c.wDense, offset: 0, index: 1); enc.setBytes(&kV, length: 4, index: 2)
                }
                right = hmatrix(c.wDense, 0, it.m, it.k)
            }
            denseMul[it.name]!.encode(commandBuffer: cb, leftMatrix: hmatrix(xDenseH[it.name]!, 0, T, it.k), rightMatrix: right,
                                      resultMatrix: hmatrix(yDense[out]![it.name]!, 0, T, it.m))
        }
    }
    // D
    var dState: (dense: MTLComputePipelineState, mid: MTLComputePipelineState, down: MTLComputePipelineState,
                 reduce: MTLComputePipelineState, xh: MTLBuffer, midh: MTLBuffer, part: MTLBuffer)?
    if let naxLib {
        dState = (try pso(naxLib, "qfn_dense_nax"), try pso(naxLib, "qfn_moe_mid_nax"), try pso(naxLib, "qfn_moe_down_nax"),
                  try pso(naxLib, "qfn_moe_reduce"), buf(T * lD * 2), buf(P * lF * 2), buf(P * lD * 4))
    }
    func encodeD(_ cb: MTLCommandBuffer, group: String, out: String) {
        let d = dState!
        if group == "fp32" { encodeFP32(cb, arm: "D"); return }
        if group == "routed" {
            toHalf(cb, xR, d.xh, T * lD)
            for (p, (inDim, outRows, rowBytes, type, input, output)) in [(lD, lF, gateRow, 16, d.xh, d.midh),
                                                                        (lF, lD, downRow, 10, d.midh, d.part)].enumerated() {
                var args = (UInt32(lTopK), UInt32(inDim), UInt32(outRows), UInt32(rowBytes), UInt32(listCap), UInt32(S),
                            UInt32(tiles), UInt32(type), UInt32(2 * p))
                run(cb, p == 0 ? d.mid : d.down, grid: MTLSize(width: (outRows + 63) / 64 * tiles, height: S, depth: 1),
                    tg: MTLSize(width: 128, height: 1, depth: 1), threadgroups: true, shmem: 16384) { enc in
                    enc.setBytes(&args, length: MemoryLayout.size(ofValue: args), index: 0)
                    enc.setBuffer(blobBuf, offset: 0, index: 1); enc.useResources(expertViews, usage: .read)
                    enc.setBuffer(partOffBuf, offset: 0, index: 2); enc.setBuffer(listsBuf, offset: 0, index: 3)
                    enc.setBuffer(countsBuf, offset: 0, index: 4); enc.setBuffer(input, offset: 0, index: 5)
                    enc.setBuffer(output, offset: 0, index: 6)
                }
            }
            var d4 = UInt32(lD / 4), kV = UInt32(lTopK)
            run(cb, d.reduce, grid: MTLSize(width: lD / 4, height: T, depth: 1), tg: MTLSize(width: 64, height: 4, depth: 1)) { enc in
                enc.setBuffer(d.part, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
                enc.setBuffer(residual, offset: 0, index: 2); enc.setBuffer(yRouted[out]!, offset: 0, index: 3)
                enc.setBytes(&d4, length: 4, index: 4); enc.setBytes(&kV, length: 4, index: 5)
            }
            return
        }
        for it in items where it.group == group {
            let (v, o) = denseView[it.name]!
            toHalf(cb, xDense[it.name]!, xDenseH[it.name]!, T * it.k)
            var args = (UInt32(T), UInt32(it.k), UInt32(it.m), UInt32(it.tensor.bytesPerRow), UInt32(it.tensor.type == .q8_0 ? 8 : 1))
            run(cb, d.dense, grid: MTLSize(width: (it.m + 63) / 64, height: (T + 63) / 64, depth: 1),
                tg: MTLSize(width: 128, height: 1, depth: 1), threadgroups: true, shmem: 16384) { enc in
                enc.setBytes(&args, length: MemoryLayout.size(ofValue: args), index: 0)
                enc.setBuffer(v, offset: o, index: 1); enc.setBuffer(xDenseH[it.name]!, offset: 0, index: 2)
                enc.setBuffer(yDense[out]![it.name]!, offset: 0, index: 3)
            }
        }
    }

    let groups = ["proj", "hc", "shexp", "fp32", "routed"]
    /// One round of an arm: GPU ms per group and the host wall of the whole arm (encode + GPU + waits).
    func round(_ arm: String) throws -> (gpu: [String: Double], wall: Double) {
        var gpu: [String: Double] = [:]
        let t0 = CFAbsoluteTimeGetCurrent()
        for g in groups {
            let cb = queue.makeCommandBuffer()!
            switch arm {
            case "A": encodeA(cb, group: g)
            case "C": encodeC(cb, group: g, out: arm)
            case "M": if g == "routed" { encodeD(cb, group: g, out: arm) } else { encodeC(cb, group: g, out: arm) }
            default: encodeD(cb, group: g, out: arm)
            }
            cb.commit(); cb.waitUntilCompleted()
            if let e = cb.error { throw e }
            gpu[g] = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        }
        return (gpu, (CFAbsoluteTimeGetCurrent() - t0) * 1000)
    }
    func line(_ arm: String, _ gpu: [String: [Double]], _ wall: [Double], _ read: [Double]) -> String {
        let med = { (v: [Double]) in v.sorted()[v.count / 2] }
        let totals = (0..<wall.count).map { i in groups.reduce(0.0) { $0 + gpu[$1]![i] } }
        var s = "  arm \(arm) GPU ms: " + groups.map { String(format: "%@ %.1f", $0, med(gpu[$0]!)) }.joined(separator: "  ")
        s += String(format: " | total %.1f (%.1f-%.1f), arm wall %.1f ms", med(totals), totals.min()!, totals.max()!, med(wall))
        if !read.isEmpty {
            let layer = (0..<wall.count).map { read[$0] * 1000 + wall[$0] }
            s += String(format: " | read %.1f ms, layer wall (read + arm) %.1f (%.1f-%.1f) ms, n %d",
                        med(read) * 1000, med(layer), layer.min()!, layer.max()!, wall.count)
        }
        return s
    }

    if cold {
        // A warm-up round first (pipelines, MPS kernel choice, the first GPU mapping of the views), then the drop.
        _ = readAll()
        try autoreleasepool { _ = try round(arms[0]) }
        denseView = [:]; expertViews = []
        // Released views unmap a little later: repeat until the dropped bytes stop growing.
        var drops: [String] = []
        var last = -1.0
        for attempt in 0..<6 {
            if attempt > 0 {
                let cb = queue.makeCommandBuffer()!
                cb.commit(); cb.waitUntilCompleted()
                usleep(200_000)
            }
            invalidate(ggufPath, denseRanges + guRanges)
            invalidate(sidecarPath, downRanges)
            (cd, ce) = coldMB()
            drops.append(String(format: "%.1f/%.1f", cd, ce))
            if cd + ce <= last + 0.1 { break }
            last = cd + ce
        }
        print("  dropped after a warm-up round, not in page cache MB (dense/experts) per attempt: " + drops.joined(separator: ", "))
        rebuildViews()
        let r = readAll()
        (cd, ce) = coldMB()
        let (gpu, wall) = try round(arms[0])
        print(String(format: "  cold read %.3f s (dense + experts, 4 threads), after it not in cache MB: dense %.1f experts %.1f", r, cd, ce))
        print(line(arms[0], gpu.mapValues { [$0] }, [wall], [r]))
        return
    }

    let r0 = readAll()
    (cd, ce) = coldMB()
    print(String(format: "  first read %.3f s, then not in cache MB: dense %.1f experts %.1f", r0, cd, ce))
    var gpu: [String: [String: [Double]]] = [:], wall: [String: [Double]] = [:], reads: [Double] = []
    for it in 0...iterations {
        let r = readAll()
        for arm in arms {
            let (g, w) = try round(arm)
            guard it > 0 else { continue }
            for (k, v) in g { gpu[arm, default: [:]][k, default: []].append(v) }
            wall[arm, default: []].append(w)
        }
        if it > 0 { reads.append(r) }
    }
    for arm in arms { print(line(arm, gpu[arm]!, wall[arm]!, reads)) }
    (cd, ce) = coldMB()
    print(String(format: "  after the rounds not in cache MB: dense %.1f experts %.1f", cd, ce))

    // Precision against A (P1: max |x - A| <= 2e-3 x max |A|; routed against max |A - residual|).
    guard arms.contains("A") else { return }
    for arm in arms where arm != "A" {
        var worst = 0.0
        var parts: [String] = []
        for it in items where it.group != "fp32" {
            let n = T * it.m
            let a = yDense["A"]![it.name]!.contents().bindMemory(to: Float.self, capacity: n)
            var d = 0.0, m = 0.0
            if arm == "C" || arm == "M" {
                let c = yDense[arm]![it.name]!.contents().bindMemory(to: Float16.self, capacity: n)
                for i in 0..<n { d = max(d, Double(abs(a[i] - Float(c[i])))); m = max(m, Double(abs(a[i]))) }
            } else {
                let c = yDense[arm]![it.name]!.contents().bindMemory(to: Float.self, capacity: n)
                for i in 0..<n { d = max(d, Double(abs(a[i] - c[i]))); m = max(m, Double(abs(a[i]))) }
            }
            worst = max(worst, d / m)
            parts.append(String(format: "%@ %.1e", it.name.replacingOccurrences(of: ".weight", with: ""), d / m))
        }
        let n = T * lD
        let a = yRouted["A"]!.contents().bindMemory(to: Float.self, capacity: n)
        let c = yRouted[arm]!.contents().bindMemory(to: Float.self, capacity: n)
        let res = residual.contents().bindMemory(to: Float.self, capacity: n)
        var d = 0.0, m = 0.0
        for i in 0..<n { d = max(d, Double(abs(a[i] - c[i]))); m = max(m, Double(abs(a[i] - res[i]))) }
        worst = max(worst, d / m)
        parts.append(String(format: "routed %.1e", d / m))
        print(String(format: "  %@ vs A, max |d| / max |A|: ", arm) + parts.joined(separator: ", ")
              + String(format: " -> worst %.2e (limit 2e-3): %@", worst, worst <= 2e-3 ? "pass" : "FAIL"))
    }
}

/// Arms C and D: each GEMM's float input -> half; arm C: Q8_0 rows -> half for MPS.
private let halfOnlySource = """
#include <metal_stdlib>
using namespace metal;
kernel void qfn_f32_to_f16(device const float4 *src [[buffer(0)]], device half4 *dst [[buffer(1)]],
                           constant uint &n4 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n4) dst[gid] = half4(src[gid]);
}
kernel void qfn_q8_0_dequant_f16(device const char *w [[buffer(0)]], device half *out [[buffer(1)]],
                                 constant uint &K [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
    device const char *b = w + ((ulong)pos.y * (K / 32) + pos.x) * 34;
    const float d = (float)*(device const half *)b;
    device half *o = out + (ulong)pos.y * K + pos.x * 32;
    for (uint i = 0; i < 32; i++) o[i] = (half)(d * (float)b[2 + i]);
}
"""
