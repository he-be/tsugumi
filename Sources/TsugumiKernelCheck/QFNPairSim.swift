import Darwin
import Foundation
@preconcurrency import Metal
import MetalPerformanceShaders
import Tsugumi

// MARK: - `--qfn-pair-sim serve|run <ip> <port> <gguf> <down sidecar> <routes prefix> A|B <k|sigma> <resident GB|chunks>`
//
// docs/qfn-prefill/12 F2 (A) and F3 (B): the GEMMs of one chunk (T = 2048) over all 48 layers, split between the MBP
// and the M6 on raw TCP over bridge0, with the real route dump of every layer (`<prefix>-l<L>.bin`).
//   M6:  --qfn-pair-sim serve <bind ip> <port> <gguf> <sidecar> <routes> A|B <k|sigma> <resident GB>
//   MBP: --qfn-pair-sim run <M6 ip> <port> <gguf> <sidecar> <routes> A|B <k|sigma> <chunks>
// A layer's work is --qfn-layer-bench's groups: the dense projections, hyper-connection, shared expert, the small
// float projections, the routed experts. The attention / GDN core, norms and element-wise work are not here.
//   MBP, today's code: dense through GGMLDenseGEMV (float32 dequant + sgemm at this T); routed as the runner's
//        `routedGemm` at T >= 1024 (experts with >= 24 pairs: float32 dequant of 4 at a time + MPS sgemm, the rest the
//        per-pair kernels). Weights are read with `preadRanges` (4 threads) into the page cache and used through
//        no-copy views, as the runner.
//   M6, P1 (arm M of F1, docs/qfn-prefill/15): dense through half dequant + MPS fp16, routed through the tensor-op
//        tiles (bench/qfn/qfn_nax.metal). Weights in the process's own buffers (docs/qfn-prefill/17): the M6's share is
//        filled in layer order into buffers kept GPU-resident by one MTLResidencySet while it fits in <resident GB>,
//        holding every expert of the share's range (another prompt selects other experts); the rest is read on every
//        use with F_NOCACHE (4 threads) into one staging buffer, only the selected experts, then dropped from the page
//        cache. <resident GB> 0 streams everything (cold).
//   A <k>: the M6 runs layers 0..<k whole and sends the hyper-connection residual (4 x T x 2560 fp32, 84 MB); the MBP
//        runs layers k..<48 after it. The M6 goes on to chunk c + 1 while the MBP runs chunk c (the layers of a chunk
//        need only the same layers of earlier chunks).
//   B <sigma>: per layer the MBP reads and runs the dense groups, sends `mixed` (T x 2560 fp32) and the top-10
//        assignment, and runs its experts [0, cut) while the M6 runs [cut, 512) and returns the partial sum
//        (T x 2560 fp32); cut = round(512 (1 - sigma)). The M6 reads a streamed share after receiving (the route is
//        known only then).
// Inputs are random. Each side prints a line per layer; the MBP prints its wall per chunk and in total.
// k 0 / sigma 0 runs the MBP alone (no socket). QFN_SIM_CHECK=1 checks the split and the buffers on one machine, then exits.
// QFN_FEED=<f> (A only, set on both sides; docs/qfn-prefill/19 §2, Q4-b): the last fraction f of the bytes of every streamed
//   M6 layer comes from the MBP's internal SSD over a second connection instead of the M6's SSD. The M6 sends the piece
//   list (file, offset, bytes) when it starts the layer's read, reads its own part meanwhile and receives the MBP's part
//   straight into the staging buffer; the MBP preads (F_NOCACHE, 4 threads, 16 MB blocks) and sends the blocks in order.

private let sD = 2560, sF = 640, sStride = 768, sExperts = 512, sTopK = 10, sLayers = 48
private let sMinPairs = 24, sGroup = 4, sHC = 4

private typealias Piece = (file: GGUFFile, offset: Int, byteCount: Int)

private struct SimItem {
    let name: String, group: String
    let tensor: GGUFFile.Tensor
    var m: Int { tensor.rowCount }
    var k: Int { tensor.rowWidth }
}

private func simVM() -> (pageins: UInt64, swapouts: UInt64, free: UInt64, wired: UInt64) {
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
    _ = withUnsafeMutablePointer(to: &stats) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
    }
    return (stats.pageins, stats.swapouts, UInt64(stats.free_count), UInt64(stats.wire_count))
}

/// `pread` with F_NOCACHE of `pieces` into `dst` at `at[i]`, 16 MB blocks on `threads` threads.
private func simPreadNoCache(_ pieces: [Piece], at: [Int], fds: [ObjectIdentifier: Int32], into dst: UnsafeMutableRawPointer, threads: Int) {
    var blocks: [(Int32, Int, Int, Int)] = []
    for (i, r) in pieces.enumerated() {
        var o = 0
        while o < r.byteCount {
            let n = min(16 << 20, r.byteCount - o)
            blocks.append((fds[ObjectIdentifier(r.file)]!, r.offset + o, n, at[i] + o))
            o += n
        }
    }
    let next = UnsafeMutablePointer<Int>.allocate(capacity: 1)
    next.initialize(to: 0)
    defer { next.deallocate() }
    let lock = NSLock()
    DispatchQueue.concurrentPerform(iterations: threads) { _ in
        while true {
            lock.lock(); let i = next.pointee; next.pointee += 1; lock.unlock()
            guard i < blocks.count else { return }
            let (fd, off, n, d) = blocks[i]
            var done = 0
            while done < n {
                let got = pread(fd, dst + d + done, n - done, off_t(off + done))
                precondition(got > 0, "pread")
                done += got
            }
        }
    }
}

/// Drops `pieces` from the page cache through the GGUFFile's own mapping (the M6 side makes no views of it).
private func simDrop(_ pieces: [Piece]) {
    let page = Int(getpagesize())
    for r in pieces {
        let a = r.offset / page * page, n = r.byteCount + (r.offset - a)
        precondition(msync(UnsafeMutableRawPointer(mutating: r.file.base + a), n, MS_INVALIDATE) == 0, "msync")
    }
}

/// Buffers shared by every layer of one side (one layer runs at a time).
private final class SimScratch {
    let x, zero, y: MTLBuffer                  // routed input (T x D fp32), residual 0, output
    let xDense, yDense: MTLBuffer              // fp32 dense input / output (T x max k / max m)
    var xDenseH, yDenseH, wDense: MTLBuffer?   // M6: half input / output, Q8_0 rows as half
    var rows, gu, wGU, wDown, acts, ones: MTLBuffer?   // MBP: the runner's routedGemm buffers
    var xh, midh, part: MTLBuffer?             // M6: the tensor-op tiles
    init(device: MTLDevice, T: Int, maxK: Int, maxM: Int, maxQ8: Int, nax: Bool) {
        let P = T * sTopK
        func buf(_ n: Int) -> MTLBuffer { device.makeBuffer(length: max(n, 16), options: .storageModeShared)! }
        func random(_ n: Int, _ r: ClosedRange<Float>) -> MTLBuffer {
            let b = buf(n * 4)
            let p = b.contents().bindMemory(to: Float.self, capacity: n)
            for i in 0..<n { p[i] = Float.random(in: r) }
            return b
        }
        x = random(T * sD, -1...1); zero = buf(T * sD * 4); y = buf(T * sD * 4)
        xDense = random(T * maxK, -1...1); yDense = buf(T * maxM * 4)
        if nax {
            xDenseH = buf(T * maxK * 2); yDenseH = buf(T * maxM * 2); wDense = buf(maxQ8 * 2)
            xh = buf(T * sD * 2); midh = buf(P * sF * 2); part = buf(P * sD * 4)
        } else {
            rows = buf(P * sD * 4); gu = buf(P * 2 * sF * 4); wGU = buf(sGroup * 2 * sF * sD * 4); wDown = buf(sGroup * sD * sF * 4)
            acts = buf(P * sStride * 4)
            let o = buf(P * 4)
            let p = o.contents().bindMemory(to: Float.self, capacity: P)
            for i in 0..<P { p[i] = 1 }
            ones = o
        }
    }
}

/// Pipelines and MPS objects of one side.
private final class SimKernels {
    let device: MTLDevice, queue: MTLCommandQueue
    let gemv: GGMLDenseGEMV
    // MBP (runtime)
    var gather, deqGU, deqDown, silu, scatter, phase1, phase2: MTLComputePipelineState?
    // M6 (nax + MPS fp16)
    var toHalf, q8h, mid, down, reduce: MTLComputePipelineState?
    private var muls: [[Int]: MPSMatrixMultiplication] = [:]
    init(device: MTLDevice, queue: MTLCommandQueue, nax: Bool) throws {
        self.device = device; self.queue = queue
        gemv = try GGMLDenseGEMV(device: device)
        func pso(_ l: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
            try device.makeComputePipelineState(function: l.makeFunction(name: name)!)
        }
        if nax {
            let ggml = try moeGGMLSource()
            let naxSource = try String(contentsOfFile: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .appendingPathComponent("../../bench/qfn/qfn_nax.metal").standardized.path, encoding: .utf8)
            guard #available(macOS 26.0, *) else { fatalError("the M6 side needs macOS 26 (MSL 4.0 tensor ops)") }
            let opts = MTLCompileOptions()
            opts.languageVersion = .version4_0
            let naxLib = try device.makeLibrary(source: ggml + "\n" + naxSource, options: opts)
            mid = try pso(naxLib, "qfn_moe_mid_nax"); down = try pso(naxLib, "qfn_moe_down_nax"); reduce = try pso(naxLib, "qfn_moe_reduce")
            let halfLib = try device.makeLibrary(source: simHalfSource, options: nil)
            toHalf = try pso(halfLib, "qfn_sim_f32_to_f16"); q8h = try pso(halfLib, "qfn_sim_q8_0_dequant_f16")
        } else {
            let lib = try MetalContext.moduleLibrary(device: device, module: "moe_ggml")
            gather = try pso(lib, "moe_gather_pair_rows"); deqGU = try pso(lib, "moe_iq2xxs_dequant_gate_up_f32")
            deqDown = try pso(lib, "moe_q2k_dequant_down_f32"); silu = try pso(lib, "moe_silu_mul_halves")
            scatter = try pso(lib, "moe_scatter_weighted")
            phase1 = try pso(lib, "moe_iq2xxs_phase1_gate_up_act"); phase2 = try pso(lib, "moe_q2k_phase2_down_reduce")
        }
    }
    func mul(_ rows: Int, _ cols: Int, _ interior: Int) -> MPSMatrixMultiplication {
        if let m = muls[[rows, cols, interior]] { return m }
        let m = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                        resultRows: rows, resultColumns: cols, interiorColumns: interior, alpha: 1, beta: 0)
        muls[[rows, cols, interior]] = m
        return m
    }
}

private func simRun(_ cb: MTLCommandBuffer, _ p: MTLComputePipelineState, grid: MTLSize, tg: MTLSize, threadgroups: Bool = false,
                    shmem: Int = 0, _ set: (MTLComputeCommandEncoder) -> Void) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p)
    set(enc)
    if shmem > 0 { enc.setThreadgroupMemoryLength(shmem, index: 0) }
    if threadgroups { enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg) } else { enc.dispatchThreads(grid, threadsPerThreadgroup: tg) }
    enc.endEncoding()
}

/// One layer of one side: optionally its dense groups, and its routed experts in [lo, hi).
private final class SimLayer {
    let il: Int, T: Int, lo: Int, hi: Int, dense: Bool
    let items: [SimItem]
    let S: Int, SB: Int, PS: Int, listCap: Int, tiles: Int
    let slots: [Int], count: [Int], start: [Int]
    private let orderBuf, atBuf, wBuf, posSlotBuf, listsBuf, countsBuf: MTLBuffer
    private let gateRow: Int, downRow: Int
    /// What this layer reads when it streams: the dense tensors (all of them) and the selected experts of the share.
    let readPieces: [Piece]
    /// Everything the share could need: the dense tensors and every expert of [lo, hi) (the M6's resident layers).
    let fullPieces: [Piece]
    var readBytes: Int { readPieces.reduce(0) { $0 + $1.byteCount } }
    var fullBytes: Int { fullPieces.reduce(0) { $0 + $1.byteCount } }
    // Bound weights.
    private var denseW: [String: (MTLBuffer, Int)] = [:]
    private var blobBuf: MTLBuffer, partOffBuf: MTLBuffer
    private var resources: [MTLBuffer] = []
    private let device: MTLDevice
    private let gateT, upT, downT: GGUFFile.Tensor
    private let file: GGUFFile, side: GGUFFile

    init(file: GGUFFile, side: GGUFFile, layer il: Int, route: Data, lo: Int, hi: Int, dense: Bool, runtimeOrder: Bool,
         device: MTLDevice) throws {
        self.il = il; self.lo = lo; self.hi = hi; self.dense = dense; self.device = device; self.file = file; self.side = side
        let T0 = Int(route.withUnsafeBytes { $0.load(as: Int32.self) })
        T = min(T0, 2048)
        let P = T * sTopK, P0 = T0 * sTopK
        var experts = [Int32](repeating: 0, count: P), weights = [Float](repeating: 0, count: P)
        route.withUnsafeBytes { raw in
            experts.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[4..<(4 + 4 * P)])) }
            weights.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[(4 + 4 * P0)..<(4 + 4 * P0 + 4 * P)])) }
        }
        // Slots of this share's experts. The runner lays experts with fewer than 24 pairs out last (per-pair kernels).
        var seen = [Int](repeating: 0, count: sExperts)
        var firstSeen: [Int] = []
        for p in 0..<P {
            let ex = Int(experts[p])
            guard ex >= lo && ex < hi else { continue }
            if seen[ex] == 0 { firstSeen.append(ex) }
            seen[ex] += 1
        }
        let big = firstSeen.filter { !runtimeOrder || seen[$0] >= sMinPairs }
        let small = runtimeOrder ? firstSeen.filter { seen[$0] < sMinPairs } : []
        slots = big + small
        S = slots.count; SB = big.count
        var slotOf = [Int](repeating: -1, count: sExperts)
        for (s, ex) in slots.enumerated() { slotOf[ex] = s }
        count = slots.map { seen[$0] }
        var start = [Int](repeating: 0, count: S + 1)
        for s in 0..<S { start[s + 1] = start[s] + count[s] }
        self.start = start
        PS = start[S]
        var fill = start
        var order = [UInt32](repeating: 0, count: max(PS, 1)), at = [UInt32](repeating: 0, count: P), w = weights
        var posSlot = [UInt32](repeating: 0, count: max(PS, 1))
        for p in 0..<P {
            let s = slotOf[Int(experts[p])]
            guard s >= 0 else { w[p] = 0; continue }
            order[fill[s]] = UInt32(p); at[p] = UInt32(fill[s]); posSlot[fill[s]] = UInt32(s); fill[s] += 1
        }
        listCap = max(count.max() ?? 1, 1)
        var lists = [Int32](repeating: -1, count: max(S, 1) * listCap)
        for s in 0..<S { for i in 0..<count[s] { lists[s * listCap + i] = Int32(order[start[s] + i]) } }
        tiles = min(8, (listCap + 63) / 64)
        func shared<E>(_ a: [E]) -> MTLBuffer {
            a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 16), options: .storageModeShared)! }
        }
        orderBuf = shared(order); atBuf = shared(at); wBuf = shared(w); posSlotBuf = shared(posSlot)
        listsBuf = shared(lists); countsBuf = shared(count.map { Int32($0) })
        blobBuf = device.makeBuffer(length: 3 * sExperts * 8, options: .storageModeShared)!
        partOffBuf = device.makeBuffer(length: 3 * sExperts * 4, options: .storageModeShared)!

        // Tensors (--qfn-layer-bench's groups).
        let pre = "blk.\(il)."
        let smallNames: Set<String> = ["ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "ssm_alpha.weight", "ssm_beta.weight",
                                       "hc_attn_inject.weight", "hc_ffn_inject.weight"]
        var items: [SimItem] = []
        var denseTensors: [GGUFFile.Tensor] = []
        for t in file.tensors.values where t.name.hasPrefix(pre) && !t.name.contains("_exps") {
            denseTensors.append(t)
            let short = String(t.name.dropFirst(pre.count))
            if smallNames.contains(short) {
                items.append(SimItem(name: short, group: "fp32", tensor: t))
            } else if t.dims.count == 2 && (t.type == .q8_0 || t.type == .f16) && t.rowCount >= 64 {
                items.append(SimItem(name: short, group: short.hasPrefix("hc_") ? "hc" : short.contains("shexp") ? "shexp" : "proj", tensor: t))
            }
        }
        self.items = dense ? items.sorted { $0.name < $1.name } : []
        gateT = try file.tensor(pre + "ffn_gate_exps.weight"); upT = try file.tensor(pre + "ffn_up_exps.weight")
        downT = try side.tensor(pre + "ffn_down_exps.weight")
        gateRow = gateT.bytesPerRow; downRow = downT.byteCount / (sD * sExperts)
        precondition(downRow == 212, "the down sidecar's rows are 212 B")
        let densePieces: [Piece] = dense ? denseTensors.sorted { $0.offset < $1.offset }.map { (file, $0.offset, $0.byteCount) } : []
        var sel: [Piece] = []
        for (f, t, per) in [(file, gateT, gateRow * sF), (file, upT, gateRow * sF), (side, downT, downRow * sD)] {
            for ex in slots.sorted() {
                let o = t.offset + ex * per
                if let last = sel.last, last.file === f, last.offset + last.byteCount == o { sel[sel.count - 1].byteCount += per }
                else { sel.append((f, o, per)) }
            }
        }
        readPieces = densePieces + sel
        fullPieces = densePieces + (hi > lo ? [(file, gateT.offset + lo * gateRow * sF, (hi - lo) * gateRow * sF),
                                                 (file, upT.offset + lo * gateRow * sF, (hi - lo) * gateRow * sF),
                                                 (side, downT.offset + lo * downRow * sD, (hi - lo) * downRow * sD)] : [])
    }

    /// The runner's way: no-copy views of the mapped files.
    func bindViews() {
        denseW = [:]; resources = []
        for it in items { denseW[it.name] = file.noCopyBuffer(device: device, tensor: it.tensor)! }
        let blobs = blobBuf.contents().bindMemory(to: UInt64.self, capacity: 3 * sExperts)
        let off = partOffBuf.contents().bindMemory(to: UInt32.self, capacity: 3 * sExperts)
        for (s, ex) in slots.enumerated() {
            for (k, (f, t, per)) in [(file, gateT, gateRow * sF), (file, upT, gateRow * sF), (side, downT, downRow * sD)].enumerated() {
                let (v, o) = f.noCopyBuffer(device: device, offset: t.offset + ex * per, byteCount: per)!
                blobs[k * sExperts + s] = v.gpuAddress; off[3 * s + k] = UInt32(o)
                resources.append(v)
            }
        }
        resources += denseW.values.map { $0.0 }
    }

    /// Weights packed in `buffer`: `pieces[i]` at `at[i]` (the owned fill or the staging layout).
    func bindPacked(_ buffer: MTLBuffer, pieces: [Piece], at: [Int]) {
        func locate(_ f: GGUFFile, _ o: Int) -> Int {
            for (i, p) in pieces.enumerated() where p.file === f && o >= p.offset && o < p.offset + p.byteCount { return at[i] + o - p.offset }
            fatalError("layer \(il): byte \(o) is not in the packed layout")
        }
        denseW = [:]
        for it in items { denseW[it.name] = (buffer, locate(file, it.tensor.offset)) }
        let blobs = blobBuf.contents().bindMemory(to: UInt64.self, capacity: 3 * sExperts)
        let off = partOffBuf.contents().bindMemory(to: UInt32.self, capacity: 3 * sExperts)
        for (s, ex) in slots.enumerated() {
            for (k, (f, t, per)) in [(file, gateT, gateRow * sF), (file, upT, gateRow * sF), (side, downT, downRow * sD)].enumerated() {
                blobs[k * sExperts + s] = buffer.gpuAddress + UInt64(locate(f, t.offset + ex * per)); off[3 * s + k] = 0
            }
        }
        resources = [buffer]
    }

    /// The dense groups: GPU seconds.
    func encodeDense(_ k: SimKernels, _ sc: SimScratch, nax: Bool) -> Double {
        guard !items.isEmpty else { return 0 }
        let cb = k.queue.makeCommandBuffer()!
        autoreleasepool {
            for it in items {
                let (v, o) = denseW[it.name]!
                if !nax || it.group == "fp32" {
                    k.gemv.encode(commandBuffer: cb, type: it.tensor.type, weights: v, weightsOffset: o, x: sc.xDense, y: sc.yDense,
                                  m: it.m, n: it.k, tokens: T)
                    continue
                }
                var n4 = UInt32(T * it.k / 4)
                simRun(cb, k.toHalf!, grid: MTLSize(width: Int(n4), height: 1, depth: 1), tg: MTLSize(width: 256, height: 1, depth: 1)) { enc in
                    enc.setBuffer(sc.xDense, offset: 0, index: 0); enc.setBuffer(sc.xDenseH!, offset: 0, index: 1); enc.setBytes(&n4, length: 4, index: 2)
                }
                func hm(_ b: MTLBuffer, _ off: Int, _ r: Int, _ c: Int) -> MPSMatrix {
                    MPSMatrix(buffer: b, offset: off, descriptor: MPSMatrixDescriptor(rows: r, columns: c, rowBytes: c * 2, dataType: .float16))
                }
                var right = hm(v, o, it.m, it.k)
                if it.tensor.type == .q8_0 {
                    var kV = UInt32(it.k)
                    simRun(cb, k.q8h!, grid: MTLSize(width: it.k / 32, height: it.m, depth: 1), tg: MTLSize(width: min(32, it.k / 32), height: 8, depth: 1)) { enc in
                        enc.setBuffer(v, offset: o, index: 0); enc.setBuffer(sc.wDense!, offset: 0, index: 1); enc.setBytes(&kV, length: 4, index: 2)
                    }
                    right = hm(sc.wDense!, 0, it.m, it.k)
                }
                k.mul(T, it.m, it.k).encode(commandBuffer: cb, leftMatrix: hm(sc.xDenseH!, 0, T, it.k), rightMatrix: right,
                                            resultMatrix: hm(sc.yDenseH!, 0, T, it.m))
            }
        }
        cb.commit(); cb.waitUntilCompleted()
        precondition(cb.error == nil, "layer \(il) dense: \(String(describing: cb.error))")
        return cb.gpuEndTime - cb.gpuStartTime
    }

    /// The routed experts of the share: y = residual + the share's weighted sum. GPU seconds.
    func encodeRouted(_ k: SimKernels, _ sc: SimScratch, nax: Bool, x: MTLBuffer, residual: MTLBuffer, y: MTLBuffer) -> Double {
        let cb = k.queue.makeCommandBuffer()!
        autoreleasepool { nax ? encodeNax(cb, k, sc, x: x, residual: residual, y: y) : encodeRuntime(cb, k, sc, x: x, residual: residual, y: y) }
        cb.commit(); cb.waitUntilCompleted()
        precondition(cb.error == nil, "layer \(il) routed: \(String(describing: cb.error))")
        return cb.gpuEndTime - cb.gpuStartTime
    }

    private func encodeNax(_ cb: MTLCommandBuffer, _ k: SimKernels, _ sc: SimScratch, x: MTLBuffer, residual: MTLBuffer, y: MTLBuffer) {
        var n4 = UInt32(T * sD / 4)
        simRun(cb, k.toHalf!, grid: MTLSize(width: Int(n4), height: 1, depth: 1), tg: MTLSize(width: 256, height: 1, depth: 1)) { enc in
            enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(sc.xh!, offset: 0, index: 1); enc.setBytes(&n4, length: 4, index: 2)
        }
        if S > 0 {
            for (p, (inDim, outRows, rowBytes, type, input, output)) in [(sD, sF, gateRow, 16, sc.xh!, sc.midh!),
                                                                        (sF, sD, downRow, 10, sc.midh!, sc.part!)].enumerated() {
                var args = (UInt32(sTopK), UInt32(inDim), UInt32(outRows), UInt32(rowBytes), UInt32(listCap), UInt32(S),
                            UInt32(tiles), UInt32(type), UInt32(2 * p))
                simRun(cb, p == 0 ? k.mid! : k.down!, grid: MTLSize(width: (outRows + 63) / 64 * tiles, height: S, depth: 1),
                       tg: MTLSize(width: 128, height: 1, depth: 1), threadgroups: true, shmem: 16384) { enc in
                    enc.setBytes(&args, length: MemoryLayout.size(ofValue: args), index: 0)
                    enc.setBuffer(blobBuf, offset: 0, index: 1); enc.useResources(resources, usage: .read)
                    enc.setBuffer(partOffBuf, offset: 0, index: 2); enc.setBuffer(listsBuf, offset: 0, index: 3)
                    enc.setBuffer(countsBuf, offset: 0, index: 4); enc.setBuffer(input, offset: 0, index: 5)
                    enc.setBuffer(output, offset: 0, index: 6)
                }
            }
        }
        var d4 = UInt32(sD / 4), kV = UInt32(sTopK)
        simRun(cb, k.reduce!, grid: MTLSize(width: sD / 4, height: T, depth: 1), tg: MTLSize(width: 64, height: 4, depth: 1)) { enc in
            enc.setBuffer(sc.part!, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
            enc.setBuffer(residual, offset: 0, index: 2); enc.setBuffer(y, offset: 0, index: 3)
            enc.setBytes(&d4, length: 4, index: 4); enc.setBytes(&kV, length: 4, index: 5)
        }
    }

    /// Qwen38Runner.routedGemm over the share's pairs (the other pairs scatter row 0 with weight 0).
    private func encodeRuntime(_ cb: MTLCommandBuffer, _ k: SimKernels, _ sc: SimScratch, x: MTLBuffer, residual: MTLBuffer, y: MTLBuffer) {
        let rows = sc.rows!, gu = sc.gu!, wGU = sc.wGU!, wDown = sc.wDown!
        var dV = UInt32(sD), fV = UInt32(sF), kV = UInt32(sTopK), strideV = UInt32(sStride), downRowV = UInt32(downRow)
        func fm(_ b: MTLBuffer, _ offset: Int, _ r: Int, _ c: Int, rowBytes: Int? = nil) -> MPSMatrix {
            MPSMatrix(buffer: b, offset: offset, descriptor: MPSMatrixDescriptor(rows: r, columns: c, rowBytes: rowBytes ?? c * 4, dataType: .float32))
        }
        if PS > 0 {
            simRun(cb, k.gather!, grid: MTLSize(width: (sD + 31) / 32, height: PS, depth: 1), tg: MTLSize(width: 16, height: 64, depth: 1)) { enc in
                enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(orderBuf, offset: 0, index: 1); enc.setBuffer(rows, offset: 0, index: 2)
                enc.setBytes(&dV, length: 4, index: 3); enc.setBytes(&kV, length: 4, index: 4)
            }
        }
        for g0 in stride(from: 0, to: SB, by: sGroup) {
            let n = min(sGroup, SB - g0)
            var firstSlot = UInt32(g0)
            simRun(cb, k.deqGU!, grid: MTLSize(width: sD / 32, height: 2 * sF, depth: n), tg: MTLSize(width: 8, height: 32, depth: 1)) { enc in
                enc.setBuffer(blobBuf, offset: 0, index: 0); enc.useResources(resources, usage: .read)
                enc.setBuffer(partOffBuf, offset: 0, index: 1); enc.setBuffer(wGU, offset: 0, index: 2)
                enc.setBytes(&dV, length: 4, index: 3); enc.setBytes(&fV, length: 4, index: 4); enc.setBytes(&firstSlot, length: 4, index: 5)
            }
            simRun(cb, k.deqDown!, grid: MTLSize(width: sF / 16, height: sD, depth: n), tg: MTLSize(width: 4, height: 64, depth: 1)) { enc in
                enc.setBuffer(blobBuf, offset: 0, index: 0); enc.useResources(resources, usage: .read)
                enc.setBuffer(partOffBuf, offset: 0, index: 1); enc.setBuffer(wDown, offset: 0, index: 2)
                enc.setBytes(&dV, length: 4, index: 3); enc.setBytes(&strideV, length: 4, index: 4); enc.setBytes(&fV, length: 4, index: 5)
                enc.setBytes(&firstSlot, length: 4, index: 6); enc.setBytes(&downRowV, length: 4, index: 7)
            }
            for s in g0..<(g0 + n) {
                k.mul(count[s], 2 * sF, sD).encode(commandBuffer: cb, leftMatrix: fm(rows, start[s] * sD * 4, count[s], sD),
                                                   rightMatrix: fm(wGU, (s - g0) * 2 * sF * sD * 4, 2 * sF, sD),
                                                   resultMatrix: fm(gu, (start[s] - start[g0]) * 2 * sF * 4, count[s], 2 * sF))
            }
            var first = UInt32(0)
            simRun(cb, k.silu!, grid: MTLSize(width: sF, height: start[g0 + n] - start[g0], depth: 1), tg: MTLSize(width: 64, height: 16, depth: 1)) { enc in
                enc.setBuffer(gu, offset: 0, index: 0); enc.setBytes(&fV, length: 4, index: 1); enc.setBytes(&first, length: 4, index: 2)
            }
            for s in g0..<(g0 + n) {
                k.mul(count[s], sD, sF).encode(commandBuffer: cb, leftMatrix: fm(gu, (start[s] - start[g0]) * 2 * sF * 4, count[s], sF, rowBytes: 2 * sF * 4),
                                               rightMatrix: fm(wDown, (s - g0) * sD * sF * 4, sD, sF),
                                               resultMatrix: fm(rows, start[s] * sD * 4, count[s], sD))
            }
        }
        let p0 = start[SB], nSmall = PS - p0
        if nSmall > 0 {
            var offsets = (UInt32(gateRow), UInt32(downRow))
            var oneV = UInt32(1), nV = UInt32(nSmall)
            let acts = sc.acts!
            simRun(cb, k.phase1!, grid: MTLSize(width: (nSmall * sF + 7) / 8, height: 1, depth: 1), tg: MTLSize(width: 64, height: 1, depth: 1),
                   threadgroups: true, shmem: 256 * 8 + 128) { enc in
                enc.setBuffer(blobBuf, offset: 0, index: 0); enc.useResources(resources, usage: .read)
                enc.setBytes(&offsets, length: 8, index: 1); enc.setBuffer(rows, offset: p0 * sD * 4, index: 2); enc.setBuffer(acts, offset: 0, index: 3)
                enc.setBytes(&dV, length: 4, index: 4); enc.setBytes(&fV, length: 4, index: 5); enc.setBytes(&oneV, length: 4, index: 6)
                enc.setBytes(&strideV, length: 4, index: 7); enc.setBuffer(partOffBuf, offset: 0, index: 8); enc.setBytes(&nV, length: 4, index: 9)
                enc.setBuffer(posSlotBuf, offset: p0 * 4, index: 10)
            }
            let blit = cb.makeBlitCommandEncoder()!
            blit.fill(buffer: rows, range: (p0 * sD * 4)..<(PS * sD * 4), value: 0)
            blit.endEncoding()
            simRun(cb, k.phase2!, grid: MTLSize(width: (nSmall * sD + 7) / 8, height: 1, depth: 1), tg: MTLSize(width: 64, height: 1, depth: 1),
                   threadgroups: true) { enc in
                enc.setBuffer(blobBuf, offset: 0, index: 0); enc.useResources(resources, usage: .read)
                enc.setBytes(&offsets, length: 8, index: 1); enc.setBuffer(acts, offset: 0, index: 2); enc.setBuffer(sc.ones!, offset: 0, index: 3)
                enc.setBuffer(rows, offset: p0 * sD * 4, index: 4); enc.setBuffer(rows, offset: p0 * sD * 4, index: 5)
                enc.setBytes(&dV, length: 4, index: 6); enc.setBytes(&strideV, length: 4, index: 7); enc.setBytes(&oneV, length: 4, index: 8)
                enc.setBuffer(partOffBuf, offset: 0, index: 9); enc.setBytes(&nV, length: 4, index: 10)
                enc.setBuffer(posSlotBuf, offset: p0 * 4, index: 11)
            }
        }
        simRun(cb, k.scatter!, grid: MTLSize(width: (sD + 31) / 32, height: T, depth: 1), tg: MTLSize(width: 16, height: 64, depth: 1)) { enc in
            enc.setBuffer(rows, offset: 0, index: 0); enc.setBuffer(atBuf, offset: 0, index: 1); enc.setBuffer(wBuf, offset: 0, index: 2)
            enc.setBuffer(residual, offset: 0, index: 3); enc.setBuffer(y, offset: 0, index: 4)
            enc.setBytes(&dV, length: 4, index: 5); enc.setBytes(&kV, length: 4, index: 6)
        }
    }
}

/// Packed offsets of `pieces`, each at a page boundary; returns (offsets, total bytes).
private func simLayout(_ pieces: [Piece]) -> ([Int], Int) {
    let page = 16384
    var at: [Int] = [], o = 0
    for p in pieces { at.append(o); o += (p.byteCount + page - 1) / page * page }
    return (at, max(o, page))
}

/// The M6's weights: resident layers in their own buffers, the rest streamed into one staging buffer.
private final class SimOwned {
    let resident: [Int: MTLBuffer]
    let staging: MTLBuffer
    let stagingLayout: [Int: [Int]]
    private let fds: [ObjectIdentifier: Int32]
    let fillSeconds: Double, residentBytes: Int, streamedLayers: [Int]
    private let set: MTLResidencySet

    init(layers: [SimLayer], residentGB: Double, device: MTLDevice, queue: MTLCommandQueue, gguf: String, sidecar: String,
         file: GGUFFile, side: GGUFFile) throws {
        var fds: [ObjectIdentifier: Int32] = [:]
        for (f, path) in [(file, gguf), (side, sidecar)] {
            let fd = open((path as NSString).expandingTildeInPath, O_RDONLY)
            precondition(fd >= 0 && fcntl(fd, F_NOCACHE, 1) == 0, "open \(path)")
            fds[ObjectIdentifier(f)] = fd
        }
        self.fds = fds
        // Resident: layers in order while their whole share fits.
        var res: [Int: MTLBuffer] = [:], total = 0
        let cap = Int(residentGB * 1e9)
        let t0 = CFAbsoluteTimeGetCurrent()
        for l in layers {
            let (at, n) = simLayout(l.fullPieces)
            guard total + n <= cap else { break }
            simDrop(l.fullPieces)
            let b = device.makeBuffer(length: n, options: .storageModeShared)!
            simPreadNoCache(l.fullPieces, at: at, fds: fds, into: b.contents(), threads: 4)
            simDrop(l.fullPieces)
            l.bindPacked(b, pieces: l.fullPieces, at: at)
            res[l.il] = b; total += n
        }
        fillSeconds = CFAbsoluteTimeGetCurrent() - t0
        resident = res; residentBytes = total
        streamedLayers = layers.filter { res[$0.il] == nil }.map { $0.il }
        var layouts: [Int: [Int]] = [:], maxN = 16384
        for l in layers where res[l.il] == nil {
            let (at, n) = simLayout(l.readPieces)
            layouts[l.il] = at; maxN = max(maxN, n)
        }
        stagingLayout = layouts
        staging = device.makeBuffer(length: maxN, options: .storageModeShared)!
        for l in layers where res[l.il] == nil { l.bindPacked(staging, pieces: l.readPieces, at: layouts[l.il]!) }
        let desc = MTLResidencySetDescriptor()
        desc.label = "qfn-pair-sim"
        set = try device.makeResidencySet(descriptor: desc)
        for b in res.values { set.addAllocation(b) }
        set.addAllocation(staging)
        set.commit()
        set.requestResidency()
        queue.addResidencySet(set)
    }

    /// Reads a streamed layer into the staging buffer (F_NOCACHE), then drops it from the page cache. Seconds of the read.
    func read(_ l: SimLayer) -> Double {
        guard let at = stagingLayout[l.il] else { return 0 }
        let t0 = CFAbsoluteTimeGetCurrent()
        simPreadNoCache(l.readPieces, at: at, fds: fds, into: staging.contents(), threads: 4)
        let s = CFAbsoluteTimeGetCurrent() - t0
        simDrop(l.readPieces)
        return s
    }

    /// QFN_FEED: the first 1 - f of the layer's bytes from this SSD, the rest from the MBP over `feed`, at once.
    /// Returns (wall, own read, fed wait, own MB, fed MB).
    func readFed(_ l: SimLayer, feed: Int32, fraction f: Double, files: [GGUFFile]) -> (Double, Double, Double, Double, Double) {
        guard let at = stagingLayout[l.il] else { return (0, 0, 0, 0, 0) }
        let total = l.readBytes, cut = Int(Double(total) * (1 - f))
        var own: [Piece] = [], ownAt: [Int] = [], fed: [Piece] = [], fedAt: [Int] = []
        var o = 0
        for (i, p) in l.readPieces.enumerated() {
            let a = max(0, min(p.byteCount, cut - o))
            if a > 0 { own.append((p.file, p.offset, a)); ownAt.append(at[i]) }
            if a < p.byteCount { fed.append((p.file, p.offset + a, p.byteCount - a)); fedAt.append(at[i] + a) }
            o += p.byteCount
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        var req = [Int64(fed.count)]
        for p in fed { req += [Int64(files.firstIndex { $0 === p.file }!), Int64(p.offset), Int64(p.byteCount)] }
        req.withUnsafeBytes { simSockAll(feed, $0.baseAddress!, $0.count, send: true) }
        let dst = staging.contents()
        var fedDone = 0.0
        let group = DispatchGroup()
        DispatchQueue.global().async(group: group) {
            for (i, p) in fed.enumerated() { simSockAll(feed, dst + fedAt[i], p.byteCount, send: false) }
            fedDone = CFAbsoluteTimeGetCurrent() - t0
        }
        simPreadNoCache(own, at: ownAt, fds: fds, into: dst, threads: 4)
        let ownS = CFAbsoluteTimeGetCurrent() - t0
        group.wait()
        let wall = CFAbsoluteTimeGetCurrent() - t0
        simDrop(own)
        return (wall, ownS, fedDone, Double(own.reduce(0) { $0 + $1.byteCount }) / 1e6, Double(fed.reduce(0) { $0 + $1.byteCount }) / 1e6)
    }
}

private func simSockAll(_ fd: Int32, _ p: UnsafeRawPointer, _ n: Int, send: Bool) {
    var done = 0
    while done < n {
        let r = send ? Darwin.send(fd, p + done, min(n - done, 1 << 24), 0)
                     : Darwin.recv(fd, UnsafeMutableRawPointer(mutating: p + done), min(n - done, 1 << 24), 0)
        guard r > 0 else { perror(send ? "send" : "recv"); exit(1) }
        done += r
    }
}

private func simSockTune(_ fd: Int32) {
    var one: Int32 = 1, buf: Int32 = 8 << 20
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, 4)
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, 4); setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, 4)
}

private func simAddr(_ ip: String, _ port: Int) -> sockaddr_in {
    var a = sockaddr_in()
    a.sin_family = sa_family_t(AF_INET); a.sin_port = in_port_t(UInt16(port).bigEndian)
    inet_pton(AF_INET, ip, &a.sin_addr)
    return a
}

/// The MBP's side of QFN_FEED: answers piece lists from the M6 until it sends -1. Blocks of 16 MB are read by 4 threads
/// (F_NOCACHE) into one buffer and sent in order as each completes. Prints a summary at the end.
private func simFeedServe(_ fd: Int32, paths: [String]) {
    let fds = paths.map { path -> Int32 in
        let f = open((path as NSString).expandingTildeInPath, O_RDONLY)
        precondition(f >= 0 && fcntl(f, F_NOCACHE, 1) == 0, "open \(path)")
        return f
    }
    var cap = 0
    var buf: UnsafeMutableRawPointer?
    var layers = 0, bytes = 0, busy = 0.0
    while true {
        var n: Int64 = 0
        withUnsafeMutableBytes(of: &n) { simSockAll(fd, $0.baseAddress!, 8, send: false) }
        if n < 0 { break }
        var list = [Int64](repeating: 0, count: Int(n) * 3)
        if n > 0 { list.withUnsafeMutableBytes { simSockAll(fd, $0.baseAddress!, $0.count, send: false) } }
        let t0 = CFAbsoluteTimeGetCurrent()
        var blocks: [(Int32, Int, Int, Int)] = [], total = 0
        for i in 0..<Int(n) {
            let f = fds[Int(list[3 * i])], off = Int(list[3 * i + 1]), len = Int(list[3 * i + 2])
            var o = 0
            while o < len { let b = min(16 << 20, len - o); blocks.append((f, off + o, b, total + o)); o += b }
            total += len
        }
        if total > cap { buf?.deallocate(); cap = total; buf = UnsafeMutableRawPointer.allocate(byteCount: cap, alignment: 16384) }
        let dst = buf!
        let ready = blocks.map { _ in DispatchSemaphore(value: 0) }
        let next = UnsafeMutablePointer<Int>.allocate(capacity: 1)
        next.initialize(to: 0)
        let lock = NSLock()
        let readers = DispatchGroup()
        for _ in 0..<4 {
            DispatchQueue.global().async(group: readers) {
                while true {
                    lock.lock(); let i = next.pointee; next.pointee += 1; lock.unlock()
                    guard i < blocks.count else { return }
                    let (f, off, len, d) = blocks[i]
                    var done = 0
                    while done < len { let got = pread(f, dst + d + done, len - done, off_t(off + done)); precondition(got > 0, "pread"); done += got }
                    ready[i].signal()
                }
            }
        }
        for (i, b) in blocks.enumerated() { ready[i].wait(); simSockAll(fd, dst + b.3, b.2, send: true) }
        readers.wait()
        next.deallocate()
        busy += CFAbsoluteTimeGetCurrent() - t0; layers += 1; bytes += total
    }
    print(String(format: "feed (MBP): %d layer reads, %.1f MB, %.3f s from the request to the last byte sent (%.2f GB/s)",
                 layers, Double(bytes) / 1e6, busy, busy > 0 ? Double(bytes) / busy / 1e9 : 0))
    for f in fds { close(f) }
}

func runQFNPairSim(mode: String, ip: String, port: Int, gguf: String, sidecar: String, routes: String, shape: String,
                   param: Double, last: Double) throws {
    setvbuf(stdout, nil, _IOLBF, 0)
    precondition(shape == "A" || shape == "B", "shape is A or B")
    let serve = mode == "serve"
    let file = try GGUFFile(url: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath))
    let side = try GGUFFile(url: URL(fileURLWithPath: (sidecar as NSString).expandingTildeInPath))
    let context = try MetalContext()
    let device = context.device, queue = context.queue
    let nax = serve
    let kernels = try SimKernels(device: device, queue: queue, nax: nax)
    func now() -> Double { CFAbsoluteTimeGetCurrent() }
    let page = Double(getpagesize())
    func mem() -> String {
        let v = simVM()
        return String(format: "free %.2f GB wired %.2f GB", Double(v.free) * page / 1e9, Double(v.wired) * page / 1e9)
    }

    // This side's layers.
    let k = shape == "A" ? Int(param) : 0
    let cut = shape == "B" ? Int((Double(sExperts) * (1 - param)).rounded()) : sExperts
    var mine: [Int]
    var lo = 0, hi = sExperts, dense = true
    if shape == "A" { mine = serve ? Array(0..<k) : Array(k..<sLayers) }
    else { mine = Array(0..<sLayers); (lo, hi) = serve ? (cut, sExperts) : (0, cut); dense = !serve }
    let tBuild = now()
    let layers = try mine.map { L in
        try SimLayer(file: file, side: side, layer: L, route: try Data(contentsOf: URL(fileURLWithPath: "\(routes)-l\(L).bin")),
                     lo: lo, hi: hi, dense: dense, runtimeOrder: !nax, device: device)
    }
    let T = layers.first?.T ?? 2048
    let items = layers.flatMap { $0.items }
    let scratch = SimScratch(device: device, T: T, maxK: items.map { $0.k }.max() ?? 32, maxM: items.map { $0.m }.max() ?? 32,
                             maxQ8: items.filter { $0.group != "fp32" && $0.tensor.type == .q8_0 }.map { $0.m * $0.k }.max() ?? 1, nax: nax)
    let readMB = Double(layers.reduce(0) { $0 + $1.readBytes }) / 1e6
    let label = shape == "A" ? "A k \(k): layers \(mine.first ?? 0)..<\((mine.last ?? -1) + 1)"
                             : String(format: "B sigma %.3f: experts [%d, %d) of 48 layers%@", param, lo, hi, dense ? " + dense" : "")
    print(String(format: "qfn-pair-sim %@ (%@): %@, T %d, %d layers built in %.2f s, this route reads %.1f MB",
                 mode, nax ? "M6 arm M" : "MBP today's code", label, T, layers.count, now() - tBuild, readMB))

    var owned: SimOwned?
    if nax {
        if !layers.isEmpty {
            let v0 = simVM()
            owned = try SimOwned(layers: layers, residentGB: last, device: device, queue: queue, gguf: gguf, sidecar: sidecar, file: file, side: side)
            let v1 = simVM()
            print(String(format: "  resident %d layers %.2f GB (cap %.2f) filled in %.2f s, streamed %d layers; Pageins +%.2f GB, Swapouts +%llu; %@",
                         owned!.resident.count, Double(owned!.residentBytes) / 1e9, last, owned!.fillSeconds, owned!.streamedLayers.count,
                         Double(v1.pageins - v0.pageins) * page / 1e9, v1.swapouts - v0.swapouts, mem()))
        }
    } else {
        for l in layers { l.bindViews() }
    }

    // QFN_SIM_CHECK=1: layer 0's (A: first layer's) routed experts in full (views) against [0, 256) + [256, 512), the
    // upper half bound as this side binds it (the M6: owned / staged buffers). Exits.
    if ProcessInfo.processInfo.environment["QFN_SIM_CHECK"] != nil {
        let L = mine.first ?? 0
        let route = try Data(contentsOf: URL(fileURLWithPath: "\(routes)-l\(L).bin"))
        let full = try SimLayer(file: file, side: side, layer: L, route: route, lo: 0, hi: sExperts, dense: false, runtimeOrder: !nax, device: device)
        let a = try SimLayer(file: file, side: side, layer: L, route: route, lo: 0, hi: 256, dense: false, runtimeOrder: !nax, device: device)
        let b = try SimLayer(file: file, side: side, layer: L, route: route, lo: 256, hi: sExperts, dense: false, runtimeOrder: !nax, device: device)
        full.bindViews(); a.bindViews()
        GGUFFile.preadRanges(full.readPieces, threads: 4)
        var how = "views"
        if nax {
            let (at, n) = simLayout(b.fullPieces)
            let buf = device.makeBuffer(length: n, options: .storageModeShared)!
            var fds: [ObjectIdentifier: Int32] = [:]
            for (f, path) in [(file, gguf), (side, sidecar)] { let fd = open((path as NSString).expandingTildeInPath, O_RDONLY); _ = fcntl(fd, F_NOCACHE, 1); fds[ObjectIdentifier(f)] = fd }
            simPreadNoCache(b.fullPieces, at: at, fds: fds, into: buf.contents(), threads: 4)
            b.bindPacked(buf, pieces: b.fullPieces, at: at)
            how = "an owned buffer of every expert of [256, 512)"
        } else { b.bindViews() }
        func run(_ l: SimLayer) -> [Float] {
            let y = device.makeBuffer(length: T * sD * 4, options: .storageModeShared)!
            _ = l.encodeRouted(kernels, scratch, nax: nax, x: scratch.x, residual: scratch.zero, y: y)
            return Array(UnsafeBufferPointer(start: y.contents().bindMemory(to: Float.self, capacity: T * sD), count: T * sD))
        }
        let f = run(full), ya = run(a), yb = run(b)
        var d: Float = 0, m: Float = 0
        for i in 0..<(T * sD) { d = max(d, abs(f[i] - ya[i] - yb[i])); m = max(m, abs(f[i])) }
        print(String(format: "check layer %d (%@): experts %d + %d of %d (upper half from %@), max |full - (lo + hi)| / max |full| = %.2e",
                     L, nax ? "tensor-op tiles" : "runtime routedGemm", a.S, b.S, full.S, how, d / m))
        return
    }

    let feedFraction = shape == "A" ? Double(ProcessInfo.processInfo.environment["QFN_FEED"] ?? "0") ?? 0 : 0
    let hcBytes = sHC * T * sD * 4, rowBytes = T * sD * 4, assignBytes = T * sTopK * 8
    let assign = UnsafeMutableRawPointer.allocate(byteCount: assignBytes, alignment: 16)
    memset(assign, 0, assignBytes)
    let hcOut = [UnsafeMutableRawPointer.allocate(byteCount: hcBytes, alignment: 16), UnsafeMutableRawPointer.allocate(byteCount: hcBytes, alignment: 16)]
    for b in hcOut { memset(b, 0, hcBytes) }
    let hcIn = UnsafeMutableRawPointer.allocate(byteCount: hcBytes, alignment: 16)

    if serve {
        let ls = socket(AF_INET, SOCK_STREAM, 0)
        var one: Int32 = 1
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, 4)
        var a = simAddr(ip, port)
        guard withUnsafePointer(to: &a, { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(ls, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }) == 0,
              listen(ls, 2) == 0 else { perror("bind/listen"); exit(1) }
        if feedFraction > 0 { print(String(format: "feed: the last %.3f of every streamed layer's bytes from the MBP (QFN_FEED)", feedFraction)) }
        print("listening on \(ip):\(port)")
        let fd = accept(ls, nil, nil)
        simSockTune(fd)
        var feed: Int32 = -1
        if feedFraction > 0 { feed = accept(ls, nil, nil); simSockTune(feed) }
        let byL = Dictionary(uniqueKeysWithValues: layers.map { ($0.il, $0) })
        if feedFraction > 0 { print("chunk\tlayer\tstreamed_MB\trecv_s\tread_s\tdense_gpu_s\trouted_gpu_s\tcompute_wall_s\tsend_s\town_MB\town_s\tfed_MB\tfed_s") }
        else { print("chunk\tlayer\tstreamed_MB\trecv_s\tread_s\tdense_gpu_s\trouted_gpu_s\tcompute_wall_s\tsend_s") }
        if shape == "A" {
            var header: Int32 = 0
            withUnsafeMutableBytes(of: &header) { simSockAll(fd, $0.baseAddress!, 4, send: false) }
            let sender = DispatchQueue(label: "send")
            let sent = DispatchGroup()
            let v0 = simVM()
            for c in 0..<Int(header) {
                let tc = now()
                for l in layers {
                    let fedRead = feed >= 0 && owned!.stagingLayout[l.il] != nil
                    let r = fedRead ? owned!.readFed(l, feed: feed, fraction: feedFraction, files: [file, side]) : (owned!.read(l), 0, 0, 0, 0)
                    let t1 = now()
                    let gd = l.encodeDense(kernels, scratch, nax: true)
                    let gr = l.encodeRouted(kernels, scratch, nax: true, x: scratch.x, residual: scratch.zero, y: scratch.y)
                    let streamed = owned!.stagingLayout[l.il] != nil ? Double(l.readBytes) / 1e6 : 0
                    let tail = feed >= 0 ? String(format: "\t%.1f\t%.4f\t%.1f\t%.4f", r.3, r.1, r.4, r.2) : ""
                    print(String(format: "%d\t%d\t%.1f\t0\t%.4f\t%.4f\t%.4f\t%.4f\t0", c, l.il, streamed, r.0, gd, gr, now() - t1) + tail)
                }
                let buf = hcOut[c % 2]
                sent.wait()   // the other buffer's send (18 ms) has long finished; keep one in flight
                sender.async(group: sent) { simSockAll(fd, buf, hcBytes, send: true) }
                print(String(format: "chunk %d on the M6: %.3f s (layers 0..<%d)", c, now() - tc, k))
            }
            sent.wait()
            if feed >= 0 {
                var end: Int64 = -1
                withUnsafeBytes(of: &end) { simSockAll(feed, $0.baseAddress!, 8, send: true) }
                close(feed)
            }
            let v1 = simVM()
            print(String(format: "M6 run: Pageins +%.2f GB, Swapouts +%llu; %@", Double(v1.pageins - v0.pageins) * page / 1e9, v1.swapouts - v0.swapouts, mem()))
        } else {
            var h = [Int32](repeating: 0, count: 2)
            let v0 = simVM()
            while true {
                h.withUnsafeMutableBytes { simSockAll(fd, $0.baseAddress!, 8, send: false) }
                if h[0] < 0 { break }
                let l = byL[Int(h[1])]!
                let t0 = now()
                simSockAll(fd, scratch.x.contents(), rowBytes, send: false)
                simSockAll(fd, assign, assignBytes, send: false)
                let t1 = now()
                let rd = owned!.read(l)
                let t2 = now()
                let gr = l.encodeRouted(kernels, scratch, nax: true, x: scratch.x, residual: scratch.zero, y: scratch.y)
                let t3 = now()
                simSockAll(fd, scratch.y.contents(), rowBytes, send: true)
                let t4 = now()
                let streamed = owned!.stagingLayout[l.il] != nil ? Double(l.readBytes) / 1e6 : 0
                print(String(format: "%d\t%d\t%.1f\t%.4f\t%.4f\t0\t%.4f\t%.4f\t%.4f", Int(h[0]), l.il, streamed, t1 - t0, rd, gr, t3 - t2, t4 - t3))
            }
            let v1 = simVM()
            print(String(format: "M6 run: Pageins +%.2f GB, Swapouts +%llu; %@", Double(v1.pageins - v0.pageins) * page / 1e9, v1.swapouts - v0.swapouts, mem()))
        }
        close(fd); close(ls)
        return
    }

    // MBP.
    let chunks = Int(last)
    let peer = (shape == "A" && k > 0) || (shape == "B" && cut < sExperts)
    var fd: Int32 = -1
    if peer {
        fd = socket(AF_INET, SOCK_STREAM, 0)
        var a = simAddr(ip, port)
        guard withUnsafePointer(to: &a, { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }) == 0
        else { perror("connect"); exit(1) }
        simSockTune(fd)
    }
    let sock = fd
    var feedThread: Thread?
    let feedDone = DispatchSemaphore(value: 0)
    if peer && feedFraction > 0 {
        let ffd = socket(AF_INET, SOCK_STREAM, 0)
        var a = simAddr(ip, port)
        guard withUnsafePointer(to: &a, { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(ffd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }) == 0
        else { perror("connect feed"); exit(1) }
        simSockTune(ffd)
        feedThread = Thread { simFeedServe(ffd, paths: [gguf, sidecar]); close(ffd); feedDone.signal() }
        feedThread!.start()
    }
    let peerY = UnsafeMutableRawPointer.allocate(byteCount: rowBytes, alignment: 16)
    let peerQueue = DispatchQueue(label: "peer")
    print("chunk\tlayer\tread_MB\tread_s\tdense_gpu_s\trouted_gpu_s\twait_s\tlayer_s")
    let v0 = simVM()
    let tRun = now()
    if shape == "A" && peer {
        var header = Int32(chunks)
        withUnsafeBytes(of: &header) { simSockAll(sock, $0.baseAddress!, 4, send: true) }
    }
    for c in 0..<chunks {
        let tc = now()
        var waitHC = 0.0
        if shape == "A" && peer {
            simSockAll(sock, hcIn, hcBytes, send: false)
            waitHC = now() - tc
            print(String(format: "chunk %d: residual from the M6 after %.3f s (%.3f s since the start)", c, waitHC, now() - tRun))
        }
        for l in layers {
            let t0 = now()
            GGUFFile.preadRanges(l.readPieces, threads: 4)
            let t1 = now()
            let gd = l.encodeDense(kernels, scratch, nax: false)
            let group = DispatchGroup()
            if shape == "B" && peer {
                let L = l.il
                peerQueue.async(group: group) {
                    var h: [Int32] = [Int32(c), Int32(L)]
                    h.withUnsafeMutableBytes { simSockAll(sock, $0.baseAddress!, 8, send: true) }
                    simSockAll(sock, scratch.x.contents(), rowBytes, send: true)
                    simSockAll(sock, assign, assignBytes, send: true)
                    simSockAll(sock, peerY, rowBytes, send: false)
                }
            }
            let gr = l.encodeRouted(kernels, scratch, nax: false, x: scratch.x, residual: scratch.zero, y: scratch.y)
            let t2 = now()
            group.wait()
            let t3 = now()
            print(String(format: "%d\t%d\t%.1f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f", c, l.il, Double(l.readBytes) / 1e6, t1 - t0, gd, gr, t3 - t2, t3 - t0))
        }
        print(String(format: "chunk %d: %.3f s (wait for the M6 %.3f s), %.3f s since the start", c, now() - tc, waitHC, now() - tRun))
    }
    let total = now() - tRun
    if shape == "B" && peer {
        var h: [Int32] = [-1, 0]
        h.withUnsafeMutableBytes { simSockAll(sock, $0.baseAddress!, 8, send: true) }
    }
    if feedThread != nil { feedDone.wait() }
    if sock >= 0 { close(sock) }
    let v1 = simVM()
    print(String(format: "total: %d chunks %.3f s; MBP Pageins +%.2f GB, Swapouts +%llu; %@", chunks, total,
                 Double(v1.pageins - v0.pageins) * page / 1e9, v1.swapouts - v0.swapouts, mem()))
}

/// The M6's float -> half input pass and Q8_0 rows -> half (as --qfn-layer-bench arm C).
private let simHalfSource = """
#include <metal_stdlib>
using namespace metal;
kernel void qfn_sim_f32_to_f16(device const float4 *src [[buffer(0)]], device half4 *dst [[buffer(1)]],
                               constant uint &n4 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n4) dst[gid] = half4(src[gid]);
}
kernel void qfn_sim_q8_0_dequant_f16(device const char *w [[buffer(0)]], device half *out [[buffer(1)]],
                                     constant uint &K [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
    device const char *b = w + ((ulong)pos.y * (K / 32) + pos.x) * 34;
    const float d = (float)*(device const half *)b;
    device half *o = out + (ulong)pos.y * K + pos.x * 32;
    for (uint i = 0; i < 32; i++) o[i] = (half)(d * (float)b[2 + i]);
}
"""
