import Darwin
import Foundation
@preconcurrency import Metal
import MetalPerformanceShaders
import Tsugumi

// MARK: - `--qfn-split-probe serve|run ...` (docs/qfn-prefill/07 E3)
//
// Candidate B's routed step (docs/qfn-prefill/08 §2) for one chunk, 48 layers, on two machines over raw TCP:
//   M6:  --qfn-split-probe serve <bind ip> <port> <gguf> <routes prefix> <sigma>
//   MBP: --qfn-split-probe run <server ip> <port> <gguf> <routes prefix> <sigma> [chunks]
// Per layer the MBP sends `mixed` (T x 2560 fp32, random) and the top-10 assignment (T x 10 x 8 B) and, while that is
// in flight and the answer is awaited, reads and computes its own share; the M6 receives, reads its share's selected
// experts (pread, cold), computes them and sends the partial sum (T x 2560 fp32) back. Experts [cut, 512) are the M6's,
// cut = round(512 (1 - sigma)), as bench/qfn/e2_lists.py splits the mirror. sigma 0 runs the MBP alone (no socket).
// Routes are real (`Q38_DUMP_ROUTE`, `<prefix>-l<L>.bin`, both machines hold a copy); the GEMM is Q2GemmBench's arm B
// (dequant to float32 + MPS sgemm, group 4) over the share's slots, and the down weights are the GGUF's (padded 768),
// not the sidecar, so a share reads 1,489,920 B per expert instead of the runtime's 1,387,520.
// Prints one line per layer per side: read / GEMM / (MBP) wait for the partial sum and the layer's wall time.

private let qD = 2560, qF = 640, qActStride = 768, qExperts = 512, qTopK = 10

/// One layer's routed experts restricted to the expert range [lo, hi): pread + dequant/sgemm into a partial sum.
private final class RoutedShare {
    let T: Int, P: Int, S: Int
    let ranges: [(offset: Int, byteCount: Int)]
    let bytes: Int
    private let device: MTLDevice, queue: MTLCommandQueue
    private let psoGather, psoDeqGU, psoDeqDown, psoSilu, psoScatter: MTLComputePipelineState
    private let arg, partOffBuf, orderBuf, atBuf, wBuf: MTLBuffer
    /// One no-copy view per selected expert and tensor, as the runner's expert views: a view over the whole tensor makes
    /// Metal map every expert's pages, the other machine's share included.
    private let views: [MTLBuffer]
    private let downRow: Int
    /// Scratch shared by every layer (one layer runs at a time): gathered rows, gate|up, down rows, G experts of weights.
    struct Scratch {
        let xg, gu, dn, wGU, wDown: MTLBuffer
        init(device: MTLDevice, pairs P: Int) {
            func floats(_ n: Int) -> MTLBuffer { device.makeBuffer(length: max(n, 1) * 4, options: .storageModeShared)! }
            xg = floats(P * qD); gu = floats(P * 2 * qF); dn = floats(P * qD)
            wGU = floats(4 * 2 * qF * qD); wDown = floats(4 * qD * qF)
        }
    }
    private let xg, gu, dn, wGU, wDown: MTLBuffer
    private let count, start: [Int]
    private var muls: [Int: (MPSMatrixMultiplication, MPSMatrixMultiplication)] = [:]

    init(file: GGUFFile, layer il: Int, route: Data, lo: Int, hi: Int, device: MTLDevice, queue: MTLCommandQueue,
         lib: MTLLibrary, scratch: Scratch) throws {
        self.device = device; self.queue = queue
        func pso(_ name: String) throws -> MTLComputePipelineState {
            try device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
        }
        psoGather = try pso("moe_gather_pair_rows"); psoDeqGU = try pso("moe_iq2xxs_dequant_gate_up_f32")
        psoDeqDown = try pso("moe_q2k_dequant_down_f32"); psoSilu = try pso("moe_silu_mul_halves")
        psoScatter = try pso("moe_scatter_weighted")
        let T0 = Int(route.withUnsafeBytes { $0.load(as: Int32.self) })
        T = min(T0, 2048)
        let Pall = T * qTopK, P0 = T0 * qTopK
        var experts = [Int32](repeating: 0, count: Pall), weights = [Float](repeating: 0, count: Pall)
        route.withUnsafeBytes { raw in
            experts.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[4..<(4 + 4 * Pall)])) }
            weights.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[(4 + 4 * P0)..<(4 + 4 * P0 + 4 * Pall)])) }
        }
        // Slots in first-seen order over this share's pairs; the other pairs scatter row 0 with weight 0.
        var slotOf = [Int](repeating: -1, count: qExperts)
        var slots: [Int] = []
        var pairSlot = [Int](repeating: -1, count: Pall)
        for p in 0..<Pall {
            let ex = Int(experts[p])
            guard ex >= lo && ex < hi else { continue }
            if slotOf[ex] < 0 { slotOf[ex] = slots.count; slots.append(ex) }
            pairSlot[p] = slotOf[ex]
        }
        S = slots.count
        var count = [Int](repeating: 0, count: S)
        for p in 0..<Pall where pairSlot[p] >= 0 { count[pairSlot[p]] += 1 }
        var start = [Int](repeating: 0, count: S + 1)
        for s in 0..<S { start[s + 1] = start[s] + count[s] }
        P = start[S]
        var fill = start
        var order = [UInt32](repeating: 0, count: max(P, 1))
        var at = [UInt32](repeating: 0, count: Pall)
        var w = weights
        for p in 0..<Pall {
            let s = pairSlot[p]
            if s < 0 { w[p] = 0; continue }
            order[fill[s]] = UInt32(p); at[p] = UInt32(fill[s]); fill[s] += 1
        }
        self.count = count; self.start = start

        let gateT = try file.tensor("blk.\(il).ffn_gate_exps.weight")
        let upT = try file.tensor("blk.\(il).ffn_up_exps.weight")
        let downT = try file.tensor("blk.\(il).ffn_down_exps.weight")
        let gateRow = gateT.bytesPerRow
        downRow = downT.bytesPerRow
        // The selected experts' bytes, adjacent experts merged, as the runtime's preadRanges calls.
        var r: [(offset: Int, byteCount: Int)] = []
        for (t, per) in [(gateT, gateRow * qF), (upT, gateRow * qF), (downT, downRow * qD)] {
            for ex in slots.sorted() {
                let o = t.offset + ex * per
                if let last = r.last, last.offset + last.byteCount == o { r[r.count - 1].byteCount += per } else { r.append((o, per)) }
            }
        }
        ranges = r; bytes = r.reduce(0) { $0 + $1.byteCount }
        let fn = lib.makeFunction(name: "moe_iq2xxs_dequant_gate_up_f32")!
        let argEnc = fn.makeArgumentEncoder(bufferIndex: 0)
        arg = device.makeBuffer(length: argEnc.encodedLength, options: .storageModeShared)!
        argEnc.setArgumentBuffer(arg, offset: 0)
        var partOff = [UInt32](repeating: 0, count: 3 * qExperts)
        var views: [MTLBuffer] = []
        for (s, ex) in slots.enumerated() {
            for (k, (t, per)) in [(gateT, gateRow * qF), (upT, gateRow * qF), (downT, downRow * qD)].enumerated() {
                let (v, o) = file.noCopyBuffer(device: device, offset: t.offset + ex * per, byteCount: per)!
                argEnc.setBuffer(v, offset: 0, index: k * qExperts + s)
                partOff[3 * s + k] = UInt32(o)
                views.append(v)
            }
        }
        self.views = views
        func shared<E>(_ a: [E]) -> MTLBuffer {
            a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)! }
        }
        partOffBuf = shared(partOff); orderBuf = shared(order); atBuf = shared(at); wBuf = shared(w)
        xg = scratch.xg; gu = scratch.gu; dn = scratch.dn; wGU = scratch.wGU; wDown = scratch.wDown
        for n in Set(count) {
            muls[n] = (MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                               resultRows: n, resultColumns: 2 * qF, interiorColumns: qD, alpha: 1, beta: 0),
                       MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                               resultRows: n, resultColumns: qD, interiorColumns: qF, alpha: 1, beta: 0))
        }
    }

    func read(file: GGUFFile) { if !ranges.isEmpty { file.preadRanges(ranges, threads: 4) } }
    /// Bytes of this share not in the page cache (mincore), before `read`: a warm share reads at memory speed.
    func coldBytes(file: GGUFFile) -> Int { ranges.reduce(0) { $0 + file.nonResidentBytes(offset: $1.offset, byteCount: $1.byteCount) } }

    /// y = residual + sum over this share's pairs. Returns GPU seconds.
    func compute(x: MTLBuffer, residual: MTLBuffer, y: MTLBuffer) -> Double {
        let G = 4
        func matrix(_ b: MTLBuffer, _ offset: Int, _ rows: Int, _ cols: Int, rowBytes: Int? = nil) -> MPSMatrix {
            MPSMatrix(buffer: b, offset: offset,
                      descriptor: MPSMatrixDescriptor(rows: rows, columns: cols, rowBytes: rowBytes ?? cols * 4, dataType: .float32))
        }
        var cbs: [MTLCommandBuffer] = []
        var cb = queue.makeCommandBuffer()!
        if P > 0 {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoGather)
            enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(orderBuf, offset: 0, index: 1); enc.setBuffer(xg, offset: 0, index: 2)
            var wV = UInt32(qD), kV = UInt32(qTopK)
            enc.setBytes(&wV, length: 4, index: 3); enc.setBytes(&kV, length: 4, index: 4)
            enc.dispatchThreads(MTLSize(width: (qD + 31) / 32, height: P, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 64, depth: 1))
            enc.endEncoding()
        }
        for (gi, g0) in stride(from: 0, to: S, by: G).enumerated() {
            let n = min(G, S - g0)
            var firstSlot = UInt32(g0), dV = UInt32(qD), fV = UInt32(qF)
            var enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoDeqGU)
            enc.setBuffer(arg, offset: 0, index: 0); enc.useResources(views, usage: .read)
            enc.setBuffer(partOffBuf, offset: 0, index: 1); enc.setBuffer(wGU, offset: 0, index: 2)
            enc.setBytes(&dV, length: 4, index: 3); enc.setBytes(&fV, length: 4, index: 4); enc.setBytes(&firstSlot, length: 4, index: 5)
            enc.dispatchThreads(MTLSize(width: qD / 32, height: 2 * qF, depth: n), threadsPerThreadgroup: MTLSize(width: 8, height: 32, depth: 1))
            enc.endEncoding()
            enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoDeqDown)
            enc.setBuffer(arg, offset: 0, index: 0); enc.useResources(views, usage: .read)
            enc.setBuffer(partOffBuf, offset: 0, index: 1); enc.setBuffer(wDown, offset: 0, index: 2)
            var sV = UInt32(qActStride), cV = UInt32(qF), rbV = UInt32(downRow)
            enc.setBytes(&dV, length: 4, index: 3); enc.setBytes(&sV, length: 4, index: 4); enc.setBytes(&cV, length: 4, index: 5)
            enc.setBytes(&firstSlot, length: 4, index: 6); enc.setBytes(&rbV, length: 4, index: 7)
            enc.dispatchThreads(MTLSize(width: qF / 16, height: qD, depth: n), threadsPerThreadgroup: MTLSize(width: 4, height: 64, depth: 1))
            enc.endEncoding()
            for s in g0..<(g0 + n) {
                muls[count[s]]!.0.encode(commandBuffer: cb, leftMatrix: matrix(xg, start[s] * qD * 4, count[s], qD),
                                         rightMatrix: matrix(wGU, (s - g0) * 2 * qF * qD * 4, 2 * qF, qD),
                                         resultMatrix: matrix(gu, start[s] * 2 * qF * 4, count[s], 2 * qF))
            }
            enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoSilu)
            enc.setBuffer(gu, offset: 0, index: 0)
            var firstV = UInt32(start[g0])
            enc.setBytes(&fV, length: 4, index: 1); enc.setBytes(&firstV, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: qF, height: start[g0 + n] - start[g0], depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 64, height: 16, depth: 1))
            enc.endEncoding()
            for s in g0..<(g0 + n) {
                muls[count[s]]!.1.encode(commandBuffer: cb, leftMatrix: matrix(gu, start[s] * 2 * qF * 4, count[s], qF, rowBytes: 2 * qF * 4),
                                         rightMatrix: matrix(wDown, (s - g0) * qD * qF * 4, qD, qF),
                                         resultMatrix: matrix(dn, start[s] * qD * 4, count[s], qD))
            }
            if gi % 4 == 3 { cb.commit(); cbs.append(cb); cb = queue.makeCommandBuffer()! }
        }
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psoScatter)
        enc.setBuffer(dn, offset: 0, index: 0); enc.setBuffer(atBuf, offset: 0, index: 1); enc.setBuffer(wBuf, offset: 0, index: 2)
        enc.setBuffer(residual, offset: 0, index: 3); enc.setBuffer(y, offset: 0, index: 4)
        var dV = UInt32(qD), kV = UInt32(qTopK)
        enc.setBytes(&dV, length: 4, index: 5); enc.setBytes(&kV, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: (qD + 31) / 32, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 64, depth: 1))
        enc.endEncoding()
        cb.commit(); cbs.append(cb)
        cb.waitUntilCompleted()
        return cbs.reduce(0.0) { $0 + ($1.gpuEndTime - $1.gpuStartTime) }
    }
}

private func sockAll(_ fd: Int32, _ p: UnsafeRawPointer, _ n: Int, send: Bool) {
    var done = 0
    while done < n {
        let r = send ? Darwin.send(fd, p + done, min(n - done, 1 << 24), 0)
                     : Darwin.recv(fd, UnsafeMutableRawPointer(mutating: p + done), min(n - done, 1 << 24), 0)
        guard r > 0 else { perror(send ? "send" : "recv"); exit(1) }
        done += r
    }
}

private func sockTune(_ fd: Int32) {
    var one: Int32 = 1, buf: Int32 = 8 << 20
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, 4)
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, 4); setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, 4)
}

private func sockAddr(_ ip: String, _ port: Int) -> sockaddr_in {
    var a = sockaddr_in()
    a.sin_family = sa_family_t(AF_INET); a.sin_port = in_port_t(UInt16(port).bigEndian)
    inet_pton(AF_INET, ip, &a.sin_addr)
    return a
}

func runQFNSplitProbe(mode: String, ip: String, port: Int, gguf: String, routes: String, sigma: Double, chunks: Int) throws {
    setvbuf(stdout, nil, _IOLBF, 0)
    let file = try GGUFFile(url: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath))
    let context = try MetalContext()
    let device = context.device, queue = context.queue
    let lib = try MetalContext.moduleLibrary(device: device, module: "moe_ggml")
    let cut = Int((Double(qExperts) * (1 - sigma)).rounded())
    let serve = mode == "serve"
    let (lo, hi) = serve ? (cut, qExperts) : (0, cut)
    func now() -> Double { CFAbsoluteTimeGetCurrent() }
    let tBuild = now()
    let scratch = RoutedShare.Scratch(device: device, pairs: 2048 * qTopK)
    let shares = try (0..<48).map { L in
        try RoutedShare(file: file, layer: L, route: try Data(contentsOf: URL(fileURLWithPath: "\(routes)-l\(L).bin")),
                        lo: lo, hi: hi, device: device, queue: queue, lib: lib, scratch: scratch)
    }
    let T = shares[0].T
    let rowBytes = T * qD * 4, assignBytes = T * qTopK * 8
    print(String(format: "qfn-split-probe %@: sigma %.2f, experts [%d, %d), T %d, %d layers built in %.2f s",
                 mode, sigma, lo, hi, T, shares.count, now() - tBuild))
    let x = device.makeBuffer(length: rowBytes, options: .storageModeShared)!
    let xp = x.contents().bindMemory(to: Float.self, capacity: T * qD)
    for i in 0..<(T * qD) { xp[i] = Float.random(in: -1...1) }
    let zero = device.makeBuffer(length: rowBytes, options: .storageModeShared)!
    let yOwn = device.makeBuffer(length: rowBytes, options: .storageModeShared)!
    let yPeer = UnsafeMutableRawPointer.allocate(byteCount: rowBytes, alignment: 16)
    let assign = UnsafeMutableRawPointer.allocate(byteCount: assignBytes, alignment: 16)
    memset(assign, 0, assignBytes)

    // QFN_PROBE_CHECK=1: layer 0 in full against the two shares added (the split itself, no socket), then exit.
    if ProcessInfo.processInfo.environment["QFN_PROBE_CHECK"] != nil {
        let route = try Data(contentsOf: URL(fileURLWithPath: "\(routes)-l0.bin"))
        let full = try RoutedShare(file: file, layer: 0, route: route, lo: 0, hi: qExperts, device: device, queue: queue, lib: lib, scratch: scratch)
        let a = try RoutedShare(file: file, layer: 0, route: route, lo: 0, hi: cut, device: device, queue: queue, lib: lib, scratch: scratch)
        let b = try RoutedShare(file: file, layer: 0, route: route, lo: cut, hi: qExperts, device: device, queue: queue, lib: lib, scratch: scratch)
        full.read(file: file)
        let yF = device.makeBuffer(length: rowBytes, options: .storageModeShared)!
        let yA = device.makeBuffer(length: rowBytes, options: .storageModeShared)!
        let yB = device.makeBuffer(length: rowBytes, options: .storageModeShared)!
        _ = full.compute(x: x, residual: zero, y: yF); _ = a.compute(x: x, residual: zero, y: yA); _ = b.compute(x: x, residual: zero, y: yB)
        let f = yF.contents().bindMemory(to: Float.self, capacity: T * qD)
        let pa = yA.contents().bindMemory(to: Float.self, capacity: T * qD), pb = yB.contents().bindMemory(to: Float.self, capacity: T * qD)
        var d: Float = 0, m: Float = 0
        for i in 0..<(T * qD) { d = max(d, abs(f[i] - pa[i] - pb[i])); m = max(m, abs(f[i])) }
        print(String(format: "check: experts %d + %d of %d, max |full - (a + b)| / max |full| = %.2e", a.S, b.S, full.S, d / m))
        return
    }

    if serve {
        let ls = socket(AF_INET, SOCK_STREAM, 0)
        var one: Int32 = 1
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, 4)
        var a = sockAddr(ip, port)
        guard withUnsafePointer(to: &a, { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(ls, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }) == 0,
              listen(ls, 1) == 0 else { perror("bind/listen"); exit(1) }
        print("listening on \(ip):\(port)")
        let fd = accept(ls, nil, nil)
        sockTune(fd)
        print("chunk\tlayer\tbytes_MB\trecv_s\tread_s\tgemm_gpu_s\tgemm_wall_s\tsend_s\tcold_MB")
        var layerHeader = [Int32](repeating: 0, count: 2)
        while true {
            layerHeader.withUnsafeMutableBytes { sockAll(fd, $0.baseAddress!, 8, send: false) }
            if layerHeader[0] < 0 { break }
            let (c, L) = (Int(layerHeader[0]), Int(layerHeader[1]))
            let t0 = now()
            sockAll(fd, x.contents(), rowBytes, send: false)
            sockAll(fd, assign, assignBytes, send: false)
            let cold = shares[L].coldBytes(file: file)
            let t1 = now()
            shares[L].read(file: file)
            let t2 = now()
            let gpu = shares[L].compute(x: x, residual: zero, y: yOwn)
            let t3 = now()
            sockAll(fd, yOwn.contents(), rowBytes, send: true)
            let t4 = now()
            print(String(format: "%d\t%d\t%.1f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.1f", c, L, Double(shares[L].bytes) / 1e6,
                         t1 - t0, t2 - t1, gpu, t3 - t2, t4 - t3, Double(cold) / 1e6))
        }
        close(fd); close(ls)
        return
    }

    var fd: Int32 = -1
    if sigma > 0 {
        fd = socket(AF_INET, SOCK_STREAM, 0)
        var a = sockAddr(ip, port)
        guard withUnsafePointer(to: &a, { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }) == 0
        else { perror("connect"); exit(1) }
        sockTune(fd)
    }
    print("chunk\tlayer\tbytes_MB\tread_s\tgemm_gpu_s\tgemm_wall_s\twait_s\tlayer_s\tcold_MB")
    let peerQueue = DispatchQueue(label: "peer")
    let sock = fd
    for c in 0..<chunks {
        let tChunk = now()
        for L in 0..<48 {
            let t0 = now()
            let group = DispatchGroup()
            if sock >= 0 {
                peerQueue.async(group: group) {
                    var h: [Int32] = [Int32(c), Int32(L)]
                    h.withUnsafeMutableBytes { sockAll(sock, $0.baseAddress!, 8, send: true) }
                    sockAll(sock, x.contents(), rowBytes, send: true)
                    sockAll(sock, assign, assignBytes, send: true)
                    sockAll(sock, yPeer, rowBytes, send: false)
                }
            }
            let cold = shares[L].coldBytes(file: file)   // mincore of a few hundred ranges: well under a millisecond
            shares[L].read(file: file)
            let t1 = now()
            let gpu = shares[L].compute(x: x, residual: zero, y: yOwn)
            let t2 = now()
            group.wait()
            let t3 = now()
            print(String(format: "%d\t%d\t%.1f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.1f", c, L, Double(shares[L].bytes) / 1e6,
                         t1 - t0, gpu, t2 - t1, t3 - t2, t3 - t0, Double(cold) / 1e6))
        }
        print(String(format: "chunk %d: %.3f s", c, now() - tChunk))
    }
    if fd >= 0 {
        var h: [Int32] = [-1, 0]
        h.withUnsafeMutableBytes { sockAll(fd, $0.baseAddress!, 8, send: true) }
        close(fd)
    }
}
