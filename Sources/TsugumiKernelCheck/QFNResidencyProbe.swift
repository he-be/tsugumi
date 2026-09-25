import Darwin
import Foundation
import Metal
import Tsugumi

// MARK: - `--qfn-residency <gguf> <down sidecar> <routes prefix> layers|experts <GB> [passes] [scratch GB]`
//
// docs/qfn-prefill/12 F4: how much of an M6 share stays in the page cache when the same route is streamed again.
// The share is built from the route dumps (`<prefix>-l<L>.bin`, T = 2048):
//   layers   candidate A: layers 0, 1, ... whole (dense tensors + the selected experts), until the share reaches <GB>.
//   experts  candidate B: in every layer the selected experts in [cut, 512) (gate/up from the GGUF, down from the
//            sidecar), cut lowered until the share reaches <GB>.
// The share's ranges are dropped from the page cache first (`msync(MS_INVALIDATE)`, as --qfn-layer-bench cold).
// Each pass then goes layer by layer as the runner does: pread the layer's share ranges (4 threads), wrap them in
// no-copy views, run a GPU kernel that reads one word of every page, release the views. Per pass it prints the
// share's bytes not in the page cache before the pass (mincore), the read seconds, and the pass's Pageins /
// Swapouts deltas (host_statistics64). `scratch GB` of anonymous memory (touched) stands for the process's working
// buffers (arm M of --qfn-layer-bench holds about 0.6 GB at T = 2048); default 1.

private func vmCounters() -> (pageins: UInt64, swapouts: UInt64, free: UInt64, wired: UInt64, file: UInt64) {
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
    _ = withUnsafeMutablePointer(to: &stats) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
    }
    return (stats.pageins, stats.swapouts, UInt64(stats.free_count), UInt64(stats.wire_count), UInt64(stats.external_page_count))
}

typealias QFNRange = (file: GGUFFile, offset: Int, byteCount: Int)

/// The M6 share of `targetGB` from the route dumps, per layer: `layers` = layers 0, 1, ... whole (dense + selected
/// experts), `experts` = every layer's selected experts in [cut, 512).
func qfnShare(gguf: String, sidecar: String, routes: String, shape: String, targetGB: Double) throws
    -> (file: GGUFFile, side: GGUFFile, share: [[QFNRange]], label: String) {
    let file = try GGUFFile(url: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath))
    let side = try GGUFFile(url: URL(fileURLWithPath: (sidecar as NSString).expandingTildeInPath))
    let nLayers = 48, nExperts = 512, D = 2560, F = 640, topK = 10
    precondition(shape == "layers" || shape == "experts")
    let target = Int(targetGB * 1e9)

    // Selected experts per layer (first 2048 tokens of each dump).
    var selected: [[Int]] = []
    for il in 0..<nLayers {
        let route = try Data(contentsOf: URL(fileURLWithPath: "\(routes)-l\(il).bin"))
        let T0 = Int(route.withUnsafeBytes { $0.load(as: Int32.self) })
        let P = min(T0, 2048) * topK
        var seen = Set<Int>()
        route.withUnsafeBytes { raw in
            for p in 0..<P { seen.insert(Int(raw.load(fromByteOffset: 4 + 4 * p, as: Int32.self))) }
        }
        selected.append(seen.sorted())
    }
    typealias Range = QFNRange
    func expertRanges(_ il: Int, lo: Int) -> [Range] {
        var r: [Range] = []
        let gate = try! file.tensor("blk.\(il).ffn_gate_exps.weight"), up = try! file.tensor("blk.\(il).ffn_up_exps.weight")
        let down = try! side.tensor("blk.\(il).ffn_down_exps.weight")
        let gPer = gate.bytesPerRow * F, dPer = down.byteCount / nExperts
        precondition(dPer == 212 * D)
        for (f, t, per) in [(file, gate, gPer), (file, up, gPer), (side, down, dPer)] {
            for ex in selected[il] where ex >= lo {
                let o = t.offset + ex * per
                if let last = r.last, last.file === f, last.offset + last.byteCount == o { r[r.count - 1].byteCount += per }
                else { r.append((f, o, per)) }
            }
        }
        return r
    }
    func denseRanges(_ il: Int) -> [Range] {
        file.tensors.values.filter { $0.name.hasPrefix("blk.\(il).") && !$0.name.contains("_exps") }
            .map { (file, $0.offset, $0.byteCount) }
    }
    let bytes = { (r: [Range]) in r.reduce(0) { $0 + $1.byteCount } }
    var share: [[Range]] = []   // per layer
    var label = ""
    if shape == "layers" {
        var total = 0
        for il in 0..<nLayers where total < target {
            let r = denseRanges(il) + expertRanges(il, lo: 0)
            share.append(r); total += bytes(r)
        }
        label = "layers 0..<\(share.count)"
    } else {
        var cut = nExperts
        while cut > 0 {
            cut -= 1
            if (0..<nLayers).reduce(0, { $0 + bytes(expertRanges($1, lo: cut)) }) >= target { break }
        }
        share = (0..<nLayers).map { expertRanges($0, lo: cut) }
        label = String(format: "experts [%d, 512) of every layer (sigma %.3f)", cut, Double(nExperts - cut) / Double(nExperts))
    }
    return (file, side, share, label)
}

func runQFNResidencyProbe(gguf: String, sidecar: String, routes: String, shape: String, targetGB: Double,
                          passes: Int, scratchGB: Double) throws {
    let (file, side, share, label) = try qfnShare(gguf: gguf, sidecar: sidecar, routes: routes, shape: shape, targetGB: targetGB)
    _ = (file, side)
    let device = MTLCreateSystemDefaultDevice()!, queue = device.makeCommandQueue()!
    let all = share.flatMap { $0 }
    let shareBytes = all.reduce(0) { $0 + $1.byteCount }
    func nonResident() -> Int { all.reduce(0) { $0 + $1.file.nonResidentBytes(offset: $1.offset, byteCount: $1.byteCount) } }
    let page = Int(getpagesize())
    func invalidate() {
        for r in all {
            let a = r.offset / page * page, n = r.byteCount + (r.offset - a)
            let p = UnsafeMutableRawPointer(mutating: r.file.base + a)
            // The GGUFFile's own mapping: no views exist yet, so nothing pins the pages.
            precondition(msync(p, n, MS_INVALIDATE) == 0, "msync")
        }
    }
    let wiredLimit = { () -> Int in var v = 0; var n = MemoryLayout<Int>.size; sysctlbyname("iogpu.wired_limit_mb", &v, &n, nil, 0); return v }()
    print(String(format: "qfn-residency %@: %@, share %.2f GB in %d ranges, %d passes, scratch %.1f GB, iogpu.wired_limit_mb %d",
                 shape, label, Double(shareBytes) / 1e9, all.count, passes, scratchGB, wiredLimit))

    // Scratch: anonymous memory the process keeps for the whole run (touched once).
    let scratchBytes = Int(scratchGB * 1e9)
    let scratch = scratchBytes > 0 ? UnsafeMutableRawPointer.allocate(byteCount: scratchBytes, alignment: page) : nil
    if let scratch { for o in stride(from: 0, to: scratchBytes, by: page) { scratch.storeBytes(of: UInt8(1), toByteOffset: o, as: UInt8.self) } }
    defer { scratch?.deallocate() }

    let lib = try device.makeLibrary(source: """
    #include <metal_stdlib>
    using namespace metal;
    kernel void qfn_touch_pages(device const uint *p [[buffer(0)]], device atomic_uint *sink [[buffer(1)]],
                                constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
        if (gid < n && p[gid * 4096u] == 0xFFFFFFFFu) atomic_fetch_add_explicit(sink, 1u, memory_order_relaxed);
    }
    """, options: nil)
    let pso = try device.makeComputePipelineState(function: lib.makeFunction(name: "qfn_touch_pages")!)
    let sink = device.makeBuffer(length: 4, options: .storageModeShared)!

    invalidate()
    print(String(format: "  dropped: not in page cache %.2f GB of %.2f", Double(nonResident()) / 1e9, Double(shareBytes) / 1e9))
    for pass in 1...passes {
        let before = nonResident()
        let v0 = vmCounters()
        let t0 = CFAbsoluteTimeGetCurrent()
        var readS = 0.0
        for layer in share where !layer.isEmpty {
            let tr = CFAbsoluteTimeGetCurrent()
            GGUFFile.preadRanges(layer, threads: 4)
            readS += CFAbsoluteTimeGetCurrent() - tr
            autoreleasepool {
                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pso)
                enc.setBuffer(sink, offset: 0, index: 1)
                for r in layer {
                    let (v, _) = r.file.noCopyBuffer(device: device, offset: r.offset, byteCount: r.byteCount)!
                    var n = UInt32(v.length / page)
                    enc.setBuffer(v, offset: 0, index: 0)
                    enc.setBytes(&n, length: 4, index: 2)
                    enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                }
                enc.endEncoding()
                cb.commit(); cb.waitUntilCompleted()
            }
        }
        let wall = CFAbsoluteTimeGetCurrent() - t0
        let v1 = vmCounters()
        print(String(format: "  pass %d: not in cache before %.2f GB, read %.2f s, pass %.2f s, Pageins +%.2f GB, Swapouts +%llu pages, free %.2f GB wired %.2f GB file-backed %.2f GB",
                     pass, Double(before) / 1e9, readS, wall, Double(v1.pageins - v0.pageins) * Double(page) / 1e9,
                     v1.swapouts - v0.swapouts, Double(v1.free) * Double(page) / 1e9, Double(v1.wired) * Double(page) / 1e9,
                     Double(v1.file) * Double(page) / 1e9))
    }
}

// MARK: - `--qfn-residency-owned <gguf> <down sidecar> <routes prefix> layers|experts <share GB> <resident GB> [passes] [scratch GB]`
//
// F4 again without the page cache (docs/qfn-prefill/17): the M6 runs inference only, so its share need not stay
// evictable. The share's layers are taken in order while their bytes fit in <resident GB>; each of those layers is read
// once with F_NOCACHE `pread` (4 threads) into its own MTLBuffer, and the buffers are kept GPU-resident in one
// MTLResidencySet. The other layers are read on every pass with F_NOCACHE into one fixed staging buffer (the largest
// layer). Each pass goes layer by layer: (streamed layers) read, then a GPU kernel reads one word of every page.
// Read ranges are dropped from the page cache right after their read (msync MS_INVALIDATE; its seconds are printed and
// not counted in the pass), so no second copy stays in the cache and every streamed read comes from the SSD.
// Prints the fill, then per pass the streamed read seconds, the pass seconds and the Pageins / Swapouts deltas.

/// `pread` of `ranges` with F_NOCACHE into `dst`, packed in order, on `threads` threads in 16 MB pieces.
private func preadNoCache(_ ranges: [QFNRange], fds: [ObjectIdentifier: Int32], into dst: UnsafeMutableRawPointer, threads: Int) {
    var pieces: [(Int32, Int, Int, Int)] = []   // fd, file offset, bytes, dst offset
    var at = 0
    for r in ranges {
        var o = 0
        while o < r.byteCount {
            let n = min(16 << 20, r.byteCount - o)
            pieces.append((fds[ObjectIdentifier(r.file)]!, r.offset + o, n, at + o))
            o += n
        }
        at += r.byteCount
    }
    let next = UnsafeMutablePointer<Int>.allocate(capacity: 1)
    next.initialize(to: 0)
    defer { next.deallocate() }
    let lock = NSLock()
    DispatchQueue.concurrentPerform(iterations: threads) { _ in
        while true {
            lock.lock(); let i = next.pointee; next.pointee += 1; lock.unlock()
            guard i < pieces.count else { return }
            let (fd, off, n, d) = pieces[i]
            var done = 0
            while done < n {
                let got = pread(fd, dst + d + done, n - done, off_t(off + done))
                precondition(got > 0, "pread")
                done += got
            }
        }
    }
}

/// Drops `ranges` from the page cache through the GGUFFile's own mapping (no views of it exist in this probe).
private func dropFromCache(_ ranges: [QFNRange]) {
    let page = Int(getpagesize())
    for r in ranges {
        let a = r.offset / page * page, n = r.byteCount + (r.offset - a)
        precondition(msync(UnsafeMutableRawPointer(mutating: r.file.base + a), n, MS_INVALIDATE) == 0, "msync")
    }
}

func runQFNResidencyOwned(gguf: String, sidecar: String, routes: String, shape: String, targetGB: Double,
                          residentGB: Double, passes: Int, scratchGB: Double) throws {
    let (file, side, share, label) = try qfnShare(gguf: gguf, sidecar: sidecar, routes: routes, shape: shape, targetGB: targetGB)
    let device = MTLCreateSystemDefaultDevice()!, queue = device.makeCommandQueue()!
    let page = Int(getpagesize())
    let layerBytes = share.map { $0.reduce(0) { $0 + $1.byteCount } }
    var nResident = 0, residentBytes = 0
    while nResident < share.count && residentBytes + layerBytes[nResident] <= Int(residentGB * 1e9) {
        residentBytes += layerBytes[nResident]; nResident += 1
    }
    let streamed = Array(nResident..<share.count).filter { layerBytes[$0] > 0 }
    let streamedBytes = streamed.reduce(0) { $0 + layerBytes[$1] }
    let wiredLimit = { () -> Int in var v = 0; var n = MemoryLayout<Int>.size; sysctlbyname("iogpu.wired_limit_mb", &v, &n, nil, 0); return v }()
    print(String(format: "qfn-residency-owned %@: %@, share %.2f GB; resident %d layers %.2f GB, streamed %d layers %.2f GB; %d passes, scratch %.1f GB, iogpu.wired_limit_mb %d, recommendedMaxWorkingSetSize %.2f GB",
                 shape, label, Double(residentBytes + streamedBytes) / 1e9, nResident, Double(residentBytes) / 1e9,
                 streamed.count, Double(streamedBytes) / 1e9, passes, scratchGB, wiredLimit,
                 Double(device.recommendedMaxWorkingSetSize) / 1e9))

    var fds: [ObjectIdentifier: Int32] = [:]
    for (f, path) in [(file, gguf), (side, sidecar)] {
        let fd = open((path as NSString).expandingTildeInPath, O_RDONLY)
        precondition(fd >= 0 && fcntl(fd, F_NOCACHE, 1) == 0, "open \(path)")
        fds[ObjectIdentifier(f)] = fd
    }
    defer { for fd in fds.values { close(fd) } }

    let scratchBytes = Int(scratchGB * 1e9)
    let scratch = scratchBytes > 0 ? UnsafeMutableRawPointer.allocate(byteCount: scratchBytes, alignment: page) : nil
    if let scratch { for o in stride(from: 0, to: scratchBytes, by: page) { scratch.storeBytes(of: UInt8(1), toByteOffset: o, as: UInt8.self) } }
    defer { scratch?.deallocate() }

    let lib = try device.makeLibrary(source: """
    #include <metal_stdlib>
    using namespace metal;
    kernel void qfn_touch_pages(device const uint *p [[buffer(0)]], device atomic_uint *sink [[buffer(1)]],
                                constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
        if (gid < n && p[gid * 4096u] == 0xFFFFFFFFu) atomic_fetch_add_explicit(sink, 1u, memory_order_relaxed);
    }
    """, options: nil)
    let pso = try device.makeComputePipelineState(function: lib.makeFunction(name: "qfn_touch_pages")!)
    let sink = device.makeBuffer(length: 4, options: .storageModeShared)!
    func touch(_ b: MTLBuffer, bytes: Int) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        var n = UInt32((bytes + page - 1) / page)
        enc.setBuffer(b, offset: 0, index: 0); enc.setBuffer(sink, offset: 0, index: 1); enc.setBytes(&n, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
    }
    func mem() -> String {
        let v = vmCounters()
        return String(format: "free %.2f GB wired %.2f GB file-backed %.2f GB", Double(v.free) * Double(page) / 1e9,
                      Double(v.wired) * Double(page) / 1e9, Double(v.file) * Double(page) / 1e9)
    }

    // Fill the resident layers. The whole share is dropped from the page cache first, and every read range again right
    // after its read: with the GGUF also mapped, F_NOCACHE reads were found to stay in the cache (MBP trial).
    dropFromCache(share.flatMap { $0 })
    let nr = share.flatMap { $0 }.reduce(0) { $0 + $1.file.nonResidentBytes(offset: $1.offset, byteCount: $1.byteCount) }
    print(String(format: "  dropped: not in page cache %.2f GB", Double(nr) / 1e9))
    let v0 = vmCounters()
    let t0 = CFAbsoluteTimeGetCurrent()
    var owned: [MTLBuffer] = []
    for il in 0..<nResident where layerBytes[il] > 0 {
        let b = device.makeBuffer(length: (layerBytes[il] + page - 1) / page * page, options: .storageModeShared)!
        preadNoCache(share[il], fds: fds, into: b.contents(), threads: 4)
        dropFromCache(share[il])
        owned.append(b)
    }
    let fillS = CFAbsoluteTimeGetCurrent() - t0
    let setDesc = MTLResidencySetDescriptor()
    setDesc.label = "qfn-share"
    let set = try device.makeResidencySet(descriptor: setDesc)
    for b in owned { set.addAllocation(b) }
    set.commit()
    set.requestResidency()
    queue.addResidencySet(set)
    let v1 = vmCounters()
    print(String(format: "  fill: %.2f GB in %.2f s, Pageins +%.2f GB, Swapouts +%llu pages, allocated %.2f GB (residency set), %@",
                 Double(residentBytes) / 1e9, fillS, Double(v1.pageins - v0.pageins) * Double(page) / 1e9, v1.swapouts - v0.swapouts,
                 Double(set.allocatedSize) / 1e9, mem()))

    let staging = device.makeBuffer(length: max(((streamed.map { layerBytes[$0] }.max() ?? 0) + page - 1) / page * page, page),
                                    options: .storageModeShared)!
    for pass in 1...passes {
        let a = vmCounters()
        let tp = CFAbsoluteTimeGetCurrent()
        var readS = 0.0, dropS = 0.0
        var oi = 0
        for il in 0..<share.count where layerBytes[il] > 0 {
            if il < nResident {
                touch(owned[oi], bytes: layerBytes[il]); oi += 1
            } else {
                let tr = CFAbsoluteTimeGetCurrent()
                preadNoCache(share[il], fds: fds, into: staging.contents(), threads: 4)
                readS += CFAbsoluteTimeGetCurrent() - tr
                touch(staging, bytes: layerBytes[il])
                let td = CFAbsoluteTimeGetCurrent()
                dropFromCache(share[il])
                dropS += CFAbsoluteTimeGetCurrent() - td
            }
        }
        let wall = CFAbsoluteTimeGetCurrent() - tp - dropS
        let b = vmCounters()
        print(String(format: "  pass %d: streamed read %.2f s, pass %.2f s (drop %.2f s not counted), Pageins +%.2f GB, Swapouts +%llu pages, %@",
                     pass, readS, wall, dropS, Double(b.pageins - a.pageins) * Double(page) / 1e9, b.swapouts - a.swapouts, mem()))
    }
    queue.removeResidencySet(set)
}
