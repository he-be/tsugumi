import Darwin
import Foundation
import Metal

/// The M6's trunk weights in its own buffers (`docs/qfn-prefill/22` §4, measured in 17, 20 and 21 as the simulation
/// `--qfn-pair-sim`): layers 0 ..< `resident` whole in buffers kept GPU-resident, and every later layer read for each
/// chunk into one of `slabs` staging buffers, the next layers read while the GPU runs this one. A streamed layer is read
/// whole (dense tensors and every expert): its selected experts are known only at its router, and a chunk of 2,048
/// selects 286-470 of 512 per layer (median 402, the F2 route dumps). With `feed` > 0 and a feed link, the last `feed`
/// fraction of each streamed layer's bytes comes from the MBP's SSD over that link (`Qwen38Pair`, Q4 arm b); the rest
/// is read here. Reads are `pread` with `F_NOCACHE`, 16 MB blocks on 4 threads; the file's pages are then dropped with
/// `msync(MS_INVALIDATE)` on the runner's mapping (the mapping keeps them cached otherwise: +8.5 GB file-backed and
/// swapping in the first resident fill without it, as the simulation's `simDrop`).
///
/// Installs itself as the runner's `layerWillRun` / `layerDidFinish`. A layer's slab is released when its last GPU
/// buffer completes, which with the runner's pipeline is after the next layer has begun: `slabs` must be 2 or more.
package final class Qwen38LayerStore: @unchecked Sendable {
    package typealias Piece = (file: GGUFFile, offset: Int, byteCount: Int)
    package let resident: Int, slabs: Int, feed: Double
    private let runner: Qwen38Runner
    private let pieces: [[Piece]]
    private let layouts: [[Int]]
    private var fds: [ObjectIdentifier: Int32] = [:]
    private var owned: [MTLBuffer] = []
    private var staging: [MTLBuffer] = []
    private let free: [DispatchSemaphore], ready: [DispatchSemaphore]
    private var consumed = 0
    private var slabOf: [Int: Int] = [:]
    private var feedFD: Int32 = -1
    private var reader: Thread?
    package private(set) var fillSeconds = 0.0
    package var residentBytes: Int { owned.reduce(0) { $0 + $1.length } }
    package var stagingBytes: Int { staging.reduce(0) { $0 + $1.length } }

    /// Seconds since the round began: the runner waiting for a slab, and the reader's own reads, fed receives and
    /// whole layer reads (summed over layers).
    package struct Stats { package var wait = 0.0, own = 0.0, fed = 0.0, read = 0.0, ownBytes = 0, fedBytes = 0 }
    private let statsLock = NSLock()
    private var _stats = Stats()
    package var stats: Stats { statsLock.lock(); defer { statsLock.unlock() }; return _stats }

    package init(runner: Qwen38Runner, resident: Int, slabs: Int, feed: Double) throws {
        precondition(slabs >= 2, "the pipeline releases a slab after the next layer began: 2 slabs or more")
        self.runner = runner
        self.resident = min(resident, runner.nTrunk)
        self.slabs = slabs
        self.feed = feed
        let page = 16384
        var pieces: [[Piece]] = [], layouts: [[Int]] = []
        for il in 0..<runner.nTrunk {
            let p = try runner.layerPieces(il)
            var at: [Int] = [], o = 0
            for x in p { at.append(o); o += (x.byteCount + page - 1) / page * page }
            pieces.append(p)
            layouts.append(at)
        }
        self.pieces = pieces
        self.layouts = layouts
        free = (0..<slabs).map { _ in DispatchSemaphore(value: 1) }
        ready = (0..<slabs).map { _ in DispatchSemaphore(value: 0) }
        for f in runner.weightFiles {
            let fd = open(f.url.path, O_RDONLY)
            guard fd >= 0, fcntl(fd, F_NOCACHE, 1) == 0 else { throw GGUFFile.Error.open("\(f.url.path): errno \(errno)") }
            fds[ObjectIdentifier(f)] = fd
        }
        let device = runner.device
        let t0 = CFAbsoluteTimeGetCurrent()
        for il in 0..<self.resident {
            guard let b = device.makeBuffer(length: layoutBytes(il), options: .storageModeShared) else {
                throw GGUFFile.Error.format("layer \(il): no buffer of \(layoutBytes(il)) bytes")
            }
            Self.drop(pieces[il])
            Self.pread(pieces[il], at: layouts[il], fds: fds, into: b.contents())
            Self.drop(pieces[il])
            try runner.install(layer: il, buffer: b, at: layouts[il])
            owned.append(b)
        }
        fillSeconds = CFAbsoluteTimeGetCurrent() - t0
        let slabBytes = (self.resident..<runner.nTrunk).map(layoutBytes).max() ?? 0
        if slabBytes > 0 {
            for _ in 0..<slabs {
                guard let b = device.makeBuffer(length: slabBytes, options: .storageModeShared) else {
                    throw GGUFFile.Error.format("no staging buffer of \(slabBytes) bytes")
                }
                staging.append(b)
            }
        }
        try runner.makeResident(owned + staging)
        runner.layerWillRun = { [unowned self] il in self.willRun(il) }
        runner.layerDidFinish = { [unowned self] il in self.didFinish(il) }
    }

    deinit { for fd in fds.values { close(fd) } }

    private func layoutBytes(_ il: Int) -> Int { layouts[il].last! + (pieces[il].last!.byteCount + 16383) / 16384 * 16384 }

    /// Starts reading the streamed layers of a round: `chunks` forwards of layers 0 ..< `k`. `feed`: the feed link's
    /// socket, or -1 to read everything here.
    package func beginRound(chunks: Int, k: Int, feed fd: Int32) {
        precondition(reader == nil || reader!.isFinished, "a round is still reading")
        consumed = 0
        feedFD = fd
        statsLock.lock(); _stats = Stats(); statsLock.unlock()
        let layers = Array(resident..<max(resident, k))
        guard !layers.isEmpty else { reader = nil; return }
        let t = Thread { [self] in
            var i = 0
            for _ in 0..<chunks {
                for il in layers {
                    let s = i % slabs
                    free[s].wait()
                    read(il, into: staging[s])
                    ready[s].signal()
                    i += 1
                }
            }
        }
        t.stackSize = 1 << 20
        reader = t
        t.start()
    }

    private func willRun(_ il: Int) {
        guard il >= resident else { return }
        let s = consumed % slabs
        let t0 = CFAbsoluteTimeGetCurrent()
        ready[s].wait()
        let waited = CFAbsoluteTimeGetCurrent() - t0
        statsLock.lock(); _stats.wait += waited; statsLock.unlock()
        try! runner.install(layer: il, buffer: staging[s], at: layouts[il])
        slabOf[il] = s
        consumed += 1
    }

    private func didFinish(_ il: Int) {
        guard let s = slabOf.removeValue(forKey: il) else { return }
        runner.uninstall(layer: il)
        free[s].signal()
    }

    /// Layer `il` into `slab`: the first 1 - `feed` of its bytes (in piece order) here, the rest from the feed link.
    private func read(_ il: Int, into slab: MTLBuffer) {
        let p = pieces[il], at = layouts[il]
        let total = p.reduce(0) { $0 + $1.byteCount }
        let cut = feedFD >= 0 && feed > 0 ? Int(Double(total) * (1 - feed)) : total
        var own: [Piece] = [], ownAt: [Int] = [], fed: [Piece] = [], fedAt: [Int] = []
        var o = 0
        for (i, x) in p.enumerated() {
            let a = max(0, min(x.byteCount, cut - o))
            if a > 0 { own.append((x.file, x.offset, a)); ownAt.append(at[i]) }
            if a < x.byteCount { fed.append((x.file, x.offset + a, x.byteCount - a)); fedAt.append(at[i] + a) }
            o += x.byteCount
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        let dst = slab.contents()
        var fedSeconds = 0.0
        let group = DispatchGroup()
        if !fed.isEmpty {
            let files = runner.weightFiles
            var req = [Int64(fed.count)]
            for x in fed { req += [Int64(files.firstIndex { $0 === x.file }!), Int64(x.offset), Int64(x.byteCount)] }
            let fd = feedFD
            req.withUnsafeBytes { Self.io(fd, UnsafeMutableRawPointer(mutating: $0.baseAddress!), $0.count, send: true) }
            DispatchQueue.global().async(group: group) {
                for (i, x) in fed.enumerated() { Self.io(fd, dst + fedAt[i], x.byteCount, send: false) }
                fedSeconds = CFAbsoluteTimeGetCurrent() - t0
            }
        }
        Self.pread(own, at: ownAt, fds: fds, into: dst)
        let ownSeconds = CFAbsoluteTimeGetCurrent() - t0
        Self.drop(own)
        group.wait()
        let wall = CFAbsoluteTimeGetCurrent() - t0
        statsLock.lock()
        _stats.own += ownSeconds; _stats.fed += fedSeconds; _stats.read += wall
        _stats.ownBytes += own.reduce(0) { $0 + $1.byteCount }
        _stats.fedBytes += fed.reduce(0) { $0 + $1.byteCount }
        statsLock.unlock()
    }

    /// Whole-buffer socket I/O for the feed link; a broken link ends the process (the round cannot finish).
    private static func io(_ fd: Int32, _ p: UnsafeMutableRawPointer, _ n: Int, send: Bool) {
        var done = 0
        while done < n {
            let r = send ? Darwin.send(fd, p + done, min(n - done, 16 << 20), 0) : Darwin.recv(fd, p + done, min(n - done, 16 << 20), 0)
            if r <= 0 {
                FileHandle.standardError.write("Qwen38LayerStore: feed link \(send ? "send" : "recv") failed (\(r), errno \(errno))\n".data(using: .utf8)!)
                exit(1)
            }
            done += r
        }
    }

    /// Drops `pieces` from the page cache through the file's mapping (no GPU view of these ranges exists).
    package static func drop(_ pieces: [Piece]) {
        let page = Int(getpagesize())
        for r in pieces {
            let a = r.offset / page * page
            _ = msync(UnsafeMutableRawPointer(mutating: r.file.base + a), r.byteCount + (r.offset - a), MS_INVALIDATE)
        }
    }

    /// `pieces[i]` into `dst + at[i]` with `pread` in 16 MB blocks on 4 threads (the descriptors carry F_NOCACHE).
    package static func pread(_ pieces: [Piece], at: [Int], fds: [ObjectIdentifier: Int32], into dst: UnsafeMutableRawPointer) {
        var blocks: [(Int32, Int, Int, Int)] = []
        for (i, r) in pieces.enumerated() {
            var o = 0
            while o < r.byteCount {
                let n = min(16 << 20, r.byteCount - o)
                blocks.append((fds[ObjectIdentifier(r.file)]!, r.offset + o, n, at[i] + o))
                o += n
            }
        }
        let lock = NSLock()
        let next = UnsafeMutablePointer<Int>.allocate(capacity: 1)
        next.initialize(to: 0)
        defer { next.deallocate() }
        DispatchQueue.concurrentPerform(iterations: 4) { _ in
            while true {
                lock.lock(); let i = next.pointee; next.pointee += 1; lock.unlock()
                guard i < blocks.count else { return }
                let (fd, off, n, d) = blocks[i]
                var done = 0
                while done < n {
                    let got = Darwin.pread(fd, dst + d + done, n - done, off_t(off + done))
                    precondition(got > 0, "pread: errno \(errno)")
                    done += got
                }
            }
        }
    }
}
