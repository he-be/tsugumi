import Darwin
import Foundation

// Two machines split the prefill by layers (`docs/qfn-prefill/22` §2, split A of `08` §1).
//
// The MBP owns every piece of state. A prefill round sends the M6 the prompt, the chunk cut, the recurrent state of
// layers 0 ..< k (GDN state, conv history, the PLE history when the PLE layer is among them) and the K / V and indexer
// rows of those layers it lacks (decode rows since the last round, or everything after a new connection). The M6 runs
// layers 0 ..< k of every chunk and answers each with the residual rows (T x hc x e floats), the chunk's K / V and
// indexer rows of its layers, and, after the chunks the MBP marks (checkpoint cuts and the last), the recurrent state
// of its layers; for a checkpoint cut inside a chunk (`Qwen38Runner.cutRows`, docs/qfn-prefill/29 C1) the state of
// its layers at that row too. The MBP runs layers k ..< 48, the head and the MTP head, and decodes alone.
//
// On connecting, the M6 says how it holds its layers (`Qwen38LayerStore`: resident layers, staging slabs, the feed
// fraction) and the sizes of its weight files, which must be the MBP's. With a feed fraction the MBP opens a second
// connection (port + 1) and serves the M6's piece lists from its own SSD (F_NOCACHE, 16 MB blocks on 4 threads).
//
// Raw TCP on bridge0 (link-local); both ends are arm64 (native byte order). No authentication: bind only the
// link-local address.

private let pairMagic: Int64 = 0x5146_4E50   // "QFNP"
private let pairVersion: Int64 = 3
/// Doubles per chunk answer: the M6's forward and profile, then its layer store's (`Qwen38LayerStore.Stats`).
private let pairTimes = 14

private func pairFileSize(_ f: GGUFFile) -> Int64 {
    var st = stat()
    return stat(f.url.path, &st) == 0 ? Int64(st.st_size) : -1
}

private func pairConnect(_ host: String, _ port: Int) throws -> Int32 {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    guard fd >= 0 else { throw Qwen38Pair.Error.link("socket: \(String(cString: strerror(errno)))") }
    pairTune(fd)
    var a = try pairAddress(host, port)
    let r = withUnsafePointer(to: &a) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    guard r == 0 else {
        let why = String(cString: strerror(errno))
        close(fd)
        throw Qwen38Pair.Error.link("connect \(host):\(port): \(why)")
    }
    return fd
}

private func pairListen(_ host: String, _ port: Int) throws -> Int32 {
    let ls = socket(AF_INET, SOCK_STREAM, 0)
    var one: Int32 = 1
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, 4)
    var a = try pairAddress(host, port)
    guard withUnsafePointer(to: &a, { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
        bind(ls, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }) == 0, listen(ls, 1) == 0 else {
        throw Qwen38Pair.Error.link("bind/listen \(host):\(port): \(String(cString: strerror(errno)))")
    }
    return ls
}

/// Blocking whole-buffer I/O on a socket.
private func pairIO(_ fd: Int32, _ p: UnsafeRawPointer, _ n: Int, send: Bool) throws {
    var done = 0
    while done < n {
        let r = send ? Darwin.send(fd, p + done, min(n - done, 64 << 20), 0)
                     : Darwin.recv(fd, UnsafeMutableRawPointer(mutating: p + done), min(n - done, 64 << 20), 0)
        if r < 0 && (errno == EINTR || errno == EAGAIN) { continue }
        guard r > 0 else { throw Qwen38Pair.Error.link(send ? "send: \(r == 0 ? "closed" : String(cString: strerror(errno)))"
                                                             : "recv: \(r == 0 ? "closed" : String(cString: strerror(errno)))") }
        done += r
    }
}

private func pairSend<T>(_ fd: Int32, _ values: [T]) throws {
    try values.withUnsafeBytes { try pairIO(fd, $0.baseAddress!, $0.count, send: true) }
}

private func pairRecv<T>(_ fd: Int32, _ count: Int, as: T.Type) throws -> [T] {
    guard count > 0 else { return [] }
    return try [T](unsafeUninitializedCapacity: count) { b, n in
        try pairIO(fd, UnsafeRawPointer(b.baseAddress!), count * MemoryLayout<T>.stride, send: false)
        n = count
    }
}

private func pairTune(_ fd: Int32) {
    var one: Int32 = 1, buf: Int32 = 8 << 20
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, 4)
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, 4)
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, 4)
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, 4)
}

private func pairAddress(_ host: String, _ port: Int) throws -> sockaddr_in {
    var a = sockaddr_in()
    a.sin_family = sa_family_t(AF_INET)
    a.sin_port = in_port_t(UInt16(port).bigEndian)
    guard inet_pton(AF_INET, host, &a.sin_addr) == 1 else { throw Qwen38Pair.Error.link("bad address \(host)") }
    return a
}

/// Every region `visit` names, sent or received in order.
private func pairRegions(_ fd: Int32, send: Bool, _ visit: ((UnsafeMutableRawPointer, Int) throws -> Void) throws -> Void) throws {
    try visit { p, n in try pairIO(fd, p, n, send: send) }
}

package enum Qwen38Pair {
    package enum Error: Swift.Error, CustomStringConvertible {
        case link(String)
        package var description: String {
            switch self { case .link(let s): return "pair link: \(s)" }
        }
    }

    /// `Q38_PAIR=<M6 bridge0 IPv4>[:port]` and `Q38_PAIR_K` (layers the M6 runs, default 46: the lowest median of the k grid 44 / 46 / 48, docs/qfn-prefill/27). Unset: the MBP alone.
    package static func configured() -> (host: String, port: Int, k: Int)? {
        let env = ProcessInfo.processInfo.environment
        guard let v = env["Q38_PAIR"], !v.isEmpty, v != "0" else { return nil }
        let parts = v.split(separator: ":")
        let port = parts.count > 1 ? Int(parts[1]) ?? defaultPort : defaultPort
        let k = Int(env["Q38_PAIR_K"] ?? "") ?? 46
        return (String(parts[0]), port, k)
    }
    package static let defaultPort = 5310

    /// One chunk's answer from the M6.
    package struct Chunk {
        package var index = 0, T = 0
        /// T x hc x e floats: the residual after layer k - 1.
        package var residual: [Float] = []
        /// The chunk's rows of layers 0 ..< k (`forEachKVRegion` order).
        package var kv: [UInt8] = []
        /// The recurrent state of layers 0 ..< k after the chunk (`forEachRecurrentRegion(layers:)` order), if asked.
        package var state: [UInt8]?
        /// The same at each of the chunk's cut rows, in order.
        package var cutStates: [[UInt8]] = []
        package var plePrev: [Int] = []
        /// The M6's forward seconds, its profile (ple, preRouter, route, routed, preGPU, routedGPU, routeAdvise) and its
        /// layer store's (slab wait, own read, fed, layer reads, own MB, fed MB) for the chunk.
        package var times: [Double] = []
        /// Seconds the MBP's receiver spent on this message after its header arrived.
        package var recvSeconds = 0.0
    }

    // MARK: - MBP

    /// The MBP's end: connects once, then one `begin` + `next` x chunks per prefill round. A receiver thread takes the
    /// M6's answers as they come, so the M6 never waits on the MBP's layers.
    package final class Client: @unchecked Sendable {
        package let k: Int
        private let fd: Int32
        private let lock = NSCondition()
        private var queue: [Chunk] = []
        private var failure: Swift.Error?
        private var expected = 0
        private var receiver: Thread?

        /// What the M6 said on connecting: resident layers, staging slabs, feed fraction.
        package let resident: Int, slabs: Int, feed: Double
        private var feedFD: Int32 = -1

        package init(host: String, port: Int, k: Int, runner: Qwen38Runner) throws {
            self.k = k
            let fd = try pairConnect(host, port)
            self.fd = fd
            do {
                let hello: [Int64] = try pairRecv(fd, 4, as: Int64.self)
                let sizes: [Int64] = try pairRecv(fd, Int(hello[3]), as: Int64.self)
                let mine = runner.weightFiles.map(pairFileSize)
                guard sizes == mine else { throw Error.link("weight files differ: M6 \(sizes), here \(mine)") }
                resident = Int(hello[0]); slabs = Int(hello[1]); feed = Double(hello[2]) / 1e6
                if feed > 0 {
                    let ffd = try pairConnect(host, port + 1)
                    feedFD = ffd
                    let paths = runner.weightFiles.map { $0.url.path }
                    let t = Thread { Qwen38Pair.feedServe(ffd, paths: paths) }
                    t.stackSize = 1 << 20
                    t.start()
                }
            } catch {
                close(fd)
                throw error
            }
        }

        deinit {
            close(fd)
            if feedFD >= 0 { shutdown(feedFD, SHUT_RDWR) }
        }

        /// Starts a round: the M6 runs layers 0 ..< k of `tokens` at `startPos` in chunks of `chunks`, and sends the
        /// recurrent state after the chunks marked in `stateAfter` and at each chunk's `cutRows`. `kvFrom`: the first
        /// position whose rows the M6 lacks (rows `kvFrom ..< startPos` go with the request).
        package func begin(runner: Qwen38Runner, tokens: [Int], startPos: Int, chunks: [Int], stateAfter: [Bool],
                           cutRows: [[Int]], kvFrom: Int) throws {
            precondition(chunks.reduce(0, +) == tokens.count && stateAfter.count == chunks.count
                         && cutRows.count == chunks.count && kvFrom <= startPos)
            let layers = 0..<k
            let header: [Int64] = [pairMagic, pairVersion, Int64(k), Int64(startPos), Int64(tokens.count),
                                   Int64(chunks.count), Int64(kvFrom), Int64(runner.recurrentBytes(layers: layers))]
            try pairSend(fd, header)
            try pairSend(fd, tokens.map { Int32($0) })
            try pairSend(fd, chunks.map { Int32($0) })
            try pairSend(fd, stateAfter.map { Int32($0 ? 1 : 0) })
            try pairSend(fd, cutRows.map { Int32($0.count) })
            try pairSend(fd, cutRows.flatMap { $0 }.map { Int32($0) })
            try pairSend(fd, runner.plePrevious.map { Int64($0) })
            try pairRegions(fd, send: true) { try runner.forEachRecurrentRegion(layers: layers, $0) }
            try pairRegions(fd, send: true) { try runner.forEachKVRegion(layers: layers, positions: kvFrom..<startPos, $0) }
            lock.lock()
            queue = []
            failure = nil
            expected = chunks.count
            lock.unlock()
            var chunkStart = startPos
            let kvBytes = chunks.map { T in
                defer { chunkStart += T }
                return runner.kvRegionBytes(layers: layers, positions: chunkStart..<(chunkStart + T))
            }
            let stateBytes = runner.recurrentBytes(layers: layers)
            let W = runner.hc * runner.e
            let t = Thread { [self] in
                do {
                    for (i, T) in chunks.enumerated() {
                        let head: [Int64] = try pairRecv(fd, 4, as: Int64.self)
                        let t0 = CFAbsoluteTimeGetCurrent()
                        guard head[0] == Int64(i), head[1] == Int64(T) else { throw Error.link("chunk \(i): got \(head)") }
                        var c = Chunk()
                        c.index = i; c.T = T
                        c.times = try pairRecv(fd, pairTimes, as: Double.self)
                        c.plePrev = try pairRecv(fd, 2, as: Int64.self).map { Int($0) }
                        c.residual = try pairRecv(fd, T * W, as: Float.self)
                        c.kv = try pairRecv(fd, kvBytes[i], as: UInt8.self)
                        if head[2] != 0 { c.state = try pairRecv(fd, stateBytes, as: UInt8.self) }
                        guard head[3] == Int64(cutRows[i].count) else { throw Error.link("chunk \(i): \(head[3]) cut states") }
                        for _ in 0..<cutRows[i].count { c.cutStates.append(try pairRecv(fd, stateBytes, as: UInt8.self)) }
                        c.recvSeconds = CFAbsoluteTimeGetCurrent() - t0
                        lock.lock(); queue.append(c); lock.signal(); lock.unlock()
                    }
                } catch {
                    lock.lock(); failure = error; lock.signal(); lock.unlock()
                }
            }
            t.stackSize = 1 << 20
            receiver = t
            t.start()
        }

        /// The next chunk's answer, in order; blocks until it is in. Returns the seconds waited too.
        package func next() throws -> (Chunk, Double) {
            let t0 = CFAbsoluteTimeGetCurrent()
            lock.lock()
            defer { lock.unlock() }
            while queue.isEmpty && failure == nil { lock.wait() }
            if let failure { throw failure }
            return (queue.removeFirst(), CFAbsoluteTimeGetCurrent() - t0)
        }

        /// Copies a chunk's rows and state into the MBP's runner (layers 0 ..< k). `chunkStart`: the chunk's first
        /// position.
        package func apply(_ c: Chunk, to runner: Qwen38Runner, chunkStart: Int) {
            var at = 0
            c.kv.withUnsafeBytes { src in
                runner.forEachKVRegion(layers: 0..<k, positions: chunkStart..<(chunkStart + c.T)) { p, n in
                    p.copyMemory(from: src.baseAddress! + at, byteCount: n)
                    at += n
                }
            }
            precondition(at == c.kv.count, "chunk \(c.index): KV \(at) of \(c.kv.count) bytes")
            if let s = c.state {
                at = 0
                s.withUnsafeBytes { src in
                    runner.forEachRecurrentRegion(layers: 0..<k) { p, n in
                        p.copyMemory(from: src.baseAddress! + at, byteCount: n)
                        at += n
                    }
                }
                precondition(at == s.count)
            }
            for (j, s) in c.cutStates.enumerated() {
                at = 0
                s.withUnsafeBytes { src in
                    runner.forEachCutRegion(j, layers: 0..<k) { p, n in
                        p.copyMemory(from: src.baseAddress! + at, byteCount: n)
                        at += n
                    }
                }
                precondition(at == s.count)
            }
        }
    }

    /// The MBP's side of the feed link: answers the M6's piece lists (count, then file index, offset, bytes each) until
    /// the link closes. Blocks of 16 MB are read by 4 threads (F_NOCACHE) and sent in order as each completes.
    fileprivate static func feedServe(_ fd: Int32, paths: [String]) {
        let fds = paths.map { path -> Int32 in
            let f = open(path, O_RDONLY)
            _ = fcntl(f, F_NOCACHE, 1)
            return f
        }
        defer { for f in fds { close(f) }; close(fd) }
        var cap = 0
        var buf: UnsafeMutableRawPointer?
        defer { buf?.deallocate() }
        func note(_ why: String) { FileHandle.standardError.write("Qwen3.8 pair feed: \(why)\n".data(using: .utf8)!) }
        if let bad = fds.firstIndex(where: { $0 < 0 }) { note("cannot open \(paths[bad])"); return }
        while true {
            let n: Int64, list: [Int64]
            // The M6 closes the link between requests when its connection ends: the normal end, said nothing about.
            guard let head = try? pairRecv(fd, 1, as: Int64.self) else { return }
            n = head[0]
            guard n >= 0 else { note("request \(n)"); return }
            do {
                list = try pairRecv(fd, Int(n) * 3, as: Int64.self)
            } catch {
                note("\(error)")
                return
            }
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
                        while done < len {
                            let got = pread(f, dst + d + done, len - done, off_t(off + done))
                            precondition(got > 0, "feed pread: errno \(errno)")
                            done += got
                        }
                        ready[i].signal()
                    }
                }
            }
            var ok = true
            for (i, b) in blocks.enumerated() {
                ready[i].wait()
                if ok { ok = (try? pairIO(fd, dst + b.3, b.2, send: true)) != nil }
            }
            readers.wait()
            next.deallocate()
            if !ok { note("send failed"); return }
        }
    }

    // MARK: - M6

    /// The M6's end: serves rounds on one connection at a time with its own runner, layers 0 ..< k (k from the request).
    /// `log` gets a line per chunk.
    /// `store`: the M6's own buffers for its layers (nil: the runner's mapped files). The feed link (port + 1) is
    /// accepted after the main one when the store has a feed fraction.
    package static func serve(runner: Qwen38Runner, store: Qwen38LayerStore?, host: String, port: Int,
                              log: (String) -> Void) throws -> Never {
        let ls = try pairListen(host, port)
        let feedLS = (store?.feed ?? 0) > 0 ? try pairListen(host, port + 1) : -1
        log("listening on \(host):\(port)" + (feedLS >= 0 ? " (feed \(port + 1))" : ""))
        while true {
            let fd = accept(ls, nil, nil)
            guard fd >= 0 else { continue }
            pairTune(fd)
            var feedFD: Int32 = -1
            do {
                let files = runner.weightFiles.map(pairFileSize)
                try pairSend(fd, [Int64(store?.resident ?? 0), Int64(store?.slabs ?? 0), Int64(((store?.feed ?? 0) * 1e6).rounded()),
                                  Int64(files.count)] + files)
                if feedLS >= 0 {
                    feedFD = accept(feedLS, nil, nil)
                    pairTune(feedFD)
                }
                log("connected" + (feedFD >= 0 ? " (feed)" : ""))
                while true {
                    // A link that ends between rounds is the MBP leaving; one that ends inside a round leaves the store
                    // mid-read, and the process ends (serveRound).
                    guard let h = try? pairRecv(fd, 8, as: Int64.self) else { break }
                    try serveRound(runner: runner, store: store, fd: fd, feedFD: feedFD, header: h, log: log)
                }
                log("connection ended")
            } catch {
                log("round failed: \(error)")
                exit(1)
            }
            close(fd)
            if feedFD >= 0 { close(feedFD) }
        }
    }

    private static func serveRound(runner: Qwen38Runner, store: Qwen38LayerStore?, fd: Int32, feedFD: Int32,
                                   header h: [Int64], log: (String) -> Void) throws {
        guard h[0] == pairMagic, h[1] == pairVersion else { throw Error.link("bad header \(h)") }
        let k = Int(h[2]), startPos = Int(h[3]), n = Int(h[4]), nChunks = Int(h[5]), kvFrom = Int(h[6])
        guard k >= 1 && k <= runner.nTrunk, startPos + n <= runner.capacity, kvFrom <= startPos else {
            throw Error.link("round k \(k) at \(startPos) + \(n) (capacity \(runner.capacity)), kv from \(kvFrom)")
        }
        let layers = 0..<k
        guard Int(h[7]) == runner.recurrentBytes(layers: layers) else { throw Error.link("recurrent bytes \(h[7])") }
        let t0 = CFAbsoluteTimeGetCurrent()
        let tokens = try pairRecv(fd, n, as: Int32.self).map { Int($0) }
        let chunks = try pairRecv(fd, nChunks, as: Int32.self).map { Int($0) }
        let stateAfter = try pairRecv(fd, nChunks, as: Int32.self).map { $0 != 0 }
        let cutCounts = try pairRecv(fd, nChunks, as: Int32.self).map { Int($0) }
        var cutFlat = try pairRecv(fd, cutCounts.reduce(0, +), as: Int32.self).map { Int($0) }[...]
        let cutRows = cutCounts.map { n in defer { cutFlat = cutFlat.dropFirst(n) }; return Array(cutFlat.prefix(n)) }
        runner.plePrevious = try pairRecv(fd, 2, as: Int64.self).map { Int($0) }
        try pairRegions(fd, send: false) { try runner.forEachRecurrentRegion(layers: layers, $0) }
        try pairRegions(fd, send: false) { try runner.forEachKVRegion(layers: layers, positions: kvFrom..<startPos, $0) }
        let tIn = CFAbsoluteTimeGetCurrent() - t0
        log(String(format: "round k %d at %d: %d tokens in %d chunks, kv rows %d, request %.3f s", k, startPos, n, nChunks,
                   startPos - kvFrom, tIn))
        if runner.plePrefetch && k > runner.pleLayerIndex { try runner.prefetchPLE(tokens: tokens) }
        store?.beginRound(chunks: nChunks, k: k, feed: feedFD)
        var last = Qwen38LayerStore.Stats()
        runner.layerRange = layers
        runner.trunkOnly = true
        defer { runner.layerRange = nil; runner.trunkOnly = false }
        let W = runner.hc * runner.e
        var pos = startPos, done = 0
        for (i, T) in chunks.enumerated() {
            let tf = CFAbsoluteTimeGetCurrent()
            runner.cutRows = cutRows[i]
            defer { runner.cutRows = []; runner.releaseCutBuffers() }
            _ = try runner.forward(tokens: Array(tokens[done..<(done + T)]), startPos: pos)
            let forward = CFAbsoluteTimeGetCurrent() - tf
            let p = runner.lastProfile
            let ts = CFAbsoluteTimeGetCurrent()
            try pairSend(fd, [Int64(i), Int64(T), stateAfter[i] ? 1 : 0, Int64(cutRows[i].count)])
            let st = store?.stats ?? Qwen38LayerStore.Stats()
            let ls = [st.wait - last.wait, st.own - last.own, st.fed - last.fed, st.read - last.read,
                      Double(st.ownBytes - last.ownBytes) / 1e6, Double(st.fedBytes - last.fedBytes) / 1e6]
            last = st
            try pairSend(fd, [forward, p.ple, p.preRouter, p.route, p.routed, p.preGPU, p.routedGPU, p.routeAdvise] + ls)
            try pairSend(fd, runner.plePrevious.map { Int64($0) })
            try pairIO(fd, runner.hidden(row: 0), T * W * 4, send: true)
            try pairRegions(fd, send: true) { try runner.forEachKVRegion(layers: layers, positions: pos..<(pos + T), $0) }
            if stateAfter[i] { try pairRegions(fd, send: true) { try runner.forEachRecurrentRegion(layers: layers, $0) } }
            for j in cutRows[i].indices { try pairRegions(fd, send: true) { try runner.forEachCutRegion(j, layers: layers, $0) } }
            log(String(format: "chunk %d at %d T %d: forward %.3f s (ple %.3f pre %.3f route %.3f advise %.3f routed %.3f, gpu %.3f + %.3f), send %.3f s",
                       i, pos, T, forward, p.ple, p.preRouter, p.route, p.routeAdvise, p.routed, p.preGPU, p.routedGPU,
                       CFAbsoluteTimeGetCurrent() - ts)
                + (store == nil ? "" : String(format: "; slabs: wait %.3f s, reads %.3f s (own %.3f s %.0f MB, fed %.3f s %.0f MB)",
                                              ls[0], ls[3], ls[1], ls[4], ls[2], ls[5])))
            pos += T
            done += T
        }
        // The round's scratch goes back to the decode size here (the MBP's first decode step does the same); a round
        // whose last chunk was 1-8 tokens (a checkpoint cut ending the chunk) had already shrunk it.
        runner.shrinkBatchAfterRound()
        let mem = runner.memoryReport().filter { $0.1 > 0 }.map { String(format: "%@ %.2f", $0.0, Double($0.1) / 1e9) }
        log("memory GB after the round: " + mem.joined(separator: ", "))
    }
}
