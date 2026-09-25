import Darwin
import Foundation

// Two machines split the prefill by layers (`docs/qfn-prefill/22` §2, split A of `08` §1).
//
// The MBP owns every piece of state. A prefill round sends the M6 the prompt, the chunk cut, the recurrent state of
// layers 0 ..< k (GDN state, conv history, the PLE history when the PLE layer is among them) and the K / V and indexer
// rows of those layers it lacks (decode rows since the last round, or everything after a new connection). The M6 runs
// layers 0 ..< k of every chunk and answers each with the residual rows (T x hc x e floats), the chunk's K / V and
// indexer rows of its layers, and, after the chunks the MBP marks (checkpoint cuts and the last), the recurrent state
// of its layers. The MBP runs layers k ..< 48, the head and the MTP head, and decodes alone.
//
// Raw TCP on bridge0 (link-local), one connection; both ends are arm64 (native byte order). No authentication: bind
// only the link-local address.

private let pairMagic: Int64 = 0x5146_4E50   // "QFNP"
private let pairVersion: Int64 = 1

/// Blocking whole-buffer I/O on a socket.
private func pairIO(_ fd: Int32, _ p: UnsafeRawPointer, _ n: Int, send: Bool) throws {
    var done = 0
    while done < n {
        let r = send ? Darwin.send(fd, p + done, min(n - done, 64 << 20), 0)
                     : Darwin.recv(fd, UnsafeMutableRawPointer(mutating: p + done), min(n - done, 64 << 20), 0)
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

    /// `Q38_PAIR=<M6 bridge0 IPv4>[:port]` and `Q38_PAIR_K` (layers the M6 runs, default 48). Unset: the MBP alone.
    package static func configured() -> (host: String, port: Int, k: Int)? {
        let env = ProcessInfo.processInfo.environment
        guard let v = env["Q38_PAIR"], !v.isEmpty, v != "0" else { return nil }
        let parts = v.split(separator: ":")
        let port = parts.count > 1 ? Int(parts[1]) ?? defaultPort : defaultPort
        let k = Int(env["Q38_PAIR_K"] ?? "") ?? 48
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
        package var plePrev: [Int] = []
        /// The M6's forward seconds and its profile (ple, preRouter, route, routed, preGPU, routedGPU).
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

        package init(host: String, port: Int, k: Int) throws {
            self.k = k
            let fd = socket(AF_INET, SOCK_STREAM, 0)
            guard fd >= 0 else { throw Error.link("socket: \(String(cString: strerror(errno)))") }
            pairTune(fd)
            var a = try pairAddress(host, port)
            let r = withUnsafePointer(to: &a) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
            }
            guard r == 0 else {
                let why = String(cString: strerror(errno))
                close(fd)
                throw Error.link("connect \(host):\(port): \(why)")
            }
            self.fd = fd
        }

        deinit { close(fd) }

        /// Starts a round: the M6 runs layers 0 ..< k of `tokens` at `startPos` in chunks of `chunks`, and sends the
        /// recurrent state after the chunks marked in `stateAfter`. `kvFrom`: the first position whose rows the M6
        /// lacks (rows `kvFrom ..< startPos` go with the request).
        package func begin(runner: Qwen38Runner, tokens: [Int], startPos: Int, chunks: [Int], stateAfter: [Bool],
                           kvFrom: Int) throws {
            precondition(chunks.reduce(0, +) == tokens.count && stateAfter.count == chunks.count && kvFrom <= startPos)
            let layers = 0..<k
            let header: [Int64] = [pairMagic, pairVersion, Int64(k), Int64(startPos), Int64(tokens.count),
                                   Int64(chunks.count), Int64(kvFrom), Int64(runner.recurrentBytes(layers: layers))]
            try pairSend(fd, header)
            try pairSend(fd, tokens.map { Int32($0) })
            try pairSend(fd, chunks.map { Int32($0) })
            try pairSend(fd, stateAfter.map { Int32($0 ? 1 : 0) })
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
                        c.times = try pairRecv(fd, 8, as: Double.self)
                        c.plePrev = try pairRecv(fd, 2, as: Int64.self).map { Int($0) }
                        c.residual = try pairRecv(fd, T * W, as: Float.self)
                        c.kv = try pairRecv(fd, kvBytes[i], as: UInt8.self)
                        if head[2] != 0 { c.state = try pairRecv(fd, stateBytes, as: UInt8.self) }
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
        }
    }

    // MARK: - M6

    /// The M6's end: serves rounds on one connection at a time with its own runner, layers 0 ..< k (k from the request).
    /// `log` gets a line per chunk.
    package static func serve(runner: Qwen38Runner, host: String, port: Int, log: (String) -> Void) throws -> Never {
        let ls = socket(AF_INET, SOCK_STREAM, 0)
        var one: Int32 = 1
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, 4)
        var a = try pairAddress(host, port)
        guard withUnsafePointer(to: &a, { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(ls, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }) == 0, listen(ls, 1) == 0 else {
            throw Error.link("bind/listen \(host):\(port): \(String(cString: strerror(errno)))")
        }
        log("listening on \(host):\(port)")
        while true {
            let fd = accept(ls, nil, nil)
            guard fd >= 0 else { continue }
            pairTune(fd)
            log("connected")
            do {
                while true { try serveRound(runner: runner, fd: fd, log: log) }
            } catch {
                log("connection ended: \(error)")
            }
            close(fd)
        }
    }

    private static func serveRound(runner: Qwen38Runner, fd: Int32, log: (String) -> Void) throws {
        let h: [Int64] = try pairRecv(fd, 8, as: Int64.self)
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
        runner.plePrevious = try pairRecv(fd, 2, as: Int64.self).map { Int($0) }
        try pairRegions(fd, send: false) { try runner.forEachRecurrentRegion(layers: layers, $0) }
        try pairRegions(fd, send: false) { try runner.forEachKVRegion(layers: layers, positions: kvFrom..<startPos, $0) }
        let tIn = CFAbsoluteTimeGetCurrent() - t0
        log(String(format: "round k %d at %d: %d tokens in %d chunks, kv rows %d, request %.3f s", k, startPos, n, nChunks,
                   startPos - kvFrom, tIn))
        if runner.plePrefetch && k > runner.pleLayerIndex { try runner.prefetchPLE(tokens: tokens) }
        runner.layerRange = layers
        runner.trunkOnly = true
        defer { runner.layerRange = nil; runner.trunkOnly = false }
        let W = runner.hc * runner.e
        var pos = startPos, done = 0
        for (i, T) in chunks.enumerated() {
            let tf = CFAbsoluteTimeGetCurrent()
            _ = try runner.forward(tokens: Array(tokens[done..<(done + T)]), startPos: pos)
            let forward = CFAbsoluteTimeGetCurrent() - tf
            let p = runner.lastProfile
            let ts = CFAbsoluteTimeGetCurrent()
            try pairSend(fd, [Int64(i), Int64(T), stateAfter[i] ? 1 : 0, 0])
            try pairSend(fd, [forward, p.ple, p.preRouter, p.route, p.routed, p.preGPU, p.routedGPU, p.routeAdvise])
            try pairSend(fd, runner.plePrevious.map { Int64($0) })
            try pairIO(fd, runner.hidden(row: 0), T * W * 4, send: true)
            try pairRegions(fd, send: true) { try runner.forEachKVRegion(layers: layers, positions: pos..<(pos + T), $0) }
            if stateAfter[i] { try pairRegions(fd, send: true) { try runner.forEachRecurrentRegion(layers: layers, $0) } }
            log(String(format: "chunk %d at %d T %d: forward %.3f s (ple %.3f pre %.3f route %.3f advise %.3f routed %.3f, gpu %.3f + %.3f), send %.3f s",
                       i, pos, T, forward, p.ple, p.preRouter, p.route, p.routeAdvise, p.routed, p.preGPU, p.routedGPU,
                       CFAbsoluteTimeGetCurrent() - ts))
            pos += T
            done += T
        }
    }
}
