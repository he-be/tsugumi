import Foundation
import Tsugumi

// MARK: - Qwen3.8 decode split by layers over two machines (docs/qfn-prefill/23)
//
// `--qwen38-split-decode serve|run <ip> <port> <j> <prompt token file>` with `--q38-gguf`, `--q38-ple`, `--q38-new`
// (400), `--q38-chunk` (2048), `--q38-seed` (1), `--q38-force <out file of the g run>`, `--q38-out`.
// Both sides prefill the whole prompt through all 48 layers on their own (the MBP also the MTP head). Then the MTP
// spec loop with one draft (chain 1): the MBP drafts d, sends [rollback keep, y, d, pos] to the M6, which runs
// layers 0 ..< j on [y, d] and answers the two residual rows (and its times); the MBP runs layers j ..< 48, the
// head, and decides. j 0 is the MBP alone (no socket), j 48 leaves the MBP only the head. Without `--q38-force` the
// MBP samples (instruct, `--q38-seed`) and stops at EOS: the g run, which writes `ids:` and `top1:` (each token's
// trunk argmax). With it, the tokens are the g run's: a draft is kept when it equals the next forced token, and the
// run goes on to `--q38-new` tokens.

private func sdVM() -> (pageins: UInt64, swapouts: UInt64, wired: UInt64) {
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
    _ = withUnsafeMutablePointer(to: &stats) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
    }
    return (stats.pageins, stats.swapouts, UInt64(stats.wire_count))
}

private func sdSockAll(_ fd: Int32, _ p: UnsafeRawPointer, _ n: Int, send: Bool) {
    var done = 0
    while done < n {
        let r = send ? Darwin.send(fd, p + done, n - done, 0)
                     : Darwin.recv(fd, UnsafeMutableRawPointer(mutating: p + done), n - done, 0)
        guard r > 0 else { perror(send ? "send" : "recv"); exit(1) }
        done += r
    }
}

private func sdAddr(_ ip: String, _ port: Int) -> sockaddr_in {
    var a = sockaddr_in()
    a.sin_family = sa_family_t(AF_INET); a.sin_port = in_port_t(UInt16(port).bigEndian)
    inet_pton(AF_INET, ip, &a.sin_addr)
    return a
}

private func sdTune(_ fd: Int32) {
    var one: Int32 = 1
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, 4)
}

func runQwen38SplitDecode(mode: String, ip: String, port: Int, j: Int, tokenFile: String, gguf: String, ple: String,
                          newTokens: Int, chunk: Int, seed: UInt64, force: String?, out: String) throws {
    precondition(mode == "serve" || mode == "run", "--qwen38-split-decode serve|run")
    setvbuf(stdout, nil, _IOLBF, 0)
    let serve = mode == "serve"
    let text = try String(contentsOfFile: tokenFile, encoding: .utf8)
    let prompt = text.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    let n = prompt.count
    let runner = try Qwen38Runner(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                                  ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                                  capacity: n + newTokens + 2, maxBatch: chunk)
    let nTrunk = 48
    precondition(j >= 0 && j <= nTrunk)
    precondition(!serve || j > 0, "the M6 serves layers 0 ..< j with j > 0")
    let W = runner.hc * runner.e
    let rowBytes = W * 4
    func now() -> Double { CFAbsoluteTimeGetCurrent() }
    let v0 = sdVM()
    print("Qwen3.8 split decode (\(mode)): prompt \(n) tokens (\(tokenFile)), chunk \(chunk), j \(j), new \(newTokens)"
          + (serve ? "" : ", " + (force.map { "forced by \($0)" } ?? "sampler instruct seed \(seed)")))

    var listen_: Int32 = -1
    if serve {
        listen_ = socket(AF_INET, SOCK_STREAM, 0)
        var one: Int32 = 1
        setsockopt(listen_, SOL_SOCKET, SO_REUSEADDR, &one, 4)
        var a = sdAddr(ip, port)
        guard withUnsafePointer(to: &a, { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(listen_, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }) == 0,
              listen(listen_, 1) == 0 else { perror("bind/listen"); exit(1) }
        print("listening on \(ip):\(port)")
    }

    // Prefill: all layers on both sides (the MBP also the MTP head, as runQwen38Generate).
    runner.exportHidden = !serve
    var pendingH = [Float](repeating: 0, count: W)
    var lastLogits = [Float]()
    var trunkPrefill = 0.0, mtpPrefill = 0.0
    var start = 0
    while start < n {
        let T = min(chunk, n - start)
        let t0 = now()
        lastLogits = Array(try runner.forward(tokens: Array(prompt[start..<(start + T)]), startPos: start))
        trunkPrefill += now() - t0
        if !serve {
            let t1 = now()
            var rows = [Float](repeating: 0, count: T * W)
            rows.withUnsafeMutableBufferPointer { b in
                for r in 0..<T {
                    if r == 0 { b.baseAddress!.update(from: pendingH, count: W) }
                    else { (b.baseAddress! + r * W).update(from: runner.hidden(row: r - 1), count: W) }
                }
            }
            var s = 0
            while s < T {
                let m = min(256, T - s)
                _ = try rows.withUnsafeBufferPointer {
                    try runner.mtpForward(tokens: Array(prompt[(start + s)..<(start + s + m)]), startPos: start + s,
                                          hidden: $0.baseAddress! + s * W)
                }
                s += m
            }
            pendingH = Array(UnsafeBufferPointer(start: runner.hidden(row: T - 1), count: W))
            mtpPrefill += now() - t1
        }
        start += T
    }
    let vPrefill = sdVM()
    print(String(format: "  prefill %d tokens: trunk %.1f s%@", n, trunkPrefill, serve ? "" : String(format: ", mtp %.1f s", mtpPrefill)))

    if serve {
        let fd = accept(listen_, nil, nil)
        sdTune(fd)
        var ready: UInt8 = 1   // both prefills are done: the MBP starts the decode clock after this
        sdSockAll(fd, &ready, 1, send: true)
        runner.layerRange = 0..<j
        runner.trunkOnly = true
        var header = [Int32](repeating: 0, count: 4)
        var reply = [UInt8](repeating: 0, count: 2 * rowBytes + 4 * 8)
        var steps = 0
        var trunkMs: [Double] = []
        var wiredMax = sdVM().wired
        print("step\tpos\tkeep\ttrunk_ms\tkernel_gpu_ms\tadvise_ms\trollback_ms")
        while true {
            header.withUnsafeMutableBytes { sdSockAll(fd, $0.baseAddress!, 16, send: false) }
            if header[1] < 0 { break }
            let keep = Int(header[0]), y = Int(header[1]), d = Int(header[2]), pos = Int(header[3])
            var rMs = 0.0
            if keep > 0 {
                let tr = now()
                runner.rollback(keep: keep)
                rMs = (now() - tr) * 1000
            }
            let t = now()
            runner.snapshotRows = 1
            _ = try runner.forward(tokens: [y, d], startPos: pos)
            runner.snapshotRows = 0
            let ms = (now() - t) * 1000
            let p = runner.lastProfile
            reply.withUnsafeMutableBytes { b in
                b.baseAddress!.copyMemory(from: runner.hidden(row: 0), byteCount: rowBytes)
                (b.baseAddress! + rowBytes).copyMemory(from: runner.hidden(row: 1), byteCount: rowBytes)
                let times = [ms, p.routedKernelToGPU * 1000, (p.routeAdvise - p.missTime) * 1000, rMs]
                times.withUnsafeBytes { (b.baseAddress! + 2 * rowBytes).copyMemory(from: $0.baseAddress!, byteCount: 32) }
            }
            reply.withUnsafeBytes { sdSockAll(fd, $0.baseAddress!, $0.count, send: true) }
            print(String(format: "%d\t%d\t%d\t%.1f\t%.1f\t%.1f\t%.1f", steps, pos, keep, ms, p.routedKernelToGPU * 1000,
                         (p.routeAdvise - p.missTime) * 1000, rMs))
            trunkMs.append(ms)
            steps += 1
            wiredMax = max(wiredMax, sdVM().wired)
        }
        close(fd)
        let v1 = sdVM(), page = UInt64(getpagesize())
        let sorted = trunkMs.sorted()
        print(String(format: "  served %d steps, trunk median %.1f ms; decode Pageins +%llu, Swapouts +%llu (run: Pageins +%llu, Swapouts +%llu), wired max %.2f GB",
                     steps, sorted.isEmpty ? .nan : sorted[sorted.count / 2], v1.pageins - vPrefill.pageins, v1.swapouts - vPrefill.swapouts,
                     v1.pageins - v0.pageins, v1.swapouts - v0.swapouts, Double(wiredMax * page) / 1e9))
        return
    }

    // MBP.
    var forced: [Int]? = nil, forcedTop1: [Int] = []
    if let f = force {
        let body = try String(contentsOfFile: f, encoding: .utf8)
        func line(_ key: String) -> [Int] {
            guard let l = body.split(separator: "\n").first(where: { $0.hasPrefix(key) }) else { return [] }
            return l.dropFirst(key.count).split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        }
        forced = line("ids: ")
        forcedTop1 = line("top1: ")
        precondition(forced!.count >= newTokens, "the forced sequence has \(forced!.count) tokens, fewer than \(newTokens)")
    }
    var fd: Int32 = -1
    if j > 0 {
        fd = socket(AF_INET, SOCK_STREAM, 0)
        var a = sdAddr(ip, port)
        guard withUnsafePointer(to: &a, { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }) == 0
        else { perror("connect"); exit(1) }
        sdTune(fd)
        var ready: UInt8 = 0
        let tw = now()
        sdSockAll(fd, &ready, 1, send: false)
        print(String(format: "  waited %.1f s for the M6's prefill", now() - tw))
        runner.layerRange = j..<nTrunk
    }
    let V = runner.vocab
    let eos: Set<Int> = [248_046, 248_044]
    var sampler = Q38Sampler(greedy: false, seed: seed)
    var generated: [Int] = []
    var top1: [Int] = [lastLogits.withUnsafeBufferPointer { Q38Sampler.argmax($0) }]
    var y = forced?[0] ?? lastLogits.withUnsafeBufferPointer { sampler.sample($0) }
    generated.append(y)
    var mtpTokens: [Int] = [], mtpRows: [Float] = []
    var pos = n, pendingKeep: Int32 = 0
    var header = [Int32](repeating: 0, count: 4)
    var reply = [UInt8](repeating: 0, count: 2 * rowBytes + 4 * 8)
    struct Step { var first: Int; var tokens: Int; var kept: Int; var wall, draft, m6, mbp, link, m6KG, m6Adv, m6Rb, kg, adv, rb: Double }
    var steps: [Step] = []
    var wiredMax = sdVM().wired
    print("step\tpos\tfirst\ttokens\tkept\twall_ms\tdraft_ms\tm6_trunk_ms\tmbp_trunk_ms\tlink_ms\tm6_kernel_gpu_ms\tm6_advise_ms\tm6_rollback_ms\tmbp_kernel_gpu_ms\tmbp_advise_ms\tmbp_rollback_ms")
    loop: while generated.count < newTokens && (forced != nil || !eos.contains(y)) {
        let tStep = now()
        let first = generated.count
        let hidden = mtpRows + pendingH
        let d = Q38Sampler.argmax(try hidden.withUnsafeBufferPointer {
            try runner.mtpForward(tokens: mtpTokens + [y], startPos: pos - mtpTokens.count, hidden: $0.baseAddress!)
        })
        let dMs = (now() - tStep) * 1000
        var m6 = [Double](repeating: 0, count: 4), linkMs = 0.0
        if j > 0 {
            let tl = now()
            header = [pendingKeep, Int32(y), Int32(d), Int32(pos)]
            header.withUnsafeBytes { sdSockAll(fd, $0.baseAddress!, 16, send: true) }
            reply.withUnsafeMutableBytes { sdSockAll(fd, $0.baseAddress!, $0.count, send: false) }
            let wall = (now() - tl) * 1000
            reply.withUnsafeBytes { b in
                runner.residualIn = Array(UnsafeBufferPointer(start: b.baseAddress!.assumingMemoryBound(to: Float.self), count: 2 * W))
                m6 = Array(UnsafeBufferPointer(start: (b.baseAddress! + 2 * rowBytes).assumingMemoryBound(to: Double.self), count: 4))
            }
            linkMs = wall - m6[0] - m6[3]
        }
        let tv = now()
        runner.snapshotRows = 1
        let logits = Array(try runner.forward(tokens: [y, d], startPos: pos, allLogits: true))
        runner.snapshotRows = 0
        let vMs = (now() - tv) * 1000
        let vp = runner.lastProfile
        let rows = (0...1).map { Array(UnsafeBufferPointer(start: runner.hidden(row: $0), count: W)) }
        func row(_ i: Int) -> ArraySlice<Float> { logits[(i * V)..<((i + 1) * V)] }
        func pick(_ i: Int) -> Int {
            if let f = forced { return f[generated.count] }
            return row(i).withUnsafeBufferPointer { sampler.sample($0) }
        }
        var kept = 0
        var t = pick(0)
        top1.append(row(0).withUnsafeBufferPointer { Q38Sampler.argmax($0) })
        generated.append(t)
        if t == d && (forced != nil || !eos.contains(t)) && generated.count < newTokens {
            kept = 1
            t = pick(1)
            top1.append(row(1).withUnsafeBufferPointer { Q38Sampler.argmax($0) })
            generated.append(t)
        }
        var rMs = 0.0
        if kept < 1 {
            let tr = now()
            runner.rollback(keep: 1)
            rMs = (now() - tr) * 1000
        }
        pendingKeep = kept < 1 ? 1 : 0
        mtpTokens = kept == 1 ? [d] : []
        mtpRows = kept == 1 ? rows[0] : []
        pendingH = rows[kept]
        pos += kept + 1
        let st = Step(first: first, tokens: kept + 1, kept: kept, wall: (now() - tStep) * 1000, draft: dMs, m6: m6[0], mbp: vMs,
                      link: linkMs, m6KG: m6[1], m6Adv: m6[2], m6Rb: m6[3], kg: vp.routedKernelToGPU * 1000,
                      adv: (vp.routeAdvise - vp.missTime) * 1000, rb: rMs)
        steps.append(st)
        print(String(format: "%d\t%d\t%d\t%d\t%d\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f", steps.count - 1, pos - kept - 1,
                     st.first, st.tokens, st.kept, st.wall, st.draft, st.m6, st.mbp, st.link, st.m6KG, st.m6Adv, st.m6Rb, st.kg, st.adv, st.rb))
        wiredMax = max(wiredMax, sdVM().wired)
        y = t
        if forced == nil && eos.contains(t) { break loop }
    }
    if j > 0 {
        header = [0, -1, 0, 0]
        header.withUnsafeBytes { sdSockAll(fd, $0.baseAddress!, 16, send: true) }
        close(fd)
    }
    let v1 = sdVM(), page = UInt64(getpagesize())
    // The window (docs/qfn-prefill/23 §0): from the step that emitted token 100 to the one that emitted token 399.
    func window(_ s: ArraySlice<Step>) -> (Double, Int, Int) {
        let ms = s.reduce(0) { $0 + $1.wall }, tok = s.reduce(0) { $0 + $1.tokens }
        return (ms / 1000 / Double(max(tok, 1)), tok, s.count)
    }
    let w0 = steps.firstIndex { $0.first + $0.tokens > 100 } ?? steps.count
    let w1 = steps.firstIndex { $0.first + $0.tokens > 399 } ?? (steps.count - 1)
    let early = window(steps[0..<w0]), main = window(w0 <= w1 && w1 < steps.count ? steps[w0...w1] : [])
    let accepted = steps.filter { $0.kept == 1 }.count
    let top1Match = forcedTop1.isEmpty ? -1 : zip(top1, forcedTop1).filter { $0 == $1 }.count
    func med(_ k: KeyPath<Step, Double>) -> Double {
        let v = (w0 <= w1 && w1 < steps.count ? steps[w0...w1] : []).map { $0[keyPath: k] }.sorted()
        return v.isEmpty ? .nan : v[v.count / 2]
    }
    var summary = String(format: "  j %d: %d steps, %d tokens, accepted %d, top1 agree %d / %d\n", j, steps.count, generated.count, accepted,
                         top1Match, min(top1.count, forcedTop1.count))
    summary += String(format: "  window tokens 100-399: %.4f s/token (%d tokens in %d steps); before: %.4f s/token (%d tokens in %d steps)\n",
                      main.0, main.1, main.2, early.0, early.1, early.2)
    summary += String(format: "  window step medians ms: wall %.1f draft %.1f m6 %.1f mbp %.1f link %.1f m6 kernel>gpu %.1f advise %.1f | mbp kernel>gpu %.1f advise %.1f\n",
                      med(\.wall), med(\.draft), med(\.m6), med(\.mbp), med(\.link), med(\.m6KG), med(\.m6Adv), med(\.kg), med(\.adv))
    summary += String(format: "  decode Pageins +%llu, Swapouts +%llu (run: Pageins +%llu, Swapouts +%llu), wired max %.2f GB",
                      v1.pageins - vPrefill.pageins, v1.swapouts - vPrefill.swapouts, v1.pageins - v0.pageins, v1.swapouts - v0.swapouts,
                      Double(wiredMax * page) / 1e9)
    print(summary)
    let body = "ids: " + generated.map(String.init).joined(separator: ",") + "\n"
        + "top1: " + top1.map(String.init).joined(separator: ",") + "\n" + summary + "\n"
    try body.write(toFile: out, atomically: true, encoding: .utf8)
    print("  written \(out)")
}
