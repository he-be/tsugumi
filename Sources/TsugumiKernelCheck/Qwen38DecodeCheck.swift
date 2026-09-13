import Foundation
import Tsugumi

// MARK: - Qwen3.8-Flash-Next Q2 decode against the CPU reference
//
// `Qwen38Runner` (Metal, GGUF-direct) runs the reference's prompt one token at
// a time and greedy-extends it; every position's top-1 is compared with the
// reference log (`Scripts/qwen38/reference_forward.py`, lines
// `[ i] prefill ... top1=X` and `[ i] token= X`). With `--q38-ref-logits`
// (the reference's `--dump-logits` file, [n, vocab] float32) it also reports
// the relative logit error per position.
//
//     .build/release/TsugumiKernelCheck --qwen38-decode scratch/qwen38/ref-france.log \
//         [--q38-ref-logits scratch/qwen38/ref-france.logits] [--q38-gguf ...] [--q38-ple ...]

private func parseQwen38RefLog(_ path: String) throws -> (prompt: [Int], top1: [Int]) {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    var prompt: [Int] = []
    var top1: [Int: Int] = [:]
    for line in text.split(separator: "\n") {
        if line.hasPrefix("入力"), let open = line.firstIndex(of: "["), let close = line.lastIndex(of: "]") {
            prompt = line[line.index(after: open)..<close].split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        } else if line.hasPrefix("["), let close = line.firstIndex(of: "]"),
                  let pos = Int(line[line.index(after: line.startIndex)..<close].trimmingCharacters(in: .whitespaces)) {
            func field(_ key: String) -> Int? {
                guard let r = line.range(of: key) else { return nil }
                let rest = line[r.upperBound...].drop(while: { $0 == " " })
                return Int(rest.prefix(while: { $0.isNumber }))
            }
            if line.contains("prefill") { top1[pos] = field("top1=") } else { top1[pos] = field("token=") }
        }
    }
    let n = (top1.keys.max() ?? -1) + 1
    return (prompt, (0..<n).map { top1[$0] ?? -1 })
}

func runQwen38DecodeCheck(refLog: String, refLogits: String?, gguf: String, ple: String,
                          indexerTopK: Int? = nil) throws -> Bool {
    let ref = try parseQwen38RefLog(refLog)
    let n = ref.top1.count
    precondition(!ref.prompt.isEmpty && n > 0, "reference log has no prompt or positions")
    let runner = try Qwen38Runner(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                                  ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                                  capacity: n + 1)
    runner.splitPreRouter = ProcessInfo.processInfo.environment["Q38_SPLIT_PRE"] != nil
    if let indexerTopK {
        runner.indexerTopK = indexerTopK
        print("  indexer top_k override: \(indexerTopK) tokens")
    }
    var refLogitsData: Data?
    if let refLogits { refLogitsData = try Data(contentsOf: URL(fileURLWithPath: refLogits)) }
    print("Qwen3.8 Q2 decode check: prompt \(ref.prompt.count) tokens, \(n) positions (\(refLog))")
    var seq = ref.prompt
    var mismatches = 0
    var worstLogit = 0.0
    for pos in 0..<n {
        let t0 = Date()
        let logits = try runner.step(token: seq[pos], pos: pos)
        var best = 0
        for i in 1..<logits.count where logits[i] > logits[best] { best = i }
        if pos + 1 >= ref.prompt.count { seq.append(best) }
        var logitNote = ""
        if let data = refLogitsData {
            let v = logits.count
            data.withUnsafeBytes { raw in
                let r = raw.bindMemory(to: Float.self)
                var maxDiff = 0.0, refMax = 0.0
                for i in 0..<v {
                    refMax = max(refMax, abs(Double(r[pos * v + i])))
                    maxDiff = max(maxDiff, abs(Double(logits[i]) - Double(r[pos * v + i])))
                }
                worstLogit = max(worstLogit, maxDiff / refMax)
                logitNote = String(format: "  logit rel err %.2e", maxDiff / refMax)
            }
        }
        let ok = best == ref.top1[pos]
        if !ok { mismatches += 1 }
        let pr = runner.lastProfile
        print(String(format: "  [%3d] input %7d  top1 %7d  ref %7d  %@  %.2fs (ple %.0f pre %.0f [gpu %.0f] route %.0f routed %.0f [gpu %.0f] head %.0f ms, miss %.0f MB)%@",
                     pos, seq[pos], best, ref.top1[pos], ok ? "ok  " : "DIFF",
                     Date().timeIntervalSince(t0), pr.ple * 1000, pr.preRouter * 1000, pr.preGPU * 1000, pr.route * 1000,
                     pr.routed * 1000, pr.routedGPU * 1000, pr.head * 1000,
                     Double(pr.missBytes) / 1e6, logitNote))
        if !pr.sections.isEmpty {
            print("        pre-router GPU ms: " + pr.sections.sorted { $0.key < $1.key }
                .map { String(format: "%@ %.1f", $0.key as NSString, $0.value) }.joined(separator: "  "))
        }
        // Generated tokens follow the runner's own choice; after a divergence the
        // reference's later positions are a different sequence, so stop there.
        if !ok && pos + 1 >= ref.prompt.count { break }
    }
    let pass = mismatches == 0 && (refLogitsData == nil || worstLogit < 1e-3)
    print("  top-1 mismatches: \(mismatches)" + (refLogitsData != nil ? String(format: ", worst logit rel err %.2e", worstLogit) : ""))
    print("  \(pass ? "PASS" : "FAIL")")
    return pass
}

// MARK: - Prefill: the same sequence in chunks (`--qwen38-prefill <ref log> [--q38-chunk N]`)
//
// The reference's prompt plus its greedy continuation, fed through
// `Qwen38Runner.forward` `chunk` tokens at a time with every position's logits
// kept, and compared position by position like the decode check. A chunk
// boundary inside the sequence exercises the carried state (GDN conv and
// recurrent state, PLE history, KV and indexer caches).
func runQwen38PrefillCheck(refLog: String, refLogits: String?, gguf: String, ple: String,
                           indexerTopK: Int?, chunk: Int) throws -> Bool {
    let ref = try parseQwen38RefLog(refLog)
    let n = ref.top1.count
    precondition(!ref.prompt.isEmpty && n > 0, "reference log has no prompt or positions")
    var seq = ref.prompt
    while seq.count < n { seq.append(ref.top1[seq.count - 1]) }
    let runner = try Qwen38Runner(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                                  ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                                  capacity: n + 1, maxBatch: chunk)
    if let indexerTopK {
        runner.indexerTopK = indexerTopK
        print("  indexer top_k override: \(indexerTopK) tokens")
    }
    var refLogitsData: Data?
    if let refLogits { refLogitsData = try Data(contentsOf: URL(fileURLWithPath: refLogits)) }
    print("Qwen3.8 Q2 prefill check: \(n) positions in chunks of \(chunk) (\(refLog))")
    var mismatches = 0
    var worstLogit = 0.0
    var start = 0
    while start < n {
        let T = min(chunk, n - start)
        let t0 = Date()
        let all = try runner.forward(tokens: Array(seq[start..<(start + T)]), startPos: start, allLogits: true)
        let pr = runner.lastProfile
        var worstChunk = 0.0
        var line = ""
        for t in 0..<T {
            let pos = start + t
            let logits = UnsafeBufferPointer(rebasing: all[(t * runner.vocab)..<((t + 1) * runner.vocab)])
            var best = 0
            for i in 1..<logits.count where logits[i] > logits[best] { best = i }
            if best != ref.top1[pos] { mismatches += 1; line += " DIFF@\(pos)" }
            if let data = refLogitsData {
                data.withUnsafeBytes { raw in
                    let r = raw.bindMemory(to: Float.self)
                    let v = runner.vocab
                    var maxDiff = 0.0, refMax = 0.0
                    for i in 0..<v {
                        refMax = max(refMax, abs(Double(r[pos * v + i])))
                        maxDiff = max(maxDiff, abs(Double(logits[i]) - Double(r[pos * v + i])))
                    }
                    worstChunk = max(worstChunk, maxDiff / refMax)
                }
            }
        }
        worstLogit = max(worstLogit, worstChunk)
        print(String(format: "  [%3d..<%3d] %.2fs (ple %.0f pre %.0f [gpu %.0f] route %.0f routed %.0f [gpu %.0f] head %.0f ms, experts %d)%@%@",
                     start, start + T, Date().timeIntervalSince(t0), pr.ple * 1000, pr.preRouter * 1000,
                     pr.preGPU * 1000, pr.route * 1000, pr.routed * 1000, pr.routedGPU * 1000, pr.head * 1000,
                     pr.distinctExperts,
                     refLogitsData != nil ? String(format: "  logit rel err %.2e", worstChunk) : "", line))
        start += T
    }
    let pass = mismatches == 0 && (refLogitsData == nil || worstLogit < 1e-3)
    print("  top-1 mismatches: \(mismatches)" + (refLogitsData != nil ? String(format: ", worst logit rel err %.2e", worstLogit) : ""))
    print("  \(pass ? "PASS" : "FAIL")")
    return pass
}

// MARK: - Prefill speed (`--qwen38-prefill-bench <token file> [--q38-tokens N] [--q38-chunk C]`)
//
// N token ids (comma-separated file) through `forward` C at a time, last logits only,
// then one decode step at the end of the context. Prints each chunk's stages.
// `--q38-dump-logits F` saves the prefill's last logits and the decode step's; `--q38-compare-logits F`
// compares against such a file (the long-context check: no CPU reference reaches 8K, so a run of
// another path, e.g. `Q38_ATTN_MPS_MIN_T=0`, is the oracle).
func runQwen38PrefillBench(tokenFile: String, tokens: Int, chunk: Int, gguf: String, ple: String,
                           dumpLogits: String? = nil, compareLogits: String? = nil) throws -> Bool {
    let text = try String(contentsOfFile: tokenFile, encoding: .utf8)
    let ids = text.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    let n = min(tokens, ids.count)
    setvbuf(stdout, nil, _IOLBF, 0)   // lines survive a guard kill
    let benchWidths = (ProcessInfo.processInfo.environment["Q38_BENCH_WIDTHS"] ?? "").split(separator: ",").compactMap { Int($0) }
    let benchReps = Int(ProcessInfo.processInfo.environment["Q38_BENCH_REPS"] ?? "") ?? 5
    let runner = try Qwen38Runner(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                                  ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                                  capacity: n + 2 + benchWidths.reduce(0, +) * benchReps,
                                  maxBatch: chunk)
    runner.splitPreRouter = ProcessInfo.processInfo.environment["Q38_SPLIT_PRE"] != nil
    print("Qwen3.8 Q2 prefill bench: \(n) tokens in chunks of \(chunk) (\(tokenFile))")
    let tAll = Date()
    var start = 0
    var last = 0
    var prefillLogits: [Float] = []
    while start < n {
        let T = min(chunk, n - start)
        let t0 = Date()
        let logits = try runner.forward(tokens: Array(ids[start..<(start + T)]), startPos: start)
        for i in 1..<logits.count where logits[i] > logits[last] { last = i }
        if start + T == n { prefillLogits = Array(logits) }
        let pr = runner.lastProfile
        let s = Date().timeIntervalSince(t0)
        print(String(format: "  [%5d..<%5d] %6.2fs %6.1f tok/s (ple %.0f pre %.0f [gpu %.0f] route %.0f routed %.0f [gpu %.0f] head %.0f ms, experts %d; route = topk %.0f views %.0f advise %.0f (%d calls))",
                     start, start + T, s, Double(T) / s, pr.ple * 1000, pr.preRouter * 1000, pr.preGPU * 1000,
                     pr.route * 1000, pr.routed * 1000, pr.routedGPU * 1000, pr.head * 1000, pr.distinctExperts,
                     pr.routeTopK * 1000, pr.routeViews * 1000, pr.routeAdvise * 1000, pr.adviseCalls))
        if ProcessInfo.processInfo.environment["Q38_MEM_LOG"] != nil {
            print("        mem MB: " + runner.memoryReport().map { String(format: "%@ %.0f", $0.0 as NSString, Double($0.1) / 1e6) }.joined(separator: "  "))
        }
        if !pr.sections.isEmpty {
            print("        pre-router GPU ms: " + pr.sections.sorted { $0.key < $1.key }
                .map { String(format: "%@ %.0f", $0.key as NSString, $0.value) }.joined(separator: "  "))
        }
        start += T
    }
    let total = Date().timeIntervalSince(tAll)
    print(String(format: "  prefill %d tokens: %.1f s, %.1f tok/s", n, total, Double(n) / total))
    var best = 0
    let t0 = Date()
    let logits = Array(try runner.step(token: ids.count > n ? ids[n] : last, pos: n))   // valid only until the next forward
    for i in 1..<logits.count where logits[i] > logits[best] { best = i }
    let pr = runner.lastProfile
    print(String(format: "  decode at %d: %.2fs (pre %.0f [gpu %.0f] route %.0f routed %.0f [gpu %.0f] ms)",
                 n, Date().timeIntervalSince(t0), pr.preRouter * 1000, pr.preGPU * 1000,
                 pr.route * 1000, pr.routed * 1000, pr.routedGPU * 1000))
    // `Q38_BENCH_WIDTHS=1,2,3,5 Q38_BENCH_REPS=R`: after the decode step, batches of k following prompt tokens
    // (the verify pass of a k-1 draft), the widths interleaved R times, positions advancing through the text.
    if !benchWidths.isEmpty {
        let widths = benchWidths, reps = benchReps
        var pos = n + 1
        var rows: [Int: [(wall: Double, pr: Qwen38Runner.StepProfile)]] = [:]
        rep: for _ in 0..<reps {
            for k in widths {
                guard pos + k <= ids.count else { break rep }
                let t = Date()
                _ = try runner.forward(tokens: Array(ids[pos..<(pos + k)]), startPos: pos, allLogits: true)
                rows[k, default: []].append((Date().timeIntervalSince(t), runner.lastProfile))
                pos += k
            }
        }
        func med(_ v: [Double]) -> Double { let s = v.sorted(); return s.isEmpty ? .nan : s[s.count / 2] }
        if ProcessInfo.processInfo.environment["Q38_MEM_LOG"] != nil {
            print("  mem MB after widths: " + runner.memoryReport().map { String(format: "%@ %.0f", $0.0 as NSString, Double($0.1) / 1e6) }.joined(separator: "  "))
        }
        print("  verify widths from pos \(n + 1) (median of each width, ms):")
        for k in widths {
            guard let r = rows[k], !r.isEmpty else { continue }
            print(String(format: "    T=%d n=%d  wall %.0f (min %.0f max %.0f)  ple %.0f pre %.0f [gpu %.0f] route %.0f (topk %.0f views %.0f advise %.0f) routed %.0f [gpu %.0f] head %.0f  experts %.0f",
                         k, r.count, med(r.map { $0.wall }) * 1000, r.map { $0.wall }.min()! * 1000, r.map { $0.wall }.max()! * 1000,
                         med(r.map { $0.pr.ple }) * 1000, med(r.map { $0.pr.preRouter }) * 1000, med(r.map { $0.pr.preGPU }) * 1000,
                         med(r.map { $0.pr.route }) * 1000, med(r.map { $0.pr.routeTopK }) * 1000, med(r.map { $0.pr.routeViews }) * 1000,
                         med(r.map { $0.pr.routeAdvise }) * 1000, med(r.map { $0.pr.routed }) * 1000, med(r.map { $0.pr.routedGPU }) * 1000,
                         med(r.map { $0.pr.head }) * 1000, med(r.map { Double($0.pr.distinctExperts) })))
            if let keys = r.first?.pr.sections.keys, !keys.isEmpty {
                print("        sections GPU ms (median): " + keys.sorted().map { key in
                    String(format: "%@ %.1f", key as NSString, med(r.map { $0.pr.sections[key] ?? 0 }))
                }.joined(separator: "  "))
            }
        }
    }
    if ProcessInfo.processInfo.environment["Q38_MEM_LOG"] != nil {
        print("  mem MB at end: " + runner.memoryReport().map { String(format: "%@ %.0f", $0.0 as NSString, Double($0.1) / 1e6) }.joined(separator: "  "))
    }
    let both = prefillLogits + logits
    if let path = dumpLogits {
        try both.withUnsafeBytes { Data($0) }.write(to: URL(fileURLWithPath: path))
        print("  logits written to \(path)")
    }
    guard let path = compareLogits else { return true }
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let ref = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    precondition(ref.count == both.count, "logit file size")
    let v = logits.count
    var pass = true
    for (label, o) in [("prefill last", 0), ("decode", v)] {
        func argmax(_ a: ArraySlice<Float>) -> Int { a.indices.max { a[$0] < a[$1] }! - a.startIndex }
        var maxDiff = 0.0, refMax = 0.0
        for i in 0..<v {
            refMax = max(refMax, abs(Double(ref[o + i])))
            maxDiff = max(maxDiff, abs(Double(both[o + i]) - Double(ref[o + i])))
        }
        let same = argmax(both[o..<(o + v)]) == argmax(ref[o..<(o + v)])
        pass = pass && same && maxDiff / refMax < 1e-3
        print(String(format: "  %@ vs %@: top-1 %@, logit rel err %.2e", label as NSString, path as NSString, same ? "same" : "DIFFERS", maxDiff / refMax))
    }
    print("  \(pass ? "PASS" : "FAIL")")
    return pass
}
