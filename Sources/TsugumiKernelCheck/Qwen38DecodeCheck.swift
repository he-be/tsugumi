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
func runQwen38PrefillBench(tokenFile: String, tokens: Int, chunk: Int, gguf: String, ple: String) throws {
    let text = try String(contentsOfFile: tokenFile, encoding: .utf8)
    let ids = text.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    let n = min(tokens, ids.count)
    let runner = try Qwen38Runner(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                                  ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                                  capacity: n + 2, maxBatch: chunk)
    runner.splitPreRouter = ProcessInfo.processInfo.environment["Q38_SPLIT_PRE"] != nil
    print("Qwen3.8 Q2 prefill bench: \(n) tokens in chunks of \(chunk) (\(tokenFile))")
    let tAll = Date()
    var start = 0
    var last = 0
    while start < n {
        let T = min(chunk, n - start)
        let t0 = Date()
        let logits = try runner.forward(tokens: Array(ids[start..<(start + T)]), startPos: start)
        for i in 1..<logits.count where logits[i] > logits[last] { last = i }
        let pr = runner.lastProfile
        let s = Date().timeIntervalSince(t0)
        print(String(format: "  [%5d..<%5d] %6.2fs %6.1f tok/s (ple %.0f pre %.0f [gpu %.0f] route %.0f routed %.0f [gpu %.0f] head %.0f ms, experts %d; route = topk %.0f views %.0f advise %.0f (%d calls))",
                     start, start + T, s, Double(T) / s, pr.ple * 1000, pr.preRouter * 1000, pr.preGPU * 1000,
                     pr.route * 1000, pr.routed * 1000, pr.routedGPU * 1000, pr.head * 1000, pr.distinctExperts,
                     pr.routeTopK * 1000, pr.routeViews * 1000, pr.routeAdvise * 1000, pr.adviseCalls))
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
    let logits = try runner.step(token: ids.count > n ? ids[n] : last, pos: n)
    for i in 1..<logits.count where logits[i] > logits[best] { best = i }
    let pr = runner.lastProfile
    print(String(format: "  decode at %d: %.2fs (pre %.0f [gpu %.0f] route %.0f routed %.0f [gpu %.0f] ms)",
                 n, Date().timeIntervalSince(t0), pr.preRouter * 1000, pr.preGPU * 1000,
                 pr.route * 1000, pr.routed * 1000, pr.routedGPU * 1000))
}
