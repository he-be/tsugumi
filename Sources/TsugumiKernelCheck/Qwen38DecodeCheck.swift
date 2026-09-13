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
        print(String(format: "  [%3d] input %7d  top1 %7d  ref %7d  %@  %.2fs (ple %.0f pre %.0f route %.0f routed %.0f head %.0f ms)%@",
                     pos, seq[pos], best, ref.top1[pos], ok ? "ok  " : "DIFF",
                     Date().timeIntervalSince(t0), pr.ple * 1000, pr.preRouter * 1000, pr.route * 1000,
                     pr.routed * 1000, pr.head * 1000, logitNote))
        // Generated tokens follow the runner's own choice; after a divergence the
        // reference's later positions are a different sequence, so stop there.
        if !ok && pos + 1 >= ref.prompt.count { break }
    }
    let pass = mismatches == 0 && (refLogitsData == nil || worstLogit < 1e-3)
    print("  top-1 mismatches: \(mismatches)" + (refLogitsData != nil ? String(format: ", worst logit rel err %.2e", worstLogit) : ""))
    print("  \(pass ? "PASS" : "FAIL")")
    return pass
}
