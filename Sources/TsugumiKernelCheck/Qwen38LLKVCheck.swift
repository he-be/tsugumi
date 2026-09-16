import Foundation
import Tsugumi

// `--qwen38-llkv-check <token file> [--q38-tokens N] [--q38-chunk C] [--q38-new G] [--q38-split L]
//  [--q38-fills exact,mean,hcmix] [--q38-suffixes 256,1]`: the LLKVApprox prefill (docs/qwen38/28) against the
// whole one. The first N tokens are prefilled C at a time, the last chunk with only its last `suffix` tokens through
// layers L...; then G decode steps are teacher-forced with the whole prefill's greedy tokens. Per arm: the prefill
// time and its fill part, the last prefill logits against the whole prefill's (max |Δ| over the vocabulary and over
// its top 20, KL at temperature 0.7, top-1, P(`<tool_call>`)), and over the G steps the top-1 agreement and the mean / max KL. The whole
// prefill runs again at the end (warm time, run-to-run noise).
// `exact` differs from the whole prefill only in where the batch is cut (docs/qwen38/25 §2 measured that noise).
func runQwen38LLKVCheck(tokenFile: String, tokens: Int, chunk: Int, newTokens: Int, split: Int,
                        fills: [Qwen38Runner.LLKVFill], suffixes: [Int], gguf: String, ple: String) throws {
    setvbuf(stdout, nil, _IOLBF, 0)
    let ids = try String(contentsOfFile: tokenFile, encoding: .utf8).split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    let n = min(tokens, ids.count)
    let e = try Qwen38Engine(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                             ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                             capacity: n + newTokens + 2, prefillChunk: chunk, speculative: false)
    let runner = e.runner
    let V = runner.vocab
    print("Qwen3.8 LLKVApprox check: \(n) tokens in chunks of \(chunk), split \(split), \(newTokens) teacher-forced steps (\(tokenFile))")

    func probs(_ l: ArraySlice<Float>) -> [Double] {
        let s = l.map { Double($0) / 0.7 }, m = s.max()!
        let z = s.reduce(0) { $0 + exp($1 - m) }
        return s.map { exp($0 - m) / z }
    }
    func kl(_ p: [Double], _ q: [Double]) -> Double {
        zip(p, q).reduce(0.0) { $0 + ($1.0 > 0 ? $1.0 * log($1.0 / max($1.1, 1e-300)) : 0) }
    }
    func argmax(_ l: ArraySlice<Float>) -> Int { l.indices.max { l[$0] < l[$1] }! - l.startIndex }

    /// Prefill (tail nil: whole), then `forced` (or greedy when nil) decode steps; the prefill's last logits and each step's.
    func arm(tail: Int?, forced: [Int]?) throws -> (prefill: [Float], steps: [[Float]], tokens: [Int], seconds: Double, fill: Double) {
        e.reset()
        let t = Date()
        var fill = 0.0
        var start = 0
        var last: [Float] = []
        while start < n {
            let T = min(chunk, n - start)
            let isLast = start + T == n
            let l = try runner.forward(tokens: Array(ids[start..<(start + T)]), startPos: start,
                                       exactTail: tail.map { isLast ? $0 : 0 })
            fill += runner.lastProfile.llkvFill
            if isLast { last = Array(l.prefix(V)) }
            start += T
        }
        let seconds = Date().timeIntervalSince(t)
        var steps: [[Float]] = []
        var toks: [Int] = []
        var next = argmax(last[...])
        for g in 0..<newTokens {
            let input = forced?[g] ?? next
            toks.append(input)
            let l = Array(try runner.step(token: input, pos: n + g).prefix(V))
            steps.append(l)
            next = argmax(l[...])
        }
        return (last, steps, toks, seconds, fill)
    }

    runner.llkvSplit = 0
    let base = try arm(tail: nil, forced: nil)
    let baseP = probs(base.prefill[...])
    let baseTop1 = argmax(base.prefill[...])
    let baseStepP = base.steps.map { probs($0[...]) }
    let baseStepTop = base.steps.map { argmax($0[...]) }
    let toolCall = 248_058
    print(String(format: "  whole           prefill %6.1f s  top1 %6d  P %.3f  P(tool_call) %.4f", base.seconds, baseTop1,
                 baseP[baseTop1], baseP[toolCall]))
    print("  greedy tokens: " + base.tokens.map(String.init).joined(separator: ","))
    runner.llkvSplit = split
    for fill in fills {
        runner.llkvFill = fill
        for suffix in suffixes {
            let r = try arm(tail: suffix, forced: base.tokens)
            let p = probs(r.prefill[...])
            let top1 = argmax(r.prefill[...])
            let maxAll = zip(r.prefill, base.prefill).map { abs($0 - $1) }.max()!
            let top20 = base.prefill.indices.sorted { base.prefill[$0] > base.prefill[$1] }.prefix(20)
            let maxTop = top20.map { abs(r.prefill[$0] - base.prefill[$0]) }.max()!
            var agree = 0, klSum = 0.0, klMax = 0.0
            for (g, l) in r.steps.enumerated() {
                if argmax(l[...]) == baseStepTop[g] { agree += 1 }
                let d = kl(baseStepP[g], probs(l[...]))
                klSum += d
                klMax = max(klMax, d)
            }
            print(String(format: "  %-6@ tail %4d  prefill %6.1f s (fill %5.1f)  top1 %6d %@  P(base top1) %.3f  max|Δ| %.3f  top20 %.3f  KL %.4f  P(tool_call) %.4f  | steps: top1 %d/%d  KL mean %.4f max %.4f",
                         fill.rawValue as NSString, suffix, r.seconds, r.fill, top1, top1 == baseTop1 ? "same" : "DIFF",
                         p[baseTop1], maxAll, maxTop, kl(baseP, p), p[toolCall], agree, newTokens,
                         klSum / Double(max(newTokens, 1)), klMax))
        }
    }
    // The first arm ran on a colder page cache; the whole prefill again gives its warm time and the run-to-run noise.
    runner.llkvSplit = 0
    let again = try arm(tail: nil, forced: base.tokens)
    print(String(format: "  whole again     prefill %6.1f s  max|Δ| %.3f", again.seconds,
                 zip(again.prefill, base.prefill).map { abs($0 - $1) }.max()!))
}
