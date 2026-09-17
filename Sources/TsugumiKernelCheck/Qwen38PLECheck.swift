import Foundation
import Tsugumi

// `--qwen38-ple-check <token file> --q38-ple-ref <BF16 table> [--q38-ple <Q4_1 table>] [--q38-tokens N]
//  [--q38-chunk C] [--q38-new G]`: the PLE table's quantization (docs/qwen38/31). The first N − G tokens are
// prefilled C at a time and the prompt's own last G tokens are teacher-forced, so every row the run reads is one the
// sparse BF16 table holds, and the last step's logits are the prompt end's (a branch point's P(`<tool_call>`)).
// Arms, in order: the BF16 table (reference), Q4_1, no PLE at all (the scale), BF16 again (run-to-run noise).
// Per arm against the reference: the prefill's last logits (max |Δ|, KL at temperature 0.7, top-1) and over the G
// steps the top-1 agreement, mean / max KL, max |Δ|, and the last step's P(`<tool_call>`).
func runQwen38PLECheck(tokenFile: String, tokens: Int, chunk: Int, newTokens: Int,
                       gguf: String, ple: String, pleRef: String) throws {
    setvbuf(stdout, nil, _IOLBF, 0)
    let ids = try String(contentsOfFile: tokenFile, encoding: .utf8).split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    let n = min(tokens, ids.count)
    let G = min(newTokens, n - 1)
    let url = { (p: String) in URL(fileURLWithPath: (p as NSString).expandingTildeInPath) }
    let e = try Qwen38Engine(gguf: url(gguf), ple: url(pleRef), capacity: n + 2, prefillChunk: chunk, speculative: false)
    let runner = e.runner
    let V = runner.vocab
    let toolCall = 248_058
    print("Qwen3.8 PLE check: \(n - G) tokens prefilled in chunks of \(chunk), \(G) prompt tokens teacher-forced (\(tokenFile))")

    func probs(_ l: ArraySlice<Float>) -> [Double] {
        let s = l.map { Double($0) / 0.7 }, m = s.max()!
        let z = s.reduce(0) { $0 + exp($1 - m) }
        return s.map { exp($0 - m) / z }
    }
    func kl(_ p: [Double], _ q: [Double]) -> Double {
        zip(p, q).reduce(0.0) { $0 + ($1.0 > 0 ? $1.0 * log($1.0 / max($1.1, 1e-300)) : 0) }
    }
    func argmax(_ l: ArraySlice<Float>) -> Int { l.indices.max { l[$0] < l[$1] }! - l.startIndex }
    func maxAbs(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).map { abs($0 - $1) }.max()! }

    func arm() throws -> (prefill: [Float], steps: [[Float]], seconds: Double) {
        e.reset()
        let t = Date()
        var start = 0
        var last: [Float] = []
        while start < n - G {
            let T = min(chunk, n - G - start)
            let l = try runner.forward(tokens: Array(ids[start..<(start + T)]), startPos: start)
            if start + T == n - G { last = Array(l.prefix(V)) }
            start += T
        }
        let steps = try (0..<G).map { g in Array(try runner.step(token: ids[n - G + g], pos: n - G + g).prefix(V)) }
        return (last, steps, Date().timeIntervalSince(t))
    }

    let ref = try arm()
    let refP = probs(ref.prefill[...]), refTop = argmax(ref.prefill[...])
    let refStepP = ref.steps.map { probs($0[...]) }, refStepTop = ref.steps.map { argmax($0[...]) }
    let refLastP = refStepP.last ?? refP
    let forcedHits = (0..<G).filter { g in g + 1 < G && refStepTop[g] == ids[n - G + g + 1] }.count
    print(String(format: "  bf16      %6.1f s  prefill top1 %6d P %.3f  | last step top1 %6d P %.3f  P(tool_call) %.4f  | ref top1 == next prompt token %d/%d",
                 ref.seconds, refTop, refP[refTop], refStepTop.last ?? -1, refLastP[refStepTop.last ?? 0], refLastP[toolCall],
                 forcedHits, max(G - 1, 0)))
    for (name, setup) in [("q4_1", { try runner.replacePLE(url(ple)) }),
                          ("no-ple", { runner.ablatePLE = true }),
                          ("bf16 again", { runner.ablatePLE = false; try runner.replacePLE(url(pleRef)) })] as [(String, () throws -> Void)] {
        try setup()
        let r = try arm()
        let p = probs(r.prefill[...]), top = argmax(r.prefill[...])
        var agree = 0, klSum = 0.0, klMax = 0.0
        var dMax: Float = 0
        for (g, l) in r.steps.enumerated() {
            if argmax(l[...]) == refStepTop[g] { agree += 1 }
            let d = kl(refStepP[g], probs(l[...]))
            klSum += d
            klMax = max(klMax, d)
            dMax = max(dMax, maxAbs(l, ref.steps[g]))
        }
        let lastP = r.steps.last.map { probs($0[...]) } ?? p
        print(String(format: "  %-10@%6.1f s  prefill top1 %@ max|Δ| %.3f KL %.4f  | steps: top1 %d/%d  KL mean %.4f max %.4f  max|Δ| %.3f  | last P(tool_call) %.4f",
                     name as NSString, r.seconds, top == refTop ? "same" : "DIFF", maxAbs(r.prefill, ref.prefill),
                     kl(refP, p), agree, G, klSum / Double(max(G, 1)), klMax, dMax, lastP[toolCall]))
    }
}
