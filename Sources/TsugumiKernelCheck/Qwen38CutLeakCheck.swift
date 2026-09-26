import Foundation
import Tsugumi

// MARK: - Qwen3.8 cut copies across forwards (docs/qfn-prefill/32)
//
// `--qwen38-cut-leak <prompt>` (a token file): the M6's pair serve loop on one runner, `--q38-rounds N` times —
// a forward of the prompt at position 0 with one cut row (`cutRows`, C1), the cut regions read as the serve sends them
// (`forEachCutRegion`), `releaseCutBuffers`, `shrinkBatchAfterRound` — and the Metal allocation after each. The cut
// copies are freed every round, so the allocation must not grow with N. `--q38-layers k` limits the forward to layers
// 0 ..< k with `trunkOnly` as on the M6 (default 46). Exit 1 when the last round's allocation exceeds the first's
// by more than one cut row's copies.

func runQwen38CutLeak(prompt: String, rounds: Int, layers k: Int, gguf: String, ple: String) throws -> Bool {
    setvbuf(stdout, nil, _IOLBF, 0)
    let tokens = try String(contentsOfFile: prompt, encoding: .utf8).split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    let engine = try Qwen38Engine(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                                  ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                                  capacity: tokens.count + 8, prefillChunk: 2048, speculative: false)
    let runner = engine.runner
    let layers = 0..<k
    let cutBytes = runner.recurrentBytes(layers: layers)
    // The weight views are made as experts are first used (the MBP maps them), so they are taken out.
    func allocated() -> Double {
        let report = runner.memoryReport()
        let views = report.filter { $0.0.contains("views") }.reduce(0) { $0 + $1.1 }
        return Double((report.first { $0.0 == "metal allocated" }?.1 ?? 0) - views) / 1e9
    }
    print(String(format: "Qwen3.8 cut leak: %d tokens, layers 0..<%d, one cut at %d, %.1f MiB per cut, allocated less views %.2f GB",
                 tokens.count, k, tokens.count / 2, Double(cutBytes) / 1_048_576, allocated()))
    runner.layerRange = layers
    runner.trunkOnly = true
    var first = 0.0, last = 0.0
    for r in 1...rounds {
        runner.cutRows = [tokens.count / 2]
        _ = try runner.forward(tokens: tokens, startPos: 0)
        var read = 0
        try runner.forEachCutRegion(0, layers: layers) { _, n in read += n }
        runner.cutRows = []
        runner.releaseCutBuffers()
        runner.shrinkBatchAfterRound()
        last = allocated()
        if r == 1 { first = last }
        print(String(format: "  round %d: allocated less views %.2f GB (cut regions %.1f MiB)", r, last, Double(read) / 1_048_576))
    }
    let grown = last - first
    let ok = grown * 1e9 <= Double(cutBytes)
    print(String(format: "  grown %.2f GB over %d rounds%@", grown, rounds - 1, ok ? "" : "  <-- FAIL"))
    return ok
}
