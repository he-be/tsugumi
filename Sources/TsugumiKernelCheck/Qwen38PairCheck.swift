import Foundation
import Tsugumi

// MARK: - Qwen3.8 prefill split by layers over two machines (docs/qfn-prefill/22 §4, 04 Q6)
//
// M6:  --qwen38-pair-serve <bind ip> <port> [--q38-capacity N] [--q38-chunk C] with --q38-gguf, --q38-ple
//      Serves `Qwen38Pair` rounds with its own runner (layers 0 ..< k, k from the MBP).
// MBP: --qwen38-pair-check <prompt token file> --q38-pair <M6 ip>[:port] [--q38-pair-k K] [--q38-tokens N] [--q38-new 64]
//      Three prefills of the prompt's first N tokens through the engine (MTP off), each followed by decode:
//        solo    the MBP alone, then greedy `--q38-new` tokens (the reference tokens)
//        forced  the pair, then the reference tokens fed one by one: every row's logits against solo's
//        free    the pair, then its own greedy tokens (the text is recorded; a difference does not stop the check)
//      Per row: max |diff| / max |solo|, top-1 same, and for a swapped top-1 whether each side's top-1 is in the
//      other's top-5 (04 Q6's bar). Exit 1 when a swapped row fails that.

private func top(_ l: ArraySlice<Float>, _ n: Int) -> [Int] {
    var best: [(Float, Int)] = []
    for (i, v) in zip(l.indices, l) where best.count < n || v > best[best.count - 1].0 {
        best.append((v, i - l.startIndex))
        best.sort { $0.0 > $1.0 }
        if best.count > n { best.removeLast() }
    }
    return best.map { $0.1 }
}

func runQwen38PairServe(ip: String, port: Int, gguf: String, ple: String, capacity: Int, chunk: Int) throws -> Never {
    setvbuf(stdout, nil, _IOLBF, 0)
    let runner = try Qwen38Runner(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                                  ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                                  capacity: capacity, maxBatch: chunk)
    print("Qwen3.8 pair serve: capacity \(capacity), chunk \(chunk), KV \(runner.kvQ8 ? "q8_0" : "f32")")
    try Qwen38Pair.serve(runner: runner, host: ip, port: port) { print($0) }
}

func runQwen38PairCheck(tokenFile: String, pair: String, k: Int, tokens: Int?, newTokens: Int, chunk: Int,
                        gguf: String, ple: String) throws -> Bool {
    setvbuf(stdout, nil, _IOLBF, 0)
    var prompt = try String(contentsOfFile: tokenFile, encoding: .utf8).split(separator: ",")
        .compactMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    if let tokens { prompt = Array(prompt.prefix(tokens)) }
    let n = prompt.count
    let engine = try Qwen38Engine(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                                  ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                                  capacity: n + newTokens + 8, prefillChunk: chunk, speculative: false)
    let runner = engine.runner
    let V = runner.vocab
    let parts = pair.split(separator: ":")
    let config = (host: String(parts[0]), port: parts.count > 1 ? Int(parts[1])! : Qwen38Pair.defaultPort, k: k)
    print("Qwen3.8 pair check: prompt \(n) tokens (\(tokenFile)), chunk \(chunk), k \(k), M6 \(config.host):\(config.port), new \(newTokens)")

    /// Prefill (alone or on the pair) and return its seconds.
    func prefill(pair: Bool) throws -> Double {
        engine.reset()
        engine.pairConfig = pair ? config : nil
        let t = Date()
        _ = try engine.runCompletion(promptTokens: prompt, cachedPromptTokens: 0, maxNewTokens: 1, stopTokens: [],
                                     constraint: nil, greedy: true, seed: 0)
        return Date().timeIntervalSince(t)
    }
    /// Decode from the prompt's end: feeds `forced` if given, else each row's argmax. Returns every row's logits
    /// (the prefill's last row first) and the tokens fed.
    func decode(forced: [Int]?) throws -> ([[Float]], [Int]) {
        var rows = [engine.lastPrefillLogits]
        var fed: [Int] = []
        for i in 0..<newTokens {
            let y = forced?[i] ?? top(rows[i][...], 1)[0]
            fed.append(y)
            rows.append(Array(try runner.forward(tokens: [y], startPos: n + i)))
        }
        return (rows, fed)
    }

    let soloS = try prefill(pair: false)
    let (solo, ref) = try decode(forced: nil)
    print(String(format: "solo: prefill %.2f s", soloS))
    let pairS = try prefill(pair: true)
    guard engine.pairConfig != nil else { print("the pair link failed"); return false }
    let (forced, _) = try decode(forced: ref)
    print(String(format: "forced: prefill %.2f s (pair)", pairS))
    let freeS = try prefill(pair: true)
    let (_, free) = try decode(forced: nil)
    print(String(format: "free: prefill %.2f s (pair)", freeS))

    print("row\tmax|diff|\tmax|solo|\trel\ttop1\tsolo_top1\tpair_top1\tin_top5")
    var same = 0, bad = 0
    var rels: [Double] = []
    for r in 0..<solo.count {
        let a = solo[r], b = forced[r]
        precondition(a.count == V && b.count == V)
        var d: Float = 0, m: Float = 0
        for i in 0..<V { d = max(d, abs(a[i] - b[i])); m = max(m, abs(a[i])) }
        let ta = top(a[...], 5), tb = top(b[...], 5)
        let ok = ta[0] == tb[0]
        let inTop5 = ok || (tb.contains(ta[0]) && ta.contains(tb[0]))
        if ok { same += 1 } else if !inTop5 { bad += 1 }
        rels.append(Double(d / m))
        print(String(format: "%d\t%.4f\t%.3f\t%.2e\t%@\t%d\t%d\t%@", r, d, m, d / m, ok ? "same" : "SWAP", ta[0], tb[0],
                     inTop5 ? "yes" : "NO"))
    }
    rels.sort()
    print(String(format: "rows %d: top-1 same %d, swapped outside top-5 %d; rel diff median %.2e max %.2e",
                 solo.count, same, bad, rels[rels.count / 2], rels.last!))
    print("ref ids: " + ref.map(String.init).joined(separator: ","))
    print("pair ids: " + free.map(String.init).joined(separator: ","))
    let firstDiff = zip(ref, free).enumerated().first { $0.element.0 != $0.element.1 }?.offset
    print("greedy \(newTokens): " + (firstDiff.map { "first difference at token \($0)" } ?? "same"))
    return bad == 0
}
