import Foundation
import Tsugumi

// `--qwen38-branch-probe <manifest.json>`: how the next token at one decision point moves when only the tail of the
// prompt changes (docs/qwen38/25). The trunk is prefilled once with a checkpoint at every variant's divergence from
// it; each variant restores its checkpoint, runs only its own tail, and reports P(`<tool_call>`) as the first token
// (softmax at the sampler's temperature 0.7, before top-k / top-p), the top 5 ids, and `samples` short draws with the
// official sampler.
//
// manifest: {"trunk": "<token file>", "samples": 4, "sampleTokens": 16, "capacity": 16384, "cuts": [2018, …],
//            "variants": [{"label": "…", "tokens": "<token file>"}, …]}
// Token files are comma-separated ids. Prints one JSON line per variant.

private struct BranchManifest: Decodable {
    struct Variant: Decodable { let label: String; let tokens: String }
    let trunk: String
    let samples: Int?
    let sampleTokens: Int?
    /// The engine's capacity (default: the longest prompt + the draws); the app's is its context.
    let capacity: Int?
    /// Extra chunk boundaries in the trunk's prefill, e.g. where the app resumed each round.
    let cuts: [Int]?
    let variants: [Variant]
}

func runQwen38BranchProbe(manifest path: String, chunk: Int, gguf: String, ple: String) throws {
    setvbuf(stdout, nil, _IOLBF, 0)
    let base = URL(fileURLWithPath: path).deletingLastPathComponent()
    func load(_ file: String) throws -> [Int32] {
        let url = file.hasPrefix("/") ? URL(fileURLWithPath: file) : base.appendingPathComponent(file)
        return try String(contentsOf: url, encoding: .utf8).split(separator: ",")
            .compactMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    }
    let manifest = try JSONDecoder().decode(BranchManifest.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    let trunk = try load(manifest.trunk)
    let variants = try manifest.variants.map { ($0.label, try load($0.tokens)) }
    let samples = manifest.samples ?? 4, sampleTokens = manifest.sampleTokens ?? 16
    let toolCall = 248_058
    let stops: Set<Int32> = [248_046, 248_044]

    // Each variant resumes from its longest common prefix with the trunk, one short of its own end so it has a tail.
    let cuts = variants.map { _, tokens in
        min(zip(trunk, tokens).prefix { $0 == $1 }.count, tokens.count - 1, trunk.count - 1)
    }
    precondition(cuts.allSatisfy { $0 > 0 }, "every variant must share a prefix with the trunk")
    let longest = max(trunk.count, variants.map(\.1.count).max()!)
    let engine = try Qwen38Engine(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                                  ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                                  capacity: max(manifest.capacity ?? 0, longest + sampleTokens + 8), prefillChunk: chunk, speculative: false)
    var checkpoints: [Int: Qwen38Checkpoint] = [:]
    engine.reset()
    let t0 = Date()
    let prefix = Array(trunk[..<(cuts.max()! + 1)])
    _ = try engine.runCompletion(promptTokens: prefix, cachedPromptTokens: 0, maxNewTokens: 0, stopTokens: stops,
                                 constraint: nil, greedy: true, seed: 1, checkpointsAt: Array(Set(cuts + (manifest.cuts ?? []))),
                                 onCheckpoint: { checkpoints[$0.position] = $0 })
    print(String(format: "{\"trunk\": %d, \"prefilled\": %d, \"capacity\": %d, \"checkpoints\": %d, \"prefill_s\": %.1f}",
                 trunk.count, prefix.count, engine.capacity, checkpoints.count, Date().timeIntervalSince(t0)))

    // A checkpoint holds the recurrent state only; the KV rows before its position must still be the trunk's. A tail
    // written from a lower checkpoint overwrites the rows a higher one relies on, so the highest cuts go first.
    for ((label, tokens), cut) in zip(variants, cuts).sorted(by: { $0.1 > $1.1 }) {
        guard let checkpoint = checkpoints[cut] else { print("{\"label\": \"\(label)\", \"error\": \"no checkpoint at \(cut)\"}"); continue }
        let tail = Array(tokens[cut...])
        let t = Date()
        try engine.restore(checkpoint)
        var logits = [Float]()
        var done = 0
        while done < tail.count {
            let n = min(chunk, tail.count - done)
            logits = Array(try engine.runner.forward(tokens: tail[done..<(done + n)].map { Int($0) }, startPos: cut + done))
            done += n
        }
        let V = engine.runner.vocab
        let scaled = logits.prefix(V).map { Double($0) / 0.7 }
        let top = scaled.max()!
        let z = scaled.reduce(0) { $0 + exp($1 - top) }
        let probability = { (id: Int) in exp(scaled[id] - top) / z }
        let ranked = (0..<V).sorted { scaled[$0] > scaled[$1] }.prefix(5)
        var draws: [[Int32]] = []
        for seed in 1...max(samples, 1) where samples > 0 {
            try engine.restore(checkpoint)
            let run = try engine.runCompletion(promptTokens: tail, cachedPromptTokens: cut, maxNewTokens: sampleTokens,
                                               stopTokens: stops, constraint: nil, greedy: false, seed: UInt64(seed))
            draws.append(run.tokens)
        }
        let topText = ranked.map { String(format: "[%d, %.4f]", $0, probability($0)) }.joined(separator: ", ")
        let drawText = draws.map { "[" + $0.map(String.init).joined(separator: ",") + "]" }.joined(separator: ", ")
        print(String(format: "{\"label\": \"%@\", \"tokens\": %d, \"from\": %d, \"p_tool_call\": %.4f, \"top\": [%@], \"draws\": [%@], \"s\": %.1f}",
                     label as NSString, tokens.count, cut, probability(toolCall), topText as NSString,
                     drawText as NSString, Date().timeIntervalSince(t)))
    }
}

// `--qwen38-path-check <token file> --q38-cuts a,b,… --q38-resume-at N`: the last position's logits of one prompt
// through different prefill paths (docs/qwen38/25 §2). Paths: straight in 2,048 chunks twice (run-to-run), the same
// with the capacity at 16,384, chunks cut where the app resumed each round (`--q38-cuts`), and a checkpoint at
// `--q38-resume-at` restored before the tail (the branch probe's path), at both capacities. Prints each path against
// the first: max |Δlogit| over the vocabulary, over the first path's top 20, the KL divergence at temperature 0.7,
// the top-1 id and P(`<tool_call>`).
func runQwen38PathCheck(tokenFile: String, cuts appCuts: [Int], resumeAt: Int, gguf: String, ple: String) throws {
    setvbuf(stdout, nil, _IOLBF, 0)
    let tokens = try String(contentsOfFile: tokenFile, encoding: .utf8).split(separator: ",")
        .compactMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    let stops: Set<Int32> = [248_046, 248_044]
    precondition(resumeAt > 0 && resumeAt < tokens.count)
    var first: [Float]?
    func report(_ label: String, _ logits: [Float], seconds: Double) {
        let V = 248_320
        let l = Array(logits.prefix(V))
        func softmax(_ x: [Float]) -> [Double] {
            let s = x.map { Double($0) / 0.7 }; let m = s.max()!
            let z = s.reduce(0) { $0 + exp($1 - m) }
            return s.map { exp($0 - m) / z }
        }
        let p = softmax(l)
        let top1 = l.indices.max { l[$0] < l[$1] }!
        var line = String(format: "%-26@ top1 %6d  P(tool_call) %.4f  %.0f s", label as NSString, top1, p[248_058], seconds)
        if let base = first {
            let q = softmax(base)
            let maxAll = zip(l, base).map { abs($0 - $1) }.max()!
            let top20 = base.indices.sorted { base[$0] > base[$1] }.prefix(20)
            let maxTop = top20.map { abs(l[$0] - base[$0]) }.max()!
            let kl = zip(q, p).reduce(0.0) { $0 + ($1.0 > 0 ? $1.0 * log($1.0 / max($1.1, 1e-300)) : 0) }
            line += String(format: "  max|Δ| %.3f  top20 %.3f  KL %.4f", maxAll, maxTop, kl)
        } else {
            first = l
        }
        print(line)
    }
    func engine(_ capacity: Int) throws -> Qwen38Engine {
        try Qwen38Engine(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                         ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                         capacity: capacity, prefillChunk: 2048, speculative: false)
    }
    /// Prefill with chunk boundaries at `at`; with `resume`, stop at it, checkpoint, restore and run the tail.
    func path(_ e: Qwen38Engine, at: [Int], resume: Int?) throws -> [Float] {
        e.reset()
        var logits = [Float]()
        let end = resume ?? (tokens.count - 1)
        var checkpoint: Qwen38Checkpoint?
        _ = try e.runCompletion(promptTokens: Array(tokens[..<end]), cachedPromptTokens: 0, maxNewTokens: 0,
                                stopTokens: stops, constraint: nil, greedy: true, seed: 1,
                                checkpointsAt: at.filter { $0 < end } + (resume == nil ? [] : [end - 1]),
                                onCheckpoint: { if $0.position == end - 1 { checkpoint = $0 } })
        if resume != nil {
            // Back one token from the stopping point, the way the probe resumes a variant from a shared prefix.
            try e.restore(checkpoint!)
            logits = Array(try e.runner.forward(tokens: tokens[(end - 1)...].map { Int($0) }, startPos: end - 1))
        } else {
            logits = Array(try e.runner.forward(tokens: [Int(tokens.last!)], startPos: end))
        }
        return logits
    }
    print("Qwen3.8 path check: \(tokens.count) tokens, cuts \(appCuts), resume at \(resumeAt)")
    // One engine at a time: each holds its own buffers.
    do {
        let small = try engine(tokens.count + 32)
        var t = Date()
        report("straight", try path(small, at: [], resume: nil), seconds: Date().timeIntervalSince(t))
        t = Date(); report("straight again", try path(small, at: [], resume: nil), seconds: Date().timeIntervalSince(t))
        t = Date(); report("app cuts", try path(small, at: appCuts, resume: nil), seconds: Date().timeIntervalSince(t))
        t = Date(); report("checkpoint", try path(small, at: [], resume: resumeAt), seconds: Date().timeIntervalSince(t))
    }
    let large = try engine(16_384)
    var t = Date()
    report("16K straight", try path(large, at: [], resume: nil), seconds: Date().timeIntervalSince(t))
    t = Date(); report("16K app cuts + checkpoint", try path(large, at: appCuts, resume: resumeAt), seconds: Date().timeIntervalSince(t))
}

// `--qwen38-restore-check <token file> <trunk file> <position>`: the tail after `position` from three states
// (docs/qwen38/25 §2): live (prefilled to `position`, no checkpoint), restored right away (prefilled to `position` + 1,
// back to `position`), and restored after the trunk ran on past it (the branch probe's path). The trunk shares the
// token file's first `position` tokens.
func runQwen38RestoreCheck(tokenFile: String, trunkFile: String, position: Int, gguf: String, ple: String) throws {
    setvbuf(stdout, nil, _IOLBF, 0)
    func load(_ path: String) throws -> [Int] {
        try String(contentsOfFile: path, encoding: .utf8).split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    }
    let tokens = try load(tokenFile), trunk = try load(trunkFile)
    precondition(Array(tokens[..<position]) == Array(trunk[..<position]), "the trunk must share the first \(position) tokens")
    let stops: Set<Int32> = [248_046, 248_044]
    let e = try Qwen38Engine(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                             ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                             capacity: max(tokens.count, trunk.count) + 32, prefillChunk: 2048, speculative: false)
    func prefill(_ ids: [Int], checkpointAt: Int?) throws -> Qwen38Checkpoint? {
        e.reset()
        var kept: Qwen38Checkpoint?
        _ = try e.runCompletion(promptTokens: ids.map { Int32($0) }, cachedPromptTokens: 0, maxNewTokens: 0,
                                stopTokens: stops, constraint: nil, greedy: true, seed: 1,
                                checkpointsAt: checkpointAt.map { [$0] } ?? [], onCheckpoint: { kept = $0 })
        return kept
    }
    var base: [Float]?
    func report(_ label: String, _ l: [Float], _ t: Date) {
        let s = l.prefix(248_320).map { Double($0) / 0.7 }, m = s.max()!
        let z = s.reduce(0) { $0 + exp($1 - m) }
        var line = String(format: "%-22@ P(tool_call) %.4f  %.0f s", label as NSString, exp(s[248_058] - m) / z, Date().timeIntervalSince(t))
        if let base { line += String(format: "  max|Δ| vs live %.3f", zip(l, base).map { abs($0 - $1) }.max()!) } else { base = l }
        print(line)
    }
    let tail = Array(tokens[position...])
    print("Qwen3.8 restore check: \(tokens.count) tokens, trunk \(trunk.count), position \(position), tail \(tail.count)")
    var t = Date()
    _ = try prefill(Array(tokens[..<position]), checkpointAt: nil)
    report("live", Array(try e.runner.forward(tokens: tail, startPos: position)), t)
    t = Date()
    let near = try prefill(Array(tokens[...position]), checkpointAt: position)!
    try e.restore(near)
    report("restored right away", Array(try e.runner.forward(tokens: tail, startPos: position)), t)
    t = Date()
    let far = try prefill(trunk, checkpointAt: position)!
    try e.restore(far)
    report("restored after trunk", Array(try e.runner.forward(tokens: tail, startPos: position)), t)
    t = Date()
    try e.restore(near)
    report("near again after trunk", Array(try e.runner.forward(tokens: tail, startPos: position)), t)
    // The branch probe draws a few tokens between variants: decode steps, then a restore and a batched forward.
    for (label, greedy) in [("after 12 greedy tokens", true), ("after 12 sampled tokens", false)] {
        t = Date()
        try e.restore(near)
        let run = try e.runCompletion(promptTokens: tail.map { Int32($0) }, cachedPromptTokens: position, maxNewTokens: 12,
                                      stopTokens: stops, constraint: nil, greedy: greedy, seed: 1)
        try e.restore(near)
        report(label + " (\(run.tokens.count))", Array(try e.runner.forward(tokens: tail, startPos: position)), t)
    }
    t = Date()
    try e.restore(near)
    report("again, no decode", Array(try e.runner.forward(tokens: tail, startPos: position)), t)
}
