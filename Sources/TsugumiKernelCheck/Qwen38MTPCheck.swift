import Foundation
import Tsugumi

// MARK: - Qwen3.8 MTP head against the CPU reference (docs/qwen38/10)
//
// `--qwen38-mtp-dump <token file> <out>`: the trunk runs the prompt in one batch with `exportHidden`, then the
// MTP head runs the same tokens, token p paired with the trunk's residual row p - 1 (zeros for p = 0, as
// llama.cpp's first `pending_h`), in batches of `Q38_MTP_CHUNKS` (default 48,1,1,2,3, then ones) so its KV
// continues across the dense sgemm path (T >= 32) and the host lanes. Writes Int32 T, Int32 vocab, T Int32 tokens,
// T x hc x e Float32 input rows, T x vocab Float32 logits, for `Scripts/qwen38/mtp_reference.py`.

func runQwen38MTPDump(tokenFile: String, out: String, tokens maxTokens: Int, gguf: String, ple: String) throws {
    let text = try String(contentsOfFile: tokenFile, encoding: .utf8)
    let ids = Array(text.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }.prefix(maxTokens))
    let T = ids.count
    let runner = try Qwen38Runner(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                                  ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                                  capacity: T + 1, maxBatch: T)
    runner.exportHidden = true
    let W = runner.hc * runner.e
    let t0 = Date()
    _ = try runner.forward(tokens: ids, startPos: 0)
    var hidden = [Float](repeating: 0, count: T * W)
    for p in 1..<T { hidden.withUnsafeMutableBufferPointer { ($0.baseAddress! + p * W).update(from: runner.hidden(row: p - 1), count: W) } }
    print(String(format: "  trunk %d tokens: %.2f s", T, Date().timeIntervalSince(t0)))

    let chunks = (ProcessInfo.processInfo.environment["Q38_MTP_CHUNKS"] ?? "48,1,1,2,3").split(separator: ",").compactMap { Int($0) }
    var logits = [Float]()
    logits.reserveCapacity(T * runner.vocab)
    var start = 0, ci = 0
    while start < T {
        let n = min(ci < chunks.count ? chunks[ci] : 1, T - start)
        ci += 1
        let t = Date()
        let l = try hidden.withUnsafeBufferPointer {
            try runner.mtpForward(tokens: Array(ids[start..<(start + n)]), startPos: start, hidden: $0.baseAddress! + start * W, allLogits: true)
        }
        logits.append(contentsOf: l)
        let pr = runner.lastMTPProfile
        print(String(format: "  mtp [%3d..<%3d] %.0f ms (pre %.0f [gpu %.0f] route %.0f routed %.0f [gpu %.0f] head %.0f, experts %d)",
                     start, start + n, Date().timeIntervalSince(t) * 1000, pr.preRouter * 1000, pr.preGPU * 1000,
                     pr.route * 1000, pr.routed * 1000, pr.routedGPU * 1000, pr.head * 1000, pr.distinctExperts))
        start += n
    }
    var data = Data()
    withUnsafeBytes(of: Int32(T)) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: Int32(runner.vocab)) { data.append(contentsOf: $0) }
    ids.map { Int32($0) }.withUnsafeBytes { data.append(contentsOf: $0) }
    hidden.withUnsafeBytes { data.append(contentsOf: $0) }
    logits.withUnsafeBytes { data.append(contentsOf: $0) }
    try data.write(to: URL(fileURLWithPath: out))
    var agree = 0
    for p in 0..<(T - 1) {
        let row = logits[(p * runner.vocab)..<((p + 1) * runner.vocab)]
        if row.indices.max(by: { row[$0] < row[$1] })! - row.startIndex == ids[p + 1] { agree += 1 }
    }
    // Row p (token p, h_{p-1}) drafts token p + 1: how often its argmax is the prompt's next token.
    print("  draft argmax == prompt's next token at \(agree) of \(T - 1) positions (from row 1: h_{p-1} real)")
    print("  written \(out)")
}

// MARK: - Generation (docs/qwen38/10 §5-1, §5-3)

/// The trunk forward's route and routed host time split (docs/qwen38/11): route = top-k + views + advise (+ the
/// `Q38_COUNT_MISS` mincore time, printed apart), routed = commit->kernel start + kernel->GPU start + GPU + after.
func routeDetail(_ p: Qwen38Runner.StepProfile) -> String {
    String(format: "  | route topk %.1f views %.1f advise %.1f (miss %.1f) calls %d | routed commit>kernel %.1f kernel>gpu %.1f (behind shared %.1f, shared gpu %.1f) gpu %.1f after %.1f | experts %d miss %.1f MB new views %.1f MB",
           p.routeTopK * 1000, p.routeViews * 1000, (p.routeAdvise - p.missTime) * 1000, p.missTime * 1000, p.adviseCalls,
           p.routedToKernel * 1000, p.routedKernelToGPU * 1000, p.routedBehindShared * 1000, p.sharedGPU * 1000, p.routedGPU * 1000, p.routedAfterGPU * 1000,
           p.distinctExperts, Double(p.missBytes) / 1e6, Double(p.newViewBytes) / 1e6)
        + (p.previewActual > 0 ? String(format: " | preview hit %d/%d named %d host %.1f",
                                        p.previewHit, p.previewActual, p.previewNamed, p.previewAdvise * 1000) : "")
}

/// Host sampler for the generation checks. `greedy`: argmax (lowest id on ties). `instruct`: the official non-thinking
/// settings (temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5 over the generated tokens, `HANDOVER-llm-server.md`),
/// in Hugging Face's order: penalty, temperature, top_k, top_p over the tempered top-k mass, draw (xorshift64*).
struct Q38Sampler {
    let greedy: Bool
    let temperature: Float = 0.7, topK = 20, topP: Float = 0.8, presence: Float = 1.5
    var state: UInt64
    var seen = Set<Int>()

    init(greedy: Bool, seed: UInt64) {
        self.greedy = greedy
        state = seed &* 0x9E37_79B9_7F4A_7C15 | 1
    }

    static func argmax(_ l: UnsafeBufferPointer<Float>) -> Int {
        var best = 0
        for i in 1..<l.count where l[i] > l[best] { best = i }
        return best
    }

    mutating func sample(_ l: UnsafeBufferPointer<Float>) -> Int {
        if greedy { return Q38Sampler.argmax(l) }
        var top: [(id: Int, z: Float)] = []
        top.reserveCapacity(topK + 1)
        for i in 0..<l.count {
            let z = seen.contains(i) ? l[i] - presence : l[i]
            if top.count == topK, z <= top[topK - 1].z { continue }
            var at = top.count
            while at > 0 && top[at - 1].z < z { at -= 1 }
            top.insert((i, z), at: at)
            if top.count > topK { top.removeLast() }
        }
        let zMax = top[0].z
        var p = top.map { Double(exp(($0.z - zMax) / temperature)) }
        let sum = p.reduce(0, +)
        p = p.map { $0 / sum }
        var keep = 0, acc = 0.0
        while keep < p.count { acc += p[keep]; keep += 1; if acc >= Double(topP) { break } }
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27
        let u = Double((state &* 0x2545_F491_4F6C_DD1D) >> 11) / Double(1 << 53) * acc
        var run = 0.0
        var id = top[0].id
        for j in 0..<keep { run += p[j]; if u < run { id = top[j].id; break } }
        seen.insert(id)
        return id
    }
}

/// `--qwen38-generate <prompt token file>`: prefill in chunks of `--q38-chunk`, then up to `--q38-new` tokens with
/// `--q38-sampler greedy|instruct` (`--q38-seed`). `--q38-mtp off|shadow`: shadow also runs the MTP head over the
/// prompt (in batches of `Q38_MTP_CHUNK`, default 256) and, before every decode step, drafts the next token from
/// (the token being fed, the previous position's residual) and throws the draft away, so the acceptance
/// (draft argmax == the token the target then samples) and the draft cost are measured without a rollback.
/// `spec`: the speculative loop with one draft (§5-4): draft from the MTP head (after it takes the previous step's
/// accepted row), verify [token, draft] in one T = 2 forward, keep the draft if the target's sample at the first row
/// equals it (and sample the second row), else roll the recurrent state back to after the first token.
/// `reject`: the same verify pass with every draft rejected, the neutrality control (its tokens must equal `off`).
/// Writes `ids: ...` and one line per token (or step) to `--q38-out`.
func runQwen38Generate(tokenFile: String, newTokens: Int, chunk: Int, greedy: Bool, seed: UInt64, mtp: String,
                       out: String, gguf: String, ple: String) throws {
    let text = try String(contentsOfFile: tokenFile, encoding: .utf8)
    let prompt = text.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    setvbuf(stdout, nil, _IOLBF, 0)
    let shadow = mtp == "shadow"
    let spec = mtp == "spec" || mtp == "reject"
    precondition(mtp == "off" || shadow || spec, "--q38-mtp off|shadow|spec|reject")
    let useMTP = shadow || spec
    let n = prompt.count
    let runner = try Qwen38Runner(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                                  ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                                  capacity: n + newTokens + 2, maxBatch: chunk)
    runner.exportHidden = useMTP
    let W = runner.hc * runner.e
    let mtpChunk = Int(ProcessInfo.processInfo.environment["Q38_MTP_CHUNK"] ?? "") ?? 256
    let eos: Set<Int> = [248_046, 248_044]
    print("Qwen3.8 generate: prompt \(n) tokens (\(tokenFile)), chunk \(chunk), sampler \(greedy ? "greedy" : "instruct") seed \(seed), mtp \(mtp)")

    var pendingH = [Float](repeating: 0, count: W)
    var lastLogits = [Float]()
    var trunkPrefill = 0.0, mtpPrefill = 0.0
    var start = 0
    while start < n {
        let T = min(chunk, n - start)
        let t0 = Date()
        lastLogits = Array(try runner.forward(tokens: Array(prompt[start..<(start + T)]), startPos: start))
        let dt = Date().timeIntervalSince(t0)
        trunkPrefill += dt
        var line = String(format: "  prefill [%5d..<%5d] %.2f s", start, start + T, dt)
        if useMTP {
            let t1 = Date()
            var s = 0
            while s < T {
                let m = min(mtpChunk, T - s)
                var rows = [Float](repeating: 0, count: m * W)
                rows.withUnsafeMutableBufferPointer { b in
                    for r in 0..<m {
                        if start + s + r == start { (b.baseAddress! + r * W).update(from: pendingH, count: W) }
                        else { (b.baseAddress! + r * W).update(from: runner.hidden(row: s + r - 1), count: W) }
                    }
                }
                _ = try rows.withUnsafeBufferPointer {
                    try runner.mtpForward(tokens: Array(prompt[(start + s)..<(start + s + m)]), startPos: start + s, hidden: $0.baseAddress!)
                }
                s += m
            }
            pendingH = Array(UnsafeBufferPointer(start: runner.hidden(row: T - 1), count: W))
            let dm = Date().timeIntervalSince(t1)
            mtpPrefill += dm
            line += String(format: ", mtp %.2f s", dm)
        }
        print(line)
        start += T
    }
    print(String(format: "  prefill %d tokens: trunk %.1f s%@", n, trunkPrefill, useMTP ? String(format: ", mtp %.1f s", mtpPrefill) : ""))
    if spec {
        try speculativeLoop(runner: runner, n: n, lastLogits: lastLogits, pendingH: pendingH, newTokens: newTokens,
                            greedy: greedy, seed: seed, forceReject: mtp == "reject", out: out)
        return
    }

    var sampler = Q38Sampler(greedy: greedy, seed: seed)
    var generated: [Int] = []
    var y = lastLogits.withUnsafeBufferPointer { sampler.sample($0) }
    generated.append(y)
    var lines: [String] = []
    var trunkMs: [Double] = [], mtpMs: [Double] = []
    var accepted = 0, drafted = 0
    var pos = n
    while generated.count < newTokens && !eos.contains(y) {
        var draft = -1
        var mtpLine = ""
        if shadow {
            let t = Date()
            let l = try pendingH.withUnsafeBufferPointer { try runner.mtpForward(tokens: [y], startPos: pos, hidden: $0.baseAddress!) }
            draft = Q38Sampler.argmax(l)
            let ms = Date().timeIntervalSince(t) * 1000
            mtpMs.append(ms)
            let pr = runner.lastMTPProfile
            mtpLine = String(format: "  mtp %.0f (pre %.0f [gpu %.0f] route %.0f routed %.0f [gpu %.0f] head %.0f)",
                             ms, pr.preRouter * 1000, pr.preGPU * 1000, pr.route * 1000, pr.routed * 1000, pr.routedGPU * 1000, pr.head * 1000)
        }
        let t = Date()
        let l = try runner.forward(tokens: [y], startPos: pos)
        let ms = Date().timeIntervalSince(t) * 1000
        trunkMs.append(ms)
        let pr = runner.lastProfile
        let next = sampler.sample(l)
        if shadow {
            pendingH = Array(UnsafeBufferPointer(start: runner.hidden(row: 0), count: W))
            drafted += 1
            if draft == next { accepted += 1 }
        }
        lines.append(String(format: "  [%5d] in %6d out %6d logit %.6f%@  trunk %.0f (pre %.0f [gpu %.0f] route %.0f routed %.0f [gpu %.0f] head %.0f)%@",
                            pos, y, next, l[next], shadow ? String(format: " draft %6d %@", draft, draft == next ? "hit " : "miss") : "",
                            ms, pr.preRouter * 1000, pr.preGPU * 1000, pr.route * 1000, pr.routed * 1000, pr.routedGPU * 1000,
                            pr.head * 1000, mtpLine) + routeDetail(pr))
        print(lines.last!)
        generated.append(next)
        y = next
        pos += 1
    }
    func med(_ v: [Double]) -> Double { let s = v.sorted(); return s.isEmpty ? .nan : s[s.count / 2] }
    // The first decode step after a prefill pages the dense weights back in (docs/qwen38/09 §2-1): reported apart.
    let steady = Array(trunkMs.dropFirst())
    var summary = String(format: "  decode %d steps: first %.0f ms, steady median %.0f ms (mean %.0f), %.2f tok/s steady",
                         trunkMs.count, trunkMs.first ?? .nan, med(steady), steady.reduce(0, +) / Double(max(steady.count, 1)),
                         1000 * Double(steady.count) / max(steady.reduce(0, +), 1e-9))
    if shadow {
        let mSteady = Array(mtpMs.dropFirst())
        summary += String(format: "\n  shadow: accepted %d / %d = %.3f, draft first %.0f ms, steady median %.0f ms (mean %.0f)",
                          accepted, drafted, Double(accepted) / Double(max(drafted, 1)), mtpMs.first ?? .nan,
                          med(mSteady), mSteady.reduce(0, +) / Double(max(mSteady.count, 1)))
    }
    print(summary)
    let body = "ids: " + generated.map(String.init).joined(separator: ",") + "\n" + lines.joined(separator: "\n") + "\n" + summary + "\n"
    try body.write(toFile: out, atomically: true, encoding: .utf8)
    print("  written \(out)")
}

private func speculativeLoop(runner: Qwen38Runner, n: Int, lastLogits: [Float], pendingH h: [Float], newTokens: Int,
                             greedy: Bool, seed: UInt64, forceReject: Bool, out: String) throws {
    let V = runner.vocab, W = runner.hc * runner.e
    let eos: Set<Int> = [248_046, 248_044]
    var sampler = Q38Sampler(greedy: greedy, seed: seed)
    var pendingH = h
    var generated: [Int] = []
    var y = lastLogits.withUnsafeBufferPointer { sampler.sample($0) }
    generated.append(y)
    var mtpTokens: [Int] = []       // accepted rows the MTP KV has not taken yet, paired with...
    var mtpRows: [Float] = []       // ...the trunk residual of the position before each
    var pos = n
    var lines: [String] = []
    var stepMs: [Double] = [], stepTokens: [Int] = []
    var draftMs: [Double] = [], verifyMs: [Double] = [], rollbackMs: [Double] = []
    var accepted = 0, steps = 0
    loop: while generated.count < newTokens && !eos.contains(y) {
        let tStep = Date()
        let pos0 = pos, draftT = mtpTokens.count + 1
        let hidden = mtpRows + pendingH
        let dl = try hidden.withUnsafeBufferPointer {
            try runner.mtpForward(tokens: mtpTokens + [y], startPos: pos - mtpTokens.count, hidden: $0.baseAddress!)
        }
        let d = Q38Sampler.argmax(dl)
        let dMs = Date().timeIntervalSince(tStep) * 1000
        let mp = runner.lastMTPProfile
        let tVerify = Date()
        runner.snapshotFirst = true
        let logits = Array(try runner.forward(tokens: [y, d], startPos: pos, allLogits: true))
        runner.snapshotFirst = false
        let vMs = Date().timeIntervalSince(tVerify) * 1000
        let vp = runner.lastProfile
        let h0 = Array(UnsafeBufferPointer(start: runner.hidden(row: 0), count: W))
        let h1 = Array(UnsafeBufferPointer(start: runner.hidden(row: 1), count: W))
        let t0 = logits.withUnsafeBufferPointer { sampler.sample(UnsafeBufferPointer(rebasing: $0[0..<V])) }
        let logit0 = logits[t0]
        var emitted = [t0]
        var rMs = 0.0
        let hit = t0 == d
        if hit && !forceReject {
            accepted += 1
            if !eos.contains(d) && generated.count + 1 < newTokens {
                let t1 = logits.withUnsafeBufferPointer { sampler.sample(UnsafeBufferPointer(rebasing: $0[V..<(2 * V)])) }
                emitted.append(t1)
            }
            mtpTokens = [d]
            mtpRows = h0
            pendingH = h1
            pos += 2
        } else {
            let tr = Date()
            runner.rollbackToFirst()
            rMs = Date().timeIntervalSince(tr) * 1000
            mtpTokens = []
            mtpRows = []
            pendingH = h0
            pos += 1
        }
        steps += 1
        draftMs.append(dMs); verifyMs.append(vMs); rollbackMs.append(rMs)
        stepMs.append(Date().timeIntervalSince(tStep) * 1000)
        stepTokens.append(emitted.count)
        lines.append(String(format: "  [%5d] in %6d draft %6d %@ out %@ logit %.6f  step %.0f = draft %.0f (T=%d pre %.0f route %.0f routed %.0f head %.0f) + verify %.0f (pre %.0f [gpu %.0f] route %.0f routed %.0f [gpu %.0f] head %.0f) + rollback %.1f",
                            pos0, y, d, hit ? "hit " : "miss", emitted.map(String.init).joined(separator: ",") as NSString, logit0,
                            stepMs.last!, dMs, draftT, mp.preRouter * 1000, mp.route * 1000,
                            mp.routed * 1000, mp.head * 1000, vMs, vp.preRouter * 1000, vp.preGPU * 1000, vp.route * 1000,
                            vp.routed * 1000, vp.routedGPU * 1000, vp.head * 1000, rMs) + routeDetail(vp))
        print(lines.last!)
        for t in emitted {
            generated.append(t)
            if eos.contains(t) { y = t; break loop }
        }
        y = emitted.last!
    }
    func med(_ v: [Double]) -> Double { let s = v.sorted(); return s.isEmpty ? .nan : s[s.count / 2] }
    let steadyMs = stepMs.dropFirst().reduce(0, +), steadyTokens = stepTokens.dropFirst().reduce(0, +)
    let summary = String(format: "  %@ %d steps, %d tokens: accepted %d / %d = %.3f, tokens/step %.2f; first step %.0f ms; steady %.2f tok/s (%d tokens in %.1f s); median step %.0f ms = draft %.0f + verify %.0f + rollback %.1f",
                         (forceReject ? "reject" : "spec") as NSString, steps, stepTokens.reduce(0, +), accepted, steps,
                         Double(accepted) / Double(max(steps, 1)), Double(stepTokens.reduce(0, +)) / Double(max(steps, 1)),
                         stepMs.first ?? .nan, 1000 * Double(steadyTokens) / max(steadyMs, 1e-9), steadyTokens, steadyMs / 1000,
                         med(stepMs), med(draftMs), med(verifyMs), med(rollbackMs))
    print(summary)
    let body = "ids: " + generated.map(String.init).joined(separator: ",") + "\n" + lines.joined(separator: "\n") + "\n" + summary + "\n"
    try body.write(toFile: out, atomically: true, encoding: .utf8)
    print("  written \(out)")
}
