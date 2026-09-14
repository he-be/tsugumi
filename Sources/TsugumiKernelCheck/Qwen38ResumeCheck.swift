import Foundation
import Tsugumi

// MARK: - Qwen3.8 checkpoints against a whole recompute (docs/qwen38/15 §2 G-0, 18)
//
// `--qwen38-resume <prompt A> <prompt B>` (token files; B is A with a line added to the last user message) greedy,
// `--q38-mtp off|spec`, `--q38-new N`, `--q38-chunk C`:
//   whole   A from position 0, no checkpoints
//   cut     A cut at L = LCP(A, B) and at P - 1, a checkpoint at each and before every `<tool_call>` fed in decode
//   P-1     restore P - 1, feed A's last token
//   L -> B  restore L, feed B[L...], against B from position 0
//   neg     restore P - 1, feed another token (must differ)
//   call    restore before the first `<tool_call>`, feed it, against the rest of `whole`
// Each line says same / DIFFERENT and the run's prompt_n and kv position. Exit 1 when an expected match fails.

func runQwen38Resume(promptA: String, promptB: String, newTokens: Int, chunk: Int, mtp: String,
                     gguf: String, ple: String) throws -> Bool {
    func load(_ path: String) throws -> [Int32] {
        try String(contentsOfFile: path, encoding: .utf8).split(separator: ",")
            .compactMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    }
    setvbuf(stdout, nil, _IOLBF, 0)
    let A = try load(promptA), B = try load(promptB)
    precondition(mtp == "off" || mtp == "spec", "--q38-mtp off|spec")
    let engine = try Qwen38Engine(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                                  ple: URL(fileURLWithPath: (ple as NSString).expandingTildeInPath),
                                  capacity: max(A.count, B.count) + newTokens + 8, prefillChunk: chunk,
                                  speculative: mtp == "spec")
    let stops: Set<Int32> = [248_046, 248_044]
    let marker: Int32 = 248_058
    var L = 0
    while L < min(A.count, B.count) && A[L] == B[L] { L += 1 }
    let P = A.count
    print("Qwen3.8 resume: A \(P) tokens, B \(B.count), LCP \(L), mtp \(mtp), chunk \(chunk), new \(newTokens)")

    var checkpoints: [Qwen38Checkpoint] = []
    func run(_ label: String, _ tokens: ArraySlice<Int32>, cached: Int, new: Int = newTokens,
             at: [Int] = [], before: Set<Int32> = []) throws -> [Int32] {
        let t = Date()
        let r = try engine.runCompletion(promptTokens: Array(tokens), cachedPromptTokens: cached, maxNewTokens: new,
                                         stopTokens: stops, constraint: nil, greedy: true, seed: 1,
                                         checkpointsAt: at, checkpointBefore: before,
                                         onCheckpoint: { checkpoints.append($0) })
        print(String(format: "  %-7@ prompt_n %5d from %5d, %3d tokens, kv %5d, prefill %.2f s, total %.2f s%@",
                     label as NSString, r.promptTokens, cached, r.newTokens, r.kvPosition, r.prefillSeconds,
                     Date().timeIntervalSince(t), mtp == "spec" ? " (accepted \(r.accepted) / \(r.passes))" : ""))
        return r.tokens
    }
    var ok = true
    func expect(_ label: String, _ got: [Int32], _ want: [Int32], same: Bool = true) {
        let equal = got == want
        let n = zip(got, want).prefix { $0 == $1 }.count
        print("  \(label): \(equal ? "same" : "DIFFERENT (first \(n) agree, \(got.count) vs \(want.count))")\(equal == same ? "" : "  <-- FAIL")")
        if equal != same { ok = false }
    }
    func checkpoint(at position: Int) -> Qwen38Checkpoint? { checkpoints.first { $0.position == position } }

    engine.reset()
    let whole = try run("whole", A[...], cached: 0)
    engine.reset()
    let tc = Date()
    let cut = try run("cut", A[...], cached: 0, at: [L, P - 1], before: [marker])
    print(String(format: "  checkpoints at %@ (%.1f MiB each, cut run %.2f s)",
                 checkpoints.map { String(format: "%d (%.0f ms)", $0.position, $0.captureSeconds * 1000) }.joined(separator: ", ") as NSString,
                 Double(engine.runner.recurrentBytes) / 1_048_576, Date().timeIntervalSince(tc)))
    expect("cut == whole", cut, whole)

    if let c = checkpoint(at: P - 1) {
        try engine.restore(c)
        expect("P-1 == whole", try run("P-1", A[(P - 1)...], cached: P - 1), whole)
        try engine.restore(c)
        let other: Int32 = A[P - 1] == 198 ? 220 : 198
        expect("neg != whole", try run("neg", [other][...], cached: P - 1), whole, same: false)
    } else { print("  no checkpoint at P-1  <-- FAIL"); ok = false }

    if L > 0, L < B.count, let c = checkpoint(at: L) {
        try engine.restore(c)
        let resumed = try run("L->B", B[L...], cached: L)
        engine.reset()
        expect("L->B == B whole", resumed, try run("B", B[...], cached: 0))
    } else { print("  no usable L (\(L))") }

    if let k = whole.firstIndex(of: marker), let c = checkpoint(at: P + k) {
        try engine.restore(c)
        let rest = try run("call", [marker][...], cached: P + k, new: whole.count - k - 1)
        expect("call == whole after <tool_call>", rest, Array(whole[(k + 1)...]))
    } else {
        print("  no <tool_call> in whole\(whole.contains(marker) ? " but no checkpoint  <-- FAIL" : "")")
        if whole.contains(marker) { ok = false }
    }
    print(ok ? "  PASS" : "  FAIL")
    return ok
}
