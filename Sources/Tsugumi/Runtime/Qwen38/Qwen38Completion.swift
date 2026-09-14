import Foundation

/// What one Qwen3.8 completion produced, in `QwenGreedyRun`'s words (RSP-3 `prompt_n` / `cache_n` / `kvPosition`).
package struct Qwen38Run: Sendable {
    /// The generated tokens, the stop token included when one ended the run.
    package let tokens: [Int32]
    package let promptTokens: Int
    package let cachedPromptTokens: Int
    package let prefillSeconds: Double
    package let decodeSeconds: Double
    package let timeToFirstTokenSeconds: Double
    package let reason: StopReason
    /// The trunk position when the run returned: how many tokens the state holds. The loop feeds token t to draw
    /// t + 1, so this is usually one short of prompt + generated; a stop string that lands on a kept draft leaves it
    /// equal.
    package let kvPosition: Int
    /// Speculative only: verify passes and the drafts they kept (n_max 1, one draft a pass).
    package let passes: Int
    package let accepted: Int

    package var newTokens: Int { tokens.count }
}

/// The Qwen3.8 generation loop over `Qwen38Runner` for the server and the app: chunked prefill, then plain decode or
/// the MTP loop with one draft (`docs/qwen38/10` §5-4, n_max 1 = the default of `14`), the host sampler, a grammar
/// gate, and the one state a family with a recurrent state can continue from without a checkpoint — its own live
/// position (`docs/qwen38/15` §2 G). The loop is the one `--qwen38-generate` measures (`Qwen38MTPCheck.swift`), with
/// the callbacks a server needs.
package final class Qwen38Engine {
    package let runner: Qwen38Runner
    package let speculative: Bool
    package let prefillChunk: Int
    package var capacity: Int { runner.capacity }
    /// Trunk positions the state holds.
    package private(set) var position = 0
    /// The MTP head's input for the token at `position`: the trunk residual of `position - 1` (zeros at 0, as
    /// llama.cpp's first `pending_h`).
    private var pendingH: [Float]
    /// Kept drafts whose rows the trunk holds but the MTP KV does not yet, each with the trunk residual of the position
    /// before it (`14` §4).
    private var mtpTokens: [Int] = []
    private var mtpRows: [Float] = []
    private let mtpChunk = Int(ProcessInfo.processInfo.environment["Q38_MTP_CHUNK"] ?? "") ?? 256

    package init(gguf: URL, ple: URL, capacity: Int, prefillChunk: Int, speculative: Bool) throws {
        runner = try Qwen38Runner(gguf: gguf, ple: ple, capacity: capacity, maxBatch: prefillChunk)
        runner.exportHidden = speculative
        self.speculative = speculative
        self.prefillChunk = prefillChunk
        pendingH = [Float](repeating: 0, count: runner.hc * runner.e)
    }

    package func reset() {
        runner.reset()
        position = 0
        pendingH = [Float](repeating: 0, count: runner.hc * runner.e)
        mtpTokens = []
        mtpRows = []
    }

    /// Prefill `promptTokens` from `position` (= `cachedPromptTokens`) and generate up to `maxNewTokens`.
    /// `onPrefill(done, total)` after every chunk; `onToken(index, id)` for every emitted token, before the next draw
    /// (a caller that suppresses the grammar inside a thought block sets it there). `shouldStop` is asked after each
    /// token. Throws `CancellationError` between chunks and steps; the state is then unnamed and the caller resets.
    package func runCompletion(promptTokens: [Int32],
                               cachedPromptTokens: Int,
                               maxNewTokens: Int,
                               stopTokens: Set<Int32>,
                               constraint: (any GenerationConstraint)?,
                               greedy: Bool,
                               seed: UInt64,
                               shouldStop: () -> Bool = { false },
                               onPrefill: ((Int, Int) -> Void)? = nil,
                               onToken: ((Int, Int32) throws -> Void)? = nil) throws -> Qwen38Run {
        precondition(!promptTokens.isEmpty, "the prompt must have at least one token")
        precondition(cachedPromptTokens == position,
                     "a resumed run must name the state it continues (cached \(cachedPromptTokens), state \(position))")
        precondition(position + promptTokens.count + maxNewTokens + (speculative ? 1 : 0) <= capacity,
                     "prompt + generation exceeds the capacity \(capacity)")
        let gate = constraint.map { ConstraintGate(constraint: $0, endOfGenerationTokenIDs: stopTokens) }
        var sampler = Qwen38Sampler(greedy: greedy, seed: seed)
        let started = Date()
        let prompt = promptTokens.map { Int($0) }
        let W = runner.hc * runner.e

        // Prefill. The MTP head takes the rows it has not taken yet first, then each chunk paired with the trunk
        // residual of the position before each token.
        if speculative && !mtpTokens.isEmpty {
            _ = try mtpRows.withUnsafeBufferPointer {
                try runner.mtpForward(tokens: mtpTokens, startPos: position - mtpTokens.count, hidden: $0.baseAddress!)
            }
            mtpTokens = []
            mtpRows = []
        }
        var lastLogits = [Float]()
        var done = 0
        while done < prompt.count {
            try Task.checkCancellation()
            let T = min(prefillChunk, prompt.count - done)
            let chunk = Array(prompt[done..<(done + T)])
            let l = try runner.forward(tokens: chunk, startPos: position)
            if done + T == prompt.count { lastLogits = Array(l) }
            if speculative {
                var s = 0
                while s < T {
                    let m = min(mtpChunk, T - s)
                    var rows = [Float](repeating: 0, count: m * W)
                    rows.withUnsafeMutableBufferPointer { b in
                        for r in 0..<m {
                            let dst = b.baseAddress! + r * W
                            if s + r == 0 { dst.update(from: pendingH, count: W) }
                            else { dst.update(from: runner.hidden(row: s + r - 1), count: W) }
                        }
                    }
                    _ = try rows.withUnsafeBufferPointer {
                        try runner.mtpForward(tokens: Array(chunk[s..<(s + m)]), startPos: position + s, hidden: $0.baseAddress!)
                    }
                    s += m
                }
                pendingH = Array(UnsafeBufferPointer(start: runner.hidden(row: T - 1), count: W))
            }
            position += T
            done += T
            onPrefill?(done, prompt.count)
        }
        let prefillSeconds = Date().timeIntervalSince(started)
        let decodeStart = Date()

        var produced: [Int32] = []
        var reason: StopReason = .maxTokens
        var firstToken = 0.0
        var passes = 0, accepted = 0
        /// Appends and reports one token; false when the run ends on it.
        func emit(_ id: Int) throws -> Bool {
            let token = Int32(id)
            produced.append(token)
            if produced.count == 1 { firstToken = Date().timeIntervalSince(started) }
            try gate?.accept(token)
            try onToken?(produced.count - 1, token)
            if stopTokens.contains(token) { reason = .endOfTurn; return false }
            if shouldStop() { reason = .stopString; return false }
            return produced.count < maxNewTokens
        }

        var y = try lastLogits.withUnsafeBufferPointer { try sampler.sample($0, gate: gate, position: 0) }
        if maxNewTokens > 0, try emit(y) {
            if speculative {
                let V = runner.vocab
                loop: while true {
                    try Task.checkCancellation()
                    let draft = Qwen38Sampler.argmax(try (mtpRows + pendingH).withUnsafeBufferPointer {
                        try runner.mtpForward(tokens: mtpTokens + [y], startPos: position - mtpTokens.count,
                                              hidden: $0.baseAddress!)
                    })
                    runner.snapshotRows = 1
                    let verify: [Float]
                    do {
                        defer { runner.snapshotRows = 0 }
                        verify = Array(try runner.forward(tokens: [y, draft], startPos: position, allLogits: true))
                    }
                    let rows = (0...1).map { Array(UnsafeBufferPointer(start: runner.hidden(row: $0), count: W)) }
                    passes += 1
                    let t0 = try verify.withUnsafeBufferPointer {
                        try sampler.sample(UnsafeBufferPointer(rebasing: $0[0..<V]), gate: gate, position: produced.count)
                    }
                    let kept = t0 == draft && !stopTokens.contains(Int32(t0)) && produced.count + 1 < maxNewTokens ? 1 : 0
                    if kept == 0 { runner.rollback(keep: 1) }
                    accepted += kept
                    mtpTokens = kept == 1 ? [draft] : []
                    mtpRows = kept == 1 ? rows[0] : []
                    pendingH = rows[kept]
                    position += kept + 1
                    y = t0
                    guard try emit(t0) else { break loop }
                    if kept == 1 {
                        y = try verify.withUnsafeBufferPointer {
                            try sampler.sample(UnsafeBufferPointer(rebasing: $0[V..<(2 * V)]), gate: gate, position: produced.count)
                        }
                        guard try emit(y) else { break loop }
                    }
                }
            } else {
                while true {
                    try Task.checkCancellation()
                    let l = try runner.forward(tokens: [y], startPos: position)
                    position += 1
                    y = try sampler.sample(l, gate: gate, position: produced.count)
                    guard try emit(y) else { break }
                }
            }
        }
        return Qwen38Run(tokens: produced,
                         promptTokens: prompt.count,
                         cachedPromptTokens: cachedPromptTokens,
                         prefillSeconds: prefillSeconds,
                         decodeSeconds: Date().timeIntervalSince(decodeStart),
                         timeToFirstTokenSeconds: firstToken,
                         reason: reason,
                         kvPosition: position,
                         passes: passes,
                         accepted: accepted)
    }
}

/// A Qwen3.8 install as the server and the app open it: not a `.moepack` but a directory whose `manifest.json` declares
/// `arch.family` `qwen4exp` and names the GGUF and the PLE GGUF (relative to the directory), with the tokenizer sidecar
/// in `tokenizer/` (`GFTokenizer.tokenizerFolder`). The weight sidecars (`down/`, `bf16/`) are found next to the GGUF
/// by `Qwen38Runner` itself.
package struct Qwen38ModelDirectory: Sendable {
    package static let family = "qwen4exp"
    package let gguf: URL
    package let ple: URL

    package init(directory: URL) throws {
        struct Manifest: Decodable {
            struct Arch: Decodable { let family: String? }
            struct Files: Decodable { let gguf: String; let ple: String }
            let arch: Arch
            let qwen38: Files
        }
        let url = directory.appendingPathComponent("manifest.json")
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: url))
        guard manifest.arch.family == Self.family else {
            throw GGUFFile.Error.format("\(url.path): arch.family is not \(Self.family)")
        }
        func resolve(_ path: String) throws -> URL {
            let expanded = (path as NSString).expandingTildeInPath
            let file = expanded.hasPrefix("/") ? URL(fileURLWithPath: expanded)
                : directory.appendingPathComponent(expanded)
            guard FileManager.default.fileExists(atPath: file.path) else {
                throw GGUFFile.Error.open("\(file.path) is missing")
            }
            return file.standardizedFileURL
        }
        gguf = try resolve(manifest.qwen38.gguf)
        ple = try resolve(manifest.qwen38.ple)
    }
}
