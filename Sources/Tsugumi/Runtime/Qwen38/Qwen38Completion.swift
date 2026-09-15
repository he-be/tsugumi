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

/// A copy of everything `Qwen38Engine` needs to continue from `position` (`docs/qwen38/15` §2 G-0): the runner's
/// recurrent regions (117.6 MiB) and the engine's own MTP bookkeeping. The positional state (KV, indexer keys, MTP KV)
/// is not copied: the rows before `position` are unchanged by anything that runs after it, as long as the tokens
/// before `position` are the same.
///
/// Where the bytes live is `Qwen38Engine.checkpointStore`: `ssd` (default: a file written and read with `F_NOCACHE`,
/// unlinked as soon as it is created so only the descriptor holds it) or `ram` (anonymous memory). Four copies in RAM
/// (450 MB) swapped under the guard at 4K twice (`docs/qwen38/18`).
package final class Qwen38Checkpoint: @unchecked Sendable {
    package let position: Int
    package let bytes: Int
    /// Seconds the copy took.
    package let captureSeconds: Double
    private let memory: UnsafeMutableRawPointer?
    private let fd: Int32
    fileprivate let plePrev: [Int]
    fileprivate let pendingH: [Float]
    fileprivate let mtpTokens: [Int]
    fileprivate let mtpRows: [Float]

    fileprivate init(engine: Qwen38Engine) throws {
        let t = Date()
        position = engine.position
        let runner = engine.runner
        bytes = runner.recurrentBytes
        plePrev = runner.plePrevious
        pendingH = engine.pendingH
        mtpTokens = engine.mtpTokens
        mtpRows = engine.mtpRows
        switch engine.checkpointStore {
        case .ram:
            let m = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 16)
            var at = 0
            runner.forEachRecurrentRegion { p, n in (m + at).copyMemory(from: p, byteCount: n); at += n }
            memory = m
            fd = -1
        case .ssd(let directory):
            let file = directory.appendingPathComponent("q38-checkpoint-\(UUID().uuidString).bin").path
            let fd = open(file, O_CREAT | O_EXCL | O_RDWR, 0o600)
            guard fd >= 0 else { throw GGUFFile.Error.open("\(file): errno \(errno)") }
            unlink(file)
            _ = fcntl(fd, F_NOCACHE, 1)
            var ok = false
            defer { if !ok { close(fd) } }
            var at = 0
            try runner.forEachRecurrentRegion { p, n in
                var done = 0
                while done < n {
                    let w = pwrite(fd, p + done, n - done, off_t(at + done))
                    guard w > 0 else { throw GGUFFile.Error.open("\(file): write errno \(errno)") }
                    done += w
                }
                at += n
            }
            ok = true
            memory = nil
            self.fd = fd
        }
        captureSeconds = Date().timeIntervalSince(t)
    }

    fileprivate func read(into runner: Qwen38Runner) throws {
        if let memory {
            var at = 0
            runner.forEachRecurrentRegion { p, n in p.copyMemory(from: memory + at, byteCount: n); at += n }
        } else {
            var at = 0
            try runner.forEachRecurrentRegion { p, n in
                var done = 0
                while done < n {
                    let r = pread(fd, p + done, n - done, off_t(at + done))
                    guard r > 0 else { throw GGUFFile.Error.open("checkpoint read errno \(errno)") }
                    done += r
                }
                at += n
            }
        }
        runner.plePrevious = plePrev
    }

    deinit {
        memory?.deallocate()
        if fd >= 0 { close(fd) }
    }
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
    fileprivate var pendingH: [Float]
    /// Kept drafts whose rows the trunk holds but the MTP KV does not yet, each with the trunk residual of the position
    /// before it (`14` §4).
    fileprivate var mtpTokens: [Int] = []
    fileprivate var mtpRows: [Float] = []
    private let mtpChunk = Int(ProcessInfo.processInfo.environment["Q38_MTP_CHUNK"] ?? "") ?? 256

    package enum CheckpointStore {
        case ram
        case ssd(URL)
    }
    /// `Q38_CHECKPOINT_STORE=ssd|ram` (default ssd, in `Q38_CHECKPOINT_DIR` or the temporary directory).
    package var checkpointStore: CheckpointStore = {
        let env = ProcessInfo.processInfo.environment
        if env["Q38_CHECKPOINT_STORE"] == "ram" { return .ram }
        return .ssd(URL(fileURLWithPath: env["Q38_CHECKPOINT_DIR"] ?? NSTemporaryDirectory()))
    }()

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

    /// A copy of the state at `position` (needs `position > 0`).
    package func captureCheckpoint() throws -> Qwen38Checkpoint {
        precondition(position > 0, "nothing to capture at position 0")
        return try Qwen38Checkpoint(engine: self)
    }

    /// Back to where `checkpoint` was taken. The caller vouches that the tokens before its position are the ones the
    /// next prompt starts with.
    package func restore(_ checkpoint: Qwen38Checkpoint) throws {
        try checkpoint.read(into: runner)
        position = checkpoint.position
        pendingH = checkpoint.pendingH
        mtpTokens = checkpoint.mtpTokens
        mtpRows = checkpoint.mtpRows
    }

    /// Prefill `promptTokens` from `position` (= `cachedPromptTokens`) and generate up to `maxNewTokens`.
    /// `onPrefill(done, total)` in prompt positions (the cached ones included, as `CompletionPrefill` reports): once
    /// before any work, after every trunk layer of a chunk, and at every chunk's end; `onToken(index, id)` for every emitted token, before the next draw
    /// (a caller that suppresses the grammar inside a thought block sets it there). `shouldStop` is asked after each
    /// token. Throws `CancellationError` between chunks and steps; the state is then unnamed and the caller resets.
    ///
    /// Checkpoints (`docs/qwen38/15` §2 G): the prefill is cut at every position of `checkpointsAt` inside it and
    /// `onCheckpoint` gets a copy there (after the MTP head has taken the same rows). In decode, before a token of
    /// `checkpointBefore` is fed, `onCheckpoint` gets a copy at that token's position; the speculative loop never keeps
    /// such a token as a draft, so it is always the token fed next and never inside a verified pair.
    package func runCompletion(promptTokens: [Int32],
                               cachedPromptTokens: Int,
                               maxNewTokens: Int,
                               stopTokens: Set<Int32>,
                               constraint: (any GenerationConstraint)?,
                               greedy: Bool,
                               seed: UInt64,
                               checkpointsAt: [Int] = [],
                               checkpointBefore: Set<Int32> = [],
                               shouldStop: () -> Bool = { false },
                               onPrefill: ((Int, Int) -> Void)? = nil,
                               onCheckpoint: ((Qwen38Checkpoint) -> Void)? = nil,
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
        let cuts = checkpointsAt.filter { $0 > position && $0 < position + prompt.count }.sorted()
        // Report the total before the first chunk: at ~95 tok/s one chunk of 2,048 is over 20 s.
        let total = position + prompt.count
        onPrefill?(position, total)
        while done < prompt.count {
            try Task.checkCancellation()
            let T = min(prefillChunk, prompt.count - done, (cuts.first { $0 > position } ?? Int.max) - position)
            let chunk = Array(prompt[done..<(done + T)])
            let chunkStart = position
            let layers = runner.nTrunk
            // Inside the chunk the count stays below its end: `done == total` is what tells the app decode began.
            let l = try runner.forward(tokens: chunk, startPos: position, onLayer: onPrefill.map { report in
                { n in report(chunkStart + min(T * n / layers, T - 1), total) }
            })
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
            if cuts.contains(position), let onCheckpoint { onCheckpoint(try captureCheckpoint()) }
            onPrefill?(position, total)
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
                    if checkpointBefore.contains(Int32(y)), let onCheckpoint { onCheckpoint(try captureCheckpoint()) }
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
                    let kept = t0 == draft && !stopTokens.contains(Int32(t0)) && !checkpointBefore.contains(Int32(t0))
                        && produced.count + 1 < maxNewTokens ? 1 : 0
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
                    if checkpointBefore.contains(Int32(y)), let onCheckpoint { onCheckpoint(try captureCheckpoint()) }
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
