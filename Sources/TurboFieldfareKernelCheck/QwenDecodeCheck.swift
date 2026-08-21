import Foundation
import Metal
import TurboFieldfare

// Phase 3's exit condition: decode the production Qwen3.5-MoE install and get
// the same tokens the CPU float32 reference got
// (`docs/qwen35moe/04-PHASES.md` Phase 3).
//
//   swift run -c release TurboFieldfareKernelCheck \
//     --qwen-decode scratch/ornith-oq4e-g64.gturbo
//
// Unlike `--gdn` and `--qwen`, this one needs the model: it is the first check
// in this plan that runs weights rather than synthetic inputs. Every kernel it
// exercises has already been scored against a CPU reference on its own
// (39 cases in `--qwen`, 15 in `--gdn`); what is unproven until here is the
// *wiring* — which tensor reaches which kernel, in what order, with the state
// carried between tokens.
//
// The comparison is greedy token ids, not activations. Two reasons. Activations
// would need the reference to dump 40 layers of hidden state for every step,
// and the reference is 47 s a token
// (`docs/qwen35moe/14-REFERENCE.md` §6) — the fixture would cost hours to make
// and would have to be remade for every prompt. And a token id is the quantity
// the exit condition is about: FP16 activations are not going to match a float32
// reference bit for bit, while the argmax over 248,077 rows either agrees or
// does not.
//
// A mismatch prints the step it happened at. That is where a layer-by-layer
// hidden-state comparison would start, and it is deliberately not automated
// here: the cheap check runs first.
//
// After the positive case, five deliberately mis-wired runs have to *disagree*
// with the reference. A whole-model comparison that has only ever been seen to
// pass says nothing about whether it could have failed
// (`PLAN_VISION.md` §6-3), and unlike a kernel case there is no error floor
// here to calibrate against — the only calibration available is watching the
// same comparison reject models that are wrong in ways this one nearly was.
// `--qwen-decode-no-faults` skips them.

private enum QwenDecodeError: Error, CustomStringConvertible {
    case noMetalDevice
    case badFixture(String)

    var description: String {
        switch self {
        case .noMetalDevice: return "no Metal device"
        case .badFixture(let detail): return "fixture: \(detail)"
        }
    }
}

/// Prompt and continuation from `docs/qwen35moe/14-REFERENCE.md` §6 — the
/// generation smoke that `Scripts/qwen35/reference_forward.py` ran over
/// `~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64` in float32, greedy, at 47.5 s a token.
///
/// The prompt is
/// `<|im_start|>user\n日本の首都はどこですか。一文で答えてください。<|im_end|>\n<|im_start|>assistant\n`
/// as the upstream tokenizer encodes it. **This built-in continuation stops at
/// 41 tokens because that reference run was stopped from outside, not because
/// the model stopped** (`14-REFERENCE.md` §6). The run was taken again and the
/// model stopped itself at 55 with `<|im_end|>`, so the fixture that closes the
/// phase is `scratch/qwen35/decode-fixture-55.json` via `--qwen-decode-fixture`
/// (`docs/qwen35moe/21-PHASE4-PREFILL.md` §1). Matching 41 is already decisive
/// about the wiring — a mis-bound tensor does not survive one token, let alone
/// forty.
struct QwenDecodeFixture: Decodable {
    let prompt: [Int32]
    let expected: [Int32]

    static let referenceSmoke = QwenDecodeFixture(
        prompt: [248045, 846, 198, 161607, 110161, 236576, 173399, 1710, 123070,
                 15685, 96517, 149657, 68798, 1710, 248046, 198, 248045, 74455, 198],
        expected: [248068, 198, 760, 1156, 369, 9859, 303, 10452, 25, 328, 8791,
                   369, 279, 6511, 314, 6124, 30, 5044, 4087, 303, 799, 11316,
                   1149, 271, 760, 6511, 314, 6124, 369, 25358, 318, 115197, 553,
                   271, 40, 1220, 4087, 303, 799, 11316, 303])

    init(prompt: [Int32], expected: [Int32]) {
        self.prompt = prompt
        self.expected = expected
    }

    static func load(path: String) throws -> QwenDecodeFixture {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let fixture = try JSONDecoder().decode(QwenDecodeFixture.self, from: data)
        guard !fixture.prompt.isEmpty, !fixture.expected.isEmpty else {
            throw QwenDecodeError.badFixture("prompt and expected must both be non-empty")
        }
        return fixture
    }
}

/// Tokens each negative control is given to disagree in, by default. A
/// mis-wiring that survives them is not a mis-wiring this check can see, and
/// saying so is the point — it would be a hole in the evidence, not a pass.
/// `--qwen-decode-fault-tokens` widens it.
let defaultFaultTokens = 16

private func runOnce(runner: QwenForwardRunner,
                     fixture: QwenDecodeFixture,
                     wanted: Int,
                     verbose: Bool,
                     prefillChunk: Int? = nil,
                     routedPath: QwenForwardRunner.PrefillRoutedPath = .perPair)
    throws -> (tokens: [Int32], firstMismatch: Int?) {
    runner.prefillRoutedPath = routedPath
    runner.reset()
    var firstMismatch: Int?
    var lastTokenEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    let onToken: (Int, Int32) -> Void = { step, token in
        let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let ms = Double(now - lastTokenEnd) / 1e6
        lastTokenEnd = now
        let want = fixture.expected[step]
        let agrees = token == want
        if !agrees, firstMismatch == nil { firstMismatch = step }
        guard verbose else { return }
        print(String(format: "  [%3d] %@ token=%6d  expected=%6d  %6.0f ms",
                     step, agrees ? "ok  " : "DIFF", token, want, ms))
    }
    let tokens: [Int32]
    if let chunk = prefillChunk {
        // Phase 4: the prompt goes through the T-row path, the continuation
        // through the same decode path Phase 3 matched. The handover — the
        // recurrent state, the conv window and the K/V cursor a chunk leaves
        // behind — is what this arm adds to the comparison.
        tokens = try runner.generateGreedyPrefilled(promptTokens: fixture.prompt,
                                                    maxNewTokens: wanted,
                                                    chunkWidth: chunk,
                                                    stopTokens: [],
                                                    onToken: onToken)
    } else {
        tokens = try runner.generateGreedy(promptTokens: fixture.prompt,
                                           maxNewTokens: wanted,
                                           stopTokens: [],
                                           onToken: onToken)
    }
    return (tokens, firstMismatch)
}

func runQwenDecodeCheck(modelPath: String,
                        fixturePath: String?,
                        maxNewTokens: Int?,
                        slotCount: Int,
                        runFaults: Bool,
                        faultTokens: Int,
                        prefillChunks: [Int] = []) throws -> Bool {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw QwenDecodeError.noMetalDevice
    }
    let fixture = try fixturePath.map { try QwenDecodeFixture.load(path: $0) }
        ?? QwenDecodeFixture.referenceSmoke
    let wanted = min(maxNewTokens ?? fixture.expected.count, fixture.expected.count)

    print(prefillChunks.isEmpty
          ? "=== qwen3_5_moe decode against the float32 reference "
            + "(docs/qwen35moe/04-PHASES.md Phase 3) ==="
          : "=== qwen3_5_moe prefill against the float32 reference "
            + "(docs/qwen35moe/04-PHASES.md Phase 4) ===")
    print("  model    \(modelPath)")
    print("  fixture  \(fixturePath ?? "built in (14-REFERENCE.md §6)")"
          + " — prompt \(fixture.prompt.count), expecting \(wanted)")

    let context = try MetalContext()
    var stats = ModelLoadStats()
    let loadStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    let model = try Model.load(directoryURL: URL(fileURLWithPath: modelPath),
                               device: device,
                               expecting: .ornith1_5_35B_A3B,
                               streamingMode: .pread(slotCount: slotCount),
                               loadStats: &stats)
    let loadMs = Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - loadStart) / 1e6
    print("  loaded in \(String(format: "%.0f", loadMs)) ms, \(slotCount) expert slots")

    let maxContext = fixture.prompt.count + wanted + 8
    let runner = try QwenForwardRunner(model: model, context: context, maxContext: maxContext)
    print("  scoring \(runner.scoredVocab) vocabulary rows of \(model.config.vocabSize)")
    print("")

    let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    let run = try runOnce(runner: runner, fixture: fixture, wanted: wanted, verbose: true,
                          prefillChunk: prefillChunks.first)
    let elapsed = Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started) / 1e6
    let produced = run.tokens
    let perToken = produced.isEmpty ? 0 : elapsed / Double(produced.count)
    print("")
    // The prompt runs one token at a time here, so this number is a decode rate
    // over prompt+generation, not a tok/s anyone should quote: there is no
    // chunked prefill on this path yet and nothing is overlapped
    // (`QwenForwardRunner` class note). Phase 6 measures.
    print(String(format: "  %.0f ms for %d tokens (%.0f ms each, serial, no prefill path)",
                 elapsed, produced.count, perToken))
    print("")

    var passed = true
    if produced.count != wanted {
        print("FAIL  produced \(produced.count) tokens, wanted \(wanted)")
        passed = false
    } else if let step = run.firstMismatch {
        print("FAIL  diverged at step \(step): got \(produced[step]), "
              + "reference \(fixture.expected[step])")
        passed = false
    } else {
        print("PASS  \(wanted) tokens, every one equal to the float32 reference")
    }

    // The chunk width is invisible to the model, so every width has to give
    // the same tokens. A prompt shorter than the width is one chunk and says
    // nothing about the handover; the narrow widths are the ones that cut the
    // prompt into three and make the state carry.
    for chunk in prefillChunks.dropFirst() {
        let again = try runOnce(runner: runner, fixture: fixture, wanted: wanted,
                                verbose: false, prefillChunk: chunk)
        let chunks = (fixture.prompt.count + chunk - 1) / chunk
        if again.tokens.count == wanted, again.firstMismatch == nil {
            print("PASS  chunk \(chunk) (\(chunks) chunks) — the same \(wanted) tokens")
        } else {
            let step = again.firstMismatch.map(String.init) ?? "-"
            print("FAIL  chunk \(chunk) (\(chunks) chunks) diverged at step \(step)")
            passed = false
        }
    }

    // The routed experts have two kernel families and Phase 4 only ever ran
    // one of them (`QwenForwardRunner.prefillRoutedPath`). They read the same
    // sorted pairs and write the same `routePartials`, so a disagreement here
    // is a bug in one of them and not a property of the prompt.
    if !prefillChunks.isEmpty {
        // Every chunk width, not just the first. The narrow width is where the
        // tiled path's batch planner sees the smallest groups — a chunk of 8
        // gives an expert one or two rows of a 64-row block — and that is the
        // shape the per-pair path never has to think about.
        for path in QwenForwardRunner.PrefillRoutedPath.allCases where path != .perPair {
            for chunk in prefillChunks {
                let again = try runOnce(runner: runner, fixture: fixture, wanted: wanted,
                                        verbose: false, prefillChunk: chunk,
                                        routedPath: path)
                let chunks = (fixture.prompt.count + chunk - 1) / chunk
                if again.tokens.count == wanted, again.firstMismatch == nil {
                    print("PASS  routed experts on the \(path.rawValue) path, "
                          + "chunk \(chunk) (\(chunks) chunks) — the same \(wanted) tokens")
                } else {
                    let step = again.firstMismatch.map(String.init) ?? "-"
                    print("FAIL  routed experts on the \(path.rawValue) path, "
                          + "chunk \(chunk) diverged at step \(step)")
                    passed = false
                }
            }
        }
        runner.prefillRoutedPath = .perPair
    }

    guard runFaults else { return passed }

    print("")
    print("  negative controls — each must disagree within \(faultTokens) tokens:")
    let faultWanted = min(faultTokens, wanted)
    for fault in QwenForwardRunner.DecodeFault.allCases where fault != .none {
        let faulty = try QwenForwardRunner(model: model,
                                           context: context,
                                           maxContext: maxContext,
                                           fault: fault)
        let result = try runOnce(runner: faulty, fixture: fixture,
                                 wanted: faultWanted, verbose: false,
                                 prefillChunk: prefillChunks.first)
        if let step = result.firstMismatch {
            print(String(format: "    PASS  %-20@ diverged at step %d (%d, not %d)",
                         fault.rawValue as NSString, step,
                         Int(result.tokens[step]), Int(fixture.expected[step])))
        } else {
            print("    FAIL  \(fault.rawValue) produced the reference's "
                  + "\(faultWanted) tokens anyway — this check cannot see it")
            passed = false
        }
    }
    return passed
}

// MARK: - prefill の時間 (`docs/qwen35moe/04-PHASES.md` Phase 4 / Phase 6)
//
// 検査ではなく**測定**。`--qwen-prefill` が答えるのは「同じトークンが出るか」で、
// こちらは「1 チャンクに何 ms 掛かるか」を見る。
//
//   swift run -c release TurboFieldfareKernelCheck \
//     --qwen-prefill-bench scratch/ornith-oq4e-g64.gturbo \
//     --qwen-prefill-bench-tokens 512
//
// **運用値ではない。**この経路は直列で (`QwenPrefill.swift`)、タイルごとに
// エキスパートを取り終えるまで GPU を止める。入力も合成で、トークン ID を
// 語彙に散らしただけのものなので、router の当たり方は本物の文章のそれではない
// (エキスパートキャッシュのヒット率がここの数字を大きく動かす)。
// 運用の数字は Phase 6 で `bench.sh` の作法に沿って取る。
func runQwenPrefillBench(modelPath: String,
                         tokens: Int,
                         chunk: Int,
                         slotCount: Int,
                         iterations: Int,
                         routedPath: QwenForwardRunner.PrefillRoutedPath,
                         cooldownSeconds: Double) throws -> Bool {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw QwenDecodeError.noMetalDevice
    }
    print("=== qwen3_5_moe prefill の時間 (測定、n=\(iterations)) ===")
    print("  model    \(modelPath)")
    print("  合成プロンプト \(tokens) トークン / チャンク \(chunk) / スロット \(slotCount)")
    print("  routed expert は \(routedPath.rawValue) の経路")
    print(String(format: "  クールダウン %.0f 秒 (Phase 6 の作法。0 なら連続)", cooldownSeconds))

    let context = try MetalContext()
    var stats = ModelLoadStats()
    let model = try Model.load(directoryURL: URL(fileURLWithPath: modelPath),
                               device: device,
                               expecting: .ornith1_5_35B_A3B,
                               streamingMode: .pread(slotCount: slotCount),
                               loadStats: &stats)
    let runner = try QwenForwardRunner(model: model, context: context,
                                       maxContext: tokens + 8)
    runner.prefillRoutedPath = routedPath
    // 語彙に散らした ID。同じ ID を並べると 40 層の router が毎トークン同じ
    // エキスパートを引き、キャッシュのヒット率が本物と懸け離れる。
    let prompt = (0..<tokens).map { Int32(($0 &* 7919) % runner.scoredVocab) }
    print("")
    // エキスパートの取得は pread で、キャッシュに無ければ 1.69 MiB を 1 個読む。
    // 総時間だけを見ていると、GPU が遅いのか I/O が遅いのかが分からない —
    // `docs/qwen35moe/21-PHASE4-PREFILL.md` §5 の説明の付いていない行
    // (2048 トークン / チャンク 512 だけ 2・3 回目が遅い) がまさにそれで、
    // 内訳の 3 列はその区別のために出す。
    print("  回  prefill (ms)  1 トークン    GPU (ms)   取得 (ms)   ホスト (ms)"
          + "  ヒット率   エキスパート  footprint (GB)")
    for iteration in 0..<iterations {
        if iteration > 0, cooldownSeconds > 0 { Thread.sleep(forTimeInterval: cooldownSeconds) }
        runner.reset()
        runner.resetProfile()
        model.telemetry.reset()
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        _ = try runner.prefill(tokens: prompt, chunkWidth: chunk)
        let ms = Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started) / 1e6
        let counters = model.telemetry.snapshot().prefill
        let fetchMs = Double(counters.fetchNanos) / 1e6
        let gpuMs = runner.gpuSeconds * 1e3
        let footprint = Double(ProcessMemoryFootprint.current().physFootprintBytes) / 1e9
        print(String(format: "  %2d %13.0f %11.2f %11.0f %11.0f %12.0f %9.1f%% %13d %15.2f",
                     iteration + 1, ms, ms / Double(tokens), gpuMs, fetchMs,
                     ms - gpuMs - fetchMs,
                     100.0 * counters.hitRate, counters.experts, footprint))
    }
    print("")
    print("  1 回目はエキスパートキャッシュが冷たい。**解釈は書かない** (n=\(iterations))")
    return true
}
