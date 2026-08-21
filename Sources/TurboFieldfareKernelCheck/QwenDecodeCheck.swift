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
/// as the upstream tokenizer encodes it. **The continuation stops at 41 tokens
/// because the reference run was stopped from outside, not because the model
/// stopped** (`14-REFERENCE.md` §6); Phase 3 asks for 64, so a longer fixture
/// passed with `--qwen-decode-fixture` is what closes the phase. Matching 41 is
/// already decisive about the wiring — a mis-bound tensor does not survive one
/// token, let alone forty.
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
                     verbose: Bool) throws -> (tokens: [Int32], firstMismatch: Int?) {
    runner.reset()
    var firstMismatch: Int?
    var lastTokenEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    let tokens = try runner.generateGreedy(
        promptTokens: fixture.prompt,
        maxNewTokens: wanted,
        stopTokens: [],
        onToken: { step, token in
            let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let ms = Double(now - lastTokenEnd) / 1e6
            lastTokenEnd = now
            let want = fixture.expected[step]
            let agrees = token == want
            if !agrees, firstMismatch == nil { firstMismatch = step }
            guard verbose else { return }
            print(String(format: "  [%3d] %@ token=%6d  expected=%6d  %6.0f ms",
                         step, agrees ? "ok  " : "DIFF", token, want, ms))
        })
    return (tokens, firstMismatch)
}

func runQwenDecodeCheck(modelPath: String,
                        fixturePath: String?,
                        maxNewTokens: Int?,
                        slotCount: Int,
                        runFaults: Bool,
                        faultTokens: Int) throws -> Bool {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw QwenDecodeError.noMetalDevice
    }
    let fixture = try fixturePath.map { try QwenDecodeFixture.load(path: $0) }
        ?? QwenDecodeFixture.referenceSmoke
    let wanted = min(maxNewTokens ?? fixture.expected.count, fixture.expected.count)

    print("=== qwen3_5_moe decode against the float32 reference "
          + "(docs/qwen35moe/04-PHASES.md Phase 3) ===")
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
    let run = try runOnce(runner: runner, fixture: fixture, wanted: wanted, verbose: true)
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
                                 wanted: faultWanted, verbose: false)
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
