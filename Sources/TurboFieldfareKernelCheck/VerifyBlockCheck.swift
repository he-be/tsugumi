import Foundation
import Metal
import TurboFieldfare
import TurboFieldfareValidationSupport

// M4 (docs/mtp/04-PHASES.md): the speculative verify pass and the KV rewind,
// on the real 26B target.
//
//   swift run -c release TurboFieldfareKernelCheck \
//     --verify-block scratch/gemma4-qat.gturbo
//
// Two claims, and a negative control for each.
//
//   A. `verifyBlock` on k tokens produces, at every one of its k positions,
//      what `produce` produces one token at a time. Same KV, same kernels
//      below the head — only the head widens (03-DESIGN D3). If this is false
//      the MTP output-equality gate (04-PHASES §3 gate 1) cannot hold, because
//      a round would accept a token plain decode never would.
//
//   B. After `rewind`, generation continues exactly as in the world that never
//      speculated. The control breaks the rewind by one position in each
//      direction, and skips it entirely; all three have to diverge, or the
//      comparison is not evidence (PLAN_VISION §6-3).
//
// The A rounds use the rewind themselves — reference produces first, then
// rewind, then the block — which cannot manufacture a false PASS: the
// reference rows are already captured before the cursor moves, so a broken
// rewind corrupts only the block being tested and shows up as a mismatch.

enum VerifyBlockPrompt {
    /// Prose rather than a puzzle: the point is to walk the model through a
    /// long run of ordinary decode positions, not to be hard.
    static let `default` = """
        Explain, in three sentences, why speculative decoding can be faster \
        than plain autoregressive decoding on a memory-bound accelerator.
        """
}

/// One position's comparison between the block head and the scalar head.
private struct RowComparison {
    let position: Int
    /// Index of this row inside its block. Row `i` attends to the `i` rows the
    /// block wrote before it, so if the two paths' KV rows differ at all, the
    /// difference has `i` layers of attention to grow through.
    let blockIndex: Int
    let referenceArgmax: Int
    let blockArgmax: Int
    /// max |block - scalar| over the vocabulary, divided by max |scalar|.
    let relative: Double
    /// Gap between the reference's top-1 and top-2 logits, in the same units.
    /// A row whose margin is under its own numeric disagreement is a row where
    /// the two heads *could* have picked differently.
    let margin: Double
    let absolute: Double
    let referenceNorm: Double

    var contested: Bool { margin <= relative }
}

private func argmax(_ values: [Float]) -> Int {
    var best = 0
    for i in 1..<values.count where values[i] > values[best] { best = i }
    return best
}

/// Top-1 index plus the top-1 to top-2 gap.
private func topTwo(_ values: [Float]) -> (index: Int, margin: Float) {
    var best = 0
    var second = -1
    for i in 1..<values.count {
        if values[i] > values[best] {
            second = best
            best = i
        } else if second < 0 || values[i] > values[second] {
            second = i
        }
    }
    return (best, values[best] - values[second])
}

private struct RowProfile {
    let relative: Double
    let margin: Double
    /// max |block - scalar| in raw logit units.
    let absolute: Double
    /// max |scalar| — the denominator. A row with a flat distribution has a
    /// small one, which inflates `relative` for an unchanged `absolute`.
    let referenceNorm: Double
}

private func rowProfile(block: [Float], reference: [Float]) -> RowProfile {
    precondition(block.count == reference.count, "length mismatch")
    var maxDiff = 0.0
    var refNorm = 0.0
    for i in 0..<block.count {
        let b = Double(block[i])
        let r = Double(reference[i])
        precondition(r.isFinite, "reference logit is not finite at \(i) — harness bug")
        refNorm = Swift.max(refNorm, abs(r))
        guard b.isFinite else {
            return RowProfile(relative: .infinity, margin: 0, absolute: .infinity,
                              referenceNorm: refNorm)
        }
        maxDiff = Swift.max(maxDiff, abs(b - r))
    }
    precondition(refNorm > 1e-4, "reference row has no signal — harness bug")
    return RowProfile(relative: maxDiff / refNorm,
                      margin: Double(topTwo(reference).margin) / refNorm,
                      absolute: maxDiff,
                      referenceNorm: refNorm)
}

enum VerifyTolerance {
    /// One row of logits through the chunk path against the same row through
    /// the scalar decode path.
    ///
    /// This is not an FP16 noise floor and should not be read as one. The two
    /// paths run different kernels over a 30-layer, 2816-deep stack — tiled
    /// `simdgroup_matrix` QMM versus per-row GEMV — so they accumulate in a
    /// different order at every layer, and the gap they open is a property of
    /// the runtime that predates MTP.
    ///
    /// Set from measurement (`docs/mtp/15-M4-RESULTS.md` §2): over 128 decode
    /// positions the median row scores 1.0e-3 and the worst 3.8e-2, and a
    /// one-token block — the chunk path with nothing speculative in it — scores
    /// the same 3.5e-2 on the same row. 8e-2 is that worst case with the ~2x
    /// headroom this repo's fixture checks use, and it sits 20x under the
    /// smallest fault the detection case produces (1.7).
    ///
    /// What a new pass has to prove here is that it is no worse than the chunk
    /// path already is, not that it matches scalar decode bit for bit. The
    /// argmax cases carry the claim that actually decides a token.
    static let chunkVsScalarRow: Float = 8e-2
}

enum VerifyBlockCheckError: Error, CustomStringConvertible {
    case allocation(String)
    case promptTooShort(Int)

    var description: String {
        switch self {
        case .allocation(let what):
            return "\(what) allocation failed"
        case .promptTooShort(let count):
            return "prompt encodes to \(count) tokens; need at least 2"
        }
    }
}

private func sharedBuffer(_ device: MTLDevice, bytes: Int, label: String) throws -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: max(bytes, 1), options: .storageModeShared) else {
        throw VerifyBlockCheckError.allocation(label)
    }
    buffer.label = label
    return buffer
}

private func makeRunner(model: Model,
                        context: MetalContext,
                        maxContext: Int,
                        forceLogitsHead: Bool) throws -> RealForwardRunner {
    try RealForwardRunner(
        model: model,
        context: context,
        maxContext: maxContext,
        runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: forceLogitsHead))
}

/// Prefill the prompt and return the cursor it leaves behind.
@discardableResult
private func prefillPrompt(runner: RealForwardRunner,
                           promptIds: [Int32],
                           into logits: MTLBuffer) async throws -> Int {
    runner.reset()
    let result = try await runner.prefillChunked(
        tokens: promptIds[0...],
        startPosition: 0,
        outputMode: runner.usesFusedGreedyHead ? .greedyIfAvailable : .logits,
        config: .production(chunkTokens: 2048),
        vision: nil,
        into: logits) { _ in }
    return result.newPosition
}

// MARK: - A. verify rows against scalar decode

func runVerifyBlockChecks(modelPath: String,
                          blockTokens: Int,
                          rounds: Int,
                          prompt: String,
                          costOnly: Bool = false) async throws -> [CaseResult] {
    precondition(blockTokens >= 2 && blockTokens <= SpeculativeBlock.maxTokens,
                 "block tokens must be in 2...\(SpeculativeBlock.maxTokens)")
    let directoryURL = URL(fileURLWithPath: modelPath)
    let context = try MetalContext()
    let tokenizer = try await GFTokenizer.load(forModelDirectory: directoryURL)
    let promptIds = tokenizer.encode(prompt, addBOS: true)
    guard promptIds.count >= 2 else {
        throw VerifyBlockCheckError.promptTooShort(promptIds.count)
    }

    let runtime = RuntimeConfiguration()
    let model = try Model.load(directoryURL: directoryURL,
                               device: context.device,
                               streamingMode: .pread(slotCount: runtime.expertCacheSlots),
                               expertCachePolicy: runtime.modelExpertCachePolicy,
                               integrityPolicy: nil)
    let vocab = model.config.vocabSize
    let maxContext = promptIds.count
        + max(rounds * blockTokens, CostProbePlan.tokens) + 64

    print("=== MTP verify pass + KV rewind (docs/mtp 04-PHASES M4) ===")
    print("  model    \(directoryURL.lastPathComponent)")
    print("  prompt   \(promptIds.count) tok, block k=\(blockTokens), rounds=\(rounds)")

    var results: [CaseResult] = []
    // The correctness phases each build their own runner, and a runner owns a
    // prefill scratch sized for the configured chunk width. Timing after them
    // measures the leftovers, so `--verify-cost-only` gives the probe a process
    // with exactly one runner in it.
    if costOnly {
        try await costProbe(model: model, context: context, promptIds: promptIds,
                            vocab: vocab, maxContext: maxContext,
                            blockTokens: blockTokens)
        return results
    }
    results.append(contentsOf: try await prefillPathBaseline(
        model: model, context: context, promptIds: promptIds, vocab: vocab,
        maxContext: maxContext))
    // Width 1 first, over the same number of positions: the chunk path with
    // nothing speculative about it, and therefore the bar the real block has to
    // meet. Anything the k-token block adds shows up as a difference between
    // these two distributions.
    let width1 = try await verifyRowsAgainstScalarDecode(
        model: model, context: context, promptIds: promptIds, vocab: vocab,
        maxContext: maxContext, blockTokens: 1, rounds: rounds * blockTokens)
    let wide = try await verifyRowsAgainstScalarDecode(
        model: model, context: context, promptIds: promptIds, vocab: vocab,
        maxContext: maxContext, blockTokens: blockTokens, rounds: rounds)
    results.append(contentsOf: width1.cases)
    results.append(contentsOf: wide.cases)
    results.append(result("verify/rows/wide-vs-width1",
                          groupSize: context.affineGroupSize,
                          rel: wide.worst / Swift.max(width1.worst, 1e-9),
                          tolerance: 3,
                          detail: String(format: "k=%d worst=%.3e over k=1 worst=%.3e; "
                                         + "medians %.3e / %.3e",
                                         blockTokens, wide.worst, width1.worst,
                                         wide.median, width1.median)))
    results.append(contentsOf: try await verifyGreedyRows(
        model: model, context: context, promptIds: promptIds, vocab: vocab,
        maxContext: maxContext, blockTokens: blockTokens, rounds: rounds))
    results.append(contentsOf: try await rewindEquivalence(
        model: model, context: context, promptIds: promptIds, vocab: vocab,
        maxContext: maxContext, blockTokens: blockTokens))
    try await costProbe(model: model, context: context, promptIds: promptIds,
                        vocab: vocab, maxContext: maxContext,
                        blockTokens: blockTokens)
    return results
}

/// The gap the runtime already ships with: the *last prompt position*, scored
/// through the chunk path against the scalar path.
///
/// That row is not hypothetical. Every ordinary run takes it from the chunk
/// path and samples the first generated token from it (`RawCompletion.swift`,
/// `PrefillSeed`), so whatever separates the two paths here is pre-existing
/// behavior that MTP inherits rather than creates. The sweep below turns the
/// same comparison into a distribution.
private func prefillPathBaseline(model: Model,
                                 context: MetalContext,
                                 promptIds: [Int32],
                                 vocab: Int,
                                 maxContext: Int) async throws -> [CaseResult] {
    let runner = try makeRunner(model: model, context: context,
                                maxContext: maxContext, forceLogitsHead: true)
    let device = context.device
    let rowBytes = vocab * MemoryLayout<Float16>.stride
    let logits = try sharedBuffer(device, bytes: rowBytes, label: "baseline.logits")

    // The whole prompt through the chunk path.
    _ = try await prefillPrompt(runner: runner, promptIds: promptIds, into: logits)
    let chunkRow = Fp16Buffer.read(logits, count: vocab)

    // The same final position through the scalar path.
    let head = Array(promptIds.dropLast())
    _ = try await prefillPrompt(runner: runner, promptIds: head, into: logits)
    try await runner.produce(token: promptIds[promptIds.count - 1],
                             position: head.count, into: logits)
    let scalarRow = Fp16Buffer.read(logits, count: vocab)
    let baseline = rowProfile(block: chunkRow, reference: scalarRow)

    return [
        result("verify/baseline/prefill-vs-decode",
               groupSize: context.affineGroupSize,
               rel: baseline.relative,
               tolerance: Double(VerifyTolerance.chunkVsScalarRow),
               detail: String(format: "last prompt position, abs=%.3f |ref|=%.1f margin=%.3e",
                              baseline.absolute, baseline.referenceNorm, baseline.margin)),
    ]
}

/// Claim A on the logits head: every row of the block against the same row
/// from `produce`, as a number rather than as an argmax.
private struct RowSweep {
    let cases: [CaseResult]
    let worst: Double
    let median: Double
}

private func verifyRowsAgainstScalarDecode(model: Model,
                                           context: MetalContext,
                                           promptIds: [Int32],
                                           vocab: Int,
                                           maxContext: Int,
                                           blockTokens: Int,
                                           rounds: Int) async throws -> RowSweep {
    let runner = try makeRunner(model: model, context: context,
                                maxContext: maxContext, forceLogitsHead: true)
    let device = context.device
    let rowBytes = vocab * MemoryLayout<Float16>.stride
    let scalarLogits = try sharedBuffer(device, bytes: rowBytes, label: "verify.scalarLogits")
    let blockLogits = try sharedBuffer(device,
                                       bytes: SpeculativeBlock.logitRowsBytes(
                                        vocab: vocab, blockTokens: blockTokens),
                                       label: "verify.blockLogits")

    var position = try await prefillPrompt(runner: runner, promptIds: promptIds,
                                           into: scalarLogits)
    var nextToken = Int32(argmax(Fp16Buffer.read(scalarLogits, count: vocab)))

    var comparisons: [RowComparison] = []
    var shiftedRelative = 0.0
    var produceSeconds = 0.0
    var verifySeconds = 0.0
    let started = Date()
    for _ in 0..<rounds {
        // The block is the greedy continuation: exactly what a fully accepted
        // speculative round would hold.
        var blockIDs: [Int32] = []
        var referenceRows: [[Float]] = []
        for i in 0..<blockTokens {
            blockIDs.append(nextToken)
            let producedAt = Date()
            try await runner.produce(token: nextToken, position: position + i,
                                     into: scalarLogits)
            produceSeconds += Date().timeIntervalSince(producedAt)
            let row = Fp16Buffer.read(scalarLogits, count: vocab)
            referenceRows.append(row)
            nextToken = Int32(argmax(row))
        }

        try runner.rewind(to: position)
        let verifiedAt = Date()
        try await runner.verifyBlock(tokens: blockIDs[0...],
                                     startPosition: position,
                                     into: blockLogits,
                                     greedyTokens: nil)
        verifySeconds += Date().timeIntervalSince(verifiedAt)

        for i in 0..<blockTokens {
            let row = Fp16Buffer.read(
                blockLogits, count: vocab,
                byteOffset: i * vocab * MemoryLayout<Float16>.stride)
            let profile = rowProfile(block: row, reference: referenceRows[i])
            comparisons.append(RowComparison(position: position + i,
                                             blockIndex: i,
                                             referenceArgmax: argmax(referenceRows[i]),
                                             blockArgmax: argmax(row),
                                             relative: profile.relative,
                                             margin: profile.margin,
                                             absolute: profile.absolute,
                                             referenceNorm: profile.referenceNorm))
            // Detection power: the same block row against its *neighbour's*
            // reference. Rows that differ only by numeric noise would score the
            // same against either, and this case would stop clearing its floor.
            if i + 1 < blockTokens {
                shiftedRelative = Swift.max(
                    shiftedRelative,
                    rowProfile(block: row, reference: referenceRows[i + 1]).relative)
            }
        }
        position += blockTokens
    }
    let seconds = Date().timeIntervalSince(started)

    let sorted = comparisons.map(\.relative).sorted()
    let worstRelative = sorted.last ?? 0
    let median = sorted.isEmpty ? 0 : sorted[sorted.count / 2]
    let disagreements = comparisons.filter { $0.referenceArgmax != $0.blockArgmax }
    // Rows where the reference's own top-1/top-2 gap is narrower than that
    // row's numeric disagreement: positions where the two paths were entitled
    // to pick differently and happened not to. Reported because it is the
    // number that bounds the output-equality gate, not because it is a failure.
    let contested = comparisons.filter(\.contested)

    // Does the gap grow with depth into the block? Row i attends to i rows the
    // chunk path wrote, so a systematic KV difference would show up here as a
    // rising column and would put a ceiling on the block size.
    print("  k=\(blockTokens) rows:        " + (0..<blockTokens).map { i in
        let column = comparisons.filter { $0.blockIndex == i }.map(\.relative).sorted()
        let med = column.isEmpty ? 0 : column[column.count / 2]
        return String(format: "i=%d med=%.1e max=%.1e", i, med, column.last ?? 0)
    }.joined(separator: "  "))
    // The relative score divides by the row's own logit range, so a flat row
    // scores high on an unchanged absolute difference. Print both for the
    // widest rows, which is the only way to tell the two apart.
    let widest = comparisons.sorted { $0.relative > $1.relative }.prefix(3)
    print("  widest rows:       " + widest.map {
        String(format: "pos=%d i=%d rel=%.1e abs=%.2f |ref|=%.1f margin=%.1e",
               $0.position, $0.blockIndex, $0.relative, $0.absolute,
               $0.referenceNorm, $0.margin)
    }.joined(separator: "  "))
    print(String(format: "  absolute logits:   median=%.3f worst=%.3f",
                 comparisons.map(\.absolute).sorted()[comparisons.count / 2],
                 comparisons.map(\.absolute).max() ?? 0))
    // Interleaved, so not a cost measurement: each verify evicts what the
    // produces around it just streamed into the 48-slot expert cache, and both
    // sides pay for it (produce lands 2-3x above its own baseline here). The
    // steady-state numbers come from `costProbe`; these are printed only so the
    // reader can see that this run is not one.
    print(String(format: "  interleaved:       produce=%.1f ms/tok  verify=%.1f ms/block",
                 produceSeconds * 1000 / Double(comparisons.count),
                 verifySeconds * 1000 / Double(rounds)))

    var results: [CaseResult] = []
    results.append(result("verify/rows/logits k=\(blockTokens) n=\(comparisons.count)",
                          groupSize: context.affineGroupSize,
                          rel: worstRelative,
                          tolerance: Double(VerifyTolerance.chunkVsScalarRow),
                          detail: String(format: "median=%.3e worst=%.3e over %d positions, "
                                         + "%d contested, %.1f s",
                                         median, worstRelative, comparisons.count,
                                         contested.count, seconds)))
    results.append(result("verify/rows/argmax k=\(blockTokens) n=\(comparisons.count)",
                          groupSize: context.affineGroupSize,
                          rel: disagreements.isEmpty ? 0 : 1,
                          tolerance: 0,
                          detail: disagreements.isEmpty
                            ? "block and scalar heads agree at every position; "
                              + "\(contested.count) row(s) with top-1 margin <= rel"
                            : disagreements.prefix(4).map {
                                "pos \($0.position): block=\($0.blockArgmax) "
                                + "scalar=\($0.referenceArgmax) margin="
                                + String(format: "%.3e", $0.margin)
                              }.joined(separator: "; ")))
    if blockTokens > 1 {
        results.append(detectionResult("verify/rows/shifted-by-one",
                                       groupSize: context.affineGroupSize,
                                       rel: shiftedRelative,
                                       floor: 10 * Double(VerifyTolerance.chunkVsScalarRow),
                                       detail: "row i vs reference row i+1"))
    }
    return RowSweep(cases: results, worst: worstRelative, median: median)
}

/// Claim A on the fused greedy head — the path a temperature-0 run takes.
/// Nothing but the argmax leaves the GPU there, so this is a token comparison.
private func verifyGreedyRows(model: Model,
                              context: MetalContext,
                              promptIds: [Int32],
                              vocab: Int,
                              maxContext: Int,
                              blockTokens: Int,
                              rounds: Int) async throws -> [CaseResult] {
    let runner = try makeRunner(model: model, context: context,
                                maxContext: maxContext, forceLogitsHead: false)
    let device = context.device
    let unusedLogits = try sharedBuffer(device, bytes: vocab * MemoryLayout<Float16>.stride,
                                        label: "verify.greedyUnusedLogits")
    let greedyRows = try sharedBuffer(device,
                                      bytes: blockTokens * MemoryLayout<UInt32>.stride,
                                      label: "verify.greedyRows")

    var position = try await prefillPrompt(runner: runner, promptIds: promptIds,
                                           into: unusedLogits)
    var nextToken = Int32(bitPattern: runner.lastGreedyToken)
    var checked = 0
    var mismatches: [String] = []

    for _ in 0..<rounds {
        var blockIDs: [Int32] = []
        var referenceTokens: [Int32] = []
        for i in 0..<blockTokens {
            blockIDs.append(nextToken)
            try await runner.produce(token: nextToken, position: position + i,
                                     into: unusedLogits)
            nextToken = Int32(bitPattern: runner.lastGreedyToken)
            referenceTokens.append(nextToken)
        }
        try runner.rewind(to: position)
        try await runner.verifyBlock(tokens: blockIDs[0...],
                                     startPosition: position,
                                     into: unusedLogits,
                                     greedyTokens: greedyRows)
        for i in 0..<blockTokens {
            let produced = Int32(bitPattern: greedyRows.contents().load(
                fromByteOffset: i * MemoryLayout<UInt32>.stride, as: UInt32.self))
            checked += 1
            if produced != referenceTokens[i] {
                mismatches.append("pos \(position + i): block=\(produced) "
                                  + "scalar=\(referenceTokens[i])")
            }
        }
        position += blockTokens
    }

    return [result("verify/rows/greedy n=\(checked)",
                   groupSize: context.affineGroupSize,
                   rel: mismatches.isEmpty ? 0 : 1,
                   tolerance: 0,
                   detail: mismatches.isEmpty
                    ? "fused greedy head agrees at every block position"
                    : mismatches.prefix(4).joined(separator: "; "))]
}

// MARK: - Cost of a verify block

/// What one round of speculation costs on the target side, measured rather
/// than derived.
///
/// `10-M0-RESULTS.md` §2 priced a k-token verify at 1.32 / 1.60 / 1.85 decode
/// steps for k = 2 / 3 / 4 from an expert-I/O argument. That factor is half of
/// the speed model the whole plan rests on and it has never been run. This
/// runs it: the same token stream over the same positions, once through
/// `produce` and once through `verifyBlock`, each in steady state on its own
/// warm cache. The fused greedy head is used on both sides, because reading a
/// 512 KiB logits row per position back to the host would measure the harness.
/// Widths are swept rather than measured at one k: a verify that costs the
/// same at k=1 and k=4 is paying for the call, and a verify that scales with k
/// is paying for the tokens. The two have different fixes and only the sweep
/// tells them apart.
private enum CostProbePlan {
    static let timedBlocks = 16
    static let warmupBlocks = 2
    /// Positions one pass walks. Every width replays the same stream, so the
    /// widest one sizes it.
    static let tokens = (timedBlocks + warmupBlocks) * SpeculativeBlock.maxTokens
}

private func costProbe(model: Model,
                       context: MetalContext,
                       promptIds: [Int32],
                       vocab: Int,
                       maxContext: Int,
                       blockTokens: Int) async throws {
    let widths = Array(Set([1, 2, blockTokens, SpeculativeBlock.maxTokens])).sorted()
    let timedBlocks = CostProbePlan.timedBlocks
    let warmupBlocks = CostProbePlan.warmupBlocks
    let total = CostProbePlan.tokens

    let runner = try makeRunner(model: model, context: context,
                                maxContext: maxContext, forceLogitsHead: false)
    let device = context.device
    let logits = try sharedBuffer(device, bytes: vocab * MemoryLayout<Float16>.stride,
                                  label: "cost.logits")
    let greedyRows = try sharedBuffer(device,
                                      bytes: SpeculativeBlock.maxTokens
                                        * MemoryLayout<UInt32>.stride,
                                      label: "cost.greedyRows")

    // The decode baseline is measured twice, once on each side of the verify
    // sweep, because this machine drifts: the same phase has come out anywhere
    // between 45 and 62 ms/tok across runs. Bracketing it makes the drift
    // visible instead of silently scaling every ratio below it.
    let produceWarmup = warmupBlocks * (widths.last ?? blockTokens)
    var tokens: [Int32] = []
    var promptEnd = 0

    func decodePass(record: Bool) async throws -> Double {
        var position = try await prefillPrompt(runner: runner, promptIds: promptIds,
                                               into: logits)
        if record { promptEnd = position }
        var token = Int32(bitPattern: runner.lastGreedyToken)
        var seconds = 0.0
        for step in 0..<total {
            if record { tokens.append(token) }
            let at = Date()
            try await runner.produce(token: token, position: position, into: logits)
            if step >= produceWarmup { seconds += Date().timeIntervalSince(at) }
            position += 1
            token = Int32(bitPattern: runner.lastGreedyToken)
        }
        return seconds * 1000 / Double(total - produceWarmup)
    }

    let decodeBefore = try await decodePass(record: true)
    var position = 0

    var line = ""
    for k in widths {
        position = try await prefillPrompt(runner: runner, promptIds: promptIds, into: logits)
        precondition(position == promptEnd, "prefill is not reproducible — harness bug")
        var seconds = 0.0
        for block in 0..<(warmupBlocks + timedBlocks) {
            let lower = block * k
            let at = Date()
            try await runner.verifyBlock(tokens: tokens[lower..<(lower + k)],
                                         startPosition: position,
                                         into: logits,
                                         greedyTokens: greedyRows)
            if block >= warmupBlocks { seconds += Date().timeIntervalSince(at) }
            position += k
        }
        let msPerBlock = seconds * 1000 / Double(timedBlocks)
        line += String(format: "  |  k=%d %.0f ms/block", k, msPerBlock)
    }
    let decodeAfter = try await decodePass(record: false)
    let decode = (decodeBefore + decodeAfter) / 2
    print(String(format: "  cost: decode=%.1f ms/tok (%.1f before, %.1f after)%@",
                 decode, decodeBefore, decodeAfter, line))
    // Which side of the round is paying: expert bytes, or everything else.
    let telemetry = model.telemetry.snapshot()
    print(String(format: "  expert io: decode hit=%.1f%% %d/%d io=%.2fs  |  "
                 + "prefill hit=%.1f%% %d/%d io=%.2fs",
                 telemetry.decode.hitRate * 100, telemetry.decode.hits,
                 telemetry.decode.experts, Double(telemetry.decode.fetchNanos) / 1e9,
                 telemetry.prefill.hitRate * 100, telemetry.prefill.hits,
                 telemetry.prefill.experts, Double(telemetry.prefill.fetchNanos) / 1e9))
}

// MARK: - B. rewind equivalence and its controls

/// How a run is allowed to be wrong about where the cursor should land.
private enum RewindFault: String, CaseIterable {
    case correct
    case shortByOne = "short-by-one"
    case longByOne = "long-by-one"
    case skipped
}

private func rewindEquivalence(model: Model,
                               context: MetalContext,
                               promptIds: [Int32],
                               vocab: Int,
                               maxContext: Int,
                               blockTokens: Int) async throws -> [CaseResult] {
    let runner = try makeRunner(model: model, context: context,
                                maxContext: maxContext, forceLogitsHead: true)
    let device = context.device
    let scalarLogits = try sharedBuffer(device, bytes: vocab * MemoryLayout<Float16>.stride,
                                        label: "rewind.scalarLogits")
    let blockLogits = try sharedBuffer(device,
                                       bytes: SpeculativeBlock.logitRowsBytes(
                                        vocab: vocab, blockTokens: blockTokens),
                                       label: "rewind.blockLogits")
    let continuation = 16
    let accepted = blockTokens / 2

    // The world that never speculated: plain scalar decode, `blockTokens +
    // continuation` tokens long.
    var reference: [Int32] = []
    var position = try await prefillPrompt(runner: runner, promptIds: promptIds,
                                           into: scalarLogits)
    let promptEnd = position
    var token = Int32(argmax(Fp16Buffer.read(scalarLogits, count: vocab)))
    for _ in 0..<(blockTokens + continuation) {
        reference.append(token)
        try await runner.produce(token: token, position: position, into: scalarLogits)
        position += 1
        token = Int32(argmax(Fp16Buffer.read(scalarLogits, count: vocab)))
    }

    var results: [CaseResult] = []
    for fault in RewindFault.allCases {
        position = try await prefillPrompt(runner: runner, promptIds: promptIds,
                                           into: scalarLogits)
        precondition(position == promptEnd, "prefill is not reproducible — harness bug")
        try await runner.verifyBlock(tokens: reference[0..<blockTokens],
                                     startPosition: position,
                                     into: blockLogits,
                                     greedyTokens: nil)
        let target: Int
        switch fault {
        case .correct:    target = position + accepted
        case .shortByOne: target = position + accepted - 1
        case .longByOne:  target = position + accepted + 1
        case .skipped:    target = position + blockTokens
        }
        try runner.rewind(to: target)

        // The loop's own bookkeeping says `accepted` tokens were committed, so
        // it feeds the next one — at whatever cursor the rewind actually left.
        var cursor = target
        var produced: [Int32] = []
        var feed = reference[accepted]
        for _ in 0..<continuation {
            try await runner.produce(token: feed, position: cursor, into: scalarLogits)
            cursor += 1
            feed = Int32(argmax(Fp16Buffer.read(scalarLogits, count: vocab)))
            produced.append(feed)
        }

        let expected = Array(reference[(accepted + 1)...].prefix(continuation))
        let matched = zip(produced, expected).prefix(while: { $0 == $1 }).count
        let agrees = matched == expected.count
        let detail = "accepted=\(accepted)/\(blockTokens) target=\(target) "
            + "matched \(matched)/\(expected.count)"
        if fault == .correct {
            results.append(result("rewind/continuation-matches-unspeculated",
                                  groupSize: context.affineGroupSize,
                                  rel: agrees ? 0 : 1,
                                  tolerance: 0,
                                  detail: detail))
        } else {
            // A broken rewind has to change the text. Scored as "fraction of the
            // continuation that still agrees", which must stay under 1.
            results.append(detectionResult("rewind/detect/\(fault.rawValue)",
                                           groupSize: context.affineGroupSize,
                                           rel: 1.0 - Double(matched) / Double(expected.count),
                                           floor: 1.0 / Double(expected.count),
                                           detail: detail))
        }
    }
    return results
}
