import Foundation
import Metal

/// Per-generation buffers the speculative loop needs on top of
/// `RawCompletionScratch`: one block of logit rows (or greedy tokens), and the
/// post-norm hidden rows the drafter conditions on.
///
/// Allocated only when `--draft-block-size > 0`, so a text-only run keeps the
/// memory profile it had before MTP existed (04-PHASES §3 gate 4).
public struct SpeculativeScratch: @unchecked Sendable {
    /// Block width `bs`: one bonus token plus `bs - 1` proposals.
    public let blockTokens: Int
    let logitRows: MTLBuffer
    let greedyRows: MTLBuffer
    let hiddenRows: MTLBuffer

    /// - Parameters:
    ///   - blockTokens: 2...`SpeculativeBlock.maxTokens`.
    ///   - fusedGreedy: whether the producer drives the fused greedy head. The
    ///     logits rows are 512 KiB each at the pinned vocabulary, so a greedy
    ///     run allocates one row instead of `bs`.
    public init(context: MetalContext,
                vocab: Int,
                hiddenSize: Int,
                blockTokens: Int,
                fusedGreedy: Bool) throws {
        guard blockTokens >= 2, blockTokens <= SpeculativeBlock.maxTokens else {
            throw SpeculativeDraftError.blockSizeUnsupported(
                "draft block size \(blockTokens) is outside 2...\(SpeculativeBlock.maxTokens)")
        }
        let logitRowCount = fusedGreedy ? 1 : blockTokens
        guard let logitRows = context.device.makeBuffer(
                  length: SpeculativeBlock.logitRowsBytes(vocab: vocab,
                                                          blockTokens: logitRowCount),
                  options: .storageModeShared),
              let greedyRows = context.device.makeBuffer(
                  length: blockTokens * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let hiddenRows = context.device.makeBuffer(
                  length: blockTokens * hiddenSize * MemoryLayout<Float16>.stride,
                  options: .storageModePrivate)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        self.blockTokens = blockTokens
        self.logitRows = logitRows
        self.greedyRows = greedyRows
        self.hiddenRows = hiddenRows
    }
}

/// What the round bookkeeping saw. Diagnostics only: 22-GOAL-RESET §6 keeps
/// these out of any pass/fail column — the score is the end-to-end wall clock
/// of the goal task, measured against a same-session MTP-off run.
public struct SpeculativeStats: Sendable {
    public let blockTokens: Int
    public let rounds: Int
    public let proposed: Int
    public let accepted: Int
    /// `acceptedHistogram[a]` = rounds that accepted `a` drafts.
    public let acceptedHistogram: [Int]
    public let draftSeconds: Double
    public let verifySeconds: Double
    public let verifyBlocks: Int

    /// Mean accepted draft length per round — 14-M3.5 §4's `a`, measured in the
    /// loop instead of by the probe.
    public var meanAcceptedLength: Double {
        rounds > 0 ? Double(accepted) / Double(rounds) : 0
    }
}

public struct SpeculativeDecodeResult: Sendable {
    public let decode: RawDecodeResult
    public let speculative: SpeculativeStats
}

/// The MTP decode loop (`docs/mtp/03-DESIGN.md` D6).
///
/// One round, anchored on a bonus token `t` that has been emitted but has no
/// K/V row yet:
///
/// 1. the drafter proposes `bs - 1` tokens from `(t, the target hidden that
///    produced t, the target's K/V up to t)`;
/// 2. `verifyBlock` runs `[t, proposals...]` in one pass, leaving one target
///    token per row;
/// 3. the accept rule (D5) takes the longest prefix of proposals the target
///    itself would have drawn, and the target's own draw at the first mismatch
///    becomes the next bonus token;
/// 4. the K/V is rewound to just past the accepted tokens, so the cache is
///    exactly what a non-speculative run would have had.
///
/// Because every accepted token is a token the target drew, with the same
/// sampler position and seed a plain `runRawCompletion` would have used, the
/// text is the text of the non-speculative run. That identity is the gate, not
/// an aspiration (04-PHASES §3 gate 1).
public func runSpeculativeCompletion(producer: any LogitProducer,
                                     tokenizer: GFTokenizer,
                                     promptIds: [Int32],
                                     config: GenerationConfig,
                                     context: MetalContext,
                                     scratch: RawCompletionScratch,
                                     speculative: SpeculativeScratch,
                                     prefillConfig: PrefillRuntimeConfig = .defaultChunked,
                                     vision: VisionPrefillInput? = nil,
                                     start: RawCompletionStart = .reset,
                                     shouldStop: () -> Bool = { false },
                                     onProgress: (RawDecodeProgress) -> Void) async throws
    -> SpeculativeDecodeResult {
    guard var drafting = producer as? any SpeculativeDrafting else {
        throw SpeculativeDraftError.unsupportedConfig(
            "this producer cannot draft; run with --draft-block-size 0")
    }
    guard drafting.isDraftInstalled else {
        throw SpeculativeDraftError.notInstalled(
            "the model has no drafter installed; reinstall with --include-draft "
            + "or run with --draft-block-size 0")
    }
    guard speculative.blockTokens <= drafting.maxSpeculativeBlockTokens else {
        throw SpeculativeDraftError.blockSizeUnsupported(
            "draft block size \(speculative.blockTokens) is wider than the producer's "
            + "\(drafting.maxSpeculativeBlockTokens)-token ceiling")
    }
    // D5 accepts a proposal exactly when the target's own draw equals it, and
    // that draw has to be reproducible from `(position, seed, history)` alone.
    // A repetition penalty makes it depend on a history that the round has not
    // committed yet, so the honest answer is to refuse rather than to drift.
    guard config.repetitionPenalty == 1.0 else {
        throw SpeculativeDraftError.unsupportedConfig(
            "--repetition-penalty other than 1.0 cannot be verified; "
            + "run with --draft-block-size 0")
    }

    // Bound before prefill: the seed token's own producing row is the last row
    // of the prompt, and that head emission is where it gets captured.
    drafting.speculativeHiddenRows = speculative.hiddenRows
    defer { drafting.speculativeHiddenRows = nil }

    let prepared = try await prepareGeneration(producer: producer,
                                               promptIds: promptIds,
                                               config: config,
                                               scratch: scratch,
                                               prefillConfig: prefillConfig,
                                               vision: vision,
                                               start: start,
                                               onProgress: onProgress)
    guard let prefillSeed = prepared.prefillSeed else {
        throw SpeculativeDraftError.unsupportedConfig(
            "speculative decode needs chunked prefill; run with --prefill on")
    }

    let fusedGreedy = prepared.fusedGreedy
    var detok = GFDetokenizer(tokenizer: tokenizer,
                              barrierTokenIDs: tokenizer.structuralMarkerIDs)
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var history = prepared.history
    /// Where the next uncommitted token belongs; the K/V holds `[0, position)`.
    var position = prepared.position
    var generated = 0
    var reason: StopReason = .maxTokens
    var uncommittedBoundaryTokenIDs: [Int32] = []
    var timeToFirstToken: Double = 0

    // The producer's counters are cumulative across generations (a server reuses
    // one runner), so the stats this call reports are differences.
    let draftNanosBefore = drafting.draftNanos
    let verifyNanosBefore = drafting.verifyBlockNanos
    let verifyBlocksBefore = drafting.verifyBlockCount

    var rounds = 0
    var proposedTotal = 0
    var acceptedTotal = 0
    var histogram = [Int](repeating: 0, count: speculative.blockTokens)

    /// Tokens the target has already drawn but the loop has not emitted, with
    /// the K/V position each one occupies. The last entry of a round is the
    /// next bonus token, which has no row yet — its position is where the round
    /// left the cursor.
    var queue: [(token: Int32, position: Int)] = []
    /// Row of `speculative.hiddenRows` holding the post-norm hidden that
    /// produced the bonus token. Prefill leaves the prompt's final row at 0.
    var bonusHiddenRow = 0

    let hiddenRows = speculative.hiddenRows
    let vocab = scratch.sampler.vocab
    // Diagnostic control (04-PHASES §3 gate 1): `TF_MTP_DRAFTS=0` keeps the loop
    // and the verify path but proposes nothing, so every round is a one-token
    // block. If the text still differs from a plain decode of the same prompt,
    // the difference is the chunk path against the scalar decode path
    // (15-M4 §2) and not the speculation.
    let draftCap = ProcessInfo.processInfo.environment["TF_MTP_DRAFTS"].flatMap { Int($0) }

    /// The target's own token for row `row` of the verified block, drawn the way
    /// the non-speculative loop would have drawn the token with that generation
    /// index (D5).
    func targetToken(row: Int, generationIndex: Int) throws -> Int32 {
        if fusedGreedy {
            return Int32(bitPattern: speculative.greedyRows.contents().load(
                fromByteOffset: row * MemoryLayout<UInt32>.stride, as: UInt32.self))
        }
        // The sampler reads one vocabulary-wide buffer, so the row is staged
        // into it. 512 KiB per drawn row, and only for rows the accept loop
        // actually reaches.
        guard let cb = context.queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else {
            throw MetalError.noDevice
        }
        blit.copy(from: speculative.logitRows,
                  sourceOffset: row * vocab * MemoryLayout<Float16>.stride,
                  to: scratch.logits, destinationOffset: 0,
                  size: vocab * MemoryLayout<Float16>.stride)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        try checkCommandBufferError(cb.error)
        return try sampleOnce(scratch: scratch, context: context,
                              history: history, config: config,
                              position: generationIndex)
    }

    // The first bonus token is the prompt's own continuation, drawn exactly as
    // `runRawCompletion` draws it.
    let seedToken: Int32
    switch prefillSeed {
    case .greedyToken(let token):
        seedToken = Int32(bitPattern: token)
    case .logitsWritten:
        seedToken = try sampleOnce(scratch: scratch, context: context,
                                   history: history, config: config, position: 0)
    }
    queue.append((seedToken, position))
    /// The token a round is anchored on: the one emitted last, still uncommitted.
    var bonusToken = seedToken

    emit: while true {
        try Task.checkCancellation()

        if queue.isEmpty {
            // A round is anchored on the token emitted last, which is still
            // uncommitted: `position` is where it goes.
            let bonus = bonusToken
            let remaining = config.maxNewTokens - generated
            var draftCount = min(speculative.blockTokens - 1, max(0, remaining - 1))
            if let draftCap { draftCount = min(draftCount, draftCap) }
            let proposals = try drafting.draftProposals(bonusToken: bonus,
                                                        position: position,
                                                        hidden: hiddenRows,
                                                        hiddenRow: bonusHiddenRow,
                                                        count: draftCount)
            var block = [bonus]
            block.append(contentsOf: proposals)
            try await drafting.verifyBlock(tokens: block[0...],
                                           startPosition: position,
                                           into: speculative.logitRows,
                                           greedyTokens: fusedGreedy
                                               ? speculative.greedyRows : nil)

            var accepted = 0
            var next = try targetToken(row: 0, generationIndex: generated)
            while accepted < proposals.count, proposals[accepted] == next {
                accepted += 1
                next = try targetToken(row: accepted, generationIndex: generated + accepted)
            }

            for i in 0..<accepted {
                queue.append((proposals[i], position + 1 + i))
            }
            // The next bonus token has no row of its own; the cache stops at it.
            let bonusPosition = position + accepted + 1
            queue.append((next, bonusPosition))
            try drafting.rewind(to: bonusPosition)
            position = bonusPosition
            bonusHiddenRow = accepted

            rounds += 1
            proposedTotal += proposals.count
            acceptedTotal += accepted
            histogram[accepted] += 1
        }

        let (tokenID, tokenPosition) = queue.removeFirst()
        generated += 1
        if generated == 1 {
            timeToFirstToken = Date().timeIntervalSince(prepared.prefillStart)
        }
        uncommittedBoundaryTokenIDs = [tokenID]

        if tokenizer.stopTokenIDs.contains(tokenID) || config.extraStopTokens.contains(tokenID) {
            if tokenID == tokenizer.endOfTurnID {
                reason = .endOfTurn
            } else if tokenID == tokenizer.toolResponseID {
                reason = .toolCalls
            } else {
                reason = .eos
            }
            // A stop token is never committed, and neither is anything a round
            // speculated past it: the cache has to end where the text does.
            try drafting.rewind(to: tokenPosition)
            position = tokenPosition
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            break emit
        }

        let delta = detok.push(tokenID)
        let visible = stopMatcher.push(delta)
        onProgress(.token(index: generated - 1, id: tokenID, delta: visible))

        let hitStopString = stopMatcher.isStopped || shouldStop()
        let hitMax = generated >= config.maxNewTokens
        if hitStopString || hitMax {
            try drafting.rewind(to: tokenPosition)
            position = tokenPosition
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            reason = hitStopString ? .stopString : .maxTokens
            break emit
        }

        history.append(tokenID)
        bonusToken = tokenID
        uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
    }

    let decode = RawDecodeResult(prefillTokens: promptIds.count,
                                 cachedPromptTokens: prepared.cachedPromptTokens,
                                 computedPrefillTokens: prepared.computedPrefillTokens,
                                 prefillSeconds: prepared.prefillSeconds,
                                 newTokens: generated,
                                 decodeSeconds: Date().timeIntervalSince(prepared.decodeStart),
                                 timeToFirstTokenSeconds: timeToFirstToken,
                                 reason: reason,
                                 kvPosition: position,
                                 kvBackedTokenIDs: history,
                                 uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs)
    let stats = SpeculativeStats(blockTokens: speculative.blockTokens,
                                 rounds: rounds,
                                 proposed: proposedTotal,
                                 accepted: acceptedTotal,
                                 acceptedHistogram: histogram,
                                 draftSeconds: Double(drafting.draftNanos - draftNanosBefore) / 1e9,
                                 verifySeconds: Double(drafting.verifyBlockNanos - verifyNanosBefore) / 1e9,
                                 verifyBlocks: drafting.verifyBlockCount - verifyBlocksBefore)
    return SpeculativeDecodeResult(decode: decode, speculative: stats)
}
