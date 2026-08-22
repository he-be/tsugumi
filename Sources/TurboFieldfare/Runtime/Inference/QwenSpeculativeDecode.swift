import Foundation
import Metal

/// Width-2 speculative decoding with the grafted MTP head
/// (`docs/qwen35moe/36-MTP-DECODE.md`).
///
/// One pass of the loop:
///
///   1. the head drafts `d`, the token it thinks follows the one just emitted
///      (`QwenMTPDrafter`);
///   2. the body runs **two rows in one pass** — the emitted token and `d` —
///      through the same T-row path the prompt uses;
///   3. one pass over the 508 MB `lm_head` scores both rows
///      (`QwenLMHeadChainInt8.encodeGreedyDecodeRows`);
///   4. row 0's argmax is the real next token. If `d` matched it, row 1's
///      argmax is real too and the pass emitted **two** tokens; otherwise row 1
///      is discarded.
///
/// Why width 2 and not 4: the verify pass's cost grows with the **union** of
/// the experts its rows route to, not with the number of rows — 1.55x for two
/// rows, 2.43x for four — while the acceptance chain pays a whole drafter pass
/// per extra token and returns 78.7%, then 41.5%, then 14.2%
/// (`33-MTP-ACCEPTANCE.md` §2-2, §3-1). Three is a wash on paper and four
/// loses.
///
/// **Greedy only, and the output must be identical to the non-speculative
/// loop.** That is the correctness statement this path is checked by: a
/// rejected draft leaves nothing behind (`kv.rewind` for the ten attention
/// layers, the snapshot for the thirty recurrent ones), so the token stream is
/// a function of the model, not of what the head guessed.
public struct QwenSpeculativeStats: Sendable {
    /// Verify passes run.
    public var passes = 0
    /// Passes whose draft matched — the ones that emitted two tokens.
    public var accepted = 0
    /// Tokens the loop emitted.
    public var tokens = 0
    /// Seconds inside the drafter (its own command buffers, host time).
    public var draftSeconds: Double = 0
    /// Seconds inside the verify pass, snapshot and restore included.
    public var verifySeconds: Double = 0
    /// Seconds spent moving the recurrent state around a rejected row.
    ///
    /// **Zero by construction, and printed anyway.** The speculative row writes
    /// a shadow buffer and acceptance is a pointer swap, so there is no copy to
    /// time (`33-MTP-ACCEPTANCE.md` §3-6 (a)). A non-zero value here would mean
    /// the loop had fallen back to blitting 61.4 MiB a pass.
    public var snapshotSeconds: Double = 0

    /// P1 as this run saw it. The measurement in `33` §2-1 is the same
    /// quantity computed off-policy in float32; this one is on-policy and
    /// quantized.
    public var acceptanceRate: Double {
        passes == 0 ? 0 : Double(accepted) / Double(passes)
    }
    /// Tokens per verify pass — `a(k=2)` measured rather than derived.
    public var acceptanceLength: Double {
        passes == 0 ? 0 : Double(tokens) / Double(passes)
    }
}

extension QwenForwardRunner {

    public enum SpeculativeError: Error, CustomStringConvertible {
        case noHead
        case constrained

        public var description: String {
            switch self {
            case .noHead:
                return "speculative decoding needs an MTP head; "
                    + "build one with Scripts/qwen35/build_mtp_sidecar.py"
            case .constrained:
                return "speculative decoding and a grammar constraint are not "
                    + "wired together yet; drop --tools or --qwen-mtp"
            }
        }
    }

    /// Load the sidecar and build the drafter. Idempotent.
    public func attachMTPHead(directory: String = QwenMTPSidecar.defaultDirectory) throws {
        if mtpDrafter != nil { return }
        let sidecar = try QwenMTPSidecar(directory: directory, device: ctx.device)
        // The MoE kernels take one `MoEExpertOffsets` for all eight blobs, so a
        // sidecar laid out differently from the body's `packed_experts` would
        // read plausible bytes with the wrong meaning.
        try sidecar.checkExpertLayout(matches: model.routedExpertOffsets(layer: 0))
        mtpDrafter = try QwenMTPDrafter(sidecar: sidecar,
                                        context: ctx,
                                        model: model,
                                        config: cfg,
                                        rms: rms,
                                        int8: int8,
                                        qwen: qwen,
                                        attention: attention,
                                        moe: moe,
                                        head: head,
                                        embed: embed,
                                        unitFeatureScale: unitFeatureScale,
                                        unitExpertScale: unitExpertScale,
                                        scoredVocab: scoredVocab,
                                        maxContext: maxContext,
                                        rmsEps: rmsEps)
    }

    public var hasMTPHead: Bool { mtpDrafter != nil }

    /// Negative control: throw every draft away, so the loop still runs the
    /// width-2 verify pass and the rollback but can never accept.
    ///
    /// A speculative loop that only ever agrees with itself is not evidence
    /// (`PLAN_VISION.md` §6-3). With this on, the emitted stream is produced by
    /// row 0 alone — the same rows, the same rollback, one token per pass — so
    /// **a difference against this arm is the accept path, and a difference
    /// between this arm and plain decode is the T-row kernels.** The two
    /// questions are separated by one environment variable.
    public static let forceReject =
        ProcessInfo.processInfo.environment["TF_QWEN_MTP_FORCE_REJECT"] == "1"

    /// Second negative control: run the loop with **width 1** — no draft, one
    /// row per pass — but still through the T-row chunk path.
    ///
    /// This separates the two things that could make a speculative step slow.
    /// A decode step costs 50 ms/tok on the per-token kernels
    /// (`27-PHASE6-THROUGHPUT.md` §2); if one row through the chunk path costs
    /// the same, then everything above it is the price of the second row, which
    /// is what `33` §3-7 measured as 1.27x. If one row already costs much more,
    /// the chunk path is simply the wrong machinery for two rows and the
    /// speculative arithmetic never gets a chance.
    public static let noDraft =
        ProcessInfo.processInfo.environment["TF_QWEN_MTP_NO_DRAFT"] == "1"

    /// Which routed-expert kernels the verify pass runs on, independently of
    /// what the prompt used.
    ///
    /// **Per-pair, against the prompt's default.** The tiled path won at 512
    /// rows and every width `24-PREFILL-MOE-PATH.md` measured, but a 64-row
    /// GEMM tile carrying two rows throws away 97% of itself: measured on the
    /// coding task, the verify pass costs 153.9 ms tiled and 117.9 ms per-pair
    /// (`36-MTP-DECODE.md` §4-2).
    public static let verifyRoutedPath: PrefillRoutedPath =
        ProcessInfo.processInfo.environment["TF_QWEN_MTP_VERIFY_MOE"]
            .flatMap(PrefillRoutedPath.init(rawValue:)) ?? .perPair

    /// Keep the prompt's command-buffer structure in the verify pass. The
    /// control arm for the compaction (`36-MTP-DECODE.md` §4-2).
    public static let wideCommandBuffers =
        ProcessInfo.processInfo.environment["TF_QWEN_MTP_WIDE_CB"] == "1"

    /// The same contract as `runGreedyCompletion`, with the MTP head in the
    /// loop. The token stream is identical; the wall clock is not.
    public func runGreedyCompletionMTP(
        promptTokens: [Int32],
        maxNewTokens: Int,
        chunkWidth: Int = 512,
        stopTokens: Set<Int32> = [],
        constraint: (any GenerationConstraint)? = nil,
        shouldStop: () -> Bool = { false },
        onPrefill: ((Int, Double) -> Void)? = nil,
        onToken: ((Int, Int32) throws -> Void)? = nil,
        onStats: ((QwenSpeculativeStats) -> Void)? = nil
    ) throws -> QwenGreedyRun {
        guard let drafter = mtpDrafter else { throw SpeculativeError.noHead }
        guard constraint == nil else { throw SpeculativeError.constrained }
        precondition(!promptTokens.isEmpty, "the prompt must have at least one token")
        precondition(promptTokens.count + maxNewTokens + 1 <= maxContext,
                     "prompt + generation exceeds maxContext \(maxContext)")

        let D = hiddenSize
        let rowBytes = D * MemoryLayout<Float16>.size
        // The chunk scratch the prompt already built is wide enough for two
        // rows; `runChunkLayers` sizes every dispatch from `tokens.count`, so a
        // second, narrower context would only cost memory.
        let verify = try prefillContext(width: 2)
        let (normed, rowTokens) = try verifyBuffers()
        try state.ensureShadow(device: ctx.device)

        var stats = QwenSpeculativeStats()
        let startedAt = DispatchTime.now()
        let firstToken = try prefill(tokens: promptTokens, chunkWidth: chunkWidth)
        // The prompt's last row, already through `model.norm` — the head chain
        // left it in `normalizedHidden` and nothing has run since.
        memcpy(normed.contents(), head.normalizedHidden.contents(), rowBytes)
        let prefillSeconds = Self.seconds(since: startedAt)
        onPrefill?(promptTokens.count, prefillSeconds)

        let decodeStart = DispatchTime.now()
        var produced: [Int32] = []
        var reason: StopReason = .maxTokens
        var timeToFirstToken: Double = 0

        /// The token that has been emitted but not yet fed to the body.
        var pending = firstToken
        /// The body position whose hidden sits in row `hiddenRow` of `normed`.
        var lastPosition = kv.position - 1
        var hiddenRow = 0
        /// Positions the head still owes its cache a `(k, v)` for, oldest first.
        var catchUp: [QwenMTPDrafter.Row] = []

        func emit(_ token: Int32) throws -> Bool {
            produced.append(token)
            stats.tokens += 1
            if produced.count == 1 { timeToFirstToken = Self.seconds(since: startedAt) }
            try onToken?(produced.count - 1, token)
            if stopTokens.contains(token) { reason = .endOfTurn; return false }
            if shouldStop() { reason = .stopString; return false }
            if produced.count >= maxNewTokens { return false }
            return true
        }

        var running = try emit(pending)
        while running {
            try Task.checkCancellation()

            // --- 1. draft ----------------------------------------------------
            let draftStart = DispatchTime.now()
            var draft: Int32 = 0
            if !Self.noDraft {
                let rows = catchUp + [QwenMTPDrafter.Row(
                    hiddenOffset: hiddenRow * rowBytes,
                    token: pending,
                    position: lastPosition)]
                draft = try drafter.draft(baseNormed: normed, rows: rows)
            }
            // The control feeds the body a token the head did not pick, so row
            // 1 is certain to be rejected. Any token that is not row 0's argmax
            // would do; the body still has to run it, so it has to be a real id.
            if Self.forceReject { draft = draft == 0 ? 1 : 0 }
            catchUp.removeAll(keepingCapacity: true)
            stats.draftSeconds += Self.seconds(since: draftStart)

            // --- 2. verify ---------------------------------------------------
            let verifyStart = DispatchTime.now()
            let start = kv.position
            let savedPath = prefillRoutedPath
            prefillRoutedPath = Self.verifyRoutedPath
            compactChunkCommandBuffers = !Self.wideCommandBuffers
            // Row 1 is the guess: its recurrent state goes to the shadow so
            // that rejecting it is doing nothing at all.
            // Width 1 needs no shadow: there is no speculative row to discard.
            speculativeLastRow = !Self.noDraft
            try runChunkLayers(Self.noDraft ? [pending] : [pending, draft],
                               scratch: verify)
            speculativeLastRow = false
            compactChunkCommandBuffers = false
            prefillRoutedPath = savedPath
            let lm = try model.qwenLMHead
            let finalNorm = model.finalNorm
            try runSync("mtp verify head") { cb in
                verify.rms.encodeBF16W(commandBuffer: cb,
                                       x: verify.hidden,
                                       weight: finalNorm.buffer,
                                       weightOffset: Int(finalNorm.offset),
                                       out: normed,
                                       t: Self.noDraft ? 1 : 2,
                                       d: UInt32(D), eps: self.rmsEps)
                self.head.encodeGreedyDecodeRows(
                    commandBuffer: cb,
                    hiddenNormed: normed,
                    weights: lm.buffer, weightsOffset: Int(lm.offset),
                    scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                    outTokens: rowTokens,
                    rows: Self.noDraft ? 1 : 2,
                    d: UInt32(D),
                    vocab: UInt32(self.scoredVocab))
            }
            let ids = rowTokens.contents().bindMemory(to: UInt32.self, capacity: 2)
            let y0 = Int32(bitPattern: ids[0])
            let y1 = Int32(bitPattern: ids[1])
            stats.passes += 1

            // --- 3. accept or roll back ---------------------------------------
            if !Self.noDraft && draft == y0 {
                stats.accepted += 1
                // Both rows stand. The skipped position still owes the head a
                // `(k, v)`; it is the row that consumed `pending`.
                // Row 1's recurrent state was written to the shadow; taking it
                // is two pointer swaps.
                state.adoptShadow()
                catchUp.append(QwenMTPDrafter.Row(hiddenOffset: 0,
                                                  token: y0,
                                                  position: start))
                lastPosition = start + 1
                hiddenRow = 1
                pending = y1
                stats.verifySeconds += Self.seconds(since: verifyStart)
                running = try emit(y0)
                if running { running = try emit(y1) }
            } else {
                // Nothing to undo on the recurrent side: row 1's state went to
                // the shadow and the state itself still holds row 0's. The
                // attention layers only need their cursor moved back.
                if !Self.noDraft { kv.rewind(to: start + 1) }
                lastPosition = start
                hiddenRow = 0
                pending = y0
                stats.verifySeconds += Self.seconds(since: verifyStart)
                running = try emit(y0)
            }
        }

        onStats?(stats)
        return QwenGreedyRun(tokens: produced,
                             promptTokens: promptTokens.count,
                             prefillSeconds: prefillSeconds,
                             decodeSeconds: Self.seconds(since: decodeStart),
                             timeToFirstTokenSeconds: timeToFirstToken,
                             reason: reason)
    }

    /// The two buffers the verify pass and the drafter share. Allocated on
    /// first use, so a non-speculative run never has them.
    private func verifyBuffers() throws -> (MTLBuffer, MTLBuffer) {
        let rows = QwenLMHeadChainInt8.maxHiddenRows
        if verifyNormedBuffer == nil {
            guard let normed = ctx.device.makeBuffer(
                      length: rows * hiddenSize * MemoryLayout<Float16>.size,
                      options: .storageModeShared),
                  let tokens = ctx.device.makeBuffer(
                      length: rows * MemoryLayout<UInt32>.size,
                      options: .storageModeShared) else {
                throw MetalError.noDevice
            }
            normed.label = "qwen.verifyNormed"
            tokens.label = "qwen.verifyTokens"
            verifyNormedBuffer = normed
            verifyTokensBuffer = tokens
        }
        return (verifyNormedBuffer!, verifyTokensBuffer!)
    }

    private static func seconds(since start: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds &- start.uptimeNanoseconds) / 1e9
    }
}
