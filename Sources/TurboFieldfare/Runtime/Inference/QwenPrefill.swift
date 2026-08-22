import Foundation
import Metal

/// Chunked prefill for Qwen3.5-MoE (`docs/qwen35moe/04-PHASES.md` Phase 4).
///
/// Phase 3 ran the prompt one token at a time through the decode path, which
/// is what made its comparison against the reference a statement about decode
/// alone (`docs/qwen35moe/20-PHASE3-DECODE.md`). This is the T-row path: the
/// same forty layers, the same order, everything widened from a vector to a
/// chunk of rows.
///
/// **Still deliberately serial.** One command buffer per stage, waited on
/// before the next is encoded, for the same reason decode is: Phase 4's exit
/// condition is that prefill and decode produce the same tokens, and a serial
/// loop is the shape in which a disagreement can be named. The three-command
/// pipeline, the tile scheduler and the row-path A/B that `RealForwardRunner`
/// carries are measurements against Gemma 4, and belong to Phase 6 here.
///
/// What is new, and what is borrowed:
///
/// | stage | kernel |
/// | --- | --- |
/// | embedding | `qwen_embed_lookup_int8_block` (new — the table is 8-bit) |
/// | norm | `prefill_rmsnorm_bf16w_block` (Gemma's, unchanged) |
/// | 4-bit projections | `PrefillInt4QMM` (Gemma's, unchanged) |
/// | 8-bit projections | `QwenPrefillInt8QMM` (new — see its header) |
/// | linear attention | `qwen_delta_*` + `qwen_delta_rule` at T > 1 |
/// | full attention | `qwen_qkv_epilogue`, `qwen_query_compact` (new), `attention_prefill_causal_qblock_d256` |
/// | router | `prefill_router_gemma4*_block` with unit scales |
/// | routed experts | `prefill_grouped_routed_moe_batched_*`, SiLU by function constant |
/// | shared expert | INT8 QMMs + `qwen_silu_mul` + `qwen_moe_shared_gate_logit_block` (new) |
///
/// The routed experts have **two** paths, picked by `prefillRoutedPath`: the
/// per-pair GEMVs Phase 4 wired (no alignment precondition, no batch planner,
/// so the first prefill that ran was the one with the fewest moving parts
/// between the router and the answer) and the tiled `prefill_moe_gemm_int4`.
/// Which one is faster is `docs/qwen35moe/05-RISKS.md` §1-2's question, and it
/// needed a measurement rather than a guess
/// (`docs/qwen35moe/24-PREFILL-MOE-PATH.md`).

/// Everything a chunk of width `width` needs, allocated once.
///
/// Sized for the widest chunk the caller asked for, so a narrower final chunk
/// reuses the same buffers. Nothing here is per-layer: a layer's work is
/// drained before the next layer's is encoded.
final class QwenPrefillContext {
    let width: Int

    // Kernels.
    let rms: PrefillRMSNorm
    let int4: PrefillInt4QMM
    let int8: QwenPrefillInt8QMM
    let router: PrefillRouter
    let routedMoE: PrefillGroupedRoutedMoE
    let moeReduce: PrefillMoE
    let attention: PrefillAttention

    // Activations.
    let tokenIDs: MTLBuffer        // [T] UInt32
    let hidden: MTLBuffer          // [T, D] FP16 — the residual stream
    let normed: MTLBuffer          // [T, D] FP16
    let branchOut: MTLBuffer       // [T, D] FP16
    let wide: MTLBuffer            // [T, max(qkvWidth, 2*NQ*HD)] FP16
    let zBuf: MTLBuffer            // [T, Hv*Dv] FP16
    let aBuf: MTLBuffer            // [T, Hv] FP16
    let bBuf: MTLBuffer            // [T, Hv] FP16
    let gBuf: MTLBuffer            // [T, Hv] FP32
    let betaBuf: MTLBuffer         // [T, Hv] FP32
    let qDelta: MTLBuffer          // [T, Hk*Dk] FP16
    let kDelta: MTLBuffer          // [T, Hk*Dk] FP16
    let vDelta: MTLBuffer          // [T, Hv*Dv] FP16
    let yDelta: MTLBuffer          // [T, Hv*Dv] FP16
    let oDelta: MTLBuffer          // [T, Hv*Dv] FP16
    let qCompact: MTLBuffer        // [T, NQ*HD] FP16
    let attnOut: MTLBuffer         // [T, NQ*HD] FP16
    let sharedGateAct: MTLBuffer   // [T, F] FP16
    let sharedUpAct: MTLBuffer     // [T, F] FP16
    let sharedAct: MTLBuffer       // [T, F] FP16
    let sharedOut: MTLBuffer       // [T, D] FP16
    let sharedGateLogit: MTLBuffer // [T] FP16
    let routedOut: MTLBuffer       // [T, D] FP16
    let routerIndices: MTLBuffer   // [T, topK] UInt32
    let routerWeights: MTLBuffer   // [T, topK] FP16
    /// The **next** layer's router, run over this layer's `normed` — the
    /// cross-layer read-ahead's guess (`docs/qwen35moe/39-RESIDENCY-COMMIT.md`).
    /// Written only when the verify pass asks for it; 24 KB at width 512.
    let previewIndices: MTLBuffer  // [T, topK] UInt32
    let previewWeights: MTLBuffer  // [T, topK] FP16
    let routePartials: MTLBuffer   // [T*topK, D] FP16
    let gateUpAct: MTLBuffer       // [3, max(microbatch, gemmBatchRows), F] FP16
    let downScratch: MTLBuffer     // [microbatch, D] FP16
    /// The conv window the chunk hands to the next chunk.
    ///
    /// `qwen_delta_qkv_prepare` refuses to alias its state in and out for a
    /// dispatch with more than one token block — the group that writes the
    /// state and the group that reads it would be different threadgroups with
    /// no ordering between them (`QwenKernels.encodeDeltaQKVPrepare`). Decode
    /// is always one block and aliases; a chunk writes here and the layer
    /// blits it back.
    let convStateOut: MTLBuffer
    let greedyToken: MTLBuffer     // [1] UInt32

    /// Route pairs one `encodeStreamedBatched` dispatch carries. The scratch
    /// it writes is `3 * rows * F` halves, so this is a memory knob, not a
    /// correctness one.
    static let pairMicrobatchRows = 32

    /// The most rows one `encodeStreamedTiled` batch may describe, before the
    /// chunk's own pair count caps it. Gemma's prefill layout uses the same
    /// 2048 (`PrefillChunkScratch`), and the planner rounds down to the 64-row
    /// tile anyway.
    static let gemmBatchRowCap = 2048

    /// Rows a tiled batch actually gets, given this chunk's width. A chunk of
    /// `width` tokens produces at most `width * topK` pairs, so a wider budget
    /// than that would only make `gateUpAct` bigger.
    let gemmBatchRows: Int

    init(device: MTLDevice,
         context: MetalContext,
         config cfg: ArchConfig,
         width: Int,
         qkvWidth: Int,
         valueWidth: Int,
         keyWidth: Int,
         numValueHeads: Int,
         routerWeightBits: Int,
         gateActivation: MoE.GateActivation) throws {
        precondition(width > 0, "prefill chunk width must be positive")
        self.width = width
        self.rms = try PrefillRMSNorm(context: context)
        self.int4 = try PrefillInt4QMM(context: context)
        self.int8 = try QwenPrefillInt8QMM(context: context)
        self.router = try PrefillRouter(context: context, routerWeightBits: routerWeightBits)
        self.routedMoE = try PrefillGroupedRoutedMoE(context: context,
                                                     gateActivation: gateActivation)
        self.moeReduce = try PrefillMoE(context: context)
        self.attention = try PrefillAttention(context: context)

        func buffer(_ count: Int,
                    _ stride: Int = MemoryLayout<Float16>.size,
                    label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(count * stride, 4),
                                                 options: .storageModeShared) else {
                throw QwenForwardRunner.QwenRunnerError.commandBufferFailed(
                    "could not allocate prefill scratch \(label)")
            }
            buffer.label = "qwen.prefill.\(label)"
            return buffer
        }

        let d = cfg.hiddenSize
        let f = cfg.moeIntermediateSize
        let topK = cfg.topKExperts
        let fullWidth = cfg.numHeads * cfg.fullHeadDim
        let rows = Self.pairMicrobatchRows
        // Sized for both routed paths at once, not for the one this run picks:
        // `prefillRoutedPath` is settable between runs (the A/B in
        // `docs/qwen35moe/24-PREFILL-MOE-PATH.md`) and the scratch is cached by
        // width, so a buffer sized for the path in force at allocation time
        // would be a stale-scratch bug the first time the A/B flipped.
        self.gemmBatchRows = max(64, min(Self.gemmBatchRowCap,
                                         ((width * topK + 63) / 64) * 64))
        let actRows = max(rows, gemmBatchRows)

        self.tokenIDs = try buffer(width, MemoryLayout<UInt32>.size, label: "tokenIDs")
        self.hidden = try buffer(width * d, label: "hidden")
        self.normed = try buffer(width * d, label: "normed")
        self.branchOut = try buffer(width * d, label: "branchOut")
        self.wide = try buffer(width * max(qkvWidth, 2 * fullWidth), label: "wide")
        self.zBuf = try buffer(width * valueWidth, label: "z")
        self.aBuf = try buffer(width * numValueHeads, label: "a")
        self.bBuf = try buffer(width * numValueHeads, label: "b")
        self.gBuf = try buffer(width * numValueHeads, MemoryLayout<Float>.size, label: "g")
        self.betaBuf = try buffer(width * numValueHeads, MemoryLayout<Float>.size, label: "beta")
        self.qDelta = try buffer(width * keyWidth, label: "qDelta")
        self.kDelta = try buffer(width * keyWidth, label: "kDelta")
        self.vDelta = try buffer(width * valueWidth, label: "vDelta")
        self.yDelta = try buffer(width * valueWidth, label: "yDelta")
        self.oDelta = try buffer(width * valueWidth, label: "oDelta")
        self.qCompact = try buffer(width * fullWidth, label: "qCompact")
        self.attnOut = try buffer(width * fullWidth, label: "attnOut")
        self.sharedGateAct = try buffer(width * f, label: "sharedGateAct")
        self.sharedUpAct = try buffer(width * f, label: "sharedUpAct")
        self.sharedAct = try buffer(width * f, label: "sharedAct")
        self.sharedOut = try buffer(width * d, label: "sharedOut")
        self.sharedGateLogit = try buffer(width, label: "sharedGateLogit")
        self.routedOut = try buffer(width * d, label: "routedOut")
        self.routerIndices = try buffer(width * topK, MemoryLayout<UInt32>.size, label: "routerIndices")
        self.routerWeights = try buffer(width * topK, label: "routerWeights")
        self.previewIndices = try buffer(width * topK, MemoryLayout<UInt32>.size,
                                         label: "previewIndices")
        self.previewWeights = try buffer(width * topK, label: "previewWeights")
        self.routePartials = try buffer(width * topK * d, label: "routePartials")
        self.gateUpAct = try buffer(3 * actRows * f, label: "gateUpAct")
        self.downScratch = try buffer(rows * d, label: "downScratch")
        self.convStateOut = try buffer((QwenKernels.convKernel - 1) * qkvWidth,
                                       label: "convStateOut")
        self.greedyToken = try buffer(1, MemoryLayout<UInt32>.size, label: "greedyToken")
    }
}

/// What one greedy completion produced, and the partition of its wall clock.
///
/// The Gemma path answers this with `RawDecodeResult`, which carries a K/V
/// snapshot and a cached-prefix count besides. Neither has a meaning here:
/// this family's prompt cache is off because a recurrent state cannot be
/// rewound (`docs/qwen35moe/03-DESIGN.md` §5), so every run starts at position
/// zero and the whole prompt is computed.
public struct QwenGreedyRun: Sendable {
    /// The generated tokens, stop token included when one ended the run — the
    /// caller decides whether to show it, exactly as the CLI does.
    public let tokens: [Int32]
    /// SPEC §9 RSP-3 `prompt_n`: the prompt this run actually computed.
    public let promptTokens: Int
    /// RSP-3 `prompt_ms`.
    public let prefillSeconds: Double
    /// RSP-3 `predicted_ms`.
    public let decodeSeconds: Double
    /// From the start of prefill to the first generated token.
    public let timeToFirstTokenSeconds: Double
    public let reason: StopReason

    public var newTokens: Int { tokens.count }
}

extension QwenForwardRunner {

    /// Runs `tokens` through the model in chunks and returns the greedy
    /// continuation of `maxNewTokens`, generated one token at a time by
    /// `step`.
    ///
    /// The split is the point of the phase: the prompt goes through the T-row
    /// path, and everything after it through the decode path Phase 3 already
    /// matched against the reference. Both write the same recurrent state and
    /// the same K/V, so the handover is the thing under test.
    public func generateGreedyPrefilled(promptTokens: [Int32],
                                        maxNewTokens: Int,
                                        chunkWidth: Int = 512,
                                        stopTokens: Set<Int32> = [],
                                        constraint: (any GenerationConstraint)? = nil,
                                        onToken: ((Int, Int32) throws -> Void)? = nil) throws -> [Int32] {
        try runGreedyCompletion(promptTokens: promptTokens,
                                maxNewTokens: maxNewTokens,
                                chunkWidth: chunkWidth,
                                stopTokens: stopTokens,
                                constraint: constraint,
                                onToken: onToken).tokens
    }

    /// The same loop, with the three things a server needs and a check does
    /// not: a way to be stopped from outside, a way to be cancelled, and the
    /// partition of the wall clock that SPEC §9 RSP-3 reports.
    ///
    /// One loop, not two. `generateGreedyPrefilled` above is this function with
    /// its extra answers dropped — so the rule that decides where a constrained
    /// draw comes from, and the rule that a stop token is *emitted* and then
    /// ends the run, exist once (`docs/qwen35moe/26-PHASE8-SERVER.md` §2).
    ///
    /// `shouldStop` is asked after the caller has seen the token, which is what
    /// makes a stop *string* work: the matcher upstream only knows the answer
    /// once the text of that token has reached it. A run ended that way reports
    /// `.stopString`, and the token that triggered it has already been emitted
    /// — the caller is the one holding text back, not this loop.
    public func runGreedyCompletion(
        promptTokens: [Int32],
        maxNewTokens: Int,
        chunkWidth: Int = 512,
        stopTokens: Set<Int32> = [],
        constraint: (any GenerationConstraint)? = nil,
        shouldStop: () -> Bool = { false },
        onPrefill: ((Int, Double) -> Void)? = nil,
        onToken: ((Int, Int32) throws -> Void)? = nil
    ) throws -> QwenGreedyRun {
        precondition(!promptTokens.isEmpty, "the prompt must have at least one token")
        precondition(promptTokens.count + maxNewTokens <= maxContext,
                     "prompt + generation exceeds maxContext \(maxContext)")

        // GEN-7 is applied to the token the prompt produces too: the first
        // generated token is as much part of the constrained sequence as the
        // rest, and with `tool_choice: required` it is the one the preamble
        // rule is about.
        let gate = constraint.map {
            ConstraintGate(constraint: $0, endOfGenerationTokenIDs: stopTokens)
        }
        let startedAt = DispatchTime.now()
        let rawToken = try prefill(tokens: promptTokens, chunkWidth: chunkWidth)
        // The prompt's own clock stops when the last chunk has run — the head
        // pass that produced `rawToken` is part of it, and the rescore that a
        // constraint may add on top of it is not: that one is spent on the
        // first *generated* token.
        let prefillSeconds = Self.seconds(since: startedAt)
        onPrefill?(promptTokens.count, prefillSeconds)
        let decodeStart = DispatchTime.now()
        var next = try constrained(rawToken, gate: gate, position: 0)
        var produced: [Int32] = []
        var reason: StopReason = .maxTokens
        var timeToFirstToken: Double = 0
        for index in 0..<maxNewTokens {
            produced.append(next)
            if index == 0 { timeToFirstToken = Self.seconds(since: startedAt) }
            try gate?.accept(next)
            try onToken?(index, next)
            if stopTokens.contains(next) { reason = .endOfTurn; break }
            if shouldStop() { reason = .stopString; break }
            if produced.count == maxNewTokens { break }
            // Between two tokens is the only place a cancelled request can be
            // dropped: one step is forty layers of command buffers that are
            // waited on, and nothing inside them is interruptible.
            try Task.checkCancellation()
            next = try constrained(try step(token: next, emitToken: true),
                                   gate: gate, position: index + 1)
        }
        return QwenGreedyRun(tokens: produced,
                             promptTokens: promptTokens.count,
                             prefillSeconds: prefillSeconds,
                             decodeSeconds: Self.seconds(since: decodeStart),
                             timeToFirstTokenSeconds: timeToFirstToken,
                             reason: reason)
    }

    private static func seconds(since start: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds &- start.uptimeNanoseconds) / 1e9
    }

    /// Runs `tokens` through the T-row path and returns the greedy token that
    /// follows them.
    ///
    /// The chunk boundary is invisible to the model: the recurrent state and
    /// the K/V cursor carry across it exactly as they carry across a token in
    /// decode. Only the last chunk pays for the LM head, and only for its last
    /// row.
    @discardableResult
    public func prefill(tokens: [Int32], chunkWidth: Int = 512) throws -> Int32 {
        precondition(!tokens.isEmpty, "prefill needs at least one token")
        precondition(chunkWidth > 0, "chunk width must be positive")
        guard kv.position + tokens.count <= maxContext else {
            throw QwenRunnerError.geometryMismatch(
                "prefill of \(tokens.count) from position \(kv.position) "
                + "exceeds maxContext \(maxContext)")
        }
        // **要求された幅で切る。**scratch は「その幅を載せられる大きさ」でしかない
        // ので、前の呼び出しが作った広い scratch を使い回すのは正しいが、
        // *切り方*までそれに合わせてはいけない。ここが `ctxScratch.width` だった
        // 間、1 プロセスの中で幅を変えても 2 本目以降は最初の幅で走っていた
        // (`docs/qwen35moe/35-PREFILL-CHUNK-WIDTH.md`)。
        let width = min(chunkWidth, tokens.count)
        let ctxScratch = try prefillContext(width: width)
        var token: Int32 = 0
        var offset = 0
        while offset < tokens.count {
            let count = min(width, tokens.count - offset)
            let last = offset + count == tokens.count
            token = try prefillChunk(Array(tokens[offset..<(offset + count)]),
                                     scratch: ctxScratch,
                                     emitToken: last)
            offset += count
        }
        return token
    }

    /// Allocates the chunk scratch on first use and reuses it after.
    func prefillContext(width: Int) throws -> QwenPrefillContext {
        if let existing = prefillScratch, existing.width >= width { return existing }
        let created = try QwenPrefillContext(
            device: ctx.device,
            context: ctx,
            config: cfg,
            width: width,
            qkvWidth: qkvWidth,
            valueWidth: valueWidth,
            keyWidth: numKeyHeads * keyHeadDim,
            numValueHeads: numValueHeads,
            routerWeightBits: model.routerWeightBits,
            gateActivation: fault == .routedActivationGelu ? .geluPytorchTanh : .silu)
        prefillScratch = created
        return created
    }

    // MARK: - One chunk

    private func prefillChunk(_ tokens: [Int32],
                              scratch s: QwenPrefillContext,
                              emitToken: Bool) throws -> Int32 {
        try runChunkLayers(tokens, scratch: s)
        guard emitToken else { return 0 }

        // Only the last row is scored: the token that follows the chunk is the
        // argmax over the last position's logits, and the 248,077-row table is
        // 508 MB to read (`docs/qwen35moe/19-LM-HEAD-INT8.md`).
        let lm = try model.qwenLMHead
        let finalNorm = model.finalNorm
        let T = tokens.count
        try runSync("prefill head") { cb in
            self.head.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: s.hidden,
                hiddenOffset: (T - 1) * self.hiddenSize * MemoryLayout<Float16>.size,
                normWeight: finalNorm.buffer, normOffset: Int(finalNorm.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: s.greedyToken,
                d: UInt32(self.hiddenSize),
                vocab: UInt32(self.scoredVocab),
                rmsEps: self.rmsEps)
        }
        return Int32(bitPattern: s.greedyToken.contents().load(as: UInt32.self))
    }

    /// Everything a chunk does below the head: forty layers, the K/V cursor and
    /// the recurrent state.
    ///
    /// Split out because the speculative verify pass runs the same T rows and
    /// then scores **all** of them rather than the last
    /// (`QwenSpeculativeDecode.swift`). Nothing here knows which caller it has.
    func runChunkLayers(_ tokens: [Int32],
                        scratch s: QwenPrefillContext) throws {
        let T = tokens.count
        let start = kv.position
        let D = UInt32(hiddenSize)
        // The routed-expert counters are per phase; a chunk is prefill, and
        // the step it is stamped with is where the chunk starts
        // (`ExpertTelemetry`, same convention as `RealForwardRunner`).
        model.telemetry.beginPhase(.prefill, step: start)
        if fault == .forgetRecurrentState { state.reset() }

        // The cross-layer read-ahead, when the verify pass asked for it. It is
        // per call rather than per runner: a chunk that throws must not leave a
        // read in flight against the next one's slots.
        let prefetching = chunkExpertPrefetch
            && QwenForwardRunner.verifyPrefetchTopN > 0
            && model.usesMappedExperts
            && QwenForwardRunner.pipelineEnabled
        var chunkPrefetch: (layer: Int, handle: RoutedExpertFetchHandle)?
        var chunkPredicted: [Int]?
        defer { if let pending = chunkPrefetch { _ = try? pending.handle.wait() } }

        let ids = s.tokenIDs.contents().bindMemory(to: UInt32.self, capacity: T)
        for (index, token) in tokens.enumerated() { ids[index] = UInt32(bitPattern: token) }

        let embedding = model.embedding
        try stage(.embed) {
            try runSync("prefill embed") { cb in
                self.embed.encodeBlock(commandBuffer: cb,
                                       table: embedding.buffer, tableOffset: Int(embedding.offset),
                                       scales: embedding.buffer,
                                       scalesOffset: Int(embedding.scaleOffset),
                                       biases: embedding.buffer,
                                       biasesOffset: Int(embedding.biasOffset),
                                       out: s.hidden,
                                       tokens: s.tokenIDs,
                                       d: D,
                                       seqLen: T)
            }
        }

        for L in 0..<cfg.numLayers {
            let w = layers[L]
            try stage(.preRouter) {
                let cb = try commandBuffer()
                s.rms.encodeBF16W(commandBuffer: cb,
                                  x: s.hidden,
                                  weight: w.inputNorm.buffer, weightOffset: Int(w.inputNorm.offset),
                                  out: s.normed, t: UInt32(T), d: D, eps: rmsEps)
                if let linear = w.linear {
                    encodePrefillRecurrent(cb, layer: L, tokens: T, w: linear, scratch: s)
                } else if let full = w.full {
                    encodePrefillAttention(cb, layer: L, tokens: T, start: start,
                                           w: full, scratch: s)
                }
                qwen.encodeResidualAdd(commandBuffer: cb, hidden: s.hidden, y: s.branchOut,
                                       count: T * hiddenSize)
                s.rms.encodeBF16W(commandBuffer: cb,
                                  x: s.hidden,
                                  weight: w.postAttnNorm.buffer,
                                  weightOffset: Int(w.postAttnNorm.offset),
                                  out: s.normed, t: UInt32(T), d: D, eps: rmsEps)
                encodePrefillRouter(cb, tokens: T, w: w.moe, scratch: s)
                // The next layer's router over *this* layer's `normed`, in the
                // same command buffer — the guess costs no extra join
                // (`27-PHASE6-THROUGHPUT.md` §9-4).
                if prefetching, L + 1 < cfg.numLayers {
                    encodePrefillRouter(cb, tokens: T, w: layers[L + 1].moe, scratch: s,
                                        indices: s.previewIndices, weights: s.previewWeights)
                }
                // This join also drains the previous layer's committed tail, so
                // its wall clock is "everything the GPU still owed" — the same
                // shape decode's own pre-router join has.
                try wait(cb, "prefill layer \(L) pre-router")
            }

            // The routes cannot be grouped before the router has run, and the
            // experts of a tile cannot be fetched before the grouping names
            // them. This readback is the reason a layer is more than one
            // command buffer, as it is in decode.
            let routes = try stage(.routeReadback) {
                try readPrefillRoutes(tokens: T, scratch: s)
            }
            let nextGuess = prefetching && L + 1 < cfg.numLayers
                ? readChunkPreviewExperts(tokens: T, scratch: s)
                : nil
            // A guess for *this* layer has to have landed before this layer
            // plans, or the plan can hand out the slot the read is filling.
            if let started = chunkPrefetch, started.layer == L {
                try stage(.expertIO) {
                    let began = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    _ = try? started.handle.wait()
                    notePrefetchWait(nanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - began)
                    chunkPrefetch = nil
                }
            }
            if let predicted = chunkPredicted {
                let actual = routes.tiles.indices.flatMap {
                    (try? PrefillStreamedTileBinding.expertIDs(forTile: $0, routes: routes)) ?? []
                }
                notePreview(predicted: predicted, actual: actual, missed: [])
            }
            chunkPredicted = nextGuess
            // The next layer's read goes out before this layer waits on its
            // own, so its bytes — and, on the mapped arm, its residency
            // commit — move while this layer's MoE is encoded and run.
            // A refusal costs nothing: that layer will fetch in its turn.
            if let guess = nextGuess, chunkPrefetch == nil, !guess.isEmpty {
                if let speculative = try model.planSpeculativeRoutedExperts(layer: L + 1,
                                                                            experts: guess) {
                    notePrefetchIssued(reads: speculative.misses.count)
                    chunkPrefetch = (L + 1, try model.startRoutedExpertFetch(plan: speculative))
                } else {
                    notePrefetchDeclined()
                }
            }

            try stage(.sharedExpert) {
                let shared = try commandBuffer()
                encodePrefillSharedExpert(shared, tokens: T, w: w.moe, scratch: s)
                // Committed, not joined: the branch that does not depend on which
                // experts the router picked runs on the GPU while the first tile's
                // experts are read (Phase 6, `27-PHASE6-THROUGHPUT.md`).
                if QwenForwardRunner.pipelineEnabled {
                    commitDeferred(shared, "prefill layer \(L) shared expert")
                } else {
                    try wait(shared, "prefill layer \(L) shared expert")
                }
            }

            let residuals: (MTLCommandBuffer) -> Void = { cb in
                self.qwen.encodeResidualAdd(commandBuffer: cb, hidden: s.hidden,
                                            y: s.sharedOut, count: T * self.hiddenSize)
                self.qwen.encodeResidualAdd(commandBuffer: cb, hidden: s.hidden,
                                            y: s.routedOut, count: T * self.hiddenSize)
            }
            try encodePrefillRoutedExperts(layer: L, tokens: T, routes: routes,
                                           scratch: s,
                                           tail: compactChunkCommandBuffers ? residuals : nil)
            if compactChunkCommandBuffers { continue }

            try stage(.reduceTail) {
                let tail = try commandBuffer()
                residuals(tail)
                // The next layer's pre-router buffer is the join.
                if QwenForwardRunner.pipelineEnabled {
                    commitDeferred(tail, "prefill layer \(L) residual")
                } else {
                    try wait(tail, "prefill layer \(L) residual")
                }
            }
        }
        // The head reads the last row of `hidden`, and a chunk that emits
        // nothing still has to have landed before the next one is encoded.
        try stage(.drain) { try drainDeferred() }

        kv.advance(by: T)
    }

    // MARK: - Layer halves

    private func encodePrefillRecurrent(_ cb: MTLCommandBuffer,
                                        layer L: Int,
                                        tokens T: Int,
                                        w: LinearLayerWeights,
                                        scratch s: QwenPrefillContext) {
        encodePrefillProjection(cb, w.inProjQKV, x: s.normed, y: s.wide, tokens: T, scratch: s)
        encodePrefillProjection(cb, w.inProjZ, x: s.normed, y: s.zBuf, tokens: T, scratch: s)
        encodePrefillProjection(cb, w.inProjA, x: s.normed, y: s.aBuf, tokens: T, scratch: s)
        encodePrefillProjection(cb, w.inProjB, x: s.normed, y: s.bBuf, tokens: T, scratch: s)

        let convOffset = state.convOffset(layer: L)
        let convBytes = (QwenKernels.convKernel - 1) * qkvWidth * MemoryLayout<Float16>.size
        if speculativeLastRow {
            encodeSplitRecurrent(cb, layer: L, tokens: T, w: w, scratch: s)
            return
        }
        qwen.encodeDeltaQKVPrepare(commandBuffer: cb,
                                   qkv: s.wide,
                                   convWeight: w.conv1d.buffer,
                                   convWeightOffset: Int(w.conv1d.offset),
                                   stateIn: state.convBuffer, stateInOffset: convOffset,
                                   stateOut: s.convStateOut, stateOutOffset: 0,
                                   q: s.qDelta, k: s.kDelta, v: s.vDelta,
                                   seqLen: T,
                                   numKHeads: numKeyHeads,
                                   numVHeads: numValueHeads,
                                   headDim: keyHeadDim)
        // Encoders inside one command buffer run in order, so the window the
        // next chunk reads is written before anything reads it again.
        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: s.convStateOut, sourceOffset: 0,
                      to: state.convBuffer, destinationOffset: convOffset,
                      size: convBytes)
            blit.endEncoding()
        }
        qwen.encodeDeltaGates(commandBuffer: cb,
                              a: s.aBuf, b: s.bBuf,
                              aLog: w.aLog.buffer, aLogOffset: Int(w.aLog.offset),
                              dtBias: w.dtBias.buffer, dtBiasOffset: Int(w.dtBias.offset),
                              g: s.gBuf, beta: s.betaBuf,
                              seqLen: T, numVHeads: numValueHeads)
        // The recurrence is sequential in the sequence, so one threadgroup owns
        // a slice of the state for the whole chunk: in and out may alias, as
        // they do in decode.
        let stateOffset = state.stateOffset(layer: L)
        delta.encode(commandBuffer: cb,
                     q: s.qDelta, k: s.kDelta, v: s.vDelta,
                     g: s.gBuf, beta: s.betaBuf,
                     stateIn: state.stateBuffer, stateInOffset: stateOffset,
                     y: s.yDelta,
                     stateOut: state.stateBuffer, stateOutOffset: stateOffset,
                     seqLen: T,
                     numKHeads: numKeyHeads,
                     numVHeads: numValueHeads,
                     keyHeadDim: keyHeadDim,
                     valueHeadDim: valueHeadDim)
        qwen.encodeDeltaNormGate(commandBuffer: cb,
                                 o: s.yDelta, z: s.zBuf,
                                 weight: w.norm.buffer, weightOffset: Int(w.norm.offset),
                                 out: s.oDelta,
                                 seqLen: T,
                                 numVHeads: numValueHeads,
                                 headDim: valueHeadDim,
                                 eps: rmsEps)
        encodePrefillProjection(cb, w.outProj, x: s.oDelta, y: s.branchOut, tokens: T, scratch: s)
    }

    /// The same recurrence, with the **last row's** state written somewhere else.
    ///
    /// This is what makes a speculative row discardable. `qwen_delta_rule` folds
    /// every row of a chunk into one state, and there is no subtraction that
    /// takes the last one back out (`03-DESIGN.md` §3-4) — so the chunk is split
    /// into `T-1` confirmed rows, which update the state in place, and the last
    /// row, which reads that state and writes the shadow. Accepting is
    /// `adoptShadow()`; rejecting is doing nothing.
    ///
    /// The cost is one extra pair of dispatches per recurrent layer and no copy
    /// at all: `33-MTP-ACCEPTANCE.md` §3-6 measured the split at +0.14 ms a
    /// token against +1.8 ms for blitting the 61.4 MiB aside.
    private func encodeSplitRecurrent(_ cb: MTLCommandBuffer,
                                      layer L: Int,
                                      tokens T: Int,
                                      w: LinearLayerWeights,
                                      scratch s: QwenPrefillContext) {
        precondition(T >= 1, "the split path needs at least one row")
        guard let shadowState = state.shadowStateBuffer,
              let shadowConv = state.shadowConvBuffer else {
            preconditionFailure("the recurrent shadow was not allocated")
        }
        let convOffset = state.convOffset(layer: L)
        let stateOffset = state.stateOffset(layer: L)
        let keyWidth = numKeyHeads * keyHeadDim
        let valueWidth = numValueHeads * valueHeadDim
        let half = MemoryLayout<Float16>.size
        let confirmed = T - 1

        func prepare(rows: Int, from: Int,
                     convIn: MTLBuffer, convInOffset: Int,
                     convOut: MTLBuffer, convOutOffset: Int) {
            qwen.encodeDeltaQKVPrepare(commandBuffer: cb,
                                       qkv: s.wide, qkvOffset: from * qkvWidth * half,
                                       convWeight: w.conv1d.buffer,
                                       convWeightOffset: Int(w.conv1d.offset),
                                       stateIn: convIn, stateInOffset: convInOffset,
                                       stateOut: convOut, stateOutOffset: convOutOffset,
                                       q: s.qDelta, qOffset: from * keyWidth * half,
                                       k: s.kDelta, kOffset: from * keyWidth * half,
                                       v: s.vDelta, vOffset: from * valueWidth * half,
                                       seqLen: rows,
                                       numKHeads: numKeyHeads,
                                       numVHeads: numValueHeads,
                                       headDim: keyHeadDim)
        }
        func recur(rows: Int, from: Int, out: MTLBuffer, outOffset: Int) {
            delta.encode(commandBuffer: cb,
                         q: s.qDelta, qOffset: from * keyWidth * half,
                         k: s.kDelta, kOffset: from * keyWidth * half,
                         v: s.vDelta, vOffset: from * valueWidth * half,
                         g: s.gBuf, gOffset: from * numValueHeads * MemoryLayout<Float>.size,
                         beta: s.betaBuf,
                         betaOffset: from * numValueHeads * MemoryLayout<Float>.size,
                         stateIn: state.stateBuffer, stateInOffset: stateOffset,
                         y: s.yDelta, yOffset: from * valueWidth * half,
                         stateOut: out, stateOutOffset: outOffset,
                         seqLen: rows,
                         numKHeads: numKeyHeads,
                         numVHeads: numValueHeads,
                         keyHeadDim: keyHeadDim,
                         valueHeadDim: valueHeadDim)
        }

        // The confirmed rows first, in place. `encodeDeltaQKVPrepare` refuses an
        // aliased state for more than one token block, so wider confirmed spans
        // stage through `convStateOut` exactly as the ordinary path does.
        if confirmed > 0 {
            let convBytes = (QwenKernels.convKernel - 1) * qkvWidth * half
            let aliasable = confirmed <= QwenKernels.tokensPerGroup
            prepare(rows: confirmed, from: 0,
                    convIn: state.convBuffer, convInOffset: convOffset,
                    convOut: aliasable ? state.convBuffer : s.convStateOut,
                    convOutOffset: aliasable ? convOffset : 0)
            if !aliasable, let blit = cb.makeBlitCommandEncoder() {
                blit.copy(from: s.convStateOut, sourceOffset: 0,
                          to: state.convBuffer, destinationOffset: convOffset,
                          size: convBytes)
                blit.endEncoding()
            }
        }
        // The speculative row reads the state the confirmed rows left and writes
        // the shadow. Encoders inside one command buffer run in order.
        prepare(rows: 1, from: confirmed,
                convIn: state.convBuffer, convInOffset: convOffset,
                convOut: shadowConv, convOutOffset: convOffset)

        qwen.encodeDeltaGates(commandBuffer: cb,
                              a: s.aBuf, b: s.bBuf,
                              aLog: w.aLog.buffer, aLogOffset: Int(w.aLog.offset),
                              dtBias: w.dtBias.buffer, dtBiasOffset: Int(w.dtBias.offset),
                              g: s.gBuf, beta: s.betaBuf,
                              seqLen: T, numVHeads: numValueHeads)

        if confirmed > 0 {
            recur(rows: confirmed, from: 0,
                  out: state.stateBuffer, outOffset: stateOffset)
        }
        recur(rows: 1, from: confirmed, out: shadowState, outOffset: stateOffset)

        qwen.encodeDeltaNormGate(commandBuffer: cb,
                                 o: s.yDelta, z: s.zBuf,
                                 weight: w.norm.buffer, weightOffset: Int(w.norm.offset),
                                 out: s.oDelta,
                                 seqLen: T,
                                 numVHeads: numValueHeads,
                                 headDim: valueHeadDim,
                                 eps: rmsEps)
        encodePrefillProjection(cb, w.outProj, x: s.oDelta, y: s.branchOut,
                                tokens: T, scratch: s)
    }

    private func encodePrefillAttention(_ cb: MTLCommandBuffer,
                                        layer L: Int,
                                        tokens T: Int,
                                        start: Int,
                                        w: FullLayerWeights,
                                        scratch s: QwenPrefillContext) {
        let headDim = cfg.fullHeadDim
        let kvHeads = cfg.numFullKVHeads
        let fullWidth = cfg.numHeads * headDim
        // With the ring off, a chunk's positions are contiguous slots and the
        // per-token stride is exactly `numFullKVHeads * fullHeadDim` halves —
        // the same row the projection writes. The K/V of the chunk therefore
        // land in the cache directly, with no staging copy.
        let kRange = kv.kRange(layer: L, start: start, count: T)
        let vRange = kv.vRange(layer: L, start: start, count: T)

        encodePrefillProjection(cb, w.qProj, x: s.normed, y: s.wide, tokens: T, scratch: s)
        encodePrefillProjection(cb, w.kProj, x: s.normed,
                                y: kRange.buffer, yOffset: kRange.offset, tokens: T, scratch: s)
        encodePrefillProjection(cb, w.vProj, x: s.normed,
                                y: vRange.buffer, yOffset: vRange.offset, tokens: T, scratch: s)

        qwen.encodeQKVEpilogue(commandBuffer: cb,
                               q: s.wide,
                               k: kRange.buffer, kOffset: kRange.offset,
                               qWeight: w.qNorm.buffer, qWeightOffset: Int(w.qNorm.offset),
                               kWeight: w.kNorm.buffer, kWeightOffset: Int(w.kNorm.offset),
                               seqLen: T,
                               numQHeads: cfg.numHeads,
                               numKVHeads: kvHeads,
                               headDim: headDim,
                               rotaryDim: rotaryDim,
                               position: start,
                               theta: Float(cfg.fullRopeTheta),
                               eps: rmsEps)
        qwen.encodeQueryCompact(commandBuffer: cb,
                                wide: s.wide, out: s.qCompact,
                                seqLen: T,
                                numQHeads: cfg.numHeads,
                                headDim: headDim)

        // `scale` is 1.0 because `q_norm.weight` carries `head_dim ** -0.5`
        // (`Scripts/qwen35/bake_snapshot.py`), the same as in decode.
        let params = PrefillAttentionParams(
            startPosition: UInt32(start),
            queryCount: UInt32(T),
            headDim: UInt32(headDim),
            numQHeads: UInt32(cfg.numHeads),
            numKVHeads: UInt32(kvHeads),
            kvValidCount: UInt32(start + T),
            slidingWindow: 0,
            kvTokenStrideElements: UInt32(kvHeads * headDim),
            qTokenStrideElements: UInt32(fault == .uncompactedQuery ? 2 * fullWidth : fullWidth),
            oTokenStrideElements: UInt32(fullWidth),
            scale: 1.0)
        // Two kernels for the same sum, chosen by how many rows there are.
        // The rows path needs the compacted query (its stride is
        // `numQHeads * headDim`), so the uncompacted-query fault keeps the
        // query-blocked kernel — that negative control exists to be caught by
        // the prefill check, and it has to stay on the path it was written for.
        if chunkRowsAttention && T <= Attention.maxRows && fault != .uncompactedQuery {
            // **One dispatch per row, not one for the block.**
            // `Attention.rowsGeometry` cuts the KV range into chunks over
            // `[0, startPosition + rows)`, so a block's cut — and with it the
            // order the log-sum-exp merge adds the chunks in — depends on where
            // the block starts. A token that comes out as row 1 of a pass and
            // the same token as row 0 of the next would then be summed
            // differently, and the speculative loop would stop being neutral:
            // the force-reject control lost its token-for-token match with the
            // production arm at 95/96 (`38-MTP-VERIFY-PATH.md` §4). Asking for
            // one row at a time makes the geometry a function of that row's own
            // position and nothing else, which restores the match exactly. It
            // costs a second walk of the KV range — 2.5 ms a pass at 2,700
            // positions against the shared walk, out of a 30 ms saving.
            let rowStride = cfg.numHeads * headDim * MemoryLayout<Float16>.size
            for row in 0..<T {
                attention.encodeRows(commandBuffer: cb,
                                     q: s.qCompact, qOffset: row * rowStride,
                                     k: kRange.buffer, kOffset: 0,
                                     v: vRange.buffer, vOffset: 0,
                                     out: s.attnOut, outOffset: row * rowStride,
                                     headDim: UInt32(headDim),
                                     numQHeads: UInt32(cfg.numHeads),
                                     numKVHeads: UInt32(kvHeads),
                                     rows: 1,
                                     startPosition: start + row,
                                     window: 0,
                                     scale: 1.0)
            }
        } else {
            s.attention.encodeCausal(commandBuffer: cb,
                                     q: fault == .uncompactedQuery ? s.wide : s.qCompact,
                                     k: kRange.buffer, kOffset: 0,
                                     v: vRange.buffer, vOffset: 0,
                                     out: s.attnOut,
                                     params: params)
        }
        qwen.encodeAttnOutputGate(commandBuffer: cb,
                                  o: s.attnOut, qGate: s.wide,
                                  seqLen: T,
                                  numQHeads: cfg.numHeads,
                                  headDim: headDim)
        encodePrefillProjection(cb, w.oProj, x: s.attnOut, y: s.branchOut, tokens: T, scratch: s)
    }

    // MARK: - MoE

    private func encodePrefillRouter(_ cb: MTLCommandBuffer,
                                     tokens T: Int,
                                     w: MoEWeights,
                                     scratch s: QwenPrefillContext,
                                     indices: MTLBuffer? = nil,
                                     weights: MTLBuffer? = nil) {
        let outIndices = indices ?? s.routerIndices
        let outWeights = weights ?? s.routerWeights
        // Both scale buffers are the multiplicative identity: this family has
        // neither Gemma's `router.scale` nor its per-expert weight scale, and
        // the remaining `logits -> top-k -> renormalize` is identical to the
        // reference (`docs/qwen35moe/20-PHASE3-DECODE.md` §4).
        let router = w.router
        if s.router.routerWeightBits == 16 {
            s.router.encodeGemma4BF16Block(commandBuffer: cb,
                                           weights: router.buffer,
                                           weightsOffset: Int(router.offset),
                                           hidden: s.normed,
                                           effectiveScale: unitFeatureScale,
                                           perExpertScale: unitExpertScale,
                                           outIndices: outIndices,
                                           outWeights: outWeights,
                                           queryCount: UInt32(T),
                                           numExperts: UInt32(cfg.numExperts),
                                           d: UInt32(hiddenSize),
                                           topK: UInt32(cfg.topKExperts),
                                           hiddenStrideElements: UInt32(hiddenSize))
        } else {
            s.router.encodeGemma4Block(commandBuffer: cb,
                                       weights: router.buffer,
                                       weightsOffset: Int(router.offset),
                                       scales: router.buffer,
                                       scalesOffset: Int(router.scaleOffset),
                                       biases: router.buffer,
                                       biasesOffset: Int(router.biasOffset),
                                       hidden: s.normed,
                                       effectiveScale: unitFeatureScale,
                                       perExpertScale: unitExpertScale,
                                       outIndices: outIndices,
                                       outWeights: outWeights,
                                       queryCount: UInt32(T),
                                       numExperts: UInt32(cfg.numExperts),
                                       d: UInt32(hiddenSize),
                                       topK: UInt32(cfg.topKExperts),
                                       hiddenStrideElements: UInt32(hiddenSize))
        }
    }

    /// The next layer's predicted experts, strongest first, as one list for the
    /// whole chunk.
    ///
    /// A verify pass fetches the **union** of its rows' picks, so the guess is
    /// the union of the rows' predictions — interleaved by rank rather than
    /// concatenated per row, because the read-ahead is truncated at
    /// `verifyPrefetchTopN` per row and rank 1 of row 1 is a better bet than
    /// rank 4 of row 0 (`27-PHASE6-THROUGHPUT.md` §9-4: rank 1 is used 98.6% of
    /// the time, rank 8 44%).
    private func readChunkPreviewExperts(tokens T: Int,
                                         scratch s: QwenPrefillContext) -> [Int] {
        let topK = cfg.topKExperts
        let depth = min(QwenForwardRunner.verifyPrefetchTopN, topK)
        let ptr = s.previewIndices.contents().bindMemory(to: UInt32.self,
                                                         capacity: T * topK)
        var seen = Set<Int>()
        var out: [Int] = []
        out.reserveCapacity(T * depth)
        for rank in 0..<depth {
            for row in 0..<T {
                let expert = min(Int(ptr[row * topK + rank]), cfg.numExperts - 1)
                if seen.insert(expert).inserted { out.append(expert) }
            }
        }
        return out
    }

    private func readPrefillRoutes(tokens T: Int,
                                   scratch s: QwenPrefillContext) throws
        -> PrefillMoEGroupedRoutes {
        let topK = cfg.topKExperts
        let indexPtr = s.routerIndices.contents().bindMemory(to: UInt32.self,
                                                             capacity: T * topK)
        let weightPtr = s.routerWeights.contents().bindMemory(to: Float16.self,
                                                              capacity: T * topK)
        var indices = [UInt32](repeating: 0, count: T * topK)
        var weights = [Float16](repeating: 0, count: T * topK)
        for i in 0..<(T * topK) {
            indices[i] = min(indexPtr[i], UInt32(cfg.numExperts - 1))
            weights[i] = weightPtr[i]
        }
        let pairs = PrefillRouter.makeTokenExpertPairs(indices: indices,
                                                       weights: weights,
                                                       queryCount: T,
                                                       topK: topK)
        return try PrefillMoEGrouping.groupTokenExpertPairs(pairs,
                                                            queryCount: T,
                                                            topK: topK,
                                                            numExperts: cfg.numExperts)
    }

    /// `sigmoid(w_gate . x) * down(silu(gate(x)) * up(x))` for every row.
    private func encodePrefillSharedExpert(_ cb: MTLCommandBuffer,
                                           tokens T: Int,
                                           w: MoEWeights,
                                           scratch s: QwenPrefillContext) {
        encodePrefillProjection(cb, w.gateProj, x: s.normed, y: s.sharedGateAct,
                                tokens: T, scratch: s)
        encodePrefillProjection(cb, w.upProj, x: s.normed, y: s.sharedUpAct,
                                tokens: T, scratch: s)
        qwen.encodeSiluMul(commandBuffer: cb,
                           gate: s.sharedGateAct, up: s.sharedUpAct, out: s.sharedAct,
                           count: T * cfg.intermediateSize)
        encodePrefillProjection(cb, w.downProj, x: s.sharedAct, y: s.sharedOut,
                                tokens: T, scratch: s)

        if fault == .sharedGateSkipped { return }
        // The production gate is a quantized `[1, D]` row, so the dot product
        // is a QMM of one column and the kernel only applies the sigmoid
        // (`docs/qwen35moe/18-MIXED-BITS.md` §3). The BF16 fused kernel decode
        // keeps for unquantized checkpoints has no T-row sibling; a BF16 gate
        // would come through here as a projection instead.
        //
        // `sharedGateAsBF16` is decode's name for "the gate is read at the
        // wrong width". There is no BF16 T-row sibling to point it at, so on
        // this path the same mistake is the nibble one: an 8-bit row read as
        // 4-bit. The case still tests what it is there to test.
        let gateBits = (fault == .sharedGateAsBF16) ? 4 : (w.sharedGateBits ?? 8)
        encodePrefillGEMM(cb, bits: gateBits, view: w.sharedGate,
                          x: s.normed, y: s.sharedGateLogit,
                          t: T, n: 1, k: hiddenSize, scratch: s)
        qwen.encodeMoESharedGateLogitBlock(commandBuffer: cb,
                                           y: s.sharedOut,
                                           logit: s.sharedGateLogit,
                                           hiddenSize: hiddenSize,
                                           seqLen: T)
    }

    /// The routed half: one tile of at most 16 experts at a time, each tile
    /// fetched, run and drained before the next one is planned.
    ///
    /// Draining is not tidiness. The expert cache hands a tile its slots, and
    /// the next tile's fetch may be given the same ones; overlapping them is
    /// what `RealForwardRunner` uses `avoidingSlots` and a slot lifetime for.
    /// A serial loop needs neither, and pays for that in wall time.
    ///
    /// Which kernels a tile runs on is `prefillRoutedPath`: the per-pair GEMVs
    /// this phase was wired with, or the 64-row tiled GEMM
    /// (`docs/qwen35moe/05-RISKS.md` §1-2 is the question those two answer
    /// differently). Everything either path reads and writes is the same —
    /// same sorted pairs, same argument buffer, same `routePartials`, written
    /// once per pair — so the reduction below and the fetch above do not know
    /// which one ran.
    private func encodePrefillRoutedExperts(layer L: Int,
                                            tokens T: Int,
                                            routes: PrefillMoEGroupedRoutes,
                                            scratch s: QwenPrefillContext,
                                            tail: ((MTLCommandBuffer) -> Void)? = nil) throws {
        let device = ctx.device
        let metadata = try stage(.routedExperts) {
            try s.routedMoE.makeStreamedMetadataBuffers(device: device, routes: routes)
        }
        let offsets = model.routedExpertOffsets(layer: L)

        // Tile *i*'s bytes are read while tile *i-1* is on the GPU. The reads
        // are the larger half of a chunk's wall clock and they have nothing to
        // do with the GPU, so the only thing that made them serial was asking
        // for them one at a time (Phase 6).
        //
        // **Only on the mapped arm.** There a tile's blobs are views into the
        // layer file's mapping, so a read that is planned while an earlier
        // tile's command buffer is still running cannot land on top of it. On
        // the private-slot arm (`TF_EXPERT_MMAP=0`) it could: the planner would
        // have to be told to avoid every slot every unjoined buffer is reading,
        // and at 16 slots and 16 experts to a tile there is no such plan. That
        // arm keeps the serial loop — it is the measurement arm, not the
        // default (`docs/qwen35moe/27-PHASE6-THROUGHPUT.md`).
        // Asked of the model, not of the layer: a layer file is opened on its
        // first read, so asking whether *this* layer has a residency set says
        // "no" for every layer of the first chunk — which is the chunk that
        // matters.
        let pipelined = model.usesMappedExperts && QwenForwardRunner.pipelineEnabled
        struct InflightTile {
            let index: Int
            let expertIDs: [Int]
            let plan: RoutedExpertFetchPlan
            let handle: RoutedExpertFetchHandle
        }
        let active = routes.tiles.enumerated()
            .filter { $0.element.pairCount > 0 }
            .map(\.offset)
        func startTile(_ index: Int) throws -> InflightTile {
            try stage(.expertPlan) {
                let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: index,
                                                                         routes: routes)
                guard let plan = try model.planRoutedExperts(layer: L, experts: expertIDs) else {
                    throw QwenRunnerError.commandBufferFailed(
                        "prefill layer \(L) tile \(index): could not plan \(expertIDs.count) experts")
                }
                return InflightTile(index: index,
                                    expertIDs: expertIDs,
                                    plan: plan,
                                    handle: try model.startRoutedExpertFetch(plan: plan))
            }
        }
        /// The read-ahead. `avoiding` is the tile still on the GPU, and the
        /// planner may refuse: with 16 slots one tile already fills the layer's
        /// cache, and there is no plan that keeps off it. Refusing is the right
        /// answer there — the read-ahead would have to evict the bytes the GPU
        /// is reading, which on the mapped arm means dropping their residency
        /// mid-kernel. The caller falls back to reading it in its turn.
        func startTileAhead(_ index: Int, avoiding: Set<Int>) throws -> InflightTile? {
            try stage(.expertPlan) {
                let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: index,
                                                                         routes: routes)
                guard let plan = try model.planRoutedExpertsIfPossible(layer: L,
                                                                       experts: expertIDs,
                                                                       avoidingSlots: avoiding)
                else { return nil }
                return InflightTile(index: index,
                                    expertIDs: expertIDs,
                                    plan: plan,
                                    handle: try model.startRoutedExpertFetch(plan: plan))
            }
        }
        var inflight: InflightTile? = nil
        var lastTile: MTLCommandBuffer? = nil
        for (position, index) in active.enumerated() {
            let current: InflightTile
            if let started = inflight, started.index == index {
                current = started
            } else {
                // Nothing was read ahead for this tile — either this is the
                // first one, or the arm does not pipeline. Whatever is still
                // running has to be joined first on the serial arm, because the
                // read about to be planned may take its slots.
                if !pipelined { try drainDeferred() }
                current = try startTile(index)
            }
            let tile = routes.tiles[index]
            let expertIDs = current.expertIDs
            let views = try stage(.expertIO) { try current.handle.wait() }
            inflight = pipelined && position + 1 < active.count
                ? try startTileAhead(active[position + 1],
                                     avoiding: Set(current.plan.assignedSlots))
                : nil
            let binding = try stage(.routedExperts) {
                try PrefillStreamedTileBinding(expertIDs: expertIDs, views: views)
            }
            let argumentBuffer = try stage(.routedExperts) {
                try s.routedMoE.streamedArgumentBuffer(device: device,
                                                       index: index,
                                                       binding: binding)
            }
            let params = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart,
                pairCount: tile.pairCount,
                d: UInt32(hiddenSize),
                routedIntermediate: UInt32(cfg.moeIntermediateSize),
                topK: UInt32(cfg.topKExperts),
                hiddenStrideElements: UInt32(hiddenSize),
                binding: binding,
                offsets: offsets)

            let cb = try stage(.routedExperts) { try commandBuffer() }
            if let set = model.routedExpertResidencySet(layer: L) {
                cb.useResidencySet(set)
            }
            try stage(.routedExperts) {
            switch prefillRoutedPath {
            case .perPair:
                _ = s.routedMoE.encodeStreamedBatched(commandBuffer: cb,
                                                      hidden: s.normed,
                                                      sortedPairs: metadata.sortedPairs,
                                                      routePartials: s.routePartials,
                                                      gateUpActScratch: s.gateUpAct,
                                                      downScratch: s.downScratch,
                                                      argumentBuffer: argumentBuffer,
                                                      binding: binding,
                                                      params: params,
                                                      pairMicrobatchRows: QwenPrefillContext.pairMicrobatchRows)
            case .tiled:
                // No silent fallback. A run asked for this path so that its
                // wall clock could be compared with the other one's; quietly
                // giving it the other path would make the comparison a lie.
                guard s.routedMoE.usesExpertGEMMPath(d: hiddenSize,
                                                     f: cfg.moeIntermediateSize) else {
                    throw QwenRunnerError.commandBufferFailed(
                        "prefill layer \(L) tile \(index): the tiled routed path is not "
                        + "available for D=\(hiddenSize) F=\(cfg.moeIntermediateSize) "
                        + "(TF_PREFILL_MOE=scalar turns its pipelines off)")
                }
                let start = Int(tile.groupStart)
                let groups = Array(routes.groups[start..<(start + Int(tile.groupCount))])
                let batches = s.routedMoE.encodeStreamedTiled(
                    commandBuffer: cb,
                    hidden: s.normed,
                    sortedPairs: metadata.sortedPairs,
                    routePartials: s.routePartials,
                    gateUpActScratch: s.gateUpAct,
                    argumentBuffer: argumentBuffer,
                    binding: binding,
                    groups: groups,
                    params: params,
                    maxRowsPerBatch: s.gemmBatchRows)
                guard batches > 0 else {
                    throw QwenRunnerError.commandBufferFailed(
                        "prefill layer \(L) tile \(index): the tiled routed path encoded "
                        + "nothing for \(groups.count) groups / \(tile.pairCount) pairs")
                }
            }
            if compactChunkCommandBuffers && position == active.count - 1 {
                // Held back: the reduce and the residual adds go on it.
                lastTile = cb
            } else if pipelined {
                commitDeferred(cb, "prefill layer \(L) tile \(index)")
            } else {
                try wait(cb, "prefill layer \(L) tile \(index)")
            }
            }
        }

        // Every pair of every tile has written its row, so the token-major sum
        // can run once for the layer.
        //
        // In compact mode it rides on the last tile's command buffer together
        // with the caller's residual adds. Command buffers on one queue run in
        // commit order, so an earlier tile is finished before this one starts —
        // the same guarantee `commitDeferred` already relies on.
        if let lastTile, compactChunkCommandBuffers {
            try stage(.reduceTail) {
            s.moeReduce.encodeReduceTokenMajor(commandBuffer: lastTile,
                                               routePartials: s.routePartials,
                                               routeWeights: s.routerWeights,
                                               h2: s.routedOut,
                                               queryCount: UInt32(T),
                                               topK: UInt32(cfg.topKExperts),
                                               d: UInt32(hiddenSize))
            tail?(lastTile)
            if pipelined {
                commitDeferred(lastTile, "prefill layer \(L) routed tail")
            } else {
                try wait(lastTile, "prefill layer \(L) routed tail")
            }
            }
            return
        }
        try stage(.reduceTail) {
        let reduce = try commandBuffer()
        s.moeReduce.encodeReduceTokenMajor(commandBuffer: reduce,
                                           routePartials: s.routePartials,
                                           routeWeights: s.routerWeights,
                                           h2: s.routedOut,
                                           queryCount: UInt32(T),
                                           topK: UInt32(cfg.topKExperts),
                                           d: UInt32(hiddenSize))
        tail?(reduce)
        if pipelined {
            commitDeferred(reduce, "prefill layer \(L) routed reduce")
        } else {
            try wait(reduce, "prefill layer \(L) routed reduce")
        }
        }
    }

    // MARK: - Plumbing

    private func encodePrefillProjection(_ cb: MTLCommandBuffer,
                                         _ p: Projection,
                                         x: MTLBuffer, xOffset: Int = 0,
                                         y: MTLBuffer, yOffset: Int = 0,
                                         tokens T: Int,
                                         scratch s: QwenPrefillContext) {
        encodePrefillGEMM(cb, bits: p.bits, view: p.view,
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          t: T, n: Int(p.rows), k: Int(p.cols), scratch: s)
    }

    /// `Y[T, N] = X[T, K] * W[N, K]^T` at whichever width the index says this
    /// tensor is (`docs/qwen35moe/18-MIXED-BITS.md`). The two kernels are
    /// separate for the same reason the two GEMVs are in decode: the nibble
    /// unpack is the row geometry, not an argument.
    private func encodePrefillGEMM(_ cb: MTLCommandBuffer,
                                   bits: Int,
                                   view: TensorView,
                                   x: MTLBuffer, xOffset: Int = 0,
                                   y: MTLBuffer, yOffset: Int = 0,
                                   t: Int, n: Int, k: Int,
                                   scratch s: QwenPrefillContext) {
        // A verify block is one or two rows, and both QMM kernels are wrong for
        // that width: the tiled one refuses below eight rows and the scalar one
        // gives each output row a single thread walking K on its own, so nothing
        // coalesces. Decode's SIMD-per-row GEMV, widened to T activations over
        // one pass of the weights, is what these shapes want
        // (`docs/qwen35moe/36-MTP-DECODE.md` §4-3). Bit-identical to the
        // one-row GEMV at T=1, which is not the same numbers as the QMM path —
        // so it is switched on only for the block, never for the prompt.
        if compactChunkCommandBuffers,
           t <= DequantInt8GEMV.maxRows, k % 64 == 0 {
            if bits == 8 {
                int8.encodeRows(commandBuffer: cb,
                                weights: view.buffer, weightsOffset: Int(view.offset),
                                scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                                biases: view.buffer, biasesOffset: Int(view.biasOffset),
                                x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                                rows: t, m: UInt32(n), n: UInt32(k),
                                xStride: UInt32(k), yStride: UInt32(n))
            } else {
                int4.encodeRows(commandBuffer: cb,
                                weights: view.buffer, weightsOffset: Int(view.offset),
                                scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                                biases: view.buffer, biasesOffset: Int(view.biasOffset),
                                x: x, xOffset: xOffset, xStrideElements: k,
                                y: y, yOffset: yOffset, yStrideElements: n,
                                t: t, m: UInt32(n), n: UInt32(k))
            }
            return
        }
        if bits == 8 {
            s.int8.encode(commandBuffer: cb,
                          weights: view.buffer, weightsOffset: Int(view.offset),
                          scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                          biases: view.buffer, biasesOffset: Int(view.biasOffset),
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          t: t, n: n, k: k)
        } else {
            s.int4.encode(commandBuffer: cb,
                          weights: view.buffer, weightsOffset: Int(view.offset),
                          scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                          biases: view.buffer, biasesOffset: Int(view.biasOffset),
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          t: t, n: n, k: k)
        }
    }
}
