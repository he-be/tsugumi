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
        self.routePartials = try buffer(width * topK * d, label: "routePartials")
        self.gateUpAct = try buffer(3 * actRows * f, label: "gateUpAct")
        self.downScratch = try buffer(rows * d, label: "downScratch")
        self.convStateOut = try buffer((QwenKernels.convKernel - 1) * qkvWidth,
                                       label: "convStateOut")
        self.greedyToken = try buffer(1, MemoryLayout<UInt32>.size, label: "greedyToken")
    }
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
        var next = try constrained(try prefill(tokens: promptTokens, chunkWidth: chunkWidth),
                                   gate: gate, position: 0)
        var produced: [Int32] = []
        for index in 0..<maxNewTokens {
            produced.append(next)
            try gate?.accept(next)
            try onToken?(index, next)
            if stopTokens.contains(next) { break }
            if produced.count == maxNewTokens { break }
            next = try constrained(try step(token: next, emitToken: true),
                                   gate: gate, position: index + 1)
        }
        return produced
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
        let ctxScratch = try prefillContext(width: min(chunkWidth, tokens.count))
        var token: Int32 = 0
        var offset = 0
        while offset < tokens.count {
            let count = min(ctxScratch.width, tokens.count - offset)
            let last = offset + count == tokens.count
            token = try prefillChunk(Array(tokens[offset..<(offset + count)]),
                                     scratch: ctxScratch,
                                     emitToken: last)
            offset += count
        }
        return token
    }

    /// Allocates the chunk scratch on first use and reuses it after.
    private func prefillContext(width: Int) throws -> QwenPrefillContext {
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
        let T = tokens.count
        let start = kv.position
        let D = UInt32(hiddenSize)
        // The routed-expert counters are per phase; a chunk is prefill, and
        // the step it is stamped with is where the chunk starts
        // (`ExpertTelemetry`, same convention as `RealForwardRunner`).
        model.telemetry.beginPhase(.prefill, step: start)
        if fault == .forgetRecurrentState { state.reset() }

        let ids = s.tokenIDs.contents().bindMemory(to: UInt32.self, capacity: T)
        for (index, token) in tokens.enumerated() { ids[index] = UInt32(bitPattern: token) }

        let embedding = model.embedding
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

        for L in 0..<cfg.numLayers {
            let w = layers[L]
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
            try wait(cb, "prefill layer \(L) pre-router")

            // The routes cannot be grouped before the router has run, and the
            // experts of a tile cannot be fetched before the grouping names
            // them. This readback is the reason a layer is more than one
            // command buffer, as it is in decode.
            let routes = try readPrefillRoutes(tokens: T, scratch: s)

            let shared = try commandBuffer()
            encodePrefillSharedExpert(shared, tokens: T, w: w.moe, scratch: s)
            try wait(shared, "prefill layer \(L) shared expert")

            try encodePrefillRoutedExperts(layer: L, tokens: T, routes: routes, scratch: s)

            let tail = try commandBuffer()
            qwen.encodeResidualAdd(commandBuffer: tail, hidden: s.hidden, y: s.sharedOut,
                                   count: T * hiddenSize)
            qwen.encodeResidualAdd(commandBuffer: tail, hidden: s.hidden, y: s.routedOut,
                                   count: T * hiddenSize)
            try wait(tail, "prefill layer \(L) residual")
        }

        kv.advance(by: T)
        guard emitToken else { return 0 }

        // Only the last row is scored: the token that follows the chunk is the
        // argmax over the last position's logits, and the 248,077-row table is
        // 508 MB to read (`docs/qwen35moe/19-LM-HEAD-INT8.md`).
        let lm = try model.qwenLMHead
        let finalNorm = model.finalNorm
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
                d: D,
                vocab: UInt32(self.scoredVocab),
                rmsEps: self.rmsEps)
        }
        return Int32(bitPattern: s.greedyToken.contents().load(as: UInt32.self))
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
        s.attention.encodeCausal(commandBuffer: cb,
                                 q: fault == .uncompactedQuery ? s.wide : s.qCompact,
                                 k: kRange.buffer, kOffset: 0,
                                 v: vRange.buffer, vOffset: 0,
                                 out: s.attnOut,
                                 params: params)
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
                                     scratch s: QwenPrefillContext) {
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
                                           outIndices: s.routerIndices,
                                           outWeights: s.routerWeights,
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
                                       outIndices: s.routerIndices,
                                       outWeights: s.routerWeights,
                                       queryCount: UInt32(T),
                                       numExperts: UInt32(cfg.numExperts),
                                       d: UInt32(hiddenSize),
                                       topK: UInt32(cfg.topKExperts),
                                       hiddenStrideElements: UInt32(hiddenSize))
        }
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
                                            scratch s: QwenPrefillContext) throws {
        let device = ctx.device
        let metadata = try s.routedMoE.makeStreamedMetadataBuffers(device: device, routes: routes)
        let offsets = model.routedExpertOffsets(layer: L)

        for (index, tile) in routes.tiles.enumerated() where tile.pairCount > 0 {
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: index,
                                                                     routes: routes)
            guard let plan = try model.planRoutedExperts(layer: L, experts: expertIDs) else {
                throw QwenRunnerError.commandBufferFailed(
                    "prefill layer \(L) tile \(index): could not plan \(expertIDs.count) experts")
            }
            let views = try model.startRoutedExpertFetch(plan: plan).wait()
            let binding = try PrefillStreamedTileBinding(expertIDs: expertIDs, views: views)
            let argumentBuffer = try s.routedMoE.streamedArgumentBuffer(device: device,
                                                                        index: index,
                                                                        binding: binding)
            let params = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart,
                pairCount: tile.pairCount,
                d: UInt32(hiddenSize),
                routedIntermediate: UInt32(cfg.moeIntermediateSize),
                topK: UInt32(cfg.topKExperts),
                hiddenStrideElements: UInt32(hiddenSize),
                binding: binding,
                offsets: offsets)

            let cb = try commandBuffer()
            if let set = model.routedExpertResidencySet(layer: L) {
                cb.useResidencySet(set)
            }
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
            try wait(cb, "prefill layer \(L) tile \(index)")
        }

        // Every pair of every tile has written its row, so the token-major sum
        // can run once for the layer.
        let reduce = try commandBuffer()
        s.moeReduce.encodeReduceTokenMajor(commandBuffer: reduce,
                                           routePartials: s.routePartials,
                                           routeWeights: s.routerWeights,
                                           h2: s.routedOut,
                                           queryCount: UInt32(T),
                                           topK: UInt32(cfg.topKExperts),
                                           d: UInt32(hiddenSize))
        try wait(reduce, "prefill layer \(L) routed reduce")
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
