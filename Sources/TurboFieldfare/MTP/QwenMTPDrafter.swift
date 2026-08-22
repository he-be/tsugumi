import Foundation
import Metal

/// The grafted MTP head, running on the GPU (`docs/qwen35moe/36-MTP-DECODE.md`).
///
/// One transformer block plus a projection, in the order
/// `Scripts/qwen35/mtp_acceptance.py` `MTPHead.step` walks it — the same
/// reference the acceptance rate in [33](../../docs/qwen35moe/33-MTP-ACCEPTANCE.md)
/// was measured against:
///
///     h  = fc([ pre_fc_norm_embedding(embed(x_{t+1})) ; pre_fc_norm_hidden(h_t) ])
///     h += o_proj(gate(attn(rms(h, input_layernorm))))
///     h += moe(rms(h, post_attention_layernorm))
///     draft = argmax lm_head(rms(h, mtp.norm))
///
/// Everything below the block is the body's: the embedding table, the 508 MB
/// `lm_head`, `Attention`, `MoE`, `RMSNorm`, the dequantizing GEMVs. What is
/// its own is 42 tensors (`QwenMTPSidecar`), a one-layer K/V cache, and the
/// scratch.
///
/// **The K/V cache holds only what generation put there.** The measurement fed
/// the head a causal cache over the prompt as well, because it ran the whole
/// sequence in teacher forcing; reproducing that here would mean a T-row prefill
/// of `fc` and the k/v projections. `Scripts/qwen35/mtp_history_ablation.py`
/// priced dropping it first: P1 falls 78.40% -> 76.05% averaged over the four
/// tasks, and 87.43% -> 85.34% on the coding one. Two points of acceptance is
/// not worth a second prefill path, so the cache starts empty at the first
/// generated token and grows one entry per committed position.
///
/// **Positions are absolute.** Row `t` carries RoPE position `t+1`, as the
/// reference does — the cache being shorter than the sequence changes which
/// keys are visible, not what any of them is.
final class QwenMTPDrafter {

    /// One row the head has to walk: the body hidden it reads, the token it
    /// eats, and the position that hidden sits at.
    struct Row {
        /// Byte offset of this row inside the caller's post-norm hidden buffer.
        let hiddenOffset: Int
        /// `x_{t+1}` — the token that follows the hidden.
        let token: Int32
        /// `t`, the body position the hidden belongs to. RoPE gets `t+1`.
        let position: Int
    }

    let sidecar: QwenMTPSidecar

    private let ctx: MetalContext
    private let model: Model
    private let rms: RMSNorm
    private let int8: DequantInt8GEMV
    private let qwen: QwenKernels
    private let attention: Attention
    private let moe: MoE
    private let head: QwenLMHeadChainInt8
    private let embed: QwenEmbedLookupInt8
    private let unitFeatureScale: MTLBuffer
    private let unitExpertScale: MTLBuffer
    private let scoredVocab: Int
    private let rmsEps: Float

    private let hiddenSize: Int
    private let numHeads: Int
    private let numKVHeads: Int
    private let headDim: Int
    private let rotaryDim: Int
    private let topK: Int
    private let maxContext: Int

    // Scratch. Allocated once; a draft never allocates.
    private let concat: MTLBuffer        // [2D] FP16 — [embedding ; hidden]
    private let hBuf: MTLBuffer          // [D]  FP16 — the block's residual stream
    private let normed: MTLBuffer        // [D]  FP16
    private let branchOut: MTLBuffer     // [D]  FP16
    private let wide: MTLBuffer          // [NQ * 2 * HD] FP16 — query and gate
    private let qCompact: MTLBuffer      // [NQ * HD] FP16
    private let attnOut: MTLBuffer       // [NQ * HD] FP16
    private let sharedGateAct: MTLBuffer // [F] FP16
    private let sharedUpAct: MTLBuffer   // [F] FP16
    private let sharedAct: MTLBuffer     // [F] FP16
    private let sharedOut: MTLBuffer     // [D] FP16
    private let sharedGateLogit: MTLBuffer
    private let outIndices: MTLBuffer    // [topK] UInt32
    private let outWeights: MTLBuffer    // [topK] FP16
    private let moeActs: MTLBuffer       // [topK * F] FP16
    private let draftToken: MTLBuffer    // [1] UInt32
    private let routedArgBuffer: MTLBuffer
    private let kCache: MTLBuffer
    private let vCache: MTLBuffer

    /// Entries the head has written. Also the `seqLen` its attention reads.
    private(set) var cachedRows = 0

    /// GPU time this head has spent, on the same clock the runner keeps.
    private(set) var gpuSeconds: Double = 0
    private(set) var commandBuffers = 0
    /// Where a draft's wall clock goes (`docs/qwen35moe/39-...`).
    ///
    /// `38-MTP-VERIFY-PATH.md` §7-2 left this stage as the one with no meter:
    /// 6.0 ms a pass, GPU time uncounted because the head runs its own command
    /// buffers rather than the runner's. The cut is the head's own structure —
    /// the catch-up rows that only owe the cache a `(k, v)`, the first buffer of
    /// the last row (embed through the router, joined because the host has to
    /// name the eight experts), and the tail (the MoE and the 508 MB head).
    private(set) var profile = DraftProfile()

    struct DraftProfile: Sendable {
        var passes = 0
        var rows = 0
        var catchUpRows = 0
        /// Wall and GPU per region, seconds.
        var catchUpWall: Double = 0, catchUpGPU: Double = 0
        var preRouterWall: Double = 0, preRouterGPU: Double = 0
        var tailWall: Double = 0, tailGPU: Double = 0
        var totalGPU: Double { catchUpGPU + preRouterGPU + tailGPU }
    }

    init(sidecar: QwenMTPSidecar,
         context: MetalContext,
         model: Model,
         config cfg: ArchConfig,
         rms: RMSNorm,
         int8: DequantInt8GEMV,
         qwen: QwenKernels,
         attention: Attention,
         moe: MoE,
         head: QwenLMHeadChainInt8,
         embed: QwenEmbedLookupInt8,
         unitFeatureScale: MTLBuffer,
         unitExpertScale: MTLBuffer,
         scoredVocab: Int,
         maxContext: Int,
         rmsEps: Float) throws {
        self.sidecar = sidecar
        self.ctx = context
        self.model = model
        self.rms = rms
        self.int8 = int8
        self.qwen = qwen
        self.attention = attention
        self.moe = moe
        self.head = head
        self.embed = embed
        self.unitFeatureScale = unitFeatureScale
        self.unitExpertScale = unitExpertScale
        self.scoredVocab = scoredVocab
        self.maxContext = maxContext
        self.rmsEps = rmsEps

        let arch = sidecar.arch
        // The head is a layer of the same model; a mismatch here means the
        // sidecar was built from a different checkpoint than the pack.
        guard arch.hiddenSize == cfg.hiddenSize,
              arch.numExperts == cfg.numExperts,
              arch.topK == cfg.topKExperts,
              arch.moeIntermediateSize == cfg.moeIntermediateSize,
              arch.headDim == cfg.fullHeadDim,
              arch.numHeads == cfg.numHeads,
              arch.numKVHeads == cfg.numFullKVHeads else {
            throw QwenMTPSidecar.SidecarError.geometry(
                "the sidecar's geometry differs from the pack's arch section")
        }
        self.hiddenSize = arch.hiddenSize
        self.numHeads = arch.numHeads
        self.numKVHeads = arch.numKVHeads
        self.headDim = arch.headDim
        self.rotaryDim = Int(Double(arch.headDim) * arch.partialRotaryFactor)
        self.topK = arch.topK

        let device = context.device
        func buffer(_ count: Int, _ stride: Int = MemoryLayout<Float16>.size,
                    _ label: String) throws -> MTLBuffer {
            guard let created = device.makeBuffer(length: max(count * stride, 4),
                                                  options: .storageModeShared) else {
                throw MetalError.noDevice
            }
            created.label = "mtp.\(label)"
            return created
        }
        let D = arch.hiddenSize
        let fullWidth = arch.numHeads * arch.headDim
        let kvWidth = arch.numKVHeads * arch.headDim
        self.concat = try buffer(2 * D, MemoryLayout<Float16>.size, "concat")
        self.hBuf = try buffer(D, MemoryLayout<Float16>.size, "h")
        self.normed = try buffer(D, MemoryLayout<Float16>.size, "normed")
        self.branchOut = try buffer(D, MemoryLayout<Float16>.size, "branchOut")
        self.wide = try buffer(2 * fullWidth, MemoryLayout<Float16>.size, "wide")
        self.qCompact = try buffer(fullWidth, MemoryLayout<Float16>.size, "qCompact")
        self.attnOut = try buffer(fullWidth, MemoryLayout<Float16>.size, "attnOut")
        self.sharedGateAct = try buffer(arch.sharedIntermediateSize,
                                        MemoryLayout<Float16>.size, "sharedGate")
        self.sharedUpAct = try buffer(arch.sharedIntermediateSize,
                                      MemoryLayout<Float16>.size, "sharedUp")
        self.sharedAct = try buffer(arch.sharedIntermediateSize,
                                    MemoryLayout<Float16>.size, "sharedAct")
        self.sharedOut = try buffer(D, MemoryLayout<Float16>.size, "sharedOut")
        self.sharedGateLogit = try buffer(1, MemoryLayout<Float16>.size, "sharedGateLogit")
        self.outIndices = try buffer(arch.topK, MemoryLayout<UInt32>.size, "outIndices")
        self.outWeights = try buffer(arch.topK, MemoryLayout<Float16>.size, "outWeights")
        self.moeActs = try buffer(arch.topK * arch.moeIntermediateSize,
                                  MemoryLayout<Float16>.size, "moeActs")
        self.draftToken = try buffer(1, MemoryLayout<UInt32>.size, "draftToken")
        self.kCache = try buffer(maxContext * kvWidth, MemoryLayout<Float16>.size, "k")
        self.vCache = try buffer(maxContext * kvWidth, MemoryLayout<Float16>.size, "v")
        guard let argBuffer = device.makeBuffer(length: moe.routedArgumentBufferLength,
                                                options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        argBuffer.label = "mtp.routedArgs"
        self.routedArgBuffer = argBuffer
    }

    /// Back to an empty head. Called wherever the body's caches are reset.
    func reset() { cachedRows = 0 }

    func resetProfile() {
        gpuSeconds = 0
        commandBuffers = 0
        profile = DraftProfile()
    }

    private static func seconds(since start: UInt64) -> Double {
        Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) &- start) / 1e9
    }

    /// Run the head over `rows` and return the token it predicts after the last
    /// one.
    ///
    /// Only the last row pays for attention, the MoE and the 508 MB head. The
    /// earlier rows exist because a verify pass that accepted two tokens
    /// skipped a position, and all that position owes the cache is its `(k, v)`
    /// — the depth-1 chain never reads its output (`33` §2-4 is about depth 2
    /// and beyond, which width 2 does not use).
    func draft(baseNormed: MTLBuffer, rows: [Row]) throws -> Int32 {
        precondition(!rows.isEmpty, "the head needs at least one row")
        guard cachedRows + rows.count <= maxContext else {
            throw QwenMTPSidecar.SidecarError.geometry(
                "MTP cache is full at \(cachedRows) rows")
        }

        let fc = try sidecar.view("fc")
        let preEmbed = try sidecar.view("pre_fc_norm_embedding")
        let preHidden = try sidecar.view("pre_fc_norm_hidden")
        let inputNorm = try sidecar.view("input_layernorm")
        let postNorm = try sidecar.view("post_attention_layernorm")
        let finalNorm = try sidecar.view("final_norm")
        let D = UInt32(hiddenSize)
        let embedding = model.embedding
        profile.passes += 1
        profile.rows += rows.count
        for (index, row) in rows.enumerated() {
            let last = index == rows.count - 1
            let slot = cachedRows
            let rowStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)

            let cb = try commandBuffer()
            // `fc` eats [embedding ; hidden]; the two halves are normalized
            // separately and written straight into the halves of `concat`.
            embed.encode(commandBuffer: cb,
                         table: embedding.buffer, tableOffset: Int(embedding.offset),
                         scales: embedding.buffer, scalesOffset: Int(embedding.scaleOffset),
                         biases: embedding.buffer, biasesOffset: Int(embedding.biasOffset),
                         out: concat,
                         tokenId: UInt32(bitPattern: row.token),
                         d: D)
            rms.encodeBF16W(commandBuffer: cb,
                            x: concat,
                            weight: preEmbed.buffer, weightOffset: Int(preEmbed.offset),
                            out: concat, d: D, eps: rmsEps)
            rms.encodeBF16W(commandBuffer: cb,
                            x: baseNormed, xOffset: row.hiddenOffset,
                            weight: preHidden.buffer, weightOffset: Int(preHidden.offset),
                            out: concat,
                            outOffset: hiddenSize * MemoryLayout<Float16>.size,
                            d: D, eps: rmsEps)
            qwen.encodeBF16GEMV(commandBuffer: cb,
                                weights: fc.buffer, weightsOffset: Int(fc.offset),
                                x: concat, y: hBuf,
                                m: hiddenSize, k: 2 * hiddenSize)
            rms.encodeBF16W(commandBuffer: cb,
                            x: hBuf,
                            weight: inputNorm.buffer, weightOffset: Int(inputNorm.offset),
                            out: normed, d: D, eps: rmsEps)
            try encodeQKV(cb, slot: slot, position: row.position + 1)
            if !last {
                profile.catchUpGPU += try wait(cb, "mtp catch-up row")
                profile.catchUpWall += Self.seconds(since: rowStart)
                profile.catchUpRows += 1
                cachedRows += 1
                continue
            }

            // The last row is the one that produces a draft.
            let qNormBytes = headDim * MemoryLayout<Float16>.size
            if let blit = cb.makeBlitCommandEncoder() {
                for h in 0..<numHeads {
                    blit.copy(from: wide, sourceOffset: h * 2 * qNormBytes,
                              to: qCompact, destinationOffset: h * qNormBytes,
                              size: qNormBytes)
                }
                blit.endEncoding()
            }
            attention.encodeFull(commandBuffer: cb,
                                 q: qCompact,
                                 k: kCache, kOffset: 0,
                                 v: vCache, vOffset: 0,
                                 out: attnOut,
                                 headDim: UInt32(headDim),
                                 numQHeads: UInt32(numHeads),
                                 numKVHeads: UInt32(numKVHeads),
                                 seqLen: UInt32(slot + 1),
                                 // `q_norm` carries `head_dim ** -0.5`, baked
                                 // at repack for the head too (`30` §6-1).
                                 scale: 1.0)
            qwen.encodeAttnOutputGate(commandBuffer: cb,
                                      o: attnOut, qGate: wide,
                                      seqLen: 1, numQHeads: numHeads, headDim: headDim)
            try gemv("o_proj", cb, x: attnOut, y: branchOut)
            qwen.encodeResidualAdd(commandBuffer: cb, hidden: hBuf, y: branchOut,
                                   count: hiddenSize)
            rms.encodeBF16W(commandBuffer: cb,
                            x: hBuf,
                            weight: postNorm.buffer, weightOffset: Int(postNorm.offset),
                            out: normed, d: D, eps: rmsEps)
            try encodeRouter(cb)
            try encodeSharedExpert(cb)
            // The router's pick has to reach the host before the eight blobs
            // can be named, exactly as it does in the body's decode loop — but
            // here nothing is fetched, so the readback is the only reason this
            // is two command buffers rather than one.
            profile.preRouterGPU += try wait(cb, "mtp attention and router")
            profile.preRouterWall += Self.seconds(since: rowStart)

            let tailStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let experts = readRoutedExperts()
            let tail = try commandBuffer()
            tail.useResidencySet(sidecar.residencySet)
            encodeRoutedExperts(tail, experts: experts)
            qwen.encodeResidualAdd(commandBuffer: tail, hidden: hBuf, y: sharedOut,
                                   count: hiddenSize)
            rms.encodeBF16W(commandBuffer: tail,
                            x: hBuf,
                            weight: finalNorm.buffer, weightOffset: Int(finalNorm.offset),
                            out: normed, d: D, eps: rmsEps)
            let lm = try model.qwenLMHead
            head.encodeGreedyDecodeRows(commandBuffer: tail,
                                        hiddenNormed: normed,
                                        weights: lm.buffer, weightsOffset: Int(lm.offset),
                                        scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                        biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                        outTokens: draftToken,
                                        rows: 1,
                                        d: D,
                                        vocab: UInt32(scoredVocab))
            profile.tailGPU += try wait(tail, "mtp moe and head")
            profile.tailWall += Self.seconds(since: tailStart)
            cachedRows += 1
        }
        return Int32(bitPattern: draftToken.contents().load(as: UInt32.self))
    }

    /// **The head's cache never needs rewinding at width 2.** It only ever
    /// writes rows the body has already committed: the drafted token is fed to
    /// the *body*, not to the head, so a rejection leaves no entry behind. A
    /// wider width would break that and need a cursor here.

    // MARK: - Halves

    private func encodeQKV(_ cb: MTLCommandBuffer, slot: Int, position: Int) throws {
        let kvWidth = numKVHeads * headDim
        let rowBytes = kvWidth * MemoryLayout<Float16>.size
        try gemv("q_proj", cb, x: normed, y: wide)
        try gemv("k_proj", cb, x: normed, y: kCache, yOffset: slot * rowBytes)
        try gemv("v_proj", cb, x: normed, y: vCache, yOffset: slot * rowBytes)
        let qNorm = try sidecar.view("q_norm")
        let kNorm = try sidecar.view("k_norm")
        qwen.encodeQKVEpilogue(commandBuffer: cb,
                               q: wide,
                               k: kCache, kOffset: slot * rowBytes,
                               qWeight: qNorm.buffer, qWeightOffset: Int(qNorm.offset),
                               kWeight: kNorm.buffer, kWeightOffset: Int(kNorm.offset),
                               seqLen: 1,
                               numQHeads: numHeads,
                               numKVHeads: numKVHeads,
                               headDim: headDim,
                               rotaryDim: rotaryDim,
                               position: position,
                               theta: sidecar.arch.ropeTheta,
                               eps: rmsEps)
    }

    private func encodeRouter(_ cb: MTLCommandBuffer) throws {
        let router = try sidecar.view("router")
        moe.encodeRouterGemma4BF16(commandBuffer: cb,
                                   weights: router.buffer,
                                   weightsOffset: Int(router.offset),
                                   hidden: normed,
                                   effectiveScale: unitFeatureScale,
                                   perExpertScale: unitExpertScale,
                                   perExpertScaleOffset: 0,
                                   outIndices: outIndices, outWeights: outWeights,
                                   numExperts: UInt32(sidecar.arch.numExperts),
                                   d: UInt32(hiddenSize),
                                   topK: UInt32(topK))
    }

    private func encodeSharedExpert(_ cb: MTLCommandBuffer) throws {
        let F = sidecar.arch.sharedIntermediateSize
        try gemv("shared_gate_proj", cb, x: normed, y: sharedGateAct)
        try gemv("shared_up_proj", cb, x: normed, y: sharedUpAct)
        qwen.encodeSiluMul(commandBuffer: cb,
                           gate: sharedGateAct, up: sharedUpAct, out: sharedAct,
                           count: F)
        try gemv("shared_down_proj", cb, x: sharedAct, y: sharedOut)
        try gemv("shared_expert_gate", cb, x: normed, y: sharedGateLogit)
        qwen.encodeMoESharedGateLogit(commandBuffer: cb,
                                      y: sharedOut,
                                      logit: sharedGateLogit,
                                      hiddenSize: hiddenSize)
    }

    private func readRoutedExperts() -> [Int] {
        let ptr = outIndices.contents().bindMemory(to: UInt32.self, capacity: topK)
        return (0..<topK).map { min(Int(ptr[$0]), sidecar.arch.numExperts - 1) }
    }

    private func encodeRoutedExperts(_ cb: MTLCommandBuffer, experts: [Int]) {
        let blobs = experts.map { sidecar.expertBuffers[$0] }
        let offsets = sidecar.expertOffsets
        let k = UInt32(topK)
        moe.encodeRoutedArgumentBuffer(into: routedArgBuffer, routedBlobs: blobs, topK: k)
        moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                routedArgBuffer: routedArgBuffer,
                                                routedBlobs: blobs,
                                                routedOffsets: offsets,
                                                x: normed,
                                                acts: moeActs,
                                                d: UInt32(hiddenSize),
                                                f: UInt32(sidecar.arch.moeIntermediateSize),
                                                topK: k)
        moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: cb,
                                               routedArgBuffer: routedArgBuffer,
                                               routedBlobs: blobs,
                                               routedOffsets: offsets,
                                               acts: moeActs,
                                               routingWeights: outWeights,
                                               residual: sharedOut,
                                               y: sharedOut,
                                               d: UInt32(hiddenSize),
                                               f: UInt32(sidecar.arch.moeIntermediateSize),
                                               topK: k)
    }

    private func gemv(_ name: String, _ cb: MTLCommandBuffer,
                      x: MTLBuffer, xOffset: Int = 0,
                      y: MTLBuffer, yOffset: Int = 0) throws {
        let entry = try sidecar.tensor(name)
        int8.encode(commandBuffer: cb,
                    weights: sidecar.coreBuffer, weightsOffset: entry.offset,
                    scales: sidecar.coreBuffer, scalesOffset: entry.scaleOffset,
                    biases: sidecar.coreBuffer, biasesOffset: entry.biasOffset,
                    x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                    m: entry.rows, n: entry.cols)
    }

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw QwenForwardRunner.QwenRunnerError.commandBufferFailed(
                "makeCommandBuffer returned nil")
        }
        return cb
    }

    /// Commit, join, and hand back this buffer's GPU seconds so the caller can
    /// put them in the right bucket of `profile`.
    @discardableResult
    private func wait(_ cb: MTLCommandBuffer, _ label: String) throws -> Double {
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error {
            throw QwenForwardRunner.QwenRunnerError.commandBufferFailed("\(label): \(error)")
        }
        let gpu = cb.gpuEndTime - cb.gpuStartTime
        gpuSeconds += gpu
        commandBuffers += 1
        return gpu
    }
}
