import Foundation
import Metal

/// Decode for Qwen3.5-MoE (`docs/qwen35moe/04-PHASES.md` Phase 3).
///
/// `RealForwardRunner` is not extended to cover this family and will not be.
/// Its decode loop is a hand-folded three-command-buffer pipeline that overlaps
/// a layer's routed-expert I/O with the next layer's attention, tuned against
/// measurements this repository treats as assets; threading a third kind of
/// layer through it would move those numbers (`03-DESIGN.md` §3-2). What is
/// shared instead is everything below the loop: `Model`, the streamers, the
/// router and routed-expert kernels, the attention PSOs, `Attention`,
/// `RMSNorm`, and the dequantizing GEMVs.
///
/// **This runner is deliberately serial.** One command buffer per layer,
/// waited on before the next is encoded. Phase 3's exit condition is a
/// token-for-token match against the CPU float32 reference, and a serial loop
/// is the shape in which a divergence can be localized to a layer and a stage.
/// Overlapping I/O with compute is Phase 6's subject, not this one's.
///
/// One layer, in the order the reference walks it
/// (`Scripts/qwen35/reference_forward.py` `forward`):
///
///     normed = rmsnorm(hidden, input_layernorm)
///     hidden += recurrent(normed)   or   attention(normed)
///     normed = rmsnorm(hidden, post_attention_layernorm)
///     hidden += moe(normed)
///
/// There is no sandwich norm, no layer scalar, and no logit softcap; the three
/// Gemma fusions that fold those away therefore have no counterpart here.
public final class QwenForwardRunner {

    public enum QwenRunnerError: Error, CustomStringConvertible {
        case wrongFamily(String)
        case missingLinearAttentionGeometry
        case unsupportedWidth(tensor: String, bits: Int?)
        case geometryMismatch(String)
        case commandBufferFailed(String)

        public var description: String {
            switch self {
            case .wrongFamily(let family):
                return "QwenForwardRunner needs a qwen3_5_moe install; manifest says \(family)"
            case .missingLinearAttentionGeometry:
                return "manifest has no arch.linearAttention section"
            case .unsupportedWidth(let tensor, let bits):
                return "\(tensor) is \(bits.map(String.init) ?? "not an affine") "
                    + "weight; this path needs 4-bit or 8-bit"
            case .geometryMismatch(let detail):
                return "Qwen geometry mismatch: \(detail)"
            case .commandBufferFailed(let detail):
                return "command buffer failed: \(detail)"
            }
        }
    }

    /// A deliberate mis-wiring, for the check that proves the comparison
    /// against the reference has detection power.
    ///
    /// A whole-model check that has only ever been seen to pass is not evidence
    /// (`PLAN_VISION.md` §6-3, and the six negative controls in
    /// `docs/qwen35moe/17-PHASE2-KERNELS.md` §2). Every case here is a road this
    /// runner could plausibly have taken: four of them were live possibilities
    /// while it was being written, and each produces a model that runs to
    /// completion, allocates the same bytes and emits fluent-looking tokens.
    ///
    /// The mirror of `VisionPrefillFault`, but taken as an argument rather than
    /// an environment variable: `routedActivationGelu` decides a pipeline at
    /// construction, so the check builds one runner per case.
    public enum DecodeFault: String, Sendable, CaseIterable {
        case none = "none"
        /// Read the 8-bit `shared_expert_gate.weight` with the kernel that
        /// expects BF16. The bytes are there and the dot product comes out
        /// finite — it is simply a different number.
        case sharedGateAsBF16 = "shared-gate-bf16"
        /// Drop the shared expert's sigmoid gate entirely.
        case sharedGateSkipped = "shared-gate-skipped"
        /// Gemma 4's activation inside the routed experts instead of SiLU.
        case routedActivationGelu = "routed-gelu"
        /// Hand attention the doubled-width `q_proj` rows without compacting
        /// them, so every head reads its neighbour's output gate as queries.
        case uncompactedQuery = "uncompacted-query"
        /// Zero the recurrent state before each token: 30 of 40 layers then see
        /// only the token in front of them.
        case forgetRecurrentState = "forget-state"
    }

    /// One resident projection plus the width the index says it is. The width
    /// is per tensor, not per slot: on `oQ4e-g64` five roles mix 4-bit and
    /// 8-bit inside one slot (`docs/qwen35moe/18-MIXED-BITS.md` §3).
    struct Projection {
        let view: TensorView
        let bits: Int
        let rows: UInt32
        let cols: UInt32
    }

    struct LinearLayerWeights {
        let inProjQKV: Projection
        let inProjZ: Projection
        let inProjA: Projection
        let inProjB: Projection
        let outProj: Projection
        let conv1d: TensorView
        let aLog: TensorView
        let dtBias: TensorView
        let norm: TensorView
    }

    struct FullLayerWeights {
        let qProj: Projection
        let kProj: Projection
        let vProj: Projection
        let oProj: Projection
        let qNorm: TensorView
        let kNorm: TensorView
    }

    struct MoEWeights {
        let router: TensorView
        let sharedGate: TensorView
        /// `nil` when `shared_expert_gate.weight` is BF16, which the fused
        /// kernel reads directly.
        let sharedGateBits: Int?
        let gateProj: Projection
        let upProj: Projection
        let downProj: Projection
    }

    struct LayerWeights {
        let inputNorm: TensorView
        let postAttnNorm: TensorView
        let linear: LinearLayerWeights?
        let full: FullLayerWeights?
        let moe: MoEWeights
    }

    // MARK: - Collaborators

    let model: Model
    let ctx: MetalContext
    let cfg: ArchConfig
    let kv: KVCacheManager
    let state: RecurrentStateManager

    let embed: QwenEmbedLookupInt8
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let int8: DequantInt8GEMV
    let qwen: QwenKernels
    let delta: GatedDeltaNet
    let attention: Attention
    let moe: MoE
    let head: QwenLMHeadChainInt8

    let layers: [LayerWeights]
    /// `.none` in every run that is not the detection-power check.
    public let fault: DecodeFault

    // MARK: - Geometry

    let hiddenSize: Int
    let numKeyHeads: Int
    let numValueHeads: Int
    let keyHeadDim: Int
    let valueHeadDim: Int
    let qkvWidth: Int
    let valueWidth: Int
    let rotaryDim: Int
    /// Vocabulary rows the head scores. Upstream pads `vocab_size` past the
    /// tokenizer's last piece and those rows were never trained
    /// (`docs/qwen35moe/10-MLX4BIT-AUDIT.md` §3).
    public let scoredVocab: Int
    public let maxContext: Int
    let rmsEps: Float = 1e-6

    // MARK: - Scratch (allocated once; the decode loop never allocates)

    private let hidden: MTLBuffer        // [D] FP16 — the residual stream
    private let normed: MTLBuffer        // [D] FP16
    private let branchOut: MTLBuffer     // [D] FP16 — attention or recurrence output
    private let wide: MTLBuffer          // [max(qkvWidth, 2*numHeads*fullHeadDim)] FP16
    private let zBuf: MTLBuffer          // [Hv*Dv] FP16
    private let aBuf: MTLBuffer          // [Hv] FP16
    private let bBuf: MTLBuffer          // [Hv] FP16
    private let gBuf: MTLBuffer          // [Hv] FP32
    private let betaBuf: MTLBuffer       // [Hv] FP32
    private let qDelta: MTLBuffer        // [Hk*Dk] FP16
    private let kDelta: MTLBuffer        // [Hk*Dk] FP16
    private let vDelta: MTLBuffer        // [Hv*Dv] FP16
    private let yDelta: MTLBuffer        // [Hv*Dv] FP16
    private let oDelta: MTLBuffer        // [Hv*Dv] FP16
    private let qCompact: MTLBuffer      // [numHeads*fullHeadDim] FP16
    private let attnOut: MTLBuffer       // [numHeads*fullHeadDim] FP16
    private let sharedGateAct: MTLBuffer // [F] FP16
    private let sharedUpAct: MTLBuffer   // [F] FP16
    private let sharedAct: MTLBuffer     // [F] FP16
    private let sharedOut: MTLBuffer     // [D] FP16 — the whole MoE output
    private let sharedGateLogit: MTLBuffer // [1] FP16
    private let outIndices: MTLBuffer    // [topK] UInt32
    private let outWeights: MTLBuffer    // [topK] FP16
    private let moeActs: MTLBuffer       // [topK * F] FP16
    private let greedyToken: MTLBuffer   // [1] UInt32
    /// The T-row scratch, allocated on the first `prefill` call and reused
    /// (`QwenPrefill.swift`). A decode-only run never pays for it.
    /// All ones. Gemma's router folds a per-feature `router.scale` and a
    /// per-expert weight scale into the same two kernels; this family has
    /// neither, so both are the multiplicative identity rather than a second
    /// pair of router kernels (`docs/qwen35moe/03-DESIGN.md` §4-1).
    let unitFeatureScale: MTLBuffer  // [D] BF16 = 1
    let unitExpertScale: MTLBuffer   // [numExperts] BF16 = 1

    var prefillScratch: QwenPrefillContext?

    // MARK: - Init

    public init(model: Model,
                context: MetalContext,
                maxContext: Int,
                fault: DecodeFault = .none) throws {
        self.fault = fault
        guard model.modelFamily == "qwen3_5_moe" else {
            throw QwenRunnerError.wrongFamily(model.modelFamily)
        }
        guard let linear = model.qwenLinearAttention else {
            throw QwenRunnerError.missingLinearAttentionGeometry
        }
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.maxContext = maxContext

        self.hiddenSize = cfg.hiddenSize
        self.numKeyHeads = linear.numKeyHeads
        self.numValueHeads = linear.numValueHeads
        self.keyHeadDim = linear.keyHeadDim
        self.valueHeadDim = linear.valueHeadDim
        self.qkvWidth = 2 * linear.numKeyHeads * linear.keyHeadDim
            + linear.numValueHeads * linear.valueHeadDim
        self.valueWidth = linear.numValueHeads * linear.valueHeadDim
        self.rotaryDim = Int((Double(cfg.fullHeadDim) * cfg.partialRotaryFactor).rounded())
        self.scoredVocab = min(cfg.vocabSize, QwenLMHeadChainInt8.ornithScoredVocab)

        guard rotaryDim > 0, rotaryDim % 2 == 0, rotaryDim <= cfg.fullHeadDim else {
            throw QwenRunnerError.geometryMismatch(
                "rotary_dim \(rotaryDim) from partialRotaryFactor \(cfg.partialRotaryFactor)")
        }
        guard linear.convKernelDim == QwenKernels.convKernel else {
            throw QwenRunnerError.geometryMismatch(
                "conv kernel \(linear.convKernelDim) != the \(QwenKernels.convKernel) "
                + "the shader carries in registers")
        }
        // The library was compiled with the context's group size and scheme
        // baked in. A model quantized at another one would be read with the
        // wrong per-group scale — plausible output, not a crash.
        guard context.affineGroupSize == model.affineGroupSize,
              context.affineScheme == model.affineScheme else {
            throw QwenRunnerError.geometryMismatch(
                "the shader library is built for group \(context.affineGroupSize) "
                + "\(context.affineScheme), the model is group \(model.affineGroupSize) "
                + "\(model.affineScheme)")
        }
        // Both ends of the model go through width-specific kernels: the nibble
        // unpack is a row geometry, not an argument
        // (`docs/qwen35moe/19-LM-HEAD-INT8.md`). A 4-bit table read eight bits
        // at a time produces finite nonsense, so the width is a gate.
        for name in ["language_model.model.embed_tokens.weight",
                     "language_model.lm_head.weight"] {
            let bits = model.residentWeightBits(name)
            guard bits == 8 else {
                throw QwenRunnerError.unsupportedWidth(tensor: name, bits: bits)
            }
        }

        // The recurrent layers hold no K/V, so the cache is told about them and
        // allocates nothing (`03-DESIGN.md` §3-3). The ring is off: this family
        // has no sliding-window layer for it to serve, and `slidingWindow == 0`
        // would size it at the prefill chunk width instead.
        var recurrent = Set<Int>()
        for layer in 0..<cfg.numLayers where cfg.fullAttentionLayerMask[layer] == 0 {
            recurrent.insert(layer)
        }
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
                                     fp16RingEnabled: false,
                                     recurrentLayers: recurrent)
        self.state = try RecurrentStateManager(device: context.device,
                                               config: cfg,
                                               linear: linear)

        self.embed = try QwenEmbedLookupInt8(context: context)
        self.rms = try RMSNorm(context: context)
        self.int4 = try DequantInt4GEMV(context: context)
        self.int8 = try DequantInt8GEMV(context: context)
        self.qwen = try QwenKernels(context: context)
        self.delta = try GatedDeltaNet(context: context)
        self.attention = try Attention(context: context)
        self.moe = try MoE(context: context,
                           routerWeightBits: model.routerWeightBits,
                           gateActivation: fault == .routedActivationGelu
                               ? .geluPytorchTanh : .silu)
        self.head = try QwenLMHeadChainInt8(context: context,
                                            maxD: cfg.hiddenSize,
                                            maxVocab: cfg.vocabSize)

        // Resolve every resident tensor once. A decode step then binds offsets
        // and never looks a name up.
        func projection(_ name: String,
                        _ view: TensorView,
                        rows: Int, cols: Int) throws -> Projection {
            guard let bits = model.residentWeightBits(name), bits == 4 || bits == 8 else {
                throw QwenRunnerError.unsupportedWidth(tensor: name,
                                                       bits: model.residentWeightBits(name))
            }
            guard Int(view.shape.0) == rows, Int(view.shape.1) == cols else {
                throw QwenRunnerError.geometryMismatch(
                    "\(name) is \(view.shape.0)x\(view.shape.1), expected \(rows)x\(cols)")
            }
            return Projection(view: view, bits: bits, rows: UInt32(rows), cols: UInt32(cols))
        }

        var built: [LayerWeights] = []
        built.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            let prefix = "language_model.model.layers.\(L)"
            var linearWeights: LinearLayerWeights?
            var fullWeights: FullLayerWeights?

            if cfg.fullAttentionLayerMask[L] == 0 {
                let p = "\(prefix).linear_attn"
                linearWeights = LinearLayerWeights(
                    inProjQKV: try projection("\(p).in_proj_qkv.weight",
                                              try model.qwenInProjQKV(layer: L),
                                              rows: qkvWidth, cols: cfg.hiddenSize),
                    inProjZ: try projection("\(p).in_proj_z.weight",
                                            try model.qwenInProjZ(layer: L),
                                            rows: valueWidth, cols: cfg.hiddenSize),
                    inProjA: try projection("\(p).in_proj_a.weight",
                                            try model.qwenInProjA(layer: L),
                                            rows: numValueHeads, cols: cfg.hiddenSize),
                    inProjB: try projection("\(p).in_proj_b.weight",
                                            try model.qwenInProjB(layer: L),
                                            rows: numValueHeads, cols: cfg.hiddenSize),
                    outProj: try projection("\(p).out_proj.weight",
                                            try model.qwenOutProj(layer: L),
                                            rows: cfg.hiddenSize, cols: valueWidth),
                    conv1d: try model.qwenConv1D(layer: L),
                    aLog: try model.qwenALog(layer: L),
                    dtBias: try model.qwenDtBias(layer: L),
                    norm: try model.qwenDeltaNorm(layer: L))
            } else {
                let p = "\(prefix).self_attn"
                let queryDim = cfg.numHeads * cfg.fullHeadDim
                let kvDim = cfg.numFullKVHeads * cfg.fullHeadDim
                fullWeights = FullLayerWeights(
                    // 2x: the second half of every head's row is the output
                    // gate (`01-MODEL.md` §3-2).
                    qProj: try projection("\(p).q_proj.weight", try model.qProj(layer: L),
                                          rows: 2 * queryDim, cols: cfg.hiddenSize),
                    kProj: try projection("\(p).k_proj.weight", try model.kProj(layer: L),
                                          rows: kvDim, cols: cfg.hiddenSize),
                    vProj: try projection("\(p).v_proj.weight", try model.vProj(layer: L),
                                          rows: kvDim, cols: cfg.hiddenSize),
                    oProj: try projection("\(p).o_proj.weight", try model.oProj(layer: L),
                                          rows: cfg.hiddenSize, cols: queryDim),
                    qNorm: try model.qNorm(layer: L),
                    kNorm: try model.kNorm(layer: L))
            }

            let sharedGateName = "\(prefix).mlp.shared_expert_gate.weight"
            let sharedGateBits = model.residentWeightBits(sharedGateName)
            if let bits = sharedGateBits, bits != 4, bits != 8 {
                throw QwenRunnerError.unsupportedWidth(tensor: sharedGateName, bits: bits)
            }
            let moeWeights = MoEWeights(
                router: try model.qwenRouter(layer: L),
                sharedGate: try model.qwenSharedExpertGate(layer: L),
                sharedGateBits: sharedGateBits,
                gateProj: try projection("\(prefix).mlp.shared_expert.gate_proj.weight",
                                         try model.qwenSharedExpertGateProj(layer: L),
                                         rows: cfg.intermediateSize, cols: cfg.hiddenSize),
                upProj: try projection("\(prefix).mlp.shared_expert.up_proj.weight",
                                       try model.qwenSharedExpertUp(layer: L),
                                       rows: cfg.intermediateSize, cols: cfg.hiddenSize),
                downProj: try projection("\(prefix).mlp.shared_expert.down_proj.weight",
                                         try model.qwenSharedExpertDown(layer: L),
                                         rows: cfg.hiddenSize, cols: cfg.intermediateSize))

            built.append(LayerWeights(inputNorm: try model.inputNorm(layer: L),
                                      postAttnNorm: try model.postAttnNorm(layer: L),
                                      linear: linearWeights,
                                      full: fullWeights,
                                      moe: moeWeights))
        }
        self.layers = built

        func buffer(_ count: Int, _ stride: Int = MemoryLayout<Float16>.size,
                    _ label: String) throws -> MTLBuffer {
            guard let buf = context.device.makeBuffer(length: max(1, count * stride),
                                                      options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buf.label = "qwen.\(label)"
            return buf
        }

        let queryDim = cfg.numHeads * cfg.fullHeadDim
        let f = cfg.intermediateSize
        self.hidden = try buffer(hiddenSize, MemoryLayout<Float16>.size, "hidden")
        self.normed = try buffer(hiddenSize, MemoryLayout<Float16>.size, "normed")
        self.branchOut = try buffer(hiddenSize, MemoryLayout<Float16>.size, "branchOut")
        self.wide = try buffer(max(qkvWidth, 2 * queryDim), MemoryLayout<Float16>.size, "wide")
        self.zBuf = try buffer(valueWidth, MemoryLayout<Float16>.size, "z")
        self.aBuf = try buffer(numValueHeads, MemoryLayout<Float16>.size, "a")
        self.bBuf = try buffer(numValueHeads, MemoryLayout<Float16>.size, "b")
        self.gBuf = try buffer(numValueHeads, MemoryLayout<Float>.size, "g")
        self.betaBuf = try buffer(numValueHeads, MemoryLayout<Float>.size, "beta")
        self.qDelta = try buffer(numKeyHeads * keyHeadDim, MemoryLayout<Float16>.size, "qDelta")
        self.kDelta = try buffer(numKeyHeads * keyHeadDim, MemoryLayout<Float16>.size, "kDelta")
        self.vDelta = try buffer(valueWidth, MemoryLayout<Float16>.size, "vDelta")
        self.yDelta = try buffer(valueWidth, MemoryLayout<Float16>.size, "yDelta")
        self.oDelta = try buffer(valueWidth, MemoryLayout<Float16>.size, "oDelta")
        self.qCompact = try buffer(queryDim, MemoryLayout<Float16>.size, "qCompact")
        self.attnOut = try buffer(queryDim, MemoryLayout<Float16>.size, "attnOut")
        self.sharedGateAct = try buffer(f, MemoryLayout<Float16>.size, "sharedGateAct")
        self.sharedUpAct = try buffer(f, MemoryLayout<Float16>.size, "sharedUpAct")
        self.sharedAct = try buffer(f, MemoryLayout<Float16>.size, "sharedAct")
        self.sharedOut = try buffer(hiddenSize, MemoryLayout<Float16>.size, "sharedOut")
        self.sharedGateLogit = try buffer(1, MemoryLayout<Float16>.size, "sharedGateLogit")
        self.outIndices = try buffer(cfg.topKExperts, MemoryLayout<UInt32>.size, "routerIndices")
        self.outWeights = try buffer(cfg.topKExperts, MemoryLayout<Float16>.size, "routerWeights")
        self.moeActs = try buffer(cfg.topKExperts * cfg.moeIntermediateSize,
                                  MemoryLayout<Float16>.size, "moeActs")
        self.greedyToken = try buffer(1, MemoryLayout<UInt32>.size, "greedyToken")
        self.unitFeatureScale = try buffer(hiddenSize, MemoryLayout<UInt16>.size, "unitFeature")
        self.unitExpertScale = try buffer(cfg.numExperts, MemoryLayout<UInt16>.size, "unitExpert")

        // BF16 1.0 is 0x3F80.
        let one = Quantization.bf16Bits(1.0)
        let features = unitFeatureScale.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<hiddenSize { features[i] = one }
        let experts = unitExpertScale.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<cfg.numExperts { experts[i] = one }
    }

    // MARK: - Generation

    /// Back to the start of a conversation: the K/V cursor and both recurrent
    /// tensors.
    public func reset() {
        kv.reset()
        state.reset()
    }

    /// Position the next token will occupy.
    public var position: Int { kv.position }

    /// Runs `tokens` through the model one at a time and returns the greedy
    /// continuation of `maxNewTokens` tokens.
    ///
    /// Prompt tokens go through the same per-token path as generated ones —
    /// there is no chunked prefill here. That is Phase 4; running the prompt as
    /// 1-token steps is what makes Phase 3's comparison against the reference a
    /// statement about the decode path alone.
    public func generateGreedy(promptTokens: [Int32],
                               maxNewTokens: Int,
                               stopTokens: Set<Int32> = [],
                               onToken: ((Int, Int32) -> Void)? = nil) throws -> [Int32] {
        precondition(!promptTokens.isEmpty, "the prompt must have at least one token")
        precondition(promptTokens.count + maxNewTokens <= maxContext,
                     "prompt + generation exceeds maxContext \(maxContext)")

        var produced: [Int32] = []
        var next: Int32 = 0
        for (index, token) in promptTokens.enumerated() {
            let emit = index == promptTokens.count - 1
            next = try step(token: token, emitToken: emit)
        }
        for index in 0..<maxNewTokens {
            produced.append(next)
            onToken?(index, next)
            // A stop token is reported, like the reference's `generate`, and
            // then ends the run. The last wanted token is not followed by a
            // forward pass nobody reads.
            if stopTokens.contains(next) { break }
            if produced.count == maxNewTokens { break }
            next = try step(token: next, emitToken: true)
        }
        return produced
    }

    /// One token in, one greedy token out (or 0 when `emitToken` is false, in
    /// which case only the caches moved).
    @discardableResult
    public func step(token: Int32, emitToken: Bool) throws -> Int32 {
        let position = kv.position
        guard position < maxContext else {
            throw QwenRunnerError.geometryMismatch(
                "position \(position) reached maxContext \(maxContext)")
        }
        if fault == .forgetRecurrentState { state.reset() }
        let D = UInt32(hiddenSize)
        let embedding = model.embedding

        try runSync("embed") { cb in
            self.embed.encode(commandBuffer: cb,
                              table: embedding.buffer, tableOffset: Int(embedding.offset),
                              scales: embedding.buffer, scalesOffset: Int(embedding.scaleOffset),
                              biases: embedding.buffer, biasesOffset: Int(embedding.biasOffset),
                              out: self.hidden,
                              tokenId: UInt32(bitPattern: token),
                              d: D)
        }

        for L in 0..<cfg.numLayers {
            let w = layers[L]
            let cb = try commandBuffer()
            rms.encodeBF16W(commandBuffer: cb,
                            x: hidden,
                            weight: w.inputNorm.buffer, weightOffset: Int(w.inputNorm.offset),
                            out: normed, d: D, eps: rmsEps)
            if let linear = w.linear {
                encodeRecurrent(cb, layer: L, w: linear)
            } else if let full = w.full {
                encodeAttention(cb, layer: L, position: position, w: full)
            }
            qwen.encodeResidualAdd(commandBuffer: cb, hidden: hidden, y: branchOut,
                                   count: hiddenSize)
            rms.encodeBF16W(commandBuffer: cb,
                            x: hidden,
                            weight: w.postAttnNorm.buffer,
                            weightOffset: Int(w.postAttnNorm.offset),
                            out: normed, d: D, eps: rmsEps)
            encodeRouter(cb, w: w.moe)
            try wait(cb, "layer \(L) pre-router")

            // The routed-expert blobs cannot be chosen before the router has
            // run, and cannot be read without leaving the GPU. This readback is
            // the reason a layer is two command buffers rather than one.
            let experts = readRoutedExperts()
            // The synchronous arm of the streamer: `startRoutedExpertFetch`
            // hands back a handle whose `wait()` blocks, which is what a serial
            // encode loop wants. The `async` sibling exists for
            // `RealForwardRunner`, whose whole point is to have something else
            // running while the read is in flight.
            guard let plan = try model.planRoutedExperts(layer: L, experts: experts) else {
                throw QwenRunnerError.commandBufferFailed(
                    "layer \(L): the expert streamer could not plan \(experts.count) experts")
            }
            let blobs = try model.startRoutedExpertFetch(plan: plan).wait()

            let tail = try commandBuffer()
            if let set = model.routedExpertResidencySet(layer: L) {
                tail.useResidencySet(set)
            }
            encodeSharedExpert(tail, w: w.moe)
            encodeRoutedExperts(tail, layer: L, blobs: blobs)
            qwen.encodeResidualAdd(commandBuffer: tail, hidden: hidden, y: sharedOut,
                                   count: hiddenSize)
            try wait(tail, "layer \(L) moe")
        }

        kv.advance()
        guard emitToken else { return 0 }

        let lm = try model.qwenLMHead
        let finalNorm = model.finalNorm
        try runSync("head") { cb in
            self.head.encodeGreedyDecode(commandBuffer: cb,
                                         hidden: self.hidden,
                                         normWeight: finalNorm.buffer,
                                         normOffset: Int(finalNorm.offset),
                                         weights: lm.buffer, weightsOffset: Int(lm.offset),
                                         scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                         biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                         outToken: self.greedyToken,
                                         d: D,
                                         vocab: UInt32(self.scoredVocab),
                                         rmsEps: self.rmsEps)
        }
        return Int32(bitPattern: greedyToken.contents().load(as: UInt32.self))
    }

    // MARK: - Layer halves

    /// One Gated DeltaNet layer. `branchOut` holds `out_proj`'s result on exit.
    private func encodeRecurrent(_ cb: MTLCommandBuffer, layer L: Int, w: LinearLayerWeights) {
        encodeProjection(cb, w.inProjQKV, x: normed, y: wide)
        encodeProjection(cb, w.inProjZ, x: normed, y: zBuf)
        encodeProjection(cb, w.inProjA, x: normed, y: aBuf)
        encodeProjection(cb, w.inProjB, x: normed, y: bBuf)

        let convOffset = state.convOffset(layer: L)
        // Decode is one token, so the whole dispatch is a single token block
        // and the convolution history may be updated in place — the wrapper
        // refuses the alias for any wider dispatch.
        qwen.encodeDeltaQKVPrepare(commandBuffer: cb,
                                   qkv: wide,
                                   convWeight: w.conv1d.buffer,
                                   convWeightOffset: Int(w.conv1d.offset),
                                   stateIn: state.convBuffer, stateInOffset: convOffset,
                                   stateOut: state.convBuffer, stateOutOffset: convOffset,
                                   q: qDelta, k: kDelta, v: vDelta,
                                   seqLen: 1,
                                   numKHeads: numKeyHeads,
                                   numVHeads: numValueHeads,
                                   headDim: keyHeadDim)
        qwen.encodeDeltaGates(commandBuffer: cb,
                              a: aBuf, b: bBuf,
                              aLog: w.aLog.buffer, aLogOffset: Int(w.aLog.offset),
                              dtBias: w.dtBias.buffer, dtBiasOffset: Int(w.dtBias.offset),
                              g: gBuf, beta: betaBuf,
                              seqLen: 1, numVHeads: numValueHeads)
        let stateOffset = state.stateOffset(layer: L)
        delta.encode(commandBuffer: cb,
                     q: qDelta, k: kDelta, v: vDelta,
                     g: gBuf, beta: betaBuf,
                     stateIn: state.stateBuffer, stateInOffset: stateOffset,
                     y: yDelta,
                     stateOut: state.stateBuffer, stateOutOffset: stateOffset,
                     seqLen: 1,
                     numKHeads: numKeyHeads,
                     numVHeads: numValueHeads,
                     keyHeadDim: keyHeadDim,
                     valueHeadDim: valueHeadDim)
        qwen.encodeDeltaNormGate(commandBuffer: cb,
                                 o: yDelta, z: zBuf,
                                 weight: w.norm.buffer, weightOffset: Int(w.norm.offset),
                                 out: oDelta,
                                 seqLen: 1,
                                 numVHeads: numValueHeads,
                                 headDim: valueHeadDim,
                                 eps: rmsEps)
        encodeProjection(cb, w.outProj, x: oDelta, y: branchOut)
    }

    /// One full-attention layer. `branchOut` holds `o_proj`'s result on exit.
    private func encodeAttention(_ cb: MTLCommandBuffer,
                                 layer L: Int,
                                 position: Int,
                                 w: FullLayerWeights) {
        let headDim = cfg.fullHeadDim
        let kSlot = kv.kSlot(layer: L, position: position)
        let vSlot = kv.vSlot(layer: L, position: position)

        encodeProjection(cb, w.qProj, x: normed, y: wide)
        encodeProjection(cb, w.kProj, x: normed, y: kSlot.buffer, yOffset: kSlot.offset)
        encodeProjection(cb, w.vProj, x: normed, y: vSlot.buffer, yOffset: vSlot.offset)

        qwen.encodeQKVEpilogue(commandBuffer: cb,
                               q: wide,
                               k: kSlot.buffer, kOffset: kSlot.offset,
                               qWeight: w.qNorm.buffer, qWeightOffset: Int(w.qNorm.offset),
                               kWeight: w.kNorm.buffer, kWeightOffset: Int(w.kNorm.offset),
                               seqLen: 1,
                               numQHeads: cfg.numHeads,
                               numKVHeads: cfg.numFullKVHeads,
                               headDim: headDim,
                               rotaryDim: rotaryDim,
                               position: position,
                               theta: Float(cfg.fullRopeTheta),
                               eps: rmsEps)

        // `wide` interleaves each head's query with its output gate, and the
        // attention kernels index queries at a stride of `headDim`. Copying the
        // 16 query halves out is a blit, not a kernel: the alternative is a
        // second output buffer on the epilogue, which would make the gate the
        // easy thing to forget to carry forward.
        if let blit = cb.makeBlitCommandEncoder() {
            let rowBytes = headDim * MemoryLayout<Float16>.size
            for h in 0..<cfg.numHeads {
                blit.copy(from: wide, sourceOffset: h * 2 * rowBytes,
                          to: qCompact, destinationOffset: h * rowBytes,
                          size: rowBytes)
            }
            blit.endEncoding()
        }

        // `scale` is 1.0 because `q_norm.weight` carries `head_dim ** -0.5`,
        // baked at repack (`Scripts/qwen35/bake_snapshot.py`, `03-DESIGN.md`
        // §1-1). RoPE is a rotation, so moving the scale before it is exact.
        attention.encodeFull(commandBuffer: cb,
                             q: fault == .uncompactedQuery ? wide : qCompact,
                             k: kSlot.buffer, kOffset: 0,
                             v: vSlot.buffer, vOffset: 0,
                             out: attnOut,
                             headDim: UInt32(headDim),
                             numQHeads: UInt32(cfg.numHeads),
                             numKVHeads: UInt32(cfg.numFullKVHeads),
                             seqLen: UInt32(position + 1),
                             scale: 1.0)
        qwen.encodeAttnOutputGate(commandBuffer: cb,
                                  o: attnOut, qGate: wide,
                                  seqLen: 1,
                                  numQHeads: cfg.numHeads,
                                  headDim: headDim)
        encodeProjection(cb, w.oProj, x: attnOut, y: branchOut)
    }

    // MARK: - MoE

    private func encodeRouter(_ cb: MTLCommandBuffer, w: MoEWeights) {
        let router = w.router
        if moe.routerWeightBits == 16 {
            moe.encodeRouterGemma4BF16(commandBuffer: cb,
                                       weights: router.buffer,
                                       weightsOffset: Int(router.offset),
                                       hidden: normed,
                                       effectiveScale: unitFeatureScale,
                                       perExpertScale: unitExpertScale,
                                       perExpertScaleOffset: 0,
                                       outIndices: outIndices, outWeights: outWeights,
                                       numExperts: UInt32(cfg.numExperts),
                                       d: UInt32(hiddenSize),
                                       topK: UInt32(cfg.topKExperts))
        } else {
            moe.encodeRouterGemma4(commandBuffer: cb,
                                   weights: router.buffer, weightsOffset: Int(router.offset),
                                   scales: router.buffer, scalesOffset: Int(router.scaleOffset),
                                   biases: router.buffer, biasesOffset: Int(router.biasOffset),
                                   hidden: normed,
                                   effectiveScale: unitFeatureScale,
                                   perExpertScale: unitExpertScale,
                                   perExpertScaleOffset: 0,
                                   outIndices: outIndices, outWeights: outWeights,
                                   numExperts: UInt32(cfg.numExperts),
                                   d: UInt32(hiddenSize),
                                   topK: UInt32(cfg.topKExperts))
        }
    }

    private func readRoutedExperts() -> [Int] {
        let ptr = outIndices.contents().bindMemory(to: UInt32.self, capacity: cfg.topKExperts)
        return (0..<cfg.topKExperts).map { min(Int(ptr[$0]), cfg.numExperts - 1) }
    }

    /// `sigmoid(w_gate . x) * down(silu(gate(x)) * up(x))` into `sharedOut`.
    private func encodeSharedExpert(_ cb: MTLCommandBuffer, w: MoEWeights) {
        encodeProjection(cb, w.gateProj, x: normed, y: sharedGateAct)
        encodeProjection(cb, w.upProj, x: normed, y: sharedUpAct)
        qwen.encodeSiluMul(commandBuffer: cb,
                           gate: sharedGateAct, up: sharedUpAct, out: sharedAct,
                           count: cfg.intermediateSize)
        encodeProjection(cb, w.downProj, x: sharedAct, y: sharedOut)

        let gate = w.sharedGate
        if fault == .sharedGateSkipped {
            return
        }
        if let bits = w.sharedGateBits, fault != .sharedGateAsBF16 {
            // A quantized `[1, D]` row: the generic GEMV dequantizes it and the
            // gate kernel only applies the sigmoid.
            encodeGEMV(cb, bits: bits, view: gate,
                       x: normed, xOffset: 0,
                       y: sharedGateLogit, yOffset: 0,
                       m: 1, n: UInt32(hiddenSize))
            qwen.encodeMoESharedGateLogit(commandBuffer: cb,
                                          y: sharedOut,
                                          logit: sharedGateLogit,
                                          hiddenSize: hiddenSize)
        } else {
            qwen.encodeMoESharedGate(commandBuffer: cb,
                                     y: sharedOut, x: normed,
                                     weight: gate.buffer, weightOffset: Int(gate.offset),
                                     hiddenSize: hiddenSize)
        }
    }

    /// The eight routed experts, summed on top of the shared branch. Phase 2's
    /// reduce takes a residual, so `sharedOut` is both the residual and the
    /// destination: each threadgroup owns one output element and reads it
    /// before writing it.
    private func encodeRoutedExperts(_ cb: MTLCommandBuffer, layer L: Int, blobs: [TensorView]) {
        let routedBufs = blobs.map { $0.buffer }
        let offsets = model.routedExpertOffsets(layer: L)
        let topK = UInt32(cfg.topKExperts)
        let argBuf = moe.makeReusedRoutedArgumentBuffer(routedBlobs: routedBufs, topK: topK)
        moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                routedArgBuffer: argBuf,
                                                routedBlobs: routedBufs,
                                                routedOffsets: offsets,
                                                x: normed,
                                                acts: moeActs,
                                                d: UInt32(hiddenSize),
                                                f: UInt32(cfg.moeIntermediateSize),
                                                topK: topK)
        moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: cb,
                                               routedArgBuffer: argBuf,
                                               routedBlobs: routedBufs,
                                               routedOffsets: offsets,
                                               acts: moeActs,
                                               routingWeights: outWeights,
                                               residual: sharedOut,
                                               y: sharedOut,
                                               d: UInt32(hiddenSize),
                                               f: UInt32(cfg.moeIntermediateSize),
                                               topK: topK)
    }

    // MARK: - Plumbing

    private func encodeProjection(_ cb: MTLCommandBuffer,
                                  _ p: Projection,
                                  x: MTLBuffer, xOffset: Int = 0,
                                  y: MTLBuffer, yOffset: Int = 0) {
        encodeGEMV(cb, bits: p.bits, view: p.view,
                   x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                   m: p.rows, n: p.cols)
    }

    private func encodeGEMV(_ cb: MTLCommandBuffer,
                            bits: Int,
                            view: TensorView,
                            x: MTLBuffer, xOffset: Int,
                            y: MTLBuffer, yOffset: Int,
                            m: UInt32, n: UInt32) {
        if bits == 8 {
            int8.encode(commandBuffer: cb,
                        weights: view.buffer, weightsOffset: Int(view.offset),
                        scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                        biases: view.buffer, biasesOffset: Int(view.biasOffset),
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: m, n: n)
        } else {
            int4.encode(commandBuffer: cb,
                        weights: view.buffer, weightsOffset: Int(view.offset),
                        scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                        biases: view.buffer, biasesOffset: Int(view.biasOffset),
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: m, n: n)
        }
    }

    func commandBuffer() throws -> MTLCommandBuffer {
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw QwenRunnerError.commandBufferFailed("makeCommandBuffer returned nil")
        }
        return cb
    }

    func wait(_ cb: MTLCommandBuffer, _ label: String) throws {
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error {
            throw QwenRunnerError.commandBufferFailed("\(label): \(error)")
        }
    }

    func runSync(_ label: String, _ body: (MTLCommandBuffer) -> Void) throws {
        let cb = try commandBuffer()
        body(cb)
        try wait(cb, label)
    }
}
