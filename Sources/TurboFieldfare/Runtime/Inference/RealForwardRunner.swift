import Foundation
import Metal

public enum RDAdvicePolicyMode: String, Codable, Sendable, Equatable {
    case `default`
    case off
    case bounded
    case adaptive

    public static func parse(_ raw: String?) -> RDAdvicePolicyMode {
        switch raw?.lowercased() {
        case "off", "none", "disabled":
            return .off
        case "bounded":
            return .bounded
        case "adaptive":
            return .adaptive
        default:
            return .default
        }
    }
}

public struct RDAdviceAdaptivePolicyConfig: Sendable, Equatable {
    public var missCap: Int
    public var byteCap: UInt64
    public var slowCallNanos: UInt64

    public init(missCap: Int,
                byteCap: UInt64,
                slowCallNanos: UInt64) {
        self.missCap = missCap
        self.byteCap = byteCap
        self.slowCallNanos = slowCallNanos
    }

    public static let conservative = RDAdviceAdaptivePolicyConfig(
        missCap: 12,
        byteCap: 384 * 1_048_576,
        slowCallNanos: 1_000_000)
}

struct RDAdviceAdaptivePolicyState: Sendable, Equatable {
    var config: RDAdviceAdaptivePolicyConfig
    private var skipUntilPosition: Int = -1
    private(set) var recentSlowCallNanos: UInt64 = 0

    init(config: RDAdviceAdaptivePolicyConfig = .conservative) {
        self.config = config
    }

    mutating func reset() {
        skipUntilPosition = -1
        recentSlowCallNanos = 0
    }

    func shouldSkip(position: Int,
                    requestedMisses: Int,
                    estimatedBytes: UInt64,
                    canOverlapUsefulGPUWork: Bool) -> Bool {
        position <= skipUntilPosition ||
        !canOverlapUsefulGPUWork ||
        requestedMisses > config.missCap ||
        estimatedBytes > config.byteCap
    }

    mutating func update(after result: ExpertIOAdviceResult,
                                position: Int) {
        recentSlowCallNanos = max(recentSlowCallNanos, result.maxCallNanos)
        if result.maxCallNanos >= config.slowCallNanos {
            skipUntilPosition = max(skipUntilPosition, position)
        }
    }
}

/// Gemma 4 real-forward decode pass.
///
/// Composes the production kernels against the `.gturbo` model:
///
///   embed_lookup_int4(token) * sqrt(H)
///   for L in 0..<30:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     Q = q_proj(a)    K = k_proj(a)    V = (SWA) v_proj(a) | (full) k_proj(a)
///     per-head q/k_norm (bf16w), per-head v_norm (no_scale)
///     NeoX RoPE on Q + K (default for SWA, proportional for full)
///     write K and V into separate cache slots
///     attn = attention(scale=1.0, SWA window or full causal)
///     attn = o_proj(attn)
///     h = h + rmsnorm_bf16w(attn, post_attention_layernorm)
///     h1 = rmsnorm_bf16w(h, pre_feedforward_layernorm)
///     h1 = SharedExpertInt8(h1)
///     h1 = rmsnorm_bf16w(h1, post_feedforward_layernorm_1)
///     // router + routed branch
///     xr   = rmsnorm_no_scale(h)
///     idx, w = router_topk_gemma4(xr, effective_scale[L], per_expert_scale[L])
///     h2 = rmsnorm_bf16w(h, pre_feedforward_layernorm_2)
///     h2 = moe_fused_ffn_streamed_routed(h2, residual=0, routedBlobs=fetch(idx), w)
///     h2 = rmsnorm_bf16w(h2, post_feedforward_layernorm_2)
///     h = h + rmsnorm_bf16w(h1 + h2, post_feedforward_layernorm)
///     h = h * layer_scalar[L]
///   logits = DequantInt4GEMV(rmsnorm_bf16w(h, model.norm), embed_table^T)
///   // final softcap and softmax happen in the Sampler.
///
/// Direct against `Model`; this is the only production decode forward path.
internal enum PrefillProjectionFamily: Sendable, Equatable {
    case q
    case kv
    case o
    case shared
    case routed
}

internal enum PrefillProjectionDispatch: Sendable, Equatable {
    case repeatedGEMV
    case qmm
}

internal enum PrefillProjectionDispatchPolicy {
    /// `tiledQMM` is whether `PrefillInt4QMM` can serve this shape with the
    /// `simdgroup_matrix` kernel rather than the scalar one. Q is the widest
    /// projection (N = numQHeads * headDim), which is exactly where the scalar
    /// QMM's 8x8 threadgroup lost to a GEMV per row; the tiled kernel reverses
    /// that, so Q only stays on `repeatedGEMV` when the tiled path is off.
    static func selectedDispatch(for family: PrefillProjectionFamily,
                                 chunkTokens: Int,
                                 tiledQMM: Bool) -> PrefillProjectionDispatch {
        guard chunkTokens >= 32 else {
            return .repeatedGEMV
        }
        switch family {
        case .q:
            return tiledQMM ? .qmm : .repeatedGEMV
        case .kv, .o, .shared, .routed:
            return .qmm
        }
    }
}

/// A deliberate deviation in how a prompt's image rows are prefilled.
///
/// `VisionTowerFault` does the same job inside the tower; these two are the
/// mistakes that live *outside* it, in the two places PLAN_VISION §6-3 predicts
/// a plausible-looking wrong answer rather than a crash. Selected by
/// `TF_VISION_PREFILL_FAULT` and never by a normal run.
enum VisionPrefillFault: String, Sendable, CaseIterable {
    case none = ""
    /// Soft tokens multiplied by sqrt(text hidden), the scaling that belongs to
    /// the text embedding lookup alone (§2-6). 53x too large.
    case softTokensScaled = "soft-tokens-scaled"
    /// No bidirectional span: every soft token sees only the ones before it.
    case causalOnly = "causal-only"

    static let selected: VisionPrefillFault = {
        let raw = ProcessInfo.processInfo.environment["TF_VISION_PREFILL_FAULT"] ?? ""
        guard let fault = VisionPrefillFault(rawValue: raw) else {
            preconditionFailure("unknown TF_VISION_PREFILL_FAULT \(raw); "
                                + "expected one of \(VisionPrefillFault.allCases.map(\.rawValue))")
        }
        return fault
    }()
}

public final class RealForwardRunner: ChunkedPrefillRunner, ContextWindowReporting, ContinuableLogitProducer, SpeculativeDrafting, @unchecked Sendable {
    private struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        let postF1: TensorView
    }

    private let model: Model
    private let ctx: MetalContext
    private let kv: KVCacheManager?
    private let cfg: ArchConfig

    // Kernels
    private let embedInt4: EmbedLookupInt4
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let attention: Attention
    private let shared: SharedExpertRuntime
    private let moe: MoE
    private let fusionHead: LMHeadChainInt4
    private let fusedQKVGEMV: FusedQKVGEMV
    private let fusedQKVEpilogue: FusedQKVEpilogue
    private let fusedPostAttentionSetup: FusedPostAttentionSetup
    private let fusedTail: FusedLayerTail

    // Prefill kernels. These are initialized once per runner so the chunk path
    // cannot accidentally rebuild PSOs inside a per-layer loop.
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    private let prefillMPPAffineInt4: MPPPrefillInt4QMM?
    private let prefillQKVEpilogue: PrefillQKVEpilogue
    private let prefillAttention: PrefillAttention
    private let prefillPostAttention: PrefillPostAttentionSetup
    private let prefillRouter: PrefillRouter
    private let prefillSharedExpert: PrefillSharedExpert
    private let prefillGroupedMoE: PrefillGroupedRoutedMoE
    private let prefillMoE: PrefillMoE
    private let prefillLayerTail: PrefillLayerTail
    private let prefillFinalRowHead: PrefillFinalRowHeadInt4

    // Scratch — preallocated per spec'd D / F / vocab.
    private let hidden: MTLBuffer        // [D] FP16
    private let normed: MTLBuffer        // [D] FP16
    private let attnOut: MTLBuffer       // [N_HEADS * head_dim] FP16
    private let qScratch: MTLBuffer      // [N_HEADS * head_dim] FP16
    private let kStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let vStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let oOut: MTLBuffer          // [D] FP16
    private let h1Buf: MTLBuffer         // [D] FP16 (dense MLP output)
    private let h2Buf: MTLBuffer         // [D] FP16 (routed output)
    private let routedX: MTLBuffer       // [D] FP16 (pre_feedforward_layernorm_2 output)
    private let denseX: MTLBuffer        // [D] FP16 (pre_feedforward_layernorm output)
    private let denseScratchGate: MTLBuffer // [F=2112] FP16
    private let denseScratchUp: MTLBuffer   // [F=2112] FP16
    private let denseScratchAct: MTLBuffer  // [F=2112] FP16
    private let routerInput: MTLBuffer   // [D] FP16 (rmsnorm_no_scale(h))
    private let zeroResidual: MTLBuffer  // [D] FP16 zeros — for routed branch base
    private let outIndices: MTLBuffer    // [topK] UInt32
    private let outWeights: MTLBuffer    // [topK] FP16
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
    private let moeActs: MTLBuffer       // [topK * FmoE] FP16
    private let moeHitActiveSlots: MTLBuffer // [topK] UInt32
    private let moeMissActiveSlots: MTLBuffer // [topK] UInt32
    private let greedyTokenBuf: MTLBuffer // 4 B UInt32 fused-head output
    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?
    /// The chunk width this runner was built for. `verifyBlock` reuses it so a
    /// 4-token block binds the scratch prefill already allocated instead of
    /// swapping it for a narrower layout every round.
    private let prefillRuntimeConfig: PrefillRuntimeConfig
    private var prefillGPUProfile = PrefillGPUProfile()
    private var prefillHostProfile = PrefillHostProfile()
    /// 28-M8 §3 の果実 B の前提 (層 L の hidden から層 L+d の routing を
    /// 当てられるか) を実装の前に潰すための計器。`TF_MTP_ROUTER_PREVIEW=1`
    /// のときだけ動き、出力にも fetch にも触らない。
    private var routerPreviewProbe = RouterPreviewProbe()
    /// 予測の書き出し先。距離ごとに 1 本、`maxTokens * topK` の UInt32。
    /// 計器が有効なときだけ確保する。
    private var routerPreviewIDs: [MTLBuffer] = []
    private var routerPreviewWeights: [MTLBuffer] = []
    /// The same busy/idle split for plain decode, so a block's occupancy has
    /// something to be compared against: `verify(k)` is a ratio of two wall
    /// clocks, and it says nothing about whether either side is GPU-bound.
    /// One line per `decodeQueueWindow` steps, on the same env var.
    private var decodeQueue = GPUQueueOccupancy()
    private var decodeQueueSteps = 0
    private var decodeQueueRecording = false
    private static let decodeQueueWindow = 32
    /// Decode's own per-stage GPU split, in the block's buckets: `attn` is the
    /// one buffer that carries norm/QKV/RoPE/attention/o/router, `moe` the
    /// routed experts, `head` the LM head. Which bucket a wait belongs to is
    /// set by the site that waits.
    private var decodeStageProfile = PrefillGPUProfile()
    private var decodeStage = PrefillGPUProfile.Stage.attention

    /// Vision state, built on the first image and never for a text-only run:
    /// the tower owns 83 MB of scratch and its weights are 1.15 GB, so a prompt
    /// without an image must not construct either (`PLAN_VISION.md` §4-1).
    private var visionTower: VisionTower?
    private var visionScatter: VisionSoftTokenScatter?
    /// Wall seconds the tower has spent on this runner, and images it ran on.
    /// Reported in the CLI footer: the tower is the whole of the vision cost and
    /// it lands on TTFT (§7 gate 4).
    public private(set) var visionTowerSeconds: Double = 0
    public private(set) var visionTowerImages: Int = 0
    public private(set) var visionLoadSeconds: Double = 0

    private static let rdadviseBoundedMissCap = 12
    private static let rdadviseBoundedMaxCallNanos: UInt64 = 250_000
    private static let rdadviseAdaptiveMissCap = 12
    private static let rdadviseAdaptiveByteCap: UInt64 = 384 * 1_048_576
    private static let rdadviseAdaptiveSlowCallNanos: UInt64 = 1_000_000
    private static let prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig()
    /// Narrowest chunk that still uses the tiled routed-MoE GEMM. 0 keeps the
    /// pre-M4 behavior (always tiled).
    private static let groupedGEMMMinChunkTokens = Int(
        ProcessInfo.processInfo.environment["TF_PREFILL_MOE_GEMM_MIN_TOKENS"] ?? "") ?? 0
    /// The M4.5 k-row paths, each separately switchable so a measurement can
    /// attribute a change to one of them (docs/mtp/16-M4.5-PLAN.md §5).
    /// Unset means on; `0` puts that call back on the path it had before.
    private static func rowsPathEnabled(_ name: String) -> Bool {
        ProcessInfo.processInfo.environment["TF_MTP_ROWS_\(name)"] != "0"
    }
    private static let rowsDenseEnabled = rowsPathEnabled("DENSE")
    private static let rowsMoEEnabled = rowsPathEnabled("MOE")
    private static let rowsHeadEnabled = rowsPathEnabled("HEAD")
    private static let rowsRouterEnabled = rowsPathEnabled("ROUTER")
    private static let rowsAttentionEnabled = rowsPathEnabled("ATTENTION")
    /// Inside the rows attention path: `1` (default) walks the KV range once
    /// for the whole block, `0` keeps M5.5's row-at-a-time decode calls, which
    /// read it `k` times (docs/mtp/25-M5.6-RESULTS.md).
    private static let rowsSharedAttentionEnabled = rowsPathEnabled("ATTENTION_SHARED")
    private static let blockPipelineEnabled = rowsPathEnabled("PIPELINE")
    private static let blockTileLookaheadEnabled = rowsPathEnabled("LOOKAHEAD")
    /// One expert-cache plan per layer instead of one per tile
    /// (`TF_MTP_ROWS_LAYER_PLAN=0` restores the per-tile planning).
    private static let blockLayerPlanEnabled = rowsPathEnabled("LAYER_PLAN")
    /// Resident experts first in a block's tile order
    /// (`TF_MTP_ROWS_RESIDENT_FIRST=0` keeps the physical order for every tile).
    private static let blockResidentFirstEnabled = rowsPathEnabled("RESIDENT_FIRST")
    /// The block's router as one dispatch pair instead of one per row
    /// (`TF_MTP_ROWS_ROUTER_FOLD=0` keeps the row loop).
    private static let blockRowsRouterFoldEnabled = rowsPathEnabled("ROUTER_FOLD")
    /// Set on the sort key of an expert that is not in a slot. Above every
    /// physical offset (a layer's stream is well under 2^40 bytes), so it
    /// splits the order in two without disturbing either half.
    private static let nonResidentSortBit: UInt64 = 1 << 40

    /// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    private let effectiveScaleBuffers: [MTLBuffer]
    private let sharedExpertProjections: [LayerSharedExpertProjections]

    public let maxContext: Int

    /// Per-instance head and RDADVISE modes. The fused head (default) skips the
    /// 512 KB logits write and leaves a greedy argmax in `lastGreedyToken`;
    /// callers that sample from the logits buffer (non-greedy configs) must pass
    /// `forceLogitsHead: true` or they read a never-written buffer.
    private let useFusedGreedyHead: Bool
    private let prefillAttentionPath: RuntimePrefillAttentionPath
    public let rdadviseEnabled: Bool
    public let rdadvisePolicyMode: RDAdvicePolicyMode
    private var rdadviseSkipUntilPosition: Int = -1
    private var rdadviseAdaptiveState: RDAdviceAdaptivePolicyState
    private var rdadviseAdaptivePosition: Int = -1
    private var rdadviseAdaptivePositionBytes: UInt64 = 0

    /// M3's acceptance-rate diagnostic (`docs/mtp/04-PHASES.md` §2), off unless
    /// `TF_DRAFT_PROBE` is set. Resolved once at init so the decode path pays
    /// an optional-nil test, and the 236 MB drafter is only loaded if a decode
    /// step actually runs.
    private let draftProbeSettings: (path: String, drafts: Int, mode: DraftAcceptanceProbe.Mode)?
    private var draftProbe: DraftAcceptanceProbe?

    /// M5's production drafter (`docs/mtp/04-PHASES.md` M5), built on the first
    /// round of a speculative generation and never during a plain run — the
    /// 236 MB drafter stays unread when `--draft-block-size 0` (D1's lazy load).
    private var speculativeDrafter: SpeculativeDrafter?
    /// Set by the speculative loop; see `SpeculativeDrafting`.
    public var speculativeHiddenRows: MTLBuffer?
    public private(set) var verifyBlockCount = 0
    public private(set) var verifyBlockNanos: UInt64 = 0
    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.maxContext = maxContext
        // The shader library bakes in the affine group size, so it has to be
        // chosen before the first pipeline is built — which is what the kernel
        // constructors below do. `MetalContext` is created before `Model.load`
        // at every entry point, so this is the first place the manifest value
        // is known.
        try context.setAffineGroupSize(model.affineGroupSize)
        // Reject a cache that would not stay resident before allocating any of
        // it: overshooting the working set trades SSD reads for OS compression,
        // which is the opposite of what a large cache is for.
        let prefillConfig = runtimeConfiguration.prefillConfig
        self.prefillRuntimeConfig = prefillConfig
        let budget = model.expertCacheBudget(
            slotCount: runtimeConfiguration.expertCacheSlots,
            maxContext: maxContext,
            prefillConfig: prefillConfig)
        guard budget.fitsRecommendedWorkingSet else {
            throw ExpertCacheBudgetError.exceedsRecommendedWorkingSet(budget)
        }
        self.memoryBudget = budget
        self.useFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
        self.prefillAttentionPath = runtimeConfiguration.prefillAttentionPath
        let useFP16Ring = runtimeConfiguration.fp16RingEnabled
        self.rdadvisePolicyMode = runtimeConfiguration.rdadvisePolicy
        self.rdadviseAdaptiveState = RDAdviceAdaptivePolicyState(
            config: RDAdviceAdaptivePolicyConfig(
                missCap: Self.rdadviseAdaptiveMissCap,
                byteCap: Self.rdadviseAdaptiveByteCap,
                slowCallNanos: Self.rdadviseAdaptiveSlowCallNanos))
        self.rdadviseEnabled = runtimeConfiguration.rdadviseEnabled
        self.draftProbeSettings = DraftAcceptanceProbe.settings()
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
                                     fp16RingEnabled: useFP16Ring,
                                     slidingWindow: cfg.slidingWindow,
                                     maxPrefillChunkTokens: prefillConfig.chunkTokens,
                                     maxSpeculativeBlockTokens: SpeculativeBlock.maxTokens)

        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.rms       = try RMSNorm(context: context)
        self.int4      = try DequantInt4GEMV(context: context)
        self.attention = try Attention(context: context)
        self.shared    = try SharedExpertRuntime(context: context,
                                                  weightBits: model.sharedExpertWeightBits)
        self.moe       = try MoE(context: context,
                                 routerWeightBits: model.routerWeightBits)
        self.fusionHead = try LMHeadChainInt4(context: context,
                                              maxD: cfg.hiddenSize,
                                              maxVocab: cfg.vocabSize)
        self.fusedQKVGEMV = try FusedQKVGEMV(context: context)
        self.fusedQKVEpilogue = try FusedQKVEpilogue(context: context)
        self.fusedPostAttentionSetup = try FusedPostAttentionSetup(context: context)
        self.fusedTail = try FusedLayerTail(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(context: context)
        self.prefillMPPAffineInt4 = MPPPrefillInt4QMM(context: context)
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(context: context)
        self.prefillAttention = try PrefillAttention(context: context)
        self.prefillPostAttention = try PrefillPostAttentionSetup(context: context)
        self.prefillRouter = try PrefillRouter(context: context,
                                               routerWeightBits: model.routerWeightBits)
        self.prefillSharedExpert = try PrefillSharedExpert(
            context: context,
            weightBits: model.sharedExpertWeightBits)
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(context: context)
        self.prefillMoE = try PrefillMoE(context: context)
        self.prefillLayerTail = try PrefillLayerTail(context: context)
        self.prefillFinalRowHead = try PrefillFinalRowHeadInt4(context: context,
                                                               maxD: cfg.hiddenSize)

        let device = context.device
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let maxQ = cfg.numHeads * max(cfg.headDim, cfg.fullHeadDim)

        func buf(_ count: Int, _ stride: Int = MemoryLayout<Float16>.size) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * stride,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return b
        }
        self.hidden        = try buf(D)
        self.normed        = try buf(D)
        self.attnOut       = try buf(maxQ)
        self.qScratch      = try buf(maxQ)
        self.kStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.vStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.oOut          = try buf(D)
        self.h1Buf         = try buf(D)
        self.h2Buf         = try buf(D)
        self.routedX       = try buf(D)
        self.denseX        = try buf(D)
        self.denseScratchGate = try buf(F)
        self.denseScratchUp   = try buf(F)
        self.denseScratchAct  = try buf(F)
        self.routerInput   = try buf(D)
        self.zeroResidual  = try buf(D)
        // The routed MoE kernel seeds y[d] = residual[d]; pinning this buffer
        // to zero once at init makes the routed branch's residual contribution
        // exactly zero (it's combined with the dense MLP downstream).
        memset(self.zeroResidual.contents(), 0, self.zeroResidual.length)
        self.outIndices    = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.outWeights    = try buf(cfg.topKExperts)
        self.moeActs       = try buf(cfg.topKExperts * cfg.moeIntermediateSize)
        self.moeHitActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.moeMissActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        guard let tok = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.greedyTokenBuf = tok

        func sharedProj(_ view: TensorView, rows: UInt32, cols: UInt32) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer,
                                 scales: view.buffer,
                                 biases: view.buffer,
                                 weightsOffset: Int(view.offset),
                                 scalesOffset: Int(view.scaleOffset),
                                 biasesOffset: Int(view.biasOffset),
                                 rows: rows,
                                 cols: cols)
        }
        var sharedViews: [LayerSharedExpertProjections] = []
        sharedViews.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            let gate = try model.sharedExpertGate(layer: L)
            let up = try model.sharedExpertUp(layer: L)
            let down = try model.sharedExpertDown(layer: L)
            sharedViews.append(LayerSharedExpertProjections(
                gate: sharedProj(gate, rows: UInt32(F), cols: UInt32(D)),
                up: sharedProj(up, rows: UInt32(F), cols: UInt32(D)),
                down: sharedProj(down, rows: UInt32(D), cols: UInt32(F)),
                postF1: try model.postFFN1(layer: L)))
        }
        self.sharedExpertProjections = sharedViews

        // Pre-fold 1/sqrt(D) into router.scale per layer. Each layer gets its
        // own BF16 [D] buffer — the kernel reads `effective_scale[i]` and we
        // pay for the multiply once per generation, not per token.
        var perLayer: [MTLBuffer] = []
        perLayer.reserveCapacity(cfg.numLayers)
        let invSqrtD = Float(1.0) / Float(D).squareRoot()
        let dInts = D
        for L in 0..<cfg.numLayers {
            let scaleView = try model.routerScale(layer: L)
            guard let buf = device.makeBuffer(length: dInts * MemoryLayout<UInt16>.size,
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            let src = scaleView.buffer.contents()
                .advanced(by: Int(scaleView.offset))
                .assumingMemoryBound(to: UInt16.self)
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<dInts {
                let v = Quantization.bf16ToFloat(src[i]) * invSqrtD
                dst[i] = Quantization.bf16Bits(v)
            }
            buf.label = "effective_scale.L\(L)"
            perLayer.append(buf)
        }
        self.effectiveScaleBuffers = perLayer
    }

    public func reset() {
        kv?.reset()
        resetTransientState()
    }

    public var continuationPosition: Int {
        kv?.position ?? 0
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard let kv else {
            throw PrefillError.prefillCursorMismatch(
                "continuation requires an initialized KV cache")
        }
        guard expectedPosition > 0, kv.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected KV position \(expectedPosition), current \(kv.position)")
        }
        resetTransientState()
    }

    private func resetTransientState() {
        // A reset or a continuation breaks the (hidden, next token) chain the
        // probe pairs up, and ends one generation's worth of rounds.
        draftProbe?.flushSummary()
        prefillChunkState.reset()
        rdadviseSkipUntilPosition = -1
        rdadviseAdaptiveState.reset()
        rdadviseAdaptivePosition = -1
        rdadviseAdaptivePositionBytes = 0
    }

    /// Memory this configuration committed to at init, for diagnostics.
    public let memoryBudget: ExpertCacheBudget

    public private(set) var totalIoNanos: UInt64 = 0
    public private(set) var totalCb1Nanos: UInt64 = 0
    public private(set) var totalCb2Nanos: UInt64 = 0
    public private(set) var totalHeadNanos: UInt64 = 0
    public private(set) var totalHeadFusedNanos: UInt64 = 0
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }
    public private(set) var totalRDAdviseNanos: UInt64 = 0
    public private(set) var totalRDAdviseCalls: UInt64 = 0
    public private(set) var totalRDAdviseBytes: UInt64 = 0
    public private(set) var totalRDAdviseFailures: UInt64 = 0
    public private(set) var totalRDAdviseSkipped: UInt64 = 0

    private func recordRDAdvice(_ result: ExpertIOAdviceResult, wallNanos: UInt64) {
        totalRDAdviseNanos &+= wallNanos
        totalRDAdviseCalls &+= UInt64(result.calls)
        totalRDAdviseBytes &+= result.bytes
        totalRDAdviseFailures &+= UInt64(result.failed)
        totalRDAdviseSkipped &+= UInt64(result.skipped)
    }

    private func shouldSkipRDAdvice(position: Int,
                                    requestedMisses: Int,
                                    estimatedBytes: UInt64,
                                    canOverlapUsefulGPUWork: Bool) -> ExpertIOAdviceResult? {
        switch rdadvisePolicyMode {
        case .bounded:
            if position <= rdadviseSkipUntilPosition {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            if requestedMisses > Self.rdadviseBoundedMissCap {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            return nil
        case .adaptive:
            if position != rdadviseAdaptivePosition {
                rdadviseAdaptivePosition = position
                rdadviseAdaptivePositionBytes = 0
            }
            let cumulativeEstimatedBytes = rdadviseAdaptivePositionBytes &+ estimatedBytes
            let shouldSkip = rdadviseAdaptiveState.shouldSkip(
                position: position,
                requestedMisses: requestedMisses,
                estimatedBytes: cumulativeEstimatedBytes,
                canOverlapUsefulGPUWork: canOverlapUsefulGPUWork)
            rdadviseAdaptivePositionBytes = cumulativeEstimatedBytes
            guard shouldSkip else { return nil }
            return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                bytes: estimatedBytes)
        case .default, .off:
            return nil
        }
    }

    private func updateRDAdvicePolicy(after result: ExpertIOAdviceResult,
                                      position: Int) {
        switch rdadvisePolicyMode {
        case .bounded:
            if result.maxCallNanos > Self.rdadviseBoundedMaxCallNanos {
                rdadviseSkipUntilPosition = max(rdadviseSkipUntilPosition, position + 1)
            }
        case .adaptive:
            rdadviseAdaptiveState.update(after: result, position: position)
        case .default, .off:
            break
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        model.telemetry.beginPhase(.decode, step: position)
        try await produceToken(token: token,
                               position: position,
                               into: logits,
                               emitHead: true,
                               outputMode: .greedyIfAvailable)
    }

    public var maxSpeculativeBlockTokens: Int { SpeculativeBlock.maxTokens }

    /// Score a block of proposed tokens in one pass (`docs/mtp/03-DESIGN.md` D3).
    ///
    /// This is the prefill chunk path with the head widened from "last row" to
    /// "every row" — no second forward implementation, which is the point: the
    /// tokens a speculative round accepts have to be the tokens plain decode
    /// would have produced, and the cheapest way to keep that true is for the
    /// two paths to share every kernel below the head. That they still agree is
    /// not assumed; it is the M4 exit condition
    /// (`TurboFieldfareKernelCheck --verify-block`).
    public func verifyBlock(tokens: ArraySlice<Int32>,
                            startPosition: Int,
                            into logitRows: MTLBuffer,
                            greedyTokens: MTLBuffer?) async throws {
        try prefillChunkState.requireClean(operation: "verifyBlock")
        guard !tokens.isEmpty else { return }
        guard tokens.count <= SpeculativeBlock.maxTokens else {
            throw SpeculativeVerifyError.blockTooWide(
                "verifyBlock got \(tokens.count) tokens, more than the "
                + "\(SpeculativeBlock.maxTokens)-token block ceiling")
        }
        guard let kv else {
            throw SpeculativeVerifyError.cursorMismatch("verifyBlock requires FP16 KV")
        }
        guard kv.position == startPosition else {
            throw SpeculativeVerifyError.cursorMismatch(
                "verifyBlock cursor \(kv.position) != startPosition \(startPosition)")
        }
        guard startPosition + tokens.count <= maxContext else {
            throw SpeculativeVerifyError.cursorMismatch(
                "verifyBlock range [\(startPosition), \(startPosition + tokens.count)) "
                + "exceeds maxContext \(maxContext)")
        }
        // The two heads write different things to different buffers, and a
        // caller that picked the wrong one would read a buffer nothing wrote —
        // plausible stale bytes rather than a crash. Name the mismatch instead.
        let outputMode: PrefillOutputMode
        if let greedyTokens {
            guard useFusedGreedyHead else {
                throw SpeculativeVerifyError.headMismatch(
                    "verifyBlock was given a greedy-token buffer but this runner "
                    + "was built with the logits head")
            }
            outputMode = .greedyIfAvailable
            let needed = tokens.count * MemoryLayout<UInt32>.stride
            guard greedyTokens.length >= needed else {
                throw SpeculativeVerifyError.bufferTooSmall(
                    "greedyTokens holds \(greedyTokens.length) bytes, needs \(needed)")
            }
        } else {
            outputMode = .logits
            let needed = SpeculativeBlock.logitRowsBytes(vocab: cfg.vocabSize,
                                                         blockTokens: tokens.count)
            guard logitRows.length >= needed else {
                throw SpeculativeVerifyError.bufferTooSmall(
                    "logitRows holds \(logitRows.length) bytes, needs \(needed) for "
                    + "\(tokens.count) rows of \(cfg.vocabSize)")
            }
        }

        let scratch = try ensurePrefillScratch(config: prefillRuntimeConfig)
        // Verify is generation, not prompt processing: its expert traffic
        // belongs in the decode column of the telemetry, or the MTP round's
        // cost lands in a bucket nobody reads during decode.
        model.telemetry.beginPhase(.decode, step: startPosition)
        let expertsBefore = PrefillHostProfile.isEnabled
            ? model.telemetry.snapshot().decode : ExpertPhaseCounters()
        let verifyStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        routerPreviewProbe.beginBlock()
        defer {
            verifyBlockNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - verifyStart
            verifyBlockCount += 1
        }
        try await executePrefillChunk(tokens: tokens,
                                      startPosition: startPosition,
                                      outputMode: outputMode,
                                      logits: logitRows,
                                      scratch: scratch,
                                      config: prefillRuntimeConfig,
                                      head: .allRows,
                                      greedyRows: greedyTokens,
                                      speculativeBlock: true)
        // A verify block is one chunk, so the profile has to be flushed here or
        // it accumulates across rounds with nothing to print it.
        if PrefillGPUProfile.isEnabled {
            FileHandle.standardError.write(Data((prefillGPUProfile.summary() + "\n").utf8))
            prefillGPUProfile.reset()
        }
        if PrefillHostProfile.isEnabled {
            let after = model.telemetry.snapshot().decode
            prefillHostProfile.recordExperts(
                experts: after.experts - expertsBefore.experts,
                misses: after.misses - expertsBefore.misses,
                ioNanos: after.fetchNanos - expertsBefore.fetchNanos)
            FileHandle.standardError.write(Data((prefillHostProfile.summary + "\n").utf8))
            prefillHostProfile.reset()
        }
        if let line = routerPreviewProbe.summary {
            FileHandle.standardError.write(Data((line + "\n").utf8))
        }
    }

    /// 計器用のバッファ。距離ごとに ID と重みを 1 本ずつ、`maxTokens * topK`
    /// ぶん。8 × 8 × 4 B = 256 B なので、確保しても測定の邪魔にならない。
    private func ensureRouterPreviewBuffers(topK: Int) throws {
        guard routerPreviewIDs.count < RouterPreviewProbe.maxDistance else { return }
        let elements = SpeculativeBlock.maxTokens * topK
        while routerPreviewIDs.count < RouterPreviewProbe.maxDistance {
            guard let ids = ctx.device.makeBuffer(
                    length: elements * MemoryLayout<UInt32>.stride,
                    options: .storageModeShared),
                  let weights = ctx.device.makeBuffer(
                    length: elements * MemoryLayout<Float16>.stride,
                    options: .storageModeShared)
            else { throw ModelError.residentBufferWrapFailed }
            ids.label = "prefill.routerPreviewIDs"
            weights.label = "prefill.routerPreviewWeights"
            routerPreviewIDs.append(ids)
            routerPreviewWeights.append(weights)
        }
    }

    /// Drop every KV row at or after `position` (`docs/mtp/03-DESIGN.md` D4).
    public func rewind(to position: Int) throws {
        try prefillChunkState.requireClean(operation: "rewind")
        guard let kv else {
            throw SpeculativeVerifyError.cursorMismatch("rewind requires FP16 KV")
        }
        guard position >= 0, position <= kv.position else {
            throw SpeculativeVerifyError.cursorMismatch(
                "rewind target \(position) is outside [0, \(kv.position)]")
        }
        kv.rewind(to: position)
    }

    // MARK: - SpeculativeDrafting

    public var isDraftInstalled: Bool { model.hasDraft }
    public var speculativeHiddenRowStride: Int { cfg.hiddenSize }
    public var draftStepCount: Int { speculativeDrafter?.steps ?? 0 }
    public var draftNanos: UInt64 { speculativeDrafter?.nanos ?? 0 }

    public func draftProposals(bonusToken: Int32,
                               position: Int,
                               hidden: MTLBuffer,
                               hiddenRow: Int,
                               count: Int) throws -> [Int32] {
        guard let kv else {
            throw SpeculativeVerifyError.cursorMismatch("drafting requires FP16 KV")
        }
        guard kv.position == position else {
            throw SpeculativeVerifyError.cursorMismatch(
                "draft cursor \(kv.position) != bonus position \(position)")
        }
        let drafter = try ensureSpeculativeDrafter()
        return try drafter.propose(bonusToken: bonusToken,
                                   position: position,
                                   hidden: hidden,
                                   hiddenRow: hiddenRow,
                                   count: count,
                                   kv: kv)
    }

    /// Built on the first speculative round, so a `--draft-block-size 0` run
    /// never opens the drafter file (D1's lazy load; the acceptance-probe path
    /// above does the same thing for the same reason).
    private func ensureSpeculativeDrafter() throws -> SpeculativeDrafter {
        if let speculativeDrafter { return speculativeDrafter }
        guard model.hasDraft else {
            throw DraftError.notInstalled(path: model.directoryURL.path)
        }
        let weights = try model.draftWeights()
        try weights.config.crossCheck(against: cfg)
        // The drafter is packed at a different affine group size than the
        // target (64 vs 32 for the pinned pair), and that size is baked into
        // the shader library, so it needs a context of its own.
        let draftContext: MetalContext
        if weights.config.quantGroupSize == ctx.affineGroupSize {
            draftContext = ctx
        } else {
            draftContext = try MetalContext(sharingDeviceWith: ctx)
            try draftContext.setAffineGroupSize(weights.config.quantGroupSize)
        }
        let drafter = try SpeculativeDrafter(context: ctx,
                                             draftContext: draftContext,
                                             weights: weights,
                                             embedTable: model.embedding,
                                             backboneHiddenSize: cfg.hiddenSize)
        speculativeDrafter = drafter
        return drafter
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               vision: VisionPrefillInput? = nil,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        let scratch = try ensurePrefillScratch(config: config)
        // The tower runs to completion before the first chunk is encoded: it is
        // one 3.5 TFLOP pass per image and it does not belong inside the chunk
        // loop, where it would run again for every chunk that touched the image
        // (PLAN_VISION §4-4).
        let images = try runVisionTower(vision: vision, chunkTokens: config.chunkTokens,
                                        tokenCount: tokens.count)
        let imageSpans = vision?.spans ?? []
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              chunkTokens: config.chunkTokens,
                                              imageSpans: imageSpans)
        for (spanIndex, span) in spans.enumerated() {
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            model.telemetry.beginPhase(.prefill, step: span.startPosition)
            try await executePrefillChunk(
                tokens: tokens[lower..<upper],
                startPosition: span.startPosition,
                outputMode: outputMode,
                logits: logits,
                scratch: scratch,
                config: config,
                vision: VisionChunkPlan(chunkTokenOffset: span.tokenOffset,
                                        tokenCount: span.tokenCount,
                                        promptStartPosition: startPosition,
                                        spans: imageSpans,
                                        softTokens: images),
                head: spanIndex == spans.count - 1 ? .finalRow : .none)
            onProgress(span.completedCount)
        }
        if PrefillGPUProfile.isEnabled {
            FileHandle.standardError.write(Data((prefillGPUProfile.summary() + "\n").utf8))
            prefillGPUProfile.reset()
        }
        if PrefillHostProfile.isEnabled {
            FileHandle.standardError.write(Data((prefillHostProfile.summary + "\n").utf8))
            prefillHostProfile.reset()
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    /// What one prefill chunk has to do about images: which of its rows are
    /// soft tokens, and which of its queries attend bidirectionally.
    ///
    /// A chunk boundary never falls inside an image span (`PrefillChunkPlanner`
    /// arranges that), so an image is either wholly inside this chunk or wholly
    /// outside it. The intersection is still computed rather than assumed —
    /// this is the code that would write into the wrong rows if that invariant
    /// ever broke, and clamping makes it write nothing instead.
    private struct VisionChunkPlan {
        let chunkTokenOffset: Int
        let tokenCount: Int
        let promptStartPosition: Int
        let spans: [VisionImageSpan]
        let softTokens: [VisionTowerOutput]

        static var none: VisionChunkPlan {
            VisionChunkPlan(chunkTokenOffset: 0,
                            tokenCount: 0,
                            promptStartPosition: 0,
                            spans: [],
                            softTokens: [])
        }

        struct Scatter {
            let soft: MTLBuffer
            let hiddenRow: Int
            let softRow: Int
            let rowCount: Int
        }

        var scatters: [Scatter] {
            var out: [Scatter] = []
            let chunkEnd = chunkTokenOffset + tokenCount
            for (index, span) in spans.enumerated() where index < softTokens.count {
                let first = Swift.max(span.tokenOffset, chunkTokenOffset)
                let last = Swift.min(span.tokenEnd, chunkEnd)
                guard first < last else { continue }
                out.append(Scatter(soft: softTokens[index].softTokens,
                                   hiddenRow: first - chunkTokenOffset,
                                   softRow: first - span.tokenOffset,
                                   rowCount: last - first))
            }
            return out
        }

        /// Exclusive visible end per query row, or nil when no image touches
        /// this chunk — in which case the mask stays off and the attention
        /// kernels take the path they took before vision existed.
        var spanEnds: [UInt32]? {
            let chunkEnd = chunkTokenOffset + tokenCount
            var ends = [UInt32](repeating: 0, count: tokenCount)
            var touched = false
            for span in spans {
                let first = Swift.max(span.tokenOffset, chunkTokenOffset)
                let last = Swift.min(span.tokenEnd, chunkEnd)
                guard first < last else { continue }
                touched = true
                let end = UInt32(promptStartPosition + last)
                for row in first..<last {
                    ends[row - chunkTokenOffset] = end
                }
            }
            return touched ? ends : nil
        }
    }

    /// Turn the attached images into soft tokens, once, before any chunk runs.
    private func runVisionTower(vision: VisionPrefillInput?,
                                chunkTokens: Int,
                                tokenCount: Int) throws -> [VisionTowerOutput] {
        guard let vision, !vision.isEmpty else { return [] }
        if let reason = PrefillChunkPlanner.imageSpanRejection(imageSpans: vision.spans,
                                                              tokenCount: tokenCount,
                                                              chunkTokens: chunkTokens) {
            throw PrefillError.chunkedUnsupported(reason)
        }
        // Opening the future direction is only equivalent to full bidirectional
        // visibility while the span fits inside the sliding window; past that,
        // the window's lower edge would cut the span's own head off (§2-7).
        if let longest = vision.spans.map(\.tokenCount).max(), longest > cfg.slidingWindow {
            throw PrefillError.chunkedUnsupported(
                "image span of \(longest) tokens exceeds the \(cfg.slidingWindow)-token "
                + "sliding window, so it cannot be made bidirectional")
        }

        let tower = try ensureVisionTower()
        var outputs: [VisionTowerOutput] = []
        outputs.reserveCapacity(vision.images.count)
        for image in vision.images {
            guard tower.accepts(image.geometry) else {
                throw VisionError.patchBudgetExceeded(width: image.geometry.targetWidth,
                                                      height: image.geometry.targetHeight,
                                                      patches: image.geometry.patchCount,
                                                      budget: tower.maxPatchCount)
            }
            let byteCount = image.patches.count * MemoryLayout<Float16>.size
            guard let pixels = image.patches.withUnsafeBytes({ raw in
                ctx.device.makeBuffer(bytes: raw.baseAddress!,
                                      length: byteCount,
                                      options: .storageModeShared)
            }) else {
                throw ModelError.residentBufferWrapFailed
            }
            pixels.label = "vision.pixels"
            guard let cb = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            let start = Date()
            let output = try tower.encode(commandBuffer: cb,
                                          pixels: pixels,
                                          geometry: image.geometry)
            cb.commit()
            waitUntilCompleted(cb)
            try checkCommandBufferError(cb.error)
            visionTowerSeconds += Date().timeIntervalSince(start)
            visionTowerImages += 1
            outputs.append(output)
        }
        return outputs
    }

    private func ensureVisionTower() throws -> VisionTower {
        if let visionTower { return visionTower }
        guard model.hasVisionTower else {
            throw VisionError.towerNotInstalled(path: model.directoryURL.path)
        }
        let start = Date()
        let weights = try model.visionWeights()
        let tower = try VisionTower(context: ctx, weights: weights)
        let scatter = try VisionSoftTokenScatter(context: ctx)
        visionLoadSeconds += Date().timeIntervalSince(start)
        visionTower = tower
        visionScatter = scatter
        return tower
    }

    /// Built on the first decode step of a probing run — never during a plain
    /// run, and never before the drafter is actually needed (D1's lazy load).
    private func ensureDraftProbe() throws -> DraftAcceptanceProbe? {
        guard let settings = draftProbeSettings else { return nil }
        if let draftProbe { return draftProbe }
        guard model.hasDraft else {
            throw DraftError.notInstalled(path: model.directoryURL.path)
        }
        let weights = try model.draftWeights()
        try weights.config.crossCheck(against: cfg)
        // The drafter is packed at a different affine group size than the
        // target (64 vs 32 for the pinned pair), and that size is baked into
        // the shader library, so it needs a context of its own.
        let draftContext: MetalContext
        if weights.config.quantGroupSize == ctx.affineGroupSize {
            draftContext = ctx
        } else {
            draftContext = try MetalContext(sharingDeviceWith: ctx)
            try draftContext.setAffineGroupSize(weights.config.quantGroupSize)
        }
        let probe = try DraftAcceptanceProbe(context: ctx,
                                             draftContext: draftContext,
                                             weights: weights,
                                             embedTable: model.embedding,
                                             backboneHiddenSize: cfg.hiddenSize,
                                             path: settings.path,
                                             drafts: settings.drafts,
                                             mode: settings.mode)
        draftProbe = probe
        return probe
    }

    @discardableResult
    private func ensurePrefillScratch(config: PrefillRuntimeConfig) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg, runtime: config)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)
        prefillScratch = scratch
        return scratch
    }

    /// Which rows of a chunk the lm_head runs on.
    private enum PrefillHeadRows {
        case none
        case finalRow
        case allRows
    }

    /// `speculativeBlock` is true only for `verifyBlock`. The M4.5 k-row paths
    /// (`docs/mtp/16-M4.5-PLAN.md` §4) are a third route through this function,
    /// and prompt prefill — whose chunks are wide, and whose speed and numbers
    /// are measured somewhere else entirely — keeps the path it had.
    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     vision: VisionChunkPlan = .none,
                                     head: PrefillHeadRows,
                                     greedyRows: MTLBuffer? = nil,
                                     speculativeBlock: Bool = false) async throws {
        guard !tokens.isEmpty else { return }
        guard kv != nil else {
            throw PrefillError.chunkedUnsupported("chunked prefill attention requires FP16 KV")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard startPosition >= 0, startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)")
        }
        if let kv, kv.fp16RingEnabled, let ringLayer = (0..<cfg.numLayers).first(where: {
            kv.ringCapacity(layer: $0) > 0
        }) {
            let requiredCapacity = min(maxContext, cfg.slidingWindow + config.chunkTokens)
            let ringCapacity = kv.ringCapacity(layer: ringLayer)
            guard requiredCapacity <= ringCapacity else {
                throw PrefillError.chunkedUnsupported(
                    "FP16 KV ring capacity \(ringCapacity) cannot hold required capacity \(requiredCapacity) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and prefillChunkTokens \(config.chunkTokens)")
            }
        }

        struct LayerPrefillQKVViews {
            let inputNorm: TensorView
            let q: TensorView
            let k: TensorView
            let v: TensorView
            let o: TensorView
            let postAttention: TensorView
            let preFFN: TensorView
            let preFFN2: TensorView
            let postFFN2: TensorView
            let postFFN: TensorView
            let layerScalar: TensorView
            let qNorm: TensorView
            let kNorm: TensorView
            let router: TensorView
            let routerPerExpertScale: TensorView
        }

        let layerViews = try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            return LayerPrefillQKVViews(
                inputNorm: try model.inputNorm(layer: L),
                q: try model.qProj(layer: L),
                k: try model.kProj(layer: L),
                v: isFull ? (try model.kProj(layer: L)) : (try model.vProj(layer: L)),
                o: try model.oProj(layer: L),
                postAttention: try model.postAttnNorm(layer: L),
                preFFN: try model.preFFN(layer: L),
                preFFN2: try model.preFFN2(layer: L),
                postFFN2: try model.postFFN2(layer: L),
                postFFN: try model.postFFN(layer: L),
                layerScalar: try model.layerScalar(layer: L),
                qNorm: try model.qNorm(layer: L),
                kNorm: try model.kNorm(layer: L),
                router: try model.router(layer: L),
                routerPerExpertScale: try model.routerPerExpertScale(layer: L))
        }

        let tokenIDs = tokens.map { UInt32(bitPattern: $0) }
        guard let tokenBuffer = ctx.device.makeBuffer(bytes: tokenIDs,
                                                      length: tokenIDs.count * MemoryLayout<UInt32>.stride,
                                                      options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let sqrtHidden = Float(D).squareRoot()
        let t = tokens.count
        let emb = model.embedding

        func encodeInt4Projection(commandBuffer: MTLCommandBuffer,
                                  family: PrefillProjectionFamily,
                                  weights: TensorView,
                                  x: MTLBuffer,
                                  y: MTLBuffer,
                                  rows: Int,
                                  columns: Int,
                                  tokenCount: Int,
                                  xStrideElements: Int,
                                  yStrideElements: Int) throws {
            if tokenCount >= 32,
               family == .q || family == .kv || family == .o,
               let candidate = prefillMPPAffineInt4 {
                let path = candidate.encode(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: tokenCount,
                    n: rows,
                    k: columns)
                if path == .affineThreadgroupF16 {
                    return
                }
            }
            if PrefillProjectionDispatchPolicy.selectedDispatch(
                for: family,
                chunkTokens: tokenCount,
                tiledQMM: prefillQMM.usesSimdgroupPath(k: columns)) == .qmm {
                prefillQMM.encode(commandBuffer: commandBuffer,
                                  weights: weights.buffer,
                                  weightsOffset: Int(weights.offset),
                                  scales: weights.buffer,
                                  scalesOffset: Int(weights.scaleOffset),
                                  biases: weights.buffer,
                                  biasesOffset: Int(weights.biasOffset),
                                  x: x,
                                  y: y,
                                  t: tokenCount,
                                  n: rows,
                                  k: columns)
                return
            }
            // A block narrow enough for the k-row GEMV reads the weights once
            // for all of its rows. The per-element arithmetic is the same as
            // the loop below, so this is a bandwidth change and not a numeric
            // one (docs/mtp/16-M4.5-PLAN.md §4 c).
            if Self.rowsDenseEnabled, tokenCount >= 1,
               tokenCount <= DequantInt4GEMV.maxRows {
                try int4.encodeRowsWide(commandBuffer: commandBuffer,
                                        weights: weights.buffer,
                                        weightsOffset: Int(weights.offset),
                                        scales: weights.buffer,
                                        scalesOffset: Int(weights.scaleOffset),
                                        biases: weights.buffer,
                                        biasesOffset: Int(weights.biasOffset),
                                        x: x,
                                        xStrideElements: xStrideElements,
                                        y: y,
                                        yStrideElements: yStrideElements,
                                        t: tokenCount,
                                        m: UInt32(rows),
                                        n: UInt32(columns))
                return
            }
            for row in 0..<tokenCount {
                int4.encode(commandBuffer: commandBuffer,
                            weights: weights.buffer,
                            weightsOffset: Int(weights.offset),
                            scales: weights.buffer,
                            scalesOffset: Int(weights.scaleOffset),
                            biases: weights.buffer,
                            biasesOffset: Int(weights.biasOffset),
                            x: x,
                            xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                            y: y,
                            yOffset: row * yStrideElements * MemoryLayout<Float16>.stride,
                            m: UInt32(rows),
                            n: UInt32(columns))
            }
        }

        func copyPrefillKV(commandBuffer: MTLCommandBuffer,
                           source: MTLBuffer,
                           destination: (buffer: MTLBuffer, offset: Int, stride: Int),
                           sourceTokenOffset: Int,
                           tokenCount: Int,
                           bytesPerToken: Int) throws {
            guard tokenCount > 0 else { return }
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(from: source,
                      sourceOffset: sourceTokenOffset * bytesPerToken,
                      to: destination.buffer,
                      destinationOffset: destination.offset,
                      size: tokenCount * bytesPerToken)
            blit.endEncoding()
        }

        func copyPrefillKVToCache(commandBuffer: MTLCommandBuffer,
                                  kv: KVCacheManager,
                                  layer: Int,
                                  startPosition: Int,
                                  tokenCount: Int,
                                  keySource: MTLBuffer,
                                  valueSource: MTLBuffer,
                                  bytesPerToken: Int) throws {
            let capacity = kv.capacity(layer: layer)
            let physicalStart = startPosition % capacity
            let firstSpan = min(tokenCount, capacity - physicalStart)
            let keyFirst = kv.kRange(layer: layer, start: startPosition, count: firstSpan)
            let valueFirst = kv.vRange(layer: layer, start: startPosition, count: firstSpan)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keyFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            guard firstSpan < tokenCount else { return }

            let secondCount = tokenCount - firstSpan
            let secondStart = startPosition + firstSpan
            let keySecond = kv.kRange(layer: layer, start: secondStart, count: secondCount)
            let valueSecond = kv.vRange(layer: layer, start: secondStart, count: secondCount)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keySecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueSecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
        }

        let callStart = PrefillHostProfile.mark()
        defer { prefillHostProfile.addCall(since: callStart) }
        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: tokens.count)

        // A speculative verify block runs its routed branch as one command
        // buffer per layer and waits for it only when the next layer needs the
        // expert slots back. At k=4 the per-tile pipeline spent more of the
        // block on submissions than on the GPU (docs/mtp/16-M4.5-PLAN.md §4 d).
        let blockPipeline = Self.blockPipelineEnabled && speculativeBlock
            && t <= SpeculativeBlock.maxTokens
        struct PendingBlockTile {
            let commandBuffer: MTLCommandBuffer
            let fetch: PrefillStreamedTileFetchResult
            let argumentBuffer: PrefillStreamedTileArgumentBuffer
        }
        struct PendingBlockLayer {
            let commandBuffer: MTLCommandBuffer
            let sharedCommandBuffer: MTLCommandBuffer?
            let metadata: PrefillGroupedRoutedMoEStreamedMetadataBuffers
            let tiles: [PendingBlockTile]
        }
        var pendingBlockLayer: PendingBlockLayer?
        /// Waits for the previous layer's routed buffer, which is what makes
        /// its expert slots reusable. Holding the bindings until here is what
        /// keeps the fetched blobs alive while the GPU reads them.
        func drainPendingBlockLayer() throws {
            guard let pending = pendingBlockLayer else { return }
            pendingBlockLayer = nil
            try withExtendedLifetime((pending.metadata, pending.tiles)) {
                for tile in pending.tiles {
                    try waitProfiled(.moe, tile.commandBuffer)
                }
                if let shared = pending.sharedCommandBuffer {
                    try waitProfiled(.shared, shared)
                }
                try waitProfiled(.tail, pending.commandBuffer)
            }
        }

        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }

        /// Detail mode only: close `cb` and open a fresh one, so the work
        /// encoded since the previous cut gets its own GPU timestamps. The
        /// segments are committed as they are cut and collected at the end of
        /// the layer, so nothing blocks the CPU mid-layer.
        var profiledSegments: [(PrefillGPUProfile.Detail, MTLCommandBuffer)] = []
        func cutProfiled(_ detail: PrefillGPUProfile.Detail) throws {
            guard PrefillGPUProfile.isDetailed else { return }
            cb.commit()
            profiledSegments.append((detail, cb))
            guard let next = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            cb = next
        }
        func flushProfiledSegments() throws {
            for (detail, segment) in profiledSegments {
                try waitProfiled(.attention, segment, detail: detail)
            }
            profiledSegments.removeAll(keepingCapacity: true)
        }

        prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer,
                            tableOffset: Int(emb.offset),
                            scales: emb.buffer,
                            scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer,
                            biasesOffset: Int(emb.biasOffset),
                            tokens: tokenBuffer,
                            out: scratch.hidden,
                            t: UInt32(t),
                            d: UInt32(D),
                            outScale: sqrtHidden)
        // Image rows are replaced, not scaled: `outScale` above is the text
        // embedding's sqrt(hidden), and upstream does not apply it to soft
        // tokens (PLAN_VISION §2-6).
        for scatter in vision.scatters {
            guard let visionScatter else {
                throw PrefillError.chunkedUnsupported(
                    "a prefill chunk carries image rows but no vision scatter kernel")
            }
            visionScatter.encode(commandBuffer: cb,
                                 hidden: scratch.hidden,
                                 soft: scatter.soft,
                                 d: D,
                                 hiddenRow: scatter.hiddenRow,
                                 softRow: scatter.softRow,
                                 rowCount: scatter.rowCount,
                                 hiddenStrideElements: D,
                                 scale: VisionPrefillFault.selected == .softTokensScaled
                                     ? sqrtHidden : 1.0)
        }
        try cutProfiled(.embed)

        // One `[queryCount]` upload per chunk that holds an image, and none at
        // all otherwise (`spanEnds` is nil then, and every attention dispatch
        // below takes the same path it took before vision existed).
        let spanEnds = VisionPrefillFault.selected == .causalOnly ? nil : vision.spanEnds
        let spanEndBuffer: MTLBuffer? = try spanEnds.map { ends in
            guard let buffer = ctx.device.makeBuffer(
                bytes: ends,
                length: ends.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = "prefill.spanEnd"
            return buffer
        }

        for L in 0..<cfg.numLayers {
            let frontStart = PrefillHostProfile.mark()
            model.beginOpeningRoutedExpertStreamer(layer: L)
            let views = layerViews[L]
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVHeads = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim = cfg.numHeads * headDim
            let kvDim = numKVHeads * headDim

            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: views.inputNorm.buffer,
                                   weightOffset: Int(views.inputNorm.offset),
                                   out: scratch.normed,
                                   t: UInt32(t),
                                   d: UInt32(D),
                                   eps: eps)
            try cutProfiled(.norm)
            try encodeInt4Projection(commandBuffer: cb,
                                 family: .q,
                                 weights: views.q,
                                 x: scratch.normed,
                                 y: scratch.q,
                                 rows: qDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: qDim)
            try encodeInt4Projection(commandBuffer: cb,
                                 family: .kv,
                                 weights: views.k,
                                 x: scratch.normed,
                                 y: scratch.kStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)
            try encodeInt4Projection(commandBuffer: cb,
                                 family: .kv,
                                 weights: views.v,
                                 x: scratch.normed,
                                 y: scratch.vStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)
            try cutProfiled(.qkvProjection)

            let rotatedPairs = isFull
                ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                : UInt32(headDim / 2)
            prefillQKVEpilogue.encode(commandBuffer: cb,
                                       q: scratch.q,
                                       k: scratch.kStage,
                                       v: scratch.vStage,
                                       qWeight: views.qNorm.buffer,
                                       qWeightOffset: Int(views.qNorm.offset),
                                       kWeight: views.kNorm.buffer,
                                       kWeightOffset: Int(views.kNorm.offset),
                                       startPosition: UInt32(startPosition),
                                       queryCount: UInt32(t),
                                       headDim: UInt32(headDim),
                                       numQHeads: UInt32(cfg.numHeads),
                                       numKVHeads: UInt32(numKVHeads),
                                       qTokenStrideElements: UInt32(qDim),
                                       kvTokenStrideElements: UInt32(kvDim),
                                       theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                       rotatedPairs: rotatedPairs,
                                       eps: eps)
            try cutProfiled(.rope)

            if let kv {
                let bytes = t * kvDim * MemoryLayout<Float16>.stride
                try copyPrefillKVToCache(commandBuffer: cb,
                                         kv: kv,
                                         layer: L,
                                         startPosition: startPosition,
                                         tokenCount: t,
                                         keySource: scratch.kStage,
                                         valueSource: scratch.vStage,
                                         bytesPerToken: bytes / t)
            }
            try cutProfiled(.kvCopy)
            // Upstream ORs the image mask into the *sliding* layers only and
            // leaves the five full-attention layers causal (PLAN_VISION §2-7).
            // That is upstream's shape, possibly its bug, and it is what the
            // reference we compare against does.
            let layerSpanEnd = isFull ? nil : spanEndBuffer
            let params = PrefillAttentionParams(
                    startPosition: UInt32(startPosition),
                    queryCount: UInt32(t),
                    headDim: UInt32(headDim),
                    numQHeads: UInt32(cfg.numHeads),
                    numKVHeads: UInt32(numKVHeads),
                    kvValidCount: UInt32(startPosition + t),
                    slidingWindow: isFull ? UInt32(startPosition + t) : UInt32(cfg.slidingWindow),
                    kvTokenStrideElements: UInt32(kvDim),
                    qTokenStrideElements: UInt32(qDim),
                    oTokenStrideElements: UInt32(qDim),
                    scale: 1.0,
                    spanMaskEnabled: layerSpanEnd != nil)
            if let kv {
                    let keyBuffer = kv.keyBuffer(layer: L, validTokenCount: startPosition + t)
                    let valueBuffer = kv.valueBuffer(layer: L, validTokenCount: startPosition + t)
                    let ringCapacity = kv.ringCapacity(layer: L)
                    let activeRingCapacity = ringCapacity > 0 && startPosition + t > ringCapacity
                        ? UInt32(ringCapacity)
                        : 0
                    // A k-token block is a decode step's shape, not a prompt's.
                    // The query-blocked prefill kernel gives one simdgroup four
                    // queries and never splits the key range, so at k <= 8 it
                    // runs 16 threadgroups with one live simdgroup each and the
                    // GPU sits at a few percent occupancy: 15 ms a block, more
                    // than plain decode spends on its whole forward
                    // (docs/mtp/24-M5.5-RESULTS.md §2). Decode's split-KV
                    // kernel has the parallelism the shape needs, so the block
                    // takes it a row at a time — the same trade the router made
                    // in M4.5, and for the same second reason: it puts the
                    // block on decode's kernel instead of near it.
                    if Self.rowsAttentionEnabled, speculativeBlock, layerSpanEnd == nil,
                       t <= SpeculativeBlock.maxTokens {
                        // One dispatch pair for the whole block: the rows
                        // kernel gives each row its own SIMD group, so the KV
                        // range is walked once instead of once per row
                        // (24-M5.5 §7-1). A one-row block stays on the decode
                        // call itself — same kernel, same numbers as decode.
                        if Self.rowsSharedAttentionEnabled, t > 1 {
                            let activeRing = ringCapacity > 0 && startPosition + t > ringCapacity
                                ? UInt32(ringCapacity)
                                : 0
                            attention.encodeRows(commandBuffer: cb,
                                                 q: scratch.q,
                                                 k: keyBuffer,
                                                 v: valueBuffer,
                                                 out: scratch.attentionOutput,
                                                 headDim: UInt32(headDim),
                                                 numQHeads: UInt32(cfg.numHeads),
                                                 numKVHeads: UInt32(numKVHeads),
                                                 rows: t,
                                                 startPosition: startPosition,
                                                 window: isFull ? 0 : UInt32(cfg.slidingWindow),
                                                 scale: 1.0,
                                                 ringCapacity: isFull ? 0 : activeRing)
                        } else {
                            let rowStride = qDim * MemoryLayout<Float16>.stride
                            for row in 0..<t {
                                let seqLen = UInt32(startPosition + row + 1)
                                if isFull {
                                    attention.encodeFull(commandBuffer: cb,
                                                         q: scratch.q, qOffset: row * rowStride,
                                                         k: keyBuffer, kOffset: 0,
                                                         v: valueBuffer, vOffset: 0,
                                                         out: scratch.attentionOutput,
                                                         outOffset: row * rowStride,
                                                         headDim: UInt32(headDim),
                                                         numQHeads: UInt32(cfg.numHeads),
                                                         numKVHeads: UInt32(numKVHeads),
                                                         seqLen: seqLen,
                                                         scale: 1.0)
                                } else {
                                    let rowRing = ringCapacity > 0 && Int(seqLen) > ringCapacity
                                        ? UInt32(ringCapacity)
                                        : 0
                                    attention.encodeSWA(commandBuffer: cb,
                                                        q: scratch.q, qOffset: row * rowStride,
                                                        k: keyBuffer, kOffset: 0,
                                                        v: valueBuffer, vOffset: 0,
                                                        out: scratch.attentionOutput,
                                                        outOffset: row * rowStride,
                                                        headDim: UInt32(headDim),
                                                        numQHeads: UInt32(cfg.numHeads),
                                                        numKVHeads: UInt32(numKVHeads),
                                                        seqLen: seqLen,
                                                        window: UInt32(cfg.slidingWindow),
                                                        scale: 1.0,
                                                        ringCapacity: rowRing)
                                }
                            }
                        }
                    } else {
                        prefillAttention.encodeCausal(commandBuffer: cb,
                                                      q: scratch.q,
                                                      k: keyBuffer,
                                                      v: valueBuffer,
                                                      out: scratch.attentionOutput,
                                                      params: params,
                                                      spanEnd: layerSpanEnd,
                                                      kvRingCapacity: activeRingCapacity,
                                                      path: prefillAttentionPath)
                    }
            } else {
                throw PrefillError.chunkedUnsupported(
                    "chunked prefill attention requires FP16 KV")
            }
            try cutProfiled(isFull ? .attentionFull : .attentionSWA)
            try encodeInt4Projection(commandBuffer: cb,
                                     family: .o,
                                     weights: views.o,
                                     x: scratch.attentionOutput,
                                     y: scratch.h1,
                                     rows: D,
                                     columns: qDim,
                                     tokenCount: t,
                                     xStrideElements: qDim,
                                     yStrideElements: D)
            try cutProfiled(.outputProjection)
            prefillPostAttention.encode(commandBuffer: cb,
                                            hidden: scratch.hidden,
                                            attn: scratch.h1,
                                            denseX: scratch.denseX,
                                            routedX: scratch.routedX,
                                            routerX: scratch.routerX,
                                            postAttentionWeight: views.postAttention.buffer,
                                            postAttentionWeightOffset: Int(views.postAttention.offset),
                                            preFFNWeight: views.preFFN.buffer,
                                            preFFNWeightOffset: Int(views.preFFN.offset),
                                            preFFN2Weight: views.preFFN2.buffer,
                                            preFFN2WeightOffset: Int(views.preFFN2.offset),
                                            queryCount: UInt32(t),
                                            d: UInt32(D),
                                            hiddenStrideElements: UInt32(D),
                                            attnStrideElements: UInt32(D),
                                            denseStrideElements: UInt32(D),
                                            routedStrideElements: UInt32(D),
                                            routerStrideElements: UInt32(D),
                                            eps: eps)
            // A block narrow enough for the row-at-a-time decode router takes
            // it. The router is 720 KB a layer, so reading it once per row
            // costs about a millisecond a block; what it buys is that the
            // block's expert choices are decode's choices. The two routers
            // reduce the same dot product in a different order, and a near-tie
            // between two experts resolves differently on either side — a
            // discrete difference no tolerance covers
            // (docs/mtp/16-M4.5-PLAN.md §4).
            if Self.rowsRouterEnabled, speculativeBlock, t <= SpeculativeBlock.maxTokens,
               moe.routerWeightBits == 16, cfg.topKExperts == 8 {
                // One dispatch pair for the block when the rows kernels are
                // there, the row loop when they are not. Both produce the same
                // logits: the rows kernel gives each row the threadgroup and
                // lane layout the single-row dispatch gave it.
                let folded = Self.blockRowsRouterFoldEnabled && moe.encodeRouterGemma4BF16Rows(
                    commandBuffer: cb,
                    weights: views.router.buffer,
                    weightsOffset: Int(views.router.offset),
                    hidden: scratch.routerX,
                    hiddenStrideElements: UInt32(D),
                    effectiveScale: effectiveScaleBuffers[L],
                    perExpertScale: views.routerPerExpertScale.buffer,
                    perExpertScaleOffset: Int(views.routerPerExpertScale.offset),
                    outIndices: scratch.routeIDs,
                    outWeights: scratch.routeWeights,
                    rowCount: t,
                    numExperts: UInt32(cfg.numExperts),
                    d: UInt32(D),
                    topK: UInt32(cfg.topKExperts))
                for row in 0..<(folded ? 0 : t) {
                    moe.encodeRouterGemma4BF16(
                        commandBuffer: cb,
                        weights: views.router.buffer,
                        weightsOffset: Int(views.router.offset),
                        hidden: scratch.routerX,
                        hiddenOffset: row * D * MemoryLayout<Float16>.stride,
                        effectiveScale: effectiveScaleBuffers[L],
                        perExpertScale: views.routerPerExpertScale.buffer,
                        perExpertScaleOffset: Int(views.routerPerExpertScale.offset),
                        outIndices: scratch.routeIDs,
                        outIndicesOffset: row * cfg.topKExperts * MemoryLayout<UInt32>.stride,
                        outWeights: scratch.routeWeights,
                        outWeightsOffset: row * cfg.topKExperts * MemoryLayout<Float16>.stride,
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts))
                }
            } else if prefillRouter.routerWeightBits == 16 {
                prefillRouter.encodeGemma4BF16Block(
                        commandBuffer: cb,
                        weights: views.router.buffer,
                        weightsOffset: Int(views.router.offset),
                        hidden: scratch.routerX,
                        effectiveScale: effectiveScaleBuffers[L],
                        perExpertScale: views.routerPerExpertScale.buffer,
                        perExpertScaleOffset: Int(views.routerPerExpertScale.offset),
                        outIndices: scratch.routeIDs,
                        outWeights: scratch.routeWeights,
                        queryCount: UInt32(t),
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D))
            } else {
                prefillRouter.encodeGemma4Block(
                        commandBuffer: cb,
                        weights: views.router.buffer,
                        weightsOffset: Int(views.router.offset),
                        scales: views.router.buffer,
                        scalesOffset: Int(views.router.scaleOffset),
                        biases: views.router.buffer,
                        biasesOffset: Int(views.router.biasOffset),
                        hidden: scratch.routerX,
                        effectiveScale: effectiveScaleBuffers[L],
                        perExpertScale: views.routerPerExpertScale.buffer,
                        perExpertScaleOffset: Int(views.routerPerExpertScale.offset),
                        outIndices: scratch.routeIDs,
                        outWeights: scratch.routeWeights,
                        queryCount: UInt32(t),
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D))
            }

            // 計器 (28-M8 §3 の果実 B): 層 L の router 入力に層 L+d の router
            // 重みを当てて、層 L+d が選ぶエキスパートを先に出しておく。同じ
            // command buffer の後ろに積むだけなので `routerLogits` の共用は
            // 直列で安全で、結果は数えるだけ — 本物のルーティングにも fetch
            // にも渡さないので出力は 1 バイトも動かない。
            if RouterPreviewProbe.isEnabled, speculativeBlock,
               t <= SpeculativeBlock.maxTokens,
               moe.routerWeightBits == 16, cfg.topKExperts == 8 {
                try ensureRouterPreviewBuffers(topK: cfg.topKExperts)
                for distance in 1...RouterPreviewProbe.maxDistance
                where L + distance < cfg.numLayers {
                    let ahead = layerViews[L + distance]
                    moe.encodeRouterGemma4BF16Rows(
                        commandBuffer: cb,
                        weights: ahead.router.buffer,
                        weightsOffset: Int(ahead.router.offset),
                        hidden: scratch.routerX,
                        hiddenStrideElements: UInt32(D),
                        effectiveScale: effectiveScaleBuffers[L + distance],
                        perExpertScale: ahead.routerPerExpertScale.buffer,
                        perExpertScaleOffset: Int(ahead.routerPerExpertScale.offset),
                        outIndices: routerPreviewIDs[distance - 1],
                        outWeights: routerPreviewWeights[distance - 1],
                        rowCount: t,
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts))
                }
            }

                    // The shared expert needs nothing from the router, so it
                    // is encoded and submitted before the host stops to read
                    // the routing back: it runs on the GPU while the host
                    // groups the pairs and the streamer preads the misses,
                    // instead of adding a round trip of its own
                    // (docs/mtp/16-M4.5-PLAN.md §4 d).
                    guard let sharedCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    let sharedProj = sharedExpertProjections[L]
                    try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                                        x: scratch.denseX,
                                                        y: scratch.h1,
                                                        gate: sharedProj.gate,
                                                        up: sharedProj.up,
                                                        down: sharedProj.down,
                                                        scratchGate: scratch.sharedGateScratch,
                                                        scratchUp: scratch.sharedUpScratch,
                                                        scratchAct: scratch.sharedActScratch,
                                                        queryCount: t,
                                                        d: D,
                                                        intermediate: cfg.intermediateSize,
                                                        xStrideElements: D,
                                                        yStrideElements: D,
                                                        rowsPath: speculativeBlock)
                    prefillRMS.encodeBF16W(commandBuffer: sharedCB,
                                           x: scratch.h1,
                                           weight: sharedProj.postF1.buffer,
                                           weightOffset: Int(sharedProj.postF1.offset),
                                           out: scratch.h1,
                                           t: UInt32(t),
                                           d: UInt32(D),
                                           eps: eps)
                    cb.commit()
                    sharedCB.commit()
                    prefillHostProfile.add(.encodeFront, since: frontStart)
                    let frontWaitStart = PrefillHostProfile.mark()
                    try waitProfiled(.attention, cb, detail: .post)
                    try flushProfiledSegments()
                    prefillHostProfile.add(.waitFront, since: frontWaitStart)
                    // The previous layer's routed buffer has to be done before
                    // this layer fetches an expert, or a fetch could land in a
                    // slot the GPU is still reading. Draining here rather than
                    // at the top of the layer lets it finish while this layer's
                    // attention runs.
                    let drainStart = PrefillHostProfile.mark()
                    try drainPendingBlockLayer()
                    prefillHostProfile.add(.drain, since: drainStart)

                    let routeStart = PrefillHostProfile.mark()
                    let routeCount = t * cfg.topKExperts
                    let idPtr = scratch.routeIDs.contents()
                        .bindMemory(to: UInt32.self, capacity: routeCount)
                    let weightPtr = scratch.routeWeights.contents()
                        .bindMemory(to: Float16.self, capacity: routeCount)
                    var routeIDs = [UInt32]()
                    routeIDs.reserveCapacity(routeCount)
                    var routeWeights = [Float16]()
                    routeWeights.reserveCapacity(routeCount)
                    for i in 0..<routeCount {
                        routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
                        routeWeights.append(weightPtr[i])
                    }
                    let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                                   weights: routeWeights,
                                                                   queryCount: t,
                                                                   topK: cfg.topKExperts)
                    // この層について前もって出しておいた予測を採点し、続けて
                    // 先の層の予測を控える。順番が要る: `compare` が層 L の
                    // 予測を消費してから、`record` が L+d 用を積む。
                    if RouterPreviewProbe.isEnabled, speculativeBlock {
                        let requested = Array(Set(pairs.map { Int($0.expert) })).sorted()
                        let resident = try model.routedExpertResidency(layer: L,
                                                                       experts: requested)
                        routerPreviewProbe.compare(layer: L,
                                                   actual: requested,
                                                   resident: resident)
                        let previewCount = t * cfg.topKExperts
                        for distance in 1...RouterPreviewProbe.maxDistance
                        where L + distance < cfg.numLayers
                            && distance <= routerPreviewIDs.count {
                            let ptr = routerPreviewIDs[distance - 1].contents()
                                .bindMemory(to: UInt32.self, capacity: previewCount)
                            let wPtr = routerPreviewWeights[distance - 1].contents()
                                .bindMemory(to: Float16.self, capacity: previewCount)
                            // 予測の強さ = その層の全行が付けた router 重みの和。
                            // 複数の行が同じエキスパートを選ぶほど確かになる。
                            var score = [Int: Float]()
                            for i in 0..<previewCount {
                                let expert = Int(min(ptr[i],
                                                     UInt32(cfg.numExperts - 1)))
                                score[expert, default: 0] += Float(wPtr[i])
                            }
                            let predicted = Set(score.keys)
                            // 先読みが実際に読むのは「予測のうちまだ載って
                            // いないもの」だけなので、その時点の常駐を引く。
                            let order = predicted.sorted()
                            let ahead = try model.routedExpertResidency(
                                layer: L + distance, experts: order)
                            let nonResident = zip(order, ahead)
                                .filter { !$0.1 }
                                .map(\.0)
                                .sorted { (score[$0] ?? 0, $1) > (score[$1] ?? 0, $0) }
                            routerPreviewProbe.record(fromLayer: L,
                                                      distance: distance,
                                                      predicted: predicted,
                                                      nonResident: nonResident)
                        }
                    }
                    let schedulerConfig = Self.prefillRoutedTileSchedulerConfig
                    let routeTileExpertCount: Int
                    if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
                        guard schedulerConfig.fitsSlotBudget(slotCount: slotCount) else {
                            throw PrefillError.chunkedUnsupported(
                                "prefill routed tile depth \(schedulerConfig.maxPendingDepth) with \(schedulerConfig.tileExperts) experts/tile needs \((schedulerConfig.maxPendingDepth + 1) * schedulerConfig.tileExperts) slots, has \(slotCount)")
                        }
                        routeTileExpertCount = min(schedulerConfig.tileExperts, slotCount)
                    } else {
                        routeTileExpertCount = schedulerConfig.tileExperts
                    }
                    // Tiles are cut out of the experts in this order, so the
                    // order decides which tile has to wait for a read. A
                    // speculative block sorts the experts it already holds to
                    // the front and cuts the tiles there: those tiles are pure
                    // hits, so the GPU can run them — about three quarters of
                    // the layer's routed work — while the misses are still
                    // landing. Without it every tile of the layer holds a miss
                    // or two and the GPU has nothing to do until the slowest
                    // read of the layer is in (docs/mtp/27-M7-RESULTS.md §4).
                    // Ties keep the physical order, so a tile's misses are
                    // still read in file order.
                    var sortKeys = model.routedExpertPhysicalOffsets(layer: L)
                    var residentGroupCount: Int?
                    if blockPipeline && Self.blockResidentFirstEnabled {
                        let requested = Array(Set(pairs.map { Int($0.expert) })).sorted()
                        let resident = try model.routedExpertResidency(layer: L,
                                                                       experts: requested)
                        for (index, expert) in requested.enumerated() where !resident[index] {
                            sortKeys[expert] |= Self.nonResidentSortBit
                        }
                        let residentCount = resident.filter { $0 }.count
                        // All resident or none: the cut would fall on a tile
                        // boundary anyway, and passing it would only pin the
                        // first tile to a partial width.
                        residentGroupCount = (residentCount > 0
                                              && residentCount < requested.count)
                            ? residentCount : nil
                    }
                    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                        pairs,
                        queryCount: t,
                        topK: cfg.topKExperts,
                        numExperts: cfg.numExperts,
                        tileExpertCount: routeTileExpertCount,
                        expertSortKeys: sortKeys,
                        tileBreakAfterGroup: residentGroupCount)
                    prefillHostProfile.add(.route, since: routeStart)

                    let expertSlotCount = model.routedExpertCacheSlotCount(layer: L)
                    // Every tile's experts are known as soon as the routing is
                    // grouped, so a block issues all of the layer's reads here
                    // and waits for each one only when its tile is encoded.
                    // The bytes then land while the GPU runs the tiles already
                    // submitted, which is where half of a block's wall clock
                    // used to go (docs/mtp/18-M4.6-RESULTS.md §3). Only when
                    // the layer's whole expert union fits the cache: otherwise
                    // a tile has to hand slots back mid-layer, and a read in
                    // flight could be writing into one of them.
                    let tileLookahead = Self.blockTileLookaheadEnabled && blockPipeline
                        && (expertSlotCount.map { routes.groups.count <= $0 } ?? false)
                    struct StartedTileFetch {
                        let tileIndex: Int
                        let expertIDs: [Int]
                        let plan: RoutedExpertFetchPlan
                        let handle: RoutedExpertFetchHandle
                    }
                    func startTileFetch(_ tileIndex: Int,
                                        avoiding: Set<Int>) throws -> StartedTileFetch? {
                        guard tileIndex < routes.tiles.count else { return nil }
                        let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                            forTile: tileIndex,
                            routes: routes)
                        guard let plan = try model.planRoutedExpertsIfPossible(
                            layer: L,
                            experts: expertIDs,
                            avoidingSlots: avoiding) else { return nil }
                        return StartedTileFetch(
                            tileIndex: tileIndex,
                            expertIDs: expertIDs,
                            plan: plan,
                            handle: try model.startRoutedExpertFetch(plan: plan))
                    }
                    /// One plan for the layer, one fetch per tile. Returns an
                    /// empty array when the layer cannot be placed in one call,
                    /// which puts the caller back on the per-tile loop.
                    func startLayerFetches(routes: PrefillMoEGroupedRoutes,
                                           layer: Int) throws -> [StartedTileFetch] {
                        var tileExperts: [[Int]] = []
                        tileExperts.reserveCapacity(routes.tiles.count)
                        for index in routes.tiles.indices {
                            tileExperts.append(try PrefillStreamedTileBinding.expertIDs(
                                forTile: index,
                                routes: routes))
                        }
                        let layerExperts = tileExperts.flatMap { $0 }
                        guard !layerExperts.isEmpty,
                              let layerPlan = try model.planRoutedExpertsIfPossible(
                                layer: layer,
                                experts: layerExperts)
                        else { return [] }
                        var started: [StartedTileFetch] = []
                        started.reserveCapacity(tileExperts.count)
                        var cursor = 0
                        for (index, experts) in tileExperts.enumerated() {
                            let plan = layerPlan.slice(cursor..<(cursor + experts.count))
                            cursor += experts.count
                            started.append(StartedTileFetch(
                                tileIndex: index,
                                expertIDs: experts,
                                plan: plan,
                                handle: try model.startRoutedExpertFetch(plan: plan)))
                        }
                        return started
                    }
                    // Every tile at once, not one ahead: the misses of a layer
                    // are not spread evenly over its tiles, so a single tile of
                    // lookahead leaves the layer's one expensive read with only
                    // one tile of GPU work to hide behind. Planned in order
                    // with an accumulating reservation, so no read lands in a
                    // slot another read of this layer is filling.
                    //
                    // Planned in one call for the whole layer, not once per
                    // tile: the tiles of a layer hold disjoint experts, so one
                    // plan places exactly the same slots — except that a
                    // per-tile plan does not know the experts of the tiles
                    // behind it and can evict one of them, which the layer then
                    // reads back from the file (docs/mtp/27-M7-RESULTS.md §3).
                    // The reads still go one handle per tile, so a tile whose
                    // experts are all resident can start while the others land.
                    var startedFetches: [StartedTileFetch] = []
                    if tileLookahead {
                        if Self.blockLayerPlanEnabled {
                            startedFetches = try startLayerFetches(routes: routes, layer: L)
                        }
                        var reservedSlots = Set<Int>(
                            startedFetches.flatMap(\.plan.assignedSlots))
                        for index in routes.tiles.indices where index >= startedFetches.count {
                            guard let started = try startTileFetch(index,
                                                                   avoiding: reservedSlots)
                            else { break }
                            reservedSlots.formUnion(started.plan.assignedSlots)
                            startedFetches.append(started)
                        }
                        // A layer whose union fits the cache always places, so
                        // this is the impossible case; drain what was issued
                        // rather than leave reads writing into slots the
                        // fallback path is about to reuse.
                        if startedFetches.count != routes.tiles.count {
                            for started in startedFetches { _ = try started.handle.wait() }
                            startedFetches.removeAll()
                        }
                    }

                    // Nothing on the host needs the shared expert's result —
                    // the layer tail reads `h1` on the GPU, and it is behind
                    // this buffer on the same queue. So a block does not stop
                    // for it here; it is checked with the rest of the layer at
                    // the next drain, and until then its dispatches are GPU
                    // work the expert reads below can hide behind.
                    if !blockPipeline || !tileLookahead {
                        let sharedWaitStart = PrefillHostProfile.mark()
                        try waitProfiled(.shared, sharedCB)
                        prefillHostProfile.add(.waitShared, since: sharedWaitStart)
                    }

                    let metaStart = PrefillHostProfile.mark()
                    let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
                        device: ctx.device,
                        routes: routes)
                    prefillHostProfile.add(.meta, since: metaStart)
                    let routedOffsets = model.routedExpertOffsets(layer: L)
                    // A speculative block is narrow enough that a layer's whole
                    // routed branch fits one command buffer: the tiles, the
                    // reduce and the tail go in together and nothing waits for
                    // them until the next layer needs the slots back. A prompt
                    // chunk keeps the per-tile pipeline, where a tile's GPU work
                    // is what hides the next tile's pread
                    // (docs/mtp/16-M4.5-PLAN.md §4 d).
                    var heldTileState: [PendingBlockTile] = []
                    var blockUsedSlots = Set<Int>()
                    /// A tile of this layer holds its slots until the layer is
                    /// drained, so a layer whose expert union does not fit the
                    /// cache has to hand some back mid-layer. That costs the
                    /// round trip the block path exists to avoid, so it happens
                    /// only when the cache is genuinely too small for the layer.
                    func flushBlockTiles() throws {
                        guard !heldTileState.isEmpty else { return }
                        let held = heldTileState
                        heldTileState.removeAll(keepingCapacity: true)
                        let waitStart = PrefillHostProfile.mark()
                        try withExtendedLifetime(held) {
                            for tile in held { try waitProfiled(.moe, tile.commandBuffer) }
                        }
                        prefillHostProfile.add(.waitTile, since: waitStart)
                        blockUsedSlots.removeAll(keepingCapacity: true)
                    }
                    struct PendingPrefillTile {
                        let tileIndex: Int
                        let commandBuffer: MTLCommandBuffer
                        let fetch: PrefillStreamedTileFetchResult
                        let argumentBuffer: PrefillStreamedTileArgumentBuffer
                    }
                    var pendingTiles: [PendingPrefillTile] = []
                    var tileLifetime = PrefillStreamedTileSlotLifetime()
                    func drainOldestPendingTile() throws {
                        guard !pendingTiles.isEmpty else { return }
                        let pending = pendingTiles.removeFirst()
                        let waitStart = PrefillHostProfile.mark()
                        try withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                            try waitProfiled(.moe, pending.commandBuffer)
                        }
                        prefillHostProfile.add(.waitTile, since: waitStart)
                        if !pending.fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.complete(tileIndex: pending.tileIndex)
                        }
                    }

                    let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
                    // Tiles are independent — each writes its own range of
                    // `routePartials`, and the reduce runs after all of them —
                    // so a layer is free to encode the tiles whose experts are
                    // already resident first. That is what gives the reads
                    // still in flight something to hide behind: without it the
                    // GPU waits for the layer's one expensive read before it
                    // has anything at all to run
                    // (docs/mtp/18-M4.6-RESULTS.md §3).
                    let tileOrder: [Int] = startedFetches.isEmpty
                        ? Array(routes.tiles.indices)
                        : routes.tiles.indices.sorted {
                            let lhs = startedFetches[$0].plan.misses.count
                            let rhs = startedFetches[$1].plan.misses.count
                            return lhs == rhs ? $0 < $1 : lhs < rhs
                        }
                    for tileIndex in tileOrder {
                        let tile = routes.tiles[tileIndex]
                        let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                            forTile: tileIndex,
                            routes: routes)
                        var plannedFetch: RoutedExpertFetchPlan?
                        if blockPipeline {
                            // Nothing of this layer has been submitted yet, so
                            // the only slots to keep clear are the ones this
                            // layer's earlier tiles were fetched into — and if
                            // that leaves the cache without room for this tile,
                            // the layer is cut here instead of failing to place
                            // a miss.
                            if let expertSlotCount,
                               blockUsedSlots.count + expertIDs.count > expertSlotCount {
                                try flushBlockTiles()
                            }
                        } else if !pendingTiles.isEmpty {
                            let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                            if !pendingAssignedSlots.isEmpty {
                                let pendingSlots = Set(pendingAssignedSlots)
                                let plan = try model.planRoutedExpertsIfPossible(
                                    layer: L,
                                    experts: expertIDs,
                                    avoidingSlots: pendingSlots)
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: pendingAssignedSlots,
                                        avoidingSlotPlanAvailable: plan != nil))
                                switch decision {
                                case .prefetchNext:
                                    guard let plan else {
                                        throw ModelError.indexCorrupt(
                                            detail: "routed tile scheduler requested missing plan")
                                    }
                                    plannedFetch = plan
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler ignored pending tile")
                                }
                            } else {
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: [],
                                        avoidingSlotPlanAvailable: false))
                                switch decision {
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending, .prefetchNext:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler failed to drain empty-slot pending tile")
                                }
                            }
                        } else {
                            let decision = routedTileScheduler.decide(
                                PrefillRoutedTileSchedulerInput(
                                    hasPendingTile: false,
                                    pendingAssignedSlots: [],
                                    avoidingSlotPlanAvailable: false))
                            switch decision {
                            case .issueWithoutPending:
                                break
                            case .prefetchNext, .drainBeforeIssue:
                                throw ModelError.indexCorrupt(
                                    detail: "routed tile scheduler requested pending action without pending tile")
                            }
                        }
                        let fetchStart = PrefillHostProfile.mark()
                        let fetch: PrefillStreamedTileFetchResult
                        let started = tileIndex < startedFetches.count
                            ? startedFetches[tileIndex] : nil
                        prefillHostProfile.recordTile(lookahead: started != nil)
                        if let started {
                            let views = try started.handle.wait()
                            fetch = PrefillStreamedTileFetchResult(
                                expertIDs: started.expertIDs,
                                binding: try PrefillStreamedTileBinding(
                                    expertIDs: started.expertIDs,
                                    views: views),
                                usedPlannedFetch: true,
                                plannedHits: started.plan.hits,
                                plannedMissIndices: started.plan.misses,
                                plannedAssignedSlots: started.plan.assignedSlots,
                                plannedMissSlots: started.plan.misses.map {
                                    started.plan.assignedSlots[$0]
                                })
                        } else {
                            fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                                model: model,
                                layer: L,
                                tileIndex: tileIndex,
                                routes: routes,
                                plannedFetch: plannedFetch,
                                avoidingSlots: blockPipeline
                                    ? blockUsedSlots
                                    : Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
                        }
                        try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                              pairStart: Int(tile.pairStart),
                                                              pairCount: Int(tile.pairCount))
                        prefillHostProfile.add(.fetch, since: fetchStart)
                        let moeEncodeStart = PrefillHostProfile.mark()
                        if !blockPipeline, !fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.begin(tileIndex: tileIndex,
                                                   plannedSlots: fetch.plannedMissSlots)
                        }
                        let argumentBuffer = blockPipeline
                            ? try prefillGroupedMoE.streamedArgumentBuffer(
                                device: ctx.device,
                                index: tileIndex,
                                binding: fetch.binding)
                            : try prefillGroupedMoE.makeStreamedArgumentBuffer(
                                device: ctx.device,
                                binding: fetch.binding)
                        let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                            pairStart: tile.pairStart,
                            pairCount: tile.pairCount,
                            d: UInt32(D),
                            routedIntermediate: UInt32(cfg.moeIntermediateSize),
                            topK: UInt32(cfg.topKExperts),
                            hiddenStrideElements: UInt32(D),
                            binding: fetch.binding,
                            offsets: routedOffsets)
                        guard let tileCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        let groupStart = Int(tile.groupStart)
                        let tileGroups = Array(
                            routes.groups[groupStart..<(groupStart + Int(tile.groupCount))])
                        // A block narrow enough that no expert holds more than
                        // eight rows takes the rows path: weights read once,
                        // decode's arithmetic per row. That is the speculative
                        // verify case, and it is the item the verify bill was
                        // mostly made of (docs/mtp/16-M4.5-PLAN.md §4 a).
                        let maxPairsPerExpert = tileGroups.map { Int($0.pairCount) }.max() ?? 0
                        let rowsBlocks = Self.rowsMoEEnabled && speculativeBlock
                            && prefillGroupedMoE.usesExpertRowsPath(
                                maxPairsPerExpert: maxPairsPerExpert)
                            ? prefillGroupedMoE.encodeStreamedRows(
                                commandBuffer: tileCB,
                                hidden: scratch.routedX,
                                sortedPairs: metadata.sortedPairs,
                                routePartials: scratch.routePartials,
                                gateUpActScratch: scratch.routedGateUpActScratch,
                                argumentBuffer: argumentBuffer,
                                binding: fetch.binding,
                                groups: tileGroups,
                                params: streamedParams,
                                maxRows: scratch.layout.routedGEMMBatchRows)
                            : 0
                        if rowsBlocks > 0 {
                            // The rows path took the tile.
                        } else if t >= Self.groupedGEMMMinChunkTokens,
                           // The tiled path computes a whole 64-row block per
                           // expert. A prompt chunk fills those blocks; a
                           // 4-token speculative verify block leaves ~1 useful
                           // row in each, so the same answer costs an order of
                           // magnitude more arithmetic.
                           // `TF_PREFILL_MOE_GEMM_MIN_TOKENS` is the knob that
                           // measurement uses to find where the crossover is
                           // (docs/mtp/15-M4-RESULTS.md §4).
                           prefillGroupedMoE.usesExpertGEMMPath(d: D, f: cfg.moeIntermediateSize) {
                            _ = prefillGroupedMoE.encodeStreamedTiled(
                                commandBuffer: tileCB,
                                hidden: scratch.routedX,
                                sortedPairs: metadata.sortedPairs,
                                routePartials: scratch.routePartials,
                                gateUpActScratch: scratch.routedGateUpActScratch,
                                argumentBuffer: argumentBuffer,
                                binding: fetch.binding,
                                groups: tileGroups,
                                params: streamedParams,
                                maxRowsPerBatch: scratch.layout.routedGEMMBatchRows)
                        } else {
                            _ = prefillGroupedMoE.encodeStreamedBatched(
                                commandBuffer: tileCB,
                                hidden: scratch.routedX,
                                sortedPairs: metadata.sortedPairs,
                                routePartials: scratch.routePartials,
                                gateUpActScratch: scratch.routedGateUpActScratch,
                                downScratch: scratch.routedDownScratch,
                                argumentBuffer: argumentBuffer,
                                binding: fetch.binding,
                                params: streamedParams,
                                pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
                        }
                        if blockPipeline {
                            // Committed, not waited for: the next tile's pread
                            // runs against this tile's GPU work exactly as it
                            // did before, but the host no longer stops in
                            // between (docs/mtp/16-M4.5-PLAN.md §4 d).
                            tileCB.commit()
                            blockUsedSlots.formUnion(fetch.plannedAssignedSlots)
                            heldTileState.append(PendingBlockTile(
                                commandBuffer: tileCB,
                                fetch: fetch,
                                argumentBuffer: argumentBuffer))
                            prefillHostProfile.add(.encodeMoE, since: moeEncodeStart)
                            continue
                        }
                        tileCB.commit()
                        pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                               commandBuffer: tileCB,
                                                               fetch: fetch,
                                                               argumentBuffer: argumentBuffer))
                        prefillHostProfile.add(.encodeMoE, since: moeEncodeStart)
                        while pendingTiles.count > schedulerConfig.maxPendingDepth {
                            try drainOldestPendingTile()
                        }
                    }
                    while !pendingTiles.isEmpty {
                        try drainOldestPendingTile()
                    }
                    let tailStart = PrefillHostProfile.mark()
                    guard let tailCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                                      routePartials: scratch.routePartials,
                                                      routeWeights: scratch.routeWeights,
                                                      h2: scratch.h2,
                                                      queryCount: UInt32(t),
                                                      topK: UInt32(cfg.topKExperts),
                                                      d: UInt32(D))
                    let scalarBits = views.layerScalar.buffer.contents()
                        .advanced(by: Int(views.layerScalar.offset))
                        .assumingMemoryBound(to: UInt16.self)[0]
                    prefillLayerTail.encode(commandBuffer: tailCB,
                                            h2: scratch.h2,
                                            h1: scratch.h1,
                                            hidden: scratch.hidden,
                                            postFFN2Weight: views.postFFN2.buffer,
                                            postFFN2WeightOffset: Int(views.postFFN2.offset),
                                            postFFNWeight: views.postFFN.buffer,
                                            postFFNWeightOffset: Int(views.postFFN.offset),
                                            queryCount: UInt32(t),
                                            d: UInt32(D),
                                            h2StrideElements: UInt32(D),
                                            h1StrideElements: UInt32(D),
                                            hiddenStrideElements: UInt32(D),
                                            eps: eps,
                                            layerScalar: Quantization.bf16ToFloat(scalarBits))
                    tailCB.commit()
                    if blockPipeline {
                        // Deferred: the next layer's first buffer cannot start
                        // before this one finishes, so the only reason to stop
                        // here would be to hand the expert slots back — and
                        // that is what the drain at the top of the next layer
                        // is for.
                        pendingBlockLayer = PendingBlockLayer(
                            commandBuffer: tailCB,
                            sharedCommandBuffer: tileLookahead ? sharedCB : nil,
                            metadata: metadata,
                            tiles: heldTileState)
                    } else {
                        try withExtendedLifetime(metadata) {
                            try waitProfiled(.tail, tailCB)
                        }
                    }
                    prefillHostProfile.add(.tail, since: tailStart)
                    if L + 1 < cfg.numLayers {
                        guard let nextCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        cb = nextCB
                    }
                    continue
        }

        // Which rows of this chunk get an lm_head. Prompt prefill wants the last
        // row of the last chunk and nothing else; a speculative verify block
        // wants every row, because each one scores a different proposed token
        // (`docs/mtp/03-DESIGN.md` D3). Everything below the head is identical
        // either way — that identity is what the M4 comparison checks.
        let finalDrainStart = PrefillHostProfile.mark()
        try drainPendingBlockLayer()
        prefillHostProfile.add(.drain, since: finalDrainStart)

        let headStart = PrefillHostProfile.mark()
        defer { prefillHostProfile.add(.head, since: headStart) }
        let headRows: Range<Int>
        switch head {
        case .none:
            headRows = t..<t
        case .finalRow:
            headRows = (t - 1)..<t
        case .allRows:
            headRows = 0..<t
        }
        if !headRows.isEmpty {
            let finalNorm = model.finalNorm
            let lm = model.lmHead
            let greedy = outputMode == .greedyIfAvailable && useFusedGreedyHead
            guard let finalCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            // The head applies the final norm inside its own kernel, so the
            // post-norm hidden the drafter conditions on exists nowhere unless
            // it is produced here — one hidden-wide norm per head row, and only
            // while a speculative loop has a buffer bound (D6).
            if let hiddenRowsOut = speculativeHiddenRows {
                prefillRMS.encodeBF16W(
                    commandBuffer: finalCB,
                    x: scratch.hidden,
                    xOffset: headRows.lowerBound * D * MemoryLayout<Float16>.stride,
                    weight: finalNorm.buffer, weightOffset: Int(finalNorm.offset),
                    out: hiddenRowsOut, outOffset: 0,
                    t: UInt32(headRows.count),
                    d: UInt32(D),
                    eps: eps)
            }
            // A block head reads the 461 MB lm-head table once instead of once
            // per row. At k=4 the per-row loop reads 1.84 GB where the block
            // reads 0.46 GB, which is the second largest item in the verify
            // bill (docs/mtp/16-M4.5-PLAN.md §2, §4 b).
            let blockHead = Self.rowsHeadEnabled && headRows.count > 1
                && headRows.count <= DequantInt4GEMV.maxRows
                // The fused head writes one token per row, so it needs the
                // caller's k-wide buffer and not the single-token scratch.
                && (!greedy || greedyRows != nil)
            if blockHead {
                if greedy, let greedyRows {
                    try fusionHead.encodeGreedyDecodeBlock(
                        commandBuffer: finalCB,
                        hidden: scratch.hidden,
                        hiddenOffset: headRows.lowerBound * D * MemoryLayout<Float16>.stride,
                        normWeight: finalNorm.buffer,
                        normOffset: Int(finalNorm.offset),
                        weights: lm.buffer,
                        weightsOffset: Int(lm.offset),
                        scales: lm.buffer,
                        scalesOffset: Int(lm.scaleOffset),
                        biases: lm.buffer,
                        biasesOffset: Int(lm.biasOffset),
                        outTokens: greedyRows,
                        rows: headRows.count,
                        d: UInt32(D),
                        vocab: UInt32(cfg.vocabSize),
                        rmsEps: eps)
                } else {
                    try prefillFinalRowHead.encodeLogitsRows(
                        commandBuffer: finalCB,
                        hiddenBlock: scratch.hidden,
                        firstRow: headRows.lowerBound,
                        rowCount: headRows.count,
                        rowStrideElements: D,
                        normWeight: finalNorm.buffer,
                        normWeightOffset: Int(finalNorm.offset),
                        weights: lm.buffer,
                        weightsOffset: Int(lm.offset),
                        scales: lm.buffer,
                        scalesOffset: Int(lm.scaleOffset),
                        biases: lm.buffer,
                        biasesOffset: Int(lm.biasOffset),
                        logits: logits,
                        logitsOffset: 0,
                        d: UInt32(D),
                        vocab: UInt32(cfg.vocabSize),
                        rmsEps: eps)
                }
                finalCB.commit()
                try waitProfiled(.head, finalCB)
                if greedy, let greedyRows {
                    lastGreedyToken = greedyRows.contents().load(
                        fromByteOffset: (headRows.count - 1) * MemoryLayout<UInt32>.stride,
                        as: UInt32.self)
                }
                kv?.advance(by: tokens.count)
                prefillChunkState.markCommitted()
                return
            }
            for row in headRows {
                let outputRow = row - headRows.lowerBound
                if greedy {
                    fusionHead.encodeGreedyDecode(
                        commandBuffer: finalCB,
                        hidden: scratch.hidden,
                        hiddenOffset: row * D * MemoryLayout<Float16>.stride,
                        normWeight: finalNorm.buffer,
                        normOffset: Int(finalNorm.offset),
                        weights: lm.buffer,
                        weightsOffset: Int(lm.offset),
                        scales: lm.buffer,
                        scalesOffset: Int(lm.scaleOffset),
                        biases: lm.buffer,
                        biasesOffset: Int(lm.biasOffset),
                        outToken: greedyRows ?? greedyTokenBuf,
                        outTokenOffset: greedyRows == nil
                            ? 0 : outputRow * MemoryLayout<UInt32>.stride,
                        d: UInt32(D),
                        vocab: UInt32(cfg.vocabSize),
                        rmsEps: eps)
                } else {
                    prefillFinalRowHead.encodeLogits(commandBuffer: finalCB,
                                                     hiddenBlock: scratch.hidden,
                                                     row: row,
                                                     rowStrideElements: D,
                                                     normWeight: finalNorm.buffer,
                                                     normWeightOffset: Int(finalNorm.offset),
                                                     weights: lm.buffer,
                                                     weightsOffset: Int(lm.offset),
                                                     scales: lm.buffer,
                                                     scalesOffset: Int(lm.scaleOffset),
                                                     biases: lm.buffer,
                                                     biasesOffset: Int(lm.biasOffset),
                                                     logits: logits,
                                                     logitsOffset: outputRow * cfg.vocabSize
                                                         * MemoryLayout<Float16>.stride,
                                                     d: UInt32(D),
                                                     vocab: UInt32(cfg.vocabSize),
                                                     rmsEps: eps)
                }
            }
            finalCB.commit()
            try waitProfiled(.head, finalCB)
            if greedy, let greedyRows {
                lastGreedyToken = greedyRows.contents().load(
                    fromByteOffset: (headRows.count - 1) * MemoryLayout<UInt32>.stride,
                    as: UInt32.self)
            } else if greedy {
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            }
        }

        kv?.advance(by: tokens.count)
        prefillChunkState.markCommitted()
    }

    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode) async throws {
        let kvPosition = kv?.position ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        let D    = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let eps: Float = 1e-6
        let sqrtHidden = Float(cfg.hiddenSize).squareRoot()
        // A decode step's own busy/idle split, on the same env var as the block
        // one. The last buffers of a step are waited for at the top of the next
        // step, so the window is closed a step late by construction — over 32
        // steps that is a 3% edge effect on the span, and none at all on busy.
        let profilingDecodeQueue = PrefillHostProfile.isEnabled
        if profilingDecodeQueue {
            decodeQueueRecording = true
            decodeQueueSteps += 1
        }
        defer {
            if profilingDecodeQueue {
                decodeQueueRecording = false
                if decodeQueueSteps >= Self.decodeQueueWindow {
                    if let line = decodeQueue.summary(label: "decode gpuq",
                                                      extra: " tok=\(decodeQueueSteps)") {
                        FileHandle.standardError.write(Data((line + "\n").utf8))
                    }
                    if PrefillGPUProfile.isEnabled {
                        let line = decodeStageProfile.summary(label: "decode gpu")
                        FileHandle.standardError.write(Data((line + "\n").utf8))
                        decodeStageProfile.reset()
                    }
                    decodeQueue.reset()
                    decodeQueueSteps = 0
                }
            }
        }
        struct PendingRoutedCommand {
            let cb: MTLCommandBuffer
            let sharedCB: MTLCommandBuffer?
            let phase1HitCB: MTLCommandBuffer?
            let encodeAndCommitNanos: UInt64
        }
        var pendingRoutedCommand: PendingRoutedCommand?

        func finishPendingRoutedCommand(_ pending: PendingRoutedCommand,
                                        waitIfNeeded: Bool) throws {
            if waitIfNeeded {
                if let sharedCB = pending.sharedCB {
                    decodeStage = .shared
                    waitUntilCompleted(sharedCB)
                }
                decodeStage = .moe
                if let phase1HitCB = pending.phase1HitCB {
                    waitUntilCompleted(phase1HitCB)
                }
                waitUntilCompleted(pending.cb)
            } else if decodeQueueRecording {
                // Not waited for, but finished: these buffers sit ahead of the
                // front buffer this step already blocked on, so their GPU spans
                // are readable. Recording them here is what keeps the decode
                // side of the busy/idle split honest — a layer's routed experts
                // are most of its GPU time, and skipping them would book that
                // time as an idle queue.
                if let sharedCB = pending.sharedCB {
                    decodeQueue.record(sharedCB)
                    decodeStageProfile.record(.shared, sharedCB)
                }
                if let phase1HitCB = pending.phase1HitCB {
                    decodeQueue.record(phase1HitCB)
                    decodeStageProfile.record(.moe, phase1HitCB)
                }
                decodeQueue.record(pending.cb)
                decodeStageProfile.record(.moe, pending.cb)
            }
            if let sharedCB = pending.sharedCB {
                try checkCommandBufferError(sharedCB.error)
            }
            if let phase1HitCB = pending.phase1HitCB {
                try checkCommandBufferError(phase1HitCB.error)
            }
            try checkCommandBufferError(pending.cb.error)
            totalCb2Nanos &+= pending.encodeAndCommitNanos
        }

        func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
            let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<slots.count { ptr[i] = slots[i] }
        }

        // The M3 probe checks the rounds it has open against this step's input
        // token before the target touches anything; it drafts at the end of the
        // step, once this position's K/V and hidden exist. It writes neither
        // the KV cache nor any buffer the target reads.
        if draftProbeSettings != nil, let probe = try ensureDraftProbe() {
            probe.observe(token: token, position: position)
        }

        // Embed lookup + sqrt(H) fused.
        let emb = model.embedding
        do {
            try runSync { cb in
                embedInt4.encode(commandBuffer: cb,
                                 table:  emb.buffer, tableOffset:  Int(emb.offset),
                                 scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                 biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                 out: hidden,
                                 tokenId: UInt32(bitPattern: token),
                                 d: D,
                                 outScale: sqrtHidden)
            }
        }

        for L in 0..<cfg.numLayers {
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            let headDimL = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVL   = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim     = UInt32(cfg.numHeads * headDimL)
            let kvDim    = UInt32(numKVL * headDimL)
            let kSlot    = kv?.kSlot(layer: L, position: position) ?? (buffer: kStage, offset: 0)
            let vSlot    = kv?.vSlot(layer: L, position: position) ?? (buffer: vStage, offset: 0)
            let seqLen   = UInt32(position + 1)

            let inNorm   = try model.inputNorm(layer: L)
            let q        = try model.qProj(layer: L)
            let k        = try model.kProj(layer: L)
            // v_proj only exists on SWA layers; full layers reuse k_proj.
            let vProj    = isFull ? k : (try model.vProj(layer: L))
            let o        = try model.oProj(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            let qNorm    = try model.qNorm(layer: L)
            let kNorm    = try model.kNorm(layer: L)
            let preFFN   = try model.preFFN(layer: L)
            let preFFN2  = try model.preFFN2(layer: L)
            let sharedProj = sharedExpertProjections[L]
            let postF2   = try model.postFFN2(layer: L)
            let postF    = try model.postFFN(layer: L)
            let routerW  = try model.router(layer: L)
            let perExpertScale = try model.routerPerExpertScale(layer: L)
            let layerScalarView = try model.layerScalar(layer: L)

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // Everything up to and including the router runs in a single CB:
            // the only reason to break is the CPU readback of router indices
            // needed to issue I/O for the routed-expert blobs.
            let gInputNorm: (MTLCommandBuffer) -> Void = { [self] cb in
                rms.encodeBF16W(commandBuffer: cb,
                                x: hidden,
                                weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                                out: normed,
                                d: D, eps: eps)
            }

            let gQKV: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedQKVGEMV.encode(commandBuffer: cb,
                                    qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                                    qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                                    qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                                    kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                                    kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                                    kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                                    vWeights: vProj.buffer, vWeightsOffset: Int(vProj.offset),
                                    vScales: vProj.buffer, vScalesOffset: Int(vProj.scaleOffset),
                                    vBiases: vProj.buffer, vBiasesOffset: Int(vProj.biasOffset),
                                    x: normed,
                                    qOut: qScratch,
                                    kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                                    vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                                    qRows: qDim,
                                    kvRows: kvDim,
                                    n: D)
            }

            let gQKVEpilogue: (MTLCommandBuffer) -> Void = { [self] cb in
                let rotated = isFull
                    ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                    : UInt32(headDimL / 2)
                fusedQKVEpilogue.encode(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer,
                                        kOffset: kSlot.offset,
                                        v: vSlot.buffer,
                                        vOffset: vSlot.offset,
                                        qWeight: qNorm.buffer,
                                        qWeightOffset: Int(qNorm.offset),
                                        kWeight: kNorm.buffer,
                                        kWeightOffset: Int(kNorm.offset),
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        position: UInt32(position),
                                        theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                        rotatedPairs: rotated,
                                        eps: eps)
            }

            let gAttention: (MTLCommandBuffer) -> Void = { [self] cb in
                guard kv != nil else {
                    preconditionFailure("FP16 attention requires an FP16 KV cache")
                }
                if isFull {
                    attention.encodeFull(commandBuffer: cb,
                                         q: qScratch,
                                         k: kSlot.buffer, kOffset: 0,
                                         v: vSlot.buffer, vOffset: 0,
                                         out: attnOut,
                                         headDim: UInt32(headDimL),
                                         numQHeads: UInt32(cfg.numHeads),
                                         numKVHeads: UInt32(numKVL),
                                         seqLen: seqLen,
                                         scale: 1.0)
                } else {
                    let ringCapacity = kv?.ringCapacity(layer: L) ?? 0
                    let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                        ? UInt32(ringCapacity)
                        : 0
                    attention.encodeSWA(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer, kOffset: 0,
                                        v: vSlot.buffer, vOffset: 0,
                                        out: attnOut,
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        seqLen: seqLen,
                                        window: UInt32(cfg.slidingWindow),
                                        scale: 1.0,
                                        ringCapacity: activeRingCapacity)
                }
            }
            let gOProj: (MTLCommandBuffer) -> Void = { [self] cb in
                int4.encode(commandBuffer: cb,
                            weights: o.buffer, weightsOffset: Int(o.offset),
                            scales:  o.buffer, scalesOffset:  Int(o.scaleOffset),
                            biases:  o.buffer, biasesOffset:  Int(o.biasOffset),
                            x: attnOut, y: oOut, m: D, n: qDim)
            }

            let gPostAttnSetup: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedPostAttentionSetup.encode(commandBuffer: cb,
                                               hidden: hidden,
                                               attn: oOut,
                                               denseX: denseX,
                                               routedX: routedX,
                                               routerX: routerInput,
                                               postAttentionWeight: postAttn.buffer,
                                               postAttentionWeightOffset: Int(postAttn.offset),
                                               preFFNWeight: preFFN.buffer,
                                               preFFNWeightOffset: Int(preFFN.offset),
                                               preFFN2Weight: preFFN2.buffer,
                                               preFFN2WeightOffset: Int(preFFN2.offset),
                                               d: D,
                                               eps: eps)
            }

            let gRouter: (MTLCommandBuffer) -> Void = { [self] cb in
                if moe.routerWeightBits == 16 {
                    moe.encodeRouterGemma4BF16(commandBuffer: cb,
                        weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                        hidden: routerInput,
                        effectiveScale: effectiveScaleBuffers[L],
                        perExpertScale: perExpertScale.buffer,
                        perExpertScaleOffset: Int(perExpertScale.offset),
                        outIndices: outIndices, outWeights: outWeights,
                        numExperts: UInt32(cfg.numExperts), d: D,
                        topK: UInt32(cfg.topKExperts))
                } else {
                    moe.encodeRouterGemma4(commandBuffer: cb,
                        weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                        scales:  routerW.buffer, scalesOffset:  Int(routerW.scaleOffset),
                        biases:  routerW.buffer, biasesOffset:  Int(routerW.biasOffset),
                        hidden: routerInput,
                        effectiveScale: effectiveScaleBuffers[L],
                        perExpertScale: perExpertScale.buffer,
                        perExpertScaleOffset: Int(perExpertScale.offset),
                        outIndices: outIndices, outWeights: outWeights,
                        numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))
                }
            }

            let cb = ctx.queue.makeCommandBuffer()!
            gInputNorm(cb)
            gQKV(cb)
            gQKVEpilogue(cb)
            gAttention(cb)
            gOProj(cb)
            gPostAttnSetup(cb)
            gRouter(cb)
            cb.commit()
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            decodeStage = .attention
            waitUntilCompleted(cb)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            if let pending = pendingRoutedCommand {
                try finishPendingRoutedCommand(pending, waitIfNeeded: false)
                pendingRoutedCommand = nil
            }
            try checkCommandBufferError(cb.error)
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos

            // CPU readback to fetch routed-expert blobs from disk.
            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                          capacity: cfg.topKExperts)
            var experts = [Int](repeating: 0, count: cfg.topKExperts)
            for i in 0..<cfg.topKExperts {
                experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1)
            }

            let routedOffsets = model.routedExpertOffsets(layer: L)
            let topK = UInt32(cfg.topKExperts)
            let canPlanPhase1HitSplit =
                cfg.topKExperts <= MoE.maxStreamedExperts
            let plannedFetch = canPlanPhase1HitSplit
                ? try model.planRoutedExperts(layer: L, experts: experts)
                : nil
            var phase1HitCB: MTLCommandBuffer?
            var phase1HitSplitArgBuf: MTLBuffer?
            var phase1HitSplitRoutedBufs: [MTLBuffer] = []
            var phase1HitSlots: [UInt32] = []
            var phase1MissSlots: [UInt32] = []

            if let plan = plannedFetch {
                let missSet = Set(plan.misses)
                phase1HitSlots = (0..<cfg.topKExperts)
                    .filter { !missSet.contains($0) }
                    .map { UInt32($0) }
                phase1MissSlots = plan.misses.map { UInt32($0) }
            }
            func encodeRoutedPhase1Full(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MTLBuffer]
            ) {
                moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                        routedArgBuffer: argBuf,
                                                        routedBlobs: routedBufs,
                                                        routedOffsets: routedOffsets,
                                                        x: routedX,
                                                        acts: moeActs,
                                                        d: D,
                                                        f: FmoE,
                                                        topK: topK)
            }

            func encodeRoutedPhase1Subset(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MTLBuffer],
                activeSlots: MTLBuffer,
                activeSlotIndices: [UInt32],
                activeCount: UInt32
            ) {
                moe.encodeRoutedPersistentPhase1SubsetU16Load(
                    commandBuffer: cb,
                    routedArgBuffer: argBuf,
                    routedBlobs: routedBufs,
                    routedOffsets: routedOffsets,
                    x: routedX,
                    acts: moeActs,
                    activeSlots: activeSlots,
                    activeSlotIndices: activeSlotIndices,
                    activeCount: activeCount,
                    d: D,
                    f: FmoE,
                    topK: topK)
            }

            if let plan = plannedFetch,
               plan.hits > 0,
               !plan.misses.isEmpty {
                let plannedBlobs = try model.routedExpertBuffers(for: plan)
                phase1HitSplitRoutedBufs = plannedBlobs.map { $0.buffer }
                phase1HitSplitArgBuf = moe.makeRoutedArgumentBuffer(
                    routedBlobs: phase1HitSplitRoutedBufs,
                    topK: topK)
                if let argBuf = phase1HitSplitArgBuf, plan.hits > 0, !plan.misses.isEmpty {
                    writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                    let cb = ctx.queue.makeCommandBuffer()!
                    encodeRoutedPhase1Subset(
                        cb,
                        argBuf: argBuf,
                        routedBufs: phase1HitSplitRoutedBufs,
                        activeSlots: moeHitActiveSlots,
                        activeSlotIndices: phase1HitSlots,
                        activeCount: UInt32(phase1HitSlots.count))
                    phase1HitCB = cb
                }
            }

            // The shared dense MLP depends only on denseX, not on the routed
            // experts. Commit it without waiting so its GPU work overlaps the
            // routed-expert pread. The routed CB follows it on the same queue,
            // so the combine sees h1Buf.
            let gSharedFFN: (MTLCommandBuffer) -> Void = { [self] cb in
                try! shared.encode(commandBuffer: cb,
                                   x: denseX,
                                   gate: sharedProj.gate,
                                   up: sharedProj.up,
                                   down: sharedProj.down,
                                   y: h1Buf,
                                   scratchGate: denseScratchGate,
                                   scratchUp: denseScratchUp,
                                   scratchAct: denseScratchAct)
            }
            let gSharedNorm: (MTLCommandBuffer) -> Void = { [self] cb in
                rms.encodeBF16W(commandBuffer: cb, x: h1Buf,
                                weight: sharedProj.postF1.buffer,
                                weightOffset: Int(sharedProj.postF1.offset),
                                out: h1Buf, d: D, eps: eps)
            }
            let sharedCB = ctx.queue.makeCommandBuffer()!
            gSharedFFN(sharedCB)
            gSharedNorm(sharedCB)
            sharedCB.commit()
            if let cb = phase1HitCB {
                cb.commit()
            }
            if rdadviseEnabled && rdadvisePolicyMode != .off {
                let requestedMisses = plannedFetch?.misses.count ?? experts.count
                let estimatedAdviceBytes = try model.routedExpertAdviceByteEstimate(
                    layer: L,
                    missCount: requestedMisses)
                if let skipped = shouldSkipRDAdvice(position: position,
                                                    requestedMisses: requestedMisses,
                                                    estimatedBytes: estimatedAdviceBytes,
                                                    canOverlapUsefulGPUWork: true) {
                    recordRDAdvice(skipped, wallNanos: 0)
                } else {
                    let tAdvice = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    let result: ExpertIOAdviceResult
                    if let plannedFetch {
                        result = try model.adviseRoutedExperts(plan: plannedFetch)
                    } else {
                        result = try model.adviseRoutedExperts(layer: L, experts: experts)
                    }
                    let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tAdvice
                    recordRDAdvice(result, wallNanos: wallNanos)
                    updateRDAdvicePolicy(after: result, position: position)
                }
            }

            // Routed-expert pread — overlaps the shared MLP GPU work above.
            let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let blobs: [TensorView]
            if let plannedFetch {
                blobs = try await model.fetchRoutedExperts(plan: plannedFetch)
            } else {
                blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
            }
            let layerIo = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart
            totalIoNanos &+= layerIo
            let routedBufs = blobs.map { $0.buffer }
            let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let scalarPtr = layerScalarView.buffer.contents()
                .advanced(by: Int(layerScalarView.offset))
                .assumingMemoryBound(to: UInt16.self)
            let layerScalar = Quantization.bf16ToFloat(scalarPtr[0])

            let gTail: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedTail.encode(commandBuffer: cb,
                                 h2: h2Buf,
                                 h1: h1Buf,
                                 hidden: hidden,
                                 postFFN2Weight: postF2.buffer,
                                 postFFN2WeightOffset: Int(postF2.offset),
                                 postFFNWeight: postF.buffer,
                                 postFFNWeightOffset: Int(postF.offset),
                                 d: D,
                                 eps: eps,
                                 layerScalar: layerScalar)
            }
            let routedCB = ctx.queue.makeCommandBuffer()!
            let splitArgBuf = phase1HitCB != nil && !phase1MissSlots.isEmpty
                ? phase1HitSplitArgBuf
                : nil
            let argBuf = splitArgBuf ?? moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: routedBufs,
                topK: topK)
            if splitArgBuf != nil {
                writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
                encodeRoutedPhase1Subset(
                    routedCB,
                    argBuf: argBuf,
                    routedBufs: routedBufs,
                    activeSlots: moeMissActiveSlots,
                    activeSlotIndices: phase1MissSlots,
                    activeCount: UInt32(phase1MissSlots.count))
            } else {
                encodeRoutedPhase1Full(routedCB,
                                       argBuf: argBuf,
                                       routedBufs: routedBufs)
            }
            moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: routedCB,
                                                   routedArgBuffer: argBuf,
                                                   routedBlobs: routedBufs,
                                                   routedOffsets: routedOffsets,
                                                   acts: moeActs,
                                                   routingWeights: outWeights,
                                                   residual: zeroResidual,
                                                   y: h2Buf,
                                                   d: D,
                                                   f: FmoE,
                                                   topK: topK)
            gTail(routedCB)
            routedCB.commit()
            precondition(pendingRoutedCommand == nil,
                         "routed command-buffer pipeline drained before queuing the next layer")
            pendingRoutedCommand = PendingRoutedCommand(
                cb: routedCB,
                sharedCB: sharedCB,
                phase1HitCB: phase1HitCB,
                encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
            continue
        }
        if let pending = pendingRoutedCommand {
            try finishPendingRoutedCommand(pending, waitIfNeeded: true)
            pendingRoutedCommand = nil
        }

        // The fused head skips the vocab buffer and leaves a greedy token in
        // greedyTokenBuf; the logits path writes the complete vector.
        let fNorm = model.finalNorm
        let lm    = model.embedding
        let gFinalNorm: (MTLCommandBuffer) -> Void = { cb in
            self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                 weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                 out: self.normed, d: D, eps: eps)
        }
        let gLmHead: (MTLCommandBuffer) -> Void = { cb in
            self.int4.encode(commandBuffer: cb,
                             weights: lm.buffer, weightsOffset: Int(lm.offset),
                             scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                             biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                             x: self.normed, y: logits, m: UInt32(self.cfg.vocabSize), n: D)
        }
        let gFusionHead: (MTLCommandBuffer) -> Void = { cb in
            self.fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: self.hidden,
                normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: self.greedyTokenBuf,
                d: D, vocab: UInt32(self.cfg.vocabSize),
                rmsEps: eps)
        }
        // 02-RUNTIME-FIT §N3: the head applies the final RMSNorm inside its own
        // kernel, so the post-norm hidden the drafter needs exists nowhere.
        // One extra 2816-wide norm produces it — only when probing. The round
        // then runs on this position's own (token, hidden, KV) triple, before
        // `advance()`, so the cache ends exactly at the bonus token.
        if let probe = draftProbe, let kv {
            try runSync { cb in
                self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                     weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                     out: probe.hiddenCaptureBuffer, d: D, eps: eps)
                // M3.5: the pre-norm residual and the norm weight go out too, so
                // the replay can tell "wrong quantity" from "wrong scale".
                probe.capturePreNorm(commandBuffer: cb, hidden: self.hidden,
                                     normWeight: fNorm.buffer,
                                     normWeightOffset: Int(fNorm.offset),
                                     count: Int(D))
            }
            try probe.draft(bonusToken: token, position: position, kv: kv)
        }

        if emitHead {
            let useFusedHeadForThisToken = useFusedGreedyHead && outputMode == .greedyIfAvailable
            let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            decodeStage = .head
            if useFusedHeadForThisToken {
                try runSync(gFusionHead)
                totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            } else {
                try runSync { cb in
                    gFinalNorm(cb)
                    gLmHead(cb)
                }
                totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            }
        }

        kv?.advance()
    }

    private func runSync(_ body: (MTLCommandBuffer) -> Void) throws {
        let cb = ctx.queue.makeCommandBuffer()!
        body(cb)
        cb.commit()
        try waitForCompletion(cb)
    }

    private nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) throws {
        waitUntilCompleted(cb)
        try checkCommandBufferError(cb.error)
    }

    /// `waitForCompletion` plus the command buffer's own GPU span, bucketed by
    /// stage. Off unless `TF_PREFILL_GPU_PROFILE=1`, and the accounting is
    /// GPU-side, so a stage that overlaps expert I/O shows only its GPU cost.
    private func waitProfiled(_ stage: PrefillGPUProfile.Stage,
                              _ cb: MTLCommandBuffer,
                              detail: PrefillGPUProfile.Detail? = nil) throws {
        try waitForCompletion(cb)
        prefillGPUProfile.record(stage, cb, detail: detail)
        prefillHostProfile.recordGPUInterval(start: cb.gpuStartTime, end: cb.gpuEndTime)
    }

    private func waitUntilCompleted(_ cb: MTLCommandBuffer) {
        cb.waitUntilCompleted()
        guard decodeQueueRecording else { return }
        decodeQueue.record(cb)
        decodeStageProfile.record(decodeStage, cb)
    }

}
