import Foundation
import Metal

struct PrefillAttentionParams: Sendable, Equatable {
    var startPosition: UInt32
    var queryCount: UInt32
    var headDim: UInt32
    var numQHeads: UInt32
    var numKVHeads: UInt32
    var kvValidCount: UInt32
    var slidingWindow: UInt32
    var kvTokenStrideElements: UInt32
    var qTokenStrideElements: UInt32
    var oTokenStrideElements: UInt32
    var scale: Float
    /// Non-zero when the `spanEnd` buffer is bound and carries a per-query
    /// visible end (`PLAN_VISION.md` §4-5-b). Zero is the text-only path, which
    /// is bit-identical to the kernel before image spans existed.
    var spanMaskEnabled: UInt32

    init(startPosition: UInt32,
                queryCount: UInt32,
                headDim: UInt32,
                numQHeads: UInt32,
                numKVHeads: UInt32,
                kvValidCount: UInt32,
                slidingWindow: UInt32,
                kvTokenStrideElements: UInt32,
                qTokenStrideElements: UInt32,
                oTokenStrideElements: UInt32,
                scale: Float,
                spanMaskEnabled: Bool = false) {
        self.startPosition = startPosition
        self.queryCount = queryCount
        self.headDim = headDim
        self.numQHeads = numQHeads
        self.numKVHeads = numKVHeads
        self.kvValidCount = kvValidCount
        self.slidingWindow = slidingWindow
        self.kvTokenStrideElements = kvTokenStrideElements
        self.qTokenStrideElements = qTokenStrideElements
        self.oTokenStrideElements = oTokenStrideElements
        self.scale = scale
        self.spanMaskEnabled = spanMaskEnabled ? 1 : 0
    }
}


final class PrefillAttention {
    /// A compiled specialisation of the query-blocked kernel. The head
    /// dimension and the queries per simdgroup are template arguments, so the
    /// host has to dispatch the exact shape a kernel was built for; a head
    /// dimension with no entry here falls back to the tiled kernel.
    private struct QBlockSpecialisation {
        let function: String
        let queriesPerSimdgroup: Int
    }

    private static let qBlockThreads = 256
    private static let qBlockSpecialisations: [UInt32: QBlockSpecialisation] = [
        256: QBlockSpecialisation(function: "attention_prefill_causal_qblock_d256",
                                  queriesPerSimdgroup: 4),
        512: QBlockSpecialisation(function: "attention_prefill_causal_qblock_d512",
                                  queriesPerSimdgroup: 2),
    ]

    /// `TF_PREFILL_ATTN=sgmm` sends `causalQBlock`'s headDim-256 dispatches
    /// to the simdgroup-matrix kernel, and `TF_PREFILL_ATTN=qblock` sends the
    /// Gemma default (`causalSimdgroupMatrix`) back to the query-blocked one,
    /// so an A/B runs on one binary.
    private static let forcedSimdgroupMatrix =
        ProcessInfo.processInfo.environment["TF_PREFILL_ATTN"] == "sgmm"
    private static let forcedQBlock =
        ProcessInfo.processInfo.environment["TF_PREFILL_ATTN"] == "qblock"
    /// The Gemma default (`causalSimdgroupMatrix`) runs its headDim-256
    /// dispatches on the `matmul2d` kernel where the library has it and the
    /// shape allows, and on the simdgroup-matrix one otherwise
    /// (docs/m6-prefill/25・26). `TF_PREFILL_ATTN=sgmm` keeps them on the
    /// simdgroup-matrix kernel, `=qblock` on the query-blocked one.
    private static let swaMPPDisabled = forcedSimdgroupMatrix || forcedQBlock
    private static let swaMPPRows = 64
    /// The fused fallback for the full-attention layers (headDim 512) where
    /// the unfused path below does not fit: the same `matmul2d` kernel as the
    /// sliding-window layers, 64 rows on 8 simdgroups (docs/m6-prefill/36).
    /// `TF_PREFILL_ATTN_FULL=mpp32` takes 32 rows on 4, and `=qblock` (or
    /// `TF_PREFILL_ATTN=sgmm`/`=qblock`) keeps them on the query-blocked kernel.
    private static let fullMPPShape: (function: String, rows: Int, threads: Int)? = {
        if swaMPPDisabled { return nil }
        switch ProcessInfo.processInfo.environment["TF_PREFILL_ATTN_FULL"] {
        case "qblock": return nil
        case "mpp32": return ("attention_prefill_full_mpp_d512_r32", 32, 128)
        default: return ("attention_prefill_full_mpp_d512_r64", 64, 256)
        }
    }()
    private static let fullMPPHeadDim: UInt32 = 512
    /// The full layers by default, as MLX runs headDim 512: a GEMM for the
    /// scores, a softmax, a GEMM for P·V, over as many query heads at a time as
    /// fit a bounded scratch (docs/m6-prefill/36). A shape whose one head does
    /// not fit (a long prefix) falls back to the fused kernel above. Any
    /// `TF_PREFILL_ATTN_FULL` value (`mpp64`, `mpp32`, `qblock`) turns it off.
    private static let fullUnfusedRequested =
        !swaMPPDisabled && (ProcessInfo.processInfo.environment["TF_PREFILL_ATTN_FULL"] ?? "unfused") == "unfused"
    /// The sliding-window layers (headDim 256) on the same three kernels, each
    /// 64-row block keeping only its band of keys (docs/m6-prefill/37). A
    /// candidate for now, `TF_PREFILL_ATTN_SWA=unfused`.
    private static let swaUnfusedRequested =
        !swaMPPDisabled && ProcessInfo.processInfo.environment["TF_PREFILL_ATTN_SWA"] == "unfused"
    private static let unfusedScratchBudget = 64 << 20
    private static let swaMPPKeys: UInt32 = 64
    private static let swaMPPThreads = 128
    private static let sgmmHeadDim: UInt32 = 256
    private static let sgmmQueries = 32
    private static let sgmmThreads = 128
    /// With two query heads per KV head (the pinned model's sliding-window
    /// layers, 16/8) both heads share one threadgroup and its K/V tiles.
    private static let sgmmGqaHeads: UInt32 = 2
    private static let sgmmGqaThreads = 256

    private let context: MetalContext
    private let psoCausalTiled: MTLComputePipelineState
    private let psoFullTensorOps2DValidityV2: MTLComputePipelineState?
    /// Whether the library has the `matmul2d` sliding-window kernel at all;
    /// the ring-specialised pipelines are built on first use.
    private let hasSwaMPP: Bool
    private let hasFullMPP: Bool
    private typealias UnfusedPipelines = (scores: MTLComputePipelineState,
                                          softmax: MTLComputePipelineState,
                                          pv: MTLComputePipelineState)
    /// Keyed by head dimension: 512 when the full layers use them, 256 when
    /// the sliding-window ones do.
    private let unfusedPipelines: [UInt32: UnfusedPipelines]
    private var unfusedScratch: MTLBuffer?

    init(context: MetalContext) throws {
        self.context = context
        self.psoCausalTiled = try context.pipeline("attention_prefill_causal_tiled")
        // Gated on the pipeline itself, not on a GPU family: the kernel only
        // exists where the library was built with tensor support, and a family
        // flag is today's answer rather than the next machine's
        // (docs/m6-prefill/05-RISKS.md §7).
        self.psoFullTensorOps2DValidityV2 =
            try? context.pipeline("attention_prefill_full_tensorops_2d_validity_v2")
        self.hasSwaMPP = !Self.swaMPPDisabled
            && (try? context.pipeline("attention_prefill_swa_mpp_d256")) != nil
        var unfused: [UInt32: UnfusedPipelines] = [:]
        for (headDim, requested) in [(UInt32(512), Self.fullUnfusedRequested),
                                     (UInt32(256), Self.swaUnfusedRequested)] where requested {
            if let scores = try? context.pipeline("attention_prefill_unfused_scores_d\(headDim)"),
               let softmax = try? context.pipeline("attention_prefill_unfused_softmax"),
               let pv = try? context.pipeline("attention_prefill_unfused_pv_d\(headDim)") {
                unfused[headDim] = (scores, softmax, pv)
            }
        }
        self.unfusedPipelines = unfused
        self.hasFullMPP = Self.fullMPPShape.map {
            (try? context.pipeline($0.function)) != nil
        } ?? false
    }

    func encodeCausal(commandBuffer: MTLCommandBuffer,
                             q: MTLBuffer, qOffset: Int = 0,
                             k: MTLBuffer, kOffset: Int = 0,
                             v: MTLBuffer, vOffset: Int = 0,
                             out: MTLBuffer, outOffset: Int = 0,
                             params: PrefillAttentionParams,
                             spanEnd: MTLBuffer? = nil,
                             spanEndOffset: Int = 0,
                             kvRingCapacity: UInt32 = 0,
                             path: RuntimePrefillAttentionPath = .causalTiled) {
        validate(params)
        let path: RuntimePrefillAttentionPath =
            path == .causalQBlock && Self.forcedSimdgroupMatrix ? .causalSimdgroupMatrix
            : path == .causalSimdgroupMatrix && Self.forcedQBlock ? .causalQBlock
            : path
        precondition((params.spanMaskEnabled != 0) == (spanEnd != nil),
                     "the span mask flag and the span-end buffer must agree")

        let requestsTensorOps = path == .fullTensorOps2DPreferred
            || path == .fullTensorOps2DValidityV2
        // The pinned model uses 512/16/2 only for full attention; its
        // sliding-window layers use 256/16/8. A future model that reuses this
        // shape for sliding attention must add a full-visibility check here.
        //
        // The TensorOps kernel has no span mask: it is a causal-only path, and
        // letting it serve a masked dispatch would silently drop the image
        // span's bidirectional visibility on Apple10 hardware (PLAN_VISION
        // §4-5-b). Excluded by shape, not trusted to be unreachable.
        let tensorOpsShape = requestsTensorOps
            && params.spanMaskEnabled == 0
            && kvRingCapacity == 0
            && params.headDim == 512
            && params.numQHeads == 16
            && params.numKVHeads == 2
            && params.scale == 1.0
        let tensorOpsPipeline = tensorOpsShape ? psoFullTensorOps2DValidityV2 : nil
        // K and V are sliced a key block at a time, so a ring has to be a
        // whole number of blocks; the kernel has no span mask.
        let swaMPPShape = hasSwaMPP
            && path == .causalSimdgroupMatrix
            && params.headDim == Self.sgmmHeadDim
            && params.spanMaskEnabled == 0
            && kvRingCapacity % Self.swaMPPKeys == 0
        if let pipelines = unfusedPipelines[params.headDim],
           path == .causalSimdgroupMatrix,
           params.spanMaskEnabled == 0,
           encodeUnfused(pipelines, commandBuffer: commandBuffer,
                         q: q, qOffset: qOffset, k: k, kOffset: kOffset,
                         v: v, vOffset: vOffset, out: out, outOffset: outOffset,
                         params: params, kvRingCapacity: kvRingCapacity) {
            return
        }
        let fullMPPShape = hasFullMPP
            && path == .causalSimdgroupMatrix
            && params.headDim == Self.fullMPPHeadDim
            && params.spanMaskEnabled == 0
            && kvRingCapacity % Self.swaMPPKeys == 0
        enum Variant { case tensorOps, swaMPP, fullMPP, sgmm, sgmmGqa, qBlock, tiled }
        let variant: Variant
        let pipeline: MTLComputePipelineState
        var qBlock: QBlockSpecialisation?
        if let tensorOpsPipeline {
            variant = .tensorOps
            pipeline = tensorOpsPipeline
        } else if tensorOpsShape && path == .fullTensorOps2DValidityV2 {
            preconditionFailure(
                "TensorOps 2D prefill attention requires Apple10 MPP tensor support")
        } else if swaMPPShape {
            variant = .swaMPP
            pipeline = sgmmPipeline("attention_prefill_swa_mpp_d256", kvRingCapacity: kvRingCapacity)
        } else if fullMPPShape, let shape = Self.fullMPPShape {
            variant = .fullMPP
            pipeline = sgmmPipeline(shape.function, kvRingCapacity: kvRingCapacity)
        } else if path == .causalSimdgroupMatrix && params.headDim == Self.sgmmHeadDim {
            let gqa = params.numQHeads == Self.sgmmGqaHeads * params.numKVHeads
            variant = gqa ? .sgmmGqa : .sgmm
            pipeline = sgmmPipeline(
                gqa ? "attention_prefill_causal_sgmm_gqa_d256" : "attention_prefill_causal_sgmm_d256",
                kvRingCapacity: kvRingCapacity)
        } else if let specialisation = Self.qBlockSpecialisations[params.headDim] {
            // Both attention kinds: 256 is the sliding-window layers, 512 the
            // full ones.
            variant = .qBlock
            qBlock = specialisation
            pipeline = qBlockPipeline(specialisation, kvRingCapacity: kvRingCapacity)
        } else {
            // Explicit mode also falls back for incompatible shapes. Benchmark
            // fixtures must use 512/16/2 to prove that TensorOps ran.
            variant = .tiled
            pipeline = causalTiledPipeline(kvRingCapacity: kvRingCapacity)
        }
        let headDim = Int(params.headDim)
        let threadWidth = max(1, pipeline.threadExecutionWidth)
        let threadCount: Int
        switch variant {
        case .tensorOps: threadCount = 128
        case .swaMPP: threadCount = Self.swaMPPThreads
        case .fullMPP: threadCount = Self.fullMPPShape?.threads ?? 0
        case .sgmm: threadCount = Self.sgmmThreads
        case .sgmmGqa: threadCount = Self.sgmmGqaThreads
        case .qBlock: threadCount = Self.qBlockThreads
        case .tiled: threadCount = roundUp(max(threadWidth, headDim), toMultipleOf: threadWidth)
        }
        precondition(threadCount <= pipeline.maxTotalThreadsPerThreadgroup,
                     "tiled prefill attention requires headDim <= maxTotalThreadsPerThreadgroup")

        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
        // Bound only when the mask is on; the kernels read it under the same
        // flag, so an unbound slot is never dereferenced.
        enc.setBuffer(spanEnd, offset: spanEnd == nil ? 0 : spanEndOffset, index: 5)
        let groups: MTLSize
        switch variant {
        case .tensorOps:
            groups = MTLSize(width: Int(params.queryCount),
                             height: Int(params.numQHeads) / 8,
                             depth: 1)
        case .swaMPP:
            groups = MTLSize(
                width: (Int(params.queryCount) + Self.swaMPPRows - 1) / Self.swaMPPRows,
                height: Int(params.numQHeads),
                depth: 1)
        case .fullMPP:
            let rows = Self.fullMPPShape?.rows ?? 1
            groups = MTLSize(
                width: (Int(params.queryCount) + rows - 1) / rows,
                height: Int(params.numQHeads),
                depth: 1)
        case .sgmm, .sgmmGqa:
            groups = MTLSize(
                width: (Int(params.queryCount) + Self.sgmmQueries - 1) / Self.sgmmQueries,
                height: variant == .sgmmGqa ? Int(params.numKVHeads) : Int(params.numQHeads),
                depth: 1)
        case .qBlock:
            // One simdgroup per `queriesPerSimdgroup` queries, and the kernel
            // derives the same figure from `simdgroups_per_threadgroup`.
            let queriesPerGroup = (qBlock?.queriesPerSimdgroup ?? 1) * (threadCount / threadWidth)
            groups = MTLSize(
                width: (Int(params.queryCount) + queriesPerGroup - 1) / queriesPerGroup,
                height: Int(params.numQHeads),
                depth: 1)
        case .tiled:
            groups = MTLSize(width: Int(params.queryCount),
                             height: Int(params.numQHeads),
                             depth: 1)
        }
        enc.dispatchThreadgroups(
            groups,
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
        enc.endEncoding()
    }


    /// `PrefillUnfusedParams` in prefill.metal.
    private struct UnfusedParams {
        var headBase: UInt32
        var scoreStride: UInt32
        var headRows: UInt32
        var ring: UInt32
    }

    /// Encodes the unfused attention, or returns false (encoding nothing)
    /// when one head's scores do not fit the scratch budget. Each 64-row
    /// block keeps `scoreStride` keys from its band start: every key for a
    /// window that hides none (the full layers pass the whole prefix), the
    /// window plus two blocks otherwise.
    private func encodeUnfused(_ pipelines: UnfusedPipelines,
                               commandBuffer: MTLCommandBuffer,
                               q: MTLBuffer, qOffset: Int,
                               k: MTLBuffer, kOffset: Int,
                               v: MTLBuffer, vOffset: Int,
                               out: MTLBuffer, outOffset: Int,
                               params: PrefillAttentionParams,
                               kvRingCapacity: UInt32) -> Bool {
        let rows = Int(params.queryCount)
        let kv = Int(params.kvValidCount)
        let window = Int(params.slidingWindow)
        var stride = roundUp(kv, toMultipleOf: 64)
        if window > 0 && window < kv {
            stride = min(stride, roundUp(window + 127, toMultipleOf: 64))
        }
        let sBytesPerHead = rows * stride * MemoryLayout<Float>.stride
        let pBytesPerHead = rows * stride * MemoryLayout<Float16>.stride
        let sumBytesPerHead = rows * MemoryLayout<Float>.stride
        let bytesPerHead = sBytesPerHead + pBytesPerHead + sumBytesPerHead
        let heads = Int(params.numQHeads)
        var group = heads
        while group > 1 && (group * bytesPerHead > Self.unfusedScratchBudget || heads % group != 0) {
            group -= 1
        }
        guard group * bytesPerHead <= Self.unfusedScratchBudget else { return false }
        if unfusedScratch == nil {
            guard let buffer = context.device.makeBuffer(length: Self.unfusedScratchBudget,
                                                         options: .storageModePrivate) else {
                return false
            }
            buffer.label = "prefill unfused attention scratch"
            unfusedScratch = buffer
        }
        guard let scratch = unfusedScratch,
              let enc = commandBuffer.makeComputeCommandEncoder() else { return false }
        let pOffset = group * sBytesPerHead
        let sumOffset = pOffset + group * pBytesPerHead
        var p = params
        let rowTiles = (rows + 63) / 64
        let threads = MTLSize(width: 128, height: 1, depth: 1)
        for headBase in Swift.stride(from: 0, to: heads, by: group) {
            var u = UnfusedParams(headBase: UInt32(headBase), scoreStride: UInt32(stride),
                                  headRows: UInt32(rows), ring: kvRingCapacity)
            enc.setComputePipelineState(pipelines.scores)
            enc.setBuffer(q, offset: qOffset, index: 0)
            enc.setBuffer(k, offset: kOffset, index: 1)
            enc.setBuffer(scratch, offset: 0, index: 2)
            enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
            enc.setBytes(&u, length: MemoryLayout<UnfusedParams>.stride, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: stride / 64, height: rowTiles, depth: group),
                                     threadsPerThreadgroup: threads)

            enc.setComputePipelineState(pipelines.softmax)
            enc.setBuffer(scratch, offset: pOffset, index: 3)
            enc.setBuffer(scratch, offset: sumOffset, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: (rows + 3) / 4, height: group, depth: 1),
                                     threadsPerThreadgroup: threads)

            enc.setComputePipelineState(pipelines.pv)
            enc.setBuffer(out, offset: outOffset, index: 0)
            enc.setBuffer(v, offset: vOffset, index: 1)
            enc.dispatchThreadgroups(
                MTLSize(width: Int(params.headDim) / 64, height: rowTiles, depth: group),
                threadsPerThreadgroup: threads)
        }
        enc.endEncoding()
        return true
    }

    private func validate(_ params: PrefillAttentionParams) {
        precondition(params.headDim > 0, "headDim must be positive")
        precondition(params.queryCount > 0, "queryCount must be positive")
        precondition(params.numQHeads > 0, "numQHeads must be positive")
        precondition(params.numKVHeads > 0, "numKVHeads must be positive")
        precondition(params.numQHeads % params.numKVHeads == 0,
                     "numQHeads must be divisible by numKVHeads")
        precondition(params.qTokenStrideElements >= params.numQHeads * params.headDim,
                     "q token stride is too small")
        precondition(params.oTokenStrideElements >= params.numQHeads * params.headDim,
                     "output token stride is too small")
        precondition(params.kvTokenStrideElements >= params.numKVHeads * params.headDim,
                     "KV token stride is too small")
        precondition(params.startPosition + params.queryCount <= params.kvValidCount,
                     "kvValidCount must include all in-flight query rows")
    }


    private func roundUp(_ value: Int, toMultipleOf multiple: Int) -> Int {
        ((value + multiple - 1) / multiple) * multiple
    }

    private func causalTiledPipeline(kvRingCapacity: UInt32) -> MTLComputePipelineState {
        guard kvRingCapacity > 0 else { return psoCausalTiled }
        do {
            return try context.pipeline(
                "attention_prefill_causal_tiled",
                constants: [MetalFunctionConstant(index: 76, value: .uint32(kvRingCapacity))])
        } catch {
            preconditionFailure("failed to build FP16 KV ring prefill attention pipeline: \(error)")
        }
    }

    private func sgmmPipeline(_ function: String, kvRingCapacity: UInt32) -> MTLComputePipelineState {
        let constants = kvRingCapacity > 0
            ? [MetalFunctionConstant(index: 76, value: .uint32(kvRingCapacity))]
            : []
        do {
            return try context.pipeline(function, constants: constants)
        } catch {
            preconditionFailure("failed to build simdgroup-matrix prefill attention pipeline: \(error)")
        }
    }

    private func qBlockPipeline(_ specialisation: QBlockSpecialisation,
                                kvRingCapacity: UInt32) -> MTLComputePipelineState {
        let constants = kvRingCapacity > 0
            ? [MetalFunctionConstant(index: 76, value: .uint32(kvRingCapacity))]
            : []
        do {
            return try context.pipeline(specialisation.function, constants: constants)
        } catch {
            preconditionFailure("failed to build query-blocked prefill attention pipeline: \(error)")
        }
    }
}
