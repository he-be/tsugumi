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

    private let context: MetalContext
    private let psoCausalTiled: MTLComputePipelineState
    private let psoFullTensorOps2DValidityV2: MTLComputePipelineState?

    init(context: MetalContext) throws {
        self.context = context
        self.psoCausalTiled = try context.pipeline("attention_prefill_causal_tiled")
        self.psoFullTensorOps2DValidityV2 = context.device.supportsFamily(.apple10)
            ? try? context.pipeline("attention_prefill_full_tensorops_2d_validity_v2")
            : nil
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
        enum Variant { case tensorOps, qBlock, tiled }
        let variant: Variant
        let pipeline: MTLComputePipelineState
        var qBlock: QBlockSpecialisation?
        if let tensorOpsPipeline {
            variant = .tensorOps
            pipeline = tensorOpsPipeline
        } else if tensorOpsShape && path == .fullTensorOps2DValidityV2 {
            preconditionFailure(
                "TensorOps 2D prefill attention requires Apple10 MPP tensor support")
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
