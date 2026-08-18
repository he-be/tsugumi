import Foundation
import Metal

@frozen
public struct MoEExpertOffsets {
    public var gateWOff: UInt32
    public var gateSOff: UInt32
    public var gateBOff: UInt32
    public var upWOff: UInt32
    public var upSOff: UInt32
    public var upBOff: UInt32
    public var downWOff: UInt32
    public var downSOff: UInt32
    public var downBOff: UInt32

    public init(gateWOff: UInt32, gateSOff: UInt32, gateBOff: UInt32,
                upWOff: UInt32, upSOff: UInt32, upBOff: UInt32,
                downWOff: UInt32, downSOff: UInt32, downBOff: UInt32) {
        self.gateWOff = gateWOff
        self.gateSOff = gateSOff
        self.gateBOff = gateBOff
        self.upWOff = upWOff
        self.upSOff = upSOff
        self.upBOff = upBOff
        self.downWOff = downWOff
        self.downSOff = downSOff
        self.downBOff = downBOff
    }
}

public enum RouterError: Error, CustomStringConvertible {
    case unsupportedWeightBits(Int)

    public var description: String {
        switch self {
        case .unsupportedWeightBits(let bits):
            return "Unsupported router weight bits: \(bits) (expected 8 or 16)"
        }
    }
}

package final class MoE {
    static let maxStreamedExperts = 8
    /// Rows the shared router-logit staging is sized for. A verify block is at
    /// most `SpeculativeBlock.maxTokens` wide; decode uses row 0 alone.
    static let maxRouterRows = 8

    private static let realDecodeD: UInt32 = 2816
    private static let realDecodeF: UInt32 = 704
    private static let realDecodeTopK: UInt32 = 8
    private static let realDecodeNumExperts: UInt32 = 128
    private static let realDecodeMoEConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 0, value: .uint32(realDecodeD)),
        MetalFunctionConstant(index: 1, value: .uint32(realDecodeF)),
        MetalFunctionConstant(index: 2, value: .uint32(realDecodeTopK)),
        MetalFunctionConstant(index: 3, value: .bool(true)),
    ]
    private static let realDecodeRouterConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 40, value: .uint32(realDecodeNumExperts)),
        MetalFunctionConstant(index: 41, value: .uint32(realDecodeD)),
        MetalFunctionConstant(index: 42, value: .uint32(realDecodeTopK)),
        MetalFunctionConstant(index: 43, value: .bool(true)),
    ]

    /// Router GEMV pipelines for whichever weight format this model uses. Only
    /// one of the two kernels is ever compiled — see `routerWeightBits`.
    private let routerGemvPSO: MTLComputePipelineState
    private let routerGemvSpecializedPSO: MTLComputePipelineState
    private let routerSelectK8PSO: MTLComputePipelineState
    private let routerSelectK8SpecializedPSO: MTLComputePipelineState
    /// The same two router kernels with the block's rows in the grid, so a
    /// k-row verify block spends 2 dispatches a layer instead of 2k
    /// (docs/mtp/27-M7-RESULTS.md §5).
    private let routerGemvRowsPSO: MTLComputePipelineState?
    private let routerGemvRowsSpecializedPSO: MTLComputePipelineState?
    private let routerSelectK8RowsPSO: MTLComputePipelineState
    private let routerSelectK8RowsSpecializedPSO: MTLComputePipelineState
    private let routerLogits: MTLBuffer
    private let phase1U16PSO: MTLComputePipelineState
    private let phase1U16SpecializedPSO: MTLComputePipelineState
    private let phase1SubsetU16PSO: MTLComputePipelineState
    private let phase1SubsetU16SpecializedPSO: MTLComputePipelineState
    private let phase2ReduceK8PSO: MTLComputePipelineState
    private let phase2ReduceK8SpecializedPSO: MTLComputePipelineState
    private let routedArgEncoder: MTLArgumentEncoder
    private let reusableRoutedArgBuffer: MTLBuffer

    private let affineGroupSize: Int
    /// 8 for affine INT8 router weights, 16 for the unquantized BF16 router
    /// that the QAT checkpoints ship.
    let routerWeightBits: Int

    package init(context: MetalContext, routerWeightBits: Int = 8) throws {
        self.affineGroupSize = context.affineGroupSize
        self.routerWeightBits = routerWeightBits
        let routerName: String
        switch routerWeightBits {
        case 8:  routerName = "router_gemv_gemma4_r4"
        case 16: routerName = "router_gemv_gemma4_bf16_r4"
        default: throw RouterError.unsupportedWeightBits(routerWeightBits)
        }
        self.routerGemvPSO = try context.pipeline(
            routerName,
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)
        self.routerGemvSpecializedPSO = try context.pipeline(
            routerName,
            constants: Self.realDecodeRouterConstants,
            maxTotalThreadsPerThreadgroup: 512)
        self.routerSelectK8PSO = try context.pipeline("router_topk_select_k8")
        self.routerSelectK8SpecializedPSO = try context.pipeline(
            "router_topk_select_k8",
            constants: Self.realDecodeRouterConstants)
        // Only the BF16 router has a rows kernel: the INT8 router is not the
        // format a speculative block runs on (`rowsRouterEnabled` requires 16).
        self.routerGemvRowsPSO = routerWeightBits == 16
            ? try context.pipeline("router_gemv_gemma4_bf16_rows",
                                   constants: [],
                                   maxTotalThreadsPerThreadgroup: 512)
            : nil
        self.routerGemvRowsSpecializedPSO = routerWeightBits == 16
            ? try context.pipeline("router_gemv_gemma4_bf16_rows",
                                   constants: Self.realDecodeRouterConstants,
                                   maxTotalThreadsPerThreadgroup: 512)
            : nil
        self.routerSelectK8RowsPSO = try context.pipeline("router_topk_select_k8_rows")
        self.routerSelectK8RowsSpecializedPSO = try context.pipeline(
            "router_topk_select_k8_rows",
            constants: Self.realDecodeRouterConstants)
        self.phase1U16PSO = try context.pipeline("moe_phase1_gate_up_act_u16load")
        self.phase1U16SpecializedPSO = try context.pipeline(
            "moe_phase1_gate_up_act_u16load",
            constants: Self.realDecodeMoEConstants)
        self.phase1SubsetU16PSO = try context.pipeline("moe_phase1_gate_up_act_subset_u16load")
        self.phase1SubsetU16SpecializedPSO = try context.pipeline(
            "moe_phase1_gate_up_act_subset_u16load",
            constants: Self.realDecodeMoEConstants)
        self.phase2ReduceK8PSO = try context.pipeline("moe_phase2_down_reduce_k8")
        self.phase2ReduceK8SpecializedPSO = try context.pipeline(
            "moe_phase2_down_reduce_k8",
            constants: Self.realDecodeMoEConstants)

        guard let logits = context.device.makeBuffer(
            length: Self.maxRouterRows * 256 * MemoryLayout<Float>.stride,
            options: .storageModeShared),
              let phase1Function = try context.library.makeFunction(
                name: "moe_phase1_gate_up_act_u16load") else {
            throw MetalError.noDevice
        }
        self.routerLogits = logits
        self.routedArgEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        guard let reusable = context.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.reusableRoutedArgBuffer = reusable
    }

    package func encodeRouterGemma4(commandBuffer: MTLCommandBuffer,
                                   weights: MTLBuffer, weightsOffset: Int = 0,
                                   scales: MTLBuffer, scalesOffset: Int = 0,
                                   biases: MTLBuffer, biasesOffset: Int = 0,
                                   hidden: MTLBuffer,
                                   effectiveScale: MTLBuffer, effectiveScaleOffset: Int = 0,
                                   perExpertScale: MTLBuffer, perExpertScaleOffset: Int = 0,
                                   outIndices: MTLBuffer,
                                   outWeights: MTLBuffer,
                                   numExperts: UInt32,
                                   d: UInt32,
                                   topK: UInt32) {
        precondition(routerWeightBits == 8,
                     "INT8 router encode on a \(routerWeightBits)-bit router")
        precondition(d.isMultiple(of: UInt32(affineGroupSize)))
        // The kernel walks the row in fixed 64-element steps (32 lanes x 2).
        precondition(d.isMultiple(of: 64))
        precondition(numExperts <= 256)
        precondition(topK == UInt32(Self.maxStreamedExperts))

        var expertCount = numExperts
        var dimension = d
        let useSpecialized = numExperts == Self.realDecodeNumExperts
            && d == Self.realDecodeD
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(
                useSpecialized ? routerGemvSpecializedPSO : routerGemvPSO)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(scales, offset: scalesOffset, index: 1)
            encoder.setBuffer(biases, offset: biasesOffset, index: 2)
            encoder.setBuffer(hidden, offset: 0, index: 3)
            encoder.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 4)
            encoder.setBuffer(routerLogits, offset: 0, index: 5)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encoder.endEncoding()
        }

        encodeRouterSelect(commandBuffer: commandBuffer,
                           perExpertScale: perExpertScale,
                           perExpertScaleOffset: perExpertScaleOffset,
                           outIndices: outIndices,
                           outWeights: outWeights,
                           numExperts: numExperts,
                           useSpecialized: useSpecialized)
    }

    /// BF16 (unquantized) router weights. Same logits and top-k contract as
    /// `encodeRouterGemma4`, minus the scale/bias buffers.
    /// The offsets exist for the speculative verify block, which routes k rows
    /// through this kernel one row at a time so that its routing decisions are
    /// decode's rather than a second implementation's: the two reduce the same
    /// dot product in a different order, and a near-tie between two experts
    /// resolves differently on either side (docs/mtp/16-M4.5-PLAN.md §4).
    /// Plain decode passes zeros and is unaffected.
    package func encodeRouterGemma4BF16(commandBuffer: MTLCommandBuffer,
                                weights: MTLBuffer, weightsOffset: Int = 0,
                                hidden: MTLBuffer, hiddenOffset: Int = 0,
                                effectiveScale: MTLBuffer, effectiveScaleOffset: Int = 0,
                                perExpertScale: MTLBuffer, perExpertScaleOffset: Int = 0,
                                outIndices: MTLBuffer, outIndicesOffset: Int = 0,
                                outWeights: MTLBuffer, outWeightsOffset: Int = 0,
                                numExperts: UInt32,
                                d: UInt32,
                                topK: UInt32) {
        precondition(routerWeightBits == 16,
                     "BF16 router encode on a \(routerWeightBits)-bit router")
        precondition(numExperts <= 256)
        precondition(topK == UInt32(Self.maxStreamedExperts))

        var expertCount = numExperts
        var dimension = d
        let useSpecialized = numExperts == Self.realDecodeNumExperts
            && d == Self.realDecodeD
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(
                useSpecialized ? routerGemvSpecializedPSO : routerGemvPSO)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(hidden, offset: hiddenOffset, index: 1)
            encoder.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 2)
            encoder.setBuffer(routerLogits, offset: 0, index: 3)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.dispatchThreadgroups(
                MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encoder.endEncoding()
        }

        encodeRouterSelect(commandBuffer: commandBuffer,
                           perExpertScale: perExpertScale,
                           perExpertScaleOffset: perExpertScaleOffset,
                           outIndices: outIndices,
                           outIndicesOffset: outIndicesOffset,
                           outWeights: outWeights,
                           outWeightsOffset: outWeightsOffset,
                           numExperts: numExperts,
                           useSpecialized: useSpecialized)
    }

    /// The router for a k-row block: one GEMV dispatch and one select dispatch
    /// for the whole block instead of one pair per row.
    ///
    /// Each row keeps its own threadgroup, so the per-row reduction order — and
    /// with it the expert choice on a near-tie — is the one the per-row
    /// dispatch produced (docs/mtp/16-M4.5-PLAN.md §4). Returns false when the
    /// rows kernels are not available for this model, and the caller keeps the
    /// row loop.
    @discardableResult
    package func encodeRouterGemma4BF16Rows(commandBuffer: MTLCommandBuffer,
                                            weights: MTLBuffer, weightsOffset: Int = 0,
                                            hidden: MTLBuffer, hiddenOffset: Int = 0,
                                            hiddenStrideElements: UInt32,
                                            effectiveScale: MTLBuffer,
                                            effectiveScaleOffset: Int = 0,
                                            perExpertScale: MTLBuffer,
                                            perExpertScaleOffset: Int = 0,
                                            outIndices: MTLBuffer, outIndicesOffset: Int = 0,
                                            outWeights: MTLBuffer, outWeightsOffset: Int = 0,
                                            rowCount: Int,
                                            numExperts: UInt32,
                                            d: UInt32,
                                            topK: UInt32) -> Bool {
        precondition(routerWeightBits == 16,
                     "BF16 router encode on a \(routerWeightBits)-bit router")
        precondition(numExperts <= 256)
        precondition(topK == UInt32(Self.maxStreamedExperts))
        guard rowCount > 0, rowCount <= Self.maxRouterRows,
              let rowsPSO = routerGemvRowsPSO,
              let rowsSpecializedPSO = routerGemvRowsSpecializedPSO else { return false }

        var expertCount = numExperts
        var dimension = d
        var hiddenStride = hiddenStrideElements
        let useSpecialized = numExperts == Self.realDecodeNumExperts
            && d == Self.realDecodeD
        guard let gemv = commandBuffer.makeComputeCommandEncoder() else { return false }
        gemv.setComputePipelineState(useSpecialized ? rowsSpecializedPSO : rowsPSO)
        gemv.setBuffer(weights, offset: weightsOffset, index: 0)
        gemv.setBuffer(hidden, offset: hiddenOffset, index: 1)
        gemv.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 2)
        gemv.setBuffer(routerLogits, offset: 0, index: 3)
        gemv.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
        gemv.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 5)
        gemv.setBytes(&hiddenStride, length: MemoryLayout<UInt32>.stride, index: 6)
        gemv.dispatchThreadgroups(
            MTLSize(width: (Int(numExperts) + 3) / 4, height: rowCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        gemv.endEncoding()

        guard let select = commandBuffer.makeComputeCommandEncoder() else { return false }
        select.setComputePipelineState(
            useSpecialized ? routerSelectK8RowsSpecializedPSO : routerSelectK8RowsPSO)
        select.setBuffer(routerLogits, offset: 0, index: 0)
        select.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 1)
        select.setBuffer(outIndices, offset: outIndicesOffset, index: 2)
        select.setBuffer(outWeights, offset: outWeightsOffset, index: 3)
        select.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
        select.dispatchThreadgroups(MTLSize(width: 1, height: rowCount, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        select.endEncoding()
        return true
    }

    private func encodeRouterSelect(commandBuffer: MTLCommandBuffer,
                                    perExpertScale: MTLBuffer,
                                    perExpertScaleOffset: Int,
                                    outIndices: MTLBuffer,
                                    outIndicesOffset: Int = 0,
                                    outWeights: MTLBuffer,
                                    outWeightsOffset: Int = 0,
                                    numExperts: UInt32,
                                    useSpecialized: Bool) {
        var expertCount = numExperts
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(
                useSpecialized ? routerSelectK8SpecializedPSO : routerSelectK8PSO)
            encoder.setBuffer(routerLogits, offset: 0, index: 0)
            encoder.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 1)
            encoder.setBuffer(outIndices, offset: outIndicesOffset, index: 2)
            encoder.setBuffer(outWeights, offset: outWeightsOffset, index: 3)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    package func makeRoutedArgumentBuffer(routedBlobs: [MTLBuffer],
                                         topK: UInt32) -> MTLBuffer? {
        validate(routedBlobs: routedBlobs, topK: topK)
        guard let buffer = routedBlobs.first?.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            return nil
        }
        encodeRoutedArgumentBuffer(buffer, routedBlobs: routedBlobs)
        return buffer
    }

    func makeReusedRoutedArgumentBuffer(routedBlobs: [MTLBuffer],
                                               topK: UInt32) -> MTLBuffer {
        validate(routedBlobs: routedBlobs, topK: topK)
        encodeRoutedArgumentBuffer(reusableRoutedArgBuffer, routedBlobs: routedBlobs)
        return reusableRoutedArgBuffer
    }

    package func encodeRoutedPersistentPhase1U16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32
    ) {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        var expertCount = topK
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f)
                ? phase1U16SpecializedPSO
                : phase1U16PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(topK * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase1SubsetU16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        activeSlots: MTLBuffer,
        activeSlotIndices: [UInt32],
        activeCount: UInt32,
        d: UInt32,
        f: UInt32,
        topK: UInt32
    ) {
        guard activeCount > 0 else { return }
        validate(routedBlobs: routedBlobs, topK: topK)
        precondition(activeSlotIndices.count == Int(activeCount))
        var dimension = d
        var intermediate = f
        var expertCount = topK
        var active = activeCount
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f)
                ? phase1SubsetU16SpecializedPSO
                : phase1SubsetU16PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for slot in activeSlotIndices {
            encoder.useResource(routedBlobs[Int(slot)], usage: .read)
        }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(activeSlots, offset: 0, index: 7)
        encoder.setBytes(&active, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(activeCount * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    package func encodeRoutedPersistentPhase2Reduce(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        acts: MTLBuffer,
        routingWeights: MTLBuffer,
        residual: MTLBuffer,
        y: MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32
    ) {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f)
                ? phase2ReduceK8SpecializedPSO
                : phase2ReduceK8PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: 0, index: 2)
        encoder.setBuffer(routingWeights, offset: 0, index: 3)
        encoder.setBuffer(residual, offset: 0, index: 4)
        encoder.setBuffer(y, offset: 0, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func validate(routedBlobs: [MTLBuffer], topK: UInt32) {
        precondition(topK == UInt32(Self.maxStreamedExperts))
        precondition(routedBlobs.count == Int(topK))
    }

    private func encodeRoutedArgumentBuffer(_ buffer: MTLBuffer,
                                            routedBlobs: [MTLBuffer]) {
        routedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for (index, blob) in routedBlobs.enumerated() {
            routedArgEncoder.setBuffer(blob, offset: 0, index: index)
        }
    }

    private func useRealDecodeConstants(d: UInt32, f: UInt32) -> Bool {
        d == Self.realDecodeD && f == Self.realDecodeF
    }
}
