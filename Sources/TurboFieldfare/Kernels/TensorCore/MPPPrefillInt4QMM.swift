import Foundation
import Metal

final class MPPPrefillInt4QMM {
    enum Path: String, Sendable {
        case affineThreadgroupF16 = "affine-threadgroup-f16"
        case unavailable
    }

    static let tileM = 64
    static let tileN = 32
    /// The MPP tile walks K in fixed 64-element steps, so this path only
    /// applies to group-64 models. A group-32 model has two scale/bias pairs
    /// per tile and cannot be expressed by this kernel — the wrapper reports
    /// `.unavailable` rather than silently reading the wrong scales.
    static let tileK = 64

    private var pipeline: MTLComputePipelineState?

    init(context: MetalContext) {
        guard context.affineGroupSize == Self.tileK else {
            self.pipeline = nil
            return
        }
        do {
            let library = try Self.compileTensorOpsLibrary(
                device: context.device,
                affineGroupSize: context.affineGroupSize)
            guard let function = library.makeFunction(
                name: "mpp_prefill_affine_threadgroup_f16") else {
                throw MetalError.missingFunction("mpp_prefill_affine_threadgroup_f16")
            }
            self.pipeline = try context.device.makeComputePipelineState(function: function)
        } catch {
            self.pipeline = nil
        }
    }

    var isAvailable: Bool {
        pipeline != nil
    }

    @discardableResult
    func encode(commandBuffer: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       m: Int,
                       n: Int,
                       k: Int) -> Path {
        guard m > 0,
              n > 0,
              k > 0,
              k.isMultiple(of: Self.tileK),
              weightsOffset >= 0,
              scalesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              biasesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              xOffset.isMultiple(of: MemoryLayout<Float16>.stride),
              yOffset.isMultiple(of: MemoryLayout<Float16>.stride),
              let pipeline,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return .unavailable
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = UInt32(m)
        var nValue = UInt32(n)
        var kValue = UInt32(k)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + Self.tileN - 1) / Self.tileN,
                    height: (m + Self.tileM - 1) / Self.tileM,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth * 4,
                                           height: 1,
                                           depth: 1))
        encoder.endEncoding()
        return .affineThreadgroupF16
    }

    private static func compileTensorOpsLibrary(device: MTLDevice,
                                                affineGroupSize: Int) throws -> MTLLibrary {
        try MetalContext.moduleLibrary(device: device,
                                       module: "tensorops",
                                       affineGroupSize: affineGroupSize)
    }
}
