import Foundation
import Metal

package final class MPPPrefillInt4QMM {
    package enum Path: String, Sendable {
        case affineThreadgroupF16 = "affine-threadgroup-f16"
        case affineTile64F16 = "affine-tile64-f16"
        case unavailable
    }

    /// Which `matmul2d` kernel this instance dispatches.
    package enum Kernel: String, Sendable {
        /// `mpp_prefill_affine_threadgroup_f16`: 64x32 tile, A read from
        /// device memory, a `multiply` per K step added by hand (G1).
        case threadgroup
        /// `mpp_prefill_affine_tile64_f16`: 64x64 tile, A and B staged in
        /// threadgroup memory, one `multiply_accumulate` destination
        /// (docs/m6-prefill/12-G3-MOE-MATMUL2D.md's routed kernel, dense).
        case tile64
    }

    static let tileM = 64
    static let tileN = 32
    /// The MPP tile walks K in fixed 64-element steps. The kernel indexes
    /// scale/bias by the global K position, so any group that divides the tile
    /// (32 or 64) is expressible; a group-32 tile spans two scale/bias pairs.
    static let tileK = 64

    /// `TF_PREFILL_QMM` picks the Q/K/V/O projection path so an A/B runs on one
    /// binary (docs/m6-prefill/04-GATES.md G1): `mpp` takes this kernel at
    /// any supported group size, `simdgroup` never takes it. Unset (or any
    /// other value) keeps the default, which is this kernel for group-64
    /// models only — group 32 stays on the simdgroup path until its numbers
    /// are in.
    /// `mpp64` takes the `tile64` kernel for the projections *and* the shared
    /// MLP (`isSharedSelected`), so one arm covers step 2 of
    /// docs/m6-prefill/17-NAX-NOT-USED.md §3.
    private static let forcedPath = ProcessInfo.processInfo.environment["TF_PREFILL_QMM"]

    /// Whether the prefill runner should offer projections to this kernel.
    package static func isSelected(affineGroupSize: Int) -> Bool {
        switch forcedPath {
        case "mpp", "mpp64": return true
        case "simdgroup": return false
        default: return affineGroupSize == tileK
        }
    }

    /// Whether the prefill shared MLP should offer its three GEMMs to this
    /// kernel. Only the `mpp64` arm; unset keeps the simdgroup QMM.
    package static var isSharedSelected: Bool { forcedPath == "mpp64" }

    /// The kernel the environment asks for when the caller does not say.
    package static var selectedKernel: Kernel { forcedPath == "mpp64" ? .tile64 : .threadgroup }

    package let kernel: Kernel
    private var pipeline: MTLComputePipelineState?

    package init(context: MetalContext, kernel: Kernel? = nil) {
        let kernel = kernel ?? Self.selectedKernel
        self.kernel = kernel
        guard Self.tileK.isMultiple(of: context.affineGroupSize) else {
            self.pipeline = nil
            return
        }
        let name = kernel == .tile64
            ? "mpp_prefill_affine_tile64_f16"
            : "mpp_prefill_affine_threadgroup_f16"
        do {
            let library = try Self.compileTensorOpsLibrary(
                device: context.device,
                affineGroupSize: context.affineGroupSize,
                affineScheme: context.affineScheme)
            guard let function = library.makeFunction(name: name) else {
                throw MetalError.missingFunction(name)
            }
            self.pipeline = try context.device.makeComputePipelineState(function: function)
        } catch {
            self.pipeline = nil
        }
    }

    package var isAvailable: Bool {
        pipeline != nil
    }

    @discardableResult
    package func encode(commandBuffer: MTLCommandBuffer,
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
              // tile64 loads the activations as half4.
              kernel == .threadgroup || xOffset.isMultiple(of: 8),
              let pipeline,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return .unavailable
        }
        let tileN = kernel == .tile64 ? 64 : Self.tileN

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
            MTLSize(width: (n + tileN - 1) / tileN,
                    height: (m + Self.tileM - 1) / Self.tileM,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth * 4,
                                           height: 1,
                                           depth: 1))
        encoder.endEncoding()
        return kernel == .tile64 ? .affineTile64F16 : .affineThreadgroupF16
    }

    /// `tensorops` is compiled outside the combined runtime library, so it has
    /// to be handed the same two whole-model constants the combined one bakes
    /// in. Missing the scheme here would leave this path loading a bias array
    /// that a `sym` model does not have.
    private static func compileTensorOpsLibrary(device: MTLDevice,
                                                affineGroupSize: Int,
                                                affineScheme: Quantization.AffineScheme) throws
        -> MTLLibrary {
        try MetalContext.moduleLibrary(device: device,
                                       module: "tensorops",
                                       affineGroupSize: affineGroupSize,
                                       affineScheme: affineScheme)
    }
}
