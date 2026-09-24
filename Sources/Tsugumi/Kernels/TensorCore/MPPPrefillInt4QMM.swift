import Foundation
import Metal

package final class MPPPrefillInt4QMM {
    package enum Path: String, Sendable {
        case affineThreadgroupF16 = "affine-threadgroup-f16"
        case affineTile64F16 = "affine-tile64-f16"
        case affineNAXF16 = "affine-nax-f16"
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
        /// `mpp_prefill_affine_nax_f16`: MLX's `affine_qmm_t_nax` ported
        /// (docs/m6-prefill/31). 64x64x64 tile, 2x2 simdgroups each holding
        /// 32x32 of C in registers, A read from device memory into fragments,
        /// B alone staged, `matmul2d` 16x32x16 on one simdgroup.
        case nax
    }

    static let tileM = 64
    static let tileN = 32
    /// The MPP tile walks K in fixed 64-element steps. The kernel indexes
    /// scale/bias by the global K position, so any group that divides the tile
    /// (32 or 64) is expressible; a group-32 tile spans two scale/bias pairs.
    static let tileK = 64

    /// `TF_PREFILL_QMM` picks the Q/K/V/O projection path so an A/B runs on one
    /// binary (docs/m6-prefill/04-GATES.md G1): `mpp` takes the threadgroup
    /// kernel for the projections only, `simdgroup` never takes this kernel.
    /// `mpp64` takes the `tile64` kernel for the projections *and* the shared
    /// MLP (`isSharedSelected`, docs/m6-prefill/17-NAX-NOT-USED.md §3 step 2).
    /// `nax` is the `mpp64` arm with the `nax` kernel, and unset is `nax`, the
    /// default since docs/m6-prefill/32 (`mpp64` was, from docs/m6-prefill/23).
    /// Where the device cannot build the kernel, all of them fall back to the
    /// simdgroup QMM.
    private static let forcedPath = ProcessInfo.processInfo.environment["TF_PREFILL_QMM"]

    /// Whether the prefill runner should offer projections to this kernel.
    package static func isSelected(affineGroupSize: Int) -> Bool {
        switch forcedPath {
        case "simdgroup": return false
        case "mpp", "mpp64", "nax", nil: return true
        default: return affineGroupSize == tileK
        }
    }

    /// Whether the prefill shared MLP should offer its three GEMMs to this
    /// kernel. The `mpp64` and `nax` arms, and unset.
    package static var isSharedSelected: Bool {
        forcedPath == nil || forcedPath == "mpp64" || forcedPath == "nax"
    }

    /// The kernel the environment asks for when the caller does not say.
    package static var selectedKernel: Kernel {
        switch forcedPath {
        case nil, "nax": .nax
        case "mpp64": .tile64
        default: .threadgroup
        }
    }

    package let kernel: Kernel
    private var pipeline: MTLComputePipelineState?

    package init(context: MetalContext, kernel: Kernel? = nil) {
        let kernel = kernel ?? Self.selectedKernel
        self.kernel = kernel
        guard Self.tileK.isMultiple(of: context.affineGroupSize) else {
            self.pipeline = nil
            return
        }
        let name = switch kernel {
        case .threadgroup: "mpp_prefill_affine_threadgroup_f16"
        case .tile64: "mpp_prefill_affine_tile64_f16"
        case .nax: "mpp_prefill_affine_nax_f16"
        }
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
              // tile64 and nax load the activations as half4.
              kernel == .threadgroup || xOffset.isMultiple(of: 8),
              let pipeline,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return .unavailable
        }
        let tileN = kernel == .threadgroup ? Self.tileN : 64

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
        switch kernel {
        case .threadgroup: return .affineThreadgroupF16
        case .tile64: return .affineTile64F16
        case .nax: return .affineNAXF16
        }
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
