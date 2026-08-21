import Foundation
import Metal

/// `Y[T, N] = X[T, K] * W[N, K]^T` for an 8-bit affine weight
/// (`docs/qwen35moe/04-PHASES.md` Phase 4).
///
/// `PrefillInt4QMM` is the same product one width down and cannot be widened —
/// the nibble unpack is the row geometry, not an argument, the same reason
/// `QwenLMHeadChainInt8` and `QwenEmbedLookupInt8` exist
/// (`docs/qwen35moe/19-LM-HEAD-INT8.md`). On the production checkpoint the
/// prefill path needs both widths in the same layer: `in_proj_{z,a,b}`,
/// `out_proj` and the whole shared expert are 8-bit, while `in_proj_qkv` and
/// `self_attn.*` mix 4-bit and 8-bit layer by layer
/// (`docs/qwen35moe/18-MIXED-BITS.md` §3).
///
/// Both kernels live in `qwen.metal` rather than `prefill.metal` so that the
/// Gemma 4 measurements stay frozen (`docs/qwen35moe/README.md`, operating
/// rules). Nothing here is Qwen-specific beyond that.
public final class QwenPrefillInt8QMM {
    /// Which kernel served a call. `encode` returns it so a check can assert
    /// the path it meant to exercise actually ran.
    public enum Path: String, Sendable {
        /// One thread per (token, row); the K reduction stays in FP32.
        case scalarBlock = "scalar-block"
        /// 64x64 output tile per threadgroup, 8x8 `simdgroup_matrix` products,
        /// dequantized weights staged as FP16.
        case simdgroupMatrix = "simdgroup-matrix"
    }

    public static let tileM = 64
    public static let tileN = 64
    public static let tileK = 32
    public static let threadsPerGroup = 128

    /// `TF_QWEN_QMM=scalar` forces the scalar kernel. It is the way back to
    /// FP32 weight arithmetic — the tiled path stages its dequantized weights
    /// as FP16, one rounding per weight — which is what separates a bug in the
    /// kernel from the staging error. Anything else, including unset, takes the
    /// tiled path.
    private static let forcedPath = ProcessInfo.processInfo.environment["TF_QWEN_QMM"]

    private let affineGroupSize: Int
    private let scalarPSO: MTLComputePipelineState
    private let tiledPSO: MTLComputePipelineState?

    public init(context: MetalContext) throws {
        self.affineGroupSize = context.affineGroupSize
        let library = try MetalContext.moduleLibrary(device: context.device,
                                                     module: "qwen",
                                                     affineGroupSize: context.affineGroupSize,
                                                     safeMath: true)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw MetalError.missingFunction(name)
            }
            return try context.device.makeComputePipelineState(function: function)
        }
        self.scalarPSO = try pipeline("qwen_int8_qmm_f16_block")
        if Self.forcedPath == "scalar" {
            self.tiledPSO = nil
        } else {
            // The tile shape fixes the threadgroup at 128 threads; a build
            // where register pressure caps it lower cannot run this kernel.
            let candidate = try? pipeline("qwen_int8_qmm_simdgroup_f16")
            self.tiledPSO = (candidate?.maxTotalThreadsPerThreadgroup ?? 0) >= Self.threadsPerGroup
                ? candidate
                : nil
        }
    }

    /// The tiled kernel walks K in 32-element steps and reads one scale/bias
    /// pair per 8-wide chunk, so K must be aligned to the tile and to the
    /// affine group. A single-row product (`N == 1`, the shared expert gate)
    /// would throw away 63 of 64 tile columns, so it goes scalar too.
    public func usesTiledPath(t: Int, n: Int, k: Int) -> Bool {
        tiledPSO != nil && k % Self.tileK == 0 && k % affineGroupSize == 0
            && n >= Self.tileN && t >= 8
    }

    @discardableResult
    public func encode(commandBuffer: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       t: Int,
                       n: Int,
                       k: Int,
                       forcedPath: Path? = nil) -> Path {
        precondition(k % affineGroupSize == 0,
                     "K must be a multiple of \(affineGroupSize)")
        // `forcedPath` exists for the check, which runs the same inputs through
        // both kernels and scores them against each other: the scalar one keeps
        // the weights in FP32, so a disagreement is the tiling, not the affine
        // arithmetic. The tiled path is still refused for a shape it cannot
        // serve.
        let tiled: Bool
        switch forcedPath {
        case .scalarBlock: tiled = false
        case .simdgroupMatrix: tiled = tiledPSO != nil && k % Self.tileK == 0
        case nil: tiled = usesTiledPath(t: t, n: n, k: k)
        }
        guard t > 0, n > 0, k > 0,
              let enc = commandBuffer.makeComputeCommandEncoder() else {
            return tiled ? .simdgroupMatrix : .scalarBlock
        }
        enc.setComputePipelineState(tiled ? tiledPSO! : scalarPSO)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(x, offset: xOffset, index: 3)
        enc.setBuffer(y, offset: yOffset, index: 4)
        var tVar = UInt32(t)
        var nVar = UInt32(n)
        var kVar = UInt32(k)
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 7)
        if tiled {
            enc.dispatchThreadgroups(
                MTLSize(width: (n + Self.tileN - 1) / Self.tileN,
                        height: (t + Self.tileM - 1) / Self.tileM,
                        depth: 1),
                threadsPerThreadgroup: MTLSize(width: Self.threadsPerGroup, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(
                MTLSize(width: (n + 7) / 8, height: (t + 7) / 8, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        }
        enc.endEncoding()
        return tiled ? .simdgroupMatrix : .scalarBlock
    }
}
