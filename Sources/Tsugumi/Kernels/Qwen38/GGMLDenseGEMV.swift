import Foundation
import Metal
import MetalPerformanceShaders

/// `ggml_dense.metal`: y = W x over a GGML tensor that stays in its GGUF bytes
/// (Q8_0, F16 or F32), float32 activations, for T tokens at once.
/// `ggml_iq.metal` adds the IQ / K types of the Qwen3.8-27B GGUF (docs/qwen38-27b/01 §3-1);
/// those have no dequant kernel yet and always take the direct kernels.
///
/// Up to `mpsMinTokens` tokens the kernels read the GGUF bytes directly: at one
/// token the cost is the weight read and those kernels sit at the memory
/// bandwidth. From there on the tensor is dequantized to float32 into a scratch
/// buffer and multiplied with MPS sgemm, which is compute-bound and several times
/// faster (attn_qkv, T = 512: 1.3 + 5.4 ms against 45 ms; T = 16 is the crossover).
package final class GGMLDenseGEMV {
    private let device: MTLDevice
    private let q8: MTLComputePipelineState
    private let f16: MTLComputePipelineState
    private let f32: MTLComputePipelineState
    private let f16Chunk: MTLComputePipelineState
    private let f32Chunk: MTLComputePipelineState
    private let bf16: MTLComputePipelineState
    private let bf16Chunk: MTLComputePipelineState
    private let bf16Dequant: MTLComputePipelineState
    private let q8Dequant: MTLComputePipelineState
    private let f16Dequant: MTLComputePipelineState
    private let iq: [GGUFFile.GGMLType: MTLComputePipelineState]
    /// Token count from which the dequant + sgemm path runs (`Q38_MPS_MIN_T`, 0 = never).
    package var mpsMinTokens = Int(ProcessInfo.processInfo.environment["Q38_MPS_MIN_T"] ?? "") ?? 32
    /// Tensors above this many weights stay on the direct kernels (the LM head).
    package var mpsMaxWeights = 64 << 20
    private var scratch: MTLBuffer?
    package var scratchBytes: Int { scratch?.length ?? 0 }
    package func dropScratch() { scratch = nil }
    private var multiplications: [[Int]: MPSMatrixMultiplication] = [:]

    package init(device: MTLDevice) throws {
        self.device = device
        let library = try MetalContext.moduleLibrary(device: device, module: "ggml_dense")
        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw GGUFFile.Error.format("ggml_dense: \(name) missing")
            }
            return try device.makeComputePipelineState(function: fn)
        }
        q8 = try pso("ggml_q8_0_gemv")
        f16 = try pso("ggml_f16_gemv")
        f32 = try pso("ggml_f32_gemv")
        f16Chunk = try pso("ggml_f16_gemv_chunk")
        f32Chunk = try pso("ggml_f32_gemv_chunk")
        q8Dequant = try pso("ggml_q8_0_dequant_f32")
        f16Dequant = try pso("ggml_f16_dequant_f32")
        bf16 = try pso("ggml_bf16_gemv")
        bf16Chunk = try pso("ggml_bf16_gemv_chunk")
        bf16Dequant = try pso("ggml_bf16_dequant_f32")
        let iqLibrary = try MetalContext.moduleLibrary(device: device, module: "ggml_iq")
        func iqPSO(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = iqLibrary.makeFunction(name: name) else {
                throw GGUFFile.Error.format("ggml_iq: \(name) missing")
            }
            return try device.makeComputePipelineState(function: fn)
        }
        var iq: [GGUFFile.GGMLType: MTLComputePipelineState] = [:]
        for (type, name) in Self.iqKernels { iq[type] = try iqPSO(name) }
        self.iq = iq
    }

    private static let iqKernels: [GGUFFile.GGMLType: String] = [
        .q2_K: "ggml_q2_K_gemv", .q4_K: "ggml_q4_K_gemv", .q6_K: "ggml_q6_K_gemv",
        .iq2_xxs: "ggml_iq2_xxs_gemv", .iq2_xs: "ggml_iq2_xs_gemv", .iq2_s: "ggml_iq2_s_gemv",
        .iq3_xxs: "ggml_iq3_xxs_gemv", .iq3_s: "ggml_iq3_s_gemv",
        .iq1_m: "ggml_iq1_m_gemv", .iq4_xs: "ggml_iq4_xs_gemv",
    ]

    package static func supports(_ type: GGUFFile.GGMLType) -> Bool {
        hasDequant(type) || iqKernels[type] != nil
    }

    private static func hasDequant(_ type: GGUFFile.GGMLType) -> Bool {
        type == .q8_0 || type == .f16 || type == .f32 || type == .bf16
    }

    /// `y[t] = W x[t]` for `t < tokens`: `x` holds `tokens * n` floats, `y` `tokens * m`.
    package func encode(commandBuffer: MTLCommandBuffer,
                        type: GGUFFile.GGMLType,
                        weights: MTLBuffer, weightsOffset: Int,
                        x: MTLBuffer, xOffset: Int = 0,
                        y: MTLBuffer, yOffset: Int = 0,
                        m: Int, n: Int, tokens: Int = 1) {
        precondition(n % 32 == 0, "row width must be a multiple of 32")
        if mpsMinTokens > 0 && tokens >= mpsMinTokens && m * n <= mpsMaxWeights && Self.hasDequant(type) {
            encodeSgemm(commandBuffer: commandBuffer, type: type, weights: weights, weightsOffset: weightsOffset,
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: m, n: n, tokens: tokens)
            return
        }
        let pso: MTLComputePipelineState
        switch type {
        case .q8_0: pso = q8
        // Float rows of 1024+ read 32-element chunks per lane: 3-9x less GPU time than
        // one element per lane on the hyper-connection and router shapes. Narrower
        // rows (320) leave most lanes idle in the chunk form and keep the stride form.
        case .f16: pso = n >= 1024 ? f16Chunk : f16
        case .f32: pso = n >= 1024 ? f32Chunk : f32
        // BF16 reads back the F32 bits exactly (docs/qwen38/15 §2 W-4), same forms as F32.
        case .bf16: pso = n >= 1024 ? bf16Chunk : bf16
        default:
            guard let kernel = iq[type] else { preconditionFailure("GGMLDenseGEMV: unsupported type \(type)") }
            precondition(n % 256 == 0, "\(type) row width must be a multiple of 256")
            pso = kernel
        }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(x, offset: xOffset, index: 1)
        enc.setBuffer(y, offset: yOffset, index: 2)
        var mv = UInt32(m), nv = UInt32(n)
        enc.setBytes(&mv, length: 4, index: 3)
        enc.setBytes(&nv, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: (m + 7) / 8, height: tokens, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeSgemm(commandBuffer: MTLCommandBuffer, type: GGUFFile.GGMLType,
                             weights: MTLBuffer, weightsOffset: Int,
                             x: MTLBuffer, xOffset: Int, y: MTLBuffer, yOffset: Int,
                             m: Int, n: Int, tokens: Int) {
        var wBuffer = weights, wOffset = weightsOffset
        if type != .f32 {
            if (scratch?.length ?? 0) < m * n * 4 {
                scratch = device.makeBuffer(length: m * n * 4, options: .storageModePrivate)
            }
            let enc = commandBuffer.makeComputeCommandEncoder()!
            enc.setComputePipelineState(type == .q8_0 ? q8Dequant : type == .bf16 ? bf16Dequant : f16Dequant)
            enc.setBuffer(weights, offset: weightsOffset, index: 0)
            enc.setBuffer(scratch, offset: 0, index: 1)
            var nv = UInt32(n)
            enc.setBytes(&nv, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: n / 32, height: m, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(32, n / 32), height: 16, depth: 1))
            enc.endEncoding()
            wBuffer = scratch!
            wOffset = 0
        }
        let key = [tokens, m, n]
        let mul = multiplications[key] ?? MPSMatrixMultiplication(
            device: device, transposeLeft: false, transposeRight: true,
            resultRows: tokens, resultColumns: m, interiorColumns: n, alpha: 1, beta: 0)
        multiplications[key] = mul
        let xm = MPSMatrix(buffer: x, offset: xOffset,
                           descriptor: MPSMatrixDescriptor(rows: tokens, columns: n, rowBytes: n * 4, dataType: .float32))
        let wm = MPSMatrix(buffer: wBuffer, offset: wOffset,
                           descriptor: MPSMatrixDescriptor(rows: m, columns: n, rowBytes: n * 4, dataType: .float32))
        let ym = MPSMatrix(buffer: y, offset: yOffset,
                           descriptor: MPSMatrixDescriptor(rows: tokens, columns: m, rowBytes: m * 4, dataType: .float32))
        mul.encode(commandBuffer: commandBuffer, leftMatrix: xm, rightMatrix: wm, resultMatrix: ym)
    }
}
