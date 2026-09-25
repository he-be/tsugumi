import Foundation
import Metal
import MetalPerformanceShaders

/// `ggml_dense.metal`: y = W x over a GGML tensor that stays in its GGUF bytes
/// (Q8_0, F16 or F32), float32 activations, for T tokens at once.
/// `ggml_iq.metal` adds the K / IQ types of the Qwen3.8-27B GGUF (docs/qwen38-27b/01 §3-1, 03),
/// with both the direct and the dequant kernels.
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
    private let iqDequant: [GGUFFile.GGMLType: MTLComputePipelineState]
    /// PQ2_0 has its own GEMV (x held in registers across the group's rows): one column reads
    /// 32 weights per lane over 4 rows, more columns 16 weights over 8 rows (docs/qwen38-27b/14 §1).
    private let pq2Rows4: MTLComputePipelineState
    private let pq2Rows8W16: MTLComputePipelineState
    /// Token count from which the dequant + sgemm path runs (`Q38_MPS_MIN_T`, 0 = never).
    package var mpsMinTokens = Int(ProcessInfo.processInfo.environment["Q38_MPS_MIN_T"] ?? "") ?? 32
    /// Tensors above this many weights stay on the direct kernels (the LM head).
    package var mpsMaxWeights = 64 << 20
    private var scratch: MTLBuffer?
    /// The half products (`encodeHalf`): conversions and the Q8_0 dequant to half, set by `useHalf`.
    private var toHalf, toFloat, q8Half: MTLComputePipelineState?
    private var xHalf, wHalf, yHalf: MTLBuffer?
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
        var iqDequant: [GGUFFile.GGMLType: MTLComputePipelineState] = [:]
        for (type, name) in Self.iqKernels {
            iq[type] = try iqPSO("ggml_\(name)_gemv")
            iqDequant[type] = try iqPSO("ggml_\(name)_dequant_f32")
        }
        self.iq = iq
        self.iqDequant = iqDequant
        pq2Rows4 = try iqPSO("ggml_pq2_0_gemv_r4")
        pq2Rows8W16 = try iqPSO("ggml_pq2_0_gemv_r8w16")
    }

    /// Kernel name stem in `ggml_iq.metal` (`ggml_<stem>_gemv`, `ggml_<stem>_dequant_f32`).
    private static let iqKernels: [GGUFFile.GGMLType: String] = [
        .q2_K: "q2_K", .q4_K: "q4_K", .q6_K: "q6_K",
        .iq2_xxs: "iq2_xxs", .iq2_xs: "iq2_xs", .iq2_s: "iq2_s",
        .iq3_xxs: "iq3_xxs", .iq3_s: "iq3_s", .iq1_m: "iq1_m", .iq4_xs: "iq4_xs",
        .pq2_0: "pq2_0",
    ]

    /// Weights per block: the IQ / K types are 256, Prism's PQ2_0 is 128.
    private static func elementsPerBlock(_ type: GGUFFile.GGMLType) -> Int {
        type == .pq2_0 ? 128 : 256
    }

    package static func supports(_ type: GGUFFile.GGMLType) -> Bool {
        type == .q8_0 || type == .f16 || type == .f32 || type == .bf16 || iqKernels[type] != nil
    }

    /// `y[t] = W x[t]` for `t < tokens`: `x` holds `tokens * n` floats, `y` `tokens * m`.
    package func encode(commandBuffer: MTLCommandBuffer,
                        type: GGUFFile.GGMLType,
                        weights: MTLBuffer, weightsOffset: Int,
                        x: MTLBuffer, xOffset: Int = 0,
                        y: MTLBuffer, yOffset: Int = 0,
                        m: Int, n: Int, tokens: Int = 1) {
        precondition(n % 32 == 0, "row width must be a multiple of 32")
        if mpsMinTokens > 0 && tokens >= mpsMinTokens && m * n <= mpsMaxWeights {
            encodeSgemm(commandBuffer: commandBuffer, type: type, weights: weights, weightsOffset: weightsOffset,
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: m, n: n, tokens: tokens)
            return
        }
        let pso: MTLComputePipelineState
        var rowsPerThreadgroup = 8
        switch type {
        case .q8_0: pso = q8
        // Float rows of 1024+ read 32-element chunks per lane: 3-9x less GPU time than
        // one element per lane on the hyper-connection and router shapes. Narrower
        // rows (320) leave most lanes idle in the chunk form and keep the stride form.
        case .f16: pso = n >= 1024 ? f16Chunk : f16
        case .f32: pso = n >= 1024 ? f32Chunk : f32
        // BF16 reads back the F32 bits exactly (docs/qwen38/15 §2 W-4), same forms as F32.
        case .bf16: pso = n >= 1024 ? bf16Chunk : bf16
        case .pq2_0:
            precondition(n % 128 == 0, "pq2_0 row width must be a multiple of 128")
            pso = tokens == 1 ? pq2Rows4 : pq2Rows8W16
            rowsPerThreadgroup = tokens == 1 ? 8 : 16
        default:
            guard let kernel = iq[type] else { preconditionFailure("GGMLDenseGEMV: unsupported type \(type)") }
            let epb = Self.elementsPerBlock(type)
            precondition(n % epb == 0, "\(type) row width must be a multiple of \(epb)")
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
        enc.dispatchThreadgroups(MTLSize(width: (m + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                                         height: tokens, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Pipelines of `encodeHalf` (the `qwen38_nax` module: `q38_p1_f32_to_f16`, `q38_p1_f16_to_f32`, `q38_p1_q8_0_dequant_f16`).
    package func useHalf(toHalf: MTLComputePipelineState, toFloat: MTLComputePipelineState, q8Half: MTLComputePipelineState) {
        self.toHalf = toHalf
        self.toFloat = toFloat
        self.q8Half = q8Half
    }

    /// `y = W x` as `encode`, through MPS fp16 (P1 of `docs/qfn-prefill/15`, arm C): x to half, Q8_0 rows dequantized
    /// to half (F16 rows used as they are), the half product, and the result back to float32. Q8_0 / F16 only.
    package func encodeHalf(commandBuffer cb: MTLCommandBuffer, type: GGUFFile.GGMLType,
                            weights: MTLBuffer, weightsOffset: Int,
                            x: MTLBuffer, xOffset: Int = 0, y: MTLBuffer, yOffset: Int = 0,
                            m: Int, n: Int, tokens: Int) {
        precondition(type == .q8_0 || type == .f16, "encodeHalf: \(type)")
        precondition(n % 32 == 0 && m % 4 == 0 && xOffset % 16 == 0 && yOffset % 16 == 0)
        func ensure(_ b: inout MTLBuffer?, _ bytes: Int) {
            if (b?.length ?? 0) < bytes { b = device.makeBuffer(length: bytes, options: .storageModePrivate) }
        }
        ensure(&xHalf, tokens * n * 2)
        ensure(&yHalf, tokens * m * 2)
        func convert(_ pso: MTLComputePipelineState, _ src: MTLBuffer, _ srcOffset: Int, _ dst: MTLBuffer, _ dstOffset: Int,
                     _ count: Int) {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso)
            enc.setBuffer(src, offset: srcOffset, index: 0)
            enc.setBuffer(dst, offset: dstOffset, index: 1)
            var n4 = UInt32(count / 4)
            enc.setBytes(&n4, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: count / 4, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        convert(toHalf!, x, xOffset, xHalf!, 0, tokens * n)
        var wBuffer = weights, wOffset = weightsOffset
        if type == .q8_0 {
            ensure(&wHalf, m * n * 2)
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(q8Half!)
            enc.setBuffer(weights, offset: weightsOffset, index: 0)
            enc.setBuffer(wHalf, offset: 0, index: 1)
            var nv = UInt32(n)
            enc.setBytes(&nv, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: n / 32, height: m, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(32, n / 32), height: 8, depth: 1))
            enc.endEncoding()
            wBuffer = wHalf!
            wOffset = 0
        }
        let key = [tokens, m, n, 2]
        let mul = multiplications[key] ?? MPSMatrixMultiplication(
            device: device, transposeLeft: false, transposeRight: true,
            resultRows: tokens, resultColumns: m, interiorColumns: n, alpha: 1, beta: 0)
        multiplications[key] = mul
        func half(_ b: MTLBuffer, _ offset: Int, _ r: Int, _ c: Int) -> MPSMatrix {
            MPSMatrix(buffer: b, offset: offset, descriptor: MPSMatrixDescriptor(rows: r, columns: c, rowBytes: c * 2, dataType: .float16))
        }
        mul.encode(commandBuffer: cb, leftMatrix: half(xHalf!, 0, tokens, n), rightMatrix: half(wBuffer, wOffset, m, n),
                   resultMatrix: half(yHalf!, 0, tokens, m))
        convert(toFloat!, yHalf!, 0, y, yOffset, tokens * m)
    }

    /// Rows `[m, n]` of a Q8_0 / F16 / BF16 / K / IQ tensor -> float32 `[m][n]` at `out` (the token embedding rows
    /// of the Qwen3.8-27B runner).
    package func encodeDequant(commandBuffer: MTLCommandBuffer, type: GGUFFile.GGMLType,
                               weights: MTLBuffer, weightsOffset: Int,
                               out: MTLBuffer, outOffset: Int = 0, m: Int, n: Int) {
        precondition(n % 32 == 0 && type != .f32)
        let enc = commandBuffer.makeComputeCommandEncoder()!
        enc.setComputePipelineState(iqDequant[type] ?? (type == .q8_0 ? q8Dequant : type == .bf16 ? bf16Dequant : f16Dequant))
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var nv = UInt32(n)
        enc.setBytes(&nv, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: n / 32, height: m, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(32, n / 32), height: min(16, m), depth: 1))
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
            enc.setComputePipelineState(iqDequant[type] ?? (type == .q8_0 ? q8Dequant : type == .bf16 ? bf16Dequant : f16Dequant))
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
