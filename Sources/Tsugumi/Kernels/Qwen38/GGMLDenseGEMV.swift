import Foundation
import Metal

/// `ggml_dense.metal`: y = W x over a GGML tensor that stays in its GGUF bytes
/// (Q8_0, F16 or F32), float32 activations.
package final class GGMLDenseGEMV {
    private let q8: MTLComputePipelineState
    private let f16: MTLComputePipelineState
    private let f32: MTLComputePipelineState
    private let f16Chunk: MTLComputePipelineState
    private let f32Chunk: MTLComputePipelineState

    package init(device: MTLDevice) throws {
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
    }

    package static func supports(_ type: GGUFFile.GGMLType) -> Bool {
        type == .q8_0 || type == .f16 || type == .f32
    }

    /// Rows `row0 ..< row0 + m` of `tensor` times `x` (`n` = row width).
    package func encode(commandBuffer: MTLCommandBuffer,
                        type: GGUFFile.GGMLType,
                        weights: MTLBuffer, weightsOffset: Int,
                        x: MTLBuffer, xOffset: Int = 0,
                        y: MTLBuffer, yOffset: Int = 0,
                        m: Int, n: Int) {
        precondition(n % 32 == 0, "row width must be a multiple of 32")
        let pso: MTLComputePipelineState
        switch type {
        case .q8_0: pso = q8
        // Float rows of 1024+ read 32-element chunks per lane: 3-9x less GPU time than
        // one element per lane on the hyper-connection and router shapes. Narrower
        // rows (320) leave most lanes idle in the chunk form and keep the stride form.
        case .f16: pso = n >= 1024 ? f16Chunk : f16
        case .f32: pso = n >= 1024 ? f32Chunk : f32
        default: preconditionFailure("GGMLDenseGEMV: unsupported type \(type)")
        }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(x, offset: xOffset, index: 1)
        enc.setBuffer(y, offset: yOffset, index: 2)
        var mv = UInt32(m), nv = UInt32(n)
        enc.setBytes(&mv, length: 4, index: 3)
        enc.setBytes(&nv, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: (m + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }
}
