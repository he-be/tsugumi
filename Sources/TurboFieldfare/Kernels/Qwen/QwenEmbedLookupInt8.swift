import Foundation
import Metal

/// 8-bit affine embedding lookup for Qwen3.5-MoE.
///
/// `EmbedLookupInt4` is the same lookup one width down and cannot be widened:
/// the nibble unpack is the row geometry, not an argument — the same reason
/// `QwenLMHeadChainInt8` exists rather than a parameter on `LMHeadChainInt4`
/// (`docs/qwen35moe/19-LM-HEAD-INT8.md`). On the production checkpoint both
/// ends of the model are 8-bit (`docs/qwen35moe/18-MIXED-BITS.md` §3).
///
/// There is no `outScale`. Gemma multiplies the embedded row by
/// `sqrt(hidden_size)`; this family does not, and an argument that must be
/// passed as 1.0 is a way to be quietly wrong (`03-DESIGN.md` §2-9).
public final class QwenEmbedLookupInt8 {
    private let pso: MTLComputePipelineState
    private let blockPSO: MTLComputePipelineState
    private let affineGroupSize: Int

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
        self.pso = try pipeline("qwen_embed_lookup_int8")
        self.blockPSO = try pipeline("qwen_embed_lookup_int8_block")
    }

    /// `table`, `scales` and `biases` are normally the one resident buffer with
    /// three offsets into it.
    @discardableResult
    public func encode(commandBuffer: MTLCommandBuffer,
                       table: MTLBuffer, tableOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       tokenId: UInt32,
                       d: UInt32) -> Bool {
        guard d > 0, Int(d) % affineGroupSize == 0,
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(table, offset: tableOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var token = tokenId
        var width = d
        encoder.setBytes(&token, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.size, index: 5)
        let threads = min(pso.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: Int(d), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
        return true
    }

    /// The same lookup for a whole prefill chunk: `out[t, :]` is the row for
    /// `tokens[t]`. `tokens` is a `[T]` UInt32 buffer.
    @discardableResult
    public func encodeBlock(commandBuffer: MTLCommandBuffer,
                            table: MTLBuffer, tableOffset: Int = 0,
                            scales: MTLBuffer, scalesOffset: Int = 0,
                            biases: MTLBuffer, biasesOffset: Int = 0,
                            out: MTLBuffer, outOffset: Int = 0,
                            tokens: MTLBuffer, tokensOffset: Int = 0,
                            d: UInt32,
                            seqLen: Int) -> Bool {
        guard d > 0, seqLen > 0, Int(d) % affineGroupSize == 0,
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }
        encoder.setComputePipelineState(blockPSO)
        encoder.setBuffer(table, offset: tableOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        encoder.setBuffer(tokens, offset: tokensOffset, index: 4)
        var width = d
        var t = UInt32(seqLen)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 6)
        let threads = min(blockPSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: Int(d), height: seqLen, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
        return true
    }
}
