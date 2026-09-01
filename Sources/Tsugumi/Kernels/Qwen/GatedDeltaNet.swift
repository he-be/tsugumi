import Foundation
import Metal

/// Swift wrapper for `qwen_delta_rule` — the Gated DeltaNet recurrence that
/// carries 30 of Qwen3.5-MoE's 40 layers (`docs/qwen35moe/03-DESIGN.md` §2-6).
///
///     S      = g[t] * S
///     kv_mem = S . k[t]
///     delta  = (v[t] - kv_mem) * beta[t]
///     S     += outer(delta, k[t])
///     y[t]   = S . q[t]
///
/// One dispatch covers a whole chunk of one layer. The state never leaves
/// registers inside the chunk; it is read once at the top and written once at
/// the bottom. Decode is the same kernel with `seqLen == 1`.
///
/// Layouts (batch 1):
///   `q`, `k`    half  `[T, Hk, Dk]`
///   `v`         half  `[T, Hv, Dv]`
///   `g`, `beta` float `[T, Hv]`
///   state       float `[Hv, Dv, Dk]`
///   `y`         half  `[T, Hv, Dv]`
///
/// `stateIn` and `stateOut` may be the same buffer: each thread owns one
/// `[dv][16]` fragment and reads it before anything writes.
public final class GatedDeltaNet {

    /// Time-block sizes the shader ships. Phase 4 measures all three
    /// (`docs/qwen35moe/04-PHASES.md`); 32 is the omlx default for half input.
    public enum TimeBlock: Int, CaseIterable, Sendable {
        case tb16 = 16
        case tb32 = 32
        case tb48 = 48

        var functionName: String { "qwen_delta_rule_tb\(rawValue)" }
    }

    /// `linear_key_head_dim`. The staged k/q tiles are sized on it at compile
    /// time, so a checkpoint with another value needs another shader rather
    /// than another argument.
    public static let keyHeadDim = 128
    /// dv rows per threadgroup — `Dv / dvBlock` threadgroups cover one v head.
    public static let dvBlock = 32
    public static let threadsPerGroup = 256

    private struct Params {
        var seqLen:    UInt32
        var numKHeads: UInt32
        var numVHeads: UInt32
        var valueDim:  UInt32
    }

    private var pipelines: [TimeBlock: MTLComputePipelineState] = [:]

    public init(context: MetalContext) throws {
        let library = try MetalContext.moduleLibrary(device: context.device,
                                                     module: "gdn")
        for block in TimeBlock.allCases {
            guard let function = library.makeFunction(name: block.functionName) else {
                throw MetalError.missingFunction(block.functionName)
            }
            pipelines[block] = try context.device.makeComputePipelineState(function: function)
        }
    }

    /// Threadgroup memory the shader reserves for a time block, in bytes.
    /// Metal's limit is 32,768; TB=48 lands at 30,336.
    public static func threadgroupBytes(_ block: TimeBlock) -> Int {
        let kStride = keyHeadDim + 8
        let vStride = dvBlock + 8
        return block.rawValue * (2 * kStride * MemoryLayout<Float16>.size
                                 + vStride * MemoryLayout<Float16>.size
                                 + 2 * MemoryLayout<Float>.size)
    }

    @discardableResult
    public func encode(commandBuffer: MTLCommandBuffer,
                       q: MTLBuffer, qOffset: Int = 0,
                       k: MTLBuffer, kOffset: Int = 0,
                       v: MTLBuffer, vOffset: Int = 0,
                       g: MTLBuffer, gOffset: Int = 0,
                       beta: MTLBuffer, betaOffset: Int = 0,
                       stateIn: MTLBuffer, stateInOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       stateOut: MTLBuffer, stateOutOffset: Int = 0,
                       seqLen: Int,
                       numKHeads: Int,
                       numVHeads: Int,
                       keyHeadDim: Int,
                       valueHeadDim: Int,
                       timeBlock: TimeBlock = .tb32) -> Bool {
        guard seqLen > 0,
              keyHeadDim == Self.keyHeadDim,
              valueHeadDim % Self.dvBlock == 0,
              numKHeads > 0, numVHeads > 0,
              numVHeads % numKHeads == 0,
              let pipeline = pipelines[timeBlock],
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q,        offset: qOffset,        index: 0)
        encoder.setBuffer(k,        offset: kOffset,        index: 1)
        encoder.setBuffer(v,        offset: vOffset,        index: 2)
        encoder.setBuffer(g,        offset: gOffset,        index: 3)
        encoder.setBuffer(beta,     offset: betaOffset,     index: 4)
        encoder.setBuffer(stateIn,  offset: stateInOffset,  index: 5)
        encoder.setBuffer(y,        offset: yOffset,        index: 6)
        encoder.setBuffer(stateOut, offset: stateOutOffset, index: 7)
        var params = Params(seqLen: UInt32(seqLen),
                            numKHeads: UInt32(numKHeads),
                            numVHeads: UInt32(numVHeads),
                            valueDim: UInt32(valueHeadDim))
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: valueHeadDim / Self.dvBlock, height: numVHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadsPerGroup, height: 1, depth: 1))
        encoder.endEncoding()
        return true
    }
}
