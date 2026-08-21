import Foundation
import Metal

/// Final RMSNorm, INT8 affine lm-head projection, and greedy argmax, for
/// Qwen3.5-MoE (`docs/qwen35moe/03-DESIGN.md` §2-8).
///
/// `LMHeadChainInt4` is the same chain one width down. It cannot be reused
/// here and it cannot simply be widened either: the production checkpoint's
/// `lm_head` is 8-bit g64 (`docs/qwen35moe/17-PHASE2-KERNELS.md` §5), and the
/// nibble unpack is not a parameter of that kernel — it is its row geometry.
///
/// What the chain buys is the same thing it buys for Gemma: the vocabulary-wide
/// logits are never written. One pass over the 508 MB table leaves a per-
/// threadgroup argmax behind, and a second, tiny dispatch folds those into one
/// token id.
///
/// **Pass the real vocabulary, not `vocab_size`.** Upstream rounds the config's
/// `vocab_size` up to 248,320 while the tokenizer defines 248,077 pieces, so
/// the last 243 rows were never trained (`docs/qwen35moe/10-MLX4BIT-AUDIT.md`
/// §3). Scoring fewer rows is the whole fix; there is no mask.
public final class QwenLMHeadChainInt8 {

    /// Vocabulary rows one threadgroup scores — one per SIMD group.
    public static let rowsPerThreadgroup = 8
    /// `[max value, token id as bits]` per threadgroup.
    static let rowSummaryStride = 2

    /// Rows the tokenizer actually defines. `ArchConfig.vocabSize` is the
    /// padded 248,320.
    public static let ornithScoredVocab = 248_077

    private let rms: RMSNorm
    private let rowGreedy: MTLComputePipelineState
    private let rowReduce: MTLComputePipelineState
    private let xNormedBuffer: MTLBuffer
    private let summariesBuffer: MTLBuffer
    private let affineGroupSize: Int
    private let maxD: Int
    private let maxVocab: Int

    public init(context: MetalContext,
                maxD: Int = 2048,
                maxVocab: Int = 248_320) throws {
        self.affineGroupSize = context.affineGroupSize
        self.maxD = maxD
        self.maxVocab = maxVocab
        self.rms = try RMSNorm(context: context)
        // Same module — and therefore the same safe-math build — as the rest of
        // the Qwen kernels (`docs/qwen35moe/17-PHASE2-KERNELS.md` §3). Nothing
        // here evaluates a transcendental, so the setting costs this chain
        // nothing; it keeps the module to one library.
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
        self.rowGreedy = try pipeline("qwen_lm_head_greedy_int8_rows_chunk_raw")
        self.rowReduce = try pipeline("qwen_lm_head_greedy_int8_rows_reduce")

        let rowGroups = (maxVocab + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
        // Both staging buffers are shared rather than private so a check can
        // score the chain stage by stage. Together they are under 300 KB
        // against the 540 MB the head reads per token, so the storage mode is
        // not what this path is paying for.
        guard let xNormedBuffer = context.device.makeBuffer(
                  length: max(maxD, 1) * MemoryLayout<Float16>.size,
                  options: .storageModeShared),
              let summariesBuffer = context.device.makeBuffer(
                  length: rowGroups * Self.rowSummaryStride * MemoryLayout<Float>.size,
                  options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.xNormedBuffer = xNormedBuffer
        self.summariesBuffer = summariesBuffer
    }

    /// The normalized activation the last `encodeGreedyDecode` projected. The
    /// chain has two numeric stages and this is the seam between them, so a
    /// failing check can say which one moved.
    public var normalizedHidden: MTLBuffer { xNormedBuffer }

    /// The per-threadgroup argmaxes of the last `encodeGreedyDecode`. Shared
    /// storage so a check can score the whole reduction, not only its final
    /// token: entry `i` is the largest logit among vocabulary rows
    /// `8i ..< 8i+8`, which covers every row that was scored.
    public var rowSummaries: MTLBuffer { summariesBuffer }

    /// How many summary slots `vocab` rows fill.
    public func rowGroupCount(vocab: Int) -> Int {
        (vocab + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
    }

    /// One hidden row in, one token id out at `outTokenOffset`.
    ///
    /// The two dispatches are separate encoders and encoders run in submission
    /// order, so the shared `xNormedBuffer` and `summariesBuffer` are never
    /// live for two rows at once — several rows may be encoded onto the same
    /// command buffer.
    @discardableResult
    public func encodeGreedyDecode(commandBuffer: MTLCommandBuffer,
                                   hidden: MTLBuffer, hiddenOffset: Int = 0,
                                   normWeight: MTLBuffer, normOffset: Int = 0,
                                   weights: MTLBuffer, weightsOffset: Int = 0,
                                   scales: MTLBuffer, scalesOffset: Int = 0,
                                   biases: MTLBuffer, biasesOffset: Int = 0,
                                   outToken: MTLBuffer, outTokenOffset: Int = 0,
                                   d: UInt32,
                                   vocab: UInt32,
                                   rmsEps: Float) -> Bool {
        guard d > 0, vocab > 0,
              Int(d) <= maxD, Int(vocab) <= maxVocab,
              Int(d) % affineGroupSize == 0,
              // The row body walks 64 elements at a time.
              d % 64 == 0,
              hiddenOffset >= 0,
              outTokenOffset >= 0, outTokenOffset % 4 == 0 else { return false }

        let rowGroups = rowGroupCount(vocab: Int(vocab))
        rms.encodeBF16W(commandBuffer: commandBuffer,
                        x: hidden, xOffset: hiddenOffset,
                        weight: normWeight, weightOffset: normOffset,
                        out: xNormedBuffer, d: d, eps: rmsEps)

        guard let head = commandBuffer.makeComputeCommandEncoder() else { return false }
        head.setComputePipelineState(rowGreedy)
        head.setBuffer(xNormedBuffer, offset: 0, index: 0)
        head.setBuffer(weights, offset: weightsOffset, index: 1)
        head.setBuffer(scales, offset: scalesOffset, index: 2)
        head.setBuffer(biases, offset: biasesOffset, index: 3)
        head.setBuffer(summariesBuffer, offset: 0, index: 4)
        var dValue = d
        var vocabValue = vocab
        head.setBytes(&dValue, length: MemoryLayout<UInt32>.size, index: 5)
        head.setBytes(&vocabValue, length: MemoryLayout<UInt32>.size, index: 6)
        head.dispatchThreadgroups(
            MTLSize(width: rowGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                           height: 1, depth: 1))
        head.endEncoding()

        guard let reduce = commandBuffer.makeComputeCommandEncoder() else { return false }
        reduce.setComputePipelineState(rowReduce)
        reduce.setBuffer(summariesBuffer, offset: 0, index: 0)
        reduce.setBuffer(outToken, offset: outTokenOffset, index: 1)
        var rowGroupValue = UInt32(rowGroups)
        reduce.setBytes(&rowGroupValue, length: MemoryLayout<UInt32>.size, index: 2)
        let width = MTLSize(width: 256, height: 1, depth: 1)
        reduce.dispatchThreads(width, threadsPerThreadgroup: width)
        reduce.endEncoding()
        return true
    }
}
