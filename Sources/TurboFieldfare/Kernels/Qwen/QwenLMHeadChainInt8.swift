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
    /// Hidden rows one pass over the 508 MB table may score
    /// (`qwen_lm_head_greedy_int8_rows_chunk_raw_multi`). The verify pass of a
    /// width-2 speculative step needs two; the kernel's scratch is sized for
    /// four so a wider width does not need a second kernel.
    public static let maxHiddenRows = 4
    /// `[max value, token id as bits]` per threadgroup.
    static let rowSummaryStride = 2

    /// Rows the tokenizer actually defines. `ArchConfig.vocabSize` is the
    /// padded 248,320.
    public static let ornithScoredVocab = 248_077

    private let rms: RMSNorm
    private let rowGreedy: MTLComputePipelineState
    private let rowGreedyMulti: MTLComputePipelineState
    private let rowMasked: MTLComputePipelineState
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
        self.rowGreedyMulti = try pipeline("qwen_lm_head_greedy_int8_rows_chunk_raw_multi")
        self.rowMasked = try pipeline("qwen_lm_head_greedy_int8_rows_chunk_masked")
        self.rowReduce = try pipeline("qwen_lm_head_greedy_int8_rows_reduce")

        let rowGroups = (maxVocab + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
            * Self.maxHiddenRows
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

    /// Words a `vocab`-wide allow mask occupies, one bit per token id.
    public static func maskWordCount(vocab: Int) -> Int {
        (vocab + 31) / 32
    }

    /// Score again over the allowed rows only, and hand back the argmax among
    /// them.
    ///
    /// This is GEN-7's rejection path for a head that never writes the logits
    /// (`docs/qwen35moe/25-CLI-TOOLS.md` §2). The caller has already run
    /// `encodeGreedyDecode` for this hidden row and had the token it produced
    /// refused by a constraint; what is missing is not a number the host could
    /// compute from what is on the GPU — the logits were never materialized —
    /// so the table is read a second time with the rejected rows switched off.
    ///
    /// **The normalization is not repeated.** `xNormedBuffer` still holds the
    /// row `encodeGreedyDecode` normalized, and re-deriving it would be the one
    /// way for the two passes to disagree about the same token.
    ///
    /// `allowedBits` is one bit per token id, lowest id in the lowest bit of
    /// word 0, at least `maskWordCount(vocab:)` words long. A mask with every
    /// bit set produces the same token as `encodeGreedyDecode` did, bit for
    /// bit: both passes run the same row function in the same order.
    @discardableResult
    public func encodeMaskedRescore(commandBuffer: MTLCommandBuffer,
                                    weights: MTLBuffer, weightsOffset: Int = 0,
                                    scales: MTLBuffer, scalesOffset: Int = 0,
                                    biases: MTLBuffer, biasesOffset: Int = 0,
                                    allowedBits: MTLBuffer, allowedBitsOffset: Int = 0,
                                    outToken: MTLBuffer, outTokenOffset: Int = 0,
                                    d: UInt32,
                                    vocab: UInt32) -> Bool {
        guard d > 0, vocab > 0,
              Int(d) <= maxD, Int(vocab) <= maxVocab,
              Int(d) % affineGroupSize == 0,
              d % 64 == 0,
              allowedBitsOffset >= 0, allowedBitsOffset % 4 == 0,
              allowedBits.length - allowedBitsOffset
                  >= Self.maskWordCount(vocab: Int(vocab)) * MemoryLayout<UInt32>.size,
              outTokenOffset >= 0, outTokenOffset % 4 == 0 else { return false }

        let rowGroups = rowGroupCount(vocab: Int(vocab))
        guard let head = commandBuffer.makeComputeCommandEncoder() else { return false }
        head.setComputePipelineState(rowMasked)
        head.setBuffer(xNormedBuffer, offset: 0, index: 0)
        head.setBuffer(weights, offset: weightsOffset, index: 1)
        head.setBuffer(scales, offset: scalesOffset, index: 2)
        head.setBuffer(biases, offset: biasesOffset, index: 3)
        head.setBuffer(summariesBuffer, offset: 0, index: 4)
        var dValue = d
        var vocabValue = vocab
        head.setBytes(&dValue, length: MemoryLayout<UInt32>.size, index: 5)
        head.setBytes(&vocabValue, length: MemoryLayout<UInt32>.size, index: 6)
        head.setBuffer(allowedBits, offset: allowedBitsOffset, index: 7)
        head.dispatchThreadgroups(
            MTLSize(width: rowGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                           height: 1, depth: 1))
        head.endEncoding()

        encodeReduce(commandBuffer: commandBuffer, rowGroups: rowGroups,
                     outToken: outToken, outTokenOffset: outTokenOffset)
        return true
    }

    /// Fold the per-threadgroup argmaxes into one token id.
    private func encodeReduce(commandBuffer: MTLCommandBuffer,
                              rowGroups: Int,
                              summariesOffset: Int = 0,
                              outToken: MTLBuffer, outTokenOffset: Int) {
        guard let reduce = commandBuffer.makeComputeCommandEncoder() else { return }
        reduce.setComputePipelineState(rowReduce)
        reduce.setBuffer(summariesBuffer, offset: summariesOffset, index: 0)
        reduce.setBuffer(outToken, offset: outTokenOffset, index: 1)
        var rowGroupValue = UInt32(rowGroups)
        reduce.setBytes(&rowGroupValue, length: MemoryLayout<UInt32>.size, index: 2)
        let width = MTLSize(width: 256, height: 1, depth: 1)
        reduce.dispatchThreads(width, threadsPerThreadgroup: width)
        reduce.endEncoding()
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

        encodeReduce(commandBuffer: commandBuffer, rowGroups: rowGroups,
                     outToken: outToken, outTokenOffset: outTokenOffset)
        return true
    }

    /// `rows` already-normalized hidden rows in, `rows` token ids out — with
    /// **one** pass over the 508 MB table.
    ///
    /// This is the width-2 verify pass's head
    /// (`docs/qwen35moe/33-MTP-ACCEPTANCE.md` §3-8 measurement 3). Two calls to
    /// `encodeGreedyDecode` would read the table twice and cost a second 4.0 ms
    /// (`19-LM-HEAD-INT8.md`); the weights are bandwidth-bound and independent
    /// of how many activations are projected against them, so the second row is
    /// arithmetic on bytes that are already in registers.
    ///
    /// **The normalization is the caller's.** Decode's `encodeGreedyDecode`
    /// folds it in because it has exactly one row to normalize; here the rows
    /// come out of a T-row chunk that has its own multi-row RMSNorm, and
    /// re-deriving them would be the one way for the two to disagree.
    @discardableResult
    public func encodeGreedyDecodeRows(commandBuffer: MTLCommandBuffer,
                                       hiddenNormed: MTLBuffer,
                                       hiddenNormedOffset: Int = 0,
                                       weights: MTLBuffer, weightsOffset: Int = 0,
                                       scales: MTLBuffer, scalesOffset: Int = 0,
                                       biases: MTLBuffer, biasesOffset: Int = 0,
                                       outTokens: MTLBuffer, outTokensOffset: Int = 0,
                                       rows: Int,
                                       d: UInt32,
                                       vocab: UInt32) -> Bool {
        guard rows > 0, rows <= Self.maxHiddenRows,
              d > 0, vocab > 0,
              Int(d) <= maxD, Int(vocab) <= maxVocab,
              Int(d) % affineGroupSize == 0, d % 64 == 0,
              outTokensOffset >= 0, outTokensOffset % 4 == 0 else { return false }

        let rowGroups = rowGroupCount(vocab: Int(vocab))
        guard let head = commandBuffer.makeComputeCommandEncoder() else { return false }
        head.setComputePipelineState(rowGreedyMulti)
        head.setBuffer(hiddenNormed, offset: hiddenNormedOffset, index: 0)
        head.setBuffer(weights, offset: weightsOffset, index: 1)
        head.setBuffer(scales, offset: scalesOffset, index: 2)
        head.setBuffer(biases, offset: biasesOffset, index: 3)
        head.setBuffer(summariesBuffer, offset: 0, index: 4)
        var dValue = d
        var vocabValue = vocab
        var rowValue = UInt32(rows)
        head.setBytes(&dValue, length: MemoryLayout<UInt32>.size, index: 5)
        head.setBytes(&vocabValue, length: MemoryLayout<UInt32>.size, index: 6)
        head.setBytes(&rowValue, length: MemoryLayout<UInt32>.size, index: 7)
        head.dispatchThreadgroups(
            MTLSize(width: rowGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                           height: 1, depth: 1))
        head.endEncoding()

        // One reduce per hidden row. Each reads its own slice of `summaries`,
        // which the multi-row kernel wrote as [row][threadgroup].
        for row in 0..<rows {
            encodeReduce(commandBuffer: commandBuffer,
                         rowGroups: rowGroups,
                         summariesOffset: row * rowGroups
                             * Self.rowSummaryStride * MemoryLayout<Float>.size,
                         outToken: outTokens,
                         outTokenOffset: outTokensOffset + row * MemoryLayout<UInt32>.size)
        }
        return true
    }
}
