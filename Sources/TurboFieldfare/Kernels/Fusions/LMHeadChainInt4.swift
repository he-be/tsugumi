import Metal

/// Final BF16 RMSNorm, INT4 affine lm-head projection, and greedy argmax.
/// The hot path writes one token ID without materializing vocab-sized logits.
package final class LMHeadChainInt4 {
    static let rowsPerThreadgroup = 8

    private static let rowSummaryStride = 2
    private static let realDecodeD: UInt32 = 2816
    private static let realDecodeVocab: UInt32 = 262144
    private static let realDecodeHeadConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 10, value: .uint32(realDecodeD)),
        MetalFunctionConstant(index: 11, value: .uint32(realDecodeVocab)),
        MetalFunctionConstant(index: 13, value: .bool(true)),
    ]

    /// Rows one `encodeGreedyDecodeBlock` may score. Matches
    /// `kLMHeadMaxBlockRows`.
    package static let maxBlockRows = 8

    private let rms: RMSNorm
    private let blockRMS: PrefillRMSNorm
    private let rowGreedy: MTLComputePipelineState
    private let rowGreedySpecialized: MTLComputePipelineState
    private let blockGreedy: MTLComputePipelineState
    private let blockGreedySpecialized: MTLComputePipelineState
    private let rowReducer: MTLComputePipelineState
    private let xNormedBuffer: MTLBuffer
    private let rowSummariesBuffer: MTLBuffer
    private let device: MTLDevice
    /// Block staging, allocated on the first block call: a runner that never
    /// speculates never pays for it (2 MB of summaries at production vocab).
    private var blockXNormedBuffer: MTLBuffer?
    private var blockSummariesBuffer: MTLBuffer?
    private let maxD: Int
    private let maxVocab: Int

    private let affineGroupSize: Int

    package init(context: MetalContext,
                 maxD: Int = 2816,
                 maxVocab: Int = 262144) throws {
        self.affineGroupSize = context.affineGroupSize
        self.device = context.device
        self.rms = try RMSNorm(context: context)
        self.blockRMS = try PrefillRMSNorm(context: context)
        self.rowGreedy = try context.pipeline("lm_head_greedy_int4_rows_chunk_raw")
        self.rowGreedySpecialized = try context.pipeline(
            "lm_head_greedy_int4_rows_chunk_raw",
            constants: Self.realDecodeHeadConstants)
        self.blockGreedy = try context.pipeline("lm_head_greedy_int4_rows_chunk_block")
        self.blockGreedySpecialized = try context.pipeline(
            "lm_head_greedy_int4_rows_chunk_block",
            constants: Self.realDecodeHeadConstants)
        self.rowReducer = try context.pipeline("lm_head_greedy_int4_rows_reduce")
        self.maxD = maxD
        self.maxVocab = maxVocab

        let rowGroups = (maxVocab + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
        let xLength = max(maxD, 1) * MemoryLayout<Float16>.size
        let summaryLength = rowGroups * Self.rowSummaryStride * MemoryLayout<Float>.size
        guard let xNormedBuffer = context.device.makeBuffer(
                  length: xLength,
                  options: .storageModePrivate),
              let rowSummariesBuffer = context.device.makeBuffer(
                  length: summaryLength,
                  options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        self.xNormedBuffer = xNormedBuffer
        self.rowSummariesBuffer = rowSummariesBuffer
    }

    /// One row in, one token ID out at `outTokenOffset`.
    ///
    /// Several rows may be encoded onto the same command buffer — that is how
    /// the speculative verify block gets its per-position argmaxes. Each stage
    /// is its own encoder and encoders run in submission order, so the shared
    /// `xNormedBuffer` and `rowSummariesBuffer` are never live for two rows at
    /// once.
    package func encodeGreedyDecode(commandBuffer: MTLCommandBuffer,
                            hidden: MTLBuffer,
                            hiddenOffset: Int = 0,
                            normWeight: MTLBuffer,
                            normOffset: Int = 0,
                            weights: MTLBuffer,
                            weightsOffset: Int = 0,
                            scales: MTLBuffer,
                            scalesOffset: Int = 0,
                            biases: MTLBuffer,
                            biasesOffset: Int = 0,
                            outToken: MTLBuffer,
                            outTokenOffset: Int = 0,
                            d: UInt32,
                            vocab: UInt32,
                            rmsEps: Float = 1e-6) {
        precondition(Int(d) <= maxD, "d=\(d) exceeds wrapper maxD=\(maxD)")
        precondition(outTokenOffset >= 0 && outTokenOffset % MemoryLayout<UInt32>.stride == 0,
                     "outTokenOffset must be a non-negative multiple of 4")
        precondition(Int(vocab) <= maxVocab,
                     "vocab=\(vocab) exceeds wrapper maxVocab=\(maxVocab)")
        precondition(Int(d) % affineGroupSize == 0,
                     "d must be a multiple of \(affineGroupSize)")
        precondition(hiddenOffset >= 0, "hiddenOffset must be non-negative")
        precondition(weightsOffset % 2 == 0,
                     "lm_head_greedy_int4_rows_chunk_raw needs a 2-aligned weightsOffset")

        let rowGroups = (Int(vocab) + Self.rowsPerThreadgroup - 1)
            / Self.rowsPerThreadgroup
        rms.encodeBF16W(commandBuffer: commandBuffer,
                        x: hidden,
                        xOffset: hiddenOffset,
                        weight: normWeight,
                        weightOffset: normOffset,
                        out: xNormedBuffer,
                        d: d,
                        eps: rmsEps)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            let specialized = d == Self.realDecodeD && vocab == Self.realDecodeVocab
            encoder.setComputePipelineState(specialized ? rowGreedySpecialized : rowGreedy)
            encoder.setBuffer(xNormedBuffer, offset: 0, index: 0)
            encoder.setBuffer(weights, offset: weightsOffset, index: 1)
            encoder.setBuffer(scales, offset: scalesOffset, index: 2)
            encoder.setBuffer(biases, offset: biasesOffset, index: 3)
            encoder.setBuffer(rowSummariesBuffer, offset: 0, index: 4)
            var dValue = d
            var vocabValue = vocab
            encoder.setBytes(&dValue, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&vocabValue, length: MemoryLayout<UInt32>.size, index: 6)

            let threadgroupSize = MTLSize(
                width: 32 * Self.rowsPerThreadgroup,
                height: 1,
                depth: 1)
            encoder.dispatchThreadgroups(
                MTLSize(width: rowGroups, height: 1, depth: 1),
                threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(rowReducer)
            encoder.setBuffer(rowSummariesBuffer, offset: 0, index: 0)
            encoder.setBuffer(outToken, offset: outTokenOffset, index: 1)
            var rowGroupCount = UInt32(rowGroups)
            encoder.setBytes(&rowGroupCount, length: MemoryLayout<UInt32>.size, index: 2)

            let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
            encoder.dispatchThreads(threadgroupSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
    }

    /// `rows` consecutive hidden rows in, `rows` token IDs out.
    ///
    /// `encodeGreedyDecode` in a loop reads the 461 MB lm-head table once per
    /// row; a speculative block needs every row and the rows differ only in the
    /// activation, so this reads the table once for all of them
    /// (docs/mtp/16-M4.5-PLAN.md §4 b). Per (row, vocabulary row) the reduction
    /// is `encodeGreedyDecode`'s, so a one-row block is bit-identical to it.
    ///
    /// The hidden rows must be contiguous with stride `d`, which is how every
    /// prefill chunk stages them.
    package func encodeGreedyDecodeBlock(commandBuffer: MTLCommandBuffer,
                                         hidden: MTLBuffer,
                                         hiddenOffset: Int = 0,
                                         normWeight: MTLBuffer,
                                         normOffset: Int = 0,
                                         weights: MTLBuffer,
                                         weightsOffset: Int = 0,
                                         scales: MTLBuffer,
                                         scalesOffset: Int = 0,
                                         biases: MTLBuffer,
                                         biasesOffset: Int = 0,
                                         outTokens: MTLBuffer,
                                         outTokensOffset: Int = 0,
                                         rows: Int,
                                         d: UInt32,
                                         vocab: UInt32,
                                         rmsEps: Float = 1e-6) throws {
        precondition(rows >= 1 && rows <= Self.maxBlockRows,
                     "rows=\(rows) is outside 1...\(Self.maxBlockRows)")
        precondition(Int(d) <= maxD, "d=\(d) exceeds wrapper maxD=\(maxD)")
        precondition(Int(vocab) <= maxVocab,
                     "vocab=\(vocab) exceeds wrapper maxVocab=\(maxVocab)")
        precondition(Int(d) % affineGroupSize == 0,
                     "d must be a multiple of \(affineGroupSize)")
        precondition(hiddenOffset >= 0, "hiddenOffset must be non-negative")
        precondition(outTokensOffset >= 0
                     && outTokensOffset % MemoryLayout<UInt32>.stride == 0,
                     "outTokensOffset must be a non-negative multiple of 4")
        precondition(weightsOffset % 2 == 0,
                     "lm_head_greedy_int4_rows_chunk_block needs a 2-aligned weightsOffset")

        let rowGroups = (Int(vocab) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
        let (xNormed, summaries) = try blockStaging()

        blockRMS.encodeBF16W(commandBuffer: commandBuffer,
                             x: hidden,
                             xOffset: hiddenOffset,
                             weight: normWeight,
                             weightOffset: normOffset,
                             out: xNormed,
                             t: UInt32(rows),
                             d: d,
                             eps: rmsEps)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            let specialized = d == Self.realDecodeD && vocab == Self.realDecodeVocab
            encoder.setComputePipelineState(specialized ? blockGreedySpecialized : blockGreedy)
            encoder.setBuffer(xNormed, offset: 0, index: 0)
            encoder.setBuffer(weights, offset: weightsOffset, index: 1)
            encoder.setBuffer(scales, offset: scalesOffset, index: 2)
            encoder.setBuffer(biases, offset: biasesOffset, index: 3)
            encoder.setBuffer(summaries, offset: 0, index: 4)
            var dValue = d
            var vocabValue = vocab
            var rowsValue = UInt32(rows)
            var rowGroupCount = UInt32(rowGroups)
            encoder.setBytes(&dValue, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&vocabValue, length: MemoryLayout<UInt32>.size, index: 6)
            encoder.setBytes(&rowsValue, length: MemoryLayout<UInt32>.size, index: 7)
            encoder.setBytes(&rowGroupCount, length: MemoryLayout<UInt32>.size, index: 8)

            encoder.dispatchThreadgroups(
                MTLSize(width: rowGroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                               height: 1, depth: 1))
            encoder.endEncoding()
        }

        // One reducer per row over that row's slice of the summaries. Each is a
        // single 256-thread threadgroup, so this is the same work the one-row
        // path did, not a new cost.
        let summaryRowBytes = rowGroups * Self.rowSummaryStride * MemoryLayout<Float>.stride
        for row in 0..<rows {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }
            encoder.setComputePipelineState(rowReducer)
            encoder.setBuffer(summaries, offset: row * summaryRowBytes, index: 0)
            encoder.setBuffer(outTokens,
                              offset: outTokensOffset + row * MemoryLayout<UInt32>.stride,
                              index: 1)
            var rowGroupCount = UInt32(rowGroups)
            encoder.setBytes(&rowGroupCount, length: MemoryLayout<UInt32>.size, index: 2)
            let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
            encoder.dispatchThreads(threadgroupSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
    }

    private func blockStaging() throws -> (xNormed: MTLBuffer, summaries: MTLBuffer) {
        if let blockXNormedBuffer, let blockSummariesBuffer {
            return (blockXNormedBuffer, blockSummariesBuffer)
        }
        let rowGroups = (maxVocab + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
        let xLength = Self.maxBlockRows * max(maxD, 1) * MemoryLayout<Float16>.size
        let summaryLength = Self.maxBlockRows * rowGroups * Self.rowSummaryStride
            * MemoryLayout<Float>.size
        guard let xNormed = device.makeBuffer(length: xLength, options: .storageModePrivate),
              let summaries = device.makeBuffer(length: summaryLength,
                                                options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        xNormed.label = "lmHead.block.xNormed"
        summaries.label = "lmHead.block.summaries"
        blockXNormedBuffer = xNormed
        blockSummariesBuffer = summaries
        return (xNormed, summaries)
    }
}
