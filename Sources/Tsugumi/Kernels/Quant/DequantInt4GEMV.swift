import Foundation
import Metal

/// MLX-affine INT4 matrix-vector multiplication.
/// Eight SIMD groups process eight output rows per threadgroup.
package final class DequantInt4GEMV {
    private struct Shape: Hashable {
        var m: UInt32
        var n: UInt32
    }

    private static let rowsPerThreadgroup = 8
    private static let realDecodeShapes: [Shape] = [
        Shape(m: 4096, n: 2816),
        Shape(m: 2048, n: 2816),
        Shape(m: 2816, n: 4096),
        Shape(m: 8192, n: 2816),
        Shape(m: 1024, n: 2816),
        Shape(m: 2816, n: 8192),
    ]

    /// Rows one `encodeRows` dispatch may carry. Matches `kInt4MaxRows`.
    package static let maxRows = 8

    private let pipeline: MTLComputePipelineState
    private let rowsPipeline: MTLComputePipelineState
    private let specializedPipelines: [Shape: MTLComputePipelineState]

    /// Kept so a non-default `rowsPerSIMDGroup` (which only `--rows-bench`
    /// asks for) can still build its pipeline; the context caches under a lock.
    private let context: MetalContext
    private let groupSums: Int4RowGroupSums
    /// One pipeline per block width at the default `rowsPerSIMDGroup`, built
    /// here rather than on first use: building one costs more than a whole
    /// verify block, and the first block of each width is exactly what a cost
    /// measurement reads.
    private let widePipelines: [Int: MTLComputePipelineState]

    private let affineGroupSize: Int

    package init(context: MetalContext) throws {
        self.context = context
        self.affineGroupSize = context.affineGroupSize
        self.groupSums = try Int4RowGroupSums(context: context)
        var widePipelines: [Int: MTLComputePipelineState] = [:]
        if Self.wideEnabled {
            for t in 1...Self.maxRowsPerWideDispatch {
                widePipelines[t] = try Self.widePipeline(
                    context: context,
                    rowsPerSIMDGroup: Self.defaultRowsPerSIMDGroup,
                    t: t, specializeT: true)
            }
        }
        self.widePipelines = widePipelines
        self.pipeline = try context.pipeline(
            "dequant_int4_gemv_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)
        self.rowsPipeline = try context.pipeline(
            "dequant_int4_gemv_rows_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)

        var specializedPipelines: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            specializedPipelines[shape] = try context.pipeline(
                "dequant_int4_gemv_simd",
                constants: [
                    MetalFunctionConstant(index: 20, value: .uint32(shape.m)),
                    MetalFunctionConstant(index: 21, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 22, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPipelines = specializedPipelines
    }

    package func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer,
                weightsOffset: Int = 0,
                scales: MTLBuffer,
                scalesOffset: Int = 0,
                biases: MTLBuffer,
                biasesOffset: Int = 0,
                x: MTLBuffer,
                xOffset: Int = 0,
                y: MTLBuffer,
                yOffset: Int = 0,
                m: UInt32,
                n: UInt32) {
        precondition(n % UInt32(affineGroupSize) == 0,
                     "N must be a multiple of \(affineGroupSize)")
        // The kernel reads packed weights through a `ushort*`; the repacker
        // guarantees two-byte sub-tensor alignment but not four-byte alignment.
        precondition(weightsOffset % 2 == 0,
                     "dequant_int4_gemv_simd needs a 2-aligned weightsOffset, got \(weightsOffset)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            specializedPipelines[Shape(m: m, n: n)] ?? pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = m
        var nValue = n
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)

        let threadgroupSize = MTLSize(
            width: 32 * Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(m) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        encoder.dispatchThreadgroups(threadgroupCount,
                                     threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    /// `t` activation rows against the same weights in one dispatch.
    ///
    /// `encode` called in a loop reads `W` once per row; this reads it once for
    /// the whole block, which is what makes a k-token speculative verify cost
    /// the bytes of one forward instead of k (docs/mtp/16-M4.5-PLAN.md §4 c).
    /// Per output element the arithmetic is `encode`'s, in the same order, so
    /// `t == 1` is bit-identical to it.
    package func encodeRows(commandBuffer: MTLCommandBuffer,
                            weights: MTLBuffer,
                            weightsOffset: Int = 0,
                            scales: MTLBuffer,
                            scalesOffset: Int = 0,
                            biases: MTLBuffer,
                            biasesOffset: Int = 0,
                            x: MTLBuffer,
                            xOffset: Int = 0,
                            xStrideElements: Int,
                            y: MTLBuffer,
                            yOffset: Int = 0,
                            yStrideElements: Int,
                            t: Int,
                            m: UInt32,
                            n: UInt32) {
        precondition(n % UInt32(affineGroupSize) == 0,
                     "N must be a multiple of \(affineGroupSize)")
        precondition(weightsOffset % 2 == 0,
                     "dequant_int4_gemv_rows_simd needs a 2-aligned weightsOffset, got \(weightsOffset)")
        precondition(t >= 1 && t <= Self.maxRows,
                     "row count \(t) is outside 1...\(Self.maxRows)")
        precondition(xStrideElements >= Int(n), "x stride must cover N")
        precondition(yStrideElements >= Int(m), "y stride must cover M")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(rowsPipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = m
        var nValue = n
        var tValue = UInt32(t)
        var xStride = UInt32(xStrideElements)
        var yStride = UInt32(yStrideElements)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&tValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&xStride, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.setBytes(&yStride, length: MemoryLayout<UInt32>.size, index: 9)

        let threadgroupSize = MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                      height: 1,
                                      depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(m) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        encoder.dispatchThreadgroups(threadgroupCount,
                                     threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
}

// MARK: - k-row GEMV with the per-row work taken out of the weight loop

extension DequantInt4GEMV {
    /// Weight rows one SIMD group may carry. Matches `kInt4MaxRowsPerSG`.
    package static let maxRowsPerSIMDGroup = 4

    /// Activation rows one wide dispatch may carry. Matches `kInt4WideMaxT`.
    ///
    /// Not a capability limit — a tuning one. `acc` is `rowsPerSIMDGroup * t`
    /// registers, and past four rows the block no longer fits: at t=5 the head
    /// costs 9.4 ms against 3.4 ms at t=4, while two dispatches of four cost
    /// 6.8 ms even though they read the weights twice
    /// (`docs/mtp/20-M4.8-RESULTS.md` §2). So a wider block splits.
    package static let maxRowsPerWideDispatch = 4

    /// `TF_MTP_ROWS_WIDE=0` sends every wide call back to `encodeRows`.
    ///
    /// The two are bit-identical, so this is a timing switch: it is how
    /// `docs/mtp/20-M4.8-RESULTS.md` §3 measures the change against the shipped
    /// kernel inside one run, rather than across two builds.
    package static let wideEnabled =
        ProcessInfo.processInfo.environment["TF_MTP_ROWS_WIDE"] != "0"

    /// `encodeRows` with the activation-side work amortized.
    ///
    /// Same result, same order, two fewer things charged per activation row:
    /// the affine bias term's activation sum is precomputed once per block
    /// (`int4_rows_group_sums`) instead of once per weight row, and
    /// `rowsPerSIMDGroup` weight rows share one activation load. See
    /// `docs/mtp/19-M4.7-RESULTS.md` §5 for why those two dominate the k-scaling
    /// of the `head` and `shared` stages.
    ///
    /// `specializeT` picks a pipeline per block width, which unrolls the
    /// activation loop; those are built in `init`. Blocks wider than
    /// `maxRowsPerWideDispatch` split into several dispatches.
    package func encodeRowsWide(commandBuffer: MTLCommandBuffer,
                                weights: MTLBuffer,
                                weightsOffset: Int = 0,
                                scales: MTLBuffer,
                                scalesOffset: Int = 0,
                                biases: MTLBuffer,
                                biasesOffset: Int = 0,
                                x: MTLBuffer,
                                xOffset: Int = 0,
                                xStrideElements: Int,
                                y: MTLBuffer,
                                yOffset: Int = 0,
                                yStrideElements: Int,
                                t: Int,
                                m: UInt32,
                                n: UInt32,
                                rowsPerSIMDGroup: Int = defaultRowsPerSIMDGroup,
                                specializeT: Bool = true) throws {
        precondition(t >= 1 && t <= Self.maxRows,
                     "row count \(t) is outside 1...\(Self.maxRows)")
        guard Self.wideEnabled else {
            encodeRows(commandBuffer: commandBuffer,
                       weights: weights, weightsOffset: weightsOffset,
                       scales: scales, scalesOffset: scalesOffset,
                       biases: biases, biasesOffset: biasesOffset,
                       x: x, xOffset: xOffset, xStrideElements: xStrideElements,
                       y: y, yOffset: yOffset, yStrideElements: yStrideElements,
                       t: t, m: m, n: n)
            return
        }
        let halfBytes = MemoryLayout<Float16>.stride
        var first = 0
        while first < t {
            let count = min(Self.maxRowsPerWideDispatch, t - first)
            try encodeRowsWideChunk(
                commandBuffer: commandBuffer,
                weights: weights, weightsOffset: weightsOffset,
                scales: scales, scalesOffset: scalesOffset,
                biases: biases, biasesOffset: biasesOffset,
                x: x, xOffset: xOffset + first * xStrideElements * halfBytes,
                xStrideElements: xStrideElements,
                y: y, yOffset: yOffset + first * yStrideElements * halfBytes,
                yStrideElements: yStrideElements,
                t: count, m: m, n: n,
                rowsPerSIMDGroup: rowsPerSIMDGroup, specializeT: specializeT)
            first += count
        }
    }

    private func encodeRowsWideChunk(commandBuffer: MTLCommandBuffer,
                                     weights: MTLBuffer,
                                     weightsOffset: Int,
                                     scales: MTLBuffer,
                                     scalesOffset: Int,
                                     biases: MTLBuffer,
                                     biasesOffset: Int,
                                     x: MTLBuffer,
                                     xOffset: Int,
                                     xStrideElements: Int,
                                     y: MTLBuffer,
                                     yOffset: Int,
                                     yStrideElements: Int,
                                     t: Int,
                                     m: UInt32,
                                     n: UInt32,
                                     rowsPerSIMDGroup: Int,
                                     specializeT: Bool) throws {
        precondition(n % UInt32(affineGroupSize) == 0,
                     "N must be a multiple of \(affineGroupSize)")
        precondition(weightsOffset % 2 == 0,
                     "dequant_int4_gemv_rows_wide_simd needs a 2-aligned weightsOffset, "
                        + "got \(weightsOffset)")
        precondition(t >= 1 && t <= Self.maxRowsPerWideDispatch,
                     "row count \(t) is outside 1...\(Self.maxRowsPerWideDispatch)")
        precondition(rowsPerSIMDGroup >= 1 && rowsPerSIMDGroup <= Self.maxRowsPerSIMDGroup,
                     "rowsPerSIMDGroup \(rowsPerSIMDGroup) is outside "
                        + "1...\(Self.maxRowsPerSIMDGroup)")
        precondition(xStrideElements >= Int(n), "x stride must cover N")
        precondition(yStrideElements >= Int(m), "y stride must cover M")

        let (xsum, xsumStride) = try groupSums.encode(commandBuffer: commandBuffer,
                                                     x: x, xOffset: xOffset,
                                                     xStrideElements: xStrideElements,
                                                     t: t, n: Int(n))

        let prebuilt = specializeT && rowsPerSIMDGroup == Self.defaultRowsPerSIMDGroup
            ? widePipelines[t] : nil
        let state = try prebuilt ?? Self.widePipeline(context: context,
                                                      rowsPerSIMDGroup: rowsPerSIMDGroup,
                                                      t: t, specializeT: specializeT)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(state)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = m
        var nValue = n
        var tValue = UInt32(t)
        var xStride = UInt32(xStrideElements)
        var yStride = UInt32(yStrideElements)
        var sumStride = UInt32(xsumStride)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&tValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&xStride, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.setBytes(&yStride, length: MemoryLayout<UInt32>.size, index: 9)
        encoder.setBuffer(xsum, offset: 0, index: 10)
        encoder.setBytes(&sumStride, length: MemoryLayout<UInt32>.size, index: 11)

        let rowsPerThreadgroup = Self.rowsPerThreadgroupWide(rowsPerSIMDGroup)
        let threadgroupSize = MTLSize(width: 32 * 8, height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(m) + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: 1,
            depth: 1)
        encoder.dispatchThreadgroups(threadgroupCount,
                                     threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    /// Weight rows per SIMD group. Two: one activation load then pays for two
    /// output rows, and a third does not fit alongside `acc`
    /// (`docs/mtp/20-M4.8-RESULTS.md` §2).
    package static let defaultRowsPerSIMDGroup = 2

    private static func widePipeline(context: MetalContext,
                                     rowsPerSIMDGroup: Int,
                                     t: Int,
                                     specializeT: Bool) throws -> MTLComputePipelineState {
        var constants = [MetalFunctionConstant(index: 27,
                                               value: .uint32(UInt32(rowsPerSIMDGroup)))]
        if specializeT {
            constants.append(MetalFunctionConstant(index: 28, value: .uint32(UInt32(t))))
        }
        return try context.pipeline("dequant_int4_gemv_rows_wide_simd",
                                    constants: constants,
                                    maxTotalThreadsPerThreadgroup: 512)
    }

    private static func rowsPerThreadgroupWide(_ rowsPerSIMDGroup: Int) -> Int {
        8 * rowsPerSIMDGroup
    }
}
