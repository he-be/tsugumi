import Metal

/// The activation half of the affine bias term, computed once per block.
///
/// An INT4 affine GEMV row is `s_g * Σ q x + b_g * Σ x` per group, and the
/// second sum does not depend on the weight row — but the k-row kernels
/// recomputed it inside the weight loop, so the LM head's 262144 rows each
/// summed the same activations again. This precomputes them, in the same order
/// and to the same bits, so the wide kernels can read the sum instead
/// (`docs/mtp/20-M4.8-RESULTS.md` §2).
///
/// One entry per (vectorized block, lane) per activation row, which is the
/// eight contiguous elements a lane covers. A reduction length that is not a
/// whole number of 256-element blocks leaves a scalar tail; the kernels keep
/// computing that tail's two-element sum inline, so nothing here has to know
/// about it.
package final class Int4RowGroupSums {
    private let pipeline: MTLComputePipelineState
    private let device: MTLDevice
    private var scratch: MTLBuffer?

    package init(context: MetalContext) throws {
        self.pipeline = try context.pipeline("int4_rows_group_sums")
        self.device = context.device
    }

    /// Entries per activation row for a reduction of length `n`.
    package static func stride(n: Int) -> Int { (n / 256) * 32 }

    /// Encodes the prepass and returns the buffer the wide kernel should bind.
    ///
    /// The buffer is reused across calls: compute encoders in one command
    /// buffer run in order, so each dispatch sees its own prepass.
    package func encode(commandBuffer: MTLCommandBuffer,
                        x: MTLBuffer,
                        xOffset: Int,
                        xStrideElements: Int,
                        t: Int,
                        n: Int) throws -> (buffer: MTLBuffer, stride: Int) {
        let stride = Self.stride(n: n)
        let buffer = try scratchBuffer(rows: t, stride: stride)
        guard stride > 0, let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return (buffer, stride)
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(buffer, offset: 0, index: 1)
        var xStride = UInt32(xStrideElements)
        var sumStride = UInt32(stride)
        encoder.setBytes(&xStride, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&sumStride, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 64)
        encoder.dispatchThreads(MTLSize(width: stride, height: t, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        return (buffer, stride)
    }

    private func scratchBuffer(rows: Int, stride: Int) throws -> MTLBuffer {
        let needed = max(1, rows * stride) * MemoryLayout<Float>.stride
        if let scratch, scratch.length >= needed { return scratch }
        guard let buffer = device.makeBuffer(length: needed, options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        buffer.label = "int4.rowsWide.xsum"
        scratch = buffer
        return buffer
    }
}
