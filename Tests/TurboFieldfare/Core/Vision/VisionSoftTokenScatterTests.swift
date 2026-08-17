import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// The soft-token scatter (`PLAN_VISION.md` §4-5-c): image rows of a prefill
/// chunk's embedding block are replaced by the tower's output, unscaled, and
/// nothing else in the block is touched.
@Suite struct VisionSoftTokenScatterTests {
    private static let d = 8
    private static let stride = 11   // wider than d, as the real block is not

    private func run(hiddenRows: Int,
                     softRows: Int,
                     hiddenRow: Int,
                     softRow: Int,
                     rowCount: Int) throws -> (before: [Float], after: [Float], soft: [Float]) {
        let ctx = try MetalContext()
        let scatter = try VisionSoftTokenScatter(context: ctx)
        let d = Self.d
        let stride = Self.stride

        var hidden = [Float](repeating: 0, count: hiddenRows * stride)
        for row in 0..<hiddenRows {
            for column in 0..<stride {
                hidden[row * stride + column] = Float(100 + row) + Float(column) / 32
            }
        }
        var soft = [Float](repeating: 0, count: softRows * d)
        for row in 0..<softRows {
            for column in 0..<d {
                soft[row * d + column] = -Float(row + 1) - Float(column) / 64
            }
        }

        guard let hiddenBuffer = Fp16Buffer.make(ctx.device, values: hidden),
              let softBuffer = Fp16Buffer.make(ctx.device, values: soft) else {
            Issue.record("alloc failed")
            return ([], [], [])
        }
        let cb = ctx.queue.makeCommandBuffer()!
        scatter.encode(commandBuffer: cb,
                       hidden: hiddenBuffer,
                       soft: softBuffer,
                       d: d,
                       hiddenRow: hiddenRow,
                       softRow: softRow,
                       rowCount: rowCount,
                       hiddenStrideElements: stride)
        cb.commit()
        cb.waitUntilCompleted()
        try checkCommandBufferError(cb.error)
        return (hidden, Fp16Buffer.read(hiddenBuffer, count: hidden.count), soft)
    }

    @Test func writesTheImageRowsAndOnlyThoseRows() throws {
        let (before, after, soft) = try run(hiddenRows: 12, softRows: 6,
                                            hiddenRow: 3, softRow: 0, rowCount: 6)
        let d = Self.d
        let stride = Self.stride
        for row in 0..<12 {
            for column in 0..<stride {
                let index = row * stride + column
                let isImageRow = (3..<9).contains(row)
                if isImageRow && column < d {
                    // FP16 round trip, so equality is to the FP16 grid.
                    let expected = Float(Float16(soft[(row - 3) * d + column]))
                    #expect(after[index] == expected,
                            "row \(row) column \(column): \(after[index]) != \(expected)")
                } else {
                    #expect(after[index] == Float(Float16(before[index])),
                            "row \(row) column \(column) was modified")
                }
            }
        }
    }

    /// The general form: a chunk that holds only the tail of an image writes
    /// from the middle of the soft-token buffer. The runner never produces this
    /// today — a chunk boundary never falls inside an image — but the kernel is
    /// what would silently write the wrong rows if that ever changed.
    @Test func writesAPartialSpanFromTheMiddleOfTheSoftTokens() throws {
        let (before, after, soft) = try run(hiddenRows: 10, softRows: 8,
                                            hiddenRow: 0, softRow: 5, rowCount: 3)
        let d = Self.d
        let stride = Self.stride
        for row in 0..<3 {
            for column in 0..<d {
                let expected = Float(Float16(soft[(5 + row) * d + column]))
                #expect(after[row * stride + column] == expected)
            }
        }
        for row in 3..<10 {
            for column in 0..<stride {
                let index = row * stride + column
                #expect(after[index] == Float(Float16(before[index])))
            }
        }
    }
}
