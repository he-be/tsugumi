import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// The k-row router of a speculative block against the row-at-a-time dispatch
/// it replaced (docs/mtp/27-M7-RESULTS.md §5).
///
/// The bar is bit equality, not a tolerance: the reason the block ran the decode
/// router once per row was that a near-tie between two experts has to resolve
/// the way decode resolves it (docs/mtp/16-M4.5-PLAN.md §4). The rows kernel is
/// only allowed to change how the rows are dispatched, so every row's logits —
/// and with them the chosen experts and their weights — must come out identical.
@Suite struct RouterRowsTests {
    private static let topK = 8

    private struct Result: Equatable {
        let indices: [UInt32]
        let weightBits: [UInt16]
    }

    @Test func foldedRouterMatchesRowLoopBitForBit() throws {
        try Self.compare(experts: 16, dimension: 128, rows: 4, seed: 0x51D2_88AF)
    }

    @Test func foldedRouterMatchesRowLoopOnTheSpecializedShape() throws {
        // 128 experts and D = 2816 is the shape the production router compiles
        // function constants for, so this is the pipeline the model runs.
        try Self.compare(experts: 128, dimension: 2816, rows: 4, seed: 0x1F0C_7B29)
    }

    @Test func foldedRouterMatchesRowLoopOnNearTies() throws {
        // Experts 7 and 8 sit either side of the top-8 cut, one part in 10^4
        // apart: the position where a different reduction order would show.
        try Self.compare(experts: 16, dimension: 128, rows: 3, seed: 0x2B44_9E10,
                         nearTie: true)
    }

    @Test func foldedRouterMatchesRowLoopAtWidthOne() throws {
        try Self.compare(experts: 16, dimension: 128, rows: 1, seed: 0x77A1_0455)
    }

    private static func compare(experts: Int,
                                dimension: Int,
                                rows: Int,
                                seed: UInt64,
                                nearTie: Bool = false) throws {
        var rng = SplitMix64(seed: seed)
        var gains = (0..<experts).map { _ in rng.uniform(0.6, 1.4) }
        if nearTie {
            gains[7] = 0.5
            gains[8] = 0.5 * (1.0 + 1e-4)
        }
        let pattern = (0..<dimension).map { _ in rng.uniform(-0.05, 0.05) }
        let weights = gains.flatMap { gain in pattern.map { $0 * gain } }
        let hidden = (0..<(rows * dimension)).map { _ in rng.uniform(-1.0, 1.0) }
        let invSqrtD = 1.0 / Float(dimension).squareRoot()
        let effectiveScale = (0..<dimension).map { _ in rng.uniform(0.5, 1.5) * invSqrtD }
        let expertScale = (0..<experts).map { _ in rng.uniform(0.6, 1.4) }

        let context = try MetalContext()
        let kernel = try MoE(context: context, routerWeightBits: 16)
        guard let weightBuffer = context.device.makeBuffer(
                  bytes: weights.map(Quantization.bf16Bits),
                  length: weights.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let hiddenBuffer = Fp16Buffer.make(context.device, values: hidden),
              let effectiveBuffer = context.device.makeBuffer(
                  bytes: effectiveScale.map(Quantization.bf16Bits),
                  length: effectiveScale.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let expertScaleBuffer = context.device.makeBuffer(
                  bytes: expertScale.map(Quantization.bf16Bits),
                  length: expertScale.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared) else {
            throw CocoaError(.fileReadUnknown)
        }

        let perRow = try run(rows: rows, experts: experts, dimension: dimension,
                             context: context, kernel: kernel,
                             weights: weightBuffer, hidden: hiddenBuffer,
                             effectiveScale: effectiveBuffer,
                             expertScale: expertScaleBuffer,
                             folded: false)
        let folded = try run(rows: rows, experts: experts, dimension: dimension,
                             context: context, kernel: kernel,
                             weights: weightBuffer, hidden: hiddenBuffer,
                             effectiveScale: effectiveBuffer,
                             expertScale: expertScaleBuffer,
                             folded: true)
        #expect(folded == perRow)
        // A row that picked nothing would compare equal to another empty row,
        // so check that the dispatch ran at all.
        #expect(Set(perRow.indices).count > 1)
    }

    private static func run(rows: Int,
                            experts: Int,
                            dimension: Int,
                            context: MetalContext,
                            kernel: MoE,
                            weights: MTLBuffer,
                            hidden: MTLBuffer,
                            effectiveScale: MTLBuffer,
                            expertScale: MTLBuffer,
                            folded: Bool) throws -> Result {
        guard let indexBuffer = context.device.makeBuffer(
                  length: rows * topK * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let weightBuffer = Fp16Buffer.make(context.device, count: rows * topK),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        if folded {
            let encoded = kernel.encodeRouterGemma4BF16Rows(
                commandBuffer: commandBuffer,
                weights: weights,
                hidden: hidden,
                hiddenStrideElements: UInt32(dimension),
                effectiveScale: effectiveScale,
                perExpertScale: expertScale,
                outIndices: indexBuffer,
                outWeights: weightBuffer,
                rowCount: rows,
                numExperts: UInt32(experts),
                d: UInt32(dimension),
                topK: UInt32(topK))
            #expect(encoded)
        } else {
            for row in 0..<rows {
                kernel.encodeRouterGemma4BF16(
                    commandBuffer: commandBuffer,
                    weights: weights,
                    hidden: hidden,
                    hiddenOffset: row * dimension * MemoryLayout<Float16>.stride,
                    effectiveScale: effectiveScale,
                    perExpertScale: expertScale,
                    outIndices: indexBuffer,
                    outIndicesOffset: row * topK * MemoryLayout<UInt32>.stride,
                    outWeights: weightBuffer,
                    outWeightsOffset: row * topK * MemoryLayout<Float16>.stride,
                    numExperts: UInt32(experts),
                    d: UInt32(dimension),
                    topK: UInt32(topK))
            }
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let indexPointer = indexBuffer.contents().bindMemory(
            to: UInt32.self, capacity: rows * topK)
        let weightPointer = weightBuffer.contents().bindMemory(
            to: UInt16.self, capacity: rows * topK)
        return Result(indices: (0..<(rows * topK)).map { indexPointer[$0] },
                      weightBits: (0..<(rows * topK)).map { weightPointer[$0] })
    }
}
