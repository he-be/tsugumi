import Foundation
import Testing
@testable import Tsugumi

/// Geometry is the one part of the vision path with an exact reference: it is
/// integer arithmetic all the way down, so agreement with upstream is a literal
/// equality rather than a tolerance.
///
/// The expectations below are the measured output of
/// `transformers 5.6.2`'s `get_aspect_ratio_preserving_size` on the fixture
/// images (PLAN_VISION §0-B-2), transcribed so this suite runs without the
/// 195 MB of fixtures present.
@Suite("VisionGeometry")
struct VisionGeometryTests {

    private struct Expectation {
        let width: Int, height: Int, budget: Int
        let patchesWide: Int, patchesHigh: Int, softTokens: Int
    }

    private static let reference: [Expectation] = [
        .init(width: 512, height: 512, budget: 70, patchesWide: 24, patchesHigh: 24, softTokens: 64),
        .init(width: 512, height: 512, budget: 280, patchesWide: 48, patchesHigh: 48, softTokens: 256),
        .init(width: 480, height: 1200, budget: 70, patchesWide: 15, patchesHigh: 39, softTokens: 65),
        .init(width: 480, height: 1200, budget: 280, patchesWide: 30, patchesHigh: 78, softTokens: 260),
        .init(width: 1024, height: 768, budget: 70, patchesWide: 27, patchesHigh: 21, softTokens: 63),
        .init(width: 1024, height: 768, budget: 280, patchesWide: 57, patchesHigh: 42, softTokens: 266),
    ]

    @Test("Patch grids match the transformers reference")
    func matchesReference() throws {
        for expected in Self.reference {
            let config = try VisionPreprocessorConfig(maxSoftTokens: expected.budget)
            let geometry = try config.geometry(imageWidth: expected.width, imageHeight: expected.height)
            #expect(geometry.patchesWide == expected.patchesWide,
                    "\(expected.width)x\(expected.height)@\(expected.budget) width")
            #expect(geometry.patchesHigh == expected.patchesHigh,
                    "\(expected.width)x\(expected.height)@\(expected.budget) height")
            #expect(geometry.softTokenCount == expected.softTokens,
                    "\(expected.width)x\(expected.height)@\(expected.budget) soft tokens")
        }
    }

    /// The claim the whole prompt layout rests on: a budget of 280 is a ceiling
    /// that no real image reaches. Anything that writes 280 as a constant is
    /// wrong (PLAN_VISION §2-1).
    @Test("No fixture image consumes its full soft-token budget")
    func budgetIsACeiling() throws {
        for expected in Self.reference {
            let config = try VisionPreprocessorConfig(maxSoftTokens: expected.budget)
            let geometry = try config.geometry(imageWidth: expected.width, imageHeight: expected.height)
            #expect(geometry.softTokenCount < expected.budget)
            #expect(geometry.softTokenCount > 0)
        }
    }

    @Test("Sides are multiples of pooling * patch, and patches fit the budget")
    func invariants() throws {
        let config = try VisionPreprocessorConfig()
        // A spread of aspect ratios rather than the fixtures alone, so the
        // invariants are exercised where the rounding is least convenient.
        for width in [17, 64, 300, 640, 1024, 4000] {
            for height in [17, 64, 300, 640, 1024, 4000] {
                let geometry = try config.geometry(imageWidth: width, imageHeight: height)
                #expect(geometry.targetWidth % config.sideMultiple == 0)
                #expect(geometry.targetHeight % config.sideMultiple == 0)
                #expect(geometry.patchCount <= config.maxPatches)
                #expect(geometry.patchCount == geometry.softTokenCount * 9)
                #expect(geometry.targetWidth == geometry.patchesWide * config.patchSize)
                #expect(geometry.targetHeight == geometry.patchesHigh * config.patchSize)
            }
        }
    }

    /// Extreme aspect ratios round one side to zero; upstream rescues them with
    /// a branch that uses the integer side ratio. Reproduced rather than
    /// reasoned about, because the fixtures come from that branch.
    @Test("Degenerate aspect ratios take the rescue branch")
    func degenerateAspectRatios() throws {
        let config = try VisionPreprocessorConfig()
        for (width, height) in [(4000, 8), (8, 4000), (10000, 16), (16, 10000)] {
            let geometry = try config.geometry(imageWidth: width, imageHeight: height)
            #expect(geometry.patchCount > 0)
            #expect(geometry.patchCount <= config.maxPatches)
            #expect(min(geometry.targetWidth, geometry.targetHeight) == config.sideMultiple)
        }
    }

    @Test("Patch positions are row-major and pooling is row-major over the pooled grid")
    func patchOrdering() throws {
        let config = try VisionPreprocessorConfig(maxSoftTokens: 70)
        let geometry = try config.geometry(imageWidth: 480, imageHeight: 1200)
        #expect(geometry.patchesWide == 15 && geometry.patchesHigh == 39)

        #expect(geometry.position(ofPatch: 0) == (x: 0, y: 0))
        #expect(geometry.position(ofPatch: 14) == (x: 14, y: 0))
        #expect(geometry.position(ofPatch: 15) == (x: 0, y: 1))

        // Pooled index = x/3 + (patchesWide/3) * (y/3).
        #expect(geometry.pooledIndex(ofPatch: 0, poolingKernelSize: 3) == 0)
        #expect(geometry.pooledIndex(ofPatch: 2, poolingKernelSize: 3) == 0)
        #expect(geometry.pooledIndex(ofPatch: 3, poolingKernelSize: 3) == 1)
        // Row 3 of the patch grid starts the second pooled row (15/3 = 5 wide).
        #expect(geometry.pooledIndex(ofPatch: 45, poolingKernelSize: 3) == 5)

        // Every pooled cell must receive exactly nine patches.
        var counts = [Int](repeating: 0, count: geometry.softTokenCount)
        for patch in 0..<geometry.patchCount {
            counts[geometry.pooledIndex(ofPatch: patch, poolingKernelSize: 3)] += 1
        }
        #expect(counts.allSatisfy { $0 == 9 })
    }

    @Test("Unsupported soft-token budgets are rejected")
    func rejectsUnsupportedBudget() {
        #expect(throws: VisionError.self) { try VisionPreprocessorConfig(maxSoftTokens: 100) }
        #expect(throws: VisionError.self) { try VisionPreprocessorConfig(maxSoftTokens: 0) }
    }
}
