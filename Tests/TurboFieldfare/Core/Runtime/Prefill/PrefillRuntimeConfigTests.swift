import Testing
@testable import TurboFieldfare

@Suite struct PrefillRuntimeConfigTests {
    @Test(arguments: RuntimeConfiguration.allowedPrefillChunkTokens)
    func productionUsesCompleteChunkedPath(_ chunkTokens: Int) throws {
        let config = PrefillRuntimeConfig.production(chunkTokens: chunkTokens)
        #expect(config.mode == .chunked)
        #expect(config.chunkTokens == chunkTokens)
    }

    @Test func offDisablesChunkedPrefill() {
        let config = PrefillRuntimeConfig.off
        #expect(config.mode == .off)
        #expect(!config.enabled)
    }

    @Test func plannerUsesConfiguredChunkSize() {
        let spans = PrefillChunkPlanner.spans(
            tokenCount: 130,
            startPosition: 7,
            config: .production(chunkTokens: 64))
        #expect(spans.map(\.tokenCount) == [64, 64, 2])
        #expect(spans.map(\.startPosition) == [7, 71, 135])
    }

    // MARK: - Image spans (PLAN_VISION §0-A-1)

    private func imageSpan(_ offset: Int, _ count: Int, index: Int = 0) -> VisionImageSpan {
        VisionImageSpan(imageIndex: index, tokenOffset: offset, tokenCount: count)
    }

    @Test func plannerCutsBeforeAnImageRatherThanThroughIt() {
        // A 512-token chunk would cut at 512, inside the image at [400, 656).
        let spans = PrefillChunkPlanner.spans(
            tokenCount: 1_200,
            startPosition: 0,
            chunkTokens: 512,
            imageSpans: [imageSpan(400, 256)])
        #expect(spans.map(\.tokenOffset) == [0, 400, 912])
        #expect(spans.map(\.tokenCount) == [400, 512, 288])
        for span in spans {
            let end = span.tokenOffset + span.tokenCount
            #expect(!(span.tokenOffset > 400 && span.tokenOffset < 656),
                    "chunk starts inside the image at \(span.tokenOffset)")
            #expect(!(end > 400 && end < 656), "chunk ends inside the image at \(end)")
        }
    }

    /// An image that begins exactly on a chunk boundary needs no adjustment,
    /// and one that ends on it needs none either.
    @Test func plannerLeavesAlignedImagesAlone() {
        let spans = PrefillChunkPlanner.spans(
            tokenCount: 256,
            startPosition: 0,
            chunkTokens: 64,
            imageSpans: [imageSpan(64, 64)])
        #expect(spans.map(\.tokenOffset) == [0, 64, 128, 192])
        #expect(spans.allSatisfy { $0.tokenCount == 64 })
    }

    @Test func plannerHandlesSeveralImagesInOneChunk() {
        let spans = PrefillChunkPlanner.spans(
            tokenCount: 300,
            startPosition: 11,
            chunkTokens: 2048,
            imageSpans: [imageSpan(10, 64, index: 0), imageSpan(80, 64, index: 1)])
        #expect(spans.count == 1)
        #expect(spans[0].tokenCount == 300)
        #expect(spans[0].startPosition == 11)
    }

    @Test func plannerWithoutImagesIsUnchanged() {
        let withEmptySpans = PrefillChunkPlanner.spans(
            tokenCount: 130, startPosition: 7, chunkTokens: 64, imageSpans: [])
        let baseline = PrefillChunkPlanner.spans(
            tokenCount: 130, startPosition: 7, config: .production(chunkTokens: 64))
        #expect(withEmptySpans.map(\.tokenOffset) == baseline.map(\.tokenOffset))
        #expect(withEmptySpans.map(\.tokenCount) == baseline.map(\.tokenCount))
    }

    @Test func imageWiderThanTheChunkIsRejectedWithAWayOut() {
        let rejection = PrefillChunkPlanner.imageSpanRejection(
            imageSpans: [imageSpan(10, 256)],
            tokenCount: 400,
            chunkTokens: 128)
        #expect(rejection?.contains("--prefill-chunk-tokens") == true,
                "rejection should say what to change, got \(rejection ?? "nil")")
        #expect(PrefillChunkPlanner.imageSpanRejection(imageSpans: [imageSpan(10, 256)],
                                                       tokenCount: 400,
                                                       chunkTokens: 2048) == nil)
    }

    @Test func overlappingOrOutOfRangeSpansAreRejected() {
        #expect(PrefillChunkPlanner.imageSpanRejection(
            imageSpans: [imageSpan(10, 40, index: 0), imageSpan(30, 40, index: 1)],
            tokenCount: 400,
            chunkTokens: 2048) != nil)
        #expect(PrefillChunkPlanner.imageSpanRejection(
            imageSpans: [imageSpan(380, 40)],
            tokenCount: 400,
            chunkTokens: 2048) != nil)
    }

    @Test func diagnosticsPreserveUnknownValues() {
        let diagnostics = PrefillExecutionDiagnostics(
            config: .production(chunkTokens: 128),
            executedMode: .unsupported,
            kvStorageMode: nil,
            unsupportedReason: "unavailable")
        #expect(diagnostics.kvStorageMode == nil)
        #expect(diagnostics.chunkCompleteness == .unsupported)
        #expect(diagnostics.unsupportedReason == "unavailable")
    }
}
