import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppContextLengthOptionTests {
    @Test func optionsUseSupportedContextLengthsInAscendingOrder() {
        #expect(AppContextLengthOption.allCases.map(\.tokens)
            == [4_096, 8_192, 16_384, 32_768, 65_536, 131_072])
    }

    @Test func optionsReportProductionFP16KVAllocation() {
        let mebibytes = AppContextLengthOption.allCases.map {
            $0.fp16KVBytes / 1_048_576
        }
        // The base moved +375 MiB when the prefill chunk default went 128 ->
        // 2048: sliding rows are `window + chunkTokens`. The per-context
        // deltas below are unchanged, so the slope is the same.
        #expect(mebibytes == [680, 760, 920, 1_240, 1_880, 3_160])
        #expect(AppContextLengthOption.allCases.map(\.menuLabel) == [
            "4K, Default",
            "8K, +85 MB",
            "16K, +250 MB",
            "32K, +590 MB",
            "64K, +1.26 GB",
            "128K, +2.6 GB",
        ])
    }
}
