import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// SPEC §9 **RSP-3**: the `timings` object.
///
/// These need neither weights nor Metal — every claim here is arithmetic over
/// numbers the decode loop already reports (`RawDecodeResult`) plus a clock the
/// test supplies, which is what keeps the per-token half checkable at all.
@Suite("RSP-3 timings")
struct ServerTimingsTests {
    private static func result(cached: Int,
                               computed: Int,
                               prefillSeconds: Double,
                               generated: Int,
                               decodeSeconds: Double) -> RawDecodeResult {
        RawDecodeResult(prefillTokens: cached + computed,
                        cachedPromptTokens: cached,
                        computedPrefillTokens: computed,
                        prefillSeconds: prefillSeconds,
                        newTokens: generated,
                        decodeSeconds: decodeSeconds,
                        timeToFirstTokenSeconds: prefillSeconds,
                        reason: .endOfTurn,
                        kvPosition: cached + computed + generated,
                        kvBackedTokenIDs: [],
                        uncommittedBoundaryTokenIDs: [])
    }

    /// RSP-3 names eight fields and the object carries those eight — no more
    /// (the reference's `prompt_per_token_ms` and its draft counters are not
    /// in the SPEC line) and no fewer.
    @Test func RSP_3_timings_object_carries_the_eight_named_fields() {
        let timings = ServerTimings(cacheTokens: 3,
                                    promptTokens: 7,
                                    promptMilliseconds: 100,
                                    predictedTokens: 5,
                                    predictedMilliseconds: 200)
        let object = timings.jsonObject

        #expect(Set(object.keys) == [
            "cache_n", "prompt_n", "prompt_ms", "prompt_per_second",
            "predicted_n", "predicted_ms", "predicted_per_token_ms",
            "predicted_per_second",
        ])
        #expect(object["cache_n"] as? Int == 3)
        #expect(object["prompt_n"] as? Int == 7)
        #expect(object["predicted_n"] as? Int == 5)
        #expect(object["prompt_ms"] as? Double == 100)
        #expect(object["predicted_ms"] as? Double == 200)
    }

    /// RSP-3's last sentence: a client computes context usage as
    /// `prompt_n + cache_n + predicted_n`, so those three have to partition the
    /// prompt exactly the way `usage` does (RSP-1) — `cache_n + prompt_n` is
    /// `usage.prompt_tokens` and `cache_n` is `cached_tokens`.
    @Test func RSP_3_prompt_n_and_cache_n_partition_the_prompt_like_usage() {
        let decoded = Self.result(cached: 120,
                                  computed: 40,
                                  prefillSeconds: 0.4,
                                  generated: 11,
                                  decodeSeconds: 1.0)
        let timings = ServerTimings(decoded)
        let usage = OpenAIUsage(promptTokens: decoded.prefillTokens,
                                completionTokens: decoded.newTokens,
                                totalTokens: decoded.prefillTokens + decoded.newTokens,
                                cachedTokens: decoded.cachedPromptTokens)

        #expect(timings.cacheTokens + timings.promptTokens == usage.promptTokens)
        #expect(timings.cacheTokens == usage.promptTokensDetails.cachedTokens)
        #expect(timings.predictedTokens == usage.completionTokens)
        #expect(timings.contextTokens == 171)
    }

    /// The rates are the reference implementation's (`server-common.h` at the
    /// pin): the prompt rate is over the tokens that were computed, and the
    /// generated rate is over decode *steps* — the first token is free, it comes
    /// out of the logits the last prompt batch already wrote.
    @Test func RSP_3_rates_follow_the_reference_arithmetic() {
        let timings = ServerTimings(cacheTokens: 0,
                                    promptTokens: 200,
                                    promptMilliseconds: 500,
                                    predictedTokens: 11,
                                    predictedMilliseconds: 1_000)

        #expect(timings.promptTokensPerSecond == 400)
        #expect(timings.predictedMillisecondsPerToken == 100)
        #expect(timings.predictedTokensPerSecond == 10)
    }

    /// Every rate is zero when its divisor is not known yet, exactly as the
    /// reference returns 0.0 — a fully cached prompt spends no prefill time and
    /// a one-token answer takes no decode step.
    @Test func RSP_3_rates_are_zero_when_the_divisor_is_unknown() {
        let cachedWhole = ServerTimings(cacheTokens: 160,
                                        promptTokens: 0,
                                        promptMilliseconds: 0,
                                        predictedTokens: 1,
                                        predictedMilliseconds: 12)

        #expect(cachedWhole.promptTokensPerSecond == 0)
        #expect(cachedWhole.predictedMillisecondsPerToken == 0)
        #expect(cachedWhole.predictedTokensPerSecond == 0)
    }

    /// RSP-3 puts `timings` on the non-stream body and the final chunk, so the
    /// completion the route is handed has to carry them.
    @Test func RSP_3_completion_carries_the_timings_of_its_decode() {
        let decoded = Self.result(cached: 8,
                                  computed: 2,
                                  prefillSeconds: 0.1,
                                  generated: 3,
                                  decodeSeconds: 0.2)
        let completion = ServerCompletion(
            content: "hi",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 10, completionTokens: 3, totalTokens: 13),
            timings: ServerTimings(decoded))

        #expect(completion.timings == ServerTimings(cacheTokens: 8,
                                                    promptTokens: 2,
                                                    promptMilliseconds: 100,
                                                    predictedTokens: 3,
                                                    predictedMilliseconds: 200))
    }

    /// `timings_per_token: true` wants a `timings` on *every* chunk, so the
    /// running numbers have to exist before the decode loop returns. The prompt
    /// half is settled before the first token; the generated half counts up.
    @Test func RSP_3_timings_per_token_counts_up_while_generating() {
        let start = Date(timeIntervalSince1970: 1_000)
        var live = ServerLiveTimings(cacheTokens: 40, promptTokens: 60, startedAt: start)

        #expect(live.observe(.prefill(done: 60, total: 100),
                             at: start.addingTimeInterval(0.2)) == nil)

        let first = live.observe(.token(index: 0, id: 7, delta: "a"),
                                 at: start.addingTimeInterval(0.5))
        #expect(first?.cacheTokens == 40)
        #expect(first?.promptTokens == 60)
        #expect(first?.promptMilliseconds == 500)
        #expect(first?.predictedTokens == 1)
        #expect(first?.predictedMilliseconds == 0)

        let third = live.observe(.token(index: 2, id: 9, delta: "c"),
                                 at: start.addingTimeInterval(1.5))
        #expect(third?.promptMilliseconds == 500)
        #expect(third?.predictedTokens == 3)
        #expect(third?.predictedMilliseconds == 1_000)

        #expect(live.observe(.tail("!"), at: start.addingTimeInterval(1.6)) == nil)
    }

    /// The box the route reads while it writes a chunk: empty until the first
    /// token, then the latest snapshot.
    @Test func RSP_3_timings_monitor_publishes_the_latest_snapshot() {
        let monitor = ServerTimingsMonitor()
        #expect(monitor.current == nil)

        let timings = ServerTimings(cacheTokens: 1,
                                    promptTokens: 2,
                                    promptMilliseconds: 3,
                                    predictedTokens: 4,
                                    predictedMilliseconds: 5)
        monitor.record(timings)
        #expect(monitor.current == timings)
    }
}
