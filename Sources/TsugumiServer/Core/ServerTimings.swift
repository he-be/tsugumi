import Foundation
import Synchronization
import Tsugumi

/// SPEC §9 **RSP-3**: what one completion cost.
///
/// The field names and the way each is derived are the reference
/// implementation's (`server_slot_stats::to_json` / the rate helpers in
/// `server-common.h` at the pin), because SPEC §0 makes the reference the
/// norm for behaviour detail. SPEC names eight of them and this type carries
/// exactly those eight.
///
/// The prompt is partitioned the same way `usage` partitions it (RSP-1):
/// `cache_n` are the prompt tokens the KV already held and `prompt_n` are the
/// ones this request computed, so `cache_n + prompt_n` is `usage.prompt_tokens`
/// and `cache_n + prompt_n + predicted_n` is what the context holds after the
/// answer — which is the sum SPEC's last sentence promises a client can take.
public struct ServerTimings: Equatable, Sendable {
    /// `cache_n`: prompt tokens served out of the KV without being computed.
    public let cacheTokens: Int
    /// `prompt_n`: prompt tokens this request actually prefilled.
    public let promptTokens: Int
    /// `prompt_ms`: wall time spent prefilling those tokens.
    public let promptMilliseconds: Double
    /// `predicted_n`: tokens generated.
    public let predictedTokens: Int
    /// `predicted_ms`: wall time spent generating them.
    public let predictedMilliseconds: Double

    /// `draft_n`: proposals the drafter made across the whole request. Zero for
    /// a request that did not speculate, and the two keys then stay off the
    /// wire entirely (`jsonObject`).
    public let draftTokens: Int
    /// `draft_n_accepted`: how many of those the target's own draw agreed with.
    public let draftAcceptedTokens: Int

    public init(cacheTokens: Int,
                promptTokens: Int,
                promptMilliseconds: Double,
                predictedTokens: Int,
                predictedMilliseconds: Double,
                draftTokens: Int = 0,
                draftAcceptedTokens: Int = 0) {
        self.cacheTokens = cacheTokens
        self.promptTokens = promptTokens
        self.promptMilliseconds = promptMilliseconds
        self.predictedTokens = predictedTokens
        self.predictedMilliseconds = predictedMilliseconds
        self.draftTokens = draftTokens
        self.draftAcceptedTokens = draftAcceptedTokens
    }

    /// The measurements the decode loop already takes, read off its result.
    public init(_ result: RawDecodeResult) {
        self.init(cacheTokens: result.cachedPromptTokens,
                  promptTokens: result.computedPrefillTokens,
                  promptMilliseconds: result.prefillSeconds * 1_000,
                  predictedTokens: result.newTokens,
                  predictedMilliseconds: result.decodeSeconds * 1_000)
    }

    /// RSP-3 / GEN-14: the same measurements plus what the speculative loop
    /// reported, for a request that ran it. `nil` — the plain path — leaves the
    /// two counters at zero, which is what keeps them off the wire.
    public init(_ result: RawDecodeResult, speculative: ServerSpeculativeSummary?) {
        self.init(cacheTokens: result.cachedPromptTokens,
                  promptTokens: result.computedPrefillTokens,
                  promptMilliseconds: result.prefillSeconds * 1_000,
                  predictedTokens: result.newTokens,
                  predictedMilliseconds: result.decodeSeconds * 1_000,
                  draftTokens: speculative?.proposed ?? 0,
                  draftAcceptedTokens: speculative?.accepted ?? 0)
    }

    /// RSP-3's last sentence: what this request left in the context.
    public var contextTokens: Int { cacheTokens + promptTokens + predictedTokens }

    /// Decode steps spent generating. The first token is free — it comes out of
    /// the logits the last prompt batch already wrote — so it is not a step,
    /// which is the divisor the reference uses for both generated rates
    /// (`server_slot_stats::n_gen_steps`).
    var predictedSteps: Int { predictedTokens > 0 ? predictedTokens - 1 : 0 }

    /// `prompt_per_second`.
    public var promptTokensPerSecond: Double {
        promptMilliseconds > 0 ? 1_000 / promptMilliseconds * Double(promptTokens) : 0
    }

    /// `predicted_per_token_ms`.
    public var predictedMillisecondsPerToken: Double {
        predictedSteps > 0 ? predictedMilliseconds / Double(predictedSteps) : 0
    }

    /// `predicted_per_second`.
    public var predictedTokensPerSecond: Double {
        predictedMilliseconds > 0 ? 1_000 / predictedMilliseconds * Double(predictedSteps) : 0
    }

    /// The `timings` object as it goes on the wire.
    ///
    /// The two draft counters are written **only when something was drafted**,
    /// which is the reference's own gate (`server_slot_stats::to_json` writes
    /// them under `n_draft_tokens > 0`, pin `34af94cd9`). That is what lets a
    /// client tell "MTP did not run" from "MTP ran and accepted nothing": the
    /// first has no keys, the second has `draft_n > 0` with
    /// `draft_n_accepted == 0`. CONFORMANCE §5 measures the everyday client
    /// exactly this way.
    public var jsonObject: [String: Any] {
        var object: [String: Any] = [
            "cache_n": cacheTokens,
            "prompt_n": promptTokens,
            "prompt_ms": promptMilliseconds,
            "prompt_per_second": promptTokensPerSecond,
            "predicted_n": predictedTokens,
            "predicted_ms": predictedMilliseconds,
            "predicted_per_token_ms": predictedMillisecondsPerToken,
            "predicted_per_second": predictedTokensPerSecond,
        ]
        if draftTokens > 0 {
            object["draft_n"] = draftTokens
            object["draft_n_accepted"] = draftAcceptedTokens
        }
        return object
    }
}

/// RSP-3's `timings_per_token: true`: the timings of a generation still in
/// flight.
///
/// Generation runs inside the backend actor and a route writes its SSE chunks
/// from the synchronous event callback, so there is no point at which the route
/// could `await` the actor for a number about the very work that is holding it.
/// The backend writes into this box as each token lands and the route reads it
/// while writing the chunk that token produced.
public final class ServerTimingsMonitor: Sendable {
    private let state = Mutex<ServerTimings?>(nil)

    public init() {}

    /// The timings as of the most recent token, or nil before the first one.
    public var current: ServerTimings? { state.withLock { $0 } }

    func record(_ timings: ServerTimings) {
        state.withLock { $0 = timings }
    }
}

/// The running clock behind `ServerTimingsMonitor`.
///
/// The prompt half is known before generation starts — the prompt cache decided
/// how much of the prompt is computed and how much is reused — so the only
/// thing this has to follow is the generated half and the wall clock.
///
/// **The prompt wall time here is not the one the finished response carries.**
/// Neither are the draft counters: the round bookkeeping is reported when the
/// speculative loop returns, so a running snapshot has nothing to say about it
/// and says nothing rather than a made-up zero. The final chunk and the
/// non-stream body carry them (RSP-3 / GEN-14).
/// The decode loop measures its own prefill and only reports it in
/// `RawDecodeResult`, when the generation is over; a request that asked for
/// per-token timings needs a number before that, so `prompt_ms` is measured
/// here as the wall time from the start of the completion call to the first
/// token — time to first token, which includes the first sample. The final
/// chunk and the non-stream body carry the loop's own measurement instead.
struct ServerLiveTimings {
    private let cacheTokens: Int
    private let promptTokens: Int
    private let startedAt: Date
    private var promptMilliseconds: Double = 0
    private var firstTokenAt: Date?

    init(cacheTokens: Int, promptTokens: Int, startedAt: Date) {
        self.cacheTokens = cacheTokens
        self.promptTokens = promptTokens
        self.startedAt = startedAt
    }

    /// The timings as of this progress event, or nil for an event that is not a
    /// generated token — prefill progress and the flushed tail move no counter.
    mutating func observe(_ progress: RawDecodeProgress, at now: Date) -> ServerTimings? {
        guard case .token(let index, _, _) = progress else { return nil }
        let generationStart: Date
        if let firstTokenAt {
            generationStart = firstTokenAt
        } else {
            firstTokenAt = now
            generationStart = now
            promptMilliseconds = now.timeIntervalSince(startedAt) * 1_000
        }
        return ServerTimings(
            cacheTokens: cacheTokens,
            promptTokens: promptTokens,
            promptMilliseconds: promptMilliseconds,
            predictedTokens: index + 1,
            predictedMilliseconds: now.timeIntervalSince(generationStart) * 1_000)
    }
}
