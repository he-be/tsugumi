import Foundation

/// SPEC §3 **EP-6** `GET /slots`: one generation slot's state.
///
/// This machine has exactly one (DEV-3), so the array a route answers with is
/// always one element long. What the reference reports beside these two — the
/// whole sampler `params` block, `chat_format`, `lora`, the speculative limits,
/// the prompt and generated text of the request in flight — is not in the
/// EP-6 line and is not here.
public struct ServerSlotState: Equatable, Sendable {
    public let id: Int
    /// The reference's `is_processing`: this slot is generating right now.
    public let isProcessing: Bool

    public init(id: Int, isProcessing: Bool) {
        self.id = id
        self.isProcessing = isProcessing
    }
}

/// EP-6: what the queue in front of the generation slot looks like right now.
///
/// This is the thing the runbook currently reads out of stderr — whether the
/// server is busy and how many requests are stacked behind it.
public struct ServerQueueState: Equatable, Sendable {
    /// `GET /slots`, one element (DEV-3).
    public let slots: [ServerSlotState]
    /// `/metrics` `requests_processing`.
    public let processingCount: Int
    /// `/metrics` `requests_deferred`: admitted requests waiting for the slot.
    public let deferredCount: Int

    public init(slots: [ServerSlotState], processingCount: Int, deferredCount: Int) {
        self.slots = slots
        self.processingCount = processingCount
        self.deferredCount = deferredCount
    }
}

/// EP-6 `GET /metrics`: the counters the server has run up since it started.
///
/// Every number here is a sum of what RSP-3 already measures per request, so
/// the two cannot disagree about how much work was done. The prompt count is
/// the *processed* tokens only, as the reference's `prompt_tokens_total` is —
/// a token served out of the KV cost no time and would flatter the rate.
public struct ServerMetricsSnapshot: Equatable, Sendable {
    /// `llamacpp:prompt_tokens_total`.
    public let promptTokensTotal: Int
    /// `llamacpp:prompt_seconds_total`.
    public let promptSecondsTotal: Double
    /// `llamacpp:tokens_predicted_total`.
    public let predictedTokensTotal: Int
    /// `llamacpp:tokens_predicted_seconds_total`.
    public let predictedSecondsTotal: Double

    public static let zero = ServerMetricsSnapshot(promptTokensTotal: 0,
                                                   promptSecondsTotal: 0,
                                                   predictedTokensTotal: 0,
                                                   predictedSecondsTotal: 0)

    public init(promptTokensTotal: Int,
                promptSecondsTotal: Double,
                predictedTokensTotal: Int,
                predictedSecondsTotal: Double) {
        self.promptTokensTotal = promptTokensTotal
        self.promptSecondsTotal = promptSecondsTotal
        self.predictedTokensTotal = predictedTokensTotal
        self.predictedSecondsTotal = predictedSecondsTotal
    }

    /// `llamacpp:prompt_tokens_seconds`.
    public var promptTokensPerSecond: Double { 0 }

    /// `llamacpp:predicted_tokens_seconds`.
    public var predictedTokensPerSecond: Double { 0 }

    /// One finished completion added to the running totals.
    public func adding(_ timings: ServerTimings) -> ServerMetricsSnapshot { self }
}
