import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

private actor TextEndpointGate {
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        await withCheckedContinuation { continuation = $0 }
    }

    func open() {
        continuation?.resume()
        continuation = nil
    }
}

/// SPEC §3 **EP-5** (`/tokenize`, `/detokenize`, `/apply-template`) and
/// **EP-6** (`/slots`, `/metrics`) on the inference side of the boundary.
///
/// EP-5 needs the tokenizer only, so it is C2; EP-6's queue half is the
/// coordinator and its counter half is arithmetic, so neither needs weights.
@Suite("EP-5 / EP-6 backend surface")
struct ServerEndpointBackendTests {
    private static let sentence = "The capital of France is Paris."

    /// EP-5: `/detokenize` is the inverse of `/tokenize`. A client that
    /// pre-computes a token count and then asks what those tokens say has to
    /// get its text back.
    @Test func EP_5_detokenize_inverts_tokenize() async throws {
        let endpoints = ServerTextEndpoints(tokenizer: try await GFTokenizer.load())
        let tokens = endpoints.tokenize(Self.sentence, addSpecial: false)

        #expect(!tokens.isEmpty)
        #expect(endpoints.detokenize(tokens) == Self.sentence)
    }

    /// EP-5: `add_special` is the reference's flag for prepending `<bos>`, and
    /// its default there is false.
    @Test func EP_5_tokenize_add_special_prepends_bos() async throws {
        let tokenizer = try await GFTokenizer.load()
        let endpoints = ServerTextEndpoints(tokenizer: tokenizer)

        let plain = endpoints.tokenize(Self.sentence, addSpecial: false)
        let special = endpoints.tokenize(Self.sentence, addSpecial: true)

        #expect(special.first == tokenizer.bosID)
        #expect(Array(special.dropFirst()) == plain)
    }

    /// EP-5 with SPEC §12 **DEV-12**: `/apply-template` must describe the
    /// template this server actually prefills with, not the one the checkpoint
    /// ships. The check is the strongest one available: the string it answers
    /// with has to encode to the very tokens `ServerPromptRenderer` builds.
    @Test(arguments: [false, true], [false, true])
    func EP_5_apply_template_renders_the_variant_the_server_prefills_with(
        thinking: Bool,
        tools: Bool
    ) async throws {
        let tokenizer = try await GFTokenizer.load()
        let request = try Self.request(thinking: thinking, tools: tools)
        let rendered = try ServerTextEndpoints(tokenizer: tokenizer)
            .applyChatTemplate(request)

        #expect(tokenizer.encode(rendered, addBOS: false)
            == (try ServerPromptRenderer(tokenizer: tokenizer).promptIDs(request)))
    }

    /// EP-6 `/slots` and the two request gauges of `/metrics`: the one
    /// generation slot (DEV-3) and how many requests are behind it. This is
    /// the runbook's "is it jammed" question, answered without stderr.
    @Test func EP_6_queue_state_reports_the_slot_and_the_waiting_requests() async throws {
        let coordinator = ServerCoordinator(queueLimit: 2)
        let idle = await coordinator.queueState
        #expect(idle.slots == [ServerSlotState(id: 0, isProcessing: false)])
        #expect(idle.processingCount == 0)
        #expect(idle.deferredCount == 0)

        let gate = TextEndpointGate()
        let active = Task { try await coordinator.run { await gate.wait() } }
        while await !coordinator.isActive { await Task.yield() }
        let queued = Task { try await coordinator.run {} }
        while await coordinator.queuedCount != 1 { await Task.yield() }

        let busy = await coordinator.queueState
        #expect(busy.slots == [ServerSlotState(id: 0, isProcessing: true)])
        #expect(busy.processingCount == 1)
        #expect(busy.deferredCount == 1)

        await gate.open()
        try await active.value
        try await queued.value
        #expect(await coordinator.queueState.deferredCount == 0)
    }

    /// EP-6 `/metrics`: the counters are sums of what RSP-3 measured, so the
    /// two can never tell different stories about the same work. The prompt
    /// count is the processed half only — a cached token cost no time.
    @Test func EP_6_metrics_accumulate_the_timings_of_every_completion() {
        let first = ServerTimings(cacheTokens: 100,
                                  promptTokens: 20,
                                  promptMilliseconds: 100,
                                  predictedTokens: 11,
                                  predictedMilliseconds: 1_000)
        let second = ServerTimings(cacheTokens: 0,
                                   promptTokens: 80,
                                   promptMilliseconds: 400,
                                   predictedTokens: 21,
                                   predictedMilliseconds: 1_000)
        let metrics = ServerMetricsSnapshot.zero.adding(first).adding(second)

        #expect(metrics.promptTokensTotal == 100)
        #expect(metrics.promptSecondsTotal == 0.5)
        #expect(metrics.predictedTokensTotal == 32)
        #expect(metrics.predictedSecondsTotal == 2.0)
        #expect(metrics.promptTokensPerSecond == 200)
        #expect(metrics.predictedTokensPerSecond == 16)
    }

    /// A server that has answered nothing reports zeros rather than a rate it
    /// cannot compute.
    @Test func EP_6_metrics_start_at_zero() {
        #expect(ServerMetricsSnapshot.zero.promptTokensPerSecond == 0)
        #expect(ServerMetricsSnapshot.zero.predictedTokensPerSecond == 0)
    }

    private static func request(thinking: Bool, tools: Bool) throws -> ValidatedChatRequest {
        let toolsJSON = #"""
        ,"tools":[{"type":"function","function":{
          "name":"lookup","description":"Look something up",
          "parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}]
        """#
        let body = #"{"model":"m","messages":[{"role":"user","content":"Where is Paris?"}]"#
            + (tools ? toolsJSON : "")
            + #","chat_template_kwargs":{"enable_thinking":"# + "\(thinking)}}"
        return try ChatRequestParser.parse(Data(body.utf8))
    }
}
