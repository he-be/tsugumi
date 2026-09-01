import Foundation
import NIOCore
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

private actor MonitoringGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private(set) var isWaiting = false

    func wait() async {
        isWaiting = true
        await withCheckedContinuation { continuation = $0 }
        isWaiting = false
    }

    func open() {
        continuation?.resume()
        continuation = nil
    }
}

/// A backend that holds the one generation slot until it is let go, and reports
/// a fixed set of counters. Both halves are what EP-6 answers with.
private actor MonitoringStubBackend: ServerInferenceBackend {
    static let snapshot = ServerMetricsSnapshot(promptTokensTotal: 240,
                                                promptSecondsTotal: 1.5,
                                                predictedTokensTotal: 64,
                                                predictedSecondsTotal: 4)

    let gate: MonitoringGate?

    init(gate: MonitoringGate? = nil) {
        self.gate = gate
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        await gate?.wait()
        return ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }

    func metrics() async -> ServerMetricsSnapshot { Self.snapshot }
}

/// SPEC §3 **EP-6**: `GET /slots` and `GET /metrics`, and the startup flags
/// that gate them.
///
/// The flag spellings are the reference implementation's at the pin
/// `34af94cd9` (`tools/server/README.md`): `--slots` / `--no-slots`, enabled by
/// default, and `--metrics`, disabled by default. A request to an endpoint that
/// was not started is EP-7's answer — 501 `not_supported_error`.
@Suite("EP-6 slots and metrics", .serialized)
struct ServerSlotsMetricsTests {
    private static func started(
        gate: MonitoringGate? = nil,
        slots: Bool = true,
        metrics: Bool = false
    ) async throws -> (TurboFieldfareHTTPServer, Int) {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 2,
            backend: MonitoringStubBackend(gate: gate),
            slotsEndpointEnabled: slots,
            metricsEndpointEnabled: metrics)
        let channel = try await server.start(port: 0)
        guard let port = channel.localAddress?.port else {
            throw ServerRequestError.invalid(message: "no port")
        }
        return (server, port)
    }

    static func get(_ path: String, port: Int) async throws -> (Int, String, Data) {
        let (data, response) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        let http = response as? HTTPURLResponse
        return (http?.statusCode ?? 0,
                http?.value(forHTTPHeaderField: "content-type") ?? "",
                data)
    }

    /// EP-6 with §12 **DEV-3**: this machine has exactly one generation slot,
    /// so the array is one element long and its id is 0.
    @Test func EP_6_slots_reports_the_one_generation_slot() async throws {
        let (server, port) = try await Self.started()
        let (status, contentType, data) = try await Self.get("/slots", port: port)

        #expect(status == 200)
        #expect(contentType.hasPrefix("application/json"))
        let slots = try #require(
            JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        #expect(slots.count == 1)
        #expect(slots.first?["id"] as? Int == 0)
        #expect(slots.first?["is_processing"] as? Bool == false)

        try await server.shutdown()
    }

    /// EP-6's reason for existing: "is it jammed", answered without reading
    /// stderr. While a generation holds the slot, `/slots` says so.
    @Test func EP_6_slots_reports_the_slot_as_processing_while_it_generates() async throws {
        let gate = MonitoringGate()
        let (server, port) = try await Self.started(gate: gate)

        var completion = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        completion.httpMethod = "POST"
        completion.setValue("application/json", forHTTPHeaderField: "content-type")
        completion.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """#.utf8)
        let inFlight = Task { try await URLSession.shared.data(for: completion) }
        while await !gate.isWaiting { await Task.yield() }

        let (status, _, data) = try await Self.get("/slots", port: port)
        #expect(status == 200)
        let slots = try #require(
            JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        #expect(slots.first?["is_processing"] as? Bool == true)

        await gate.open()
        _ = try await inFlight.value
        try await server.shutdown()
    }

    /// EP-6 `/metrics` is Prometheus text, not JSON: the names, the `# HELP` and
    /// `# TYPE` lines and the content type are all what a scraper reads.
    @Test func EP_6_metrics_answers_with_prometheus_text() async throws {
        let (server, port) = try await Self.started(metrics: true)
        let (status, contentType, data) = try await Self.get("/metrics", port: port)
        let text = String(decoding: data, as: UTF8.self)

        #expect(status == 200)
        #expect(contentType.hasPrefix("text/plain"))
        for name in ["prompt_tokens_total", "prompt_seconds_total",
                     "tokens_predicted_total", "tokens_predicted_seconds_total",
                     "prompt_tokens_seconds", "predicted_tokens_seconds",
                     "requests_processing", "requests_deferred"] {
            #expect(text.contains("# HELP llamacpp:\(name) "), "\(name)")
            #expect(text.contains("# TYPE llamacpp:\(name) "), "\(name)")
        }
        #expect(text.contains("llamacpp:prompt_tokens_total 240\n"))
        #expect(text.contains("llamacpp:prompt_seconds_total 1.5\n"))
        #expect(text.contains("llamacpp:tokens_predicted_total 64\n"))
        #expect(text.contains("llamacpp:tokens_predicted_seconds_total 4\n"))
        #expect(text.contains("llamacpp:prompt_tokens_seconds 160\n"))
        #expect(text.contains("llamacpp:predicted_tokens_seconds 16\n"))
        #expect(text.contains("llamacpp:requests_processing 0\n"))
        #expect(text.contains("llamacpp:requests_deferred 0\n"))

        try await server.shutdown()
    }

    /// EP-6: both endpoints are gated by a startup flag, as in the reference,
    /// and an ungated request gets EP-7's shape — 501 `not_supported_error`,
    /// with the flag named so an operator can turn it on.
    @Test func EP_6_ungated_slots_and_metrics_answer_501_not_supported() async throws {
        let (server, port) = try await Self.started(slots: false, metrics: false)

        for (path, flag) in [("/slots", "--slots"), ("/metrics", "--metrics")] {
            let (status, _, data) = try await Self.get(path, port: port)
            #expect(status == 501, "\(path)")
            let body = try #require(
                JSONSerialization.jsonObject(with: data) as? [String: Any])
            let error = try #require(body["error"] as? [String: Any])
            #expect(error["type"] as? String == "not_supported_error", "\(path)")
            #expect((error["message"] as? String)?.contains(flag) == true, "\(path)")
        }

        try await server.shutdown()
    }

    /// EP-6's defaults are the reference's: `/slots` is on unless it is turned
    /// off, `/metrics` is off unless it is turned on.
    @Test func EP_6_endpoint_flags_follow_the_reference_defaults() throws {
        let bare = try ServerArguments.parse(["--model", "/tmp/model"])
        #expect(bare.slotsEndpointEnabled)
        #expect(!bare.metricsEndpointEnabled)

        let off = try ServerArguments.parse(["--model", "/tmp/model", "--no-slots"])
        #expect(!off.slotsEndpointEnabled)

        let on = try ServerArguments.parse(
            ["--model", "/tmp/model", "--slots", "--metrics"])
        #expect(on.slotsEndpointEnabled)
        #expect(on.metricsEndpointEnabled)
    }

    /// The three take no value, so the flag after them is a flag and not their
    /// argument — the parser's "requires a value" rule must not swallow it.
    @Test func EP_6_endpoint_flags_take_no_value() throws {
        let arguments = try ServerArguments.parse(
            ["--metrics", "--no-slots", "--model", "/tmp/model", "--port", "9999"])

        #expect(arguments.metricsEndpointEnabled)
        #expect(!arguments.slotsEndpointEnabled)
        #expect(arguments.model == "/tmp/model")
        #expect(arguments.port == 9_999)
    }
}
