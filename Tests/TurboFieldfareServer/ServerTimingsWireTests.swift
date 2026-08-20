import Foundation
import NIOCore
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// A backend that measures. Three tokens, with the running timings published
/// into the monitor just before the event each token produced, which is the
/// order `ServerModelSession` publishes them in.
private actor TimingsStubBackend: ServerInferenceBackend {
    static let deltas = ["a", "b", "c"]

    /// What the finished completion cost — the numbers RSP-3 puts on the
    /// non-stream body and the last chunk.
    static let final = ServerTimings(cacheTokens: 40,
                                     promptTokens: 60,
                                     promptMilliseconds: 500,
                                     predictedTokens: 3,
                                     predictedMilliseconds: 200)

    /// The running timings as of token `index`.
    static func running(_ index: Int) -> ServerTimings {
        ServerTimings(cacheTokens: 40,
                      promptTokens: 60,
                      promptMilliseconds: 500,
                      predictedTokens: index + 1,
                      predictedMilliseconds: Double(index) * 100)
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        try await generate(ServerPreparedRequest(request: request),
                           monitor: nil,
                           onEvent: onEvent)
    }

    func generate(
        _ prepared: ServerPreparedRequest,
        monitor: ServerTimingsMonitor?,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        for (index, delta) in Self.deltas.enumerated() {
            monitor?.record(Self.running(index))
            onEvent(.content(delta))
        }
        return ServerCompletion(
            content: Self.deltas.joined(),
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 100,
                               completionTokens: 3,
                               totalTokens: 103,
                               cachedTokens: 40),
            timings: Self.final)
    }
}

/// SPEC §9 **RSP-3** on the wire.
///
/// `ServerTimingsTests` checks the arithmetic; this checks that the object
/// reaches the client — on the non-stream body, on the last chunk, and on
/// every chunk when the request asked for `timings_per_token`.
@Suite("RSP-3 timings on the wire", .serialized)
struct ServerTimingsWireTests {
    private static func started() async throws -> (TurboFieldfareHTTPServer, Int) {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: TimingsStubBackend())
        let channel = try await server.start(port: 0)
        guard let port = channel.localAddress?.port else {
            throw ServerRequestError.invalid(message: "no port")
        }
        return (server, port)
    }

    private static func completion(port: Int, options: String) async throws -> Data {
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data("""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]\(options)}
        """.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        return data
    }

    /// The eight fields of RSP-3 read back off a chunk or a body.
    private static func expectTimings(_ object: [String: Any]?,
                                      equal expected: ServerTimings,
                                      _ label: Comment) {
        guard let timings = object else {
            Issue.record("\(label): no timings object")
            return
        }
        #expect(Set(timings.keys) == Set(expected.jsonObject.keys), label)
        #expect(timings["cache_n"] as? Int == expected.cacheTokens, label)
        #expect(timings["prompt_n"] as? Int == expected.promptTokens, label)
        #expect(timings["prompt_ms"] as? Double == expected.promptMilliseconds, label)
        #expect(timings["predicted_n"] as? Int == expected.predictedTokens, label)
        #expect(timings["predicted_ms"] as? Double == expected.predictedMilliseconds,
                label)
    }

    /// RSP-3: the non-stream body carries the completion's `timings`, and the
    /// object partitions the prompt the same way `usage` does — which is what
    /// lets a client add the three counts up into context usage.
    @Test func RSP_3_non_stream_body_carries_the_completion_timings() async throws {
        let (server, port) = try await Self.started()
        let body = try #require(JSONSerialization.jsonObject(
            with: try await Self.completion(port: port, options: "")) as? [String: Any])

        Self.expectTimings(body["timings"] as? [String: Any],
                           equal: TimingsStubBackend.final,
                           "non-stream body")
        let usage = try #require(body["usage"] as? [String: Any])
        let timings = try #require(body["timings"] as? [String: Any])
        #expect((timings["cache_n"] as? Int ?? 0) + (timings["prompt_n"] as? Int ?? 0)
                == usage["prompt_tokens"] as? Int)

        try await server.shutdown()
    }

    /// RSP-3: "the final chunk". Without `include_usage` that is the
    /// `finish_reason` chunk, and the chunks before it carry none — the request
    /// did not ask for per-token timings.
    @Test func RSP_3_final_sse_chunk_carries_the_completion_timings() async throws {
        let (server, port) = try await Self.started()
        let chunks = ServerFingerprintWireTests.streamChunks(
            try await Self.completion(port: port, options: #","stream":true"#))

        let last = try #require(chunks.last)
        let choices = try #require(last["choices"] as? [[String: Any]])
        #expect(choices.first?["finish_reason"] as? String == "stop")
        Self.expectTimings(last["timings"] as? [String: Any],
                           equal: TimingsStubBackend.final,
                           "final chunk")
        for chunk in chunks.dropLast() {
            #expect(chunk["timings"] == nil, "chunk \(chunk)")
        }

        try await server.shutdown()
    }

    /// With `include_usage` the usage chunk is the last one written, so that is
    /// where the timings ride — the reference puts them on `deltas.back()`
    /// (`server-task.cpp` at the pin), whichever chunk that turns out to be.
    @Test func RSP_3_timings_ride_the_usage_chunk_when_it_is_last() async throws {
        let (server, port) = try await Self.started()
        let chunks = ServerFingerprintWireTests.streamChunks(
            try await Self.completion(
                port: port,
                options: #","stream":true,"stream_options":{"include_usage":true}"#))

        let last = try #require(chunks.last)
        #expect(last["usage"] != nil)
        Self.expectTimings(last["timings"] as? [String: Any],
                           equal: TimingsStubBackend.final,
                           "usage chunk")
        for chunk in chunks.dropLast() {
            #expect(chunk["timings"] == nil, "chunk \(chunk)")
        }

        try await server.shutdown()
    }

    /// RSP-3's `timings_per_token: true`: every chunk carries the timings as of
    /// the token that produced it. The role chunk is written before the first
    /// token exists, so it has nothing to carry — a number there would be a
    /// made-up one.
    @Test func RSP_3_timings_per_token_puts_running_timings_on_every_chunk() async throws {
        let (server, port) = try await Self.started()
        let chunks = ServerFingerprintWireTests.streamChunks(
            try await Self.completion(
                port: port,
                options: #","stream":true,"timings_per_token":true"#))

        // role → three content chunks → finish_reason.
        #expect(chunks.count == TimingsStubBackend.deltas.count + 2)
        #expect(chunks.first?["timings"] == nil)
        for (index, chunk) in chunks.dropFirst().dropLast().enumerated() {
            let choices = try #require(chunk["choices"] as? [[String: Any]])
            let delta = try #require(choices.first?["delta"] as? [String: Any])
            #expect(delta["content"] as? String == TimingsStubBackend.deltas[index])
            Self.expectTimings(chunk["timings"] as? [String: Any],
                               equal: TimingsStubBackend.running(index),
                               "content chunk \(index)")
        }
        Self.expectTimings(chunks.last?["timings"] as? [String: Any],
                           equal: TimingsStubBackend.final,
                           "final chunk")

        try await server.shutdown()
    }
}
