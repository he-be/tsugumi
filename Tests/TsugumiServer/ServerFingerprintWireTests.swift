import Foundation
import NIOCore
import Testing
@testable import Tsugumi
@testable import TsugumiServerCore

private actor FingerprintStubBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("hello"))
        return ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 1, totalTokens: 4))
    }
}

/// SPEC §9 **RSP-5** (`system_fingerprint`) and §3 **EP-4** (`build_info`) on
/// the wire.
///
/// EP-4 says the two are the same value. That is only checkable from outside if
/// both are read in one test off one running server, which is what this suite
/// is for — `ServerBuildIdentityTests` checks the value itself, this one checks
/// that both places answer with it.
@Suite("RSP-5 / EP-4 fingerprint on the wire", .serialized)
struct ServerFingerprintWireTests {
    private static func started() async throws -> (TsugumiHTTPServer, Int) {
        let server = TsugumiHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: FingerprintStubBackend(),
            properties: ServerProperties(
                modelPath: "/tmp/tsugumi-tests/gemma4.moepack",
                contextLength: 16_384,
                chatTemplate: "template"))
        let channel = try await server.start(port: 0)
        guard let port = channel.localAddress?.port else {
            throw ServerRequestError.invalid(message: "no port")
        }
        return (server, port)
    }

    private static func completion(port: Int, stream: Bool) async throws -> Data {
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data("""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],
         "stream":\(stream),"stream_options":{"include_usage":true}}
        """.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        return data
    }

    /// Every `data:` payload of an SSE body except the `[DONE]` sentinel.
    static func streamChunks(_ data: Data) -> [[String: Any]] {
        String(decoding: data, as: UTF8.self)
            .components(separatedBy: "\n\n")
            .compactMap { block -> [String: Any]? in
                guard block.hasPrefix("data: ") else { return nil }
                let payload = String(block.dropFirst("data: ".count))
                guard payload != "[DONE]" else { return nil }
                return try? JSONSerialization.jsonObject(with: Data(payload.utf8))
                    as? [String: Any]
            }
    }

    /// EP-4: `build_info` is the one build-identity string, not a constant that
    /// says only which program is answering.
    @Test func EP_4_props_build_info_is_the_build_fingerprint() async throws {
        let (server, port) = try await Self.started()
        let (data, _) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/props")!)
        let props = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])

        #expect(props["build_info"] as? String == ServerBuildIdentity.fingerprint)

        try await server.shutdown()
    }

    /// RSP-5 with EP-4: the non-stream body carries `system_fingerprint`, and
    /// it is the *same value* `/props` reports — a client that pinned a cache
    /// on one has to be able to compare it with the other.
    @Test func RSP_5_non_stream_body_carries_the_props_build_info() async throws {
        let (server, port) = try await Self.started()
        let (propsData, _) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/props")!)
        let props = try #require(
            JSONSerialization.jsonObject(with: propsData) as? [String: Any])
        let body = try #require(JSONSerialization.jsonObject(
            with: try await Self.completion(port: port, stream: false)) as? [String: Any])

        let fingerprint = try #require(body["system_fingerprint"] as? String)
        #expect(fingerprint == props["build_info"] as? String)
        #expect(fingerprint == ServerBuildIdentity.fingerprint)

        try await server.shutdown()
    }

    /// RSP-5: OpenAI puts `system_fingerprint` on every `chat.completion.chunk`
    /// as well, the reference implementation included (`server-task.cpp`'s
    /// `add_delta` at the pin) — the role chunk, the content chunks, the
    /// `finish_reason` chunk and the usage chunk alike.
    @Test func RSP_5_every_sse_chunk_carries_the_system_fingerprint() async throws {
        let (server, port) = try await Self.started()
        let chunks = Self.streamChunks(
            try await Self.completion(port: port, stream: true))

        // role → content → finish_reason → usage.
        #expect(chunks.count >= 4)
        for chunk in chunks {
            #expect(chunk["system_fingerprint"] as? String
                == ServerBuildIdentity.fingerprint,
                "chunk \(chunk)")
        }

        try await server.shutdown()
    }
}
