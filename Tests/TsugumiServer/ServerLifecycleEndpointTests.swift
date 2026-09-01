import Foundation
import NIOCore
import Testing
@testable import Tsugumi
@testable import TsugumiServerCore

private actor EndpointStubBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

/// EP-7. The routing table's answer to a path it knows about but does not
/// implement, and to one it has never heard of.
@Suite("Server endpoint routing", .serialized)
struct ServerLifecycleEndpointTests {
    private static func started() async throws -> (TsugumiHTTPServer, Int) {
        let server = TsugumiHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: EndpointStubBackend())
        let channel = try await server.start(port: 0)
        guard let port = channel.localAddress?.port else {
            throw ServerRequestError.invalid(message: "no port")
        }
        return (server, port)
    }

    private static func send(
        method: String, path: String, port: Int
    ) async throws -> (Int, JSONValue) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        request.httpMethod = method
        if method == "POST" {
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            request.httpBody = Data("{}".utf8)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        return (status, try JSONDecoder().decode(JSONValue.self, from: data))
    }

    /// SPEC §3's list of known paths this server does not adopt (DEV-7). 501
    /// says "this server will never do that", which is what a client needs to
    /// stop retrying — 404 would say "you have the path wrong".
    @Test func EP_7_known_unsupported_paths_return_501_not_supported() async throws {
        let (server, port) = try await Self.started()

        let posted = [
            "/v1/embeddings", "/embedding", "/reranking", "/rerank", "/infill",
            "/v1/responses", "/v1/messages", "/v1/chat/completions/control",
            "/props", "/lora-adapters", "/v1/completions", "/completion",
            "/slots/0",
        ]
        for path in posted {
            let (status, body) = try await Self.send(
                method: "POST", path: path, port: port)
            #expect(status == 501, "POST \(path)")
            #expect(body.objectValue?["error"]?.objectValue?["type"]
                    == .string("not_supported_error"), "POST \(path)")
        }

        // `/lora-adapters` and `/slots/{id}?action=…` are read as well as
        // written in the reference implementation; neither is adopted.
        for path in ["/lora-adapters", "/slots/0?action=erase"] {
            let (status, body) = try await Self.send(
                method: "GET", path: path, port: port)
            #expect(status == 501, "GET \(path)")
            #expect(body.objectValue?["error"]?.objectValue?["type"]
                    == .string("not_supported_error"), "GET \(path)")
        }

        try await server.shutdown()
    }

    /// EP-7: only a genuinely unknown path is a 404 — and ERR-2 fixes the type
    /// that goes with the number.
    @Test func EP_7_unknown_paths_stay_404_not_found() async throws {
        let (server, port) = try await Self.started()

        for path in ["/nope", "/v1/nope/deeper"] {
            let (status, body) = try await Self.send(
                method: "GET", path: path, port: port)
            #expect(status == 404, "GET \(path)")
            #expect(body.objectValue?["error"]?.objectValue?["type"]
                    == .string("not_found_error"), "GET \(path)")
        }

        try await server.shutdown()
    }

    /// EP-8: the reference implementation serves `/models` and
    /// `/chat/completions` under both spellings, so a client configured with a
    /// base URL that has no `/v1` on it works either way. Same handler, same
    /// body — not a redirect.
    @Test func EP_8_models_and_completions_answer_without_the_v1_prefix() async throws {
        let (server, port) = try await Self.started()

        let (aliasStatus, aliasBody) = try await Self.send(
            method: "GET", path: "/models", port: port)
        let (canonicalStatus, canonicalBody) = try await Self.send(
            method: "GET", path: "/v1/models", port: port)
        #expect(aliasStatus == 200)
        #expect(canonicalStatus == 200)
        #expect(aliasBody == canonicalBody)

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let completion = try JSONDecoder().decode(JSONValue.self, from: data)
        #expect(completion.objectValue?["object"] == .string("chat.completion"))

        try await server.shutdown()
    }

    /// EP-8 / ERR-2: an alias is the same endpoint, so the wrong verb on it is
    /// the same 405 — not a 404 and not a 501.
    @Test func EP_8_aliases_answer_405_for_the_wrong_method() async throws {
        let (server, port) = try await Self.started()

        let (models, modelsBody) = try await Self.send(
            method: "POST", path: "/models", port: port)
        #expect(models == 405)
        #expect(modelsBody.objectValue?["error"]?.objectValue?["code"]
                == .string("method_not_allowed"))

        let (completions, completionsBody) = try await Self.send(
            method: "GET", path: "/chat/completions", port: port)
        #expect(completions == 405)
        #expect(completionsBody.objectValue?["error"]?.objectValue?["code"]
                == .string("method_not_allowed"))

        try await server.shutdown()
    }

    /// EP-4 vs EP-7: the same path, two answers. `GET /props` is the
    /// capability endpoint; `POST /props` is the reference implementation's
    /// write side, which this server does not adopt.
    @Test func EP_7_props_is_readable_but_not_writable() async throws {
        let (server, port) = try await Self.started()

        let read = try await Self.send(method: "GET", path: "/props", port: port)
        #expect(read.0 == 200)

        let write = try await Self.send(method: "POST", path: "/props", port: port)
        #expect(write.0 == 501)

        try await server.shutdown()
    }
}
