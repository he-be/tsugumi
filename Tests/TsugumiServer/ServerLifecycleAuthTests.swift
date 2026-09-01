import Foundation
import NIOCore
import Testing
@testable import Tsugumi
@testable import TsugumiServerCore

private actor AuthStubBackend: ServerInferenceBackend {
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

/// FLAG-5, the `--api-key` half. The key guards the endpoints that carry the
/// model; the ones a client needs before it can authenticate anything are
/// exempt (EP-1, and the reference's public set at the pin).
@Suite("Server API key", .serialized)
struct ServerLifecycleAuthTests {
    private static let key = "sk-test-key"
    private static let otherKey = "sk-second-key"

    private static func started(
        keys: [String],
        loaded: Bool = true
    ) async throws -> (TsugumiHTTPServer, Int) {
        let server = TsugumiHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: loaded ? AuthStubBackend() : nil,
            apiKeys: keys)
        let channel = try await server.start(port: 0)
        guard let port = channel.localAddress?.port else {
            throw ServerRequestError.invalid(message: "no port")
        }
        return (server, port)
    }

    /// One GET, with whatever authorization headers the caller wants.
    private static func get(
        _ path: String, port: Int, headers: [String: String] = [:]
    ) async throws -> (Int, JSONValue) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        for (name, value) in headers {
            request.setValue(value, forHTTPHeaderField: name)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        return ((response as? HTTPURLResponse)?.statusCode ?? 0,
                (try? JSONDecoder().decode(JSONValue.self, from: data)) ?? .null)
    }

    private static func postCompletion(
        port: Int, headers: [String: String] = [:]
    ) async throws -> (Int, JSONValue) {
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        for (name, value) in headers {
            request.setValue(value, forHTTPHeaderField: name)
        }
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        return ((response as? HTTPURLResponse)?.statusCode ?? 0,
                (try? JSONDecoder().decode(JSONValue.self, from: data)) ?? .null)
    }

    private static func errorField(_ body: JSONValue, _ name: String) -> JSONValue? {
        body.objectValue?["error"]?.objectValue?[name]
    }

    /// FLAG-5: with no `--api-key`, nothing changes. The 127.0.0.1 bind is the
    /// whole defence, exactly as it was, so a server started the way every
    /// existing runbook starts one is not suddenly refusing its own client.
    @Test func FLAG_5_without_the_flag_no_request_needs_a_key() async throws {
        let (server, port) = try await Self.started(keys: [])

        #expect(try await Self.postCompletion(port: port).0 == 200)
        #expect(try await Self.get("/props", port: port).0 == 200)
        #expect(try await Self.get("/v1/models", port: port).0 == 200)

        try await server.shutdown()
    }

    /// FLAG-5 / ERR-1: a missing key is an envelope, not a bare status. The
    /// shape is OpenAI's own answer to this case — 401 carrying
    /// `invalid_request_error` and `code: "invalid_api_key"` — which outranks
    /// the reference for `/v1/*` wire format (SPEC §0).
    @Test func FLAG_5_a_request_without_a_key_is_401_invalid_api_key() async throws {
        let (server, port) = try await Self.started(keys: [Self.key])

        let (status, body) = try await Self.postCompletion(port: port)
        #expect(status == 401)
        #expect(Self.errorField(body, "type") == .string("invalid_request_error"))
        #expect(Self.errorField(body, "code") == .string("invalid_api_key"))
        #expect(Self.errorField(body, "param") == .null)
        if case .string(let message)? = Self.errorField(body, "message") {
            #expect(!message.isEmpty)
        } else {
            Issue.record("the envelope carries no message")
        }

        try await server.shutdown()
    }

    /// FLAG-5: a key that is not one of the configured ones is the same 401 as
    /// no key at all — the answer must not tell the caller which of the two it
    /// got wrong.
    @Test func FLAG_5_a_wrong_key_is_401() async throws {
        let (server, port) = try await Self.started(keys: [Self.key])

        let (status, body) = try await Self.postCompletion(
            port: port, headers: ["Authorization": "Bearer sk-not-the-key"])
        #expect(status == 401)
        #expect(Self.errorField(body, "code") == .string("invalid_api_key"))

        try await server.shutdown()
    }

    /// FLAG-5: the three spellings the reference accepts at the pin —
    /// `Authorization: Bearer <key>`, the same header with the bare value, and
    /// Anthropic's `X-Api-Key`.
    @Test func FLAG_5_bearer_bare_and_x_api_key_headers_are_accepted() async throws {
        let (server, port) = try await Self.started(keys: [Self.key])

        for headers in [
            ["Authorization": "Bearer \(Self.key)"],
            ["Authorization": Self.key],
            ["X-Api-Key": Self.key],
        ] {
            let (status, _) = try await Self.postCompletion(port: port, headers: headers)
            #expect(status == 200, "\(headers.keys.sorted().joined())")
        }

        try await server.shutdown()
    }

    /// FLAG-5: several keys, so a key can be rotated without a restart that
    /// drops every client at once.
    @Test func FLAG_5_every_configured_key_is_accepted() async throws {
        let (server, port) = try await Self.started(keys: [Self.key, Self.otherKey])

        for key in [Self.key, Self.otherKey] {
            let (status, _) = try await Self.postCompletion(
                port: port, headers: ["Authorization": "Bearer \(key)"])
            #expect(status == 200, "\(key)")
        }

        try await server.shutdown()
    }

    /// EP-1 and the reference's public set: `/health`, its `/v1` alias, and
    /// both spellings of the model list answer without a key — a client polls
    /// health before it has been configured with anything, and the model list
    /// is what tells it which model it is talking to. Everything that reaches
    /// the model needs the key.
    @Test func FLAG_5_health_and_models_need_no_key_but_props_and_completions_do() async throws {
        let (server, port) = try await Self.started(keys: [Self.key])

        for path in ["/health", "/v1/health", "/models", "/v1/models"] {
            #expect(try await Self.get(path, port: port).0 == 200, "\(path)")
        }
        #expect(try await Self.get("/props", port: port).0 == 401)
        #expect(try await Self.postCompletion(port: port).0 == 401)

        try await server.shutdown()
    }

    /// FLAG-5 with LIF-2 / LIF-6: the load gate is decided first, so a keyless
    /// request during the load is 503 and not 401. The reference orders its two
    /// middlewares the same way (`server-http.cpp:302`).
    @Test func FLAG_5_the_load_gate_is_answered_before_the_key_check() async throws {
        let (server, port) = try await Self.started(keys: [Self.key], loaded: false)

        let (status, body) = try await Self.postCompletion(port: port)
        #expect(status == 503)
        #expect(Self.errorField(body, "code") == .string("model_loading"))

        await server.modelDidLoad(AuthStubBackend())
        #expect(try await Self.postCompletion(port: port).0 == 401)

        try await server.shutdown()
    }

    /// FLAG-5 with EP-7: the key is checked ahead of the routing table, so an
    /// unauthenticated caller cannot map which paths exist by reading 404s and
    /// 501s off a server it has no key for.
    @Test func FLAG_5_an_unknown_path_answers_401_before_404() async throws {
        let (server, port) = try await Self.started(keys: [Self.key])

        #expect(try await Self.get("/nope", port: port).0 == 401)
        #expect(try await Self.get("/v1/embeddings", port: port).0 == 401)

        let authorized = ["Authorization": "Bearer \(Self.key)"]
        #expect(try await Self.get("/nope", port: port, headers: authorized).0 == 404)
        #expect(try await Self.get("/v1/embeddings", port: port,
                                   headers: authorized).0 == 501)

        try await server.shutdown()
    }

    /// FLAG-5: `--api-key` takes one key or a comma-separated list, the way the
    /// reference spells it at the pin.
    @Test func FLAG_5_the_flag_takes_a_comma_separated_list() throws {
        let single = try ServerArguments.parse(["--model", "m.moepack",
                                                "--api-key", "one"])
        #expect(single.apiKeys == ["one"])

        let several = try ServerArguments.parse(["--model", "m.moepack",
                                                 "--api-key", "one,two , three"])
        #expect(several.apiKeys == ["one", "two", "three"])

        let repeated = try ServerArguments.parse(["--model", "m.moepack",
                                                  "--api-key", "one",
                                                  "--api-key", "two"])
        #expect(repeated.apiKeys == ["one", "two"])

        #expect(try ServerArguments.parse(["--model", "m.moepack"]).apiKeys == [])

        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--model", "m.moepack", "--api-key", " , "])
        }
    }
}
