import Foundation
import NIOCore
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

private actor GateStubBackend: ServerInferenceBackend {
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

/// The gates the five endpoints of **EP-5** and **EP-6** stand behind.
///
/// None of them is new: LIF-2/LIF-6 puts the load gate ahead of the routing
/// table, FLAG-5 puts the key between the two, and FLAG-6 answers preflight
/// ahead of both. What is new is five routes, and a route that was added past
/// one of those gates would be a hole nobody would notice — so the placement is
/// checked here for each of them rather than assumed from where the code sits.
@Suite("EP-5 / EP-6 gates", .serialized)
struct ServerEndpointGateTests {
    private static let key = "sk-gate-key"

    /// The five, with the verb each one takes.
    private static let endpoints: [(method: String, path: String)] = [
        ("POST", "/tokenize"),
        ("POST", "/detokenize"),
        ("POST", "/apply-template"),
        ("GET", "/slots"),
        ("GET", "/metrics"),
    ]

    private static func started(
        loaded: Bool = true,
        keys: [String] = [],
        cors: ServerCORSPolicy = .disabled,
        slots: Bool = true,
        metrics: Bool = true
    ) async throws -> (TurboFieldfareHTTPServer, Int) {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: loaded ? GateStubBackend() : nil,
            apiKeys: keys,
            corsPolicy: cors,
            slotsEndpointEnabled: slots,
            metricsEndpointEnabled: metrics)
        let channel = try await server.start(port: 0)
        guard let port = channel.localAddress?.port else {
            throw ServerRequestError.invalid(message: "no port")
        }
        return (server, port)
    }

    private static func send(
        _ method: String,
        _ path: String,
        port: Int,
        headers: [String: String] = [:]
    ) async throws -> (Int, JSONValue, HTTPURLResponse?) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        request.httpMethod = method
        if method == "POST" {
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            request.httpBody = Data(#"""
            {"model":"test-model","content":"hi","tokens":[],
             "messages":[{"role":"user","content":"hi"}]}
            """#.utf8)
        }
        for (name, value) in headers {
            request.setValue(value, forHTTPHeaderField: name)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = response as? HTTPURLResponse
        return (http?.statusCode ?? 0,
                (try? JSONDecoder().decode(JSONValue.self, from: data)) ?? .null,
                http)
    }

    /// LIF-2 / LIF-6: the load gate is ahead of the routing table, so every one
    /// of the five is 503 `model_loading` while the model comes up — including
    /// `/metrics` with its flag turned **off**, which would be a 501 if the
    /// table were consulted first. That difference is the whole point of LIF-6:
    /// a loading server gives away nothing about its routes.
    @Test func LIF_2_ep5_and_ep6_answer_503_while_the_model_loads() async throws {
        let (server, port) = try await Self.started(
            loaded: false, slots: false, metrics: false)

        for endpoint in Self.endpoints {
            let (status, body, _) = try await Self.send(
                endpoint.method, endpoint.path, port: port)
            #expect(status == 503, "\(endpoint.method) \(endpoint.path)")
            #expect(body.objectValue?["error"]?.objectValue?["code"]
                    == .string("model_loading"), "\(endpoint.method) \(endpoint.path)")
        }

        try await server.shutdown()
    }

    /// FLAG-5: none of the five is in the key-free set — that is EP-1 and EP-2
    /// only. They all reach the model's tokenizer or its counters, so a caller
    /// with no key gets 401 and cannot tell them apart from a route that does
    /// not exist.
    @Test func FLAG_5_ep5_and_ep6_are_behind_the_api_key() async throws {
        let (server, port) = try await Self.started(keys: [Self.key])

        for endpoint in Self.endpoints {
            let (refused, body, _) = try await Self.send(
                endpoint.method, endpoint.path, port: port)
            #expect(refused == 401, "\(endpoint.method) \(endpoint.path)")
            #expect(body.objectValue?["error"]?.objectValue?["code"]
                    == .string("invalid_api_key"), "\(endpoint.method) \(endpoint.path)")

            let (accepted, _, _) = try await Self.send(
                endpoint.method, endpoint.path, port: port,
                headers: ["authorization": "Bearer \(Self.key)"])
            #expect(accepted == 200, "\(endpoint.method) \(endpoint.path)")
        }

        try await server.shutdown()
    }

    /// FLAG-6 with LIF-6: preflight is answered ahead of the load gate and
    /// ahead of the key, identically for every path. A browser never puts an
    /// `Authorization` on a preflight and cannot read a 503 body it could not
    /// preflight for, so both gates have to stand behind it.
    @Test func FLAG_6_preflight_for_ep5_and_ep6_answers_ahead_of_both_gates() async throws {
        let origin = "https://example.test"
        let (server, port) = try await Self.started(
            loaded: false,
            keys: [Self.key],
            cors: .origins([origin]),
            slots: false,
            metrics: false)

        for endpoint in Self.endpoints {
            let (status, _, response) = try await Self.send(
                "OPTIONS", endpoint.path, port: port, headers: ["origin": origin])
            #expect(status == 200, "OPTIONS \(endpoint.path)")
            #expect(response?.value(forHTTPHeaderField: "access-control-allow-origin")
                    == origin, "OPTIONS \(endpoint.path)")
            #expect(response?.value(forHTTPHeaderField: "access-control-allow-methods")
                    == "GET, POST, OPTIONS", "OPTIONS \(endpoint.path)")
        }

        try await server.shutdown()
    }
}
