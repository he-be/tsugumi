import Foundation
import NIOCore
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// C1: a real HTTP server whose backend is a stub, so the load state can be
/// held open for as long as the test likes without weights or Metal
/// (CONFORMANCE §1).
private actor LifecycleStubBackend: ServerInferenceBackend {
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

@Suite("Server lifecycle", .serialized)
struct ServerLifecycleTests {
    /// SPEC LIF-2, quoted. The body is spelled out in the spec, so the test
    /// holds the whole value rather than a predicate over parts of it. The
    /// comparison is on the decoded JSON because an object's key order is not
    /// part of the value (and `JSONEncoder` does not promise one).
    private static let loadingBody =
        #"{"error":{"message":"Loading model","type":"unavailable_error","param":null,"code":"model_loading"}}"#

    private static let healthyBody = #"{"status":"ok"}"#

    private static func json(_ data: Data) throws -> JSONValue {
        try JSONDecoder().decode(JSONValue.self, from: data)
    }

    private static func json(literal: String) throws -> JSONValue {
        try json(Data(literal.utf8))
    }

    /// LIF-1: the process opens the port before the model is loaded. What the
    /// client must be able to tell apart is "refused" from "loading", and the
    /// only evidence for that is an answered request while no model exists.
    @Test func LIF_1_port_is_open_before_the_model_is_loaded() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: nil)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (_, response) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/health")!)
        #expect((response as? HTTPURLResponse)?.statusCode == 503)
        #expect(await !server.isReady)

        try await server.shutdown()
    }

    /// LIF-2: `/health`'s loading body, verbatim.
    @Test func LIF_2_health_returns_503_while_loading() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: nil)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (data, response) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/health")!)
        #expect((response as? HTTPURLResponse)?.statusCode == 503)
        #expect(try Self.json(data) == Self.json(literal: Self.loadingBody))

        try await server.shutdown()
    }

    /// LIF-2: *every* endpoint, not only `/health`. A client that posts a
    /// completion during the load gets a status it can retry, never a hang and
    /// never a refused connection.
    @Test func LIF_2_every_endpoint_returns_503_unavailable_while_loading() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: nil)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        for path in ["/health", "/v1/health", "/v1/models", "/props"] {
            let (data, response) = try await URLSession.shared.data(
                from: URL(string: "http://127.0.0.1:\(port)\(path)")!)
            #expect((response as? HTTPURLResponse)?.statusCode == 503,
                    "GET \(path) while loading")
            #expect(String(decoding: data, as: UTF8.self)
                .contains(#""type":"unavailable_error""#), "GET \(path) while loading")
        }

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """#.utf8)
        let (body, completion) = try await URLSession.shared.data(for: request)
        #expect((completion as? HTTPURLResponse)?.statusCode == 503)
        #expect(String(decoding: body, as: UTF8.self)
            .contains(#""code":"model_loading""#))

        try await server.shutdown()
    }

    /// LIF-3: once the load lands, `/health` is `200 {"status":"ok"}` — and the
    /// endpoints that were 503 a moment ago answer for real.
    @Test func LIF_3_health_is_ok_once_the_model_is_loaded() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: nil)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let loading = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/health")!)
        #expect((loading.1 as? HTTPURLResponse)?.statusCode == 503)

        await server.modelDidLoad(LifecycleStubBackend())
        #expect(await server.isReady)

        let (data, response) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/health")!)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        #expect(try Self.json(data) == Self.json(literal: Self.healthyBody))

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """#.utf8)
        let completion = try await URLSession.shared.data(for: request)
        #expect((completion.1 as? HTTPURLResponse)?.statusCode == 200)
        #expect(String(decoding: completion.0, as: UTF8.self).contains("hello"))

        try await server.shutdown()
    }

    /// EP-1: `/v1/health` is an alias of `/health`, in both states, with no API
    /// key. Clients that only know the `/v1` prefix must not have to special
    /// case this one path.
    @Test func EP_1_health_answers_on_both_spellings() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: nil)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        for path in ["/health", "/v1/health"] {
            let (data, response) = try await URLSession.shared.data(
                from: URL(string: "http://127.0.0.1:\(port)\(path)")!)
            #expect((response as? HTTPURLResponse)?.statusCode == 503, "\(path)")
            #expect(try Self.json(data) == Self.json(literal: Self.loadingBody), "\(path)")
        }

        await server.modelDidLoad(LifecycleStubBackend())

        for path in ["/health", "/v1/health"] {
            let (data, response) = try await URLSession.shared.data(
                from: URL(string: "http://127.0.0.1:\(port)\(path)")!)
            #expect((response as? HTTPURLResponse)?.statusCode == 200, "\(path)")
            #expect(try Self.json(data) == Self.json(literal: Self.healthyBody), "\(path)")
        }

        try await server.shutdown()
    }

    /// LIF-5: a signal that arrives during the load has to be able to end the
    /// process, which only means anything now that the port is open first
    /// (LIF-1). Shutting a server down before it is ready closes the listener
    /// and returns rather than waiting for a model that never arrives.
    @Test func LIF_5_shutdown_while_loading_closes_the_listener() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: nil)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let health = URL(string: "http://127.0.0.1:\(port)/health")!
        _ = try await URLSession.shared.data(from: health)

        try await server.shutdown()

        await #expect(throws: (any Error).self) {
            _ = try await URLSession.shared.data(from: health)
        }
    }
}
