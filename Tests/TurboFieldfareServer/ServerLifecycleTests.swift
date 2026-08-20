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

    /// LIF-6: the 503 is decided ahead of the routing table, so during the load
    /// even a path this server has never heard of answers 503 and not 404 —
    /// the reference implementation refuses from a middleware in front of its
    /// own routes (`server-http.cpp:255`). A client cannot conclude anything
    /// about a route from a server that has not finished starting.
    @Test func LIF_6_unknown_paths_are_503_while_loading_and_404_after() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: nil)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let unknown = URL(string: "http://127.0.0.1:\(port)/nope")!
        // `/v1/embeddings` is a known-but-unsupported path (EP-7): its 501 is
        // decided by the same table, so it is 503 while loading too.
        let unsupported = URL(string: "http://127.0.0.1:\(port)/v1/embeddings")!

        for url in [unknown, unsupported] {
            let (data, response) = try await URLSession.shared.data(from: url)
            #expect((response as? HTTPURLResponse)?.statusCode == 503, "\(url.path)")
            #expect(try Self.json(data) == Self.json(literal: Self.loadingBody),
                    "\(url.path)")
        }

        await server.modelDidLoad(LifecycleStubBackend())

        let after = try await URLSession.shared.data(from: unknown)
        #expect((after.1 as? HTTPURLResponse)?.statusCode == 404)
        let unsupportedAfter = try await URLSession.shared.data(from: unsupported)
        #expect((unsupportedAfter.1 as? HTTPURLResponse)?.statusCode == 501)

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

    /// LIF-7: a load that fails ends the process — the port is already open by
    /// then (LIF-1), so the alternative would be a server that answers 503
    /// `model_loading` forever for a model that is never coming.
    ///
    /// Driven as a subprocess because the exit is `main.swift`'s. The model
    /// path does not exist, so this never loads weights and never touches
    /// Metal: it fails in the tokenizer folder lookup, before any of that.
    @Test(.enabled(if: ServerLifecycleTests.freshServerBinary != nil))
    func LIF_7_a_failed_load_exits_the_process() async throws {
        let binary = try #require(Self.freshServerBinary)
        // A port that was free a moment ago, taken the way the tests take one.
        let probe = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: nil)
        let port = try #require(try await probe.start(port: 0).localAddress?.port)
        try await probe.shutdown()

        let process = Process()
        process.executableURL = binary
        process.arguments = [
            "--model", "/nonexistent/turbofieldfare-tests/no-such-model.gturbo",
            "--port", "\(port)",
        ]
        let output = Pipe()
        let errors = Pipe()
        process.standardOutput = output
        process.standardError = errors
        try process.run()
        let stdout = String(
            decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        let stderr = String(
            decoding: errors.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        process.waitUntilExit()

        // LIF-1 happened first: the port was open while the load was running.
        #expect(stdout.contains("listening"))
        // LIF-7: the reason is on stderr and the status is 1.
        #expect(stderr.hasPrefix("error: "))
        #expect(process.terminationStatus == 1)
    }

    /// The built server binary, but only when it is at least as new as the
    /// sources it would be built from: `swift test` does not build an
    /// executable product, so an older one would be answering for code that is
    /// no longer here.
    private static let freshServerBinary: URL? = {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sources = [
            "Sources/TurboFieldfareServer/Command/main.swift",
            "Sources/TurboFieldfareServer/Core/HTTPServer.swift",
        ].compactMap {
            modificationDate(root.appendingPathComponent($0))
        }
        guard let newestSource = sources.max() else { return nil }
        for configuration in ["debug", "release"] {
            let candidate = root.appendingPathComponent(
                ".build/\(configuration)/TurboFieldfareServer")
            guard FileManager.default.isExecutableFile(atPath: candidate.path),
                  let built = modificationDate(candidate),
                  built >= newestSource else { continue }
            return candidate
        }
        return nil
    }()

    private static func modificationDate(_ url: URL) -> Date? {
        try? FileManager.default
            .attributesOfItem(atPath: url.path)[.modificationDate] as? Date
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
