import Foundation
import NIOCore
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

private actor CORSStubBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("hello"))
        return ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

/// FLAG-6. CORS is the browser's enforcement, not ours: everything here is
/// about which headers go out, never about refusing a request.
@Suite("Server CORS", .serialized)
struct ServerLifecycleCORSTests {
    private static let allowed = "http://allowed.example"
    private static let alsoAllowed = "https://second.example"
    private static let stranger = "http://stranger.example"

    private static func started(
        cors: ServerCORSPolicy,
        keys: [String] = [],
        loaded: Bool = true
    ) async throws -> (TurboFieldfareHTTPServer, Int) {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: loaded ? CORSStubBackend() : nil,
            apiKeys: keys,
            corsPolicy: cors)
        let channel = try await server.start(port: 0)
        guard let port = channel.localAddress?.port else {
            throw ServerRequestError.invalid(message: "no port")
        }
        return (server, port)
    }

    private static func send(
        method: String,
        path: String,
        port: Int,
        origin: String? = nil,
        headers: [String: String] = [:],
        body: String? = nil
    ) async throws -> HTTPURLResponse {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        request.httpMethod = method
        if let origin { request.setValue(origin, forHTTPHeaderField: "Origin") }
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            request.httpBody = Data(body.utf8)
        }
        let (_, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ServerRequestError.invalid(message: "not an HTTP response")
        }
        return http
    }

    private static let completionBody =
        #"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#

    private static func allowOrigin(_ response: HTTPURLResponse) -> String? {
        response.value(forHTTPHeaderField: "Access-Control-Allow-Origin")
    }

    private static func vary(_ response: HTTPURLResponse) -> String? {
        response.value(forHTTPHeaderField: "Vary")
    }

    /// FLAG-6 / DEV-20: with no `--cors-origins`, no response carries a CORS
    /// header and `OPTIONS` is not a preflight — the 127.0.0.1 bind is the
    /// whole defence, exactly as it was. The reference's `*`-by-default would
    /// open this server to any page the browser happens to have open.
    @Test func FLAG_6_without_the_flag_no_response_carries_a_cors_header() async throws {
        let (server, port) = try await Self.started(cors: .disabled)

        for path in ["/health", "/v1/models", "/props"] {
            let response = try await Self.send(
                method: "GET", path: path, port: port, origin: Self.allowed)
            #expect(response.statusCode == 200, "\(path)")
            #expect(Self.allowOrigin(response) == nil, "\(path)")
            #expect(Self.vary(response) == nil, "\(path)")
        }

        let completion = try await Self.send(
            method: "POST", path: "/v1/chat/completions", port: port,
            origin: Self.allowed, body: Self.completionBody)
        #expect(completion.statusCode == 200)
        #expect(Self.allowOrigin(completion) == nil)

        // No CORS configured is no preflight: OPTIONS is just a verb this
        // endpoint does not take (ERR-2's 405), the way it is today.
        let preflight = try await Self.send(
            method: "OPTIONS", path: "/v1/chat/completions", port: port,
            origin: Self.allowed)
        #expect(preflight.statusCode == 405)
        #expect(Self.allowOrigin(preflight) == nil)

        try await server.shutdown()
    }

    /// FLAG-6 / DEV-20: a listed origin gets **itself** back, never the list —
    /// `Access-Control-Allow-Origin` takes one origin or `*`, and the
    /// reference's comma-separated value is one no browser accepts. The answer
    /// varies by `Origin`, so it says so.
    @Test func FLAG_6_a_listed_origin_gets_itself_back_with_vary() async throws {
        let (server, port) = try await Self.started(
            cors: .origins([Self.allowed, Self.alsoAllowed]))

        for origin in [Self.allowed, Self.alsoAllowed] {
            let response = try await Self.send(
                method: "GET", path: "/v1/models", port: port, origin: origin)
            #expect(response.statusCode == 200, "\(origin)")
            #expect(Self.allowOrigin(response) == origin, "\(origin)")
            #expect(Self.vary(response)?.contains("Origin") == true, "\(origin)")
        }

        try await server.shutdown()
    }

    /// FLAG-6: an origin that is not on the list simply gets no header. The
    /// request is still answered — enforcement is the browser's, and refusing
    /// here would break every non-browser client that sends an `Origin`.
    @Test func FLAG_6_an_unlisted_origin_gets_no_header_and_is_still_answered() async throws {
        let (server, port) = try await Self.started(cors: .origins([Self.allowed]))

        let response = try await Self.send(
            method: "GET", path: "/v1/models", port: port, origin: Self.stranger)
        #expect(response.statusCode == 200)
        #expect(Self.allowOrigin(response) == nil)

        let completion = try await Self.send(
            method: "POST", path: "/v1/chat/completions", port: port,
            origin: Self.stranger, body: Self.completionBody)
        #expect(completion.statusCode == 200)
        #expect(Self.allowOrigin(completion) == nil)

        try await server.shutdown()
    }

    /// FLAG-6: `*` answers `*`, and needs no `Vary` because the answer does not
    /// depend on the request's `Origin` at all.
    @Test func FLAG_6_star_echoes_star_without_vary() async throws {
        let (server, port) = try await Self.started(cors: .any)

        let withOrigin = try await Self.send(
            method: "GET", path: "/v1/models", port: port, origin: Self.stranger)
        #expect(Self.allowOrigin(withOrigin) == "*")
        #expect(Self.vary(withOrigin)?.contains("Origin") != true)

        // Even a request that sends no Origin at all.
        let withoutOrigin = try await Self.send(
            method: "GET", path: "/v1/models", port: port)
        #expect(Self.allowOrigin(withoutOrigin) == "*")

        try await server.shutdown()
    }

    /// FLAG-6 with LIF-6: preflight is answered ahead of the load gate. A
    /// browser that cannot preflight cannot read the 503 either, so it could
    /// not even show that the model is still loading.
    @Test func FLAG_6_preflight_is_answered_while_the_model_is_loading() async throws {
        let (server, port) = try await Self.started(
            cors: .origins([Self.allowed]), loaded: false)

        let preflight = try await Self.send(
            method: "OPTIONS", path: "/v1/chat/completions", port: port,
            origin: Self.allowed)
        #expect(preflight.statusCode == 200)
        #expect(Self.allowOrigin(preflight) == Self.allowed)
        #expect(preflight.value(forHTTPHeaderField: "Access-Control-Allow-Methods")
                == "GET, POST, OPTIONS")
        #expect(preflight.value(forHTTPHeaderField: "Access-Control-Allow-Headers")
                == "authorization, content-type, x-api-key")

        // And the request it is a preflight for is still 503 (LIF-2).
        let real = try await Self.send(
            method: "POST", path: "/v1/chat/completions", port: port,
            origin: Self.allowed, body: Self.completionBody)
        #expect(real.statusCode == 503)

        try await server.shutdown()
    }

    /// FLAG-6 with FLAG-5: preflight carries no `Authorization` — browsers do
    /// not put one on it — so the key check cannot stand in front of it.
    @Test func FLAG_6_preflight_is_answered_without_a_key() async throws {
        let (server, port) = try await Self.started(
            cors: .origins([Self.allowed]), keys: ["sk-test-key"])

        let preflight = try await Self.send(
            method: "OPTIONS", path: "/v1/chat/completions", port: port,
            origin: Self.allowed)
        #expect(preflight.statusCode == 200)
        #expect(Self.allowOrigin(preflight) == Self.allowed)

        // The real request behind it still needs the key.
        let real = try await Self.send(
            method: "POST", path: "/v1/chat/completions", port: port,
            origin: Self.allowed, body: Self.completionBody)
        #expect(real.statusCode == 401)

        try await server.shutdown()
    }

    /// FLAG-6 with EP-7: the same preflight answer for a path that exists, a
    /// path this server does not adopt, and a path it has never heard of — so
    /// which routes exist is not readable from preflight.
    @Test func FLAG_6_preflight_answers_identically_for_every_path() async throws {
        let (server, port) = try await Self.started(cors: .origins([Self.allowed]))

        var seen: Set<String> = []
        for path in ["/v1/chat/completions", "/v1/embeddings", "/nope"] {
            let response = try await Self.send(
                method: "OPTIONS", path: path, port: port, origin: Self.allowed)
            #expect(response.statusCode == 200, "\(path)")
            seen.insert([
                "\(response.statusCode)",
                Self.allowOrigin(response) ?? "-",
                response.value(forHTTPHeaderField: "Access-Control-Allow-Methods") ?? "-",
                response.value(forHTTPHeaderField: "Access-Control-Allow-Headers") ?? "-",
            ].joined(separator: "|"))
        }
        #expect(seen.count == 1)

        try await server.shutdown()
    }

    /// FLAG-6: never, under any policy, on any response. This server's
    /// authentication is a header the client sets on purpose, not a cookie the
    /// browser attaches on its own, so there is no reason to let a credentialed
    /// cross-origin request through.
    @Test func FLAG_6_no_response_ever_allows_credentials() async throws {
        for policy in [ServerCORSPolicy.disabled, .any, .origins([Self.allowed])] {
            let (server, port) = try await Self.started(cors: policy)
            for (method, path) in [("GET", "/v1/models"),
                                   ("OPTIONS", "/v1/chat/completions")] {
                let response = try await Self.send(
                    method: method, path: path, port: port, origin: Self.allowed)
                #expect(response.value(
                    forHTTPHeaderField: "Access-Control-Allow-Credentials") == nil,
                        "\(policy) \(method) \(path)")
            }
            try await server.shutdown()
        }
    }

    /// FLAG-6: a streamed answer writes its own response head, so it is its own
    /// path to get this wrong.
    @Test func FLAG_6_a_streamed_answer_carries_the_header_too() async throws {
        let (server, port) = try await Self.started(cors: .origins([Self.allowed]))

        let response = try await Self.send(
            method: "POST", path: "/v1/chat/completions", port: port,
            origin: Self.allowed,
            body: #"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#)
        #expect(response.statusCode == 200)
        #expect(response.value(forHTTPHeaderField: "content-type") == "text/event-stream")
        #expect(Self.allowOrigin(response) == Self.allowed)

        try await server.shutdown()
    }

    /// FLAG-6: `--cors-origins` takes a comma-separated list or `*`, and its
    /// absence is not the reference's `*` (DEV-20).
    @Test func FLAG_6_the_flag_takes_a_list_or_a_star() throws {
        #expect(try ServerArguments.parse(["--model", "m.gturbo"]).corsPolicy
                == .disabled)
        #expect(try ServerArguments.parse(["--model", "m.gturbo",
                                           "--cors-origins", "*"]).corsPolicy == .any)
        #expect(try ServerArguments.parse(
            ["--model", "m.gturbo",
             "--cors-origins", "http://a.example, https://b.example"]).corsPolicy
                == .origins(["http://a.example", "https://b.example"]))

        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--model", "m.gturbo", "--cors-origins", " , "])
        }
    }
}
