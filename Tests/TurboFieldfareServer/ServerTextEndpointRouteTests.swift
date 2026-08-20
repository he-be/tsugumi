import Foundation
import NIOCore
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// A tokenizer that is arithmetic rather than a model: one id per Unicode
/// scalar, with a fixed id standing in for `<bos>`. That makes every expected
/// value in this suite writable by hand, which is what keeps these tests about
/// the HTTP contract (C1) and not about the tokenizer (C2, and already covered
/// by `ServerEndpointBackendTests`).
private actor TextRouteStubBackend: ServerInferenceBackend {
    static let bos: Int32 = 2

    static func ids(_ text: String) -> [Int32] {
        text.unicodeScalars.map { Int32($0.value) }
    }

    static func text(_ ids: [Int32]) -> String {
        String(String.UnicodeScalarView(ids.compactMap { Unicode.Scalar(UInt32($0)) }))
    }

    static func rendered(_ request: ValidatedChatRequest) -> String {
        "<start thinking=\(request.enableThinking)>"
            + request.messages
                .map { "\($0.role.rawValue):\($0.content ?? "")" }
                .joined(separator: "|")
    }

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

    func tokenize(_ text: String, addSpecial: Bool) async throws -> [Int32] {
        (addSpecial ? [Self.bos] : []) + Self.ids(text)
    }

    func detokenize(_ tokens: [Int32]) async throws -> String {
        Self.text(tokens)
    }

    func applyChatTemplate(_ request: ValidatedChatRequest) async throws -> String {
        Self.rendered(request)
    }
}

/// SPEC §3 **EP-5**: `POST /tokenize`, `/detokenize`, `/apply-template`.
///
/// The shapes are the reference implementation's at the pin `34af94cd9`
/// (`tools/server/README.md` and `server-context.cpp`): `{"tokens": [...]}`,
/// `{"content": "..."}` and `{"prompt": "..."}`.
@Suite("EP-5 tokenizer endpoints", .serialized)
struct ServerTextEndpointRouteTests {
    private static func started() async throws -> (TurboFieldfareHTTPServer, Int) {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: TextRouteStubBackend())
        let channel = try await server.start(port: 0)
        guard let port = channel.localAddress?.port else {
            throw ServerRequestError.invalid(message: "no port")
        }
        return (server, port)
    }

    static func post(_ path: String,
                     body: String,
                     port: Int,
                     method: String = "POST") async throws -> (Int, [String: Any]) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        if method != "GET" {
            request.httpBody = Data(body.utf8)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        return (status, object ?? [:])
    }

    /// EP-5: `content` in, `tokens` out. This is the token count a client
    /// pre-computes a request against.
    @Test func EP_5_tokenize_answers_with_the_token_ids() async throws {
        let (server, port) = try await Self.started()
        let (status, body) = try await Self.post(
            "/tokenize", body: #"{"content":"hi"}"#, port: port)

        #expect(status == 200)
        #expect(body["tokens"] as? [Int] == TextRouteStubBackend.ids("hi").map(Int.init))

        try await server.shutdown()
    }

    /// EP-5: `add_special` is the reference's flag for the leading `<bos>`, and
    /// its default there is `false` — a client asking "how long is this text"
    /// must not be told a length that includes a token it never sent.
    @Test func EP_5_tokenize_add_special_defaults_to_false() async throws {
        let (server, port) = try await Self.started()

        let (_, plain) = try await Self.post(
            "/tokenize", body: #"{"content":"hi"}"#, port: port)
        let (_, special) = try await Self.post(
            "/tokenize", body: #"{"content":"hi","add_special":true}"#, port: port)

        #expect(plain["tokens"] as? [Int] == TextRouteStubBackend.ids("hi").map(Int.init))
        #expect(special["tokens"] as? [Int]
            == [Int(TextRouteStubBackend.bos)]
                + TextRouteStubBackend.ids("hi").map(Int.init))

        try await server.shutdown()
    }

    /// EP-5: a body with no `content` is not an error in the reference — it is
    /// an empty tokenization (`server-context.cpp` at the pin).
    @Test func EP_5_tokenize_without_content_answers_with_no_tokens() async throws {
        let (server, port) = try await Self.started()
        let (status, body) = try await Self.post("/tokenize", body: "{}", port: port)

        #expect(status == 200)
        #expect(body["tokens"] as? [Int] == [])

        try await server.shutdown()
    }

    /// EP-5: `tokens` in, `content` out — the inverse direction, and the field
    /// name the reference answers with.
    @Test func EP_5_detokenize_answers_with_the_content() async throws {
        let (server, port) = try await Self.started()
        let ids = TextRouteStubBackend.ids("hi").map(String.init).joined(separator: ",")
        let (status, body) = try await Self.post(
            "/detokenize", body: #"{"tokens":[\#(ids)]}"#, port: port)

        #expect(status == 200)
        #expect(body["content"] as? String == "hi")

        try await server.shutdown()
    }

    /// EP-5: as with `/tokenize`, a missing field is the empty answer.
    @Test func EP_5_detokenize_without_tokens_answers_with_empty_content() async throws {
        let (server, port) = try await Self.started()
        let (status, body) = try await Self.post("/detokenize", body: "{}", port: port)

        #expect(status == 200)
        #expect(body["content"] as? String == "")

        try await server.shutdown()
    }

    /// EP-5 with §12 **DEV-12**: `/apply-template` answers with `prompt`, and it
    /// gets there through the same `ChatRequestParser` `/chat/completions` uses
    /// — so the prompt it describes is the one a completion of that very body
    /// would prefill, thought channel and all.
    @Test func EP_5_apply_template_answers_with_the_rendered_prompt() async throws {
        let (server, port) = try await Self.started()
        let (status, body) = try await Self.post(
            "/apply-template",
            body: #"""
            {"model":"test-model","messages":[{"role":"user","content":"hi"}],
             "chat_template_kwargs":{"enable_thinking":true}}
            """#,
            port: port)

        #expect(status == 200)
        #expect(body["prompt"] as? String == "<start thinking=true>user:hi")

        try await server.shutdown()
    }

    /// EP-5 / ERR-3: the body goes through the request parser, so a body the
    /// parser refuses is refused here in the same words — with the field named
    /// in `param`, not a blanket "malformed JSON".
    @Test func EP_5_apply_template_refuses_a_body_the_parser_refuses() async throws {
        let (server, port) = try await Self.started()
        let (status, body) = try await Self.post(
            "/apply-template", body: "{}", port: port)

        #expect(status == 400)
        let error = try #require(body["error"] as? [String: Any])
        #expect(error["type"] as? String == "invalid_request_error")
        #expect(error["param"] as? String == "messages")

        try await server.shutdown()
    }

    /// ERR-2: these are POST endpoints, so another verb on them is the wrong
    /// method on a route that exists — not a 404 and not EP-7's 501.
    @Test func EP_5_endpoints_answer_405_for_the_wrong_method() async throws {
        let (server, port) = try await Self.started()

        for path in ["/tokenize", "/detokenize", "/apply-template"] {
            let (status, body) = try await Self.post(
                path, body: "", port: port, method: "GET")
            #expect(status == 405, "GET \(path)")
            let error = body["error"] as? [String: Any]
            #expect(error?["code"] as? String == "method_not_allowed", "GET \(path)")
        }

        try await server.shutdown()
    }
}
