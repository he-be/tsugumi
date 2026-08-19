import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// Emits a thought channel and an answer, in the order a reasoning generation
/// produces them.
private actor ReasoningBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.reasoning("weighing it"))
        onEvent(.content("hello"))
        return ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 5, totalTokens: 8),
            reasoningContent: "weighing it")
    }
}

/// Answers without reasoning, so the response must not carry the field at all.
private actor PlainBackend: ServerInferenceBackend {
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

/// Records what the request asked the template to do.
private actor ThinkingRecordingBackend: ServerInferenceBackend {
    private(set) var sawThinking: Bool?

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        sawThinking = request.enableThinking
        onEvent(.content("ok"))
        return ServerCompletion(
            content: "ok",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

@Suite("Server reasoning requests")
struct ServerReasoningRequestTests {
    private func request(_ body: String) -> Data { Data(body.utf8) }

    /// pi's `qwen-chat-template` thinking format sends exactly this, including
    /// the `preserve_thinking` key this template has no use for.
    @Test func chatTemplateKwargsTurnTheThoughtChannelOn() throws {
        let validated = try ChatRequestParser.parse(
            request(#"""
            {"model":"m","messages":[{"role":"user","content":"hi"}],
             "chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}
            """#))
        #expect(validated.enableThinking)
    }

    /// REQ-reasoning-effort: this template has one thought channel and no
    /// budget, so a level is read for its on/off sense — and the set of levels
    /// is not enumerated. "none" is the one word with a fixed meaning.
    @Test func REQ_reasoning_effort_any_level_asks_to_reason() throws {
        for effort in ["minimal", "low", "medium", "high", "max", "ultra", "off"] {
            let validated = try ChatRequestParser.parse(
                request("""
                {"model":"m","messages":[{"role":"user","content":"hi"}],
                 "reasoning_effort":"\(effort)"}
                """))
            #expect(validated.enableThinking, "\(effort) should reason")
            #expect(validated.reasoningEffort == effort)
        }
    }

    @Test func REQ_reasoning_effort_none_turns_the_channel_off() throws {
        let validated = try ChatRequestParser.parse(
            request(#"""
            {"model":"m","messages":[{"role":"user","content":"hi"}],
             "reasoning_effort":"none"}
            """#),
            defaults: ChatRequestDefaults(thinking: .on))
        #expect(!validated.enableThinking)
    }

    /// RSN-2. Two spellings reach this server and neither is refused for
    /// disagreeing with the other: `enable_thinking` is the template's own
    /// switch and wins, and `reasoning_effort: "none"` is applied after it,
    /// which is the order the reference implementation resolves them in
    /// (`server-common.cpp:1278-1304`).
    @Test func RSN_2_both_spellings_resolve_without_a_refusal() throws {
        let kwargsWin = try ChatRequestParser.parse(
            request(#"""
            {"model":"m","messages":[{"role":"user","content":"hi"}],
             "reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":false}}
            """#))
        #expect(!kwargsWin.enableThinking)

        let noneWins = try ChatRequestParser.parse(
            request(#"""
            {"model":"m","messages":[{"role":"user","content":"hi"}],
             "reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":true}}
            """#))
        #expect(!noneWins.enableThinking)

        let agreeing = try ChatRequestParser.parse(
            request(#"""
            {"model":"m","messages":[{"role":"user","content":"hi"}],
             "reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":true}}
            """#))
        #expect(agreeing.enableThinking)
    }

    /// RSN-1: a budget of zero says "do not reason" the third way.
    @Test func REQ_reasoning_budget_zero_closes_the_channel() throws {
        let validated = try ChatRequestParser.parse(
            request(#"""
            {"model":"m","messages":[{"role":"user","content":"hi"}],
             "chat_template_kwargs":{"enable_thinking":true},"reasoning_budget_tokens":0}
            """#))
        #expect(!validated.enableThinking)
        #expect(validated.reasoningBudgetTokens == 0)
    }

    @Test func malformedChatTemplateKwargsAreRefused() throws {
        for body in [
            #"{"model":"m","messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":"yes"}}"#,
            #"{"model":"m","messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":[1]}"#,
        ] {
            let decoded = request(body)
            let error = #expect(throws: ServerRequestError.self) {
                try ChatRequestParser.parse(decoded)
            }
            #expect(error?.param == "chat_template_kwargs")
        }
    }

    @Test func theProcessDefaultAppliesOnlyWhenTheRequestIsSilent() throws {
        let silent = request(#"""
        {"model":"m","messages":[{"role":"user","content":"hi"}]}
        """#)
        #expect(try ChatRequestParser.parse(
            silent, defaults: ChatRequestDefaults(thinking: .on)).enableThinking)
        #expect(!(try ChatRequestParser.parse(silent).enableThinking))

        let opinionated = request(#"""
        {"model":"m","messages":[{"role":"user","content":"hi"}],
         "chat_template_kwargs":{"enable_thinking":false}}
        """#)
        #expect(!(try ChatRequestParser.parse(
            opinionated, defaults: ChatRequestDefaults(thinking: .on)).enableThinking))
    }

    @Test func serverArgumentsCarryTheThinkingPolicy() throws {
        #expect(try ServerArguments.parse(["--model", "m.gturbo"]).thinkingPolicy == .off)
        #expect(try ServerArguments.parse(
            ["--model", "m.gturbo", "--thinking", "on"]).thinkingPolicy == .on)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--model", "m.gturbo", "--thinking", "yes"])
        }
    }

    /// Since S3 the tool-calling template reasons too: `enableThinking` is
    /// passed through instead of being pinned to false, and the marker lands in
    /// the same system turn that carries the tool declarations.
    @Test func toolPromptsCarryTheThoughtMarkerWhenThinkingIsOn() async throws {
        let tokenizer = try await GFTokenizer.load()
        let tools = [GFTokenizer.FunctionDefinition(
            name: "read",
            description: "Read a file",
            parameters: .object(["type": .string("object")]))]
        let messages = [GFTokenizer.Message(role: .user, content: "read /tmp/a")]

        let reasoning = try tokenizer.encodeToolChat(
            messages: messages, tools: tools, enableThinking: true)
        #expect(tokenizer.decode(reasoning, skipSpecialTokens: false).contains("<|think|>"))

        let plain = try tokenizer.encodeToolChat(messages: messages, tools: tools)
        #expect(!tokenizer.decode(plain, skipSpecialTokens: false).contains("<|think|>"))
    }

    /// A tool request with no image renders byte for byte as it did before the
    /// parts path existed — the body is still sent as a string.
    @Test func textOnlyToolPromptsAreUnchangedByThePartsPath() async throws {
        let tokenizer = try await GFTokenizer.load()
        let tools = [GFTokenizer.FunctionDefinition(
            name: "read",
            description: "Read a file",
            parameters: .object(["type": .string("object")]))]
        let asMessages = try tokenizer.encodeToolChat(
            messages: [GFTokenizer.Message(role: .user, content: "read /tmp/a")],
            tools: tools)
        let asParts = try tokenizer.encodeToolChat(
            messages: [GFTokenizer.ToolChatMessage(role: .user, parts: [.text("read /tmp/a")])],
            tools: tools)
        #expect(asMessages == asParts)
    }

    /// Reasoning requests reuse their prefix like any other request; only the
    /// image exclusion and the mode remain (S3.5). Whether two *different*
    /// modes may share a prefix is the cache entry's question, not this one
    /// (`ServerPromptCacheTests`).
    @Test func promptCacheParticipationDoesNotDependOnReasoning() throws {
        #expect(ServerModelSession.promptCacheParticipates(mode: .singlePrefix))
        #expect(!ServerModelSession.promptCacheParticipates(mode: .off))
    }

    @Test func decoderSurfacesTheThoughtChannelAsReasoning() async throws {
        let tokenizer = try await GFTokenizer.load()
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer,
                                                 allowedTools: [],
                                                 emitsReasoning: true)
        #expect(try decoder.consume(tokenID: tokenizer.channelStartID, delta: "").isEmpty)
        // The label line names the channel and carries no text of its own.
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "thought\n").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "weighing") == [
            .reasoning("weighing"),
        ])
        #expect(try decoder.consumeTail(" it") == [.reasoning(" it")])
        #expect(try decoder.consume(tokenID: tokenizer.channelEndID, delta: "") == [])
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "answer") == [
            .content("answer"),
        ])
    }

    /// The label line names the channel; text that arrives in the same delta
    /// after the newline belongs to the channel it just named.
    @Test func reasoningInTheLabelDeltaIsNotLost() async throws {
        let tokenizer = try await GFTokenizer.load()
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer,
                                                 allowedTools: [],
                                                 emitsReasoning: true)
        #expect(try decoder.consume(tokenID: tokenizer.channelStartID, delta: "").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "thought\nfirst") == [
            .reasoning("first"),
        ])
    }

    /// Without the flag the decoder behaves exactly as it did before reasoning
    /// existed: the thought channel is dropped, not relabeled.
    @Test func decoderStillHidesThoughtWhenReasoningIsNotRequested() async throws {
        let tokenizer = try await GFTokenizer.load()
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer, allowedTools: [])
        #expect(try decoder.consume(tokenID: tokenizer.channelStartID, delta: "").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "thought\nhidden").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "more").isEmpty)
    }
}

@Suite("Server reasoning responses")
struct ServerReasoningResponseTests {
    @Test func nonStreamingCarriesReasoningContentBesideTheAnswer() async throws {
        let server = TurboFieldfareHTTPServer(modelID: "test-model",
                                              queueLimit: 1,
                                              backend: ReasoningBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        defer { Task { try? await server.shutdown() } }

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],
         "chat_template_kwargs":{"enable_thinking":true}}
        """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(object["choices"] as? [[String: Any]])
        let message = try #require(choices[0]["message"] as? [String: Any])
        #expect(message["content"] as? String == "hello")
        #expect(message["reasoning_content"] as? String == "weighing it")
    }

    @Test func aRequestThatDidNotReasonHasNoReasoningField() async throws {
        let server = TurboFieldfareHTTPServer(modelID: "test-model",
                                              queueLimit: 1,
                                              backend: PlainBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        defer { Task { try? await server.shutdown() } }

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """#.utf8)
        let (data, _) = try await URLSession.shared.data(for: request)
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(object["choices"] as? [[String: Any]])
        let message = try #require(choices[0]["message"] as? [String: Any])
        #expect(message["reasoning_content"] == nil)
    }

    @Test func streamingSendsReasoningInItsOwnDeltaField() async throws {
        let server = TurboFieldfareHTTPServer(modelID: "test-model",
                                              queueLimit: 1,
                                              backend: ReasoningBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        defer { Task { try? await server.shutdown() } }

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],
         "stream":true,"reasoning_effort":"medium"}
        """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""reasoning_content":"weighing it""#))
        #expect(text.contains(#""content":"hello""#))
        // The reasoning arrives before the answer and never inside it.
        let reasoningIndex = try #require(text.range(of: #""reasoning_content""#))
        let contentIndex = try #require(text.range(of: #""content":"hello""#))
        #expect(reasoningIndex.lowerBound < contentIndex.lowerBound)
        #expect(text.hasSuffix("data: [DONE]\n\n"))
    }

    @Test func theRequestsThinkingChoiceReachesTheBackend() async throws {
        let backend = ThinkingRecordingBackend()
        let server = TurboFieldfareHTTPServer(modelID: "test-model",
                                              queueLimit: 1,
                                              backend: backend,
                                              defaults: ChatRequestDefaults(thinking: .on))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        defer { Task { try? await server.shutdown() } }

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """#.utf8)
        _ = try await URLSession.shared.data(for: request)
        #expect(await backend.sawThinking == true)
    }
}
