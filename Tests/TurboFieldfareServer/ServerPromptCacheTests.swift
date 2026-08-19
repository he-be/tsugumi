import Foundation
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("Server prompt cache")
struct ServerPromptCacheTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 16_384,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    @Test func textContinuationUsesActualGeneratedHistoryAndOnlyPrefillsSuffix() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let generated = tokenizer.encode("answer", addBOS: false)
        let kvBacked = initialPrompt + generated
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let continuation = request(messages: initial.messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(continuation.messages),
            addBOS: false)
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached, _) = match else {
            Issue.record("expected text continuation hit")
            return
        }
        let bridge = try tokenizer.encodeTextContinuation(userContent: "second")
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
        #expect(effective[cached] == tokenizer.endOfTurnID)
    }

    /// A reasoning session reuses its prefix like any other. Without this the
    /// second turn of a pi session re-prefills the whole conversation, which is
    /// what reasoning cost when it shipped (S3.5).
    @Test func reasoningContinuationHitsTheCache() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(
            messages: [GFTokenizer.Message(role: .user, content: "first")],
            enableThinking: true)
        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages, enableThinking: true),
            addBOS: false)
        let generated = tokenizer.encode("answer", addBOS: false)
        let kvBacked = initialPrompt + generated
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let messages = initial.messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ]
        let continuation = request(messages: messages, enableThinking: true)
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages, enableThinking: true),
            addBOS: false)
        guard case .hit(let effective, let cached, _) = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer) else {
            Issue.record("expected a reasoning continuation hit")
            return
        }
        #expect(cached == kvBacked.count)
        let bridge = try tokenizer.encodeTextContinuation(
            userContent: "second", enableThinking: true)
        #expect(effective == kvBacked + bridge)

        // Switching the mode mid-session cannot ride that prefix: the cached
        // system turn carries the marker this request would not render.
        let plain = request(messages: messages, enableThinking: false)
        #expect(cache.match(domain: domain,
                            request: plain,
                            renderedPromptIDs: rendered,
                            tokenizer: tokenizer) == .miss(.thinking))
    }

    /// The bridge has to end the way the generation prompt of its own mode
    /// does: reasoning leaves the thought channel open, plain closes it.
    @Test func continuationBridgeMatchesTheModesGenerationPrompt() async throws {
        let tokenizer = try await GFTokenizer.load()
        let reasoning = try tokenizer.encodeTextContinuation(
            userContent: "second", enableThinking: true)
        let plain = try tokenizer.encodeTextContinuation(userContent: "second")
        #expect(reasoning != plain)
        #expect(plain.count > reasoning.count)
        #expect(plain.starts(with: reasoning))
        let plainTail = tokenizer.decode(Array(plain.dropFirst(reasoning.count)),
                                         skipSpecialTokens: false)
        #expect(plainTail == "<|channel>thought\n<channel|>")
    }

    @Test func capturedOpenCodeToolResultUsesFrozenToolBoundary() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = try validatedFixture("opencode-1.15.11-initial.json")
        let continuation = try validatedFixture("opencode-1.15.11-tool-result.json")
        let initialPrompt = try tokenizer.encodeToolChat(
            messages: initial.messages,
            tools: initial.tools)
        let assistant = continuation.messages[initial.messages.count]
        let prefix = try tokenizer.encodeToolChat(
            messages: initial.messages + [assistant],
            tools: initial.tools)
        let callStart = try #require(prefix.lastIndex(of: tokenizer.toolCallStartID))
        let callEnd = try #require(prefix.lastIndex(of: tokenizer.toolCallEndID))
        let generatedCall = Array(prefix[callStart...callEnd])
        let kvBacked = initialPrompt + generatedCall
        let historicalCall = try #require(assistant.toolCalls.first)
        let parsedCall = ParsedToolCall(
            id: historicalCall.id,
            name: historicalCall.name,
            arguments: historicalCall.arguments,
            argumentsJSON: try historicalCall.arguments.encoded())
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "",
            calls: [parsedCall],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.toolResponseID,
                reason: .toolCalls))
        let rendered = try tokenizer.encodeToolChat(
            messages: continuation.messages,
            tools: continuation.tools)

        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached, _) = match else {
            Issue.record("expected captured OpenCode tool-result hit")
            return
        }
        let bridge = try tokenizer.encodeToolResultContinuation(
            cachedMessages: initial.messages,
            assistant: assistant,
            incomingMessages: continuation.messages,
            tools: continuation.tools)
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        #expect(bridge.first == tokenizer.toolResponseID)
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
    }

    /// An agent commonly sends the tool result and the user's next message in
    /// one request. That shape used to miss, which cost a coding session a full
    /// prefill on exactly the turns where the user typed something.
    @Test func toolResultFollowedByANewUserTurnStillResumes() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = try validatedFixture("opencode-1.15.11-initial.json")
        let resultTurn = try validatedFixture("opencode-1.15.11-tool-result.json")
        let initialPrompt = try tokenizer.encodeToolChat(
            messages: initial.messages, tools: initial.tools)
        let assistant = resultTurn.messages[initial.messages.count]
        let prefix = try tokenizer.encodeToolChat(
            messages: initial.messages + [assistant], tools: initial.tools)
        let callStart = try #require(prefix.lastIndex(of: tokenizer.toolCallStartID))
        let callEnd = try #require(prefix.lastIndex(of: tokenizer.toolCallEndID))
        let kvBacked = initialPrompt + Array(prefix[callStart...callEnd])
        let historicalCall = try #require(assistant.toolCalls.first)
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "",
            calls: [ParsedToolCall(
                id: historicalCall.id,
                name: historicalCall.name,
                arguments: historicalCall.arguments,
                argumentsJSON: try historicalCall.arguments.encoded())],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.toolResponseID,
                reason: .toolCalls))

        // The tool result, and the user's next message behind it.
        let withUserTurn = ValidatedChatRequest(
            messages: resultTurn.messages + [
                GFTokenizer.Message(role: .user, content: "now caption this"),
            ],
            tools: resultTurn.tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
        let rendered = try tokenizer.encodeToolChat(
            messages: withUserTurn.messages, tools: withUserTurn.tools)
        guard case .hit(let effective, let cached, _) = cache.match(
            domain: domain,
            request: withUserTurn,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer) else {
            Issue.record("expected a hit for tool result + user turn")
            return
        }
        #expect(cached == kvBacked.count)
        let bridge = Array(effective.dropFirst(cached))
        #expect(bridge.first == tokenizer.toolResponseID)
        // The bridge carries both the tool response and the new user turn.
        let bridgeText = tokenizer.decode(bridge, skipSpecialTokens: false)
        #expect(bridgeText.contains("now caption this"))
        #expect(bridgeText.contains("<|tool_response>"))

        // A tool result for a call this KV never made is still a miss.
        let foreign = ValidatedChatRequest(
            messages: initial.messages + [
                assistant,
                GFTokenizer.Message(role: .tool,
                                    content: "x",
                                    toolCallID: "call_ffffffffffffffffffffffff"),
            ],
            tools: resultTurn.tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
        #expect(cache.match(domain: domain,
                            request: foreign,
                            renderedPromptIDs: rendered,
                            tokenizer: tokenizer) == .miss(.continuationShape("tool")))
    }

    @Test func mismatchedLineageDomainAndUnsafeStopsMiss() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var cache = ServerPromptCache()

        for reason in [StopReason.stopString, .eos] {
            cache.publish(
                domain: domain,
                request: initial,
                content: "answer",
                calls: [],
                result: rawResult(
                    prompt: prompt,
                    kvBacked: prompt,
                    boundary: tokenizer.eosID,
                    reason: reason))
            #expect(cache.entry == nil)
        }

        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt + tokenizer.encode("answer", addBOS: false),
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let changed = request(messages: [
            GFTokenizer.Message(role: .user, content: "changed"),
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(changed.messages),
            addBOS: false)
        #expect(cache.match(
            domain: domain,
            request: changed,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer) == .miss(.history))
    }

    @Test func tailCompletedStopStringDoesNotPublishPrefix() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var matcher = StreamingStopMatcher(stops: ["🌳stop"])
        #expect(matcher.push("answer 🌳") == "answer ")
        #expect(matcher.push("stop") == "")
        #expect(matcher.isStopped)

        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer ",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn),
            stopStringFiltered: matcher.isStopped)
        #expect(cache.entry == nil)
    }

    private func request(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition] = [],
        enableThinking: Bool = false
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages,
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16,
            enableThinking: enableThinking)
    }

    private func rawResult(
        prompt: [Int32],
        kvBacked: [Int32],
        boundary: Int32,
        reason: StopReason
    ) -> RawDecodeResult {
        RawDecodeResult(
            prefillTokens: prompt.count,
            cachedPromptTokens: 0,
            computedPrefillTokens: prompt.count,
            prefillSeconds: 0,
            newTokens: 1,
            decodeSeconds: 0,
            timeToFirstTokenSeconds: 0,
            reason: reason,
            kvPosition: kvBacked.count,
            kvBackedTokenIDs: kvBacked,
            uncommittedBoundaryTokenIDs: [boundary])
    }

    private func validatedFixture(_ name: String) throws -> ValidatedChatRequest {
        let url = try #require(Bundle.module.url(
            forResource: name,
            withExtension: nil,
            subdirectory: "Fixtures"))
        return try ChatRequestParser.parse(try Data(contentsOf: url))
    }
}
