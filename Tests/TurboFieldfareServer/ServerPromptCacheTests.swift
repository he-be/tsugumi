import Foundation
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// C2 (CONFORMANCE §1): the reuse rule against conversations a real client
/// actually sent.
///
/// `PromptCacheLCPTests` states SPEC §7 on invented token arrays; this file
/// puts the same claims behind the captured OpenCode session, so the numbers
/// come from a prompt with a system turn, tool declarations, and a tool call
/// in it. The claims are still only ever about how many tokens two sequences
/// share — never "this shape hits".
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

    /// The turn a coding agent spends most of its life in: the assistant asked
    /// for a tool, the client sends the result back, and everything in front of
    /// the result is already in the KV.
    @Test func CACHE_1_captured_tool_result_resumes_from_the_whole_kv() async throws {
        let tokenizer = try await GFTokenizer.load()
        let renderer = ServerPromptRenderer(tokenizer: tokenizer)
        let initial = try validatedFixture("opencode-1.15.11-initial.json")
        let continuation = try validatedFixture("opencode-1.15.11-tool-result.json")

        // What the KV held when the tool call finished: the prompt it was
        // generated from, plus the call itself.
        let initialPrompt = try renderer.promptIDs(initial)
        let assistant = continuation.messages[initial.messages.count]
        let withCall = try renderer.promptIDs(request(
            messages: initial.messages + [assistant], tools: initial.tools))
        let callStart = try #require(withCall.lastIndex(of: tokenizer.toolCallStartID))
        let callEnd = try #require(withCall.lastIndex(of: tokenizer.toolCallEndID))
        let kvBacked = initialPrompt + Array(withCall[callStart...callEnd])

        var cache = ServerPromptCache()
        cache.publish(domain: domain, request: initial, result: rawResult(kvBacked: kvBacked))

        let rendered = try renderer.promptIDs(continuation)
        // INV-1 stated on a real session: the redraw covers the whole KV.
        #expect(commonPrefixLength(kvBacked, rendered) == kvBacked.count)

        guard case .hit(let effective, let cached, _) = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            maximumRewind: 2048) else {
            Issue.record("expected the captured tool result to resume")
            return
        }
        #expect(cached == kvBacked.count)
        #expect(effective == rendered)
        // Only the tool response and the turn it opens are prefilled again.
        #expect(effective.count - cached == rendered.count - kvBacked.count)
    }

    /// The same session with the user's next message sent alongside the tool
    /// result — one request, two new turns. Nothing about the rule changes.
    @Test func CACHE_1_a_tool_result_plus_a_user_turn_resumes_the_same_way() async throws {
        let tokenizer = try await GFTokenizer.load()
        let renderer = ServerPromptRenderer(tokenizer: tokenizer)
        let initial = try validatedFixture("opencode-1.15.11-initial.json")
        let resultTurn = try validatedFixture("opencode-1.15.11-tool-result.json")

        let initialPrompt = try renderer.promptIDs(initial)
        let assistant = resultTurn.messages[initial.messages.count]
        let withCall = try renderer.promptIDs(request(
            messages: initial.messages + [assistant], tools: initial.tools))
        let callStart = try #require(withCall.lastIndex(of: tokenizer.toolCallStartID))
        let callEnd = try #require(withCall.lastIndex(of: tokenizer.toolCallEndID))
        let kvBacked = initialPrompt + Array(withCall[callStart...callEnd])

        var cache = ServerPromptCache()
        cache.publish(domain: domain, request: initial, result: rawResult(kvBacked: kvBacked))

        let withUserTurn = request(
            messages: resultTurn.messages
                + [GFTokenizer.Message(role: .user, content: "and now run the tests")],
            tools: resultTurn.tools)
        let rendered = try renderer.promptIDs(withUserTurn)

        guard case .hit(_, let cached, _) = cache.match(
            domain: domain,
            request: withUserTurn,
            renderedPromptIDs: rendered,
            maximumRewind: 2048) else {
            Issue.record("expected the tool result plus user turn to resume")
            return
        }
        #expect(cached == kvBacked.count)
    }

    /// The lineage guard, which is the one thing that is *not* a fact about the
    /// conversation: the same token ids mean something else under a different
    /// model, runtime, or template, and the rows behind them describe it.
    @Test func CACHE_1_a_different_lineage_is_not_the_same_prefix() async throws {
        let tokenizer = try await GFTokenizer.load()
        let renderer = ServerPromptRenderer(tokenizer: tokenizer)
        let initial = try validatedFixture("opencode-1.15.11-initial.json")
        let prompt = try renderer.promptIDs(initial)

        var cache = ServerPromptCache()
        cache.publish(domain: domain, request: initial,
                      result: rawResult(kvBacked: prompt))

        var otherTemplate = domain
        otherTemplate = ServerPromptCacheDomain(
            modelID: domain.modelID,
            sourceSnapshotHash: domain.sourceSnapshotHash,
            runtimeProfileHash: domain.runtimeProfileHash,
            maximumContext: domain.maximumContext,
            kvStorage: domain.kvStorage,
            fp16RingEnabled: domain.fp16RingEnabled,
            templateSHA256: "a different template")
        #expect(cache.match(domain: otherTemplate,
                            request: initial,
                            renderedPromptIDs: prompt + [7],
                            maximumRewind: 2048) == .miss)
    }

    /// An entry is only usable if the tokens it names are exactly the rows the
    /// KV holds. A result that cannot say that publishes nothing.
    @Test func an_entry_records_exactly_the_rows_the_kv_holds() async throws {
        let tokenizer = try await GFTokenizer.load()
        let renderer = ServerPromptRenderer(tokenizer: tokenizer)
        let initial = try validatedFixture("opencode-1.15.11-initial.json")
        let prompt = try renderer.promptIDs(initial)

        var cache = ServerPromptCache()
        cache.publish(domain: domain, request: initial, result: rawResult(kvBacked: prompt))
        #expect(cache.entry?.tokenIDs == prompt)
        #expect(cache.entry?.kvPosition == prompt.count)

        var inconsistent = rawResult(kvBacked: prompt)
        inconsistent = RawDecodeResult(
            prefillTokens: inconsistent.prefillTokens,
            cachedPromptTokens: 0,
            computedPrefillTokens: inconsistent.computedPrefillTokens,
            prefillSeconds: 0,
            newTokens: 1,
            decodeSeconds: 0,
            timeToFirstTokenSeconds: 0,
            reason: .endOfTurn,
            kvPosition: prompt.count - 3,
            kvBackedTokenIDs: prompt,
            uncommittedBoundaryTokenIDs: [])
        cache.publish(domain: domain, request: initial, result: inconsistent)
        #expect(cache.entry == nil)
    }

    // MARK: - plumbing

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

    private func rawResult(kvBacked: [Int32]) -> RawDecodeResult {
        RawDecodeResult(
            prefillTokens: kvBacked.count,
            cachedPromptTokens: 0,
            computedPrefillTokens: kvBacked.count,
            prefillSeconds: 0,
            newTokens: 1,
            decodeSeconds: 0,
            timeToFirstTokenSeconds: 0,
            reason: .endOfTurn,
            kvPosition: kvBacked.count,
            kvBackedTokenIDs: kvBacked,
            uncommittedBoundaryTokenIDs: [])
    }

    private func validatedFixture(_ name: String) throws -> ValidatedChatRequest {
        let url = try #require(Bundle.module.url(
            forResource: name,
            withExtension: nil,
            subdirectory: "Fixtures"))
        return try ChatRequestParser.parse(try Data(contentsOf: url))
    }
}
