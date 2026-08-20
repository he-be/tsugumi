import Foundation
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// C1 (CONFORMANCE §1): SPEC §7, the reuse rule stated as token arithmetic.
///
/// Every claim here is "these two token sequences share N, so N is what the
/// next request resumes from". Nothing is said about roles, tools, thinking,
/// or which client shape it is — CACHE-1 exists precisely to stop the cache
/// from consulting any of that, and the reference implementation spends one
/// line on the whole question (`get_common_prefix`).
@Suite("C1 prompt cache — LCP")
struct PromptCacheLCPTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 16_384,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    // MARK: - plumbing
    //
    // The two helpers are the only part of this file that knows the cache's
    // current signatures. The tests below are the SPEC lines and do not move
    // when those signatures do.

    /// A cache whose entry holds exactly `kv` in the KV.
    private func cache(holding kv: [Int32],
                       tools: [GFTokenizer.FunctionDefinition] = []) -> ServerPromptCache {
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: request(tools: tools),
            content: "answer",
            calls: [],
            result: RawDecodeResult(
                prefillTokens: kv.count,
                cachedPromptTokens: 0,
                computedPrefillTokens: kv.count,
                prefillSeconds: 0,
                newTokens: 1,
                decodeSeconds: 0,
                timeToFirstTokenSeconds: 0,
                reason: .endOfTurn,
                kvPosition: kv.count,
                kvBackedTokenIDs: kv,
                uncommittedBoundaryTokenIDs: [99]))
        return cache
    }

    /// How many tokens a request resumes from, or nil when it starts over.
    private func reuse(_ cache: ServerPromptCache,
                       prompt: [Int32],
                       tools: [GFTokenizer.FunctionDefinition] = []) async throws -> Int? {
        let tokenizer = try await GFTokenizer.load()
        switch cache.match(domain: domain,
                           request: request(tools: tools),
                           renderedPromptIDs: prompt,
                           tokenizer: tokenizer) {
        case .hit(_, let cached, _): return cached
        default: return nil
        }
    }

    private func request(
        tools: [GFTokenizer.FunctionDefinition] = []
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: [GFTokenizer.Message(role: .user, content: "hi")],
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
    }

    // MARK: - SPEC §7

    /// The ordinary turn: the client sends back everything the KV holds and
    /// adds to it, so everything the KV holds is reused.
    @Test("CACHE-1: a prompt that extends the KV resumes at the end of it")
    func CACHE_1_extension_resumes_at_the_end() async throws {
        let cache = cache(holding: [1, 2, 3, 4])
        #expect(try await reuse(cache, prompt: [1, 2, 3, 4, 5, 6]) == 4)
    }

    /// CACHE-1's real content: the answer is a property of the two token
    /// arrays and of nothing else. This request declares a tool the cached one
    /// did not, which the old design refused outright; the tokens still agree,
    /// so the prefix is still valid.
    @Test("CACHE-1: the shape of the conversation is not consulted")
    func CACHE_1_shape_is_not_consulted() async throws {
        let tool = GFTokenizer.FunctionDefinition(
            name: "lookup", description: "look up", parameters: .object([:]))
        let cache = cache(holding: [1, 2, 3, 4])
        #expect(try await reuse(cache, prompt: [1, 2, 3, 4, 5], tools: [tool]) == 4)
    }

    /// Nothing in common: nothing to resume from.
    @Test("CACHE-1: a prompt that shares no prefix starts over")
    func CACHE_1_no_shared_prefix_starts_over() async throws {
        let cache = cache(holding: [1, 2, 3, 4])
        #expect(try await reuse(cache, prompt: [7, 8, 9]) == nil)
    }

    /// CACHE-2: the client edited its history — compacted it, dropped a turn,
    /// rewrote a tool result. The prefix in front of the edit is still valid
    /// and is served; only the tail is prefilled again. Refusing the whole
    /// entry here is what makes a compaction cost a full prompt.
    @Test("CACHE-2: a prompt that diverges inside the KV resumes at the divergence")
    func CACHE_2_partial_prefix_is_served() async throws {
        let cache = cache(holding: [1, 2, 3, 4, 5])
        #expect(try await reuse(cache, prompt: [1, 2, 3, 9]) == 3)
    }

    /// CACHE-3: the KV already holds every token of this prompt, so there is
    /// nothing left to draw logits from. The reference backs the cursor up by
    /// one and re-decodes that token (`n_past--`).
    @Test("CACHE-3: a prompt the KV already holds re-decodes its last token")
    func CACHE_3_full_match_re_decodes_one_token() async throws {
        let cache = cache(holding: [1, 2, 3, 4])
        #expect(try await reuse(cache, prompt: [1, 2, 3, 4]) == 3)
    }

    /// The KV holds more than this prompt does, and the shared part is one
    /// token: still a resume, of one token.
    @Test("CACHE-2: a one-token prefix is still a prefix")
    func CACHE_2_single_token_prefix() async throws {
        let cache = cache(holding: [1, 2, 3, 4, 5])
        #expect(try await reuse(cache, prompt: [1, 9, 9]) == 1)
    }
}
