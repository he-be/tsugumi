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
        switch cache.match(domain: domain,
                           request: request(tools: tools),
                           renderedPromptIDs: prompt,
                           maximumRewind: 2048) {
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

    // MARK: - SPEC CACHE-4
    //
    // The pictures are stated the same way: token arithmetic, with `9` standing
    // for the soft token every photograph widens into. That id is the same id
    // whatever the camera saw, which is the entire reason the walk has to be
    // told about chunks — the reference compares each chunk's id and token
    // count in the middle of the same loop (`server-common.cpp:678`).

    private func chunk(_ offset: Int, _ count: Int, _ digest: String)
        -> ServerPromptMediaChunk {
        ServerPromptMediaChunk(tokenOffset: offset, tokenCount: count, digest: digest)
    }

    /// Two photographs, the same words, the same ids — and the walk must still
    /// stop where the pictures part.
    @Test("CACHE-4: a different picture ends the walk where its chunk starts")
    func CACHE_4_a_different_picture_ends_the_walk() {
        let lhs: [Int32] = [1, 2, 9, 9, 9, 3, 4]
        let rhs: [Int32] = [1, 2, 9, 9, 9, 3, 4]
        #expect(commonPrefixLength(lhs, rhs) == 7,
                "the ids agree all the way — that is what makes CACHE-4 necessary")
        #expect(commonPrefixLength(lhs, rhs,
                                   lhsMedia: [chunk(2, 3, "photo-a")],
                                   rhsMedia: [chunk(2, 3, "photo-b")]) == 2)
    }

    /// The same photograph is crossed in one step, and the text behind it is
    /// compared as text — the walk does not end at a picture, it steps over it.
    @Test("CACHE-4: the same picture is skipped whole and the walk goes on")
    func CACHE_4_the_same_picture_is_skipped_whole() {
        let lhs: [Int32] = [1, 2, 9, 9, 9, 3, 4]
        let rhs: [Int32] = [1, 2, 9, 9, 9, 3, 8]
        #expect(commonPrefixLength(lhs, rhs,
                                   lhsMedia: [chunk(2, 3, "photo-a")],
                                   rhsMedia: [chunk(2, 3, "photo-a")]) == 6)
    }

    /// Same photograph, different size: its rows are a different length, so the
    /// prefix ends at the chunk rather than part way into it.
    @Test("CACHE-4: the same picture at a different token count ends the walk")
    func CACHE_4_a_different_token_count_ends_the_walk() {
        let lhs: [Int32] = [1, 2, 9, 9, 9, 3]
        let rhs: [Int32] = [1, 2, 9, 9, 3, 5]
        #expect(commonPrefixLength(lhs, rhs) == 4,
                "the ids agree until the shorter picture runs out")
        #expect(commonPrefixLength(lhs, rhs,
                                   lhsMedia: [chunk(2, 3, "photo-a")],
                                   rhsMedia: [chunk(2, 2, "photo-a")]) == 2)
    }

    /// A picture on one side and ordinary text on the other. The reference sees
    /// `LLAMA_TOKEN_NULL` against a real token and returns that index; here the
    /// ids can even agree, and the answer is the same.
    @Test("CACHE-4: a picture against text ends the walk where the picture starts")
    func CACHE_4_a_picture_against_text_ends_the_walk() {
        let lhs: [Int32] = [1, 2, 9, 9, 9, 3]
        let rhs: [Int32] = [1, 2, 9, 9, 9, 3]
        #expect(commonPrefixLength(lhs, rhs,
                                   lhsMedia: [chunk(2, 3, "photo-a")],
                                   rhsMedia: []) == 2)
    }

    /// The walk keeps comparing after a picture matches: a conversation whose
    /// second photograph differs still keeps everything in front of it.
    @Test("CACHE-4: a later picture ends the walk at its own chunk")
    func CACHE_4_a_later_picture_ends_the_walk() {
        let lhs: [Int32] = [1, 9, 9, 2, 9, 9, 3]
        let rhs: [Int32] = [1, 9, 9, 2, 9, 9, 3]
        #expect(commonPrefixLength(lhs, rhs,
                                   lhsMedia: [chunk(1, 2, "photo-a"), chunk(4, 2, "photo-b")],
                                   rhsMedia: [chunk(1, 2, "photo-a"), chunk(4, 2, "photo-c")]) == 4)
    }
}
