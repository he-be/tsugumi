import Foundation
import Testing
@testable import TurboFieldfareServerCore

/// C0 (CONFORMANCE §1) for SPEC §7 on the Ornith (Qwen 3.5-MoE) family: what
/// `QwenServerSession` decides *before* it touches the runner
/// (`docs/qwen35moe/41-PROMPT-CACHE.md`).
///
/// `ServerPromptCacheTests`' sibling. The rule under test is much smaller than
/// Gemma's — thirty of the forty layers are recurrent, so the reusable prefix
/// is all of the state or none of it — and that is exactly why it is worth
/// pinning: every case here is a way for a continuation to be silently wrong
/// rather than slow.
@Suite("C0 Qwen prompt cache")
struct QwenPromptCacheTests {
    private static func published(prompt: [Int32],
                                  generated: [Int32],
                                  kvPosition: Int? = nil) -> QwenPromptCache {
        var cache = QwenPromptCache()
        cache.publish(promptIDs: prompt,
                      generated: generated,
                      kvPosition: kvPosition ?? (prompt.count + generated.count - 1))
        return cache
    }

    @Test("an empty cache never hits")
    func emptyMisses() {
        #expect(QwenPromptCache().match([1, 2, 3]) == .miss(divergedAt: nil))
    }

    @Test("a prompt that extends the state hits with the whole state")
    func extensionHits() {
        let cache = Self.published(prompt: [1, 2, 3], generated: [4, 5])
        // The state holds 1 2 3 4 — the last generated token was never fed back.
        #expect(cache.tokens == [1, 2, 3, 4])
        #expect(cache.match([1, 2, 3, 4, 5, 6, 7]) == .hit(cached: 4))
    }

    @Test("one differing token anywhere is a whole miss, and says where")
    func divergenceMisses() {
        let cache = Self.published(prompt: [1, 2, 3], generated: [4, 5])
        #expect(cache.match([1, 2, 9, 4, 5, 6]) == .miss(divergedAt: 2))
        #expect(cache.match([9, 2, 3, 4, 5, 6]) == .miss(divergedAt: 0))
    }

    @Test("a prompt no longer than the state is a miss: prefill needs a row")
    func identicalPromptMisses() {
        let cache = Self.published(prompt: [1, 2, 3], generated: [4, 5])
        #expect(cache.match([1, 2, 3, 4]) == .miss(divergedAt: 4))
        #expect(cache.match([1, 2, 3]) == .miss(divergedAt: 3))
    }

    @Test("CACHE-5: an opted-out request neither reads nor writes")
    func optedOut() {
        var cache = Self.published(prompt: [1, 2, 3], generated: [4, 5])
        #expect(cache.match([1, 2, 3, 4, 5, 6], cachePrompt: false)
                == .miss(divergedAt: nil))
        cache.publish(promptIDs: [1, 2, 3, 4, 5, 6], generated: [7],
                      kvPosition: 6, cachePrompt: false)
        #expect(cache.tokens.isEmpty)
    }

    /// The bookkeeping the whole thing rests on, in both of its shapes. The
    /// speculative loop can end with the state holding *everything* — an
    /// accepted draft on the last pass consumes the token it predicted — and
    /// asserting the shorter form cost a whole turn's cache the first time this
    /// was measured (`41-PROMPT-CACHE.md` §3-2).
    @Test("the runner's position decides the length, not the token counts")
    func positionDecides() {
        let short = Self.published(prompt: [1, 2], generated: [3, 4], kvPosition: 3)
        #expect(short.tokens == [1, 2, 3])
        let whole = Self.published(prompt: [1, 2], generated: [3, 4], kvPosition: 4)
        #expect(whole.tokens == [1, 2, 3, 4])
        #expect(whole.match([1, 2, 3, 4, 5]) == .hit(cached: 4))
    }

    @Test("a position past what the request could have produced is refused")
    func impossiblePositionRefused() {
        var cache = Self.published(prompt: [1, 2, 3], generated: [4])
        let ok = cache.publish(promptIDs: [1, 2, 3], generated: [4], kvPosition: 9)
        #expect(!ok)
        #expect(cache.tokens.isEmpty)
    }

    @Test("a prefill-only run leaves exactly the prompt")
    func prefillOnlyRun() {
        // `max_tokens: 1` consumes the prompt and none of the answer, which is
        // what makes a resumed prefill comparable to a whole one.
        let cache = Self.published(prompt: [1, 2, 3, 4], generated: [5], kvPosition: 4)
        #expect(cache.tokens == [1, 2, 3, 4])
        #expect(cache.match([1, 2, 3, 4, 5]) == .hit(cached: 4))
    }

    @Test("invalidate drops the entry")
    func invalidateDrops() {
        var cache = Self.published(prompt: [1, 2, 3], generated: [4, 5])
        cache.invalidate()
        #expect(cache.tokens.isEmpty)
        #expect(cache.match([1, 2, 3, 4, 5, 6]) == .miss(divergedAt: nil))
    }
}
