import Foundation
import Testing
@testable import TurboFieldfareServerCore

/// C2 (CONFORMANCE §1) for SPEC **CACHE-2 / CACHE-8** on the Ornith family.
///
/// **This suite is red on purpose** (CONFORMANCE §2, 2026-08-23). The claims are
/// written the way C2 requires — "the divergence is at N, so what gets
/// re-prefilled is N onward" — never "this shape hits".
///
/// The rule under test, read off SPEC §7:
///
/// - CACHE-2: a partial match is used as it is. A family whose memory cannot be
///   truncated to the reuse point uses CACHE-8's checkpoints instead of
///   throwing the conversation away.
/// - CACHE-8: a checkpoint is taken at the end of the prompt whatever the
///   spacing (the reference forces one at the last user message / near the
///   prompt end, `server-context.cpp:3529`, because that is where the next turn
///   diverges). On reuse, the newest checkpoint at or before the reuse point is
///   restored; only when there is none does the whole prompt get re-processed.
///
/// Every case here is the shape that was measured in pi's session
/// `01a02a00-0f89-792f-a434-3c59e41f0bb9` (2026-08-22): the client appends a
/// tool result, the conversation is unchanged before that, and the divergence
/// lands *inside the assistant turn the server itself just wrote*. Today all of
/// those are a full miss.
@Suite("C2 Ornith prompt cache — CACHE-2 / CACHE-8")
struct QwenPromptCacheCheckpointTests {
    /// A published state, the way `QwenServerSession` publishes one: the prompt
    /// it prefilled plus the tokens it generated, minus the last token the loop
    /// never fed back — and the checkpoint the session takes when the prefill
    /// ends (`captureAtPromptEnd`, CACHE-8's forced one).
    ///
    /// The checkpoint is an *input* to the rule under test: this type only ever
    /// names positions the session says it holds a copy of. That the session
    /// really takes one at every prompt end is a claim about the session and
    /// needs the runner, so it is C3's.
    private static func published(prompt: [Int32], generated: [Int32]) -> QwenPromptCache {
        var cache = QwenPromptCache()
        cache.publish(promptIDs: prompt,
                      generated: generated,
                      kvPosition: prompt.count + generated.count - 1)
        cache.recordCheckpoint(at: prompt.count)
        return cache
    }

    private static func ids(_ range: Range<Int>) -> [Int32] {
        range.map { Int32(truncatingIfNeeded: $0 &* 2_654_435_761) }
    }

    /// The next request: the same conversation, the assistant turn described
    /// back with `divergeAfter` of its tokens agreeing, then something else.
    private static func nextPrompt(prompt: [Int32],
                                   generated: [Int32],
                                   divergeAfter: Int,
                                   tail: Int) -> [Int32] {
        prompt + generated.prefix(divergeAfter) + ids(9_000_000..<(9_000_000 + tail))
    }

    @Test("CACHE-8: a divergence inside the assistant turn re-prefills from the prompt's end, not from zero")
    func divergenceInsideTheTurn() {
        let prompt = Self.ids(0..<600)
        let generated = Self.ids(600..<2_600)
        let cache = Self.published(prompt: prompt, generated: generated)
        let next = Self.nextPrompt(prompt: prompt, generated: generated,
                                   divergeAfter: 71, tail: 400)
        // The checkpoint at the prompt's end is at or before the divergence, so
        // it is the one restored. 600 of the 671 agreeing tokens are kept.
        #expect(cache.match(next) == .hit(cached: prompt.count))
    }

    @Test("CACHE-8: the prompt survives even when the very first generated token differs")
    func divergenceAtTheFirstGeneratedToken() {
        let prompt = Self.ids(0..<600)
        let generated = Self.ids(600..<2_600)
        let cache = Self.published(prompt: prompt, generated: generated)
        let next = Self.nextPrompt(prompt: prompt, generated: generated,
                                   divergeAfter: 0, tail: 400)
        #expect(cache.match(next) == .hit(cached: prompt.count))
    }

    @Test("CACHE-8: a divergence inside the prompt has no checkpoint behind it, so it is a full re-process")
    func divergenceInsideThePrompt() {
        let prompt = Self.ids(0..<600)
        let generated = Self.ids(600..<2_600)
        let cache = Self.published(prompt: prompt, generated: generated)
        // Longer than the held state, so this is the divergence rule and not
        // CACHE-3's "a prompt no longer than the state is a miss".
        let next = Array(prompt.prefix(410)) + Self.ids(9_000_000..<9_002_400)
        #expect(next.count > cache.tokens.count)
        #expect(cache.match(next) == .miss(divergedAt: 410))
    }

    @Test("CACHE-2: a strict extension still reuses the whole state")
    func strictExtensionUnchanged() {
        let prompt = Self.ids(0..<600)
        let generated = Self.ids(600..<2_600)
        let cache = Self.published(prompt: prompt, generated: generated)
        let next = prompt + generated + Self.ids(9_000_000..<9_000_400)
        #expect(cache.match(next) == .hit(cached: prompt.count + generated.count - 1))
    }

    /// The measured turn, at its measured sizes: request `chatcmpl-52298b62`
    /// (2026-08-22 16:31:15Z). The server logged
    /// `miss diverged_at=33873 held=35836` and re-prefilled 35871 tokens to
    /// produce 2101. The prompt it had already prefilled was 33802 tokens.
    @Test("CACHE-2: the measured turn keeps 33802 of its 35871 tokens instead of none")
    func measuredTurn() {
        let prompt = Self.ids(0..<33_802)
        let generated = Self.ids(33_802..<35_837)
        let cache = Self.published(prompt: prompt, generated: generated)
        #expect(cache.tokens.count == 35_836)
        let next = Self.nextPrompt(prompt: prompt, generated: generated,
                                   divergeAfter: 71, tail: 1_998)
        #expect(next.count == 35_871)
        #expect(cache.match(next) == .hit(cached: 33_802))
    }
}
