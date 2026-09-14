import Foundation
import Testing
@testable import TsugumiServerCore

/// C0 for SPEC §7 on Qwen3.8 (`docs/qwen38/15` §2 G, `18`): where `Qwen38ServerSession` starts a request from,
/// before it touches the engine. One case per row of 15's requirement table, in token ids.
@Suite("C0 Qwen3.8 prompt cache")
struct Qwen38PromptCacheTests {
    // prompt = 10 11 12 (user body) 99 (<|im_end|>) 50 51 (generation prompt): P = 6, U = 3, P - 1 = 5.
    static let prompt: [Int32] = [10, 11, 12, 99, 50, 51]

    private static func afterFirstTurn() -> Qwen38PromptCache {
        var cache = Qwen38PromptCache()
        // Generated 20 21 99; the live state holds everything but the last token.
        cache.publish(prompt: prompt, generated: [20, 21, 99], kvPosition: 8)
        return cache
    }

    @Test("nothing held is a miss")
    func emptyMisses() {
        #expect(Qwen38PromptCache().decide(Self.prompt, checkpoints: [3, 5]).0 == .miss)
    }

    @Test("the next turn continues from the live state")
    func nextTurnIsLive() {
        let next = Self.prompt + [20, 21, 99, 7, 13, 99, 50, 51]
        let (decision, agreed) = Self.afterFirstTurn().decide(next, checkpoints: [3, 5])
        #expect(decision == .live(8))
        #expect(agreed == 8)
    }

    @Test("a regenerate of the same prompt restores P - 1, the first turn included")
    func regenerateRestoresPMinusOne() {
        #expect(Self.afterFirstTurn().decide(Self.prompt, checkpoints: [3, 5]).0 == .restore(5))
    }

    @Test("an instructed regenerate restores the end of the user body")
    func instructedRegenerateRestoresU() {
        let instructed: [Int32] = [10, 11, 12, 30, 31, 99, 50, 51]
        #expect(Self.afterFirstTurn().decide(instructed, checkpoints: [3, 5]).0 == .restore(3))
    }

    @Test("a redraw that diverges inside the answer restores the newest checkpoint behind it")
    func redrawDivergenceUsesCheckpointBehind() {
        // A checkpoint before a `<tool_call>` at 7; the client redraws token 7 the same and token 8 differently.
        var cache = Qwen38PromptCache()
        cache.publish(prompt: Self.prompt, generated: [20, 77, 40, 41, 99], kvPosition: 10)
        let redrawn = Self.prompt + [20, 77, 42, 41, 99, 9, 99, 50, 51]
        #expect(cache.decide(redrawn, checkpoints: [3, 5, 7]).0 == .restore(7))
    }

    @Test("after an interrupted run the live state is not a candidate, the checkpoints are")
    func interruptedKeepsCheckpoints() {
        var cache = Self.afterFirstTurn()
        cache.publishInterrupted(prompt: Self.prompt)
        #expect(cache.livePosition == nil)
        #expect(cache.decide(Self.prompt, checkpoints: [3, 5]).0 == .restore(5))
        #expect(cache.decide(Self.prompt + [20], checkpoints: []).0 == .miss)
    }

    @Test("a checkpoint past the divergence or at the whole prompt is never used")
    func unusableCheckpoints() {
        let cache = Self.afterFirstTurn()
        let diverged: [Int32] = [10, 11, 8, 99, 50, 51]
        let (decision, agreed) = cache.decide(diverged, checkpoints: [3, 5])
        #expect(decision == .miss)
        #expect(agreed == 2)
        // The live state (8) at or past the prompt's length leaves nothing to draw from.
        #expect(cache.decide(Self.prompt + [20, 21], checkpoints: []).0 == .miss)
    }

    @Test("the live state wins over a checkpoint at or behind it")
    func liveAgainstCheckpoint() {
        var cache = Qwen38PromptCache()
        cache.publish(prompt: Self.prompt, generated: [20], kvPosition: 6)
        let next = Self.prompt + [20, 9, 9]
        #expect(cache.decide(next, checkpoints: [6]).0 == .live(6))
        #expect(cache.decide(next, checkpoints: [3]).0 == .live(6))
    }

    // `docs/qwen38/19`: the real split the app hit. The model generated `、` `前回` `答` (5205 247988 96517); rendering
    // the answer back encodes `、` `前` `回答` (5205 95990 97913). Byte-level pieces from the checkpoint's tokenizer.json.
    static let pieces: [Int32: String] = [
        5205: "ãĢģ", 247988: "åīįåĽŀ", 96517: "çŃĶ", 95990: "åīį", 97913: "åĽŀçŃĶ",
        172182: "ãģŁ", 393: " $", 87: "x", 1088: " \\", 20: "a", 21: "b", 7: "c",
    ]
    static let added: Set<Int32> = [99, 50, 51]
    static func piece(_ id: Int32) -> String? { added.contains(id) ? nil : pieces[id] }

    private static func afterSplitAnswer() -> Qwen38PromptCache {
        var cache = Qwen38PromptCache()
        cache.publish(prompt: prompt, generated: [20, 5205, 247988, 96517, 172182, 393, 87, 99], kvPosition: 13)
        return cache
    }

    @Test("an answer re-rendered with a different split still continues from the live state")
    func resplitAnswerIsLive() {
        let cache = Self.afterSplitAnswer()
        let rendered = Self.prompt + [20, 5205, 95990, 97913, 172182, 393, 87, 99, 7, 99, 50, 51]
        #expect(cache.decide(rendered, checkpoints: [5]).0 == .restore(5))
        let (aligned, spans) = cache.aligned(rendered, piece: Self.piece)
        #expect(spans == 1)
        #expect(aligned == Self.prompt + [20, 5205, 247988, 96517, 172182, 393, 87, 99, 7, 99, 50, 51])
        #expect(cache.decide(aligned, checkpoints: [5]).0 == .live(13))
    }

    @Test("different bytes are not a re-split: the prompt is left as rendered")
    func differentTextIsKept() {
        let cache = Self.afterSplitAnswer()
        // `前回` `ã` in place of `前回` `答`, and a trailing piece the answer did not have.
        let changed = Self.prompt + [20, 5205, 247988, 172182, 172182, 393, 87, 99, 50, 51]
        #expect(cache.aligned(changed, piece: Self.piece).tokens == changed)
        let rendered = Self.prompt + [20, 5205, 95990, 97913, 172182, 1088, 87, 99, 50, 51]
        let (aligned, spans) = cache.aligned(rendered, piece: Self.piece)
        // The re-split before the difference is still taken; the rest stays rendered and the live state is passed over.
        #expect(spans == 1)
        #expect(aligned == Self.prompt + [20, 5205, 247988, 96517, 172182, 1088, 87, 99, 50, 51])
        #expect(cache.decide(aligned, checkpoints: [5]).0 == .restore(5))
    }

    @Test("an added token is never re-split")
    func addedTokensStay() {
        var cache = Qwen38PromptCache()
        cache.publish(prompt: [20, 99], generated: [21], kvPosition: 3)
        let rendered: [Int32] = [20, 50, 21, 7]
        #expect(cache.aligned(rendered, piece: Self.piece).tokens == rendered)
    }

    @Test("cache_prompt false neither reads nor keeps anything")
    func optOut() {
        var cache = Self.afterFirstTurn()
        #expect(cache.decide(Self.prompt, checkpoints: [5], cachePrompt: false).0 == .miss)
        cache.publish(prompt: Self.prompt, generated: [20], kvPosition: 6, cachePrompt: false)
        #expect(cache.tokens.isEmpty)
    }
}
