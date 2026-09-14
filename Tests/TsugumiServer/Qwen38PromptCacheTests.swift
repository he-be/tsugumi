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

    @Test("cache_prompt false neither reads nor keeps anything")
    func optOut() {
        var cache = Self.afterFirstTurn()
        #expect(cache.decide(Self.prompt, checkpoints: [5], cachePrompt: false).0 == .miss)
        cache.publish(prompt: Self.prompt, generated: [20], kvPosition: 6, cachePrompt: false)
        #expect(cache.tokens.isEmpty)
    }
}
