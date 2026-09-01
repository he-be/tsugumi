import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// SPEC §8 **RSN-4** where it actually has to hold: inside the decode loop.
///
/// The model here is the measured defect made deterministic — it opens the
/// thought channel and then never closes it. Without a forcer the completion
/// runs to `max_tokens` with an empty answer, which is precisely what
/// CONFORMANCE §2 recorded (`max_tokens: 80` で reasoning 357 字・content 空).
/// With one, the closing tag goes in and the answer follows.
@Suite("RSN-4 forced tokens in the decode loop")
struct RawCompletionForcedTokenTests {

    /// Records what the constraint was fed, in order. GEN-7's mechanism is not
    /// under test here; the ordering promise is.
    private final class RecordingConstraint: GenerationConstraint, @unchecked Sendable {
        private(set) var accepted: [Int32] = []
        var mayEndHere: Bool { true }
        func allows(tokenID: Int32) -> Bool { true }
        func fillAllowedMask(_ allowed: UnsafeMutableBufferPointer<Bool>) throws {
            for i in 0..<allowed.count { allowed[i] = true }
        }
        func accept(tokenID: Int32) throws { accepted.append(tokenID) }
    }

    private struct Outcome {
        let emitted: [Int32]
        let result: RawDecodeResult
        let content: String
    }

    /// A model that opens the thought channel, thinks forever, and — if it is
    /// ever let out of the channel — writes one word and stops.
    private func runEndlessThinker(
        tokenizer: GFTokenizer,
        maxNewTokens: Int,
        forcer: (any ForcedTokenSource)?,
        constraint: (any GenerationConstraint)? = nil
    ) async throws -> Outcome {
        let thought = tokenizer.encode("a", addBOS: false).first!
        let answer = tokenizer.encode("b", addBOS: false).first!
        let next: [Int32: Int32] = [
            tokenizer.channelStartID: thought,
            thought: thought,
            tokenizer.channelEndID: answer,
            answer: tokenizer.eosID,
        ]
        let producer = ScriptedLogitProducer(vocabSize: tokenizer.vocabSize) { input, _ in
            .argmax(next[input] ?? tokenizer.channelStartID)
        }
        let ctx = try MetalContext()
        let scratch = try RawCompletionScratch(context: ctx, vocab: tokenizer.vocabSize)
        var emitted: [Int32] = []
        var content = ""
        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: tokenizer.encode("go", addBOS: true),
            config: GenerationConfig(maxNewTokens: maxNewTokens, temperature: 0),
            constraint: constraint,
            forcer: forcer,
            context: ctx,
            scratch: scratch,
            prefillConfig: .off) { progress in
                switch progress {
                case .token(_, let id, let delta):
                    emitted.append(id)
                    content += delta
                case .tail(let text):
                    content += text
                case .prefill:
                    break
                }
            }
        return Outcome(emitted: emitted, result: result, content: content)
    }

    private func forcer(_ tokenizer: GFTokenizer, budget: Int) -> ReasoningBudgetForcer {
        ReasoningBudgetForcer(startTokenID: tokenizer.channelStartID,
                              endTokenID: tokenizer.channelEndID,
                              forcedTokenIDs: [tokenizer.channelEndID],
                              budget: budget,
                              deadline: Int.max)
    }

    /// The defect itself, so the fix is measured against something. No forcer:
    /// the run ends on `max_tokens` having said nothing outside the channel.
    @Test func RSN_4_without_a_budget_the_thought_channel_eats_the_whole_answer() async throws {
        let tok = try await GFTokenizer.load()
        let outcome = try await runEndlessThinker(tokenizer: tok,
                                                  maxNewTokens: 12,
                                                  forcer: nil)
        #expect(outcome.result.reason == .maxTokens)
        #expect(!outcome.emitted.contains(tok.channelEndID))
        #expect(outcome.emitted.count == 12)
    }

    /// RSN-4. The tag is forced in, the model leaves the channel, and the
    /// answer is not empty.
    @Test func RSN_4_the_closing_tag_is_forced_and_the_answer_follows() async throws {
        let tok = try await GFTokenizer.load()
        let answer = tok.encode("b", addBOS: false).first!
        let outcome = try await runEndlessThinker(tokenizer: tok,
                                                  maxNewTokens: 32,
                                                  forcer: forcer(tok, budget: 4))
        // <|channel> + 4 thought tokens + the forced <channel|> + "b". The
        // end-of-generation token that follows ends the loop before it is
        // reported, as it does for every other completion.
        let thought = tok.encode("a", addBOS: false).first!
        #expect(outcome.emitted
                == [tok.channelStartID] + [Int32](repeating: thought, count: 4)
                + [tok.channelEndID, answer])
        #expect(outcome.result.reason == .eos)
        #expect(outcome.content.contains("b"), "本文が空でない (RSN-4)")
    }

    /// SPEC §7 INV-1. A forced token is in the KV like any other token, so it
    /// has to be in the token list the prompt cache publishes — at the same
    /// index, in the same order. If it were not, the next turn's redraw would
    /// disagree with the KV at exactly that position.
    @Test func RSN_4_a_forced_token_reaches_the_prompt_cache_in_order() async throws {
        let tok = try await GFTokenizer.load()
        let outcome = try await runEndlessThinker(tokenizer: tok,
                                                  maxNewTokens: 32,
                                                  forcer: forcer(tok, budget: 4))
        let prompt = tok.encode("go", addBOS: true)
        let generated = Array(outcome.result.kvBackedTokenIDs.dropFirst(prompt.count))
        #expect(generated == outcome.emitted)
        #expect(generated.contains(tok.channelEndID))
    }

    /// GEN-6 / GEN-13. A forced token is fed to the constraint exactly where a
    /// sampled one would have been — the constraint decides what to do with it
    /// (a lazy grammar is suppressed inside the block and ignores it; a
    /// non-lazy one has the thought block in its `root` and consumes it).
    /// Skipping this feed is how the grammar state and the KV drift apart.
    @Test func RSN_4_a_forced_token_is_fed_to_the_constraint_like_a_sampled_one() async throws {
        let tok = try await GFTokenizer.load()
        let constraint = RecordingConstraint()
        let outcome = try await runEndlessThinker(tokenizer: tok,
                                                  maxNewTokens: 32,
                                                  forcer: forcer(tok, budget: 4),
                                                  constraint: constraint)
        // Every kept token bar the end-of-generation id, which the gate rules
        // on with `mayEndHere` and never forwards.
        #expect(constraint.accepted == outcome.emitted)
        #expect(constraint.accepted.contains(tok.channelEndID))
    }
}
