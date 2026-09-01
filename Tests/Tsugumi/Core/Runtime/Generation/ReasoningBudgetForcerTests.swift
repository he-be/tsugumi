import Testing
import Foundation
@testable import Tsugumi

/// SPEC §8 **RSN-4**: 思考の予算が尽きたら終了タグを強制挿入して本文へ移らせる。
///
/// The state machine only — no tokenizer, no Metal, no server. Start and end
/// are plain ids here for the same reason the reference's tests use synthetic
/// tokens: what is under test is the counting and the transitions, not which
/// token this template happens to close a thought block with.
@Suite("RSN-4 reasoning budget forcer")
struct ReasoningBudgetForcerTests {
    private let start: Int32 = 100
    private let end: Int32 = 200
    private let word: Int32 = 7

    private func forcer(budget: Int, deadline: Int = Int.max) -> ReasoningBudgetForcer {
        ReasoningBudgetForcer(startTokenID: start,
                              endTokenID: end,
                              forcedTokenIDs: [end],
                              budget: budget,
                              deadline: deadline)
    }

    /// Feed one token the way the decode loop does and answer with what the
    /// next position would emit.
    @discardableResult
    private func step(_ subject: ReasoningBudgetForcer,
                      _ tokenID: Int32,
                      at index: Int,
                      completesCharacter: Bool = true) -> Int32? {
        subject.accept(tokenID: tokenID,
                       generationIndex: index,
                       completesCharacter: completesCharacter)
        return subject.nextForcedToken()
    }

    /// Nothing is forced outside a thought block, however long the answer runs.
    @Test func RSN_4_forces_nothing_until_the_thought_block_opens() {
        let subject = forcer(budget: 1)
        for index in 0..<8 {
            #expect(step(subject, word, at: index) == nil)
        }
        #expect(subject.state == .idle)
    }

    /// The measured defect, in miniature: the block opens, spends its budget,
    /// and is closed from the outside so the answer can start.
    @Test func RSN_4_forces_the_closing_tag_when_the_token_budget_runs_out() {
        let subject = forcer(budget: 3)
        #expect(step(subject, start, at: 0) == nil)
        #expect(subject.state == .counting)
        #expect(step(subject, word, at: 1) == nil)
        #expect(step(subject, word, at: 2) == nil)
        // The third thought token spends the last of the budget, so the next
        // position is not the model's to choose.
        #expect(step(subject, word, at: 3) == end)
        #expect(subject.state == .forcing)
        // The loop emits it and feeds it back, which completes the sequence.
        #expect(step(subject, end, at: 4) == nil)
        #expect(subject.state == .done)
    }

    /// A model that closes the block on its own is never interfered with — the
    /// budget is a ceiling, not a schedule.
    @Test func RSN_4_a_natural_end_tag_spends_no_forcing() {
        let subject = forcer(budget: 5)
        step(subject, start, at: 0)
        step(subject, word, at: 1)
        #expect(step(subject, end, at: 2) == nil)
        #expect(subject.state == .done)
    }

    /// RSN-4's second half: `max_tokens` bounds the block even when the
    /// explicit budget does not. `deadline` is the last index at which the tag
    /// may still start.
    @Test func RSN_4_the_max_tokens_deadline_closes_an_unlimited_budget() {
        let subject = forcer(budget: ReasoningBudgetForcer.unlimited, deadline: 5)
        step(subject, start, at: 0)
        for index in 1...3 {
            #expect(step(subject, word, at: index) == nil)
        }
        #expect(step(subject, word, at: 4) == end, "the tag must start at the deadline")
        #expect(subject.state == .forcing)
    }

    /// The reference promotes a zero budget straight to forcing; so does this.
    /// (A request that asks for `reasoning_budget_tokens: 0` never gets here —
    /// RSN-1 closes the channel in the prompt instead — but a `max_tokens` so
    /// small that the deadline has already passed lands in exactly this state.)
    @Test func RSN_4_a_spent_budget_forces_as_soon_as_the_block_opens() {
        let subject = forcer(budget: ReasoningBudgetForcer.unlimited, deadline: 0)
        #expect(step(subject, start, at: 0) == end)
        #expect(subject.state == .forcing)
    }

    /// `WAITING_UTF8` in the reference. The token that spent the budget left a
    /// multi-byte character half-written, so the tag waits for it to close
    /// rather than cutting it in two.
    @Test func RSN_4_a_half_written_character_finishes_before_the_tag_goes_in() {
        let subject = forcer(budget: 2)
        step(subject, start, at: 0)
        step(subject, word, at: 1)
        #expect(step(subject, word, at: 2, completesCharacter: false) == nil)
        #expect(subject.state == .waitingUTF8)
        #expect(step(subject, word, at: 3, completesCharacter: false) == nil,
                "still inside the character")
        #expect(step(subject, word, at: 4) == end)
        #expect(subject.state == .forcing)
    }

    /// A model may open more than one thought block in one answer; each gets
    /// its own window (the reference re-arms from `DONE` on a start tag).
    @Test func RSN_4_a_second_thought_block_gets_a_fresh_budget() {
        let subject = forcer(budget: 2)
        step(subject, start, at: 0)
        step(subject, end, at: 1)
        #expect(subject.state == .done)
        #expect(step(subject, word, at: 2) == nil)
        step(subject, start, at: 3)
        #expect(subject.state == .counting)
        step(subject, word, at: 4)
        #expect(step(subject, word, at: 5) == end)
    }

    /// The whole forced sequence goes out, one token per position — the
    /// reference forces "message + end tag" and nothing here assumes a
    /// one-token tag.
    @Test func RSN_4_a_multi_token_forced_sequence_is_emitted_in_order() {
        let subject = ReasoningBudgetForcer(startTokenID: start,
                                            endTokenID: end,
                                            forcedTokenIDs: [42, 43, end],
                                            budget: 1,
                                            deadline: Int.max)
        step(subject, start, at: 0)
        #expect(step(subject, word, at: 1) == 42)
        #expect(step(subject, 42, at: 2) == 43)
        #expect(step(subject, 43, at: 3) == end)
        #expect(step(subject, end, at: 4) == nil)
        #expect(subject.state == .done)
    }
}
