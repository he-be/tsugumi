import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// SPEC §6 GEN-7 (拘束は棄却サンプリングで入れる) and §12 DEV-14 (拘束のある
/// 要求は投機デコードを使わない), checked with stub constraints only — no
/// grammar, no schema, no weights.
///
/// The stubs stand in for whatever a real constraint turns out to be: the
/// sampler is only ever told "is this id allowed", "which ids are allowed", and
/// "here is the token that was kept".
@Suite struct GenerationConstraintTests {

    // MARK: - Stubs

    /// Allows an explicit id set, which may change with the number of tokens
    /// accepted so far. `nil` from `allowedAtStep` means "everything".
    final class StubConstraint: GenerationConstraint, @unchecked Sendable {
        private let allowedAtStep: @Sendable (Int) -> Set<Int32>?
        private let endsAfter: Int
        private(set) var accepted: [Int32] = []
        /// How many times the whole-vocabulary mask was requested — i.e. how
        /// many draws took the rejection path.
        private(set) var maskFills = 0

        init(endsAfter: Int = 0,
             allowedAtStep: @escaping @Sendable (Int) -> Set<Int32>?) {
            self.endsAfter = endsAfter
            self.allowedAtStep = allowedAtStep
        }

        convenience init(allowing allowed: Set<Int32>, endsAfter: Int = 0) {
            self.init(endsAfter: endsAfter, allowedAtStep: { _ in allowed })
        }

        /// Allows every id; only `mayEndHere` is interesting.
        convenience init(endsAfter: Int) {
            self.init(endsAfter: endsAfter, allowedAtStep: { _ in nil })
        }

        var mayEndHere: Bool { accepted.count >= endsAfter }

        func allows(tokenID: Int32) -> Bool {
            allowedAtStep(accepted.count)?.contains(tokenID) ?? true
        }

        func fillAllowedMask(_ allowed: UnsafeMutableBufferPointer<Bool>) throws {
            maskFills += 1
            let set = allowedAtStep(accepted.count)
            for i in 0..<allowed.count {
                allowed[i] = set?.contains(Int32(i)) ?? true
            }
        }

        func accept(tokenID: Int32) throws { accepted.append(tokenID) }
    }

    /// Allows exactly one fixed token sequence, then anything (so the sequence
    /// can be followed by a stop token). It may only end once the sequence is
    /// complete.
    final class SequenceConstraint: GenerationConstraint, @unchecked Sendable {
        private let sequence: [Int32]
        private(set) var accepted: [Int32] = []
        private(set) var maskFills = 0

        init(_ sequence: [Int32]) { self.sequence = sequence }

        var mayEndHere: Bool { accepted.count >= sequence.count }

        func allows(tokenID: Int32) -> Bool {
            accepted.count < sequence.count ? tokenID == sequence[accepted.count] : true
        }

        func fillAllowedMask(_ allowed: UnsafeMutableBufferPointer<Bool>) throws {
            maskFills += 1
            for i in 0..<allowed.count { allowed[i] = allows(tokenID: Int32(i)) }
        }

        func accept(tokenID: Int32) throws { accepted.append(tokenID) }
    }

    /// The shape a real grammar has, which the other stubs do not: it knows
    /// nothing about stop tokens, so it rejects them like any other id that does
    /// not continue the structure, and once the structure is complete nothing
    /// continues it at all. `accept` on an id it did not allow is a hard error —
    /// the reference's "no surviving stacks".
    final class GrammarLikeConstraint: GenerationConstraint, @unchecked Sendable {
        struct NoSurvivingStacks: Error {}

        private let body: [Int32]
        private(set) var accepted: [Int32] = []
        private(set) var maskFills = 0

        init(_ body: [Int32]) { self.body = body }

        var mayEndHere: Bool { accepted.count >= body.count }

        func allows(tokenID: Int32) -> Bool {
            accepted.count < body.count && tokenID == body[accepted.count]
        }

        func fillAllowedMask(_ allowed: UnsafeMutableBufferPointer<Bool>) throws {
            maskFills += 1
            for i in 0..<allowed.count { allowed[i] = allows(tokenID: Int32(i)) }
        }

        func accept(tokenID: Int32) throws {
            guard allows(tokenID: tokenID) else { throw NoSurvivingStacks() }
            accepted.append(tokenID)
        }
    }

    /// The pathological state GEN-7 calls an error: nothing at all is allowed.
    final class RejectEverythingConstraint: GenerationConstraint, @unchecked Sendable {
        var mayEndHere: Bool { false }
        func allows(tokenID: Int32) -> Bool { false }
        func fillAllowedMask(_ allowed: UnsafeMutableBufferPointer<Bool>) throws {
            for i in 0..<allowed.count { allowed[i] = false }
        }
        func accept(tokenID: Int32) throws {}
    }

    /// A producer that answers with a GPU argmax and never writes logits — the
    /// fused head, without the model behind it.
    final class FusedGreedyOnlyProducer: FusedGreedyReporting, @unchecked Sendable {
        let usesFusedGreedyHead = true
        func reset() {}
        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {}
    }

    // MARK: - Rigs

    static func logits(_ weights: [Int32: Float]) -> ScriptedLogitProducer.Step {
        let maxID = Int(weights.keys.max() ?? 0)
        var values = [Float](repeating: -30, count: maxID + 1)
        for (id, weight) in weights { values[Int(id)] = weight }
        return .vector(values)
    }

    struct LoopOutcome {
        let emitted: [Int32]
        let result: RawDecodeResult
    }

    func runLoop(producer: any LogitProducer,
                 tokenizer: GFTokenizer,
                 config: GenerationConfig,
                 constraint: (any GenerationConstraint)? = nil) async throws -> LoopOutcome {
        let ctx = try MetalContext()
        let scratch = try RawCompletionScratch(context: ctx, vocab: tokenizer.vocabSize)
        var emitted: [Int32] = []
        let result = try await runRawCompletion(producer: producer,
                                                tokenizer: tokenizer,
                                                promptIds: tokenizer.encode("go", addBOS: true),
                                                config: config,
                                                constraint: constraint,
                                                context: ctx,
                                                scratch: scratch,
                                                prefillConfig: .off) { progress in
            if case .token(_, let id, _) = progress { emitted.append(id) }
        }
        return LoopOutcome(emitted: emitted, result: result)
    }

    // MARK: - GEN-7: the fast path

    /// The common path: the drawn token satisfies the constraint, so it is kept
    /// as drawn and the whole-vocabulary mask is never built.
    @Test func GEN_7_allowedDrawIsKeptWithoutBuildingTheMask() async throws {
        let tok = try await GFTokenizer.load()
        let idA = tok.encode("a", addBOS: false).first!
        let idB = tok.encode("b", addBOS: false).first!
        let next: [Int32: Int32] = [idA: idB, idB: tok.eosID]
        let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize) { input, _ in
            .argmax(next[input] ?? idA)
        }
        let constraint = StubConstraint(endsAfter: 0)
        let outcome = try await runLoop(producer: producer, tokenizer: tok,
                                        config: GenerationConfig(maxNewTokens: 50,
                                                                 temperature: 0),
                                        constraint: constraint)
        #expect(outcome.emitted == [idA, idB])
        #expect(outcome.result.reason == .eos)
        #expect(constraint.maskFills == 0, "the mask ran on the fast path")
    }

    /// A constrained run and an unconstrained run of the same automaton emit the
    /// same tokens when the constraint allows what the model draws.
    @Test func GEN_7_constraintThatAllowsTheDrawChangesNothing() async throws {
        let tok = try await GFTokenizer.load()
        let idA = tok.encode("a", addBOS: false).first!
        let idB = tok.encode("b", addBOS: false).first!
        let next: [Int32: Int32] = [idA: idB, idB: idA]
        let config = GenerationConfig(maxNewTokens: 6, temperature: 1.0, seed: 11)
        func run(_ constraint: (any GenerationConstraint)?) async throws -> [Int32] {
            let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize) { input, _ in
                var weights: [Int32: Float] = [idA: 8, idB: 8]
                weights[next[input] ?? idA] = 30
                return Self.logits(weights)
            }
            return try await runLoop(producer: producer, tokenizer: tok,
                                     config: config, constraint: constraint).emitted
        }
        let plain = try await run(nil)
        let constrained = try await run(StubConstraint(endsAfter: 0))
        #expect(plain == constrained)
    }

    // MARK: - GEN-7: the rejection path

    /// Greedy: the model's argmax is forbidden, so the draw is repeated against
    /// the masked vocabulary and the best *allowed* token comes out.
    @Test func GEN_7_rejectedGreedyDrawResamplesIntoAnAllowedToken() async throws {
        let tok = try await GFTokenizer.load()
        let idA = tok.encode("a", addBOS: false).first!
        let idB = tok.encode("b", addBOS: false).first!
        let idC = tok.encode("c", addBOS: false).first!
        let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize) { _, _ in
            Self.logits([idA: 30, idB: 10, idC: 4])
        }
        let constraint = StubConstraint(allowing: [idB, idC])
        let outcome = try await runLoop(producer: producer, tokenizer: tok,
                                        config: GenerationConfig(maxNewTokens: 3,
                                                                 temperature: 0),
                                        constraint: constraint)
        #expect(outcome.emitted == [idB, idB, idB])
        #expect(constraint.maskFills == 3)
        #expect(constraint.accepted == [idB, idB, idB])
    }

    /// Sampled: same rule, with a seeded temperature draw. Every emitted token
    /// is inside the allowed set — not "usually", every one.
    @Test func GEN_7_rejectedSampledDrawResamplesIntoAnAllowedToken() async throws {
        let tok = try await GFTokenizer.load()
        let idA = tok.encode("a", addBOS: false).first!
        let idB = tok.encode("b", addBOS: false).first!
        let idC = tok.encode("c", addBOS: false).first!
        let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize) { _, _ in
            Self.logits([idA: 30, idB: 6, idC: 6])
        }
        let constraint = StubConstraint(allowing: [idB, idC])
        let outcome = try await runLoop(producer: producer, tokenizer: tok,
                                        config: GenerationConfig(maxNewTokens: 8,
                                                                 temperature: 1.0,
                                                                 seed: 4242),
                                        constraint: constraint)
        #expect(outcome.emitted.count == 8)
        #expect(outcome.emitted.allSatisfy { $0 == idB || $0 == idC },
                "emitted \(outcome.emitted) outside the allowed set")
        #expect(constraint.maskFills >= 1)
    }

    /// A constraint that spells out one token sequence gets exactly that
    /// sequence, whatever the model prefers.
    @Test func GEN_7_fixedSequenceConstraintIsFollowedExactly() async throws {
        let tok = try await GFTokenizer.load()
        let want = tok.encode("hello", addBOS: false)
        let other = tok.encode("zzz", addBOS: false).first!
        let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize) { _, _ in
            .argmax(other)
        }
        let constraint = SequenceConstraint(want)
        let outcome = try await runLoop(producer: producer, tokenizer: tok,
                                        config: GenerationConfig(maxNewTokens: want.count,
                                                                 temperature: 0),
                                        constraint: constraint)
        #expect(outcome.emitted == want)
    }

    /// The rejection path is visible: the same buffer draws an unconstrained
    /// token on the fast path and a resampled one under a constraint.
    @Test func GEN_7_samplePathReportsTheResample() throws {
        let ctx = try MetalContext()
        let vocab = 2048
        let scratch = try RawCompletionScratch(context: ctx, vocab: vocab)
        let ptr = scratch.logits.contents().bindMemory(to: Float16.self, capacity: vocab)
        for i in 0..<vocab { ptr[i] = Float16(-8.0) }
        ptr[100] = Float16(9.0); ptr[200] = Float16(4.0); ptr[300] = Float16(2.0)
        let config = GenerationConfig(temperature: 0)

        let unconstrained = try sampleOnce(scratch: scratch, context: ctx, history: [],
                                           config: config, position: 0, constraint: nil)
        #expect(unconstrained.id == 100)
        #expect(unconstrained.path == .greedyGPU)

        let allowing = StubConstraint(allowing: [100, 200])
        let fast = try sampleOnce(scratch: scratch, context: ctx, history: [],
                                  config: config, position: 0,
                                  constraint: ConstraintGate(constraint: allowing,
                                                             endOfGenerationTokenIDs: []))
        #expect(fast.id == 100)
        #expect(fast.path == .greedyGPU)
        #expect(allowing.maskFills == 0)

        let rejecting = StubConstraint(allowing: [200, 300])
        let resampled = try sampleOnce(scratch: scratch, context: ctx, history: [],
                                       config: config, position: 0,
                                       constraint: ConstraintGate(constraint: rejecting,
                                                                  endOfGenerationTokenIDs: []))
        #expect(resampled.id == 200)
        #expect(resampled.path == .constraintResampled)
        #expect(rejecting.maskFills == 1)
    }

    // MARK: - GEN-7: accept

    /// `accept` sees exactly the tokens the loop kept, in order. The loop hands
    /// every one of them to the gate, including the token that ends generation;
    /// the gate withholds that one when it is an end-of-generation id the
    /// constraint never ruled on.
    @Test func GEN_7_acceptReceivesEveryGeneratedTokenInOrder() async throws {
        let tok = try await GFTokenizer.load()
        let idA = tok.encode("a", addBOS: false).first!
        let idB = tok.encode("b", addBOS: false).first!
        let next: [Int32: Int32] = [idA: idB, idB: tok.endOfTurnID]
        let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize) { input, _ in
            .argmax(next[input] ?? idA)
        }
        let constraint = StubConstraint(endsAfter: 0)
        let outcome = try await runLoop(producer: producer, tokenizer: tok,
                                        config: GenerationConfig(maxNewTokens: 50,
                                                                 temperature: 0),
                                        constraint: constraint)
        #expect(outcome.emitted == [idA, idB])
        #expect(constraint.accepted == [idA, idB])
        #expect(outcome.result.newTokens == 3)
    }

    // MARK: - GEN-7: mayEndHere

    /// End-of-generation tokens are masked while the constraint cannot end, and
    /// are drawn normally the moment it can.
    @Test func GEN_7_endOfGenerationIsWithheldUntilMayEndHere() async throws {
        let tok = try await GFTokenizer.load()
        let idA = tok.encode("a", addBOS: false).first!
        let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize) { _, _ in
            Self.logits([tok.endOfTurnID: 30, idA: 10])
        }
        let constraint = StubConstraint(endsAfter: 3)
        let outcome = try await runLoop(producer: producer, tokenizer: tok,
                                        config: GenerationConfig(maxNewTokens: 20,
                                                                 temperature: 0),
                                        constraint: constraint)
        #expect(outcome.emitted == [idA, idA, idA])
        #expect(outcome.result.reason == .endOfTurn)
        #expect(outcome.result.newTokens == 4)
        #expect(constraint.maskFills == 3, "the stop token was withheld \(constraint.maskFills) times")
        #expect(constraint.accepted == [idA, idA, idA])
    }

    /// An end-of-generation id is never run through the constraint: it is
    /// allowed iff the constraint may end here, on the probe and on the mask
    /// alike. The reference excludes eog ids from the candidate set entirely
    /// (`llama_grammar_apply_impl`) — they are decided by `allow_eog` and by
    /// nothing else.
    @Test func GEN_7_endOfGenerationBypassesTheConstraintOnProbeAndMask() throws {
        let eog: Set<Int32> = [1, 2]

        // Mid-structure: the stop ids are masked, the grammar's own id is not.
        let mid = GrammarLikeConstraint([5, 6])
        let midGate = ConstraintGate(constraint: mid, endOfGenerationTokenIDs: eog)
        #expect(midGate.allows(1) == false)
        #expect(midGate.allows(5) == true)
        #expect(midGate.allows(6) == false)

        // Complete: the stop ids are allowed even though the grammar itself
        // rejects everything, and the grammar's ids stay rejected.
        let done = GrammarLikeConstraint([])
        let doneGate = ConstraintGate(constraint: done, endOfGenerationTokenIDs: eog)
        #expect(doneGate.allows(1) == true)
        #expect(doneGate.allows(2) == true)
        #expect(doneGate.allows(5) == false)

        var midMask = [Bool](repeating: false, count: 8)
        try midMask.withUnsafeMutableBufferPointer { try midGate.fillAllowedMask($0) }
        #expect(midMask[1] == false)
        #expect(midMask[5] == true)

        var doneMask = [Bool](repeating: false, count: 8)
        try doneMask.withUnsafeMutableBufferPointer { try doneGate.fillAllowedMask($0) }
        #expect(doneMask[1] == true)
        #expect(doneMask[2] == true)
        #expect(doneMask[5] == false)
    }

    /// The stop token that ends a completed constraint is not handed to the
    /// constraint at all — a real one has no state left to advance and throws.
    /// (`llama_grammar_accept_impl` returns early for an eog token whenever a
    /// stack is empty.)
    @Test func GEN_7_endOfGenerationIsNotForwardedToTheConstraint() throws {
        let done = GrammarLikeConstraint([])
        let gate = ConstraintGate(constraint: done, endOfGenerationTokenIDs: [1, 2])
        try gate.accept(1)
        #expect(done.accepted.isEmpty)
    }

    /// End to end: a constraint that has been satisfied can terminate. Every
    /// non-stop id is rejected at that point, so the stop token is the only way
    /// out — and it has to be allowed, accepted, and reported as the reason.
    @Test func GEN_7_completedConstraintCanStop() async throws {
        let tok = try await GFTokenizer.load()
        let idA = tok.encode("a", addBOS: false).first!
        let idB = tok.encode("b", addBOS: false).first!
        let next: [Int32: Int32] = [idA: idB, idB: tok.endOfTurnID]
        let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize) { input, _ in
            .argmax(next[input] ?? idA)
        }
        let constraint = GrammarLikeConstraint([idA, idB])
        let outcome = try await runLoop(producer: producer, tokenizer: tok,
                                        config: GenerationConfig(maxNewTokens: 20,
                                                                 temperature: 0),
                                        constraint: constraint)
        #expect(outcome.emitted == [idA, idB])
        #expect(outcome.result.reason == .endOfTurn)
        #expect(outcome.result.newTokens == 3)
        #expect(constraint.accepted == [idA, idB])
    }

    /// The caller's own extra stop tokens follow the same rule.
    @Test func GEN_7_extraStopTokensAreWithheldTheSameWay() async throws {
        let tok = try await GFTokenizer.load()
        let idA = tok.encode("a", addBOS: false).first!
        let idB = tok.encode("b", addBOS: false).first!
        let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize) { _, _ in
            Self.logits([idB: 30, idA: 10])
        }
        let constraint = StubConstraint(endsAfter: 2)
        let outcome = try await runLoop(producer: producer, tokenizer: tok,
                                        config: GenerationConfig(maxNewTokens: 20,
                                                                 temperature: 0,
                                                                 extraStopTokens: [idB]),
                                        constraint: constraint)
        #expect(outcome.emitted == [idA, idA])
        #expect(outcome.result.reason == .eos)
        #expect(constraint.accepted == [idA, idA])
    }

    // MARK: - GEN-7: no token at all

    /// "No token is allowed" is an error, never a silent truncation and never a
    /// fallback to an unconstrained token.
    @Test func GEN_7_noAllowedTokenIsAnError() async throws {
        let tok = try await GFTokenizer.load()
        let idA = tok.encode("a", addBOS: false).first!
        let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize) { _, _ in
            .argmax(idA)
        }
        await #expect(throws: GenerationConstraintError.noAllowedToken(position: 0)) {
            _ = try await runLoop(producer: producer, tokenizer: tok,
                                  config: GenerationConfig(maxNewTokens: 4, temperature: 0),
                                  constraint: RejectEverythingConstraint())
        }
    }

    // MARK: - GEN-7: the producer has to supply logits

    /// A constraint cannot be applied to logits that were never written, so the
    /// fused-greedy shortcut is refused rather than silently taken.
    @Test func GEN_7_fusedGreedyProducerWithAConstraintIsRefused() async throws {
        let tok = try await GFTokenizer.load()
        do {
            _ = try await runLoop(producer: FusedGreedyOnlyProducer(), tokenizer: tok,
                                  config: GenerationConfig(maxNewTokens: 4, temperature: 0),
                                  constraint: StubConstraint(endsAfter: 0))
            Issue.record("the fused-greedy producer accepted a constraint")
        } catch let error as GenerationConstraintError {
            guard case .logitsUnavailable = error else {
                Issue.record("wrong refusal: \(error)")
                return
            }
        }
    }

    // MARK: - GEN-14

    // 拘束のある要求が投機デコードを使えることは
    // `SpeculativeCompletionLoopTests` が持つ (SPEC §6 GEN-14)。DEV-14 に
    // 残ったのは強制挿入 (RSN-4) と `repeat_penalty != 1` の 2 つだけで、
    // どちらも同じファイルの `repetitionPenaltyIsRefused` /
    // `ServerReasoningPlan` の側で検定してある。ここにあった
    // 「投機ループは拘束を断る」は、断らなくなったので消した。
}
