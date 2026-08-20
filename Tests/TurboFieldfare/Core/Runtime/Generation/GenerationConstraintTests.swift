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

    /// `accept` sees exactly the tokens the loop kept, in order, including the
    /// stop token that ends generation.
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
        #expect(constraint.accepted == [idA, idB, tok.endOfTurnID])
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
        #expect(constraint.accepted == [idA, idA, idA, tok.endOfTurnID])
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
        #expect(constraint.accepted == [idA, idA, idB])
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

    // MARK: - DEV-14

    /// A request with a constraint does not use speculative decoding: the
    /// speculative entry point refuses it before it looks at anything else.
    @Test func DEV_14_speculativeCompletionRefusesAConstraint() async throws {
        let tok = try await GFTokenizer.load()
        let ctx = try MetalContext()
        let idA = tok.encode("a", addBOS: false).first!
        // Not a drafting producer: if the refusal were not first, this would
        // fail with `notInstalled` / "cannot draft" instead.
        let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize) { _, _ in .argmax(idA) }
        let scratch = try RawCompletionScratch(context: ctx, vocab: tok.vocabSize)
        let speculative = try SpeculativeScratch(context: ctx, vocab: tok.vocabSize,
                                                 hiddenSize: 8, blockTokens: 2,
                                                 fusedGreedy: false)
        do {
            _ = try await runSpeculativeCompletion(producer: producer,
                                                   tokenizer: tok,
                                                   promptIds: tok.encode("go", addBOS: true),
                                                   config: GenerationConfig(maxNewTokens: 4,
                                                                            temperature: 0),
                                                   constraint: StubConstraint(endsAfter: 0),
                                                   context: ctx,
                                                   scratch: scratch,
                                                   speculative: speculative) { _ in }
            Issue.record("the speculative loop accepted a constraint")
        } catch let error as SpeculativeDraftError {
            guard case .unsupportedConfig(let reason) = error else {
                Issue.record("wrong refusal: \(error)")
                return
            }
            #expect(reason.contains("DEV-14"))
        }
    }
}
