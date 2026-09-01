import Metal
import Testing
@testable import Tsugumi
import TsugumiValidationSupport

/// GEN-7 says an empty mask is an error. It does **not** say that an allowed
/// set the model happens to think is unlikely is one.
///
/// `QwenSampler` keeps its probabilities as FP16 over the whole vocabulary, so
/// anything under `2^-24` is stored as zero. The masked renormalization used to
/// divide by the sum of the allowed probabilities and call a zero sum "the
/// constraint allows no token" — which is how a live grammar, sitting on ids the
/// row had rounded away, became a 500 mid-answer. The other family never had
/// this: `Sampler.maskedSoftmax` renormalizes from the *logits* with the
/// maximum over the allowed ids, so its denominator is at least 1.
///
/// These pin the two halves of the claim: an underflowed allowed set is drawn
/// from the logits, and only a genuinely empty mask is still an error.
@Suite("Qwen masked draw under FP16 underflow")
struct QwenSamplerMaskedUnderflowTests {
    private static let vocab = 512
    /// Far enough below the row maximum that `exp(-40)` is zero in FP16 — the
    /// state the failure needs — while staying an ordinary logit.
    private static let buried: Float = -40

    /// Only these are allowed, and `high` outranks `low` by logit. Both are
    /// zero in the FP16 probabilities.
    private static let low: Int32 = 400
    private static let high: Int32 = 401

    /// Allows exactly `allowed`, and never ends.
    private final class FixedConstraint: GenerationConstraint, @unchecked Sendable {
        private let allowed: Set<Int32>
        init(_ allowed: Set<Int32>) { self.allowed = allowed }

        var mayEndHere: Bool { false }
        func allows(tokenID: Int32) -> Bool { allowed.contains(tokenID) }
        func fillAllowedMask(_ mask: UnsafeMutableBufferPointer<Bool>) throws {
            for i in 0..<mask.count { mask[i] = allowed.contains(Int32(i)) }
        }
        func accept(tokenID: Int32) throws {}
    }

    /// One row whose mass sits on id 0 and whose two allowed ids are buried.
    private static func buriedLogits() -> [Float] {
        var values = [Float](repeating: buried, count: vocab)
        values[0] = 0
        values[Int(low)] = buried
        values[Int(high)] = buried + 2
        return values
    }

    private static func sampler(_ context: MetalContext, rows: [[Float]]) throws -> QwenSampler {
        let sampler = try QwenSampler(context: context, vocab: vocab, maxRows: rows.count)
        guard let logits = Fp16Buffer.make(context.device, values: rows.flatMap { $0 }),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.noDevice
        }
        sampler.encodeProbabilities(commandBuffer: commandBuffer, logits: logits, rows: rows.count)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return sampler
    }

    private static func gate(allowing allowed: Set<Int32>) -> ConstraintGate {
        ConstraintGate(constraint: FixedConstraint(allowed), endOfGenerationTokenIDs: [])
    }

    private static let sampled = GenerationConfig(maxNewTokens: 1, temperature: 0.6,
                                                  topK: 20, topP: 0.95, seed: 1)
    private static let greedy = GenerationConfig(maxNewTokens: 1, temperature: 0,
                                                 topK: 20, topP: 0.95, seed: 1)

    /// The row rounded both allowed ids to zero. The draw is still the allowed
    /// pair, ranked the way the logits rank them.
    @Test func anUnderflowedAllowedSetIsDrawnFromTheLogits() throws {
        let context = try MetalContext()
        let sampler = try Self.sampler(context, rows: [Self.buriedLogits()])

        // The premise: nothing is left of these two in the FP16 row.
        let drawn = try sampler.categorical(row: 0, config: Self.sampled,
                                            gate: Self.gate(allowing: [Self.low, Self.high]),
                                            position: 7)
        #expect(!drawn.isEmpty)
        #expect(Set(drawn.ids) == [Self.low, Self.high])
        #expect(drawn.ids.first == Self.high, "the larger logit has to come first")
    }

    /// Greedy used to answer with the *lowest* allowed id, because every allowed
    /// probability tied at zero. The logits break the tie.
    @Test func greedyUnderUnderflowPicksTheLargestAllowedLogit() throws {
        let context = try MetalContext()
        let sampler = try Self.sampler(context, rows: [Self.buriedLogits()])

        let drawn = try sampler.categorical(row: 0, config: Self.greedy,
                                            gate: Self.gate(allowing: [Self.low, Self.high]),
                                            position: 7)
        #expect(drawn.ids == [Self.high])
    }

    /// The half GEN-7 still owns: no allowed id at all is an error, and it
    /// names the position the caller passed — not the sampling row, which is
    /// what it used to report.
    @Test func anEmptyMaskIsStillAnErrorAndNamesTheGenerationPosition() throws {
        let context = try MetalContext()
        // Row 2 on purpose: the draft row is the one whose number used to be
        // reported as a generation position.
        let rows = [Self.buriedLogits(), Self.buriedLogits(), Self.buriedLogits()]
        let sampler = try Self.sampler(context, rows: rows)

        #expect(throws: GenerationConstraintError.noAllowedToken(position: 7)) {
            _ = try sampler.categorical(row: 2, config: Self.sampled,
                                        gate: Self.gate(allowing: []),
                                        position: 7)
        }
        #expect(throws: GenerationConstraintError.noAllowedToken(position: 7)) {
            _ = try sampler.categorical(row: 2, config: Self.greedy,
                                        gate: Self.gate(allowing: []),
                                        position: 7)
        }
    }

    /// The fallback is reached only when there is nothing to renormalize: a row
    /// that leaves the allowed ids any mass at all keeps the FP16 path.
    @Test func aRowWithAllowedMassIsUnchanged() throws {
        let context = try MetalContext()
        var values = [Float](repeating: Self.buried, count: Self.vocab)
        values[0] = 0
        values[Int(Self.low)] = -1
        values[Int(Self.high)] = -2
        let sampler = try Self.sampler(context, rows: [values])

        let drawn = try sampler.categorical(row: 0, config: Self.sampled,
                                            gate: Self.gate(allowing: [Self.low, Self.high]),
                                            position: 7)
        #expect(drawn.ids == [Self.low, Self.high], "the larger logit still comes first")
        #expect(drawn.weights.count == 2)
    }
}
