import Testing
import Foundation
import Metal
@testable import TurboFieldfare

/// M5's loop, checked without the kernel stack: a scripted target automaton
/// plus a scripted drafter (`docs/mtp/04-PHASES.md` M5).
///
/// The gate these exercise is gate 1 (04-PHASES §3): the speculative loop must
/// emit the tokens plain decode emits, whatever the drafter proposes. Here that
/// is a machine check — the same automaton is run through `runRawCompletion` and
/// through `runSpeculativeCompletion`, and the two token streams are compared —
/// with the drafter set to always right, always wrong, and half right.
@Suite struct SpeculativeCompletionLoopTests {

    /// A target whose next token is a fixed function of its input, plus a K/V
    /// cursor that enforces `verifyBlock`'s contract the way the real runner
    /// does. It is not a `RealForwardRunner`, so the loop takes the logits path
    /// and every token goes through the real sampler.
    final class ScriptedSpeculativeProducer: LogitProducer, ChunkedPrefillRunner,
                                             SpeculativeDrafting, @unchecked Sendable {
        let vocabSize: Int
        private let next: [Int32: Int32]
        private let firstToken: Int32
        private let propose: @Sendable (Int32, Int) -> [Int32]

        /// Simulated K/V cursor: what the real runner keeps in `KVCacheManager`.
        private(set) var cursor = 0
        private(set) var blocks: [[Int32]] = []
        private(set) var rewinds: [Int] = []
        private(set) var draftCalls: [(bonus: Int32, position: Int, count: Int)] = []

        var speculativeHiddenRows: MTLBuffer?
        var isDraftInstalled = true
        var speculativeHiddenRowStride: Int { 8 }
        var maxSpeculativeBlockTokens: Int { SpeculativeBlock.maxTokens }
        var draftStepCount = 0
        var draftNanos: UInt64 = 0
        var verifyBlockCount = 0
        var verifyBlockNanos: UInt64 = 0

        init(vocabSize: Int,
             next: [Int32: Int32],
             firstToken: Int32,
             propose: @escaping @Sendable (Int32, Int) -> [Int32]) {
            self.vocabSize = vocabSize
            self.next = next
            self.firstToken = firstToken
            self.propose = propose
        }

        func reset() {
            cursor = 0
            blocks = []
            rewinds = []
            draftCalls = []
        }

        private func writeArgmax(_ token: Int32, into buffer: MTLBuffer, row: Int) {
            let ptr = buffer.contents().bindMemory(to: Float16.self,
                                                   capacity: vocabSize * (row + 1))
            for i in 0..<vocabSize { ptr[row * vocabSize + i] = Float16(-30.0) }
            ptr[row * vocabSize + Int(token)] = Float16(30.0)
        }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            writeArgmax(next[token] ?? firstToken, into: logits, row: 0)
            cursor = position + 1
        }

        func prefillChunked(tokens: ArraySlice<Int32>,
                            startPosition: Int,
                            outputMode: PrefillOutputMode,
                            config: PrefillRuntimeConfig,
                            vision: VisionPrefillInput?,
                            into logits: MTLBuffer,
                            onProgress: (Int) -> Void) async throws -> PrefillResult {
            writeArgmax(firstToken, into: logits, row: 0)
            cursor = startPosition + tokens.count
            onProgress(tokens.count)
            return PrefillResult(newPosition: cursor, seed: .logitsWritten)
        }

        func verifyBlock(tokens: ArraySlice<Int32>,
                         startPosition: Int,
                         into logitRows: MTLBuffer,
                         greedyTokens: MTLBuffer?) async throws {
            #expect(startPosition == cursor)
            #expect(tokens.count <= maxSpeculativeBlockTokens)
            #expect(greedyTokens == nil)
            blocks.append(Array(tokens))
            for (row, token) in tokens.enumerated() {
                writeArgmax(next[token] ?? firstToken, into: logitRows, row: row)
            }
            cursor = startPosition + tokens.count
            verifyBlockCount += 1
        }

        func rewind(to position: Int) throws {
            #expect(position >= 0 && position <= cursor)
            rewinds.append(position)
            cursor = position
        }

        func draftProposals(bonusToken: Int32,
                            position: Int,
                            hidden: MTLBuffer,
                            hiddenRow: Int,
                            count: Int) throws -> [Int32] {
            #expect(position == cursor)
            draftCalls.append((bonusToken, position, count))
            draftStepCount += count
            return Array(propose(bonusToken, count).prefix(count))
        }
    }

    /// `seq` walked as an automaton, ending on `end`.
    private func automaton(_ seq: [Int32], end: Int32) -> [Int32: Int32] {
        var next: [Int32: Int32] = [:]
        for i in 0..<max(0, seq.count - 1) { next[seq[i]] = seq[i + 1] }
        if let last = seq.last { next[last] = end }
        return next
    }

    private struct Run {
        var tokens: [Int32] = []
        var text = ""
        var result: RawDecodeResult
        var stats: SpeculativeStats?
    }

    private func plainRun(seq: [Int32], end: Int32,
                          config: GenerationConfig) async throws -> Run {
        let ctx = try MetalContext()
        let tok = try await GFTokenizer.load()
        let next = automaton(seq, end: end)
        let producer = ScriptedSpeculativeProducer(vocabSize: tok.vocabSize,
                                                   next: next,
                                                   firstToken: seq[0]) { _, _ in [] }
        let promptIds = tok.encode("go", addBOS: true)
        let scratch = try RawCompletionScratch(context: ctx, vocab: tok.vocabSize)
        var run = Run(result: RawDecodeResult(prefillTokens: 0, cachedPromptTokens: 0,
                                              computedPrefillTokens: 0, prefillSeconds: 0,
                                              newTokens: 0, decodeSeconds: 0,
                                              timeToFirstTokenSeconds: 0, reason: .maxTokens,
                                              kvPosition: 0, kvBackedTokenIDs: [],
                                              uncommittedBoundaryTokenIDs: []))
        run.result = try await runRawCompletion(producer: producer, tokenizer: tok,
                                                promptIds: promptIds, config: config,
                                                context: ctx, scratch: scratch,
                                                prefillConfig: .defaultChunked) { progress in
            if case .token(_, let id, let delta) = progress {
                run.tokens.append(id)
                run.text += delta
            }
            if case .tail(let tail) = progress { run.text += tail }
        }
        return run
    }

    private func speculativeRun(seq: [Int32], end: Int32,
                                config: GenerationConfig,
                                blockTokens: Int,
                                propose: @escaping @Sendable (Int32, Int) -> [Int32])
        async throws -> (Run, ScriptedSpeculativeProducer) {
        let ctx = try MetalContext()
        let tok = try await GFTokenizer.load()
        let next = automaton(seq, end: end)
        let producer = ScriptedSpeculativeProducer(vocabSize: tok.vocabSize,
                                                   next: next,
                                                   firstToken: seq[0],
                                                   propose: propose)
        let promptIds = tok.encode("go", addBOS: true)
        let scratch = try RawCompletionScratch(context: ctx, vocab: tok.vocabSize)
        let speculative = try SpeculativeScratch(context: ctx,
                                                 vocab: tok.vocabSize,
                                                 hiddenSize: 8,
                                                 blockTokens: blockTokens,
                                                 fusedGreedy: false)
        var run = Run(result: RawDecodeResult(prefillTokens: 0, cachedPromptTokens: 0,
                                              computedPrefillTokens: 0, prefillSeconds: 0,
                                              newTokens: 0, decodeSeconds: 0,
                                              timeToFirstTokenSeconds: 0, reason: .maxTokens,
                                              kvPosition: 0, kvBackedTokenIDs: [],
                                              uncommittedBoundaryTokenIDs: []))
        let result = try await runSpeculativeCompletion(producer: producer, tokenizer: tok,
                                                        promptIds: promptIds, config: config,
                                                        context: ctx, scratch: scratch,
                                                        speculative: speculative,
                                                        prefillConfig: .defaultChunked) { progress in
            if case .token(_, let id, let delta) = progress {
                run.tokens.append(id)
                run.text += delta
            }
            if case .tail(let tail) = progress { run.text += tail }
        }
        run.result = result.decode
        run.stats = result.speculative
        return (run, producer)
    }

    /// The drafter that is always right: it walks the same automaton the target
    /// does, so every round should accept `bs - 1` drafts.
    private func perfectDrafter(_ next: [Int32: Int32],
                                firstToken: Int32) -> @Sendable (Int32, Int) -> [Int32] {
        { bonus, count in
            var out: [Int32] = []
            var token = bonus
            for _ in 0..<count {
                token = next[token] ?? firstToken
                out.append(token)
            }
            return out
        }
    }

    private let sequence: [Int32] = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    @Test("a perfect drafter changes the token count per round, not the tokens")
    func perfectDrafterMatchesPlainDecode() async throws {
        let config = GenerationConfig(maxNewTokens: 12, temperature: 0, seed: 1)
        let plain = try await plainRun(seq: sequence, end: 1, config: config)
        let next = automaton(sequence, end: 1)
        let (spec, producer) = try await speculativeRun(
            seq: sequence, end: 1, config: config, blockTokens: 4,
            propose: perfectDrafter(next, firstToken: sequence[0]))

        #expect(spec.tokens == plain.tokens)
        #expect(spec.text == plain.text)
        #expect(spec.result.newTokens == plain.result.newTokens)
        #expect(spec.result.reason == plain.result.reason)
        #expect(spec.result.kvPosition == plain.result.kvPosition)
        #expect(spec.result.kvBackedTokenIDs == plain.result.kvBackedTokenIDs)
        // Every proposal is accepted, so a round emits as many tokens as it
        // verified. The last round is narrower because the token budget caps it.
        let stats = try #require(spec.stats)
        #expect(stats.accepted == stats.proposed)
        #expect(stats.rounds * 4 >= spec.result.newTokens)
        #expect(producer.blocks.allSatisfy { $0.count <= 4 })
    }

    @Test("a drafter that is always wrong still produces the plain-decode text")
    func uselessDrafterMatchesPlainDecode() async throws {
        let config = GenerationConfig(maxNewTokens: 12, temperature: 0, seed: 1)
        let plain = try await plainRun(seq: sequence, end: 1, config: config)
        let (spec, _) = try await speculativeRun(
            seq: sequence, end: 1, config: config, blockTokens: 4,
            propose: { _, count in Array(repeating: Int32(9999), count: count) })

        #expect(spec.tokens == plain.tokens)
        #expect(spec.text == plain.text)
        #expect(spec.result.newTokens == plain.result.newTokens)
        #expect(spec.result.reason == plain.result.reason)
        #expect(spec.result.kvPosition == plain.result.kvPosition)
        let stats = try #require(spec.stats)
        #expect(stats.accepted == 0)
        // One emitted token per round when nothing is accepted.
        #expect(stats.rounds == spec.result.newTokens - 1 || stats.rounds == spec.result.newTokens)
    }

    @Test("a drafter that is right about its first proposal only")
    func partialDrafterMatchesPlainDecode() async throws {
        let config = GenerationConfig(maxNewTokens: 12, temperature: 0, seed: 1)
        let plain = try await plainRun(seq: sequence, end: 1, config: config)
        let next = automaton(sequence, end: 1)
        let (spec, _) = try await speculativeRun(
            seq: sequence, end: 1, config: config, blockTokens: 4,
            propose: { bonus, count in
                var out: [Int32] = [next[bonus] ?? 9999]
                out.append(contentsOf: Array(repeating: Int32(9999), count: max(0, count - 1)))
                return out
            })

        #expect(spec.tokens == plain.tokens)
        #expect(spec.result.kvBackedTokenIDs == plain.result.kvBackedTokenIDs)
        // Exactly the first proposal of each round survives, so no round ever
        // lands above 1 accepted draft.
        let stats = try #require(spec.stats)
        #expect(stats.acceptedHistogram[2...].allSatisfy { $0 == 0 })
        #expect(stats.accepted >= stats.rounds - 1)
    }

    @Test("the stop token ends the run even when a round speculated past it")
    func stopTokenInsideAnAcceptedRunTruncates() async throws {
        // The automaton walks 3 tokens and then hits the tokenizer's end-of-turn
        // id, which the drafter also predicts — so the block contains tokens
        // after the stop, and none of them may be emitted or committed.
        let tok = try await GFTokenizer.load()
        let end = tok.endOfTurnID
        let short: [Int32] = [11, 12, 13]
        let config = GenerationConfig(maxNewTokens: 32, temperature: 0, seed: 1)
        let plain = try await plainRun(seq: short, end: end, config: config)
        let next = automaton(short, end: end)
        let (spec, producer) = try await speculativeRun(
            seq: short, end: end, config: config, blockTokens: 4,
            propose: perfectDrafter(next, firstToken: short[0]))

        #expect(spec.tokens == plain.tokens)
        #expect(spec.result.reason == .endOfTurn)
        #expect(spec.result.newTokens == plain.result.newTokens)
        // The K/V ends where the text does: the stop token is uncommitted, and
        // so is everything the round ran past it.
        #expect(spec.result.kvPosition == plain.result.kvPosition)
        #expect(producer.cursor == spec.result.kvPosition)
        #expect(spec.result.uncommittedBoundaryTokenIDs == [end])
    }

    @Test("the token budget is never overrun by a round")
    func maxNewTokensIsExact() async throws {
        let next = automaton(sequence, end: 1)
        for maxNew in [1, 2, 3, 5, 7] {
            let config = GenerationConfig(maxNewTokens: maxNew, temperature: 0, seed: 1)
            let (spec, producer) = try await speculativeRun(
                seq: sequence, end: 1, config: config, blockTokens: 4,
                propose: perfectDrafter(next, firstToken: sequence[0]))
            #expect(spec.result.newTokens == maxNew)
            #expect(spec.result.reason == .maxTokens)
            #expect(producer.cursor == spec.result.kvPosition)
        }
    }

    @Test("a repetition penalty is refused rather than mis-verified")
    func repetitionPenaltyIsRefused() async throws {
        let config = GenerationConfig(maxNewTokens: 4, temperature: 0,
                                      repetitionPenalty: 1.1, seed: 1)
        await #expect(throws: SpeculativeDraftError.self) {
            _ = try await speculativeRun(seq: sequence, end: 1, config: config,
                                         blockTokens: 4, propose: { _, count in
                Array(repeating: Int32(9999), count: count)
            })
        }
    }
}
