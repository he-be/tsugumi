import Foundation
import Metal

/// Everything the decode loop needs from the phase before it: the validated
/// configuration, the prompt in the KV cache, and the seed the first token is
/// drawn from.
///
/// Factored out because there are two decode loops over the same front half —
/// `runRawCompletion` and `runSpeculativeCompletion` (`docs/mtp/03-DESIGN.md`
/// D6). Sharing the phase is what keeps "MTP off" and "MTP on" answering the
/// same request: one prompt path, one prefill, one seed rule.
struct PreparedGeneration {
    let fusedRunner: RealForwardRunner?
    let fusedGreedy: Bool
    let cachedPromptTokens: Int
    let computedPrefillTokens: Int
    /// Absolute KV position after prefill: where the first generated token goes.
    let position: Int
    let prefillSeed: PrefillSeed?
    let history: [Int32]
    let prefillStart: Date
    let decodeStart: Date
    let prefillSeconds: Double
}

/// Validate, reset or resume, and prefill the prompt. Byte-for-byte the front
/// half `runRawCompletion` had before M5 split it out.
func prepareGeneration(producer: any LogitProducer,
                       promptIds: [Int32],
                       config: GenerationConfig,
                       scratch: RawCompletionScratch,
                       prefillConfig: PrefillRuntimeConfig,
                       vision: VisionPrefillInput?,
                       start: RawCompletionStart,
                       onProgress: (RawDecodeProgress) -> Void) async throws -> PreparedGeneration {
    try config.validate()
    guard !promptIds.isEmpty else {
        throw GeneratorError.emptyPrompt
    }
    let fusedRunner = producer as? RealForwardRunner
    let fusedGreedy = fusedRunner?.usesFusedGreedyHead == true
    guard !fusedGreedy || config.isPureGreedy else {
        throw PrefillError.unsupportedPrefillSeed(
            "the fused-head producer cannot serve this sampling configuration; use a logits head")
    }

    let cachedPromptTokens: Int
    switch start {
    case .reset:
        cachedPromptTokens = 0
    case .resume(let count):
        guard count > 0, count < promptIds.count else {
            throw GeneratorError.invalidContinuation(
                "cached prompt token count must be greater than zero and less than the effective prompt")
        }
        // Image spans are offsets into the slice that is actually prefilled, so
        // a served prefix would shift every one of them. The prompt cache is
        // disabled for image requests upstream of here (PLAN_VISION §4-6); this
        // is the check that makes that a contract rather than a convention.
        guard vision == nil else {
            throw GeneratorError.invalidContinuation(
                "a cached prompt prefix cannot be combined with attached images")
        }
        guard producer is any ContinuableLogitProducer else {
            throw GeneratorError.invalidContinuation(
                "producer does not support continuation")
        }
        cachedPromptTokens = count
    }
    let computedPrefillTokens = promptIds.count - cachedPromptTokens

    var history = Array(promptIds.prefix(cachedPromptTokens))
    history.reserveCapacity(promptIds.count + config.maxNewTokens)

    if let context = producer as? any ContextWindowReporting,
       promptIds.count + config.maxNewTokens > context.maxContext {
        throw GeneratorError.contextOverflow(prompt: promptIds.count,
                                             maxNew: config.maxNewTokens,
                                             maxContext: context.maxContext)
    }
    switch start {
    case .reset:
        producer.reset()
    case .resume:
        let continuable = producer as! any ContinuableLogitProducer
        try continuable.prepareForContinuation(expectedPosition: cachedPromptTokens)
    }
    let prefillStart = Date()
    var position = cachedPromptTokens
    var prefillSeed: PrefillSeed?
    let prefillTokens = promptIds[cachedPromptTokens...]
    switch prefillConfig.mode {
    case .chunked where producer is any ChunkedPrefillRunner:
        let chunked = producer as! any ChunkedPrefillRunner
        let mode: PrefillOutputMode = fusedGreedy ? .greedyIfAvailable : .logits
        let result = try await chunked.prefillChunked(tokens: prefillTokens,
                                                      startPosition: position,
                                                      outputMode: mode,
                                                      config: prefillConfig,
                                                      vision: vision,
                                                      into: scratch.logits) { done in
            onProgress(.prefill(done: cachedPromptTokens + done, total: promptIds.count))
        }
        if mode == .logits, result.seed != .logitsWritten {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion chunked prefill requested logits but producer returned \(result.seed)")
        }
        if case .greedyToken = result.seed, !config.isPureGreedy {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion chunked prefill returned a greedy token for a sampling config")
        }
        position = result.newPosition
        prefillSeed = result.seed
        history.append(contentsOf: prefillTokens)
    case .chunked:
        throw PrefillError.chunkedUnsupported(
            PrefillError.chunkedRequiresChunkedRunnerReason)
    case .off:
        // The scalar replay path has nowhere to put a soft token: it embeds one
        // token at a time through the text lookup. Refusing is the only honest
        // answer — running it would answer about an image it never saw.
        guard vision == nil else {
            throw PrefillError.chunkedUnsupported(
                "images require chunked prefill; run with --prefill on")
        }
        for t in prefillTokens {
            try Task.checkCancellation()
            try await producer.produce(token: t, position: position, into: scratch.logits)
            position += 1
            history.append(t)
            onProgress(.prefill(done: position, total: promptIds.count))
        }
    }

    let decodeStart = Date()
    return PreparedGeneration(fusedRunner: fusedRunner,
                              fusedGreedy: fusedGreedy,
                              cachedPromptTokens: cachedPromptTokens,
                              computedPrefillTokens: computedPrefillTokens,
                              position: position,
                              prefillSeed: prefillSeed,
                              history: history,
                              prefillStart: prefillStart,
                              decodeStart: decodeStart,
                              prefillSeconds: decodeStart.timeIntervalSince(prefillStart))
}

/// One draw from the logits buffer, with the acceptance probe's hook. Shared by
/// both loops so a sampled token is produced identically either way (D5 leans
/// on that: same `position`, same history, same config ⇒ same token).
func sampleOnce(scratch: RawCompletionScratch, context: MetalContext,
                history: [Int32], config: GenerationConfig, position: Int) throws -> Int32 {
    let cb = context.queue.makeCommandBuffer()!
    scratch.sampler.sample(commandBuffer: cb, logits: scratch.logits, probs: scratch.probs,
                           history: history, config: config, position: position,
                           outToken: scratch.outToken)
    cb.commit(); cb.waitUntilCompleted()
    try checkCommandBufferError(cb.error)
    let token = Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
    AcceptanceProbe.shared?.record(step: position, drawn: token,
                                   probs: scratch.probs,
                                   vocab: scratch.sampler.vocab, config: config)
    return token
}
