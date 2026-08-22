import Foundation
import TurboFieldfare

/// SPEC's server over the Ornith (Qwen 3.5-MoE) family — Phase 8
/// (`docs/qwen35moe/04-PHASES.md`).
///
/// `ServerModelSession`'s sibling, not a branch of it, for the reason every
/// other Ornith type is one: nothing below the tokenizer is shared. The
/// producer is `QwenForwardRunner` (serial, greedy, its own K/V *and* a
/// recurrent state — and, with `--draft-block-size 2`, its own width-2 MTP
/// loop), the framing is the checkpoint's own ChatML template, and the vision
/// assembler and the sampler have no Ornith side yet. What **is** shared is everything above the backend: the HTTP
/// layer, the request validator, the queue, the timings and the response
/// shapes all speak `ServerInferenceBackend` and never ask which family
/// answered.
///
/// Four things this backend cannot do, each refused or recorded rather than
/// silently dropped:
///
/// | | why | what happens |
/// | --- | --- | --- |
/// | images | Phase 9 | 400 `unsupported_image` |
/// | speculative decoding wider than one draft | the head drafts one token and the verify pass carries two rows (`docs/qwen35moe/36-MTP-DECODE.md`) | `--draft-block-size 2` is the only width accepted; 3...8 are refused at startup |
/// | prompt cache | a recurrent state cannot be rewound ([03](../../../docs/qwen35moe/03-DESIGN.md) §5) | every request computes the whole prompt; `cached_tokens` is 0 |
/// | sampling | the fused head writes no logits ([19](../../../docs/qwen35moe/19-LM-HEAD-INT8.md)) | greedy, with the sampler the client asked for named in `approximations` (R3) |
public actor QwenServerSession: ServerInferenceBackend {
    private let context: MetalContext
    private let model: Model
    private let tokenizer: QwenTokenizer
    private let runner: QwenForwardRunner
    private let prefillChunkTokens: Int
    private let maxContext: Int
    /// Whether `--draft-block-size 2` put the MTP head in the decode loop.
    private let speculative: Bool
    /// RSN-1 / FLAG-1, and RSN-3.
    private let reasoningBudget: Int
    private let reasoningFormat: ReasoningFormat
    /// SPEC §6 GEN-1: the id → piece table every grammar constraint matches
    /// against, built once beside the tokenizer for the same reason the Gemma
    /// session builds it once — 248 070 ids cost more to walk than the
    /// generation they constrain.
    private let grammarVocabulary: GrammarVocabulary
    private let markers: QwenToolCallMarkers
    private var accumulatedMetrics = ServerMetricsSnapshot.zero

    public static func load(modelDirectory: URL,
                            maxContext: Int,
                            runtimeConfiguration: RuntimeConfiguration,
                            integrityPolicy: ModelIntegrityPolicy = .fullSha256,
                            draftBlockSize: Int = 0,
                            mtpHeadDirectory: String = QwenMTPSidecar.defaultDirectory,
                            reasoningBudget: Int = -1,
                            reasoningFormat: ReasoningFormat = .auto) async throws -> QwenServerSession {
        try validateFlags(draftBlockSize: draftBlockSize)
        let tokenizer = try await QwenTokenizer.load(forModelDirectory: modelDirectory)
        let context = try MetalContext()
        let runtime = runtimeConfiguration
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            expecting: .ornith1_5_35B_A3B,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: integrityPolicy)
        let runner = try QwenForwardRunner(model: model,
                                           context: context,
                                           maxContext: maxContext)
        // The head is a 480 MB sidecar beside the checkpoint, not a section of
        // the `.gturbo` (`docs/qwen35moe/30-MTP-HEAD-GRAFT.md` §6), so it is
        // loaded here rather than by `Model.load` — and loaded at startup, so a
        // server told to speculate without one fails to start instead of
        // failing the first completion.
        if draftBlockSize != 0 { try runner.attachMTPHead(directory: mtpHeadDirectory) }
        return QwenServerSession(context: context,
                                 model: model,
                                 tokenizer: tokenizer,
                                 runner: runner,
                                 prefillChunkTokens: runtime.prefillChunkTokens,
                                 maxContext: maxContext,
                                 speculative: draftBlockSize != 0,
                                 reasoningBudget: reasoningBudget,
                                 reasoningFormat: reasoningFormat)
    }

    /// The flags this family cannot honour, refused before anything is loaded.
    ///
    /// Called twice on purpose: once from `main` while the arguments are being
    /// resolved, so an unusable combination exits with usage before the
    /// listener opens, and once from `load` — which is the contract for any
    /// other caller and the reason the message exists in one place.
    ///
    /// Refused rather than ignored, exactly as the Gemma session refuses a
    /// drafter it does not have: a server told to speculate over a family with
    /// no speculative path is misconfigured, and finding that out on the first
    /// completion is worse than not starting.
    ///
    /// **Two is the only width.** Gemma's drafter proposes a block of `n`
    /// tokens; this family's MTP head proposes exactly one and the verify pass
    /// carries the two rows that follow from it, so a request for 4 is not a
    /// slower version of what this backend does — it is a shape it has no
    /// kernels for (`docs/qwen35moe/33-MTP-ACCEPTANCE.md` §3-4: the verify cost
    /// grows with the union of the experts the rows route to, and four rows
    /// lose on paper before any of it is written).
    public static func validateFlags(draftBlockSize: Int) throws {
        guard draftBlockSize == 0 || draftBlockSize == 2 else {
            throw ServerArgumentError.invalid(
                "--draft-block-size \(draftBlockSize) has no Ornith path: this "
                + "family drafts one token a pass, so the width is 2 "
                + "(docs/qwen35moe/36-MTP-DECODE.md); run with "
                + "--draft-block-size 2 or 0")
        }
    }

    private init(context: MetalContext,
                 model: Model,
                 tokenizer: QwenTokenizer,
                 runner: QwenForwardRunner,
                 prefillChunkTokens: Int,
                 maxContext: Int,
                 speculative: Bool,
                 reasoningBudget: Int,
                 reasoningFormat: ReasoningFormat) {
        self.context = context
        self.model = model
        self.tokenizer = tokenizer
        self.runner = runner
        self.prefillChunkTokens = prefillChunkTokens
        self.maxContext = maxContext
        self.speculative = speculative
        self.reasoningBudget = reasoningBudget
        self.reasoningFormat = reasoningFormat
        self.grammarVocabulary = GrammarVocabulary.shared(for: tokenizer)
        self.markers = QwenToolCallMarkers(tokenizer: tokenizer)
    }

    // MARK: - Prompt

    /// The checkpoint's own `chat_template.jinja`, rendered and encoded.
    ///
    /// There is no server variant here and no INV-1 redraw: Gemma's server owns
    /// its framing because upstream ships none, whereas this checkpoint ships
    /// the template that wrote every assistant turn the model has ever seen —
    /// including the XML of a finished tool call, which `QwenToolCallGrammar`
    /// accepts and `QwenToolCallParser` reads back
    /// (`docs/qwen35moe/23-PHASE5-TOOLS.md`). Redrawing it ourselves would be a
    /// second answer to a question the checkpoint already answered.
    private func renderPrompt(_ request: ValidatedChatRequest) throws -> [Int32] {
        guard request.vision == nil else {
            throw ServerRequestError.invalid(
                message: "this model has no vision path yet "
                    + "(docs/qwen35moe/04-PHASES.md Phase 9)",
                param: "messages",
                code: "unsupported_image")
        }
        return try tokenizer.applyChatTemplate(request.messages,
                                               tools: request.tools,
                                               enableThinking: request.enableThinking)
    }

    public func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        ServerPreparedRequest(request: request, promptIDs: try renderPrompt(request), vision: nil)
    }

    // MARK: - Generation

    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        try await generate(try await prepare(request), monitor: nil, onEvent: onEvent)
    }

    public func generate(
        _ prepared: ServerPreparedRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        try await generate(prepared, monitor: nil, onEvent: onEvent)
    }

    public func generate(
        _ prepared: ServerPreparedRequest,
        monitor: ServerTimingsMonitor?,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let request = prepared.request
        // CACHE: there is nothing to invalidate and nothing to keep. The
        // recurrent state of the thirty linear layers cannot be rewound or
        // shortened (03-DESIGN §5), so a conversation is replayed from token
        // zero every turn and the runner starts each request clean — including
        // one that threw half way through the last.
        runner.reset()
        let promptIDs = try prepared.promptIDs ?? renderPrompt(request)
        guard !promptIDs.isEmpty else {
            throw ServerRequestError.invalid(message: "the rendered prompt is empty",
                                             param: "messages",
                                             code: "invalid_message")
        }
        guard promptIDs.count < maxContext else {
            throw ServerRequestError.exceedContextSize(
                message: "prompt of \(promptIDs.count) tokens reaches the "
                    + "configured context of \(maxContext)",
                param: "messages",
                code: "context_length_exceeded")
        }
        let plan = QwenGenerationPlan(request: request, markers: markers)

        // REQ-max-tokens: -1 is everything the context has left, 0 is prefill
        // only, and every other value is still bounded by the context.
        // The speculative loop runs a row for the drafted token before it
        // knows whether the token is real, so it asserts on one position more
        // than it will emit. Giving it the whole context would turn a
        // `max_tokens: -1` request into a trap.
        let contextRemaining = maxContext - promptIDs.count - (speculative ? 1 : 0)
        let maxNewTokens = request.maximumCompletionTokens < 0
            ? contextRemaining
            : min(request.maximumCompletionTokens, contextRemaining)
        if request.maximumCompletionTokens == 0 {
            return ServerCompletion(
                content: "",
                toolCalls: [],
                finishReason: "length",
                usage: OpenAIUsage(promptTokens: promptIDs.count,
                                   completionTokens: 0,
                                   totalTokens: promptIDs.count,
                                   cachedTokens: 0),
                approximations: plan.approximations,
                timings: ServerTimings(cacheTokens: 0,
                                       promptTokens: promptIDs.count,
                                       promptMilliseconds: 0,
                                       predictedTokens: 0,
                                       predictedMilliseconds: 0))
        }

        // SPEC §8. Only RSN-3's routing is honoured here: RSN-4 places a
        // closing tag without drawing it, and this loop has no forcer, so a
        // request that asked for a bounded thought block is told the bound was
        // not applied rather than being quietly given one.
        let reasoning = ServerReasoningPlan(request: request,
                                            defaultBudget: reasoningBudget,
                                            defaultFormat: reasoningFormat,
                                            maxNewTokens: maxNewTokens,
                                            contextRemaining: contextRemaining,
                                            forcedTokenCount: 1)
        var approximations = plan.approximations
        if reasoning.forcesClosingTag {
            approximations.append(
                "reasoning/budget-not-enforced: this family has no forced closing tag yet "
                + "(docs/qwen35moe/04-PHASES.md Phase 8)")
        }

        // GEN-1 / GEN-3 / GEN-5. Built after the prefill-only exit for the
        // reason the Gemma session builds it there: parsing a grammar is real
        // work and that request samples nothing.
        let constraint: GrammarTokenConstraint?
        if let grammarText = plan.grammar {
            do {
                constraint = try GrammarTokenConstraint(
                    grammarText,
                    vocabulary: grammarVocabulary,
                    trigger: plan.trigger.map { .token($0.tokenID) })
            } catch {
                throw ServerGrammarBuildFailure(shape: plan.shape, underlying: error)
            }
        } else {
            constraint = nil
        }

        // Two producers, one channel rule (`QwenReasoningSplitter`'s header).
        // With no tools declared a `<tool_call>` the model writes unasked is
        // text; with tools it is a call, and the decoder collects its token ids
        // and parses them once.
        let startsInsideReasoning = QwenStructuredAssistantDecoder
            .promptEndsInsideReasoning(promptIDs, tokenizer: tokenizer)
        // `emitsReasoning: true` unconditionally, and not `enableThinking`:
        // with it false the decoder **drops** thought text instead of routing
        // it, so a `<think>` block the model opened on a thinking-off request
        // would vanish from the answer with nothing said. The splitter arm
        // cannot drop it either, so this keeps the two arms telling the same
        // story. Where the text ends up is RSN-3's question, and
        // `ServerReasoningPlan.route` is the one place that answers it.
        let decoder = request.tools.isEmpty
            ? nil
            : QwenStructuredAssistantDecoder(tokenizer: tokenizer,
                                             tools: request.tools,
                                             emitsReasoning: true,
                                             startsInReasoning: startsInsideReasoning)
        var splitter = request.tools.isEmpty
            ? QwenReasoningSplitter(tokenizer: tokenizer,
                                    startsInsideReasoning: startsInsideReasoning)
            : nil
        var detokenizer = tokenizer.makeDetokenizer(skipSpecialTokens: true)
        var stopMatcher = StreamingStopMatcher(stops: request.generationConfig.stopStrings)

        var content = ""
        var reasoningContent = ""
        var calls: [ParsedToolCall] = []
        var shouldStop = false
        var live: ServerLiveTimings?
        if monitor != nil {
            live = ServerLiveTimings(cacheTokens: 0,
                                     promptTokens: promptIDs.count,
                                     startedAt: Date())
        }

        func handle(_ events: [StructuredAssistantEvent]) {
            for event in events {
                switch reasoning.route(event) {
                case .content(let text):
                    let visible = stopMatcher.push(text)
                    if !visible.isEmpty {
                        content += visible
                        onEvent(.content(visible))
                    }
                    if stopMatcher.isStopped { shouldStop = true }
                case .reasoning(let text):
                    // Stop strings match the answer, not the thinking: a
                    // client's stop sequence is about the text it will show.
                    reasoningContent += text
                    onEvent(.reasoning(text))
                case .toolCall(let call):
                    calls.append(call)
                    onEvent(.toolCall(call))
                }
            }
        }

        let run: QwenGreedyRun
        var speculativeStats: QwenSpeculativeStats?
        do {
            // One callback, two loops. The MTP loop's contract is the
            // non-speculative one — same tokens out, same stop rule, same
            // constraint applied in the same order (`36-MTP-DECODE.md` §3,
            // `40-MTP-GRAMMAR.md` §2) — so the request-shaped work above and
            // below it does not branch on which one ran.
            let onToken: (Int, Int32) throws -> Void = { index, id in
                    let delta = detokenizer.push(id)
                    if let monitor, let timings = live?.observe(
                        .token(index: index, id: id, delta: delta), at: Date()) {
                        monitor.record(timings)
                    }
                    let events: [StructuredAssistantEvent]
                    if let decoder {
                        events = try decoder.consume(tokenID: id, delta: delta)
                        // GEN-6, and the reason it is set here rather than in
                        // the runner: the decoder owns the channel state, and
                        // the state *this* token leaves behind is the one the
                        // next draw is judged by. A lazy grammar that armed
                        // inside the thought block would constrain reasoning
                        // the model is allowed to write freely.
                        constraint?.setSuppressed(decoder.isInsideReasoning)
                    } else {
                        events = splitter!.consume(tokenID: id, delta: delta)
                    }
                    handle(events)
            }
            if speculative {
                run = try runner.runGreedyCompletionMTP(
                    promptTokens: promptIDs,
                    maxNewTokens: maxNewTokens,
                    chunkWidth: prefillChunkTokens,
                    stopTokens: tokenizer.stopTokenIDs,
                    constraint: constraint,
                    shouldStop: { shouldStop },
                    onToken: onToken,
                    onStats: { speculativeStats = $0 })
            } else {
                run = try runner.runGreedyCompletion(
                    promptTokens: promptIDs,
                    maxNewTokens: maxNewTokens,
                    chunkWidth: prefillChunkTokens,
                    stopTokens: tokenizer.stopTokenIDs,
                    constraint: constraint,
                    shouldStop: { shouldStop },
                    onToken: onToken)
            }
        } catch let error as GenerationConstraintError {
            // GEN-7 calls a no-allowed-token state an error rather than a stop,
            // so it escapes as a 500 instead of becoming a truncated answer
            // under a constrained request.
            throw ServerGrammarBuildFailure(shape: plan.shape, underlying: error)
        }

        let tail = detokenizer.flush()
        if let decoder {
            do {
                handle(try decoder.consumeTail(tail))
                // Refuses a call the model left half written rather than
                // reporting the ones before it as if the turn were complete.
                try decoder.finish()
            } catch {
                throw QwenStructuredOutputFailure(shape: plan.shape,
                                                  promptTokens: promptIDs.count,
                                                  newTokens: run.newTokens,
                                                  decodedCalls: calls.count,
                                                  visibleBytes: content.utf8.count,
                                                  underlying: error)
            }
        } else {
            handle(splitter!.consumeTail(tail))
        }
        let held = stopMatcher.finish()
        if !held.isEmpty {
            content += held
            onEvent(.content(held))
        }

        let reason: String
        if !calls.isEmpty {
            reason = "tool_calls"
        } else if run.reason == .maxTokens {
            reason = "length"
        } else {
            reason = "stop"
        }
        let timings = ServerTimings(cacheTokens: 0,
                                    promptTokens: run.promptTokens,
                                    promptMilliseconds: run.prefillSeconds * 1_000,
                                    predictedTokens: run.newTokens,
                                    predictedMilliseconds: run.decodeSeconds * 1_000)
        accumulatedMetrics = accumulatedMetrics.adding(timings)
        return ServerCompletion(
            content: content,
            toolCalls: calls,
            finishReason: reason,
            usage: OpenAIUsage(promptTokens: run.promptTokens,
                               completionTokens: run.newTokens,
                               totalTokens: run.promptTokens + run.newTokens,
                               cachedTokens: 0),
            speculative: speculativeStats.map {
                // RSP-3's vocabulary over this family's shape: a round is one
                // verify pass, and each pass proposes exactly one token.
                ServerSpeculativeSummary(blockTokens: 2,
                                         rounds: $0.passes,
                                         proposed: $0.passes,
                                         accepted: $0.accepted)
            },
            reasoningContent: reasoningContent,
            approximations: approximations,
            timings: timings)
    }

    // MARK: - EP-5 / EP-6

    public func tokenize(_ text: String, addSpecial: Bool) -> [Int32] {
        // There is no BOS to add: this checkpoint declares none
        // (`tokenizer_config.json`, `add_bos_token: false`), so `add_special`
        // has nothing to switch on and the two answers are the same one.
        tokenizer.encode(text)
    }

    public func detokenize(_ tokens: [Int32]) -> String {
        tokenizer.decode(tokens, skipSpecialTokens: false)
    }

    /// EP-5 `/apply-template`: the prompt as text.
    ///
    /// Rendered and encoded by the same call generation uses, then decoded
    /// back — so what a client reads here is the token sequence the model would
    /// actually be prefilled with, and not a second rendering that could drift
    /// from it. The round trip is exact: the decoder is the encoder's inverse
    /// on any id sequence, specials included.
    public func applyChatTemplate(_ request: ValidatedChatRequest) throws -> String {
        tokenizer.decode(try renderPrompt(request), skipSpecialTokens: false)
    }

    public func metrics() -> ServerMetricsSnapshot { accumulatedMetrics }
}

/// The Ornith half of `StructuredOutputFailure`: a tool call the decoder could
/// not finish reading.
///
/// It escapes untyped into a 500 `server_error`, and exists so the stderr line
/// names the request shape and the counts rather than the conversation. Never
/// prompt text.
struct QwenStructuredOutputFailure: Error, CustomDebugStringConvertible, Sendable {
    let shape: String
    let promptTokens: Int
    let newTokens: Int
    let decodedCalls: Int
    let visibleBytes: Int
    let underlying: String

    init(shape: String,
         promptTokens: Int,
         newTokens: Int,
         decodedCalls: Int,
         visibleBytes: Int,
         underlying: Error) {
        self.shape = shape
        self.promptTokens = promptTokens
        self.newTokens = newTokens
        self.decodedCalls = decodedCalls
        self.visibleBytes = visibleBytes
        self.underlying = String(reflecting: underlying)
    }

    var debugDescription: String {
        "qwen_structured_output_failure shape=\(shape) prompt_n=\(promptTokens) "
        + "predicted_n=\(newTokens) calls=\(decodedCalls) visible_bytes=\(visibleBytes) "
        + "error=\(underlying)"
    }
}
