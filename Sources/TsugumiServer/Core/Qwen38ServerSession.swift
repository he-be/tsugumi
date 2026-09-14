import Foundation
import Tsugumi

/// SPEC's server over Qwen3.8-Flash-Next (qwen4exp, DS4-IQ2 GGUF + PLE) — `docs/qwen38/15` §2 G-1.
///
/// `QwenServerSession`'s sibling: the producer is `Qwen38Engine` over `Qwen38Runner` instead of `QwenForwardRunner`,
/// and everything from the tokenizer up is Ornith's — `QwenTokenizer` reading this checkpoint's own
/// `chat_template.jinja` (the vocabulary, merges and pre-tokenizer are Ornith's; the added tokens differ by seven audio
/// ids), the XML tool-call grammar and decoder, the reasoning splitter and `QwenPromptCache`.
///
/// What it does not do yet, each refused or recorded rather than dropped:
///
/// | | what happens |
/// | --- | --- |
/// | images | 400 `unsupported_image` |
/// | speculative decoding wider than one draft | `--draft-block-size 2` (n_max 1) is the only width; n_max 2 stays a check (`docs/qwen38/14`) |
/// | prompt cache | the live state or a checkpoint at or before the divergence; checkpoints at P − 1, at the end of the last user message (U − m) and before each `<tool_call>` generated (`docs/qwen38/15` §2 G, `18`) |
/// | sampling | temperature 0.7 / top_p 0.8 / top_k 20 / presence 1.5 (`Qwen38Sampler`), whatever the request asked for, the override named in `approximations` |
public actor Qwen38ServerSession: ServerInferenceBackend {
    private let engine: Qwen38Engine
    private let tokenizer: QwenTokenizer
    private let maxContext: Int
    private let speculative: Bool
    private var promptCache = Qwen38PromptCache()
    /// CACHE-8: the copies, by position. Kept across requests only while a later request uses or retakes them.
    private var checkpoints: [Int: Qwen38Checkpoint] = [:]
    private let reasoningBudget: Int
    private let reasoningFormat: ReasoningFormat
    private let grammarVocabulary: GrammarVocabulary
    private let markers: QwenToolCallMarkers
    private var accumulatedMetrics = ServerMetricsSnapshot.zero
    /// Checks only (`Q38_SERVER_GREEDY=1`): argmax instead of the official sampler, to compare token for token with
    /// `--qwen38-generate --q38-sampler greedy`.
    private let greedy = ProcessInfo.processInfo.environment["Q38_SERVER_GREEDY"] == "1"

    /// Prefill chunk: 2,048 is what the 12K runs measured (`docs/qwen38/09`); 4,096 does not fit the RAM (`08` §4-1).
    public static let defaultPrefillChunk = 2_048

    public static func isQwen38(modelDirectory: URL) -> Bool {
        Model.declaredFamily(at: modelDirectory) == Qwen38ModelDirectory.family
    }

    public static func load(modelDirectory: URL,
                            maxContext: Int,
                            draftBlockSize: Int = 0,
                            reasoningBudget: Int = -1,
                            reasoningFormat: ReasoningFormat = .auto,
                            prefillChunk: Int = defaultPrefillChunk) async throws -> Qwen38ServerSession {
        try validateFlags(draftBlockSize: draftBlockSize)
        let files = try Qwen38ModelDirectory(directory: modelDirectory)
        let tokenizer = try await QwenTokenizer.load(forModelDirectory: modelDirectory)
        let engine = try Qwen38Engine(gguf: files.gguf, ple: files.ple, capacity: maxContext,
                                      prefillChunk: min(prefillChunk, maxContext), speculative: draftBlockSize != 0)
        return Qwen38ServerSession(engine: engine, tokenizer: tokenizer, maxContext: maxContext,
                                   reasoningBudget: reasoningBudget, reasoningFormat: reasoningFormat)
    }

    /// Two is the only width: the loop drafts one token a pass and verifies two rows (n_max 1).
    public static func validateFlags(draftBlockSize: Int) throws {
        guard draftBlockSize == 0 || draftBlockSize == 2 else {
            throw ServerArgumentError.invalid(
                "--draft-block-size \(draftBlockSize) has no Qwen3.8 path: the server loop drafts one token a pass, "
                + "so the width is 2 (docs/qwen38/10 §5-4); run with --draft-block-size 2 or 0")
        }
    }

    private init(engine: Qwen38Engine, tokenizer: QwenTokenizer, maxContext: Int,
                 reasoningBudget: Int, reasoningFormat: ReasoningFormat) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.maxContext = maxContext
        self.speculative = engine.speculative
        self.reasoningBudget = reasoningBudget
        self.reasoningFormat = reasoningFormat
        self.grammarVocabulary = GrammarVocabulary.shared(for: tokenizer, identity: "qwen3.8-flash-next")
        self.markers = QwenToolCallMarkers(tokenizer: tokenizer)
    }

    // MARK: - Prompt

    /// The checkpoint's own `chat_template.jinja`, rendered and encoded (no server variant yet: INV-1 is G-1's
    /// remaining work, `docs/qwen38/15` §2 G).
    private func renderPrompt(_ request: ValidatedChatRequest) throws -> [Int32] {
        guard request.vision == nil else {
            throw ServerRequestError.invalid(
                message: "this model has no vision path",
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

    // MARK: - Checkpoints (CACHE-8)

    /// Where this prompt's prefill takes a checkpoint (`docs/qwen38/15` §2 G's table): P − 1, which a regenerate of the
    /// same prompt restores to redraw one token, and U − m when the last message is the user's — U the position of
    /// that message's `<|im_end|>`, m the trailing tokens of its body that change when a line is appended (an
    /// instructed regenerate). m is found by encoding the last 16 body tokens as text with and without a newline suffix.
    private func checkpointPositions(promptIDs: [Int32], request: ValidatedChatRequest) -> [Int] {
        var positions = [promptIDs.count - 1]
        if request.messages.last?.role == .user,
           let u = promptIDs.lastIndex(of: tokenizer.imEndID), u > 0 {
            let window = Array(promptIDs[max(0, u - 16)..<u])
            let text = tokenizer.decode(window, skipSpecialTokens: false)
            let base = tokenizer.encode(text)
            var m = 0
            // A line appended (U5 appends "\n\n" and the instruction); a letter glued to the body is not a line.
            for suffix in ["\n", "\n\n", "\n\n("] {
                let other = tokenizer.encode(text + suffix)
                let same = zip(base, other).prefix { $0 == $1 }.count
                m = max(m, base.count - same)
            }
            positions.append(u - min(m, window.count))
        }
        return Array(Set(positions.filter { $0 > 0 })).sorted()
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
        try await generate(prepared, monitor: monitor, onPrefill: nil, onEvent: onEvent)
    }

    /// The full generation with a prefill-progress hook (the Mac app's progress bar).
    public func generate(
        _ prepared: ServerPreparedRequest,
        monitor: ServerTimingsMonitor?,
        onPrefill: (@Sendable (Int, Int) -> Void)?,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let request = prepared.request
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
        let plan = QwenGenerationPlan(request: request, markers: markers, official: .qwen38)

        // The verify pass writes the drafted row before it knows whether the token is real: one position more.
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

        // CACHE-1 / CACHE-2 / CACHE-3 / CACHE-8 (`docs/qwen38/15` §2 G): the newest of the live state and the
        // checkpoints at or before the divergence; nothing behind it means position 0.
        let (decision, agreed) = promptCache.decide(promptIDs, checkpoints: Array(checkpoints.keys),
                                                    cachePrompt: request.cachePrompt)
        var reused = 0
        var restoreSeconds = 0.0
        switch decision {
        case .live(let position):
            reused = position
        case .restore(let position):
            let t = Date()
            try engine.restore(checkpoints[position]!)
            restoreSeconds = Date().timeIntervalSince(t)
            reused = position
        case .miss:
            engine.reset()
            promptCache.invalidate()
        }
        // A checkpoint past the divergence belongs to a sequence this prompt is not.
        checkpoints = checkpoints.filter { $0.key <= agreed && decision != .miss }
        let wanted = checkpointPositions(promptIDs: promptIDs, request: request)
        var taken: [Qwen38Checkpoint] = []
        ServerLog.promptCache("qwen38 \(decision) agreed=\(agreed) prompt=\(promptIDs.count) "
            + "held=\(promptCache.tokens.count) take=\(wanted.filter { $0 > reused }) "
            + "checkpoints=\(checkpoints.keys.sorted())"
            + (restoreSeconds > 0 ? String(format: " restore_ms=%.0f", restoreSeconds * 1000) : ""))
        let promptSuffix = Array(promptIDs[reused...])

        let reasoning = ServerReasoningPlan(request: request,
                                            defaultBudget: reasoningBudget,
                                            defaultFormat: reasoningFormat,
                                            maxNewTokens: maxNewTokens,
                                            contextRemaining: contextRemaining,
                                            forcedTokenCount: 1)
        var approximations = plan.approximations
        if reasoning.forcesClosingTag {
            approximations.append(
                "reasoning/budget-not-enforced: this family has no forced closing tag yet")
        }

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

        let startsInsideReasoning = QwenStructuredAssistantDecoder
            .promptEndsInsideReasoning(promptIDs, tokenizer: tokenizer)
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
            live = ServerLiveTimings(cacheTokens: reused,
                                     promptTokens: promptSuffix.count,
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
                    reasoningContent += text
                    onEvent(.reasoning(text))
                case .toolCall(let call):
                    calls.append(call)
                    onEvent(.toolCall(call))
                }
            }
        }

        let run: Qwen38Run
        // Anything that leaves by throwing leaves a state whose position nothing can name.
        var completed = false
        // What survives this request: what it took, what it restored, and the positions it wanted that are already
        // held (U - m behind a restored P - 1).
        func retain(upTo limit: Int) {
            for checkpoint in taken { checkpoints[checkpoint.position] = checkpoint }
            let keep = Set(taken.map(\.position) + wanted + (reused > 0 ? [reused] : []))
            checkpoints = checkpoints.filter { keep.contains($0.key) && $0.key <= limit }
        }
        defer {
            if !completed {
                // The live state is unnamed; the checkpoints the prompt holds are not.
                engine.reset()
                promptCache.publishInterrupted(prompt: promptIDs)
                retain(upTo: promptIDs.count - 1)
            }
        }
        do {
            run = try engine.runCompletion(
                promptTokens: promptSuffix,
                cachedPromptTokens: reused,
                maxNewTokens: maxNewTokens,
                stopTokens: tokenizer.stopTokenIDs,
                constraint: constraint,
                greedy: greedy,
                seed: UInt64.random(in: 1...UInt64.max),
                checkpointsAt: request.cachePrompt ? wanted : [],
                checkpointBefore: request.cachePrompt && !request.tools.isEmpty ? [tokenizer.toolCallStartID] : [],
                shouldStop: { shouldStop || Task.isCancelled },
                onPrefill: { done, total in onPrefill?(done, total) },
                onCheckpoint: { taken.append($0) },
                onToken: { index, id in
                    let delta = detokenizer.push(id)
                    if let monitor, let timings = live?.observe(
                        .token(index: index, id: id, delta: delta), at: Date()) {
                        monitor.record(timings)
                    }
                    let events: [StructuredAssistantEvent]
                    if let decoder {
                        events = try decoder.consume(tokenID: id, delta: delta)
                        constraint?.setSuppressed(decoder.isInsideReasoning)
                    } else {
                        events = splitter!.consume(tokenID: id, delta: delta)
                    }
                    handle(events)
                })
            if !promptCache.publish(prompt: promptIDs,
                                    generated: run.tokens,
                                    kvPosition: run.kvPosition,
                                    cachePrompt: request.cachePrompt) {
                ServerLog.promptCache("dropped state=\(run.kvPosition) beyond "
                    + "prompt+generated=\(promptIDs.count + run.tokens.count)")
                engine.reset()
                checkpoints.removeAll()
            } else {
                retain(upTo: run.kvPosition)
            }
            ServerLog.promptCache(String(format: "qwen38 took=%@ capture_ms=%.0f held=%@ kv=%d",
                                         "\(taken.map(\.position))",
                                         taken.reduce(0) { $0 + $1.captureSeconds } * 1000,
                                         "\(checkpoints.keys.sorted())", run.kvPosition))
            completed = true
        } catch let error as GenerationConstraintError {
            throw ServerGrammarBuildFailure(shape: plan.shape, underlying: error)
        }

        let tail = detokenizer.flush()
        if let decoder {
            do {
                handle(try decoder.consumeTail(tail))
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
        let timings = ServerTimings(cacheTokens: run.cachedPromptTokens,
                                    promptTokens: run.promptTokens,
                                    promptMilliseconds: run.prefillSeconds * 1_000,
                                    predictedTokens: run.newTokens,
                                    predictedMilliseconds: run.decodeSeconds * 1_000)
        accumulatedMetrics = accumulatedMetrics.adding(timings)
        return ServerCompletion(
            content: content,
            toolCalls: calls,
            finishReason: reason,
            usage: OpenAIUsage(promptTokens: run.cachedPromptTokens + run.promptTokens,
                               completionTokens: run.newTokens,
                               totalTokens: run.cachedPromptTokens + run.promptTokens + run.newTokens,
                               cachedTokens: run.cachedPromptTokens),
            speculative: speculative
                ? ServerSpeculativeSummary(blockTokens: 2,
                                           rounds: run.passes,
                                           proposed: run.passes,
                                           accepted: run.accepted)
                : nil,
            reasoningContent: reasoningContent,
            approximations: approximations,
            timings: timings)
    }

    // MARK: - EP-5 / EP-6

    public func tokenize(_ text: String, addSpecial: Bool) -> [Int32] {
        tokenizer.encode(text)
    }

    public func detokenize(_ tokens: [Int32]) -> String {
        tokenizer.decode(tokens, skipSpecialTokens: false)
    }

    public func applyChatTemplate(_ request: ValidatedChatRequest) throws -> String {
        tokenizer.decode(try renderPrompt(request), skipSpecialTokens: false)
    }

    public func metrics() -> ServerMetricsSnapshot { accumulatedMetrics }
}
