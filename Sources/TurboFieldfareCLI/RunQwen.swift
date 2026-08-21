import Foundation
import Metal
import TurboFieldfare

// The Ornith (Qwen 3.5-MoE) arm of the CLI — Phase 5's exit condition
// (`docs/qwen35moe/04-PHASES.md`): a prompt in Japanese or English goes in and
// an answer comes out, with the `<think>` block separated from it.
//
//   swift run -c release TurboFieldfareCLI \
//     --model scratch/ornith-oq4e-g64.gturbo \
//     --messages-file scratch/qwen35/ja.json --thinking
//
// It is a separate path from `run(args:)`, not a branch inside it, because
// nothing downstream of the tokenizer is shared: the framing is ChatML, the
// producer is `QwenForwardRunner` (serial, greedy, its own K/V and recurrent
// state), and the sampler, the speculative loop and the vision assembler have
// no Ornith side yet. Sharing `RawCompletion` would mean generalizing three
// types that carry Gemma 4 measurements, which is the one thing this plan does
// not do (`docs/qwen35moe/README.md` 運用ルール).
//
// What that costs, and where it is written down: greedy only (the runner runs
// the fused head, so `--temperature` and friends are refused rather than
// ignored), `--stop` strings are not applied (token stops are), and there is no
// speculative or vision path.
//
// `--tools` adds the second half of Phase 5 (`docs/qwen35moe/25-CLI-TOOLS.md`):
// the declarations go into the system turn the checkpoint's own template
// writes, a `QwenToolCallGrammar` constrains the call, and
// `QwenStructuredAssistantDecoder` turns the generated tokens back into
// arguments. Greedy stays greedy — the constraint is applied to the argmax
// rather than to a distribution (`QwenForwardRunner.constrained`).

/// Which turn a piece of the model's output belongs to.
///
/// Ornith writes its reasoning inline, between `<think>` and `</think>`, and
/// the template *opens* that block in the generation prompt — so a run with
/// thinking on starts inside it and the closing marker is the only signal.
/// Tracking the marker IDs rather than the text means a token that merely
/// spells `<think>` in the answer cannot move the boundary.
private struct QwenReasoningSplitter {
    private let thinkStart: Int32
    private let thinkEnd: Int32
    private(set) var insideReasoning: Bool

    init(tokenizer: QwenTokenizer, startsInsideReasoning: Bool) {
        self.thinkStart = tokenizer.thinkStartID
        self.thinkEnd = tokenizer.thinkEndID
        self.insideReasoning = startsInsideReasoning
    }

    /// Where `id`'s text belongs, and — for a marker — the marker's own
    /// spelling, which is framing rather than content and is not printed.
    ///
    /// The spelling has to come back out because these markers are *not*
    /// special tokens (`tokenizer.json` declares them `special: false`), so the
    /// detokenizer emits them as text, preceded by whatever bytes it was
    /// holding back from before them. Dropping the whole delta would drop those
    /// bytes too — a codepoint that straddles the token in front of `</think>`
    /// would vanish from the answer.
    mutating func route(_ id: Int32) -> (channel: Channel, marker: String?) {
        if id == thinkStart {
            insideReasoning = true
            return (.reasoning, "<think>")
        }
        if id == thinkEnd {
            // The held-back bytes belong to the reasoning that ends here.
            insideReasoning = false
            return (.reasoning, "</think>")
        }
        return (insideReasoning ? .reasoning : .content, nil)
    }

    enum Channel { case reasoning, content }
}

/// One turn from `--messages-file`, in the subset Ornith's template renders.
/// Images are refused rather than dropped: the vision tower for this family is
/// Phase 9, and a silently text-only answer about an image is the failure mode
/// worth the most to avoid.
private struct QwenMessageJSON: Decodable {
    let role: String
    let content: String
}

/// One entry of `--tools`.
///
/// Both shapes are accepted: the OpenAI envelope
/// (`{"type":"function","function":{…}}`) that a client would send, and the
/// bare declaration, which is what a hand-written file usually holds. The
/// envelope is the one the server sees, so accepting only the bare form would
/// make a file that works here fail there.
struct QwenToolJSON: Decodable {
    let definition: GFTokenizer.FunctionDefinition

    private struct Declaration: Decodable {
        let name: String
        let description: String?
        let parameters: JSONValue?
    }

    private enum CodingKeys: String, CodingKey { case type, function }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let declaration: Declaration
        if let nested = try? container.decode(Declaration.self, forKey: .function) {
            let type = try? container.decode(String.self, forKey: .type)
            if let type, type != "function" {
                throw ArgsError.invalidValue(flag: "--tools",
                                             value: "unsupported tool type \(type)")
            }
            declaration = nested
        } else {
            declaration = try Declaration(from: decoder)
        }
        // An empty schema is a function of no arguments, not a malformed one:
        // the template renders `parameters` verbatim and the grammar spells no
        // parameter block for it.
        self.definition = GFTokenizer.FunctionDefinition(
            name: declaration.name,
            description: declaration.description ?? "",
            parameters: declaration.parameters ?? .object([:]))
    }
}

/// `--tools`, decoded. Split from the file read so the two accepted shapes can
/// be checked without a filesystem.
enum QwenToolFile {
    static func decode(_ data: Data) throws -> [GFTokenizer.FunctionDefinition] {
        let rows = try JSONDecoder().decode([QwenToolJSON].self, from: data)
        let tools = rows.map(\.definition)
        var seen = Set<String>()
        for tool in tools where !seen.insert(tool.name).inserted {
            // Two declarations of one name make the grammar ambiguous and the
            // parser's schema lookup arbitrary; neither failure would be
            // visible in the output.
            throw ArgsError.invalidValue(flag: "--tools",
                                         value: "duplicate tool name \(tool.name)")
        }
        return tools
    }
}

private func qwenTools(args: Args) throws -> [GFTokenizer.FunctionDefinition] {
    guard let path = args.toolsFile else { return [] }
    return try QwenToolFile.decode(
        Data(contentsOf: URL(fileURLWithPath: path), options: [.mappedIfSafe]))
}

/// The grammar `--tool-choice` asks for, or nil for no constraint.
///
/// This is the CLI's half of what `QwenChatGrammarBuilder` does for the server,
/// and deliberately only that half: which tools the choice selects and whether
/// the grammar is lazy. The grammar text itself is the checkpoint's, and comes
/// from the same `QwenToolCallGrammar` both callers share — so a call the CLI
/// accepts is one the server would accept.
private func qwenToolConstraint(
    tools: [GFTokenizer.FunctionDefinition],
    choice: CLIToolChoice,
    parallelToolCalls: Bool,
    tokenizer: QwenTokenizer
) throws -> (constraint: GrammarTokenConstraint?, approximations: [String]) {
    guard !tools.isEmpty else { return (nil, []) }
    let selected: [GFTokenizer.FunctionDefinition]
    switch choice {
    case .none:
        // GEN-4: the declarations still go into the prompt — the model is told
        // what it has — but nothing constrains it.
        return (nil, [])
    case .auto, .required:
        selected = tools
    case .function(let name):
        selected = tools.filter { $0.name == name }
        guard !selected.isEmpty else {
            throw ArgsError.invalidValue(
                flag: "--tool-choice",
                value: "\(name) is not one of the declared tools")
        }
    }
    let isLazy = choice == .auto
    let markers = QwenToolCallMarkers(tokenizer: tokenizer)
    let result = QwenToolCallGrammar.grammar(tools: selected,
                                             parallelToolCalls: parallelToolCalls,
                                             withPreamble: !isLazy,
                                             markers: markers)
    let constraint = try GrammarTokenConstraint(
        result.grammar,
        vocabulary: GrammarVocabulary(tokenizer),
        // GEN-5: a lazy grammar sleeps until the model itself opens a call.
        trigger: isLazy ? GrammarTrigger.token(markers.toolCallStartTokenID) : nil)
    return (constraint, result.approximations)
}

private func qwenPrompt(args: Args,
                        tools: [GFTokenizer.FunctionDefinition],
                        tokenizer: QwenTokenizer) throws -> [Int32] {
    if let raw = args.prompt {
        // Same meaning as the Gemma arm: verbatim, no framing. There is no BOS
        // to prepend — this checkpoint has none (`tokenizer_config.json`,
        // `add_bos_token: false`).
        return tokenizer.encode(raw)
    }
    guard let messagesFile = args.messagesFile else { throw ArgsError.modeMissing }
    let data = try Data(contentsOf: URL(fileURLWithPath: messagesFile),
                        options: [.mappedIfSafe])
    let rows = try JSONDecoder().decode([QwenMessageJSON].self, from: data)
    let messages: [GFTokenizer.Message] = try rows.map { row in
        guard let role = GFTokenizer.Role(rawValue: row.role) else {
            throw QwenTokenizerError.invalidChatMessages("unsupported role \(row.role)")
        }
        return GFTokenizer.Message(role: role, content: row.content)
    }
    return try tokenizer.applyChatTemplate(messages,
                                          tools: tools,
                                          enableThinking: args.thinking)
}

func runQwen(args: Args,
             stdout: FileHandle = .standardOutput,
             stderr: FileHandle = .standardError) async -> RunResult {
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        guard args.images.isEmpty else {
            return qwenErrored(stderr, "this family has no vision path yet "
                               + "(docs/qwen35moe/04-PHASES.md Phase 9); drop --image", 2)
        }
        guard args.draftBlockSize == 0 else {
            return qwenErrored(stderr, "this family has no speculative path yet "
                               + "(Phase 7); use --draft-block-size 0", 2)
        }
        let config = GenerationConfig(maxNewTokens: args.maxNew,
                                      temperature: args.temperature,
                                      topK: args.topK,
                                      topP: args.topP,
                                      repetitionPenalty: args.repetitionPenalty,
                                      seed: args.seed,
                                      stopStrings: args.stops,
                                      extraStopTokens: [])
        guard config.isPureGreedy else {
            return qwenErrored(stderr, "this family decodes greedily only for now "
                               + "(the runner uses the fused head); pass "
                               + "--temperature 0 --repetition-penalty 1", 2)
        }
        if !args.stops.isEmpty {
            stderr.write(Data(("warning: --stop strings are not applied on this "
                               + "family; the token stops are\n").utf8))
        }

        let tokenizer = try await QwenTokenizer.load(forModelDirectory: modelURL)
        let tools = try qwenTools(args: args)
        let (constraint, approximations) = try qwenToolConstraint(
            tools: tools,
            choice: args.toolChoice,
            parallelToolCalls: args.parallelToolCalls,
            tokenizer: tokenizer)
        // SPEC §12: an approximation is a place the grammar is looser than the
        // schema. Silence would be the wrong default — the caller is asking for
        // a guarantee — so they are named on stderr, once, before generation.
        for note in approximations where !args.quiet {
            stderr.write(Data("note: grammar approximation: \(note)\n".utf8))
        }
        let promptIds = try qwenPrompt(args: args, tools: tools, tokenizer: tokenizer)
        guard !promptIds.isEmpty else { return qwenErrored(stderr, "empty prompt", 2) }
        guard promptIds.count < args.maxContext else {
            return qwenErrored(
                stderr,
                "context overflow: prompt \(promptIds.count) reaches maxContext \(args.maxContext)",
                2)
        }
        let maxNew = min(args.maxNew, args.maxContext - promptIds.count)

        guard MTLCreateSystemDefaultDevice() != nil else {
            return qwenErrored(stderr, "no Metal device", 1)
        }
        let runtime = try args.resolvedRuntimeConfiguration(forceLogitsHead: false)
        let loadStart = Date()
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            expecting: .ornith1_5_35B_A3B,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: args.verification)
        let runner = try QwenForwardRunner(model: model,
                                           context: context,
                                           maxContext: args.maxContext)
        let loadSeconds = Date().timeIntervalSince(loadStart)

        if let tracePath = args.dumpExpertTrace {
            try model.telemetry.startTrace(path: tracePath, header: [
                "model": model.modelID,
                "slots": "\(runtime.expertCacheSlots)",
                "policy": runtime.expertCachePolicy.rawValue,
                "experts": "\(model.config.numExperts)",
                "topK": "\(model.config.topKExperts)",
                "layers": "\(model.config.numLayers)",
                "prefillChunkTokens": "\(runtime.prefillChunkTokens)",
            ])
        }
        defer { model.telemetry.finishTrace() }

        // The template leaves the reasoning block open when thinking is on, so
        // the first generated token is already inside it.
        // Read off the last marker in the prompt, so a `<think>` quoted in an
        // earlier turn cannot decide it. The decoder owns that rule; the
        // splitter arm asks it the same question rather than keeping a second
        // copy that could drift.
        let startsInsideReasoning = QwenStructuredAssistantDecoder
            .promptEndsInsideReasoning(promptIds, tokenizer: tokenizer)
        // Two producers, one channel rule. Without `--tools` the text-only
        // splitter runs, exactly as it did before this flag existed — a
        // `<tool_call>` the model writes unasked stays text, which is what a
        // run that declared no tools should show. With `--tools` the structured
        // decoder runs instead: it applies the same rule to `<think>` and adds
        // the tool body, which it collects as token ids and parses once.
        var splitter = tools.isEmpty
            ? QwenReasoningSplitter(tokenizer: tokenizer,
                                    startsInsideReasoning: startsInsideReasoning)
            : nil
        let decoder = tools.isEmpty
            ? nil
            : QwenStructuredAssistantDecoder(tokenizer: tokenizer,
                                             tools: tools,
                                             emitsReasoning: true,
                                             startsInReasoning: startsInsideReasoning)
        var detokenizer = tokenizer.makeDetokenizer(skipSpecialTokens: true)
        var reasoningText = ""
        var answerText = ""
        var toolCalls: [ParsedToolCall] = []
        var firstTokenAt: Date?
        let decodeStart = Date()

        func emit(_ events: [StructuredAssistantEvent]) {
            for event in events {
                switch event {
                case .reasoning(let text):
                    reasoningText += text
                    if !args.quiet { stderr.write(Data(text.utf8)) }
                case .content(let text):
                    answerText += text
                    stdout.write(Data(text.utf8))
                case .toolCall(let call):
                    toolCalls.append(call)
                }
            }
        }

        let generated = try runner.generateGreedyPrefilled(
            promptTokens: promptIds,
            maxNewTokens: maxNew,
            chunkWidth: runtime.prefillChunkTokens,
            stopTokens: tokenizer.stopTokenIDs,
            constraint: constraint
        ) { _, id in
            if firstTokenAt == nil { firstTokenAt = Date() }
            let delta = detokenizer.push(id)
            guard let decoder else {
                let route = splitter!.route(id)
                var text = delta
                if let marker = route.marker, text.hasSuffix(marker) {
                    text = String(text.dropLast(marker.count))
                }
                guard !text.isEmpty else { return }
                switch route.channel {
                case .reasoning:
                    reasoningText += text
                    if !args.quiet { stderr.write(Data(text.utf8)) }
                case .content:
                    answerText += text
                    stdout.write(Data(text.utf8))
                }
                return
            }
            let events = try decoder.consume(tokenID: id, delta: delta)
            // GEN-6, and the reason this is set here rather than inside the
            // runner: the decoder owns the channel state, and the state it is
            // left in by *this* token is the one the next draw is judged by.
            // A lazy grammar that armed inside the thought block would
            // constrain reasoning the model is allowed to write freely.
            constraint?.setSuppressed(decoder.isInsideReasoning)
            emit(events)
        }
        let tail = detokenizer.flush()
        if let decoder {
            emit(try decoder.consumeTail(tail))
            // Refuses a call the model left half-written rather than reporting
            // the ones before it as if the turn were complete.
            try decoder.finish()
        } else if !tail.isEmpty {
            if splitter!.insideReasoning {
                reasoningText += tail
                if !args.quiet { stderr.write(Data(tail.utf8)) }
            } else {
                answerText += tail
                stdout.write(Data(tail.utf8))
            }
        }
        stdout.write(Data("\n".utf8))
        // One call per line, after the text: a shell reads them with `tail`,
        // and the arguments are the parser's own JSON rather than the XML the
        // model wrote, so what is printed is what a client would be sent.
        for call in toolCalls {
            stdout.write(Data(("{\"id\":\"\(call.id)\",\"name\":\"\(call.name)\","
                               + "\"arguments\":\(call.argumentsJSON)}\n").utf8))
        }
        let decodeSeconds = Date().timeIntervalSince(decodeStart)

        if !args.quiet {
            let stopped = generated.last.map { tokenizer.stopTokenIDs.contains($0) } ?? false
            let tokensPerSecond = decodeSeconds > 0
                ? Double(generated.count) / decodeSeconds : 0
            let ttft = firstTokenAt.map { $0.timeIntervalSince(decodeStart) } ?? 0
            let memory = ProcessMemoryFootprint.current()
            let telemetry = model.telemetry.snapshot()
            var footer = "\n[stop=\(stopped ? "stopToken" : "maxNewTokens")"
            footer += " prefill=\(promptIds.count)tok new=\(generated.count)tok"
            footer += " decode=\(String(format: "%.2f", decodeSeconds))s"
            footer += " tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            footer += "[load=\(String(format: "%.3f", loadSeconds))s"
            footer += " ttft=\(String(format: "%.3f", ttft))s"
            footer += " reasoning=\(reasoningText.count)ch answer=\(answerText.count)ch"
            if !tools.isEmpty { footer += " toolCalls=\(toolCalls.count)" }
            // What the grammar cost, in the only currency this path spends:
            // one extra pass over the 508 MB head per refused token.
            if constraint != nil { footer += " rescored=\(runner.constraintRescores)" }
            footer += " peak=\(String(format: "%.2f", Double(memory.peakPhysFootprintBytes) / 1e9))GB]\n"
            footer += "[expert prefill hit="
            footer += "\(String(format: "%.1f", telemetry.prefill.hitRate * 100))%"
            footer += " \(telemetry.prefill.hits)/\(telemetry.prefill.experts)"
            footer += " | decode hit="
            footer += "\(String(format: "%.1f", telemetry.decode.hitRate * 100))%"
            footer += " \(telemetry.decode.hits)/\(telemetry.decode.experts)]\n"
            stderr.write(Data(footer.utf8))
        }
        return RunResult(exitCode: 0)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return qwenErrored(stderr, "\(error)", 1)
    }
}

private func qwenErrored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}
