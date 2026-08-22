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

        // The run reports its own partition of the wall clock (RSP-3), which is
        // the one Phase 6 measures with: `decode=` here is decode only, as it is
        // on the Gemma path, so the two families' tok/s mean the same thing.
        //
        // The profile is read at the phase boundary as well as at the end, so
        // the GPU seconds can be split the same way the wall clock is. Both
        // paths are serial, so each phase's three parts add up to its wall
        // clock and the remainder is host time
        // (`docs/qwen35moe/24-PREFILL-MOE-PATH.md` §4).
        runner.resetProfile()
        var prefillGPUSeconds = 0.0
        var prefillCommandBuffers = 0
        var speculative: QwenSpeculativeStats?
        // `--qwen-mtp` swaps the decode loop, not the prompt or the framing:
        // the prompt still goes through the T-row prefill and the answer is
        // still greedy, so the tokens must come out the same as without it
        // (`docs/qwen35moe/36-MTP-DECODE.md` §3).
        if let head = args.qwenMTP { try runner.attachMTPHead(directory: head) }
        let onPrefillProfile: (Int, Double) -> Void = { _, _ in
            prefillGPUSeconds = runner.gpuSeconds
            prefillCommandBuffers = runner.gpuCommandBuffers
        }
        let onTokenEmit: (Int, Int32) throws -> Void = { _, id in
                let delta = detokenizer.push(id)
                guard let decoder else {
                    emit(splitter!.consume(tokenID: id, delta: delta))
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
        let run = args.qwenMTP != nil
            ? try runner.runGreedyCompletionMTP(
                promptTokens: promptIds,
                maxNewTokens: maxNew,
                chunkWidth: runtime.prefillChunkTokens,
                stopTokens: tokenizer.stopTokenIDs,
                constraint: constraint,
                onPrefill: onPrefillProfile,
                onToken: onTokenEmit,
                onStats: { speculative = $0 })
            : try runner.runGreedyCompletion(
                promptTokens: promptIds,
                maxNewTokens: maxNew,
                chunkWidth: runtime.prefillChunkTokens,
                stopTokens: tokenizer.stopTokenIDs,
                constraint: constraint,
                onPrefill: onPrefillProfile,
                onToken: onTokenEmit)
        let generated = run.tokens
        // The ids, not the text. Re-tokenizing what was printed does not
        // recover them: BPE can re-segment the same bytes, and the MTP
        // acceptance measurement teacher-forces on exactly this sequence
        // (`docs/qwen35moe/33-MTP-ACCEPTANCE.md` §1).
        if let path = args.dumpTokens {
            let payload = "{\"prompt\":[\(promptIds.map(String.init).joined(separator: ","))],"
                + "\"generated\":[\(generated.map(String.init).joined(separator: ","))]}\n"
            try Data(payload.utf8).write(to: URL(fileURLWithPath: path))
        }
        let tail = detokenizer.flush()
        if let decoder {
            emit(try decoder.consumeTail(tail))
            // Refuses a call the model left half-written rather than reporting
            // the ones before it as if the turn were complete.
            try decoder.finish()
        } else {
            emit(splitter!.consumeTail(tail))
        }
        stdout.write(Data("\n".utf8))
        // One call per line, after the text: a shell reads them with `tail`,
        // and the arguments are the parser's own JSON rather than the XML the
        // model wrote, so what is printed is what a client would be sent.
        for call in toolCalls {
            stdout.write(Data(("{\"id\":\"\(call.id)\",\"name\":\"\(call.name)\","
                               + "\"arguments\":\(call.argumentsJSON)}\n").utf8))
        }
        if !args.quiet {
            let stopped = run.reason != StopReason.maxTokens
            let tokensPerSecond = run.decodeSeconds > 0
                ? Double(generated.count) / run.decodeSeconds : 0
            let memory = ProcessMemoryFootprint.current()
            let telemetry = model.telemetry.snapshot()
            var footer = "\n[stop=\(stopped ? "stopToken" : "maxNewTokens")"
            footer += " prefill=\(promptIds.count)tok new=\(generated.count)tok"
            footer += " decode=\(String(format: "%.2f", run.decodeSeconds))s"
            footer += " tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            footer += "[load=\(String(format: "%.3f", loadSeconds))s"
            footer += " prefill=\(String(format: "%.3f", run.prefillSeconds))s"
            footer += " ttft=\(String(format: "%.3f", run.timeToFirstTokenSeconds))s"
            footer += " reasoning=\(reasoningText.count)ch answer=\(answerText.count)ch"
            if !tools.isEmpty { footer += " toolCalls=\(toolCalls.count)" }
            // What the grammar cost, in the only currency this path spends:
            // one extra pass over the 508 MB head per refused token.
            if constraint != nil { footer += " rescored=\(runner.constraintRescores)" }
            // What the head bought and what it cost, in the two numbers that
            // decide it: how often the draft was right, and how many tokens one
            // verify pass produced (`docs/qwen35moe/36-MTP-DECODE.md` §4).
            if let spec = speculative {
                footer += " mtpP1=\(String(format: "%.1f", spec.acceptanceRate * 100))%"
                footer += " a=\(String(format: "%.3f", spec.acceptanceLength))"
                footer += " passes=\(spec.passes)"
            }
            footer += " peak=\(String(format: "%.2f", Double(memory.peakPhysFootprintBytes) / 1e9))GB]\n"
            footer += "[expert prefill hit="
            footer += "\(String(format: "%.1f", telemetry.prefill.hitRate * 100))%"
            footer += " \(telemetry.prefill.hits)/\(telemetry.prefill.experts)"
            footer += " | decode hit="
            footer += "\(String(format: "%.1f", telemetry.decode.hitRate * 100))%"
            footer += " \(telemetry.decode.hits)/\(telemetry.decode.experts)"
            // The expert I/O each phase waited on, and what one decode token
            // paid for it — the currency `bench/expert_sim.py --io-ms` converts
            // a miss count into.
            footer += " | io prefill=\(String(format: "%.2f", Double(telemetry.prefill.fetchNanos) / 1e9))s"
            footer += " decode=\(String(format: "%.2f", Double(telemetry.decode.fetchNanos) / 1e9))s]\n"
            // Where each phase's wall clock went. `host` is the remainder —
            // encode, readback, and the driver's own overhead — and on this
            // serial path it is the cost of committing and waiting on a command
            // buffer, which is why the count is printed next to it.
            func split(_ name: String,
                       wall: Double, gpu: Double, io: Double,
                       units: Int, buffers: Int) -> String {
                guard units > 0 else { return "" }
                let ms = { (s: Double) in String(format: "%.2f", s * 1e3 / Double(units)) }
                return "[\(name)/tok gpu=\(ms(gpu))ms io=\(ms(io))ms"
                    + " host=\(ms(wall - gpu - io))ms"
                    + " cb=\(String(format: "%.1f", Double(buffers) / Double(units)))]\n"
            }
            footer += split("prefill",
                            wall: run.prefillSeconds,
                            gpu: prefillGPUSeconds,
                            io: Double(telemetry.prefill.fetchNanos) / 1e9,
                            units: promptIds.count,
                            buffers: prefillCommandBuffers)
            footer += split("decode",
                            wall: run.decodeSeconds,
                            gpu: runner.gpuSeconds - prefillGPUSeconds,
                            io: Double(telemetry.decode.fetchNanos) / 1e9,
                            units: generated.count,
                            buffers: runner.gpuCommandBuffers - prefillCommandBuffers)
            // Where a speculative step's wall clock goes. The three parts add
            // up to `decode` because the loop is serial: draft, then verify
            // (the snapshot is inside it), then the host's own bookkeeping.
            if let spec = speculative, spec.passes > 0 {
                let perPass = { (s: Double) in
                    String(format: "%.2f", s * 1e3 / Double(spec.passes))
                }
                footer += "[mtp/pass draft=\(perPass(spec.draftSeconds))ms"
                footer += " verify=\(perPass(spec.verifySeconds))ms"
                footer += " snapshot=\(perPass(spec.snapshotSeconds))ms"
                footer += " tok=\(String(format: "%.3f", spec.acceptanceLength))]\n"
            }
            // What the mapped arm spends inside that `io` number: the residency
            // set churn is host work on the fetch thread, and the rest is the
            // page-in the GPU would otherwise fault on.
            let residency = MmapExpertMapping.stats
            if residency.syncs > 0 && !generated.isEmpty {
                let ms = Double(residency.nanos) / 1e6 / Double(generated.count)
                footer += "[residency syncs=\(residency.syncs)"
                footer += " added=\(residency.added) removed=\(residency.removed)"
                let part = { (v: UInt64) in
                    String(format: "%.2f", Double(v) / 1e6 / Double(generated.count))
                }
                footer += " \(String(format: "%.2f", ms))ms/tok"
                footer += " (edit=\(part(residency.editNanos))"
                footer += " commit=\(part(residency.commitNanos))"
                footer += " request=\(part(residency.requestNanos)))]\n"
            }
            let preview = runner.routerPreview
            if preview.comparisons > 0 {
                let pct = { (n: Int, d: Int) in
                    d > 0 ? String(format: "%.1f", Double(n) / Double(d) * 100) : "-"
                }
                footer += "[preview pairs=\(preview.comparisons)"
                // The denominator is how many ranks the preview produced, which
                // is eight on the select-kernel path and whatever
                // `TF_QWEN_PREVIEW_TOPN` asked for on the wide one.
                footer += " ranks=\(preview.rankUsed.count)"
                footer += " overlap=\(pct(preview.overlap, preview.comparisons * max(preview.rankUsed.count, 1)))%"
                footer += " missCovered=\(pct(preview.missedCovered, preview.missed))%"
                footer += " (\(preview.missedCovered)/\(preview.missed))"
                footer += " rankUsed=" + preview.rankUsed
                    .map { pct($0, preview.comparisons) }.joined(separator: "/")
                footer += " rankMiss=" + preview.rankMissed
                    .map { pct($0, preview.comparisons) }.joined(separator: "/")
                footer += "]\n"
            }
            // The read-ahead's own three numbers: was the guess refused, did it
            // move anything, and had it landed when the layer arrived
            // (`docs/qwen35moe/28-PREFETCH-IDEAS.md` §3-2 / §3-3).
            let prefetch = runner.expertPrefetch
            if prefetch.issuedPlans + prefetch.declined > 0 {
                footer += "[prefetch issued=\(prefetch.issuedPlans)"
                footer += " declined=\(prefetch.declined)"
                footer += " reads=\(prefetch.reads)"
                footer += " waits=\(prefetch.waits)"
                footer += " slow=\(prefetch.slowWaits)"
                footer += " max=\(String(format: "%.2f", Double(prefetch.maxWaitNanos) / 1e6))ms"
                let waitMs = Double(prefetch.waitNanos) / 1e6
                footer += " wait=\(String(format: "%.1f", waitMs))ms"
                if !generated.isEmpty {
                    footer += " (\(String(format: "%.3f", waitMs / Double(generated.count)))ms/tok)"
                }
                footer += "]\n"
            }
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
