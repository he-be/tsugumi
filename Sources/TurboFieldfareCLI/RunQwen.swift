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

    /// Where `id`'s text belongs, and whether the marker itself should be
    /// printed (it should not: it is framing, not content).
    mutating func route(_ id: Int32) -> (channel: Channel, isMarker: Bool) {
        if id == thinkStart {
            insideReasoning = true
            return (.reasoning, true)
        }
        if id == thinkEnd {
            insideReasoning = false
            return (.reasoning, true)
        }
        return (insideReasoning ? .reasoning : .content, false)
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

private func qwenPrompt(args: Args, tokenizer: QwenTokenizer) throws -> [Int32] {
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
    return try tokenizer.applyChatTemplate(messages, enableThinking: args.thinking)
}

/// Whether the prompt leaves the reasoning block open — the template does when
/// thinking is on (`…<|im_start|>assistant\n<think>\n`) and closes it again when
/// it is off (`<think>\n\n</think>\n\n`). Read off the last marker in the
/// prompt, so a `<think>` quoted earlier in the conversation cannot decide it.
private func promptEndsInsideReasoning(_ ids: [Int32], tokenizer: QwenTokenizer) -> Bool {
    for id in ids.reversed() {
        if id == tokenizer.thinkStartID { return true }
        if id == tokenizer.thinkEndID { return false }
    }
    return false
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
        let promptIds = try qwenPrompt(args: args, tokenizer: tokenizer)
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
        var splitter = QwenReasoningSplitter(
            tokenizer: tokenizer,
            startsInsideReasoning: promptEndsInsideReasoning(promptIds,
                                                             tokenizer: tokenizer))
        var detokenizer = tokenizer.makeDetokenizer(skipSpecialTokens: true)
        var reasoningText = ""
        var answerText = ""
        var firstTokenAt: Date?
        let decodeStart = Date()

        let generated = try runner.generateGreedyPrefilled(
            promptTokens: promptIds,
            maxNewTokens: maxNew,
            chunkWidth: runtime.prefillChunkTokens,
            stopTokens: tokenizer.stopTokenIDs
        ) { _, id in
            if firstTokenAt == nil { firstTokenAt = Date() }
            let route = splitter.route(id)
            let delta = detokenizer.push(id)
            guard !route.isMarker, !delta.isEmpty else { return }
            switch route.channel {
            case .reasoning:
                reasoningText += delta
                if !args.quiet { stderr.write(Data(delta.utf8)) }
            case .content:
                answerText += delta
                stdout.write(Data(delta.utf8))
            }
        }
        let tail = detokenizer.flush()
        if !tail.isEmpty {
            if splitter.insideReasoning {
                reasoningText += tail
                if !args.quiet { stderr.write(Data(tail.utf8)) }
            } else {
                answerText += tail
                stdout.write(Data(tail.utf8))
            }
        }
        stdout.write(Data("\n".utf8))
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
