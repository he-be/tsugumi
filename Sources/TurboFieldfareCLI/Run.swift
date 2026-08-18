import Foundation
import Metal
import TurboFieldfare

/// A chat message from `--messages-file`.
///
/// `content` is either a string, as it has always been, or a list of parts so a
/// turn can carry images (PLAN_VISION §4-6). An image part names a file rather
/// than a data URI: this is a local CLI, and a path is what a shell has.
private struct MessageJSON: Decodable {
    enum Part {
        case text(String)
        case image(path: String)
    }

    let role: String
    let parts: [Part]

    private enum CodingKeys: String, CodingKey { case role, content }

    private struct PartJSON: Decodable {
        let type: String
        let text: String?
        let path: String?
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.role = try container.decode(String.self, forKey: .role)
        if let text = try? container.decode(String.self, forKey: .content) {
            self.parts = [.text(text)]
            return
        }
        let rawParts = try container.decode([PartJSON].self, forKey: .content)
        self.parts = try rawParts.map { part in
            switch part.type {
            case "text":
                guard let text = part.text else {
                    throw GFTokenizerError.invalidChatTemplate("text part without \"text\"")
                }
                return .text(text)
            case "image":
                guard let path = part.path else {
                    throw GFTokenizerError.invalidChatTemplate("image part without \"path\"")
                }
                return .image(path: path)
            default:
                throw GFTokenizerError.invalidChatTemplate(
                    "unsupported content part type \(part.type)")
            }
        }
    }
}

/// The prompt, plus the images whose soft tokens sit inside it.
private struct PreparedPrompt {
    let ids: [Int32]
    let vision: VisionPrefillInput?
    let softTokenCounts: [Int]
}

private func preparePrompt(args: Args, tokenizer: GFTokenizer) throws -> PreparedPrompt {
    if let rawPrompt = args.prompt {
        return PreparedPrompt(ids: tokenizer.encode(rawPrompt, addBOS: true),
                              vision: nil,
                              softTokenCounts: [])
    }
    guard let messagesFile = args.messagesFile else {
        throw ArgsError.modeMissing
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: messagesFile),
                        options: [.mappedIfSafe])
    let rows = try JSONDecoder().decode([MessageJSON].self, from: data)

    var messages: [GFTokenizer.MultimodalMessage] = []
    var imagePaths: [String] = []
    for row in rows {
        guard let role = GFTokenizer.Role(rawValue: row.role) else {
            throw GFTokenizerError.invalidChatTemplate("unsupported role \(row.role)")
        }
        var parts: [GFTokenizer.ContentPart] = []
        for part in row.parts {
            switch part {
            case .text(let text):
                parts.append(.text(text))
            case .image(let path):
                parts.append(.image)
                imagePaths.append(path)
            }
        }
        messages.append(GFTokenizer.MultimodalMessage(role: role, parts: parts))
    }

    // `--image` attaches to the last user turn, after its text.
    if !args.images.isEmpty {
        guard let last = messages.lastIndex(where: { $0.role == .user }) else {
            throw GFTokenizerError.invalidChatTemplate(
                "--image needs a user turn in \(messagesFile) to attach to")
        }
        messages[last] = GFTokenizer.MultimodalMessage(
            role: .user,
            parts: messages[last].parts + args.images.map { _ in .image })
        imagePaths.append(contentsOf: args.images)
    }

    let rendered = try tokenizer.applyChatTemplate(multimodal: messages,
                                                   enableThinking: args.thinking)
    let tokens = tokenizer.encode(rendered, addBOS: false)
    guard !imagePaths.isEmpty else {
        return PreparedPrompt(ids: tokens, vision: nil, softTokenCounts: [])
    }

    let preprocessorConfig = try VisionPreprocessorConfig(maxSoftTokens: args.imageTokens)
    let images = try imagePaths.map { path in
        try VisionImagePreprocessor.preprocess(contentsOf: URL(fileURLWithPath: path),
                                               config: preprocessorConfig)
    }
    let ids = try VisionMediaTokenIDs(tokenizer: tokenizer)
    let prompt = try VisionPromptAssembler.makePrefillPrompt(tokens: tokens,
                                                             images: images,
                                                             ids: ids)
    return PreparedPrompt(ids: prompt.tokens,
                          vision: prompt.vision,
                          softTokenCounts: images.map(\.geometry.softTokenCount))
}

public struct RunResult: Equatable, Sendable {
    public let exitCode: Int32
    public init(exitCode: Int32) { self.exitCode = exitCode }
}

public func run(args: Args,
                stdout: FileHandle = .standardOutput,
                stderr: FileHandle = .standardError) async -> RunResult {
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        let prepared = try preparePrompt(args: args, tokenizer: tokenizer)
        let promptIds = prepared.ids
        guard !promptIds.isEmpty else { return errored(stderr, "empty prompt", 2) }
        guard promptIds.count < args.maxContext else {
            return errored(
                stderr,
                "context overflow: prompt \(promptIds.count) reaches maxContext \(args.maxContext)",
                2)
        }
        let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
        let config = GenerationConfig(
            maxNewTokens: effectiveMaxNew,
            temperature: args.temperature,
            topK: args.topK,
            topP: args.topP,
            repetitionPenalty: args.repetitionPenalty,
            seed: args.seed,
            stopStrings: args.stops,
            extraStopTokens: [])
        let runtime = try args.resolvedRuntimeConfiguration(
            forceLogitsHead: !config.isPureGreedy)

        guard MTLCreateSystemDefaultDevice() != nil else {
            return errored(stderr, "no Metal device", 1)
        }
        let loadStart = Date()
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: args.verification)
        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize)
        let loadSeconds = Date().timeIntervalSince(loadStart)

        if let tracePath = args.dumpExpertTrace {
            try model.telemetry.startTrace(path: tracePath, header: [
                "model": model.modelID,
                "slots": "\(runtime.expertCacheSlots)",
                "policy": runtime.expertCachePolicy.rawValue,
                "experts": "\(model.config.numExperts)",
                "topK": "\(model.config.topKExperts)",
                "layers": "\(model.config.numLayers)",
                "prefill": runtime.prefillPolicy.rawValue,
                "prefillChunkTokens": "\(runtime.prefillChunkTokens)",
            ])
        }
        defer { model.telemetry.finishTrace() }
        let onProgress: (RawDecodeProgress) -> Void = { progress in
            switch progress {
            case .prefill:
                break
            case .token(_, _, let delta):
                if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
            case .tail(let tail):
                stdout.write(Data(tail.utf8))
            }
        }
        let stats: RawDecodeResult
        var speculativeStats: SpeculativeStats?
        if args.draftBlockSize > 0 {
            let speculative = try SpeculativeScratch(
                context: context,
                vocab: model.config.vocabSize,
                hiddenSize: model.config.hiddenSize,
                blockTokens: args.draftBlockSize,
                fusedGreedy: runner.usesFusedGreedyHead)
            let result = try await runSpeculativeCompletion(
                producer: runner,
                tokenizer: tokenizer,
                promptIds: promptIds,
                config: config,
                context: context,
                scratch: scratch,
                speculative: speculative,
                prefillConfig: runtime.prefillConfig,
                vision: prepared.vision,
                onProgress: onProgress)
            stats = result.decode
            speculativeStats = result.speculative
        } else {
            stats = try await runRawCompletion(
                producer: runner,
                tokenizer: tokenizer,
                promptIds: promptIds,
                config: config,
                context: context,
                scratch: scratch,
                prefillConfig: runtime.prefillConfig,
                vision: prepared.vision,
                onProgress: onProgress)
        }

        if !args.quiet {
            let tokensPerSecond = stats.decodeSeconds > 0
                ? Double(stats.newTokens) / stats.decodeSeconds
                : 0
            var footer = "\n[stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            footer += statsFooter(loadSeconds: loadSeconds,
                                  stats: stats,
                                  telemetry: model.telemetry.snapshot(),
                                  runner: runner,
                                  softTokenCounts: prepared.softTokenCounts,
                                  speculative: speculativeStats)
            stderr.write(Data(footer.utf8))
        }
        return RunResult(exitCode: 0)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

/// Phase-0 instrumentation lines. Kept on their own lines below the original
/// footer so existing footer parsers keep working unchanged.
private func statsFooter(loadSeconds: Double,
                         stats: RawDecodeResult,
                         telemetry: ExpertTelemetrySnapshot,
                         runner: RealForwardRunner,
                         softTokenCounts: [Int] = [],
                         speculative: SpeculativeStats? = nil) -> String {
    func s(_ value: Double) -> String { String(format: "%.3f", value) }
    func ms(_ nanos: UInt64, per count: Int) -> String {
        guard count > 0 else { return "n/a" }
        return String(format: "%.2f", Double(nanos) / 1e6 / Double(count))
    }
    func gb(_ bytes: UInt64) -> String {
        String(format: "%.2f", Double(bytes) / 1e9)
    }
    func pct(_ value: Double) -> String { String(format: "%.1f", value * 100) }

    let memory = ProcessMemoryFootprint.current()
    let newTokens = max(stats.newTokens, 1)

    var lines = "[load=\(s(loadSeconds))s"
    lines += " layerVerify=\(s(telemetry.layerVerifySeconds))s/\(telemetry.layersOpened)layers"
    lines += " prefill=\(s(stats.prefillSeconds))s"
    lines += " ttft=\(s(stats.timeToFirstTokenSeconds))s"
    lines += " peak=\(gb(memory.peakPhysFootprintBytes))GB"
    lines += " rss=\(gb(memory.peakResidentBytes))GB]\n"

    let p = telemetry.prefill
    let d = telemetry.decode
    lines += "[expert prefill hit=\(pct(p.hitRate))% \(p.hits)/\(p.experts)"
    lines += " io=\(s(Double(p.fetchNanos) / 1e9))s"
    lines += " | decode hit=\(pct(d.hitRate))% \(d.hits)/\(d.experts)"
    lines += " io=\(s(Double(d.fetchNanos) / 1e9))s]\n"

    if runner.visionTowerImages > 0 {
        lines += "[vision images=\(runner.visionTowerImages)"
        lines += " soft=\(softTokenCounts.map(String.init).joined(separator: ","))"
        lines += " tower=\(s(runner.visionTowerSeconds))s"
        lines += " towerLoad=\(s(runner.visionLoadSeconds))s]\n"
    }

    lines += "[decode/tok io=\(ms(runner.totalIoNanos, per: newTokens))ms"
    lines += " cb1=\(ms(runner.totalCb1Nanos, per: newTokens))ms"
    lines += " cb2=\(ms(runner.totalCb2Nanos, per: newTokens))ms"
    lines += " head=\(ms(runner.totalHeadNanos &+ runner.totalHeadFusedNanos, per: newTokens))ms]\n"

    // MTP diagnostics. 22-GOAL-RESET §6: these say where the time is, they are
    // not a score — the score is the end-to-end wall clock against a
    // same-session MTP-off run.
    if let mtp = speculative {
        lines += "[mtp bs=\(mtp.blockTokens) rounds=\(mtp.rounds)"
        lines += " accept=\(String(format: "%.3f", mtp.meanAcceptedLength))"
        lines += "/\(mtp.blockTokens - 1)"
        lines += " tok/round=\(String(format: "%.3f", mtp.rounds > 0 ? Double(newTokens) / Double(mtp.rounds) : 0))"
        lines += " draft=\(s(mtp.draftSeconds))s verify=\(s(mtp.verifySeconds))s"
        lines += " accepted=\(mtp.acceptedHistogram.map(String.init).joined(separator: ","))]\n"
    }

    return lines
}

private func errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}
