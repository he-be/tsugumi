import Foundation
import Metal
import TurboFieldfare

private struct MessageJSON: Decodable {
    let role: String
    let content: String
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
        let promptIds: [Int32]
        if let rawPrompt = args.prompt {
            promptIds = tokenizer.encode(rawPrompt, addBOS: true)
        } else if let messagesFile = args.messagesFile {
            let data = try Data(contentsOf: URL(fileURLWithPath: messagesFile),
                                options: [.mappedIfSafe])
            let rows = try JSONDecoder().decode([MessageJSON].self, from: data)
            let messages = try rows.map { row -> GFTokenizer.Message in
                guard let role = GFTokenizer.Role(rawValue: row.role) else {
                    throw GFTokenizerError.invalidChatTemplate("unsupported role \(row.role)")
                }
                return GFTokenizer.Message(role: role, content: row.content)
            }
            let rendered = try tokenizer.applyChatTemplate(messages,
                                                           enableThinking: args.thinking)
            promptIds = tokenizer.encode(rendered, addBOS: false)
        } else {
            return errored(stderr, "one of --prompt or --messages-file is required", 2)
        }
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
        let stats = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIds,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: runtime.prefillConfig) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                case .tail(let tail):
                    stdout.write(Data(tail.utf8))
                }
            }

        if !args.quiet {
            let tokensPerSecond = stats.decodeSeconds > 0
                ? Double(stats.newTokens) / stats.decodeSeconds
                : 0
            var footer = "\n[stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            footer += statsFooter(loadSeconds: loadSeconds,
                                  stats: stats,
                                  telemetry: model.telemetry.snapshot(),
                                  runner: runner)
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
                         runner: RealForwardRunner) -> String {
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

    lines += "[decode/tok io=\(ms(runner.totalIoNanos, per: newTokens))ms"
    lines += " cb1=\(ms(runner.totalCb1Nanos, per: newTokens))ms"
    lines += " cb2=\(ms(runner.totalCb2Nanos, per: newTokens))ms"
    lines += " head=\(ms(runner.totalHeadNanos &+ runner.totalHeadFusedNanos, per: newTokens))ms]\n"

    return lines
}

private func errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}
