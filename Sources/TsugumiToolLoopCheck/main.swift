import Foundation
import TsugumiAppCore

// The Mac app's tool loop, run without the window (`docs/qwen38/20`).
//
// `AppModel` is the app's own — lookups, the Online policy, the fold into history, the retry after a structured-output
// failure, the round budget — and so are the executors (Serper / Brave, the page readers, the local Wikipedia) and the
// settings files it reads. The one substitution is the inference client: `RealInferenceClient` in this process (the
// session the decode service runs) behind a recorder that keeps each round's request and diagnostics. The model is
// this process, so `Scripts/qwen38/guarded.sh` can watch and kill it.
//
//     .build/release/TsugumiToolLoopCheck --out DIR [--model DIR] [--conversations FILE] [--only a,b] [--repeats N]
//                                         [--network online|offline|model] [--context N] [--page-chars N]
//                                         [--web-store DIR] [--budget-notes limit|rounds|context]
//
// `--web-store DIR` answers the web tools' HTTP requests from DIR and records the ones it does not have
// (`RecordedHTTPTransport`), so a second run reads the same search results and pages (`docs/qwen38/21` §4 E-1).
//
// Writes `rounds.jsonl` (one line per round), `turns.jsonl` (one line per turn: answer, trace, checks) and prints a
// summary. Exit status 1 when a check failed. Checks per turn:
//   live      every round after the first continues from the live state the previous round left
//             (cached == previous prompt + previous generated − 1); the first round of a later turn likewise
//             continues from the previous turn's last round
//   error     no error
//   answer    a non-empty answer that is not a tool call written as text (`<tool_call>` / `<function=`, docs/qwen38/21 §3)
//   online    (Online) a web search ran, a page read was tried, and the answer names a source
//   progress  every round whose prefill took over 5 s reported its progress within 1 s of the round's start, starting
//             at the cached position, and never went more than 5 s without a report until the prefill ended (the
//             app's prefill counter, docs/qwen38/24 §1)

struct Options {
    var model = NSString(string: "~/LLM/Qwen3.8-Flash-Next-DS4-IQ2").expandingTildeInPath
    var out = ""
    var conversations = "Scripts/qwen38/tool_loop_conversations.json"
    var only: Set<String> = []
    var repeats = 1
    var network = AppNetworkMode.online
    var context: Int?
    /// Overrides the saved page text limit for this run only (the settings file is not written).
    var pageCharacters: Int?
    var webStore: String?
    var budgetNotes = AppToolBudgetNotes.limitOnly

    init(_ arguments: [String]) {
        var iterator = arguments.dropFirst().makeIterator()
        while let flag = iterator.next() {
            let value = iterator.next() ?? ""
            switch flag {
            case "--model": model = NSString(string: value).expandingTildeInPath
            case "--out": out = value
            case "--conversations": conversations = value
            case "--only": only = Set(value.split(separator: ",").map(String.init))
            case "--repeats": repeats = Int(value) ?? 1
            case "--network": network = AppNetworkMode(rawValue: value) ?? .online
            case "--context": context = Int(value)
            case "--page-chars": pageCharacters = Int(value)
            case "--web-store": webStore = value
            case "--budget-notes":
                guard let notes = AppToolBudgetNotes(rawValue: value) else {
                    FileHandle.standardError.write(Data("--budget-notes takes limit, rounds or context\n".utf8))
                    exit(2)
                }
                budgetNotes = notes
            default:
                FileHandle.standardError.write(Data("unknown flag \(flag)\n".utf8))
                exit(2)
            }
        }
        if out.isEmpty {
            FileHandle.standardError.write(Data("--out DIR is required\n".utf8))
            exit(2)
        }
    }
}

struct Conversation: Decodable {
    let name: String
    let turns: [String]
}

/// One round as it left and came back.
struct RoundRecord: Sendable {
    var request: AppGenerationRequest
    var started = Date()
    var ended: Date?
    var outcome = "running"
    var diagnostics: AppDiagnostics?
    var error: String?
    var toolCalls: [AppToolCall] = []
    /// `(seconds after started, done, total)` for each prefill progress event.
    var prefill: [(seconds: Double, done: Int, total: Int)] = []
}

/// `RealInferenceClient`, recording each generation.
final class RecordingClient: AppModelLifecycleClient, AppInferenceRuntimeReporting, @unchecked Sendable {
    private let inner = RealInferenceClient()
    private let lock = NSLock()
    private var records: [RoundRecord] = []

    var loadedRuntimeOwnBytes: UInt64? { inner.loadedRuntimeOwnBytes }

    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int, options: AppRuntimeOptions, forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        try await inner.ensureLoaded(modelDirectory: modelDirectory, maxContextTokens: maxContextTokens,
                                     options: options, forceLogitsHead: forceLogitsHead, onState: onState)
    }

    func unload() async { await inner.unload() }

    func cancel() { inner.cancel() }

    private func update(_ index: Int, _ change: (inout RoundRecord) -> Void) {
        lock.lock(); defer { lock.unlock() }
        change(&records[index])
    }

    func take() -> [RoundRecord] {
        lock.lock(); defer { lock.unlock() }
        defer { records = [] }
        return records
    }

    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        lock.lock()
        let index = records.count
        records.append(RoundRecord(request: request))
        lock.unlock()
        let stream = inner.generate(request)
        logLine("round \(index + 1) start: history=\(request.history.count) continuation=\(request.continuation.count) "
            + "tools=\(request.tools.count) choice=\(request.toolChoice.rawValue)")
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await event in stream {
                        switch event {
                        case .toolCall(let call): self.update(index) { $0.toolCalls.append(call) }
                        case .prefillProgress(let done, let total):
                            self.update(index) {
                                $0.prefill.append((Date().timeIntervalSince($0.started), done, total))
                            }
                        case .finished(let d):
                            self.update(index) { $0.outcome = "finished"; $0.diagnostics = d; $0.ended = Date() }
                        case .cancelled(let d):
                            self.update(index) { $0.outcome = "cancelled"; $0.diagnostics = d; $0.ended = Date() }
                        case .failed(let e, let d):
                            self.update(index) {
                                $0.outcome = "failed"; $0.diagnostics = d; $0.error = e.userMessage; $0.ended = Date()
                            }
                        default: break
                        }
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    let text = "\(error)"
                    self.update(index) {
                        if $0.outcome == "running" { $0.outcome = "threw"; $0.error = text; $0.ended = Date() }
                    }
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

func logLine(_ text: String) {
    FileHandle.standardError.write(Data("[\(Date().formatted(.iso8601))] check \(text)\n".utf8))
}

func jsonLine(_ object: [String: Any], to handle: FileHandle) {
    guard let data = try? JSONSerialization.data(withJSONObject: object,
                                                 options: [.sortedKeys, .withoutEscapingSlashes]) else { return }
    handle.write(data)
    handle.write(Data("\n".utf8))
}

func appendHandle(_ url: URL) -> FileHandle {
    FileManager.default.createFile(atPath: url.path, contents: nil)
    return try! FileHandle(forWritingTo: url)
}

@MainActor
func waitUntil(timeout: Double, _ condition: () -> Bool) async -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while !condition() {
        if Date() > deadline { return false }
        try? await Task.sleep(nanoseconds: 200_000_000)
    }
    return true
}

@MainActor
func runCheck() async -> Int32 {
    let options = Options(CommandLine.arguments)
    let outDirectory = URL(fileURLWithPath: options.out, isDirectory: true)
    try? FileManager.default.createDirectory(at: outDirectory, withIntermediateDirectories: true)
    let conversations: [Conversation]
    do {
        conversations = try JSONDecoder().decode([Conversation].self,
                                                 from: Data(contentsOf: URL(fileURLWithPath: options.conversations)))
            .filter { options.only.isEmpty || options.only.contains($0.name) }
    } catch {
        logLine("cannot read \(options.conversations): \(error)")
        return 2
    }

    let client = RecordingClient()
    var webStore: RecordedHTTPTransport?
    if let path = options.webStore {
        do {
            webStore = try RecordedHTTPTransport(directory: URL(fileURLWithPath: path, isDirectory: true))
        } catch {
            logLine("cannot use --web-store \(path): \(error)")
            return 2
        }
    }
    let toolExecutorProvider: ((WebSearchConfiguration, AppNetworkMode) throws -> (any AppToolExecutor)?)? =
        webStore.map { store in { try AppModel.makeToolExecutor(configuration: $0, mode: $1, transport: store) } }
    let model = AppModel(modelDirectory: URL(fileURLWithPath: options.model), client: client,
                         turnMetricsLog: AppTurnMetricsLog(fileURL: outDirectory.appendingPathComponent("turn-metrics.jsonl")),
                         webSearchConfigurationURL: WebSearchConfigurationStore.defaultFileURL,
                         personaURL: AppPersonaStore.defaultFileURL,
                         toolExecutorProvider: toolExecutorProvider)
    if let context = options.context { model.maxContextTokens = context }
    if let pageCharacters = options.pageCharacters {
        model.webSearchConfiguration.pageCharacterLimit = pageCharacters
    }
    model.networkMode = options.network
    model.toolBudgetNotes = options.budgetNotes
    logLine("model \(model.selectedModelKind.rawValue) context=\(model.maxContextTokens) mtp=\(model.runtimeOptions.mtpEnabled) "
        + "thinking=\(model.thinkingEnabled) network=\(model.effectiveNetworkMode.rawValue) "
        + "rounds=\(model.webSearchConfiguration.resolved().maxToolRounds) page=\(model.webSearchConfiguration.resolved().pageCharacterLimit) "
        + "budget-notes=\(model.toolBudgetNotes.rawValue)")
    model.loadModel()
    guard await waitUntil(timeout: 600, { model.loadState.isReady || model.error != nil }),
          model.loadState.isReady else {
        logLine("load failed: \(model.loadState) \(String(describing: model.error))")
        return 2
    }

    let rounds = appendHandle(outDirectory.appendingPathComponent("rounds.jsonl"))
    let turns = appendHandle(outDirectory.appendingPathComponent("turns.jsonl"))
    var failures: [String] = []
    var summary: [String] = []

    for repeatIndex in 1...max(options.repeats, 1) {
        for conversation in conversations {
            model.newChat()
            // Live position the previous round left: prompt + generated − 1 (the stop token is sampled, not fed).
            var livePosition: Int?
            for (turnIndex, question) in conversation.turns.enumerated() {
                let label = "\(conversation.name)#\(repeatIndex) turn \(turnIndex + 1)"
                logLine("\(label) ask: \(question)")
                model.promptText = question
                let started = Date()
                model.run()
                _ = await waitUntil(timeout: 3_600) { !model.isRunning }
                let wall = Date().timeIntervalSince(started)
                let records = client.take()
                let web = webStore?.takeCounts()

                var checks: [String: Bool] = [:]
                var liveNotes: [String] = []
                var progressNotes: [String] = []
                for (roundIndex, record) in records.enumerated() {
                    let d = record.diagnostics
                    let cached = d?.cachedPromptTokens ?? 0
                    let prompt = d?.promptTokenCount ?? 0
                    let generated = d?.generatedTokens ?? 0
                    var shortfall: Int?
                    if let expected = livePosition {
                        shortfall = expected - cached
                        if shortfall != 0 {
                            liveNotes.append("round \(roundIndex + 1) cached \(cached) of live \(expected)")
                        }
                    }
                    var row: [String: Any] = [
                        "conversation": conversation.name, "repeat": repeatIndex, "turn": turnIndex + 1,
                        "round": roundIndex + 1, "outcome": record.outcome,
                        "history": record.request.history.count, "continuation": record.request.continuation.count,
                        "tools": record.request.tools.count, "choice": record.request.toolChoice.rawValue,
                        "prompt": prompt, "cached": cached, "generated": generated,
                        "calls": record.toolCalls.map { "\($0.name) \($0.argumentsJSON)" },
                        // Epoch seconds, to line up with `GUARD_LOG` (Scripts/qwen38/guarded.sh).
                        "started": record.started.timeIntervalSince1970,
                        "ended": record.ended?.timeIntervalSince1970 ?? NSNull(),
                    ]
                    if let shortfall { row["shortfall"] = shortfall }
                    if let d {
                        row["stop"] = d.stopReason.rawValue
                        row["prefill_s"] = d.prefillSeconds ?? 0
                        row["decode_s"] = d.decodeSeconds
                        row["tok_s"] = d.tokensPerSecond
                        if let spec = d.speculative { row["draft"] = "\(spec.accepted)/\(spec.proposed)" }
                    }
                    if let error = record.error { row["error"] = error }
                    let times = record.prefill.map(\.seconds)
                    let gaps = zip(times.dropFirst(), times).map { $0 - $1 }
                    row["prefill_events"] = record.prefill.count
                    if let first = record.prefill.first {
                        row["prefill_first_s"] = first.seconds
                        row["prefill_first_done"] = first.done
                        row["prefill_max_gap_s"] = gaps.max() ?? 0
                    }
                    if let seconds = d?.prefillSeconds, seconds > 5 {
                        let first = record.prefill.first
                        if first == nil || first!.seconds > 1 || first!.done != cached || (gaps.max() ?? 0) > 5
                            || record.prefill.last.map({ $0.done != $0.total }) ?? true {
                            progressNotes.append(String(
                                format: "round %d prefill %.1f s: %d events, first %@ at %@, max gap %.1f s",
                                roundIndex + 1, seconds, record.prefill.count,
                                first.map { "\($0.done)/\($0.total)" } ?? "-",
                                first.map { String(format: "%.1f s", $0.seconds) } ?? "-", gaps.max() ?? 0))
                        }
                    }
                    jsonLine(row, to: rounds)
                    logLine("\(label) round \(roundIndex + 1): \(record.outcome) prompt=\(prompt) cached=\(cached) "
                        + "generated=\(generated) shortfall=\(shortfall.map(String.init) ?? "-") calls=\(row["calls"]!)")
                    // A round that did not finish leaves nothing to continue from.
                    livePosition = record.outcome == "finished" && prompt > 0 ? prompt + generated - 1 : nil
                }
                checks["live"] = liveNotes.isEmpty
                checks["progress"] = progressNotes.isEmpty
                checks["error"] = model.error == nil
                let answer = model.outputText
                checks["answer"] = !answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    && !answer.contains("<tool_call>") && !answer.contains("<function=")
                let trace = model.outputToolTrace
                let grounding = AppAnswerGrounding.of(trace)
                if model.effectiveNetworkMode == .online {
                    let fetchTried = trace.contains { $0.name == "fetch_page" }
                    checks["online"] = grounding.webSearches > 0 && fetchTried
                        && AppAnswerGrounding.citesSources(answer)
                }
                let failed = checks.filter { !$0.value }.map(\.key).sorted()
                if !failed.isEmpty {
                    failures.append("\(label): \(failed.joined(separator: ", "))"
                        + (liveNotes.isEmpty ? "" : " [\(liveNotes.joined(separator: "; "))]")
                        + (progressNotes.isEmpty ? "" : " [\(progressNotes.joined(separator: "; "))]")
                        + (model.error.map { " error: \($0.userMessage)" } ?? ""))
                }
                let japanese = answer.unicodeScalars.filter { (0x3040...0x30FF).contains($0.value) || (0x4E00...0x9FFF).contains($0.value) }.count
                jsonLine([
                    "conversation": conversation.name, "repeat": repeatIndex, "turn": turnIndex + 1,
                    "question": question, "answer": answer, "wall_s": wall, "rounds": records.count,
                    "trace": trace.map { ["name": $0.name, "subject": $0.subject, "status": $0.status.rawValue,
                                          "summary": $0.summary] },
                    "web_searches": grounding.webSearches, "pages_read": grounding.pagesRead,
                    "wikipedia_steps": grounding.wikipediaSteps, "cites": AppAnswerGrounding.citesSources(answer),
                    "japanese_ratio": answer.isEmpty ? 0 : Double(japanese) / Double(answer.count),
                    "checks": checks, "error": model.error?.userMessage ?? NSNull(),
                    "web_replayed": web.map { $0.replayed } ?? NSNull(), "web_recorded": web.map { $0.recorded } ?? NSNull(),
                    "continuation": model.outputContinuationTurns.map { turn -> [String: Any] in
                        ["role": turn.role.rawValue, "text": turn.text, "name": turn.toolName ?? "",
                         "calls": turn.toolCalls.map { "\($0.name) \($0.argumentsJSON)" }]
                    },
                ], to: turns)
                summary.append(String(format: "%@: %d rounds, %.0f s, %@", label, records.count, wall,
                                      failed.isEmpty ? "ok" : "FAIL " + failed.joined(separator: ","))
                    + (web.map { ", web replayed \($0.replayed) recorded \($0.recorded)" } ?? ""))
                logLine(summary.last!)
                if model.error != nil { break }
            }
        }
    }
    try? rounds.close()
    try? turns.close()
    print(summary.joined(separator: "\n"))
    print(failures.isEmpty ? "all checks passed" : "FAILURES:\n" + failures.joined(separator: "\n"))
    return failures.isEmpty ? 0 : 1
}

let status = await runCheck()
exit(status)
