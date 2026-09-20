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
//                                         [--web-store DIR] [--max-rounds N] [--thinking on|off]
//                                         [--endpoint URL --remote-model ID [--remote-direct]] [--replay RUN_DIR]
//                                         [--search-budget N] [--pin-search] [--sample-forced TOOL:N]
//                                         [--stop-after-round K] [--cancel-at-tokens N] [--section-embed DIR]
//                                         [--gather DIR] [--then-models DIR,DIR] [--server-args "A B"]
//
// `--web-store DIR` answers the web tools' HTTP requests from DIR and records the ones it does not have
// (`RecordedHTTPTransport`), so a second run reads the same search results and pages (`docs/qwen38/21` §4 E-1).
//
// `--endpoint URL --remote-model ID` runs the model on a llama-server behind llama-swap instead of this Mac
// (`RemoteInferenceClient`). `--model` then only picks the kind — the prompts, sampler and call syntax — and is not
// loaded; the context defaults to the server's slot. The `live` and `progress` checks are about this Mac's session
// and are skipped. `--remote-direct` says the endpoint is a bare llama-server, whose `props`, `tokenize` and
// `apply-template` sit at the root instead of under llama-swap's `/upstream/ID/`. `--max-rounds` overrides the saved
// tool round budget for this run only.
//
// `--replay RUN_DIR` plays a recorded run back without the model (`ReplayInferenceClient`): each round returns the
// recorded calls, the app runs them, and each round's prompt is written to `--out`/prompts/ (docs/qwen38/32). Only the
// first repeat's first turn of each conversation is replayed.
//
// Metered searches (docs/qwen38/32 §5), with `--web-store`:
//   `--search-budget N`   at most N searches reach Serper / Brave in this run; a recorded one is not counted.
//   `--pin-search`        every search of a conversation gets the first result it got (`PinnedSearchTransport`),
//                         whatever the query. A conversation's `pin` shares another's result, and `searchOrder`
//                         (1-based) re-ranks it.
// `--sample-forced TOOL:N` (with `--endpoint`) draws N seeded calls at each forced TOOL round into `samples.jsonl`.
// `--section-embed DIR` (a Ruri v3 Core ML directory) makes each Wikipedia search carry the sections of the found
// articles closest to the question (R1-d, docs/qwen38/39); the settings file is not written.
// `--gather DIR` (a Ruri v3 Core ML directory, Offline) runs the turns without tools: the model writes search terms or
// an answer under a grammar, and the app hands over the closest chunks of the found articles (`GatherLoop.swift`,
// docs/qwen38/40). The app's own tool loop is not used. With `--endpoint` the model is the llama-server; without, the
// app's Qwen3.8 session in this process (docs/qwen38/42), each round's disk reads and swap-outs in `rounds.jsonl` and,
// with `Q38_ROUND_LOG=FILE`, the runner's per-chunk breakdown in FILE.
// `--gather-force FILE` (with `--gather`) makes round 1 of each listed conversation's first turn write the recorded
// text: `{"<conversation>": "調査\n- ...\n"}` (docs/qwen38/42 §3-1).
// `--gather-answer quotes` (with `--gather`): the model writes one sentence and sentence numbers, and the loop copies
// those sentences into the answer (docs/qwen38/47). `written` (the default) has the model write the whole answer.
// `--stop-after-round K` ends a turn after K rounds: round K+1 answers empty without the model, and the turn's checks
// are skipped.
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
    /// A Ruri v3 Core ML directory for this run only: Wikipedia searches carry the near sections (R1-d, docs/qwen38/39).
    var sectionEmbed: String?
    /// A Ruri v3 Core ML directory: run the turns in the gather mode (docs/qwen38/40) instead of the app's tool loop.
    var gather: String?
    /// With `--gather`: score the cases of this file without the model (`runGatherProbe`).
    var gatherProbe: String?
    /// With `--gather-probe`: write each case's gathered text (what the loop would hand over) to this JSONL file
    /// instead of printing the ranking (docs/qwen38/41).
    var gatherProbeOut: String?
    /// With `--gather`: how the gatherer finds articles, `question,qfirst=K,titles,articles=N` (docs/qwen38/41). Empty is 40's.
    var gatherSearch = ""
    /// With `--gather`: round 1's text by conversation (docs/qwen38/42 §3-1).
    var gatherForce: String?
    /// With `--gather`: `--gather-answer quotes` has the model write one sentence and sentence numbers, and the loop
    /// copies the numbered sentences into the answer (docs/qwen38/47). Default: the model writes the whole answer.
    var gatherQuotes = false
    var webStore: String?
    var maxRounds: Int?
    var thinking: Bool?
    var endpoint: URL?
    var remoteModel: String?
    /// `--remote-direct`: `--endpoint` is a bare llama-server, not llama-swap (`RemoteInferenceClient.Routing`).
    var remoteDirect = false
    var replay: String?
    var searchBudget: Int?
    var pinSearch = false
    var sampleForced: [String: Int] = [:]
    var stopAfterRound: Int?
    /// `--cancel-at-tokens N`: the first turn of each conversation is cancelled (the app's Stop) once N tokens of text
    /// have streamed; the turn's check is then `cancel` — the app was idle within 10 s with no error — and the next
    /// turn has to answer as usual (docs/qwen38-27b/10 §3-1 3).
    var cancelAtTokens: Int?
    /// `--then-models DIR,DIR`: after the conversations, select each directory in turn as the app's model menu does,
    /// load it and ask one question without tools. The check is that each answers and that a llama-server runs only
    /// while its kind is the selected one (docs/qwen38-27b/10 S5).
    var thenModels: [String] = []
    /// `--server-args "A B"`: arguments added to the child llama-server's (`LlamaServerInferenceClient`).
    var serverArguments: [String] = []
    /// `--extract-endpoint URL --extract-model ID` (Online): the web tools answer with a light model's extract of
    /// the pages (`ExtractLoop.swift`, docs/qwen38/48).
    var extractEndpoint: URL?
    var extractModel: String?
    var extractPages = 3
    var extractPageCharacters = 8000
    var extractMaxTokens = 1024
    /// `--extract-launch SCRIPT`: start the light model's server for each tool call and stop it after
    /// (`ExtractorLauncher`); `--extract-prefetch A,B` are the weight files read into the page cache first.
    var extractLaunch: String?
    var extractPrefetch: [String] = []

    init(_ arguments: [String]) {
        var iterator = arguments.dropFirst().makeIterator()
        while let flag = iterator.next() {
            if flag == "--pin-search" {
                pinSearch = true
                continue
            }
            if flag == "--remote-direct" {
                remoteDirect = true
                continue
            }
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
            case "--section-embed": sectionEmbed = NSString(string: value).expandingTildeInPath
            case "--gather": gather = NSString(string: value).expandingTildeInPath
            case "--gather-probe": gatherProbe = value
            case "--gather-probe-out": gatherProbeOut = value
            case "--gather-search": gatherSearch = value
            case "--gather-force": gatherForce = value
            case "--gather-answer":
                guard value == "quotes" || value == "written" else {
                    FileHandle.standardError.write(Data("--gather-answer is quotes or written\n".utf8))
                    exit(2)
                }
                gatherQuotes = value == "quotes"
            case "--web-store": webStore = value
            case "--max-rounds": maxRounds = Int(value)
            case "--thinking": thinking = value == "on"
            case "--endpoint": endpoint = URL(string: value)
            case "--remote-model": remoteModel = value
            case "--replay": replay = value
            case "--search-budget": searchBudget = Int(value)
            case "--sample-forced":
                let parts = value.split(separator: ":")
                if parts.count == 2, let count = Int(parts[1]) { sampleForced[String(parts[0])] = count }
            case "--stop-after-round": stopAfterRound = Int(value)
            case "--cancel-at-tokens": cancelAtTokens = Int(value)
            case "--server-args": serverArguments = value.split(separator: " ").map(String.init)
            case "--then-models":
                thenModels = value.split(separator: ",").map { NSString(string: String($0)).expandingTildeInPath }
            case "--extract-endpoint": extractEndpoint = URL(string: value)
            case "--extract-model": extractModel = value
            case "--extract-pages": extractPages = Int(value) ?? 3
            case "--extract-page-chars": extractPageCharacters = Int(value) ?? 8000
            case "--extract-max-tokens": extractMaxTokens = Int(value) ?? 1024
            case "--extract-launch": extractLaunch = NSString(string: value).expandingTildeInPath
            case "--extract-prefetch":
                extractPrefetch = value.split(separator: ",").map { NSString(string: String($0)).expandingTildeInPath }
            default:
                FileHandle.standardError.write(Data("unknown flag \(flag)\n".utf8))
                exit(2)
            }
        }
        if out.isEmpty {
            FileHandle.standardError.write(Data("--out DIR is required\n".utf8))
            exit(2)
        }
        if gather != nil, gatherProbe == nil, network != .offline || sectionEmbed != nil || replay != nil {
            FileHandle.standardError.write(Data("--gather needs --network offline, without --section-embed or --replay\n".utf8))
            exit(2)
        }
        if gatherQuotes, gather == nil {
            FileHandle.standardError.write(Data("--gather-answer goes with --gather\n".utf8))
            exit(2)
        }
        if gatherForce != nil, gather == nil {
            FileHandle.standardError.write(Data("--gather-force goes with --gather\n".utf8))
            exit(2)
        }
        if (extractEndpoint == nil) != (extractModel == nil) {
            FileHandle.standardError.write(Data("--extract-endpoint and --extract-model go together\n".utf8))
            exit(2)
        }
        if (endpoint == nil) != (remoteModel == nil) {
            FileHandle.standardError.write(Data("--endpoint and --remote-model go together\n".utf8))
            exit(2)
        }
    }
}

struct Conversation: Decodable {
    let name: String
    let turns: [String]
    /// `--pin-search`: the pin whose search this conversation reads (default: its name).
    var pin: String?
    /// `--pin-search`: the pinned results re-ranked, 1-based.
    var searchOrder: [Int]?
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

/// The session's client (`RealInferenceClient`, or `RemoteInferenceClient` with `--endpoint`), recording each generation.
final class RecordingClient: AppModelLifecycleClient, AppInferenceRuntimeReporting, @unchecked Sendable {
    private let inner: any AppModelLifecycleClient & AppInferenceRuntimeReporting
    private let lock = NSLock()
    private var records: [RoundRecord] = []
    /// `--stop-after-round`: rounds past this many end the turn without the model.
    var stopAfterRound: Int?

    init(inner: any AppModelLifecycleClient & AppInferenceRuntimeReporting) {
        self.inner = inner
    }

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

    /// Rounds started in the current turn (the 1-based number of the running one).
    var roundCount: Int {
        lock.lock(); defer { lock.unlock() }
        return records.count
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
        if let stopAfterRound, index >= stopAfterRound {
            // `--stop-after-round`: the round past the cut ends the turn at once with an empty answer, without the
            // model, so the turn finishes on the app's own path (a cancel races the tools between rounds).
            update(index) { $0.outcome = "stopped"; $0.ended = Date() }
            return AsyncThrowingStream { continuation in
                continuation.yield(.finished(AppDiagnostics(
                    generatedTokens: 0, stopReason: .endOfTurn, promptTokenCount: nil, cachedPromptTokens: nil,
                    speculative: nil, prefillSeconds: nil, timeToFirstTokenSeconds: nil, decodeSeconds: 0,
                    tokensPerSecond: 0, peakMemoryBytes: nil, runtimeOptions: request.runtimeOptions)))
                continuation.finish()
            }
        }
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

/// `samples.jsonl` for `--sample-forced`: each forced round's draws, tagged with the turn being run.
final class SampleLog: @unchecked Sendable {
    private let handle: FileHandle?
    private let lock = NSLock()
    private var current: [String: Any] = [:]

    init(handle: FileHandle?) {
        self.handle = handle
    }

    var context: [String: Any] {
        get { lock.lock(); defer { lock.unlock() }; return current }
        set { lock.lock(); defer { lock.unlock() }; current = newValue }
    }

    func write(_ fields: [String: Any]) {
        guard let handle else { return }
        lock.lock(); defer { lock.unlock() }
        jsonLine(current.merging(fields) { $1 }, to: handle)
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

    var remote: RemoteInferenceClient?
    if let endpoint = options.endpoint, let remoteModel = options.remoteModel {
        guard let kind = AppModelKind.probe(modelDirectory: URL(fileURLWithPath: options.model)) else {
            logLine("--model \(options.model) names no model kind (it picks the prompts and call syntax for --endpoint)")
            return 2
        }
        remote = RemoteInferenceClient(endpoint: endpoint, modelID: remoteModel, dialect: .init(kind: kind),
                                       routing: options.remoteDirect ? .direct : .llamaSwap)
    } else if !options.sampleForced.isEmpty {
        logLine("--sample-forced needs --endpoint")
        return 2
    }
    var replay: ReplayInferenceClient?
    if let run = options.replay {
        do {
            replay = try ReplayInferenceClient(run: URL(fileURLWithPath: run, isDirectory: true), out: outDirectory)
        } catch {
            logLine("cannot replay \(run): \(error)")
            return 2
        }
    }
    let inner: any AppModelLifecycleClient & AppInferenceRuntimeReporting
    var real: RealInferenceClient?
    // A kind that runs in a llama-server goes through the app's own router, which starts and stops the server
    // (`KindRoutingInferenceClient`, docs/qwen38-27b/10 S4-S5).
    var childServer: LlamaServerInferenceClient?
    if let replay { inner = replay } else if let remote { inner = remote } else {
        real = RealInferenceClient()
        let directories = [options.model] + options.thenModels
        if directories.contains(where: {
            AppModelKind.probe(modelDirectory: URL(fileURLWithPath: $0))?.runsOnLlamaServer == true
        }) {
            childServer = LlamaServerInferenceClient(
                stateDirectory: outDirectory.appendingPathComponent("llama-server", isDirectory: true),
                extraArguments: options.serverArguments)
            inner = KindRoutingInferenceClient(engine: real!, llamaServer: childServer!)
        } else {
            inner = real!
        }
    }
    let client = RecordingClient(inner: inner)
    client.stopAfterRound = options.stopAfterRound
    var webStore: RecordedHTTPTransport?
    var budget: SearchBudgetTransport?
    var pinned: PinnedSearchTransport?
    var webTransport: (any HTTPTransport)?
    if let path = options.webStore {
        do {
            let directory = URL(fileURLWithPath: path, isDirectory: true)
            let store = try RecordedHTTPTransport(directory: directory)
            webStore = store
            webTransport = store
            if let limit = options.searchBudget {
                budget = SearchBudgetTransport(budget: limit, store: store)
                webTransport = budget
            }
            if options.pinSearch {
                pinned = try PinnedSearchTransport(directory: directory, inner: webTransport!)
                webTransport = pinned
            }
        } catch {
            logLine("cannot use --web-store \(path): \(error)")
            return 2
        }
    } else if options.searchBudget != nil || options.pinSearch {
        logLine("--search-budget and --pin-search need --web-store")
        return 2
    }
    var toolExecutorProvider: ((WebSearchConfiguration, AppNetworkMode) throws -> (any AppToolExecutor)?)? =
        webTransport.map { transport in
            { try AppModel.makeToolExecutor(configuration: $0, mode: $1, transport: transport) }
        }
    if let extractEndpoint = options.extractEndpoint, let extractModel = options.extractModel {
        let extractor = PageExtractor(endpoint: extractEndpoint, modelID: extractModel,
                                      pageCharacters: options.extractPageCharacters, maxTokens: options.extractMaxTokens,
                                      log: outDirectory.appendingPathComponent("extract.jsonl"))
        if let script = options.extractLaunch {
            extractor.launcher = ExtractorLauncher(script: script, prefetch: options.extractPrefetch, endpoint: extractEndpoint)
        }
        let transport: any HTTPTransport = webTransport ?? URLSessionTransport()
        let pages = options.extractPages
        toolExecutorProvider = { configuration, mode in
            guard mode == .online else {
                return try AppModel.makeToolExecutor(configuration: configuration, mode: mode, transport: transport)
            }
            let resolved = configuration.resolved()
            guard resolved.canSearch else {
                throw AppInferenceError.invalidRequest("Online needs a Serper or Brave API key.")
            }
            // Wikipedia などの他のツールは Offline と同じ形で作り、Web の 2 つだけを差し替える
            let base = try AppModel.makeToolExecutor(configuration: configuration, mode: .offline, transport: transport)
            return ExtractingWebExecutor(base: base,
                                         web: WebSearchToolExecutor(configuration: resolved, transport: transport),
                                         extractor: extractor, pagesPerSearch: pages)
        }
        logLine("extract: \(extractModel) at \(extractEndpoint.absoluteString) pages=\(pages) "
            + "page_chars=\(options.extractPageCharacters) max_tokens=\(options.extractMaxTokens)")
    }
    let model = AppModel(modelDirectory: URL(fileURLWithPath: options.model), client: client,
                         turnMetricsLog: AppTurnMetricsLog(fileURL: outDirectory.appendingPathComponent("turn-metrics.jsonl")),
                         webSearchConfigurationURL: WebSearchConfigurationStore.defaultFileURL,
                         personaURL: AppPersonaStore.defaultFileURL,
                         toolExecutorProvider: toolExecutorProvider)
    if let context = options.context { model.maxContextTokens = context }
    if let sectionEmbed = options.sectionEmbed {
        model.webSearchConfiguration.sectionEmbeddingPath = sectionEmbed
    }
    if let pageCharacters = options.pageCharacters {
        model.webSearchConfiguration.pageCharacterLimit = pageCharacters
    }
    if let maxRounds = options.maxRounds { model.webSearchConfiguration.maxToolRounds = maxRounds }
    if let thinking = options.thinking { model.thinkingEnabled = thinking }
    model.networkMode = options.network
    if let probe = options.gatherProbe, let gather = options.gather {
        return runGatherProbe(file: probe, gatherDirectory: gather, model: model, out: options.gatherProbeOut,
                              searchSpec: options.gatherSearch)
    }
    logLine("model \(model.selectedModelKind.rawValue) context=\(model.maxContextTokens) mtp=\(model.runtimeOptions.mtpEnabled) "
        + "thinking=\(model.thinkingEnabled) network=\(model.effectiveNetworkMode.rawValue) "
        + "rounds=\(model.webSearchConfiguration.resolved().maxToolRounds) page=\(model.webSearchConfiguration.resolved().pageCharacterLimit)")
    if let remote {
        do {
            try await remote.prepare()
        } catch {
            logLine("remote \(remote.modelID) at \(remote.endpoint.absoluteString) is not usable: \(error)")
            return 2
        }
        let slot = remote.serverContext ?? 0
        if options.context == nil, slot > 0 {
            model.maxContextTokens = slot
        } else if slot > 0, model.maxContextTokens > slot {
            logLine("warning: --context \(model.maxContextTokens) is over the server slot's \(slot)")
        }
        logLine("remote \(remote.modelID) at \(remote.endpoint.absoluteString) dialect=\(remote.dialect.rawValue) "
            + "slot=\(slot) context=\(model.maxContextTokens)")
    }
    guard model.canLoadModel else {
        logLine("cannot load \(model.modelPathText): installation \(model.installationStatus) state \(model.loadState)")
        return 2
    }
    model.loadModel()
    guard await waitUntil(timeout: 600, { model.loadState.isReady || model.error != nil }),
          model.loadState.isReady else {
        logLine("load failed: \(model.loadState) \(String(describing: model.error))")
        return 2
    }

    if let gather = options.gather {
        let generate: GatherGenerate
        if let remote {
            generate = { try await remote.constrained(messages: $0, grammar: $1, sampling: $2, maxTokens: $3) }
        } else if let real {
            generate = { try await real.constrained(messages: $0, grammar: $1, sampling: $2, maxTokens: $3) }
        } else {
            logLine("--gather needs --endpoint or this Mac's session")
            return 2
        }
        return await runGather(options: options, gatherDirectory: gather, model: model, generate: generate,
                               local: remote == nil, conversations: conversations, outDirectory: outDirectory)
    }
    let rounds = appendHandle(outDirectory.appendingPathComponent("rounds.jsonl"))
    let turns = appendHandle(outDirectory.appendingPathComponent("turns.jsonl"))
    let sampleLog = SampleLog(handle: options.sampleForced.isEmpty
        ? nil : appendHandle(outDirectory.appendingPathComponent("samples.jsonl")))
    if let remote {
        remote.forcedSamples = options.sampleForced
        remote.onForcedSamples = { tool, calls in
            sampleLog.write(["tool": tool, "round": client.roundCount, "calls": calls])
        }
    }
    var failures: [String] = []
    var summary: [String] = []

    for repeatIndex in 1...max(options.repeats, 1) {
        for conversation in conversations {
            model.newChat()
            replay?.start(conversation: conversation.name)
            pinned?.start(pin: conversation.pin ?? conversation.name, order: conversation.searchOrder)
            // Live position the previous round left: prompt + generated − 1 (the stop token is sampled, not fed).
            var livePosition: Int?
            for (turnIndex, question) in conversation.turns.enumerated() {
                let label = "\(conversation.name)#\(repeatIndex) turn \(turnIndex + 1)"
                logLine("\(label) ask: \(question)")
                model.promptText = question
                sampleLog.context = ["conversation": conversation.name, "repeat": repeatIndex, "turn": turnIndex + 1]
                let started = Date()
                model.run()
                var cancelSeconds: Double?
                if let at = options.cancelAtTokens, turnIndex == 0 {
                    _ = await waitUntil(timeout: 3_600) { !model.isRunning || model.liveTokenCount >= at }
                    if model.isRunning {
                        let issued = Date()
                        logLine("\(label) cancel at \(model.liveTokenCount) tokens")
                        model.cancel()
                        _ = await waitUntil(timeout: 3_600) { !model.isRunning }
                        cancelSeconds = Date().timeIntervalSince(issued)
                    }
                }
                _ = await waitUntil(timeout: 3_600) { !model.isRunning }
                let wall = Date().timeIntervalSince(started)
                let records = client.take()
                // A turn cut on purpose (`--stop-after-round`, `--cancel-at-tokens`) has no answer to check.
                let stopped = records.contains { $0.outcome == "stopped" } || cancelSeconds != nil
                let web = webStore?.takeCounts()
                let pins = pinned?.takeCounts()

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
                if let cancelSeconds {
                    // The app reports a Stop as `.cancelled`; anything else is a failure of the cancel itself.
                    checks["cancel"] = cancelSeconds < 10 && model.error == .cancelled
                    logLine("\(label) idle \(String(format: "%.2f", cancelSeconds)) s after cancel")
                } else if stopped {
                    // Cut on purpose (`--stop-after-round`): there is no answer to check.
                } else if remote == nil && replay == nil && !model.selectedModelKind.runsOnLlamaServer {
                    checks["live"] = liveNotes.isEmpty
                    checks["progress"] = progressNotes.isEmpty
                }
                let answer = model.outputText
                if !stopped {
                    checks["error"] = model.error == nil
                    checks["answer"] = !answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        && !answer.contains("<tool_call>") && !answer.contains("<function=")
                        && !answer.contains("<|tool_call>")
                }
                let trace = model.outputToolTrace
                let grounding = AppAnswerGrounding.of(trace)
                if model.effectiveNetworkMode == .online, !stopped {
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
                    "search_pinned": pins.map { $0.pinned } ?? NSNull(),
                    "search_fetched": pins.map { $0.fetched } ?? NSNull(),
                    "search_budget_used": budget.map { $0.used } ?? NSNull(),
                    "stopped": stopped,
                    "continuation": model.outputContinuationTurns.map { turn -> [String: Any] in
                        ["role": turn.role.rawValue, "text": turn.text, "name": turn.toolName ?? "",
                         "calls": turn.toolCalls.map { "\($0.name) \($0.argumentsJSON)" }]
                    },
                ], to: turns)
                summary.append(String(format: "%@: %d rounds, %.0f s, %@", label, records.count, wall,
                                      stopped ? "stopped" : failed.isEmpty ? "ok" : "FAIL " + failed.joined(separator: ","))
                    + (web.map { ", web replayed \($0.replayed) recorded \($0.recorded)" } ?? "")
                    + (pins.map { ", search pinned \($0.pinned) fetched \($0.fetched)" } ?? "")
                    + (budget.map { ", search budget \($0.used)/\($0.budget)" } ?? ""))
                logLine(summary.last!)
                // A cancelled turn is followed by the next one: that the server still answers is the point.
                if cancelSeconds == nil, model.error != nil || stopped { break }
            }
        }
    }
    try? rounds.close()
    try? turns.close()
    for path in options.thenModels {
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        let before = childServer?.serverProcessIdentifier
        model.setModelURL(directory)
        let label = "switch to \(model.selectedModelKind.rawValue)"
        if let context = options.context,
           model.selectedModelKind.contextOptions.contains(where: { $0.tokens == context }) {
            model.maxContextTokens = context
        }
        if let thinking = options.thinking { model.thinkingEnabled = thinking }
        model.networkMode = .modelOnly
        guard model.canLoadModel else {
            failures.append("\(label): cannot load \(path): \(model.installationStatus)")
            break
        }
        let started = Date()
        model.loadModel()
        guard await waitUntil(timeout: 600, { model.loadState.isReady || model.error != nil }),
              model.loadState.isReady else {
            failures.append("\(label): load failed: \(model.loadState)")
            break
        }
        let loadSeconds = Date().timeIntervalSince(started)
        let server = childServer?.serverProcessIdentifier
        let wantsServer = model.selectedModelKind.runsOnLlamaServer
        let previousGone = before.map { $0 == server || kill($0, 0) != 0 } ?? true
        model.newChat()
        model.promptText = "What is the capital of France? Answer in one sentence."
        model.run()
        _ = await waitUntil(timeout: 3_600) { !model.isRunning }
        _ = client.take()
        let answer = model.outputText.trimmingCharacters(in: .whitespacesAndNewlines)
        logLine("\(label): loaded in \(String(format: "%.1f", loadSeconds)) s, llama-server \(server.map(String.init) ?? "none"), "
            + "previous server \(before.map(String.init) ?? "none") gone=\(previousGone), answer: \(answer.prefix(120))")
        if model.error != nil || answer.isEmpty {
            failures.append("\(label): no answer\(model.error.map { " error: \($0.userMessage)" } ?? "")")
        }
        if (server != nil) != wantsServer { failures.append("\(label): llama-server \(server.map(String.init) ?? "none")") }
        if !previousGone { failures.append("\(label): the previous llama-server \(before!) is still running") }
        summary.append("\(label): ok=\(model.error == nil && !answer.isEmpty) server=\(server.map(String.init) ?? "none")")
    }
    if let childServer, let pid = childServer.serverProcessIdentifier {
        await inner.unload()
        let gone = kill(pid, 0) != 0
        logLine("child llama-server pid \(pid) after unload: \(gone ? "gone" : "STILL RUNNING")")
        if !gone { failures.append("child llama-server \(pid) outlived unload") }
    }
    print(summary.joined(separator: "\n"))
    print(failures.isEmpty ? "all checks passed" : "FAILURES:\n" + failures.joined(separator: "\n"))
    return failures.isEmpty ? 0 : 1
}

let status = await runCheck()
exit(status)
