import Foundation
import Tsugumi
import TsugumiAppCore

/// A recorded run played back through the app's tool loop without the model (`--replay RUN_DIR`, docs/qwen38/32).
///
/// Each round returns what the recorded round returned — its calls, or the recorded answer on the last round — so
/// `AppModel` runs the same calls through its own executors (the local Wikipedia is deterministic) and builds the
/// same requests. Every request is rendered as the Qwen3.8 session renders it (`RealInferenceClient.qwen38PromptTokens`)
/// and written to `DIR/prompts/<conversation>-r<round>.tokens`, with `prompts.jsonl` holding the rendered length next
/// to the recorded one. The session's prompt cache may realign the history to the generated split, so a length can
/// differ by a token or two from the recording; the tool results are compared afterwards from `turns.jsonl`.
///
/// With `engine` (`--replay-model`, docs/qfn-prefill/04 Q0), each request also goes through the app's own engine with
/// one new token before the recorded round is returned: the rounds stay the recorded ones while the prefill (prompt
/// cache and checkpoints included) is the engine's, in one process. The engine's diagnostics replace the zeros, so
/// `rounds.jsonl` carries its `prefill_s`, `cached` and `prompt`. What the engine wrote is dropped; the next prompt
/// holds the recorded text instead, so each round prefills that text again (the recording had it in the cache).
/// The engine's request says tool_choice none (the sampler forbids the call's start token, so one token cannot cut a
/// call short); the rendered prompt is checked to be the same as the request's own.
final class ReplayInferenceClient: AppModelLifecycleClient, AppInferenceRuntimeReporting, @unchecked Sendable {
    struct Round {
        var calls: [(name: String, arguments: String)]
        /// What the model wrote before its calls (from `turns.jsonl`'s continuation).
        var text = ""
        var prompt: Int
        var generated: Int
    }

    private let rounds: [String: [Round]]
    private let answers: [String: String]
    private let promptDirectory: URL
    private let index: FileHandle
    private let lock = NSLock()
    private var tokenizer: QwenTokenizer?
    private var conversation = ""
    private var position = 0
    private let engine: RealInferenceClient?

    var loadedRuntimeOwnBytes: UInt64? { engine?.loadedRuntimeOwnBytes }

    /// Reads the first repeat's first turn of each conversation in `run` (`rounds.jsonl`, `turns.jsonl`).
    init(run: URL, out: URL, engine: RealInferenceClient? = nil) throws {
        self.engine = engine
        func lines(_ name: String) throws -> [[String: Any]] {
            try String(contentsOf: run.appendingPathComponent(name), encoding: .utf8)
                .split(separator: "\n")
                .compactMap { try JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any] }
                .filter { ($0["repeat"] as? Int) == 1 && ($0["turn"] as? Int) == 1 }
        }
        var rounds: [String: [Round]] = [:]
        for row in try lines("rounds.jsonl") {
            let calls = (row["calls"] as? [String] ?? []).map { call -> (String, String) in
                let parts = call.split(separator: " ", maxSplits: 1)
                return (String(parts[0]), parts.count > 1 ? String(parts[1]) : "{}")
            }
            rounds[row["conversation"] as! String, default: []].append(
                Round(calls: calls, prompt: row["prompt"] as? Int ?? 0, generated: row["generated"] as? Int ?? 0))
        }
        var answers: [String: String] = [:]
        for row in try lines("turns.jsonl") {
            let name = row["conversation"] as! String
            answers[name] = row["answer"] as? String ?? ""
            // The model's turns in order; the app's own lookup turns come first and are not rounds.
            let spoken = (row["continuation"] as? [[String: Any]] ?? []).filter {
                ($0["role"] as? String) == "assistant"
                    && !(($0["calls"] as? [String]) ?? []).contains { $0.hasPrefix("wikipedia_lookup ") }
            }
            var turn = 0
            for i in (rounds[name] ?? []).indices where !(rounds[name]![i].calls.isEmpty) {
                guard turn < spoken.count else { break }
                rounds[name]![i].text = spoken[turn]["text"] as? String ?? ""
                turn += 1
            }
        }
        self.rounds = rounds
        self.answers = answers
        promptDirectory = out.appendingPathComponent("prompts", isDirectory: true)
        try FileManager.default.createDirectory(at: promptDirectory, withIntermediateDirectories: true)
        let indexURL = out.appendingPathComponent("prompts.jsonl")
        FileManager.default.createFile(atPath: indexURL.path, contents: nil)
        index = try FileHandle(forWritingTo: indexURL)
    }

    /// The recorded conversation the next rounds belong to.
    func start(conversation name: String) {
        lock.lock(); defer { lock.unlock() }
        conversation = name
        position = 0
    }

    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int, options: AppRuntimeOptions, forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        tokenizer = try await QwenTokenizer.load(forModelDirectory: modelDirectory)
        if let engine {
            try await engine.ensureLoaded(modelDirectory: modelDirectory, maxContextTokens: maxContextTokens,
                                          options: options, forceLogitsHead: forceLogitsHead, onState: onState)
        } else {
            onState(.ready(modelDirectory: modelDirectory, loadSeconds: 0))
        }
    }

    func unload() async { await engine?.unload() }

    func cancel() { engine?.cancel() }

    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        lock.lock()
        let name = conversation
        let roundIndex = position
        position += 1
        lock.unlock()
        let engine = self.engine
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    guard let tokenizer else { throw AppInferenceError.modelNotLoaded }
                    let recorded = rounds[name] ?? []
                    guard roundIndex < recorded.count else {
                        throw AppInferenceError.unknown("\(name): no recorded round \(roundIndex + 1)")
                    }
                    let round = recorded[roundIndex]
                    let tokens = try RealInferenceClient.qwen38PromptTokens(for: request, tokenizer: tokenizer)
                    let label = "\(name)-r\(roundIndex + 1)"
                    try tokens.map(String.init).joined(separator: ",")
                        .write(to: promptDirectory.appendingPathComponent("\(label).tokens"), atomically: true, encoding: .utf8)
                    jsonLine(["conversation": name, "round": roundIndex + 1, "file": "prompts/\(label).tokens",
                              "rendered": tokens.count, "recorded": round.prompt, "delta": tokens.count - round.prompt,
                              "recorded_calls": round.calls.map { "\($0.name) \($0.arguments)" },
                              "last": roundIndex == recorded.count - 1], to: index)

                    var measured: AppDiagnostics?
                    if let engine {
                        var prefill = request
                        prefill.maxNewTokens = 1
                        // One token that opens a tool call would fail the call parser; `none` forbids that token in
                        // the sampler. It must not change the prompt, which is checked here.
                        prefill.toolChoice = .none
                        guard try RealInferenceClient.qwen38PromptTokens(for: prefill, tokenizer: tokenizer) == tokens else {
                            throw AppInferenceError.unknown("\(label): tool_choice none renders a different prompt")
                        }
                        for try await event in engine.generate(prefill) {
                            switch event {
                            case .prefillProgress: continuation.yield(event)
                            case .finished(let d): measured = d
                            case .cancelled: throw AppInferenceError.cancelled
                            case .failed(let error, _): throw error
                            case .token, .toolCall: break
                            }
                        }
                        guard measured != nil else { throw AppInferenceError.unknown("\(label): engine gave no diagnostics") }
                    }

                    let text = round.calls.isEmpty ? answers[name] ?? "" : round.text
                    if !text.isEmpty {
                        continuation.yield(.token(AppTokenEvent(index: 0, textDelta: text, elapsedDecodeSeconds: 0)))
                    }
                    for (i, call) in round.calls.enumerated() {
                        continuation.yield(.toolCall(AppToolCall(id: "call_replay_\(roundIndex)_\(i)", name: call.name,
                                                                 argumentsJSON: call.arguments)))
                    }
                    let stop: AppStopReason = round.calls.isEmpty ? .endOfTurn : .toolCalls
                    if var measured {
                        measured.stopReason = stop
                        continuation.yield(.finished(measured))
                    } else {
                        continuation.yield(.finished(AppDiagnostics(
                            generatedTokens: round.generated, stopReason: stop,
                            promptTokenCount: tokens.count, cachedPromptTokens: 0, speculative: nil,
                            prefillSeconds: 0, timeToFirstTokenSeconds: nil, decodeSeconds: 0, tokensPerSecond: 0,
                            peakMemoryBytes: nil, runtimeOptions: request.runtimeOptions)))
                    }
                    continuation.finish()
                } catch {
                    continuation.yield(.failed(.unknown("\(error)"), partial: nil))
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
