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

    var loadedRuntimeOwnBytes: UInt64? { nil }

    /// Reads the first repeat's first turn of each conversation in `run` (`rounds.jsonl`, `turns.jsonl`).
    init(run: URL, out: URL) throws {
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
        onState(.ready(modelDirectory: modelDirectory, loadSeconds: 0))
    }

    func unload() async {}

    func cancel() {}

    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        lock.lock()
        let name = conversation
        let roundIndex = position
        position += 1
        lock.unlock()
        return AsyncThrowingStream { continuation in
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

                let text = round.calls.isEmpty ? answers[name] ?? "" : round.text
                if !text.isEmpty {
                    continuation.yield(.token(AppTokenEvent(index: 0, textDelta: text, elapsedDecodeSeconds: 0)))
                }
                for (i, call) in round.calls.enumerated() {
                    continuation.yield(.toolCall(AppToolCall(id: "call_replay_\(roundIndex)_\(i)", name: call.name,
                                                             argumentsJSON: call.arguments)))
                }
                continuation.yield(.finished(AppDiagnostics(
                    generatedTokens: round.generated,
                    stopReason: round.calls.isEmpty ? .endOfTurn : .toolCalls,
                    promptTokenCount: tokens.count, cachedPromptTokens: 0, speculative: nil,
                    prefillSeconds: 0, timeToFirstTokenSeconds: nil, decodeSeconds: 0, tokensPerSecond: 0,
                    peakMemoryBytes: nil, runtimeOptions: request.runtimeOptions)))
                continuation.finish()
            } catch {
                continuation.yield(.failed(.unknown("\(error)"), partial: nil))
                continuation.finish(throwing: error)
            }
        }
    }
}
