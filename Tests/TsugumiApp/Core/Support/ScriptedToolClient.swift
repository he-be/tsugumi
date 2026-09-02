import Foundation
import Synchronization
@testable import TsugumiAppCore

/// Plays a script of rounds: each round is either tool calls (the
/// generation ends on `toolCalls`) or answer text. Records every request so
/// a test can read what the next round was sent.
final class ScriptedToolClient: AppInferenceClient, @unchecked Sendable {
    enum Round: Sendable {
        case calls([AppToolCall], text: String = "", reasoning: String = "")
        case answer(String, reasoning: String = "")
    }

    private let rounds: Mutex<[Round]>
    private let log = Mutex<[AppGenerationRequest]>([])
    private let active = Mutex<Task<Void, Never>?>(nil)
    let delayNanos: UInt64

    init(_ rounds: [Round], delayNanos: UInt64 = 1_000_000) {
        self.rounds = Mutex(rounds)
        self.delayNanos = delayNanos
    }

    var requests: [AppGenerationRequest] { log.withLock { $0 } }

    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            do {
                try request.validate(requireModelDirectory: false)
            } catch {
                let appError = error as? AppInferenceError ?? .unknown("\(error)")
                continuation.yield(.failed(appError, partial: nil))
                continuation.finish(throwing: appError)
                return
            }
            log.withLock { $0.append(request) }
            let round: Round = rounds.withLock { script in
                script.isEmpty ? .answer("(script exhausted)") : script.removeFirst()
            }
            let task = Task { [delayNanos] in
                try? await Task.sleep(nanoseconds: delayNanos)
                continuation.yield(.prefillProgress(done: 1, total: 1))
                func diagnostics(_ stop: AppStopReason, tokens: Int) -> AppDiagnostics {
                    AppDiagnostics(generatedTokens: tokens, stopReason: stop,
                                   promptTokenCount: 1, timeToFirstTokenSeconds: nil,
                                   decodeSeconds: 0.01, tokensPerSecond: 0,
                                   peakMemoryBytes: nil, runtimeOptions: request.runtimeOptions)
                }
                switch round {
                case .calls(let calls, let text, let reasoning):
                    if !reasoning.isEmpty {
                        continuation.yield(.token(AppTokenEvent(
                            index: 0, textDelta: "", elapsedDecodeSeconds: 0,
                            reasoningDelta: reasoning)))
                    }
                    if !text.isEmpty {
                        continuation.yield(.token(AppTokenEvent(
                            index: 0, textDelta: text, elapsedDecodeSeconds: 0)))
                    }
                    for call in calls { continuation.yield(.toolCall(call)) }
                    if Task.isCancelled {
                        continuation.yield(.cancelled(diagnostics(.cancelled, tokens: 1)))
                        continuation.finish(throwing: AppInferenceError.cancelled)
                        return
                    }
                    continuation.yield(.finished(diagnostics(.toolCalls, tokens: 1)))
                case .answer(let text, let reasoning):
                    if !reasoning.isEmpty {
                        continuation.yield(.token(AppTokenEvent(
                            index: 0, textDelta: "", elapsedDecodeSeconds: 0,
                            reasoningDelta: reasoning)))
                    }
                    continuation.yield(.token(AppTokenEvent(
                        index: 0, textDelta: text, elapsedDecodeSeconds: 0.01)))
                    continuation.yield(.finished(diagnostics(.eos, tokens: 1)))
                }
                continuation.finish()
            }
            active.withLock { $0 = task }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    func cancel() {
        active.withLock { $0?.cancel() }
    }
}

/// Answers each call from a table, optionally after a delay, and records
/// the calls it saw.
final class ScriptedToolExecutor: AppToolExecutor, @unchecked Sendable {
    let definitions: [AppToolDefinition] = [
        AppToolDefinition(name: "web_search", description: "search",
                          parametersJSON: #"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#),
        AppToolDefinition(name: "fetch_page", description: "fetch",
                          parametersJSON: #"{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}"#),
    ]
    private let results: [String: AppToolResult]
    private let seen = Mutex<[AppToolCall]>([])
    let delayNanos: UInt64
    /// What `lookups` hands the app before the first round (call ids are
    /// renumbered under the prefix the app asks for); empty for nothing.
    let seeds: [AppToolLookup]
    private let lookedUp = Mutex<[String]>([])

    init(results: [String: AppToolResult], delayNanos: UInt64 = 0, seeds: [AppToolLookup] = []) {
        self.results = results
        self.delayNanos = delayNanos
        self.seeds = seeds
    }

    var lookups: [String] { lookedUp.withLock { $0 } }

    func lookups(prompt: String, callIDPrefix: String) async -> [AppToolLookup] {
        lookedUp.withLock { $0.append(prompt) }
        return seeds.enumerated().map { index, seed in
            var seed = seed
            seed.call.id = "\(callIDPrefix)\(index + 1)"
            return seed
        }
    }

    var calls: [AppToolCall] { seen.withLock { $0 } }

    func execute(_ call: AppToolCall) async -> AppToolResult {
        seen.withLock { $0.append(call) }
        if delayNanos > 0 { try? await Task.sleep(nanoseconds: delayNanos) }
        return results[call.id] ?? AppToolResult(content: "error: unscripted", isError: true,
                                                 summary: "unscripted")
    }

    func subject(of call: AppToolCall) -> String {
        call.stringArgument("query") ?? call.stringArgument("url") ?? call.argumentsJSON
    }
}
