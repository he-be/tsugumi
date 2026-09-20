import Foundation

/// The app's one client, in front of the two ways a model runs: this project's engines behind the decode service, or
/// a `llama-server` the app starts (`AppModelKind.runsOnLlamaServer`, `docs/qwen38-27b/10` §2). The kind is read from
/// the directory a load names, and the other side is unloaded first: 18 GB does not hold two models.
public final class KindRoutingInferenceClient: AppModelLifecycleClient, AppInferenceRuntimeReporting,
                                                 AppInferenceMemoryReporting, AppInferenceTranscriptReporting,
                                                 @unchecked Sendable {
    public typealias Engine = AppModelLifecycleClient & AppInferenceRuntimeReporting

    private enum Route { case engine, llamaServer }

    private let engine: any Engine
    private let llamaServer: LlamaServerInferenceClient
    private let lock = NSLock()
    private var route = Route.engine

    /// The transcript the views read while a round streams. The decode service fills its own; a llama-server round is
    /// copied into the same one, so the view has one place to look whichever side answers.
    public let generationTranscriptMailbox: GenerationTranscriptMailbox

    public init(engine: any Engine, llamaServer: LlamaServerInferenceClient = LlamaServerInferenceClient()) {
        self.engine = engine
        self.llamaServer = llamaServer
        generationTranscriptMailbox = (engine as? any AppInferenceTranscriptReporting)?.generationTranscriptMailbox
            ?? GenerationTranscriptMailbox()
    }

    private var current: Route {
        lock.lock(); defer { lock.unlock() }
        return route
    }

    private func setRoute(_ wanted: Route) {
        lock.lock(); defer { lock.unlock() }
        route = wanted
    }

    public var loadedRuntimeOwnBytes: UInt64? {
        current == .engine ? engine.loadedRuntimeOwnBytes : llamaServer.loadedRuntimeOwnBytes
    }

    /// Nil for a llama-server: its memory is another process's, and `AppModel` then samples the machine instead.
    public var currentInferenceMemoryBytes: UInt64? {
        current == .engine ? (engine as? any AppInferenceMemoryReporting)?.currentInferenceMemoryBytes : nil
    }

    public func ensureLoaded(modelDirectory: URL, maxContextTokens: Int, options: AppRuntimeOptions,
                             forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        let wanted: Route = AppModelKind.probe(modelDirectory: modelDirectory)?.runsOnLlamaServer == true
            ? .llamaServer : .engine
        if wanted != current {
            switch wanted {
            case .llamaServer: await engine.unload()
            case .engine: await llamaServer.unload()
            }
            setRoute(wanted)
        }
        switch wanted {
        case .engine:
            try await engine.ensureLoaded(modelDirectory: modelDirectory, maxContextTokens: maxContextTokens,
                                          options: options, forceLogitsHead: forceLogitsHead, onState: onState)
        case .llamaServer:
            try await llamaServer.ensureLoaded(modelDirectory: modelDirectory, maxContextTokens: maxContextTokens,
                                               options: options, forceLogitsHead: forceLogitsHead, onState: onState)
        }
    }

    public func unload() async {
        await engine.unload()
        await llamaServer.unload()
    }

    /// What the app calls on its way out: the decode service is launchd's to stop, the server is this process's child.
    public func stopChildServer() { llamaServer.stopServer() }

    public func cancel() {
        switch current {
        case .engine: engine.cancel()
        case .llamaServer: llamaServer.cancel()
        }
    }

    public func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        guard current == .llamaServer else { return engine.generate(request) }
        let mailbox = generationTranscriptMailbox
        let inner = llamaServer.generate(request)
        return AsyncThrowingStream { continuation in
            let task = Task {
                mailbox.reset()
                do {
                    for try await event in inner {
                        if case .token(let token) = event {
                            mailbox.append(token.textDelta)
                            mailbox.appendReasoning(token.reasoningDelta)
                        }
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
