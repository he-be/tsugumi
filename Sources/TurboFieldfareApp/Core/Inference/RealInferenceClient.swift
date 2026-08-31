import Foundation
import Metal
import TurboFieldfare
import TurboFieldfareServerCore
import Synchronization

final class GenerationTaskRegistry: Sendable {
    private struct Entry: Sendable {
        let id: UUID
        var task: Task<Void, Never>?
    }

    private let state = Mutex<Entry?>(nil)

    func reserve(_ id: UUID) -> Bool {
        state.withLock { entry in
            guard entry == nil else { return false }
            entry = Entry(id: id, task: nil)
            return true
        }
    }

    func attach(_ task: Task<Void, Never>, to id: UUID) {
        let shouldCancel = state.withLock { entry -> Bool in
            guard entry?.id == id else { return true }
            entry?.task = task
            return false
        }
        if shouldCancel { task.cancel() }
    }

    func take(_ id: UUID) -> Task<Void, Never>? {
        state.withLock { entry in
            guard entry?.id == id else { return nil }
            defer { entry = nil }
            return entry?.task
        }
    }

    func takeCurrent() -> Task<Void, Never>? {
        state.withLock { entry in
            defer { entry = nil }
            return entry?.task
        }
    }

    func clear(_ id: UUID) {
        state.withLock { entry in
            if entry?.id == id { entry = nil }
        }
    }

}

/// Real-model inference client for the Mac app. Since the two-model rework it
/// wraps the same family sessions the HTTP server serves from —
/// `ServerModelSession` for Gemma (vision, MTP block 4, LCP prompt cache) and
/// `QwenServerSession` for Ornith (MTP width 2, exact-extension prompt cache,
/// the official sampler pinned by S1) — behind the `AppInferenceClient` event
/// stream, with an explicit load lifecycle so the resident weights stay warm
/// across generations. The prompt goes through each checkpoint's own chat
/// template, with the thought channel opened or closed by the request.
public final class RealInferenceClient: AppModelLifecycleClient, @unchecked Sendable {
    private let session: RealInferenceSession
    private let memorySampler: AppMemorySampler
    private let generationTasks = GenerationTaskRegistry()

    public init(memorySampler: AppMemorySampler = AppMemorySampler()) {
        self.memorySampler = memorySampler
        self.session = RealInferenceSession()
    }

    public func ensureLoaded(modelDirectory: URL,
                             maxContextTokens: Int,
                             options: AppRuntimeOptions,
                             forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        try await session.ensureLoaded(
            key: SessionLoadKey(directory: modelDirectory.standardizedFileURL,
                                maxContext: maxContextTokens,
                                options: options,
                                forceLogitsHead: forceLogitsHead),
            onState: onState)
    }

    public func unload() async {
        await session.unload()
    }

    public func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            let generationID = UUID()
            guard generationTasks.reserve(generationID) else {
                continuation.yield(.failed(.generationInFlight, partial: nil))
                continuation.finish(throwing: AppInferenceError.generationInFlight)
                return
            }
            let task = Task { [self] in
                await session.run(request: request,
                                  memorySampler: memorySampler,
                                  continuation: continuation)
                generationTasks.clear(generationID)
            }
            generationTasks.attach(task, to: generationID)

            continuation.onTermination = { [generationTasks] _ in
                generationTasks.take(generationID)?.cancel()
            }
        }
    }

    public func cancel() {
        generationTasks.takeCurrent()?.cancel()
    }

}

struct SessionLoadKey: Equatable, Sendable {
    var directory: URL
    var maxContext: Int
    var options: AppRuntimeOptions
    var forceLogitsHead: Bool

    init(directory: URL,
         maxContext: Int,
         options: AppRuntimeOptions,
         forceLogitsHead: Bool = false) {
        self.directory = directory.standardizedFileURL
        self.maxContext = maxContext
        self.options = options
        self.forceLogitsHead = forceLogitsHead
    }
}

/// Owns the loaded family session and serializes load / unload / generate.
/// All Metal command-buffer waits happen inside this actor, off the main
/// actor; one cooperative-pool thread is occupied for the duration of a
/// generation, which is acceptable for the app's single session. A reload
/// releases the loaded session before constructing a replacement, so two
/// models are never alive at once.
actor RealInferenceSession {
    enum Backend {
        case gemma(ServerModelSession)
        case ornith(QwenServerSession)
    }

    private var loadedKey: SessionLoadKey?
    private var loadedKind: AppModelKind?
    private var backend: Backend?
    /// What load actually enabled, after degrading gracefully: MTP falls back
    /// to off when the checkpoint has no drafter section (Gemma) or no head
    /// sidecar (Ornith).
    private(set) var loadedMTPEnabled = false

    func ensureLoaded(key: SessionLoadKey,
                      onState: @Sendable (AppModelLoadState) -> Void) async throws {
        if loadedKey == key, backend != nil { return }

        backend = nil
        loadedKey = nil
        loadedKind = nil

        let start = Date()
        do {
            onState(.loading(.validatingDirectory))
            let manifest = key.directory.appendingPathComponent("manifest.json")
            guard FileManager.default.fileExists(atPath: manifest.path) else {
                throw AppInferenceError.modelNotFound(key.directory.path)
            }
            guard let kind = AppModelKind.probe(modelDirectory: key.directory) else {
                throw AppInferenceError.modelLoadFailed(
                    "the manifest at \(key.directory.path) names a model family this app does not ship")
            }
            try Task.checkCancellation()

            onState(.loading(.verifyingWeights))
            let runtimeConfiguration = try key.options.resolvedRuntimeConfiguration(
                forceLogitsHead: key.forceLogitsHead)
            let integrity = key.options.modelVerification.runtimeValue
            let draftBlockSize = Self.resolvedDraftBlockSize(
                kind: kind,
                modelDirectory: key.directory,
                requested: key.options.effectiveMTPEnabled)
            onState(.loading(.preparingRunner))
            switch kind {
            case .gemmaQATSym:
                let session = try await ServerModelSession.load(
                    modelDirectory: key.directory,
                    maxContext: key.maxContext,
                    runtimeConfiguration: runtimeConfiguration,
                    integrityPolicy: integrity,
                    draftBlockSize: draftBlockSize)
                backend = .gemma(session)
            case .ornith:
                let session = try await QwenServerSession.load(
                    modelDirectory: key.directory,
                    maxContext: key.maxContext,
                    runtimeConfiguration: runtimeConfiguration,
                    integrityPolicy: integrity,
                    draftBlockSize: draftBlockSize,
                    mtpHeadDirectory: Self.mtpSidecarDirectory(
                        forModelDirectory: key.directory))
                backend = .ornith(session)
            }
            try Task.checkCancellation()

            loadedMTPEnabled = draftBlockSize > 0
            loadedKey = key
            loadedKind = kind
            onState(.ready(modelDirectory: key.directory,
                           loadSeconds: Date().timeIntervalSince(start)))
        } catch is CancellationError {
            backend = nil
            throw CancellationError()
        } catch let appError as AppInferenceError {
            onState(.failed(appError))
            throw appError
        } catch {
            let appError = AppInferenceError.modelLoadFailed("\(error)")
            onState(.failed(appError))
            throw appError
        }
    }

    /// MTP degrades to off rather than failing the load: a checkpoint without
    /// the drafter section (Gemma) or without a head sidecar (Ornith) still
    /// answers, just without the speculative loop.
    static func resolvedDraftBlockSize(kind: AppModelKind,
                                       modelDirectory: URL,
                                       requested: Bool) -> Int {
        guard requested else { return 0 }
        switch kind {
        case .gemmaQATSym:
            guard let data = try? Data(
                    contentsOf: modelDirectory.appendingPathComponent("manifest.json")),
                  let object = try? JSONSerialization.jsonObject(with: data),
                  let dictionary = object as? [String: Any],
                  dictionary["draft"] != nil else {
                return 0
            }
            return kind.draftBlockSize
        case .ornith:
            let sidecar = mtpSidecarDirectory(forModelDirectory: modelDirectory)
            let index = URL(fileURLWithPath: sidecar).appendingPathComponent("mtp_head.json")
            guard FileManager.default.fileExists(atPath: index.path) else { return 0 }
            return kind.draftBlockSize
        }
    }

    /// The sidecar inside the model directory wins (that is where the
    /// installer puts it); the development machine's shared location is the
    /// fallback.
    static func mtpSidecarDirectory(forModelDirectory modelDirectory: URL) -> String {
        let installed = modelDirectory
            .appendingPathComponent(AppModelKind.mtpSidecarDirectoryName)
        if FileManager.default.fileExists(
            atPath: installed.appendingPathComponent("mtp_head.json").path) {
            return installed.path
        }
        return QwenMTPSidecar.defaultDirectory
    }

    func unload() {
        backend = nil
        loadedKey = nil
        loadedKind = nil
        loadedMTPEnabled = false
    }

    static func forceLogitsHead(for request: AppGenerationRequest) -> Bool {
        !request.isPureGreedy
    }

    func run(request: AppGenerationRequest,
             memorySampler: AppMemorySampler,
             continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation) async {
        let progress = ProgressState()
        do {
            try request.validate()
            let requestKey = SessionLoadKey(
                directory: request.modelDirectory.standardizedFileURL,
                maxContext: request.maxContextTokens,
                options: request.runtimeOptions,
                forceLogitsHead: Self.forceLogitsHead(for: request))
            guard let loadedKey else { throw AppInferenceError.modelNotLoaded }
            guard loadedKey == requestKey else { throw AppInferenceError.reloadRequired }
            guard let backend, let loadedKind else {
                throw AppInferenceError.modelLoadFailed("session lost its loaded state")
            }

            let validated = try Self.validatedChatRequest(for: request, kind: loadedKind)
            memorySampler.resetPeak()
            _ = memorySampler.sample()
            progress.prefillStart = Date()

            let monitor = ServerTimingsMonitor()
            let onPrefill: @Sendable (Int, Int) -> Void = { done, total in
                progress.observePrefill(done: done, total: total)
                continuation.yield(.prefillProgress(done: done, total: total))
            }
            let onEvent: @Sendable (ServerInferenceEvent) -> Void = { event in
                switch event {
                case .content(let text):
                    guard !text.isEmpty else { return }
                    let (index, elapsed) = progress.observeToken(
                        monitor: monitor, memorySampler: memorySampler)
                    continuation.yield(.token(AppTokenEvent(
                        index: index,
                        textDelta: text,
                        elapsedDecodeSeconds: elapsed)))
                case .reasoning(let text):
                    guard !text.isEmpty else { return }
                    let (index, elapsed) = progress.observeToken(
                        monitor: monitor, memorySampler: memorySampler)
                    continuation.yield(.token(AppTokenEvent(
                        index: index,
                        textDelta: "",
                        elapsedDecodeSeconds: elapsed,
                        reasoningDelta: text)))
                case .toolCall:
                    // The app declares no tools, so a call cannot be asked
                    // for; a `<tool_call>` the model writes unasked stays
                    // text on the no-tools path and never reaches here.
                    break
                }
            }

            let completion: ServerCompletion
            switch backend {
            case .gemma(let session):
                let prepared = try await session.prepare(validated)
                progress.promptTokenCount = prepared.promptTokenCount
                completion = try await session.generate(
                    prepared, monitor: monitor, onPrefill: onPrefill, onEvent: onEvent)
            case .ornith(let session):
                let prepared = try await session.prepare(validated)
                progress.promptTokenCount = prepared.promptTokenCount
                completion = try await session.generate(
                    prepared, monitor: monitor, onPrefill: onPrefill, onEvent: onEvent)
            }

            let cancelled = Task.isCancelled
            let diagnostics = Self.diagnostics(request: request,
                                               memorySampler: memorySampler,
                                               progress: progress,
                                               completion: completion,
                                               cancelled: cancelled)
            if cancelled {
                continuation.yield(.cancelled(diagnostics))
                continuation.finish(throwing: AppInferenceError.cancelled)
            } else {
                continuation.yield(.finished(diagnostics))
                continuation.finish()
            }
        } catch is CancellationError {
            let diagnostics = Self.partialDiagnostics(request: request,
                                                      memorySampler: memorySampler,
                                                      progress: progress,
                                                      stopReason: .cancelled)
            continuation.yield(.cancelled(diagnostics))
            continuation.finish(throwing: AppInferenceError.cancelled)
        } catch let requestError as ServerRequestError {
            failGeneration(.invalidRequest(requestError.message),
                           request: request, memorySampler: memorySampler,
                           progress: progress, continuation: continuation)
        } catch let appError as AppInferenceError {
            failGeneration(appError, request: request, memorySampler: memorySampler,
                           progress: progress, continuation: continuation)
        } catch {
            failGeneration(.unknown("\(error)"), request: request, memorySampler: memorySampler,
                           progress: progress, continuation: continuation)
        }
    }

    /// The GUI request as the sessions take it: one user turn through the
    /// checkpoint's own chat template, the thought channel as the toggle set
    /// it, the sampler as the UI shows it (Ornith's session pins S1's
    /// official values whatever arrives here — the UI shows them pinned).
    static func validatedChatRequest(for request: AppGenerationRequest,
                                     kind: AppModelKind) throws -> ValidatedChatRequest {
        let messages = [GFTokenizer.Message(role: .user, content: request.prompt)]
        var vision: ValidatedVisionRequest?
        if !request.imagePaths.isEmpty {
            guard kind.supportsVision else {
                throw AppInferenceError.invalidRequest(
                    "\(kind.shortName) has no vision tower; remove the attached images.")
            }
            let policy = ServerImagePolicy.default
            var attachments: [ServerImageAttachment] = []
            for (index, path) in request.imagePaths.enumerated() {
                let data = try Data(contentsOf: URL(fileURLWithPath: path))
                let mediaType = Self.mediaType(forPath: path)
                let dataURL = "data:\(mediaType);base64,\(data.base64EncodedString())"
                attachments.append(try ServerImageDecoder.attachment(
                    fromImageURL: dataURL, policy: policy, index: index))
            }
            let parts = request.imagePaths.map { _ in GFTokenizer.ContentPart.image }
                + [GFTokenizer.ContentPart.text(request.prompt)]
            vision = ValidatedVisionRequest(
                messages: [GFTokenizer.MultimodalMessage(role: .user, parts: parts)],
                images: attachments)
        }
        let config = GenerationConfig(
            maxNewTokens: max(request.maxNewTokens, 1),
            temperature: request.temperature,
            topK: request.topK,
            topP: request.topP,
            repetitionPenalty: request.repetitionPenalty)
        return ValidatedChatRequest(
            messages: messages,
            tools: [],
            stream: true,
            includeUsage: false,
            generationConfig: config,
            maximumCompletionTokens: request.maxNewTokens,
            vision: vision,
            enableThinking: request.enableThinking)
    }

    static func mediaType(forPath path: String) -> String {
        switch (path as NSString).pathExtension.lowercased() {
        case "png": "image/png"
        case "jpg", "jpeg": "image/jpeg"
        case "webp": "image/webp"
        case "gif": "image/gif"
        default: "image/png"
        }
    }

    private func failGeneration(_ error: AppInferenceError,
                                request: AppGenerationRequest,
                                memorySampler: AppMemorySampler,
                                progress: ProgressState,
                                continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation) {
        let partial = progress.generated > 0
            ? Self.partialDiagnostics(request: request,
                                      memorySampler: memorySampler,
                                      progress: progress,
                                      stopReason: .failed)
            : nil
        continuation.yield(.failed(error, partial: partial))
        continuation.finish(throwing: error)
    }

    private static func diagnostics(request: AppGenerationRequest,
                                    memorySampler: AppMemorySampler,
                                    progress: ProgressState,
                                    completion: ServerCompletion,
                                    cancelled: Bool) -> AppDiagnostics {
        _ = memorySampler.sample()
        let timings = completion.timings
        let generated = timings?.predictedTokens ?? completion.usage.completionTokens
        let decodeSeconds = (timings?.predictedMilliseconds ?? 0) / 1_000
        let stopReason: AppStopReason
        if cancelled {
            stopReason = .cancelled
        } else {
            switch completion.finishReason {
            case "length": stopReason = .maxTokens
            case "tool_calls": stopReason = .toolCalls
            default: stopReason = .eos
            }
        }
        return AppDiagnostics(
            generatedTokens: generated,
            stopReason: stopReason,
            promptTokenCount: completion.usage.promptTokens,
            cachedPromptTokens: timings?.cacheTokens
                ?? completion.usage.promptTokensDetails.cachedTokens,
            speculative: completion.speculative.map {
                AppSpeculativeDiagnostics(blockTokens: $0.blockTokens,
                                          proposed: $0.proposed,
                                          accepted: $0.accepted)
            },
            prefillSeconds: (timings?.promptMilliseconds).map { $0 / 1_000 },
            timeToFirstTokenSeconds: progress.timeToFirstTokenSeconds,
            decodeSeconds: decodeSeconds,
            tokensPerSecond: decodeSeconds > 0 ? Double(generated) / decodeSeconds : 0,
            peakMemoryBytes: memorySampler.peakBytes,
            runtimeOptions: request.runtimeOptions,
            prefill: PrefillExecutionDiagnostics(
                config: request.runtimeOptions.prefillConfig,
                executedMode: request.runtimeOptions.prefillConfig.mode == .chunked
                    ? .chunked : .off,
                kvStorageMode: .fp16))
    }

    private static func partialDiagnostics(request: AppGenerationRequest,
                                           memorySampler: AppMemorySampler,
                                           progress: ProgressState,
                                           stopReason: AppStopReason) -> AppDiagnostics {
        _ = memorySampler.sample()
        let decodeSeconds = progress.elapsedDecodeSeconds
        return AppDiagnostics(
            generatedTokens: progress.generated,
            stopReason: stopReason,
            promptTokenCount: progress.promptTokenCount,
            prefillSeconds: progress.elapsedPrefillSeconds,
            timeToFirstTokenSeconds: progress.timeToFirstTokenSeconds,
            decodeSeconds: decodeSeconds,
            tokensPerSecond: decodeSeconds > 0 ? Double(progress.generated) / decodeSeconds : 0,
            peakMemoryBytes: memorySampler.peakBytes,
            runtimeOptions: request.runtimeOptions,
            prefill: PrefillExecutionDiagnostics(
                config: request.runtimeOptions.prefillConfig,
                executedMode: request.runtimeOptions.prefillConfig.mode == .chunked
                    ? .chunked : .off,
                kvStorageMode: .fp16))
    }
}

/// Mutable per-generation state shared between the session's event callbacks
/// and the surrounding actor method. The callbacks run synchronously inside
/// the backend actor's generation, one at a time.
private final class ProgressState: @unchecked Sendable {
    private let state = Mutex<Inner>(Inner())

    private struct Inner {
        var generated = 0
        var promptTokenCount: Int?
        var prefillStart: Date?
        var decodeStart: Date?
        var firstTokenDate: Date?
    }

    var generated: Int { state.withLock { $0.generated } }

    var promptTokenCount: Int? {
        get { state.withLock { $0.promptTokenCount } }
        set { state.withLock { $0.promptTokenCount = newValue } }
    }

    var prefillStart: Date? {
        get { state.withLock { $0.prefillStart } }
        set { state.withLock { $0.prefillStart = newValue } }
    }

    func observePrefill(done: Int, total: Int) {
        state.withLock {
            if done >= total, $0.decodeStart == nil { $0.decodeStart = Date() }
        }
    }

    /// Returns the token index and elapsed decode seconds for one streamed
    /// event. The count tracks the monitor's per-token timings when they are
    /// available, so the HUD's tok/s is tokens and not text deltas.
    func observeToken(monitor: ServerTimingsMonitor,
                      memorySampler: AppMemorySampler) -> (index: Int, elapsed: Double) {
        state.withLock { inner in
            let now = Date()
            if inner.decodeStart == nil { inner.decodeStart = now }
            if inner.firstTokenDate == nil { inner.firstTokenDate = now }
            if let live = monitor.current {
                inner.generated = max(inner.generated, live.predictedTokens)
            } else {
                inner.generated += 1
            }
            if inner.generated % 8 == 0 { _ = memorySampler.sample() }
            let elapsed = inner.decodeStart.map { now.timeIntervalSince($0) } ?? 0
            return (max(inner.generated - 1, 0), elapsed)
        }
    }

    var elapsedDecodeSeconds: Double {
        state.withLock {
            guard let decodeStart = $0.decodeStart else { return 0 }
            return Date().timeIntervalSince(decodeStart)
        }
    }

    var elapsedPrefillSeconds: Double? {
        state.withLock {
            guard let prefillStart = $0.prefillStart else { return nil }
            let end = $0.decodeStart ?? Date()
            return max(end.timeIntervalSince(prefillStart), 0)
        }
    }

    var timeToFirstTokenSeconds: Double? {
        state.withLock {
            guard let first = $0.firstTokenDate, let start = $0.decodeStart else { return nil }
            return first.timeIntervalSince(start)
        }
    }
}
