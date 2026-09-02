import Foundation

public protocol AppInferenceClient: Sendable {
    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error>
    func cancel()
}

/// A client that owns a loadable model session. Loading is split from
/// generation so the UI can pre-load the ~1.6 GB resident weights once and
/// keep them warm across runs. Generation never loads or replaces a session.
public protocol AppModelLifecycleClient: AnyObject, AppInferenceClient {
    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                      options: AppRuntimeOptions, forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws
    func unload() async
}

public protocol AppInferenceMemoryReporting: AnyObject {
    var currentInferenceMemoryBytes: UInt64? { get }
}

/// What the loaded runtime holds for itself while the model is loaded:
/// the resident weights, the KV cache for the context length, the experts
/// it asks Metal to keep resident. Wired or private either way, so the
/// speedometer can tell it apart from other apps. Nil when unknown (not
/// loaded, or a runtime that does not budget).
public protocol AppInferenceRuntimeReporting: AnyObject {
    var loadedRuntimeOwnBytes: UInt64? { get }
}

public protocol AppInferenceTranscriptReporting: AnyObject {
    var generationTranscriptMailbox: GenerationTranscriptMailbox { get }
}
