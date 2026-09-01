import Foundation
import Synchronization

public final class GenerationTranscriptMailbox: Sendable {
    public struct Snapshot: Sendable {
        public let pendingText: String
        public let completeText: String
    }

    private struct State: Sendable {
        var pending = ""
        var complete = ""
        var reasoning = ""
    }

    private let state = Mutex(State())

    public init() {}

    public func append(_ text: String) {
        guard !text.isEmpty else { return }
        state.withLock {
            $0.pending += text
            $0.complete += text
        }
    }

    public func drain() -> Snapshot {
        state.withLock {
            let snapshot = Snapshot(pendingText: $0.pending, completeText: $0.complete)
            $0.pending = ""
            return snapshot
        }
    }

    public var completeText: String {
        state.withLock { $0.complete }
    }

    /// Thought-channel text, accumulated apart from the answer so the two
    /// never interleave in the transcript.
    public func appendReasoning(_ text: String) {
        guard !text.isEmpty else { return }
        state.withLock { $0.reasoning += text }
    }

    public var completeReasoningText: String {
        state.withLock { $0.reasoning }
    }

    public func reset() {
        state.withLock { $0 = State() }
    }
}
