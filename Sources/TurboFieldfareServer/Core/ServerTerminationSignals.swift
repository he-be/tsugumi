import Darwin
import Dispatch

public actor ServerTerminationSignals {
    private let stream: AsyncStream<Int32>
    private let continuation: AsyncStream<Int32>.Continuation
    private let sources: [any DispatchSourceSignal]
    /// The signals this object took over, so it can hand them back (LIF-5).
    private nonisolated let managedSignals: [Int32]

    public init(_ signals: [Int32] = [SIGINT, SIGTERM]) {
        var capturedContinuation: AsyncStream<Int32>.Continuation?
        let stream = AsyncStream<Int32>(bufferingPolicy: .bufferingOldest(1)) {
            capturedContinuation = $0
        }
        let continuation = capturedContinuation!

        self.stream = stream
        self.continuation = continuation
        self.managedSignals = signals
        self.sources = signals.map {
            Darwin.signal($0, SIG_IGN)
            return Self.makeSource(signal: $0, continuation: continuation)
        }
        for source in sources {
            source.resume()
        }
    }

    public func wait() async -> Int32 {
        for await signal in stream {
            return signal
        }
        preconditionFailure("termination signal stream ended without a signal")
    }

    /// LIF-5: put the default disposition back, so a second signal kills.
    ///
    /// `init` had to set SIG_IGN for the dispatch sources to see anything at
    /// all, which also means a second Ctrl-C would be swallowed. The caller
    /// hands the signals back as soon as it has taken the first one, so the
    /// shutdown it then runs can always be cut short.
    ///
    /// `nonisolated` because it is called from the moment the signal arrives,
    /// which must not have to wait for this actor's turn.
    public nonisolated func restoreDefaultDisposition() {
        for signal in managedSignals {
            Darwin.signal(signal, SIG_DFL)
        }
    }

    public func cancel() {
        for source in sources {
            source.cancel()
        }
        continuation.finish()
    }

    private nonisolated static func makeSource(
        signal: Int32,
        continuation: AsyncStream<Int32>.Continuation
    ) -> any DispatchSourceSignal {
        let source = DispatchSource.makeSignalSource(signal: signal, queue: .global())
        source.setEventHandler { @Sendable [continuation] in
            continuation.yield(signal)
        }
        return source
    }
}
