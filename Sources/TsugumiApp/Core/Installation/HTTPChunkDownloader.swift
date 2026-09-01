import Foundation

/// Streams an HTTP response body as `Data` chunks.
///
/// `URLSession.bytes` hands the body out one byte at a time, which is the
/// wrong shape for hashing and writing a 20 GB weight file; the delegate
/// callbacks already deliver whole chunks, so this exposes those directly.
/// The session is per-request and invalidated when the body ends.
enum HTTPChunkDownloader {
    static func open(request: URLRequest) async throws
        -> (statusCode: Int, body: AsyncThrowingStream<Data, Error>) {
        let (stream, continuation) = AsyncThrowingStream<Data, Error>.makeStream(
            bufferingPolicy: .unbounded)
        let delegate = StreamDelegate(continuation: continuation)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 60
        // The resource timeout must cover the whole body of a 15 GB file on a
        // slow line; a week is "no resource timeout" in practice.
        configuration.timeoutIntervalForResource = 7 * 24 * 3600
        let session = URLSession(configuration: configuration,
                                 delegate: delegate,
                                 delegateQueue: nil)
        let task = session.dataTask(with: request)
        continuation.onTermination = { termination in
            if case .cancelled = termination { task.cancel() }
            session.finishTasksAndInvalidate()
        }
        let statusCode = try await delegate.start(task: task)
        return (statusCode, stream)
    }

    private final class StreamDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
        private let continuation: AsyncThrowingStream<Data, Error>.Continuation
        private let lock = NSLock()
        private var responseContinuation: CheckedContinuation<Int, Error>?

        init(continuation: AsyncThrowingStream<Data, Error>.Continuation) {
            self.continuation = continuation
        }

        func start(task: URLSessionDataTask) async throws -> Int {
            try await withCheckedThrowingContinuation { continuation in
                lock.lock()
                responseContinuation = continuation
                lock.unlock()
                task.resume()
            }
        }

        private func takeResponseContinuation() -> CheckedContinuation<Int, Error>? {
            lock.lock()
            defer { lock.unlock() }
            let taken = responseContinuation
            responseContinuation = nil
            return taken
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                        didReceive response: URLResponse,
                        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            takeResponseContinuation()?.resume(returning: statusCode)
            completionHandler(.allow)
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                        didReceive data: Data) {
            continuation.yield(data)
        }

        func urlSession(_ session: URLSession, task: URLSessionTask,
                        didCompleteWithError error: Error?) {
            if let error {
                // A failure before the headers arrive must release `start`,
                // not only the body stream.
                takeResponseContinuation()?.resume(throwing: error)
                continuation.finish(throwing: error)
            } else {
                continuation.finish()
            }
        }
    }
}
