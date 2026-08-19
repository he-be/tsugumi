import Foundation

enum ServerLog {
    /// `thinking` is what the request asked for, not what the prompt rendered:
    /// a request that declares tools goes through the tool-calling template,
    /// which reasons in neither case. The completed line's `reasoning` is the
    /// side that says what actually happened.
    static func accepted(id: String, streaming: Bool, thinking: Bool) {
        write("request \(id) accepted streaming=\(streaming) "
            + "thinking=\(thinking ? "on" : "off")")
    }

    static func prepared(id: String, promptTokens: Int?) {
        let count = promptTokens.map(String.init) ?? "backend-managed"
        write("request \(id) prepared prompt=\(count)")
    }

    static func queued(id: String) {
        write("request \(id) queued")
    }

    static func generating(id: String) {
        write("request \(id) generating")
    }

    static func completed(id: String,
                          duration: Duration,
                          completion: ServerCompletion) {
        let usage = completion.usage
        var line = "request \(id) completed in \(format(duration)) "
            + "prompt=\(usage.promptTokens) "
            + "cached=\(usage.promptTokensDetails.cachedTokens) "
            + "completion=\(usage.completionTokens) "
            + "finish=\(completion.finishReason)"
        if !completion.reasoningContent.isEmpty {
            line += " reasoning=\(completion.reasoningContent.utf8.count)B"
        }
        if let speculative = completion.speculative {
            line += " mtp=\(speculative.blockTokens)"
                + " rounds=\(speculative.rounds)"
                + " accept=\(String(format: "%.3f", speculative.meanAcceptedLength))"
        }
        write(line)
    }

    static func failed(id: String,
                       phase: String,
                       status: UInt,
                       error: Error) {
        write("request \(id) failed phase=\(phase) status=\(status) "
            + "error=\(String(reflecting: error))")
    }

    private static func format(_ duration: Duration) -> String {
        let seconds = Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return String(format: "%.3fs", seconds)
    }

    private static func write(_ message: String) {
        let line = "[\(Date().formatted(.iso8601))] \(message)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }
}
