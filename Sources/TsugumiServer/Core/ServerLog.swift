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
        // SPEC §6 GEN-2 / §12 DEV-16: what the client asked for that could only
        // be approximated. It rides the line that already exists rather than a
        // channel of its own, and it comes from the declared schemas
        // (`tools`, `response_format`), never from `messages`.
        if let approximations = ServerApproximationLog.field(completion.approximations) {
            line += " approx=\"\(approximations)\""
        }
        write(line)
    }

    /// SPEC §7 CACHE-6: why a continuation did not happen, in numbers only.
    ///
    /// The Ornith family reuses the state or resets it, with nothing in
    /// between (`docs/qwen35moe/41-PROMPT-CACHE.md`), so "how far the new
    /// prompt agreed with what the state holds" is the whole diagnosis — a
    /// client that re-renders the last assistant turn differently shows up as a
    /// divergence a few dozen tokens from the end. **Never prompt text.**
    /// No request id: the backend decides this before the HTTP layer hands the
    /// request over, and DEV-3 keeps generation serial, so the line always sits
    /// directly above the `generating` line it belongs to. Plumbing an id down
    /// would mean teaching the shared layer which family answered.
    static func promptCache(_ detail: String) {
        write("prompt cache \(detail)")
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
