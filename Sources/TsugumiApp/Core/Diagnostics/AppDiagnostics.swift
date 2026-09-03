import Foundation
import Tsugumi

public enum AppStopReason: String, Equatable, Sendable {
    case maxTokens
    case cancelled
    case failed
    case eos
    case endOfTurn
    case stopString
    case toolCalls
}

public struct AppRunnerDiagnostics: Equatable, Sendable {
    public var cb1MillisecondsPerToken: Double
    public var ioMillisecondsPerToken: Double
    public var cb2MillisecondsPerToken: Double
    public var headMillisecondsPerToken: Double
    public var rdadviseMillisecondsPerToken: Double
    public var rdadviseCallsPerToken: Double
    public var rdadviseMegabytesPerToken: Double
    public var rdadviseSkippedPerToken: Double
    public var rdadviseFailures: UInt64

    public init(cb1MillisecondsPerToken: Double = 0,
                ioMillisecondsPerToken: Double = 0,
                cb2MillisecondsPerToken: Double = 0,
                headMillisecondsPerToken: Double = 0,
                rdadviseMillisecondsPerToken: Double = 0,
                rdadviseCallsPerToken: Double = 0,
                rdadviseMegabytesPerToken: Double = 0,
                rdadviseSkippedPerToken: Double = 0,
                rdadviseFailures: UInt64 = 0) {
        self.cb1MillisecondsPerToken = cb1MillisecondsPerToken
        self.ioMillisecondsPerToken = ioMillisecondsPerToken
        self.cb2MillisecondsPerToken = cb2MillisecondsPerToken
        self.headMillisecondsPerToken = headMillisecondsPerToken
        self.rdadviseMillisecondsPerToken = rdadviseMillisecondsPerToken
        self.rdadviseCallsPerToken = rdadviseCallsPerToken
        self.rdadviseMegabytesPerToken = rdadviseMegabytesPerToken
        self.rdadviseSkippedPerToken = rdadviseSkippedPerToken
        self.rdadviseFailures = rdadviseFailures
    }
}

/// What one request's speculative loop did: proposals and acceptances, for
/// the HUD only — accepted tokens are tokens the target itself drew, so this
/// says how the wall clock was spent, never what the answer was.
public struct AppSpeculativeDiagnostics: Equatable, Sendable {
    public var blockTokens: Int
    public var proposed: Int
    public var accepted: Int

    public init(blockTokens: Int, proposed: Int, accepted: Int) {
        self.blockTokens = blockTokens
        self.proposed = proposed
        self.accepted = accepted
    }

    public var acceptanceRate: Double {
        proposed > 0 ? Double(accepted) / Double(proposed) : 0
    }
}

public struct AppDiagnostics: Equatable, Sendable {
    public var generatedTokens: Int
    public var stopReason: AppStopReason
    public var promptTokenCount: Int?
    /// Prompt tokens the session served from its cache instead of
    /// recomputing (exact-extension for Ornith, LCP for Gemma).
    public var cachedPromptTokens: Int?
    /// Present only when the MTP loop actually ran.
    public var speculative: AppSpeculativeDiagnostics?
    public var prefillSeconds: Double?
    public var timeToFirstTokenSeconds: Double?
    public var decodeSeconds: Double
    public var tokensPerSecond: Double
    public var peakMemoryBytes: UInt64?
    public var runtimeOptions: AppRuntimeOptions
    public var prefill: PrefillExecutionDiagnostics?
    public var runner: AppRunnerDiagnostics?
    /// Bytes the round read from the SSD, prefill and decode apart. Present
    /// when the decode service measured them.
    public var diskRead: AppDiskReadDiagnostics?

    public var requestStartTimeToFirstTokenSeconds: Double? {
        guard let prefillSeconds, let timeToFirstTokenSeconds else { return nil }
        return prefillSeconds + timeToFirstTokenSeconds
    }

    public var prefillTokensPerSecond: Double? {
        guard let promptTokenCount,
              promptTokenCount > 0,
              let prefillSeconds,
              prefillSeconds.isFinite,
              prefillSeconds > 0 else {
            return nil
        }
        let rate = Double(promptTokenCount) / prefillSeconds
        return rate.isFinite ? rate : nil
    }

    public init(generatedTokens: Int,
                stopReason: AppStopReason,
                promptTokenCount: Int? = nil,
                cachedPromptTokens: Int? = nil,
                speculative: AppSpeculativeDiagnostics? = nil,
                prefillSeconds: Double? = nil,
                timeToFirstTokenSeconds: Double?,
                decodeSeconds: Double,
                tokensPerSecond: Double,
                peakMemoryBytes: UInt64?,
                runtimeOptions: AppRuntimeOptions,
                prefill: PrefillExecutionDiagnostics? = nil,
                runner: AppRunnerDiagnostics? = nil,
                diskRead: AppDiskReadDiagnostics? = nil) {
        self.generatedTokens = generatedTokens
        self.stopReason = stopReason
        self.promptTokenCount = promptTokenCount
        self.cachedPromptTokens = cachedPromptTokens
        self.speculative = speculative
        self.prefillSeconds = prefillSeconds
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.decodeSeconds = decodeSeconds
        self.tokensPerSecond = tokensPerSecond
        self.peakMemoryBytes = peakMemoryBytes
        self.runtimeOptions = runtimeOptions
        self.prefill = prefill
        self.runner = runner
        self.diskRead = diskRead
    }
}

public struct AppTokenEvent: Equatable, Sendable {
    public var index: Int
    public var textDelta: String
    public var elapsedDecodeSeconds: Double
    /// Thought-channel text this event carries. Kept apart from `textDelta`
    /// so the UI can render reasoning as reasoning instead of answer text.
    public var reasoningDelta: String = ""
}

public enum AppInferenceEvent: Equatable, Sendable {
    case prefillProgress(done: Int, total: Int)
    case token(AppTokenEvent)
    /// A function call the model wrote. The generation then ends with
    /// `stopReason == .toolCalls`; the app runs the call and starts the
    /// next round.
    case toolCall(AppToolCall)
    case finished(AppDiagnostics)
    case cancelled(AppDiagnostics)
    case failed(AppInferenceError, partial: AppDiagnostics?)
}
