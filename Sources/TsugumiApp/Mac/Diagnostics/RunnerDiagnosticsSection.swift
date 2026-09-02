import TsugumiAppCore
import SwiftUI

struct RunnerDiagnosticsSection: View {
    let diagnostics: AppDiagnostics?

    var body: some View {
        Section(L("Last run")) {
            if let diagnostics {
                groupLabel(L("Result"))
                DiagnosticRow(L("Settings"), diagnostics.runtimeOptions.resultSummary, multiline: true)
                DiagnosticRow(L("Prompt tokens"), diagnostics.promptTokenCount.map(String.init) ?? L("unknown"))
                if let cached = diagnostics.cachedPromptTokens, cached > 0 {
                    DiagnosticRow(L("From prompt cache"), "\(cached)")
                }
                DiagnosticRow(L("Output tokens"), "\(diagnostics.generatedTokens)")
                if let speculative = diagnostics.speculative {
                    DiagnosticRow(
                        L("MTP acceptance"),
                        "\(speculative.accepted)/\(speculative.proposed) (\(MetricFormat.percent(speculative.acceptanceRate * 100)))")
                }
                DiagnosticRow(L("Stop"), diagnostics.stopReason.rawValue)

                groupLabel(L("Performance"))
                DiagnosticRow(L("Prompt prefill"), MetricFormat.seconds(diagnostics.prefillSeconds))
                if let prefillRate = diagnostics.prefillTokensPerSecond {
                    DiagnosticRow(L("Prefill rate"), "\(MetricFormat.rate(prefillRate)) \(L("tok/s"))")
                }
                DiagnosticRow(L("First token wait"), MetricFormat.seconds(diagnostics.timeToFirstTokenSeconds))
                DiagnosticRow(L("Request TTFT"), MetricFormat.seconds(diagnostics.requestStartTimeToFirstTokenSeconds))
                DiagnosticRow(L("Decode duration"), MetricFormat.seconds(diagnostics.decodeSeconds))
                DiagnosticRow(L("Decode rate"), "\(MetricFormat.rate(diagnostics.tokensPerSecond)) \(L("tok/s"))")
                DiagnosticRow(L("Peak memory"), MetricFormat.memory(diagnostics.peakMemoryBytes))
                DiagnosticRow(L("I/O / token"),
                              MetricFormat.milliseconds(diagnostics.runner?.ioMillisecondsPerToken))

                if hasIssues(diagnostics) {
                    groupLabel(L("Issues"))
                    issueRows(diagnostics)
                }

                if let prefill = diagnostics.prefill {
                    DisclosureGroup(L("Prefill details")) {
                        VStack(spacing: 8) {
                            DiagnosticRow(L("Mode"), "\(prefill.requestedMode.rawValue) -> \(prefill.executedMode.rawValue)")
                            DiagnosticRow(L("KV storage"), prefill.kvStorageMode?.rawValue ?? L("unknown"))
                            DiagnosticRow(L("Completeness"), prefill.chunkCompleteness.rawValue)
                            if let reason = prefill.unsupportedReason, !reason.isEmpty {
                                DiagnosticRow(L("Unsupported reason"), reason, multiline: true)
                            }
                        }
                        .frame(maxWidth: .infinity)
                    }
                }
                if let runner = diagnostics.runner {
                    DisclosureGroup(L("Decode runner")) {
                        AdvancedRunnerDiagnosticsView(runner: runner)
                    }
                }
            } else {
                Text(L("No runs yet"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    @ViewBuilder
    private func issueRows(_ diagnostics: AppDiagnostics) -> some View {
        if let prefill = diagnostics.prefill {
            if prefill.requestedMode.rawValue != prefill.executedMode.rawValue {
                DiagnosticRow(L("Prefill mode"),
                              "\(prefill.requestedMode.rawValue) -> \(prefill.executedMode.rawValue)")
            }
            if prefill.chunkCompleteness != .complete {
                DiagnosticRow(L("Prefill status"), prefill.chunkCompleteness.rawValue)
            }
            if let reason = prefill.unsupportedReason, !reason.isEmpty {
                DiagnosticRow(L("Unsupported reason"), reason, multiline: true)
            }
        }
        if let failures = diagnostics.runner?.rdadviseFailures, failures > 0 {
            DiagnosticRow(L("RDADVISE failures"), "\(failures)")
        }
    }

    private func hasIssues(_ diagnostics: AppDiagnostics) -> Bool {
        let prefillHasIssue = diagnostics.prefill.map {
            $0.requestedMode.rawValue != $0.executedMode.rawValue
                || $0.chunkCompleteness != .complete
                || !($0.unsupportedReason?.isEmpty ?? true)
        } ?? false
        return prefillHasIssue || (diagnostics.runner?.rdadviseFailures ?? 0) > 0
    }

    private func groupLabel(_ title: String) -> some View {
        Text(title)
            .font(.caption)
            .textCase(.uppercase)
            .foregroundStyle(.tertiary)
            .accessibilityHeading(.h3)
    }
}

private struct AdvancedRunnerDiagnosticsView: View {
    let runner: AppRunnerDiagnostics

    var body: some View {
        VStack(spacing: 8) {
            DiagnosticRow("cb1 / token", MetricFormat.milliseconds(runner.cb1MillisecondsPerToken))
            DiagnosticRow("cb2 / token", MetricFormat.milliseconds(runner.cb2MillisecondsPerToken))
            DiagnosticRow(L("Head / token"), MetricFormat.milliseconds(runner.headMillisecondsPerToken))
            if hasRDAdviceActivity {
                DiagnosticRow("RDADVISE / token",
                              MetricFormat.milliseconds(runner.rdadviseMillisecondsPerToken))
                DiagnosticRow(L("RDADVISE calls"), MetricFormat.perToken(runner.rdadviseCallsPerToken))
                DiagnosticRow(L("RDADVISE data"),
                              MetricFormat.megabytesPerToken(runner.rdadviseMegabytesPerToken))
                DiagnosticRow(L("RDADVISE skipped"), MetricFormat.perToken(runner.rdadviseSkippedPerToken))
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var hasRDAdviceActivity: Bool {
        runner.rdadviseMillisecondsPerToken > 0
            || runner.rdadviseCallsPerToken > 0
            || runner.rdadviseMegabytesPerToken > 0
            || runner.rdadviseSkippedPerToken > 0
    }
}

private struct DiagnosticRow: View {
    let label: String
    let value: String
    let multiline: Bool

    init(_ label: String, _ value: String, multiline: Bool = false) {
        self.label = label
        self.value = value
        self.multiline = multiline
    }

    var body: some View {
        LabeledContent {
            Text(value)
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .multilineTextAlignment(.trailing)
                .lineLimit(multiline ? nil : 2)
                .fixedSize(horizontal: false, vertical: true)
        } label: {
            Text(label)
        }
    }
}
