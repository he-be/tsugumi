import TsugumiAppCore
import TsugumiMacPresentation
import SwiftUI

struct StatusHUDView: View {
    let model: AppModel

    var body: some View {
        strip
            .padding(.top, 10)
            .padding(.leading, 84)
            .padding(.trailing, 20)
    }

    private var strip: some View {
        HStack(spacing: 12) {
            ModelStatusBadge(model: model)
            Divider().frame(height: 16)
            PhaseLabel(model: model)
            Spacer(minLength: 12)
            if showsMetrics {
                HUDMetricView(value: rateText, label: L("tok/s"), animated: !model.isRunning)
                HUDMetricView(value: tokensText, label: L("tokens"), animated: !model.isRunning)
                HUDMetricView(value: contextText, label: L("context"), animated: !model.isRunning)
                    .help(L("Tokens the last round occupied, of the context the model was loaded with."))
                HUDMetricView(value: memoryText, label: L("memory"), animated: !model.isRunning)
            }
            if !model.requiresModelInstallation {
                Divider().frame(height: 16)
                HeadroomGaugeView(headroom: model.machineHeadroom)
            }
        }
        .frame(height: 30)
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background {
            Capsule()
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    Capsule().stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
        .gesture(WindowDragGesture())
        .task {
            // The speedometer's clock: one host_statistics64 call every
            // two seconds, generating or not.
            while !Task.isCancelled {
                model.refreshMachineHeadroom()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    private var rateText: String {
        if model.phase == .decode { return MetricFormat.rate(model.liveTokensPerSecond) }
        if let d = model.diagnostics { return MetricFormat.rate(d.tokensPerSecond) }
        return "\u{2014}"
    }

    private var tokensText: String {
        if model.isRunning { return "\(model.liveTokenCount)" }
        if let d = model.diagnostics { return "\(d.generatedTokens)" }
        return "\u{2014}"
    }

    private var memoryText: String {
        MetricFormat.memory(model.currentProcessMemoryBytes)
    }

    /// "5.2K / 32K": the prompt as prefilled plus the answer, against the
    /// window. The number to watch on a long chat — past it the oldest
    /// turns fall out of what the model sees.
    private var contextText: String {
        let limit = MetricFormat.kiloTokens(model.maxContextTokens)
        guard let used = model.contextUsedTokens else { return "\u{2014} / \(limit)" }
        return "\(MetricFormat.kiloTokens(used)) / \(limit)"
    }

    private var showsMetrics: Bool {
        model.loadState.isReady || model.isRunning || model.diagnostics != nil
    }
}

private struct PhaseLabel: View {
    let model: AppModel

    var body: some View {
        HStack(spacing: 6) {
            switch content {
            case .loading(let label):
                ProgressView().controlSize(.mini)
                Text(label)
            case .pulse(let label):
                PulsingDot()
                Text(label)
                    .monospacedDigit()
                    .contentTransition(.numericText())
            case .steady(let label):
                Circle().fill(TsugumiMacTheme.accentColor).frame(width: 7, height: 7)
                Text(label).contentTransition(.opacity)
            case .quiet(let label):
                Text(label)
                    .foregroundStyle(.secondary)
                    .contentTransition(.opacity)
            }
        }
        .font(.caption.weight(.medium))
        .lineLimit(1)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(L("Model status"))
        .accessibilityValue(model.presentation.label)
    }

    private enum Content {
        case loading(String)
        case pulse(String)
        case steady(String)
        case quiet(String)
    }

    private var content: Content {
        let presentation = model.presentation
        if presentation.showsActivity { return .loading(presentation.label) }
        if model.isRunning && model.phase == .prefill { return .pulse(presentation.label) }
        if model.isRunning && model.phase == .decode { return .steady(presentation.label) }
        return .quiet(presentation.label)
    }
}

private struct PulsingDot: View {
    var body: some View {
        Circle()
            .fill(TsugumiMacTheme.accentColor)
            .frame(width: 7, height: 7)
            .phaseAnimator([0.4, 1.0]) { dot, opacity in
                dot.opacity(opacity)
            } animation: { _ in
                .easeInOut(duration: 0.7)
            }
    }
}
