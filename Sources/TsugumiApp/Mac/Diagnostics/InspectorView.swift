import AppKit
import TsugumiAppCore
import SwiftUI

struct InspectorView: View {
    @Bindable var model: AppModel
    @State private var samplingExpanded = false

    var body: some View {
        Form {
            modelSection
            memorySection
            generationSection
            webSearchSection
            runtimeSection
            RunnerDiagnosticsSection(diagnostics: model.diagnostics)
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var modelSection: some View {
        Section(L("Model")) {
            Picker(L("Model"), selection: Binding(
                get: { model.selectedModelKind },
                set: { model.selectModel($0) })) {
                ForEach(AppModelKind.allCases) { kind in
                    Text(kind.displayName).tag(kind)
                }
            }
            .pickerStyle(.menu)
            .disabled(model.isRunning || model.isInstallingModel || model.loadState.isLoading)
            LabeledContent(L("Path")) {
                HStack(spacing: 6) {
                    Text(model.modelPathText)
                        .font(.caption)
                        .truncationMode(.middle)
                        .lineLimit(1)
                        .foregroundStyle(.secondary)
                        .help(model.modelPathText)
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(model.modelPathText, forType: .string)
                    } label: {
                        Label(L("Copy model path"), systemImage: "doc.on.doc")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                    .help(L("Copy model path"))
                }
            }
            if model.canUnloadModel {
                Button(L("Unload Model"), action: model.unloadModel)
            }
            LabeledContent(L("State")) {
                Text(model.presentation.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if model.requiresModelInstallation {
                LabeledContent(L("Download")) {
                    Text(MetricFormat.storage(model.installDescriptor.approximateDownloadBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                LabeledContent(L("Installed size")) {
                    Text(MetricFormat.storage(model.installDescriptor.installedBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if let requirement = model.installRequirement {
                    LabeledContent(L("Available")) {
                        Text(MetricFormat.storage(requirement.availableBytes))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .disabled(model.isRunning || model.isInstallingModel)
    }

    private var memorySection: some View {
        Section(L("Memory")) {
            LabeledContent(L("Context")) {
                Picker(L("Context"), selection: $model.maxContextTokens) {
                    ForEach(AppContextLengthOption.allCases) { option in
                        Text(option.menuLabel).tag(option.tokens)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            LabeledContent(L("Slots")) {
                Picker(L("Slots"), selection: $model.runtimeOptions.expertCacheSlots) {
                    ForEach(AppRuntimeOptions.allowedSlotCounts, id: \.self) { slots in
                        Text(AppRuntimeOptions.slotsLabel(for: slots)).tag(slots)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            if model.maxContextTokens == AppContextLengthOption.oneTwentyEightK.tokens {
                Text(L("128K fits, but decode can be about half as fast. 32K–64K is the everyday range."))
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var generationSection: some View {
        Section(L("Generation")) {
            Toggle(L("Thinking"), isOn: $model.thinkingEnabled)
                .toggleStyle(.switch)
            DisclosureGroup(L("Sampling"), isExpanded: $samplingExpanded) {
                if model.isSamplingLocked {
                    lockedSampling
                } else {
                    editableSampling
                }
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    /// The web tools: mode, the keys, and the two limits that decide how
    /// much of the context a turn may spend on search results.
    private var webSearchSection: some View {
        Section(L("Web search")) {
            if model.webSearchAvailable {
                Picker(L("Mode"), selection: $model.webSearchMode) {
                    ForEach(AppWebSearchMode.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                Text(model.webSearchMode.help)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text(L("Web search is available with Gemma only."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            SecureField(L("Serper API key"), text: $model.webSearchConfiguration.serperAPIKey)
                .textFieldStyle(.roundedBorder)
            SecureField(L("Brave Search API key (fallback)"),
                        text: $model.webSearchConfiguration.braveAPIKey)
                .textFieldStyle(.roundedBorder)
            SecureField(L("Jina Reader API key (optional)"),
                        text: $model.webSearchConfiguration.jinaAPIKey)
                .textFieldStyle(.roundedBorder)
            if !model.webSearchConfiguration.resolved().canUseTools {
                Text(L("Add a Serper or Brave key, or a local Wikipedia index."))
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            LocalWikipediaField(path: $model.webSearchConfiguration.wikipediaIndexPath)
            Toggle(L("Read pages with Jina Reader first"),
                   isOn: $model.webSearchConfiguration.preferJinaReader)
                .toggleStyle(.switch)
            LabeledContent(L("Page text limit")) {
                Stepper(value: $model.webSearchConfiguration.pageCharacterLimit,
                        in: 1_000...40_000, step: 1_000) {
                    Text(L("\(model.webSearchConfiguration.pageCharacterLimit) chars"))
                        .monospacedDigit()
                }
                .fixedSize()
            }
            Picker(L("Thinking before the first search"),
                   selection: $model.webSearchConfiguration.preSearchThinkingBudget) {
                Text(L("Off")).tag(0)
                Text(L("\(256) tokens")).tag(256)
                Text(L("\(512) tokens")).tag(512)
                Text(L("\(1024) tokens")).tag(1024)
                Text(L("Unlimited")).tag(-1)
            }
            LabeledContent(L("Tool rounds per answer")) {
                Stepper(value: $model.webSearchConfiguration.maxToolRounds,
                        in: 1...12, step: 1) {
                    Text("\(model.webSearchConfiguration.maxToolRounds)")
                        .monospacedDigit()
                }
                .fixedSize()
            }
        }
        .onChange(of: model.webSearchConfiguration) {
            model.saveWebSearchConfiguration()
        }
        .disabled(model.isRunning)
    }

    private var lockedSampling: some View {
        let kind = model.selectedModelKind
        let temperature = kind.officialTemperature.formatted(.number.precision(.fractionLength(1)))
        let topP = kind.officialTopP.formatted(.number.precision(.fractionLength(2)))
        return Text(L("temp \(temperature) · top-k \(kind.officialTopK) · top-p \(topP) (official, fixed)"))
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
    }

    @ViewBuilder
    private var editableSampling: some View {
        LabeledContent(L("Temperature")) {
            HStack(spacing: 8) {
                Slider(value: $model.temperature, in: 0...2, step: 0.05)
                Text(model.temperature, format: .number.precision(.fractionLength(2)))
                    .monospacedDigit()
                    .frame(width: 36, alignment: .trailing)
            }
        }
        Toggle("Top-K", isOn: $model.topKEnabled)
            .toggleStyle(.switch)
        if model.topKEnabled {
            LabeledContent(L("K value")) {
                Stepper(value: $model.topK, in: 1...256, step: 1) {
                    Text("\(model.topK)").monospacedDigit()
                }
                .fixedSize()
            }
        }
        Toggle("Top-P", isOn: $model.topPEnabled)
            .toggleStyle(.switch)
            .disabled(!model.topKEnabled)
        if model.topKEnabled && model.topPEnabled {
            LabeledContent(L("P value")) {
                HStack(spacing: 8) {
                    Slider(value: $model.topP, in: 0.01...1, step: 0.01)
                    Text(model.topP, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            }
        }
    }

    private var runtimeSection: some View {
        Section(L("Runtime")) {
            Toggle(L("Prefill"), isOn: $model.runtimeOptions.prefillEnabled)
            Toggle(L("MTP speculative decoding"), isOn: $model.runtimeOptions.mtpEnabled)
                .disabled(!model.runtimeOptions.prefillEnabled)
            VStack(alignment: .leading, spacing: 8) {
                Text("RDADVISE")
                Picker("RDADVISE", selection: $model.runtimeOptions.rdadvisePolicy) {
                    ForEach(AppRDAdvicePolicy.allCases) { policy in
                        Text(policy.label).tag(policy)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            Text(L("RDADVISE is experimental. Changes apply after reloading the model."))
                .font(.caption)
                .foregroundStyle(.secondary)
            if model.hasStaleLoadedRuntime {
                Text(L("Reload required"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

}

/// The local Wikipedia index: its path, a chooser, and what the file says
/// about itself (or why it cannot be opened). The probe opens the file on
/// a background task so a slow disk does not stall the Inspector.
private struct LocalWikipediaField: View {
    @Binding var path: String
    @State private var status: String = ""
    @State private var isError = false

    var body: some View {
        HStack {
            TextField(L("Local Wikipedia index (wikipedia-ja.sqlite)"), text: $path)
                .textFieldStyle(.roundedBorder)
            Button(L("Choose…")) { choose() }
        }
        Text(status.isEmpty
             ? L("Optional. Searches Wikipedia offline; needs no API key.")
             : status)
            .font(.caption)
            .foregroundStyle(isError ? .orange : .secondary)
            .task(id: path) { await probe() }
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.database, .data]
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
        }
    }

    private func probe() async {
        let trimmed = path.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else {
            status = ""
            isError = false
            return
        }
        let expanded = (trimmed as NSString).expandingTildeInPath
        let result = await Task.detached { LocalWikipediaIndex.probe(path: expanded) }.value
        guard !Task.isCancelled else { return }
        switch result {
        case .success(let summary):
            let date = summary.dumpDateJapanese ?? L("unknown date")
            status = L("Japanese Wikipedia, \(summary.articles.formatted()) articles, dump of \(date).")
            isError = false
        case .failure(let error):
            status = "\(error)"
            isError = true
        }
    }
}
