import AppKit
import TsugumiAppCore
import SwiftUI

struct InspectorView: View {
    @Bindable var model: AppModel

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
        Section("Model") {
            Picker("Model", selection: Binding(
                get: { model.selectedModelKind },
                set: { model.selectModel($0) })) {
                ForEach(AppModelKind.allCases) { kind in
                    Text(kind.displayName).tag(kind)
                }
            }
            .pickerStyle(.menu)
            .disabled(model.isRunning || model.isInstallingModel || model.loadState.isLoading)
            Text(model.supportsVision
                 ? "Vision and MTP speculative decoding."
                 : "Text only, MTP speculative decoding. Sampling is pinned to the official values.")
                .font(.caption)
                .foregroundStyle(.secondary)
            LabeledContent("Path") {
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
                        Label("Copy model path", systemImage: "doc.on.doc")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                    .help("Copy model path")
                }
            }
            if model.canUnloadModel {
                Button("Unload Model", action: model.unloadModel)
            }
            LabeledContent("State") {
                Text(model.presentation.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if model.requiresModelInstallation {
                LabeledContent("Download") {
                    Text(MetricFormat.storage(model.installDescriptor.approximateDownloadBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Installed size") {
                    Text(MetricFormat.storage(model.installDescriptor.installedBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if let requirement = model.installRequirement {
                    LabeledContent("Available") {
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
        Section("Memory") {
            LabeledContent("Context") {
                Picker("Context", selection: $model.maxContextTokens) {
                    ForEach(AppContextLengthOption.allCases) { option in
                        Text(option.menuLabel).tag(option.tokens)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            LabeledContent("Slots") {
                Picker("Slots", selection: $model.runtimeOptions.expertCacheSlots) {
                    ForEach(AppRuntimeOptions.allowedSlotCounts, id: \.self) { slots in
                        Text(AppRuntimeOptions.slotsLabel(for: slots)).tag(slots)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            Text("More slots can improve decode speed by keeping more experts in memory, but they also use more RAM. Changes are compared with 4K context and 32 slots and apply after reloading the model.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if model.maxContextTokens == AppContextLengthOption.oneTwentyEightK.tokens {
                Text("128K fits, but decode can halve unless the GPU wired limit is raised (sudo sysctl iogpu.wired_limit_mb). 32K–64K is the everyday range.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var generationSection: some View {
        Section("Generation") {
            Toggle("Thinking", isOn: $model.thinkingEnabled)
                .toggleStyle(.switch)
            Text(model.selectedModelKind.thinkingDefault
                 ? "This model reasons before answering by default. Turning it off answers directly."
                 : "Off by default for this model. Turning it on lets the model reason before answering.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if model.isSamplingLocked {
                lockedSampling
            } else {
                editableSampling
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    /// The web tools: mode, the keys, and the two limits that decide how
    /// much of the context a turn may spend on search results.
    private var webSearchSection: some View {
        Section("Web search") {
            if model.webSearchAvailable {
                Picker("Mode", selection: $model.webSearchMode) {
                    ForEach(AppWebSearchMode.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                Text(model.webSearchMode.help)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("Web search is available with Gemma only.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            SecureField("Serper API key", text: $model.webSearchConfiguration.serperAPIKey)
                .textFieldStyle(.roundedBorder)
            SecureField("Brave Search API key (fallback)",
                        text: $model.webSearchConfiguration.braveAPIKey)
                .textFieldStyle(.roundedBorder)
            SecureField("Jina Reader API key (optional)",
                        text: $model.webSearchConfiguration.jinaAPIKey)
                .textFieldStyle(.roundedBorder)
            if !model.webSearchConfiguration.resolved().canUseTools {
                Text("Add a Serper or Brave key, or a local Wikipedia index, to enable the tools. Keys are stored in ~/Library/Application Support/Tsugumi/web-search.json (owner-readable only).")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            LocalWikipediaField(path: $model.webSearchConfiguration.wikipediaIndexPath)
            Toggle("Read pages with Jina Reader first",
                   isOn: $model.webSearchConfiguration.preferJinaReader)
                .toggleStyle(.switch)
            Text(model.webSearchConfiguration.preferJinaReader
                 ? "Jina Reader renders JavaScript pages; the app's own fetch is the fallback."
                 : "The app fetches pages itself; Jina Reader is the fallback for pages with little text.")
                .font(.caption)
                .foregroundStyle(.secondary)
            LabeledContent("Page text limit") {
                Stepper(value: $model.webSearchConfiguration.pageCharacterLimit,
                        in: 1_000...40_000, step: 1_000) {
                    Text("\(model.webSearchConfiguration.pageCharacterLimit) chars")
                        .monospacedDigit()
                }
                .fixedSize()
            }
            Picker("Thinking before the first search",
                   selection: $model.webSearchConfiguration.preSearchThinkingBudget) {
                Text("Off").tag(0)
                Text("256 tokens").tag(256)
                Text("512 tokens").tag(512)
                Text("1024 tokens").tag(1024)
                Text("Unlimited").tag(-1)
            }
            Text("With Thinking on, the round that decides the first search is held to this many thought tokens; rounds that read results think freely. Gemma 4 otherwise spends this round doubting the date.")
                .font(.caption)
                .foregroundStyle(.secondary)
            LabeledContent("Tool rounds per answer") {
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
        LabeledContent("Sampling") {
            let kind = model.selectedModelKind
            Text("temp \(kind.officialTemperature, format: .number.precision(.fractionLength(1))) · top-k \(kind.officialTopK) · top-p \(kind.officialTopP, format: .number.precision(.fractionLength(2))) (official, fixed)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var editableSampling: some View {
            LabeledContent("Temperature") {
                HStack(spacing: 8) {
                    Slider(value: $model.temperature, in: 0...2, step: 0.05)
                    Text(model.temperature, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            }
            Text("0 uses deterministic greedy decoding. Higher values make sampling more varied.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Toggle("Top-K", isOn: $model.topKEnabled)
                .toggleStyle(.switch)
            if model.topKEnabled {
                LabeledContent("K value") {
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
                LabeledContent("P value") {
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
        Section("Runtime") {
            Toggle("Prefill", isOn: $model.runtimeOptions.prefillEnabled)
            Toggle("MTP speculative decoding", isOn: $model.runtimeOptions.mtpEnabled)
                .disabled(!model.runtimeOptions.prefillEnabled)
            Text("MTP drafts tokens ahead and verifies them, often 1.2–1.4× faster decode. Requires prefill. Applies after reloading the model.")
                .font(.caption)
                .foregroundStyle(.secondary)
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
            Text("RDADVISE is experimental. It may speed up short decodes but slow down long decodes.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if model.hasStaleLoadedRuntime {
                Text("Reload required")
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
            TextField("Local Wikipedia index (wikipedia-ja.sqlite)", text: $path)
                .textFieldStyle(.roundedBorder)
            Button("Choose…") { choose() }
        }
        Text(status.isEmpty
             ? "Optional. Built from a Wikimedia dump by Scripts/wiki/build_jawiki_index.py; the tools then search Wikipedia offline. Works without any API key."
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
            let date = summary.dumpDateJapanese ?? "unknown date"
            status = "Japanese Wikipedia, \(summary.articles.formatted()) articles, dump of \(date)."
            isError = false
        case .failure(let error):
            status = "\(error)"
            isError = true
        }
    }
}
