import AppKit
import TsugumiAppCore
import TsugumiMacPresentation
import SwiftUI
import UniformTypeIdentifiers

struct PromptComposerView: View {
    @Bindable var model: AppModel
    @FocusState private var promptFocused: Bool
    @State private var showingPromptTips = false
    @State private var showingImageImporter = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.attachedImagePaths.isEmpty {
                attachmentsRow
            }
            editor
            footer
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: 22)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 22)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
    }

    private var editor: some View {
        TextEditor(text: $model.promptText)
            .accessibilityLabel("Prompt")
            .font(.body)
            .scrollContentBackground(.hidden)
            .focused($promptFocused)
            .onKeyPress(.return, phases: [.down, .repeat]) { keyPress in
                switch PromptSubmissionPolicy.decision(
                    newlineShortcut: model.newlineShortcut,
                    modifiers: keyPress.modifiers,
                    canRun: model.canRun,
                    hasMarkedText: promptHasMarkedText,
                    isRepeat: keyPress.phase.contains(.repeat)) {
                case .submit:
                    model.run()
                    return .handled
                case .consume:
                    return .handled
                case .deferToEditor:
                    return .ignored
                }
            }
            .frame(height: editorHeight)
            .overlay(alignment: .topLeading) {
                if model.promptText.isEmpty {
                    // Matches the NSTextView text origin: 5pt line fragment
                    // padding, no vertical inset.
                    Text("Prompt")
                        .font(.body)
                        .foregroundStyle(.tertiary)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
            }
    }

    private var promptHasMarkedText: Bool {
        (NSApp.keyWindow?.firstResponder as? NSTextView)?.hasMarkedText() == true
    }

    private var editorHeight: CGFloat {
        model.promptText.isEmpty ? 46 : 84
    }

    private var footer: some View {
        HStack(spacing: 10) {
            promptTips
            if model.supportsVision {
                attachImagesButton
            }
            if model.webSearchAvailable {
                webSearchControl
            }
            Spacer()
            clearAction
            GenerateControl(model: model)
        }
    }

    private var attachImagesButton: some View {
        Button {
            showingImageImporter = true
        } label: {
            Label("Attach images", systemImage: "photo.badge.plus")
                .labelStyle(.iconOnly)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .disabled(model.isRunning || model.attachedImagePaths.count >= 4)
        .help("Attach up to 4 images (PNG, JPEG, WebP)")
        .fileImporter(isPresented: $showingImageImporter,
                      allowedContentTypes: [.png, .jpeg, .webP, .gif],
                      allowsMultipleSelection: true) { result in
            guard case .success(let urls) = result else { return }
            model.attachImages(urls.map(\.path))
        }
    }

    /// Off / Auto / Always for the web tools. Tinted when on, so the state
    /// is readable without opening the menu.
    private var webSearchControl: some View {
        Menu {
            Picker("Web search", selection: $model.webSearchMode) {
                ForEach(AppWebSearchMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
            .pickerStyle(.inline)
            .labelsHidden()
            Divider()
            Text(model.webSearchMode.help)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "globe")
                if model.webSearchMode != .off {
                    Text(model.webSearchMode.label)
                        .font(.caption.weight(.medium))
                }
            }
            .frame(height: 28)
            .padding(.horizontal, model.webSearchMode == .off ? 0 : 6)
            .contentShape(Capsule())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .foregroundStyle(model.webSearchMode == .off
                         ? AnyShapeStyle(.secondary)
                         : AnyShapeStyle(TsugumiMacTheme.accentColor))
        .background {
            if model.webSearchMode != .off {
                Capsule().fill(TsugumiMacTheme.accentColor.opacity(0.12))
            }
        }
        .disabled(model.isRunning)
        .help("Web search: \(model.webSearchMode.label) — \(model.webSearchMode.help)")
        .accessibilityLabel("Web search \(model.webSearchMode.label)")
    }

    private var attachmentsRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(model.attachedImagePaths, id: \.self) { path in
                    attachmentThumbnail(path)
                }
            }
        }
        .frame(height: 52)
    }

    private func attachmentThumbnail(_ path: String) -> some View {
        HStack(spacing: 6) {
            if let image = NSImage(contentsOfFile: path) {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 44, height: 44)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                Image(systemName: "photo")
                    .frame(width: 44, height: 44)
            }
            Text((path as NSString).lastPathComponent)
                .font(.caption)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: 120)
            Button {
                model.removeAttachedImage(path)
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .symbolRenderingMode(.hierarchical)
            }
            .buttonStyle(.borderless)
            .help("Remove image")
        }
        .padding(4)
        .background {
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(nsColor: .windowBackgroundColor))
        }
    }

    private var promptTips: some View {
        Button {
            showingPromptTips.toggle()
        } label: {
            Label("Prompt tips", systemImage: "questionmark.circle")
                .labelStyle(.iconOnly)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help("Prompt tips")
        .popover(isPresented: $showingPromptTips,
                 attachmentAnchor: .point(.top),
                 arrowEdge: .top) {
            promptGuide
        }
    }

    private var promptGuide: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Prompting this model")
                .font(.headline)

            tipSection("Ask for a clear task",
                       "Say what you want the model to create, explain, plan, or transform. Put the essential context in the same prompt.")
            tipSection("Shape the answer",
                       "Specify a useful length, sections, tone, or output format. Concrete constraints work better than a long list of vague preferences.")
            tipSection("Anchor important facts",
                       "Include facts the answer must preserve and say what should be checked. Generated factual claims can still be wrong or outdated.")
            tipSection("For code and calculations",
                       "Provide types, dimensions, interfaces, edge cases, or a small scaffold. Compile or run the result before relying on it.")
            tipSection("Try a focused revision",
                       "If the answer drifts, shorten the task and make the missing requirement explicit. The sampler defaults to each model's official recommendation.")
        }
        .font(.callout)
        .frame(width: 390, alignment: .leading)
        .padding(18)
    }

    private func tipSection(_ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .fontWeight(.semibold)
            Text(detail)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private var clearAction: some View {
        if !model.isRunning && model.hasOutputTranscript {
            Button {
                model.clearOutput()
            } label: {
                Label("Clear output", systemImage: "trash")
                    .labelStyle(.iconOnly)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Clear output")
        } else if !model.isRunning && !model.promptText.isEmpty {
            Button {
                model.promptText = ""
                promptFocused = true
            } label: {
                Label("Clear prompt", systemImage: "xmark.circle.fill")
                    .labelStyle(.iconOnly)
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Clear prompt")
        }
    }
}
