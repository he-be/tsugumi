import AppKit
import TsugumiAppCore
import TsugumiMacPresentation
import SwiftUI
import UniformTypeIdentifiers

struct PromptComposerView: View {
    @Bindable var model: AppModel
    @FocusState private var promptFocused: Bool
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
            .accessibilityLabel(L("Prompt"))
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
                    Text(L("Prompt"))
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
            if model.supportsVision {
                attachImagesButton
            }
            if model.toolsAvailable {
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
            Label(L("Attach images"), systemImage: "photo.badge.plus")
                .labelStyle(.iconOnly)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .disabled(model.isRunning || model.attachedImagePaths.count >= 4)
        .help(L("Attach up to 4 images (PNG, JPEG, WebP)"))
        .fileImporter(isPresented: $showingImageImporter,
                      allowedContentTypes: [.png, .jpeg, .webP, .gif],
                      allowsMultipleSelection: true) { result in
            guard case .success(let urls) = result else { return }
            model.attachImages(urls.map(\.path))
        }
    }

    /// Offline / Online. Tinted when online, so that anything leaving the
    /// Mac is readable without opening the menu.
    private var webSearchControl: some View {
        Menu {
            Picker(L("Network"), selection: $model.networkMode) {
                ForEach(AppNetworkMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
            .pickerStyle(.inline)
            .labelsHidden()
            Divider()
            Text(model.networkMode.help)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "globe")
                if model.networkMode == .online {
                    Text(model.networkMode.label)
                        .font(.caption.weight(.medium))
                }
            }
            .frame(height: 28)
            .padding(.horizontal, model.networkMode == .offline ? 0 : 6)
            .contentShape(Capsule())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .foregroundStyle(model.networkMode == .offline
                         ? AnyShapeStyle(.secondary)
                         : AnyShapeStyle(TsugumiMacTheme.accentColor))
        .background {
            if model.networkMode == .online {
                Capsule().fill(TsugumiMacTheme.accentColor.opacity(0.12))
            }
        }
        .disabled(model.isRunning)
        .help("\(L("Network")): \(model.networkMode.label) — \(model.networkMode.help)")
        .accessibilityLabel("\(L("Network")) \(model.networkMode.label)")
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
            .help(L("Remove image"))
        }
        .padding(4)
        .background {
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(nsColor: .windowBackgroundColor))
        }
    }

    @ViewBuilder
    private var clearAction: some View {
        if !model.isRunning && model.hasOutputTranscript {
            Button {
                model.clearOutput()
            } label: {
                Label(L("Clear output"), systemImage: "trash")
                    .labelStyle(.iconOnly)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help(L("Clear output"))
        } else if !model.isRunning && !model.promptText.isEmpty {
            Button {
                model.promptText = ""
                promptFocused = true
            } label: {
                Label(L("Clear prompt"), systemImage: "xmark.circle.fill")
                    .labelStyle(.iconOnly)
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help(L("Clear prompt"))
        }
    }
}
