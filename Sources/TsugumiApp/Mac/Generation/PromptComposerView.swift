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
                searchControl
                webSearchControl
            }
            thinkingControl
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

    /// Two toggles for the three modes (`AppNetworkMode`): whether the
    /// turn looks anything up, and whether it may leave the Mac. Both off is
    /// Model only; Online switches Search on with it, and Search off takes
    /// Online down too, so each press is one sensible step.
    private var searchControl: some View {
        modeToggle(on: model.networkMode.usesTools,
                   systemImage: "magnifyingglass",
                   title: L("Search"),
                   help: model.networkMode.usesTools
                       ? L("Search is on: the model may look things up (Wikipedia from the local index; the web when Online).")
                       : model.networkMode.help,
                   accessibilityLabel: L("Search")) {
            model.networkMode = model.networkMode.usesTools ? .modelOnly : .offline
        }
    }

    private var webSearchControl: some View {
        modeToggle(on: model.networkMode == .online,
                   systemImage: "globe",
                   title: AppNetworkMode.online.label,
                   help: model.networkMode == .online
                       ? AppNetworkMode.online.help
                       : L("Online is off: nothing leaves this Mac."),
                   accessibilityLabel: L("Network")) {
            model.networkMode = model.networkMode == .online ? .offline : .online
        }
    }

    /// The chip the three mode controls share: icon only when off, icon and
    /// title tinted when on.
    private func modeToggle(on: Bool, systemImage: String, title: String, help: String,
                            accessibilityLabel: String,
                            action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: systemImage)
                if on {
                    Text(title)
                        .font(.caption.weight(.medium))
                }
            }
            .frame(height: 28)
            .padding(.horizontal, on ? 6 : 0)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .fixedSize()
        .foregroundStyle(on
                         ? AnyShapeStyle(TsugumiMacTheme.accentColor)
                         : AnyShapeStyle(.secondary))
        .background {
            if on {
                Capsule().fill(TsugumiMacTheme.accentColor.opacity(0.12))
            }
        }
        .disabled(model.isRunning)
        .help(help)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityValue(on ? L("On") : L("Off"))
    }

    /// The thought channel, on or off for the next turn. The same value as
    /// the Inspector's switch; here because it is the one knob that changes
    /// how long an answer takes.
    private var thinkingControl: some View {
        modeToggle(on: model.thinkingEnabled,
                   systemImage: "brain",
                   title: L("Thinking"),
                   help: model.thinkingEnabled
                       ? L("Thinking is on: the model reasons before it answers. Slower, better on hard questions.")
                       : L("Thinking is off: the model answers directly."),
                   accessibilityLabel: L("Thinking")) {
            model.thinkingEnabled.toggle()
        }
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
