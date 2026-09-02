import AppKit
import TsugumiAppCore
import TsugumiMacPresentation
import SwiftUI

struct OutputPaneView: View {
    let model: AppModel
    @State private var responseCopyFeedbackID: UUID?
    @State private var reasoningExpanded = false
    @State private var toolTraceExpanded = true

    var body: some View {
        Group {
            if model.hasOutputTranscript {
                transcript
            } else {
                placeholder
            }
        }
        .task(id: responseCopyFeedbackID) {
            guard let feedbackID = responseCopyFeedbackID else { return }
            try? await Task.sleep(for: .seconds(1.2))
            guard !Task.isCancelled, responseCopyFeedbackID == feedbackID else { return }
            withAnimation(.easeOut(duration: 0.15)) {
                responseCopyFeedbackID = nil
            }
        }
        .contextMenu {
            Button(L("Copy response")) {
                copyResponse()
            }
            .disabled(model.outputResponsePlainText.isEmpty)

            Button(L("Copy prompt")) {
                copy(model.outputPromptText)
            }
            .disabled(model.outputPromptText.isEmpty)

            Button(L("Copy conversation")) {
                copy(model.outputConversationPlainText)
            }
            .disabled(model.outputConversationPlainText.isEmpty)

            Divider()

            Button(L("Clear")) { model.clearOutput() }
                .disabled(model.isRunning || !model.hasOutputTranscript)
        }
    }

    private var placeholder: some View {
        EmptyConversationLayout(spacing: 8) {
            EmptyPlaceholderIcon(systemName: placeholderSymbol)
                .frame(width: 32, height: 32)

            emptyPlaceholderContent
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// Completed turns paired for display. `conversationTurns` is appended in
    /// user/assistant pairs by the fold, but the pairing loop tolerates a
    /// stray order rather than trapping on it.
    private var completedTurns: [InstructionTranscriptDocumentController.CompletedTurn] {
        var turns: [InstructionTranscriptDocumentController.CompletedTurn] = []
        var pendingPrompt: String?
        for turn in model.conversationTurns {
            switch turn.role {
            case .user:
                pendingPrompt = turn.text
            case .assistant:
                // A tool-calling assistant turn has no answer of its own;
                // the answer is the assistant turn after the tool results.
                guard turn.toolCalls.isEmpty else { continue }
                turns.append(InstructionTranscriptDocumentController.CompletedTurn(
                    prompt: pendingPrompt ?? "", response: turn.text))
                pendingPrompt = nil
            case .tool:
                continue
            }
        }
        if let pendingPrompt {
            turns.append(InstructionTranscriptDocumentController.CompletedTurn(
                prompt: pendingPrompt, response: ""))
        }
        return turns
    }

    private var transcript: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.outputToolTrace.isEmpty {
                toolTraceSection
            }
            if !model.outputReasoningText.isEmpty {
                reasoningSection
            }
            IncrementalTranscriptView(
                history: completedTurns,
                prompt: model.outputPromptText,
                output: model.outputText,
                mailbox: model.selectedChatTranscriptMailbox,
                isTerminal: !model.isSelectedChatGenerating,
                showsPrefillPlaceholder: model.isSelectedChatGenerating
                    && model.outputResponsePlainText.isEmpty
                    && model.outputReasoningText.isEmpty)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .overlay(alignment: .topTrailing) {
                    if !model.isSelectedChatGenerating && !model.outputResponsePlainText.isEmpty {
                        copyResponseButton
                            .padding(8)
                    }
                }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 20)
    }

    /// The thought channel, kept apart from the answer. While the model is
    /// still thinking (no answer text yet) the tail streams live; once the
    /// answer starts it collapses to a disclosure.
    private var reasoningSection: some View {
        let isThinkingLive = model.isSelectedChatGenerating
            && model.outputResponsePlainText.isEmpty
        return VStack(alignment: .leading, spacing: 6) {
            Button {
                reasoningExpanded.toggle()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "brain")
                        .font(.caption)
                    Text(isThinkingLive ? L("Thinking…") : L("Thought process"))
                        .font(.caption.weight(.medium))
                    Image(systemName: reasoningExpanded ? "chevron.down" : "chevron.right")
                        .font(.caption2)
                }
                .foregroundStyle(.secondary)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if reasoningExpanded || isThinkingLive {
                ScrollViewReader { proxy in
                    ScrollView {
                        // Live: only the tail, so per-token layout cost stays
                        // bounded however long the model thinks.
                        Text(isThinkingLive
                             ? ReasoningLivePresentation.liveTail(
                                of: model.outputReasoningText)
                             : model.outputReasoningText)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Color.clear.frame(height: 1).id("reasoning-tail")
                    }
                    .frame(maxHeight: isThinkingLive ? 160 : 280)
                    .onChange(of: model.outputReasoningText) {
                        guard isThinkingLive else { return }
                        proxy.scrollTo("reasoning-tail", anchor: .bottom)
                    }
                }
                .padding(10)
                .background {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color(nsColor: .controlBackgroundColor).opacity(0.6))
                }
            }
        }
    }

    /// The web search steps of the live turn: each query and each page, with
    /// what came back. Open while the loop runs; collapsible afterwards.
    private var toolTraceSection: some View {
        let entries = model.outputToolTrace
        let isLive = model.isSelectedChatGenerating
            && entries.contains { $0.status == .running }
        return VStack(alignment: .leading, spacing: 6) {
            Button {
                toolTraceExpanded.toggle()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "globe")
                        .font(.caption)
                    Text(isLive ? L("Searching the web…") : L("Web search (\(entries.count) steps)"))
                        .font(.caption.weight(.medium))
                    Image(systemName: toolTraceExpanded ? "chevron.down" : "chevron.right")
                        .font(.caption2)
                }
                .foregroundStyle(.secondary)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if toolTraceExpanded {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(entries) { entry in
                        toolTraceRow(entry)
                    }
                }
                .padding(10)
                .background {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color(nsColor: .controlBackgroundColor).opacity(0.6))
                }
            }
        }
    }

    private func toolTraceRow(_ entry: AppToolTraceEntry) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: entry.name == "fetch_page" ? "doc.text" : "magnifyingglass")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 14)
            VStack(alignment: .leading, spacing: 1) {
                Text(entry.subject)
                    .font(.callout)
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                if !entry.summary.isEmpty {
                    Text(entry.summary)
                        .font(.caption)
                        .foregroundStyle(entry.status == .failed ? .red : .secondary)
                        .lineLimit(2)
                }
            }
            Spacer(minLength: 0)
            switch entry.status {
            case .running:
                ProgressView().controlSize(.mini)
            case .done:
                Image(systemName: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            case .failed:
                Image(systemName: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
    }

    private var copyResponseButton: some View {
        Button {
            copyResponse()
        } label: {
            Image(systemName: responseCopyFeedbackID == nil
                  ? "doc.on.doc"
                  : "checkmark.circle.fill")
                .font(.callout.weight(.medium))
                .contentTransition(.symbolEffect(.replace))
                .foregroundStyle(responseCopyFeedbackID == nil
                                 ? Color.secondary
                                 : TsugumiMacTheme.accentColor)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
                .background(.regularMaterial, in: Circle())
                .overlay {
                    Circle().stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(responseCopyFeedbackID == nil
                            ? L("Copy response")
                            : L("Response copied"))
        .accessibilityHint(L("Copies only the generated answer"))
        .help(responseCopyFeedbackID == nil
              ? L("Copy response")
              : L("Response copied"))
    }

    private var emptyPlaceholderContent: some View {
        VStack(spacing: 8) {
            if !needsModelLoad {
                Text(L("Write a prompt to begin."))
                    .font(.headline)
            }
            if isLoadingModel {
                LoadingModelText()
                    .font(.callout)
                    .foregroundStyle(.tertiary)
            } else if let placeholderHint {
                Text(placeholderHint)
                    .font(.callout)
                    .foregroundStyle(.tertiary)
            }
            if let detail = model.presentation.detail {
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(model.presentation.severity == .error ? .red : .secondary)
                    .multilineTextAlignment(.center)
            }
            if model.canLoadModel {
                Button(model.loadState.isFailed ? L("Retry Load") : L("Load Model"),
                       action: model.loadModel)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
            } else if isLoadingModel {
                Button(L("Load Model"), action: {})
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .hidden()
                    .accessibilityHidden(true)
            } else if model.canReloadModel {
                Button(L("Reload Model"), action: model.reloadModel)
                    .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var needsModelLoad: Bool {
        !model.loadState.isReady
    }

    private var isLoadingModel: Bool {
        if case .loading = model.loadState { return true }
        return false
    }

    private var placeholderSymbol: String {
        "cube.transparent"
    }

    private var placeholderHint: String? {
        if model.loadState.isFailed { return L("The model could not be loaded") }
        if model.hasStaleLoadedRuntime { return L("Reload the model to use changed settings") }
        return needsModelLoad ? L("Load the model to begin") : nil
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    private func copyResponse() {
        copy(model.outputResponsePlainText)
        withAnimation(.easeIn(duration: 0.15)) {
            responseCopyFeedbackID = UUID()
        }
    }
}

private struct EmptyPlaceholderIcon: View {
    let systemName: String

    var body: some View {
        Image(systemName: systemName)
            .font(.title2)
            .foregroundStyle(.quaternary)
            .accessibilityHidden(true)
    }
}

private struct EmptyConversationLayout: Layout {
    let spacing: CGFloat

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) -> CGSize {
        proposal.replacingUnspecifiedDimensions()
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        guard subviews.count == 2 else { return }

        let iconSize = subviews[0].sizeThatFits(.unspecified)
        let iconCenter = CGPoint(x: bounds.midX, y: bounds.midY)
        subviews[0].place(
            at: iconCenter,
            anchor: .center,
            proposal: ProposedViewSize(
                width: iconSize.width,
                height: iconSize.height))

        subviews[1].place(
            at: CGPoint(
                x: bounds.midX,
                y: iconCenter.y + iconSize.height / 2 + spacing),
            anchor: .top,
            proposal: ProposedViewSize(width: bounds.width, height: nil))
    }
}

private struct LoadingModelText: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var animationStart = Date()

    var body: some View {
        if reduceMotion {
            label(dotCount: 3)
        } else {
            TimelineView(.periodic(from: .now, by: 0.25)) { context in
                let elapsed = max(0, context.date.timeIntervalSince(animationStart))
                label(dotCount: Int(elapsed / 0.25) % 4)
            }
        }
    }

    private func label(dotCount: Int) -> some View {
        ZStack(alignment: .leading) {
            Text(L("Loading Model") + "...").hidden()
            Text(L("Loading Model") + String(repeating: ".", count: dotCount))
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(L("Loading Model"))
    }
}

private struct IncrementalTranscriptView: NSViewRepresentable {
    var history: [InstructionTranscriptDocumentController.CompletedTurn] = []
    var prompt: String
    var output: String
    var mailbox: GenerationTranscriptMailbox?
    var isTerminal: Bool
    var showsPrefillPlaceholder: Bool

    @MainActor
    final class Coordinator: NSObject {
        weak var scrollView: NSScrollView?
        weak var textView: NSTextView?
        var mailbox: GenerationTranscriptMailbox?
        var history: [InstructionTranscriptDocumentController.CompletedTurn] = []
        var prompt = ""
        var isTerminal = false
        var showsPrefillPlaceholder = false
        var timer: Timer?
        var prefillAnimationTimer: Timer?
        let documentController = InstructionTranscriptDocumentController()

        func attach(scrollView: NSScrollView, textView: NSTextView) {
            self.scrollView = scrollView
            self.textView = textView
            guard timer == nil else { return }
            let timer = Timer(timeInterval: 0.1, target: self,
                              selector: #selector(drainMailbox),
                              userInfo: nil, repeats: true)
            timer.tolerance = 0.02
            RunLoop.main.add(timer, forMode: .common)
            self.timer = timer
        }

        func synchronize(
            history: [InstructionTranscriptDocumentController.CompletedTurn],
            prompt: String,
            output: String,
            mailbox: GenerationTranscriptMailbox?,
            isTerminal: Bool,
            showsPrefillPlaceholder: Bool
        ) {
            self.mailbox = mailbox
            self.history = history
            self.prompt = prompt
            self.isTerminal = isTerminal
            self.showsPrefillPlaceholder = showsPrefillPlaceholder
            let response = mailbox?.drain().completeText ?? output
            apply(
                prompt: prompt,
                response: response,
                isTerminal: isTerminal,
                showsPrefillPlaceholder: showsPrefillPlaceholder)
        }

        @objc private func drainMailbox() {
            guard let mailbox else { return }
            let snapshot = mailbox.drain()
            guard !snapshot.pendingText.isEmpty
                    || snapshot.completeText != documentController.response else {
                return
            }
            apply(prompt: prompt,
                  response: snapshot.completeText,
                  isTerminal: isTerminal,
                  showsPrefillPlaceholder: showsPrefillPlaceholder)
        }

        @objc private func animatePrefillPlaceholderIfNeeded() {
            guard documentController.showsPrefillPlaceholder,
                  let scrollView,
                  let textView,
                  let storage = textView.textStorage else { return }
            let wasAtBottom = isAtBottom(scrollView)
            let selection = textView.selectedRanges.map(\.rangeValue)

            storage.beginEditing()
            let changed = documentController.advancePrefillAnimation(storage: storage)
            storage.endEditing()
            guard changed else { return }

            let restored = InstructionTranscriptDocumentController.clampedRanges(
                selection,
                toLength: storage.length)
            if restored.isEmpty {
                textView.setSelectedRange(NSRange(location: storage.length, length: 0))
            } else {
                textView.selectedRanges = restored.map(NSValue.init(range:))
            }
            if wasAtBottom { textView.scrollToEndOfDocument(nil) }
        }

        func invalidate() {
            timer?.invalidate()
            timer = nil
            stopPrefillAnimationTimer()
            mailbox = nil
        }

        private func updatePrefillAnimationTimer() {
            if documentController.showsPrefillPlaceholder {
                guard prefillAnimationTimer == nil else { return }
                let timer = Timer(
                    timeInterval: 0.25,
                    target: self,
                    selector: #selector(animatePrefillPlaceholderIfNeeded),
                    userInfo: nil,
                    repeats: true)
                timer.tolerance = 0.025
                RunLoop.main.add(timer, forMode: .common)
                prefillAnimationTimer = timer
            } else {
                stopPrefillAnimationTimer()
            }
        }

        private func stopPrefillAnimationTimer() {
            prefillAnimationTimer?.invalidate()
            prefillAnimationTimer = nil
        }

        private func apply(
            prompt: String,
            response: String,
            isTerminal: Bool,
            showsPrefillPlaceholder: Bool
        ) {
            guard let scrollView, let textView, let storage = textView.textStorage else { return }
            let wasAtBottom = isAtBottom(scrollView)
            let selection = textView.selectedRanges.map(\.rangeValue)

            storage.beginEditing()
            let update = documentController.synchronize(
                storage: storage,
                history: history,
                prompt: prompt,
                response: response,
                isTerminal: isTerminal,
                showsPrefillPlaceholder: showsPrefillPlaceholder)
            storage.endEditing()
            updatePrefillAnimationTimer()

            guard update.mutation != .none else { return }
            let restored = InstructionTranscriptDocumentController.clampedRanges(
                selection,
                toLength: storage.length)
            if restored.isEmpty {
                textView.setSelectedRange(NSRange(location: storage.length, length: 0))
            } else {
                textView.selectedRanges = restored.map(NSValue.init(range:))
            }
            if InstructionTranscriptDocumentController.shouldScrollToBottom(
                wasAtBottom: wasAtBottom,
                mutation: update.mutation
            ) {
                if let textContainer = textView.textContainer {
                    textView.layoutManager?.ensureLayout(for: textContainer)
                }
                textView.scrollToEndOfDocument(nil)
            }
        }

        private func isAtBottom(_ scrollView: NSScrollView) -> Bool {
            guard let document = scrollView.documentView else { return true }
            let visible = scrollView.contentView.bounds
            return visible.maxY >= document.bounds.maxY - 24
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.drawsBackground = false

        let textView = NSTextView()
        textView.isEditable = false
        textView.isSelectable = true
        textView.isRichText = true
        textView.drawsBackground = false
        textView.textContainerInset = NSSize(width: 0, height: 4)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.lineFragmentPadding = 0
        textView.isAutomaticLinkDetectionEnabled = false
        textView.isAutomaticDataDetectionEnabled = false
        textView.setAccessibilityLabel(L("Conversation transcript"))
        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? NSTextView else { return }
        context.coordinator.attach(scrollView: scrollView, textView: textView)
        context.coordinator.synchronize(
            history: history,
            prompt: prompt,
            output: output,
            mailbox: mailbox,
            isTerminal: isTerminal,
            showsPrefillPlaceholder: showsPrefillPlaceholder)
    }

    static func dismantleNSView(_ nsView: NSScrollView, coordinator: Coordinator) {
        coordinator.invalidate()
    }
}

#if DEBUG
private struct TranscriptPreview: View {
    let response: String
    let isTerminal: Bool
    var showsPrefillPlaceholder = false

    var body: some View {
        IncrementalTranscriptView(
            prompt: "Explain this clearly.",
            output: response,
            mailbox: nil,
            isTerminal: isTerminal,
            showsPrefillPlaceholder: showsPrefillPlaceholder)
            .padding(24)
            .frame(width: 720, height: 420)
    }
}

#Preview("Empty") {
    VStack(spacing: 8) {
        Image(systemName: "cube.transparent")
            .font(.title2)
            .foregroundStyle(.quaternary)
        Text("Write a prompt to begin.")
            .font(.headline)
    }
    .frame(width: 720, height: 420)
}

#Preview("Streaming") {
    TranscriptPreview(
        response: "A response arriving one readable piece at a time...",
        isTerminal: false)
}

#Preview("Prefilling") {
    TranscriptPreview(
        response: "",
        isTerminal: false,
        showsPrefillPlaceholder: true)
}

#Preview("Completed prose") {
    TranscriptPreview(
        response: "# A clear answer\n\nHere is a concise explanation with **useful emphasis**.\n\n- First point\n- Second point",
        isTerminal: true)
}

#Preview("Completed code") {
    TranscriptPreview(
        response: "Use `fibonacci(7)`:\n\n```python\ndef fibonacci(n: int) -> list[int]:\n    return []\n```",
        isTerminal: true)
}

#Preview("Incomplete Markdown fallback") {
    TranscriptPreview(
        response: "The partial answer remains readable.\n\n```python\nprint('unfinished')",
        isTerminal: true)
}
#endif
