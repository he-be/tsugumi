import AppKit
import TsugumiAppCore
import TsugumiMacPresentation
import SwiftUI

/// The row under the last answer: what the answer leaned on, which of the
/// answers to this question is on display, and the ways to ask again.
/// Only the live turn has it — earlier turns are history the model has
/// already seen, and a change there would mean a different conversation.
struct AnswerActionsRow: View {
    let model: AppModel
    @State private var copyFeedbackID: UUID?

    var body: some View {
        HStack(spacing: 8) {
            if model.toolsAvailable {
                groundingBadge
                if model.outputLacksCitation {
                    Label(L("No sources named"), systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                        .help(L("The model searched but named no page or URL: the answer was written from memory."))
                }
            }
            if let directive = model.outputDirective {
                tag(directive.label)
            }
            if model.answerCount > 1 {
                variantSwitcher
            }
            Spacer(minLength: 8)
            if model.canSearchAgain {
                chip(AppAnswerDirective.searched.label, systemImage: "magnifyingglass",
                     tint: .orange) {
                    model.regenerate(.searched)
                }
            }
            chip(AppAnswerDirective.concise.label) { model.regenerate(.concise) }
            chip(AppAnswerDirective.blunt.label) { model.regenerate(.blunt) }
            chip(AppFollowUp.opposite.label) { model.askFollowUp(.opposite) }
            iconButton("arrow.clockwise", help: AppAnswerDirective.again.label) {
                model.regenerate(.again)
            }
            copyButton
        }
        .font(.caption)
        .task(id: copyFeedbackID) {
            guard let feedbackID = copyFeedbackID else { return }
            try? await Task.sleep(for: .seconds(1.2))
            guard !Task.isCancelled, copyFeedbackID == feedbackID else { return }
            withAnimation(.easeOut(duration: 0.15)) { copyFeedbackID = nil }
        }
    }

    // MARK: Grounding

    /// Web steps first (searches and pages read counted apart: a search
    /// with no page read is orange, the model saw snippets only), Wikipedia
    /// alone second, nothing third. "No search" is orange when the model
    /// could have searched: that is the case the app exists to make visible.
    private var groundingBadge: some View {
        let grounding = model.outputGrounding
        let couldHaveSearched = model.webSearchConfiguration.resolved().canSearch
        let (symbol, text, color): (String, String, Color) = {
            if grounding.webSteps > 0 {
                return ("globe",
                        L("Web · \(grounding.webSearches) searched · \(grounding.pagesRead) read"),
                        grounding.pagesRead > 0 ? .secondary : .orange)
            }
            if grounding.wikipediaSteps > 0 {
                return ("book.closed", L("Wikipedia · \(grounding.wikipediaSteps)"), .secondary)
            }
            return ("exclamationmark.triangle", L("No search"),
                    couldHaveSearched ? .orange : .secondary)
        }()
        return Label(text, systemImage: symbol)
            .foregroundStyle(color)
            .help(grounding.isEmpty
                  ? L("The answer came from the model alone.")
                  : grounding.webSearches > 0 && grounding.pagesRead == 0
                  ? L("The model searched but read no page: it saw snippets only.")
                  : L("Steps the answer was grounded on; open the list above for each one."))
    }

    private func tag(_ text: String) -> some View {
        Text(text)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Capsule().fill(Color(nsColor: .controlBackgroundColor)))
    }

    // MARK: Variants

    private var variantSwitcher: some View {
        HStack(spacing: 2) {
            iconButton("chevron.left", help: L("Previous answer")) {
                model.selectVariant(model.selectedVariantIndex - 1)
            }
            .disabled(model.selectedVariantIndex == 0 || model.isRunning)
            Text("\(model.selectedVariantIndex + 1)/\(model.answerCount)")
                .monospacedDigit()
                .foregroundStyle(.secondary)
            iconButton("chevron.right", help: L("Next answer")) {
                model.selectVariant(model.selectedVariantIndex + 1)
            }
            .disabled(model.selectedVariantIndex + 1 >= model.answerCount || model.isRunning)
        }
    }

    // MARK: Buttons

    private func chip(_ title: String, systemImage: String? = nil,
                      tint: Color = .secondary, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                if let systemImage {
                    Image(systemName: systemImage)
                }
                Text(title)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .foregroundStyle(tint)
        .background {
            Capsule().stroke(.separator, lineWidth: 0.5)
        }
        .disabled(!model.canRegenerate)
        .help(L("Asks the same question again this way. The current answer stays as an earlier one."))
    }

    private func iconButton(_ systemImage: String, help: String,
                            action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .frame(width: 22, height: 22)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
        .disabled(!model.canRegenerate)
        .help(help)
        .accessibilityLabel(help)
    }

    private var copyButton: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(model.outputResponsePlainText, forType: .string)
            withAnimation(.easeOut(duration: 0.15)) { copyFeedbackID = UUID() }
        } label: {
            Image(systemName: copyFeedbackID == nil ? "doc.on.doc" : "checkmark.circle.fill")
                .contentTransition(.symbolEffect(.replace))
                .foregroundStyle(copyFeedbackID == nil ? Color.secondary : TsugumiMacTheme.accentColor)
                .frame(width: 22, height: 22)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .help(copyFeedbackID == nil ? L("Copy response") : L("Response copied"))
        .accessibilityLabel(copyFeedbackID == nil ? L("Copy response") : L("Response copied"))
    }
}
