import Foundation
import Observation

/// One conversation: its prompt draft, completed turns, and the live turn's
/// output fields. `AppModel` owns the list and routes generation events to
/// the session that started them, so a run keeps streaming into its own
/// chat while another one is on screen.
@MainActor
@Observable
public final class AppChatSession: Identifiable {
    public let id = UUID()
    public var promptText: String = ""
    public var attachedImagePaths: [String] = []
    /// Completed turns, oldest first; the live turn stays in the output
    /// fields below until the next run folds it in.
    public internal(set) var conversationTurns: [AppChatTurn] = []
    public internal(set) var outputPromptText: String = ""
    public internal(set) var outputImagePaths: [String] = []
    public var outputText: String = ""
    public internal(set) var outputReasoningText: String = ""
    /// The live turn's tool rounds so far: assistant turns that asked for
    /// calls, and the `tool` turns answering them. Folded between the user
    /// turn and the final answer.
    public internal(set) var outputContinuationTurns: [AppChatTurn] = []
    /// The same rounds as the UI lists them.
    public internal(set) var outputToolTrace: [AppToolTraceEntry] = []
    /// The line added to the question for this answer (a regeneration), or
    /// nil for the question as typed. `outputPromptText` stays the question.
    public internal(set) var outputDirective: AppAnswerDirective?
    /// Earlier answers to the live turn, oldest first, set aside by
    /// regenerations. The live output fields hold the one on display.
    public internal(set) var outputVariants: [AppAnswerVariant] = []
    /// Which answer is on display: an index into `outputVariants`, or
    /// `outputVariants.count` for the latest one (the live fields).
    public internal(set) var selectedVariantIndex: Int = 0

    public init() {}

    /// The user turn as the model saw it: the question plus the directive.
    public var outputPromptAsSent: String {
        outputDirective?.apply(to: outputPromptText) ?? outputPromptText
    }

    /// One more than the variants set aside: the answers there are to
    /// choose between.
    public var answerCount: Int { outputVariants.count + 1 }

    /// Sidebar label: the first sent prompt's first line, else a stand-in.
    public var title: String {
        let source = conversationTurns.first(where: { $0.role == .user })?.text
            ?? outputPromptText
        let firstLine = source.split(whereSeparator: \.isNewline).first
            .map(String.init) ?? ""
        let trimmed = firstLine.trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty ? AppLocalization.string("New Chat") : trimmed
    }

    /// True when nothing has been typed, attached, sent, or generated.
    public var isEmpty: Bool {
        conversationTurns.isEmpty && outputPromptText.isEmpty
            && outputText.isEmpty && outputReasoningText.isEmpty
            && outputContinuationTurns.isEmpty
            && promptText.isEmpty && attachedImagePaths.isEmpty
    }
}
