import Foundation

/// A way to ask the same question again. The instruction travels as one
/// line at the end of the user turn — after the question, so the cached
/// prefix up to it stays valid — and the folded history keeps that line,
/// because the next request has to redraw the turn as the model saw it.
public enum AppAnswerDirective: String, Codable, Equatable, Sendable, CaseIterable, Identifiable {
    /// The same question, nothing added: another draw of the sampler.
    case again
    case concise
    case blunt
    /// Search first: the run goes online and the first round must call a
    /// tool, whatever the switch says.
    case searched

    public var id: String { rawValue }

    /// The line appended to the question; empty for `again`.
    public var instruction: String {
        switch self {
        case .again: return ""
        case .concise: return "(短く答えてください。要点だけを数文で。)"
        case .blunt: return "(率直に答えてください。遠回しな表現や両論併記を避け、自分の判断を言い切ってください。)"
        case .searched:
            return "(まず web_search で調べ、検索結果から 1〜2 ページを fetch_page で読んでください。答えはその本文に基づいて書き、使ったページの URL を最後に「参照:」として挙げてください。)"
        }
    }

    public var label: String {
        switch self {
        case .again: return AppLocalization.string("Regenerate")
        case .concise: return AppLocalization.string("Shorter")
        case .blunt: return AppLocalization.string("Blunt")
        case .searched: return AppLocalization.string("Search and answer again")
        }
    }

    /// The user turn as it is sent: the question, then the instruction.
    public func apply(to prompt: String) -> String {
        instruction.isEmpty ? prompt : prompt + "\n\n" + instruction
    }
}

/// A question that opens a new turn under the current answer instead of
/// replacing it. The text is what the composer would have sent.
public enum AppFollowUp: String, Codable, Equatable, Sendable, CaseIterable, Identifiable {
    case opposite

    public var id: String { rawValue }

    public var prompt: String {
        switch self {
        case .opposite:
            return "反対の立場で答えてください。さっきの回答に対して、逆の見方をする人ならどう主張するかを、その人の立場から述べてください。"
        }
    }

    public var label: String {
        switch self {
        case .opposite: return AppLocalization.string("Opposite view")
        }
    }
}

/// One earlier answer to the live turn, set aside by a regeneration. Holds
/// everything the live output fields hold, so switching back is a swap.
public struct AppAnswerVariant: Codable, Equatable, Sendable {
    public var directive: AppAnswerDirective?
    public var text: String
    public var reasoningText: String
    public var continuationTurns: [AppChatTurn]
    public var toolTrace: [AppToolTraceEntry]
    public var networkMode: AppNetworkMode?

    public init(directive: AppAnswerDirective?,
                text: String,
                reasoningText: String = "",
                continuationTurns: [AppChatTurn] = [],
                toolTrace: [AppToolTraceEntry] = [],
                networkMode: AppNetworkMode? = nil) {
        self.directive = directive
        self.networkMode = networkMode
        self.text = text
        self.reasoningText = reasoningText
        self.continuationTurns = continuationTurns
        self.toolTrace = toolTrace
    }
}

/// What the live turn's grounding looks like in the trace: how many web
/// searches, how many pages read, how many Wikipedia steps. The badge under
/// the answer reads this. A search with no page read is shown as such —
/// a recipe found and not opened is a recipe the model wrote itself.
public struct AppAnswerGrounding: Equatable, Sendable {
    public var webSearches: Int
    public var pagesRead: Int
    public var wikipediaSteps: Int

    public init(webSearches: Int = 0, pagesRead: Int = 0, wikipediaSteps: Int = 0) {
        self.webSearches = webSearches
        self.pagesRead = pagesRead
        self.wikipediaSteps = wikipediaSteps
    }

    public var webSteps: Int { webSearches + pagesRead }

    public var isEmpty: Bool { webSteps == 0 && wikipediaSteps == 0 }

    public static func of(_ trace: [AppToolTraceEntry]) -> AppAnswerGrounding {
        var grounding = AppAnswerGrounding()
        for entry in trace where entry.status == .done {
            if entry.name.hasPrefix("wikipedia") {
                grounding.wikipediaSteps += 1
            } else if entry.name == "fetch_page" {
                grounding.pagesRead += 1
            } else {
                grounding.webSearches += 1
            }
        }
        return grounding
    }

    /// Whether an answer names its sources: a "参照:" line, or a URL. The
    /// prompt asks for the line; an answer without one after a search wrote
    /// from memory, and the badge says so.
    public static func citesSources(_ answer: String) -> Bool {
        if answer.contains("参照:") || answer.contains("参照：") { return true }
        return answer.contains("https://") || answer.contains("http://")
    }
}
