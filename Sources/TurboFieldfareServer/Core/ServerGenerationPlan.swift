import Foundation
import TurboFieldfare

// P2-G4c-2: 未実装の入口。赤テストを置くためにツリーがビルドできる形だけを
// 先に置く (TODO §2-4)。中身は緑コミットで書く。

/// SPEC §6: one request's *decision* about how generation is constrained,
/// with nothing of the inference loop in it.
struct ServerGenerationPlan: Equatable, Sendable {
    /// GBNF text rooted at `root`, or `nil` when this request asks for no
    /// constraint at all.
    let grammar: String?
    /// GEN-5.
    let isLazy: Bool
    /// GEN-5. Non-nil exactly when `isLazy`.
    let trigger: ChatGrammarTrigger?
    /// GEN-2 / DEV-16: what the client asked for that could only be
    /// approximated, declaration side first, grammar side after.
    let approximations: [String]
    /// The request shape an error message may name. Never prompt text.
    let shape: String

    var isConstrained: Bool { grammar != nil }
    /// DEV-14.
    var allowsSpeculativeDecoding: Bool { !isConstrained }
    /// GEN-7.
    var requiresLogitsHead: Bool { isConstrained }

    init(request: ValidatedChatRequest, markers: ChatGrammarMarkers) {
        self.grammar = nil
        self.isLazy = false
        self.trigger = nil
        self.approximations = []
        self.shape = ""
    }
}

/// GEN-6: whether the grammar is suppressed right now, because the model is
/// inside its thought block.
final class ServerThoughtSuppression: @unchecked Sendable {
    let channelStartID: Int32
    let channelEndID: Int32
    private(set) var isSuppressed = false

    init(channelStartID: Int32, channelEndID: Int32) {
        self.channelStartID = channelStartID
        self.channelEndID = channelEndID
    }

    convenience init(tokenizer: GFTokenizer) {
        self.init(channelStartID: tokenizer.channelStartID,
                  channelEndID: tokenizer.channelEndID)
    }

    @discardableResult
    func observe(tokenID: Int32, events: [StructuredAssistantEvent]) -> Bool {
        isSuppressed
    }
}

/// GEN-2: a grammar that will not build is a bug on our side, never a client
/// error, so it has to reach the wire as a 500 that says which request shape
/// produced it.
struct ServerGrammarBuildFailure: Error, CustomDebugStringConvertible, Sendable {
    let shape: String
    let underlying: String

    init(shape: String, underlying: Error) {
        self.shape = shape
        self.underlying = String(reflecting: underlying)
    }

    var debugDescription: String {
        "grammar_build_failure shape=\(shape) error=\(underlying)"
    }
}

/// GEN-2 / DEV-16: the approximations as one field of the request-lifecycle
/// stderr line.
enum ServerApproximationLog {
    static func field(_ items: [String]) -> String? {
        nil
    }
}
