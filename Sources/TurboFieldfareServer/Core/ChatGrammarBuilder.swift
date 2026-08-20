import Foundation
import TurboFieldfare

/// The marker text and token ids the tool-call grammar is written around.
///
/// Injected rather than read off `GFTokenizer` so the grammar stage stays a
/// pure function: the C0 tests build one of these by hand and never load a
/// tokenizer.
public struct ChatGrammarMarkers: Equatable, Sendable {
    /// `<|tool_call>` — the section start the template writes and, in a lazy
    /// grammar, the trigger (GEN-5).
    public let toolCallStart: String
    /// `<tool_call|>` — the section end.
    public let toolCallEnd: String
    /// The id of the token whose text is `toolCallStart`. Only the lazy
    /// grammar needs it; a non-lazy grammar spells the marker as text.
    public let toolCallStartTokenID: Int32

    public init(toolCallStart: String, toolCallEnd: String, toolCallStartTokenID: Int32) {
        self.toolCallStart = toolCallStart
        self.toolCallEnd = toolCallEnd
        self.toolCallStartTokenID = toolCallStartTokenID
    }

    /// The markers as this model writes them.
    public init(tokenizer: GFTokenizer) {
        self.init(toolCallStart: ChatGrammarMarkers.gemmaToolCallStart,
                  toolCallEnd: ChatGrammarMarkers.gemmaToolCallEnd,
                  toolCallStartTokenID: tokenizer.toolCallStartID)
    }

    public static let gemmaToolCallStart = "<|tool_call>"
    public static let gemmaToolCallEnd = "<tool_call|>"
}

/// What starts a lazy grammar (GEN-5). The id is what the sampler watches for;
/// the text is what the matcher must be fed once it fires, because the marker
/// is the first thing the grammar spells.
public struct ChatGrammarTrigger: Equatable, Sendable {
    public let tokenID: Int32
    public let text: String

    public init(tokenID: Int32, text: String) {
        self.tokenID = tokenID
        self.text = text
    }
}

/// One request's generation constraint: the GBNF text plus how it is applied.
public struct ChatGrammarConstraint: Equatable, Sendable {
    /// GBNF, rooted at `root`.
    public let grammar: String
    /// GEN-5: `true` means do not apply the grammar until `trigger` fires.
    public let isLazy: Bool
    /// Non-nil exactly when `isLazy`.
    public let trigger: ChatGrammarTrigger?
    /// GEN-2 / DEV-16: the places the schema had to be approximated. Never an
    /// error; the server logs these.
    public let approximations: [String]

    public init(grammar: String,
                isLazy: Bool,
                trigger: ChatGrammarTrigger?,
                approximations: [String]) {
        self.grammar = grammar
        self.isLazy = isLazy
        self.trigger = trigger
        self.approximations = approximations
    }
}

/// SPEC §6: a request's `tools` / `tool_choice` / `parallel_tool_calls` /
/// `response_format` turned into the grammar that constrains generation.
///
/// **Not implemented yet** — every request comes back unconstrained, which is
/// what `ChatGrammarBuilderTests` is red about.
public enum ChatGrammarBuilder {
    /// REQ-response-format's three shapes, already unwrapped: `json_schema`
    /// carries `response_format.json_schema.schema` (GEN-3). A `nil` schema is
    /// "any JSON value" (DEV-18) rather than an error (GEN-2).
    public enum ResponseFormat: Equatable, Sendable {
        case text
        case jsonObject(schema: JSONValue?)
        case jsonSchema(schema: JSONValue?)
    }

    /// The whole stage. Returns `nil` when the request asks for no constraint.
    public static func constraint(
        tools: [GFTokenizer.FunctionDefinition],
        toolChoice: ChatToolChoice,
        parallelToolCalls: Bool,
        responseFormat: ResponseFormat,
        markers: ChatGrammarMarkers
    ) -> ChatGrammarConstraint? {
        nil
    }
}
