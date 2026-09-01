import Foundation
import Tsugumi

/// The marker text and token ids the tool-call grammar is written around.
///
/// Injected rather than read off `GFTokenizer` so the grammar stage stays a
/// pure function: the C0 tests build one of these by hand and never load a
/// tokenizer.
public struct ChatGrammarMarkers: Equatable, Sendable {
    /// `<|tool_call>` — the section start the template writes and, in a lazy
    /// grammar, the trigger (GEN-5). The spelling is what the *trigger* is
    /// reported as; the grammar itself is written with the ids below.
    public let toolCallStart: String
    /// `<tool_call|>` — the section end.
    public let toolCallEnd: String
    /// The id of the token whose text is `toolCallStart`.
    ///
    /// **The grammar spells both markers as this id and `toolCallEndTokenID`,
    /// never as their text.** Their spelling is also reachable as a run of
    /// ordinary tokens (`<`, `|`, `tool`, `_`, `call`, `>`), and a literal
    /// accepts that run just as happily as the single marker token — but the
    /// decoder recognises a tool call by token id (`StructuredAssistantDecoder`),
    /// and the template re-renders an assistant turn with the marker tokens
    /// (INV-1). A text-spelled marker is therefore both unparseable and
    /// non-canonical: with `tool_choice: required` the model wrote
    /// `<`,`|`,`tool`,… for the opener and the real `<tool_call|>` token for
    /// the closer, and the decoder saw a section end with no section start.
    public let toolCallStartTokenID: Int32
    /// The id of the token whose text is `toolCallEnd`, for the same reason.
    public let toolCallEndTokenID: Int32
    /// `<|channel>` / `<channel|>` — the thought block a non-lazy grammar has
    /// to swallow before the constrained body (GEN-13). Written as token ids
    /// rather than text because the body between them is "any token that is
    /// not the closer", which only `TOKEN_NOT` can say. Both `nil` opts out of
    /// the prefix entirely.
    public let channelStartTokenID: Int32?
    public let channelEndTokenID: Int32?

    public init(toolCallStart: String,
                toolCallEnd: String,
                toolCallStartTokenID: Int32,
                toolCallEndTokenID: Int32,
                channelStartTokenID: Int32? = nil,
                channelEndTokenID: Int32? = nil) {
        self.toolCallStart = toolCallStart
        self.toolCallEnd = toolCallEnd
        self.toolCallStartTokenID = toolCallStartTokenID
        self.toolCallEndTokenID = toolCallEndTokenID
        self.channelStartTokenID = channelStartTokenID
        self.channelEndTokenID = channelEndTokenID
    }

    /// The markers as this model writes them.
    public init(tokenizer: GFTokenizer) {
        self.init(toolCallStart: ChatGrammarMarkers.gemmaToolCallStart,
                  toolCallEnd: ChatGrammarMarkers.gemmaToolCallEnd,
                  toolCallStartTokenID: tokenizer.toolCallStartID,
                  toolCallEndTokenID: tokenizer.toolCallEndID,
                  channelStartTokenID: tokenizer.channelStartID,
                  channelEndTokenID: tokenizer.channelEndID)
    }

    public static let gemmaToolCallStart = "<|tool_call>"
    public static let gemmaToolCallEnd = "<tool_call|>"
}

/// What starts a lazy grammar (GEN-5). The id is what the sampler watches for
/// and what the matcher consumes once it fires — the marker is the first thing
/// the grammar spells, as a `TOKEN` element. The text is the same marker's
/// spelling, kept for the callers that report or log it.
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
/// This is the reference implementation's `build_grammar` step
/// (`~/LLM/llama.cpp` pin `34af94cd9`, `common/chat-auto-parser-generator.cpp`
/// lines 78-150 and `common/chat-peg-parser.cpp` `standard_json_tools`), in our
/// markers and our dialect: a per-tool alternative that pins the tool name as a
/// literal and expands that tool's `parameters` schema for the arguments, a
/// section rule wrapping the call in the markers, and `parallel_tool_calls`
/// deciding whether repeats are allowed.
///
/// Nothing here touches the HTTP layer, the request schema, or the inference
/// loop; it is a pure function from what the request asked for to GBNF text.
public enum ChatGrammarBuilder {
    /// REQ-response-format's three shapes, already unwrapped: `json_schema`
    /// carries `response_format.json_schema.schema` (GEN-3). A `nil` schema is
    /// "any JSON value" (DEV-18) rather than an error (GEN-2).
    public enum ResponseFormat: Equatable, Sendable {
        case text
        case jsonObject(schema: JSONValue?)
        case jsonSchema(schema: JSONValue?)

        /// The schema to constrain to, or `nil` for "do not constrain".
        var schema: JSONValue? {
            switch self {
            case .text: return nil
            case .jsonObject(let schema), .jsonSchema(let schema):
                // DEV-18: an absent schema still constrains — to the empty
                // schema, which is every JSON value.
                return schema ?? .object([:])
            }
        }
    }

    /// The whole stage. Returns `nil` when the request asks for no constraint.
    ///
    /// **A response format and tools in one request** (GEN-12): with
    /// `tool_choice` `auto` or `none` the response format wins and no tool
    /// grammar is emitted — a tool call was optional there, so nothing is
    /// taken away. That is also what the reference does at the pin
    /// (`chat-auto-parser-generator.cpp`: the response-format branch is taken
    /// before the tools branch, and `grammar_lazy = !has_response_format && …`).
    ///
    /// With `required` or a named function the two asks collide, and GEN-12
    /// makes that combination a **400 in the request layer** — so it never
    /// reaches here from the server. The branch below stays as a defensive
    /// fallback for direct callers, and says so in `approximations` rather
    /// than erroring (GEN-2).
    public static func constraint(
        tools: [GFTokenizer.FunctionDefinition],
        toolChoice: ChatToolChoice,
        parallelToolCalls: Bool,
        responseFormat: ResponseFormat,
        markers: ChatGrammarMarkers
    ) -> ChatGrammarConstraint? {
        if let schema = responseFormat.schema {
            return responseFormatConstraint(schema: schema,
                                            tools: tools,
                                            toolChoice: toolChoice,
                                            markers: markers)
        }
        return toolConstraint(tools: tools,
                              toolChoice: toolChoice,
                              parallelToolCalls: parallelToolCalls,
                              markers: markers)
    }

    // MARK: - GEN-13

    /// Writes `root`, giving a non-lazy grammar the optional leading thought
    /// block GEN-13 asks for.
    ///
    /// A non-lazy grammar is applied from the very first generated token. With
    /// thinking on, that token is the thought channel's opener and not the
    /// body, so a `root` that only spells the body leaves no token allowed at
    /// all — which GEN-7 turns into a 500. The reference reaches the same shape
    /// by prefixing `ctx.reasoning_parser` to every generated parser.
    ///
    /// **Optional, never required.** With thinking off the template writes a
    /// closed, empty `<|channel>thought\n<channel|>` into the generation prompt
    /// itself, so the first generated token is already the body. The same
    /// grammar has to serve both.
    ///
    /// A lazy grammar keeps `root` as the trigger rule: it is not applied until
    /// the trigger fires, so it never sees the thought block (GEN-6 owns that).
    private static func addRoot(_ builder: inout JSONSchemaGrammarBuilder,
                                body: String,
                                isLazy: Bool,
                                markers: ChatGrammarMarkers) {
        guard !isLazy,
              let start = markers.channelStartTokenID,
              let end = markers.channelEndTokenID else {
            _ = builder.addRule("root", body)
            return
        }
        // The opener, then any run of tokens that is not the closer, then the
        // closer. `!<[N]>` (TOKEN_NOT) is the only element that can say "any
        // token except this one", and the thought text is arbitrary.
        let thought = builder.addRule(
            "thought", "<[\(start)]> !<[\(end)]>* <[\(end)]>")
        _ = builder.addRule("root", "\(thought)? \(body)")
    }

    // MARK: - GEN-3

    private static func responseFormatConstraint(
        schema: JSONValue,
        tools: [GFTokenizer.FunctionDefinition],
        toolChoice: ChatToolChoice,
        markers: ChatGrammarMarkers
    ) -> ChatGrammarConstraint {
        var approximations: [String] = []
        switch toolChoice {
        // Unreachable from the server: GEN-12 turns this combination into a
        // 400 before the request ever gets here. Kept so that a direct caller
        // of this pure function still gets a sane answer and a note.
        case .required, .function:
            if !tools.isEmpty {
                approximations.append(
                    "response-format-overrides-tool-choice: the response format "
                    + "constrains generation, so no tool call can be produced")
            }
        case .auto, .none:
            break
        }
        let result = JSONSchemaGrammar.build(dialect: .json) { builder in
            // The schema is a named rule, not `root`: GEN-13's `root` wraps it.
            let body = builder.addSchema("response-format", schema)
            addRoot(&builder, body: body, isLazy: false, markers: markers)
        }
        return ChatGrammarConstraint(grammar: result.grammar,
                                     isLazy: false,
                                     trigger: nil,
                                     approximations: approximations + result.approximations)
    }

    // MARK: - GEN-1 / GEN-4 / GEN-8

    private static func toolConstraint(
        tools: [GFTokenizer.FunctionDefinition],
        toolChoice: ChatToolChoice,
        parallelToolCalls: Bool,
        markers: ChatGrammarMarkers
    ) -> ChatGrammarConstraint? {
        let selected: [GFTokenizer.FunctionDefinition]
        switch toolChoice {
        // GEN-4: `none` is no tool grammar at all.
        case .none:
            return nil
        case .auto, .required:
            selected = tools
        // DEV-17: a named choice pins that one function.
        case .function(let name):
            selected = tools.filter { $0.name == name }
        }
        // No alternative to spell. A named choice that names an undeclared
        // tool lands here too; refusing that request is the request layer's
        // job, not the grammar's.
        guard !selected.isEmpty else { return nil }

        let isLazy = toolChoice == .auto
        let result = JSONSchemaGrammar.build(dialect: .gemmaToolArguments) { builder in
            var alternatives: [String] = []
            for tool in selected {
                // GEN-8: the canonical template form, with no whitespace
                // anywhere. The braces come from the schema's own object rule,
                // so this rule must not add its own.
                //
                // The two markers are `TOKEN` elements, not literals: only the
                // marker token itself may open and close the section, because
                // that is what the decoder reads and what the template writes
                // back (see `ChatGrammarMarkers.toolCallStartTokenID`).
                let arguments = builder.addSchema("tool-\(tool.name)-args",
                                                  argumentsSchema(tool.parameters))
                alternatives.append(builder.addRule(
                    "tool-\(tool.name)",
                    tokenElement(markers.toolCallStartTokenID)
                        + " " + literal("call:" + tool.name)
                        + " " + arguments
                        + " " + tokenElement(markers.toolCallEndTokenID)))
            }
            let body = alternatives.count == 1
                ? alternatives[0]
                : "(" + alternatives.joined(separator: " | ") + ")"
            // The section rule — the one the trigger starts. The template
            // writes parallel calls back to back with no separator.
            let section = builder.addRule("tool-call",
                                          parallelToolCalls ? body + "+" : body)
            addRoot(&builder, body: section, isLazy: isLazy, markers: markers)
        }
        return ChatGrammarConstraint(
            grammar: result.grammar,
            isLazy: isLazy,
            // GEN-5: the tool-call start token.
            trigger: isLazy
                ? ChatGrammarTrigger(tokenID: markers.toolCallStartTokenID,
                                     text: markers.toolCallStart)
                : nil,
            approximations: result.approximations)
    }

    /// The template always writes the arguments as `{…}`, so a tool that
    /// declares nothing usable still gets the generic object rather than the
    /// generic JSON value.
    private static func argumentsSchema(_ parameters: JSONValue) -> JSONValue {
        if case .object(let members) = parameters, !members.isEmpty {
            return parameters
        }
        return .object(["type": .string("object")])
    }

    /// A `TOKEN` element — the token with this id and nothing else. Written
    /// the way `GBNFGrammar` spells one (`<[id]>`), which the matcher matches
    /// by id rather than by piece text.
    private static func tokenElement(_ tokenID: Int32) -> String {
        "<[\(tokenID)]>"
    }

    /// The reference's `GRAMMAR_LITERAL_ESCAPE_RE` = `[\r\n"\\]`.
    private static func literal(_ text: String) -> String {
        var out = "\""
        for character in text {
            switch character {
            case "\r": out += #"\r"#
            case "\n": out += #"\n"#
            case "\"": out += #"\""#
            case "\\": out += #"\\"#
            default: out.append(character)
            }
        }
        return out + "\""
    }
}
