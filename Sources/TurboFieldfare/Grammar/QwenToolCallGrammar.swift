import Foundation

/// The markers the Ornith (Qwen 3.5-MoE) tool-call grammar is written around.
///
/// Injected rather than read off `QwenTokenizer` so the grammar stage stays a
/// pure function — the tests build one by hand and never load a tokenizer.
///
/// Only three ids matter, and one of them is the odd one out: Ornith's
/// reasoning block is **opened by the template**, not by the model, so there is
/// no `<think>` to spell. `</think>` is the only reasoning marker that appears
/// in generated text.
public struct QwenToolCallMarkers: Equatable, Sendable {
    /// `<tool_call>` — the section start, and a lazy grammar's trigger.
    public let toolCallStart: String
    /// `</tool_call>` — the section end.
    public let toolCallEnd: String
    /// The id of the token whose text is `toolCallStart`.
    ///
    /// **The grammar spells both markers as ids, never as their text.** Both
    /// are ordinary (`special: false`) added tokens whose spelling is also
    /// reachable as a run of byte-level tokens (`<`, `tool`, `_`, `call`, `>`),
    /// and a literal would accept that run just as happily. The decoder
    /// recognises a call by token id (`QwenStructuredAssistantDecoder`), and
    /// the template re-renders the turn with the marker tokens, so a
    /// text-spelled marker is both unparseable and non-canonical.
    public let toolCallStartTokenID: Int32
    /// The id of the token whose text is `toolCallEnd`, for the same reason.
    public let toolCallEndTokenID: Int32
    /// `</think>` — what closes the reasoning block a grammar applied from the
    /// first token has to let the model out of. `nil` opts out of the prefix.
    public let thinkEndTokenID: Int32?

    public init(toolCallStart: String = QwenToolCallMarkers.ornithToolCallStart,
                toolCallEnd: String = QwenToolCallMarkers.ornithToolCallEnd,
                toolCallStartTokenID: Int32,
                toolCallEndTokenID: Int32,
                thinkEndTokenID: Int32? = nil) {
        self.toolCallStart = toolCallStart
        self.toolCallEnd = toolCallEnd
        self.toolCallStartTokenID = toolCallStartTokenID
        self.toolCallEndTokenID = toolCallEndTokenID
        self.thinkEndTokenID = thinkEndTokenID
    }

    /// The markers as this checkpoint writes them.
    public init(tokenizer: QwenTokenizer) {
        self.init(toolCallStartTokenID: tokenizer.toolCallStartID,
                  toolCallEndTokenID: tokenizer.toolCallEndID,
                  thinkEndTokenID: tokenizer.thinkEndID)
    }

    public static let ornithToolCallStart = "<tool_call>"
    public static let ornithToolCallEnd = "</tool_call>"
}

/// The GBNF for the XML tool call that `chat_template.jinja` writes.
///
/// Lives beside the tokenizer rather than in the server because the format is
/// the *checkpoint's*, not the HTTP layer's: the kernel-check tool exercises
/// this grammar against the real vocabulary, and the server's
/// `QwenChatGrammarBuilder` is only the adapter that turns a request's
/// `tool_choice` / `response_format` into these calls.
///
/// ```
/// <tool_call>
/// <function=get_weather>
/// <parameter=city>
/// Kyoto
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// Two consequences of that format run through everything below.
///
/// **A string parameter is written raw.** The template's
/// `args_value | string if args_value is string else args_value | tojson`
/// means a string is spelled without quotes or escapes, and everything else is
/// spelled as JSON. So the value rule is chosen per parameter from its declared
/// type, not once for the whole grammar — and the raw form needs a rule for
/// "any text that does not contain the closer", which `textValueRules` builds.
///
/// **The parameter blocks are in a fixed order.** GBNF cannot spell a
/// permutation without expanding it, so the properties go in ascending key
/// order with the optional ones marked `?` — the same deviation
/// `JSONSchemaGrammar` already registers (SPEC §12), and the order swift-jinja
/// re-renders them in (`Value(any:)` sorts a Swift dictionary's keys before the
/// template's `|items` ever sees it).
public enum QwenToolCallGrammar {
    /// What ends a parameter block, and so the one sequence a raw value may
    /// not contain.
    public static let parameterCloser = "\n</parameter>"

    /// The tool-call grammar. `withPreamble` is for a grammar that is applied
    /// from the first generated token; a lazy one is not applied until the
    /// section start fires and so never sees the text in front of it.
    public static func grammar(
        tools: [GFTokenizer.FunctionDefinition],
        parallelToolCalls: Bool,
        withPreamble: Bool,
        markers: QwenToolCallMarkers
    ) -> JSONSchemaGrammarResult {
        // GEN-8. A non-string value is written by the redraw as
        // `JSONValue.encoded()` — compact, no whitespace anywhere — so the
        // grammar spells that and only that. The ordinary `.json` dialect
        // permits whitespace the redraw never writes, and a model that used it
        // produced a turn that could not be described back: measured on
        // 2026-08-22, four `edit` calls in a row (pi session `01a02a00-…`).
        // Key order inside a free-form value is still not spellable in GBNF and
        // stays registered as §12 DEV-15.
        JSONSchemaGrammar.build(dialect: .qwenToolArguments) { builder in
            var alternatives: [String] = []
            for tool in tools {
                let blocks = parameterBlocks(&builder, tool: tool)
                alternatives.append(builder.addRule(
                    "tool-\(tool.name)",
                    tokenElement(markers.toolCallStartTokenID)
                        + " " + literal("\n<function=" + tool.name + ">\n")
                        + (blocks.isEmpty ? "" : " " + blocks.joined(separator: " "))
                        + " " + literal("</function>\n")
                        + " " + tokenElement(markers.toolCallEndTokenID)))
            }
            let body = alternatives.count == 1
                ? alternatives[0]
                : "(" + alternatives.joined(separator: " | ") + ")"
            // The template separates parallel calls with a single newline
            // (`'\n<tool_call>\n<function=' + …` for every call after the
            // first), unlike Gemma's, which writes them back to back.
            let section = builder.addRule(
                "tool-call",
                parallelToolCalls ? body + " (\"\\n\" " + body + ")*" : body)
            if withPreamble {
                _ = builder.addRule("root", "\(toolPreamble(markers)) \(section)")
            } else {
                _ = builder.addRule("root", section)
            }
        }
    }

    /// A response-format grammar, behind the reasoning block the template left
    /// open (GEN-3 / GEN-13).
    public static func responseFormatGrammar(
        schema: JSONValue,
        markers: QwenToolCallMarkers
    ) -> JSONSchemaGrammarResult {
        JSONSchemaGrammar.build(dialect: .json) { builder in
            let body = builder.addSchema("response-format", schema)
            guard let thinkEnd = markers.thinkEndTokenID else {
                _ = builder.addRule("root", body)
                return
            }
            // There is no section marker to exclude here, so this is the
            // precise form: an optional reasoning block, then the whitespace
            // the template puts after it.
            let prefix = builder.addRule(
                "reasoning", "!<[\(thinkEnd)]>* <[\(thinkEnd)]> [ \\t\\n]{0,20}")
            _ = builder.addRule("root", "\(prefix)? \(body)")
        }
    }

    // MARK: - The prefix a non-lazy grammar needs

    /// Everything the model is allowed to write before the call.
    ///
    /// A grammar applied from the first generated token has to allow it: with
    /// thinking on that token is inside the reasoning block — the template
    /// opened it — so a `root` that only spells the call leaves no token
    /// allowed at all, which GEN-7 turns into a 500.
    ///
    /// The prefix is **"any token that is not the section start"** rather than
    /// a precise `!</think>* </think>` block. Two reasons, and either is
    /// enough:
    ///
    /// 1. After `</think>` the template writes `\n\n` and *then* the content,
    ///    and this checkpoint's own system prompt says in as many words that
    ///    natural language before the call is allowed ("You may provide
    ///    optional reasoning for your function call in natural language BEFORE
    ///    the function call, but NOT after"). A grammar that forbids it
    ///    contradicts the prompt the model is reading.
    /// 2. It costs nothing in what is actually constrained. The model cannot
    ///    *stop* in the prefix — the grammar is incomplete there, so
    ///    `mayEndHere` is false and the stop token is rejected — so the only
    ///    way out is to write the call.
    private static func toolPreamble(_ markers: QwenToolCallMarkers) -> String {
        "!<[\(markers.toolCallStartTokenID)]>*"
    }

    // MARK: - One tool's parameters

    /// `<parameter=KEY>\n VALUE \n</parameter>\n` for each declared property,
    /// in ascending key order, optional ones marked `?`.
    ///
    /// A tool that declares no usable `properties` gets no blocks at all: the
    /// template writes nothing between `<function=…>` and `</function>` when
    /// `arguments` is empty, so the empty call is the canonical form and any
    /// parameter would be one the tool never declared.
    private static func parameterBlocks(
        _ builder: inout JSONSchemaGrammarBuilder,
        tool: GFTokenizer.FunctionDefinition
    ) -> [String] {
        guard case .object(let schema) = tool.parameters,
              case .object(let properties)? = schema["properties"],
              !properties.isEmpty else {
            return []
        }
        var required: Set<String> = []
        if case .array(let names)? = schema["required"] {
            for name in names {
                if case .string(let key) = name { required.insert(key) }
            }
        }
        var blocks: [String] = []
        for key in properties.keys.sorted() {
            let property = properties[key]!
            let value = valueRule(&builder, tool: tool.name, key: key, schema: property)
            let block = builder.addRule(
                "tool-\(tool.name)-p-\(key)",
                literal("<parameter=" + key + ">\n")
                    + " " + value
                    + " " + literal(parameterCloser + "\n"))
            blocks.append(required.contains(key) ? block : block + "?")
        }
        return blocks
    }

    /// One parameter's value, in the spelling the template gives its type.
    private static func valueRule(
        _ builder: inout JSONSchemaGrammarBuilder,
        tool: String,
        key: String,
        schema: JSONValue
    ) -> String {
        guard QwenToolCallParser.isStringSchema(schema) else {
            // Not a string: `tojson` wrote it, so the JSON expansion is exact.
            return builder.addSchema("tool-\(tool)-a-\(key)", schema)
        }
        // A string enum is written raw too, so the alternatives are bare
        // literals rather than the quoted ones `JSONSchemaGrammar` would emit.
        if case .object(let members) = schema, case .array(let cases)? = members["enum"] {
            let literals = cases.compactMap { value -> String? in
                guard case .string(let text) = value else { return nil }
                return literal(text)
            }
            if literals.count == cases.count, !literals.isEmpty {
                return builder.addRule("tool-\(tool)-a-\(key)",
                                       "(" + literals.joined(separator: " | ") + ")")
            }
        }
        return textValueRules(&builder)
    }

    // MARK: - Raw text that cannot close its own block

    /// "Any text that does not contain `\n</parameter>`", as the 13 states of
    /// the automaton that recognises it.
    ///
    /// This is what a string parameter is: the template writes the value with
    /// no quoting and no escapes, so the *only* thing it may not contain is the
    /// sequence that ends the block. Spelling it as `[^<]*` — the obvious
    /// approximation — would forbid every `<` in a string argument, which is
    /// most of what a tool that edits markup or code is ever asked to do.
    ///
    /// State `i` means "the last `i` characters are `closer[0..<i]`". Since the
    /// closer's only newline is its first character, a mismatch falls back to
    /// state 1 on a newline and to state 0 otherwise. Every state is accepting
    /// (a value may end on a partial closer — `…\n</param` is fine), so every
    /// rule is optional; the transition that would complete the closer is
    /// simply absent, which is what makes the sequence unspellable.
    ///
    /// The recursion is a tail call in every branch, so the matcher's stack
    /// does not grow with the length of the value.
    static func textValueRules(_ builder: inout JSONSchemaGrammarBuilder) -> String {
        let closer = Array(parameterCloser)
        let names = (0..<closer.count).map { "text-value-\($0)" }
        for index in 0..<closer.count {
            var branches: [String] = []
            // Any character that neither advances the match nor restarts it.
            let excluded: Set<Character> = index == 0 ? ["\n"] : ["\n", closer[index]]
            branches.append("\(characterClass(excluding: excluded)) \(names[0])")
            // The closer's only newline is its first character, so a newline
            // always restarts the match at state 1 — which for index 0 *is*
            // the advancing branch.
            branches.append("\"\\n\" \(names[1])")
            if index > 0, index + 1 < closer.count {
                branches.append("\(literal(String(closer[index]))) \(names[index + 1])")
            }
            // `index + 1 == closer.count` is the closer itself: no branch, and
            // that absence is the whole point of the construction.
            _ = builder.addRule(names[index],
                                "(" + branches.joined(separator: " | ") + ")?")
        }
        return names[0]
    }

    /// `[^…]` over the characters that must not take the "restart" branch.
    private static func characterClass(excluding excluded: Set<Character>) -> String {
        var out = "[^"
        for character in excluded.sorted() {
            switch character {
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "]": out += "\\]"
            case "\\": out += "\\\\"
            case "^": out += "\\^"
            case "-": out += "\\-"
            default: out.append(character)
            }
        }
        return out + "]"
    }

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
