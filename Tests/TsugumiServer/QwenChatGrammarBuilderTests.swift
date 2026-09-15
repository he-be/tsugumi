import Foundation
import Testing
@testable import Tsugumi
@testable import TsugumiServerCore

/// SPEC §6 for the Ornith (Qwen 3.5-MoE) family: what a request's `tools`,
/// `tool_choice`, `parallel_tool_calls` and `response_format` turn into as
/// grammar text, and what that grammar accepts.
///
/// C0: no weights, no Metal, no tokenizer — the markers are injected.
///
/// The through-line of the suite is that **the grammar and
/// `QwenToolCallParser` are the same statement twice**: every call the grammar
/// accepts here is parsed back and its arguments compared, and every call it
/// rejects is one the parser could not have read.
@Suite("Qwen chat grammar builder")
struct QwenChatGrammarBuilderTests {
    // MARK: - Fixtures

    /// The real ids, so that a reader can line the grammar text up with
    /// `tokenizer.json` (`<tool_call>` 248058, `</tool_call>` 248059,
    /// `</think>` 248069).
    private static let toolCallStartID: Int32 = 248_058
    private static let toolCallEndID: Int32 = 248_059
    private static let thinkEndID: Int32 = 248_069

    private static let markers = QwenToolCallMarkers(
        toolCallStart: "<tool_call>",
        toolCallEnd: "</tool_call>",
        toolCallStartTokenID: toolCallStartID,
        toolCallEndTokenID: toolCallEndID,
        thinkEndTokenID: thinkEndID)

    /// `get_weather(city: string, days: integer)`, both required. `city` is
    /// written raw and `days` as JSON — the one difference that runs through
    /// the whole format.
    private static let weather = GFTokenizer.FunctionDefinition(
        name: "get_weather",
        description: "current weather",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "city": .object(["type": .string("string")]),
                "days": .object(["type": .string("integer")]),
            ]),
            "required": .array([.string("city"), .string("days")]),
        ]))

    /// `get_time(zone: string required, format: string optional enum)`.
    private static let clock = GFTokenizer.FunctionDefinition(
        name: "get_time",
        description: "current time",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "zone": .object(["type": .string("string")]),
                "format": .object([
                    "type": .string("string"),
                    "enum": .array([.string("iso"), .string("unix")]),
                ]),
            ]),
            "required": .array([.string("zone")]),
        ]))

    /// A tool with no parameters at all.
    private static let ping = GFTokenizer.FunctionDefinition(
        name: "ping",
        description: "no arguments",
        parameters: .object(["type": .string("object")]))

    private static func constraint(
        tools: [GFTokenizer.FunctionDefinition] = [],
        toolChoice: ChatToolChoice = .auto,
        parallelToolCalls: Bool = true,
        responseFormat: ChatGrammarBuilder.ResponseFormat = .text,
        markers: QwenToolCallMarkers = markers
    ) -> ChatGrammarConstraint? {
        QwenChatGrammarBuilder.constraint(
            tools: tools,
            toolChoice: toolChoice,
            parallelToolCalls: parallelToolCalls,
            responseFormat: responseFormat,
            markers: markers)
    }

    private enum Emission {
        case token(Int32, String)
        case text(String)
    }

    private static func walk(_ grammar: String, _ script: [Emission]) throws -> Bool {
        var matcher = try GrammarMatcher(try GBNFGrammar(grammar))
        do {
            for step in script {
                switch step {
                case .token(let tokenID, let piece):
                    try matcher.accept(piece: Array(piece.utf8), tokenID: tokenID)
                case .text(let text):
                    try matcher.accept(text: text)
                }
            }
        } catch {
            return false
        }
        return matcher.isComplete
    }

    /// One call as the model emits it: the two markers are the marker
    /// **tokens**, and `body` is everything between them.
    private static func call(_ body: String) -> [Emission] {
        [.token(toolCallStartID, "<tool_call>"),
         .text(body),
         .token(toolCallEndID, "</tool_call>")]
    }

    /// The same call with the markers spelled out as ordinary text — what a
    /// model that writes `<`, `tool`, `_`, `call`, `>` produces.
    private static func callSpelledAsText(_ body: String) -> [Emission] {
        [.text("<tool_call>" + body + "</tool_call>")]
    }

    /// The template's rendering of one parameter block.
    private static func parameter(_ key: String, _ value: String) -> String {
        "<parameter=\(key)>\n\(value)\n</parameter>\n"
    }

    private static func body(_ function: String, _ parameters: String...) -> String {
        "\n<function=\(function)>\n" + parameters.joined() + "</function>\n"
    }

    /// The canonical `get_weather` call, in the exact bytes
    /// `chat_template.jinja` would re-render it as.
    private static let weatherBody = body("get_weather",
                                          parameter("city", "Kyoto"),
                                          parameter("days", "3"))

    /// Parse a body back and return its arguments, so a grammar assertion and
    /// a parser assertion are never made about different strings.
    private static func parsedArguments(_ body: String,
                                        tools: [GFTokenizer.FunctionDefinition]) throws -> JSONValue {
        try QwenToolCallParser(tools: tools).parse(body, id: "call_x").arguments
    }

    // MARK: - tool_choice

    @Test("none produces no grammar at all")
    func none_has_no_grammar() {
        #expect(Self.constraint(tools: [Self.weather], toolChoice: .none) == nil)
        #expect(Self.constraint(toolChoice: .none) == nil)
    }

    @Test("auto is lazy and triggered by the section-start token")
    func auto_is_lazy() throws {
        let constraint = try #require(
            Self.constraint(tools: [Self.weather], toolChoice: .auto))
        #expect(constraint.isLazy)
        #expect(constraint.trigger == ChatGrammarTrigger(tokenID: Self.toolCallStartID,
                                                         text: "<tool_call>"))
        #expect(constraint.approximations.isEmpty)
        // A lazy grammar is not applied until the trigger fires, so `root` is
        // the section itself and carries no preamble.
        #expect(constraint.grammar.contains("root ::= tool-call\n"))
    }

    @Test("required constrains from the first token, with no preamble")
    func required_is_not_lazy() throws {
        let required = try #require(
            Self.constraint(tools: [Self.weather], toolChoice: .required))
        #expect(!required.isLazy)
        #expect(required.trigger == nil)
        #expect(required.grammar.contains("root ::= tool-call\n"))
    }

    @Test("a named function pins that one tool")
    func named_function_pins_one_tool() throws {
        let constraint = try #require(
            Self.constraint(tools: [Self.weather, Self.clock],
                            toolChoice: .function(name: "get_time")))
        #expect(constraint.grammar.contains("tool-get-time ::="))
        #expect(!constraint.grammar.contains("tool-get-weather ::="))
        // A name no tool declared leaves nothing to spell.
        #expect(Self.constraint(tools: [Self.weather],
                                toolChoice: .function(name: "absent")) == nil)
    }

    // MARK: - The XML form

    @Test("the canonical call is accepted, and parses back to its arguments")
    func canonical_call_round_trips() throws {
        let constraint = try #require(Self.constraint(tools: [Self.weather]))
        #expect(try Self.walk(constraint.grammar, Self.call(Self.weatherBody)))
        // The same bytes, read back: `city` raw, `days` as JSON.
        #expect(try Self.parsedArguments(Self.weatherBody, tools: [Self.weather])
                == .object(["city": .string("Kyoto"), "days": .integer(3)]))
    }

    @Test("the markers are token elements, so their spelling is not a call")
    func markers_spelled_as_text_are_rejected() throws {
        let constraint = try #require(
            Self.constraint(tools: [Self.weather], toolChoice: .required))
        #expect(try !Self.walk(constraint.grammar,
                               Self.callSpelledAsText(Self.weatherBody)))
    }

    /// GEN-8 for the values the redraw *spells* rather than copies. A non-string
    /// parameter is written by `JSONValue.encoded()`, which puts no whitespace
    /// anywhere, so the grammar must not leave the model the spaced form the
    /// checkpoint was trained on — writing it would produce a turn that cannot
    /// be described back (measured: pi session `01a02a00-…`, 2026-08-22).
    @Test("a non-string value may only be written in the redraw's compact spelling")
    func non_string_values_are_compact() throws {
        let tool = GFTokenizer.FunctionDefinition(
            name: "note",
            description: "note",
            parameters: .object([
                "type": .string("object"),
                "properties": .object(["items": .object(["type": .string("array")])]),
                "required": .array([.string("items")]),
            ]))
        let constraint = try #require(Self.constraint(tools: [tool]))
        let compact = Self.body("note", Self.parameter("items", #"["a","b"]"#))
        #expect(try Self.walk(constraint.grammar, Self.call(compact)))
        for spaced in [#"["a", "b"]"#, #"[ "a","b"]"#, #"["a","b" ]"#] {
            let body = Self.body("note", Self.parameter("items", spaced))
            #expect(try !Self.walk(constraint.grammar, Self.call(body)),
                    "the grammar left \(spaced), which the redraw does not write")
        }
    }

    @Test("without a declaration text, parameters go in ascending key order")
    func parameters_are_ordered_by_key() throws {
        let constraint = try #require(Self.constraint(tools: [Self.weather]))
        let swapped = Self.body("get_weather",
                                Self.parameter("days", "3"),
                                Self.parameter("city", "Kyoto"))
        #expect(try !Self.walk(constraint.grammar, Self.call(swapped)))
    }

    /// SPEC §12 DEV-15: required parameters first, then optional ones, each in the declaration's order. Ascending keys
    /// alone put the optional `format` before `zone`, and a model that wrote `zone` first could not add `format`.
    @Test("required parameters come first, then optional ones")
    func required_and_optional_parameters() throws {
        let constraint = try #require(Self.constraint(tools: [Self.clock]))
        let zoneOnly = Self.body("get_time", Self.parameter("zone", "Asia/Tokyo"))
        #expect(try Self.walk(constraint.grammar, Self.call(zoneOnly)))
        let both = Self.body("get_time",
                             Self.parameter("zone", "Asia/Tokyo"),
                             Self.parameter("format", "iso"))
        #expect(try Self.walk(constraint.grammar, Self.call(both)))
        let optionalFirst = Self.body("get_time",
                                      Self.parameter("format", "iso"),
                                      Self.parameter("zone", "Asia/Tokyo"))
        #expect(try !Self.walk(constraint.grammar, Self.call(optionalFirst)))
        let formatOnly = Self.body("get_time", Self.parameter("format", "iso"))
        #expect(try !Self.walk(constraint.grammar, Self.call(formatOnly)))
    }

    /// The declaration text's member order wins over ascending keys: `fetch_page(url, sections?)` is written `url`
    /// then `sections`, the call the app's page tool needs (`docs/qwen38/24` §3).
    @Test("the declaration text sets the order")
    func declaration_order() throws {
        let source = #"{"type":"object","properties":{"url":{"type":"string"},"sections":{"type":"string"},"note":{"type":"string"}},"required":["url","note"]}"#
        let parameters = try JSONDecoder().decode(JSONValue.self, from: Data(source.utf8))
        let tool = GFTokenizer.FunctionDefinition(name: "fetch_page", description: "page",
                                                  parameters: parameters, parametersSource: source)
        #expect(QwenToolDeclaration.parameterOrder(tool) == ["url", "note", "sections"])
        let constraint = try #require(Self.constraint(tools: [tool]))
        let declared = Self.body("fetch_page", Self.parameter("url", "https://a.jp/"),
                                 Self.parameter("note", "n"), Self.parameter("sections", "2,3"))
        #expect(try Self.walk(constraint.grammar, Self.call(declared)))
        let ascending = Self.body("fetch_page", Self.parameter("note", "n"),
                                  Self.parameter("sections", "2,3"), Self.parameter("url", "https://a.jp/"))
        #expect(try !Self.walk(constraint.grammar, Self.call(ascending)))
        let withoutSource = GFTokenizer.FunctionDefinition(name: "fetch_page", description: "page",
                                                           parameters: parameters)
        #expect(QwenToolDeclaration.parameterOrder(withoutSource) == ["note", "url", "sections"])
    }

    @Test("a string enum is spelled raw, not quoted")
    func string_enum_is_written_raw() throws {
        let constraint = try #require(Self.constraint(tools: [Self.clock]))
        let quoted = Self.body("get_time",
                               Self.parameter("zone", "Asia/Tokyo"),
                               Self.parameter("format", "\"iso\""))
        #expect(try !Self.walk(constraint.grammar, Self.call(quoted)))
        let outside = Self.body("get_time",
                                Self.parameter("zone", "Asia/Tokyo"),
                                Self.parameter("format", "rfc"))
        #expect(try !Self.walk(constraint.grammar, Self.call(outside)))
    }

    @Test("a tool with no parameters is called with an empty function block")
    func tool_without_parameters() throws {
        let constraint = try #require(Self.constraint(tools: [Self.ping]))
        #expect(try Self.walk(constraint.grammar, Self.call(Self.body("ping"))))
        #expect(try !Self.walk(constraint.grammar,
                               Self.call(Self.body("ping", Self.parameter("x", "1")))))
    }

    @Test("an undeclared tool name has no alternative to match")
    func undeclared_tool_is_rejected() throws {
        let constraint = try #require(Self.constraint(tools: [Self.weather]))
        let other = Self.body("rm_rf", Self.parameter("path", "/"))
        #expect(try !Self.walk(constraint.grammar, Self.call(other)))
    }

    @Test("a forced call is the first thing generated: no text before it, no stop without it")
    func forced_call_has_no_preamble() throws {
        for choice in [ChatToolChoice.required, .function(name: "get_weather")] {
            let constraint = try #require(Self.constraint(tools: [Self.weather], toolChoice: choice))
            #expect(try Self.walk(constraint.grammar, Self.call(Self.weatherBody)))
            // An answer written in place of the call: refused at its first token. With text allowed there and the stop
            // refused until a call, the model could not end (docs/qwen38/23: 14,700 tokens of filler in one round).
            #expect(try !Self.walk(constraint.grammar,
                                   [.token(702, "Let me look that up.")] + Self.call(Self.weatherBody)))
            #expect(try !Self.walk(constraint.grammar, [.text("- 要点 1\n- 要点 2\n")]))
            // Nothing may follow the call either.
            #expect(try !Self.walk(constraint.grammar,
                                   Self.call(Self.weatherBody) + [.token(703, " there.")]))
        }
    }

    // MARK: - The raw string value

    @Test("a raw string value may contain angle brackets and partial closers")
    func raw_value_allows_markup() throws {
        let constraint = try #require(Self.constraint(tools: [Self.clock]))
        // `[^<]*` — the obvious approximation — would reject every one of
        // these. The automaton only forbids the closer itself.
        for value in ["<b>bold</b>",
                      "a < b && c > d",
                      "</parameter",
                      "\n</paramete>\n",
                      "</function>",
                      "line\nline\n</param",
                      "日本語 <tag/>"] {
            let text = Self.body("get_time", Self.parameter("zone", value))
            #expect(try Self.walk(constraint.grammar, Self.call(text)),
                    "grammar rejected \(value.debugDescription)")
            #expect(try Self.parsedArguments(text, tools: [Self.clock])
                    == .object(["zone": .string(value)]),
                    "parser disagreed about \(value.debugDescription)")
        }
    }

    @Test("a raw string value cannot spell the closer")
    func raw_value_cannot_close_its_own_block() throws {
        let constraint = try #require(Self.constraint(tools: [Self.clock]))
        // The value the model would have to write to smuggle a closer in. The
        // grammar has no transition that completes `\n</parameter>` inside a
        // value, so the call as a whole is not spellable.
        let smuggled = Self.body("get_time",
                                 Self.parameter("zone", "x\n</parameter>\ny"))
        #expect(try !Self.walk(constraint.grammar, Self.call(smuggled)))
    }

    @Test("an empty string value is the block with nothing in it")
    func empty_value() throws {
        let constraint = try #require(Self.constraint(tools: [Self.clock]))
        let text = Self.body("get_time", Self.parameter("zone", ""))
        #expect(try Self.walk(constraint.grammar, Self.call(text)))
        #expect(try Self.parsedArguments(text, tools: [Self.clock])
                == .object(["zone": .string("")]))
    }

    // MARK: - parallel_tool_calls

    @Test("parallel calls are separated by one newline")
    func parallel_calls() throws {
        let constraint = try #require(
            Self.constraint(tools: [Self.weather, Self.clock], parallelToolCalls: true))
        let two = Self.call(Self.weatherBody)
            + [.text("\n")]
            + Self.call(Self.body("get_time", Self.parameter("zone", "UTC")))
        #expect(try Self.walk(constraint.grammar, two))
        // Back to back, the way Gemma's template writes them, is not this
        // template's form.
        let joined = Self.call(Self.weatherBody)
            + Self.call(Self.body("get_time", Self.parameter("zone", "UTC")))
        #expect(try !Self.walk(constraint.grammar, joined))
    }

    @Test("parallel_tool_calls false allows exactly one call")
    func single_call_only() throws {
        let constraint = try #require(
            Self.constraint(tools: [Self.weather], parallelToolCalls: false))
        #expect(try Self.walk(constraint.grammar, Self.call(Self.weatherBody)))
        let two = Self.call(Self.weatherBody) + [.text("\n")] + Self.call(Self.weatherBody)
        #expect(try !Self.walk(constraint.grammar, two))
    }

    // MARK: - GEN-3

    @Test("a response format constrains from the first token, behind the reasoning block")
    func response_format_allows_the_reasoning_block() throws {
        let schema = JSONValue.object([
            "type": .string("object"),
            "properties": .object(["ok": .object(["type": .string("boolean")])]),
            "required": .array([.string("ok")]),
        ])
        let constraint = try #require(
            Self.constraint(responseFormat: .jsonSchema(schema: schema)))
        #expect(!constraint.isLazy)
        #expect(constraint.trigger == nil)
        // Thinking off: the template already closed the block, so the body is
        // the first thing generated.
        #expect(try Self.walk(constraint.grammar, [.text("{\"ok\": true}")]))
        // Thinking on: the block is open, and `</think>` is the only marker
        // the model writes. The reasoning itself is `!<[…]>` — a *token*
        // element — so it is walked as ids, the way the sampler feeds it.
        #expect(try Self.walk(constraint.grammar, [
            .token(700, "Let me"),
            .token(701, " check."),
            .token(Self.thinkEndID, "</think>"),
            .text("\n\n{\"ok\": true}"),
        ]))
    }

    @Test("a response format and a forced tool choice collide, and say so")
    func response_format_overrides_tool_choice() throws {
        let constraint = try #require(
            Self.constraint(tools: [Self.weather],
                            toolChoice: .required,
                            responseFormat: .jsonObject(schema: nil)))
        #expect(constraint.approximations.contains {
            $0.hasPrefix("response-format-overrides-tool-choice")
        })
    }
}
