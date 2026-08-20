import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// SPEC §6 GEN-1 / GEN-2 / GEN-3 / GEN-4 / GEN-8: what a request's `tools`,
/// `tool_choice`, `parallel_tool_calls` and `response_format` turn into as
/// grammar text, and whether that grammar is lazy.
///
/// C0: no weights, no Metal, no tokenizer — the markers are injected.
///
/// The assertions are about the rules **this stage writes** (`root`,
/// `tool-call`, the per-tool rules, the tool-name literals) and about what the
/// finished grammar accepts. The expansion of a tool's `parameters` schema
/// belongs to `JSONSchemaGrammar`; where a line of it is pinned here, that is
/// noted.
@Suite("Chat grammar builder")
struct ChatGrammarBuilderTests {
    // MARK: - Fixtures

    private static let markers = ChatGrammarMarkers(
        toolCallStart: "<|tool_call>",
        toolCallEnd: "<tool_call|>",
        toolCallStartTokenID: 7)

    /// `get_weather(city: string, days: integer)` — both required, so the
    /// canonical call is `{city:<|"|>Kyoto<|"|>,days:3}`.
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

    private static let clock = GFTokenizer.FunctionDefinition(
        name: "get_time",
        description: "current time",
        parameters: .object([
            "type": .string("object"),
            "properties": .object(["zone": .object(["type": .string("string")])]),
            "required": .array([.string("zone")]),
        ]))

    private static func constraint(
        tools: [GFTokenizer.FunctionDefinition] = [],
        toolChoice: ChatToolChoice = .auto,
        parallelToolCalls: Bool = true,
        responseFormat: ChatGrammarBuilder.ResponseFormat = .text
    ) -> ChatGrammarConstraint? {
        ChatGrammarBuilder.constraint(
            tools: tools,
            toolChoice: toolChoice,
            parallelToolCalls: parallelToolCalls,
            responseFormat: responseFormat,
            markers: markers)
    }

    /// Does the grammar spell exactly this string, end to end?
    private static func accepts(_ grammar: String, _ text: String) throws -> Bool {
        var matcher = try GrammarMatcher(try GBNFGrammar(grammar))
        do {
            try matcher.accept(text: text)
        } catch {
            return false
        }
        return matcher.isComplete
    }

    /// Every rule body, without the `space` rule's own definition.
    private static func ruleBodies(_ grammar: String) -> [String] {
        grammar
            .split(separator: "\n")
            .filter { !$0.hasPrefix("space ::=") }
            .compactMap { (line: Substring) -> String? in
                guard let range = line.range(of: " ::= ") else { return nil }
                return String(line[range.upperBound...])
            }
    }

    private static let canonicalCall =
        #"<|tool_call>call:get_weather{city:<|"|>Kyoto<|"|>,days:3}<tool_call|>"#

    // MARK: - GEN-4: the four tool_choice values

    @Test("GEN-4: none produces no grammar at all")
    func GEN_4_none_has_no_grammar() {
        #expect(Self.constraint(tools: [Self.weather], toolChoice: .none) == nil)
        // Not even when the client also declared no tools.
        #expect(Self.constraint(toolChoice: .none) == nil)
    }

    @Test("GEN-4/GEN-5: auto is lazy and triggered by the tool-call start token")
    func GEN_4_auto_is_lazy() throws {
        let constraint = try #require(
            Self.constraint(tools: [Self.weather], toolChoice: .auto))
        #expect(constraint.isLazy)
        #expect(constraint.trigger == ChatGrammarTrigger(tokenID: 7,
                                                         text: "<|tool_call>"))
        #expect(constraint.approximations.isEmpty)
        #expect(constraint.grammar.contains("root ::= tool-call\n"))
    }

    @Test("GEN-4: required constrains from the first token")
    func GEN_4_required_is_not_lazy() throws {
        let required = try #require(
            Self.constraint(tools: [Self.weather], toolChoice: .required))
        #expect(!required.isLazy)
        #expect(required.trigger == nil)
        // Non-lazy means the start marker is itself part of the grammar; it is
        // the same text either way, only the application differs.
        let auto = try #require(
            Self.constraint(tools: [Self.weather], toolChoice: .auto))
        #expect(required.grammar == auto.grammar)
        #expect(required.grammar.contains(#""<|tool_call>call:get_weather""#))
    }

    @Test("GEN-4/DEV-17: a named choice keeps only that function")
    func DEV_17_named_choice_pins_one_function() throws {
        let constraint = try #require(Self.constraint(
            tools: [Self.weather, Self.clock],
            toolChoice: .function(name: "get_time")))
        #expect(!constraint.isLazy)
        #expect(constraint.trigger == nil)
        #expect(constraint.grammar.contains("tool-call ::= tool-get-time"))
        #expect(constraint.grammar.contains(#""<|tool_call>call:get_time""#))
        #expect(!constraint.grammar.contains("get_weather"))
        #expect(try !Self.accepts(constraint.grammar, Self.canonicalCall))
    }

    @Test("GEN-4: a named choice that names an undeclared tool constrains nothing")
    func DEV_17_named_choice_without_that_tool() {
        // Refusing this request is the request layer's job; the grammar has no
        // alternative to spell.
        #expect(Self.constraint(tools: [Self.weather],
                                toolChoice: .function(name: "get_time")) == nil)
        #expect(Self.constraint(tools: [], toolChoice: .required) == nil)
        #expect(Self.constraint(tools: [], toolChoice: .auto) == nil)
    }

    // MARK: - GEN-1: one tool vs several, parallel_tool_calls

    @Test("GEN-1: one tool is one alternative")
    func GEN_1_single_tool() throws {
        let constraint = try #require(Self.constraint(
            tools: [Self.weather], toolChoice: .required, parallelToolCalls: false))
        #expect(constraint.grammar.contains("tool-call ::= tool-get-weather\n"))
        #expect(constraint.grammar.contains(
            "tool-get-weather ::= \"<|tool_call>call:get_weather\" "
            + "tool-get-weather-args \"<tool_call|>\"\n"))
    }

    @Test("GEN-1: several tools become alternatives of the section rule")
    func GEN_1_several_tools() throws {
        let constraint = try #require(Self.constraint(
            tools: [Self.weather, Self.clock],
            toolChoice: .required, parallelToolCalls: false))
        #expect(constraint.grammar.contains(
            "tool-call ::= (tool-get-weather | tool-get-time)\n"))
        #expect(constraint.grammar.contains(#""<|tool_call>call:get_weather""#))
        #expect(constraint.grammar.contains(#""<|tool_call>call:get_time""#))
        #expect(try Self.accepts(constraint.grammar, Self.canonicalCall))
        #expect(try Self.accepts(
            constraint.grammar,
            #"<|tool_call>call:get_time{zone:<|"|>Asia/Tokyo<|"|>}<tool_call|>"#))
    }

    @Test("GEN-1: parallel_tool_calls decides whether repeats are allowed")
    func GEN_1_parallel_tool_calls() throws {
        let two = Self.canonicalCall + Self.canonicalCall

        let parallel = try #require(Self.constraint(
            tools: [Self.weather], toolChoice: .required, parallelToolCalls: true))
        #expect(parallel.grammar.contains("tool-call ::= tool-get-weather+\n"))
        #expect(try Self.accepts(parallel.grammar, Self.canonicalCall))
        #expect(try Self.accepts(parallel.grammar, two))

        let single = try #require(Self.constraint(
            tools: [Self.weather], toolChoice: .required, parallelToolCalls: false))
        #expect(single.grammar.contains("tool-call ::= tool-get-weather\n"))
        #expect(try Self.accepts(single.grammar, Self.canonicalCall))
        #expect(try !Self.accepts(single.grammar, two))
    }

    @Test("GEN-1: a tool that declares no parameters still writes an object")
    func GEN_1_tool_without_parameters() throws {
        let ping = GFTokenizer.FunctionDefinition(
            name: "ping", description: "", parameters: .object([:]))
        let constraint = try #require(Self.constraint(
            tools: [ping], toolChoice: .required, parallelToolCalls: false))
        #expect(constraint.grammar.contains(
            "tool-ping ::= \"<|tool_call>call:ping\" tool-ping-args \"<tool_call|>\"\n"))
        // The template writes the arguments as `{…}`, never as a bare value, so
        // the fallback is the generic object and not the generic JSON value.
        // (`tool-ping-args` and `object` are `JSONSchemaGrammar`'s rules.)
        #expect(constraint.grammar.contains("tool-ping-args ::= object\n"))
        #expect(constraint.grammar.contains(#"object ::= "{""#))
    }

    // MARK: - GEN-8: only the canonical template form (round trip)

    @Test("GEN-8: the grammar spells the canonical tool call and nothing looser")
    func GEN_8_round_trip_canonical_form() throws {
        let constraint = try #require(Self.constraint(
            tools: [Self.weather], toolChoice: .required, parallelToolCalls: false))
        let grammar = constraint.grammar

        // The rules that spell `<|tool_call>call:get_weather{city:<|"|>Kyoto<|"|>,days:3}<tool_call|>`.
        // The `-args` rules below are `JSONSchemaGrammar`'s half, pinned here
        // because GEN-8 is a statement about the finished text: no whitespace,
        // bare keys, ascending order.
        #expect(grammar.contains(
            "tool-get-weather ::= \"<|tool_call>call:get_weather\" "
            + "tool-get-weather-args \"<tool_call|>\"\n"))
        #expect(grammar.contains(
            "tool-get-weather-args ::= \"{\" tool-get-weather-args-city-kv "
            + "\",\" tool-get-weather-args-days-kv \"}\"\n"))
        #expect(grammar.contains(
            "tool-get-weather-args-city-kv ::= \"city\" \":\" string\n"))
        #expect(grammar.contains(
            "tool-get-weather-args-days-kv ::= \"days\" \":\" integer\n"))

        // No `space` anywhere — neither a rule nor a reference to one.
        #expect(!grammar.contains("space"))
        #expect(!Self.ruleBodies(grammar).contains { $0.contains("space") })

        // No JSON-quoted string alternative: strings are `<|"|>…<|"|>` only.
        #expect(grammar.contains(#"string ::= "<|\"|>""#))
        #expect(!grammar.contains(#"::= "\"""#))

        #expect(try Self.accepts(grammar, Self.canonicalCall))

        // …and the non-canonical variants are not spelled.
        for variant in [
            // a space after the comma
            #"<|tool_call>call:get_weather{city:<|"|>Kyoto<|"|>, days:3}<tool_call|>"#,
            // a JSON-quoted string
            #"<|tool_call>call:get_weather{city:"Kyoto",days:3}<tool_call|>"#,
            // quoted keys
            #"<|tool_call>call:get_weather{<|"|>city<|"|>:<|"|>Kyoto<|"|>,days:3}<tool_call|>"#,
            // descending keys (DEV-15: the template's `dictsort` is ascending)
            #"<|tool_call>call:get_weather{days:3,city:<|"|>Kyoto<|"|>}<tool_call|>"#,
            // a space before the closing brace
            #"<|tool_call>call:get_weather{city:<|"|>Kyoto<|"|>,days:3 }<tool_call|>"#,
            // an unknown function
            #"<|tool_call>call:get_forecast{city:<|"|>Kyoto<|"|>,days:3}<tool_call|>"#,
        ] {
            #expect(try !Self.accepts(grammar, variant), "\(variant)")
        }
    }

    // MARK: - GEN-3: response_format

    @Test("GEN-3: text constrains nothing")
    func GEN_3_text_is_unconstrained() {
        #expect(Self.constraint(responseFormat: .text) == nil)
    }

    @Test("GEN-3/DEV-18: json_object without a schema still constrains to JSON")
    func GEN_3_json_object_without_schema() throws {
        let constraint = try #require(
            Self.constraint(responseFormat: .jsonObject(schema: nil)))
        #expect(!constraint.isLazy)
        #expect(constraint.trigger == nil)
        #expect(constraint.approximations.isEmpty)
        #expect(constraint.grammar.contains("root ::= object\n"))
        #expect(try Self.accepts(constraint.grammar, #"{"a": [1, true]}"#))
        // R4: being asked for JSON and answering with prose is the one failure
        // forbidden at every stage.
        #expect(try !Self.accepts(constraint.grammar, "Sure! Here you go."))
    }

    @Test("GEN-3: json_object with a schema uses it")
    func GEN_3_json_object_with_schema() throws {
        let constraint = try #require(Self.constraint(
            responseFormat: .jsonObject(schema: Self.citySchema)))
        #expect(!constraint.isLazy)
        #expect(constraint.trigger == nil)
        // The `.json` dialect: quoted keys, JSON-quoted strings.
        #expect(constraint.grammar.contains(#""\"city\"""#))
        #expect(try Self.accepts(constraint.grammar, #"{"city": "Kyoto"}"#))
        #expect(try !Self.accepts(constraint.grammar, #"{"town": "Kyoto"}"#))
    }

    @Test("GEN-3: json_schema is a non-lazy grammar from its schema")
    func GEN_3_json_schema() throws {
        let constraint = try #require(Self.constraint(
            responseFormat: .jsonSchema(schema: Self.citySchema)))
        #expect(!constraint.isLazy)
        #expect(constraint.trigger == nil)
        #expect(constraint.grammar.contains(#""\"city\"""#))
        #expect(!constraint.grammar.contains("tool_call"))
        #expect(try Self.accepts(constraint.grammar, #"{"city": "Kyoto"}"#))
        // A `json_schema` block that carries no schema is not an error (GEN-2);
        // it is the empty schema.
        let empty = try #require(
            Self.constraint(responseFormat: .jsonSchema(schema: nil)))
        #expect(empty.grammar.contains("root ::= object\n"))
    }

    private static let citySchema = JSONValue.object([
        "type": .string("object"),
        "properties": .object(["city": .object(["type": .string("string")])]),
        "required": .array([.string("city")]),
        "additionalProperties": .bool(false),
    ])

    // MARK: - GEN-3 × GEN-1: both in one request

    @Test("GEN-3 wins over GEN-1 when a request carries both")
    func GEN_3_response_format_wins_over_tools() throws {
        // The reference does the same at the pin: the response-format branch is
        // taken before the tools branch and the grammar is not lazy.
        let auto = try #require(Self.constraint(
            tools: [Self.weather],
            toolChoice: .auto,
            responseFormat: .jsonObject(schema: nil)))
        #expect(!auto.isLazy)
        #expect(auto.trigger == nil)
        #expect(!auto.grammar.contains("tool_call"))
        // Nothing is broken by it: with `auto` a tool call was optional.
        #expect(auto.approximations.isEmpty)

        // With `required` the two asks collide, so the collision is reported
        // for the server to log or refuse. It is never an error here (GEN-2).
        let required = try #require(Self.constraint(
            tools: [Self.weather],
            toolChoice: .required,
            responseFormat: .jsonSchema(schema: Self.citySchema)))
        #expect(!required.isLazy)
        #expect(required.approximations.contains {
            $0.hasPrefix("response-format-overrides-tool-choice")
        })

        // `none` + a response format is just a response format.
        let none = try #require(Self.constraint(
            tools: [Self.weather],
            toolChoice: .none,
            responseFormat: .jsonObject(schema: nil)))
        #expect(none.approximations.isEmpty)
    }

    // MARK: - GEN-2: schema content never produces an error

    @Test("GEN-2: an unrepresentable schema is approximated, not refused")
    func GEN_2_approximations_reach_the_caller() throws {
        let brittle = GFTokenizer.FunctionDefinition(
            name: "lookup",
            description: "",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "id": .object(["$ref": .string("#/definitions/missing")]),
                ]),
                "required": .array([.string("id")]),
            ]))
        let constraint = try #require(Self.constraint(
            tools: [brittle], toolChoice: .required, parallelToolCalls: false))
        #expect(!constraint.approximations.isEmpty)
        #expect(constraint.approximations.contains { $0.contains("#/definitions/missing") })
        // Still a grammar, and still the canonical call syntax around it: the
        // unresolvable reference fell back to the generic JSON value.
        #expect(constraint.grammar.contains(
            "tool-lookup ::= \"<|tool_call>call:lookup\" "
            + "tool-lookup-args \"<tool_call|>\"\n"))
        #expect(constraint.grammar.contains("tool-lookup-args-id ::= value\n"))

        // The same on the response-format side.
        let format = try #require(Self.constraint(
            responseFormat: .jsonSchema(schema: .object([
                "type": .string("object"),
                "properties": .object([
                    "id": .object([
                        "type": .string("string"),
                        "pattern": .string("(unclosed"),
                    ]),
                ]),
                "required": .array([.string("id")]),
            ]))))
        #expect(!format.approximations.isEmpty)
        #expect(try Self.accepts(format.grammar, #"{"id": "anything"}"#))
    }
}

