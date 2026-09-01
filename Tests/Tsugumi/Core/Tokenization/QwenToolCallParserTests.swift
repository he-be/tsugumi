import Foundation
import Testing
@testable import Tsugumi

/// `docs/qwen35moe/04-PHASES.md` Phase 5: reading back the XML tool call
/// `chat_template.jinja` writes.
///
/// C0: no weights, no Metal, no tokenizer — the parser is a pure function from
/// the text between the markers and the tool schemas to a `ParsedToolCall`.
///
/// The suite is organised around the one thing that makes this format
/// different from Gemma's: **a value's spelling depends on its declared type**,
/// because the template writes `args_value | string if args_value is string
/// else args_value | tojson`.
@Suite("Qwen tool call parser")
struct QwenToolCallParserTests {
    private static let weather = GFTokenizer.FunctionDefinition(
        name: "get_weather",
        description: "current weather",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "city": .object(["type": .string("string")]),
                "days": .object(["type": .string("integer")]),
                "metric": .object(["type": .string("boolean")]),
                "hours": .object(["type": .string("array")]),
                "extra": .object(["type": .string("object")]),
                "anything": .object([:]),
            ]),
            "required": .array([.string("city")]),
        ]))

    private static func parse(_ body: String,
                              tools: [GFTokenizer.FunctionDefinition] = [weather]) throws -> ParsedToolCall {
        try QwenToolCallParser(tools: tools).parse(body, id: "call_fixed")
    }

    private static func parameter(_ key: String, _ value: String) -> String {
        "<parameter=\(key)>\n\(value)\n</parameter>\n"
    }

    private static func body(_ function: String, _ parameters: String...) -> String {
        "\n<function=\(function)>\n" + parameters.joined() + "</function>\n"
    }

    // MARK: - The shape

    @Test("the canonical call")
    func canonical() throws {
        let call = try Self.parse(Self.body("get_weather",
                                            Self.parameter("city", "Kyoto"),
                                            Self.parameter("days", "3")))
        #expect(call.id == "call_fixed")
        #expect(call.name == "get_weather")
        #expect(call.arguments == .object(["city": .string("Kyoto"),
                                           "days": .integer(3)]))
        #expect(call.argumentsJSON == #"{"city":"Kyoto","days":3}"#)
    }

    @Test("a call with no parameters")
    func noParameters() throws {
        let call = try Self.parse(Self.body("get_weather"))
        #expect(call.arguments == .object([:]))
    }

    @Test("the section markers are tolerated if the caller left them on")
    func markersAreTolerated() throws {
        let inner = Self.body("get_weather", Self.parameter("city", "Kyoto"))
        let wrapped = "<tool_call>" + inner + "</tool_call>"
        #expect(try Self.parse(wrapped).arguments == Self.parse(inner).arguments)
        // Half a pair is not a call with a stray marker in it.
        #expect(throws: QwenToolCallParserError.malformed) {
            _ = try Self.parse("<tool_call>" + inner)
        }
    }

    @Test("an undeclared tool is refused by name")
    func unknownTool() {
        #expect(throws: QwenToolCallParserError.unknownTool("rm_rf")) {
            _ = try Self.parse(Self.body("rm_rf", Self.parameter("path", "/")))
        }
    }

    // MARK: - Typing a value

    @Test("a string parameter is read raw, quotes and all")
    func stringIsRaw() throws {
        // The template wrote no quotes, so quotes in the text are part of the
        // value — and a numeric-looking string stays a string.
        for (written, expected) in [("Kyoto", "Kyoto"),
                                    ("\"Kyoto\"", "\"Kyoto\""),
                                    ("3", "3"),
                                    ("true", "true"),
                                    ("null", "null"),
                                    ("a\\nb", "a\\nb"),
                                    ("  padded  ", "  padded  ")] {
            let call = try Self.parse(Self.body("get_weather",
                                                Self.parameter("city", written)))
            #expect(call.arguments == .object(["city": .string(expected)]),
                    "for \(written.debugDescription)")
        }
    }

    @Test("a non-string parameter is read as the JSON tojson wrote")
    func nonStringIsJSON() throws {
        let call = try Self.parse(Self.body(
            "get_weather",
            Self.parameter("city", "Kyoto"),
            Self.parameter("days", "3"),
            Self.parameter("extra", #"{"unit":"c","levels":[1,2]}"#),
            Self.parameter("hours", "[0, 12]"),
            Self.parameter("metric", "true")))
        #expect(call.arguments == .object([
            "city": .string("Kyoto"),
            "days": .integer(3),
            "extra": .object(["unit": .string("c"),
                              "levels": .array([.integer(1), .integer(2)])]),
            "hours": .array([.integer(0), .integer(12)]),
            "metric": .bool(true),
        ]))
    }

    @Test("a non-string parameter that is not JSON is refused, not guessed")
    func nonStringMustBeJSON() {
        #expect(throws: QwenToolCallParserError.unparsableArgument("days")) {
            _ = try Self.parse(Self.body("get_weather",
                                         Self.parameter("city", "Kyoto"),
                                         Self.parameter("days", "three")))
        }
    }

    @Test("a parameter with no declared type falls back to the string it looks like")
    func untypedFallsBackToString() throws {
        // Declared, but with an empty schema.
        let declared = try Self.parse(Self.body("get_weather",
                                                Self.parameter("anything", "three")))
        #expect(declared.arguments == .object(["anything": .string("three")]))
        // Never declared at all.
        let undeclared = try Self.parse(Self.body("get_weather",
                                                  Self.parameter("surprise", "three")))
        #expect(undeclared.arguments == .object(["surprise": .string("three")]))
        // …but JSON still wins where it parses.
        let json = try Self.parse(Self.body("get_weather",
                                            Self.parameter("anything", "3")))
        #expect(json.arguments == .object(["anything": .integer(3)]))
    }

    // MARK: - Where the value ends

    @Test("the value ends at the first closer, which the grammar makes the only one")
    func valueEndsAtFirstCloser() throws {
        // `\n</parameter>` inside a value is unspellable under the grammar
        // (`QwenChatGrammarBuilderTests`), so the first occurrence is the only
        // one. A parser that searched for the last would read a different
        // value than the grammar allowed.
        let smuggled = "\n<function=get_weather>\n<parameter=city>\n"
            + "x\n</parameter>\ny\n</parameter>\n</function>\n"
        #expect(throws: QwenToolCallParserError.malformed) {
            _ = try Self.parse(smuggled)
        }
    }

    @Test("a value may contain markup and partial closers")
    func valueMayContainMarkup() throws {
        for value in ["<b>bold</b>", "</parameter", "a < b > c", "line\nline",
                      "</function>", "日本語 <tag/>", ""] {
            let call = try Self.parse(Self.body("get_weather",
                                                Self.parameter("city", value)))
            #expect(call.arguments == .object(["city": .string(value)]),
                    "for \(value.debugDescription)")
        }
    }

    // MARK: - Malformed

    @Test("the pieces of the frame are all required")
    func malformedFrames() {
        let cases = [
            "get_weather(city=Kyoto)",
            "\n<function=get_weather>\n<parameter=city>\nKyoto\n</parameter>\n",
            "\n<function=get_weather\n<parameter=city>\nKyoto\n</parameter>\n</function>\n",
            "\n<function=get_weather>\n<parameter=city>Kyoto</parameter>\n</function>\n",
            "\n<function=>\n</function>\n",
            "\n<function=get_weather>\nKyoto\n</function>\n",
            "\n<function=get_weather>\n<parameter=city>\nKyoto\n</parameter>\n</function>\ntrailing",
        ]
        for text in cases {
            #expect(throws: (any Error).self, "accepted \(text.debugDescription)") {
                _ = try Self.parse(text)
            }
        }
    }

    @Test("a parameter written twice has no rendering, so it is not read")
    func duplicateParameter() {
        #expect(throws: QwenToolCallParserError.malformed) {
            _ = try Self.parse(Self.body("get_weather",
                                         Self.parameter("city", "Kyoto"),
                                         Self.parameter("city", "Osaka")))
        }
    }

    @Test("an oversized body is refused before it is scanned")
    func oversized() {
        let huge = String(repeating: "x", count: QwenToolCallParser.maximumBytes + 1)
        #expect(throws: QwenToolCallParserError.oversized) {
            _ = try Self.parse(Self.body("get_weather", Self.parameter("city", huge)))
        }
    }
}
