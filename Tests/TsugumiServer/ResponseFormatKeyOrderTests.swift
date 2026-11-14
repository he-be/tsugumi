import Foundation
import Testing
@testable import Tsugumi
@testable import TsugumiServerCore

/// SPEC §12 DEV-15 for `response_format`: the answer's keys are written in the
/// schema's declared order (required first, then optional, each as declared),
/// the order llama.cpp's `json-schema-to-grammar` uses. Ascending keys made a
/// vision reading pick `カテゴリ` before it had written `正体`, because カ
/// (U+30AB) sorts before 正 (U+6B63).
///
/// The grammars are walked byte by byte with `GrammarMatcher`, so no tokenizer
/// and no weights are involved. A Gemma tool call keeps ascending keys (GEN-8);
/// that is `ChatGrammarBuilderTests`'.
@Suite("C0 response_format key order")
struct ResponseFormatKeyOrderTests {
    private static let gemmaMarkers = ChatGrammarMarkers(
        toolCallStart: "<|tool_call>",
        toolCallEnd: "<tool_call|>",
        toolCallStartTokenID: 7,
        toolCallEndTokenID: 8)

    private static let qwenMarkers = QwenToolCallMarkers(
        toolCallStart: "<tool_call>", toolCallEnd: "</tool_call>",
        toolCallStartTokenID: 7, toolCallEndTokenID: 8, thinkEndTokenID: nil)

    /// The reading the exhibit asks for, in the order it declares.
    private static let readingSchema = #"{"type":"object","properties":{"#
        + #""正体":{"type":"string"},"カテゴリ":{"type":"string"},"#
        + #""見立て":{"type":"object","properties":{"#
        + #""選んだ一体":{"type":"string"},"選んだわけ":{"type":"string"}},"#
        + #""required":["選んだ一体","選んだわけ"]},"#
        + #""連想する語":{"type":"array","items":{"type":"string"}}},"#
        + #""required":["正体","カテゴリ","見立て"]}"#

    private static let declared = #"{"正体":"鏡","カテゴリ":"道具","#
        + #""見立て":{"選んだ一体":"月","選んだわけ":"映すから"},"連想する語":["光"]}"#
    private static let ascending = #"{"カテゴリ":"道具","正体":"鏡","#
        + #""見立て":{"選んだわけ":"映すから","選んだ一体":"月"},"連想する語":["光"]}"#

    private static func body(_ schema: String, type: String = "json_schema") -> Data {
        let format = type == "json_schema"
            ? #"{"type":"json_schema","json_schema":{"name":"r","schema":"# + schema + "}}"
            : #"{"type":"json_object","schema":"# + schema + "}"
        return Data((#"{"model":"m","messages":[{"role":"user","content":"hi"}],"#
            + #""response_format":"# + format + "}").utf8)
    }

    private static func gemmaGrammar(_ body: Data) throws -> String {
        let plan = ServerGenerationPlan(request: try ChatRequestParser.parse(body),
                                        markers: gemmaMarkers)
        return try #require(plan.grammar)
    }

    private static func qwenGrammar(_ body: Data) throws -> String {
        let plan = QwenGenerationPlan(request: try ChatRequestParser.parse(body),
                                      markers: qwenMarkers)
        return try #require(plan.grammar)
    }

    private static func matches(_ grammar: String, _ text: String) -> Bool {
        do {
            var matcher = try GrammarMatcher(try GBNFGrammar(grammar))
            try matcher.accept(text: text)
            return matcher.isComplete
        } catch {
            return false
        }
    }

    @Test("Gemma: json_schema keys follow the declaration, at every depth")
    func gemma_json_schema_is_declared_order() throws {
        let grammar = try Self.gemmaGrammar(Self.body(Self.readingSchema))
        #expect(Self.matches(grammar, Self.declared))
        #expect(!Self.matches(grammar, Self.ascending))
    }

    @Test("Gemma: json_object's schema follows the declaration too")
    func gemma_json_object_is_declared_order() throws {
        let grammar = try Self.gemmaGrammar(Self.body(Self.readingSchema, type: "json_object"))
        #expect(Self.matches(grammar, Self.declared))
        #expect(!Self.matches(grammar, Self.ascending))
    }

    @Test("Qwen: response_format keys follow the declaration")
    func qwen_json_schema_is_declared_order() throws {
        let grammar = try Self.qwenGrammar(Self.body(Self.readingSchema))
        #expect(Self.matches(grammar, Self.declared))
        #expect(!Self.matches(grammar, Self.ascending))
    }

    @Test("required keys come first, then optional ones, each as declared")
    func required_first_then_optional() throws {
        let schema = #"{"type":"object","properties":{"zeta":{"type":"string"},"#
            + #""beta":{"type":"string"},"alpha":{"type":"string"}},"required":["alpha"]}"#
        let grammar = try Self.gemmaGrammar(Self.body(schema))
        #expect(Self.matches(grammar, #"{"alpha":"a","zeta":"z","beta":"b"}"#))
        #expect(Self.matches(grammar, #"{"alpha":"a","beta":"b"}"#))
        #expect(!Self.matches(grammar, #"{"alpha":"a","beta":"b","zeta":"z"}"#))
        #expect(!Self.matches(grammar, #"{"zeta":"z","alpha":"a"}"#))
    }

    @Test("a request parsed from a JSONValue has no order to keep, so keys stay ascending")
    func value_parsed_request_keeps_ascending() throws {
        let json = try JSONDecoder().decode(JSONValue.self, from: Self.body(Self.readingSchema))
        let request = try ChatRequestParser.parse(json)
        #expect(request.responseFormatPropertyOrder.isEmpty)
        let plan = ServerGenerationPlan(request: request, markers: Self.gemmaMarkers)
        let grammar = try #require(plan.grammar)
        #expect(Self.matches(grammar, Self.ascending))
        #expect(!Self.matches(grammar, Self.declared))
    }

    @Test("the declared order does not reach a Gemma tool call, which stays ascending (GEN-8)")
    func tool_calls_stay_ascending() throws {
        let order = JSONSchemaPropertyOrder(source: try #require(OrderedJSON.parse(
            #"{"properties":{"zeta":{"type":"string"},"alpha":{"type":"string"}}}"#)))
        #expect(order.keys(for: .object(["zeta": .object(["type": .string("string")]),
                                         "alpha": .object(["type": .string("string")])]))
                == ["zeta", "alpha"])
        let body = Data((#"{"model":"m","messages":[{"role":"user","content":"hi"}],"#
            + #""tools":[{"type":"function","function":{"name":"f","description":"","#
            + #""parameters":{"type":"object","properties":{"zeta":{"type":"string"},"#
            + #""alpha":{"type":"string"}},"required":["zeta","alpha"]}}}],"#
            + #""response_format":{"type":"json_schema","json_schema":{"name":"r","#
            + #""schema":{"type":"object","properties":{"b":{"type":"string"}}}}}}"#).utf8)
        let request = try ChatRequestParser.parse(body)
        // The table is the response format's, not the tools'.
        #expect(request.responseFormatPropertyOrder.keys(
            for: .object(["zeta": .object(["type": .string("string")]),
                          "alpha": .object(["type": .string("string")])])) == nil)
    }
}
