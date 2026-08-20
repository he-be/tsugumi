import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// C0 (CONFORMANCE §1), the contract half of SPEC §4: what a request asks the
/// generation to be constrained to.
///
/// `tool_choice: required`, a named `tool_choice` and `response_format`
/// `json_object` / `json_schema` were 501 placeholders while the grammar did
/// not exist. The grammar exists, so the values are accepted and carried to
/// the inference layer instead — and the only refusals left are the two no
/// grammar can express (GEN-4) and the collision between two contracts
/// (GEN-12). None of this touches HTTP or the tokenizer.
@Suite("C0 generation constraints")
struct ChatRequestConstraintTests {
    /// One declared tool, so a named `tool_choice` has something to name.
    private static let declaredTools = #""tools":[{"type":"function","function":{"#
        + #""name":"lookup","description":"","parameters":{"type":"object","#
        + #""properties":{"q":{"type":"string"}}}}}]"#

    private static func body(_ parts: [String]) -> Data {
        let tail = parts.filter { !$0.isEmpty }.map { "," + $0 }.joined()
        return Data((#"{"model":"m","messages":[{"role":"user","content":"hi"}]"#
            + tail + "}").utf8)
    }

    private static func parse(_ parts: String...) throws -> ValidatedChatRequest {
        try ChatRequestParser.parse(body(parts))
    }

    private static func refusal(
        _ parts: String...,
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws -> ServerRequestError {
        let data = body(parts)
        let error = #expect(throws: ServerRequestError.self, sourceLocation: sourceLocation) {
            _ = try ChatRequestParser.parse(data)
        }
        return try #require(error, sourceLocation: sourceLocation)
    }

    // MARK: - GEN-4 / DEV-17: the four shapes of tool_choice

    @Test("GEN-4: auto is the default and none still hides the tools")
    func GEN_4_tool_choice_auto_and_none() throws {
        #expect(try Self.parse(Self.declaredTools).toolChoice == .auto)
        let none = try Self.parse(Self.declaredTools, #""tool_choice":"none""#)
        #expect(none.toolChoice == .none)
        // `none` means the model may not call a tool, which this template
        // expresses by not being told the tools exist.
        #expect(none.tools.isEmpty)
    }

    @Test("GEN-4: required reaches the request instead of a 501")
    func GEN_4_tool_choice_required_is_carried() throws {
        let request = try Self.parse(Self.declaredTools, #""tool_choice":"required""#)
        #expect(request.toolChoice == .required)
        #expect(request.tools.map(\.name) == ["lookup"])
    }

    @Test("DEV-17: a named tool_choice carries the function name")
    func DEV_17_named_tool_choice_is_carried() throws {
        let named = #""tool_choice":{"type":"function","function":{"name":"lookup"}}"#
        let request = try Self.parse(Self.declaredTools, named)
        #expect(request.toolChoice == .function(name: "lookup"))
        #expect(request.tools.map(\.name) == ["lookup"])
    }

    /// GEN-4: the grammar cannot pin a function that was never declared, and
    /// dropping the constraint would answer a contract parameter with a
    /// free-form completion (R4). So it is a 400, not an empty grammar.
    @Test("GEN-4: a named tool_choice must name a declared tool")
    func GEN_4_named_tool_choice_must_be_declared() throws {
        let named = #""tool_choice":{"type":"function","function":{"name":"absent"}}"#
        let error = try Self.refusal(Self.declaredTools, named)
        #expect(error.type == .invalidRequest)
        #expect(error.param == "tool_choice")
        #expect(error.message.contains("absent"))

        // The same request with no tools at all is the same refusal.
        #expect(try Self.refusal(named).type == .invalidRequest)
    }

    @Test("GEN-4: required with no tools declared is a 400")
    func GEN_4_required_without_tools_is_refused() throws {
        let error = try Self.refusal(#""tool_choice":"required""#)
        #expect(error.type == .invalidRequest)
        #expect(error.param == "tool_choice")
    }

    // MARK: - GEN-3 / DEV-18: response_format

    @Test("GEN-3: text is the default and stays the default")
    func GEN_3_response_format_defaults_to_text() throws {
        #expect(try Self.parse().responseFormat == .text)
        #expect(try Self.parse(#""response_format":{"type":"text"}"#)
            .responseFormat == .text)
    }

    @Test("GEN-3: json_schema carries response_format.json_schema.schema")
    func GEN_3_json_schema_carries_the_inner_schema() throws {
        let format = #""response_format":{"type":"json_schema","json_schema":{"#
            + #""name":"s","schema":{"type":"object","properties":{"a":{"type":"string"}}}}}"#
        let schema = JSONValue.object([
            "type": .string("object"),
            "properties": .object(["a": .object(["type": .string("string")])]),
        ])
        #expect(try Self.parse(format).responseFormat == .jsonSchema(schema: schema))
    }

    /// GEN-3 reads `json_schema.schema`; a wrapper without one is not a 400
    /// (GEN-2), it is "no schema" and the grammar stage decides what that
    /// constrains to.
    @Test("GEN-3: a json_schema wrapper without a schema carries none")
    func GEN_3_json_schema_without_schema_carries_none() throws {
        #expect(try Self.parse(#""response_format":{"type":"json_schema","json_schema":{"name":"s"}}"#)
            .responseFormat == .jsonSchema(schema: nil))
        #expect(try Self.parse(#""response_format":{"type":"json_schema"}"#)
            .responseFormat == .jsonSchema(schema: nil))
    }

    /// DEV-18: `json_object` with no schema is a real constraint, decided by
    /// the grammar stage — the request layer only records that none was sent.
    /// The reference implementation reads `response_format.schema` here
    /// (`server-common.cpp:1150`), not the `json_schema` wrapper.
    @Test("DEV-18: json_object carries response_format.schema, or none")
    func DEV_18_json_object_carries_its_own_schema() throws {
        #expect(try Self.parse(#""response_format":{"type":"json_object"}"#)
            .responseFormat == .jsonObject(schema: nil))
        let withSchema = #""response_format":{"type":"json_object","schema":{"type":"object"}}"#
        #expect(try Self.parse(withSchema).responseFormat
            == .jsonObject(schema: .object(["type": .string("object")])))
    }

    // MARK: - GEN-12: the two contracts colliding

    @Test("GEN-12: a response format with required or a named tool_choice is a 400")
    func GEN_12_response_format_collides_with_a_forced_tool_call() throws {
        let named = #""tool_choice":{"type":"function","function":{"name":"lookup"}}"#
        for choice in [#""tool_choice":"required""#, named] {
            for format in [#""response_format":{"type":"json_object"}"#,
                           #""response_format":{"type":"json_schema","json_schema":{"schema":{}}}"#] {
                let error = try Self.refusal(Self.declaredTools, choice, format)
                #expect(error.type == .invalidRequest, "\(choice) + \(format)")
            }
        }
    }

    @Test("GEN-12: auto and none let the response format win without a refusal")
    func GEN_12_auto_and_none_do_not_collide() throws {
        for choice in ["", #""tool_choice":"auto""#, #""tool_choice":"none""#] {
            #expect(throws: Never.self, "\(choice)") {
                _ = try Self.parse(Self.declaredTools, choice,
                                   #""response_format":{"type":"json_object"}"#)
            }
        }
        // `text` is not a constraint, so it never collides.
        #expect(throws: Never.self) {
            _ = try Self.parse(Self.declaredTools, #""tool_choice":"required""#,
                               #""response_format":{"type":"text"}"#)
        }
    }
}
