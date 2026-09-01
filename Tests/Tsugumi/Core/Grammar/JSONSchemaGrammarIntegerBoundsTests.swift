import Foundation
import Testing
@testable import Tsugumi

/// SPEC §6 GEN-1 / GEN-3 — 整数の `minimum` / `maximum` /
/// `exclusiveMinimum` / `exclusiveMaximum` → GBNF (`build_min_max_int`)。
/// 期待値は参照実装 `tests/test-json-schema-to-grammar.cpp` (ピン `34af94cd9`)。
@Suite("JSONSchemaGrammar integer bounds (GEN-1/GEN-3)")
struct JSONSchemaGrammarIntegerBoundsTests {
    @Test("整数の範囲が参照実装と同じ文法になる", arguments: JSONSchemaGrammarIntegerBoundsCases.all)
    func matchesReference(_ testCase: JSONSchemaGrammarCase) throws {
        let result = try JSONSchemaGrammarFixture.convert(testCase.schema)
        #expect(result.approximations.isEmpty)
        #expect(JSONSchemaGrammarFixture.trim(result.grammar) == testCase.grammar)
    }
}

enum JSONSchemaGrammarIntegerBoundsCases {
    static let all: [JSONSchemaGrammarCase] = [
        JSONSchemaGrammarCase(
            name: "min 0",
            schema: ##"""
            {
            "type": "integer",
            "minimum": 0
            }
            """##,
            grammar: ##"""
            root ::= ([0] | [1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min 1",
            schema: ##"""
            {
            "type": "integer",
            "minimum": 1
            }
            """##,
            grammar: ##"""
            root ::= ([1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min 3",
            schema: ##"""
            {
            "type": "integer",
            "minimum": 3
            }
            """##,
            grammar: ##"""
            root ::= ([1-2] [0-9]{1,15} | [3-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min 9",
            schema: ##"""
            {
            "type": "integer",
            "minimum": 9
            }
            """##,
            grammar: ##"""
            root ::= ([1-8] [0-9]{1,15} | [9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min 10",
            schema: ##"""
            {
            "type": "integer",
            "minimum": 10
            }
            """##,
            grammar: ##"""
            root ::= ([1] ([0-9]{1,15}) | [2-9] [0-9]{1,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min 25",
            schema: ##"""
            {
            "type": "integer",
            "minimum": 25
            }
            """##,
            grammar: ##"""
            root ::= ([1] [0-9]{2,15} | [2] ([0-4] [0-9]{1,14} | [5-9] [0-9]{0,14}) | [3-9] [0-9]{1,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "max 30",
            schema: ##"""
            {
            "type": "integer",
            "maximum": 30
            }
            """##,
            grammar: ##"""
            root ::= ("-" [1-9] [0-9]{0,15} | [0-9] | ([1-2] [0-9] | [3] "0"))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min -5",
            schema: ##"""
            {
            "type": "integer",
            "minimum": -5
            }
            """##,
            grammar: ##"""
            root ::= ("-" ([0-5]) | [0] | [1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min -123",
            schema: ##"""
            {
            "type": "integer",
            "minimum": -123
            }
            """##,
            grammar: ##"""
            root ::= ("-" ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-1] [0-9] | [2] [0-3])) | [0] | [1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "max -5",
            schema: ##"""
            {
            "type": "integer",
            "maximum": -5
            }
            """##,
            grammar: ##"""
            root ::= ("-" ([0-4] [0-9]{1,15} | [5-9] [0-9]{0,15}))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "max 1",
            schema: ##"""
            {
            "type": "integer",
            "maximum": 1
            }
            """##,
            grammar: ##"""
            root ::= ("-" [1-9] [0-9]{0,15} | [0-1])
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "max 100",
            schema: ##"""
            {
            "type": "integer",
            "maximum": 100
            }
            """##,
            grammar: ##"""
            root ::= ("-" [1-9] [0-9]{0,15} | [0-9] | ([1-8] [0-9] | [9] [0-9]) | "100")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min 0 max 23",
            schema: ##"""
            {
            "type": "integer",
            "minimum": 0,
            "maximum": 23
            }
            """##,
            grammar: ##"""
            root ::= ([0-9] | ([1] [0-9] | [2] [0-3]))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min 15 max 300",
            schema: ##"""
            {
            "type": "integer",
            "minimum": 15,
            "maximum": 300
            }
            """##,
            grammar: ##"""
            root ::= (([1] ([5-9]) | [2-9] [0-9]) | ([1-2] [0-9]{2} | [3] "00"))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min 5 max 30",
            schema: ##"""
            {
            "type": "integer",
            "minimum": 5,
            "maximum": 30
            }
            """##,
            grammar: ##"""
            root ::= ([5-9] | ([1-2] [0-9] | [3] "0"))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min -123 max 42",
            schema: ##"""
            {
            "type": "integer",
            "minimum": -123,
            "maximum": 42
            }
            """##,
            grammar: ##"""
            root ::= ("-" ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-1] [0-9] | [2] [0-3])) | [0-9] | ([1-3] [0-9] | [4] [0-2]))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min -10 max 10",
            schema: ##"""
            {
            "type": "integer",
            "minimum": -10,
            "maximum": 10
            }
            """##,
            grammar: ##"""
            root ::= ("-" ([0-9] | "10") | [0-9] | "10")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min + max items with min + max values across zero",
            schema: ##"""
            {
            "items": {
            "type": "integer",
            "minimum": -12,
            "maximum": 207
            },
            "minItems": 3,
            "maxItems": 5
            }
            """##,
            grammar: ##"""
            item ::= ("-" ([0-9] | "1" [0-2]) | [0-9] | ([1-8] [0-9] | [9] [0-9]) | ([1] [0-9]{2} | [2] "0" [0-7]))
            root ::= "[" space item ("," space item){2,4} space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min + max items with min + max values",
            schema: ##"""
            {
            "items": {
            "type": "integer",
            "minimum": 12,
            "maximum": 207
            },
            "minItems": 3,
            "maxItems": 5
            }
            """##,
            grammar: ##"""
            item ::= (([1] ([2-9]) | [2-9] [0-9]) | ([1] [0-9]{2} | [2] "0" [0-7]))
            root ::= "[" space item ("," space item){2,4} space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
    ]
}
