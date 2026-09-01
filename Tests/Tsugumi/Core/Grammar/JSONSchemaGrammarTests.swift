import Foundation
import Testing
@testable import Tsugumi

/// SPEC §6 GEN-1 / GEN-3 — JSON Schema → GBNF の変換表。
///
/// 期待値の出所は参照実装 `tests/test-json-schema-to-grammar.cpp`
/// (ピン `34af94cd9`)。**決定 (a)** — オブジェクトのプロパティはキーの昇順 —
/// の分だけ並びを直してあり、直した箇所にはその旨のコメントを付けた
/// (SPEC §12 に登録する逸脱)。それ以外は 1 文字も変えていない。
struct JSONSchemaGrammarCase: Sendable, CustomTestStringConvertible {
    let name: String
    let schema: String
    let grammar: String

    var testDescription: String { name }
}

enum JSONSchemaGrammarFixture {
    /// 参照実装のテストと同じ整形 (前後の空白を落とし、各行の字下げを落とす)。
    static func trim(_ text: String) -> String {
        text
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.drop { $0 == " " || $0 == "\t" } }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func convert(
        _ schema: String,
        dialect: JSONSchemaGrammarDialect = .json
    ) throws -> JSONSchemaGrammarResult {
        let value = try JSONDecoder().decode(JSONValue.self, from: Data(schema.utf8))
        return JSONSchemaGrammar.grammar(for: value, dialect: dialect)
    }
}

@Suite("JSONSchemaGrammar (GEN-1/GEN-3)")
struct JSONSchemaGrammarTests {
    @Test("参照実装の期待値と一致する", arguments: JSONSchemaGrammarCoreCases.all)
    func matchesReference(_ testCase: JSONSchemaGrammarCase) throws {
        let result = try JSONSchemaGrammarFixture.convert(testCase.schema)
        #expect(result.approximations.isEmpty)
        #expect(JSONSchemaGrammarFixture.trim(result.grammar) == testCase.grammar)
    }
}

enum JSONSchemaGrammarCoreCases {
    static let all: [JSONSchemaGrammarCase] = [
        JSONSchemaGrammarCase(
            name: "empty schema (object)",
            schema: ##"""
            {}
            """##,
            grammar: ##"""
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
            """##),
        JSONSchemaGrammarCase(
            name: "exotic formats",
            schema: ##"""
            {
            "items": [
            { "format": "date" },
            { "format": "uuid" },
            { "format": "time" },
            { "format": "date-time" }
            ]
            }
            """##,
            grammar: ##"""
            date ::= [0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )
            date-string ::= "\"" date "\""
            date-time ::= date "T" time
            date-time-string ::= "\"" date-time "\""
            root ::= "[" space tuple-0 "," space uuid "," space tuple-2 "," space tuple-3 space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            time ::= ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )
            time-string ::= "\"" time "\""
            tuple-0 ::= date-string
            tuple-2 ::= time-string
            tuple-3 ::= date-time-string
            uuid ::= "\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\""
            """##),
        JSONSchemaGrammarCase(
            name: "string",
            schema: ##"""
            {
            "type": "string"
            }
            """##,
            grammar: ##"""
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char* "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "string w/ min length 1",
            schema: ##"""
            {
            "type": "string",
            "minLength": 1
            }
            """##,
            grammar: ##"""
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char+ "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "string w/ min length 3",
            schema: ##"""
            {
            "type": "string",
            "minLength": 3
            }
            """##,
            grammar: ##"""
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{3,} "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "string w/ max length",
            schema: ##"""
            {
            "type": "string",
            "maxLength": 3
            }
            """##,
            grammar: ##"""
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{0,3} "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "string w/ min & max length",
            schema: ##"""
            {
            "type": "string",
            "minLength": 1,
            "maxLength": 4
            }
            """##,
            grammar: ##"""
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{1,4} "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "boolean",
            schema: ##"""
            {
            "type": "boolean"
            }
            """##,
            grammar: ##"""
            root ::= ("true" | "false")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "integer",
            schema: ##"""
            {
            "type": "integer"
            }
            """##,
            grammar: ##"""
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part)
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "string const",
            schema: ##"""
            {
            "const": "foo"
            }
            """##,
            grammar: ##"""
            root ::= "\"foo\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "non-string const",
            schema: ##"""
            {
            "const": 123
            }
            """##,
            grammar: ##"""
            root ::= "123"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "non-string enum",
            schema: ##"""
            {
            "enum": ["red", "amber", "green", null, 42, ["foo"]]
            }
            """##,
            grammar: ##"""
            root ::= ("\"red\"" | "\"amber\"" | "\"green\"" | "null" | "42" | "[\"foo\"]")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "string array",
            schema: ##"""
            {
            "type": "array",
            "prefixItems": { "type": "string" }
            }
            """##,
            grammar: ##"""
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "[" space (string ("," space string)*)? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            """##),
        JSONSchemaGrammarCase(
            name: "nullable string array",
            schema: ##"""
            {
            "type": ["array", "null"],
            "prefixItems": { "type": "string" }
            }
            """##,
            grammar: ##"""
            alternative-0 ::= "[" space (string ("," space string)*)? space "]"
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            null ::= "null"
            root ::= alternative-0 | null
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            """##),
        JSONSchemaGrammarCase(
            name: "tuple1",
            schema: ##"""
            {
            "prefixItems": [{ "type": "string" }]
            }
            """##,
            grammar: ##"""
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "[" space string space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            """##),
        JSONSchemaGrammarCase(
            name: "tuple2",
            schema: ##"""
            {
            "prefixItems": [{ "type": "string" }, { "type": "number" }]
            }
            """##,
            grammar: ##"""
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "[" space string "," space number space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            """##),
        JSONSchemaGrammarCase(
            name: "array with empty items",
            schema: ##"""
            {
            "type": "array",
            "items": {}
            }
            """##,
            grammar: ##"""
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            item ::= object
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= "[" space (item ("," space item)*)? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
            """##),
        JSONSchemaGrammarCase(
            name: "array with empty items and prefixItems",
            schema: ##"""
            {
            "type": "array",
            "items": {},
            "prefixItems": { "type": "string" }
            }
            """##,
            grammar: ##"""
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            item ::= object
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= "[" space (item ("," space item)*)? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
            """##),
        JSONSchemaGrammarCase(
            name: "number",
            schema: ##"""
            {
            "type": "number"
            }
            """##,
            grammar: ##"""
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "minItems",
            schema: ##"""
            {
            "items": {
            "type": "boolean"
            },
            "minItems": 2
            }
            """##,
            grammar: ##"""
            boolean ::= ("true" | "false")
            root ::= "[" space boolean ("," space boolean)+ space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "maxItems 0",
            schema: ##"""
            {
            "items": {
            "type": "boolean"
            },
            "maxItems": 0
            }
            """##,
            grammar: ##"""
            boolean ::= ("true" | "false")
            root ::= "[" space  space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "maxItems 1",
            schema: ##"""
            {
            "items": {
            "type": "boolean"
            },
            "maxItems": 1
            }
            """##,
            grammar: ##"""
            boolean ::= ("true" | "false")
            root ::= "[" space boolean? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "maxItems 2",
            schema: ##"""
            {
            "items": {
            "type": "boolean"
            },
            "maxItems": 2
            }
            """##,
            grammar: ##"""
            boolean ::= ("true" | "false")
            root ::= "[" space (boolean ("," space boolean)?)? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "min + maxItems",
            schema: ##"""
            {
            "items": {
            "type": ["number", "integer"]
            },
            "minItems": 3,
            "maxItems": 5
            }
            """##,
            grammar: ##"""
            decimal-part ::= [0-9]{1,16}
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            item ::= number | integer
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "[" space item ("," space item){2,4} space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        // 決定 (a) により宣言順ではなく昇順。参照実装の期待値と
        // 異なるのはプロパティの並びだけである (SPEC §12)。
        JSONSchemaGrammarCase(
            name: "required props in original order",
            schema: ##"""
            {
            "type": "object",
            "properties": {
            "b": {"type": "string"},
            "c": {"type": "string"},
            "a": {"type": "string"}
            },
            "required": [
            "a",
            "b",
            "c"
            ],
            "additionalProperties": false,
            "definitions": {}
            }
            """##,
            grammar: ##"""
            a-kv ::= "\"a\"" space ":" space string
            b-kv ::= "\"b\"" space ":" space string
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space a-kv "," space b-kv "," space c-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            """##),
        JSONSchemaGrammarCase(
            name: "1 optional prop",
            schema: ##"""
            {
            "properties": {
            "a": {
            "type": "string"
            }
            },
            "additionalProperties": false
            }
            """##,
            grammar: ##"""
            a-kv ::= "\"a\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space  (a-kv )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            """##),
        JSONSchemaGrammarCase(
            name: "N optional props",
            schema: ##"""
            {
            "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"},
            "c": {"type": "string"}
            },
            "additionalProperties": false
            }
            """##,
            grammar: ##"""
            a-kv ::= "\"a\"" space ":" space string
            a-rest ::= ( "," space b-kv )? b-rest
            b-kv ::= "\"b\"" space ":" space string
            b-rest ::= ( "," space c-kv )?
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space  (a-kv a-rest | b-kv b-rest | c-kv )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            """##),
        // 決定 (a) により宣言順ではなく昇順。参照実装の期待値と
        // 異なるのはプロパティの並びだけである (SPEC §12)。
        JSONSchemaGrammarCase(
            name: "required + optional props each in original order",
            schema: ##"""
            {
            "properties": {
            "b": {"type": "string"},
            "a": {"type": "string"},
            "d": {"type": "string"},
            "c": {"type": "string"}
            },
            "required": ["a", "b"],
            "additionalProperties": false
            }
            """##,
            grammar: ##"""
            a-kv ::= "\"a\"" space ":" space string
            b-kv ::= "\"b\"" space ":" space string
            c-kv ::= "\"c\"" space ":" space string
            c-rest ::= ( "," space d-kv )?
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            d-kv ::= "\"d\"" space ":" space string
            root ::= "{" space a-kv "," space b-kv ( "," space ( c-kv c-rest | d-kv ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            """##),
        JSONSchemaGrammarCase(
            name: "additional props",
            schema: ##"""
            {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "number"}}
            }
            """##,
            grammar: ##"""
            additional-kv ::= string ":" space additional-value
            additional-value ::= "[" space (number ("," space number)*)? space "]"
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space  (additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            """##),
        JSONSchemaGrammarCase(
            name: "additional props (true)",
            schema: ##"""
            {
            "type": "object",
            "additionalProperties": true
            }
            """##,
            grammar: ##"""
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
            """##),
        JSONSchemaGrammarCase(
            name: "additional props (implicit)",
            schema: ##"""
            {
            "type": "object"
            }
            """##,
            grammar: ##"""
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
            """##),
        JSONSchemaGrammarCase(
            name: "empty w/o additional props",
            schema: ##"""
            {
            "type": "object",
            "additionalProperties": false
            }
            """##,
            grammar: ##"""
            root ::= "{" space  space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "required + additional props",
            schema: ##"""
            {
            "type": "object",
            "properties": {
            "a": {"type": "number"}
            },
            "required": ["a"],
            "additionalProperties": {"type": "string"}
            }
            """##,
            grammar: ##"""
            a-kv ::= "\"a\"" space ":" space number
            additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space a-kv ( "," space ( additional-kv ( "," space additional-kv )* ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            """##),
        JSONSchemaGrammarCase(
            name: "optional + additional props",
            schema: ##"""
            {
            "type": "object",
            "properties": {
            "a": {"type": "number"}
            },
            "additionalProperties": {"type": "number"}
            }
            """##,
            grammar: ##"""
            a-kv ::= "\"a\"" space ":" space number
            a-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space number
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space  (a-kv a-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "required + optional + additional props",
            schema: ##"""
            {
            "type": "object",
            "properties": {
            "and": {"type": "number"},
            "also": {"type": "number"}
            },
            "required": ["and"],
            "additionalProperties": {"type": "number"}
            }
            """##,
            grammar: ##"""
            additional-k ::= ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space number
            also-kv ::= "\"also\"" space ":" space number
            also-rest ::= ( "," space additional-kv )*
            and-kv ::= "\"and\"" space ":" space number
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space and-kv ( "," space ( also-kv also-rest | additional-kv ( "," space additional-kv )* ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "optional props with empty name",
            schema: ##"""
            {
            "properties": {
            "": {"type": "integer"},
            "a": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
            }
            """##,
            grammar: ##"""
            -kv ::= "\"\"" space ":" space root
            -rest ::= ( "," space a-kv )? a-rest
            a-kv ::= "\"a\"" space ":" space integer
            a-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] char+ | [^"a] char* ) ["]
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part)
            root0 ::= "{" space  (-kv -rest | a-kv a-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "optional props with nested names",
            schema: ##"""
            {
            "properties": {
            "a": {"type": "integer"},
            "aa": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
            }
            """##,
            grammar: ##"""
            a-kv ::= "\"a\"" space ":" space integer
            a-rest ::= ( "," space aa-kv )? aa-rest
            aa-kv ::= "\"aa\"" space ":" space integer
            aa-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] ([a] char+ | [^"a] char*) | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= "{" space  (a-kv a-rest | aa-kv aa-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "optional props with common prefix",
            schema: ##"""
            {
            "properties": {
            "ab": {"type": "integer"},
            "ac": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
            }
            """##,
            grammar: ##"""
            ab-kv ::= "\"ab\"" space ":" space integer
            ab-rest ::= ( "," space ac-kv )? ac-rest
            ac-kv ::= "\"ac\"" space ":" space integer
            ac-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] ([b] char+ | [c] char+ | [^"bc] char*) | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= "{" space  (ab-kv ab-rest | ac-kv ac-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "top-level $ref",
            schema: ##"""
            {
            "$ref": "#/definitions/foo",
            "definitions": {
            "foo": {
            "type": "object",
            "properties": {
            "a": {
            "type": "string"
            }
            },
            "required": [
            "a"
            ],
            "additionalProperties": false
            }
            }
            }
            """##,
            grammar: ##"""
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            ref-definitions-foo ::= "{" space ref-definitions-foo-a-kv space "}"
            ref-definitions-foo-a-kv ::= "\"a\"" space ":" space string
            root ::= ref-definitions-foo
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            """##),
        JSONSchemaGrammarCase(
            name: "anyOf",
            schema: ##"""
            {
            "anyOf": [
            {"$ref": "#/definitions/foo"},
            {"$ref": "#/definitions/bar"}
            ],
            "definitions": {
            "foo": {
            "properties": {"a": {"type": "number"}}
            },
            "bar": {
            "properties": {"b": {"type": "number"}}
            }
            },
            "type": "object"
            }
            """##,
            grammar: ##"""
            alternative-0 ::= ref-definitions-foo
            alternative-1 ::= ref-definitions-bar
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            ref-definitions-bar ::= "{" space  (ref-definitions-bar-b-kv )? space "}"
            ref-definitions-bar-b-kv ::= "\"b\"" space ":" space number
            ref-definitions-foo ::= "{" space  (ref-definitions-foo-a-kv )? space "}"
            ref-definitions-foo-a-kv ::= "\"a\"" space ":" space number
            root ::= alternative-0 | alternative-1
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "anyOf $ref",
            schema: ##"""
            {
            "properties": {
            "a": {
            "anyOf": [
            {"type": "string"},
            {"type": "number"}
            ]
            },
            "b": {
            "anyOf": [
            {"$ref": "#/properties/a/anyOf/0"},
            {"type": "boolean"}
            ]
            }
            },
            "type": "object"
            }
            """##,
            grammar: ##"""
            a ::= string | number
            a-kv ::= "\"a\"" space ":" space a
            a-rest ::= ( "," space b-kv )?
            b ::= b-0 | boolean
            b-0 ::= string
            b-kv ::= "\"b\"" space ":" space b
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space  (a-kv a-rest | b-kv )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            """##),
        // 決定 (a) により宣言順ではなく昇順。参照実装の期待値と
        // 異なるのはプロパティの並びだけである (SPEC §12)。
        JSONSchemaGrammarCase(
            name: "mix of allOf, anyOf and $ref (similar to https://json.schemastore.org/tsconfig.json)",
            schema: ##"""
            {
            "allOf": [
            {"$ref": "#/definitions/foo"},
            {"$ref": "#/definitions/bar"},
            {
            "anyOf": [
            {"$ref": "#/definitions/baz"},
            {"$ref": "#/definitions/bam"}
            ]
            }
            ],
            "definitions": {
            "foo": {
            "properties": {"a": {"type": "number"}}
            },
            "bar": {
            "properties": {"b": {"type": "number"}}
            },
            "bam": {
            "properties": {"c": {"type": "number"}}
            },
            "baz": {
            "properties": {"d": {"type": "number"}}
            }
            },
            "type": "object"
            }
            """##,
            grammar: ##"""
            a-kv ::= "\"a\"" space ":" space number
            b-kv ::= "\"b\"" space ":" space number
            c-kv ::= "\"c\"" space ":" space number
            c-rest ::= ( "," space d-kv )?
            d-kv ::= "\"d\"" space ":" space number
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space a-kv "," space b-kv ( "," space ( c-kv c-rest | d-kv ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "allOf with enum schema",
            schema: ##"""
            {
            "allOf": [
            {"$ref": "#/definitions/foo"}
            ],
            "definitions": {
            "foo": {
            "type": "string",
            "enum": ["a", "b"]
            }
            }
            }
            """##,
            grammar: ##"""
            root ::= ("\"a\"" | "\"b\"")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "allOf with multiple enum schemas",
            schema: ##"""
            {
            "allOf": [
            {"$ref": "#/definitions/foo"},
            {"$ref": "#/definitions/bar"}
            ],
            "definitions": {
            "foo": {
            "type": "string",
            "enum": ["a", "b", "c"]
            },
            "bar": {
            "type": "string",
            "enum": ["b", "c", "d"]
            }
            }
            }
            """##,
            grammar: ##"""
            root ::= ("\"b\"" | "\"c\"")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "conflicting names",
            schema: ##"""
            {
            "type": "object",
            "properties": {
            "number": {
            "type": "object",
            "properties": {
            "number": {
            "type": "object",
            "properties": {
            "root": {
            "type": "number"
            }
            },
            "required": [
            "root"
            ],
            "additionalProperties": false
            }
            },
            "required": [
            "number"
            ],
            "additionalProperties": false
            }
            },
            "required": [
            "number"
            ],
            "additionalProperties": false,
            "definitions": {}
            }
            """##,
            grammar: ##"""
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            number- ::= "{" space number-number-kv space "}"
            number-kv ::= "\"number\"" space ":" space number-
            number-number ::= "{" space number-number-root-kv space "}"
            number-number-kv ::= "\"number\"" space ":" space number-number
            number-number-root-kv ::= "\"root\"" space ":" space number
            root ::= "{" space number-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "description only (no type) treated as unconstrained",
            schema: ##"""
            {"description": "The 0-based index of the last line to be retrieved (inclusive). If None, read until the end of the file."}
            """##,
            grammar: ##"""
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= value
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
            """##),
        JSONSchemaGrammarCase(
            name: "literal string with escapes",
            schema: ##"""
            {
            "properties": {
            "code": {
            "const": " \r \n \" \\ ",
            "description": "Generated code",
            "title": "Code",
            "type": "string"
            }
            },
            "required": [
            "code"
            ],
            "title": "DecoderResponse",
            "type": "object"
            }
            """##,
            grammar: ##"""
            code ::= "\" \\r \\n \\\" \\\\ \""
            code-kv ::= "\"code\"" space ":" space code
            root ::= "{" space code-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
    ]
}
