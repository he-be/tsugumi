import Foundation
import Testing
@testable import Tsugumi

/// SPEC §6 GEN-1 / GEN-3 — `pattern` (正規表現) → GBNF。
/// 期待値は参照実装 `tests/test-json-schema-to-grammar.cpp` (ピン `34af94cd9`)。
/// このファイルの表にプロパティを持つスキーマは無いので、決定 (a) は効かない。
@Suite("JSONSchemaGrammar pattern (GEN-1/GEN-3)")
struct JSONSchemaGrammarPatternTests {
    @Test("正規表現が参照実装と同じ文法になる", arguments: JSONSchemaGrammarPatternCases.all)
    func matchesReference(_ testCase: JSONSchemaGrammarCase) throws {
        let result = try JSONSchemaGrammarFixture.convert(testCase.schema)
        #expect(result.approximations.isEmpty)
        #expect(JSONSchemaGrammarFixture.trim(result.grammar) == testCase.grammar)
    }
}

enum JSONSchemaGrammarPatternCases {
    static let all: [JSONSchemaGrammarCase] = [
        JSONSchemaGrammarCase(
            name: "simple regexp",
            schema: ##"""
            {
            "type": "string",
            "pattern": "^abc?d*efg+(hij)?kl$"
            }
            """##,
            grammar: ##"""
            root ::= "\"" ("ab" "c"? "d"* "ef" "g"+ ("hij")? "kl") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "regexp escapes",
            schema: ##"""
            {
            "type": "string",
            "pattern": "^\\[\\]\\{\\}\\(\\)\\|\\+\\*\\?$"
            }
            """##,
            grammar: ##"""
            root ::= "\"" ("[]{}()|+*?") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "regexp quote",
            schema: ##"""
            {
            "type": "string",
            "pattern": "^\"$"
            }
            """##,
            grammar: ##"""
            root ::= "\"" ("\"") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "regexp with top-level alternation",
            schema: ##"""
            {
            "type": "string",
            "pattern": "^A|B|C|D$"
            }
            """##,
            grammar: ##"""
            root ::= "\"" ("A" | "B" | "C" | "D") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "regexp",
            schema: ##"""
            {
            "type": "string",
            "pattern": "^(\\([0-9]{1,3}\\))?[0-9]{3}-[0-9]{4} a{3,5}nd...$"
            }
            """##,
            grammar: ##"""
            dot ::= [^\x0A\x0D]
            root ::= "\"" (("(" root-1{1,3} ")")? root-1{3,3} "-" root-1{4,4} " " "a"{3,5} "nd" dot dot dot) "\""
            root-1 ::= [0-9]
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "regexp with non-capturing group",
            schema: ##"""
            {
            "type": "string",
            "pattern": "^(?:foo|bar)baz$"
            }
            """##,
            grammar: ##"""
            root ::= "\"" (("foo" | "bar") "baz") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        JSONSchemaGrammarCase(
            name: "regexp with nested non-capturing groups",
            schema: ##"""
            {
            "type": "string",
            "pattern": "^(?:(?:ab)+c)?d$"
            }
            """##,
            grammar: ##"""
            root ::= "\"" ((("ab")+ "c")? "d") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
    ]
}
