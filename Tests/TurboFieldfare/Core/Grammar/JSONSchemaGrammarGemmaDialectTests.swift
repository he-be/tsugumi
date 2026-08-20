import Foundation
import Testing
@testable import TurboFieldfare

/// SPEC §6 GEN-1 — tool call の引数を拘束する方言。
///
/// 参照実装に対応物が無いので期待値は参照実装からは取れない。出所は
/// **サーバーが実際に描く形と読む形の 2 つ**である:
/// - 描く側 `Sources/TurboFieldfare/Templates/server_chat_template.jinja` の
///   `format_argument` — キーは裸 (`escape_keys=False`)、文字列は
///   `<|"|>…<|"|>`、キーの並びは Jinja の `dictsort` で昇順。
/// - 読む側 `GemmaToolCallParser` — キーは `[A-Za-z0-9_\-.$]+`、文字列は
///   `<|"|>…<|"|>` と `"…"` の**どちらも**受ける。
///
/// よってこの方言は `.json` から次の 3 点だけが違う。
/// 1. `string` は 2 形式の選択。
/// 2. オブジェクトのキーは裸 (`key` / `key-char` の原始規則が入る)。
/// 3. `additional-k` の除外集合はキーの字集合の中で取る (`["]` の囲いは無く、
///    空キーは書けないので参照実装の末尾 `?` も無い)。
/// 数値・真偽・null・配列・入れ子・区切り・`space` の位置は `.json` と同じ。
@Suite("JSONSchemaGrammar gemma dialect (GEN-1)")
struct JSONSchemaGrammarGemmaDialectTests {
    @Test("tool call 引数の方言が期待どおりの文法になる",
          arguments: JSONSchemaGrammarGemmaCases.all)
    func matchesDialect(_ testCase: JSONSchemaGrammarCase) throws {
        let result = try JSONSchemaGrammarFixture.convert(
            testCase.schema, dialect: .gemmaToolArguments)
        #expect(result.approximations.isEmpty)
        #expect(JSONSchemaGrammarFixture.trim(result.grammar) == testCase.grammar)
    }
}

enum JSONSchemaGrammarGemmaCases {
    static let all: [JSONSchemaGrammarCase] = [
        // 平たいオブジェクト。キーは裸、文字列値は 2 形式。
        JSONSchemaGrammarCase(
            name: "gemma flat object",
            schema: ##"""
            {
              "type": "object",
              "properties": {
                "b": {"type": "string"},
                "a": {"type": "integer"}
              },
              "required": ["a", "b"]
            }
            """##,
            grammar: ##"""
            a-kv ::= "a" space ":" space integer
            b-kv ::= "b" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= "{" space a-kv "," space b-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" | "<|\"|>" char* "<|\"|>"
            """##),
        // 入れ子のオブジェクト。
        JSONSchemaGrammarCase(
            name: "gemma nested object",
            schema: ##"""
            {
              "type": "object",
              "properties": {
                "outer": {
                  "type": "object",
                  "properties": {"inner": {"type": "string"}},
                  "required": ["inner"]
                }
              },
              "required": ["outer"]
            }
            """##,
            grammar: ##"""
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            outer ::= "{" space outer-inner-kv space "}"
            outer-inner-kv ::= "inner" space ":" space string
            outer-kv ::= "outer" space ":" space outer
            root ::= "{" space outer-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" | "<|\"|>" char* "<|\"|>"
            """##),
        // オブジェクトの配列。区切りと繰り返しは `.json` と同じ。
        JSONSchemaGrammarCase(
            name: "gemma array of objects",
            schema: ##"""
            {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"]
              }
            }
            """##,
            grammar: ##"""
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            item ::= "{" space item-n-kv space "}"
            item-n-kv ::= "n" space ":" space integer
            root ::= "[" space (item ("," space item)*)? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        // enum。列挙の順は与えられたまま。各値が 2 形式の選択になる。
        JSONSchemaGrammarCase(
            name: "gemma string enum",
            schema: ##"""
            {
              "type": "string",
              "enum": ["red", "green"]
            }
            """##,
            grammar: ##"""
            root ::= (("\"red\"" | "<|\"|>red<|\"|>") | ("\"green\"" | "<|\"|>green<|\"|>"))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            """##),
        // additionalProperties。`additional-k` はキーの字集合から宣言済みの
        // キーを引いたもので、参照実装の `["] … ["]` の囲いと末尾 `?` は無い。
        JSONSchemaGrammarCase(
            name: "gemma additional properties",
            schema: ##"""
            {
              "type": "object",
              "properties": {"a": {"type": "integer"}},
              "required": ["a"],
              "additionalProperties": {"type": "string"}
            }
            """##,
            grammar: ##"""
            a-kv ::= "a" space ":" space integer
            additional-k ::= ( [a] key-char+ | [0-9A-Zb-z_\-.$] key-char* )
            additional-kv ::= additional-k ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            key-char ::= [0-9A-Za-z_\-.$]
            root ::= "{" space a-kv ( "," space ( additional-kv ( "," space additional-kv )* ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" | "<|\"|>" char* "<|\"|>"
            """##),
    ]
}
