import Foundation
import Testing
@testable import TurboFieldfare

/// SPEC §6 GEN-1 / **GEN-8** — tool call の引数を拘束する方言。
///
/// GEN-8: **文法はテンプレートが描き直す正準形だけを許す。**参照実装に対応物が
/// 無いので期待値の出所はテンプレートそのもの
/// (`Sources/TurboFieldfare/Templates/server_chat_template.jinja` の
/// `format_argument` と、`tool_calls` を描くループ) である。
///
/// `.json` からの違いは 3 点だけ。
/// 1. **空白を一切入れない。**テンプレートは `{k:v,k2:v2}` / `[a,b]` を空白
///    なしで描くので、参照実装が `space` を差し込む位置には何も入れない。
///    `space` 規則そのものは**空文字列しか受けない**ものとして残る (どの規則も
///    参照しない)。`.json` の `space` は参照実装のまま。
/// 2. **オブジェクトのキーは裸で昇順** (テンプレートの `dictsort`)。
///    `key` / `key-char` の原始規則が入り、`additional-k` の除外集合は
///    キーの字集合の中で取る (GBNF に交差が無いため)。
/// 3. **文字列は `<|"|>…<|"|>` のみ。**JSON の `"…"` は生成では許さない
///    (読む側 `GemmaToolCallParser` は互換のため両方受けるが、それは読みの話で
///    あって生成の契約ではない)。
///
/// 理由は §7 INV-1: 完了した tool call ターンは parse 済みの値から描き直される
/// ので、生成が正準形でなければ描き直しと必ずずれ、毎ターン LCP が切れる。
@Suite("JSONSchemaGrammar gemma dialect (GEN-1/GEN-8)")
struct JSONSchemaGrammarGemmaDialectTests {
    @Test("tool call 引数の方言が期待どおりの文法になる",
          arguments: JSONSchemaGrammarGemmaCases.all)
    func matchesDialect(_ testCase: JSONSchemaGrammarCase) throws {
        let result = try JSONSchemaGrammarFixture.convert(
            testCase.schema, dialect: .gemmaToolArguments)
        #expect(result.approximations.isEmpty)
        #expect(JSONSchemaGrammarFixture.trim(result.grammar) == testCase.grammar)
    }

    /// GEN-8: JSON の `"…"` は生成では許さない。文法本文に GBNF の二重引用符
    /// リテラル (`"\""`) が 1 つも出てこないことで検定する。
    @Test("JSON の引用符付き文字列は生成の文法に出てこない",
          arguments: JSONSchemaGrammarGemmaCases.all)
    func rejectsJSONQuotedStrings(_ testCase: JSONSchemaGrammarCase) throws {
        let result = try JSONSchemaGrammarFixture.convert(
            testCase.schema, dialect: .gemmaToolArguments)
        #expect(!result.grammar.contains(##""\"""##))
    }

    /// GEN-8: 空白を一切入れない。どの規則の本体も `space` を参照しない。
    @Test("どの規則も space を参照しない",
          arguments: JSONSchemaGrammarGemmaCases.all)
    func emitsNoSpaceReference(_ testCase: JSONSchemaGrammarCase) throws {
        let result = try JSONSchemaGrammarFixture.convert(
            testCase.schema, dialect: .gemmaToolArguments)
        for line in result.grammar.split(separator: "\n") {
            guard let separator = line.range(of: " ::= ") else { continue }
            let name = String(line[..<separator.lowerBound])
            let body = String(line[separator.upperBound...])
            if name == "space" {
                // 空文字列しか受けない。
                #expect(body == ##""""##)
                continue
            }
            #expect(!JSONSchemaGrammarFixture.references(body).contains("space"),
                    "\(name) が space を参照している")
        }
    }

    /// GEN-8 / INV-1 の本体の検定。
    ///
    /// 値をすべて `const` で固定したスキーマの文法は**ちょうど 1 本の文字列**
    /// しか受けない。その 1 本を文法から取り出し、**テンプレートが同じ引数
    /// 辞書に対して描く正準形**と 1 文字ずつ突き合わせる。期待値はテンプレート
    /// の `format_argument` を読んで書いた文字列リテラルなので、文法かテンプレ
    /// ートのどちらが動いてもこの表が壊れる。
    @Test("固定値スキーマの文法はテンプレートの正準形と 1 文字ずつ一致する",
          arguments: JSONSchemaGrammarGemmaCases.canonical)
    func matchesTemplateRedraw(_ testCase: JSONSchemaGrammarGemmaCases.CanonicalCase) throws {
        let arguments = try JSONDecoder().decode(
            JSONValue.self, from: Data(testCase.arguments.utf8))
        let schema = JSONSchemaGrammarGemmaCases.constSchema(arguments)
        let result = JSONSchemaGrammar.grammar(for: schema, dialect: .gemmaToolArguments)
        #expect(result.approximations.isEmpty)
        #expect(JSONSchemaGrammarFixture.onlyString(in: result.grammar) == testCase.canonical)
    }
}

extension JSONSchemaGrammarFixture {
    /// 規則本体が参照している規則名。
    static func references(_ body: String) -> Set<String> {
        var names: Set<String> = []
        var current = ""
        var inLiteral = false
        var escaped = false
        var inClass = false
        for character in body {
            if inLiteral {
                if escaped { escaped = false } else if character == "\\" { escaped = true }
                else if character == "\"" { inLiteral = false }
                continue
            }
            if inClass {
                if escaped { escaped = false } else if character == "\\" { escaped = true }
                else if character == "]" { inClass = false }
                continue
            }
            if character == "\"" { inLiteral = true; continue }
            if character == "[" { inClass = true; continue }
            if character.isLetter || character.isNumber || character == "-" {
                current.append(character)
                continue
            }
            if !current.isEmpty { names.insert(current); current = "" }
        }
        if !current.isEmpty { names.insert(current) }
        return names
    }

    /// 決定的な文法 (選択も繰り返しも無い) が受ける唯一の文字列。
    /// 決定的でなければ `nil`。
    static func onlyString(in grammar: String) -> String? {
        var rules: [String: String] = [:]
        for line in grammar.split(separator: "\n") {
            guard let separator = line.range(of: " ::= ") else { continue }
            rules[String(line[..<separator.lowerBound])] =
                String(line[separator.upperBound...])
        }
        return expand("root", rules, depth: 0)
    }

    private static func expand(_ name: String, _ rules: [String: String], depth: Int) -> String? {
        guard depth < 64, let body = rules[name] else { return nil }
        var out = ""
        let characters = Array(body)
        var index = 0
        while index < characters.count {
            let character = characters[index]
            if character == " " { index += 1; continue }
            if character == "\"" {
                index += 1
                while index < characters.count, characters[index] != "\"" {
                    if characters[index] == "\\", index + 1 < characters.count {
                        index += 1
                        switch characters[index] {
                        case "n": out.append("\n")
                        case "r": out.append("\r")
                        default: out.append(characters[index])
                        }
                    } else {
                        out.append(characters[index])
                    }
                    index += 1
                }
                guard index < characters.count else { return nil }
                index += 1
                continue
            }
            if character.isLetter || character.isNumber || character == "-" {
                var identifier = ""
                while index < characters.count,
                      characters[index].isLetter || characters[index].isNumber
                        || characters[index] == "-" {
                    identifier.append(characters[index])
                    index += 1
                }
                guard let expanded = expand(identifier, rules, depth: depth + 1) else {
                    return nil
                }
                out += expanded
                continue
            }
            // 選択・繰り返し・文字クラスがあれば決定的ではない。
            return nil
        }
        return out
    }
}

enum JSONSchemaGrammarGemmaCases {
    struct CanonicalCase: Sendable, CustomTestStringConvertible {
        let name: String
        /// tool call の引数辞書。
        let arguments: String
        /// テンプレートが同じ辞書に対して描く正準形 (`format_argument` を
        /// 読んで書いた文字列リテラル — ここが唯一の期待値の出所)。
        let canonical: String

        var testDescription: String { name }
    }

    /// 引数辞書から「値を全部 `const` で固定した」スキーマを作る。
    static func constSchema(_ arguments: JSONValue) -> JSONValue {
        guard case .object(let members) = arguments else { return .object([:]) }
        var properties: [String: JSONValue] = [:]
        for (key, value) in members {
            properties[key] = .object(["const": value])
        }
        return .object([
            "type": .string("object"),
            "properties": .object(properties),
            "required": .array(members.keys.sorted().map { JSONValue.string($0) }),
        ])
    }

    static let canonical: [CanonicalCase] = [
        CanonicalCase(
            name: "string and integer",
            arguments: ##"{"city": "Paris", "days": 3}"##,
            canonical: ##"{city:<|"|>Paris<|"|>,days:3}"##),
        CanonicalCase(
            name: "boolean and negative integer",
            arguments: ##"{"flag": false, "n": -2}"##,
            canonical: ##"{flag:false,n:-2}"##),
        CanonicalCase(
            name: "nested object",
            arguments: ##"{"where": {"lat": 1, "lon": 2}}"##,
            canonical: ##"{where:{lat:1,lon:2}}"##),
        CanonicalCase(
            name: "array of strings",
            arguments: ##"{"tags": ["a", "b"]}"##,
            canonical: ##"{tags:[<|"|>a<|"|>,<|"|>b<|"|>]}"##),
        CanonicalCase(
            name: "keys are sorted, not declaration order",
            arguments: ##"{"z": 1, "a": 2}"##,
            canonical: ##"{a:2,z:1}"##),
        CanonicalCase(
            name: "empty object value",
            arguments: ##"{"o": {}}"##,
            canonical: ##"{o:{}}"##),
        CanonicalCase(
            name: "empty array value",
            arguments: ##"{"o": []}"##,
            canonical: ##"{o:[]}"##),
    ]

    static let all: [JSONSchemaGrammarCase] = [
        // 平たいオブジェクト。キーは裸、文字列は学習形式のみ、空白なし。
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
            a-kv ::= "a" ":" integer
            b-kv ::= "b" ":" string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= "{" a-kv "," b-kv "}"
            space ::= ""
            string ::= "<|\"|>" char* "<|\"|>"
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
            outer ::= "{" outer-inner-kv "}"
            outer-inner-kv ::= "inner" ":" string
            outer-kv ::= "outer" ":" outer
            root ::= "{" outer-kv "}"
            space ::= ""
            string ::= "<|\"|>" char* "<|\"|>"
            """##),
        // オブジェクトの配列。区切りは `,` だけで空白は入らない。
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
            item ::= "{" item-n-kv "}"
            item-n-kv ::= "n" ":" integer
            root ::= "[" (item ("," item)*)? "]"
            space ::= ""
            """##),
        // enum。列挙の順は与えられたまま。値は学習形式の 1 本だけ。
        JSONSchemaGrammarCase(
            name: "gemma string enum",
            schema: ##"""
            {
              "type": "string",
              "enum": ["red", "green"]
            }
            """##,
            grammar: ##"""
            root ::= ("<|\"|>red<|\"|>" | "<|\"|>green<|\"|>")
            space ::= ""
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
            a-kv ::= "a" ":" integer
            additional-k ::= ( [a] key-char+ | [0-9A-Zb-z_\-.$] key-char* )
            additional-kv ::= additional-k ":" string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            key-char ::= [0-9A-Za-z_\-.$]
            root ::= "{" a-kv ( "," ( additional-kv ( "," additional-kv )* ) )? "}"
            space ::= ""
            string ::= "<|\"|>" char* "<|\"|>"
            """##),
        // 型無しの自由な値。`object` / `array` の原始規則にも空白は入らない。
        JSONSchemaGrammarCase(
            name: "gemma free-form value",
            schema: ##"""
            {
              "type": "object",
              "properties": {"payload": {}},
              "required": ["payload"]
            }
            """##,
            grammar: ##"""
            array ::= "[" ( value ("," value)* )? "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            key ::= key-char+
            key-char ::= [0-9A-Za-z_\-.$]
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" ( key ":" value ("," key ":" value)* )? "}"
            payload ::= object
            payload-kv ::= "payload" ":" payload
            root ::= "{" payload-kv "}"
            space ::= ""
            string ::= "<|\"|>" char* "<|\"|>"
            value ::= object | array | string | number | boolean | null
            """##),
    ]
}
