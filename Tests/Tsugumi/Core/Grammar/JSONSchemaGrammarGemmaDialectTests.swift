import Foundation
import Testing
@testable import Tsugumi

/// SPEC §6 GEN-1 / **GEN-8** — tool call の引数を拘束する方言。
///
/// GEN-8: **文法はテンプレートが描き直す正準形だけを許す。**参照実装に対応物が
/// 無いので期待値の出所はテンプレートそのもの
/// (`Sources/Tsugumi/Templates/server_chat_template.jinja` の
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
/// 4. **GEN-9**: 文字列の本体は `[^"\]*` — `"` と `\` を含まない任意の文字。
///    JSON のエスケープ形は一切許さない (テンプレートは値を素のまま書く)。
///    JSON の `char` 規則は使わず、この方言専用の `text-char` を使う。
/// 5. **GEN-10**: 汎用の値の選択肢から `null` を外す。スキーマが null を明示
///    的に要求したときだけ許し、近似 `null-not-redrawable` を記録する。
/// 6. **GEN-11**: 数の桁を、描き直しの `Decimal` → `Double` 往復が通る幅に
///    絞る (指数形は無し)。JSON Schema 側の制約ではなく描き直しの制約である。
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

    /// GEN-8: 空白を一切入れない。この方言では `space` という名前が文法の
    /// どこにも出てこない (参照されない規則は文法の雑音でしかない)。
    @Test("文法に space という名前が出てこない",
          arguments: JSONSchemaGrammarGemmaCases.all)
    func emitsNoSpaceRule(_ testCase: JSONSchemaGrammarCase) throws {
        let result = try JSONSchemaGrammarFixture.convert(
            testCase.schema, dialect: .gemmaToolArguments)
        #expect(!result.grammar.contains("space"))
    }

    /// GEN-9 / GEN-10: 描き直せない値は文法から外す。ただし**文法を充足不能に
    /// はしない** — 生成を止めるほうが害が大きい (GEN-2 と同じ原則)。
    @Test("描き直せない値は近似に落ちる",
          arguments: JSONSchemaGrammarGemmaCases.unrepresentable)
    func degradesUnrepresentableValues(
        _ testCase: JSONSchemaGrammarGemmaCases.UnrepresentableCase
    ) throws {
        let result = try JSONSchemaGrammarFixture.convert(
            testCase.schema, dialect: .gemmaToolArguments)
        #expect(result.approximations == testCase.approximations)
        #expect(JSONSchemaGrammarFixture.body(of: "root", in: result.grammar)
                == testCase.rootRule)
        if let absent = testCase.absent {
            // 文法がその値を綴れないことを、綴りが文法のどこにも無いことで見る。
            #expect(!result.grammar.contains(absent))
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
    /// 名前つき規則の本体。
    static func body(of name: String, in grammar: String) -> String? {
        for line in grammar.split(separator: "\n") where line.hasPrefix(name + " ::= ") {
            return String(line.dropFirst(name.count + 5))
        }
        return nil
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

    struct UnrepresentableCase: Sendable, CustomTestStringConvertible {
        let name: String
        let schema: String
        let approximations: [String]
        /// 落とし先の `root` の本体。
        let rootRule: String
        /// 文法のどこにも出てこないはずの綴り (nil なら見ない)。
        let absent: String?

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
        // GEN-9: `"` と `\` 以外は素のまま往復する。空白も句読点も括弧も、
        // テンプレートは何も逃がさずに書く。
        CanonicalCase(
            name: "string with spaces and punctuation round-trips",
            arguments: ##"{"note": "hello, world! (ok)"}"##,
            canonical: ##"{note:<|"|>hello, world! (ok)<|"|>}"##),
    ]

    /// GEN-9 / GEN-10: 描き直せない値。文法からは外れるが、文法は充足可能な
    /// まま (`string` / 明示的な `null`) にする。
    static let unrepresentable: [UnrepresentableCase] = [
        // `"` はテンプレートの終端子 `<|"|>` の一部なので、素で書くと読む側が
        // 途中で切ってしまう。const は綴りを固定できないので汎用の `string`
        // (= `[^"\]*`) に落ちる — つまり文法はこの値を綴れない。
        UnrepresentableCase(
            name: "const string containing a double quote",
            schema: ##"{"const": "say \"hi\""}"##,
            approximations: [##"unrepresentable-string-value: say "hi""##],
            rootRule: "string",
            absent: "say"),
        // `\` は読む側 (`GemmaToolCallParser.gemmaString`) がエスケープ導入と
        // 解釈するので、同じく往復しない。
        UnrepresentableCase(
            name: "const string containing a backslash",
            schema: ##"{"const": "back\\slash"}"##,
            approximations: [##"unrepresentable-string-value: back\slash"##],
            rootRule: "string",
            absent: "back"),
        // enum は書ける値だけが綴りのまま残り、書けない値は `string` に開く。
        UnrepresentableCase(
            name: "enum with one unrepresentable value",
            schema: ##"{"type": "string", "enum": ["ok", "bad\"quote"]}"##,
            approximations: [##"unrepresentable-string-value: bad"quote"##],
            rootRule: ##"("<|\"|>ok<|\"|>" | string)"##,
            absent: "quote"),
        // GEN-10: null を明示的に要求されたら許すが、記録は残す。
        UnrepresentableCase(
            name: "explicit null type",
            schema: ##"{"type": "null"}"##,
            approximations: ["null-not-redrawable"],
            rootRule: ##""null""##,
            absent: nil),
        UnrepresentableCase(
            name: "nullable string (type array)",
            schema: ##"{"type": ["string", "null"]}"##,
            approximations: ["null-not-redrawable"],
            rootRule: "string | null",
            absent: nil),
        UnrepresentableCase(
            name: "const null",
            schema: ##"{"const": null}"##,
            approximations: ["null-not-redrawable"],
            rootRule: ##""null""##,
            absent: nil),
    ]

    static let all: [JSONSchemaGrammarCase] = [
        // 平たいオブジェクト。キーは裸、文字列は学習形式のみ、空白なし。
        // 文字列の本体は JSON の `char` ではなく `text-char` (GEN-9)。
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
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,14}
            root ::= "{" a-kv "," b-kv "}"
            string ::= "<|\"|>" text-char* "<|\"|>"
            text-char ::= [^"\\]
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
            outer ::= "{" outer-inner-kv "}"
            outer-inner-kv ::= "inner" ":" string
            outer-kv ::= "outer" ":" outer
            root ::= "{" outer-kv "}"
            string ::= "<|\"|>" text-char* "<|\"|>"
            text-char ::= [^"\\]
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
            integral-part ::= [0] | [1-9] [0-9]{0,14}
            item ::= "{" item-n-kv "}"
            item-n-kv ::= "n" ":" integer
            root ::= "[" (item ("," item)*)? "]"
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
            additional-k ::= ( [a] key-char+ | [0-9A-Zb-z_\x2D.$] key-char* )
            additional-kv ::= additional-k ":" string
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,14}
            key-char ::= [0-9A-Za-z_\x2D.$]
            root ::= "{" a-kv ( "," ( additional-kv ( "," additional-kv )* ) )? "}"
            string ::= "<|\"|>" text-char* "<|\"|>"
            text-char ::= [^"\\]
            """##),
        // 型無しの自由な値。**`value` に `null` の枝が無い** (GEN-10) こと、
        // `object` / `array` の原始規則にも空白が入らないことがここで決まる。
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
            decimal-part ::= [0-9]{1,7}
            integral-part ::= [0] | [1-9] [0-9]{0,14}
            key ::= key-char+
            key-char ::= [0-9A-Za-z_\x2D.$]
            number ::= ("-"? integral-part) | ("-"? ([0] | [1-9] [0-9]{0,6}) "." decimal-part)
            object ::= "{" ( key ":" value ("," key ":" value)* )? "}"
            payload ::= object
            payload-kv ::= "payload" ":" payload
            root ::= "{" payload-kv "}"
            string ::= "<|\"|>" text-char* "<|\"|>"
            text-char ::= [^"\\]
            value ::= object | array | string | number | boolean
            """##),
    ]
}
