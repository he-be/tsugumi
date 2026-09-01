import Foundation
import Testing
@testable import Tsugumi

/// SPEC §6 **GEN-11** — tool call の数値は描き直しが通る桁数に絞る。
///
/// これは JSON Schema 側の制約ではなく**描き直しの制約**である。完了した
/// tool call ターンを描き直すとき、`Tokenizer.swift` は引数を
/// `JSONValue.jinjaSendableValue()` に通す。その `.decimal` 分岐は
/// `Decimal` → `Double` → `Decimal` の往復が値として一致しなければ
/// `GemmaToolCallParserError.malformed` を**投げる**ので、桁の多い数を生成
/// させると次のターンの描き直しが落ちる。単に LCP が縮むだけの数値のずれ
/// (GEN-8) と違い、これは応答の失敗である。
///
/// 上限の出所は `jinjaSendableValue()` が要求するもの、すなわち
/// **有効 10 進桁数 15 桁** (`Double` の `DBL_DIG`) である。GBNF は小数点を
/// またいで桁を数えられないので、十分条件として**両半分を 7 桁ずつ**
/// (7 + 7 = 14 ≤ 15、8 + 8 = 16 は超える) に絞り、小数部を持たない整数形だけ
/// は 15 桁の予算を丸ごと使う。指数形は落とす — `Decimal(string:)` が
/// `1e300` を受けないので、指数を許すと**読む側**が先に落ちる。
///
/// `.json` は参照実装のまま (16 桁)。上限は方言側の制約である。
@Suite("JSONSchemaGrammar gemma numbers (GEN-11)")
struct JSONSchemaGrammarGemmaNumberTests {
    /// 数を含む文法を出すためのスキーマ (自由な値 = `number` も `integer` も出る)。
    static let schema = ##"""
    {
      "type": "object",
      "properties": {"payload": {}, "count": {"type": "integer"}},
      "required": ["count", "payload"]
    }
    """##

    /// 規則本体の `{m,n}` の上限を左から順に。
    static func upperBounds(_ body: String) -> [Int] {
        var bounds: [Int] = []
        var rest = Substring(body)
        while let open = rest.firstIndex(of: "{") {
            guard let close = rest[open...].firstIndex(of: "}") else { break }
            let inside = rest[rest.index(after: open)..<close]
            if let comma = inside.lastIndex(of: ",") {
                bounds.append(Int(inside[inside.index(after: comma)...]) ?? -1)
            }
            rest = rest[rest.index(after: close)...]
        }
        return bounds
    }

    /// 文法が綴れるいちばん広い数を、**文法の本文から読み取って**組み立てる。
    /// 桁数をテストに直接書かないので、上限を動かせばここが自動で追随する。
    static func widestLiterals(in grammar: String) throws -> [String] {
        let integralBody = try #require(
            JSONSchemaGrammarFixture.body(of: "integral-part", in: grammar))
        let decimalBody = try #require(
            JSONSchemaGrammarFixture.body(of: "decimal-part", in: grammar))
        let numberBody = try #require(
            JSONSchemaGrammarFixture.body(of: "number", in: grammar))

        // `[0] | [1-9] [0-9]{0,N}` → 先頭 1 桁 + N 桁。
        let integerDigits = try #require(upperBounds(integralBody).first) + 1
        let fractionDigits = try #require(upperBounds(decimalBody).first)
        // `number` の小数形が内側に持つ `[0-9]{0,N}`。
        let fractionIntegerDigits = try #require(upperBounds(numberBody).first) + 1

        func nines(_ count: Int) -> String { String(repeating: "9", count: count) }
        return [
            nines(integerDigits),
            "-" + nines(integerDigits),
            nines(fractionIntegerDigits) + "." + nines(fractionDigits),
            "-" + nines(fractionIntegerDigits) + "." + nines(fractionDigits),
            "0." + nines(fractionDigits),
            "-0." + nines(fractionDigits),
        ]
    }

    private func redraw(_ literal: String) throws -> any Sendable {
        // 描き直しが通る経路そのもの (`Tokenizer.swift` の tool_calls 分岐)。
        let value = try JSONDecoder().decode(JSONValue.self, from: Data(literal.utf8))
        return try value.jinjaSendableValue()
    }

    @Test("文法が綴れるいちばん広い数は描き直しの変換を通る")
    func widestGeneratableNumberSurvivesRedraw() throws {
        let result = JSONSchemaGrammar.grammar(
            for: try JSONDecoder().decode(JSONValue.self, from: Data(Self.schema.utf8)),
            dialect: .gemmaToolArguments)
        for literal in try Self.widestLiterals(in: result.grammar) {
            #expect(throws: Never.self, "\(literal) が描き直しで落ちた") {
                try self.redraw(literal)
            }
        }
    }

    @Test("以前に落ちていた桁数はもう綴れない")
    func previouslyFailingWidthIsNotGeneratable() throws {
        // 参照実装の幅 (整数部 16 桁 / 小数部 16 桁) で綴れて、描き直しで落ちる数。
        for literal in ["9999999999.999999", "0.12345678901234567"] {
            #expect(throws: GemmaToolCallParserError.self) { try self.redraw(literal) }
        }

        let gemma = JSONSchemaGrammar.grammar(
            for: try JSONDecoder().decode(JSONValue.self, from: Data(Self.schema.utf8)),
            dialect: .gemmaToolArguments)
        let decimalBody = try #require(
            JSONSchemaGrammarFixture.body(of: "decimal-part", in: gemma.grammar))
        let numberBody = try #require(
            JSONSchemaGrammarFixture.body(of: "number", in: gemma.grammar))
        let fractionDigits = try #require(Self.upperBounds(decimalBody).first)
        let fractionIntegerDigits = try #require(Self.upperBounds(numberBody).first) + 1
        // `9999999999.999999` は整数部 10 桁、`0.1234…7` は小数部 17 桁。
        #expect(fractionIntegerDigits < 10)
        #expect(fractionDigits < 17)
        // 指数形は落とした。`Decimal(string:)` が `1e300` を受けないので、
        // 許すと読む側が先に落ちる。
        #expect(!numberBody.contains("eE"))

        // 上限は方言側の制約であって JSON Schema の制約ではない。
        // `.json` は参照実装のままの幅を出す。
        let json = JSONSchemaGrammar.grammar(
            for: try JSONDecoder().decode(JSONValue.self, from: Data(Self.schema.utf8)),
            dialect: .json)
        #expect(JSONSchemaGrammarFixture.body(of: "decimal-part", in: json.grammar)
                == "[0-9]{1,16}")
        #expect(JSONSchemaGrammarFixture.body(of: "integral-part", in: json.grammar)
                == "[0] | [1-9] [0-9]{0,15}")
    }
}
