import Foundation
import Testing
@testable import Tsugumi

/// SPEC §6 **GEN-2** — スキーマの入口検査で 400 にしない。
///
/// 参照実装 `common/json-schema-to-grammar.cpp` (ピン `34af94cd9`) が
/// `_errors` に積んで `check_errors()` で `std::invalid_argument` を投げる形は
/// 全部ここに並べてある。GEN-2 はこれを**クライアントに見える誤り**にすること
/// を禁じるので、こちらでは拘束できるいちばん近いものへ落とし、落ちた事実を
/// `approximations` に載せる。
///
/// 落とし先は次の 4 つだけ:
/// - 何であるか分からないスキーマ / 解決できない `$ref` → `value`
/// - 壊れた `pattern` → `string`
/// - 整数でない境界 → `integer` (型どおりの原始規則)
/// - 表現できないキー (`.gemmaToolArguments`) → 文法はそのまま、記録だけ
@Suite("JSONSchemaGrammar GEN-2")
struct JSONSchemaGrammarApproximationTests {
    struct ApproximationCase: Sendable, CustomTestStringConvertible {
        let name: String
        let schema: String
        let dialect: JSONSchemaGrammarDialect
        let approximations: [String]
        /// 落とし先の規則本体。`root` がこれに落ちていることを見る。
        let rootRule: String
        /// 参照実装が例外を投げていた形かどうか (厳密モードの期待)。
        let referenceRejects: Bool

        var testDescription: String { name }
    }

    @Test("表現できないスキーマ要素は 400 ではなく近似になる",
          arguments: JSONSchemaGrammarApproximationTests.cases)
    func degradesInsteadOfRejecting(_ testCase: ApproximationCase) throws {
        let result = try JSONSchemaGrammarFixture.convert(
            testCase.schema, dialect: testCase.dialect)
        #expect(result.approximations == testCase.approximations)
        #expect(rule(named: "root", in: result.grammar) == testCase.rootRule)
        // 文法は常に出る。空の文法は「拘束できない」ことなので GEN-2 違反。
        #expect(!result.grammar.isEmpty)
    }

    @Test("参照実装が弾いていた形は厳密モードでだけ弾かれる",
          arguments: JSONSchemaGrammarApproximationTests.cases)
    func strictModeStillRejects(_ testCase: ApproximationCase) throws {
        let value = try JSONDecoder().decode(
            JSONValue.self, from: Data(testCase.schema.utf8))
        if testCase.referenceRejects {
            #expect(throws: JSONSchemaGrammarStrictError.self) {
                _ = try JSONSchemaGrammar.strictGrammar(for: value, dialect: testCase.dialect)
            }
        } else {
            #expect(throws: Never.self) {
                _ = try JSONSchemaGrammar.strictGrammar(for: value, dialect: testCase.dialect)
            }
        }
    }

    private func rule(named name: String, in grammar: String) -> String? {
        for line in grammar.split(separator: "\n") where line.hasPrefix(name + " ::= ") {
            return String(line.dropFirst(name.count + 5))
        }
        return nil
    }

    static let cases: [ApproximationCase] = [
        // 参照実装のテスト表にある FAILURE 2 件。
        ApproximationCase(
            name: "unknown type",
            schema: ##"{"type": "kaboom"}"##,
            dialect: .json,
            approximations: [##"unrecognized-schema: {"type":"kaboom"}"##],
            rootRule: "value",
            referenceRejects: true),
        ApproximationCase(
            name: "invalid type",
            schema: ##"{"type": 123}"##,
            dialect: .json,
            approximations: [##"unrecognized-schema: {"type":123}"##],
            rootRule: "value",
            referenceRejects: true),
        // 真偽値スキーマ。参照実装は `visit` の最後で弾く。
        ApproximationCase(
            name: "true schema",
            schema: "true",
            dialect: .json,
            approximations: ["unrecognized-schema: true"],
            rootRule: "value",
            referenceRejects: true),
        ApproximationCase(
            name: "false schema",
            schema: "false",
            dialect: .json,
            approximations: ["unrecognized-schema: false"],
            rootRule: "value",
            referenceRejects: true),
        // $ref の 3 通り。リモート取得は採らない (ローカル専用サーバーは外へ
        // HTTP を出さない — SPEC §12 DEV-4 と同じ理由) ので未対応扱い。
        ApproximationCase(
            name: "unsupported ref",
            schema: ##"{"$ref": "definitions/foo"}"##,
            dialect: .json,
            approximations: ["unsupported-ref: definitions/foo"],
            rootRule: "value",
            referenceRejects: true),
        ApproximationCase(
            name: "remote ref",
            schema: ##"{"$ref": "https://example.com/schema.json#/definitions/foo"}"##,
            dialect: .json,
            approximations: [
                "unsupported-ref: https://example.com/schema.json#/definitions/foo",
            ],
            rootRule: "value",
            referenceRejects: true),
        ApproximationCase(
            name: "unresolvable local ref",
            schema: ##"{"$ref": "#/definitions/missing", "definitions": {}}"##,
            dialect: .json,
            approximations: ["unresolvable-ref: #/definitions/missing"],
            rootRule: "value",
            referenceRejects: true),
        // pattern の破れ 3 通り。どれも素の `string` へ落ちる。
        ApproximationCase(
            name: "pattern without anchors",
            schema: ##"{"type": "string", "pattern": "abc"}"##,
            dialect: .json,
            approximations: ["unanchored-pattern: abc"],
            rootRule: "string",
            referenceRejects: true),
        ApproximationCase(
            name: "pattern with unbalanced square brackets",
            schema: ##"{"type": "string", "pattern": "^[abc$"}"##,
            dialect: .json,
            approximations: ["unbalanced-square-brackets: ^[abc$"],
            rootRule: "string",
            referenceRejects: true),
        ApproximationCase(
            name: "pattern with bad repetition count",
            schema: ##"{"type": "string", "pattern": "^a{x}$"}"##,
            dialect: .json,
            approximations: ["invalid-repetition-count: ^a{x}$"],
            rootRule: "string",
            referenceRejects: true),
        // 先読みは参照実装でも警告どまり (捨てて先へ進む)。
        ApproximationCase(
            name: "pattern with lookahead",
            schema: ##"{"type": "string", "pattern": "^(?=a)ab$"}"##,
            dialect: .json,
            approximations: ["unsupported-pattern-syntax: ^(?=a)ab$"],
            rootRule: ##""\"" ("ab") "\"""##,
            referenceRejects: false),
        // 整数でない境界。型どおりの `integer` へ落ちる。
        ApproximationCase(
            name: "non-integer minimum",
            schema: ##"{"type": "integer", "minimum": 1.5}"##,
            dialect: .json,
            approximations: ["non-integer-bound: minimum"],
            rootRule: ##"("-"? integral-part)"##,
            referenceRejects: true),
        // `.gemmaToolArguments` で書けないキー。文法は出るが記録に残る。
        ApproximationCase(
            name: "key outside the gemma key alphabet",
            schema: ##"{"type": "object", "properties": {"a b": {"type": "integer"}}, "required": ["a b"]}"##,
            dialect: .gemmaToolArguments,
            approximations: ["unrepresentable-key: a b"],
            rootRule: ##""{" a-b-kv "}""##,
            referenceRejects: false),
    ]
}
