import Foundation
import Testing
@testable import Tsugumi

/// SPEC §6 GEN-1 — **出した文法は文法として読めなければ意味がない。**
///
/// この組の他のテストは出力を**文字列として**突き合わせるだけなので、
/// 「参照実装と同じ字面だが GBNF パーサが受け付けない」欠陥をすり抜ける。
/// 実際にすり抜けた: `key-char ::= [0-9A-Za-z_\-.$]` の `\-` は GBNF の
/// エスケープではない (`parse_char` が知るのは
/// `\x \u \U \t \r \n \\ \" \[ \]` だけ) ので、`key` 規則に届く
/// `.gemmaToolArguments` の文法は軒並み `unknownEscape` で読めなかった。
///
/// よってここでは**この組が出す文法を全部**、両方の方言で `GBNFGrammar` に
/// 通す。表が増えれば自動でここも増える。
@Suite("JSONSchemaGrammar は読める文法を出す (GEN-1)")
struct JSONSchemaGrammarParseTests {
    struct ParseCase: Sendable, CustomTestStringConvertible {
        let name: String
        let schema: String
        let dialect: JSONSchemaGrammarDialect

        var testDescription: String {
            "\(name) [\(dialect == .json ? "json" : "gemma")]"
        }
    }

    /// この組のどの表にも載っているスキーマを、**両方の方言**で。
    static let all: [ParseCase] = {
        var schemas: [(String, String)] = []
        for testCase in JSONSchemaGrammarCoreCases.all {
            schemas.append((testCase.name, testCase.schema))
        }
        for testCase in JSONSchemaGrammarPatternCases.all {
            schemas.append((testCase.name, testCase.schema))
        }
        for testCase in JSONSchemaGrammarIntegerBoundsCases.all {
            schemas.append((testCase.name, testCase.schema))
        }
        for testCase in JSONSchemaGrammarGemmaCases.all {
            schemas.append((testCase.name, testCase.schema))
        }
        for testCase in JSONSchemaGrammarGemmaCases.unrepresentable {
            schemas.append((testCase.name, testCase.schema))
        }
        for testCase in JSONSchemaGrammarApproximationTests.cases {
            schemas.append((testCase.name, testCase.schema))
        }
        return schemas.flatMap { name, schema in
            [ParseCase(name: name, schema: schema, dialect: .json),
             ParseCase(name: name, schema: schema, dialect: .gemmaToolArguments)]
        }
    }()

    @Test("出した文法は GBNF パーサが読める", arguments: JSONSchemaGrammarParseTests.all)
    func emittedGrammarParses(_ testCase: ParseCase) throws {
        let result = try JSONSchemaGrammarFixture.convert(
            testCase.schema, dialect: testCase.dialect)
        // 投げれば、そのままテストの失敗として理由が出る。
        _ = try GBNFGrammar(result.grammar, root: "root")
    }

    /// 正準形の表 (値を `const` で固定したスキーマ) も同じく読めること。
    @Test("正準形の表が出す文法も読める",
          arguments: JSONSchemaGrammarGemmaCases.canonical)
    func canonicalGrammarParses(
        _ testCase: JSONSchemaGrammarGemmaCases.CanonicalCase
    ) throws {
        let arguments = try JSONDecoder().decode(
            JSONValue.self, from: Data(testCase.arguments.utf8))
        let schema = JSONSchemaGrammarGemmaCases.constSchema(arguments)
        for dialect in [JSONSchemaGrammarDialect.json, .gemmaToolArguments] {
            let result = JSONSchemaGrammar.grammar(for: schema, dialect: dialect)
            _ = try GBNFGrammar(result.grammar, root: "root")
        }
    }

    /// 欠陥そのものの回帰。文字クラスに入る `-` は、パーサが受けるエスケープで
    /// 綴られていなければならない (`\-` は GBNF に無い)。
    @Test("文字クラスの - はパーサが受ける形で綴られる")
    func hyphenIsSpelledWithAnAcceptedEscape() throws {
        let schema = ##"{"type": "object", "additionalProperties": true}"##
        let result = try JSONSchemaGrammarFixture.convert(
            schema, dialect: .gemmaToolArguments)
        #expect(result.grammar.contains("key-char ::="))
        #expect(!result.grammar.contains(##"\-"##))
        _ = try GBNFGrammar(result.grammar, root: "root")
    }
}
