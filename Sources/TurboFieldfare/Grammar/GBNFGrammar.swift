import Foundation

// GBNF (llama.cpp の文法記法) の解析器。
//
// 規範は参照実装 `~/LLM/llama.cpp` のピン `34af94cd9`、`src/llama-grammar.cpp` の
// `llama_grammar_parser` である (SPEC §0 の優先順位 2)。SPEC §6 GEN-1 が言う
// 「JSON schema → 文法で拘束して生成する」機構の、文法側の土台にあたる。
//
// 現状は入口だけ。中身は未実装。

/// 文法規則を構成する要素の種類。参照実装の `llama_gretype` と 1:1。
public enum GBNFElementType: UInt8, Sendable, Hashable {
    case end = 0
    case alt = 1
    case ruleRef = 2
    case char = 3
    case charNot = 4
    case charRangeUpper = 5
    case charAlt = 6
    case charAny = 7
    case token = 8
    case tokenNot = 9
}

/// 規則要素 1 個。`value` はコードポイント・規則番号・トークン ID のいずれか。
public struct GBNFElement: Sendable, Hashable {
    public var type: GBNFElementType
    public var value: UInt32

    public init(_ type: GBNFElementType, _ value: UInt32 = 0) {
        self.type = type
        self.value = value
    }
}

/// 文法の解析・照合で投げる誤り。入力由来の失敗はすべてここを通る。
public enum GBNFError: Error, Equatable, Sendable {
    case notImplemented
    case unexpectedEndOfInput(offset: Int)
    case expectedName(offset: Int)
    case expectedInteger(offset: Int)
    case expectedHexDigits(count: Int, offset: Int)
    case unknownEscape(offset: Int)
    case expected(String, offset: Int)
    case expectedNewlineOrEnd(offset: Int)
    case repetitionWithoutPrecedingItem(offset: Int)
    case repetitionRangeReversed(min: UInt64, max: UInt64, offset: Int)
    case repetitionCountTooLarge(count: UInt64, offset: Int)
    case repetitionExpansionTooLarge(offset: Int)
    case namedTokenRequiresVocabulary(offset: Int)
    case undefinedRule(name: String)
    case missingRootRule(name: String)
    case leftRecursion(rule: String)
    case emptyGrammar
    case noSurvivingStacks(piece: [UInt8], tokenID: Int32?)
}

/// 解析済みの不変な文法。
public struct GBNFGrammar: Sendable {
    /// 規則番号 → 要素列。各要素列は `alt` で区切られ、末尾は必ず `end`。
    public let rules: [[GBNFElement]]
    /// 規則名 → 規則番号。合成規則 (`root_3` など) も含む。
    public let symbolIDs: [String: Int]
    /// 開始規則の番号。
    public let rootRuleIndex: Int

    /// 繰り返しの展開で作ってよい規則数の上限。参照実装の
    /// `MAX_REPETITION_THRESHOLD` と同値。
    public static let maxRepetitionThreshold: UInt64 = 2000

    public init(_ source: String, root: String = "root") throws {
        _ = source
        _ = root
        throw GBNFError.notImplemented
    }

    public func name(ofRule index: Int) -> String { "rule_\(index)" }

    public func rule(named name: String) -> [GBNFElement]? {
        guard let id = symbolIDs[name], id < rules.count else { return nil }
        return rules[id]
    }

    /// 参照実装の `llama_grammar_parser::print` と同じ形で規則を書き出す。
    public func formatted() -> String { "" }
}
