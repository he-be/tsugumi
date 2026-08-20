import Foundation

// GBNF 文法との逐次照合。
//
// 規範は参照実装 `~/LLM/llama.cpp` のピン `34af94cd9`、`src/llama-grammar.cpp` の
// `llama_grammar_*` 群 (SPEC §0 の優先順位 2)。SPEC §6 GEN-1 が名指しする
// 「文法で拘束して生成する」機構のうち、照合とマスクを担う部分。
//
// 現状は入口だけ。中身は未実装。

/// 規則要素の位置。参照実装の `const llama_grammar_element *` に対応する。
public struct GrammarPosition: Hashable, Sendable {
    public var rule: Int32
    public var element: Int32

    @inline(__always)
    init(_ rule: Int, _ element: Int) {
        self.rule = Int32(rule)
        self.element = Int32(element)
    }
}

/// 途中まで読んだ UTF-8 列。`remaining == -1` は「壊れた列」。
public struct GrammarPartialUTF8: Equatable, Sendable {
    public var value: UInt32
    public var remaining: Int

    public init(value: UInt32 = 0, remaining: Int = 0) {
        self.value = value
        self.remaining = remaining
    }
}

/// マスク対象の候補 1 個。`index` は呼び出し側の logits 配列上の位置。
public struct GrammarCandidate: Sendable {
    public var index: Int
    public var tokenID: Int32
    public var piece: [UInt8]

    public init(index: Int, tokenID: Int32, piece: [UInt8]) {
        self.index = index
        self.tokenID = tokenID
        self.piece = piece
    }
}

/// 文法との照合状態。値型で、tokenizer や Metal への参照は持たない。
public struct GrammarMatcher: Sendable {
    public let grammar: GBNFGrammar
    public private(set) var stacks: [[GrammarPosition]]
    public private(set) var partialUTF8: GrammarPartialUTF8

    public init(_ grammar: GBNFGrammar) throws {
        _ = grammar
        throw GBNFError.notImplemented
    }

    /// 空のスタックがある = ここで生成を終えてよい。
    public var isComplete: Bool { false }

    /// スタックが 1 本も無い = もう何も続けられない。
    public var isStuck: Bool { true }

    public mutating func accept(codePoint: UInt32) {
        _ = codePoint
    }

    public mutating func accept(bytes: [UInt8]) throws {
        _ = bytes
        throw GBNFError.notImplemented
    }

    public mutating func accept(text: String) throws {
        try accept(bytes: Array(text.utf8))
    }

    public mutating func accept(piece: [UInt8], tokenID: Int32) throws {
        _ = piece
        _ = tokenID
        throw GBNFError.notImplemented
    }

    /// この候補 1 個が文法を生かしたままにできるか。
    public func allows(piece: [UInt8], tokenID: Int32) -> Bool {
        _ = piece
        _ = tokenID
        return false
    }

    /// 候補集合のうち、文法が拒む候補の `index` を返す。
    public func rejectedIndices(_ candidates: [GrammarCandidate]) -> [Int] {
        candidates.map(\.index)
    }

    /// `candidates` の並び順に沿った可否マスク (`true` = 許す)。
    public func allowedMask(_ candidates: [GrammarCandidate]) -> [Bool] {
        [Bool](repeating: false, count: candidates.count)
    }
}
