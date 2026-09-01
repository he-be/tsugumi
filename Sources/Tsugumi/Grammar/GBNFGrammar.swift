// GBNF (llama.cpp の文法記法) の解析器。
//
// 規範は参照実装 `~/LLM/llama.cpp` のピン `34af94cd9`、`src/llama-grammar.cpp` の
// `llama_grammar_parser` である (SPEC §0 の優先順位 2)。SPEC §6 GEN-1 が言う
// 「JSON schema → 文法で拘束して生成する」機構の、文法側の土台にあたる。
//
// 移植にあたって意図的に変えた点は 2 つだけ:
//
//  1. 規則要素の位置を「ポインタ」ではなく `(規則番号, 要素番号)` の組で持つ
//     (`GrammarPosition`)。値型で複製できるようになり、参照実装の `clone` に
//     あるポインタ張り替えが要らなくなる。
//  2. 入力由来の失敗はすべて `GBNFError` を投げる。参照実装のように stderr へ
//     書いて `nullptr` / `false` を返したり、`GGML_ABORT` したりしない。

/// 文法規則を構成する要素の種類。参照実装の `llama_gretype` と 1:1。
public enum GBNFElementType: UInt8, Sendable, Hashable {
    /// 規則定義の終端。
    case end = 0
    /// 規則の別選択肢の区切り。
    case alt = 1
    /// 非終端: 他の規則への参照。
    case ruleRef = 2
    /// 終端: 文字 (コードポイント)。
    case char = 3
    /// 否定文字集合 (`[^a]`, `[^a-b]`, `[^abc]`)。
    case charNot = 4
    /// 直前の `char` / `charAlt` を包含範囲に変える (`[a-z]`)。
    case charRangeUpper = 5
    /// 直前の `char` / `charRangeUpper` に別候補を足す (`[ab]`, `[a-zA]`)。
    case charAlt = 6
    /// 任意の 1 文字 (`.`)。
    case charAny = 7
    /// 終端: トークン (`<[42]>`)。
    case token = 8
    /// 否定トークン (`!<[42]>`)。
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
public enum GBNFError: Error, Equatable, Sendable, CustomStringConvertible {
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

    public var description: String {
        switch self {
        case let .unexpectedEndOfInput(offset):
            return "GBNF: 入力が途中で終わった (offset \(offset))"
        case let .expectedName(offset):
            return "GBNF: 規則名が必要 (offset \(offset))"
        case let .expectedInteger(offset):
            return "GBNF: 整数が必要 (offset \(offset))"
        case let .expectedHexDigits(count, offset):
            return "GBNF: 16 進 \(count) 桁が必要 (offset \(offset))"
        case let .unknownEscape(offset):
            return "GBNF: 未知のエスケープ (offset \(offset))"
        case let .expected(what, offset):
            return "GBNF: '\(what)' が必要 (offset \(offset))"
        case let .expectedNewlineOrEnd(offset):
            return "GBNF: 改行か入力終端が必要 (offset \(offset))"
        case let .repetitionWithoutPrecedingItem(offset):
            return "GBNF: */+/?/{ の前に項目が無い (offset \(offset))"
        case let .repetitionRangeReversed(min, max, offset):
            return "GBNF: 繰り返し範囲が逆 {\(min),\(max)} (offset \(offset))"
        case let .repetitionCountTooLarge(count, offset):
            return "GBNF: 繰り返し回数 \(count) が上限を超えた (offset \(offset))"
        case let .repetitionExpansionTooLarge(offset):
            return "GBNF: 繰り返しで生成される規則数が上限を超えた (offset \(offset))"
        case let .namedTokenRequiresVocabulary(offset):
            return "GBNF: <名前> 形のトークンは語彙が要る。<[ID]> を使う (offset \(offset))"
        case let .undefinedRule(name):
            return "GBNF: 未定義の規則 '\(name)'"
        case let .missingRootRule(name):
            return "GBNF: 開始規則 '\(name)' が無い"
        case let .leftRecursion(rule):
            return "GBNF: 左再帰の規則 '\(rule)' は扱えない"
        case .emptyGrammar:
            return "GBNF: 規則が 1 つも無い"
        case let .noSurvivingStacks(piece, tokenID):
            let text = String(decoding: piece, as: UTF8.self)
            if let tokenID {
                return "GBNF: piece '\(text)' (token \(tokenID)) を受理したら候補スタックが残らなかった"
            }
            return "GBNF: piece '\(text)' を受理したら候補スタックが残らなかった"
        }
    }
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
        var parser = GBNFParser(source: source)
        try parser.parse()

        guard !parser.rules.isEmpty else { throw GBNFError.emptyGrammar }
        guard let rootID = parser.symbolIDs[root] else {
            throw GBNFError.missingRootRule(name: root)
        }

        self.rules = parser.rules
        self.symbolIDs = parser.symbolIDs
        self.rootRuleIndex = rootID

        try GBNFGrammar.rejectLeftRecursion(rules: parser.rules, names: parser.symbolIDs)
    }

    /// 規則番号 → 名前 (誤り表示と検査用)。
    public func name(ofRule index: Int) -> String {
        for (name, id) in symbolIDs where id == index { return name }
        return "rule_\(index)"
    }

    // MARK: - 左再帰の検出

    /// 参照実装の `llama_grammar_detect_left_recursion` の移植。左再帰の文法は
    /// 構築時に落とす (異常終了ではなく誤り)。
    static func rejectLeftRecursion(rules: [[GBNFElement]], names: [String: Int]) throws {
        let count = rules.count
        var visited = [Bool](repeating: false, count: count)
        var inProgress = [Bool](repeating: false, count: count)
        var mayBeEmpty = [Bool](repeating: false, count: count)

        func nameOf(_ index: Int) -> String {
            for (name, id) in names where id == index { return name }
            return "rule_\(index)"
        }

        func detect(_ index: Int) -> Int? {
            if inProgress[index] { return index }
            inProgress[index] = true

            let rule = rules[index]

            // まず空列を生みうるかを見る。次の段と混ぜられるが 2 段のほうが読める。
            var atRuleStart = true
            for element in rule {
                if element.type.isEndOfSequence {
                    if atRuleStart {
                        mayBeEmpty[index] = true
                        break
                    }
                    atRuleStart = true
                } else {
                    atRuleStart = false
                }
            }

            // 次に最左の非終端 (直前の非終端が空になりうる間は次の非終端も) へ降りる。
            var recurse = true
            for element in rule {
                if element.type == .ruleRef, recurse {
                    if let found = detect(Int(element.value)) { return found }
                    if !mayBeEmpty[Int(element.value)] { recurse = false }
                } else if element.type.isEndOfSequence {
                    recurse = true
                } else {
                    recurse = false
                }
            }

            inProgress[index] = false
            visited[index] = true
            return nil
        }

        for index in 0..<count {
            if visited[index] { continue }
            if let found = detect(index) {
                throw GBNFError.leftRecursion(rule: nameOf(found))
            }
        }
    }
}

extension GBNFElementType {
    /// 参照実装の `llama_grammar_is_end_of_sequence`。
    var isEndOfSequence: Bool { self == .end || self == .alt }

    var isTerminal: Bool {
        switch self {
        case .char, .charNot, .charAny, .token, .tokenNot: return true
        default: return false
        }
    }
}

// MARK: - 解析器

/// 参照実装の `llama_grammar_parser` の移植。
///
/// 参照実装は NUL 終端の `const char *` を歩く。ここでもバイト列を歩き、範囲外は
/// 0 を返す `byte(_:)` で同じ「終端で 0 が読める」性質を再現する。したがって
/// 文法本文に NUL が混ざれば、参照実装と同じくそこで入力が終わる。
struct GBNFParser {
    private let bytes: [UInt8]
    private(set) var pos: Int = 0
    private(set) var symbolIDs: [String: Int] = [:]
    private(set) var rules: [[GBNFElement]] = []

    init(source: String) {
        self.bytes = Array(source.utf8)
    }

    @inline(__always)
    private func byte(_ index: Int) -> UInt8 {
        index >= 0 && index < bytes.count ? bytes[index] : 0
    }

    @inline(__always)
    private var current: UInt8 { byte(pos) }

    // MARK: 記号表

    private mutating func symbolID(for name: String) -> Int {
        if let existing = symbolIDs[name] { return existing }
        let next = symbolIDs.count
        symbolIDs[name] = next
        return next
    }

    private mutating func generateSymbolID(base: String) -> Int {
        let next = symbolIDs.count
        symbolIDs["\(base)_\(next)"] = next
        return next
    }

    private mutating func addRule(_ id: Int, _ rule: [GBNFElement]) {
        if rules.count <= id {
            rules.append(contentsOf: [[GBNFElement]](repeating: [], count: id + 1 - rules.count))
        }
        rules[id] = rule
    }

    // MARK: 字句

    private static func isDigit(_ c: UInt8) -> Bool { 0x30 <= c && c <= 0x39 }

    private static func isWordChar(_ c: UInt8) -> Bool {
        (0x61...0x7A).contains(c) || (0x41...0x5A).contains(c) || c == 0x2D || isDigit(c)
    }

    /// 空白と `#` 行コメントを読み飛ばす。`newlineOK` が偽なら改行で止まる
    /// (最上位の規則本体は改行で終わるため)。
    private mutating func skipSpace(newlineOK: Bool) {
        while true {
            let c = current
            if c == 0x20 || c == 0x09 || (newlineOK && (c == 0x0D || c == 0x0A)) {
                pos += 1
            } else if c == 0x23 { // '#'
                while current != 0, current != 0x0D, current != 0x0A { pos += 1 }
            } else {
                break
            }
        }
    }

    private mutating func scanName() throws -> String {
        let start = pos
        while GBNFParser.isWordChar(current) { pos += 1 }
        guard pos != start else { throw GBNFError.expectedName(offset: start) }
        return String(decoding: bytes[start..<pos], as: UTF8.self)
    }

    private mutating func scanInt() throws -> UInt64 {
        let start = pos
        while GBNFParser.isDigit(current) { pos += 1 }
        guard pos != start else { throw GBNFError.expectedInteger(offset: start) }
        // 桁数が多すぎて UInt64 に入らない場合も「大きすぎる」として扱う。
        guard let value = UInt64(String(decoding: bytes[start..<pos], as: UTF8.self)) else {
            throw GBNFError.repetitionCountTooLarge(count: UInt64.max, offset: start)
        }
        return value
    }

    private mutating func scanHex(_ size: Int) throws -> UInt32 {
        let start = pos
        let end = pos + size
        var value: UInt32 = 0
        while pos < end, current != 0 {
            value <<= 4
            let c = current
            if 0x61...0x66 ~= c {
                value += UInt32(c - 0x61 + 10)
            } else if 0x41...0x46 ~= c {
                value += UInt32(c - 0x41 + 10)
            } else if GBNFParser.isDigit(c) {
                value += UInt32(c - 0x30)
            } else {
                break
            }
            pos += 1
        }
        guard pos == end else { throw GBNFError.expectedHexDigits(count: size, offset: start) }
        return value
    }

    /// 参照実装の 1 引数版 `decode_utf8`。妥当な UTF-8 を仮定し、行き過ぎだけ見る。
    private mutating func scanUTF8Scalar() -> UInt32 {
        let lookup = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4]
        let first = current
        let length = lookup[Int(first >> 4)]
        let mask = UInt8((1 << (8 - length)) - 1)
        var value = UInt32(first & mask)
        let end = pos + length
        var p = pos + 1
        while p < end, byte(p) != 0 {
            value = (value << 6) + UInt32(byte(p) & 0x3F)
            p += 1
        }
        pos = p
        return value
    }

    /// 参照実装の `parse_char`。対応するエスケープはここに並ぶものだけ。
    private mutating func scanChar() throws -> UInt32 {
        if current == 0x5C { // '\'
            let escapeStart = pos
            switch byte(pos + 1) {
            case 0x78: pos += 2; return try scanHex(2)               // \x
            case 0x75: pos += 2; return try scanHex(4)               // \u
            case 0x55: pos += 2; return try scanHex(8)               // \U
            case 0x74: pos += 2; return 0x09                         // \t
            case 0x72: pos += 2; return 0x0D                         // \r
            case 0x6E: pos += 2; return 0x0A                         // \n
            case 0x5C, 0x22, 0x5B, 0x5D:                             // \\ \" \[ \]
                let value = UInt32(byte(pos + 1))
                pos += 2
                return value
            default:
                throw GBNFError.unknownEscape(offset: escapeStart)
            }
        } else if current != 0 {
            return scanUTF8Scalar()
        }
        throw GBNFError.unexpectedEndOfInput(offset: pos)
    }

    /// 参照実装の `parse_token`。語彙を持たないので `<[ID]>` 形だけを受ける。
    private mutating func scanToken() throws -> UInt32 {
        guard current == 0x3C else { throw GBNFError.expected("<", offset: pos) } // '<'
        let start = pos
        pos += 1
        guard current == 0x5B else { throw GBNFError.namedTokenRequiresVocabulary(offset: start) } // '['
        pos += 1
        let id = try scanInt()
        guard current == 0x5D else { throw GBNFError.expected("]", offset: pos) }
        pos += 1
        guard current == 0x3E else { throw GBNFError.expected(">", offset: pos) }
        pos += 1
        guard id <= UInt64(UInt32.max) else { throw GBNFError.expectedInteger(offset: start) }
        return UInt32(id)
    }

    // MARK: 構文

    mutating func parse() throws {
        skipSpace(newlineOK: true)
        while current != 0 {
            try parseRule()
        }

        for index in rules.indices where rules[index].isEmpty {
            throw GBNFError.undefinedRule(name: nameOfRule(index))
        }
        for rule in rules {
            for element in rule where element.type == .ruleRef {
                let target = Int(element.value)
                if target >= rules.count || rules[target].isEmpty {
                    throw GBNFError.undefinedRule(name: nameOfRule(target))
                }
            }
        }
    }

    private func nameOfRule(_ index: Int) -> String {
        for (name, id) in symbolIDs where id == index { return name }
        return "rule_\(index)"
    }

    private mutating func parseRule() throws {
        let name = try scanName()
        skipSpace(newlineOK: false)
        let ruleID = symbolID(for: name)

        guard current == 0x3A, byte(pos + 1) == 0x3A, byte(pos + 2) == 0x3D else {
            throw GBNFError.expected("::=", offset: pos)
        }
        pos += 3
        skipSpace(newlineOK: true)

        try parseAlternates(ruleName: name, ruleID: ruleID, isNested: false)

        if current == 0x0D {
            pos += byte(pos + 1) == 0x0A ? 2 : 1
        } else if current == 0x0A {
            pos += 1
        } else if current != 0 {
            throw GBNFError.expectedNewlineOrEnd(offset: pos)
        }
        skipSpace(newlineOK: true)
    }

    private mutating func parseAlternates(ruleName: String, ruleID: Int, isNested: Bool) throws {
        var rule: [GBNFElement] = []
        try parseSequence(into: &rule, ruleName: ruleName, isNested: isNested)
        while current == 0x7C { // '|'
            rule.append(GBNFElement(.alt))
            pos += 1
            skipSpace(newlineOK: true)
            try parseSequence(into: &rule, ruleName: ruleName, isNested: isNested)
        }
        rule.append(GBNFElement(.end))
        addRule(ruleID, rule)
    }

    private mutating func parseSequence(
        into rule: inout [GBNFElement],
        ruleName: String,
        isNested: Bool
    ) throws {
        var lastSymStart = rule.count
        var previousRuleCount: UInt64 = 1

        /// 参照実装の `handle_repetitions`。`maxTimes == UInt64.max` は上限無し。
        func handleRepetitions(_ minTimes: UInt64, _ maxTimes: UInt64, at offset: Int) throws {
            let noMax = maxTimes == UInt64.max
            guard lastSymStart != rule.count else {
                throw GBNFError.repetitionWithoutPrecedingItem(offset: offset)
            }
            // 逸脱: 参照実装は {3,1} で符号なし減算が回り込み、事実上ハングする。
            // ここでは入力由来の誤りとして落とす。
            if !noMax, maxTimes < minTimes {
                throw GBNFError.repetitionRangeReversed(min: minTimes, max: maxTimes, offset: offset)
            }

            let previous = Array(rule[lastSymStart...])

            var totalRules: UInt64 = 1
            if !noMax, maxTimes > 0 {
                totalRules = maxTimes
            } else if minTimes > 0 {
                totalRules = minTimes
            }
            let product = previousRuleCount.multipliedReportingOverflow(by: totalRules)
            guard !product.overflow, product.partialValue < GBNFGrammar.maxRepetitionThreshold else {
                throw GBNFError.repetitionExpansionTooLarge(offset: offset)
            }

            if minTimes == 0 {
                rule.removeSubrange(lastSymStart...)
            } else if minTimes > 1 {
                for _ in 1..<minTimes { rule.append(contentsOf: previous) }
            }

            var lastRecursiveRuleID = 0
            let optionalCount: UInt64 = noMax ? 1 : maxTimes - minTimes

            var recursive = previous
            var i: UInt64 = 0
            while i < optionalCount {
                recursive.removeSubrange(previous.count...)
                let recursiveRuleID = generateSymbolID(base: ruleName)
                if i > 0 || noMax {
                    recursive.append(
                        GBNFElement(.ruleRef, UInt32(noMax ? recursiveRuleID : lastRecursiveRuleID))
                    )
                }
                recursive.append(GBNFElement(.alt))
                recursive.append(GBNFElement(.end))
                addRule(recursiveRuleID, recursive)
                lastRecursiveRuleID = recursiveRuleID
                i += 1
            }
            if optionalCount > 0 {
                rule.append(GBNFElement(.ruleRef, UInt32(lastRecursiveRuleID)))
            }
            previousRuleCount = product.partialValue
        }

        while current != 0 {
            if current == 0x22 { // '"' 文字列リテラル
                pos += 1
                lastSymStart = rule.count
                previousRuleCount = 1
                while current != 0x22 {
                    if current == 0 { throw GBNFError.unexpectedEndOfInput(offset: pos) }
                    rule.append(GBNFElement(.char, try scanChar()))
                }
                pos += 1
                skipSpace(newlineOK: isNested)
            } else if current == 0x5B { // '[' 文字集合
                pos += 1
                var startType = GBNFElementType.char
                if current == 0x5E { // '^'
                    pos += 1
                    startType = .charNot
                }
                lastSymStart = rule.count
                previousRuleCount = 1
                while current != 0x5D { // ']'
                    if current == 0 { throw GBNFError.unexpectedEndOfInput(offset: pos) }
                    let value = try scanChar()
                    let type: GBNFElementType = lastSymStart < rule.count ? .charAlt : startType
                    rule.append(GBNFElement(type, value))
                    if current == 0x2D, byte(pos + 1) != 0x5D { // '-' で範囲
                        if byte(pos + 1) == 0 { throw GBNFError.unexpectedEndOfInput(offset: pos) }
                        pos += 1
                        rule.append(GBNFElement(.charRangeUpper, try scanChar()))
                    }
                }
                pos += 1
                skipSpace(newlineOK: isNested)
            } else if current == 0x3C || current == 0x21 { // '<' / '!' トークン
                var type = GBNFElementType.token
                if current == 0x21 {
                    type = .tokenNot
                    pos += 1
                }
                let id = try scanToken()
                lastSymStart = rule.count
                previousRuleCount = 1
                rule.append(GBNFElement(type, id))
                skipSpace(newlineOK: isNested)
            } else if GBNFParser.isWordChar(current) { // 規則参照
                let name = try scanName()
                let refID = symbolID(for: name)
                skipSpace(newlineOK: isNested)
                lastSymStart = rule.count
                previousRuleCount = 1
                rule.append(GBNFElement(.ruleRef, UInt32(refID)))
            } else if current == 0x28 { // '(' 群
                pos += 1
                skipSpace(newlineOK: true)
                let symbolsBefore = symbolIDs.count
                let subRuleID = generateSymbolID(base: ruleName)
                try parseAlternates(ruleName: ruleName, ruleID: subRuleID, isNested: true)
                previousRuleCount = UInt64(max(1, symbolIDs.count - symbolsBefore))
                lastSymStart = rule.count
                rule.append(GBNFElement(.ruleRef, UInt32(subRuleID)))
                guard current == 0x29 else { throw GBNFError.expected(")", offset: pos) }
                pos += 1
                skipSpace(newlineOK: isNested)
            } else if current == 0x2E { // '.'
                lastSymStart = rule.count
                previousRuleCount = 1
                rule.append(GBNFElement(.charAny))
                pos += 1
                skipSpace(newlineOK: isNested)
            } else if current == 0x2A { // '*'
                let offset = pos
                pos += 1
                skipSpace(newlineOK: isNested)
                try handleRepetitions(0, UInt64.max, at: offset)
            } else if current == 0x2B { // '+'
                let offset = pos
                pos += 1
                skipSpace(newlineOK: isNested)
                try handleRepetitions(1, UInt64.max, at: offset)
            } else if current == 0x3F { // '?'
                let offset = pos
                pos += 1
                skipSpace(newlineOK: isNested)
                try handleRepetitions(0, 1, at: offset)
            } else if current == 0x7B { // '{'
                let offset = pos
                pos += 1
                skipSpace(newlineOK: isNested)

                guard GBNFParser.isDigit(current) else {
                    throw GBNFError.expectedInteger(offset: pos)
                }
                let minTimes = try scanInt()
                skipSpace(newlineOK: isNested)

                var maxTimes = UInt64.max
                if current == 0x7D { // '}'
                    maxTimes = minTimes
                    pos += 1
                    skipSpace(newlineOK: isNested)
                } else if current == 0x2C { // ','
                    pos += 1
                    skipSpace(newlineOK: isNested)
                    if GBNFParser.isDigit(current) {
                        maxTimes = try scanInt()
                        skipSpace(newlineOK: isNested)
                    }
                    guard current == 0x7D else { throw GBNFError.expected("}", offset: pos) }
                    pos += 1
                    skipSpace(newlineOK: isNested)
                } else {
                    throw GBNFError.expected(",", offset: pos)
                }

                if minTimes > GBNFGrammar.maxRepetitionThreshold {
                    throw GBNFError.repetitionCountTooLarge(count: minTimes, offset: offset)
                }
                if maxTimes != UInt64.max, maxTimes > GBNFGrammar.maxRepetitionThreshold {
                    maxTimes = UInt64.max
                }
                try handleRepetitions(minTimes, maxTimes, at: offset)
            } else {
                break
            }
        }
    }
}
