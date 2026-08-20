import Testing

import TurboFieldfare

// SPEC §6 GEN-1 が名指しする「文法で拘束して生成する」機構の土台 (P2)。
// 規範は参照実装 `~/LLM/llama.cpp` のピン `34af94cd9` の `src/llama-grammar.cpp`。
// ここは C0 相当 — 重みも Metal も tokenizer も使わない。

// MARK: - 補助

private func makeMatcher(_ source: String, root: String = "root") throws -> GrammarMatcher {
    try GrammarMatcher(GBNFGrammar(source, root: root))
}

/// テキストを丸ごと流して、文法がそこで完結できるかを返す。
private func fullyMatches(_ source: String, root: String = "root", _ text: String) -> Bool {
    do {
        var matcher = try makeMatcher(source, root: root)
        try matcher.accept(text: text)
        return matcher.isComplete
    } catch {
        return false
    }
}

/// 解析で投げられた誤りを取り出す。成功したら nil。
private func parseError(_ source: String, root: String = "root") -> GBNFError? {
    do {
        _ = try GBNFGrammar(source, root: root)
        return nil
    } catch let error as GBNFError {
        return error
    } catch {
        return nil
    }
}

private func bytes(_ text: String) -> [UInt8] { Array(text.utf8) }

private func didThrow(_ body: () throws -> Void) -> Bool {
    do {
        try body()
        return false
    } catch {
        return true
    }
}

struct GrammarMatchCase: Sendable, CustomStringConvertible {
    let name: String
    let grammar: String
    let accepted: [String]
    let rejected: [String]

    var description: String { name }
}

let grammarMatchCases: [GrammarMatchCase] = [
    GrammarMatchCase(
        name: "文字列リテラル",
        grammar: #"root ::= "abc""#,
        accepted: ["abc"],
        rejected: ["", "ab", "abcd", "abd", "ABC"]
    ),
    GrammarMatchCase(
        name: "文字範囲",
        grammar: "root ::= [a-z]",
        accepted: ["a", "q", "z"],
        rejected: ["", "A", "0", "ab"]
    ),
    GrammarMatchCase(
        name: "否定文字集合",
        grammar: "root ::= [^a-z]+",
        accepted: ["A", "0", "AB", "-"],
        rejected: ["", "a", "Ab"]
    ),
    GrammarMatchCase(
        name: "末尾のハイフンは範囲でなく文字",
        grammar: "root ::= [a-]",
        accepted: ["a", "-"],
        rejected: ["b", "z", "0"]
    ),
    GrammarMatchCase(
        name: "列挙の文字集合",
        grammar: "root ::= [abc]",
        accepted: ["a", "b", "c"],
        rejected: ["d", "ab"]
    ),
    GrammarMatchCase(
        name: "範囲と列挙の混在",
        grammar: "root ::= [a-fA-F0-9x]",
        accepted: ["a", "f", "A", "F", "0", "9", "x"],
        rejected: ["g", "G", "y"]
    ),
    GrammarMatchCase(
        name: "任意の 1 文字",
        grammar: "root ::= .",
        accepted: ["x", "あ", "-"],
        rejected: ["", "ab"]
    ),
    GrammarMatchCase(
        name: "選択",
        grammar: #"root ::= "a" | "bb" | "c""#,
        accepted: ["a", "bb", "c"],
        rejected: ["", "b", "ac", "bbb"]
    ),
    GrammarMatchCase(
        name: "括弧による入れ子",
        grammar: #"root ::= ("a" | "b") "c""#,
        accepted: ["ac", "bc"],
        rejected: ["c", "abc", "a"]
    ),
    GrammarMatchCase(
        name: "入れ子の中の入れ子",
        grammar: #"root ::= ("a" ("b" | "c")) "d""#,
        accepted: ["abd", "acd"],
        rejected: ["ad", "abcd"]
    ),
    GrammarMatchCase(
        name: "繰り返し *",
        grammar: #"root ::= "a"*"#,
        accepted: ["", "a", "aaaa"],
        rejected: ["b", "ab"]
    ),
    GrammarMatchCase(
        name: "繰り返し +",
        grammar: #"root ::= "a"+"#,
        accepted: ["a", "aa", "aaaaa"],
        rejected: ["", "b"]
    ),
    GrammarMatchCase(
        name: "繰り返し ?",
        grammar: #"root ::= "a"?"#,
        accepted: ["", "a"],
        rejected: ["aa", "b"]
    ),
    GrammarMatchCase(
        name: "繰り返し {m}",
        grammar: #"root ::= "a"{3}"#,
        accepted: ["aaa"],
        rejected: ["", "a", "aa", "aaaa"]
    ),
    GrammarMatchCase(
        name: "繰り返し {m,}",
        grammar: #"root ::= "a"{2,}"#,
        accepted: ["aa", "aaa", "aaaaaaa"],
        rejected: ["", "a"]
    ),
    GrammarMatchCase(
        name: "繰り返し {m,n}",
        grammar: #"root ::= "a"{1,3}"#,
        accepted: ["a", "aa", "aaa"],
        rejected: ["", "aaaa"]
    ),
    GrammarMatchCase(
        name: "繰り返し {0,n}",
        grammar: #"root ::= "a"{0,2}"#,
        accepted: ["", "a", "aa"],
        rejected: ["aaa"]
    ),
    GrammarMatchCase(
        name: "群への繰り返し",
        grammar: #"root ::= ("ab")+ "!""#,
        accepted: ["ab!", "abab!"],
        rejected: ["!", "aba!"]
    ),
    GrammarMatchCase(
        name: "規則参照 (前方参照を含む)",
        grammar: """
            root ::= head tail
            head ::= "x"
            tail ::= "y"
            """,
        accepted: ["xy"],
        rejected: ["x", "y", "xyz"]
    ),
    GrammarMatchCase(
        name: "右再帰",
        grammar: #"root ::= "a" root | "b""#,
        accepted: ["b", "ab", "aaab"],
        rejected: ["", "a", "ba"]
    ),
    GrammarMatchCase(
        name: "エスケープ全種",
        grammar: #"root ::= "\t\r\n\\\"\[\]""#,
        accepted: ["\t\r\n\\\"[]"],
        rejected: ["", "\t"]
    ),
    GrammarMatchCase(
        name: "16 進とユニコードのエスケープ",
        grammar: #"root ::= [\x41B] "\U0001F600""#,
        accepted: ["A\u{1F600}", "B\u{1F600}"],
        rejected: ["C\u{1F600}", "A"]
    ),
    GrammarMatchCase(
        name: "コメントと行末",
        grammar: """
            # 先頭のコメント
            root ::= "a" body  # 行末のコメント
            body ::= "b"
            """,
        accepted: ["ab"],
        rejected: ["a", "b"]
    ),
    GrammarMatchCase(
        name: "| の後の改行は自由",
        grammar: """
            root ::= "a" |
              "b"
            """,
        accepted: ["a", "b"],
        rejected: ["", "ab"]
    ),
    GrammarMatchCase(
        name: "括弧の中は改行が自由",
        grammar: """
            root ::= (
              "a"
              | "b"
              )
            """,
        accepted: ["a", "b"],
        rejected: ["", "ab"]
    ),
]

// MARK: - 表駆動の受理/拒否

@Suite struct GBNFMatchTableTests {
    @Test(arguments: grammarMatchCases)
    func acceptsAndRejects(_ testCase: GrammarMatchCase) throws {
        for text in testCase.accepted {
            #expect(
                fullyMatches(testCase.grammar, text),
                "\(testCase.name): '\(text)' は受理されるはず"
            )
        }
        for text in testCase.rejected {
            #expect(
                !fullyMatches(testCase.grammar, text),
                "\(testCase.name): '\(text)' は拒否されるはず"
            )
        }
    }
}

// MARK: - 解析器の誤り

@Suite struct GBNFParserErrorTests {
    @Test func unknownEscapeIsRejected() {
        guard case .unknownEscape? = parseError(#"root ::= "\q""#) else {
            Issue.record("未知のエスケープは誤りになるはず")
            return
        }
    }

    @Test func supportedEscapesAreAcceptedOneByOne() throws {
        let literals = [
            #""\t""#, #""\r""#, #""\n""#, #""\\""#, #""\"""#, #""\[""#, #""\]""#,
            #""\x41""#, #""A""#, #""\U00000041""#,
        ]
        for literal in literals {
            _ = try GBNFGrammar("root ::= \(literal)")
        }
    }

    @Test func shortHexEscapeIsRejected() {
        guard case .expectedHexDigits? = parseError(#"root ::= "\x4""#) else {
            Issue.record("桁の足りない \\x は誤りになるはず")
            return
        }
    }

    @Test func newlineEndsATopLevelRuleBody() {
        // `a ::= b\n | c` は誤り: 改行で規則本体が終わり、次の行が規則名から始まらない。
        let source = """
            root ::= "a"
             | "b"
            """
        guard case .expectedName? = parseError(source) else {
            Issue.record("改行の後の '|' は規則名として読まれ、誤りになるはず")
            return
        }
    }

    @Test func undefinedRuleReferenceIsRejected() {
        #expect(parseError("root ::= missing") == .undefinedRule(name: "missing"))
    }

    @Test func missingRootIsRejected() {
        #expect(parseError(#"other ::= "a""#) == .missingRootRule(name: "root"))
    }

    @Test func nonDefaultRootIsHonoured() {
        let source = """
            root ::= "a"
            other ::= "b"
            """
        #expect(fullyMatches(source, root: "other", "b"))
        #expect(!fullyMatches(source, root: "other", "a"))
    }

    @Test func repetitionWithoutPrecedingItemIsRejected() {
        guard case .repetitionWithoutPrecedingItem? = parseError("root ::= *") else {
            Issue.record("先行する項目の無い '*' は誤りになるはず")
            return
        }
    }

    @Test func reversedRepetitionRangeIsRejected() {
        // 逸脱: 参照実装は符号なし減算が回り込んで事実上ハングする。ここは誤りにする。
        guard case .repetitionRangeReversed? = parseError(#"root ::= "a"{3,1}"#) else {
            Issue.record("{3,1} は誤りになるはず")
            return
        }
    }

    @Test func unterminatedGroupIsRejected() {
        guard case .expected? = parseError(#"root ::= ("a""#) else {
            Issue.record("閉じない '(' は誤りになるはず")
            return
        }
    }
}

// MARK: - 繰り返しの低水準化と 2000 の壁

@Suite struct GBNFRepetitionLoweringTests {
    @Test func boundedRepetitionLowersToChainedOptionalRules() throws {
        // 参照実装の handle_repetitions:
        //   root   ::= "a" "a" root_2
        //   root_1 ::= "a" |
        //   root_2 ::= "a" root_1 |
        let grammar = try GBNFGrammar(#"root ::= "a"{2,4}"#)
        let a = UInt32(UInt8(ascii: "a"))
        #expect(grammar.rules.count == 3)
        #expect(grammar.rules[0] == [
            GBNFElement(.char, a), GBNFElement(.char, a), GBNFElement(.ruleRef, 2), GBNFElement(.end),
        ])
        #expect(grammar.rules[1] == [
            GBNFElement(.char, a), GBNFElement(.alt), GBNFElement(.end),
        ])
        #expect(grammar.rules[2] == [
            GBNFElement(.char, a), GBNFElement(.ruleRef, 1), GBNFElement(.alt), GBNFElement(.end),
        ])
    }

    @Test func starLowersToASelfRecursiveRule() throws {
        let grammar = try GBNFGrammar(#"root ::= "a"*"#)
        let a = UInt32(UInt8(ascii: "a"))
        #expect(grammar.rules[0] == [GBNFElement(.ruleRef, 1), GBNFElement(.end)])
        #expect(grammar.rules[1] == [
            GBNFElement(.char, a), GBNFElement(.ruleRef, 1), GBNFElement(.alt), GBNFElement(.end),
        ])
    }

    @Test func exactRepetitionJustUnderThresholdIsAccepted() throws {
        let grammar = try GBNFGrammar(#"root ::= "a"{1999}"#)
        // 1999 個の CHAR + END。
        #expect(grammar.rules[0].count == 2000)
    }

    @Test func expansionAtThresholdIsRejected() {
        guard case .repetitionExpansionTooLarge? = parseError(#"root ::= "a"{2000}"#) else {
            Issue.record("{2000} は展開上限 (>= 2000) で落ちるはず")
            return
        }
    }

    @Test func minimumBeyondThresholdIsRejected() {
        guard case .repetitionCountTooLarge? = parseError(#"root ::= "a"{2001}"#) else {
            Issue.record("{2001} は最小回数の上限で落ちるはず")
            return
        }
    }

    @Test func maximumBeyondThresholdDowngradesToUnbounded() throws {
        // {0,2001} は上限が落ちて * と同じになる (参照実装の downgrade)。
        let source = #"root ::= "a"{0,2001}"#
        let grammar = try GBNFGrammar(source)
        #expect(grammar.rules[0] == [GBNFElement(.ruleRef, 1), GBNFElement(.end)])
        #expect(fullyMatches(source, ""))
        #expect(fullyMatches(source, String(repeating: "a", count: 3000)))
    }
}

// MARK: - 左再帰

@Suite struct GBNFLeftRecursionTests {
    @Test func directLeftRecursionIsRejected() {
        guard case .leftRecursion? = parseError(#"root ::= root "a" | "b""#) else {
            Issue.record("直接の左再帰は誤りになるはず")
            return
        }
    }

    @Test func indirectLeftRecursionIsRejected() {
        let source = """
            root ::= middle
            middle ::= root "x" | "y"
            """
        guard case .leftRecursion? = parseError(source) else {
            Issue.record("間接の左再帰は誤りになるはず")
            return
        }
    }

    @Test func leftRecursionThroughAnEmptyCapableRuleIsRejected() {
        let source = """
            root ::= maybe root "a" | "b"
            maybe ::= "m" |
            """
        guard case .leftRecursion? = parseError(source) else {
            Issue.record("空になりうる規則を挟んだ左再帰も誤りになるはず")
            return
        }
    }

    @Test func rightRecursionIsAccepted() throws {
        _ = try GBNFGrammar(#"root ::= "a" root | "b""#)
    }
}

// MARK: - トークン要素

@Suite struct GBNFTokenElementTests {
    @Test func tokenElementIsConsumedByIDNotByBytes() throws {
        var matcher = try makeMatcher(#"root ::= <[42]> "ok""#)
        #expect(!matcher.isComplete)
        // piece の中身ではなく ID で消費される。
        try matcher.accept(piece: bytes("<tool_call>"), tokenID: 42)
        #expect(!matcher.isComplete)
        try matcher.accept(text: "ok")
        #expect(matcher.isComplete)
    }

    @Test func wrongTokenIDIsRejected() throws {
        var matcher = try makeMatcher(#"root ::= <[42]> "ok""#)
        #expect(matcher.allows(piece: bytes("<tool_call>"), tokenID: 42))
        #expect(!matcher.allows(piece: bytes("<tool_call>"), tokenID: 43))
        #expect(didThrow { try matcher.accept(piece: bytes("<tool_call>"), tokenID: 43) })
    }

    @Test func negatedTokenMatchesEverythingElse() throws {
        var matcher = try makeMatcher(#"root ::= !<[42]> "z""#)
        #expect(!matcher.allows(piece: bytes("x"), tokenID: 42))
        #expect(matcher.allows(piece: bytes("x"), tokenID: 7))
        try matcher.accept(piece: bytes("x"), tokenID: 7)
        try matcher.accept(text: "z")
        #expect(matcher.isComplete)
    }

    @Test func tokenToppedStackDiesUnderCharacterInput() throws {
        var matcher = try makeMatcher("root ::= <[42]>")
        #expect(didThrow { try matcher.accept(text: "a") })
    }

    @Test func tokenAndCharAlternativesCoexist() throws {
        let source = #"root ::= <[42]> | "abc""#
        var byToken = try makeMatcher(source)
        try byToken.accept(piece: bytes("<x>"), tokenID: 42)
        #expect(byToken.isComplete)

        var byText = try makeMatcher(source)
        try byText.accept(text: "abc")
        #expect(byText.isComplete)
    }

    @Test func tokenIDIsParsedFromTheBracketForm() throws {
        let grammar = try GBNFGrammar("root ::= <[105]> !<[106]>")
        #expect(grammar.rules[0] == [
            GBNFElement(.token, 105), GBNFElement(.tokenNot, 106), GBNFElement(.end),
        ])
    }
}

// MARK: - 逐次受理と UTF-8

@Suite struct GBNFIncrementalAcceptTests {
    @Test func acceptsAcrossAMultiPieceStream() throws {
        var matcher = try makeMatcher(#"root ::= "hello" " " "world""#)
        for (index, piece) in ["he", "l", "lo", " ", "wor", "ld"].enumerated() {
            try matcher.accept(piece: bytes(piece), tokenID: Int32(index + 1))
            #expect(!matcher.isStuck)
        }
        #expect(matcher.isComplete)
    }

    @Test func acceptThrowsWhenNoStackSurvives() throws {
        var matcher = try makeMatcher(#"root ::= "ab""#)
        try matcher.accept(piece: bytes("a"), tokenID: 1)
        #expect(didThrow { try matcher.accept(piece: bytes("z"), tokenID: 2) })
    }

    @Test func multiByteCharacterSplitAcrossTwoPieces() throws {
        var matcher = try makeMatcher(#"root ::= "あい""#)
        // 'あ' == E3 81 82 を 2 つの piece に割る。
        try matcher.accept(piece: [0xE3], tokenID: 1)
        #expect(!matcher.isStuck)
        #expect(!matcher.isComplete)
        #expect(matcher.partialUTF8.remaining == 2)
        try matcher.accept(piece: [0x81, 0x82], tokenID: 2)
        #expect(matcher.partialUTF8.remaining == 0)
        try matcher.accept(text: "い")
        #expect(matcher.isComplete)
    }

    @Test func partialSequenceIsMaskedByWhatItCouldStillBecome() throws {
        // 'あ' の 1 バイト目だけの piece。肯定集合なら「なりうる」で通す。
        var possible = try makeMatcher(#"root ::= "あ""#)
        #expect(possible.allows(piece: [0xE3], tokenID: 1))
        #expect(!possible.allows(piece: [0xC3], tokenID: 2))
        try possible.accept(piece: [0xE3], tokenID: 1)
        #expect(possible.allows(piece: [0x81, 0x82], tokenID: 3))
        #expect(!possible.allows(piece: [0x81, 0x84], tokenID: 4))

        // 否定集合は非対称: 重なりがあれば落とす。
        let negatedOverlap = try makeMatcher("root ::= [^あ]")
        #expect(!negatedOverlap.allows(piece: [0xE3], tokenID: 1))
        let negatedDisjoint = try makeMatcher("root ::= [^A]")
        #expect(negatedDisjoint.allows(piece: [0xE3], tokenID: 1))
    }

    @Test func invalidUTF8PieceIsMaskedOut() throws {
        let matcher = try makeMatcher("root ::= .+")
        // 継続バイトで始まる列は壊れている (n_remain == -1)。
        #expect(!matcher.allows(piece: [0x80], tokenID: 1))
        #expect(!matcher.allows(piece: [0x61, 0x80], tokenID: 2))
        #expect(matcher.allows(piece: bytes("a"), tokenID: 3))
        // 途中で切れただけの列は「まだなりうる」ので通る。
        #expect(matcher.allows(piece: [0xE3], tokenID: 4))
    }

    @Test func invalidUTF8AcceptLeavesTheInvalidPartialState() throws {
        var matcher = try makeMatcher("root ::= .+")
        try matcher.accept(piece: [0x80], tokenID: 1)
        #expect(matcher.partialUTF8.remaining == -1)
        #expect(!matcher.isStuck)
        // 壊れた状態は次の復号の開始値としては 0 と同じに振る舞う (参照実装どおり)。
        try matcher.accept(text: "a")
        #expect(matcher.isComplete)
    }

    @Test func emptyAndNULPiecesAreMaskedOut() throws {
        let matcher = try makeMatcher("root ::= .+")
        #expect(!matcher.allows(piece: [], tokenID: 1))
        #expect(!matcher.allows(piece: [0x00, 0x61], tokenID: 2))
    }
}

// MARK: - isComplete / isStuck

@Suite struct GBNFCompletionTests {
    @Test func completionFlipsExactlyWhenTheGrammarMayEnd() throws {
        var matcher = try makeMatcher(#"root ::= "ab""#)
        #expect(!matcher.isComplete)
        try matcher.accept(text: "a")
        #expect(!matcher.isComplete)
        try matcher.accept(text: "b")
        #expect(matcher.isComplete)
        #expect(!matcher.isStuck)
    }

    @Test func optionalTailIsCompleteBeforeAndAfter() throws {
        var matcher = try makeMatcher(#"root ::= "a" "b"?"#)
        #expect(!matcher.isComplete)
        try matcher.accept(text: "a")
        #expect(matcher.isComplete)
        try matcher.accept(text: "b")
        #expect(matcher.isComplete)
    }

    @Test func nullableGrammarIsCompleteAtTheStart() throws {
        let matcher = try makeMatcher(#"root ::= "a"*"#)
        #expect(matcher.isComplete)
        #expect(!matcher.isStuck)
    }
}

// MARK: - 語彙マスク

@Suite struct GBNFCandidateMaskTests {
    /// バイト列だけの合成「語彙」。
    static let vocabulary: [(id: Int32, piece: String)] = [
        (10, "{"), (11, "}"), (12, "\""), (13, ":"), (14, ","),
        (15, "a"), (16, "1"), (17, "["), (18, "]"), (19, " "),
        (20, "true"), (21, "-"),
    ]

    static func candidates() -> [GrammarCandidate] {
        vocabulary.enumerated().map { offset, entry in
            GrammarCandidate(index: offset, tokenID: entry.id, piece: Array(entry.piece.utf8))
        }
    }

    static func allowedPieces(_ matcher: GrammarMatcher) -> Set<String> {
        let mask = matcher.allowedMask(candidates())
        var allowed: Set<String> = []
        for (offset, entry) in vocabulary.enumerated() where mask[offset] {
            allowed.insert(entry.piece)
        }
        return allowed
    }

    @Test func maskSelectsOnlyTheGrammarsContinuations() throws {
        let matcher = try makeMatcher(#"root ::= "{" ("a" | "1") "}""#)
        #expect(GBNFCandidateMaskTests.allowedPieces(matcher) == ["{"])
    }

    @Test func rejectedIndicesEchoTheCallersIndices() throws {
        let matcher = try makeMatcher(#"root ::= "{""#)
        let all = GBNFCandidateMaskTests.candidates()
        let rejected = Set(matcher.rejectedIndices(all))
        #expect(rejected.count == all.count - 1)
        #expect(!rejected.contains(0)) // "{" は index 0
    }

    @Test func batchMaskAgreesWithTheSingleCandidateQuery() throws {
        let source = #"""
            root ::= "{" ws ("a" | "true") ws "}"
            ws ::= " "*
            """#
        var matcher = try makeMatcher(source)
        try matcher.accept(text: "{")
        let all = GBNFCandidateMaskTests.candidates()
        let mask = matcher.allowedMask(all)
        for (offset, candidate) in all.enumerated() {
            #expect(
                mask[offset] == matcher.allows(piece: candidate.piece, tokenID: candidate.tokenID),
                "index \(offset) で一括と単発が食い違った"
            )
        }
        #expect(GBNFCandidateMaskTests.allowedPieces(matcher) == ["a", "true", " "])
    }

    @Test func multiCharacterPieceIsMaskedOnItsWholeSpan() throws {
        // "true" は 4 文字ぶん先まで通らないと許されない。
        let onlyPrefix = try makeMatcher(#"root ::= "tru""#)
        #expect(!onlyPrefix.allows(piece: bytes("true"), tokenID: 20))
        let full = try makeMatcher(#"root ::= "true" "!""#)
        #expect(full.allows(piece: bytes("true"), tokenID: 20))
    }

    @Test func completedMatcherRejectsFurtherInput() throws {
        var matcher = try makeMatcher(#"root ::= "a""#)
        try matcher.accept(text: "a")
        #expect(matcher.isComplete)
        // 完了済みの空スタックは、さらなる入力を許さない。
        let all = GBNFCandidateMaskTests.candidates()
        #expect(matcher.rejectedIndices(all).count == all.count)
    }
}

// MARK: - 手書き JSON 文法での通し確認

@Suite struct GBNFHandWrittenJSONTests {
    // スキーマ → 文法の変換は別の担当。ここは手で書いた小さな JSON 文法。
    static let json = #"""
        root   ::= object
        object ::= "{" ws ( pair ( ws "," ws pair )* )? ws "}"
        pair   ::= string ws ":" ws value
        value  ::= object | array | string | number | "true" | "false" | "null"
        array  ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
        string ::= "\"" ( [^"\\] | "\\" ["\\/bfnrt] )* "\""
        number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
        ws     ::= [ \t\n]*
        """#

    @Test func walksAValidDocumentTokenByToken() throws {
        var matcher = try makeMatcher(GBNFHandWrittenJSONTests.json)
        let stream = [
            "{", "\"a", "\"", ":", " ", "[", "1", ",", " ", "true", "]",
            ",", " ", "\"b", "\"", ":", " ", "\"x", "\"", "}",
        ]
        for (index, piece) in stream.enumerated() {
            #expect(
                matcher.allows(piece: bytes(piece), tokenID: Int32(index + 1)),
                "piece '\(piece)' は許されるはず"
            )
            try matcher.accept(piece: bytes(piece), tokenID: Int32(index + 1))
            #expect(!matcher.isStuck)
        }
        #expect(matcher.isComplete)
    }

    @Test func rootMustStartWithAnObject() throws {
        let matcher = try makeMatcher(GBNFHandWrittenJSONTests.json)
        #expect(matcher.allows(piece: bytes("{"), tokenID: 1))
        for piece in ["[", "\"", "1", "t", "}"] {
            #expect(!matcher.allows(piece: bytes(piece), tokenID: 2), "'\(piece)' は拒まれるはず")
        }
        #expect(!matcher.isComplete)
    }

    @Test func invalidContinuationIsMaskedOutMidDocument() throws {
        var matcher = try makeMatcher(GBNFHandWrittenJSONTests.json)
        try matcher.accept(text: "{\"a\":")
        // 値が来るべき位置。
        for piece in ["1", "-", "\"", "[", "{", "t", "f", "n", " "] {
            #expect(matcher.allows(piece: bytes(piece), tokenID: 1), "'\(piece)' は許されるはず")
        }
        for piece in ["}", ",", ":", "]"] {
            #expect(!matcher.allows(piece: bytes(piece), tokenID: 2), "'\(piece)' は拒まれるはず")
        }
        #expect(!matcher.isComplete)
    }

    @Test func stringEscapesFollowTheGrammar() throws {
        var matcher = try makeMatcher(GBNFHandWrittenJSONTests.json)
        try matcher.accept(text: "{\"a\\")
        // バックスラッシュの直後は限られた文字だけ。
        for piece in ["n", "t", "\"", "\\", "/"] {
            #expect(matcher.allows(piece: bytes(piece), tokenID: 1), "'\(piece)' は許されるはず")
        }
        for piece in ["z", "1"] {
            #expect(!matcher.allows(piece: bytes(piece), tokenID: 2), "'\(piece)' は拒まれるはず")
        }
    }

    @Test func emptyObjectIsComplete() throws {
        var matcher = try makeMatcher(GBNFHandWrittenJSONTests.json)
        try matcher.accept(text: "{}")
        #expect(matcher.isComplete)
    }

    @Test func numbersAreConstrained() throws {
        var matcher = try makeMatcher(GBNFHandWrittenJSONTests.json)
        try matcher.accept(text: "{\"n\": 1")
        #expect(matcher.allows(piece: bytes("0"), tokenID: 1))
        #expect(matcher.allows(piece: bytes("."), tokenID: 2))
        #expect(matcher.allows(piece: bytes("}"), tokenID: 3))
        #expect(!matcher.allows(piece: bytes("x"), tokenID: 4))
        try matcher.accept(text: ".5}")
        #expect(matcher.isComplete)
    }
}
