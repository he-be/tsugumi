import Foundation
import Testing

@testable import TurboFieldfare

// SPEC §6 の GEN-1 / GEN-5 / GEN-6 を、実トークンの上で検定する (P2, G4a)。
// 階層は CONFORMANCE §1 の C2 — tokenizer は使うが重みも Metal も要らない。
// 規範は参照実装 `~/LLM/llama.cpp` のピン `34af94cd9`。
//
// 初回は Gemma 4 IT の tokenizer (~32 MB) を Hugging Face Hub から
// `~/.cache/huggingface/` に取りに行く。以後はオフライン。

@Suite("GrammarTokenConstraint")
struct GrammarTokenConstraintTests {
    let tok: GFTokenizer
    let vocab: GrammarVocabulary

    init() async throws {
        self.tok = try await GFTokenizer.load()
        self.vocab = GrammarVocabulary.shared(for: tok)
    }

    // MARK: - 補助

    /// 語彙表が言う piece をつないだ文字列。文法が実際に見る入力そのもの。
    private func pieceText(_ ids: [Int32]) -> String {
        var bytes: [UInt8] = []
        for id in ids { bytes.append(contentsOf: vocab.piece(for: id)) }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// 単独のトークンとして語彙にある印だけを取り出す。無ければ nil。
    private func singleToken(_ marker: String) -> Int32? {
        guard let id = tok.tokenizer.convertTokenToId(marker),
              tok.tokenizer.convertIdToToken(id) == marker,
              let value = Int32(exactly: id) else { return nil }
        return value
    }

    private func mask(_ constraint: GrammarTokenConstraint) throws -> [Bool] {
        var buffer = [Bool](repeating: false, count: tok.vocabSize)
        try buffer.withUnsafeMutableBufferPointer { try constraint.fillAllowedMask($0) }
        return buffer
    }

    /// `{` を待っている状態でも通らない、ふつうの語のトークン。
    private var letterToken: Int32 { tok.encode("hello", addBOS: false)[0] }

    // MARK: - GEN-1: 語彙の piece 表

    @Test("GEN_1_piece_table_covers_the_whole_vocabulary")
    func pieceTableCoversVocabulary() {
        #expect(vocab.count == tok.vocabSize)
        #expect(vocab.piece(for: -1).isEmpty)
        #expect(vocab.piece(for: Int32(tok.vocabSize)).isEmpty)
    }

    @Test("GEN_1_special_markers_are_present_as_their_literal_text")
    func specialMarkersArePresent() {
        // GEN-8 の tool call 文法はこの綴りのリテラルで書かれている。
        #expect(vocab.piece(for: tok.toolCallStartID) == Array("<|tool_call>".utf8))
        #expect(vocab.piece(for: tok.toolCallEndID) == Array("<tool_call|>".utf8))
        #expect(vocab.piece(for: tok.endOfTurnID) == Array("<turn|>".utf8))
    }

    @Test("GEN_1_metaspace_becomes_a_space_and_byte_fallback_becomes_one_byte")
    func metaspaceAndByteFallback() {
        let ids = tok.encode(" hello", addBOS: false)
        #expect(pieceText(ids) == " hello")

        // `<0xC3>` は 1 バイトしか足さない — UTF-8 として不正な piece。
        if let byteToken = singleToken("<0xC3>") {
            #expect(vocab.piece(for: byteToken) == [0xC3])
        }
        // `<0x00>` は NUL 始まり。参照実装が文法にかける前に落とす形。
        if let nulToken = singleToken("<0x00>") {
            #expect(vocab.piece(for: nulToken) == [0x00])
        }
    }

    // MARK: - GEN-1: トークン単位の受理とマスク

    private static let jsonish = """
        root ::= "{" pair ("," pair)* "}"
        pair ::= key ":" digits
        key ::= [a-z]+
        digits ::= [0-9]+
        """

    @Test("GEN_1_accepts_real_gemma_tokens_one_by_one")
    func acceptsRealTokens() throws {
        let target = "{a:1,bb:22}"
        let ids = tok.encode(target, addBOS: false)
        #expect(pieceText(ids) == target, "piece 表が入力を再現していない")

        let constraint = try GrammarTokenConstraint(Self.jsonish, vocabulary: vocab)
        for id in ids {
            #expect(constraint.allows(tokenID: id), "文法が \(id) を拒んだ")
            try constraint.accept(tokenID: id)
        }
        #expect(constraint.mayEndHere)
    }

    @Test("GEN_1_may_end_here_is_false_mid_grammar_and_true_at_the_end")
    func mayEndHereTracksCompletion() throws {
        let ids = tok.encode("{a:1}", addBOS: false)
        let constraint = try GrammarTokenConstraint(Self.jsonish, vocabulary: vocab)
        #expect(!constraint.mayEndHere)
        for id in ids.dropLast() {
            try constraint.accept(tokenID: id)
            #expect(!constraint.mayEndHere)
        }
        try constraint.accept(tokenID: ids[ids.count - 1])
        #expect(constraint.mayEndHere)
    }

    @Test("GEN_1_a_forbidden_token_is_masked_out")
    func forbiddenTokenIsMasked() throws {
        let constraint = try GrammarTokenConstraint(Self.jsonish, vocabulary: vocab)
        let bad = letterToken
        let open = tok.encode("{", addBOS: false)[0]
        let table = try mask(constraint)
        #expect(!constraint.allows(tokenID: bad))
        #expect(table[Int(bad)] == false)
        #expect(constraint.allows(tokenID: open))
        #expect(table[Int(open)] == true)
    }

    @Test("GEN_1_empty_and_NUL_pieces_are_rejected_like_the_reference")
    func emptyAndNULPiecesRejected() throws {
        let constraint = try GrammarTokenConstraint(Self.jsonish, vocabulary: vocab)
        let table = try mask(constraint)
        for id in 0..<tok.vocabSize {
            let piece = vocab.piece(for: Int32(id))
            guard piece.isEmpty || piece[0] == 0 else { continue }
            #expect(table[id] == false, "空 / NUL 始まりの piece \(id) が許された")
            #expect(!constraint.allows(tokenID: Int32(id)))
        }
    }

    @Test("GEN_1_batch_mask_agrees_with_the_single_token_probe")
    func batchMaskAgreesWithProbe() throws {
        let constraint = try GrammarTokenConstraint(Self.jsonish, vocabulary: vocab)
        // 途中の状態で比べる (開始状態だけだと `{` しか通らず退屈)。
        for id in tok.encode("{ab", addBOS: false) { try constraint.accept(tokenID: id) }

        let table = try mask(constraint)
        var checked = 0
        for id in stride(from: 0, to: tok.vocabSize, by: 379) {
            #expect(table[id] == constraint.allows(tokenID: Int32(id)),
                    "マスクと単発判定が食い違った (id \(id))")
            checked += 1
        }
        // 文法が通す綴りは必ず両方で true。
        for text in [":", "c", "abc"] {
            let id = tok.encode(text, addBOS: false)[0]
            #expect(table[Int(id)] == constraint.allows(tokenID: id))
        }
        #expect(checked > 100)
    }

    @Test("GEN_1_an_illegal_token_throws_on_accept")
    func illegalTokenThrows() throws {
        let constraint = try GrammarTokenConstraint(Self.jsonish, vocabulary: vocab)
        #expect(throws: (any Error).self) {
            try constraint.accept(tokenID: self.letterToken)
        }
    }

    // MARK: - GEN-8: 特殊マーカーのリテラル一致

    @Test("GEN_8_a_special_marker_piece_matches_a_literal_written_with_it")
    func specialMarkerMatchesLiteral() throws {
        guard let quote = singleToken("<|\"|>") else {
            Issue.record("語彙に <|\"|> が単独トークンとして無い")
            return
        }
        #expect(vocab.piece(for: quote) == Array("<|\"|>".utf8))

        let grammar = #"""
            root ::= "<|\"|>" body "<|\"|>"
            body ::= [a-z]+
            """#
        let constraint = try GrammarTokenConstraint(grammar, vocabulary: vocab)
        #expect(constraint.allows(tokenID: quote))
        try constraint.accept(tokenID: quote)
        #expect(!constraint.mayEndHere)
        for id in tok.encode("abc", addBOS: false) { try constraint.accept(tokenID: id) }
        #expect(constraint.allows(tokenID: quote))
        try constraint.accept(tokenID: quote)
        #expect(constraint.mayEndHere)
    }

    // MARK: - GEN-5: 遅延文法

    private var lazyGrammar: String {
        #"""
        root ::= "<|tool_call>" "call:" name "<tool_call|>"
        name ::= [a-z]+
        """#
    }

    private func makeLazy() throws -> GrammarTokenConstraint {
        try GrammarTokenConstraint(lazyGrammar,
                                   vocabulary: vocab,
                                   trigger: .token(tok.toolCallStartID))
    }

    @Test("GEN_5_everything_is_allowed_while_disarmed")
    func disarmedAllowsEverything() throws {
        let constraint = try makeLazy()
        #expect(constraint.isLazy)
        #expect(!constraint.isArmed)
        #expect(constraint.mayEndHere)

        let table = try mask(constraint)
        #expect(table.allSatisfy { $0 }, "未起動の遅延文法がトークンを落とした")
        for id in stride(from: 0, to: tok.vocabSize, by: 379) {
            #expect(constraint.allows(tokenID: Int32(id)))
        }
    }

    @Test("GEN_5_the_trigger_arms_exactly_on_the_trigger_token")
    func triggerArmsOnTheTriggerToken() throws {
        let constraint = try makeLazy()
        for id in tok.encode("thinking out loud", addBOS: false) {
            try constraint.accept(tokenID: id)
            #expect(!constraint.isArmed, "トリガでないトークンで起動した")
            #expect(constraint.mayEndHere)
        }

        try constraint.accept(tokenID: tok.toolCallStartID)
        #expect(constraint.isArmed)
        // 参照実装はトリガ自身を文法に食わせる: `<|tool_call>` は消費済み。
        #expect(!constraint.mayEndHere)
        #expect(!constraint.allows(tokenID: tok.toolCallStartID))
        for id in tok.encode("call:", addBOS: false) {
            #expect(constraint.allows(tokenID: id))
            try constraint.accept(tokenID: id)
        }
        for id in tok.encode("ab", addBOS: false) { try constraint.accept(tokenID: id) }
        #expect(constraint.allows(tokenID: tok.toolCallEndID))
        try constraint.accept(tokenID: tok.toolCallEndID)
        #expect(constraint.mayEndHere)
    }

    @Test("GEN_5_a_non_lazy_constraint_is_armed_from_the_start")
    func nonLazyIsArmed() throws {
        let constraint = try GrammarTokenConstraint(Self.jsonish, vocabulary: vocab)
        #expect(!constraint.isLazy)
        #expect(constraint.isArmed)
        #expect(constraint.isApplied)
    }

    // MARK: - GEN-6: 思考中の抑止

    @Test("GEN_6_suppression_allows_everything_and_does_not_feed_the_matcher")
    func suppressionAllowsEverything() throws {
        let constraint = try makeLazy()
        try constraint.accept(tokenID: tok.toolCallStartID)
        for id in tok.encode("call:a", addBOS: false) { try constraint.accept(tokenID: id) }

        constraint.setSuppressed(true)
        #expect(constraint.isSuppressed)
        #expect(!constraint.isApplied)
        #expect(constraint.mayEndHere)
        let table = try mask(constraint)
        #expect(table.allSatisfy { $0 })

        // 抑止中に来たトークンは文法に供給されない (参照実装
        // `common_sampler_accept` の `accept_grammar` が false)。
        try constraint.accept(tokenID: tok.endOfTurnID)
        try constraint.accept(tokenID: tok.toolCallStartID)

        constraint.setSuppressed(false)
        #expect(constraint.isApplied)
        // 抑止前の状態がそのまま続く: name の途中。
        #expect(!constraint.mayEndHere)
        #expect(constraint.allows(tokenID: tok.toolCallEndID))
        try constraint.accept(tokenID: tok.toolCallEndID)
        #expect(constraint.mayEndHere)
    }

    @Test("GEN_6_a_trigger_inside_the_thought_channel_does_not_arm")
    func triggerInsideThoughtDoesNotArm() throws {
        let constraint = try makeLazy()
        constraint.setSuppressed(true)
        try constraint.accept(tokenID: tok.toolCallStartID)
        #expect(!constraint.isArmed, "思考中のトリガで起動してしまった")

        constraint.setSuppressed(false)
        #expect(!constraint.isArmed)
        try constraint.accept(tokenID: tok.toolCallStartID)
        #expect(constraint.isArmed)
    }

    @Test("GEN_6_a_non_lazy_constraint_is_never_suppressed")
    func nonLazyIsNeverSuppressed() throws {
        let constraint = try GrammarTokenConstraint(Self.jsonish, vocabulary: vocab)
        constraint.setSuppressed(true)
        #expect(!constraint.isSuppressed)
        #expect(constraint.isApplied)
        #expect(!constraint.allows(tokenID: self.letterToken))
    }

    // MARK: - 費用 (報告用。閾値は置かない)

    @Test("GEN_1_measures_the_table_build_and_the_two_hot_paths")
    func measuresCost() throws {
        let fresh = ContinuousClock().measure { _ = GrammarVocabulary(tok) }
        let cached = ContinuousClock().measure { _ = GrammarVocabulary.shared(for: tok) }

        let constraint = try GrammarTokenConstraint(Self.jsonish, vocabulary: vocab)
        for id in tok.encode("{ab", addBOS: false) { try constraint.accept(tokenID: id) }

        let probes = 2000
        let probe = ContinuousClock().measure {
            for i in 0..<probes { _ = constraint.allows(tokenID: Int32(i % tok.vocabSize)) }
        }
        var buffer = [Bool](repeating: false, count: tok.vocabSize)
        let batch = try buffer.withUnsafeMutableBufferPointer { pointer in
            try ContinuousClock().measure { try constraint.fillAllowedMask(pointer) }
        }

        print("""
            [GrammarTokenConstraint] 語彙表の構築 \(fresh) (\(vocab.count) 個) / \
            共有の再取得 \(cached) / allows 1 回 \(probe / probes) / \
            fillAllowedMask 1 回 \(batch)
            """)
        #expect(vocab.count == tok.vocabSize)
    }
}
