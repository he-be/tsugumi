import Foundation
import Testing

@testable import Tsugumi

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

    // MARK: - GEN-7: 棄却経路の復号表 (意味を動かさない最適化)

    /// 復号は語彙表と同じ寿命で 1 度だけ払う。
    @Test("GEN_7_the_decoded_code_point_table_covers_every_candidate")
    func decodedTableCoversCandidates() {
        #expect(vocab.decodedCodePoints.count == vocab.candidates.count)
        #expect(vocab.decodedPartials.count == vocab.candidates.count)
        #expect(vocab.candidateTokenIDs.count == vocab.candidates.count)
        #expect(vocab.baseWork.count == vocab.candidates.count)

        let filled = min(vocab.decodedCodePoints.count,
                         min(vocab.decodedPartials.count,
                             min(vocab.candidateTokenIDs.count, vocab.baseWork.count)))
        for slot in stride(from: 0, to: min(vocab.candidates.count, filled), by: 379) {
            let candidate = vocab.candidates[slot]
            let reference = GrammarMatcher.decodeUTF8(candidate.piece, GrammarPartialUTF8())
            #expect(vocab.decodedCodePoints[slot] == reference.codePoints)
            #expect(vocab.decodedPartials[slot] == reference.partial)
            #expect(vocab.candidateTokenIDs[slot] == candidate.tokenID)
            #expect(vocab.baseWork[slot].slot == slot)
            #expect(vocab.baseWork[slot].offset == 0)
        }
    }

    /// 最適化前の計算そのもの: 生の `GrammarMatcher.rejectedIndices` から
    /// 組んだマスク。これと `fillAllowedMask` が**全語彙で**一致すること。
    private func referenceMask(_ matcher: GrammarMatcher) -> [Bool] {
        var table = [Bool](repeating: false, count: tok.vocabSize)
        for candidate in vocab.candidates where candidate.index < table.count {
            table[candidate.index] = true
        }
        for index in matcher.rejectedIndices(vocab.candidates) where index < table.count {
            table[index] = false
        }
        return table
    }

    private func makeMatcher(_ source: String, feeding ids: [Int32]) throws -> GrammarMatcher {
        var matcher = try GrammarMatcher(GBNFGrammar(source))
        for id in ids { try matcher.accept(piece: vocab.piece(for: id), tokenID: id) }
        return matcher
    }

    @Test("GEN_7_the_mask_equals_the_uncached_engine_result_over_the_whole_vocabulary")
    func maskEqualsUncachedResult() throws {
        for prefix in ["", "{", "{ab", "{ab:1,"] {
            let ids = prefix.isEmpty ? [] : tok.encode(prefix, addBOS: false)
            let constraint = try GrammarTokenConstraint(Self.jsonish, vocabulary: vocab)
            for id in ids { try constraint.accept(tokenID: id) }
            let mine = try mask(constraint)
            let reference = referenceMask(try makeMatcher(Self.jsonish, feeding: ids))
            #expect(mine == reference, "接頭辞 \"\(prefix)\" でマスクがずれた")
        }
    }

    /// `partialUTF8` が空でない状態 — 多バイト文字の途中で切れた piece を
    /// 食べた直後。ここでは復号表をそのまま使えないので、実装は復号し直す。
    @Test("GEN_7_a_pending_partial_utf8_still_matches_the_uncached_result")
    func partialUTF8StillMatches() throws {
        guard let lead = singleToken("<0xC3>"), let tail = singleToken("<0xA9>") else {
            Issue.record("バイトフォールバックのトークンが語彙に無い")
            return
        }
        // "é" は C3 A9。先頭バイトだけ食べると多バイト列の途中で止まる。
        let grammar = #"""
            root ::= "é" "x"
            """#
        let constraint = try GrammarTokenConstraint(grammar, vocabulary: vocab)
        try constraint.accept(tokenID: lead)

        var matcher = try GrammarMatcher(GBNFGrammar(grammar))
        try matcher.accept(piece: vocab.piece(for: lead), tokenID: lead)
        #expect(matcher.partialUTF8.remaining != 0, "この状態は partialUTF8 を残さない")

        let mine = try mask(constraint)
        #expect(mine == referenceMask(matcher), "partialUTF8 が残る状態でマスクがずれた")
        #expect(mine[Int(tail)] == true)
        #expect(constraint.allows(tokenID: tail))

        // 途中の状態でも単発判定と一致する。
        for id in stride(from: 0, to: tok.vocabSize, by: 379) {
            #expect(mine[id] == constraint.allows(tokenID: Int32(id)), "id \(id)")
        }

        try constraint.accept(tokenID: tail)
        for id in tok.encode("x", addBOS: false) { try constraint.accept(tokenID: id) }
        #expect(constraint.mayEndHere)
    }

    // MARK: - 費用 (報告用。閾値は置かない)

    @Test("GEN_1_measures_the_table_build_and_the_two_hot_paths")
    func measuresCost() throws {
        // 実メモリは 1 本ぶんの差では読めない (直前に捨てた表のページが
        // 残るので phys_footprint がほぼ動かない)。生かしたまま複数本積んで
        // 割る。
        var live: [GrammarVocabulary] = []
        let copies = 3
        let before = Self.physFootprint()
        let fresh = ContinuousClock().measure { live.append(GrammarVocabulary(tok)) }
        for _ in 1..<copies { live.append(GrammarVocabulary(tok)) }
        let after = Self.physFootprint()
        let perCopy = live.count == copies ? (after &- before) / UInt64(copies) : 0
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
        // partialUTF8 が残る = 復号表を使えない側の経路。
        let slow: Duration
        if let lead = singleToken("<0xC3>") {
            let partial = try GrammarTokenConstraint(#"root ::= "é" "x""#, vocabulary: vocab)
            try partial.accept(tokenID: lead)
            slow = try buffer.withUnsafeMutableBufferPointer { pointer in
                try ContinuousClock().measure { try partial.fillAllowedMask(pointer) }
            }
        } else {
            slow = .zero
        }

        let pieceBytes = (0..<tok.vocabSize).reduce(0) { $0 + vocab.piece(for: Int32($1)).count }
        let codePoints = vocab.decodedCodePoints.reduce(0) { $0 + $1.count }
        print("""
            [GrammarTokenConstraint] 語彙表の構築 \(fresh) (\(vocab.count) 個, \
            実メモリ 1 本あたり \(perCopy / 1_048_576) MB) / \
            共有の再取得 \(cached) / allows 1 回 \(probe / probes) / \
            fillAllowedMask 1 回 \(batch) / partialUTF8 有りの回 \(slow) / \
            piece 合計 \(pieceBytes) B / コードポイント合計 \(codePoints)
            """)
        #expect(vocab.count == tok.vocabSize)
        #expect(live.allSatisfy { $0.count == tok.vocabSize })
    }

    /// このプロセスの実メモリ使用量 (バイト)。取れなければ 0。
    private static func physFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : 0
    }
}
