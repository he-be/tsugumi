// GBNF 文法との逐次照合。
//
// 規範は参照実装 `~/LLM/llama.cpp` のピン `34af94cd9`、`src/llama-grammar.cpp` の
// `llama_grammar_*` 群 (SPEC §0 の優先順位 2)。SPEC §6 GEN-1 が名指しする
// 「文法で拘束して生成する」機構のうち、照合とマスクを担う部分。
//
// 状態は「スタックの集合」。1 本のスタックは規則要素への位置の並びで、末尾が
// いま満たすべき義務。参照実装はここを生ポインタで持つが、こちらは
// `(規則番号, 要素番号)` の組で持つ (値型として複製できる)。

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

/// 文法との照合状態。値型で、複製は配列の CoW ぶんしか要らない。
/// tokenizer や Metal への参照は持たない。
public struct GrammarMatcher: Sendable {
    public let grammar: GBNFGrammar
    /// 生きているスタックの集合。
    public private(set) var stacks: [[GrammarPosition]]
    /// 直前まで受理した piece の末尾に残った不完全な UTF-8 列。
    public private(set) var partialUTF8: GrammarPartialUTF8

    public init(_ grammar: GBNFGrammar) throws {
        self.grammar = grammar
        self.partialUTF8 = GrammarPartialUTF8()

        // 開始規則の各選択肢から初期スタックを作る。
        var accumulator = StackAccumulator()
        let rules = grammar.rules
        let root = rules[grammar.rootRuleIndex]
        var index = 0
        while true {
            var stack: [GrammarPosition] = []
            if !root[index].type.isEndOfSequence {
                stack.append(GrammarPosition(grammar.rootRuleIndex, index))
            }
            GrammarMatcher.advance(stack: stack, rules: rules, into: &accumulator)
            while !root[index].type.isEndOfSequence { index += 1 }
            if root[index].type == .alt { index += 1 } else { break }
        }
        self.stacks = accumulator.stacks
    }

    /// 空のスタックがある = ここで生成を終えてよい。
    public var isComplete: Bool { stacks.contains(where: \.isEmpty) }

    /// スタックが 1 本も無い = もう何も続けられない。
    public var isStuck: Bool { stacks.isEmpty }

    // MARK: - 受理

    /// 1 コードポイント進める。参照実装の `llama_grammar_accept`。
    public mutating func accept(codePoint: UInt32) {
        var accumulator = StackAccumulator()
        for stack in stacks {
            GrammarMatcher.acceptCodePoint(
                stack: stack, codePoint: codePoint, rules: grammar.rules, into: &accumulator
            )
        }
        stacks = accumulator.stacks
    }

    /// 文字だけで進める。参照実装の `llama_grammar_accept_str`。
    /// トークン要素 (`<[N]>`) を先頭に持つスタックは、文字入力では死ぬ。
    public mutating func accept(bytes: [UInt8]) throws {
        let decoded = GrammarMatcher.decodeUTF8(bytes, partialUTF8)
        for codePoint in decoded.codePoints {
            accept(codePoint: codePoint)
        }
        partialUTF8 = decoded.partial
        if stacks.isEmpty {
            throw GBNFError.noSurvivingStacks(piece: bytes, tokenID: nil)
        }
    }

    public mutating func accept(text: String) throws {
        try accept(bytes: Array(text.utf8))
    }

    /// トークン 1 個ぶん進める。参照実装の `llama_grammar_accept_token`。
    ///
    /// 元のスタックごとに、その系統の中だけでコードポイントを流して生き残りを
    /// 併合する (集合全体に対する 1 コードポイントずつの走査ではない)。
    /// トークン要素を先頭に持つスタックは、piece のバイト列ではなく ID で
    /// まとめて消費する。`partialUTF8` はどちらの場合も piece の復号から更新する。
    public mutating func accept(piece: [UInt8], tokenID: Int32) throws {
        let rules = grammar.rules
        let decoded = GrammarMatcher.decodeUTF8(piece, partialUTF8)

        var accumulator = StackAccumulator()
        for stack in stacks {
            guard let top = stack.last else { continue } // 空スタックは続きを持たない
            let element = rules[Int(top.rule)][Int(top.element)]

            if element.type == .token || element.type == .tokenNot {
                if GrammarMatcher.matchToken(element, tokenID) {
                    var next = stack
                    next.removeLast()
                    let after = Int(top.element) + 1
                    if !rules[Int(top.rule)][after].type.isEndOfSequence {
                        next.append(GrammarPosition(Int(top.rule), after))
                    }
                    GrammarMatcher.advance(stack: next, rules: rules, into: &accumulator)
                }
                continue
            }

            var current = [stack]
            for codePoint in decoded.codePoints {
                var step = StackAccumulator()
                for candidate in current {
                    GrammarMatcher.acceptCodePoint(
                        stack: candidate, codePoint: codePoint, rules: rules, into: &step
                    )
                }
                current = step.stacks
                if current.isEmpty { break }
            }
            for survivor in current { accumulator.insert(survivor) }
        }

        stacks = accumulator.stacks
        partialUTF8 = decoded.partial

        if stacks.isEmpty {
            throw GBNFError.noSurvivingStacks(piece: piece, tokenID: tokenID)
        }
    }

    // MARK: - マスク

    /// この候補 1 個が文法を生かしたままにできるか。
    ///
    /// 語彙全体をマスクするときは `rejectedIndices(_:)` を使うこと。こちらは
    /// 1 候補ごとに交差計算をやり直すので、262k 回呼ぶと同じ仕事を 262k 回する。
    public func allows(piece: [UInt8], tokenID: Int32) -> Bool {
        rejectedSlots([GrammarCandidate(index: 0, tokenID: tokenID, piece: piece)]).isEmpty
    }

    /// 候補集合のうち、文法が拒む候補の `index` を返す。
    ///
    /// 参照実装の `llama_grammar_reject_candidates` と同じ「累進的な交差」:
    /// スタック i の拒否集合をスタック i+1 の候補集合として渡す。したがって
    /// 交差はスタック集合につき 1 回しか走らない。
    ///
    /// 計算量: 候補数 × piece のコードポイント長 が下限で、そこにスタック本数と
    /// 規則参照の展開が乗る。piece の UTF-8 復号は候補ごとに毎回行う
    /// (`partialUTF8` が変わるため使い回せない) — 参照実装の
    /// `llama_grammar_apply_impl` と同じ形。
    public func rejectedIndices(_ candidates: [GrammarCandidate]) -> [Int] {
        rejectedSlots(candidates).map { candidates[$0].index }
    }

    /// `candidates` の並び順に沿った可否マスク (`true` = 許す)。
    public func allowedMask(_ candidates: [GrammarCandidate]) -> [Bool] {
        var mask = [Bool](repeating: true, count: candidates.count)
        for slot in rejectedSlots(candidates) { mask[slot] = false }
        return mask
    }

    /// 拒否された候補の「`candidates` 内の位置」を返す内部形。
    private func rejectedSlots(_ candidates: [GrammarCandidate]) -> [Int] {
        guard !candidates.isEmpty else { return [] }
        guard !stacks.isEmpty else { return Array(candidates.indices) }

        var context = RejectContext(
            rules: grammar.rules,
            decoded: [],
            partials: [],
            tokenIDs: []
        )
        context.decoded.reserveCapacity(candidates.count)
        context.partials.reserveCapacity(candidates.count)
        context.tokenIDs.reserveCapacity(candidates.count)

        var rejected: [Int] = []
        var work: [WorkCandidate] = []
        work.reserveCapacity(candidates.count)

        for (slot, candidate) in candidates.enumerated() {
            context.tokenIDs.append(candidate.tokenID)
            // 参照実装の `llama_grammar_apply_impl` は、空 piece と先頭が NUL の
            // piece を文法にかける前に落とす。ここでも同じく落とす。
            if candidate.piece.isEmpty || candidate.piece[0] == 0 {
                context.decoded.append([])
                context.partials.append(GrammarPartialUTF8())
                rejected.append(slot)
                continue
            }
            let decoded = GrammarMatcher.decodeUTF8(candidate.piece, partialUTF8)
            context.decoded.append(decoded.codePoints)
            context.partials.append(decoded.partial)
            work.append(WorkCandidate(slot: slot, offset: 0))
        }

        rejected.append(contentsOf: GrammarMatcher.rejectCandidates(stacks, work, context).map(\.slot))
        return rejected
    }

    // MARK: - 内部: スタックの集合

    /// 重複を除きつつ順序を保つスタック集合。参照実装が `std::find` で行う
    /// 重複除去に対応する。
    struct StackAccumulator {
        private(set) var stacks: [[GrammarPosition]] = []
        private var seen: Set<[GrammarPosition]> = []

        mutating func insert(_ stack: [GrammarPosition]) {
            if seen.insert(stack).inserted { stacks.append(stack) }
        }
    }

    /// `RULE_REF` の ε 閉包。参照実装の `llama_grammar_advance_stack`。
    ///
    /// 事後条件: 追加されるスタックは空か、末尾が終端要素
    /// (`char` / `charNot` / `charAny` / `token` / `tokenNot`) のいずれか。
    static func advance(
        stack: [GrammarPosition],
        rules: [[GBNFElement]],
        into accumulator: inout StackAccumulator
    ) {
        var todo: [[GrammarPosition]] = [stack]
        var seen: Set<[GrammarPosition]> = []

        while let current = todo.popLast() {
            if !seen.insert(current).inserted { continue }

            guard let top = current.last else {
                accumulator.insert(current)
                continue
            }
            let rule = Int(top.rule)
            let index = Int(top.element)
            let element = rules[rule][index]

            switch element.type {
            case .ruleRef:
                let target = Int(element.value)
                var sub = 0
                while true {
                    var next = current
                    next.removeLast()
                    if !rules[rule][index + 1].type.isEndOfSequence {
                        next.append(GrammarPosition(rule, index + 1))
                    }
                    if !rules[target][sub].type.isEndOfSequence {
                        next.append(GrammarPosition(target, sub))
                    }
                    todo.append(next)
                    while !rules[target][sub].type.isEndOfSequence { sub += 1 }
                    if rules[target][sub].type == .alt { sub += 1 } else { break }
                }
            case .char, .charNot, .charAny, .token, .tokenNot:
                accumulator.insert(current)
            case .end, .alt, .charAlt, .charRangeUpper:
                // 解析器の不変条件により到達しない。参照実装はここで異常終了する。
                assertionFailure("スタックの先頭に終端でない要素が来た: \(element.type)")
            }
        }
    }

    /// 参照実装の `llama_grammar_accept_chr`。
    static func acceptCodePoint(
        stack: [GrammarPosition],
        codePoint: UInt32,
        rules: [[GBNFElement]],
        into accumulator: inout StackAccumulator
    ) {
        // 空スタックは文法の終わり。これ以上の入力では死ぬ。
        guard let top = stack.last else { return }
        let rule = rules[Int(top.rule)]
        let element = rule[Int(top.element)]

        // トークン要素は文字では消費できない。
        if element.type == .token || element.type == .tokenNot { return }

        let match = matchChar(rule, Int(top.element), codePoint)
        guard match.matched else { return }

        var next = stack
        next.removeLast()
        if !rule[match.next].type.isEndOfSequence {
            next.append(GrammarPosition(Int(top.rule), match.next))
        }
        advance(stack: next, rules: rules, into: &accumulator)
    }

    // MARK: - 内部: 終端の照合

    /// 参照実装の `llama_grammar_match_char`。戻り値の `next` は文字集合の直後。
    static func matchChar(
        _ rule: [GBNFElement],
        _ index: Int,
        _ codePoint: UInt32
    ) -> (matched: Bool, next: Int) {
        var pos = index
        var found = false
        let isPositive = rule[index].type == .char || rule[index].type == .charAny

        repeat {
            if rule[pos + 1].type == .charRangeUpper {
                found = found || (rule[pos].value <= codePoint && codePoint <= rule[pos + 1].value)
                pos += 2
            } else if rule[pos].type == .charAny {
                found = true
                pos += 1
            } else {
                found = found || rule[pos].value == codePoint
                pos += 1
            }
        } while rule[pos].type == .charAlt

        return (found == isPositive, pos)
    }

    /// 参照実装の `llama_grammar_match_partial_char`。
    ///
    /// 途中までの UTF-8 列が、この文字集合を満たしうるか。肯定集合は「重なりが
    /// あれば通す」、否定集合は「重なりがあれば落とす」という非対称をそのまま写す。
    static func matchPartialChar(
        _ rule: [GBNFElement],
        _ index: Int,
        _ partial: GrammarPartialUTF8
    ) -> Bool {
        let isPositive = rule[index].type == .char || rule[index].type == .charAny
        let value = partial.value
        let remaining = partial.remaining

        // 壊れた列、または 7 bit 文字を 2 バイトに割った冗長表現。
        if remaining < 0 || (remaining == 1 && value < 2) { return false }

        var low = value << (remaining * 6)
        let high = low | ((1 << (remaining * 6)) - 1)

        if low == 0 {
            if remaining == 2 {
                low = 1 << 11
            } else if remaining == 3 {
                low = 1 << 16
            }
        }

        var pos = index
        repeat {
            if rule[pos + 1].type == .charRangeUpper {
                if rule[pos].value <= high, low <= rule[pos + 1].value { return isPositive }
                pos += 2
            } else if rule[pos].type == .charAny {
                return true
            } else {
                if low <= rule[pos].value, rule[pos].value <= high { return isPositive }
                pos += 1
            }
        } while rule[pos].type == .charAlt

        return !isPositive
    }

    /// 参照実装の `llama_grammar_match_token`。
    static func matchToken(_ element: GBNFElement, _ tokenID: Int32) -> Bool {
        switch element.type {
        case .token: return element.value == UInt32(bitPattern: tokenID)
        case .tokenNot: return element.value != UInt32(bitPattern: tokenID)
        default: return false
        }
    }

    // MARK: - 内部: 棄却

    struct WorkCandidate {
        var slot: Int
        var offset: Int
    }

    struct RejectContext {
        var rules: [[GBNFElement]]
        var decoded: [[UInt32]]
        var partials: [GrammarPartialUTF8]
        var tokenIDs: [Int32]
    }

    /// 参照実装の `llama_grammar_reject_candidates`。累進的な交差。
    static func rejectCandidates(
        _ stacks: [[GrammarPosition]],
        _ candidates: [WorkCandidate],
        _ context: RejectContext
    ) -> [WorkCandidate] {
        if candidates.isEmpty { return [] }
        // 逸脱: 参照実装はここで `GGML_ASSERT(!stacks.empty())`。行き詰まった
        // 文法では「全部拒む」が正しい答えなので、落とさずそう返す。
        if stacks.isEmpty { return candidates }

        var rejects = rejectCandidatesForStack(stacks[0], candidates, context)
        for i in 1..<stacks.count {
            rejects = rejectCandidatesForStack(stacks[i], rejects, context)
        }
        return rejects
    }

    /// 参照実装の `llama_grammar_reject_candidates_for_stack`。
    /// 1 コードポイント消費して進んだスタックへ再帰し、戻りで読み位置を 1 戻す。
    static func rejectCandidatesForStack(
        _ stack: [GrammarPosition],
        _ candidates: [WorkCandidate],
        _ context: RejectContext
    ) -> [WorkCandidate] {
        var rejects: [WorkCandidate] = []
        rejects.reserveCapacity(candidates.count)

        guard let top = stack.last else {
            // 空スタック = 文法はここで終わってよい。まだ文字が残る候補は拒む。
            for candidate in candidates {
                if candidate.offset < context.decoded[candidate.slot].count
                    || context.partials[candidate.slot].remaining != 0 {
                    rejects.append(candidate)
                }
            }
            return rejects
        }

        let rule = context.rules[Int(top.rule)]
        let element = rule[Int(top.element)]

        if element.type == .token || element.type == .tokenNot {
            for candidate in candidates {
                if candidate.offset == context.decoded[candidate.slot].count {
                    // 文字規則で食べ切った末尾。壊れた/途中の列だけ拒む。
                    if context.partials[candidate.slot].remaining != 0 {
                        rejects.append(candidate)
                    }
                } else if !matchToken(element, context.tokenIDs[candidate.slot]) {
                    rejects.append(candidate)
                }
            }
            return rejects
        }

        var next: [WorkCandidate] = []
        next.reserveCapacity(candidates.count)

        for candidate in candidates {
            let codePoints = context.decoded[candidate.slot]
            if candidate.offset == codePoints.count {
                let partial = context.partials[candidate.slot]
                if partial.remaining != 0,
                   !matchPartialChar(rule, Int(top.element), partial) {
                    rejects.append(candidate)
                }
            } else if matchChar(rule, Int(top.element), codePoints[candidate.offset]).matched {
                next.append(WorkCandidate(slot: candidate.slot, offset: candidate.offset + 1))
            } else {
                rejects.append(candidate)
            }
        }

        let after = matchChar(rule, Int(top.element), 0).next
        var advanced = stack
        advanced.removeLast()
        if !rule[after].type.isEndOfSequence {
            advanced.append(GrammarPosition(Int(top.rule), after))
        }

        var accumulator = StackAccumulator()
        advance(stack: advanced, rules: context.rules, into: &accumulator)

        for candidate in rejectCandidates(accumulator.stacks, next, context) {
            rejects.append(WorkCandidate(slot: candidate.slot, offset: candidate.offset - 1))
        }

        return rejects
    }

    // MARK: - 内部: 逐次 UTF-8 復号

    /// 参照実装の 2 引数版 `decode_utf8`。piece をまたぐ多バイト文字を扱う。
    ///
    /// 戻り値の `codePoints` には参照実装が末尾に足す番兵 0 を含めない。
    /// 壊れた列に当たったときは `codePoints` を空にし、`remaining` を -1 にする
    /// (参照実装の `n_remain == -1` の状態)。piece 中の NUL でそこで打ち切る点も
    /// 参照実装 (`std::string::c_str()` を歩く) と同じ。
    static func decodeUTF8(
        _ bytes: [UInt8],
        _ start: GrammarPartialUTF8
    ) -> (codePoints: [UInt32], partial: GrammarPartialUTF8) {
        let lookup = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4]
        var codePoints: [UInt32] = []
        codePoints.reserveCapacity(bytes.count)

        var value = start.value
        var remaining = start.remaining
        var i = 0
        let count = bytes.count

        // 前の piece から続く列を先に閉じる。
        while i < count, bytes[i] != 0, remaining > 0 {
            let next = bytes[i]
            if (next >> 6) != 2 {
                return ([], GrammarPartialUTF8(value: 0, remaining: -1))
            }
            value = (value << 6) + UInt32(next & 0x3F)
            i += 1
            remaining -= 1
        }
        if start.remaining > 0, remaining == 0 {
            codePoints.append(value)
        }

        while i < count, bytes[i] != 0 {
            let first = bytes[i]
            remaining = lookup[Int(first >> 4)] - 1
            if remaining < 0 {
                // 継続バイトで始まる = 壊れた列。ここまでの結果も捨てる。
                return ([], GrammarPartialUTF8(value: 0, remaining: remaining))
            }
            let mask = UInt8((1 << (7 - remaining)) - 1)
            value = UInt32(first & mask)
            i += 1
            while i < count, bytes[i] != 0, remaining > 0 {
                value = (value << 6) + UInt32(bytes[i] & 0x3F)
                i += 1
                remaining -= 1
            }
            if remaining == 0 {
                codePoints.append(value)
            }
        }

        return (codePoints, GrammarPartialUTF8(value: value, remaining: remaining))
    }
}
