import Foundation

// トークン水準の文法拘束 (SPEC §6 GEN-1 / GEN-5 / GEN-6)。
//
// 規範は参照実装 `~/LLM/llama.cpp` のピン `34af94cd9`:
// `src/llama-grammar.cpp` の `llama_grammar_apply_impl` /
// `llama_grammar_accept_impl`、`src/llama-vocab.cpp` の
// `cache_token_to_piece`、`common/sampling.cpp` の `grammar_should_apply`。
//
// ここは「橋」だけ。文法の照合そのものは `GrammarMatcher` が持ち、拘束の
// 当て方 (棄却サンプリング GEN-7) は `Sampler` / `ConstraintGate` が持つ。
// この型は語彙 (ID → バイト列) と照合器を束ねて `GenerationConstraint` に
// する役だけを担い、停止トークンを特別扱いしない — 終端の規則は
// `mayEndHere` を通じて `ConstraintGate` の側にある。

// MARK: - 語彙の piece 表

/// トークン ID → そのトークンが出力に足すバイト列。
///
/// 参照実装の `llama_vocab` が読み込み時に作る `cache_token_to_piece`
/// (`token_to_piece_for_cache(id, /* special = */ true)`) に対応する。
/// **special = true** が要点で、`<|tool_call>` や `<|"|>` といった制御
/// マーカーも「そのマーカーの綴りそのもの」として表に入る — GEN-8 の
/// tool call 文法はそれらのリテラルで書かれているので、ここで落とすと
/// 文法は一致しようがない。
///
/// piece は `String` ではなく**バイト列**である。BPE のバイトフォール
/// バック `<0xXX>` は 1 バイトしか足さないので、単体では UTF-8 として
/// 不正なことがあり、`<0x00>` は NUL 始まりになる。参照実装が
/// `llama_grammar_apply_impl` で「空 piece」と「先頭が NUL の piece」を
/// 文法にかける前に落とすのはそのためで、こちらも同じ扱いをする。
///
/// **費用と共有**: 262 144 個ぶんの `convertIdToToken` + 変換 + UTF-8 の
/// 復号で、実測 0.4 秒 (M3 Pro, debug)。要求ごとに払う額ではないので
/// `shared(for:)` でプロセス内に 1 本だけ作って使い回す。値型だが中身は
/// 配列の CoW なので、複製も受け渡しも参照 1 個ぶんしかかからない。
/// piece の実体は合計 2.0 MB しかないが、要素ごとに配列を 1 本ずつ確保する
/// ので実メモリはその数倍になる (`GrammarMatcher.RejectContext` が
/// `[[UInt32]]` を要求するため、復号表も平らにはできない)。
public struct GrammarVocabulary: Sendable {
    /// 表の要素数 (= 語彙数)。
    public var count: Int { pieces.count }

    /// ID 順の piece 表。
    let pieces: [[UInt8]]

    /// 文法にかけてよい候補だけを並べた不変の候補列。`index` はトークン ID。
    /// 空 piece と NUL 始まりの piece はここに入らない (= 常に拒否)。
    ///
    /// 語彙全体のマスク 1 回ごとにこの配列を組み直すと 26 万本の配列確保に
    /// なるので、表と一緒に 1 度だけ作って共有する。`piece` は `pieces` と
    /// 同じバッファを指すだけ (CoW) なので、二重に持っても中身は 1 部。
    let candidates: [GrammarCandidate]

    /// `candidates` と同じ並びの、**空の `partialUTF8` から復号した**
    /// コードポイント列 (GEN-7 の棄却経路用)。
    ///
    /// `GrammarMatcher.rejectedIndices` は候補ごとに毎回 UTF-8 を復号し直す
    /// (参照実装 `llama_grammar_apply_impl` と同じ形)。実測ではそれが棄却
    /// 1 回の 43% を占める。復号は**文法の状態に依らない** — 唯一の例外は
    /// 直前の piece から続く多バイト列 (`GrammarMatcher.partialUTF8`) で、
    /// そこが空でありさえすれば `decodeUTF8` の出力は piece だけで決まる。
    /// なので語彙表と同じ寿命で 1 度だけ作って使い回す。
    let decodedCodePoints: [[UInt32]]
    /// `decodedCodePoints` と対の、復号後に残った不完全列。
    let decodedPartials: [GrammarPartialUTF8]
    /// `candidates` と同じ並びのトークン ID (棄却経路が slot で引く)。
    let candidateTokenIDs: [Int32]
    /// 棄却経路の初期作業列 (全 slot、offset 0)。平らな配列 1 本なので、
    /// 毎回作り直しても要素ごとの確保は起きないが、共有できるものを毎回
    /// 作る理由も無い。
    let baseWork: [GrammarMatcher.WorkCandidate]

    /// 表を 1 本作る。**要求ごとに作らないこと** — `shared(for:)` を使うか、
    /// tokenizer の隣に 1 個持って毎要求に渡す。
    public init(_ tokenizer: GFTokenizer) {
        let size = tokenizer.vocabSize
        var pieces = [[UInt8]](repeating: [], count: size)
        var candidates: [GrammarCandidate] = []
        var decodedCodePoints: [[UInt32]] = []
        var decodedPartials: [GrammarPartialUTF8] = []
        var candidateTokenIDs: [Int32] = []
        var baseWork: [GrammarMatcher.WorkCandidate] = []
        candidates.reserveCapacity(size)
        decodedCodePoints.reserveCapacity(size)
        decodedPartials.reserveCapacity(size)
        candidateTokenIDs.reserveCapacity(size)
        baseWork.reserveCapacity(size)

        for id in 0..<size {
            guard let token = tokenizer.tokenizer.convertIdToToken(id) else { continue }
            // `<0xXX>` はそのバイト 1 個。それ以外は metaspace `▁` を空白に
            // 直した綴り (`GemmaDecoding` の `Replace` 段)。特殊マーカーは
            // 綴りがそのまま piece になる — 変換で消してはならない。
            let piece: [UInt8]
            if let byte = GemmaDecoding.byteValue(token) {
                piece = [byte]
            } else {
                piece = Array(GemmaDecoding.fragment(token).utf8)
            }
            pieces[id] = piece
            // 参照実装 `llama_grammar_apply_impl` と同じ足切り。
            guard !piece.isEmpty, piece[0] != 0 else { continue }
            let slot = candidates.count
            candidates.append(GrammarCandidate(index: id, tokenID: Int32(id), piece: piece))
            // 空の `partialUTF8` からの復号。棄却経路の速い側がこれを使う。
            let decoded = GrammarMatcher.decodeUTF8(piece, GrammarPartialUTF8())
            decodedCodePoints.append(decoded.codePoints)
            decodedPartials.append(decoded.partial)
            candidateTokenIDs.append(Int32(id))
            baseWork.append(GrammarMatcher.WorkCandidate(slot: slot, offset: 0))
        }

        self.pieces = pieces
        self.candidates = candidates
        self.decodedCodePoints = decodedCodePoints
        self.decodedPartials = decodedPartials
        self.candidateTokenIDs = candidateTokenIDs
        self.baseWork = baseWork
    }

    /// トークン 1 個の piece。範囲外の ID は空バイト列 (= 文法が常に拒む)。
    public func piece(for tokenID: Int32) -> [UInt8] {
        guard tokenID >= 0, Int(tokenID) < pieces.count else { return [] }
        return pieces[Int(tokenID)]
    }

    /// プロセス内で 1 度だけ作って共有する。`identity` は tokenizer の出所
    /// (既定はピンされたモデル ID)。別の tokenizer を読むテストのために鍵に
    /// してあるだけで、通常の経路は既定値でよい。
    public static func shared(for tokenizer: GFTokenizer,
                              identity: String = GFTokenizer.modelID) -> GrammarVocabulary {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if let existing = cache[identity], existing.count == tokenizer.vocabSize {
            return existing
        }
        let built = GrammarVocabulary(tokenizer)
        cache[identity] = built
        return built
    }

    private static let cacheLock = NSLock()
    nonisolated(unsafe) private static var cache: [String: GrammarVocabulary] = [:]
}

// MARK: - 遅延文法のトリガ

/// 遅延文法 (GEN-5) の起動条件。
///
/// 現状は**トークントリガ**だけ。参照実装はこれに加えてテキスト
/// (正規表現) トリガを持ち、そちらは起動までの piece を貯めて、一致した
/// 位置から貯めた分を文法に流し直す (`trigger_buffer` /
/// `trigger_buffer_positions` の再生)。GEN-5 が名指しするトリガは
/// `<|tool_call>` というトークンそのものなので、テキストトリガは**作って
/// いない**。足すときはこの型に `patterns` を足し、
/// `GrammarTokenConstraint.accept(tokenID:)` の待機分岐に再生を書く —
/// 待機分岐は 1 か所しかなく、`isArmed` から先の経路は変わらない。
public struct GrammarTrigger: Sendable, Equatable {
    /// これらのトークンのどれかが受理された瞬間に文法が起動する。
    public var tokenIDs: Set<Int32>

    public init(tokenIDs: Set<Int32>) {
        self.tokenIDs = tokenIDs
    }

    public static func token(_ tokenID: Int32) -> GrammarTrigger {
        GrammarTrigger(tokenIDs: [tokenID])
    }
}

// MARK: - 拘束本体

/// `GrammarMatcher` と語彙表を束ねて `GenerationConstraint` にする。
///
/// **呼び出し側 (サーバー) の手順**
///
/// 1. **構築**: `GrammarVocabulary.shared(for:)` を 1 度取り、要求ごとに
///    `GrammarTokenConstraint(grammar:vocabulary:trigger:)` を作る。
///    `tool_choice: auto` は `trigger: .token(tokenizer.toolCallStartID)`
///    (GEN-5)、`required` / 名前指定 / `response_format` は `trigger: nil`
///    で最初から拘束する (GEN-4)。
/// 2. **供給**: 生成ループが受理した全トークンを順に `accept(tokenID:)`
///    へ渡す (`ConstraintGate.accept` 経由。`RawCompletion` は既にそうする)。
/// 3. **抑止 (GEN-6)**: 思考チャンネルの状態を知っているのは
///    `StructuredAssistantDecoder` の側なので、そちらが
///    `setSuppressed(true)` を `<|channel>thought` に入った時点で、
///    `setSuppressed(false)` を `<channel|>` で閉じた時点で呼ぶ。
///    抑止中は文法を**当てず供給もしない** — 思考の中に出たトリガでは
///    起動しない。非遅延の文法は抑止されない (参照実装
///    `grammar_should_apply` が `grammar_lazy` のときだけ思考状態を見る)。
///
/// **停止トークンは扱わない。**参照実装は `llama_grammar_apply_impl` の中で
/// EOG を文法にかけずに `allow_eog` だけで通す。こちらではその規則は
/// `mayEndHere` として外に出ていて、適用するのは `ConstraintGate` である。
/// 注意: 現在の `ConstraintGate.allows` は `mayEndHere` が真のとき停止
/// トークンを素通しせずこの型の `allows` に落とすので、完了した文法でも
/// `<turn|>` の piece が文法にかかって拒まれる。参照実装と同じにするには
/// ゲート側が「停止 ID なら `mayEndHere` を返して終わり」にする必要がある
/// (GEN-7 の担当。ここでは特別扱いしない)。
///
/// スレッド安全性: 生成ループ 1 本から順に呼ばれる前提で、内部に錠は無い
/// (`StubConstraint` など既存の実装と同じ約束)。
///
/// **`GrammarMatcher` への結合**: 速いマスク経路は `rejectedIndices` では
/// なく、その内側の `rejectCandidates` を復号済みの `RejectContext` 付きで
/// 直に呼ぶ (`rejectedIndices` は候補ごとに毎回復号し直す作りで、復号を
/// 外から渡す入口が無い)。`rejectedIndices` が候補を組み立てる部分 —
/// 空 / NUL の足切りと slot の採番 — はこちら側で再現している。照合器の
/// 側でその前処理が変わったら、ここも合わせること。等価性は
/// `GrammarTokenConstraintTests` が全語彙で毎回突き合わせる。
public final class GrammarTokenConstraint: GenerationConstraint, @unchecked Sendable {
    private let vocabulary: GrammarVocabulary
    private let trigger: GrammarTrigger?
    private var matcher: GrammarMatcher
    /// 参照実装の `awaiting_trigger`。非遅延では最初から false。
    private var awaitingTrigger: Bool
    private var suppressed = false

    public init(grammar: GBNFGrammar,
                vocabulary: GrammarVocabulary,
                trigger: GrammarTrigger? = nil) throws {
        self.vocabulary = vocabulary
        self.trigger = trigger
        self.awaitingTrigger = trigger != nil
        self.matcher = try GrammarMatcher(grammar)
    }

    public convenience init(_ source: String,
                            root: String = "root",
                            vocabulary: GrammarVocabulary,
                            trigger: GrammarTrigger? = nil) throws {
        try self.init(grammar: try GBNFGrammar(source, root: root),
                      vocabulary: vocabulary,
                      trigger: trigger)
    }

    /// 遅延文法か (GEN-5)。
    public var isLazy: Bool { trigger != nil }
    /// 起動済みか。非遅延は最初から `true`。
    public var isArmed: Bool { !awaitingTrigger }
    /// 思考中で抑止されているか (GEN-6)。非遅延は決して抑止されない。
    public var isSuppressed: Bool { suppressed && isLazy }
    /// いま文法が効いているか (= 起動済み かつ 抑止されていない)。
    public var isApplied: Bool { isArmed && !isSuppressed }

    /// GEN-6。呼び出し側 (`StructuredAssistantDecoder` を持つ層) が駆動する。
    /// この型は思考チャンネルの状態を推測しない。
    public func setSuppressed(_ suppressed: Bool) {
        self.suppressed = suppressed
    }

    // MARK: - GenerationConstraint

    /// 未起動・抑止中は常に真 (参照実装は `apply` 自体を呼ばないので EOG も
    /// 通る)。起動していれば「文法がここで終われるか」= `isComplete`。
    public var mayEndHere: Bool {
        isApplied ? matcher.isComplete : true
    }

    public func allows(tokenID: Int32) -> Bool {
        guard isApplied else { return true }
        let piece = vocabulary.piece(for: tokenID)
        // 参照実装 `llama_grammar_apply_impl` の足切りと同じ。
        guard !piece.isEmpty, piece[0] != 0 else { return false }
        return matcher.allows(piece: piece, tokenID: tokenID)
    }

    /// 全語彙のマスク。交差計算は候補集合につき 1 回しか走らない。
    ///
    /// 速い側は語彙表の復号 (`decodedCodePoints`) をそのまま使う。
    /// `GrammarMatcher.rejectedIndices` は候補ごとに毎回 UTF-8 を復号し直す
    /// ので、262k 候補では棄却 1 回の 43% が復号になる (実測 0.349 秒中
    /// 0.148 秒)。復号が使い回せるのは
    /// **`matcher.partialUTF8.remaining == 0` のときだけ**で、根拠は
    /// `decodeUTF8` の定義そのもの: `start.remaining` が 0 なら前置ループ
    /// (`remaining > 0` が条件) は回らず、`start.remaining > 0` の分岐も
    /// 通らないので、出力は `bytes` だけの関数になる。残りが 0 でない
    /// (= 直前の piece が多バイト文字の途中で切れた) ときは合成せずに
    /// `rejectedIndices` へ落とす — 継ぎ足しでも同じ結果を作れるはずだが、
    /// 「壊れた列は途中結果ごと捨てる」という `decodeUTF8` の畳み込みまで
    /// 再現する必要があり、等価だと言い切れないので採らない。生成の
    /// ほとんどの手はこちらに来ないので、遅い側でよい。
    public func fillAllowedMask(_ allowed: UnsafeMutableBufferPointer<Bool>) throws {
        guard isApplied else {
            allowed.update(repeating: true)
            return
        }
        // 既定は拒否。ここに残るのは表に無い ID と、空 / NUL 始まりの piece
        // (参照実装 `llama_grammar_apply_impl` が文法にかける前に落とす分)。
        allowed.update(repeating: false)
        let limit = allowed.count
        let candidates = vocabulary.candidates
        for candidate in candidates where candidate.index < limit {
            allowed[candidate.index] = true
        }

        guard matcher.partialUTF8.remaining == 0 else {
            for index in matcher.rejectedIndices(candidates) where index < limit {
                allowed[index] = false
            }
            return
        }

        // `rejectedIndices` が内側でやることと同じで、復号だけが済んでいる。
        // 空 / NUL の候補は `candidates` に入っていないので前置きも要らない。
        let context = GrammarMatcher.RejectContext(
            rules: matcher.grammar.rules,
            decoded: vocabulary.decodedCodePoints,
            partials: vocabulary.decodedPartials,
            tokenIDs: vocabulary.candidateTokenIDs
        )
        for work in GrammarMatcher.rejectCandidates(matcher.stacks, vocabulary.baseWork, context) {
            let index = candidates[work.slot].index
            if index < limit { allowed[index] = false }
        }
    }

    /// 参照実装 `llama_grammar_accept_impl` の状態機械。
    ///
    /// - 抑止中は何もしない (供給しない = トリガも見ない)。
    /// - 待機中はトリガだけを見る。トリガに当たったらその場で起動し、
    ///   **そのトークン自身を**文法に食わせる (参照実装は貯めた
    ///   バッファを捨てて `llama_grammar_accept_token` を呼ぶ)。
    /// - 起動済みなら 1 トークン進める。スタックが全滅したら
    ///   `GBNFError.noSurvivingStacks` が上がる — 呼び出し側 (`Sampler`) が
    ///   失敗として扱う。飲み込んではならない。
    public func accept(tokenID: Int32) throws {
        guard !isSuppressed else { return }

        if awaitingTrigger {
            guard trigger?.tokenIDs.contains(tokenID) == true else { return }
            awaitingTrigger = false
            try matcher.accept(piece: vocabulary.piece(for: tokenID), tokenID: tokenID)
            return
        }

        try matcher.accept(piece: vocabulary.piece(for: tokenID), tokenID: tokenID)
    }
}
