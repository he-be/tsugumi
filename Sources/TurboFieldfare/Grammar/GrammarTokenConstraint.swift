import Foundation

// トークン水準の文法拘束 (SPEC §6 GEN-1 / GEN-5 / GEN-6)。
//
// 規範は参照実装 `~/LLM/llama.cpp` のピン `34af94cd9`:
// `src/llama-grammar.cpp` の `llama_grammar_apply_impl` /
// `llama_grammar_accept_impl`、`src/llama-vocab.cpp` の
// `cache_token_to_piece`、`common/sampling.cpp` の `grammar_should_apply`。
//
// ここは「橋」だけ。文法の照合そのものは `GrammarMatcher`、拘束の当て方
// (棄却サンプリング GEN-7) は `Sampler` / `ConstraintGate` が持つ。

// MARK: - 未実装 (赤コミット用)

/// 実装前の入口が投げる誤り。緑コミットで消える。
public enum GrammarTokenConstraintError: Error, CustomStringConvertible {
    case notImplemented(String)

    public var description: String {
        switch self {
        case .notImplemented(let what): return "未実装: \(what)"
        }
    }
}

// MARK: - 語彙の piece 表

/// トークン ID → そのトークンが出力に足すバイト列。
///
/// 参照実装の `llama_vocab` が読み込み時に作る `cache_token_to_piece`
/// (`token_to_piece_for_cache(id, /* special = */ true)`) に対応する。
/// **special = true** が要点で、`<|tool_call>` や `<|"|>` といった制御
/// マーカーも「そのマーカーの文字列そのもの」として表に入る — GEN-8 の
/// tool call 文法はそれらのリテラルで書かれているので、ここで落とすと
/// 文法が一致しようがない。
///
/// piece は `String` ではなく**バイト列**である。BPE のバイトフォールバック
/// `<0xXX>` は 1 バイトだけを足すので、単体では UTF-8 として不正なことが
/// あり、`<0x00>` は NUL 始まりになる。参照実装が
/// `llama_grammar_apply_impl` で「空 piece」と「先頭が NUL の piece」を
/// 文法にかける前に落とすのはそのためで、こちらも同じ扱いをする。
public struct GrammarVocabulary: Sendable {
    /// 表の要素数 (= 語彙数)。
    public var count: Int { pieces.count }

    /// ID 順の piece 表。
    let pieces: [[UInt8]]

    /// 文法にかけてよい候補だけを並べた不変の候補列。`index` はトークン ID。
    /// 空 piece と NUL 始まりの piece はここに入らない (= 常に拒否)。
    let candidates: [GrammarCandidate]

    /// 表を 1 本作る。**要求ごとに作らないこと** — `shared(for:)` を使うか、
    /// tokenizer の隣に 1 個持って毎要求に渡す。
    public init(_ tokenizer: GFTokenizer) {
        _ = tokenizer
        self.pieces = []
        self.candidates = []
    }

    /// トークン 1 個の piece。範囲外の ID は空バイト列。
    public func piece(for tokenID: Int32) -> [UInt8] {
        _ = tokenID
        return []
    }

    /// プロセス内で 1 度だけ作って共有する。`identity` は tokenizer の出所
    /// (既定はピンされたモデル ID)。
    public static func shared(for tokenizer: GFTokenizer,
                              identity: String = GFTokenizer.modelID) -> GrammarVocabulary {
        _ = identity
        return GrammarVocabulary(tokenizer)
    }
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
/// `GrammarTokenConstraint.accept(tokenID:)` の待機分岐に再生を書く。
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
public final class GrammarTokenConstraint: GenerationConstraint, @unchecked Sendable {
    public init(grammar: GBNFGrammar,
                vocabulary: GrammarVocabulary,
                trigger: GrammarTrigger? = nil) throws {
        _ = (grammar, vocabulary, trigger)
        throw GrammarTokenConstraintError.notImplemented("GrammarTokenConstraint")
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
    public var isLazy: Bool { false }
    /// 起動済みか。非遅延は最初から `true`。
    public var isArmed: Bool { false }
    /// 思考中で抑止されているか (GEN-6)。
    public var isSuppressed: Bool { false }
    /// いま文法が効いているか (= 起動済み かつ 抑止されていない)。
    public var isApplied: Bool { false }

    /// GEN-6。呼び出し側が駆動する。
    public func setSuppressed(_ suppressed: Bool) {
        _ = suppressed
    }

    public var mayEndHere: Bool { false }

    public func allows(tokenID: Int32) -> Bool {
        _ = tokenID
        return false
    }

    public func fillAllowedMask(_ allowed: UnsafeMutableBufferPointer<Bool>) throws {
        _ = allowed
        throw GrammarTokenConstraintError.notImplemented("fillAllowedMask")
    }

    public func accept(tokenID: Int32) throws {
        _ = tokenID
        throw GrammarTokenConstraintError.notImplemented("accept")
    }
}
