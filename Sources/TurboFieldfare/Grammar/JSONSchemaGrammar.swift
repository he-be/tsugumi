import Foundation

/// JSON Schema → GBNF 変換 (SPEC §6 GEN-1 / GEN-3)。
///
/// 参照実装は `~/LLM/llama.cpp` のピン `34af94cd9`、
/// `common/json-schema-to-grammar.cpp`。規則の並び・規則名・空白の入れ方まで
/// あちらに合わせる (期待値は `tests/test-json-schema-to-grammar.cpp` から取る)。
///
/// **逸脱 (SPEC §12 に登録)**: オブジェクトのプロパティは宣言順ではなく
/// **キーの昇順**で並べる。理由は 2 つあり、どちらも単独で決定的である。
/// 1. こちらのスキーマ表現 `JSONValue.object` は Swift の `Dictionary` で、
///    宣言順を保持していない。
/// 2. サーバーのチャットテンプレートは tool call の引数を Jinja の `dictsort`
///    で描く (`server_chat_template.jinja`)。生成物がキー昇順でなければ
///    「描き直し == 生成」(SPEC §7 INV-1) が破れる。
public enum JSONSchemaGrammarDialect: Sendable, Equatable {
    /// 厳密な JSON。キーは `"引用符付き"`、文字列は `"…"`。`response_format` 用。
    case json
    /// サーバーのテンプレートが描き `GemmaToolCallParser` が読む引数の方言。
    /// オブジェクトのキーは裸 (`[A-Za-z0-9_\-.$]+`)、文字列値は JSON 形式
    /// `"…"` と学習形式 `<|"|>…<|"|>` のどちらでもよい。
    case gemmaToolArguments
}

/// 変換の結果。`approximations` は GEN-2 で近似に落とした箇所の記録で、
/// サーバーはこれをログに出す。空なら参照実装と同じ厳密な変換ができている。
public struct JSONSchemaGrammarResult: Sendable, Equatable {
    public let grammar: String
    public let approximations: [String]

    public init(grammar: String, approximations: [String]) {
        self.grammar = grammar
        self.approximations = approximations
    }
}

/// 厳密モード (テスト専用) が「参照実装ならここで 400 にしていた」と言うための型。
/// 公開の入口 (`JSONSchemaGrammar.grammar(for:dialect:)`) は
/// **スキーマの内容では決して throw しない** (SPEC §6 GEN-2)。
public struct JSONSchemaGrammarStrictError: Error, Equatable {
    public let approximations: [String]
}

public enum JSONSchemaGrammar {
    /// GEN-2 の入口。表現できないスキーマ要素は拘束できる近似に落ち、
    /// 落ちた事実は `approximations` に載る。決して throw しない。
    public static func grammar(
        for schema: JSONValue,
        dialect: JSONSchemaGrammarDialect = .json
    ) -> JSONSchemaGrammarResult {
        JSONSchemaGrammarResult(grammar: "", approximations: ["not-implemented"])
    }
}
