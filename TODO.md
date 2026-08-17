# TODO

## pi (コーディングエージェント) の対話モードで画像が使えない

### 状況 (2026-08-17、**実測**)

`~/.pi/agent/models.json` の `local-turbofieldfare` プロバイダ
(`http://127.0.0.1:8091/v1`、`scratch/gemma4-qat.gturbo` を配信) は
テキスト・画像とも非対話 (`pi --print --no-tools`) では動作を確認済み。

対話モード (既定、tools 有効) で画像を送ると **400 `unsupported_content`**
(`images cannot be combined with tools`) になる。pi は対話セッションの
毎リクエストに built-in tools (`read`/`bash`/`edit`/`write`) を宣言するため、
`Sources/TurboFieldfareServer/Core/OpenAIModels.swift:445-450` の
「画像 + tools は拒否する」検証 (`PLAN_VISION.md` §0-I-4 で明示的に決めた仕様。
tool-calling 用チャットテンプレートが画像 content part を描画できないことが理由) に
必ず引っかかる。

pi 側に「画像を含むターンだけ `tool_choice: none` を送る」機構はない
(`cli/args.js` / `core/sdk.js` を確認、`tool_choice` を露出する経路自体が存在しない)。
現状の回避策は **セッション全体で `--no-tools` (`-nt`) を付けて起動する**ことのみで、
その場合は pi の bash/read/edit/write が使えなくなる。

### 選択肢 (未着手)

| 案 | 内容 | コスト |
| --- | --- | --- |
| 現状維持 | ツール使用と画像閲覧を別セッションに分ける運用で妥協する | 0 (今のワークフロー) |
| サーバー側を緩和 | tools 宣言があっても画像を許可する。ただし tool-calling 用チャットテンプレートを画像混在に対応させる改修が要る (`PLAN_VISION.md` がスコープ外と明示した領域) | 中〜大。Gemma4 のツール用テンプレート自体の検証が要る |
| pi 側の対応を待つ/出す | 画像を含むターンで `tool_choice: none` を自動送出する機能を pi に入れる (upstream 要望) | 中。pi 側の変更で、こちらのリポジトリの手が届く範囲外 |

まだどれも着手していない。次にやるならまず「サーバー側を緩和」の実現可能性
(テンプレートが画像 + tool schema を両方描画できるか) を先に調べる。

## サーバーで Reasoning (thinking) on / 128k context を使いたい

### 状況 (2026-08-17、**調査済み・未着手**)

どちらも `TurboFieldfareServer` には現状実装がない。CLI 側にのみ部分的に存在する。

**Reasoning (thinking) on:**

- `ServerArguments.swift` に `--thinking` 相当のフラグが存在しない。
  `ServerInference.swift:716-764` は `tokenizer.encodeToolChat(...)` /
  `applyChatTemplate(...)` を `enableThinking` を渡さずに呼ぶため常に `false` 扱い。
  OpenAI 互換のリクエストボディにも `reasoning` / `reasoning_effort` フィールドは
  未実装 (`OpenAIModels.swift` に該当なし)。`docs/OPENAI_SERVER.md:132-139` の pi 向け
  サンプル設定でも `"supportsReasoningEffort": false` と明記。
- tools 宣言付きリクエストは `encodeToolChat` を通るが、`Tokenizer.swift:453` で
  `enable_thinking: false` がハードコードされている。仮にサーバーに thinking
  フラグを足しても、tool 呼び出しがある限り無効化される
  (上の「pi で画像が使えない」問題と根が同じ: tools 宣言時のテンプレート制約)。
- 使えるのは CLI (`TurboFieldfareCLI --messages-file <path> --thinking on`,
  `Args.swift:151-156,319-325`, `Run.swift:106-107`) のみ。
  `--messages-file` 経由限定 (`--prompt` には効かない)。

**128k context:**

- サーバーの `--max-context` は `[4096, 8192, 16384, 32768, 65536]` のホワイトリスト
  固定 (`ServerArguments.swift:136`、デフォルト 16384)。131072 を渡すと
  `"--max-context is not supported"` で即エラー。Mac アプリの設定 UI
  (`AppContextLengthOption.swift`) も同じ 5 段階まで。
- 128k という上限はコード中どこにも存在しない (`131072`/`128k` で全文検索してもヒットなし)。
- CLI だけは `> 0` しかチェックしない (`Args.swift:242-247`) ので
  `--max-context 131072` は起動時の引数検証は通る。成否は
  `ExpertCacheBudget.swift:55-100` が FP16 KV サイズ見積もりとデバイスの
  メモリ予算を照合して判定する (未実測)。
- `ArchConfig.gemma4_26B_A4B` (`ModelTypes.swift:78-92`) の
  `fullRopeTheta: 1_000_000.0` は Gemma 系の長文脈 (~128K) チェックポイントで
  よく使われる RoPE スケーリング値なので、重み自体は 128k を想定している
  可能性が高いが、ランタイムの配線がそこまで対応していない。

### 必要な変更 (未着手)

| やりたいこと | 変更箇所 |
| --- | --- |
| サーバーで reasoning on | `ServerArguments.swift` に `--thinking` フラグ追加 + `ServerInference.swift` の `applyChatTemplate` 呼び出しに `enableThinking` を配線。tools 宣言時は `Tokenizer.swift:453` の `enable_thinking: false` ハードコードとどう両立させるか要検討 |
| サーバーで 128k context | `ServerArguments.swift:136` のホワイトリストに `131072` を追加 (usage 文言 `:25` も更新)。`ExpertCacheBudget` でのメモリ実測が必要 |
| CLI で 128k context (実験) | 変更不要、`--max-context 131072` を試してメモリ判定の挙動を確認するだけ |
