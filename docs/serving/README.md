# サービング実用化 (Serving) — 計画

QAT ウェイト・Vision・MTP に続く 4 つ目の改修。今回の対象は数字ではなく**実用性**:
CLI と Server を、Vision・MTP・Reasoning が揃った状態で、日常のクライアント
(pi / OpenCode / ブラウザデモ) から普通に使える形にする。

**現在地: 計画のみ。実装は未着手。**

## 出発点 (2026-08-19)

ベースの改修は閉じた: 32 スロットの運用点で実際の Vision タスクが 30 t/s 超。
性能系の続き (I/O を床から外す仕事) は [mtp/32 §7](../mtp/32-M8-A-ROWS-SPLIT.md)
が引き継ぎ先で、**この計画では扱わない**。

Server は既に多くが入っている ([OPENAI_SERVER.md](../OPENAI_SERVER.md)):
streaming SSE / function tools / 画像 / プロンプトキャッシュ (`cached_tokens`) /
MTP (`--draft-block-size`) / 4K〜128K。実害のある穴は [TODO.md](../../TODO.md) が
調査済みの 3 つで、**根は 1 つ** (tools 宣言時のチャットテンプレート制約):

1. **Server に thinking が無い。**`ServerArguments.swift` にフラグが無く、
   `ServerInference.swift:809/853` は `enableThinking` を渡さずテンプレートを呼ぶ。
   ゴール条件「Server で Reasoning ON」は今も CLI でしか回せない。
2. **pi 対話モードで画像が使えない。**pi は毎要求に built-in tools を宣言し、
   サーバーは画像 + tools を 400 で拒否する (`OpenAIModels.swift:449`)。
3. **tools 宣言時は thinking が構造的に無効。**`Tokenizer.swift:453` の
   `enable_thinking: false` ハードコード。

CLI は単発実行のみ: 対話モード無し、`--thinking` は `--messages-file` 限定、
ターンをまたぐ KV 再利用無し。

## ゴール

定性 3 つ + 従来の定量 1 つ:

| | 条件 |
| --- | --- |
| G1 | **pi の既定セッション (tools ON) で**、画像を貼れて、Reasoning ON で、MTP が効いた状態で使える |
| G2 | **CLI で多ターンの Vision 対話**が成立し、2 ターン目以降は前ターンの prefix を再 prefill しない |
| G3 | **フラグ無し起動**が 16GB 機の運用点になる (今は runbook の長いコマンドをコピペしている) |
| G4 | mtp のゴール条件 (単発 Vision + Reasoning ON、tg ≥ 30 t/s) を **Server 実測**で回収する — 今までは CLI 実測しかない |

採点の流儀は mtp を引き継ぐ: 出力不変が要件の変更は **md5**、速度が絡む変更は
**同一セッション A/B**。機能の計画なので、各段の出口条件は「実クライアントで通ること」を最優先にする。

## マイルストーン

### S1 — Server Reasoning ON (tools 無し経路) 【1〜2 日】

goal 条件そのものであり、tools が絡まない限り純粋な配線なので最初にやる。

- `--thinking` フラグ (プロセス既定) + 要求ごとの上書きフィールド。
  **API 表面の設計判断が 1 つ**: OpenAI の `reasoning_effort` に載せるか、
  vLLM 系の `chat_template_kwargs` / 独自フィールドか。pi
  (`supportsReasoningEffort`) と OpenCode の互換挙動を確認してから決める。
- `ServerInference.swift:809/853` の `applyChatTemplate` に `enableThinking` を配線。
- **応答側の分離**: 思考部分を `reasoning_content` (DeepSeek/vLLM 慣行) として
  本文と別のフィールドで stream する。テンプレートの thought channel
  (`<|think|>` 系) の境界を生成テキストから検出する処理が要る。
  `max_tokens` の消費・`stop`・完了ログの token 内訳もここで決める。
- 出口: pi / OpenCode から Reasoning ON の Vision 単発が Server で通る。G4 を実測。

### S2 — テンプレート検定 (probe、実装なし) 【半日〜1 日】

穴 2 と 3 は根が同じなので、実装前に事実を 1 本の文書で確定する
(mtp の [29-M8-B-PROBE](../mtp/29-M8-B-PROBE.md) と同じ型)。

- tools 宣言時のテンプレートが (a) `enable_thinking: true` (b) 画像 content part
  を描画できるか。参照実装 (transformers、`scratch/mtp-ref/`) と突き合わせる。
- 出口: S3 の設計判断ができる状態。「描画できない」という結論も可 —
  その場合の S3 は degrade 案に切り替わる。

### S3 — Vision × tools、thinking × tools 【1〜3 日、S2 の結果次第】

- 描画できるなら: `encodeToolChat` に画像と `enableThinking` を配線し、
  `OpenAIModels.swift:449` の拒否を外す。
- できないなら: 画像を含むターンだけ tool 無しテンプレートへ落とす
  (この要求だけ tool 呼び出し不可、応答は普通に返す) か、400 のまま
  pi 側へ upstream 要望を出すか。**どちらにするかは S2 の事実を見てから。**
- 出口: pi 既定セッション (tools ON) で画像を貼って回答が返る (G1)。
  既存の tool スキーマ/呼び出しテストは全通過のまま。

### S4 — CLI 対話モード 【2〜4 日、S1〜S3 と独立に並行可】

- REPL: 多ターン、streaming、`/image` で画像添付、`/think` トグル、
  ターンごとの stats footer、Ctrl-C は生成のみ中断 (セッションは残る)。
- **プロセス内 KV 持ち回り**: 前ターンの prefix を再 prefill しない
  (`ServerPromptCache` 相当のことを in-process でやる)。
- `--thinking` を `--prompt` 経路にも。セッションの保存/再開
  (`--messages-file` 形式の JSON と往復できる形)。
- 出口: 多ターン Vision + Reasoning 対話が単発と同じ t/s で回り、
  2 ターン目の prefill が cached 分だけ短いことを実測 (G2)。

### S5 — 運用点を既定に 【1 日】

- `--expert-cache-slots` の既定をデバイスのメモリ予算から自動選択
  (`ExpertCacheBudget` の算術は既にある)。drafter 入りモデルなら
  `--draft-block-size` 既定 4。明示フラグは常に勝つ。
- Server 起動時 warmup (`Scripts/demo/serve.py` がやっている text + image の
  2 発) をサーバー本体の `--warmup` へ。
- 出口: フラグ無しの `TurboFieldfareServer --model X` が M3 Pro で
  [SERVER_RUNBOOK](../SERVER_RUNBOOK.md) §1(a) と同じ数字を出す (G3)。
  runbook の手順が短くなる。

## Backlog (この計画ではやらない。必要になったら昇格)

| 項目 | メモ |
| --- | --- |
| structured output (`json_schema`) | クライアント需要が出てから |
| `p_min` 適応ドラフト ([mtp/28 §4-2](../mtp/28-M8-PROPOSAL.md)) | 日本語散文で MTP が空回りする問題への保険。効果測定は同一セッション A/B |
| I/O を床から外す ([mtp/32 §7](../mtp/32-M8-A-ROWS-SPLIT.md)) | 性能系の再開点 |
| 128K の実測 | Server 側は対応済み。`ExpertCacheBudget` のメモリ判定の実測だけ残り |

## 順序の理由

S1 が先なのはゴール条件そのものだから。S2/S3 は根が同じ制約なので probe を
1 本挟んでまとめて潰す。S4 は Server と衝突しないので並行できる。
S5 は全部の後 — 既定値は、動く形が確定してから固めるものだから。

## 運用ルール

[mtp/README.md](../mtp/README.md) と同じ: **1 文書 = 1 論点** (目安 200 行)、
進捗は S1 → `10-S1-RESULTS.md` のように**文書を増やして**書きこの索引に 1 行足す、
設計の前提が変わったら本文を直す (履歴は git log)。
