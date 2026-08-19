# サービング実用化 (Serving) — 計画

QAT ウェイト・Vision・MTP に続く 4 つ目の改修。今回の対象は数字ではなく**実用性**:
CLI と Server を、Vision・MTP・Reasoning が揃った状態で、日常のクライアント
(pi / OpenCode / ブラウザデモ) から普通に使える形にする。

**現在地: S1〜S3 完了 (2026-08-19)。残りは S4 (CLI 対話) と S5 (既定値)。**

| 段 | 状態 | 文書 |
| --- | --- | --- |
| S1 | 実装・テスト済み。**G4 は 80 スロットで達成 (37.5 t/s)、常用点の 32 スロットでは 28.6 t/s で未達**。思考 ON のプロンプトキャッシュは一度切って戻した (§5) | [10-S1-REASONING.md](10-S1-REASONING.md) |
| S2 | 完了。**テンプレートは tools・画像・思考を同時に描ける** — 制約は `encodeToolChat` 側だった | [11-S2-TEMPLATE-PROBE.md](11-S2-TEMPLATE-PROBE.md) |
| S3 | 完了。**tools + 画像 + 思考が同時に通る** (G1 到達)。tool 呼び出しも思考と両立 | [12-S3-TOOLS-VISION-THINKING.md](12-S3-TOOLS-VISION-THINKING.md) |
| S3.6 | 完了。**画像セッションもプレフィックスを再利用する**。思考 ON はキャッシュ有無で答えが変わる点に注意 | [13-S3.6-PROMPT-CACHE-IMAGES.md](13-S3.6-PROMPT-CACHE-IMAGES.md) |

## 出発点 (2026-08-19)

ベースの改修は閉じた: 32 スロットの運用点で実際の Vision タスクが 30 t/s 超。
性能系の続き (I/O を床から外す仕事) は**この計画では扱わない**。

Server は既に多くが入っている ([OPENAI_SERVER.md](../OPENAI_SERVER.md)):
streaming SSE / function tools / 画像 / プロンプトキャッシュ (`cached_tokens`) /
MTP (`--draft-block-size`) / 4K〜128K。実害のある穴は [TODO.md](../../TODO.md) が
調査済みの 3 つで、**根は 1 つ** (tools 宣言時のチャットテンプレート制約):

1. ~~**Server に thinking が無い。**~~ **S1 で塞いだ** — `--thinking on|off` と
   要求ごとの上書きが入り、思考は `reasoning_content` で返る
   ([10-S1-REASONING.md](10-S1-REASONING.md))。tools 経路は 3 のまま。
2. ~~**pi 対話モードで画像が使えない。**~~ **S3 で塞いだ。**
3. ~~**tools 宣言時は thinking が無効。**~~ **S3 で塞いだ。**

> S2 の訂正 (2026-08-19): 2 と 3 の根を「tools 宣言時のテンプレート制約」と
> 書いていたが、**テンプレートは 3 つとも描けた**。根は `encodeToolChat` が
> 思考を false 固定で渡し、content を文字列としてしか渡さないことにあった
> ([11-S2-TEMPLATE-PROBE.md](11-S2-TEMPLATE-PROBE.md))。S3 で両方直し、
> 画像 + tools の 400 も外した ([12-S3](12-S3-TOOLS-VISION-THINKING.md))。

CLI は単発実行のみ: 対話モード無し、`--thinking` は `--messages-file` 限定、
ターンをまたぐ KV 再利用無し。

## ゴール

定性 3 つ + 従来の定量 1 つ:

| | 条件 |
| --- | --- |
| G1 | **pi の既定セッション (tools ON) で**、画像を貼れて、Reasoning ON で、MTP が効いた状態で使える。**サーバー側は到達** — tools + 画像 + 思考 + MTP の同時要求が通る ([12-S3 §2](12-S3-TOOLS-VISION-THINKING.md))。pi 実クライアントでの通し確認は未了 |
| G2 | **CLI で多ターンの Vision 対話**が成立し、2 ターン目以降は前ターンの prefix を再 prefill しない |
| G3 | **フラグ無し起動**が 16GB 機の運用点になる (今は runbook の長いコマンドをコピペしている) |
| G4 | mtp のゴール条件 (単発 Vision + Reasoning ON、tg ≥ 30 t/s) を **Server 実測**で回収する — 今までは CLI 実測しかない。**80 スロットで達成 (37.5 t/s)、32 スロットでは 28.6 t/s** (10-S1 §6) |

採点の流儀は mtp を引き継ぐ: 出力不変が要件の変更は **md5**、速度が絡む変更は
**同一セッション A/B**。機能の計画なので、各段の出口条件は「実クライアントで通ること」を最優先にする。

## マイルストーン

### S1 — Server Reasoning ON (tools 無し経路) 【実装済み】

goal 条件そのものであり、tools が絡まない限り純粋な配線なので最初にやった。
決定と根拠は [10-S1-REASONING.md](10-S1-REASONING.md)。要点だけ:

- `--thinking on|off` (プロセス既定) + 要求ごとの上書き。**API 表面の判断は
  「両方受ける」で閉じた**: pi は `thinkingFormat` の設定次第で
  `chat_template_kwargs.enable_thinking` か `reasoning_effort` のどちらかしか
  送らないので、こちらで選べる話ではなかった (10-S1 §1、**実測**)。
- `applyChatTemplate` に `enableThinking` を配線 (テキスト経路・画像経路とも)。
- 応答は `reasoning_content`。thought channel の切り出しは
  `StructuredAssistantDecoder` が既に持っていた channel 状態を使う (10-S1 §2)。
- 思考 ON の要求も**プロンプトキャッシュに参加する** (10-S1 §5)。
  当初は外していたが、pi の対話が毎ターン全再 prefill になったため撤回した。
- tools 宣言時は依然として思考なし (テンプレート制約)。400 にはせず通す — S2/S3 の論点。
- 出口: pi から Reasoning ON の Vision 単発が Server で通る (**達成**)。
  G4 の数字は**スロット数で割れた** (10-S1 §6): 80 スロットで 37.5 t/s、
  常用点の 32 スロットで 28.6 t/s。出力は両者で同じ。
  どちらを運用点にするかは S5 で決める。

### S2 — テンプレート検定 (probe、実装なし) 【完了】

結果は [11-S2-TEMPLATE-PROBE.md](11-S2-TEMPLATE-PROBE.md)。

- **(a) `enable_thinking: true` も (b) 画像 content part も、tools 宣言時に描けた**
  (**実測**、サーバーと同じ swift-jinja エンジンで描画)。3 つ同時も描ける。
- 制約は `Tokenizer.encodeToolChat` 側 — 思考は false 固定、content は文字列のみ。
- 出口: **degrade 案は不要**。S3 は本筋 (エンコーダに 2 つ足して 400 を外す) で進む。
  テンプレートの事実は `ToolTemplateCapabilityTests` 5 本で固定した。

### S3 — Vision × tools、thinking × tools 【完了】

結果は [12-S3-TOOLS-VISION-THINKING.md](12-S3-TOOLS-VISION-THINKING.md)。

- `encodeToolChat` に `enableThinking` と画像 part の入口 (`ToolChatMessage`) を足し、
  `OpenAIModels.swift` の「画像 + tools は 400」を外した。
- **実測**: tools + 思考で `bash` の tool call が出る / tools + 画像 + 思考で
  画像に答える / temp 0 で md5 再現 / 画像を含まない tool 要求のトークン列は不変。
- 既存の tool スキーマ・呼び出しテストは全通過 (パッケージ全体 895 本)。
- 残り (**未確認**): tool ループ 2 手目に前の思考が載らない件、
  思考 ON の多ターンでキャッシュが効かない件 (12-S3 §4)。

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
S5 には**スロット数の決着**も乗った: 常用点 32 とゴール条件 30 t/s が
両立しないことが S1 の実測で分かっている ([10-S1 §6](10-S1-REASONING.md))。

## 運用ルール

[mtp/README.md](../mtp/README.md) と同じ: **1 文書 = 1 論点** (目安 200 行)、
進捗は S1 → `10-S1-RESULTS.md` のように**文書を増やして**書きこの索引に 1 行足す、
設計の前提が変わったら本文を直す (履歴は git log)。
