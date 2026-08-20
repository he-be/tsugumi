# 引き継ぎ — サーバー再実装ループ

最終更新: 2026-08-22。ブランチ `macos15-support`。**P0〜P5 が全部済み。**
`swift test` は **1229 本で全緑** (約 110 秒)。**意図的な赤は無い。**
CONFORMANCE §2 の赤リストは**空**であり、§5 完了の定義の 1 つ目を満たしている。

この文書は**次のセッションが同じループを再開するための唯一の入口**である。
仕様は書かない ([SPEC.md](docs/serving/SPEC.md) が唯一の規範)。作業の並べ方も
書かない ([CONFORMANCE.md](docs/serving/CONFORMANCE.md) が持っている)。
ここにあるのは「今どこにいるか」と「どう再開するか」だけ。

## 1. 再開のしかた

新しいセッションで、この 3 つを添えて指示を出す:

```
@docs/serving/SPEC.md @docs/serving/README.md @docs/serving/CONFORMANCE.md
必要ならサブエージェントを複数起動しながら、TDDでサーバーの再実装作業を継続して。
適切な粒度でコミット・プッシュすること。
```

読む順序は **SPEC → CONFORMANCE → この文書の §3**。SPEC と食い違ったら
SPEC が勝つ。この文書が古かったら、直すのはこの文書のほうである。

## 2. 守る規律 (前回もこれで回した)

1. **仕様行 → 赤テスト → 実装 → 緑**。仕様に無い挙動を書かない。テストに無い
   仕様行を実装しない。実装を先に書いてしまったら戻して赤から始める。
2. **コミットに SPEC の ID を書く。**「症状 → 実装直行」をしていないことは、
   コミットメッセージに ID があるかどうかで外から分かる。
3. **仕様を自分で決めない。**参照実装 `~/LLM/llama.cpp/tools/server`
   (ピン `34af94cd9`) か OpenAI API を読んでから SPEC に行を足す (SPEC §13)。
   逸脱は必ず SPEC §12 に登録する。
4. **赤 → 緑を別コミットにする。**赤コミットでもツリーはビルドできる状態に
   保つ (未実装の入口は「未実装」を投げる)。
5. コミットメッセージは日本語。末尾に `Co-Authored-By` 行。

## 3. 現在地

| 段 | 状態 |
| --- | --- |
| **P0** 要求スキーマ | **済** (2026-08-19)。C0 の表駆動テスト |
| **P1** プロンプトキャッシュ → LCP | **D1〜D4 済**。**残り D5 (名前を SPEC に合わせる)** — 中身は薄く、後回しでよい |
| **P2** 生成の拘束 | **済** (2026-08-22)。下の §3.1 |
| **P3** ライフサイクル / EP | **済** (2026-08-22)。listen 先行 + ロード中 503、`/v1/health`、`/props`、採らないパスの 501、`/v1` 無しの別名、ERR-2 の 401/405/415、413 の撤去 |
| **P4** 思考 | **済** (2026-08-22)。`--reasoning-budget` / `--reasoning-format` へ改名、`--thinking` 退役、予算切れの終了タグ強制 (RSN-4) |
| **P5** 残り | **済** (2026-08-22)。`timings` (RSP-3)、`system_fingerprint` (RSP-5)、`/tokenize` `/detokenize` `/apply-template` (EP-5)、`/slots` `/metrics` (EP-6、FLAG-7 でゲート)、`--api-key` (FLAG-5)、CORS (FLAG-6)、`-c/--ctx-size` と `--expert-cache-slots` の丸め (FLAG-1/FLAG-2) |

### 次の一手 — **C3 を実機で走らせる**

**赤いテストはもう無い。**残っているのは「書けたが実機で見ていない」ことで、
それは CONFORMANCE §5 完了の定義の 2 つ目と 3 つ目である:

1. **`Scripts/c3_smoke.sh` を走らせる** (14 検査)。**まだ 1 度も走っていない。**
   人が建てたサーバーに当てる — 走らせ方はスクリプト冒頭のコメントにある。
   AGENTS.md の「Test rules」を先に満たすこと。**結果は CONFORMANCE §2 に
   書き足す** — 書けたことと通ったことは別である。
2. **pi の既定セッション** (tools ON + 画像 + Reasoning ON + MTP) を通しで動かす。
3. **OpenAI 公式 Python SDK の素朴なコード**がそのまま動くことを見る。

そこで症状が出たら、直す道は 1 つしかない: 参照実装を確認 → SPEC に行を足す →
赤テスト → 実装 (SPEC §13)。**症状から実装へ直行しない。**

残っている小さな仕事:

- **P1-D5** — 名前を SPEC に合わせる。中身は薄い。
- `docs/serving/README.md` の「残る計画」(CLI 対話モード = 旧 S4、既定値の
  自動選択 = 旧 S5) は **P2 のあと**という条件を満たした。

### 3.1 P2 で入った形 (次が壊しやすい場所)

| 型 | 役割 |
| --- | --- |
| `GBNFGrammar` / `GrammarMatcher` | GBNF のパーサと逐次マッチャ。ピン `34af94cd9` の `llama-grammar.cpp` の移植。**ピンの GBNF は公開文書より新しく、`TOKEN` / `TOKEN_NOT` (`<[42]>`, `!<[42]>`) がある** — Gemma の特殊トークンを文法要素として直接書けるのはこれのおかげ |
| `JSONSchemaGrammar` | JSON Schema → GBNF。方言が 2 つ (`.json` と `.gemmaToolArguments`)。参照実装の 73 本の期待値をそのまま持っている — **`.json` 側が 1 バイトでも動いたら、それは移植を壊したということ** |
| `GrammarTokenConstraint` | 語彙の piece 表 + マッチャ → `GenerationConstraint`。`GrammarVocabulary.shared(for:)` は**プロセスに 1 つ** (0.4 秒 / 25 MB)。要求ごとに作らないこと |
| `GenerationConstraint` / `ConstraintGate` | サンプラ側の口。**終了トークンの扱いは gate が持つ** — 拘束の実装は停止トークンを特別扱いしてはいけない |
| `ChatGrammarBuilder` | 要求 → 文法テキスト + 遅延かどうか + トリガ |
| `ServerGenerationPlan` | 要求 → 「何で拘束するか」の決定。**推論を持たない純粋な型なので、ここが検定の本体**。`ServerInference` はこの決定を実行するだけ |
| `ReasoningBudgetForcer` / `ServerReasoningPlan` | RSN-4。予算を数えて終了タグを強制する状態機械 |

**踏みやすい罠:**

- **拘束は棄却サンプリングで入る** (GEN-7)。通常どおり 1 トークン引き、
  適合しなければ**全語彙マスクで引き直す**。マスクは debug で 0.14 秒/棄却。
  棄却が増える形の文法を書くと、そのぶん素直に遅くなる。
- **`-Float16.infinity` は softcap カーネルを通らない** (`tanh(-inf) = -1` で
  確率が 0 にならない)。マスクは host で softmax を計算し直して 0 を書く。
- **文法拘束と強制挿入がある要求は投機デコードを使わない** (DEV-14)。
- **tool call の文法はテンプレートの正準形しか許さない** (GEN-8〜GEN-11)。
  空白なし・裸キー昇順・`<|"|>` 文字列・エスケープ無し・null 無し・桁数制限。
  **緩めると INV-1 が破れて毎ターン LCP が切れる**。数値のずれだけは受け入れている。
- **非遅延の文法は先頭に思考ブロックを飲む** (GEN-13)。ここを外すと
  思考 ON + `response_format` が最初のトークンで詰まる。

## 4. 土台 (P0〜P4 で入った、次の段が乗る場所)

| 型 | 役割 |
| --- | --- |
| `ChatRequestSchema` | SPEC §4 の表そのもの。行を足す = 仕様を足す。値規則は `handler` に載せる (参照実装の `custom_handler` に対応) |
| `ChatRequestParser` | 正規化済みの値 → `ValidatedChatRequest`。エンジン写像 (DEV-9/10) もここ |
| `ChatMessageValidator` | SPEC §5 と §6 の tools 側だけ。旧 `OpenAIRequestValidator` の残骸 |
| `ServerError.swift` | ERR-1/ERR-2。`type` を決めれば HTTP 番号は自動で決まる |
| `ServerPromptRenderer` | テキスト/tools 経路の描画。**サーバーが描く変種はここの `static let variant` 1 か所が決める** |
| `commonPrefixLength` | CACHE-1 の本体。**D3 でキャッシュ判定はこれ 1 本になった** |
| `GFTokenizer.ChatTemplateVariant` | D2 で入った。`.modelBundled` (CLI・アプリ・KernelCheck) と `.serverRedraw` (サーバー) の 2 値。**既定は `.modelBundled`** なので、足した経路を明示的に渡さない限り描画は動かない |
| `ServerChatTemplate` | リポジトリ所有の jinja (`Sources/TurboFieldfare/Templates/server_chat_template.jinja`)。SPEC §12 DEV-12 |
| `ServerReadiness` / `ServerProperties` | P3。ロード状態は経路表より手前で見る。`/props` は `ChatRequestSchema` の表を歩いて作る — **既定値の第 2 の写しを作らないこと** |
| `Scripts/c3_smoke.sh` | C3 の 14 検査。**まだ 1 度も走っていない** (§6) |

削除済み: `OpenAIRequestValidator`、`OpenAIChatRequest`、`OpenAIStop`、
`OpenAIStreamOptions`、`OpenAIReasoning`、`ServerRequestError.payloadTooLarge` /
`.unknownModel`。

## 5. 積み残し・注意 (SPEC には書けていない実務メモ)

- **同梱テンプレートが 2 本あり、テストとサーバーで別々のものを引いていた**
  (2026-08-20 実測)。`GFTokenizer.load()` (引数なし、テストが使う) は HF キャッシュの
  `~/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/4d7ae4984b…/chat_template.jinja`
  (390 行、sha `ae53464b…`、ヘッダに `Published: 2026-07-09`)、サーバーは
  `scratch/gemma4.gturbo/tokenizer/chat_template.jinja` (362 行、sha `36e3a42e…`) を引く。
  **D2 の変種はサーバーが実際に使っている 362 行版から取った** (差分 1 ハンクに保つため)
  ので、tools 経路については**テストとサーバーが同じテンプレートを見るようになった**のは
  D2 の副産物である。ただし 390 行版にある `arguments is none` 対応・`image_url` /
  `input_audio` の別名・continuation 判定の O(1) 化・`raise_exception` は**変種に入っていない**。
  取り込むなら SPEC に行を足してから (§13)。**症状が出るまで触らない。**
- **`repetition_penalty` は効かなくなった。**SPEC REQ-repeat-penalty が
  `repeat_penalty` のみを挙げているため、旧名は R1 で無視される。既存
  クライアントがいるなら「別名を足す」を SPEC に書くところから。
- **`max_tokens: 0` (prefill のみ) は暫定実装。**生成 0 トークンで即応答するが、
  KV の暖機はしていない。LCP 化は済んだので、いつ詰めてもよい。
- **思考の既定が変わった。**`--thinking` の既定は off だったが、後継の
  `--reasoning-budget` の既定は **-1 (無制限)** なので、何も言わないクライアントは
  これから思考する。SPEC RSN-1 のとおりだが、トークンの消費が変わる。
- **`max_tokens` の 1/4 を答えのために取り置く** (DEV-21)。RSN-4 が分け方を
  決めていなかったので実装が決めた数字であり、**測って動かしてよい**。
- **`docs/mtp/34-M9-PROPOSAL.md` は未追跡のまま置いてある。**サーバーとは
  無関係 (MTP の提案書)。前回のセッション中に外から現れたもので、意図的に
  触っていない。
- `docs/OPENAI_SERVER.md` / `docs/SERVER_RUNBOOK.md` / `docs/RUNTIME_CONTROLS.md` は
  **P5 まで追随済み** (2026-08-22)。
- **追い残し (SPEC の話ではなく実装の話)**: `GrammarMatcher` の `RejectContext.decoded` が
  `[[UInt32]]` なので語彙のコードポイント表を平坦化できず、`GrammarTokenConstraint` が
  `rejectedIndices` の候補組み立てを複製している。直すならエンジン側の型から。
- **`swift test -c release` は swift-testing の suite を走らせない** (XCTest の
  shim だけが動く)。文法マスクの release 実測ができていないのはこれが理由。

## 6. 環境メモ

- テストは `./Scripts/test.sh` (直列)。`--filter <型名>` で絞る。
  C0〜C2 は重みも Metal も要らず、サーバー全体で 8 秒。
- C3 (実機スモーク) だけモデルを積む。走らせる前に AGENTS.md の
  「Test rules」を満たすこと — 特に
  `pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|…'` が空であること。
  **既にあるモデルプロセスを止めない。**
- 参照実装: `~/LLM/llama.cpp`、ピン `34af94cd9`。**作業ツリーはピンより先に
  進んでいる** (2026-08-20 時点で `fe8156f78`)。素の `grep` はピンでないコードを
  読むので、**必ず `git show 34af94cd9:tools/server/<file>` で引くこと。**
  一次資料は
  `tools/server/server-schema.cpp` (要求スキーマ)、`server-common.cpp`
  (OAI 層とエラー封筒)、`server-context.cpp` (キャッシュ)、`README.md`。
  ピンを上げるときは差分を読んで SPEC の該当行を先に直す。
- モデル同梱テンプレート: `scratch/gemma4.gturbo/tokenizer/chat_template.jinja`。
- **実機で見ていないもの** (すべて C3 送り、`Scripts/c3_smoke.sh` に検査がある):
  棄却サンプリングと遅延文法と思考中の抑止 (GEN-5/6/7) が本物のサンプラで
  動くこと、予算切れの終了タグ強制 (RSN-4) がモデルの上で本文を書かせること、
  リングより深い巻き戻し (DEV-13)、拘束された tool call が次のターンの
  描き直しと一致すること (INV-1 × GEN-8)。**「書けた」と「通った」は別**なので、
  走らせた結果は CONFORMANCE §2 に書き足すこと。
- サブエージェントを使うときは**共有インデックスに注意**。`git add` だけでは
  他のエージェントが stage したファイルを巻き込むので、**`git commit -- <paths>`**
  で経路を限ること。実際に 2 回巻き込みが起きた。
