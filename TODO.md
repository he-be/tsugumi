# 引き継ぎ — サーバー再実装ループ

最終更新: 2026-08-19。ブランチ `macos15-support`、最新 `ca5831d`。

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
| **P0** 要求スキーマ | **済** (`a668197` 赤 → `691de9c` 緑)。C0 41 本緑 |
| **P1** プロンプトキャッシュ → LCP | **D1 済** (`db52a46`、赤のまま)。**D2 は方針決定済・未着手**。D3〜D5 未着手 |
| **P2** 生成の拘束 | 未着手 (GEN-3/GEN-4 は暫定 501 で入っている) |
| **P3** ライフサイクル / EP | 未着手 |
| **P4** 思考 | 未着手 |
| **P5** 残り | 未着手 |

`swift test` は 150 本緑 + C2 の 2 本 (8 ケース) が**意図的に赤**。
赤いのは `PromptTokenInvariantTests` だけで、これが P1 の作業キューそのもの。

### 次の一手 = P1-D2

方針は決定済み (CONFORMANCE §3 の P1 行に記録):

> モデル同梱テンプレートに従うのをやめ、**サーバーが自分のテンプレート変種を
> 持つ**。完了した assistant ターンを生成時と同じ形で描き直す — 思考 OFF なら
> 空の thought channel (`<|channel>thought\n<channel|>`)、思考 ON なら思考
> ブロックそのもの。`reasoning_content` の入力 (MSG-5) も同時に通す。

手を入れる場所:

- `Sources/TurboFieldfare/Tokenization/Tokenizer.swift` の
  `applyChatTemplate` (tools 無し経路) と `encodeToolChat` (tools 経路)。
  後者は `tokenizer.applyChatTemplate(messages:chatTemplate:…)` の
  `chatTemplate` 引数に**リポジトリ所有のテンプレート文字列**を渡す形になる
  (今は `nil` を渡してモデル同梱の `chat_template.jinja` を使っている)。
  同梱テンプレートの該当箇所は `add_generation_prompt` のブロック (末尾) と、
  assistant ターンの `reasoning_content` 描画条件
  (`thinking_text and loop.index0 > last_user_idx and message.get('tool_calls')`
   — content だけのターンでは描かれないのが破れの片方)。
- `GFTokenizer.Message` に `reasoningContent` を足し、
  `ChatMessageValidator` が `messages[].reasoning_content` を読むようにする。
- 実装したら **SPEC §12 に DEV 行を登録**し、`ServerPromptCacheDomain` の
  `templateSHA256` が変種を指すようにする。
- 品質影響 (モデルが履歴をどう読むか) は重み無しでは確認できない。
  C3 の実機スモークで見る。

D1 が測った破れ幅 (これが D2 の合格ラインでもある):

| 形 | KV | 今の LCP | 取りこぼし |
| --- | --- | --- | --- |
| 思考 OFF / tools 無 | 28 | 16 | 12 |
| 思考 OFF / tools 有 | 65 | 53 | 12 |
| 思考 ON / tools 無 | 43 | 23 | 20 |
| 思考 ON / tools 有 | 75 | 55 | 20 |

**D2 より先に D3 をやらないこと。**今のブリッジ合成が 100% 使えていた KV を
LCP が取りこぼして遅くなる (CONFORMANCE §3 の警告)。

## 4. P0 で入った土台 (次の段が乗る場所)

| 型 | 役割 |
| --- | --- |
| `ChatRequestSchema` | SPEC §4 の表そのもの。行を足す = 仕様を足す。値規則は `handler` に載せる (参照実装の `custom_handler` に対応) |
| `ChatRequestParser` | 正規化済みの値 → `ValidatedChatRequest`。エンジン写像 (DEV-9/10) もここ |
| `ChatMessageValidator` | SPEC §5 と §6 の tools 側だけ。旧 `OpenAIRequestValidator` の残骸 |
| `ServerError.swift` | ERR-1/ERR-2。`type` を決めれば HTTP 番号は自動で決まる |
| `ServerPromptRenderer` | テキスト/tools 経路の描画。**D2 で直すのはここと Tokenizer** |
| `commonPrefixLength` | CACHE-1 の本体。D3 でキャッシュ判定がこれ 1 行に置き換わる |

削除済み: `OpenAIRequestValidator`、`OpenAIChatRequest`、`OpenAIStop`、
`OpenAIStreamOptions`、`OpenAIReasoning`、`ServerRequestError.payloadTooLarge` /
`.unknownModel`。

## 5. 積み残し・注意 (SPEC には書けていない実務メモ)

- **`repetition_penalty` は効かなくなった。**SPEC REQ-repeat-penalty が
  `repeat_penalty` のみを挙げているため、旧名は R1 で無視される。既存
  クライアントがいるなら「別名を足す」を SPEC に書くところから。
- **`max_tokens: 0` (prefill のみ) は暫定実装。**生成 0 トークンで即応答するが、
  KV の暖機はしていない。P1 の LCP 化のあとに詰める。
- **`--thinking` はまだ残っている** (廃止は FLAG-4 / P4)。内部では
  `ChatRequestDefaults` 経由に変えてあるので、P4 では入口の名前だけの話になる。
- **`docs/mtp/34-M9-PROPOSAL.md` は未追跡のまま置いてある。**サーバーとは
  無関係 (MTP の提案書)。前回のセッション中に外から現れたもので、意図的に
  触っていない。
- `docs/OPENAI_SERVER.md` と `docs/SERVER_RUNBOOK.md` は P0 の範囲だけ追随済み。
  FLAG の改名 (P4) とロード中 503 (P3) が入ったらまた直す。

## 6. 環境メモ

- テストは `./Scripts/test.sh` (直列)。`--filter <型名>` で絞る。
  C0〜C2 は重みも Metal も要らず、サーバー全体で 8 秒。
- C3 (実機スモーク) だけモデルを積む。走らせる前に AGENTS.md の
  「Test rules」を満たすこと — 特に
  `pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|…'` が空であること。
  **既にあるモデルプロセスを止めない。**
- 参照実装: `~/LLM/llama.cpp`、ピン `34af94cd9`。一次資料は
  `tools/server/server-schema.cpp` (要求スキーマ)、`server-common.cpp`
  (OAI 層とエラー封筒)、`server-context.cpp` (キャッシュ)、`README.md`。
  ピンを上げるときは差分を読んで SPEC の該当行を先に直す。
- モデル同梱テンプレート: `scratch/gemma4.gturbo/tokenizer/chat_template.jinja`。
  D2 で読むことになる。
