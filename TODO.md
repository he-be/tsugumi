# 引き継ぎ — サーバー再実装ループ

最終更新: 2026-08-20。ブランチ `macos15-support`。**P1-D2 まで済み**。

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
| **P1** プロンプトキャッシュ → LCP | **D1 済** (`db52a46`)、**D2 済** (2026-08-20、赤 → 緑の 2 コミット)。**D3〜D5 未着手** |
| **P2** 生成の拘束 | 未着手 (GEN-3/GEN-4 は暫定 501 で入っている) |
| **P3** ライフサイクル / EP | 未着手 |
| **P4** 思考 | 未着手 |
| **P5** 残り | 未着手 |

`swift test --filter TurboFieldfareServerTests` は **154 本すべて緑**
(C2 の 8 ケースを含む)。**意図的な赤はもう無い** — 次の赤は D3 の入口で書く。

### 次の一手 = P1-D3

**D2 が終わったので順序の縛りは解けた** (CONFORMANCE §3 の警告「D2 より先に
D3 をやるな」は満たされた)。D3 は `ServerPromptCache` の判定を捨てて
**トークン列の LCP 1 本**にすること (SPEC CACHE-1〜3、CONFORMANCE §4 の
「巻き直す」表)。

- 捨てるもの: 意味ゲート 8 個、**ブリッジ合成** (`encodeContinuation` /
  `encodeToolResultContinuation` とその呼び出し 2 か所 —
  `ServerPromptCache.swift:298`/`:356`)、`ServerPromptCacheMiss` の 11 分類
  (SPEC CACHE-6: 観測値は `cached_tokens` の数字 1 種類だけ)。
- 置く先: `commonPrefixLength` (P0 で入っている)。CACHE-2 の部分一致と
  CACHE-3 の「全一致なら末尾 1 トークン捨てて再デコード」も同時に入る。
- **赤テストの書き方**: C2 は「この 2 要求の LCP は N トークン」としか主張しない
  (CONFORMANCE §1)。「この形は hit する」を書かない。
- `ServerPromptCache.publish` (`:153`) が合成する assistant ターンは、
  LCP になれば要らなくなる。**D2 で `reasoningContent` を足したので、
  合成を残したままにすると等値比較がずれる** — D3 で消すのが正しい順序。

**D2 が測った結果** (D1 の破れ幅がそのまま埋まった。`PromptTokenInvariantTests`):

| 形 | KV | D1 の LCP | D2 の LCP |
| --- | --- | --- | --- |
| 思考 OFF / tools 無 | 28 | 16 | **28** |
| 思考 OFF / tools 有 | 65 | 53 | **65** |
| 思考 ON / tools 無 | 43 | 23 | **43** |
| 思考 ON / tools 有 | 75 | 55 | **75** |

**未確認**: 品質影響 (履歴の思考ブロックを毎回描き直すことをモデルがどう読むか)。
同梱テンプレートは「最後の user より後」の思考しか描かない設計で、こちらは
**その guard を意図的に外している**。重み無しでは確認できないので **C3 で見る**。

## 4. P0 で入った土台 (次の段が乗る場所)

| 型 | 役割 |
| --- | --- |
| `ChatRequestSchema` | SPEC §4 の表そのもの。行を足す = 仕様を足す。値規則は `handler` に載せる (参照実装の `custom_handler` に対応) |
| `ChatRequestParser` | 正規化済みの値 → `ValidatedChatRequest`。エンジン写像 (DEV-9/10) もここ |
| `ChatMessageValidator` | SPEC §5 と §6 の tools 側だけ。旧 `OpenAIRequestValidator` の残骸 |
| `ServerError.swift` | ERR-1/ERR-2。`type` を決めれば HTTP 番号は自動で決まる |
| `ServerPromptRenderer` | テキスト/tools 経路の描画。**サーバーが描く変種はここの `static let variant` 1 か所が決める** |
| `commonPrefixLength` | CACHE-1 の本体。D3 でキャッシュ判定がこれ 1 行に置き換わる |
| `GFTokenizer.ChatTemplateVariant` | D2 で入った。`.modelBundled` (CLI・アプリ・KernelCheck) と `.serverRedraw` (サーバー) の 2 値。**既定は `.modelBundled`** なので、足した経路を明示的に渡さない限り描画は動かない |
| `ServerChatTemplate` | リポジトリ所有の jinja (`Sources/TurboFieldfare/Templates/server_chat_template.jinja`)。SPEC §12 DEV-12 |

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
- 参照実装: `~/LLM/llama.cpp`、ピン `34af94cd9`。**作業ツリーはピンより先に
  進んでいる** (2026-08-20 時点で `fe8156f78`)。素の `grep` はピンでないコードを
  読むので、**必ず `git show 34af94cd9:tools/server/<file>` で引くこと。**
  一次資料は
  `tools/server/server-schema.cpp` (要求スキーマ)、`server-common.cpp`
  (OAI 層とエラー封筒)、`server-context.cpp` (キャッシュ)、`README.md`。
  ピンを上げるときは差分を読んで SPEC の該当行を先に直す。
- モデル同梱テンプレート: `scratch/gemma4.gturbo/tokenizer/chat_template.jinja`。
  D2 で読むことになる。
