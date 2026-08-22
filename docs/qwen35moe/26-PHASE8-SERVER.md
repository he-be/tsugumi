# 26. サーバー結線 — Ornith が HTTP から答える (実測(手元)、2026-08-22)

[04-PHASES.md](04-PHASES.md) の **Phase 8**。[25](25-CLI-TOOLS.md) までで
CLI は実物にツールを呼ばせたが、**サーバーは Gemma の型しか通っていなかった** —
`QwenChatGrammarBuilder` は書けていて誰も呼んでおらず、`ServerInference` /
`ChatRequestParser` / `ServerGenerationPlan` に Ornith の分岐が無かった。

```
.build/release/TurboFieldfareServer \
  --model scratch/ornith-oq4e-g64.gturbo --model-id ornith-1.5-35b-a3b \
  --port 8099 --ctx-size 8192 --verification trusted-install --metrics
→ TurboFieldfareServer ready … family=qwen3_5_moe context=8192 slots=32 \
  expert_io=mmap mtp=0 reasoning_budget=-1 reasoning_format=auto
```

| | |
| --- | --- |
| 中心の主張 | **バックエンドだけが家族を知っている。**HTTP 層・要求検証・待ち行列・timings・応答の形はどれも `ServerInferenceBackend` しか見ておらず、**1 行も変えていない**。足したのは `QwenServerSession` (`ServerModelSession` の兄弟) と、その決定を持つ純粋型 `QwenGenerationPlan` (§2) |
| 実物が答えた | 日本語の非ストリーミング / 英語のストリーミング / ツール呼び出し / ツール応答の往復 / `tokenize`・`detokenize`・`apply-template` / `max_tokens: 0` / 画像の 400 / `reasoning_format: none` / stop 文字列 / 切断 / `metrics`・`slots` (§5) |
| 書いたもの | `QwenServerSession` / `QwenGenerationPlan` / `QwenForwardRunner.runGreedyCompletion` (§1) / `QwenReasoningSplitter` (CLI の private から**ライブラリへ移した**、§3) / `QwenTokenizer.chatTemplateJinja` / `main.swift` の家族分岐 |
| できないこと 4 つ | 画像は **400**、投機は**起動時に断る**、prompt cache は**無い** (再帰状態を巻き戻せない)、サンプラは**受理して無視** (R3)。いずれも黙って落とさず、断るか `approximations` に載せる (§4) |
| 検査 | `swift test --no-parallel` が **1,338 → 1,350 件** (新規 12 本)。`--qwen` 63 / `--qwen-tools` 36 / `--qwen-constrain` 9 / `--qwen-prefill` 55 トークン / **Gemma の既定 69 本**はすべて緑のまま |
| 残り | **`tool_choice: required` が前置きで回り続ける検体がある** (§6-1、ユーザー判断)。RSN-4 の強制閉じタグ、並列呼び出しの実物、入れ子 JSON |

---

## 1. 生成ループを 1 本にした

サーバーが CLI と違って要るものが 3 つある: **外から止められること** (stop
文字列は、そのトークンの文字が上流のマッチャに届いて初めて答えが決まる)、
**キャンセルできること** (客が切ったら 600 トークン回し切らない)、
**壁時計の内訳** (SPEC §9 RSP-3 の `prompt_ms` / `predicted_ms`)。

`QwenForwardRunner.generateGreedyPrefilled` にはどれも無かった。足したのは

```swift
public func runGreedyCompletion(
    promptTokens:, maxNewTokens:, chunkWidth:, stopTokens:, constraint:,
    shouldStop: () -> Bool = { false },
    onPrefill: ((Int, Double) -> Void)? = nil,
    onToken: ((Int, Int32) throws -> Void)? = nil
) throws -> QwenGreedyRun
```

で、**`generateGreedyPrefilled` はこれの答えを削ったもの**になった。ループは
1 本しか無い — 制約つきの引きがどこから来るかの規則も、「停止トークンは
*出してから*終わる」という規則も、1 か所にしか無い。

| 見るもの | どこで |
| --- | --- |
| `shouldStop` | 呼び出し側がそのトークンを見た**後**。stop 文字列はそうでないと働かない |
| `Task.checkCancellation()` | 2 トークンの**あいだ**。1 ステップは待ち合わせ済みのコマンドバッファ 40 層で、途中に割り込む場所が無い |
| `prefillSeconds` | 最後のチャンクが走り終わるまで。**制約の再畳み込みは入らない** — あれは最初の*生成*トークンの費用である |

`QwenGreedyRun` は `RawDecodeResult` の Ornith 版だが、**K/V スナップショットと
キャッシュ済み接頭辞を持たない**。この家族にはどちらも意味が無い (§4-3)。

`--qwen-prefill` の 55 トークンは**この書き換えの後も 1 個も動いていない**
(チャンク 512 / 8、per-pair / タイルの 4 通りとも)。

## 2. 決定は純粋型に置いた

`QwenGenerationPlan` は `ServerGenerationPlan` の兄弟で、同じ形の答えを返す —
文法・遅延かどうか・引き金・`approximations`・エラーが名乗ってよい「要求の形」。
分けた理由も同じで、**セッションは重みなしに作れない**が、ここは検証済み要求と
チェックポイントのマーカー ID だけの関数なので、C0 で 12 本の検査が立つ。

Gemma 版との差は 1 つだけ、`approximations` の**タグが 3 種類ある**こと:

| タグ | 何を失ったか |
| --- | --- |
| `tools/` | 宣言がプロンプトに入るまでに落ちたもの (`GemmaToolSchema`) |
| `grammar/` | GBNF が縛れなかったもの |
| `sampling/` | **この家族がそもそもできないもの** (§4-1) |

## 3. チャンネル規則は 1 つ、生産者は 2 つ

`QwenReasoningSplitter` は `RunQwen.swift` の `private struct` だったが、
**ライブラリ (`Sources/TurboFieldfare/Tokenization/`) に移した**。サーバーも
「ツールを宣言しない要求」で同じものが要るからで、写すのではなく動かした。

移すついでに API を `QwenStructuredAssistantDecoder` に合わせた
(`consume(tokenID:delta:) -> [StructuredAssistantEvent]` /
`consumeTail`)。これで CLI もサーバーも**イベントを 1 か所で捌く**。

なぜ 2 つ要るのか: **ツールを宣言していない run では、モデルが勝手に書いた
`<tool_call>` は本文である。**デコーダは未宣言のツールとして落とす。宣言が無い
要求が見せるべきものは前者なので、分岐は「モードの違い」ではなく別の型でよい。

### 3-1. `emitsReasoning` は常に真にした

`QwenStructuredAssistantDecoder(emitsReasoning:)` が**偽だと思考テキストを
routeText で捨てる** (`return []`)。最初は Gemma 側に合わせて
`request.enableThinking` を渡していたが、それだと

- **ツールを宣言した thinking on の要求で思考が黙って消える** (実測。§5 の
  該当行がこれで、直す前は `reasoning_content` が空だった)
- 分岐しない側 (`QwenReasoningSplitter`) は捨てられないので、**同じ要求が
  ツールの有無で違う話をする**

ので、常に真を渡すことにした。**どこへ出すかは RSN-3 の問いで、
`ServerReasoningPlan.route` が 1 か所で答える** — デコーダはチャンネルを
分けるだけ、という分担にそろえた。

## 4. できないこと 4 つ

### 4-1. サンプラ — 受理して無視する (R3)

この家族の融合ヘッドは logit をどこにも書かない ([19](19-LM-HEAD-INT8.md)) ので、
分布を作る材料が無く、引きは常に argmax である。CLI は**断る**が、サーバーで
同じことをすると**既定の要求が全部 400 になる** — REQ-temp の既定は 1.0 で、
`temperature` に触れていない客もその値を送っているからである。

採ったのは SPEC §12 **DEV-5 / R3** の作法、「受理して無視」。ただし黙らない:

```
completed in 4.3s prompt=16 cached=0 completion=32 finish=stop
  approx="sampling/greedy-only: temperature=1.0 ignored; …"
```

`temperature: 0.7`・`top_p: 0.9`・`repeat_penalty: 1.1` を送った要求は
**3 つとも 1 行に名前が並ぶ**。温度を入れるには結局 [25 §1](25-CLI-TOOLS.md) の
案 A (語彙幅の logit ヘッド) が要る。

### 4-2. 画像 — 400 で断る

`400 unsupported_image`。合わせて **EP-4 の `modalities.vision` を家族で切った**
(`ServerProperties.supportsVision`) — 画像を付けてよいかを `/props` で決める客に、
1 枚送らせてから断るのは筋が悪い。Gemma 側の既定は `true` のままで、
`/props` の答えは 1 バイトも変わらない。

### 4-3. prompt cache — ~~無い~~ → **[41](41-PROMPT-CACHE.md) で入った (2026-08-22)**

> **本節の結論は撤回する。**巻き戻しが要らない形 (**新しいプロンプトが状態の
> 食ったトークン列で始まるときだけ続ける**) なら再帰状態と両立し、実測で
> エージェント形の会話の 2 ターン目以降が全部当たった。`cache_n` は常に 0
> ではなく、`runner.reset()` は条件つきになった。以下は 2026-08-22 の
> 実装前の記述として残す。

[03 §5](03-DESIGN.md) のとおり、線形 30 層の再帰状態は**巻き戻せない・
途中を捨てられない**。したがって

- `cache_n` は**常に 0**。同じ会話の 2 ターン目もプロンプトを全部計算する
- ランナーは要求ごとに `reset()` する。**前の要求が途中で落ちていても同じ**
- `ExpertCacheBudget` に足す勘定は**無かった**。[03 §5](03-DESIGN.md) の
  「slot 数 × 62.8 MiB」は*スナップショットを持つ案*の値段で、持たないと決めた
  ので、生きている状態 1 本 (62.8 MiB、`QwenForwardRunner` の中) だけが残る。
  DEV-3 の「生成スロットは 1 本」がそれを許している

**2026-08-22 追記 — 「無い」の根拠は半分だけ正しい。**
[32 §2](32-NVMAI-ADOPT.md) が持ち込んだ snapshot-restore 型
(「巻き戻さない。次の要求が**前回の厳密な延長**のときだけ続きから走る」) なら
再帰状態と両立する。[34](34-PROMPT-CACHE-ESTIMATE.md) が机上で出したところでは:

- **取り分は最大 9.2 秒** (長文文脈の 2 ターン目、TTFT 10.6 → 1.5 秒)。
  短いチャットの 2 ターン目で 0.9 秒、ツールループ 1 ホップで 1.3 秒
  ([34 §1](34-PROMPT-CACHE-ESTIMATE.md)、**導出**)
- **「延長のみ」は既にコードが言っている** — `maximumSafeRewind` は再帰層が
  1 本でもあれば 0 を返し (`KVCacheManager.swift:283-289`)、
  `ServerPromptCache.match` はそれを要求する (`ServerPromptCache.swift:239`)。
  **新しい規則を書く必要は無い** ([34 §4-1](34-PROMPT-CACHE-ESTIMATE.md))
- **本節の勘定は「無かった」のままでよい。**その場保持型なら追加は
  **0 バイト**である ([34 §2](34-PROMPT-CACHE-ESTIMATE.md))
- 上の 2 行目「ランナーは要求ごとに `reset()` する」が変える対象
  (`QwenServerSession.swift:169` の無条件 `runner.reset()`)

**着手はしていない。判断はユーザー** ([04](04-PHASES.md) #31)。
**番号の大きい [34](34-PROMPT-CACHE-ESTIMATE.md) が正。**

### 4-4. 投機 — 起動時に断る

`--draft-block-size 4` は**リスナーが開く前に** usage で落ちる
(`QwenServerSession.validateFlags`、`main` と `load` の両方から呼ぶ)。Phase 7。

## 5. 実物が答えたもの

すべて 1 台 (M3 Pro 18GB)、`--ctx-size 8192 --expert-cache-slots 32`。
**n=1 なので数字だけ置く** (README 運用ルール)。

| 何を | 結果 |
| --- | --- |
| 日本語・非ストリーミング・thinking on | `content` = 「日本の首都は東京です。」、`reasoning_content` に英語の思考。prompt 20 / predicted 52 |
| 英語・ストリーミング・thinking off | `Hello there, friend!` が 5 チャンク、最終チャンクに `timings` |
| `tools` + `tool_choice: auto` (thinking off) | `get_weather({"city":"Kyoto","days":3})`、`finish_reason: tool_calls`、prompt 292 / predicted 39 |
| `tools` + thinking on | 思考が `reasoning_content` に分かれ、`get_weather({"city":"Kyoto","days":1})` が返る。**§3-1 を直すまではこの検体の思考が黙って消えていた** |
| ツール応答の往復 | `assistant.tool_calls` → `role: tool` をテンプレートが描き直し、prompt 359 で表つきの答え |
| `tokenize` / `detokenize` | `"こんにちは world"` → `[85951, 1814]` → 往復 |
| `apply-template` | `<\|im_start\|>user\nhi<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
| `max_tokens: 0` | `finish=length`、completion 0、prompt だけ勘定される |
| 画像 | `400 unsupported_image` |
| `reasoning_format: none` | 思考が `content` に戻り、`reasoning_content` は無い |
| `stop: ["three"]` | `'One, two, '` で `finish=stop` |
| 切断 (`curl -m 3`) | ログは `generating` で終わり `completed` が無い。**次の要求が 0.779 s で通る** = スロットは即返っている |
| `/metrics` | `prompt_tokens_total 36 / prompt_seconds_total 2.149 / tokens_predicted_total 100 / tokens_predicted_seconds_total 7.185` |
| 同じ 1 ターンを 2 回 | どちらも predicted 50、`cache_n` 0。10.85 → 18.34 tok/s |

最後の行は**運用値ではない** — 2 回目が速いのはエキスパートキャッシュが温まって
いるからだが、n=1 の 2 本なので**解釈は書かない**。速度は Phase 6 の仕事である。

## 6. 見つかったもの・残したもの

### 6-1. `tool_choice: required` が前置きで回り続ける (**ユーザー判断**)

ツールと関係の無い質問に `tool_choice: required` を付けると、**呼び出しを
書かないまま `max_tokens` まで散文が続く**。700 トークンでも同じだった。

```
"Tell me a fun fact about cats." + tools=[get_weather] + tool_choice=required
→ finish=length, tool_calls=null, completion=700
  末尾: "…<system>tool ran without output</system>\n\nLet me know if you'd like…"
```

これは**結線の不具合ではなく、非遅延文法の前置きの設計そのもの**である
([`QwenToolCallGrammar.toolPreamble`](../../Sources/TurboFieldfare/Grammar/QwenToolCallGrammar.swift)):
前置きは「**セクション開始でない任意のトークン**」で、そこに置かれた保証は
「前置きの中では*止まれない* (`mayEndHere` が偽なので停止トークンが拒まれる)
から、出口は呼び出しを書くことしか無い」だった。**止まれないことと、
いつか出ることは別である** — 実物は拒まれた停止のあとに辻褄合わせの文
(`<system>tool ran without output</system>`) を書いて、また止まろうとする。

選べる手は 2 つ:

| 案 | 中身 | 対価 |
| --- | --- | --- |
| A. 前置きを締める | **同じファイルの `responseFormatGrammar` が既に使っている形** — `(!</think>* </think> [ \t\n]{0,20})?` — に揃える。thinking off ならテンプレートが既に閉じているので**1 手目から呼び出しが強制される** | チェックポイント自身のシステムプロンプト (「呼び出しの*前*なら自然文を書いてよい」) と食い違う。[23](23-PHASE5-TOOLS.md) が理由を 2 つ挙げて選んだ形なので、**既存の検査 2 本が落ちる** |
| B. そのまま | 文書に残し、`required` は「いつか呼ぶ」までしか約束しないとする | 上の検体は 400 でも 500 でもなく `finish=length` の散文で返る |

**変えていない。**[23](23-PHASE5-TOOLS.md) の判断を勝手に上書きしないため
(README 運用ルール)。**判断はユーザーに出す。**

### 6-2. RSN-4 の強制閉じタグが無い

`ServerReasoningPlan` はそのまま使えた (重みもトークナイザも見ない型なので)
が、**`ReasoningBudgetForcer` を当てる場所がこのループに無い**。強制は「引かずに
置く」ことなので、`runGreedyCompletion` に入れるなら 1 段追加が要る。今は
`reasoning/budget-not-enforced` を `approximations` に載せている。

**注意: この行は thinking on で `max_tokens` を書いた要求のほとんどで出る。**
RSN-4 の後半 (deadline) は `max_tokens` が文脈残より小さければ立つからで、
うるさいが嘘ではない。

### 6-3. 宣言は Gemma のアダプタを通っている

`ChatRequestParser` は家族を知らないので、ツールの `parameters` は
`GemmaToolSchema.adapted` を通る。Ornith のテンプレートは JSON スキーマを
そのまま描けるので、ここは**必要より少し落としている**。落としたものは
`tools/` タグで出るので黙ってはいない。手を入れるなら要求層に家族を通すことに
なり、それは Gemma 側の検査に触る。

### 6-4. まだ見ていないもの

- **並列呼び出し**を実物が書いた run ([25 §6](25-CLI-TOOLS.md) のまま)
- **入れ子 JSON の往復** ([23 §5-2](23-PHASE5-TOOLS.md) のまま)
- 待ち行列に 2 本以上積んだときの挙動 (家族に依らない層なので Gemma 側と同じ
  はずだが、Ornith では**測っていない**)

## 7. 触った共有コード

Gemma の実測値を動かさないため、共有部への変更は次の 3 つに限った。
**どれも Gemma の経路の振る舞いを変えない** — 既定 69 本と `swift test` で確認。

| 場所 | 変更 | なぜ安全か |
| --- | --- | --- |
| `ServerPreparedRequest.promptIDs` / `.vision` | `fileprivate` → モジュール内部 | 読み手が 1 つ増えただけ。`public` にはしていない |
| `ServerProperties` | `supportsVision` を追加 (既定 `true`) | Gemma の `/props` は同じ答えを返す |
| `Command/main.swift` | 家族分岐、`forceLogitsHead: !isOrnith`、起動ログに `family=` | Gemma の枝は元のまま |

`Sources/TurboFieldfareCLI/RunQwen.swift` は `QwenReasoningSplitter` を
ライブラリのものに差し替えただけで、出力は変わらない (実物で確認)。

## 8. この文書が動かした結論

| 対象 | 更新 |
| --- | --- |
| [04](04-PHASES.md) Phase 8 | **通った。**サーバーが Ornith を開き、HTTP から答える。`ExpertCacheBudget` の勘定は**足す必要が無かった** (§4-3) |
| [03 §5](03-DESIGN.md) | 「slot あたり 62.8 MiB のスナップショットを持つ」という**推奨は採らなかった。**prompt cache を持たない方に決め、`cache_n` は常に 0 |
| [04](04-PHASES.md) 次にやること | 計測 (Phase 6) と、§6-1 の判断が残る |
| [23](23-PHASE5-TOOLS.md) 「誰も呼んでいない」 | **サーバーが呼んだ。**`QwenChatGrammarBuilder` に読み手ができた |
