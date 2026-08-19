# 10. S1 — Server の Reasoning ON

実装: 2026-08-19、M3 Pro 18GB / macOS 15.7.5 / Swift 6.2 系。
表記は mtp 系と同じ: **実測** / **導出** / **未確認**。
測定は §6、触ったファイルは §7。

[README](README.md) の S1 はゴール条件 (G4) そのもので、tools が絡まない限り
純粋な配線だと書いた。実際そのとおりで、配線以外に決めることは
**API 表面 2 つ**(要求の綴り・応答のフィールド)と、
**思考をプロンプトキャッシュに乗せるか**の 3 点だった。

---

## 0. 結論 — 5 つ

1. **要求は 2 つの綴りを両方受ける。** `chat_template_kwargs.enable_thinking`
   (vLLM 慣行、pi が送るもの) と `reasoning_effort` (OpenAI 形)。
   どちらが来るかはクライアント側の実装で決まっていて、こちらでは選べない (§1)。
2. **応答は `reasoning_content`** (DeepSeek 慣行)。本文とは別のフィールドで、
   stream では `delta.reasoning_content` として流す (§2)。
3. **プロセス既定は `--thinking on|off`(既定 off)。**要求が明示したら常に要求が勝つ。
4. **tools を宣言した要求は、頼まれても思考を描画しない** (§4)。
   `Tokenizer.swift:453` の `enable_thinking: false` が理由で、S1 では直さない
   (S2 の probe で事実を確定してから)。400 にはせず、その要求は思考なしで通す。
5. **思考 ON の要求はプロンプトキャッシュに参加しない** (§5)。
   KV には思考トークンが入るが、次のターンを素で描き直すとそれは入らない —
   ヒットとミスで別のプロンプトになるため、両側とも切った。

## 1. 要求の綴り — なぜ 2 つとも受けるか (**実測**)

pi の `openai-completions` アダプタは、モデル設定の `compat.thinkingFormat` で
送り方を変える。`qwen-chat-template` の場合に送るのはこれ
(`@earendil-works/pi-ai/dist/api/openai-completions.js:584-589`):

```js
else if (compat.thinkingFormat === "qwen-chat-template" && model.reasoning) {
    params.chat_template_kwargs = {
        enable_thinking: !!options?.reasoningEffort,
        preserve_thinking: true,
    };
}
```

`reasoning_effort` を送るのは `openai` / `deepseek` / `baseten` などの
別フォーマットで、`qwen-chat-template` では**送らない**。
つまり pi 側から見ると、この 2 つは排他の設定項目である。
一方で応答の読み取りは共通で、`reasoning_content` / `reasoning` /
`reasoning_text` の順に見る (同 `:352`)。

だから採ったのは「両方受けて、応答は `reasoning_content` に固定」である。

| 要求の書き方 | 解釈 |
| --- | --- |
| `{"chat_template_kwargs": {"enable_thinking": true}}` | 思考 ON |
| `{"reasoning_effort": "minimal"｜"low"｜"medium"｜"high"｜"max"}` | 思考 ON |
| `{"reasoning_effort": "none"｜"off"}` | 思考 OFF |
| どちらも無い | `--thinking` の既定に従う |
| 両方あって食い違う | 400 `invalid_value` |
| `enable_thinking` が bool でない / kwargs がオブジェクトでない | 400 `invalid_value` |
| 未知の `reasoning_effort` | 400 `unsupported_value` |

**このテンプレートに思考の「量」は無い**ので、effort は on/off の意味だけを読む。
`chat_template_kwargs` の他のキー (pi が併せて送る `preserve_thinking`) は
**受け取って無視する**: このテンプレートに使い道が無いキーで 400 を返すと、
それ以外はまったく正しい要求が落ちるため。

## 2. 応答 — 思考をどこに出すか

生成テキストの思考部分は chat template の thought channel
(`<|channel>thought … <channel|>`) に入る。`StructuredAssistantDecoder` は
もともとこの channel を**捨てて**いた (tool 経路で本文に漏らさないため)。
S1 では捨てる代わりに名前をつけて出せるようにした
(`emitsReasoning`、既定は従来どおり捨てる)。

- 非 stream: `choices[0].message.reasoning_content`。
  思考が空のときは**フィールドごと出さない** — 「思考しなかった」と
  「思考が空だった」をクライアントが取り違えないように。
- stream: `choices[0].delta.reasoning_content`。本文の delta と同じ順序で流れる
  (思考が先、本文が後)。
- `usage.completion_tokens` は**思考トークンを含む**。思考は生成テキストであって
  無料ではない、という事実をそのまま出す。`max_tokens` も同じ予算から減る。
- `stop` 文字列は**本文にだけ**当てる。クライアントの stop は表示するテキストに
  ついての指定で、モデルがたまたま思考の中で書いた語で打ち切ると驚かれる。
- `finish_reason` の決め方は変えていない。

## 3. 思考 ON のとき何が変わるか (プロンプト側)

`enableThinking: true` のとき、テンプレートは
(a) system ターンの先頭に `<|think|>` を置き、
(b) 生成プロンプトの末尾で thought channel を**開けたまま**にする
(`Tokenizer.swift:388,406`)。off のときは末尾で開いて即閉じるので、
モデルは思考せずに答える。

off のときサーバーは生の delta をそのまま本文にしていた。
ON では channel のマーカーが stream に出てくるので、
**tools を使わない経路でも decoder を通す**必要がある
(`ServerInference.swift`、`needsToolTemplate || thinking`)。
ここを通さないと思考が本文に混ざる。

## 4. tools と思考 (S1 の範囲外)

tools を宣言した要求 (および tool / developer ターンを含む要求) は
`encodeToolChat` を通り、そこは `enable_thinking: false` を固定で渡している。
したがって S1 の配線を足しても**この経路の出力は 1 バイトも変わらない**。

要求を 400 で拒まないのは、pi の既定セッションが毎要求に built-in tools を
宣言するためである。拒むと「tools ON のセッションでは何も返らない」になる。
今は「思考なしで普通に答える」。どちらが正しいかは S2 の probe
(テンプレートが tools 宣言時に thinking を描けるか) の結果で決まる。

ログはこの区別を落とさない:

```
request chatcmpl-… accepted streaming=true thinking=on      # 要求が何を頼んだか
request chatcmpl-… completed in … finish=stop reasoning=812B # 実際に何が出たか
```

## 5. プロンプトキャッシュを切った理由

`--prompt-cache-mode single-prefix` は、直前の生成の KV をそのまま次の要求の
前置きとして使い回す。思考 ON だと、その KV には**モデルが書いた思考トークンが
入っている**。ところが次のターンをキャッシュ無しで描き直すと、クライアントが
送り返してくるのは答えだけなので、思考は**入らない**。

同じ会話がヒットしたかミスしたかで別のプロンプトになる、という状態は
この repo が採点に使っている再現性そのものを壊す。よって
`promptCacheParticipates` に thinking を足し、**読み書きの両側で外した**
(画像要求と同じ扱い)。代償は思考 ON の会話が毎ターン再 prefill することで、
これは S4 (CLI 対話) や将来の「思考を履歴に残す」設計と一緒に見直す論点として残す。

## 6. 測定 (G4) — **実測** 2026-08-19

M3 Pro 18GB / macOS 15.7.5 / `scratch/gemma4-qat.gturbo` / 16K /
`--verification trusted-install --draft-block-size 4 --thinking on` /
temp 0 / `max_tokens` 2048 / `sample_imgs/IMG_2113.JPG` /
プロンプトは `bench/mtp_goal_prompt.json` (mtp のゴールタスクと同じ)。
1 プロセスの中で warmup 2 回 → 思考 ON 3 回 → 思考 OFF 3 回。
`decode t/s` = `completion_tokens ÷ (最後のチャンク − 最初のチャンク)`。

**G4 (単発 Vision + Reasoning ON で tg ≥ 30 t/s) は 80 スロットで満たし、
常用点の 32 スロットでは満たさない。**

| | 32 スロット (常用点) | 80 スロット |
| --- | ---: | ---: |
| decode t/s (思考 ON、3 回) | 27.76 / 28.94 / 28.59 → **中央値 28.59** | 39.11 / 35.67 / 37.50 → **中央値 37.50** |
| decode t/s (思考 OFF、3 回) | 33.56 / 33.56 / 33.20 → 中央値 33.56 | 32.92 / 37.27 / 25.93 → 中央値 32.92 |
| TTFT (思考 ON、3 回) | 4.51 / 4.52 / 4.38 s | 21.07 / 10.22 / 10.33 s |
| 壁時計 (思考 ON、3 回) | 42.1 / 40.6 / 40.9 s | 47.7 / 39.5 / 38.1 s |
| 生成トークン (思考 ON) | 1,043 | 1,043 |
| 思考 / 本文 | 2,493 字 / 332 字 | 2,493 字 / 332 字 |
| finish | stop | stop |

読み取れること 4 つ:

1. **Server で Reasoning ON の Vision 単発が通る。**思考は `reasoning_content`
   に、本文は `content` に分かれて届く。3 回とも思考 2,493 字・本文 332 字で
   一致 (temp 0)。**スロット数を変えても文字数は同じ** — 速度の設定であって
   出力の設定ではない、という mtp と同じ性質がここでも保たれている。
2. **G4 の 30 t/s は 80 スロットでしか出ない** (37.50 対 28.59)。
   32 は常用点として選んだ設定で ([SERVER_RUNBOOK §1](../SERVER_RUNBOOK.md))、
   ゴール条件の数字とは別物である。**ここは埋めずに残す** — 数字を取りに行くなら
   スロットを上げる、常用を優先するなら 30 t/s を諦める、という選択で、
   決めるのは S5 (既定値の自動選択) の仕事だから。
3. **32 のほうが TTFT が 2〜5 倍速い** (4.4 s 対 10〜21 s)。壁時計で見ると
   両者はほぼ同じ (40.9 s 対 39.5 s) — 32 は「最初のトークンが早く出て、
   その後が遅い」。**未確認**: 原因はプレフィル中のキャッシュ充填 I/O の差だと
   考えているが、切り分けていない。
4. **思考は予算を食う。**別途 `max_tokens` 600 で測ったときは 3 回とも
   600 トークンすべてを思考に使い、本文が 1 字も出ないまま `finish=length` で
   終わった。**思考 ON のクライアントは `max_tokens` を上げる必要がある** —
   このタスクでは思考だけで約 700 トークン。

思考 OFF の 3 回は 80 スロット側だけばらつきが大きい (25.9〜37.3)。
3 回では足りていないので、**この行は比較に使わない**。

## 7. 触ったファイル

| ファイル | 変更 |
| --- | --- |
| `Sources/TurboFieldfare/Tokenization/StructuredAssistantDecoder.swift` | `emitsReasoning` と `.reasoning` イベント。既定は従来どおり thought を捨てる |
| `Sources/TurboFieldfareServer/Core/ServerArguments.swift` | `--thinking on\|off`、`ServerThinkingPolicy` |
| `Sources/TurboFieldfareServer/Core/OpenAIModels.swift` | `chat_template_kwargs` / `reasoning_effort` の受理と解決 (`OpenAIReasoning`)、`ValidatedChatRequest.enableThinking` |
| `Sources/TurboFieldfareServer/Core/ServerInference.swift` | テンプレートへの配線、decoder の適用条件、`reasoning_content` の蓄積、キャッシュ判定 |
| `Sources/TurboFieldfareServer/Core/HTTPServer.swift` | 応答 2 形式への `reasoning_content`、既定の受け渡し |
| `Sources/TurboFieldfareServer/Core/ServerLog.swift` | `thinking=` と `reasoning=` |
| `Sources/TurboFieldfareServer/Command/main.swift` | 起動行に `thinking=` |
| `Tests/TurboFieldfareServer/ServerReasoningTests.swift` | 新規 16 本 (要求の解釈 / decoder / キャッシュ判定 / HTTP 2 形式) |

## 8. S2 へ渡すもの

- tools 宣言時にテンプレートが `enable_thinking: true` を描けるか (§4)。
- 同じテンプレートが画像 content part を描けるか (README の穴 2)。
- 描けるなら S3 で `encodeToolChat` に両方を配線し、`OpenAIModels.swift` の
  画像 + tools 拒否を外す。描けないなら degrade 案を選ぶ。
