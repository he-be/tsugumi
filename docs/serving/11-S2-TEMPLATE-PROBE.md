# 11. S2 — テンプレート検定 (probe、実装なし)

検定: 2026-08-19、M3 Pro 18GB。対象は `scratch/gemma4-qat.gturbo/tokenizer/chat_template.jinja`
(353 行)。描画は**サーバーが本番で使うのと同じ swift-jinja エンジン**
(`Tokenizers.applyChatTemplate`) で行った。
表記は mtp 系と同じ: **実測** / **導出** / **未確認**。

[README](README.md) の穴 2 (pi で画像が使えない) と穴 3 (tools 宣言時に思考が無効)
は「根が同じ = tools 宣言時のチャットテンプレート制約」という前提で並べてあった。
**この前提は誤りだった。**

---

## 0. 結論 — 3 つ

1. **テンプレートは tools・画像・思考を同時に描ける** (§1、**実測**)。
   3 つ揃った入力を渡すと、`<|think|>` と `<|tool>` を持つ system ターン、
   `<|image|>` を含む user ターン、開いたままの生成プロンプトが出る。
2. **制約はテンプレートではなく `Tokenizer.encodeToolChat` にある** (§2)。
   この関数は (a) `enable_thinking: false` を固定で渡し、(b) message の content を
   **文字列としてしか渡さない**。画像 content part を描く経路が呼び出し側に無い。
3. **したがって S3 は degrade 案ではなく本筋で進められる** (§3)。
   `encodeToolChat` に 2 つ足して、`OpenAIModels.swift` の
   「画像 + tools は 400」を外す。**未確認なのはモデルの振る舞いのほうで**、
   描画ではない (§4)。

## 1. 何を描けるか (**実測**)

`tools` に `read` を 1 本宣言し、user ターンをテキスト part + 画像 part にして、
`enable_thinking` を振った。実際の描画結果 (特殊トークンを可視化):

```
=== enable_thinking: false ===
<bos><|turn>system
<|tool>declaration:read{description:<|"|>Read a file<|"|>,parameters:{ properties:{ path:{ description:<|"|>path<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>path<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>

<|turn>user
what is this<|image|><turn|>
<|turn>model
<|channel>thought
<channel|>

=== enable_thinking: true ===
<bos><|turn>system
<|think|>
<|tool>declaration:read{...同じ...}<tool|><turn|>

<|turn>user
what is this<|image|><turn|>
<|turn>model
```

テンプレート側の根拠 (行は `chat_template.jinja` の該当ブロック):

| 問い | テンプレートの答え |
| --- | --- |
| tools 宣言時に thinking を描くか | 描く。system ブロックの条件は `enable_thinking or tools or messages[0].role in ['system','developer']` で、`<|think|>\n` の注入は **tools と独立** |
| 生成プロンプトは | `add_generation_prompt` で `<\|turn>model\n`、`enable_thinking` が偽のときだけ `<\|channel>thought\n<channel\|>` を足す。ここにも tools は出てこない |
| tools 宣言時に画像 part を描くか | 描く。content が sequence のとき `item['type'] == 'image'` → `<\|image\|>`。この分岐は**全メッセージ共通のループの中**にあり、tools 無しの経路ではない (`audio`/`video` も同じ場所にある) |
| 画像プレースホルダの綴り | `<\|image\|>` — `VisionMediaTokenIDs.imageToken` と同じ。つまり既存の vision 組み立てがそのまま噛む |

これらは `Tests/TurboFieldfare/Core/Tokenization/ToolTemplateCapabilityTests.swift`
(5 本) に固定した。テンプレートを差し替えたら落ちる。

## 2. では制約はどこにあるか

`Sources/TurboFieldfare/Tokenization/Tokenizer.swift` の `encodeToolChat`:

```swift
var value: Tokenizers.Message = [
    "role": message.role.rawValue,
    "content": message.content,          // ← String? しか入らない
]
...
additionalContext: ["enable_thinking": false]   // ← 固定
```

- **思考**: `false` 固定。S1 で足したフラグはここに届いていない
  ([10-S1 §4](10-S1-REASONING.md))。
- **画像**: `GFTokenizer.Message` は `content: String?` で、
  画像 part を持つ型 (`MultimodalMessage`) は**テキスト経路にしかない**。
  サーバーの 400 (`images cannot be combined with tools`) は、この
  「渡す型が無い」を仕様として書き下したものだった (`PLAN_VISION §0-I-4`)。

つまり穴 2 と穴 3 は、**根が同じ**という点では当たっていたが、
その根は**テンプレートではなくこちらのエンコーダ**である。

## 3. S3 の設計 (この probe から決まること)

| やること | 場所 |
| --- | --- |
| `encodeToolChat` に `enableThinking` 引数を足し、`additionalContext` に流す | `Tokenizer.swift` |
| 画像 part を渡せる入口を足す (`MultimodalMessage` + tools)。返すのはトークン列のままでよい — `VisionPromptAssembler.makePrefillPrompt(tokens:images:ids:)` はトークン列を受け取って span を作るので、テキスト経路と同じ形で噛む | `Tokenizer.swift` / `ServerInference.renderPrompt` |
| 「画像 + tools は 400」を外す | `OpenAIModels.swift` |
| tool 経路も decoder を思考ありで通す (S1 の `rendersThinking` の条件を更新) | `ServerInference.swift` |
| プロンプトキャッシュの扱い: tool 継続 (`encodeToolResultContinuation`) は思考 ON だと S1 と同じ理由で外す必要がある | `ServerInference.swift` / `ServerPromptCache.swift` |

**先に決めておく細部が 3 つある** (どれもテンプレートを読んで分かったこと):

1. **過去ターンの思考は基本描かれない。**テンプレートは assistant の
   `reasoning` / `reasoning_content` を thought channel として描くが、条件は
   `loop.index0 > last_user_idx and message.get('tool_calls')` — つまり
   **最後の user 発話より後の、tool call を持つ assistant ターンだけ**である。
   pi が `preserve_thinking: true` で思考を送り返してきても、tool ループの
   途中以外は無視される。`encodeToolChat` は今どちらのキーも送っていない。
2. **assistant の文字列 content からは channel が剥がされる** (`strip_thinking`
   マクロ)。クライアントが思考込みの本文を送り返しても二重にならない。
3. **tool 結果は前方走査で拾われる。**assistant の `tool_calls` の直後に続く
   `role: tool` を舐めて `<|tool_response>` を作る。今のサーバーの
   メッセージ検証はこの形をすでに満たしている。

## 4. この probe が答えていないこと (**未確認**)

- **モデルが実際にどう振る舞うか。**描けることと、tools + 思考でモデルが
  正しく tool call を出せることは別である。S3 で実測する
  (少なくとも: 思考 ON + tools で tool call が 1 本出るか、
  思考 ON + 画像 + tools で画像の内容に答えるか)。
- **トークン数の増分。**`<|think|>` の注入自体は数トークンだが、思考は生成側で
  数百トークン使う ([10-S1 §6](10-S1-REASONING.md))。pi の既定セッションのように
  毎要求 tools を宣言する使い方で、コンテキストと待ち時間がどう動くかは未測定。
- **他のチェックポイント。**この検定は pin してある QAT チェックポイントに
  同梱された `chat_template.jinja` 1 本に対するものである。
