# 12. S3 — tools × 画像 × 思考

実装・実測: 2026-08-19、M3 Pro 18GB / macOS 15.7.5 / `scratch/gemma4-qat.gturbo` /
16K / 32 スロット / `--draft-block-size 4` / `--thinking on` / temp 0。
表記は mtp 系と同じ: **実測** / **導出** / **未確認**。

[11-S2](11-S2-TEMPLATE-PROBE.md) でテンプレート側に制約が無いと分かったので、
degrade 案は捨てて `encodeToolChat` を直した。**穴 2 (pi で画像が使えない) と
穴 3 (tools 宣言時に思考が無効) は同時に閉じた。**

---

## 0. 結論 — 4 つ

1. **tools + 思考でツール呼び出しが出る** (§2 の A、**実測**)。
   `finish=tool_calls`、`bash` 1 本、思考 217 字。
2. **tools + 画像 + 思考で画像に答える** (§2 の B)。G1 が求めていた形で、
   同じ要求は S2 以前は 400 だった。
3. **temp 0 で再現する** (§2 の D)。思考も本文も md5 一致。
4. **画像を含まない tool 要求のプロンプトは 1 バイトも変わっていない** (§1)。
   本文が 1 つのテキスト part の turn は、従来どおり**文字列として**
   テンプレートに渡している。

## 1. 何を直したか

`Tokenizer.encodeToolChat` は (a) `enable_thinking: false` 固定で、
(b) message の content を `String?` としてしか渡していなかった。両方直した:

| 変更 | 内容 |
| --- | --- |
| `enableThinking` 引数 | `additionalContext` にそのまま流す。既定 `false` なので、呼ばない側の挙動は変わらない |
| `ToolChatMessage` 型 | tool のメタデータ (`tool_calls` / `tool_call_id` / `name`) と、画像を持てる本文 (`[ContentPart]`) を 1 つにした turn 型。`Message` からの変換 initializer 付き |
| content の送り方 | **画像を含まない turn は今までどおり文字列**、画像を含む turn だけ `[{type:text}, {type:image}]` の配列。テンプレートは文字列本文を trim し配列の item は trim しないので、この分岐が「画像を使わない要求の出力不変」を保証する |
| サーバー側 | `renderPrompt` が「画像あり + tools あり」を tool テンプレートへ回す。`<\|image\|>` の位置は同じなので `VisionPromptAssembler` はそのまま噛む |
| 400 の削除 | `OpenAIModels.swift` の「画像 + tools」「画像 + tool 呼び出し/developer ターン」の 2 つの拒否を外した |
| 思考の判定 | S1 で入れた `rendersThinking` (tools のとき false に落とす) を削除。両テンプレートが描けるので、要求の値がそのまま prompt の値になる |

出力不変はテストで固定した (`ServerReasoningTests`):
`Message` 経由と `ToolChatMessage` 経由の同じ会話がトークン列まで一致する。

## 2. モデルは実際にどう振る舞うか (**実測**)

pi が対話セッションで宣言するのと同じ形 (`read` / `bash` / `write` を毎要求宣言、
streaming、temp 0) で 4 本。B と D は同じ要求である。

| | 要求 | finish | tool call | 思考 | 本文 | 生成 tok | 壁時計 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| A | tools + 思考、`/etc/hosts` を読ませる | `tool_calls` | **`bash` 1 本** | 217 字 | 0 字 | 86 | 6.2 s |
| B | tools + **画像** + 思考 (G1 の形) | `stop` | なし | 1,246 字 | 50 字 | 479 | 25.4 s |
| C | tools + 画像、思考 OFF | `stop` | なし | 0 字 | 55 字 | 29 | 5.6 s |
| D | B の再実行 | `stop` | なし | 1,246 字 | 50 字 | 479 | 23.3 s |

- **A**: 思考の中で「`bash` を使うべき」と決めてから呼んでいる
  (思考の冒頭: "The user wants to read the first line of the `/etc/hosts` file.
  I should use the `bash` tool…")。**思考とツール呼び出しは両立する。**
- **B / C**: どちらも画像の内容を日本語で答えた。思考 ON は「ラグやクッション」、
  OFF は「スマートフォン」に触れており、**答えの中身は変わる** (思考は生成であって
  後付けの説明ではない、という当たり前の帰結)。どちらが正確かはここでは採点しない。
- **D**: 思考・本文とも md5 が B と一致 (`84554f2482aa` / `f8f88781a544`)。
- **費用**: 思考 ON の画像 1 枚が 479 トークン / 25 s に対し、OFF は 29 トークン /
  5.6 s。**思考は 10 倍以上の生成を伴う** — 常用の既定にするかは別の判断で、
  今のところ既定は `--thinking off` のままにしてある。

## 3. これで閉じた穴

[README](README.md) の穴 2 と穴 3、そして [TODO.md](../../TODO.md) の
「pi の対話モードで画像が使えない」は、**サーバー側だけで閉じた** — pi への
upstream 要望 (画像ターンだけ `tool_choice: none` を送る) は不要になった。

pi 側の設定で必要なのは 1 つだけ: `~/.pi/agent/models.json` の当該モデルを
`"reasoning": true` にし、`compat.thinkingFormat` を `"qwen-chat-template"` に
すること (思考を pi 側から切り替えたい場合。サーバーの `--thinking on` だけでも動く)。

## 4. まだ確かめていないこと (**未確認**)

- **tool ループの中で思考がどう扱われるか。**テンプレートは assistant の
  `reasoning_content` を thought channel として描くが、条件は
  「最後の user 発話より後 かつ tool call を持つ assistant ターン」だけである
  ([11-S2 §3](11-S2-TEMPLATE-PROBE.md))。`encodeToolChat` は今そのキーを
  送っていないので、**tool 結果を返した 2 手目に前の思考は載らない**。
  これで劣化するかは測っていない。
- **プロンプトキャッシュとの相互作用。**思考 ON の要求はキャッシュに参加しない
  ([10-S1 §5](10-S1-REASONING.md)) ので、tools + 思考の多ターンは毎ターン
  再 prefill する。pi の常用でどれだけ効くかは未測定。
- **長い会話での挙動。**上の実測は全部 1 ターンである。
