# 22. Phase 5 — トークナイザとテンプレート、そして CLI が喋った (実測(手元)、2026-08-21)

[04-PHASES.md](04-PHASES.md) 次の一手 **#19**。**Phase 5 の出口条件が通った** —
`TsugumiCLI --model … --messages-file …` が日本語でも英語でも答え、
`<think>` は答えと別の流れに分かれる。

```
./.build/release/TsugumiCLI --model scratch/ornith-oq4e-g64.moepack \
  --messages-file scratch/qwen35/ja.json --thinking on \
  --max-new 128 --temperature 0 --verification trusted-install
```

| | |
| --- | --- |
| 結果 | **「日本の首都は東京です。」が stdout に出た。**推論 (192 文字) は stderr、答え (13 文字) は stdout (§5) |
| 書いたもの | `QwenTokenizer` / `ByteLevelDecoding` / `QwenDetokenizer` (§1〜2)、`RunQwen.swift` (§5)、`Model.declaredFamily` |
| 検査 | `--qwen-tokenizer` が **223 本すべて緑、うち 4 本は負例** (§3)。`swift test --no-parallel` は **1,297 件** (1,282 + 新規 15) すべて緑 |
| 上流との突き合わせ | `tokenizers` / `transformers` の実行結果を JSON に落として比較 (`Scripts/qwen35/tokenizer_fixture.py`)。エンコード 18 本・デコード 86 本・テンプレート 6 本 |
| 食い違い | **1 つだけ、tools の JSON** (§4-2)。`tool \| tojson` はホストの JSON 書き出しで、swift-jinja と Python は区切り文字とキー順が違う |
| Gemma | **1 行も動かしていない。**CLI の分岐は `manifest.arch.family` を読む 1 か所だけ (§5-1) |

---

## 1. `GFTokenizer` に分岐を足すのではなく、兄弟を書いた

[04](04-PHASES.md) Phase 5 の箇条書きは「`verifyDecoderConfiguration` に
**ByteLevel を許す分岐**」「`GemmaDecoding` の**兄弟**」と書いていた。前者は
やめて、**`QwenTokenizer` という別の型**にした。理由は 2 つある。

**1 つ目: 特殊トークンの要求が両立しない。**`GFTokenizer.init` は
`<bos>` / `<pad>` / `<turn|>` / `<|tool_call>` など 10 本を**非オプショナルの
プロパティ**として要求する。Ornith にはそのどれも無い (`bos_token` は null、
`add_bos_token` は false)。共有型にするなら 10 本を optional にするか family で
分けるかで、**サーバー・アプリ・CLI の 111 か所**がその変更を受ける。

**2 つ目: Gemma 側の検査は「よそのトークナイザを弾く」ためにある。**
`verifyDecoderConfiguration` の存在理由は `TSUGUMI_TOKENIZER_DIR` が
どこでも指せることで、そこに ByteLevel を**通す**分岐を足すと、その検査は
Gemma を守らなくなる。**片方の家族の正解はもう片方の家族の破損である。**

したがって新しい型は Gemma の検査を緩めない。代わりに**両方向**を検査に入れた
(`QwenDecoderConfigTests`): Ornith の宣言は Gemma の検査に落ち、Gemma の宣言は
Ornith の検査に落ちる。

共有したのは**作法**の方である: デコードは 1 トークンずつ・可逆、バッチは
ストリーミングの push ループ、HF の `clean_up_tokenization_spaces` は通さない。

## 2. ByteLevel — 「全部がバイト」の世界

Gemma は `Sequence[Replace(▁→␣), ByteFallback, Fuse]` で、バイトなのは
`<0xXX>` の run だけだった。Ornith は `decoder: ByteLevel` で、**すべての
トークンがバイト**である。GPT-2 の byte↔unicode 表で 1 文字が 1 バイトを表し、
**符号点はトークン境界のどこででも割れる** — 日本語では常時割れる。

### 2-1. ストリーミングの規則 — 「決着した分だけ出す」

`ByteLevelRun.push` が返すのは、**後から来るバイトではもう変えられない**分の
文字列だけ。抱えるのは末尾の「まだ完成し得る」UTF-8 列 (最大 3 バイト) **だけ**で、
**すでに壊れている列は決着済み**として即座に出す (`\u{FFFD}` になる)。だから
不正な列でストリームが凍らない。

これで**ストリーミング = バッチ**が構成上成り立つ。Swift の
`String(decoding:as:)` は maximal subpart ごとに 1 つの U+FFFD を置く規則で、
これは `huggingface/tokenizers` (Rust の lossy 変換) と同じ規則である。
切れ目は必ず「決着した境界」なので、maximal subpart が境界をまたぐことは無い。

### 2-2. 落ちた特殊トークンをまたいで **run は融合する**

上流は `skip_special_tokens=True` のとき、**復号器に入れる前に ID を落とす**。
したがって前後のバイト列は**1 本の run** になる。「マーカーで run を閉じる」と
実装すると、そこで符号点が割れていた場合だけ壊れる。

実物で確かめた (`--qwen-tokenizer` の検体):

| 列 | keep (special を残す) | skip (落とす) |
| --- | --- | --- |
| `[158, 15434]` | `∘` | `∘` |
| `[158, 248046, 15434]` | `�<\|im_end\|>��` | **`∘`** |
| `[158, 248068, 15434]` (`<think>` は special ではない) | `�<think>��` | `�<think>��` |

**乱数の ID 列ではまず当たらない。**先頭バイトで終わるトークン (26 本) と継続
バイトで始まるトークン (85 本) を表から選んで組む必要がある — fixture 生成側で
そうしている。**この 3 行が負例 `commitAtSkippedSpecial` の検出力そのもの**である。

## 3. 検査 223 本 — 上流と 2 実装で突き合わせる

`Scripts/qwen35/tokenizer_fixture.py` が上流 (`tokenizers` /
`transformers`) を**この機械で 1 度だけ**回して JSON に落とし、
`--qwen-tokenizer` がそれと比較する。**同じ `tokenizer.json` を読む別実装
どうし**なので、同じ場所で同じように間違うことは無い。

```
swift run -c release TsugumiKernelCheck \
  --qwen-tokenizer scratch/ornith-oq4e-g64.moepack \
  --qwen-tokenizer-fixture scratch/qwen35/tokenizer-fixture.json
```

| 見るもの | 本数 | 中身 |
| --- | ---: | --- |
| マーカーの ID | 9 | `<\|im_start\|>` 248045 / `<\|im_end\|>` 248046 / `<\|endoftext\|>` 248044 / `<think>` 248068 / `</think>` 248069 / tool 4 本 |
| 停止トークン | 1 | `[248046, 248044]` (`generation_config.json`) |
| 語彙 | 3 | 248,070 (= 最大 ID + 1)、追加トークン 26 本と special フラグ、**全語彙がバイト表に載る** |
| エンコード | 18 | 日本語・英語・コード・絵文字・空白・追加トークンの文字列そのもの |
| デコード | 172 | 86 検体 × (special 残す / 落とす)。うち 12 本は §2-2 の符号点またぎ |
| ストリーミング | 1 | 86 検体すべてで push ループ = バッチ |
| テンプレート | 15 | 6 検体 (§4) |
| 生成の復号 | 1 | Phase 3/4 の 55 トークン fixture が文として読める (§3-1) |
| **負例** | **4** | 下表 |

負例は「もっともらしく間違えた復号器」を同じ検体に通したもの。**どれも
どこかで外れなければならない。**

| 負例 | 外れた検体数 (172 中) | 最初の例 |
| --- | ---: | --- |
| `commitAtSkippedSpecial` (落とすマーカーで run を閉じる) | 4 | `"���"` ≠ `"໿"` |
| `perTokenBytes` (トークンごとに復号し、途中を抱えない) | 12 | `"���"` ≠ `"໿"` |
| `literalTokenText` (バイト表を通さず文字をそのまま) | 170 | `"ãģĵãĤĵãģ«ãģ¡ãģ¯"` ≠ `"こんにちは"` |
| `maskedByteAlphabet` (`scalar & 0xFF` で表の代わりにする) | 164 | `"�#5�$5…"` ≠ `"こんにちは"` |

**上 2 本の検出力が 4 と 12 しかないことに意味がある。**これは「符号点が
トークン境界で割れた検体」の数であって、英語だけを見ていれば 0 本である。

### 3-1. Phase 3/4 の 55 トークンが、初めて文になった

これまで `--qwen-decode` / `--qwen-prefill` が突き合わせていたのは ID の列で、
**このリポジトリの中で文に戻したことは無かった**:

```
<think>
The user is asking in Japanese: "Where is the capital of Japan? Please answer in one sentence."

The capital of Japan is Tokyo (東京).

I should answer in one sentence in Japanese as requested.
</think>

日本の首都は東京です。<|im_end|>
```

[14 §6](14-REFERENCE.md) が Python 側で見ていた文と同じものが、Swift の
復号器から出た。

## 4. チャットテンプレート — 上流の jinja をそのまま

Gemma 側はテンプレートを Swift で書いていた (上流に `chat_template` が無い
から)。**Ornith は持っている**ので、`tokenizer_config.json` の
`chat_template` を swift-jinja が描画する。`enable_thinking` はテンプレート
自身の変数で、`--thinking` がそのまま渡る:

| `enable_thinking` | 生成プロンプトの末尾 |
| --- | --- |
| true | `<\|im_start\|>assistant\n<think>\n` (**開けたまま渡す**) |
| false | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` (**閉じて渡す**) |

検体 6 本 (user のみ ×2、system+user、複数ターン、`reasoning_content` つき、
tools) は**ID まで一致**し、その ID を復号し直した文字列も一致する。

### 4-2. tools の JSON だけ食い違う — これは**所見**であって許容差ではない

テンプレートは `{{- tool | tojson }}` と書く。`tojson` は**ホスト側の JSON
書き出し**で、JSON が規定していない 2 点で両者は違う:

| | swift-jinja | Python (`transformers`) |
| --- | --- | --- |
| 区切り | `{"a":1,"b":2}` | `{"a": 1, "b": 2}` |
| キー順 | **昇順** (訂正、下) | 挿入順 (`type` → `function`) |
| 非 ASCII | `\uXXXX` に逃がす | 逃がさない (`ensure_ascii=False`) |

**キー順についての訂正 (2026-08-22、[23 §5-1](23-PHASE5-TOOLS.md)):** ここには
当初「辞書由来で**不定**」と書いてあったが、それは間違いだった。swift-jinja は
両端で並べ替える — `Value(any:)` が Swift の辞書を写すときにキーを昇順にし、
`tojson` が `.sortedKeys` で書き出す。**どちらの実装も決定的で、綴りが 2 通り
あるだけ**である。この訂正は設計に効く: 再描画が必ず昇順なので、
ツール呼び出しの文法が引数を昇順で綴ってよい ([23 §2](23-PHASE5-TOOLS.md))。

**トークン列は違う。**tools を使う経路は Phase 5 の出口条件に入っていない
(ツール呼び出しのパーサと GBNF は未着手、§7) ので、検査は `<tools>` ブロックを
**JSON として解析して比較**し、その外側は逐語で比較する形にした — 同じツール・
同じフィールド・同じ値であることは言えて、**同じバイト列だとは言っていない。**

**モデルはこのブロックを Python 側の書き方で学習している。**影響の測定は
Phase 6 に置いた ([23 §5-2](23-PHASE5-TOOLS.md))。

## 5. CLI — `RunQwen.swift`

`run(args:)` の中の分岐ではなく**別の経路**にした。トークナイザから下は共有
できるものが無い (ChatML の framing、`QwenForwardRunner` 自身の K/V と再帰状態、
sampler も投機も vision も Ornith 側が無い)。`RawCompletion` を共有しようと
すると、**Gemma 4 の実測値を担いだ型を 3 つ一般化する**ことになる。

| 段 | どうしたか |
| --- | --- |
| 家族の判定 | `Model.declaredFamily(at:)` が `manifest.json` の `arch.family` だけを読む。**検証ではなくヒント** — `Model.load` の関門は後で全部通る |
| プロンプト | `--prompt` は逐語 (Gemma と同じ意味、BOS は無い)、`--messages-file` はテンプレート適用 |
| 生成 | `generateGreedyPrefilled` (prefill → decode、[21](21-PHASE4-PREFILL.md)) |
| 出力 | `QwenDetokenizer` を 1 トークンずつ。`<think>` / `</think>` の **ID** で経路を切り替え、推論は stderr、答えは stdout |
| 断る条件 | `--image` (Phase 9 が無い) / `--draft-block-size > 0` (Phase 7 が無い) / greedy でない設定 (runner は融合ヘッド)。`--stop` の文字列は効かない旨を警告する |

`<think>` の判定を**文字列ではなくトークン ID** でやるのは、答えの中で
`<think>` と綴られても境界が動かないようにするため。テンプレートは thinking on の
とき**ブロックを開いたまま**渡すので、開始状態はプロンプト末尾のマーカーから読む。

### 5-1. Gemma 経路への影響

`Run.swift` に足したのは 4 行 (family を読んで Qwen なら別関数へ) だけで、
その下は 1 文字も変えていない。`swift test --no-parallel` は **1,297 件すべて緑**
(既存 1,282 + 新規 15)。

### 5-2. 数字 (**運用値ではない。n=1**)

`--temperature 0`、`--verification trusted-install`、スロット 32、
チャンク 2048。`bench.sh` の作法 (クールダウン・反復) の対象外。

| 走らせたもの | prompt | 生成 | decode | tok/s | load | TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 日本語 / thinking off | 23 | 7 | 2.23 s | 3.14 | 1.27 s | 1.62 s |
| 英語 / thinking off | 21 | 8 | 2.01 s | 3.97 | 1.22 s | 1.17 s |
| 日本語 / thinking on | 21 | 53 | 6.42 s | 8.26 | 1.22 s | 1.13 s |
| `--prompt "日本の首都は"` (逐語) | 3 | 24 | 2.85 s | 8.43 | 1.26 s | 0.57 s |

**解釈は書かない** (反復 3 未満)。運用点の測定は Phase 6。

## 6. ついでに直したもの — エキスパート計数の phase

`QwenForwardRunner` / `QwenPrefill` は `ExpertTelemetry.beginPhase` を
呼んでいなかったので、**decode の要求まで prefill 列に入っていた**
(最初の CLI 実行の footer は `decode hit=0.0% 0/0` だった)。
`RealForwardRunner` と同じ位置で stamp するようにした:

```
[expert prefill hit=0.0% 0/2282 | decode hit=64.1% 10659/16640]   ← 32 スロット、n=1
```

**Phase 6 の 1 番目の作業 (`startTrace` の TSV を 1 回取る) はこの stamp が
無いと成立しない** — TSV の phase 列が全部 prefill になる。

## 7. この Phase が動かした結論と、残したもの

| 対象 | 更新 |
| --- | --- |
| [04](04-PHASES.md) Phase 5 | **出口条件が通った。**日本語と英語で CLI が答え、`<think>` が分離される |
| [04](04-PHASES.md) 次の一手 #19 | **完了** (tokenizer / テンプレート / CLI)。**ツール呼び出しは残り** (下) |
| [10 §3](10-MLX4BIT-AUDIT.md)「tokenizer は確実に弾かれる」 | **片づいた。**弾かれるのは Gemma のローダで、それは**正しい** — Ornith 用のローダは別に書いた (§1) |
| [04](04-PHASES.md) Phase 6 | 前提が 1 つ整った (§6)。TSV の phase 列が意味を持つようになった |

**残したもの:**

- ~~**XML 形のツール呼び出し**のパーサと GBNF ビルダ、`GrammarVocabulary` の
  ByteLevel 化~~ → **完了** ([23](23-PHASE5-TOOLS.md))
- **sampler / 投機 / vision** は Ornith 側が無い。CLI は断る (§5)
- **`--stop` の文字列**は効かない (トークン停止のみ)
- **サーバー** ([03 §5](03-DESIGN.md)、Phase 8) は手つかず。prompt cache の
  スナップショットは再帰状態と噛み合わない、という問題がそのまま残っている
- 速度は n=1 (§5-2)。運用点は Phase 6
