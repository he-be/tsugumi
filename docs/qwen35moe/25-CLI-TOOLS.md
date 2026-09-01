# 25. CLI の `--tools` と、logit を書かないヘッドで文法をかける話 (実測(手元)、2026-08-22)

[04-PHASES.md](04-PHASES.md) Phase 5 に残っていた最後の 1 行 — **CLI の
`--tools`**。[23](23-PHASE5-TOOLS.md) でパーサと GBNF は書けたが
**誰も呼んでいなかった**ので、実物のモデルに呼ばせた。

```
.build/release/TurboFieldfareCLI \
  --model scratch/ornith-oq4e-g64.gturbo \
  --messages-file scratch/qwen35/tools-messages.json \
  --tools scratch/qwen35/tools.json --tool-choice required \
  --temperature 0 --top-k 0 --top-p 1
```

| | |
| --- | --- |
| 中心の主張 | **融合ヘッドのままで文法がかかる。**Gemma は制約をかけるとき `forceLogitsHead: true` で語彙幅の logit を書き出し、`Sampler` が棄却再抽選する。Ornith の融合ヘッドは logit をどこにも書かない ([19](19-LM-HEAD-INT8.md)) ので、**同じ hidden をマスクつきでもう一度畳む**方を採った (§2) |
| 書いたもの | `qwen_lm_head_greedy_int8_rows_chunk_masked` と `encodeMaskedRescore` (§2)、`QwenForwardRunner.constrained` / `rescoreGreedy` (§2-2)、CLI の `--tools` / `--tool-choice` / `--parallel-tool-calls` (§3) |
| 検査 | `--qwen` が **57 → 63 本** (新規 6、うち 2 本は負例、§4-1)、`--qwen-constrain` が **9 本** (うち 1 本は負例、§4-2)、`swift test --no-parallel` が **1,329 → 1,338 件** |
| 費用 | 拒まれたトークン 1 個につき**ヘッドをもう 1 回**。実測 **4.086 ms** (素は 4.084 ms、どちらも 132 GB/s) — **マスクを読む費用は測れない** (§2-3) |
| Gemma | カーネルもランタイムも 1 行も動かしていない。共有部は CLI の引数と `run()` の分岐だけで、`--tools` は Gemma の install に対して**拒む** (§3-4)。既定 69 本は緑 |
| 残り | サーバー結線 (Phase 8)。`QwenChatGrammarBuilder` は**まだ誰も呼んでいない** (§6) |

---

## 1. なぜ「CLI に `--tools`」が難しかったのか

[23](23-PHASE5-TOOLS.md) の時点で揃っていたのは**文法の綴りとパーサ**である。
足りなかったのは**それを生成ループに当てる手段**で、そこに家族ごとの差があった。

Gemma 側の GEN-7 はこうなっている:

1. 制約をかける要求は `forceLogitsHead: true` でランナーを作る
2. ヘッドは語彙幅の FP16 logit をバッファに書く
3. `Sampler` が普通に 1 個引き、`allows` で聞き、**駄目なときだけ**
   `fillAllowedMask` を取ってホスト側で分布を作り直し、引き直す

Ornith の `QwenForwardRunner` には (1) も (2) も無い。**融合ヘッドしか無く、
それは argmax だけを返す** ([19](19-LM-HEAD-INT8.md))。508 MB の表を 1 回読んで
threadgroup ごとの最大値だけを残す作りで、**語彙幅のベクトルを書かないことが
その経路の全部**である。だから既存の `GenerationConstraintError.logitsUnavailable`
— 「融合ヘッドでは制約をかけられないので**断る**」— がそのまま当てはまる。

選べたのは 2 つ:

| 案 | 中身 | 採らなかった理由 |
| --- | --- | --- |
| A. logit ヘッドを足す | Gemma と同じく語彙幅を書き出すカーネルを書き、`Sampler` を通す | 496 KB の往復が**毎トークン**増える。Ornith の decode はまだ計測前 (Phase 6) で、制約を使わない run まで遅くする形になる |
| B. **マスクつきで畳み直す** | 拒まれたときだけ、同じ hidden を許可行だけで畳み直す | **採った。**常の手はマスクを 1 bit も読まない。費用は棄却のときだけ発生する |

B は GEN-7 の構造そのものである。参照実装 (`common_sampler_sample`) も
「まず引く、駄目なら全語彙をマスクして引き直す」で、**高い方が棄却時にしか
走らない**ことが設計の要点だった。違うのは棄却時に何を払うかだけで、
Gemma は「ホストで分布を作り直す」、Ornith は「表をもう 1 回読む」になる。

貪欲デコード限定という制約が、これを成り立たせている。Ornith の CLI は
温度 0 しか受け付けない (`--temperature` は無視されるのではなく**断られる**)
ので、必要なのは**許可集合の中の最大値**であって分布ではない。それは
マスクつき argmax そのものである。

## 2. マスクつきの再畳み込み

### 2-1. カーネル

`qwen_lm_head_greedy_int8_rows_chunk_masked` は `_raw` との差が 1 行しかない:

```metal
const bool scored =
    row < vocab && ((allowed_bits[row >> 5] >> (row & 31u)) & 1u) != 0u;
```

- **行の演算は `qwen_head_int8_row` を共有する。**したがって全許可のマスクは
  `_raw` と**同じ答えを出す** — 検査の 1 本目がそれである (§4-1)
- `row` は SIMD group 内で一様なので**分岐は発散しない**
- **前段の RMSNorm は走らせない。**`xNormedBuffer` には直前の `_raw` が書いた
  行がそのまま入っている。ここで正規化をやり直すと、2 回の畳み込みが
  「同じトークンについて」食い違う道が 1 本生まれる。関数を分けずに
  `encodeMaskedRescore` が正規化を**持たない**ことがその保証である
- マスクは 1 行 1 bit。語彙 248,077 行で **31,010 バイト**、常駐 1 本

### 2-2. ランナー側

`QwenForwardRunner.constrained(_:gate:position:)` が GEN-7 の形を持つ:

```
argmax を取る (ヘッド 1 回、これは制約が無くても払う)
  → gate.allows(token) が真なら、それで終わり (追加費用ゼロ)
  → 偽なら rescoreGreedy: 全語彙マスクを取り、bit に詰め、畳み直す
```

`rescoreGreedy` が見ているものが 3 つある:

| 見るもの | どうするか |
| --- | --- |
| 許可が 0 本 | `noAllowedToken` を投げる。**カーネルに投げさせない** — 採点行が 1 本も無い畳み込みは番兵 (`0xFFFFFFFF`) に落ちるので、それはトークンではない |
| 畳み直した答え | もう一度 `gate.allows` に通す。単票とマスクは同じ制約が 2 回答えたものなので、食い違ったら `maskedDrawRejected`。負例 1 本がこれを踏む (§4-2) |
| 回数 | `constraintRescores` に数える。CLI の footer が `rescored=N` で出す |

`prefill` 経由の第 1 トークンにも同じものがかかる。`tool_choice: required`
では**その 1 手目こそが前置き規則の話**なので、ここを外すと文法の意味が変わる。

`ConstraintGate` (停止トークンは `mayEndHere` だけで決まり、制約には
聞かない) はランナーの中で組む。CLI からは `GenerationConstraint` を渡すだけで、
**停止 ID の規則は 1 か所にしか無い**。

### 2-3. 費用 — マスクを読む分は測れない

`--qwen --qwen-bench` の実物語彙 (248,077 行 / 540 MB):

| | 中央値 (n=20) | 帯域 |
| --- | ---: | ---: |
| `qwen_lm_head_int8` | 4.084 ms | 132 GB/s |
| `qwen_lm_head_int8 masked` (全許可) | **4.086 ms** | 132 GB/s |

差は 2 マイクロ秒で、この経路の分散より小さい。**この経路は帯域で決まって
いる**ので (540 MB を読むのが仕事の全部)、31 KB のマスクを足し読みしても
測れる差にならない。全許可で測っているのは意図的で、疎なマスクは行を飛ばす
ので**最良値**になる — 棄却されたトークンが実際に払うのはこちらの値である。

つまり **1 トークン拒まれるごとに約 4.1 ms**。§5 の実測では 63 トークンの
生成で 1 回だった。

## 3. CLI

### 3-1. 引数

| 旗 | 意味 |
| --- | --- |
| `--tools <path>` | 関数宣言の JSON 配列。**OpenAI の封筒** (`{"type":"function","function":{…}}`) と**裸の宣言** (`{"name":…}`) の両方を受ける (§3-2) |
| `--tool-choice <s>` | `auto` (既定) / `none` / `required` / 宣言済みの関数名 |
| `--parallel-tool-calls on\|off` | 1 ターンに複数の呼び出しを許すか (既定 on) |

`--tools` は `--messages-file` を要る。宣言を書くのは**テンプレートが描く
システムターン**なので、`--prompt` の素通し経路には置き場所が無い —
黙って落とすと「ツールを無視するモデル」に見えるので、そこは弾く。

### 3-2. 封筒と裸の両方を受ける理由

サーバーが受け取るのは封筒の形である (`OpenAITool`)。手で書くファイルは
たいてい裸の宣言になる。**片方しか受けないと、ここで通るファイルが
サーバーで通らない (またはその逆) という差ができる**ので、両方受ける。
`parameters` が無ければ引数を取らない関数として扱う (`.object([:])`)。
名前の重複は弾く — 文法が曖昧になり、パーサのスキーマ引きが任意になる。
どちらも**出力には現れない**壊れ方なので、入口で落とす。

### 3-3. 生成ループ

`--tools` を渡したときだけ `QwenStructuredAssistantDecoder` に切り替わる。
渡さないときは今までの `QwenReasoningSplitter` がそのまま走る:
**宣言していない run でモデルが `<tool_call>` を書いたら、それはテキスト
である**という扱いを変えないため。

GEN-6 (思考中は遅延文法を抑止する) はここで駆動する:

```swift
let events = try decoder.consume(tokenID: id, delta: delta)
constraint?.setSuppressed(decoder.isInsideReasoning)
```

サーバーが `ServerThoughtSuppression` でやっていることと同じ規則だが、
**別の型は要らなかった**。あちらは `<|channel>` の境界トークンが「その
トークンが**残す**状態」を持つので専用の型が要る。Ornith の decoder は
`isInsideReasoning` を境界の規則込みで持っているので、**そのトークンの後の
状態**をそのまま聞けばよい — 次の一手が判定される状態は、まさにそれである。

呼び出しは本文の後に **1 行 1 個の JSON** で stdout に出る:

```
{"id":"call_…","name":"get_weather","arguments":{"city":"Kyoto"}}
```

`arguments` は**パーサが戻した JSON** であって、モデルが書いた XML では
ない。つまりクライアントに送られるのと同じものが出る。

### 3-4. Gemma の install には出さない

Gemma 4 にもツールの形式・パーサ・文法はあるが、**それはサーバーに結線されて
いて CLI には無い**。`--tools` を Gemma の install に渡すと、
`Model.declaredFamily` を読んだ時点で断る (モデルは開かない)。

## 4. 検査

### 4-1. `--qwen` に 6 本 (合成入力、モデル不要)

行の演算は `_raw` と共有しているので、ここで採点しているのは
**どの行を採点したか**だけである。参照は同じファイルの double の logit。

| 見るもの | 中身 |
| --- | --- |
| マスク全許可 == 素の argmax | 7696 / 7696。マスク経路が素の経路と同じ答えを出す |
| 勝者を落とすと 2 位 | GPU 5090 / 参照 5090 |
| 疎なマスク (8192 行中 127 行) | GPU 5338 / 参照 5338。**8 行の threadgroup が丸ごと空になる**塊が生まれる形で、そこが `-INFINITY` のまま畳まれて番兵が答えに出ないことまで見る |
| 許可が 1 本 | GPU 738 / 許可 738 |
| **負例** `マスクを読まない` | 読まなければ勝者 7696 が返る。上の「2 位」の案に検出力があることを示す |
| **負例** `ワード内の bit 順が逆` | 逆順で選ばれたのは 765 (許可したのは 738)。**落ちずに別の行を採点する**形の間違い |

### 4-2. `--qwen-constrain` に 9 本 (実物、スタブ制約)

```
.build/release/TurboFieldfareKernelCheck --qwen-constrain scratch/ornith-oq4e-g64.gturbo
```

**文法ではなくスタブを当てるのが要点である。**文法の判定はモデルが書く
テキストの性質なので、1 回も棄却しない run は棄却経路について何も言わない。
このスタブは**言われたとおりに棄却する**。プロンプトと参照は
[14 §6](14-REFERENCE.md) の生成スモーク (先頭 8 トークン)。

| 見るもの | 結果 |
| --- | --- |
| 許可が全部なら参照と同じトークン | 8 本一致、`rescored=0`。**制約は見ているだけで、通り道を変えない** |
| `accept` は出たトークンを順に受け取る | 8 / 8 |
| 勝者を落とすと 1 回だけ畳み直す | step 0 が 248068 → **248058**、`rescored=1` |
| 許可が 1 本なら毎手それが出る | 4 手とも 4090、`rescored=4`。**語彙 248,077 行での bit 詰め**を見ているのはこの 1 本 |
| 許可が 0 本なら `noAllowedToken` | 落ちた (position 0) |
| **負例** 単票とマスクが食い違えば落とす | `maskedDrawRejected` (token 248068) |
| `mayEndHere` が真なら停止トークンは通る | `[248068]` で止まる、`rescored=0` |
| `mayEndHere` が偽なら停止トークンは隠れる | step 0 が 248058 に変わり、`rescored=1` |
| 通した停止トークンは `accept` に渡らない | `accepted []` |

最後の 3 本は `ConstraintGate` の規則 (停止 ID は `mayEndHere` だけで決まり、
制約には聞かない / 生成を終わらせた停止 ID は `accept` に渡さない) を、
**実物のヘッドの上で**見ている。

### 4-3. 純粋な部分

`swift test --no-parallel` は **1,338 件** (1,329 + 新規 9)。新規は
`--tool-choice` の 4 値と旗の飲み込み (`--tool-choice --quiet` が関数名
`--quiet` にならない)、`--tools` と `--prompt` の排他、宣言ファイルの
2 つの形 / `parameters` の省略 / 名前の重複 / 封筒の型違い。

## 5. 実物 — モデルが呼び出しを書いた

`get_weather(city: string, days?: integer)` を 1 本宣言し、温度 0。
同じ会話は `--tools` 無しだと **25 トークン**、`--tools` を渡すと
**292 トークン**になる — 差の 267 はツールの規約説明で、テンプレートが
システムプロンプトに動く例を書くからである ([23 §6](23-PHASE5-TOOLS.md))。

| run | `--tool-choice` | 新規 | `rescored` | 出たもの |
| --- | --- | ---: | ---: | --- |
| 「京都の天気は」 | `required` | 27 | **0** | `{"city":"Kyoto"}` |
| 同上 | `auto` (遅延) | 27 | **0** | 同じ呼び出し |
| 同上 | `none` (文法なし) | 27 | — | 同じ呼び出し |
| 同上 + `--thinking on` | `auto` | 55 | **0** | 推論 106 字 → 同じ呼び出し |
| 「フランスの首都は」 | `none` | 8 | — | `The capital of France is Paris.` |
| 「フランスの首都は」 | **`required`** | 63 | **1** | 下記 |

上 3 行が言っているのは「**この検体では文法が 1 回も効いていない**」という
ことである。モデルの素の argmax がすでに整形式で、制約は同じ道を確認して
いただけ ( `none` と `required` が同じ 27 トークンを出した)。文法の値打ちは
**そうでないときに出る**ので、外した検体を 1 つ作った。

最後の行がそれで、**素の run と最初の 8 トークンがそのまま同じ**である:

```
The capital of France is Paris.

Now, regarding your second question about the weather in Kyoto — I can help with that. Let me check the current weather for you.

{"id":"call_…","name":"get_weather","arguments":{"city":"Kyoto"}}
```

`rescored=1`。**拒まれたのは 9 手目の停止トークン 1 個だけ**である
(`required` の文法は前置きに任意のテキストを許すが、呼び出しを書くまで
`mayEndHere` が偽なので停止できない、[23 §2-3](23-PHASE5-TOOLS.md))。
モデルはそこで「では 2 つ目の質問ですが」と自分で辻褄を合わせてから
呼び出しを書いた。これは文法が綴らせたのではなく**モデルが選んだ続き**で、
文法が禁じたのは「ここで終わること」だけである。

## 6. 残したもの

| 残るもの | 中身 |
| --- | --- |
| **サーバー (Phase 8)** | `QwenChatGrammarBuilder` は書けているが**まだ誰も呼んでいない**。`ServerInference` / `ChatRequestParser` / `ServerGenerationPlan` は Gemma の型を通っており、Ornith の分岐が無い。この文書で埋まったのは「制約を実物にかける手段」までで、要求の解釈は手つかず |
| **入れ子 JSON の往復** | [23 §5-2](23-PHASE5-TOOLS.md) のまま。`--tools` は文字列引数しか実物で通していない |
| **並列呼び出し** | 文法は通す (`--parallel-tool-calls on` が既定) が、**実物が並列で書いた run はまだ見ていない** |
| **`tool_choice: auto` の保証** | [23 §7](23-PHASE5-TOOLS.md) のまま。遅延文法は「始まった呼び出しが整形式である」までしか約束しない |
| **サンプリング** | この経路は貪欲のまま。マスクつき argmax は分布を作らないので、温度を入れるには結局 §1 の案 A が要る |

## 7. この文書が動かした結論

| 対象 | 更新 |
| --- | --- |
| [04](04-PHASES.md) Phase 5 | **閉じた。**残っていた CLI の `--tools` が入り、実物が呼び出しを書いた |
| [04](04-PHASES.md) 次にやること | サーバー結線 (Phase 8) と計測 (Phase 6) だけになった |
| [19](19-LM-HEAD-INT8.md) | 融合ヘッドに**兄弟が 1 本増えた** (`_masked`)。素の経路のカーネルは 1 命令も変えていないので、[19](19-LM-HEAD-INT8.md) の 4.0 ms / 134 GB/s はそのまま |
| [23](23-PHASE5-TOOLS.md) 「誰も呼んでいない」 | **CLI が呼んだ。**サーバーはまだ |
