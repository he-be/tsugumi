# 40. 文法つきの MTP と、サーバーの `--draft-block-size 2` — エージェント経路 (pi) が投機で回る (実測(手元)、2026-08-22)

[36 §5](36-MTP-DECODE.md) から [39](39-RESIDENCY-COMMIT.md) まで、MTP には
**満たしていない受入条件が 1 つ**残っていた: **文法・ツール呼び出しと併用できない**
(`SpeculativeError.constrained`)。同時に、サーバーは投機を**起動時に断って**いた
([26 §4-4](26-PHASE8-SERVER.md))。この 2 つは同じ 1 つの用途を塞いでいる —
**pi のようなコーディングエージェントは必ず `tools` を宣言する**ので、宣言が
文法を立て、文法が投機を弾き、[39 §4](39-RESIDENCY-COMMIT.md) で MTP が一番効いた
形のターン (a1 ×1.333) がまさに使えなくなっていた。

本書はその 2 つを閉じる。**新しい Metal カーネルは 0 本**、既存の
`qwen_lm_head_greedy_int8_rows_chunk_masked` ([25 §2](25-CLI-TOOLS.md)) に
**どの行を読むか**を渡せるようにしただけである。

---

## 0. 結論を先に

| # | 論点 | 結論 |
| --- | --- | --- |
| 1 | なぜ排他だったのか | 「文法と投機は理屈が合わない」からではない。**再畳み込みが読む行が固定**だった — `encodeMaskedRescore` は `xNormedBuffer` (ヘッド chain 内部の 1 行) しか見ず、検証パスの 2 行は**別のバッファ** (`verifyNormedBuffer`) に正規化されるので、そこに置いてある行を再採点する方法が無かった (§1) |
| 2 | 直し方 | `encodeMaskedRescore` に `hiddenNormed` / `hiddenNormedOffset` を足し、`constrained(_:gate:position:hidden:)` で行を名指す。**カーネルは同じもの**、バインドするバッファが変わるだけ (§2) |
| 3 | 順序の規則 | 行 0 の判定 → 行 0 を emit (= `gate.accept`) → **その後で**行 1 の判定。制約は逐次機械なので、前の位置を受理していない状態で次を問うのは**別の質問**になる (§2-1) |
| 4 | 受理判定 | `draft == y0` の `y0` は**制約後**の値。マスクが引きを動かしても、ドラフトがその結果と一致していれば行 1 は正しい入力で走っている (§2-2) |
| 5 | 出力は動くか | **動かない。**3 腕 (素の decode / MTP / 強制棄却の MTP) が **55/55 と 63/63 でトークン一致**。文法が実際に効いた検体 (`rescored=1`) を含む (§3) |
| 6 | サーバー | **`--draft-block-size 2` を受ける**ようになった (0 か 2 のみ。3〜8 は「この家族は 1 本しかドラフトしない」と断る)。sidecar は**起動時**に載せる。ログに `mtp=2 rounds=… accept=…` が出る (§4) |
| 7 | 実タスクの取り分 (サーバー、2,935 トークンのプロンプト + tools + thinking、n=2) | 素の decode **14.301 / 13.875**、MTP **16.034 / 16.095** tok/s (中央値で **×1.14**)。prefill は両腕とも 10.3〜11.3 秒で**動かない** (§5) |
| 8 | エージェント用途で一番効く制約は投機ではない | **prompt cache が無い**こと ([26 §4-3](26-PHASE8-SERVER.md))。2,935 トークンのターンで **TTFT 11.3 秒**、しかも**毎ターン全部**払う。MTP が返すのは decode の 14% で、この 11 秒には 1 秒も効かない (§6-1)。→ **[41](41-PROMPT-CACHE.md) で入れた。続きのターンは 0.78 秒** |
| 9 | まだ直っていないもの | **MTP と素の decode では答えが変わる** ([36 §5-3](36-MTP-DECODE.md))。T 行カーネルと decode カーネルの加算順の差で、今回の長文ターンでも思考文の 494 文字目で分岐した。**投機の中立性 (強制棄却との一致) は保たれている** (§6-2) |
| 10 | 検査 | `--qwen-constrain` が **9 → 15 本** (負例 1 → 3)。`--qwen` 63 / `--qwen-tools` 36 / Gemma 既定 69 / `swift test` **1,350 件**すべて緑 (§7) |

---

## 1. 排他の正体 — 「行が固定されていた」

[25 §2](25-CLI-TOOLS.md) の GEN-7 は、融合ヘッドが logit をどこにも書かないので
**同じ hidden をマスクつきでもう一度畳む**という形をしている。その「同じ
hidden」の在処が問題だった:

```swift
// 直す前
head.setBuffer(xNormedBuffer, offset: 0, index: 0)   // ← chain 内部の 1 行
```

decode の `step` も prefill の最終チャンクも、最後の頭の 1 行を
`xNormedBuffer` に残して戻る。だから「拒まれたら同じ行をもう一度」は
引数を 1 つも要らなかった。

ところが幅 2 の検証パス ([36 §3](36-MTP-DECODE.md)) は
`encodeGreedyDecodeRows` を使い、2 行を**自分のスクラッチ**
(`verifyNormedBuffer`) に正規化する。`xNormedBuffer` には**どちらの行も無い**。
つまり `SpeculativeError.constrained` が言っていた「まだ結線されていない」は、
**設計の非互換ではなく引数 1 つの不在**だった。

## 2. 直した形

```swift
public func encodeMaskedRescore(commandBuffer:,
                                hiddenNormed: MTLBuffer? = nil,      // ← 足した
                                hiddenNormedOffset: Int = 0,         // ← 足した
                                weights:, scales:, biases:,
                                allowedBits:, outToken:, d:, vocab:) -> Bool
```

nil は `xNormedBuffer`、つまり**既存の呼び出しは 1 バイトも変わらない**。
`QwenForwardRunner.constrained` も同じ形で行を受け取り
(`hidden: (buffer: MTLBuffer, offset: Int)?`)、速度の検査 ([25 §2-3](25-CLI-TOOLS.md)
の 4.086 ms 対 4.084 ms) の対象も同じカーネルのままである。

### 2-1. 2 行にどう当てるか — 順序が規則のすべて

制約は逐次機械なので、**行 1 の判定は行 0 が受理された後**でなければならない。
ループはこう並べた:

| 順 | すること | なぜそこ |
| --- | --- | --- |
| 1 | 行 0 の argmax を `gate` に問う (拒まれたら `normed` の**行 0** を再畳み込み) | ここまでに emit されたトークンは全部 accept 済み |
| 2 | 受理判定 (§2-2) と巻き戻し | 行 1 を使うかどうかが決まる |
| 3 | 行 0 を emit — `gate.accept` → `onToken` | `onToken` の中で呼び出し側が `setSuppressed` を更新する ([26 §3-1](26-PHASE8-SERVER.md)) ので、**次の判定より前**でなければならない |
| 4 | 行 1 の argmax を `gate` に問う (拒まれたら `normed` の**行 1**) | 行 0 の accept を織り込んだ状態で問える |
| 5 | 行 1 を emit | |

`gate.accept` を `onToken` の**前**に置くのは
`runGreedyCompletion` ([26 §1](26-PHASE8-SERVER.md)) と同じ規則で、
GEN-6 が言う「このトークンが残す状態を次の引きが判定に使う」である。

### 2-2. 受理判定は「制約後」の値と比べる

```swift
let y0 = try constrained(row0Draw, gate: gate, position: produced.count,
                         hidden: (normed, 0))
if !Self.noDraft && draft == y0 { … }
```

`row0Draw` (素の argmax) ではなく `y0` (マスク後) と比べる。行 1 は
**`draft` を入力として**走っているので、実際に出たトークンが `draft` と
同じでありさえすれば、行 1 は正しい前置きの上に載っている — マスクが引きを
動かしたかどうかは関係が無い。逆に `row0Draw` と比べると、マスクが動かした
瞬間に**間違った前置きの行 1 を採用する**。

### 2-3. 検証パスの時計を止める場所

再畳み込みは 508 MB のテーブルの 2 度目の読みなので、**そのトークンの費用**で
あって検証パスの費用ではない。`stats.verifySeconds` は行の読み出し直後に
止め、再畳み込みはその外に置いた ([26 §1](26-PHASE8-SERVER.md) の
「制約の再畳み込みは prefill に入れない」と同じ線引き)。

## 3. トークンは動かない (**実測(手元)**)

CLI、`scratch/ornith-oq4e-g64.moepack`、temp 0、`--thinking on`、
`--dump-tokens` の id 同士を比較。**n=1 なので数字だけ置く**。

| 検体 | 腕 | tok/s | `rescored` | 一致 |
| --- | --- | ---: | ---: | --- |
| tools / `--tool-choice auto`「京都の天気」 | 素の decode | 11.786 | 0 | — |
| | `--qwen-mtp` (P1 89.7%, a=1.897) | 14.733 | 0 | **55/55** |
| | `--qwen-mtp` + `TF_QWEN_MTP_FORCE_REJECT=1` | 10.018 | 0 | **55/55** |
| tools / `--tool-choice required`「フランスの首都」(§3-1) | 素の decode | 12.754 | **1** | — |
| | `--qwen-mtp` (P1 77.1%, a=1.800) | 14.811 | **1** | **63/63** |
| | `--qwen-mtp` + 強制棄却 | 9.964 | **1** | **63/63** |

### 3-1. 文法が実際に効いた検体を選んである

1 行目の検体は [25 §4](25-CLI-TOOLS.md) と同じで**素の argmax がすでに整形式**
なので、文法は 1 回も引きを動かさない (`rescored=0`)。これだけでは
「行を名指す再畳み込み」を 1 度も通らない。

そこで**ツールと無関係な質問に `--tool-choice required`** を当てる検体を並べた
([26 §6-1](26-PHASE8-SERVER.md) が見つけた形)。ここでは停止トークンが 1 度
拒まれて `rescored=1` になり、**投機ループの中で `normed` の行を再採点する経路が
実際に走る**。それでも 3 腕とも 63/63 で一致し、モデルは自分で辻褄を合わせてから
`get_weather` を書いた。

## 4. サーバー結線

```
.build/release/TsugumiServer --model scratch/ornith-oq4e-g64.moepack \
  --model-id ornith-1.5-35b-a3b --port 8092 --ctx-size 32768 \
  --expert-cache-slots 32 --verification trusted-install --draft-block-size 2
→ … family=qwen3_5_moe context=32768 slots=32 expert_io=mmap mtp=2 …
```

| 決めたこと | 中身 |
| --- | --- |
| **幅は 2 だけ** | `QwenServerSession.validateFlags` は 0 か 2 しか通さない。Gemma のドラフタは `n` 本のブロックを出すが、この家族のヘッドは**1 本しか出さない**ので 4 は「遅い同じもの」ではなく**カーネルの無い形**である ([33 §3-2](33-MTP-ACCEPTANCE.md)) |
| **sidecar は起動時** | `--draft-block-size 2` なら `load` の中で `attachMTPHead`。480 MB の sidecar が無い機械では**起動が失敗する** — 最初の completion で失敗するより良い ([26 §4-4](26-PHASE8-SERVER.md) の作法をそのまま裏返した) |
| **`max_tokens` に 1 位置の余裕** | 投機ループは「まだ本物と決まっていない行」を先に走らせるので、`prompt + maxNew + 1 <= maxContext` を要求する。`max_tokens: -1` (文脈の残り全部) をそのまま渡すと precondition に当たるので、投機時は残りを 1 減らす |
| **RSP-3 の語彙に載せる** | `ServerSpeculativeSummary(blockTokens: 2, rounds: passes, proposed: passes, accepted: accepted)`。1 パス = 1 提案なので `rounds == proposed`。ログは `mtp=2 rounds=28 accept=0.893` |
| **要求の形は 1 行も変わらない** | ループの選択は `speculative` の 1 分岐だけで、`onToken` は**同じクロージャ**を両方に渡す。デコーダも文法も停止規則も家族の話であって腕の話ではない |

## 5. 実タスク (**実測(手元)**)

サーバー、`--ctx-size 32768 --expert-cache-slots 32`、temp 0 (既定)、
プロンプトは [`bench/qwen35/t4-summarize.json`](../../bench/qwen35/t4-summarize.json)
(**2,935 トークン**、テンプレートと tools 宣言込み) に `tools` 1 本と
`enable_thinking: true` を付けたもの、`max_tokens: 160`。腕ごとにプロセスを
建て直し、要求の間に 12〜15 秒空けた。**n=2、中央値ではなく両方を置く**。

| 腕 | prefill (ms) | decode (tok/s) | 受理率 |
| --- | ---: | ---: | ---: |
| `--draft-block-size 0` | 10,950 / 10,306 | 14.301 / 13.875 | — |
| `--draft-block-size 2` | 11,306 / 10,416 | **16.034 / 16.095** | 0.787 / 0.767 |

短いターン (プロンプト 273 / 314) も同じ形で通っている:
`rounds=28 accept=0.893`、`rounds=17 accept=0.941`。後者は
**assistant の `tool_calls` → `role: tool` の往復**を含む要求で、
ストリーミングで答えた。

**取り分は decode だけ、prefill は動かない。**これは設計どおり
(投機は decode ループの中にしか無い) だが、§6-1 の理由でエージェント用途では
そこが効く。

## 6. 残っているもの

### 6-1. エージェント経路で一番効く制約は prompt cache が無いこと

> **同日に片づいた ([41](41-PROMPT-CACHE.md))。**下の 11 秒は**初回ターンだけ**の
> 費用になり、続きのターンは 0.72〜0.78 秒で始まる。以下は実装前の記述。

上の表の 1 ターンは **prefill 11.3 秒 + decode 10.0 秒**である。pi のような
クライアントは**毎ターン会話全部を送り直す**ので、この 11 秒は
`cache_n` が常に 0 である限り**ターンごとに満額**かかる
([26 §4-3](26-PHASE8-SERVER.md))。MTP が返す 14% は残り 10 秒側にしか効かない。

[34](34-PROMPT-CACHE-ESTIMATE.md) の snapshot-restore 型 (「巻き戻さない。次の
要求が前回の厳密な延長のときだけ続きから走る」) はこの家族でも成立し、
机上の取り分は**最大 9.2 秒 / ターン**、追加メモリは**その場保持なら 0 バイト**。
**着手していない。判断はユーザー** ([04](04-PHASES.md) #31 (b))。

### 6-2. MTP と素の decode では答えが変わる (既知)

同じ長文ターンで、思考文の**494 文字目**から分岐した (どちらも整合した文で、
どちらかが壊れているのではない)。機序は [36 §5-3](36-MTP-DECODE.md) のとおり
**T 行カーネルと decode カーネルの加算順**で、文法の有無とは無関係である。
**投機の中立性** (強制棄却の対照とトークン列が一致) は §3 のとおり保たれている。

### 6-3. 既定は変えていない

- `--draft-block-size` の既定は **0**。MTP を使うのは明示的な指定のときだけ
- `TF_EXPERT_MMAP_RESIDENCY_ASYNC` は既定 **off** のまま
  ([39 §0 #9](39-RESIDENCY-COMMIT.md))。4 腕 4 タスクすべてで勝っているが
  Gemma と共有の経路なので**ユーザー判断**
- `TF_QWEN_MTP_PREFETCH` も既定 off ([39 §2](39-RESIDENCY-COMMIT.md))

## 7. 検査

### 7-1. `--qwen-constrain` を 9 → 15 本に (負例 1 → 3)

```
.build/release/TsugumiKernelCheck --qwen-constrain scratch/ornith-oq4e-g64.moepack
  PASS  幅 2: 許可が全部なら参照と同じトークン — rescored=0, 8 tokens
  PASS  幅 2: accept は出たトークンを順に受け取る — accepted 8 of 8
  PASS  幅 2: 勝者を落とすと素の decode と同じ別トークンが出る — step 0 = 248058 (素の decode 248058), rescored=1
  PASS  幅 2: 許可が 1 本なら毎手それが出る — produced [4090, 4090, 4090, 4090], rescored=4
  PASS  負例: 幅 2 でも許可 0 本は noAllowedToken
  PASS  負例: 幅 2 でも単票とマスクの食い違いは落とす
PASS  15 cases, 3 of them negative controls
```

**行を取り違えても「マスクが許すトークン」は返る**ので、それを捕まえるのは
答えを 1 つに固定する 2 本である: 透明な制約 (参照と全一致でなければならない) と、
**許可が 1 本だけ**の制約 (毎手その id でなければならない)。前者は
[14 §6](14-REFERENCE.md) の fixture、後者は 248,077 行のビット詰めを
そのまま踏む。sidecar が無い機械では MTP の 6 本を **SKIP** する。

### 7-2. 動かしていないもの

| 検査 | 結果 |
| --- | --- |
| `--qwen` | **63 本**緑 |
| `--qwen-tools` | **36 本**緑 (負例 6) |
| Gemma の既定 | **69 本**緑 |
| `swift test --no-parallel` | **1,350 件 / 206 スイート**緑 |

## 8. コードと文書の根拠 (2026-08-22 に確認した現物)

| 事実 | 場所 |
| --- | --- |
| 行を名指せる再畳み込み | `Sources/Tsugumi/Kernels/Qwen/QwenLMHeadChainInt8.swift` `encodeMaskedRescore(hiddenNormed:hiddenNormedOffset:)` |
| 行を受け取る GEN-7 の口 | `Sources/Tsugumi/Runtime/Inference/QwenForwardRunner.swift` `constrained(_:gate:position:hidden:)` / `rescoreGreedy` |
| 2 行への当て方と受理判定 | `Sources/Tsugumi/Runtime/Inference/QwenSpeculativeDecode.swift` (`SpeculativeError.constrained` は**消した**) |
| サーバーの幅 2 | `Sources/TsugumiServer/Core/QwenServerSession.swift` `validateFlags` / `load` の `attachMTPHead` / `runGreedyCompletionMTP` の分岐 |
| 検査 | `Sources/TsugumiKernelCheck/QwenConstrainCheck.swift` (`generateMTP`) |
| CLI の 3 腕 | `--qwen-mtp` × `--tools`、対照は `TF_QWEN_MTP_FORCE_REJECT=1`。id は `scratch/qwen35/mtp40/*.json` (**実測(手元)**) |
| サーバーの A/B | `/tmp/req.json` (t4 + tools + thinking) を `--draft-block-size 0` と `2` の 2 プロセスに (**実測(手元)**) |
| 建て方と pi の設定 | [docs/SERVER_RUNBOOK.md](../SERVER_RUNBOOK.md) §8 |
