# 34. prompt cache の取り分 — 机上 (**導出**、2026-08-22)

**検討のみ。実装もベンチも走らせていない。既定は 1 つも変えていない。**
[32 §7](32-NVMAI-ADOPT.md) の 2 番目「prompt cache の取り分を机上で出す」への答え。
GPU は 1 度も使っていない — 数字はすべて既存の実測とコードの現物からの**導出**である。

運用ルールどおり、食い違いは番号の大きい本書が正 ([26](26-PHASE8-SERVER.md) /
[32 §2](32-NVMAI-ADOPT.md) との食い違いは §0 の表に挙げてある)。

**表記の約束:**

- **実測** — 既存の文書 / `bench/qwen35-results.tsv` に数字として在るもの。出典を必ず添える
- **導出** — 上の実測から式で出したもの。式も添える
- **仮定** — 測っていない前提。そう明記する
- **未確認** — 分からないもの。埋めない

**この文書の数字はほぼ全部が導出である。**新しい測定は 1 つもしていない
(GPU を使っていない)。

---

## 0. 結論を先に

| # | 問い | 答え |
| --- | --- | --- |
| 1 | 一番大きい取り分 | **約 9.2 秒** (導出)。t4 級の長文文脈 (2,698 tok) の会話の 2 ターン目、TTFT **10.6 → 1.5 秒**。短いチャットの 2 ターン目は 0.9 秒、ツールループの 1 ホップは 1.3 秒 (§1) |
| 2 | 32 §2 の式 `前回までのトークン数 × prefill の ms/tok` | **短いところでは正しく、長いところで最大 1.2 倍ほど過大**。prefill には **1.30 秒の床**があり、ms/tok は 498 トークンを境に 3.79 → 1.86 に落ちる (§1-1)。正しい形は `C(全長) − C(新規接尾)` |
| 3 | 1 エントリの費用 | **その場保持型なら 0 バイト。**KV も GDN 状態も既に確保済みのバッファの上に在る。切り離した写真を持つなら position 8,192 で **0.232 GB** (§2) |
| 4 | 予算に入るか | **入る。**運用点 (ctx 8192 / mmap / chunk 2048) の `ExpertCacheBudget.totalBytes` は約 **2.84 GB** に対し device の推奨は **12.88 GB** (実測、[docs/mtp/19](../mtp/19-M4.7-RESULTS.md))。0.232 GB は余裕の 2.3% (§2-3) |
| 5 | 「巻き戻せない」との両立 | **コードが既に両立させている。**`KVCacheManager.maximumSafeRewind` は再帰層が 1 本でもあれば **0** を返し (`KVCacheManager.swift:283-289`)、`ServerPromptCache.match` は `kvPosition − reusable <= maximumRewind` を要求する (`ServerPromptCache.swift:239`)。**この 2 行だけで「厳密な延長のみ」が出る** — 新しい規則を書く必要がない (§4-1) |
| 6 | 足りないもの | 32 §2 の (a)(b) に加えて **3 つ**ある: ① `QwenGreedyRun` が KV-backed トークン列を返さない ② 失敗経路の `reset()` (今は毎回先頭で reset しているので隠れている) ③ **再レンダの継ぎ目** — テンプレートが assistant ターンを構造体から描き直すので、ツールループは**最もミスしやすい**形でもある (§3、§4-3) |
| 7 | 実装規模 | 新規カーネル 0 本、Metal 0 行。Swift で **約 200 行** (§3-4) |

---

## 1. 取り分 — 式と数字

### 1-1. まず prefill の実測曲線を作る (導出)

32 §2 は取り分を `前回までのトークン数 × prefill の ms/tok` と書いた。
**その ms/tok は定数ではない。**[27 §5](27-PHASE6-THROUGHPUT.md)
が既にそう言っている — 「短いプロンプトの TTFT はトークン数に比例しない。
56 トークンでも層ごとに約 150 のエキスパートを読むので 1.46 秒かかる」。

運用点 (mmap / 32 スロット / lfu / チャンク 2048 / pipeline on) の**実測**
(27 §4 と §6-3、`bench/qwen35-results.tsv`、いずれも n=3 の中央値):

| プロンプト | チャンク数 | prefill 秒 (実測) | 平均 ms/tok |
| ---: | ---: | ---: | ---: |
| 48 (t3) | 1 | 1.52 | 31.7 |
| 56 (t1) | 1 | 1.46 | 26.1 |
| 60 (t2) | 1 | 1.54 | 25.7 |
| 498 (m.json) | 1 | 3.19 | 6.41 |
| 2,378 | 2 | 8.63 | 3.63 |
| 2,698 (t4) | 2 | 10.19 | 3.78 |

ここから **1 チャンク n トークンの費用 C(n)** を組む (**導出**)。
チャンク幅スイープ (27 §6-3、2,378 トークンを 512 / 1024 / 2048 で
14.53 / 10.57 / 8.63 秒) が、チャンクの足し算という形を裏づける。

```
C(n) = 1.30 + 0.00379·n        (n ≤ 498)     ← 55 tok=1.51 s と 498 tok=3.19 s の 2 点当て
C(n) = 3.19 + 0.00186·(n−498)  (498 < n ≤ 2048)
C(2048) = 6.08                                ← 8.63 − C(330) から逆算
prefill(N) = Σ C(チャンク)                    ← 幅 2048 で切った各チャンクの和
```

| 検算 | モデル | 実測 | 差 |
| --- | ---: | ---: | ---: |
| 2,378 / チャンク 1024 (3 本) | 10.89 | 10.57 | +3% |
| 2,378 / チャンク 512 (5 本) | 15.43 | 14.53 | +6% |
| 2,698 / チャンク 2048 (2 本) | 9.55 | 10.19 | −6% |

**モデルの精度は ±6%。**多チャンクで上振れするのは、後続チャンクが前のチャンクの
温めたエキスパートのページに相乗りするからだと読める (27 §6-2 の「mmap の腕では
ミスは転送量の指標ではない」と同じ話) が、**分けて測っていない (未確認)**。

**床は 1.30 秒。**48 トークンより短いプロンプトは**測っていない (未確認)**ので、
実測が保証するのは「48 トークンで 1.45〜1.52 秒」までである。以下では
床 = 1.30 秒 (モデルの外挿) を使い、**取り分は控えめに出る側**に倒している。

### 1-2. 取り分の正しい式

```
取り分 = prefill(N_全長) − prefill(N_新規接尾)
```

`N_新規接尾 = N_全長 − (前ターンの KV-backed トークン数)`。

32 §2 の `前回までのトークン数 × ms/tok` は、**両辺が同じ線形区間に居るときだけ
これと一致する**(床が引き算で消えるため)。区間をまたぐと過大になる:

| 場面 | 32 §2 の式 | 本節の式 | 差 |
| --- | ---: | ---: | ---: |
| §1-3 シナリオ 1 ターン 2 (cached 247) | 247 × 3.79 ms = 0.94 s | 0.94 s | 0% |
| §1-5 シナリオ 3 ターン 2 (cached 2,889) | 2,889 × 3.78 ms = 10.9 s | 9.2 s | **−16%** |

### 1-3. シナリオ 1 — 多ターン会話 (t1 を起点)

**仮定** (測っていない): 各ターンの生成は 192 トークン (27 §4 のベンチ設定)、
各ユーザ発話は 30 トークン、ターンの継ぎ目のテンプレートは 14 トークン
(`<|im_end|>\n` + `<|im_start|>user\n` + `<|im_end|>\n` +
`<|im_start|>assistant\n` + `<think>\n\n</think>\n\n`)。
thinking off。前ターンの KV-backed = プロンプト + 生成 − 1
(最後の 1 個は境界トークンで KV に入らない — §3-1)。

| ターン | 全プロンプト長 | cache 無し TTFT | 新規接尾 | cache 有り TTFT | **取り分** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 56 | **1.46 (実測)** | 56 | 1.46 | — |
| 2 | 292 | 2.41 | 45 | 1.47 | **0.94 s (−39%)** |
| 3 | 528 | 3.25 | 45 | 1.47 | **1.78 s (−55%)** |
| 4 | 764 | 3.68 | 45 | 1.47 | **2.21 s (−60%)** |
| 5 | 1,000 | 4.12 | 45 | 1.47 | **2.65 s (−64%)** |

すべて**導出**(1 行目のみ実測)。

### 1-4. シナリオ 2 — エージェントのツールループ

**実測のアンカーが 2 つある** ([26 §5](26-PHASE8-SERVER.md)):
`tools` + `tool_choice: auto` の 1 手目が **prompt 292 / predicted 39**、
ツール応答の往復のあとが **prompt 359**。差は **+67 トークン**で、これが
「assistant のツール呼び出しの再レンダ + `<tool_response>` 包み」の実測値である。

**仮定**: ループが同じ形で回る (毎ホップ +67 トークン、生成 39 トークン)。

| ホップ | 全プロンプト長 | cache 無し TTFT | 新規接尾 | cache 有り TTFT | **取り分** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **292 (実測)** | 2.41 | 292 | 2.41 | — |
| 2 | **359 (実測)** | 2.66 | 29 | 1.41 | **1.25 s (−47%)** |
| 3 | 426 | 2.91 | 29 | 1.41 | **1.50 s (−52%)** |
| 4 | 493 | 3.17 | 29 | 1.41 | **1.76 s (−56%)** |

**ただしこの形が一番ミスしやすい。**§4-3 を読むこと — テンプレートは
ツール呼び出しを**構造体から描き直す**ので、モデルが書いたバイトと一致する保証が無い。

### 1-5. シナリオ 3 — 長文文脈のエージェント (t4)、**一番大きい数字**

t4 は [01-MODEL.md](01-MODEL.md) の先頭 6,000 字を読ませる
2,698 トークンのプロンプト。実測 TTFT **10.19 秒**。

**仮定**: 同じ文書について続けて質問する (生成 192 / ユーザ 30 / 継ぎ目 14)。
2,048 を超える所は実測 10.19 秒をアンカーにして slope₂ = 1.86 ms/tok で伸ばす。

| ターン | 全プロンプト長 | cache 無し TTFT | 新規接尾 | cache 有り TTFT | **取り分** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **2,698 (実測)** | **10.19 (実測)** | 2,698 | 10.19 | — |
| 2 | 2,934 | 10.63 | 45 | 1.47 | **9.16 s (−86%)** |
| 3 | 3,170 | 11.07 | 45 | 1.47 | **9.60 s (−87%)** |

**これがこの文書の一番大きい数字である: 約 9.2 秒、TTFT 10.6 → 1.5 秒。**

### 1-6. 取り分の上限と、効かないもの

- **上限**: `取り分 ≤ prefill(N) − 1.30 秒`。**どれだけ当たっても TTFT は
  1.3〜1.5 秒より下には行かない** — その 1.3 秒は「1 チャンクぶんのエキスパートを
  40 層読む」費用で、キャッシュはそれを 1 バイトも減らさない (27 §5)
- **decode は 1 ミリ秒も速くならない。**キャッシュが触るのは prefill だけである
- **文脈が長いほど decode は遅い** (t4 15.29 tok/s vs t1 22.33 tok/s、27 §4)。
  これもキャッシュでは動かない。**TTFT だけが動く**

---

## 2. 1 エントリの費用 (バイト)

### 2-1. full attention は 40 層のうち 10 層 (コードで確認)

- `01-MODEL.md` §1: `layer_types` は `[linear, linear, linear, full] × 10`、
  **full attention は層 3, 7, 11, …, 39 の 10 層**
- `ArchInfo.swift:225` — `fullAttentionLayerMask` は `layer_types` から作られる
- `ArchInfo.swift:209-212` — この族は **`numFullKVHeads = num_key_value_heads = 2`、
  `fullHeadDim = head_dim = 256`**(「この族には第 2 の global 注意の形が無いので、
  full 用のフィールドは素のものを写す」)
- `KVCacheManager.swift:96` — `fullStride = numFullKVHeads × fullHeadDim × 2 B`
- `KVCacheManager.swift:150-160` — 再帰層には **1 バイトも確保しない**。
  full 層は `capacity = maxContext` の**線形**(この族に SWA は無く `slidingWindow = 0`)

**1 トークンあたりの KV (導出):**

```
fullStride = 2 × 256 × 2 B = 1,024 B
K と V の 2 本 × 10 層 = 20 × 1,024 B = 20,480 B/token = 20.0 KiB/token
```

### 2-2. GDN 状態は 61.41 MiB、文脈長に依らない

`RecurrentStateManager.swift:82-92` (**コードの実物**):

```
S    : Hv 32 × Dv 128 × Dk 128 × 4 B (FP32) = 2,097,152 B / 層
conv : (K−1) 3 × C 8,192 × 2 B (FP16)       =    49,152 B / 層     ← conv tail
                                    小計       2,146,304 B / 層
× 30 層 = 64,389,120 B = 61.41 MiB = 0.0644 GB
```

> **文書との差 (要注意)。**`01-MODEL.md` §3 の末尾は内訳を
> `conv 状態 8192×3 **fp32** = 98,304 B` として **62.8 MiB** を出している。
> **コードは FP16 で確保している** (`RecurrentStateManager.swift:88`
> `MemoryLayout<Float16>.size`、`ExpertCacheBudget.swift:141` も同じ式)。
> [20 §1-1](20-PHASE3-DECODE.md) の表は
> **FP16 で 48 KiB / 層・30 層で 1.41 MiB** と正しく書いているのに、
> 地の文はそれを「62.8 MiB 固定の実体」と呼んでいる (60 + 1.41 = **61.41**)。
> 実際に確保されるのは **61.41 MiB**、差は 1.41 MiB (2.2%)。
> どちらが意図かは**未確認**。本節は**コードの値**を使う。

### 2-3. 1 エントリ = KV × position + 61.41 MiB

| position | KV (20.0 KiB/tok) | GDN 状態 (固定) | **1 エントリ計** |
| ---: | ---: | ---: | ---: |
| 512 | 10.00 MiB | 61.41 MiB | **71.4 MiB = 0.0749 GB** |
| 2,048 | 40.00 MiB | 61.41 MiB | **101.4 MiB = 0.1063 GB** |
| 8,192 (運用の ctx) | 160.00 MiB | 61.41 MiB | **221.4 MiB = 0.2322 GB** |

すべて**導出**(§2-1 / §2-2 の式から)。

### 2-4. 予算に入るか — `ExpertCacheBudget` と突き合わせる

`ExpertCacheBudget.swift` の勘定 (`totalBytes`、68-71 行) は
`resident + vision + expertCache + kv + recurrent + prefillScratch`。
**mmap の腕では `expertCacheBytes = 0`** で、エキスパートの 18.12 GB は
`expertResidencyRequestBytes` (**合計の外**、クラス註 30-50 行) に載る。
`MmapExpertMapping.isEnabled` は既定 on (`MmapExpertMapping.swift:40`)。

運用点 (ctx 8192 / 32 スロット / chunk 2048 / mmap) の勘定 (**導出**):

| 欄 | 値 | 出所 |
| --- | ---: | --- |
| `residentBytes` (`model_weights.bin`) | ≈ 2.34 GB | [13 §2](13-PHASE1-REPACK.md) 実測 2,340,141,312 B |
| `visionResidentBytes` | 0 | vision 333 本は install から外している (13 §2) |
| `expertCacheBytes` | **0** | mmap の腕 |
| `kvCacheBytes` (8,192 × 20 KiB) | 0.168 GB | §2-1 |
| `recurrentStateBytes` | 0.064 GB | §2-2 |
| `prefillScratchBytes` (chunk 2048) | ≈ 0.27 GB | `ExpertCacheBudget.swift:59-61` の註「2048 で約 270 MB」 |
| **`totalBytes`** | **≈ 2.84 GB** | |
| (合計外) 常駐要求 32 × 40 × 1,769,472 | 2.27 GB | |
| `recommendedWorkingSetBytes` | **12.88 GB** | 実測、[docs/mtp/19](../mtp/19-M4.7-RESULTS.md) (同じ M3 Pro 18GB) |

**判定: 入る。**余裕は約 10 GB あり、position 8,192 の 1 エントリ 0.232 GB は
その **2.3%** にすぎない。10 エントリでも 2.3 GB で入る計算になる
(ただし 10 エントリは別の理由で無意味 — §4-4)。

**しかしもっと重要なのは、1 エントリなら 0 バイトで済むことである** (§3-2)。
[26 §4-3](26-PHASE8-SERVER.md) が
「`ExpertCacheBudget` に足す勘定は無かった」と書いたのは正しく、
**その場保持型の prompt cache でもそのまま正しい。**
足し算が要るのは「切り離した写真を N 本持つ」形に行ったときだけである。

---

## 3. 本ランタイムで足りていないもの

32 §2 は「器は既にある。足りないのは (a) capture/restore、(b) `cache_n` の結線」と
書いた。コードを読むと、**(a) は思ったより小さく、代わりに 32 §2 が挙げていない
穴が 3 つある。**

### 3-1. (a) capture/restore — **コピーは要らない**

`ServerPromptCache` は**写真を持たない**。Gemma 側の実装を読むと、キャッシュが
持っているのは**トークン列だけ**で (`ServerPromptCacheEntry.tokenIDs`、
`ServerPromptCache.swift:36-46`)、KV の中身はランナーのバッファに**居座ったまま**
である。ヒットしたら `runner.reset()` を呼ばない、それだけ。

Qwen 側もその形がそのまま使える。理由が 3 つコードに在る:

| | どこ | 何が言えるか |
| --- | --- | --- |
| prefill は position 0 を仮定していない | `QwenPrefill.swift:329` `guard kv.position + tokens.count <= maxContext`、`:372` `let start = kv.position`、`:589` `startPosition: UInt32(start)` | **任意の position から続きを prefill できる。**チャンクをまたぐのと同じ機構で、既に毎回使われている |
| 再帰状態はチャンクをまたいで持ち越す | `QwenPrefill.swift:320-322` の註「チャンク境界はモデルから見えない — 再帰状態と K/V カーソルは decode のトークン境界と同じに渡る」 | **要求境界も同じである。**継ぎ目に新しい規則は要らない |
| 状態は連続 2 本 | `RecurrentStateManager.swift:44-47` (`stateBuffer` / `convBuffer`) | 写真を取るなら memcpy 2 回。**取らないなら 0 回** |

つまり **(a) の実体は「`runner.reset()` を条件つきにする」ことである。**
今それは `QwenServerSession.swift:169` に無条件で置かれている。

**その代わりに要るもの (32 §2 が挙げていない ①):**
`QwenGreedyRun` (`QwenPrefill.swift:202-217`) は
**KV-backed トークン列も position も返さない** — 型の註が
「Gemma の `RawDecodeResult` は K/V スナップショットとキャッシュ済み接頭辞を
持つが、どちらもここでは意味が無い」と明示している。
`ServerPromptCache.publish` は `RawDecodeResult` を要求する
(`ServerPromptCache.swift:179-183`) ので、ここに橋が要る。

**境界トークンの勘定 (コードから導出、重要):**
`runGreedyCompletion` のループ (`QwenPrefill.swift:291-305`) は
`produced.append(next)` してから次の `step(token: next)` を呼ぶ。
最後に出したトークンは **step を通らない**。したがって

```
kv.position = プロンプト長 P + 生成数 N − 1
KV-backed トークン列 = promptIDs + produced.dropLast()
uncommittedBoundaryTokenIDs = [produced.last]      ← 常に 1 個
```

NVMAI の `uncommittedBoundaryTokenIDs`「常に 1 個」(32 §2) と**同じ形が
既にここに在る**。`.endOfTurn` で止まった場合、その 1 個は `<|im_end|>` である。

### 3-2. (b) `cache_n` の結線

| 直す場所 | 今 | 要るもの |
| --- | --- | --- |
| `QwenServerSession.swift:370-374` | `ServerTimings(cacheTokens: 0, promptTokens: run.promptTokens, …)` | `cacheTokens` に実数、`promptTokens` は**計算した分だけ** |
| `QwenServerSession.swift:380-383` | `OpenAIUsage(… cachedTokens: 0)` | 同上 |
| `QwenServerSession.swift:196-205` | `max_tokens: 0` の早期出口も 0 固定 | 同上 |

**壊してはいけない不変量**: SPEC RSP-1 / RSP-3 の
`cache_n + prompt_n == usage.prompt_tokens` — `Tests/TurboFieldfareServer/ServerTimingsTests.swift:113-116`
と `ServerTimingsWireTests.swift:161` が検査している。今の Qwen は
`prompt_n = 全プロンプト長 / cache_n = 0` で偶然通っているだけなので、
**`run.promptTokens` を「計算した分」に変えると同時に `cacheTokens` を入れないと落ちる。**

### 3-3. 32 §2 が挙げていない穴 — 3 つ

**① `QwenGreedyRun` の返り値** — §3-1 のとおり。

**② 失敗経路の `reset()`。**今は毎回**先頭で** `runner.reset()` するので
(`QwenServerSession.swift:164-169`、註も「途中で落ちた前の要求も同じ」と言っている)、
**途中で throw した / キャンセルされた要求の汚れた状態が問題にならない。**
キャッシュを入れると reset が条件つきになり、この安全網が消える。Gemma 側は
`defer { if !completed { promptCache.invalidate(); runner.reset() } }`
(`ServerInference.swift:670-676`) で受けている。同じものが要る。

**危ないのは KV と GDN 状態の食い違いである。**`RecurrentStateManager` には
**カーソルが無い** — どの position の状態なのかを知っているのは
`KVCacheManager.position` **だけ**である
(`QwenForwardRunner.swift:645` `public var position: Int { kv.position }`)。
`prefill` の途中で throw すると `kv.advance(by: T)` (`QwenPrefill.swift:455`) は
呼ばれないが GDN 状態は**もう進んでいる**層がありうる。この不整合を
**検出する手段がコードに 1 つも無く、次の要求が静かに間違った続きを書く。**
(現状は先頭 reset がこれを潰している。)

**③ 文脈上限の precondition。**`runGreedyCompletion`
(`QwenPrefill.swift:267-268`) は
`promptTokens.count + maxNewTokens <= maxContext` を `precondition` で見る。
継続では `promptTokens` が**接尾だけ**になるので、この式は上限を見なくなる。
実際の壁は `KVCacheManager.advance` の
`precondition(position + count <= maxContext)` で、**`precondition` は
throw ではなく abort である** — サーバーが落ちる。位置込みの式に直すか、
上位で先に弾く必要がある。

### 3-4. Gemma 側 `ServerPromptCache` の流用範囲

| 部品 | 流用 | 註 |
| --- | --- | --- |
| `commonPrefixLength` (`:68-77`) | **そのまま** | 純関数。テキストのみの経路では media 版 (`:90`) は `[]` で素通り |
| 一致判定 `match` (`:204-249`) | **そのまま** | `maximumRewind = 0` を渡せば **NVMAI の「厳密な延長」規則になる** — §4-1 |
| CACHE-3 の `reusable -= 1` (`:225`) | そのまま | **副作用: 前回と完全に同じプロンプトは必ずミスになる** (1 トークン戻る必要があり、Qwen は 0 しか戻れない)。「同じ質問を 2 回」は当たらない。害はないが、そういう挙動だと知っておくこと |
| `publish` (`:179-196`) | **要改造** | `RawDecodeResult` に依存 (`:183`)。`(tokenIDs, kvPosition)` を取る小さな overload か protocol が要る。約 10 行 |
| publish 条件 | **そのまま** | Gemma は停止理由で絞らない (`:170-177` の註「短い接頭辞になるだけで、間違いにはならない」)。NVMAI は `endOfTurn`/`toolCalls`/`maxTokens` に絞るが、絞らない方が安全側で、こちらの規則で足りる |
| `ServerPromptCacheDomain` (`:11-19`) | **要判断** | `kvStorage` / `fp16RingEnabled` は Gemma の綴り。Qwen は ring を使わない。**共有型を触ると Gemma 側の検査に当たる**ので、`runtimeProfileHash` に family を混ぜる方が安い |
| domain 束縛の中身 | **そのまま** | `modelID` / `sourceSnapshotHash` / `runtimeProfileHash` / `maximumContext` / `templateSHA256`。Qwen の template digest は `QwenTokenizer.chatTemplateJinja` の SHA256 |
| tools の一致 | **無料で付いてくる** | ツール宣言はテンプレートが**先頭の system ターンに描く** (`chat_template.jinja` 43-63 行) ので、トークン接頭辞の一致がそのまま tools の一致になる。NVMAI が別に要求している条件がここでは構造から出る |
| thinking の一致 | **無料** | `enable_thinking` は生成プロンプト末尾の `<think>\n\n</think>\n\n` の有無としてトークンに出る |
| `cache_prompt` (CACHE-5) | **そのまま** | `ChatRequestParser.swift:126` で既に読んでいる。家族を知らない層 |

### 3-5. 実装規模の見積もり (導出、行数感)

| # | 場所 | 中身 | 行 |
| --- | --- | --- | ---: |
| 1 | `QwenForwardRunner.swift` (`:639-645` の隣) | `continuationPosition` / `maximumSafeRewind` (→ `kv.maximumSafeRewind`) / `rewind(to:)` | ~15 |
| 2 | `QwenPrefill.swift:202-217` `QwenGreedyRun` | `kvPosition` / `kvBackedTokenIDs` / `uncommittedBoundaryTokenIDs` / `cachedPromptTokens` / `computedPrefillTokens` を足す | ~12 |
| 3 | `QwenPrefill.swift:257-311` `runGreedyCompletion` | `start: RawCompletionStart` を取る、接尾だけ prefill、履歴を組む、precondition を位置込みに直す (§3-3 ③) | ~25 |
| 4 | `QwenServerSession.swift:164-169` | 無条件 `reset()` を外し、`defer` で失敗時 reset + invalidate | ~12 |
| 5 | `QwenServerSession.swift` (新規フィールド + generate 前段) | `promptCache` / `promptCacheDomain` / `match` / `resume` 分岐 | ~70 |
| 6 | `QwenServerSession.swift:1096 相当` (生成後) | `publish` | ~10 |
| 7 | `QwenServerSession.swift:196-205, 370-383` | `cache_n` / `cachedTokens` / `prompt_n` の結線 (§3-2) | ~10 |
| 8 | `ServerPromptCache.swift:179` | `publish` の overload | ~12 |
| 9 | `QwenServerSession.load` | domain の組み立て (template SHA256 / runtime digest) | ~20 |
| 10 | `Tests/TurboFieldfareServer/` | 一致判定・延長・全ミス・RSP-1 分割の検査 | ~60 |
| | | **計** | **約 200 行** |

**新規カーネル 0 本、`.metal` 0 行、Gemma の経路 0 行。**

---

## 4. 危険と、崩れる前提

### 4-1. 「延長しか許さない」— コードが既にそう言っている

`RecurrentStateManager` の冒頭註 (`:26-30`) がこの制約の一次資料である:

> **This state cannot be rewound or truncated.** Dropping a token from the
> middle is not an operation the recurrence has; the only way back to an
> earlier point is a snapshot taken at that point.

そしてその制約は**既に配線されている**:

```
KVCacheManager.swift:283-289
    public var maximumSafeRewind: Int {
        if kinds.contains(where: { $0 == .linear }) { return 0 }   ← 再帰層が 1 本でもあれば 0
        …
ServerPromptCache.swift:239
    guard entry.kvPosition - reusable <= maximumRewind else { return .miss }
```

`maximumRewind = 0` を渡すと、`match` が返せるヒットは
**`reusable == entry.kvPosition`、すなわち「前回の KV 全部が新しいプロンプトの
接頭辞である」形だけ**になる。これは 32 §2 が NVMAI から写そうとしていた
「厳密な延長」そのものである。**新しい規則を書く必要がない** — Gemma のために
書かれた 1 行の guard が、Qwen ではもっと強い意味を持つ。

**両立するか、の答えは「する」。**ただし両立の代償が §4-2〜§4-4 である。

### 4-2. 連続 2 バッファであること — 良い面と、影の無さ

`RecurrentStateManager` が 30 層を `stateBuffer` / `convBuffer` の**連続 2 本**で
持つ (`:44-47`) ことは、32 §1-2 が言うとおり写真を安くする (memcpy 2 回)。
だが **prompt cache には影が 1 本も要らない** — 生きている状態がそのまま
エントリだからである (§3-1)。

その代わり **「状態は 1 つしか無い」**という制約が生まれる:

- **同時に 2 本の会話を持てない。**エントリを 2 本持ったところで、状態バッファは
  1 本しか無いので、2 本目は**必ず切り離した写真**になり、復元に
  61.41 MiB + KV の memcpy が要る。§2-3 の表が意味を持つのはこの形だけ
- `reset()` は**本当にゼロ埋め**である (`:130-135`、`KVCacheManager.reset` の
  `MADV_DONTNEED` とは違う)。**ミスのたびに 64.4 MB の memset を払う。**
  6 GB/s なら約 11 ms (**導出**、memset の帯域は**未確認**)。TTFT 1.3 秒に対して
  1% 未満なので実害は無いが、「ミスはタダ」ではない
- **2 クライアントが交互に投げると、1 エントリのキャッシュは 100% ミスになる。**
  取り分は 0、費用は毎回の memset と一致判定の walk だけ。**害は小さいが、
  効きもゼロ**である。効くのは 1 本の会話が続いている間だけ

### 4-3. **一番大きい危険** — 再レンダの継ぎ目 (ツールループが最もミスしやすい)

`ServerPromptCache.match` は**新しく描き直したプロンプト**と**前回の KV の
トークン列**を突き合わせる。前回の assistant ターンは、
**モデルが書いたバイトではなく、テンプレートが構造体から描き直したバイト**である。
`scratch/ornith-oq4e-g64.gturbo/tokenizer/chat_template.jinja` を読むと、
そこに 3 つのずれの種がある:

| 種 | テンプレートの行 | 何が起きるか |
| --- | --- | --- |
| **`trim`** | `{%- set content = render_content(message.content, true)\|trim %}` / `{%- set reasoning_content = reasoning_content\|trim %}` | モデルが書いた前後の空白・改行が落ちる。BPE では**継ぎ目のトークンが変わる** |
| **thinking の往復** | `'<\|im_start\|>' + role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content` | このテンプレートは過去の思考を**捨てない**(そこは有利)。だが**クライアントが `reasoning_content` を送り返さないと**、assistant ターンは `<think>\n\n</think>\n\n` + 本文として描かれ、生成したものと別物になる |
| **ツール呼び出しの再直列化** | `{%- for args_name, args_value in tool_call.arguments\|items %}` / `args_value \| tojson` | **引数を JSON から描き直す。**キー順・数値の書式・空白がモデルの出力と一致する保証は無い |

**そして `maximumRewind = 0` なので、1 トークンでもずれたら部分再利用は無く、
**全ミス**である。**

したがって:

- **32 §2 が「まさにこの形」と呼んだエージェントのツールループが、
  実は一番ミスしやすい。**§1-4 の 1.25〜1.76 秒は「一致したら」の値であり、
  **一致率は未確認**である
- NVMAI がこれを解いている: 32 §2 の一致判定は 2 段で、直接ヒットのほかに
  **「前回の入力 + 前回の *assistant 応答* + 続き」という会話延長ヒット**を持つ。
  つまり **assistant ターンを再レンダに任せず、記録したトークンを継ぐ。**
  Gemma 側の `ServerPromptCache` には**この 2 段目が無い** (`match` は
  `commonPrefixLength` 1 本)。**これは 32 §2 の (a)(b) が挙げていない 3 つ目の欠品である**
- 逆に、**tool 応答そのものは素直な後置である**: テンプレートの `tool` 分岐
  (`{%- if loop.previtem and loop.previtem.role != "tool" %}{{- '<\|im_start\|>user' }}`)
  は直前の assistant ターンの `<|im_end|>\n` の**後ろに足すだけ**で、
  前を書き換えない。ずれるのは assistant ターン側だけである

**当たるかどうかは実測でしか決まらない。**サーバーを 1 本立てて
`cache_n` を見るのが最短で、それは実装後の話である。

### 4-4. 分岐した会話が全ミスであることの実務上の影響

「巻き戻さない」の代償は、**接頭辞が 1 トークンでも違えば取り分がゼロ**という
一様な崖である。実務でよく踏むもの:

| 場面 | 結果 | どれくらい踏むか |
| --- | --- | --- |
| 同じプロンプトの再生成 / リトライ | **全ミス** (CACHE-3 の `-1` が効くので、完全一致でも当たらない — §3-4) | よくある |
| 過去の発言を編集して投げ直す | 全ミス | よくある |
| **文脈を切り詰めるクライアント** (8,192 に収めるため古いターンを落とす) | **以後ずっと全ミス** — 接頭辞が token 0 から変わる | **非常によくある。取り分が一番大きい長文の場面 (§1-5) がまさにこれ** |
| **system プロンプトに時刻や乱数 ID が入る** | **毎ターン全ミス** | エージェント枠組みでよくある |
| 並列にツール分岐を探索する | 全ミス (状態が 1 本しか無い、§4-2) | エージェントで踏む |
| 2 クライアント同時 | 全ミス (§4-2) | サーバーとして普通 |

**費用の側は小さい** — 一致判定はトークン列の walk 1 本、外れたら
`reset()` の memset 11 ms (§4-2)。**だから「入れて損はない」とは言える。
ただし「多ターンが速くなる」と約束はできない** — 上の表のどれか 1 つを
踏んでいるクライアントでは `cache_n` は常に 0 のままである。

### 4-5. その他の崩れる前提

- **床の 1.30 秒は外挿である。**48 トークンより短い prefill を測っていない
  (**未確認**)。継続の接尾は 29〜45 トークンなので、**取り分の見積もりは
  ちょうど測っていない領域に着地している**。実測が要る一番の点はここ
- **§1 のトークン数の仮定は測っていない。**継ぎ目 14 トークン / ユーザ発話
  30 トークンは**仮定**である。ツールループの +67 トークンだけが実測 (26 §5)
- **A/B は interleaved でないと嘘になる。**26 §5 に「同じ 1 ターンを 2 回投げたら
  10.85 → 18.34 tok/s」という n=1 の観測があり、正体はエキスパートのページが
  温まったことである。**prompt cache の A/B を逐次でやると、この温めが
  そのままキャッシュの手柄に見える。**32 §5-1 の禁止がここに直接効く
- **モデルの精度は ±6%** (§1-1 の検算表)。取り分 9.2 秒は 8.6〜9.8 秒の幅で読む
- **`conv` の dtype 差** (§2-2、61.41 vs 62.8 MiB) は費用の見積もりを
  2.2% しか動かさないが、**文書とコードが食い違っている**こと自体は残る

---

## 5. この文書が動かせるもの (判断はユーザー)

- [26 §4-3](26-PHASE8-SERVER.md)「prompt cache は無い」の理由書きは、
  **「巻き戻せないから持てない」ではなく「巻き戻せないから延長しか当たらない」**が
  正確である。`maximumSafeRewind = 0` は禁止ではなく**規則**として働く (§4-1)
- 26 §4-3「`ExpertCacheBudget` に足す勘定は無かった」は、**その場保持型でも
  依然として正しい** (§2-4)
- `01-MODEL.md` / [20 §1-1](20-PHASE3-DECODE.md) の
  **62.8 MiB は、コードでは 61.41 MiB** である (§2-2)。どちらが意図かは未確認
- 32 §2 の「足りないのは (a)(b)」には、**(c) 会話延長ヒットの 2 段目**を足す必要がある
  (§4-3)。これが無いとツールループは当たらない可能性がある
