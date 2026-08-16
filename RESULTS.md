# RESULTS — PLAN Phase 0 / 1 / 2

実施: 2026-08-16 / M3 Pro 18GB / macOS 15.7.5 / `macos15-support` ブランチ
プロトコル: PLAN §6 (`.build/release` 直叩き、`--temperature 0 --seed 1`、
インターリーブ + 20s クールダウン、中央値)

表記は PLAN と同じ: **実測** / **導出** / **未確認**

---

## 0. 結論

haiku.json (510 tok prompt) / 384 tok 生成 / インターリーブ 3 回の中央値 (**実測**)。
変更前 = 16 スロット + `full-sha256` (元の既定)、変更後 = 64 スロット + `trusted-install`。

| | 変更前 | 変更後 | |
| --- | ---: | ---: | --- |
| decode | 22.96 tok/s | **31.02 tok/s** | **+35%** |
| TTFT | 12.58 s | **9.63 s** | **−2.95 s** |
| peak (phys footprint) | 2.23 GB | **7.11 GB** | 予算 12 GB 内 |

**PLAN の目標「12 GB で 30 tok/s 台」は 7.2 GB で達成。**
llama.cpp が 15 GB で 30-40 tok/s なので、**半分以下のメモリで同圏**に入った。

PLAN の想定と違った点が 2 つある (§4 と §5 に詳述)。

- ルータの偏りは想定よりはるかに強く、**96 スロットで decode ヒット率 99.2%**
  (PLAN §3 の悲観シナリオは 75%)。
- しかし **I/O は shared MLP の GPU work と重なっている**ので、I/O をゼロに
  してもその分そのまま速くはならない。**64 スロットで頭打ち**になる。

---

## 1. Phase 0 — 計装 (完了)

追加したもの:

| # | 内容 | 場所 |
| --- | --- | --- |
| 1 | footer に load / layerVerify / prefill秒 / TTFT / peak RSS | `CLI/Run.swift` |
| 2 | 層ファイル SHA-256 の秒数を個別計上 | `Model.swift:openLayerLocked` |
| 3 | expert cache の hit/miss カウンタ (prefill / decode 別) | `ExpertTelemetry.swift` |
| 4 | `--dump-expert-trace <path>` | `ExpertTelemetry.swift` |
| 5 | prefill 側の expert I/O 計装 | `ModelExpertIO.swift` |

footer は既存の 1 行目をそのまま残し、下に 3 行足す形にした
(`bench.sh` の既存パーサを壊さないため)。

```
[stop=maxTokens prefill=510tok new=384tok decode=12.42s tok/s=30.923]
[load=0.680s layerVerify=0.000s/30layers prefill=10.015s ttft=10.015s peak=7.20GB rss=3.91GB]
[expert prefill hit=42.5% 3341/7854 io=3.700s | decode hit=99.2% 91167/91920 io=0.813s]
[decode/tok io=2.15ms cb1=0.66ms cb2=0.23ms head=3.24ms]
```

### 1-1. PLAN 1-6 の仮説は正しかった (**実測**)

```
load=0.686s   layerVerify=5.422s/30layers
```

`Model.load` 本体は 0.69 s しかかからない。status 2-3 の「ロード単体 8.56 s」の
正体は**最初の forward pass の中で走る 30 層ぶんの SHA-256** だった。
12.9 GB / 5.42 s = 2.38 GB/s で、openssl の実測 2.7 GB/s と整合する。

### 1-2. ヒット率カーブは 1 回のトレースから引ける (**実測**)

`bench/expert_sim.py` が `PreadExpertStreamer.makeExpertCachePlan` の
追い出し規則をそのまま再現する。検証: slots=16 のトレースを replay すると
decode ヒット 65.0% / ミス 1260 → 実機の footer は 65.0% / 1259。**一致。**

```
$ ./bench/expert_sim.py trace.tsv --skew
```

**Phase 0 の出口条件を満たした。** 以下は m.json (518 tok prompt / 128 tok 生成)
のトレース 1 本から引いた机上値 (**導出**)。

| slots | 常駐率 | decode hit (lfu) | decode hit (lru) | prefill hit (lfu) |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 6.2% | 51.3% | 51.3% | 0.0% |
| 16 | 12.5% | 71.0% | 69.7% | 0.2% |
| 24 | 18.8% | 80.5% | 79.5% | 1.5% |
| 32 | 25.0% | 86.4% | 85.2% | 3.4% |
| 48 | 37.5% | 93.0% | 92.3% | 16.2% |
| 64 | 50.0% | 96.5% | 96.1% | 50.0% |
| 80 | 62.5% | 98.5% | 98.3% | 64.6% |
| **96** | 75.0% | **99.2%** | 99.2% | 69.8% |
| 112 | 87.5% | 99.3% | 99.3% | 69.9% |
| 128 | 100% | 99.3% | 99.3% | 69.9% |

読み取れること:

- **ルータの偏りは非常に強い。** 12.5% の常駐で 71% ヒット、50% の常駐で 96.5%。
  PLAN §3 が心配していた「偏りがなければ 75% 止まり」は杞憂だった。
- **128 スロット (全常駐) でも 99.3%** で 100% にならない。残りは初回参照の
  compulsory miss (30 層 × 約 6.6 個)。つまり 96 以上を積む意味は原理的にない。
- **lfu は lru より一貫して僅かに良い** (16 スロットで 71.0% vs 69.7%)、
  96 以上では同値。status 7-B は「lfu のままでよい」で決着。
- prefill は 48 スロット以下だとヒット率が実質ゼロ。タイルが全エキスパートを
  舐めるため。64 で急に 50% に跳ねる。

---

## 2. Phase 1 — 起動時間 (完了)

`--verification full-sha256|trusted-install` を CLI とサーバに露出した
(`ModelIntegrityPolicy` に共通の CLI 名を持たせ、Mac アプリの語彙と揃えた)。

**実測** (ページキャッシュを温めた状態、m.json):

| | TTFT |
| --- | ---: |
| `--verification full-sha256` (既定) | 14.27 s |
| `--verification trusted-install` | 10.29 s |
| | **−3.98 s** |

PLAN の見積り約 5 s に対して実測 4.0 s。差は、SHA-256 のパスが
12.9 GB をページキャッシュに載せる副作用で expert 読み出しを温めていた分
(prefill io が 4.80 s → 5.41 s に増えている)。

既定は `full-sha256` のまま。常用時に `trusted-install` を明示的に選ぶ。

---

## 3. Phase 2 — スロット上限の引き上げ (完了)

### 3-1. 変更内容

1. `allowedExpertCacheSlots = [8, 16, 24, 32, 48, 64, 80, 96, 112]`
   (128 は PLAN §8 の却下どおり入れていない)
2. **PLAN 1-9 のホットパス修正** (`PreadExpertStreamer`)
   - `expertResidency[expert]` (常駐スロット数) で **ミス判定を O(1)** に。
     0 なら確定ミス、正なら `expertSlotHint` を検証して使う。
     ヒントが古い / 重複で予約済みのときだけ従来どおり線形走査に落ちる。
   - `misses.isEmpty` なら追い出し候補の選抜を**丸ごとスキップ**
     (99% ヒットの領域ではこれが常態)
   - sort が要る場合も、必要な `misses.count` 個 (decode なら 8) だけを
     `cheapestEvictableSlots` で選ぶ。**全体 sort を廃止。**
   - `adviseExpertMisses` の `slotExpert.contains` も O(1) 化
3. **メモリガード** (`ExpertCacheBudget`)。
   `resident + numLayers*slots*expertStride + KV` を
   `device.recommendedMaxWorkingSetSize` と比較し、超えるなら
   `RealForwardRunner.init` で拒否する。
4. 既定スロット数を 16 → **64**

**書き換えが挙動を変えていないことの確認 (実測):** 同一条件で
`decode hit=71.0% 21629/30480` / `prefill 13/8612` が書き換え前後で
**完全に一致**。純粋な高速化。

### 3-2. メモリガードの動作 (**実測**)

```
$ TurboFieldfareCLI ... --expert-cache-slots 112
error: expert cache configuration exceeds this device's recommended Metal
working set — resident 1.35 GB + experts 11.29 GB (112 slots) + kv 0.32 GB
= 12.96 GB; device recommends at most 12.88 GB.
Lower --expert-cache-slots or --max-context.
```

PLAN §2 の表 (112 = 12.9 GB、推奨作業セット 12.88 GB に接触) と一致。

### 3-3. スロット数 vs 実性能 (**実測**)

haiku.json (510 tok prompt)、384 tok 生成、インターリーブ:

| slots | tok/s | ms/token | decode io ms/tok | decode hit | prefill 秒 | peak GB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 21.2 | 47.1 | 17.2 | 70.5% | 9.4 | 2.2 |
| 32 | 27.9 | 35.8 | 7.9 | 91.7% | 9.3 | 3.9 |
| 48 | 29.8 | 33.6 | 3.9 | 97.8% | 10.1 | 5.5 |
| **64** | **31.2** | **32.1** | 2.0 | 99.2% | 9.4 | **7.1** |
| 96 | 31.2 | 32.1 | 0.85 | 99.8% | 9.2 | 9.4 |

**64 が膝。** 48 は 64 比 −4.5%、96 は 64 と同値 (誤差内)。

### 3-4. プロンプトを振ったときの pp / tg (**実測**)

haiku 1 本では評価にならないので、PLAN 付録の 3 本を **64 スロット固定**で測った。
`./bench.sh ja` (384 tok 生成、インターリーブ 3 回中央値、`trusted-install`)。

> **この表だけ greedy (`--temperature 0`) で測ったもの。** 測定プロトコルは
> このあと §3-5 のとおり `temp 0.2 / top-k 64 / top-p 0.95` に変えた。
> tok/s の差は run 間の振れ (±1-4%) の範囲なので、表そのものは有効。

| prompt | ptok | tg tok/s | ms/token | prefill 秒 | TTFT | decode io ms/tok | decode hit | prefill hit | peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| haiku (寿司俳句) | 510 | 30.84 | 32.4 | 9.91 | 9.91 | 2.22 | 99.2% | 42.5% | 7.13 GB |
| story (デカ妹創作) | 96 | **31.31** | 31.9 | 2.00 | **2.00** | 3.91 | 98.5% | 0.0% | 7.05 GB |
| math (三次方程式) | 43 | **28.42** | 35.2 | 1.54 | **1.54** | 6.56 | 97.4% | 0.0% | 7.08 GB |

3 本とも 3 回の振れは ±0.5% 以内 (例: math = 28.42 / 28.31 / 28.54)。

**(a) tg はプロンプトに弱く依存する。短いプロンプトのほうが遅い。**
haiku 30.84 に対し math 28.42 で **−8%**。原因は decode ヒット率で、
prefill が短いとキャッシュが温まらないまま decode に入る
(510 tok の prefill = 4 チャンクは 42.5% ヒットまで温まるが、43 tok は 1 チャンクで
全ミス)。差は decode io に出ていて 2.22 → 6.56 ms/tok (**+4.3 ms**)。
ms/token の差は +2.8 ms なので、§4 のとおり I/O の一部は GPU work に隠れている。

→ **RESULTS §0 の 31.02 tok/s は「510 tok の長めのプロンプト」の値。**
短いプロンプトの常用では **28-31 tok/s** と読むのが正しい。それでも
PLAN の目標 (30 tok/s 台) の下限付近には収まっている。

**(b) 逆に TTFT は短いプロンプトで劇的に良い。** 1.5-2.0 s。
つまり `--verification trusted-install` + 64 スロットの構成では、
**対話用途 (短い入力) は TTFT 2 s / 28 tok/s、長文入力は TTFT 10 s / 31 tok/s** となる。

**(c) 短いプロンプトの prefill 秒は「スループット」ではなく固定費。**
math 43 tok で 1.54 s のうち **1.01 s が expert の初回充填 I/O** (prefill io)。
ptok/prefill 秒を pp と呼ぶと 28 tok/s になるが、これは意味のない数字。
実際は「1 チャンクぶんの固定費 約 1.5 s + 追加トークンぶん」で、
差分で見ると 43→96 tok が +0.46 s (115 tok/s)、96→510 tok が +7.9 s (52 tok/s)。
**後半で pp が落ちるのはチャンクが増えて live expert の I/O が増えるから**
(haiku の prefill io = 3.84 s)。Phase 4 (チャンク上限) が効くとしたらここ。

**(d) 生成品質の観察** (perf ではないが、384 tok 打ち切りで見えた範囲)。
`bench/logs/ja-*.N.out` に全文がある。

- **haiku**: 思考プロセスとして寿司ネタを列挙する途中で
  「あわび (3)」の**無限ループに落ちて 384 tok 使い切り、俳句に到達しない**
  (temperature 0、3 回とも同じ)。ベンチ用プロンプトとしては安定だが、
  **モデル評価用としては 384 tok では答えが出ない。**
- **math**: カルダノ / 三角関数への帰着を試す妥当な筋。ただし途中で
  `$x =  مقدار$` とアラビア語のトークンが 1 個混入した (3 回とも同じ位置)。
- **story**: 破綻なし。日本語として自然で、384 tok まで一貫している。

**評価目的なら 384 tok では足りない** (3 本とも `stop=maxTokens`)。
perf 測定と評価は分けて、評価側は `--max-new` を 1024 以上にすること。

### 3-5. サンプリングパラメータの固定と温度 (**実測**)

greedy はベンチには都合がよかった (3 回バイト一致) が、実際の生成条件ではない。
**測定プロトコルをサンプリングありに固定した** (`bench.sh`):

```
--temperature 1.0 --top-k 64 --top-p 0.95 --seed <run番号>
```

温度は最初 CLI の既定値 0.2 で固定したが、下の結果を受けて
**0.2 は廃止し、モデル推奨の 1.0 を全体の既定値にした** (§3-6)。

種を run 番号にしたので **run ごとに出力は変わる** (= 変動を許容する) が、
run 番号を指定すれば再現する。**tok/s の run 間の振れは ±1-4%** (下表)。

#### 温度 0.2 (既定) vs 1.0 (Gemma 推奨)

`./bench.sh temp` (384 tok 生成、6 条件インターリーブ、3 回中央値):

| prompt | temp | tg tok/s | 振れ (max−min) | decode io ms/tok | head ms/tok |
| --- | ---: | ---: | ---: | ---: | ---: |
| haiku | 0.2 | 29.93 | 1.0% | 2.27 | 3.18 |
| haiku | 1.0 | 29.21 | 1.3% | 3.00 | 3.18 |
| story | 0.2 | 30.36 | 0.6% | 3.70 | 3.29 |
| story | 1.0 | 30.51 | 1.6% | 3.70 | 3.28 |
| math | 0.2 | 27.97 | 4.1% | 6.41 | 3.18 |
| math | 1.0 | 28.39 | 1.4% | 6.07 | 3.24 |

**温度は性能に効かない。** 差はすべて run 間の振れの中。
`head` (LM head + サンプリング) が全条件 3.18-3.29 ms で一定なので、
**top-k 64 + top-p 0.95 のコストは計測に出ない**。
サンプリングを入れるかどうかも同様 (§3-4 の greedy 表と 3% 差だが、
`head` が動いていないので熱ドリフトと生成内容の違いによる)。

→ **性能を理由に温度を選ぶ必要はない。品質だけで決めてよい。**

#### 品質: 推奨温度でないとループする (**実測**、haiku のみ n=1)

`./bench.sh loopcheck` を 1024 tok で回した。**haiku の 3 温度まで取ったところで
中断したので、math / story は未測定。** seed=1、1 本ずつなので統計ではない。

| temp | 停止 | 生成 | 末尾の反復 | 結果 |
| ---: | --- | ---: | ---: | --- |
| 0 | maxTokens | 1024 | **「あわび (3)」×25** | 俳句に到達せず |
| 0.2 | maxTokens | 1024 | なし | 列挙が終わらず俳句に到達せず |
| **1.0** | **endOfTurn** | **828** | なし | **完答** |

温度 1.0 だけが自分で止まり、俳句を出した:

```
5: まぐろ(3) + えび(2) = 5
7: サーモン(4) + ほたて(3) = 7
5: うに(2) + あなご(3) = 5

まぐろえび / サーモンほたて / うにあなご
```

モーラ数も規則も満たしている (寿司ネタのみ、5・7・5)。

**「推奨パラメータでないとループする」は、この機体・このプロンプトでは
当たっていた。** 温度 0 は明確なループ、0.2 も 1024 tok で答えに到達しない。

未消化: math / story のループ検出は中断したので未測定 (`./bench.sh loopcheck`)。
**n=1 の観察なので「0.2 だと必ずループする」とまでは言えない。**
ただし温度は性能に効かない以上、**下げて得るものが何もない**ので、
0.2 を残す理由もない。

### 3-6. 既定値を temperature 1.0 に変更し、0.2 を廃止

§3-5 の 2 つ (温度は性能に効かない / 推奨より下げるとループする) から、
**推奨値 1.0 を全経路の既定値にして、0.2 は残さない**ことにした。

| 場所 | 変更 |
| --- | --- |
| `CLI/Args.swift` | 既定 `temperature` 0.2 → **1.0** (3 箇所 + `--help` 文言) |
| `Server/OpenAIModels.swift` | `request.temperature ?? 0.2` → **`?? 1.0`** |
| `App/MacAppSettings.swift` | 既定 0.2 → **1.0** (struct + init) |
| `App/AppModel.swift` | 既定 0.2 → **1.0** |
| `App/AppGenerationRequest.swift` | 既定 0.2 → **1.0** |
| `bench.sh` | `TEMP` 既定 0.2 → **1.0**。温度スイープ `cmd_temp` は役目を終えたので削除 |
| `README.md` / `docs/RUNTIME_CONTROLS.md` / `docs/COMMUNITY_BENCHMARKS.md` | 既定値の記述を 1.0 に。ループの注意を追記 |
| `Tests/` (4 ファイル) | 既定値を検証している `#expect` を 1.0 に |
| `scratch/mac-app-settings.json` | この機体の保存済み設定も 1.0 に (既定値の変更は新規インストールにしか効かないため) |

`--temperature 0` (greedy) は**フラグとしては残っている**。再現が要る比較には
使えるが、常用と品質評価には使わない。

**注意**: `docs/COMMUNITY_BENCHMARKS.md` の手順も 1.0 に書き換えた。温度は
tok/s に効かないので既発表の数字との比較可能性は保たれるが、上流の手順とは
乖離する (このブランチは既に既定スロット数などで乖離済み)。

**再測定はしていない。** §3-4 / §3-5 の表がそのまま有効
(温度差は run 間の振れの中、`head` も動かない)。

---

## 4. PLAN の想定と違った点 (1) — I/O は GPU work と重なっている

PLAN §3 の試算は「I/O がゼロになれば 41.8 → 29.3 ms」という**引き算モデル**
だった。これは成立しない。

**実測**: 16 → 96 スロットで decode io は 17.2 → 0.85 ms/tok (**−16.4 ms**) 減ったが、
実際の ms/token は 47.1 → 32.1 ms (**−15.0 ms**) にとどまる。
さらに 64 → 96 では io が 2.0 → 0.85 ms と 1.2 ms 減っても
ms/token は動かない (32.1 → 32.1)。**I/O の削減が 1:1 で効くのは
64 スロットの手前まで**で、そこから先は完全に隠れる。

理由はコードにある。`RealForwardRunner` は routed expert の pread を発行する
前に shared MLP の command buffer を commit している
(`RealForwardRunner.swift:1571-1595` のコメントどおり意図的な設計)。
つまり **`totalIoNanos` が測っているのは「pread の待ち時間」であって、
その裏で GPU が shared FFN を回している**。I/O を消しても、隠れていた
GPU 時間が表に出てくるだけ。

結果として **64 スロットから先は decode が GPU 律速**になる。
`decode/tok` の内訳 (64 スロット、**実測**):

```
io=2.15ms  cb1=0.66ms  cb2=0.23ms  head=3.24ms   計装済み合計 6.3ms
未計装 (attention / MoE 本体)                    約 26 ms
```

**次に効くのは I/O ではなく MoE / attention カーネル本体。**

## 5. PLAN の想定と違った点 (2) — Phase 3 の先読みの目的が変わる

PLAN Phase 3 は「96 スロットは lazy にしか埋まらないので先読みが要る」と
していた。**実測**では、510 tok の prefill を終えた時点で 64 スロットの
decode ヒット率が 99.2% に達している。**prefill が事実上の先読みになっている。**

一方で、**スロットを増やすと prefill 自体は遅くなる。**
上の before/after (SHA-256 込み 12.58 s / 抜き 9.63 s) を分解すると:

```
16 slots, SHA なし  : 約 7.2 s   (12.58 − SHA 5.4 s)
64 slots, SHA なし  :     9.63 s
                      差 +2.4 s  ← 30層×64スロット×3.36MB = 6.4 GB の初回充填
```

Phase 1 の −5.4 s が Phase 2 の +2.4 s を吸収して、正味 −2.95 s。
**2 つを組み合わせて初めて TTFT が下がる**ので、`trusted-install` は
スロット増と必ずセットで使う。

→ **Phase 3 の静的先読みは、やるなら「TTFT を下げる」目的でやることになる。**
「decode のヒット率を上げる」目的での必要性は消えた。

注意: prefill 秒はセッションをまたぐと 7.6-9.4 s と大きく振れる (ページキャッシュ
の状態に依存)。**スイープの内側でインターリーブした値どうしでしか比較しないこと。**

---

## 6. 副産物として解決した未確認項目

| 項目 | 結論 |
| --- | --- |
| status 5-2 (CLI とアプリで prefill が 33% 違う) | footer に prefill 秒が出るようになったので引き算誤差が消えた |
| status 5-3 (I/O にハッシュが含まれるか) | 含まれない。層オープン時 1 回のみ (**実測**) |
| status 5-5 (スロット上限 32 は構造的か) | ただの許可リスト。112 まで通した |
| status 5-6 (チャンク上限 128 は構造的か) | 未着手 (Phase 4) |
| status 7-B (lfu vs lru) | lfu が僅かに良い。据え置きで確定 |
| PLAN リスク表「LFU カウンタの飽和」 | `expertUseCount` は `[Int]` = 64bit。現実的なセッション長で飽和しない。**リスクなし** |
| TODO.md 9 番 (apple10 ゲート) | PLAN 0-1 で確認済み |

---

## 7. 次にやるなら

優先順に:

1. **Phase 5 (文脈長)** — 独立・低コスト。`--max-context` の既定を
   4096 → 16384/32768 へ。PLAN 1-7 のとおり KV は 32K でも 840 MiB で、
   64 スロット構成 (7.1 GB) なら 12 GB 予算に余裕で収まる。
2. **decode の未計装 26 ms/token を割る** — I/O 律速ではなくなったので、
   ここが唯一の残り。attention と MoE 本体に計装を入れる。
   PLAN にはないが、Phase 4/6 より先にやる価値がある。
3. **Phase 4 (チャンク上限 + `.q` 投影)** — prefill が 9-10 s と TTFT の
   ほぼ全部を占めるようになったので、相対的な重要度は上がった。
4. **Phase 3 は目的を「TTFT 短縮」に読み替えて再評価** (§5)。
5. **Phase 6 は不要** — 96 でも 128 でもヒット率が 99.3% で頭打ちなので、
   物理再配置 + mmap も非一様配分も買えるものがない。

---

## 8. 使い方

```bash
# 常用 (既定 64 スロット + 受領証を信頼)
TurboFieldfareCLI --model scratch/gemma4.gturbo --messages-file bench/haiku.json \
  --verification trusted-install

# スロット数のスイープ
./bench.sh slots

# PLAN 付録の日本語プロンプト 3 本 (64 スロット固定、§3-4)
./bench.sh japrompts   # 追加ぶんの math.json / story.json を生成 (初回のみ)
./bench.sh ja

# トレースを取って机上でヒット率カーブを引く
./bench.sh trace
./bench/expert_sim.py bench/trace.tsv --slots 16,32,48,64,96 --skew
```

## 9. 未消化

- `swift test` はこの環境で実行できない。CLT のみで Xcode がなく、
  swift-testing の `Testing` モジュールが無い (**私の変更以前からの状態**)。
  検証は `swift build -c release` と、上記の実機での挙動一致確認で代替した。
- prefill 側の `avoidingSlots` はトレースに出していないので、
  `expert_sim.py` の prefill 列は近似 (decode 列は厳密に一致する)。
