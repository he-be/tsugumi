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
