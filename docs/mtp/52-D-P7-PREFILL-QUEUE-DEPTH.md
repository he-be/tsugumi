# 52. D の prefill の負けはキュー深度だった — **`F_RDADVISE` 4 行で ttft が pread の 0.77 倍に**

実測: 2026-08-20、M3 Pro 18 GiB / macOS 15.7.5、`iogpu.wired_limit_mb` = 14336。
[51 §5](51-D-P6-MMAP-PROTOTYPE.md) が「帰属していない」と置いた prompt prefill の
負け (×1.52) を潰した。一次ログは `bench/mtp52/`。
クールダウンは **2 s** (ユーザー指定。51 の 10 s から変更)。

**結論から:**

- **prefill はスロット数に依存しない。**16/32/64 のどれでも要求 **1963 / ヒット 0%**
  で、mmap の prefill 帯域は **4.32 / 4.64 / 4.17 GB/s** と平ら (§1)。
  **51 §5 の負けはスロットの話ではなかった。**
- **51 §5 の ×1.52 のうち約 0.14 は測り方だった。**ABBA は mmap の run を必ず
  pread の run の直後に置く。pread は私有スロットに 3.2 GB を取り、prompt の
  6.4 GB をページキャッシュから追い出す。**ブロック ABBA で外すと ×1.38** (§2)
- **残りはキュー深度だった。**prompt prefill は**両腕とも `F_RDADVISE` を
  出していない**。pread が 7.9 GB/s 出るのはタイルのミスを `concurrentPerform` で
  **7.58 本同時**に投げていて、並列度がそのまま深度になるからである。
  mmap の腕はそれが `requestResidency()` **1 本**に潰れる (§3)
- **明示的に先読みを頼むと消えた。**製品差分 **20 行**、計器 1 個
  (`TF_EXPERT_MMAP_ADVISE=1`、既定 off)。prefill io **1.158 → 0.537 s**
  (5.56 → **11.99 GB/s**)、**ttft 1.617 → 0.960 s** (§4)
- **対照を取った — advise は `pread` には効かない** (prefill io ×1.03、
  tok/s ×0.985 でノイズ内)。**取り分は D 固有である** (§5)
- **advise を両腕に揃えた上で、D は 3 軸とも勝つ** — **tok/s ×1.331 /
  ttft ×0.77 / peak −3.20 GB**。51 §5 の「払っているのは ttft である」と
  47 §0 の「冷たい側で負けない」は**どちらも解消した** (§5)
- **生成文はバイト一致** (4 腕すべて 812 B / sha 同一)、ミス数も全 run 同一

## 1. スロット掃引 — **prefill はスロットに依存しない** (実測)

`math` / 256 tok / temp 0。16 と 64 は ABBA 2 ラウンド (片側 n=4)、
32 は 51 §2 の n=6 を流用。**この表は 51 と同じ ABBA なので §2 の汚染を含む。**

| slots | 腕 | prefill 要求 | prefill io | prefill GB/s | decode ミス | decode GB/s | tok/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | pread | 1963 | 0.762 s | 8.44 | 17946 | 12.32 | 23.19 |
| 16 | mmap | 1963 | 1.489 s | **4.32** | 17946 | 26.19 | 28.85 |
| 32 | pread | 1963 | 0.789 s | 8.16 | 9145 | 8.23 | 25.63 |
| 32 | mmap | 1963 | 1.387 s | **4.64** | 9145 | 16.24 | 30.25 |
| 64 | pread | 1963 | 0.968 s | 6.65 | 2403 | 3.02 | 25.43 |
| 64 | mmap | 1963 | 1.545 s | **4.17** | 2403 | 5.99 | 32.26 |

- **prefill の要求は 3 点とも 1963・ヒット 0%。**バイトが動かないので
  **prefill はスロット数に依存しない** — D を入れても同じである。
  51 §7 の「48/16 スロットで振っていない」はこれで埋まった。
- 64 スロットの `pread` は peak **7.72 GB**。`iogpu.wired_limit_mb` = 14336 前提で、
  **既定 8192 では回らない。**

## 2. **ABBA が mmap を重く測っていた** (実測)

mmap の prefill io をラウンド内の位置で割ると、**pread の私有 footprint と
向きが揃う**:

| slots | pread の peak | mmap 1 本目 (pread の直後) | 2 本目 (mmap の直後) | 差 |
| ---: | ---: | --- | --- | ---: |
| 16 | 2.88 GB | 1.967 / 1.583 | 1.381 / 1.395 | +0.387 s |
| 32 | 4.53 GB | 1.488 / 1.448 / 1.439 | 1.334 / 1.296 / 1.281 | +0.152 s |
| **64** | **7.72 GB** | **1.988 / 2.033** | **0.996 / 1.102** | **+0.962 s** |

**64 スロットでは 2 倍・生値に重なり無し。**しかも 2 本目の 1.05 s は
pread の 0.97 s とほぼ並ぶ。

⇒ **40 §4-15「プローブが自分で機械を汚す」の 2 度目**である。48 で
`--mmap-probe-gate-only` として外した同じ形が、製品経路の ABBA に入っていた。
**production (`serve.py`) には隣で回る pread の腕は居ない。**

### 2a. 汚染を外した測定 (ブロック ABBA、片側 n=10)

`mmap×6 / pread×6 / pread×6 / mmap×6`。各ブロックの `pos=1` は前ブロックの
汚染を受けるので**外す**。32 スロット / `math` / 256 tok / temp 0。

| | pos=1 | 定常 |
| --- | ---: | ---: |
| mmap | 1.347 / 1.480 s | 1.157 s |
| pread | 0.724 s | 0.815 s |

⇒ ABBA は mmap を約 **+28%** 重く、pread を約 **−11%** 軽く測っていた。
**ttft 比は ×1.52 → ×1.38。**

**ドリフト (40 §4-2 の 5% 規則)**: **`pread` だけが連続実行で劣化する** —
prefill io head 0.746 → tail 0.856 (**+14.7%、規則を超える**)。mmap は +1.3%。
run ごとに 3.3 GB の rss を積むためで、**pread の絶対値は上振れしうる。**

## 3. **負けの正体 — prompt prefill には先読みが無い** (コード)

| | |
| --- | --- |
| `RealForwardRunner.swift:1440` | `blockPipeline = ... && speculativeBlock` ⇒ **prompt prefill では `tileLookahead` が常に false**。タイルは `routedTileScheduler` の 1 タイル先行だけ |
| advise の呼び | **`:3127`/`:3129` の decode 経路だけ。**prefill/block 経路 (1380–2686) に 1 つも無い ⇒ **prompt prefill は両腕とも `F_RDADVISE` を出していない** |
| pread の腕 | `PreadExpertStreamer.swift:404` — `DispatchQueue.concurrentPerform(iterations: plan.misses.count)` = **7.58 本同時**。並列度がそのままキュー深度になる |
| mmap の腕 | `MmapExpertMapping.swift:151-161` — `commit()` + `requestResidency()` **1 本**、しかも `lock` の中。**深度 1** |

**⇒ 51 §0 の「`F_RDADVISE` は既存経路がそのまま出す。両腕で同一なので比較の外に
ある」は decode の話である。**prefill では両腕とも 0 で、「同一」ではあるが
**pread だけが並列度で代替していた。**

これが 51 §5 が「set 更新の直列化」を落として空欄のままにした帰属である。
**ロックではなく `requestResidency()` の外、深度そのものだった。**

## 4. **実装して測った — 20 行で消えた** (実測、片側 n=10)

計器 **`TF_EXPERT_MMAP_ADVISE=1`** (既定 off)。`executeExpertCachePlan` の先頭で
既存の `adviseExpertCachePlanMisses(plan)` を呼ぶだけ。**両腕に出す** (§5 の対照のため)。

32 スロット / `math` / 256 tok / temp 0 / 定常 run のみ:

| 腕 | n | prefill io | **prefill GB/s** | **ttft** | decode io | decode GB/s | **tok/s** | peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pread | 20 | 0.876 s | 7.35 | 1.236 s | 3.500 s | 8.57 | 26.03 | 4.50 GB |
| pread + advise | 10 | 0.901 s | 7.15 | 1.253 s | 3.675 s | 8.16 | 25.63 | 4.50 GB |
| mmap | 20 | 1.158 s | 5.56 | 1.617 s | 1.534 s | 19.56 | 31.65 | **1.30 GB** |
| **mmap + advise** | 10 | **0.537 s** | **11.99** | **0.960 s** | **1.095 s** | **27.41** | **34.13** | **1.30 GB** |

**4 腕とも生値に重なりが無い**:

| 腕 | prefill io | tok/s |
| --- | --- | --- |
| pread | 0.746 .. 0.935 | 25.53 .. 26.71 |
| pread + advise | 0.823 .. 0.944 | 25.34 .. 26.11 |
| mmap | 1.117 .. 1.274 | 31.36 .. 31.94 |
| **mmap + advise** | **0.521 .. 0.615** | **33.22 .. 34.29** |

- **prefill が 11.99 GB/s** — この機械の SSD 上限 **9.36 GB/s** (50 §3) の 1.28 倍。
  ⇒ **prefill のページも元から暖かかった。**深度 1 のフォールトが暖かいページすら
  遅く通していただけである。51 §4 の「運用点は推定よりずっと暖かい」が
  **prefill にも当てはまる。**
- **受け入れ**: 生成文は 4 腕すべて **812 B / sha 同一**、ミスも
  prefill 1963 / decode 9145 で全 run 同一。`F_RDADVISE` は助言なので当然だが、
  48 §9 の受け入れは形式的に満たしている。

## 5. **対照 — advise は `pread` には効かない** (実測)

40 §4-20 と同じ形を避けるため、**同じ advise を `pread` の腕にも出して**振った
(`pread+adv` vs `pread`、両腕 pread なので §2 の汚染は構造的に無い)。

| | prefill io | decode io | ttft | tok/s |
| --- | ---: | ---: | ---: | ---: |
| pread → pread + advise | ×1.03 | ×1.05 | ×1.01 | **×0.985** |
| mmap → mmap + advise | **×0.46** | **×0.71** | **×0.59** | **×1.078** |

**pread は速くならない** (むしろ僅かに遅い。ノイズ内)。**⇒ advise が埋めているのは
mmap 固有の穴 (深度 1) であって、経路共通の取りこぼしではない。**

### 5a. advise を両腕に揃えた上での D の取り分

| | prefill io | ttft | tok/s | peak |
| --- | ---: | ---: | ---: | ---: |
| `pread+adv` → `mmap+advise` | **×0.60** | **×0.77** | **×1.331** | **×0.29** (−3.20 GB) |

⇒ **3 軸とも勝つ。**51 §5 の「払っているのは ttft である」と 47 §0 の
「冷たい側で負けない」は**どちらも解消した。**
51 §2 の **+18.0%** は ABBA の汚染と先読みの欠如を両方含んだ数字で、
**揃えると +33.1% である。**

## 6. このプローブが答えていないこと (**未確認**)

- **decode も速くなった理由を分けていない** (19.56 → 27.41 GB/s)。decode 経路は
  `:3127` で既に advise を出しているはずなので、**既存の advise が
  `shouldSkipRDAdvice` に落とされている**か、**タイミングが悪い**かのどちらかである。
  **`rdadvisePolicyMode` を振っていない。**
- ~~**advise を既定にするかの判断。**~~ **決まった (§8)。**mmap の腕では常時 on、
  pread の腕では off (§5 で効かないため)。
- **51 §4a の内訳は依然として分けていない** (「コピーが消えた」と
  「ページキャッシュが 3.2 GB 広くなった」)。§4 で prefill も暖かいと分かったので、
  **anonymous バラストの腕の重みは上がった。**
- **`iogpu.wired_limit_mb` を既定 (8192) に戻して回していない** (51 §7 から持ち越し)。
- **サーバー経路 (`Scripts/demo/serve.py`) では測っていない。**既定としては通した
  (§8) が、**数字は CLI のものだけ**である。
- **MTP (bs=4) で advise を振っていない。**
- **`story` で advise を振っていない。**§4 は `math` だけである。
- **temp 0 である** (40 §4-4)。

## 7. 計器と一次資料

| | |
| --- | --- |
| 計器 | **`TF_EXPERT_MMAP_ADVISE=1`** (`MmapExpertMapping.swift`。§4/§5 を測った時点では `adviseMisses`・既定 off・両腕に出る。**§8 で `adviseOverride` になり、指定が無ければ腕に従う**) |
| 実装 | `PreadExpertStreamer.swift` の `executeExpertCachePlan` 先頭 4 行 + `MmapExpertMapping.swift` 16 行 |
| ドライバ | `bench/mtp52_slot_sweep.sh` (§1)、`bench/mtp52_single_arm.sh` (§2a)、`bench/mtp52_advise.sh` (§4)、`bench/mtp52_pread_advise.sh` (§5) |
| 一次資料 | `bench/mtp52/` — `mmap_ab_256_math_slots{16,64}.log`、`single_arm_256_math_slots32.log`、`advise_256_math_slots32.log`、`pread_advise_256_math_slots32.log` と各 `*_summary.log` |

## 8. **採用 — CLI とサーバーの既定にした** (2026-08-20、ユーザー判断)

§5a の 3 軸 (tok/s ×1.331 / ttft ×0.77 / peak −3.20 GB) を受けて、**D と advise を
既定にした。**§6 が「決めていない」と置いた 1 行はこれで閉じる。

| | |
| --- | --- |
| 既定 | **mmap + 層ごとの residency set**。`F_RDADVISE` は **mmap の腕のときだけ**出す |
| なぜ腕に紐づけるか | §5 の対照 — advise は `pread` を速くしない (prefill io ×1.03 / tok/s ×0.985)。**効かない腕で syscall を出す理由が無い** |
| 決めるのは 1 か所 | `Model.swift` の `PreadExpertStreamer(… useMmap: MmapExpertMapping.isEnabled)`。CLI・サーバー・Mac アプリ・KernelCheck が同じ既定を共有する |
| 外し方 | **`TF_EXPERT_MMAP=0`** で 51 までの私有スロット + `pread`。`TF_EXPERT_MMAP_ADVISE=1/0` は advise だけを腕と無関係に固定する |
| bench への影響 | 腕は動かない (`bench/mtp5*` は `TF_EXPERT_MMAP` を 0/1 で明示的に渡す) が、**advise は既定が変わった**ので、advise 以前のログ (51、§1、§2a) を再現するドライバには `--advise off` / `TF_EXPERT_MMAP_ADVISE=0` を書き足した (`mtp51_mmap_ab.py` の `--advise auto\|on\|off`、`mtp52_slot_sweep.sh`、`mtp52_single_arm.sh`)。**`--advise auto` は今の製品の形を測る** |
| テストへの影響 | 無い。`PreadExpertStreamer` の `useMmap:` は**既定 `false`** で、直に作るテストは `pread` のまま (17 テスト green) |
| 目印 | CLI footer の `[expert mmap layers=30 …]`、サーバー ready 行の `expert_io=mmap`、デモの見出しのピル |

**デモ (`Scripts/demo/serve.py`)**: 既定でこの経路に乗る。`--expert-io pread` で
51 までの腕に切り替えられる (同じ画像を 2 度出して ttft を体感するための口)。
**ここで数字は取っていない** — §6 の「サーバー経路では測っていない」は開いたままである。
