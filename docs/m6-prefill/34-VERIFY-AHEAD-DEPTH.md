# 34. `fetch` の中の GPU の空きは層ファイルの SHA-256 待ちだった — 検証の先回りを深さ 2 → 8 にして、台帳は 3.142 → 2.885 s (実測、2026-09-24)

[33 §3](33-ROUTER-GEMM-SHARED-LATE.md) の続き。層ごとの直列の鎖 front → `fetch` → MoE のうち、`fetch` に手を入れるつもりで始めた。
`fetch` の中身 (mmap の residency set の commit、`F_RDADVISE`) を外しても GPU の空きは減らなかった (§1.1)。GPU の空きを host の段ごとに割り付ける計器を足すと、空きの大半は、層の最初の fetch が層ファイルの SHA-256 検証を待つ区間にあった (§1.2)。
検証の先回りを深くして既定に入れた ([04 原則 2](04-GATES.md))。戻すのは `TF_PREFILL_VERIFY_AHEAD=2`。

証拠 (M6、この順に取った。どれも `bench/m6/results/2026-09-24-Mac18-5-<ラベル>/`):
`f1-host-{default,resasync,resoff,advoff}-{a,b}` (4 構成を 2 巡交互に、各 n = 3)、
`f2-idle-by-host`、`f2-idle-trusted` (失敗、§2)、`f3-idle-open`、
`f4-{default,ahead4,ahead8,ahead30}-{a,b}` (2 巡交互に、各 n = 3)、
`f5-{ahead6,ahead8,ahead12}-{a,b}`、`f5-host-ahead8`、`f5-greedy-default`・`f5-greedy-ahead8`、`ledger-verify-ahead8` (n = 6)。

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 5769e1f / minos 15.0
(候補の run は f0285fa (f1)・973a9b1 (f2)・e53f5c8 (f3〜f5)。973a9b1 と e53f5c8 は host profile に行を足しただけで、既定の経路は f0285fa と同じ)
```

コマンドは台帳の固定形 (`prefill_bench.sh`: `--expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 --max-context 4096 --verification full-sha256`)。
全 run exit 0 (`f2-idle-trusted` の 3 本を除く)、Swapouts 0。MBP の `Scripts/test.sh` は 5769e1f のソースで 1629 本通過。

**台帳** ([06](06-LEDGER.md)): 既定を変えた。`ledger-verify-ahead8` は **2.885 s (2.840〜4.045)**。前の行 (6f9ae16) の 3.142 s より **0.257 s 短い** (導出)。候補の範囲 (2.872〜4.008 s) に入る (B3)。

## 0. 結論

1. **host の `fetch` の秒は、GPU を止めている秒ではなかった。**residency の commit を別スレッドに外すか set を使わないと、`fetch` は 0.17 s 縮むが、同じ量が `wait.tile` に移り、GPU の空き (約 0.5 s) と壁時計は変わらない (§1.1)。
2. **GPU の空きの大半は、層の最初の fetch が層ファイルを開くとき (`open`) の SHA-256 待ちだった。**暖かい run で空き 0.517 s のうち 0.430 s、冷たい 1 本目で 1.71 s のうち 1.55 s (§1.2)。台帳の (c) に、[04 原則 4](04-GATES.md) が「別に書く」とした起動時の検証が入っていた。
3. **検証の先回りの深さ 8 が最短。**同じバイナリで既定と交互に 3.159 → 2.884 s (各 n = 6)。4 は 3.067 s、6 は 2.945 s、12 は 3.000 s、30 は 4.325 s (§1.3)。greedy 64 トークンの本文は既定・深さ 8 とも `c0707c48e5`。

## 1. 数字

### 1.1 `fetch` の中身を外した (2・3 本目、a / b、秒)

| 構成 | prefill | `fetch` | `wait.tile` | GPU idle | 中央値 n = 6 (導出) |
| --- | --- | --- | --- | --- | ---: |
| 既定 | 3.138 3.119 / 3.168 3.169 | 0.748〜0.803 | 0.930〜0.951 | 0.495〜0.548 | 3.168 |
| `TF_EXPERT_MMAP_RESIDENCY_ASYNC=1` | 3.156 3.147 / 3.144 3.118 | 0.580〜0.614 | 1.091〜1.102 | 0.496〜0.534 | 3.151 |
| `TF_EXPERT_MMAP_RESIDENCY=0` | 3.165 3.169 / 3.146 3.114 | 0.570〜0.607 | 1.115〜1.135 | 0.491〜0.544 | 3.167 |
| `TF_EXPERT_MMAP_ADVISE=0` | 3.181 3.136 / 3.190 3.203 | 0.727〜0.805 | 0.957〜0.971 | 0.507〜0.586 | 3.196 |

- 既定の経路 (mmap) の `fetch` はバイトを読まない。中身は `F_RDADVISE` と residency set の編集・`commit()`・`requestResidency()` と、I/O キューへの往復 ([`MmapExpertMapping`](../../Sources/Tsugumi/Infrastructure/Streaming/MmapExpertMapping.swift))。
- `wait.front` はどの構成も 1.309〜1.324 s、GPU busy は 2.509〜2.526 s。

### 1.2 GPU の空きを host の段に割り付けた (`[prefill gpuq idle by host ...]`、run 順)

973a9b1 で、command buffer の GPU 区間を合わせた後の隙間 (約 890 個) と、host の段の区間 (同じ uptime の時計) の重なりを段ごとに足す行を足した。e53f5c8 で、層の最初の fetch の前に呼ばれる層ファイルの open (冪等。検証が済んでいなければ待つ) を `open`、最初のタイルの fetch を `fetch.first` として分けた (どちらも host profile のときだけ)。

| run (`f3-idle-open`) | prefill | GPU idle | 空きのうち `open` | `fetch.first` | `fetch` | 他の段の計 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (冷たい) | 4.455 | 1.713 | 1.549 | 0.020 | 0.105 | 0.04 |
| 2 | 3.656 | 1.017 | 0.915 | 0.025 | 0.037 | 0.04 |
| 3 | 3.143 | 0.517 | 0.430 | 0.024 | 0.024 | 0.04 |

- run 2 は layerVerify 7.351 s で、検証の揺れがそのまま壁時計に出ている。
- 深さ 8 (`f5-host-ahead8`、run 順): GPU idle 0.391 / 0.267 / 0.276 s、そのうち `open` 0.308 / 0.182 / 0.184 s。

### 1.3 検証の先回りの深さ (`TF_PREFILL_VERIFY_AHEAD`、並行の既定のまま。壁時計 s)

| 深さ | 中央値 n = 6 | 最小〜最大 | layerVerify (2・3 本目) |
| --- | ---: | --- | --- |
| 2 (既定、`f4`) | 3.159 | 3.126〜4.436 | 5.96〜6.05 |
| 4 (`f4`) | 3.067 | 3.001〜4.230 | 7.10〜8.23 |
| 8 (`f4`) | **2.884** | 2.872〜4.008 | 7.94〜8.20 |
| 30 (`f4`) | 4.325 | 4.306〜4.570 | 30.5〜33.4 |
| 6 (`f5`) | 2.945 | 2.914〜3.993 | 7.50〜7.64 |
| 8 (`f5`) | 2.903 | 2.874〜3.990 | 8.12〜8.20 |
| 12 (`f5`) | 3.000 | 2.986〜3.035 | 9.91〜10.47 |

- `f4` の 4 腕は同じバイナリで交互に取った。`f5` には既定の腕が無い。
- layerVerify は層ごとの検証の秒の和で、並行に走る分だけ深いほど大きくなる。
- 検証 1 本の読みバッファは 1 MiB。peak は 1.24〜1.28 GB で深さによらない。

## 2. 分かったこと / 外れたこと

- **33 §3 の「層ごとの直列の鎖 front → `fetch` → MoE」は、`fetch` を読み違えていた。**host の段の秒は「host がそこにいた秒」で、GPU がその間に空いていたかは言わない。GPU busy と idle の合計だけでも言えない。段ごとの割り付け (§1.2) で初めて分かった。
- **host 行の `experts=0 miss=0 io=0.000s` は計器の穴。**prompt prefill の経路は `recordExperts` を呼ばない (verify block の経路だけが `telemetry.decode` の差を渡す)。同じ量は `[expert prefill ...]` 行に出ている。直していない。
- 深さ 30 は、次に要る層のハッシュが他の 29 本とコアを分け合うので遅い。深さ 12 で既に 8 より長い。
- `--verification trusted-install` で検証を外した対照は、M6 のモデルの `verified-install.json` がディレクトリの不一致で読めず、3 本とも exit 1 (`f2-idle-trusted`)。受領書もモデルも書き換えていない。
- 正しさ: 検証の意味 (全層を開く前に SHA-256) は変わらない。greedy の本文は同じ。MBP の `Scripts/test.sh` は 5769e1f のソースで 1629 本通過。

**逸脱**: 数字は (b) でなく (c)。候補の run の commit は、host profile の行を足した commit が混ざる (§ヘッダ)。

## 3. 次に渡すもの

- 到達目標 (mlx_lm 1.823 s) に対して、台帳は 2.885 s。
- 深さ 8 でも GPU の空きのうち 0.18 s は `open`。層 1 本のハッシュは 1 コアで約 0.20 s (深さ 2 の layerVerify 5.96 s ÷ 30、導出) で、層 0 のハッシュは prefill の頭で始まるので、この分は深さでは隠れないと読める (未確認)。検証をモデルの load の時点で始めれば隠れるが、その秒は load に移るだけなので、(a) と (c) の分け方 ([04 B0](04-GATES.md)) と一緒に決める。
- 残る GPU busy は 2.50 s。
