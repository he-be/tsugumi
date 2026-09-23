# 14. G5 層ファイルの検証を先に回す — prefill 壁時計 6.98 → 4.43 s。GPU キューの稼働率は 60% → 91% (実測、2026-09-24)

[13](13-PROJ-SHARED-MATMUL2D.md) で GPU を縮めても壁時計が動かず、ホストの `fetch` が伸びた。その正体を確かめ、取り除いた。
04 の G5 (チャンク境界の io 直列) のうち、io ではなく**層ファイルの SHA-256 検証**が露出していた分。

**全腕共通**: `TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64` ([12](12-G3-MOE-MATMUL2D.md)、[13](13-PROJ-SHARED-MATMUL2D.md))。違うのは `TF_PREFILL_VERIFY_AHEAD` (何層先を検証するか) と `TF_PREFILL_VERIFY_AHEAD_PARALLEL` (検証を並行に回すか) だけ。

証拠 (すべて M6):

- 直列の先読み (commit 8dee853): ホスト側の内訳 [`g5-host-ahead0`](../../bench/m6/results/2026-09-24-Mac18-5-g5-host-ahead0/summary.tsv) / [`1`](../../bench/m6/results/2026-09-24-Mac18-5-g5-host-ahead1/summary.tsv) / [`2`](../../bench/m6/results/2026-09-24-Mac18-5-g5-host-ahead2/summary.tsv) / [`4`](../../bench/m6/results/2026-09-24-Mac18-5-g5-host-ahead4/summary.tsv) (各 n = 3)。壁時計 [`g5-ahead0-a`](../../bench/m6/results/2026-09-24-Mac18-5-g5-ahead0-a/summary.tsv) → [`g5-ahead2-a`](../../bench/m6/results/2026-09-24-Mac18-5-g5-ahead2-a/summary.tsv) → `-b` の組 (腕あたり n = 6)。greedy [`g5-greedy-ahead2`](../../bench/m6/results/2026-09-24-Mac18-5-g5-greedy-ahead2/summary.tsv)
- 並行の先読み (commit 3c45f83): ホスト側の内訳 [`g5-host-par-ahead2`](../../bench/m6/results/2026-09-24-Mac18-5-g5-host-par-ahead2/summary.tsv) / [`4`](../../bench/m6/results/2026-09-24-Mac18-5-g5-host-par-ahead4/summary.tsv) (各 n = 3)。壁時計 [`g5c-ahead0-a`](../../bench/m6/results/2026-09-24-Mac18-5-g5c-ahead0-a/summary.tsv) → [`g5c-par2-a`](../../bench/m6/results/2026-09-24-Mac18-5-g5c-par2-a/summary.tsv) → [`g5c-par4-a`](../../bench/m6/results/2026-09-24-Mac18-5-g5c-par4-a/summary.tsv) → `-b` の組 (腕あたり n = 6)。greedy [`g5c-greedy-par2`](../../bench/m6/results/2026-09-24-Mac18-5-g5c-greedy-par2/summary.tsv)

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 8dee853 / minos 15.0
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 3c45f83 / minos 15.0
```

コマンドは [12](12-G3-MOE-MATMUL2D.md) と同じ固定形 (`--verification full-sha256` のまま)。全 run exit 0。

## 0. 結論

1. **`fetch` の待ちは層ファイルの SHA-256 検証だった。**各層の先頭から 1 層先の検証を別のキューで回すだけで、`fetch` 3.00 → 1.99 s、壁時計 7.25 → 6.06 s (ホスト側の内訳、各 n = 3)。
2. **直列のままだと深さ 2 で頭打ち** (壁時計 7.252 → 5.947 s、n = 6)。検証 1 本の直列 (合計 4.5 s) が最初のチャンクの下限になる。
3. **検証を並行に回すと 6.982 → 4.432 s (1.58 倍、導出)、pp 354.9 → 559.1。**GPU キューの稼働率は 60% → 91% で、prefill はほぼ GPU 律速に戻った。深さ 2 と 4 は同じ (4.432 / 4.441 s)。greedy はどの腕も [12](12-G3-MOE-MATMUL2D.md) の mpp 腕と同一。

## 1. 数字

### 1.1 ホスト側の内訳 (`TF_PREFILL_HOST_PROFILE=1`、各 n = 3、中央値 (レンジ))

| 腕 | 壁時計 s | `fetch` s | GPU busy s | GPU idle s |
| --- | ---: | ---: | ---: | ---: |
| 先読みなし | 7.253 (7.073〜7.341) | 3.004 (2.848〜3.121) | 4.337 | 2.789 |
| 直列、深さ 1 | 6.055 (5.982〜6.139) | 1.988 (1.931〜2.127) | 4.156 | 1.781 |
| 直列、深さ 2 | 5.901 (5.872〜6.023) | 1.865 (1.820〜1.982) | 4.150 | 1.667 |
| 直列、深さ 4 | 5.918 (5.915〜5.923) | 1.877 (1.872〜1.891) | 4.145 | 1.672 |
| 並行、深さ 2 | 4.654 (4.626〜5.341) | 0.664 (0.638〜1.341) | 4.107 | 0.445 |
| 並行、深さ 4 | 4.525 (4.396〜5.017) | 0.512 (0.377〜0.977) | 4.106 | 0.304 |

### 1.2 壁時計 (腕あたり n = 6、A B A B の順、中央値 (レンジ))

| 組 | 腕 | prefill 壁時計 s | pp tok/s | prefill io s | layerVerify s |
| --- | --- | ---: | ---: | ---: | ---: |
| 直列 (8dee853) | 先読みなし | 7.252 (7.106〜7.444) | 341.7 | 0.861 | 4.566 |
| | 直列、深さ 2 | 5.947 (5.841〜6.026) | 416.7 | 0.850 | 4.597 |
| 並行 (3c45f83) | 先読みなし | 6.982 (6.927〜7.006) | 354.9 | 0.622 | 4.548 |
| | 並行、深さ 2 | 4.432 (4.397〜4.972) | 559.1 | 0.380 | 5.598 |
| | 並行、深さ 4 | 4.441 (4.358〜4.833) | 557.9 | 0.422 | 5.793 |

- **並行の腕は、組の最初の 1 本だけ遅いことがある** (4.972 s と 4.833 s。layerVerify 9.49 s と 13.43 s)。どちらも、別の腕の直後に来た 1 本目。ほかの run は 4.36〜4.58 s。
- **並行の腕の layerVerify は、層ごとの検証時間の合計**なので、検証が重なると膨らむ (4.5 → 5.6 s)。壁時計の成分ではない。
- 先読みなしの腕は 2 組で 7.252 s と 6.982 s。commit も時刻も違うので、比は組の中でだけ取る。

## 2. 分かったこと / 外れたこと

- **何を変えたか** (`Model.beginVerifyingRoutedExpertLayerAhead`): 層ファイルを開き、サイズを確かめ、SHA-256 を取る。今までと同じ検証を、`streamersQueue` ではなく専用のキューで先に済ませる。
  検証した fd はそのまま、後で来る `openLayerLocked` に渡る。ハッシュを取った fd と map する fd が同じ、という今までの性質は保っている。
  先読みが失敗したときは何も残さないので、open が自分で検証し直し、今までと同じエラー (`checksumMismatch` など) を投げる。改ざんの検査は MBP のテストで確かめた (`tamperedLayerFileVerifiedAheadStillFails`)。
- **同じキューに積むだけでは効かない**と、コードを読んだ段階で分かっていた。expert へのアクセスはどれも `streamersQueue.sync` を通るので、次の層のハッシュを同じキューに積むと、今の層のアクセスがその後ろに並ぶ。
- **並行にした分の副作用**: 検証が同時に 2〜4 本走るので、CPU コアと SSD / ページキャッシュを同時に使う。decode の経路はこの口を呼ばない。16 GB 機でスワップが出ていないかは、今回は見ていない (`run-N.pre.txt` は run の前の状態だけ)。
- **prefill io も下がった** (0.622 → 0.380 s)。検証で層ファイルを先に読むことで、expert の読みがページキャッシュに当たりやすくなった可能性がある。確かめていない。

## 3. 次に渡すもの

- **既定は変えていない。**`TF_PREFILL_VERIFY_AHEAD` (既定 0) と `TF_PREFILL_VERIFY_AHEAD_PARALLEL` (既定は直列) を足しただけ。既定にするかは判断を仰ぐ。
- **G0 からの通算 (導出、別 commit・別の日なので参考)**: G0 の 9.507 s ([10](10-G0-BASELINE.md)) → 4.432 s。
- **GPU 律速に戻った**ので、手順 3 (attention 本体) が次の的になる。並行・深さ 2 の構成で `attn` を level 2 に割ると ([`a0-profile-detail`](../../bench/m6/results/2026-09-24-Mac18-5-a0-profile-detail/summary.tsv)、n = 3、commit 67a087a)、
  `attn.swa` 1.433 s (48%)、`post` 0.415 s (14%)、`qkv` 0.392 s (13%)、`attn.full` 0.299 s (10%)、`oproj` 0.233 s (8%)、`rope` 0.191 s (6%)。level 2 は `attn` を割るぶん膨らむので割合で読む ([10 §1.3](10-G0-BASELINE.md))。
- **未確認**: 並行の腕の 1 本目の遅さの原因。16 GB 機でのスワップ。
