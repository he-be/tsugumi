# 13. 射影と shared MLP を `matmul2d` 化 — GPU 合計は 5.59 → 4.73 s に縮んだが、prefill 壁時計は 7.30 → 7.23 s でほぼ動かない (実測、2026-09-24)

[17 §3](17-NAX-NOT-USED.md) の手順 2。04 の G4 のうち射影の部分と、shared MLP。どちらの腕も routed MoE は [12](12-G3-MOE-MATMUL2D.md) の mpp (`TF_PREFILL_MOE=mpp`) にそろえ、違いは `TF_PREFILL_QMM=simdgroup` と `mpp64` だけ。

証拠 (すべて M6、commit 29948fd、同じバイナリ):

- 壁時計: [`s2-simdgroup-a`](../../bench/m6/results/2026-09-24-Mac18-5-s2-simdgroup-a/summary.tsv) → [`s2-mpp64-a`](../../bench/m6/results/2026-09-24-Mac18-5-s2-mpp64-a/summary.tsv) → [`s2-simdgroup-b`](../../bench/m6/results/2026-09-24-Mac18-5-s2-simdgroup-b/summary.tsv) → [`s2-mpp64-b`](../../bench/m6/results/2026-09-24-Mac18-5-s2-mpp64-b/summary.tsv) (各 3 本、腕あたり n = 6)
- GPU 内訳: [`s2-profile-simdgroup`](../../bench/m6/results/2026-09-24-Mac18-5-s2-profile-simdgroup/summary.tsv)、[`s2-profile-mpp64`](../../bench/m6/results/2026-09-24-Mac18-5-s2-profile-mpp64/summary.tsv) (`TF_PREFILL_GPU_PROFILE=1`、各 n = 3)
- ホスト側の内訳: [`s2-host-simdgroup`](../../bench/m6/results/2026-09-24-Mac18-5-s2-host-simdgroup/summary.tsv)、[`s2-host-mpp64`](../../bench/m6/results/2026-09-24-Mac18-5-s2-host-mpp64/summary.tsv) (`TF_PREFILL_HOST_PROFILE=1`、各 n = 3)
- greedy: [`s2-greedy-mpp64`](../../bench/m6/results/2026-09-24-Mac18-5-s2-greedy-mpp64/summary.tsv) (n = 2)
- CPU 参照との一致: [`s2-kernel-check`](../../bench/m6/results/2026-09-24-Mac18-5-s2-kernel-check/output.txt)
- 失敗した腕: [`s2-trusted-simdgroup`](../../bench/m6/results/2026-09-24-Mac18-5-s2-trusted-simdgroup/run-1.log)、[`s2-trusted-mpp64`](../../bench/m6/results/2026-09-24-Mac18-5-s2-trusted-mpp64/run-1.log) (§2)

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 29948fd / minos 15.0
```

コマンドは [12](12-G3-MOE-MATMUL2D.md) と同じ (`prefill_bench.sh` の固定形)。起動は `TF_PREFILL_MOE=mpp TF_PREFILL_QMM=<simdgroup|mpp64> [TF_PREFILL_GPU_PROFILE=1 | TF_PREFILL_HOST_PROFILE=1] bench/m6/cycle.sh prefill_bench.sh <label> <n>`。
mpp64 腕のログには `[prefill shared qmm kernel=mpp-tile64]` と `[prefill gpu proj kv:affine-tile64-f16=120 o:…=60 q:…=60]` が出ている。trusted 腕を除き全 run exit 0。

## 0. 結論

1. **GPU は縮んだ。**`attn` 3.535 → 2.805 s、`shared` 0.865 → 0.569 s、GPU 合計 5.592 → 4.730 s。
2. **壁時計はほぼ動かない。**7.300 s (7.198〜7.397) → 7.227 s (7.115〜7.330)。レンジが重なっている。
3. **縮んだ分はホストの `fetch` に移った。**`wait.front` −0.74 s と `wait.shared` −0.33 s に対して、`fetch` +0.95 s と `wait.tile` +0.19 s。GPU キューの空き (idle) は 1.95 → 2.91 s に増えた。
   `fetch` が何を待っているかは測っていない。コードから読める候補は層ファイルの SHA-256 検証 (§2)。

## 1. 数字

中央値 (レンジ)。壁時計・io・layerVerify は腕あたり n = 6、ほかは n = 3。

| | simdgroup | mpp64 |
| --- | ---: | ---: |
| prefill 壁時計 s | 7.300 (7.198〜7.397) | 7.227 (7.115〜7.330) |
| pp tok/s | 339.4 (335.0〜344.3) | 342.9 (338.1〜348.3) |
| prefill io s | 0.879 (0.840〜0.912) | 0.870 (0.714〜0.924) |
| layerVerify s | 4.537 (4.494〜4.568) | 4.569 (4.501〜4.583) |
| GPU `attn` s | 3.535 (3.531〜3.535) | 2.805 (2.800〜2.805) |
| GPU `shared` s | 0.865 (0.865〜0.866) | 0.569 (0.569〜0.570) |
| GPU `moe` s | 1.150 (1.122〜1.151) | 1.312 (1.310〜1.316) |
| **GPU 合計 s** | **5.592 (5.567〜5.596)** | **4.730 (4.726〜4.735)** |
| host `wait.front` s | 3.647 (3.638〜3.648) | 2.912 (2.909〜2.915) |
| host `wait.shared` s | 0.460 (0.457〜0.460) | 0.135 (0.134〜0.137) |
| host `fetch` s | 2.156 (2.026〜2.218) | 3.110 (2.994〜3.123) |
| host `wait.tile` s | 0.883 (0.863〜0.927) | 1.073 (1.070〜1.078) |
| GPU キュー idle s (host profile の run 1〜3) | 1.818 / 2.011 / 1.953 | 2.913 / 2.909 / 2.803 |

- **KernelCheck**: tile64 カーネルが Q (64 × 4096 × 2816)、O (131 × 2816 × 4096)、shared gate/up (131 × 2112 × 2816)、shared down (37 × 2816 × 2112)、端数 (37 × 100 × 128) で、group 32/64 × affine/sym のすべてで CPU 参照と rel 2.5e-4〜4.2e-4 (許容 5e-3)。group 32 の検出力の検査 (tile 番号で scale を索く参照に対して外れる) も合格。
- **greedy** (`--max-new 64 --temperature 0`): mpp64 腕の 2 本は互いに同一で、[12 §1.3](12-G3-MOE-MATMUL2D.md) の mpp 腕 (routed MoE だけ mpp) の出力とも同一。

## 2. 分かったこと / 外れたこと

- **`matmul2d` の tile64 カーネルは、射影でも速い。**G1 の 64 × 32 カーネル ([16 §1.3](16-G1-MPP-GROUP32.md)、`attn` 3.53 → 3.23 s) より縮み幅が大きい (3.54 → 2.81 s)。ただし G1 は別の commit・別の日なので、比べるなら同じ commit で `TF_PREFILL_QMM=mpp` の腕を取り直す必要がある。今回は取っていない。
- **`moe` が 1.150 → 1.312 s に伸びた。**`moe` は tile stream の待ちを含む ([10 §1.2](10-G0-BASELINE.md))。前段が速くなって待ちが `moe` 側に回った可能性があるが、分けていない。
- **`fetch` の中身の候補 (コードを読んだもの、未計測)**: 各層の先頭で `beginOpeningRoutedExpertStreamer(L)` が層ファイルの SHA-256 検証を直列キューに積み (`RealForwardRunner.swift` の層ループ先頭)、その層の expert 取得は同じキューで `ensureLayerOpened` を待つ (`Model.swift`)。
  検証は層の先頭から始まるので、前半 (attention + shared) が速くなるほど、`fetch` が検証の終わりを待つ時間が伸びる。層 L の MoE の間はこのキューが空いている。
  layerVerify の合計は 4.5 s で、2 チャンクのうち最初のチャンクの 30 層に乗る。**「`fetch` の増分 = 検証待ち」はまだ確かめていない。**
- **確かめようとした腕は起動できなかった。**`--verification trusted-install` を足すと、6 本とも `trusted install receipt invalid: model directory mismatch` で exit 1 になった。M6 のモデルは MBP から rsync したもので、receipt が別のディレクトリを指している。receipt を作り直すにはモデルのディレクトリに書き込む必要があるので、やっていない。

## 3. 次に渡すもの

- **既定は変えていない。**`TF_PREFILL_QMM=mpp64` のときだけ、射影と shared が tile64 カーネルを使う。
- **04 G3 の決まり** (「GPU が下がって壁時計が動かないなら、露出したと記録して G5 を先にやる」) に当たる。次は attention 本体 (手順 3) ではなく、`fetch` の待ちの正体を確かめること。
  測り方の案は 2 つ。層ごとの open/verify の時刻を `TF_PREFILL_HOST_PROFILE` の行に足す。あるいは、検証を 1 層先に積んで (層 L の先頭で L+1 も積む) 壁時計が動くかを見る。
- **判断を仰ぐこと**: M6 のモデルに trusted-install の receipt を作り直してよいか (モデルのディレクトリに `verified-install.json` を書くことになる)。
