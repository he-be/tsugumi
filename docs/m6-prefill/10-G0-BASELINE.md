# 10. G0 基準線 — M6 は prefill 9.51 s (pp 261)、GPU 側の最大は routed MoE ではなく attention の束 (実測、2026-09-23)

証拠:
[`2026-09-23-Mac18-5-g0-baseline/`](../../bench/m6/results/2026-09-23-Mac18-5-g0-baseline/summary.tsv) (M6、G0-3、n = 6)、
[`2026-09-23-Mac15-6-g0-baseline/`](../../bench/m6/results/2026-09-23-Mac15-6-g0-baseline/summary.tsv) (MBP、G0-3、n = 6)、
[`2026-09-23-Mac18-5-g0-profile/`](../../bench/m6/results/2026-09-23-Mac18-5-g0-profile/summary.tsv) (M6、G0-4、`TF_PREFILL_GPU_PROFILE=1`、n = 3)、
[`2026-09-23-Mac18-5-g0-profile-detail/`](../../bench/m6/results/2026-09-23-Mac18-5-g0-profile-detail/summary.tsv) (M6、G0-4、`TF_PREFILL_GPU_PROFILE=2`、n = 3)。

```
測定: 2026-09-23 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 837586c / minos 15.0
測定: 2026-09-23 / Mac15,6 Apple M3 Pro 18GB / macOS 15.7.5 (24G624) / Swift 6.3.3 / SDK 26.5 / Xcode 26.6 / APPLE SSD AP1024Z / commit 837586c / minos 15.0
```

コマンド (M6 の `command.txt`。MBP はパスだけ違う):

```
/Users/mh/dev/tsugumi/.build/release/TsugumiCLI --model /Users/mh/dev/tsugumi/scratch/gemma4-qat-sym.gturbo --messages-file /Users/mh/dev/tsugumi/bench/l.json --expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 --max-context 4096 --verification full-sha256
```

M6 は `bench/m6/cycle.sh prefill_bench.sh <label> <n>` (MBP から)、MBP は `bench/m6/prefill_bench.sh g0-baseline 6`。全 run exit 0。
M6 は再起動 1 分後にビルドし、その直後に G0-3 → G0-4 (level 1) → G0-4 (level 2) の順で連続して取った。run 間の休止はない (約 11 s 間隔)。

## 0. 結論

1. **M6 の prefill 壁時計は中央値 9.51 s (9.19〜9.60)、pp 260.7。MBP は 10.58 s (10.44〜10.67)、pp 234.3。**同一 commit・同一コマンドで、M6 が 1.11 倍速い (導出)。
2. **GPU 合計 7.12 s のうち attention の束 (`attn`) が 3.53 s (50%)、routed MoE (`moe`) が 2.68 s (38%)。**
   04 の G2 は「最大の的は routed MoE」を前提に形状を選んでいるが、M6 では外れた。`attn` の中は注意本体 1.73 s、射影 (qkv + oproj) 1.38 s (§1.3)。
3. **io は GPU と重なっていない側に乗る**が、3 本の揺れで確かめただけ (§2)。壁時計 − GPU 合計は io より 0.56〜0.63 s 長く、io が増えた分だけ一緒に増える。

## 1. 数字

### 1.1 G0-3 (n = 6、中央値 (レンジ))

| | M6 | MBP | MBP / M6 (導出) |
| --- | ---: | ---: | ---: |
| prefill 壁時計 s | 9.507 (9.188〜9.597) | 10.579 (10.441〜10.670) | 1.11 |
| pp tok/s | 260.7 (258.2〜269.7) | 234.3 (232.2〜237.3) | — |
| prefill io s | 1.871 (1.641〜1.902) | 1.191 (1.169〜1.461) | 0.64 |
| 壁時計 − io s (**導出**、run ごとに引いた中央値) | 7.646 (7.547〜7.715) | 9.325 (9.209〜9.458) | 1.22 |
| layerVerify s (SHA-256、30 層) | 4.491 (4.398〜4.550) | 5.656 (5.548〜5.790) | 1.26 |
| peak GB | 1.21〜1.22 | 1.27〜1.28 | — |

expert の hit はどの run も 0% (5771 / 5775 取得すべてミス)。プロセスごとにキャッシュが空から始まるため。

### 1.2 G0-4 level 1 (M6、n = 3)

| 段 | 中央値 s (レンジ) | 割合 |
| --- | ---: | ---: |
| `attn` (embed・norm・qkv・RoPE・KV 書き込み・注意本体・oproj・router) | 3.534 (3.531〜3.535) | 50% |
| `moe` (routed expert、tile stream の待ちを含む) | 2.678 (2.618〜2.722) | 38% |
| `shared` | 0.866 (0.865〜0.867) | 12% |
| `tail` + `head` | 0.046 | 1% |
| **GPU 合計** | **7.123 (7.061〜7.168)** | |
| prefill 壁時計 | 9.593 (9.158〜9.630) | |
| 壁時計 − GPU 合計 (**導出**) | 2.462 (2.097〜2.470) | |
| prefill io | 1.835 (1.538〜1.883) | |

### 1.3 G0-4 level 2 — `attn` の内訳 (M6、n = 3)

level 2 は `attn` をグループごとのコマンドバッファに割るので、`attn` 自体が 3.53 → 3.96 s に膨らむ (PrefillGPUProfile.swift の注記どおり)。**割合で読み、秒は level 1 と足さない。**

| グループ | 中央値 s | `attn` 内の割合 |
| --- | ---: | ---: |
| `attn.swa` (sliding window、headDim 256) | 1.434 | 36% |
| `qkv` | 0.872 | 22% |
| `oproj` | 0.505 | 13% |
| `rope` | 0.446 | 11% |
| `post` (post norm + router) | 0.389 | 10% |
| `attn.full` (full、headDim 512) | 0.301 | 8% |
| `norm` + `kvcopy` + `embed` | 0.016 | 0% |
| **`attn` 合計 (level 2)** | **3.962** | |

## 2. 分かったこと / 外れたこと

- **合格条件 (両機 6 本の `summary.tsv`、M6 の中央値) は満たした。G0 は閉じる。**
- **「N 倍」の分母**: G0-3 の M6 prefill 壁時計 **9.507 s**、G0-4 の routed MoE **2.678 s**。G4 用に `attn` **3.534 s** も分母に加える。
- **旧 §7-4a の値 (M6 pp 288) は再現しなかった。**今回は 260.7。一方、壁時計 − io は 7.646 s で、旧値の GPU 側 7.63 s とほぼ同じ。旧値は証拠ディレクトリが無いので、何が違ったかは追わない。
- **io は重なっていない (旧 §7-4a) は、方向としては再現した。**level 1 の 3 本で、io は 1.538 → 1.883 s (+0.345)、壁時計 − GPU 合計は 2.097 → 2.470 s (+0.373)。
  ただし io の幅が 0.35 s しかなく、旧文書の「0.85 → 3.33 s」ほどの振れは出ていない。**G5 の的は残るが、その大きさは io 約 1.8 s + 0.6 s の上限**と読むのが今の証拠の範囲。
- **壁時計 − GPU 合計は io より 0.56〜0.63 s 長い。**この差 (CPU 側の encode・同期・その他) は G0 では分けていない。
- **layerVerify の 4.5 s は prefill の裏で走るが、完全には隠れていない可能性がある。**各層の先頭で `beginOpeningRoutedExpertStreamer` が streamer キューに層ファイルの SHA-256 を積み (RealForwardRunner.swift:1566)、その層の expert 取得は同じキューの `ensureLayerOpened` で完了を待つ (Model.swift `openLayerLocked`)。
  prefill 中に expert ファイル全体 (mapped 12.90 GB) を一度読んでいることになる。**この待ちが壁時計・io・`moe` のどこに何秒出ているか、読みが io を安く見せているかは未確認。**`--verification` を変えた腕は 05 §11 に従い別ラベルで取る。今回は取っていない。
- **ページキャッシュ条件**: M6 run 1 は再起動直後 (`pre.txt`: wired 95,295 ページ、空き 88%)。run 2〜6 は wired 258,694〜311,901 ページ、空き 63〜68%。
  run 1 は io が最小 (1.641 s) で壁時計も最短 (9.188 s)。冷えた状態のほうが遅いという予想 (05 §1) とは逆向き。n = 1 のセルなので解釈しない。
- **熱**: M6 の G0-3 は run 1 → 6 で 9.188 → 9.592 s。単調ではない (run 3 < run 2)。3 セット (G0-3 → level 1 → level 2) を休みなしで続けて取った。level 2 の壁時計 (9.434〜9.742 s) は分割で膨らむので比べない。

## 3. 次の Gate に渡すもの

- **G1**: 変更なし。`qkv` + `oproj` は level 2 で `attn` の 35%。level 1 に直すと約 1.2 s、GPU 合計の 17% (**導出**: 3.534 × 0.35 ÷ 7.123)。04 G1 の「1 割強」はこの値で置き換える。
- **G2**: 形状の選び方を見直す候補 (**提案、判断はユーザー**)。M6 で最大なのは `attn.swa` (sliding window の注意本体) と `qkv`。routed MoE だけを G2 の実運用形状にすると、`attn` の半分を見落とす。
- **G5**: 的は再現した (方向のみ)。M6 の io 中央値は 1.87 s。
- **未確認のまま残るもの**: 壁時計 − GPU 合計 − io の 0.6 s の中身。full-sha256 の並行読みが io と GPU に与える影響。MBP の G0-4 (04 では M6 のみと決めているので取っていない)。
