# 12. G3 routed MoE の expert GEMM を `matmul2d` 化 — routed MoE の GPU 秒 2.58 → 1.10 s、prefill 壁時計 9.01 → 7.38 s (実測、2026-09-24)

> 読み替え ([18 §3](18-RESET-ABSOLUTE-FLOOR.md)): 壁時計は新しいプロセスの 1 回目 (c) で、起動時の SHA-256 検証を含む。routed MoE の GPU 秒の縮みは有効。下限の何倍かは [04 B1](04-GATES.md) で出す。

証拠 (すべて M6、同じ commit・同じバイナリで腕は `TF_PREFILL_MOE` だけ):

- 壁時計: [`g3-simdgroup-a`](../../bench/m6/results/2026-09-24-Mac18-5-g3-simdgroup-a/summary.tsv) → [`g3-mpp-a`](../../bench/m6/results/2026-09-24-Mac18-5-g3-mpp-a/summary.tsv) → [`g3-simdgroup-b`](../../bench/m6/results/2026-09-24-Mac18-5-g3-simdgroup-b/summary.tsv) → [`g3-mpp-b`](../../bench/m6/results/2026-09-24-Mac18-5-g3-mpp-b/summary.tsv) (この順に各 3 本、腕あたり n = 6)
- GPU 内訳: [`g3-profile-simdgroup`](../../bench/m6/results/2026-09-24-Mac18-5-g3-profile-simdgroup/summary.tsv)、[`g3-profile-mpp`](../../bench/m6/results/2026-09-24-Mac18-5-g3-profile-mpp/summary.tsv) (`TF_PREFILL_GPU_PROFILE=1`、各 n = 3)
- greedy: [`g3-greedy-simdgroup`](../../bench/m6/results/2026-09-24-Mac18-5-g3-greedy-simdgroup/summary.tsv)、[`g3-greedy-mpp`](../../bench/m6/results/2026-09-24-Mac18-5-g3-greedy-mpp/summary.tsv) (各 n = 2)
- CPU 参照との一致: [`g3-kernel-check-diff`](../../bench/m6/results/2026-09-24-Mac18-5-g3-kernel-check-diff/output.txt) (直前の commit 70255d9 で取った [`g3-kernel-check`](../../bench/m6/results/2026-09-24-Mac18-5-g3-kernel-check/output.txt) は差の要素数の列が無い版)

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 7c7ad7b / minos 15.0
```

コマンド (`command.txt`。greedy は後ろに `--max-new 64 --temperature 0` が付く):

```
/Users/mh/dev/tsugumi/.build/release/TsugumiCLI --model /Users/mh/dev/tsugumi/scratch/gemma4-qat-sym.gturbo --messages-file /Users/mh/dev/tsugumi/bench/l.json --expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 --max-context 4096 --verification full-sha256
```

起動は `TF_PREFILL_MOE=<simdgroup|mpp> [TF_PREFILL_GPU_PROFILE=1] bench/m6/cycle.sh prefill_bench.sh <label> <n> [-- …]`。全 run exit 0。
mpp 腕の全 run のログに `[prefill moe tiled kernel=mpp]` が出ている (経路が本当に切り替わった証拠)。

## 0. 結論

1. **routed MoE の GPU 秒が 2.577 s → 1.096 s (2.35 倍、導出)。**GPU 合計は 7.022 → 5.543 s。`attn`・`shared` は動いていない (§1.2)。
2. **prefill 壁時計が 9.009 s → 7.383 s (1.22 倍、導出)、pp 275.1 → 335.6。**GPU の縮みがほぼそのまま壁時計に出た。壁時計 − GPU 合計は 1.669 → 1.728 s で、io はまだ露出していない。
3. **CPU 参照とは両カーネルとも一致 (rel 4.9e-4、許容 1e-2)。greedy 64 トークンは 3 行目で分かれる** (§1.3)。分かれ方は G1 の MPP 射影腕 ([16 §1.1](16-G1-MPP-GROUP32.md)) と同じ行・同じ文字列。

## 1. 数字

### 1.1 壁時計 (腕あたり n = 6、中央値 (レンジ))

| | simdgroup (既定) | mpp | simdgroup / mpp (**導出**) |
| --- | ---: | ---: | ---: |
| prefill 壁時計 s | 9.009 (8.746〜9.082) | 7.383 (7.317〜7.621) | 1.22 |
| pp tok/s | 275.1 (272.8〜283.3) | 335.6 (325.2〜338.7) | — |
| prefill io s | 1.120 (1.011〜1.212) | 1.017 (0.968〜1.190) | — |
| layerVerify s | 4.587 (4.464〜4.626) | 4.484 (4.451〜4.522) | — |

### 1.2 GPU 内訳 (level 1、各 n = 3、中央値 (レンジ))

| 段 | simdgroup | mpp |
| --- | ---: | ---: |
| `moe` | 2.577 (2.572〜2.584) | 1.096 (1.095〜1.102) |
| `attn` | 3.529 (3.528〜3.549) | 3.537 (3.529〜3.540) |
| `shared` | 0.866 (0.865〜0.872) | 0.866 (0.865〜0.866) |
| **GPU 合計** | **7.022 (7.012〜7.043)** | **5.543 (5.541〜5.547)** |
| prefill 壁時計 | 8.712 (8.648〜8.896) | 7.271 (7.207〜7.300) |
| 壁時計 − GPU 合計 (**導出**、run ごと) | 1.669 (1.636〜1.874) | 1.728 (1.660〜1.759) |

**実効 (導出)**: routed MoE の演算量 7.07 TFLOP ([17 §1](17-NAX-NOT-USED.md)) ÷ 1.096 s = **6.5 TFLOP/s** (simdgroup は 2.7)。MPS の fp16 GEMM 18.61 TFLOP/s ([03 §1](03-FACTS.md)) の約 1/3。`moe` は tile stream の待ちを含むので、カーネル単体の値はこれより高い側にある。

### 1.3 正しさ

- **KernelCheck** (D 2816 / F 704、16 expert、288 行、64 行を超える expert 1 個を含む 17 ブロック、group 32/64 × affine/sym): simdgroup と mpp の両方が CPU 参照と rel 4.86e-4〜4.95e-4。mpp と simdgroup の出力は 811,008 要素中 48,742〜59,529 個 (6〜7%) が異なり、差の最大は 0.015625 (出力の最大値付近の fp16 1 ulp)。
- **greedy** (各腕 2 本は互いに同一):

| 腕 | 3 行目 |
| --- | --- |
| simdgroup | `### 1. KVキャッシュの「逐次的な確保」によるメモリピークの抑制` |
| mpp | `### 1. KVキャッシュの逐次的な確保によるメモリピークの抑制` |

## 2. 分かったこと / 外れたこと

- **合否 ([04 G3](04-GATES.md))**: 「routed MoE の GPU 秒が下がり、壁時計も下がる」は満たした。「greedy 出力同一」は満たしていない。[17 §3](17-NAX-NOT-USED.md) で greedy の差は記録して止めないと決めたので、ここでは記録だけにする。
- **G1 と同じ分岐**: 射影だけを MPP にした G1 と、routed MoE だけを MPP にした今回が、同じ 3 行目で同じ文字列に分かれた。1 か所の丸め順の違いで入れ替わる、僅差の 2 候補があるトークンに見える。**調べていない**ので断定しない (n = 2)。
- **カーネルの中身**: 変えたのはタイルの積だけ。A (gather した活性) と B (int4 → fp16 に展開した重み) を threadgroup メモリに置き、`matmul2d` (64 × 64 × 64、`multiply_accumulate`、`execution_simdgroups<4>`) で K ループ全体を 1 つの cooperative destination に積む。展開した重みの値は simdgroup 版と同じ。K の刻みは 32 → 64 (D、F とも 64 の倍数)。
- **17 §2 の「G1 の MPP カーネルが遅い理由」との関係**: 今回のカーネルも「展開 → barrier → `matmul2d` → barrier」の直列は残している。違いは累積を `multiply_accumulate` で destination に任せ、手の加算ループと毎ステップのゼロ埋めを外したこと、タイルが 64 × 64 であること。どれが効いたかは分けていない。
- **今日の simdgroup 腕 (9.009 s) は G0 (9.507 s、commit 837586c) より速い。**commit も日も違うので比べない。この文書の比はすべて同じ commit の対。

## 3. 次に渡すもの

- **既定は変えていない。**`TF_PREFILL_MOE=mpp` のときだけ新カーネル。既定にするかは判断を仰ぐ (greedy の分岐を受け入れるかどうかと同じ問い)。
- **分母の更新**: 以降の比較の対照は、同じ commit の simdgroup 腕と mpp 腕の両方を取る。
- **残る的**: GPU 合計 5.54 s のうち `attn` 3.54 s (64%)。計画の次は射影と shared ([17 §3](17-NAX-NOT-USED.md) の 2)。routed MoE 自体も実効 6.5 TFLOP/s で、天井までは余地がある。
- **io**: 壁時計 − GPU 合計はほぼ動かず (1.67 → 1.73 s)。GPU が縮んでも io が表に出てくる兆候はまだ無い (G5 は後でよい)。
