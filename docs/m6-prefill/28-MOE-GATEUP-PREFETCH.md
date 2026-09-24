# 28. B2 の 2 段目 `moe.gateup` — 先読みと M32 の候補 `TF_PREFILL_MOE_PF=1` (2026-09-24)

**候補の測定。台帳 §1 は動かない。**[04 B2](04-GATES.md) の `moe.gateup` の閾値 (level 2 の中央値 0.50 s 以下) には**届かなかった** (0.567 s)。
壁時計 (c) は、既定と交互に各 n = 6 取った中央値が縮まず (3.698 → 3.705 s)、原則 2 の「縮んだ候補」に当たらないので既定には入れていない。

候補 (22fcb46): mpp 経路の gate/up と down を `prefill_moe_gemm_int4_mpp_pf` に替える。
次の K ステップのデバイス読み込み (集めた活性と int4 の重み) を、今のステップの `matmul2d` の前にレジスタへ出す。
32 行以下のブロックは、32 行の同じカーネル (`_m32`) に回す。2 つの dispatch は書く行が重ならないので、1 つの concurrent encoder に載せる。
値・丸め・K の順は mpp と同じ。

証拠 (M6、この順に取った):
[`b2g-kernel-check`](../../bench/m6/results/2026-09-24-Mac18-5-b2g-kernel-check/)・[`b2g-profile-detail-fused`](../../bench/m6/results/2026-09-24-Mac18-5-b2g-profile-detail-fused/) (7667010、§1 の gate/up 融合)、
単体ベンチ `b2g-moe-gemm-bench*` 7 本 (§2)、
[`b2g-pf-kernel-check`](../../bench/m6/results/2026-09-24-Mac18-5-b2g-pf-kernel-check/)・[`b2g-profile-detail-pf`](../../bench/m6/results/2026-09-24-Mac18-5-b2g-profile-detail-pf/) (db19ce9、M32 なし)、
[`b2g-pf32-kernel-check`](../../bench/m6/results/2026-09-24-Mac18-5-b2g-pf32-kernel-check/)・[`b2g-profile-detail-pf32`](../../bench/m6/results/2026-09-24-Mac18-5-b2g-profile-detail-pf32/)・
`b2g-def-a`・`b2g-pf-a`・`b2g-def-b`・`b2g-pf-b`・[`b2g-greedy-pf`](../../bench/m6/results/2026-09-24-Mac18-5-b2g-greedy-pf/) (22fcb46)。

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 22fcb46 / minos 15.0
(途中の候補とベンチは各証拠ディレクトリの hostinfo.txt の commit)
```

壁時計は台帳の固定形 (`prefill_bench.sh`)。level 2 は `TF_PREFILL_GPU_PROFILE=2` を足した。greedy は末尾に `--max-new 64 --temperature 0`。
単体ベンチは `bench/m6/moe_gemm_bench.sh` (`TsugumiKernelCheck --group-size 32 --moe-gemm-bench`)。
prefill_bench の 23 run、KernelCheck 3 回 (各 group 64 / 32)、単体ベンチ 7 回がすべて exit 0、Swapouts 0。

## 0. 結論

1. **level 2 の `moe.gateup` は 0.753 → 0.567 s** (中央値、下限 0.253 s)。閾値 0.50 s には 0.067 s 足りない。`moe.down` も 0.352 → 0.291 s。
2. **壁時計 (c) は縮んだと言えない。**既定 3.698 s、候補 3.705 s (中央値、各 n = 6、3 本ずつ交互)。各組の 1 本目 (4.8〜5.2 s) が 6 本中 2 本を占め、中央値はその隣の値で決まっている。1 本目を除いた 4 本は既定 3.525〜3.708 s、候補 3.396〜3.991 s (§3)。
3. 正しさ: KernelCheck の `mpp-pf vs mpp` は M32 の分割込みで 811008 要素すべて一致 (4 回とも)、全ケース PASS。greedy 64 トークンの本文は 2 本とも c0707c48e5 で、既定と同じ。
4. 単体ベンチで、今の mpp カーネルは 64 行ブロックが埋まっていても 8.2 TFLOP/s。先読みで 11.2。gate/up 融合・threadgroup メモリ 2 面・128 行や 128 列のタイル・half の展開は、遅いか 3% 以内だった (§2)。

## 1. 段の秒 (level 2、run 順)

| ディレクトリ | 構成 | `moe.gateup` s | `moe.down` s |
| --- | --- | --- | --- |
| b2-profile-detail ([27](27-B1-FLOOR-TABLE-ON-SWA-MPP.md)) | 既定 | 0.791、0.753、0.734 | 0.366、0.352、0.345 |
| b2g-profile-detail-fused | gate/up 融合 (7667010) | 0.849、0.804、0.805 | 0.361、0.345、0.345 |
| b2g-profile-detail-pf | 先読みだけ (db19ce9) | 0.675、0.620、0.617 | 0.346、0.322、0.320 |
| b2g-profile-detail-pf32 | 先読み + M32 (22fcb46、候補) | 0.607、0.567、0.553 | 0.309、0.291、0.285 |

- 融合は `moe.gelu` (0.010 s) を無くしたが、gate/up は長くなった。集めた活性の読み込みは律速ではなかった。
- 実効 (導出): 候補の `moe.gateup` は 4.716 TFLOP ÷ 0.567 s = 8.3 TFLOP/s (既定 6.3)。

## 2. 単体ベンチ (`--moe-gemm-bench`、16 expert の 1 タイル、gate/up の 1 dispatch、20 回の中央値、µs)

| 行 / expert | mpp | mpp-pf | mpp-pf-64/32 | mpp-pf-128s8/64/32 | mpp-pf-n128s8 | mpp-pf-hd |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 27 | 795 | 731 | 486 | 486 | 761 | 712 |
| 64 | 1051 | 782 | 783 | 783 | 844 | 764 |
| 128 | 2025 | 1476 | 1477 | 1544 | 1628 | 1439 |
| 155 | 2786 | 2128 | 1969 | 2042 | 2336 | 2066 |
| 256 | 3969 | 2893 | 2892 | 3137 | 3191 | 2817 |

- 列の出所: mpp・mpp-pf・mpp-pf-64/32 は [`-m32`](../../bench/m6/results/2026-09-24-Mac18-5-b2g-moe-gemm-bench-m32/)、`128s8`・`n128s8` は [`-s8`](../../bench/m6/results/2026-09-24-Mac18-5-b2g-moe-gemm-bench-s8/)、`hd` は [`-hd`](../../bench/m6/results/2026-09-24-Mac18-5-b2g-moe-gemm-bench-hd/)。同じ列のディレクトリ間の差は、27 行の mpp (777〜795 µs) を除いて 1% 以内。
- 他の変種: gate/up 融合は mpp より 1% 速い〜14% 遅い ([`b2g-moe-gemm-bench`](../../bench/m6/results/2026-09-24-Mac18-5-b2g-moe-gemm-bench/))。threadgroup メモリ 2 面は mpp-pf より 19〜25% 遅い ([`-db`](../../bench/m6/results/2026-09-24-Mac18-5-b2g-moe-gemm-bench-db/))。128 行を 128 スレッドで回すと 4〜7 倍遅い ([`-tiers`](../../bench/m6/results/2026-09-24-Mac18-5-b2g-moe-gemm-bench-tiers/))。
- `hd` は sym の重みを half4 の `(q − 8) × scale` で展開したもの (2〜3% 速い)。候補には入れていない。
- 1 dispatch ごとに command buffer を分けても、8 本まとめても同じ (27 行の mpp の 1 本目を除く)。
- M32 の 1 ブロックは M64 の約 65% の時間 (12・27 行)。行に比例しない分が 1 ブロックの約 3 割ある (導出)。重みの展開を half にしても 2〜3% しか減らないので、その分は展開の演算ではない (中身は未確認)。

## 3. 壁時計 (c)、既定と交互 (22fcb46、run 順)

| 組 | 既定 | 候補 `TF_PREFILL_MOE_PF=1` |
| --- | --- | --- |
| a | 4.873、3.708、3.688 | 5.149、3.991、3.418 |
| b | 4.775、3.574、3.525 | 4.903、3.417、3.396 |
| 中央値 (n = 6) | 3.698 | 3.705 |

- 候補 a の 2 本目 (3.991 s) は load 0.813 s・layerVerify 7.614 s で、他の 2 本目以降 (load 0.54 s 前後・layerVerify 6.0〜6.2 s) から外れている。
- 既定の側も、台帳の `ledger-swa-mpp-default` (3.496 s) より遅い。M6 の状態の差 (原因は調べていない)。
- greedy の decode は 32.325、31.126 tok/s (n = 2、解釈は書かない)。

## 4. 次に渡すもの

- 候補は既定に入れていない (原則 2)。n = 6 を 3 本ずつ交互に取る形では、1 本目の性質 ([22](22-GPU-CANDIDATES-GREEDY-AND-FIRST-RUN.md)) が中央値を決める。
- `moe.gateup` は閾値に届いていない。1 ブロックの固定分 (§2) の中身が次の問い。
- 先読みは dense の QMM (`mpp_prefill_affine_tile64_f16`、`qkv`・`oproj`・`shared` が使う) にも同じ形で入る。B2 の 2 番目は `shared` ([27](27-B1-FLOOR-TABLE-ON-SWA-MPP.md))。
