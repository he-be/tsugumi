# 23. GPU 側の候補 3 つを既定に — 台帳 7.184 → 4.584 s (実測、2026-09-24)

[21](21-GPU-CANDIDATES-ON-VERIFY-AHEAD.md)・[22](22-GPU-CANDIDATES-GREEDY-AND-FIRST-RUN.md) の候補 `mpp` / `mpp64` / `sgmm` を、5decec4 で Gemma の既定にした (B3)。
戻すのは `TF_PREFILL_MOE=simdgroup` / `TF_PREFILL_QMM=simdgroup` / `TF_PREFILL_ATTN=qblock`。Qwen の経路は変えていない。tensor ops の無い機械では、MoE と射影はこれまでどおり simdgroup に落ちる。sgmm はどの機械でも既定になる。

証拠 (M6、この順に取った):
[`gpu3d-kernel-check`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3d-kernel-check/)、
[`ledger-gpu3-default`](../../bench/m6/results/2026-09-24-Mac18-5-ledger-gpu3-default/)、
[`gpu3d-greedy`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3d-greedy/)、
[`gpu3d-def-a`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3d-def-a/)、
[`gpu3d-env-a`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3d-env-a/)、
[`gpu3d-env-b`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3d-env-b/)、
[`gpu3d-def-b`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3d-def-b/)。
ホスト札 (`gpu3d-kernel-check`、`ledger-gpu3-default`、`gpu3d-greedy` は 5decec4、`gpu3d-def-*` と `gpu3d-env-*` は証拠だけを足した次の commit。どちらもソースは 5decec4):

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 5decec4 / minos 15.0
```

コマンドは 21 と同じ固定形。`gpu3d-greedy` だけ末尾に `--max-new 64 --temperature 0`、`gpu3d-env-*` だけ `TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_ATTN=sgmm`。
MBP では `Scripts/test.sh` が 1627 本すべて通った (exit 0、Swift Testing 248 スイート)。

**台帳** ([06](06-LEDGER.md)): 既定が変わった。`ledger-gpu3-default` 4.584 s (4.545〜6.110)、変更前 `ledger-verify-ahead` 7.184 s、差 −2.600 s (導出)。

## 0. 結論

1. **既定の prefill 壁時計は 7.184 → 4.584 s。**[04 B3](04-GATES.md) の合格条件 (中央値が候補 all3 のレンジ 4.067〜5.111 s に入る) を満たす。
2. **同じバイナリで、既定と `TF_` 3 つを明示した構成は同じ範囲にある** (2 本目以降 4.516〜4.628 s と 4.556〜4.732 s)。既定の書き換えは、候補と同じ経路を走っている。
3. 正しさ: KernelCheck は group 64 の 65 ケース、group 32 の 81 ケースがすべて PASS、SKIP 0。greedy 64 トークンの本文は 2 本とも c0707c48e5 で、22 の候補 4 構成と同じ。Swapouts は全 run で 0、失敗 0。

## 1. 数字

| ディレクトリ | 構成 | prefill 壁時計 (run 順) | prefill io (run 順) | layerVerify 1 本目 |
| --- | --- | --- | --- | ---: |
| ledger-gpu3-default | 既定 | 6.110、4.628、4.546、4.545、4.615、4.553 | 1.642、0.995、0.910、0.924、0.995、0.927 | 10.093 |
| gpu3d-def-a | 既定 | 5.549、4.580、4.529 | 1.280、0.982、0.918 | 9.519 |
| gpu3d-env-a | `TF_` 3 つ | 5.703、4.732、4.707 | 1.440、1.108、1.091 | 9.452 |
| gpu3d-env-b | `TF_` 3 つ | 5.757、4.600、4.556 | 1.488、0.970、0.945 | 9.500 |
| gpu3d-def-b | 既定 | 5.771、4.527、4.516 | 1.509、0.929、0.899 | 9.447 |
| gpu3d-greedy | 既定 + greedy | 4.752、4.723 | 1.000、1.179 | 6.723 |

- 1 本目の layerVerify は、`gpu3d-greedy` 以外の 5 セットで 9.447〜10.093 s。[22](22-GPU-CANDIDATES-GREEDY-AND-FIRST-RUN.md) の空きのある 1 本目と同じ形。今回は空き時間を計算していない。
- greedy の decode は 28.717、29.561 tok/s (n = 2、解釈は書かない)。
- peak_gb は全 run 1.43〜1.44。

## 2. 分かったこと / 外れたこと

- 22 の all3 (2 本目以降 4.063〜4.241 s、prefill io 0.48〜0.66 s) より、今回は 2 本目以降が 0.3〜0.5 s 遅く、prefill io が 0.9〜1.1 s と大きい。同じバイナリの既定と `TF_` 明示が同じように遅いので、既定の書き換えの差ではなく、M6 の状態の差 (22 の後に 5decec4 のビルドと KernelCheck のビルドがあった)。原因は調べていない。ページキャッシュは冷やせない ([01 §3](01-ROLES.md))。
- 台帳の中央値は合格範囲に入ったが、候補の中央値 4.167 s とは 0.417 s 離れている (導出)。上の状態の差を含む。

## 3. 次に渡すもの

- 次の段は [04 B1](04-GATES.md) (段ごとの下限表)。GPU 側の段の実測は、この既定の上で取り直す。
- sgmm は MBP・M1 でも既定になったが、そこでの prefill の秒は測っていない (記録値は M6 だけ)。
- 空きのある 1 本目の +0.9 s 前後 (22) は、既定の性質として残る。
