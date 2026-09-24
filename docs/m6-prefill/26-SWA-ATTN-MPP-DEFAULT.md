# 26. sliding window の attention の `matmul2d` 版を既定に — 台帳 4.584 → 3.496 s (実測、2026-09-24)

[25](25-SWA-ATTN-MPP.md) の候補 `TF_PREFILL_ATTN=mpp` を、cd8c26b で Gemma の既定にした ([04 B3](04-GATES.md))。
使う条件は 25 と同じ (ライブラリに tensor ops がある、headDim 256、span mask なし、ring が 64 の倍数)。それ以外は今までどおり sgmm GQA 版。
戻すのは `TF_PREFILL_ATTN=sgmm` (sgmm GQA 版) か `=qblock`。Qwen の経路は `PrefillAttention` を通らないので変わらない。

証拠 (M6、この順に取った):
[`swa-default-kernel-check`](../../bench/m6/results/2026-09-24-Mac18-5-swa-default-kernel-check/) (cd8c26b)、
[`ledger-swa-mpp-default`](../../bench/m6/results/2026-09-24-Mac18-5-ledger-swa-mpp-default/)、
[`swad-greedy`](../../bench/m6/results/2026-09-24-Mac18-5-swad-greedy/)、
[`swad-env-sgmm`](../../bench/m6/results/2026-09-24-Mac18-5-swad-env-sgmm/) (`TF_PREFILL_ATTN=sgmm`)。

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit bda6631 / minos 15.0
(KernelCheck だけ cd8c26b。bda6631 は docs と証拠だけを足した commit で、ソースは cd8c26b と同じ)
```

コマンドは台帳の固定形 (25 と同じ)。`swad-greedy` だけ末尾に `--max-new 64 --temperature 0`。
MBP では cd8c26b の `Scripts/test.sh` が 1627 本通った (exit 0)。

**台帳** ([06 §1](06-LEDGER.md)): `ledger-swa-mpp-default` 3.496 s (3.442〜4.764)。変更前 `ledger-gpu3-default` 4.584 s、差 −1.088 s (導出)。

## 0. 結論

1. **既定の prefill 壁時計は 4.584 → 3.496 s。**
2. **B3 の合格範囲 (25 の候補 3.547〜4.885 s) からは、速い側に 0.051 s 外れた。**同じバイナリで sgmm に戻した対照 (`swad-env-sgmm`) も、25 の既定 (4.207〜5.329 s) より速い (4.023〜4.999 s)。prefill io も両方で 25 より小さい。既定の書き換えのミスではなく、M6 の状態の差と読む (原因は調べていない、ページキャッシュは冷やせない、[01 §3](01-ROLES.md))。
3. 正しさ: KernelCheck は group 64 の 79 ケース、group 32 の 95 ケースがすべて PASS (SWA attention の mpp 3 形を含む)。greedy 64 トークンの本文は 2 本とも c0707c48e5 で、25 の既定・候補と同じ。全 11 run で exit 0、Swapouts 0。

## 1. 数字

| ディレクトリ | 構成 | prefill 壁時計 (run 順) | prefill io (run 順) | peak_gb |
| --- | --- | --- | --- | ---: |
| ledger-swa-mpp-default | 既定 | 4.764、3.932、3.510、3.462、3.482、3.442 | 0.588、0.447、0.360、0.335、0.349、0.301 | 1.22 |
| swad-greedy | 既定 + greedy | 4.638、3.618 | 0.487、0.529 | 1.22 |
| swad-env-sgmm | `TF_PREFILL_ATTN=sgmm` | 4.999、4.124、4.023 | 0.713、0.551、0.455 | 1.44〜1.45 |

- 1 本目の layerVerify は 9.44〜9.54 s。2 本目以降は 5.88〜7.04 s。
- greedy の decode は 31.827、32.716 tok/s (n = 2、解釈は書かない)。
- `swad-env-sgmm` は n = 3 で、同じ時間帯に交互には取っていない (既定の 6 本の後)。

## 2. 変えたもの (cd8c26b)

- `PrefillAttention`: 既定の `causalSimdgroupMatrix` で、25 の条件を満たす headDim 256 の dispatch を `attention_prefill_swa_mpp_d256` に送る。`TF_PREFILL_ATTN=sgmm` か `=qblock` のときは使わない。25 の `TF_PREFILL_ATTN=mpp` は、もう付けても何も変わらない。
- `bench/m6/ledger.sh`: 既定の表 (§1) を `ledger-*` と `g0-baseline` だけにした。候補の隣で取った既定の対照 (n = 3) が §1 に混ざっていたため。

## 3. 次に渡すもの

- B2 の次は [24 §1.3](24-B1-FLOOR-TABLE.md) の順の 2 番目 `moe.gateup` (差 0.485 s)。段の実測は、この既定の上で取り直す。
- sgmm の既定化 (23) で増えた peak_gb +0.22 GB は、この既定では出ない (1.22 GB)。
