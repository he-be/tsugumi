# 21. GPU 側の候補 3 つを、検証の並行先読みが入った既定の上で — 3 つ全部で prefill 壁時計 7.171 → 4.167 s (実測、2026-09-24)

証拠 (M6、この順に取った。a 巡は下の表の順、b 巡は逆順):
[`gpu3-base-a`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3-base-a/)、
[`gpu3-moe-a`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3-moe-a/)、
[`gpu3-qmm-a`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3-qmm-a/)、
[`gpu3-attn-a`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3-attn-a/)、
[`gpu3-moeqmm-a`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3-moeqmm-a/)、
[`gpu3-all3-a`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3-all3-a/)、
[`gpu3-all3-b`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3-all3-b/)、
[`gpu3-moeqmm-b`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3-moeqmm-b/)、
[`gpu3-attn-b`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3-attn-b/)、
[`gpu3-qmm-b`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3-qmm-b/)、
[`gpu3-moe-b`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3-moe-b/)、
[`gpu3-base-b`](../../bench/m6/results/2026-09-24-Mac18-5-gpu3-base-b/)。
ホスト札 (12 ディレクトリ共通):

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 3075b56 / minos 15.0
```

3075b56 は 79eea8f (検証の並行先読みが既定) に文書の変更だけを足した commit で、ソースは同じ。

コマンド (12 ディレクトリ共通、固定形。差は `env.txt` の `TF_` だけ)。反復は各 3、構成あたり n = 6:

```
/Users/mh/dev/tsugumi/.build/release/TsugumiCLI --model /Users/mh/dev/tsugumi/scratch/gemma4-qat-sym.gturbo --messages-file /Users/mh/dev/tsugumi/bench/l.json --expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 --max-context 4096 --verification full-sha256
```

| 構成 | `TF_` |
| --- | --- |
| base | なし (既定) |
| moe | `TF_PREFILL_MOE=mpp` |
| qmm | `TF_PREFILL_QMM=mpp64` |
| attn | `TF_PREFILL_ATTN=sgmm` |
| moeqmm | `TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64` |
| all3 | `TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_ATTN=sgmm` |

**台帳** ([06](06-LEDGER.md)): どれも候補。台帳 §1 は動かない。

## 0. 結論

1. **検証の並行先読みが既定に入った後は、GPU 側の候補 3 つとも、単独で prefill 壁時計を縮める。**既定 7.171 s に対して moe −1.479 s、qmm −1.069 s、attn −0.395 s (導出)。
2. **3 つ全部で 4.167 s (4.067〜5.111)、既定との差 −3.004 s (導出)。**moeqmm (4.636 s) に attn を足した差は −0.469 s (導出)。
3. run 中の Swapouts は 36 本すべて 0、失敗 0。

## 1. 数字

prefill 壁時計は `summary.tsv` の `prefill_s`、a と b の 6 本の中央値。測った量は [04 B0](04-GATES.md) の (c)。

| 構成 | prefill 壁時計 中央値 (レンジ) | 既定との差 s (**導出**) | prefill io 中央値 | layerVerify 中央値 | peak_gb |
| --- | ---: | ---: | ---: | ---: | --- |
| base | 7.171 (7.078〜7.260) | — | 0.695 | 4.601 | 1.21〜1.22 |
| moe | 5.692 (5.553〜5.836) | −1.479 | 0.631 | 4.954 | 1.21〜1.22 |
| qmm | 6.102 (5.945〜6.174) | −1.069 | 0.697 | 4.827 | 1.21〜1.22 |
| attn | 6.776 (6.611〜6.808) | −0.395 | 0.888 | 4.701 | 1.44 |
| moeqmm | 4.636 (4.482〜5.297) | −2.535 | 0.557 | 5.765 | 1.21〜1.22 |
| all3 | 4.167 (4.067〜5.111) | −3.004 | 0.579 | 5.929 | 1.43〜1.44 |

- base の 6 本は台帳の `ledger-verify-ahead` (7.184 s、7.107〜7.275) と同じ範囲にある。
- run 中の Swapouts / Swapins (`run-N.post.txt` − `run-N.pre.txt`) は全構成・全 run で 0 / 0。
- greedy の本文と decode tok/s は、この測定では取っていない (`--max-new 1`)。

## 2. 分かったこと / 外れたこと

- [13](13-PROJ-SHARED-MATMUL2D.md) で「単独では壁時計を動かさない」とした qmm は、新しい既定の上では単独で −1.069 s。[20 §2](20-VERIFY-AHEAD-ALONE.md) の見立て (検証を裏に回すと GPU 側が律速になる) と合う。
- moeqmm と all3 は 4 セット (a、b) すべてで 1 本目だけ遅い (4.935〜5.297 s)。条件は直前の空き (24 秒以上) だった ([22](22-GPU-CANDIDATES-GREEDY-AND-FIRST-RUN.md))。その 1 本目の layerVerify は 9.498〜9.616 s で、他の run の 5.6〜5.9 s より大きい。[19 §1](19-SWA-ATTN-SGMM-GQA.md) の best4 で見えたのと同じ形。単独の 3 構成の 1 本目は layerVerify 5.269〜6.172 s、壁時計は同じ構成の他の run と 0.3 s 以内で、base では 1 本目も他と同じ。原因は調べていない。
- moeqmm と all3 の layerVerify 中央値は base より 1.2〜1.3 s 大きい。layerVerify は層ごとの合計なので、壁時計の成分ではない。
- attn は peak_gb が 0.22 GB 増える (19 と同じ)。

## 3. 次に渡すもの

- 既定に入れるかはユーザーが決める ([04 B3](04-GATES.md))。候補ごとに残っている問い:
  - moe: greedy が 3 行目で分かれる ([12 §2](12-G3-MOE-MATMUL2D.md))。
  - attn: peak_gb が +0.22 GB 増える。
  - qmm: 13 の問い (moe とセットか) には、単独でも効くという答えが出た。
- 組み合わせの 1 本目が遅い現象と候補の greedy は [22](22-GPU-CANDIDATES-GREEDY-AND-FIRST-RUN.md) で測った。
