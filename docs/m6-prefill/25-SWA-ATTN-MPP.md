# 25. B2 の 1 段目 — sliding window の attention を `matmul2d` に。attn.swa 0.873 → 0.288 s、壁時計は既定 4.522 s に対して候補 3.579 s (実測、2026-09-24)

**候補 (`TF_PREFILL_ATTN=mpp`)。台帳 §1 は動かない** (既定の行は [06 §1](06-LEDGER.md) の `ledger-gpu3-default` 4.584 s のまま)。cd8c26b で既定に入れた ([26](26-SWA-ATTN-MPP-DEFAULT.md))。
[04 B2](04-GATES.md) の順 ([24 §1.3](24-B1-FLOOR-TABLE.md)) の 1 番目 `attn.swa` (下限 0.044 s、変更前の差 0.769 s)。閾値は計測の前に 04 に置いた。

証拠 (M6、この順に取った):
[`b2-swa-kernel-check`](../../bench/m6/results/2026-09-24-Mac18-5-b2-swa-kernel-check/) (badca3a)、
[`b2-swa-profile-detail`](../../bench/m6/results/2026-09-24-Mac18-5-b2-swa-profile-detail/)、
[`b2-def-a`](../../bench/m6/results/2026-09-24-Mac18-5-b2-def-a/)、
[`b2-mpp-a`](../../bench/m6/results/2026-09-24-Mac18-5-b2-mpp-a/)、
[`b2-mpp-b`](../../bench/m6/results/2026-09-24-Mac18-5-b2-mpp-b/)、
[`b2-def-b`](../../bench/m6/results/2026-09-24-Mac18-5-b2-def-b/)、
[`b2-greedy-def`](../../bench/m6/results/2026-09-24-Mac18-5-b2-greedy-def/)、
[`b2-greedy-mpp`](../../bench/m6/results/2026-09-24-Mac18-5-b2-greedy-mpp/)。

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 2c5c0dd / minos 15.0
(KernelCheck だけ badca3a。2c5c0dd は KernelCheck の証拠を足しただけで、ソースは badca3a と同じ)
```

コマンド (全ディレクトリ共通、差は `env.txt` の `TF_` と、greedy の末尾 `--max-new 64 --temperature 0`):

```
/Users/mh/dev/tsugumi/.build/release/TsugumiCLI --model /Users/mh/dev/tsugumi/scratch/gemma4-qat-sym.gturbo --messages-file /Users/mh/dev/tsugumi/bench/l.json --expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 --max-context 4096 --verification full-sha256
```

16 run すべて exit 0、Swapouts 0。KernelCheck は group 64 の 79 ケース、group 32 の 95 ケースがすべて PASS。

## 0. 結論

1. **GPU の attn.swa は 0.873 s → 0.288 s** (level 2 の中央値、変更前は [24](24-B1-FLOOR-TABLE.md) の `b1s-profile-detail`、差 −0.585 s **導出**)。04 B2 の閾値 (0.45 s 以下) を満たす。下限 0.044 s との差は 0.244 s (導出)。
2. **prefill の壁時計 (c) は、同じバイナリで既定と交互に取って 4.522 s → 3.579 s** (各 n = 6、差 −0.943 s **導出**)。
3. 正しさ: KernelCheck の新しい検査で、CPU 参照との相対誤差は 3.6〜4.1e-4 (許容 1e-2)。greedy 64 トークンの本文は既定・候補の 4 本とも c0707c48e5 で、[23](23-GPU-CANDIDATES-DEFAULT.md) の既定と同じ。

## 1. 数字

| | 既定 (sgmm GQA 版) | 候補 `TF_PREFILL_ATTN=mpp` | 差 s (**導出**) |
| --- | ---: | ---: | ---: |
| attn.swa GPU 秒 (level 2、n = 3) | 0.873 (0.872〜0.878) `b1s-profile-detail` | 0.288 (0.288〜0.289) `b2-swa-profile-detail` | −0.585 |
| prefill 壁時計 (交互、a + b、n = 6) | 4.522 (4.207〜5.329) `b2-def-a/b` | 3.579 (3.547〜4.885) `b2-mpp-a/b` | −0.943 |
| peak_gb | 1.44 | 1.22 | −0.22 |

- 壁時計の run 順: 既定 a 5.124、4.429、4.300、候補 a 4.885、3.573、3.558、候補 b 4.850、3.585、3.547、既定 b 5.329、4.614、4.207。どのセットも 1 本目が遅い (layerVerify 9.49〜9.52 s)。中央値にはそのまま入れた。
- prefill io: 既定 0.638〜1.032 s、候補 0.410〜0.716 s。
- greedy の decode: 既定 30.094、31.556 tok/s、候補 32.192、32.937 tok/s (n = 2、解釈は書かない)。
- level 2 の他の段 (`b2-swa-profile-detail` の中央の run): `qkv` 0.394、`rope` 0.190、`attn.full` 0.300、`oproj` 0.234、`post.router` 0.399 s。`attn.swa` 以外は [24 §1.3](24-B1-FLOOR-TABLE.md) の生の秒と同じ範囲。
- peak_gb の 1.22 は、sgmm を既定にする前の値 ([19 §1](19-SWA-ATTN-SGMM-GQA.md) の 1.21〜1.22) と同じ。

単体ベンチ (`bench/m6/attn_bench.swift`、25 層 × 2 チャンク、**記録値ではない**、M6): sgmm GQA 版 0.854 s、候補 0.280 s。全 2478 行の出力の最大差は 1.2e-4、非有限値は 0。

## 2. カーネル (badca3a)

`attention_prefill_swa_mpp_d256` を `prefill.metal` の `__HAVE_TENSOR__` 節に足した。

- 1 threadgroup (4 simdgroup、128 スレッド) が 1 ヘッドの 64 行を持つ。GQA の 2 ヘッドは別の threadgroup で、K/V の共有はしていない。
- S = Q·Kᵀ は `matmul2d` (64 × 64 × 256)。Q と K は device から tensor として直接読む。threadgroup メモリに置くのは S (float、16 KB) と P (half、8 KB) と行ごとの max / sum / 補正係数だけ。
- softmax は 1 行を 2 スレッドで (各 32 キー)、P を half で threadgroup メモリに書く。
- P·V は `matmul2d` (64 × 256 × 64) の multiply_accumulate で、出力は 1 つの cooperative destination に全キーぶん積む。ブロックごとに補正係数を掛けてから積む。
- K/V はキー 64 本ずつ slice する。ring (`FC_PREFILL_KV_RING_CAP`) は 64 の倍数のときだけ使う (ブロックが ring の端をまたがない)。`kvValidCount` を越えるキーは tensor の範囲外として 0 で読まれ、マスクされる。
- ホスト側 (`PrefillAttention`): `TF_PREFILL_ATTN=mpp` で、ライブラリにこのカーネルがあり、headDim 256、span mask なし、ring が 64 の倍数のときだけ使う。それ以外は今までどおり。既定の経路は変えていない。

KernelCheck の新しい検査 (`PrefillSWAAttentionCheck.swift`) は 16/8 ヘッド、headDim 256、窓 100。
形は 1 チャンク目 (0〜149、チャンクがブロックの途中で終わる)、2 チャンク目 (150〜219)、同じ 2 チャンク目を 192 スロットの巡回した ring で。書いていないスロットは NaN で埋めた。
窓を半分にした参照とは相対誤差 0.284 で、検出力の下限 0.1 を越えた。MBP では sgmm の 3 形が通り、mpp は SKIP (tensor 経路が無い)。MBP の `Scripts/test.sh` は、この変更の後 (2c5c0dd の作業ツリー、ソースは badca3a) で 1627 本通過 (exit 0)。

## 3. 分かったこと / 外れたこと

- 壁時計の差 (−0.943 s) は GPU の attn.swa の差 (−0.585 s) より大きい。prefill io も候補の方が小さい。どちらも原因は調べていない。
- 1 本目が遅い性質 ([22](22-GPU-CANDIDATES-GREEDY-AND-FIRST-RUN.md)) は候補でも残る (4.850、4.885 s)。
- **span mask (画像のある prompt) と、64 の倍数でない ring では、この候補は使われない。**既定の ring 容量は window 1024 + チャンク幅で、チャンク幅 2048 なら 3072 (64 の倍数)。

## 4. 次に渡すもの

- 既定に入れた ([26](26-SWA-ATTN-MPP-DEFAULT.md)、台帳 3.496 s)。
- B2 の次は [24 §1.3](24-B1-FLOOR-TABLE.md) の順の 2 番目 `moe.gateup` (差 0.485 s)。
