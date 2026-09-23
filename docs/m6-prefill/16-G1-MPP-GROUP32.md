# 16. G1 MPP 経路を group 32 で開ける — 経路は開いて数値検査も通ったが、greedy 出力が変わるので不合格で止めた (実測、2026-09-23)

番号について: 04 は G2〜G6 に 11〜15 を割り当てているが、G1 には番号が無い。空いている 16 を使う。

証拠 (すべて M6):
[`2026-09-23-Mac18-5-g1-kernel-check/`](../../bench/m6/results/2026-09-23-Mac18-5-g1-kernel-check/output.txt) (合格条件 a)、
[`2026-09-23-Mac18-5-g1-greedy-simdgroup/`](../../bench/m6/results/2026-09-23-Mac18-5-g1-greedy-simdgroup/summary.tsv) と
[`2026-09-23-Mac18-5-g1-greedy-mpp/`](../../bench/m6/results/2026-09-23-Mac18-5-g1-greedy-mpp/summary.tsv) (合格条件 b、各 n = 2)。

```
測定: 2026-09-23 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 906eae2 / minos 15.0
```

コマンド (b。`command.txt` をそのまま貼る。`--max-new 1` は後ろの `--max-new 64` が上書きする):

```
/Users/mh/dev/tsugumi/.build/release/TsugumiCLI --model /Users/mh/dev/tsugumi/scratch/gemma4-qat-sym.gturbo --messages-file /Users/mh/dev/tsugumi/bench/l.json --expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 --max-context 4096 --verification full-sha256 --max-new 64 --temperature 0
```

腕は `env.txt` に残っている: `TF_PREFILL_QMM=simdgroup` / `mpp`。どちらも `TF_PREFILL_GPU_PROFILE=1`。起動は `TF_PREFILL_QMM=<arm> TF_PREFILL_GPU_PROFILE=1 bench/m6/cycle.sh prefill_bench.sh g1-greedy-<arm> 2 -- --max-new 64 --temperature 0`。
a は `bench/m6/cycle.sh kernel_check.sh g1-kernel-check`。全 run exit 0。

## 0. 結論

1. **(a) 合格。**M6 で MPP の int4 QMM が group 64 と 32、affine と sym の両方で CPU 参照と一致した (rel 2.7e-4〜4.0e-4、許容 5e-3)。
   group 32 では、scale をタイル番号で索く誤った参照に対して rel 7.7e-2〜1.9e-1 で**外れる**ことも確かめた。検査には検出力がある。
2. **(b) 不合格。**`--temperature 0` の 64 トークンで、simdgroup と mpp の出力が 3 行目から分かれた。各腕の 2 本は互いに同一なので、揺れではなく経路の差。
3. **(c) は取っていない。**(b) が落ちたので、`g1-mpp 6` の本計測に進まずに止めた。経路ログは (b) の run で出ている (§1.2)。

## 1. 数字

### 1.1 (b) の出力の食い違い

どちらも 1 行目・2 行目は同じ。3 行目 (見出し) から分かれる:

| 腕 | 3 行目 |
| --- | --- |
| simdgroup (2 本とも) | `### 1. KVキャッシュの「逐次的な確保」によるメモリピークの抑制` |
| mpp (2 本とも) | `### 1. KVキャッシュの逐次的な確保によるメモリピークの抑制` |

G0-3 は `--max-new 1` かつ temperature 1.0 (CLI の既定) で取っているので、04 の「G0-3 と同一」とはそのままでは比べられない。
代わりに、同じバイナリの simdgroup 腕 (group 32 の既定の経路で、G0-3 と同じ経路) と比べた。

### 1.2 経路ログ (新設の `[prefill gpu proj …]` 行)

| 腕 | 1 run 分のディスパッチ数 |
| --- | --- |
| simdgroup | `kv:simdgroup-matrix=120 o:simdgroup-matrix=60 q:simdgroup-matrix=60` |
| mpp | `kv:mpp=120 o:mpp=60 q:mpp=60` |

### 1.3 数字だけ (n = 2、**解釈しない**)

| | simdgroup | mpp |
| --- | ---: | ---: |
| prefill 壁時計 s | 9.503 / 9.745 | 9.587 / 9.728 |
| GPU `attn` s | 3.532 / 3.533 | 3.226 / 3.235 |
| GPU 合計 s | 7.090 / 7.164 | 6.878 / 6.835 |

## 2. 分かったこと / 外れたこと

- **「group 32 で nil になる」(03 §2、旧 §7-1a) は原因どおりだった。**カーネルが scale/bias を K タイルの番号で索いていたのを K の絶対位置で索くように直すと (tensorops.metal)、group 32 の PSO が作れ、CPU 参照とも一致した。group 64 の計算は変更前と同じ式。
- **数値検査の許容内でも、greedy の 64 トークンは同一にならない。**これが MPP の累積順・丸めの違いによるのか、別の原因かは**調べていない** (範囲外として止めた)。
- **既定は変えていない。**group 32 のモデルで MPP を通すのは `TF_PREFILL_QMM=mpp` のときだけ。group 64 のモデルは変更前と同じく MPP を使う。`TF_PREFILL_QMM=simdgroup` で MPP を外せる。

## 3. 判断を仰ぐこと / 次に渡すもの

- **(b) の合格条件をどう扱うか。**04 の「greedy 出力が同一」は、累積順の違う 2 経路には強すぎる基準かもしれない。ただし閾値は後から動かさない決まり (02 §4-6)。決めるのはユーザー。選択肢は、(1) 原因を調べる、(2) 条件を変えずに G1 を不合格のまま閉じ、tensor ops の線を G2 の単体プローブで判断する、など。
- **コード側で残るもの**: `TF_PREFILL_QMM=mpp|simdgroup`、`[prefill gpu proj …]` のログ、PrefillAttention の PSO 判定、`TsugumiKernelCheck` の mpp-qmm ケース、`bench/m6/kernel_check.sh`。どれも既定の挙動を変えない。
- **G1 の表の他の項目**: PrefillAttention の family 判定は PSO 生成の成否に置き換えた (MBP の関係テスト 21 本は緑)。attention の A/B 口 (G4) には触れていない。
