# 04. Gate — 問い・測り方・合否

**2026-09-24 に全面的に書き直した** ([18](18-RESET-ABSOLUTE-FLOOR.md))。以前の Gate は「G0 を分母にした N 倍」と「tensor ops に載せること」を目的にしていた。そのため、下限から 10〜40 倍離れた段を見落とした。

各 Gate は **証拠ディレクトリと結果文書が揃って初めて閉じる**。合否の閾値はここに先に置き、計測後に動かさない。

## 原則

1. **分母は段ごとの下限** (B1 の下限表)。結果は「下限の何倍か」で書く。前の版との比は補足。
2. **手を入れる段は、(実測 − 下限) の秒数の大きい順。**手段は段を選んでから決める。tensor ops はその手段の 1 つであって、Gate ではない。
3. **測る量を混ぜない** (B0)。起動時の完全性検証とキャッシュの冷たさは、prefill の成分として別に書く。
4. 正しさの確認 (KernelCheck で CPU 参照と一致、greedy の記録) は、どの変更にも付ける。greedy の差は記録し、それを理由に止めない ([12 §2](12-G3-MOE-MATMUL2D.md) の分岐を参照)。

## B0. 測る量を分ける

| 量 | 中身 | 測り方 | 状態 |
| --- | --- | --- | --- |
| (a) 起動と初回検証 | load、層ファイルの SHA-256 (`--verification full-sha256` の layerVerify) | 今の `prefill_bench.sh` のフッタ (`load`、`layerVerify`) | 取れる |
| (b) 同じプロセスでの 2 回目以降の prefill | 層は開いて検証済み。expert キャッシュの状態は決めて書く | **口が無い** (CLI に同じプロンプトを繰り返す手段が無い)。作る | 未着手 |
| (c) 新しいプロセスでの 1 回目の prefill | 今の `prefill_bench.sh` の `prefill` | そのまま | 取れる (10〜14 はすべてこれ) |

**合格**: (b) を測る口ができ、同じ commit で (a)(b)(c) を並べた証拠ディレクトリがある。
**判断を仰ぐこと**: 改善の目標をどの量に置くか (アプリの TTFT に近いのはどれか)。(b) のキャッシュの扱い (空から始めるか、1 回目の後のままか)。

## B1. 段ごとの下限表

段ごとに、演算量 (FLOP) と、読み書きするバイト数の両方から下限の秒数を出す。
下限 = max(FLOP ÷ 演算の天井, バイト ÷ 帯域の天井)。天井には次を使う。

- 演算の天井: MPS の fp16 GEMM 18.61 TFLOP/s ([03 §1](03-FACTS.md))。
- 帯域の天井: 公称 153 GB/s。実測が無いので、B1 でプローブを作って実測に置き換える。

| 段の粒度 | 今の profile で取れるか |
| --- | --- |
| embed、norm、qkv、rope、kvcopy、attn.swa、attn.full、oproj、post | `TF_PREFILL_GPU_PROFILE=2` (膨らむので割合で読む) |
| post の中の router と norm の分離 | **取れない**。分ける |
| shared、routed MoE (GEMM / gelu / 待ち)、tail、head | `moe` は tile stream の待ちを含む。GEMM だけの秒は取れない。分ける |
| ホスト往復 (GPU の空き) | `TF_PREFILL_HOST_PROFILE=1` の `gpuq idle` |

**作るもの**:

- 下限表の文書 (導出。式と、元にした `arch` の値を書く)。
- 帯域のプローブ。
- 上の「取れない」の 2 行を分ける profile。

**合格**: 全段に「下限 s / 実測 s / 倍率」が並び、(実測 − 下限) の大きい順が決まっている。数字は (b) で取る。B0 が済むまでは (c) で取り、そう書く。

## B2. 下限から遠い段から直す

B1 の順に 1 段ずつ直す。1 段ごとに次を満たして閉じる。

- その段の GPU 秒が下限の何倍になったかを、変更前後で書く。
- 正しさ: KernelCheck と greedy。
- 壁時計 ((b) と (c)) を記録する。

**B1 の前の候補** ([18 §1](18-RESET-ABSOLUTE-FLOOR.md)、概算であって順位ではない):

- `post` の router: トークンごとの bf16 GEMV。
- `post` と `tail` の epilogue: 1 トークンに 1 threadgroup。
- `rope`。
- `attn.swa`: 実効 0.6 TFLOP/s。
- 層ごとのホスト往復。

**合格の閾値は B1 の後に段ごとに置く** (計測の前に置き、後で動かさない)。

## 保留

- **Qwen 経路** (旧 G6): Gemma で B0〜B2 の形が回ってから。
- **足したが既定にしていない口**: `TF_PREFILL_MOE=mpp`、`TF_PREFILL_QMM=mpp|mpp64`、`TF_PREFILL_VERIFY_AHEAD[_PARALLEL]`。B1 の下限表に照らして、残すかどうかの判断を仰ぐ。

## 以前の Gate の結果 (索引だけ)

| 旧 Gate | 結果文書 | 18 以降の読み |
| --- | --- | --- |
| G0 基準線 | [10](10-G0-BASELINE.md) | (c) の値。分母には使わない |
| G1 MPP 経路を group 32 で開く | [16](16-G1-MPP-GROUP32.md) | 経路と数値検査は有効 |
| G3 routed MoE を `matmul2d` 化 | [12](12-G3-MOE-MATMUL2D.md) | routed MoE の GPU 秒 2.58 → 1.10 s は有効。下限の何倍かは B1 で |
| 射影と shared を `matmul2d` 化 | [13](13-PROJ-SHARED-MATMUL2D.md) | GPU 秒の縮みは有効。壁時計は (c) |
| G5 層ファイルの検証を先に回す | [14](14-G5-VERIFY-AHEAD.md) | (c) の中で起動時のハッシュを隠したもの。(b) の改善ではない |
| 方針 (tensor ops 化の順序) | [17](17-NAX-NOT-USED.md) | §1 の演算量だけ B1 に引き継ぐ。方針は取り消し |
