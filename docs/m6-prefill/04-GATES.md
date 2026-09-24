# 04. Gate — 問い・測り方・合否

**2026-09-24 に全面的に書き直した** ([18](18-RESET-ABSOLUTE-FLOOR.md))。以前の Gate は「G0 を分母にした N 倍」と「tensor ops に載せること」を目的にしていた。
比で進めた結果、既定の経路は基準線から一度も測り直さないまま、`TF_` の裏の数字で進捗を語っていた。

各 Gate は **証拠ディレクトリと結果文書が揃って初めて閉じる**。合否の閾値はここに先に置き、計測後に動かさない。

## 原則

1. **進捗は [06](06-LEDGER.md) の既定行の秒だけ。**既定のバイナリ・固定コマンド・新しいプロセスの 1 回目の prefill 壁時計。スクリプトが証拠ディレクトリから生成した行しか載らない。
   倍率・%・pp・「下限の何倍か」は進捗として書かない。変更の効果は秒の差で書く (「変更前 X s、変更後 Y s、差 Z s」)。
2. **`TF_` 付きの構成は候補。**候補の秒は結果文書と台帳の §2 に載るが、進捗ではない。既定に入れる判断はユーザー (B3)。
3. **手を入れる段は、(実測 − 下限) の秒数の大きい順。**下限表 (B1) は段を選ぶ道具であって、進捗の物差しではない。手段は段を選んでから決める。tensor ops はその手段の 1 つであって、Gate ではない。
4. **測る量を混ぜない** (B0)。起動時の完全性検証とキャッシュの冷たさは、prefill の成分として別に書く。
5. 正しさの確認 (KernelCheck で CPU 参照と一致、greedy の記録) は、どの変更にも付ける。greedy の差は記録し、それを理由に止めない ([12 §2](12-G3-MOE-MATMUL2D.md) の分岐を参照)。

## B0. 測る量を分ける

| 量 | 中身 | 測り方 | 状態 |
| --- | --- | --- | --- |
| (a) 起動と初回検証 | load、層ファイルの SHA-256 (`--verification full-sha256` の layerVerify) | 今の `prefill_bench.sh` のフッタ (`load`、`layerVerify`) | 取れる |
| (b) 同じプロセスでの 2 回目以降の prefill | 層は開いて検証済み。expert キャッシュの状態は決めて書く | **口が無い** (CLI に同じプロンプトを繰り返す手段が無い)。CLI の変更が要る | 判断待ち |
| (c) 新しいプロセスでの 1 回目の prefill | 今の `prefill_bench.sh` の `prefill` | そのまま。**台帳 ([06](06-LEDGER.md)) の量** | 取れる (10〜14 はすべてこれ) |

**合格**: (b) を測る口ができ、同じ commit で (a)(b)(c) を並べた証拠ディレクトリがある。(b) の中央値は台帳の 2 列目になる。
**判断を仰ぐこと**: (b) の口を CLI に足してよいか (同じプロンプトを N 回 prefill し、回ごとにフッタを出す)。
(b) のキャッシュの扱い。提案は「1 回目の後のまま、hit% をフッタに出す」。アプリの 2 ターン目がそれに近い。prompt cache は測定では切る (当たると prefill が 0 になる)。

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

**合格**: 全段に「下限 s / 実測 s / 差 s」が並び、差の大きい順が決まっている。倍率の列は作らない (下限は導出なので、下限を動かせば倍率も動く)。数字は (b) で取る。B0 が済むまでは (c) で取り、そう書く。

## B2. 下限から遠い段から直す

B1 の順に 1 段ずつ直す。1 段ごとに次を満たして閉じる。

- その段の GPU 秒を「変更前 / 変更後 / 下限」の秒で書く。
- 正しさ: KernelCheck と greedy。
- 壁時計 ((b) と (c)) を記録する。`TF_` で切り替える変更なら、台帳の §2 に候補として載る。既定を変えたなら `ledger-*` を取り、§1 の行で示す。

**B1 の前の候補** ([18 §1](18-RESET-ABSOLUTE-FLOOR.md)、概算であって順位ではない):

- `post` の router: トークンごとの bf16 GEMV。
- `post` と `tail` の epilogue: 1 トークンに 1 threadgroup。
- `rope`。
- `attn.swa`: 実効 0.6 TFLOP/s。
- 層ごとのホスト往復。

**合格の閾値は B1 の後に段ごとに置く** (計測の前に置き、後で動かさない)。閾値は秒で置く。

## B3. 候補を既定に入れる

10〜19 で足した口は、すべて既定に入った (79eea8f、5decec4)。台帳 ([06 §1](06-LEDGER.md)) の最新行は `ledger-gpu3-default` 4.584 s。

| 候補 | 結果文書 | 候補の (c)、秒 (台帳 §2) | 既定にするときの問い |
| --- | --- | --- | --- |
| ~~`TF_PREFILL_MOE=mpp`~~ | [12](12-G3-MOE-MATMUL2D.md)、[21](21-GPU-CANDIDATES-ON-VERIFY-AHEAD.md)、[22](22-GPU-CANDIDATES-GREEDY-AND-FIRST-RUN.md) | 新しい既定の上で単独 5.692 (3075b56) | **既定に入れた** (5decec4、[23](23-GPU-CANDIDATES-DEFAULT.md))。戻すのは `TF_PREFILL_MOE=simdgroup` |
| ~~`TF_PREFILL_QMM=mpp64`~~ | [13](13-PROJ-SHARED-MATMUL2D.md)、[21](21-GPU-CANDIDATES-ON-VERIFY-AHEAD.md) | 新しい既定の上で単独 6.102、mpp と併用 4.636 (3075b56) | **既定に入れた** (5decec4、[23](23-GPU-CANDIDATES-DEFAULT.md))。戻すのは `TF_PREFILL_QMM=simdgroup` |
| ~~`TF_PREFILL_ATTN=sgmm`~~ | [19](19-SWA-ATTN-SGMM-GQA.md)、[21](21-GPU-CANDIDATES-ON-VERIFY-AHEAD.md) | 新しい既定の上で単独 6.776、3 つ全部 4.167 (3075b56) | **既定に入れた** (5decec4、[23](23-GPU-CANDIDATES-DEFAULT.md))。戻すのは `TF_PREFILL_ATTN=qblock`。peak_gb +0.22 GB |
| ~~`TF_PREFILL_VERIFY_AHEAD=2` + `_PARALLEL=1`~~ | [14](14-G5-VERIFY-AHEAD.md)、[20](20-VERIFY-AHEAD-ALONE.md) | 単独で 7.229 (f603ade) | **既定に入れた** (79eea8f、`ledger-verify-ahead` 7.184 s)。Swapouts 0、decode と greedy は同じ |

**手順**: ユーザーが既定に入れると決めた候補ごとに、既定を変えた commit で `ledger-<候補名>` を n = 6 で取る。
**合格**: 台帳 §1 に新しい行があり、その中央値が候補の (c) のレンジに入っている。入っていなければ、既定にした結果として何かが違う (書き換えのミス、環境の違い)。理由が分かるまで次に進まない。

## 保留

- **Qwen 経路** (旧 G6): Gemma で B0〜B3 の形が回ってから。

## 以前の Gate の結果 (索引だけ)

| 旧 Gate | 結果文書 | 18 以降の読み |
| --- | --- | --- |
| G0 基準線 | [10](10-G0-BASELINE.md) | (c) の値。台帳 §1 の唯一の行。分母には使わない |
| G1 MPP 経路を group 32 で開く | [16](16-G1-MPP-GROUP32.md) | 経路と数値検査は有効 |
| G3 routed MoE を `matmul2d` 化 | [12](12-G3-MOE-MATMUL2D.md) | routed MoE の GPU 秒 2.58 → 1.10 s は有効。候補 (B3)。文中の倍率は読まない |
| 射影と shared を `matmul2d` 化 | [13](13-PROJ-SHARED-MATMUL2D.md) | GPU 秒の縮みは有効。壁時計は (c)。候補 (B3) |
| G5 層ファイルの検証を先に回す | [14](14-G5-VERIFY-AHEAD.md) | (c) の中で起動時のハッシュを隠したもの。(b) の改善ではない。候補 (B3)。文中の倍率は読まない |
| 方針 (tensor ops 化の順序) | [17](17-NAX-NOT-USED.md) | §1 の演算量だけ B1 に引き継ぐ。方針は取り消し |
