# 21. Phase 4 の prefill — チャンク経由でも同じトークンが出た (実測(手元)、2026-08-21)

[04-PHASES.md](04-PHASES.md) 次の一手 **#17** と **#18**。#17 (64 トークンの参照の
取り直し) が **Phase 3 を閉じ**、その参照がそのまま Phase 4 の物差しになった。

```
swift run -c release TsugumiKernelCheck \
  --qwen-prefill scratch/ornith-oq4e-g64.moepack \
  --qwen-decode-fixture scratch/qwen35/decode-fixture-55.json
```

| | |
| --- | --- |
| 結果 | **プロンプトを T 行の経路に通しても、55 トークンすべてが参照と一致。**チャンク幅 512 (1 チャンク) と 8 (3 チャンク) の両方で同じ (§4)。**ただし幅 8 の腕が実際に 3 チャンクに割れたのは 2026-08-22 の修正後である** ([35](35-PREFILL-CHUNK-WIDTH.md)) |
| 参照 | **55 トークン。**外から止めた 41 本ではなく、**モデルが `<\|im_end\|>` を出して自分で止まった**ところまで (§1) |
| 書いたもの | **INT8 の QMM** (`qwen_int8_qmm_*`、§2)、T 行版の小物 3 本、`QwenPrefill.swift` (§3) |
| 検査 | `--qwen` が **39 → 57 本**、`--qwen-prefill` の正例 2 + 負例 5。`swift test --no-parallel` は **1,282 件すべて緑** |
| Gemma | **既定の 69 本は緑のまま。**`prefill.metal` に関数定数を 1 つ足したが、Gemma はそれを定義しないので同じコードが出る (§2-2) |
| 数字 | 合成 2048 トークンの prefill が **チャンク 2048 で 5.5 ms/トークン** (§5)。**運用値ではない** |

---

## 1. #17 — 参照は 55 トークンで、そこで終わっていた

[14 §6](14-REFERENCE.md) の生成は 41 トークンで外から止めたもので、
[20 §8](20-PHASE3-DECODE.md) はその続きを取り直すことを宿題にしていた。
同じプロンプトで `--max-new 64` を回し直した (46.6 分、47.5 s/トークン):

```
生成 55 トークン / 2798.6s
<think>
The user is asking in Japanese: "Where is the capital of Japan? ..."
The capital of Japan is Tokyo (東京).
I should answer in one sentence in Japanese as requested.
</think>

日本の首都は東京です。<|im_end|>
```

**64 には届かないが、届かない理由が変わった。**41 本のときは外から止めたから
だったが、今度は **`<|im_end|>` (248046) をモデルが出した**からで、これは
`generation_config.json` の停止トークンである ([04](04-PHASES.md) Phase 5)。
**この生成には 56 番目のトークンが存在しない。**

したがって Phase 3 の出口条件「64 トークン一致」は、**その生成の全体と一致**という
形で満たされている。`--qwen-decode` を新しい fixture に対して回すと:

```
PASS  55 tokens, every one equal to the float32 reference
  shared-gate-bf16 / shared-gate-skipped / routed-gelu /
  uncompacted-query / forget-state — 負例 5 本とも 16 トークン以内に離れる
```

fixture は `scratch/qwen35/decode-fixture-55.json`
(`Scripts/qwen35/decode_fixture.py` がログから起こす)。

## 2. INT8 の QMM — prefill の最後の穴

### 2-1. なぜ要ったか

prefill には T 行を一度に通す GEMM が要る。本リポジトリは持っている
(`prefill_int4_qmm_simdgroup_f16`) が、**INT4 しか読めない。**本線
`oQ4e-g64` で prefill が通る道のうち

| ロール | 幅 |
| --- | --- |
| `linear_attn.{in_proj_z, in_proj_a, in_proj_b, out_proj}` | 30 層すべて 8-bit |
| `mlp.shared_expert.{gate,up,down}_proj` / `shared_expert_gate` | 40 層すべて 8-bit |
| `self_attn.{q,k,v,o}_proj` / `in_proj_qkv` | 層ごとに 4/8 が混ざる |

が 8-bit である ([18 §3](18-MIXED-BITS.md))。LM head ([19](19-LM-HEAD-INT8.md)) と
埋め込みのときと同じ理屈で、**ニブルの展開は行の幾何であって引数ではない**ので、
幅を渡して済ませる道は無い。

書いたのは 2 本。幾何は INT4 版を 1 行ずつ写し、**違うのは 2 か所だけ**:
行の刻みが `K/2` ではなく `K`、8 幅のチャンクを 4 バイト読んでニブルを開く
のではなく 8 バイトそのまま。置き場所は `Metal/Qwen/qwen.metal`
(Gemma の `prefill.metal` は触らない)。

| カーネル | 形 |
| --- | --- |
| `qwen_int8_qmm_f16_block` | 1 スレッドが (token, row) の K 縮約を持つ。重みは **FP32 のまま**積む |
| `qwen_int8_qmm_simdgroup_f16` | 64x64 タイル / 4 simdgroup、逆量子化した重みを **FP16** で threadgroup に置く |

scalar 版を残したのは **カーネルのバグと staging の丸めを分けるため**で、
`TF_QWEN_QMM=scalar` と `forcedPath` の両方から選べる。`N=1`
(shared expert のゲート) は tiled 版がタイルの 64 列中 63 列を捨てるので
scalar に落とす。

### 2-2. routed expert の活性化 — prefill 側の SiLU

[20 §8](20-PHASE3-DECODE.md) が残した宿題。decode では
`FC_MOE_ACT_SILU` (moe.metal、関数定数 4) で分けたが、prefill 側の 3 か所
(`..._batched_phase1` / `prefill_moe_gate_up_gelu_mul` /
`prefill_moe_rows_gate_up_act`) は gelu を焼いたままだった。

同じ形で片づけた: `FC_PREFILL_MOE_ACT_SILU` (関数定数 77) と
`prefill_moe_gate_activation`。**Gemma は定数を定義しない**ので、その PSO は
今までと同じコードを吐く — 既定の検査 69 本が緑であること、
`swift test` の 1,282 件が緑であることで見ている。

### 2-3. 検査 18 本 (`--qwen` は 39 → 57)

立て方は [17](17-PHASE2-KERNELS.md) / [19](19-LM-HEAD-INT8.md) と同じ
(合成入力 + CPU float32 の床 + CPU double の真値)。**このカーネルだけ経路が
2 本ある**ので、見るものが 1 つ増える:

| 形 | 経路 | 対 double | scalar 版 | tiled 対 scalar |
| --- | --- | ---: | ---: | ---: |
| T=72 N=192 K=2048 | tiled | 7.27e-04 | 1.82e-04 | 7.27e-04 |
| T=64 N=128 K=2048 | tiled | 7.76e-04 | 9.71e-05 | 7.76e-04 |
| **T=100 N=130** K=2048 | tiled | 6.61e-04 | 1.65e-04 | 6.61e-04 |
| T=96 **N=1** K=2048 | scalar | 6.22e-05 | — | — |

許容は 4e-3 (他の FP16 カーネルと同じ)。**tiled と scalar の差は staging の
FP16 丸めだけ**で、その値が対 double の誤差とちょうど一致している
(scalar 側は 1 桁小さい) — つまりこの経路の誤差は**丸めであってバグではない**。

**T も N も 64 の倍数でない形を必ず 1 つ入れる。**タイルの端は写し間違えても
中央が正しければ「だいたい合う」ので、実物の形 (2048 / 4096、どれも 64 の倍数)
だけを見ていると通ってしまう。

**経路そのものも採点する。**tiled のつもりで scalar が走っていれば数字は
どちらも正しいので、それでは気づけない。

負例 4 本。3 本は LM head と同じ ([19 §2](19-LM-HEAD-INT8.md))、1 本はタイル特有:

| 負例 | 相対誤差 |
| --- | ---: |
| 行の刻みが `K/2` (INT4 の幾何) | 1.44 |
| バイトを signed で読む | 1.20 |
| affine の bias 項を落とす | 0.78 |
| **重みを `[K, N]` として読む** (`Bs` を K-major に置く段の取り違え) | 1.47 |

## 3. `QwenPrefill.swift` — T 行の経路

`QwenForwardRunner` の extension として書いた。**decode と同じく直列**
(1 段 = コマンドバッファ 1 本、待ってから次を積む)。理由も同じで、Phase 4 の
出口条件は「prefill と decode が同じトークンを出す」であって速さではなく、
食い違ったときに段を名指しできる形が要る。`RealForwardRunner` が持っている
3 本パイプライン・タイルスケジューラ・rows 経路の A/B は Gemma 4 に対する
**測定値**なので、写すのは Phase 6 の仕事にする。

| 段 | 使ったもの |
| --- | --- |
| 埋め込み | `qwen_embed_lookup_int8_block` (**新**。表が 8-bit) |
| norm | `prefill_rmsnorm_bf16w_block` (Gemma の、無改造) |
| 4-bit の射影 | `PrefillInt4QMM` (Gemma の、無改造) |
| 8-bit の射影 | `QwenPrefillInt8QMM` (**新**、§2) |
| 線形注意 | `qwen_delta_*` と `qwen_delta_rule` を T>1 で (Phase 2 のまま) |
| full attention | `qwen_qkv_epilogue` → `qwen_query_compact` (**新**) → `attention_prefill_causal_qblock_d256` |
| router | `prefill_router_gemma4*_block` に**単位ベクトル**を渡す ([20 §4](20-PHASE3-DECODE.md) と同じ) |
| routed expert | `prefill_grouped_routed_moe_batched_*` (SiLU は関数定数) |
| shared expert | INT8 QMM 4 本 + `qwen_silu_mul` + `qwen_moe_shared_gate_logit_block` (**新**) |

新しいカーネルは **4 本**で、うち 3 本は decode 版を T 行にしただけ
(`qwen_embed_lookup_int8_block` / `qwen_moe_shared_gate_logit_block` /
`qwen_query_compact`)。q の詰め直しを blit ではなくカーネルにしたのは数の問題で、
decode の 16 本が T=2048 では 32,768 本になる ([20 §3](20-PHASE3-DECODE.md))。
**ゲートの持ち回り方は変えていない** — `qwen_attn_output_gate` は今までどおり
元の 2 倍幅バッファから読む。

### 3-1. チャンクをまたぐもの 3 つ

| 持ち越すもの | どうしたか |
| --- | --- |
| 再帰状態 `S` | `qwen_delta_rule` の in/out は**別名でよい** (1 threadgroup が自分の切れ端を最初に読んで最後に書く)。decode と同じ |
| conv の窓 | **別名にできない。**`qwen_delta_qkv_prepare` はトークンブロックが 2 つ以上あると、状態を書く group と読む group が別になる。チャンク用のスクラッチに書かせて、同じコマンドバッファの中で blit で戻す |
| K/V | リングが無いので**チャンクの位置は連続スロット**で、1 トークンの刻みがちょうど `numFullKVHeads * fullHeadDim`。射影の出力をキャッシュに**直接書く** (詰め替え無し) |

### 3-2. routed expert は per-pair GEMV の方を選んだ

`prefill_moe_gemm_int4` (タイル版) ではなく
`prefill_grouped_routed_moe_batched_*` (ペアごとの GEMV)。整列の前提も
バッチの計画も要らないので、**router と答えの間で動く部品が一番少ない**形から
始めた。どちらが速いかは [05 §1-2](05-RISKS.md) の問いで、測らないと分からない。

タイルは 1 つずつ**取り終えてから積み、積み終えてから次を計画する。**
これは片づけではなく必要で、エキスパートキャッシュは次のタイルに同じスロットを
渡し得る (`RealForwardRunner` が `avoidingSlots` とスロットの寿命でやっていること)。
直列ならどちらも要らず、その代わりを壁時計で払う (§5)。

## 4. 検査 — 一致 55 本 × 2 つのチャンク幅、負例 5 本

```
PASS  55 tokens, every one equal to the float32 reference     ← チャンク 512 (1 チャンク)
PASS  chunk 8 (3 chunks) — the same 55 tokens

  negative controls — each must disagree within 16 tokens:
    PASS  shared-gate-bf16     diverged at step 0 (198, not 248068)
    PASS  shared-gate-skipped  diverged at step 0 (75, not 248068)
    PASS  routed-gelu          diverged at step 8 (328, not 25)
    PASS  uncompacted-query    diverged at step 6 (728, not 303)
    PASS  forget-state         diverged at step 1 (95761, not 198)
```

> **2026-08-22 の訂正: この腕は空振りしていた。**`QwenPrefill` が要求幅ではなく
> scratch の幅で切っていたので、先に走る幅 512 の腕が作った 19 幅の scratch を
> 幅 8 の腕が使い回し、**また 1 チャンクで終わっていた**。直した後に走らせると
> `PASS chunk 8 (3 chunks)` で**通る** — 隠れていた不一致は無く、本節の結論は
> 動かない。動くのは「確かめた」の中身の方である ([35](35-PREFILL-CHUNK-WIDTH.md))。

**チャンク 8 を回すのが本題の半分である。**プロンプトが 19 トークンなので
既定の 512 では 1 チャンクで終わり、それでは §3-1 の 3 つが**一度も持ち越されない。**
幅 8 なら 3 チャンクに割れて、conv の窓・再帰状態・K/V のカーソルが
チャンク境界を越える。**チャンク幅はモデルから見えてはいけない**というのが
この 2 行の主張で、片方だけでは主張になっていない。

負例は decode と同じ 5 本を prefill 経路に通したもの。**`shared-gate-bf16` だけ
意味が違う**: prefill には BF16 の融合カーネルの兄弟が無いので、「ゲートを
間違った幅で読む」を 8-bit の行を 4-bit として読む形にした。名前は decode
のものを引き継いでいる。

`uncompacted-query` が decode の step 7 ではなく **step 6** で、`forget-state` が
step 0 ではなく **step 1** で離れるのは、プロンプトの通り方が変わったから
(19 トークンが 1 回で流れる)。**どちらも「離れる」ことに変わりはない。**

## 5. 数字 (**運用値ではない**)

`--qwen-prefill-bench`。合成プロンプト (語彙に散らした ID)、スロット 32、n=3。
**この経路は直列で、タイルごとに GPU を止める。**router の当たり方も本物の
文章のそれではない。

| トークン | チャンク | 1 回目 | 2 回目 | 3 回目 | 1 トークン (3 回目) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 512 | 10,515 ms | 4,374 | 4,351 | **8.50 ms** |
| 2048 | 512 | 24,711 ms | 35,755 | 36,456 | 17.80 ms |
| 2048 | 1024 | 19,584 ms | 14,004 | 14,154 | 6.91 ms |
| 2048 | 2048 | 17,150 ms | 11,265 | 11,338 | **5.54 ms** |

**説明が付いていない振る舞いが 1 つある:** 2048 トークン / チャンク 512 だけ、
**1 回目より 2・3 回目が遅い** (12.1 → 17.5 ms/トークン)。他の 3 行は逆
(冷たいキャッシュの 1 回目が遅い) である。数字だけ置く。

> **[24 §4](24-PREFILL-MOE-PATH.md) が説明し、この表を 1 行訂正した。**
> 上の 4 行は**クールダウン無し**で取ったもので、増えていたのは **GPU 時間**
> (8.6 → 22.7 秒)、**4 秒空けるだけで消える**。作法どおり空けると
> 2048 トークン / チャンク 512 は **19,318 ms / 1 トークン 9.43 ms** で、
> 17.80 ms ではない。**他の 3 行は空けても同じ**なので、訂正はこの 1 行だけ。

比較対象として、[20 §6](20-PHASE3-DECODE.md) の decode は 1 トークン 94.7 ms
だった。**19 トークンのプロンプトでは prefill 経路の効きは見えない** — 上の表は
そこを合成入力で埋めたものである。

## 6. この Phase が動かした結論

| 対象 | 更新 |
| --- | --- |
| [04](04-PHASES.md) Phase 3 | **閉じた。**参照の生成 55 本すべてと一致し、55 本目は `<\|im_end\|>` (§1) |
| [04](04-PHASES.md) Phase 4 | **出口条件の 1 つ目 (prefill 経由の greedy が Phase 3 と一致) が通った** (§4)。時間の条件は [17 §4-2](17-PHASE2-KERNELS.md) の #16 と一緒に扱う |
| [04](04-PHASES.md) 次の一手 #17 / #18 | **完了** |
| [20 §8](20-PHASE3-DECODE.md) 「`prefill.metal` も gelu を焼いている」 | **片づいた** (§2-2) |
| [03 §4-1](03-DESIGN.md) 「そのまま使える」表 | prefill 側も**幅がその手前で効く**。INT4 の QMM は 8-bit を読めない (§2-1) |
| [05 §1-2](05-RISKS.md) prefill の GEMM 占有率 | ~~まだ測っていない~~ → **測った** ([24 §3-1](24-PREFILL-MOE-PATH.md))。半端ブロックは実在する (タイル版の GPU 時間だけがチャンク幅で動く) が、**それでもタイル版が速い** |

## 7. 残した宿題

- **チャンク幅の運用点。**§5 は合成入力の n=3 で、`bench.sh` の作法 (temp /
  クールダウン) の対象外。**既定は変えない** — 判断はユーザー ([04](04-PHASES.md) Phase 6)
- ~~**2048/512 の逆転**が説明できていない (§5)~~ → **片づいた**
  ([24 §4](24-PREFILL-MOE-PATH.md))
- ~~**routed expert のタイル版を通していない** (§3-2)~~ → **通した。4 通りとも
  そちらが速い** ([24 §3](24-PREFILL-MOE-PATH.md))
- **直列のまま。**タイルの取得と GPU が重なっていない。Phase 6
- **speculative / server / tokenizer は依然そのまま** ([20 §8](20-PHASE3-DECODE.md))
