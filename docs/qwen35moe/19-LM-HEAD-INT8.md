# 19. INT8 の LM head chain — Phase 2 が閉じた (実測(手元)、2026-08-21)

[04-PHASES.md](04-PHASES.md) 次の一手 **#15**、[03-DESIGN.md](03-DESIGN.md) §2-8。
[17 §5](17-PHASE2-KERNELS.md) で「INT4 の specialization ではなく INT8 の chain」と
分かり、[18](18-MIXED-BITS.md) の #11 が片づくのを待っていた最後のカーネル。

```
swift run -c release TsugumiKernelCheck --qwen             # 検査 39 本
swift run -c release TsugumiKernelCheck --qwen --qwen-bench --qwen-tokens 2048
```

| | |
| --- | --- |
| 書いたもの | `qwen_lm_head_greedy_int8_rows_chunk_raw` と `..._reduce` (`qwen.metal`)、`QwenLMHeadChainInt8` |
| 検査 | **39 本すべて緑** (#15 で 10 本増、うち 4 本は負例) |
| 時間 | 実物の語彙で **1 トークン 4.01〜4.05 ms、133〜135 GB/s** (n=3) |
| 分かったこと | **LM head だけで decode の天井が約 248 tok/s。**帯域の実測値が導出の前提 150 GB/s より **11% 低い** (§4) |

**これで Phase 2 のカーネルは全部書けた。**

---

## 1. 何を書いたか

形は `lm_head_greedy_int4_rows_chunk_raw` (logit.metal) と同じ 2 段:

| 段 | すること | 出力 |
| --- | --- | --- |
| 1 | 最終 RMSNorm → 語彙 1 行 = 1 SIMD group で INT8 affine の内積 → threadgroup (8 行) ごとの argmax | `summaries [row_groups, 2]` |
| 2 | summaries を畳む | `token` 1 個 |

**語彙幅の logit をどこにも書かない**のがこの chain の値打ちで、そこは INT4 版と
同じ。行の歩き方は既存の `dequant_int8_gemv_simd` (dequant_int8.metal) を写した
— 1 SIMD group = 1 行、32 レーン × 2 バイトの 64 要素ステップ。

**INT4 から写すときに変わるのは 2 か所だけで、どちらも静かに壊れる:**

1. **行の刻みが `N/2` ではなく `N`。**ニブルの幾何を残すと 2 行に 1 行しか読まない
2. **零点が本物のデータ。**4-bit の `sym` は `bias == -8·scale` を仮定できるが
   ([13 §4-3](13-PHASE1-REPACK.md))、INT8 の group はそうではない。
   `b·Σx` の項を落とせない

置き場所は `Metal/Qwen/qwen.metal` (既存 `.metal` は 1 行も触っていない)。
モジュールは `qwen` のままなので safe math でコンパイルされるが、この chain は
超越関数を 1 つも使わないので [17 §3](17-PHASE2-KERNELS.md) の代償は掛からない。

## 2. 検査 10 本

立て方は [15](15-PHASE2-GDN.md) / [17](17-PHASE2-KERNELS.md) と同じ (合成入力 +
CPU float32 の床 + CPU double の真値)。ただし**採点するものが違う**: chain は
logit を書かないので、読むのは

- `summaries` — 8 行ごとの最大 logit。**採点した語彙行を全部覆う** FP32 の配列
- `token` — それを畳んだ argmax
- `normalizedHidden` — 2 段の継ぎ目 (どちらの段が動いたか言えるように)

| 検査 | 結果 |
| --- | ---: |
| summaries 1,024 本 (対 double) | **1.63e-07** (float32 の床 8.26e-06) |
| 前段の RMSNorm | **0** (FP16 に落ちる時点で床の下) |
| `dequant_int8_gemv_simd` の logit 8,192 行 (対 double) | 3.31e-04 |
| chain の summaries 対 `dequant_int8_gemv_simd` | 3.79e-04 |
| token (argmax) | 一致 |

**GPU が CPU float32 の床より 50 倍良い**のは、レーン方向の `simd_sum` が木で
畳むのに対し、参照の float32 が逐次に足しているため。床は「桁の落ち方の目安」で
あって下限ではない。

**二人目の証人を 2 通りに使った。**`dequant_int8_gemv_simd` は Gemma 4 の router と
shared expert がずっと使ってきた INT8 の読み方で、logit を全部書き出す。
まず**それが同じ参照に乗る**ことを見て (参照の算式が INT8 の規約どおりだという証拠)、
次にその logit を chain の summaries と**直に**突き合わせた (chain が同じ表を
同じように読んでいるという証拠)。片方だけでは「両方が同じように間違っている」場合を
排除できない。

### 負例 4 本

| 負例 | 相対誤差 |
| --- | ---: |
| 行の刻みを `N/2` にする (INT4 の幾何) | 0.77 |
| バイトを signed で読む (MLX affine は 0..255 の unsigned) | 1.85 |
| affine の bias 項 `b·Σx` を落とす | 0.63 |
| 未学習の末尾まで採点する | トークンが変わる |

正例の許容 4e-3 に対していちばん小さい負例でも 0.63 — **157 倍**離れている。

### 未学習の末尾 243 行

`vocab_size` は 248,320、tokenizer の語彙は BPE 248,044 + added 33 = **248,077**
([10 §3](10-MLX4BIT-AUDIT.md))。差の 243 行は学習されていない。
**`vocab` に 248,077 を渡すだけでよく、マスクのコードは要らない** —
[17 §5](17-PHASE2-KERNELS.md) の読みどおりだった。

これは算術ではなく**引数**の間違いなので、負例は参照側ではなく**入力側**に
仕掛けた: 末尾の行に必ず勝つ大きさの重みを置き、`vocab=248,077` 相当では
chain がそれを選ばず、`vocab=総行数` では選ぶことを見た。引数を取り違えたときに
だけ落ちる検査になっている。

## 3. 時間 — 1 トークン 4.0 ms、133〜135 GB/s

```
qwen_lm_head_int8   T=1   4.008 / 4.039 / 4.047 ms   (3 回、実物の語彙)
```

読むバイトは **540 MB** (重み 508 MB + scale/bias 32 MB)。マイクロベンチなので
[bench.sh](../../bench.sh) の作法 (temp / クールダウン) の対象ではない。

**89% of peak** (M3 Pro の公称 150 GB/s に対して)。ここは完全に帯域律速で、
ALU 側の工夫 (`Σx` を行ごとに再計算せず 1 回で済ませる `Int4RowGroupSums` 相当) を
足しても伸びしろは 11% しかない。**やらない。**

## 4. ★ 数字が 2 つ動く

### 4-1. LM head だけで decode の天井が約 248 tok/s

4.0 ms は**1 トークンあたり 1 回**掛かる。他が全部ゼロでも 248 tok/s である。
[01 §5-5](01-MODEL.md) のバイト予算は Ornith の decode 全体を 1.89 GB / トークンと
導出しており、そのうち 540 MB (29%) がこの chain だと分かったことになる。
**予算の内訳に初めて実測が 1 つ入った。**

### 4-2. 導出が使っている 150 GB/s は 11% 楽観

[01 §5-5](01-MODEL.md) は「M3 Pro の帯域を 150 GB/s とすると天井は 79 tok/s」と
書いている。**実物と同じ大きさの連続読みで出たのは 133〜135 GB/s** で、これは
理想的な条件 (1 本の巨大な連続読み、分岐なし、再利用なし) での値である。
同じ計算を 134 GB/s でやり直すと **1.89 GB / 134 GB/s = 14.1 ms → 71 tok/s**。
予算表の数字は動かさない (あれは導出として正しい) が、**天井は 79 ではなく
71 tok/s の側で読むべき**である。

なお `lm_head` が 8-bit であること自体は選択の結果 ([16 §1](16-QUALITY.md) で
`oQ4e-g64` を本線に決めた対価)。4-bit に打ち直せば 540 → 286 MB で 2 ms 縮むが、
**本線の品質判断をやり直す話**なので、ここでは数字だけ置く。

## 5. この Phase が動かした結論

| 対象 | 更新 |
| --- | --- |
| [04](04-PHASES.md) Phase 2 | **完了。**カーネルは全部書けた |
| [04](04-PHASES.md) 次の一手 #15 | **完了** |
| [03 §2-8](03-DESIGN.md) | 「INT8 の chain を書く」→ **書いた。**`vocab` に 248,077 を渡すだけでよいことも確かめた |
| [01 §5-5](01-MODEL.md) の天井 | 79 tok/s は 150 GB/s 前提。**実測の 134 GB/s なら 71 tok/s** (§4-2) |
| 残り | **#14 (Phase 3 の結線) が次。**カーネル側の宿題はもう無い |

## 6. まだ無いもの

| | 何 | いつ |
| --- | --- | --- |
| a | **ブロック版** (`encodeGreedyDecodeBlock` 相当)。MTP の verify は k 行を一度に採点するので、表を 1 回だけ読む形が要る ([17](17-PHASE2-KERNELS.md) と同じ理由) | Phase 7 |
| b | **サンプリング経路**。`temp > 0` は logit を全部要るので `DequantInt8GEMV` をそのまま使う (M=248,077 / N=2048 で通ることは §2 で確認済み)。specialization は未追加 | Phase 3 |
| c | **実物の重みでの検査。**ここまで全部合成入力 | Phase 3 の結線後 |
