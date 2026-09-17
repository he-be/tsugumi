# 04. ランナーの参照一致と、最初の速度

実測: 2026-09-17、M3 Pro 18 GB、macOS 15、`iogpu.wired_limit_mb` = 0 (既定)。[01](01-FEASIBILITY.md) §5 の 4 段 (§3-2)。
表記は [qwen38/01](../qwen38/01-Q2-FIRST-LIGHT.md) と同じ (実測 / 導出 / 未確認)。**速度とメモリの数字はどれも 1 回の実行**で、解釈は書かない。

## 0. 結論

1. **`Qwen38DenseRunner` (Metal、GGUF を直接読む) が CPU 参照と全位置で一致した。**KV f32 で、France 10 位置・Fuji 52 位置の top-1 がすべて一致し、logit の相対誤差は最大 2.2e-5 / 4.3e-5。
   Fuji は 1 トークンずつ・20 トークンずつ・52 トークン一度の 3 通りとも同じ結果 (GDN の WY 形と sgemm の注意を通る) (§2)。
2. **KV Q8_0 (運用点) も top-1 は 52 位置すべて一致**。logit の相対誤差は最大 1.68e-3 で、閾値 1e-3 を超える。位置 0 から 1.4e-3 あり、位置が進んでも増えない。
   参照器の中で Q8_0 と f32 を比べた差は位置 0 で 6.7e-3 (§2-1)。
3. **既定の wired 上限で、重みをそのまま no-copy で参照すると、decode 1 トークンは 3.9 秒 (GPU 時間は 0.36 秒)。重みのビューを常駐セットに入れると 0.105〜0.108 秒になった。**
   ただし常駐セットありでは wired が 13.2〜13.8 GB に達し、スワップアウトが出た。52 トークンの prefill の後には、2 秒で 574 MB 出て見張りが止めた (§3)。常駐セットは `Q27_RESIDENT=1` のときだけ有効。

## 1. 形

`Sources/Tsugumi/Runtime/Qwen38/Qwen38DenseRunner.swift`。`Qwen38Runner` から hc・PLE・indexer・MoE を外し、
`x += mixer(rms(x, attn_norm))`、`x += ffn(rms(x, post_attention_norm))` にした。

- GDN と注意は `qwen38.metal` のカーネルをそのまま使う (注意の各カーネルは KV ヘッド数を引数で受けるので 4 でも同じ)。
  **GDN の出力ノルムのゲートは function constant 1 で SiLU に切り替える** (`q38_gdn_norm_gate`、既定は sigmoid のまま。02 §2)。
- dense の行列積は 03 の `ggml_iq.metal`。FFN (89M 重み) は、最初は `mpsMaxWeights` (64M) を超えて prefill でも直接カーネルを通っていた。§3-1 から上限を 96M にして、逆量子化 + sgemm に載せた。
- 注意は全位置が prefix 全体を見る (indexer が無い)。T < 32 はレーンの 5 パス、T ≥ 32 は KV グループごとの sgemm。
- `token_embd` (IQ2_S、0.38 GiB) は丸ごと載せない。トークンごとに、その行のページだけを no-copy で見て GPU で逆量子化する (`GGMLDenseGEMV.encodeDequant`)。
- 残差の加算は `q38_hc_combine` をストリーム 1 本・係数 1 で使う。
- 層ごとに 1 本のコマンドバッファを待たずに積み、LM head の後で 1 回だけ待つ (ホストが途中で読むものが無い)。
- MTP・チェックポイント・投機の巻き戻しはまだ無い。

## 2. 参照との一致 (実測、`TsugumiKernelCheck --q27-decode`)

参照は 02 の `ref-france` (5 + 6 トークン、10 位置) と `ref-fuji` (53 トークン、52 位置)。

| 文 | KV | 流し方 | top-1 | logit 相対誤差 (最大) |
| --- | --- | --- | --- | ---: |
| France | f32 | 1 トークンずつ | 10/10 | 2.18e-5 |
| Fuji | f32 | 1 トークンずつ | 52/52 | 4.29e-5 |
| Fuji | f32 | 20 ずつ (20 + 20 + 12) | 52/52 | 4.30e-5 |
| Fuji | f32 | 52 一度 | 52/52 | 4.31e-5 |
| Fuji | q8_0 (参照も `--kv-type q8_0`) | 1 トークンずつ | 52/52 | 1.68e-3 |
| Fuji | q8_0 (同) | 52 一度 | 52/52 | 1.68e-3 |

Flash-Next の同じ検査 (48 層) では f32 の最大が 1.7e-6 だった ([qwen38/01](../qwen38/01-Q2-FIRST-LIGHT.md) §4-2)。

### 2-1. KV Q8_0 の誤差 (実測)

- ランナーと参照の差 (位置ごと): 位置 0 で 1.42e-3、52 位置の範囲は 1.5e-4〜1.68e-3。位置が進んでも増えない。
- 参照器の中で `--kv-type q8_0` と f32 を比べた差: 位置 0 で 6.68e-3、中央値 1.63e-3、top-1 は 52/52 同じ。
- Flash-Next では、Q8_0 の段差をまたぐ丸めの揺れとして、閾値を `Q38_LOGIT_TOL` で明示して扱った ([qwen38/23](../qwen38/23-KV-Q8-32K.md) §1)。
  ここでは原因を調べていない。top-1 が一致していることと、上の数字だけを記録する。

## 3. 最初の速度とメモリ (実測、1 回ずつ)

France、1 トークンずつ、KV q8_0。wired の最大は `guarded.sh` の 2 秒ごとの `vm_stat`。開始前の wired は 4.6〜4.7 GB。

| 条件 | 位置 1〜9 の 1 トークン | GPU 時間 | wired の最大 | Swapouts |
| --- | ---: | ---: | ---: | ---: |
| 常駐セットなし | 3.93〜4.03 s | 348〜362 ms | 8.61 GB | 0 |
| 常駐セットあり (`Q27_RESIDENT=1`) | 0.105〜0.108 s | 106〜109 ms | 13.84 GB | +15,776 ページ (246 MB) |

- 常駐セットありの位置 0 は 6.46 s (最初の常駐要求を含む)。
- 常駐セットなしでは、実時間と GPU 時間の差の約 3.5 秒が、コマンドバッファが GPU で走り出す前にある。
- **常駐セットありで Fuji 52 トークンを一度に流した回 (KV q8_0)**: prefill 6.98 s (GPU 4.05 s)、top-1 52/52、最大 7.06e-4。
  終了間際の 2 秒で wired 13.20 GB、ファイルキャッシュ 6.08 → 1.69 GB、**Swapouts +36,748 ページ (574 MB)** で、見張りがプロセスを止めた (結果の出力は済んでいた)。
- 常駐セットなしの prefill (KV f32): 52 一度で 4.30 s (GPU 4.09 s)、20 ずつで 1 チャンク 4.0 s 前後 (GPU 1.8 s)。

01 §2 の見積もり (32K で 12.18 GiB + 作業域、OS とアプリに約 4 GiB) は、開始前の wired 4.6 GB を置いていない。
この機体で重みを常駐させたまま使える条件 (wired 上限・他のプロセスの wired・作業域) は、まだ決めていない。

### 3-1. wired 上限を上げた後 (実測、2026-09-17、`iogpu.wired_limit_mb` = 14336)

- `recommendedMaxWorkingSetSize` は 12.0 → 14.0 GiB に追従した。開始前の wired は 4.63〜4.69 GB のまま (常駐メモリの大きいユーザープロセスは無く、最大 0.34 GB)。
- 注意の sgemm をクエリの小分けにし (`Q27_ATTN_SCORE_MB`、既定 256)、FFN を逆量子化 + sgemm に載せた (`Q27_MPS_MAX_W`、既定 96M) 後も、Fuji 52 トークン一度 (KV f32、常駐あり) は top-1 52/52・最大 4.31e-5。
  1 行ずつの小分け (`Q27_ATTN_SCORE_MB=0`) でも 52/52・最大 4.29e-5。
- その 2 回の Swapouts は +9,760 ページ (152 MB、wired 最大 12.75 GB) と +35,024 ページ (547 MB、60 秒の上限を超えて見張りが停止、出力は済んでいた)。
- **4K の bench** (`--q27-bench prompt-code.tokens --q27-tokens 4096 --q27-chunk 512`、KV q8_0、常駐あり): 最初の 512 トークンは 9.47 s (GPU 6.99 s、54.0 tok/s)。
  その後 wired が 4 秒目 13.49 GB → 10 秒目 17.05 GB、ファイルキャッシュ 2.88 → 0.20 GB、Swapouts 60 秒で +43,648 ページ (682 MB) で、見張りが停止した。
  停止後のスワップ使用量は 2.59 GB / 3.07 GB。

## 4. 再現

```bash
swift build -c release --product TsugumiKernelCheck
B=.build/release/TsugumiKernelCheck; S=scratch/qwen38_27b
Q38_KV_TYPE=f32 Scripts/qwen38/guarded.sh $S/out.txt $B --q27-decode $S/ref-france.log --q27-ref-logits $S/ref-france.logits
Q38_KV_TYPE=f32 Scripts/qwen38/guarded.sh $S/out.txt $B --q27-decode $S/ref-fuji-ok.log --q27-ref-logits $S/ref-fuji.logits --q27-chunk 20
Q38_LOGIT_TOL=5e-3 Scripts/qwen38/guarded.sh $S/out.txt $B --q27-decode $S/ref-fuji-kvq8.log --q27-ref-logits $S/ref-fuji-kvq8.logits
```

`ref-fuji-kvq8` は `reference_forward.py --tokens $(cat $S/fuji.tokens) --new 0 --kv-type q8_0 --dump-logits $S/ref-fuji-kvq8.logits`。
**`Q27_RESIDENT=1` はスワップが出るので、メモリの条件を決めるまで見張りなしで使わない。**

## 5. 次

01 §5 の 5 段 (メモリと速度) は、重みを常駐させる条件を決めてからになる。`iogpu.wired_limit_mb` を上げるのは sudo でユーザーが行う (01 §2-3)。
