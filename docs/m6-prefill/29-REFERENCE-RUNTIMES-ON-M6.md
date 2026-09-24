# 29. 同じ M6 で外部ランタイムの prefill を取った — llama.cpp 2.39 s (tensor API を切ると 4.49 s)、mlx_lm 1.82 s (実測、2026-09-24)

**校正の測定。台帳 §1 は動かない。**この repo のコードは変えていない。
[04 規則 6](04-GATES.md) (他プロジェクトの数字を引かず、この repo のプローブで取り直す) に従い、
同じ M6・同じ 2478 トークンの本文で、llama.cpp と mlx_lm が何秒で prompt を処理するかを取った。
「期待通りか」を比で語らないための絶対秒 ([18](18-RESET-ABSOLUTE-FLOOR.md))。

証拠 (M6、この順に取った):
[`ref-llamacpp-iq3xxs`](../../bench/m6/results/2026-09-24-Mac18-5-ref-llamacpp-iq3xxs/) (n = 6)、
[`ref-llamacpp-iq3xxs-tensor-off`](../../bench/m6/results/2026-09-24-Mac18-5-ref-llamacpp-iq3xxs-tensor-off/) (`GGML_METAL_TENSOR_DISABLE=1`、n = 3)、
[`ref-mlx-3bit`](../../bench/m6/results/2026-09-24-Mac18-5-ref-mlx-3bit/) (n = 6)。
スクリプトは `bench/m6/ref_llamacpp.sh` (88dd7af)、`bench/m6/ref_mlx.sh` (88112df)。

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 88dd7af
llama.cpp 4c5957c / gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf 11416548832 bytes
mlx 0.32.2 mlx_lm 0.31.3 / gemma-4-26b-a4b-it-UD-MLX-3bit (mlx の証拠は commit 88112df)
```

```
llama-completion -m gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf -f prompt.txt -n 1 -c 4096 -b 2048 -ub 2048 -fa on --temp 0 -no-cnv --no-display-prompt -fit off -lv 4
python -m mlx_lm generate --model gemma-4-26b-a4b-it-UD-MLX-3bit --prompt - --max-tokens 1 --temp 0 < prompt.txt
```

`iogpu.wired_limit_mb=14336` (sudo、再起動で 0 に戻る)。既定の 0 では llama.cpp の見積り 13644 MiB に対し空きが 12123 MiB で、
`-fit on` のまま走らせると expert を黙って CPU に移す (試走で確認、その run は捨てた)。

## 0. 結論

1. **llama.cpp (IQ3_XXS、tensor API あり) の prompt eval は 2.39 s (2.385〜2.811、n = 6、2490 トークン)。**
   Tsugumi の既定は台帳 (c) で 3.496 s ([06 §1](06-LEDGER.md))、GPU busy は 2.97〜3.00 s ([27 §1.2](27-B1-FLOOR-TABLE-ON-SWA-MPP.md))。
   相手は起動時の SHA-256 を含まない暖まったプロセスの数字なので、比べる相手は (c) ではなく GPU busy 側。**差は約 0.6 s (導出)。**
2. **tensor API を切った llama.cpp は 4.49 s (4.494〜4.791、n = 3)。**同じバイナリ・同じ prompt で 2.1 s の差 (導出)。
   M6 の Neural Accelerator を llama.cpp が使えていることは、ログの行ではなくこの A/B で確かめた (`-lv 4` でも device の行は出なかった)。
3. **mlx_lm (3-bit) の Prompt は 1.82 s (1.788〜3.828、n = 6、2481 トークン)。**ただし peak memory 13.3 GB で、run ごとに Swapouts が 25k〜64k ページ (0.4〜1.0 GB、導出) 出た。
   16 GB の M6 では MLX のこの構成は収まっておらず、この秒はスワップの起きた run の値である。取り直していない (Swapouts の出る run は繰り返さない)。

品質は見ていない (IQ3_XXS と 3-bit は QAT int4 とは別の量子化)。prefill の演算量はビット幅で変わらず、両ランタイムとも大きい M では fp16 に展開して GEMM に入るので、秒の比較に使う。

## 1. 数字

### 1.1 llama.cpp、tensor API あり (`ref-llamacpp-iq3xxs`、run 順)

| run | prompt eval s | load s | Swapouts (run 中の差、ページ) |
| ---: | ---: | ---: | ---: |
| 1 | 2.811 | 1.322 | 24 |
| 2 | 2.398 | 0.571 | 0 |
| 3 | 2.398 | 0.559 | 0 |
| 4 | 2.385 | 0.557 | 0 |
| 5 | 2.387 | 0.546 | 0 |
| 6 | 2.388 | 0.549 | 0 |

中央値 2.393 s。1 本目はページキャッシュが冷たい (load 1.3 s) 分だけ遅い。

### 1.2 llama.cpp、tensor API なし (`ref-llamacpp-iq3xxs-tensor-off`、run 順)

| run | prompt eval s | load s |
| ---: | ---: | ---: |
| 1 | 4.791 | 5.352 |
| 2 | 4.494 | 0.569 |
| 3 | 4.496 | 0.561 |

中央値 4.494 s。

### 1.3 mlx_lm、3-bit (`ref-mlx-3bit`、run 順)

| run | Prompt tok/s | 秒 (**導出**、2481 ÷ tok/s) | peak GB | Swapouts (run 中の差、ページ) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 648.2 | 3.828 | 13.308 | 63976 |
| 2 | 1339.1 | 1.853 | 13.308 | 29388 |
| 3 | 1387.3 | 1.788 | 13.308 | 32216 |
| 4 | 1385.1 | 1.791 | 13.308 | 24864 |
| 5 | 1341.1 | 1.850 | 13.308 | 31916 |
| 6 | 1381.0 | 1.797 | 13.308 | 33884 |

中央値 1.823 s (tok/s の中央値 1361)。mlx_lm は秒を出さないので tok/s から導出した。1 本目はページキャッシュが冷たい。
Swapouts は `run-N.pre.txt` と `run-N.post.txt` の差。6 本で 216k ページ (3.4 GB、導出) のスワップ書き込みになった。

### 1.4 並べる (すべて同じ M6、同じ本文)

| 実装 | 量 | 秒 | 出所 |
| --- | --- | ---: | --- |
| Tsugumi 既定 (bda6631) | (c) 新プロセス 1 回目、起動時ハッシュ込み | 3.496 | [06 §1](06-LEDGER.md) |
| Tsugumi 既定 (cd8c26b) | GPU busy (ホスト profile) | 2.97〜3.00 | [27 §1.2](27-B1-FLOOR-TABLE-ON-SWA-MPP.md) |
| Tsugumi 段の下限の合計 | 導出 | 0.95 | [27 §1.1](27-B1-FLOOR-TABLE-ON-SWA-MPP.md) |
| llama.cpp、tensor API あり | 暖まったプロセスの prompt eval | 2.393 | §1.1 |
| llama.cpp、tensor API なし | 同上 | 4.494 | §1.2 |
| mlx_lm 3-bit | 暖まったプロセスの Prompt (スワップあり) | 1.823 | §1.3 |

トークン数は Tsugumi 2478、llama.cpp 2490 (bos と turn の印の分、+12)、mlx_lm 2481 (chat template 込み)。

## 2. 分かったこと / 外れたこと

- 「16 GB では参照が成立しない」(凍結文書 §5-3) は、llama.cpp では低ビットの量子化と wired limit の引き上げで外れた。剪定は要らなかった。
  MLX は 3-bit でも peak 13.3 GB で、スワップ無しには回らなかった (§1.3)。
- Tsugumi の GPU busy 3.0 s は、tensor API を切った llama.cpp (4.5 s) より速く、使った llama.cpp (2.4 s) より遅い。
  段の下限の合計 0.95 s との間に約 2 s あり、そのうち llama.cpp が実際に詰めているのは約 0.6 s 分、mlx_lm は約 1.2 s 分 (導出)。
  同じ機械で同じ本文を 1.8 s で処理する実装が存在する。
- llama.cpp は `-fit on` で黙って CPU に落とす。参照を取るときは `-fit off` で落ちるようにしておく。

## 3. 次に渡すもの

- B2 の順序は [27 §1.1](27-B1-FLOOR-TABLE-ON-SWA-MPP.md) のまま。この文書は目標値を置かない (置くと分母になる)。
- 同じ条件で decode は取っていない。取るなら別の文書。
- (b) を測る口 ([04 B0](04-GATES.md)) があれば、参照と同じ量で並べられる。今は GPU busy で代用している。
