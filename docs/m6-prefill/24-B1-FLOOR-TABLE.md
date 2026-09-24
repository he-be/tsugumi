# 24. B1 の下限表 — 既定 (6791e58) の上の段の秒と、M6 の帯域の実測 (2026-09-24)

**候補ではなく既定の測定。台帳 §1 は動かない** (変えたのは profile の level 2 だけ。進捗の行は [06 §1](06-LEDGER.md) の `ledger-gpu3-default` 4.584 s のまま)。
[04 B1](04-GATES.md) の材料を揃えた。**B1 は §1.3 で閉じた。**数字は (c) (新しいプロセスの 1 回目) で取った。B0 の (b) の口が無いため。

証拠 (M6、この順に取った):
[`b1-profile-detail`](../../bench/m6/results/2026-09-24-Mac18-5-b1-profile-detail/) (`TF_PREFILL_GPU_PROFILE=2`)、
[`b1-profile`](../../bench/m6/results/2026-09-24-Mac18-5-b1-profile/) (`=1`)、
[`b1-host`](../../bench/m6/results/2026-09-24-Mac18-5-b1-host/) (`TF_PREFILL_HOST_PROFILE=1`)、各 n = 3。
[`gpu_bw_probe`](../../bench/m6/results/2026-09-24-Mac18-5-gpu_bw_probe/) (`swift bench/m6/gpu_bw_probe.swift 1024`)。
分けた行 (§1.3) は 61619a9 で level 2 に `post.norm` / `post.router` と `moe` の gate/up・gelu・down を足してから取った:
[`b1s-profile-detail`](../../bench/m6/results/2026-09-24-Mac18-5-b1s-profile-detail/) (`=2`)、[`b1s-profile`](../../bench/m6/results/2026-09-24-Mac18-5-b1s-profile/) (`=1`)、各 n = 3。

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 6791e58 / minos 15.0
(帯域プローブは commit 1695b92、`b1s-*` は 61619a9。61619a9 が変えたのは level 2 のコマンドバッファの切り方だけで、level 1 と既定の経路は同じ)
```

コマンドは台帳と同じ固定形 (`prefill_bench.sh`: `--expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 --max-context 4096 --verification full-sha256`) に、上の `TF_` を 1 つずつ付けた。
15 run すべて exit 0、Swapouts 0。profile 付きなので、どの壁時計も台帳には載らない (§2 の候補欄に出る)。

## 0. 結論

1. **M6 の GPU の帯域は、読み 144.4 GB/s、書き 143.0 GB/s、コピー (読み + 書き) 136.2 GB/s** (中央値、各 7 本、1 GiB)。公称 153 GB/s の代わりに、下限表の天井には読みの 144.4 を使う。
2. **下限の合計は 0.924 s (導出)、同じ段の実測の合計は 3.966 s。**差の大きい順は `attn.swa` 0.764 → `moe` 0.722 → `shared` 0.427 → `post` 0.374 → `attn.full` 0.250 → `qkv` 0.175 → `rope` 0.170 → `oproj` 0.110 → `tail` 0.035 (§1)。
3. **`post` はほぼ router** (0.366 s、norm は 0.017 s)。**`moe` の GPU 秒は GEMM だけ** (gate/up 0.738、down 0.346、gelu 0.010、tile の残り 0.000 s)。tile stream の待ちは GPU の秒に入っていない。
4. **GPU の外の空き (`gpuq idle`) は 2 本目以降 0.806、0.838 s** (1 本目 1.650 s)。下限は 0 s なので、差としては `moe` と `shared` の間に入る。ただし GPU の段の表とは別の量で、足せない (§1.2)。

## 1. 数字

### 1.1 GPU の段 (`bench/m6/floor_table.py` の出力)

```
python3 bench/m6/floor_table.py bench/m6/results/2026-09-24-Mac18-5-b1-profile bench/m6/results/2026-09-24-Mac18-5-b1-profile-detail
```

| 段 | 演算量 TFLOP | バイト GB | 演算の下限 s | 帯域の下限 s | 下限 s | 実測 s | 差 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `attn.swa` | 0.825 | 1.522 | 0.044 | 0.011 | 0.044 | 0.809 | 0.764 |
| `moe` | 7.074 | 13.684 | 0.380 | 0.095 | 0.380 | 1.102 | 0.722 |
| `shared` | 2.653 | 1.138 | 0.143 | 0.008 | 0.143 | 0.570 | 0.427 |
| `post` | 0.054 | 1.696 | 0.003 | 0.012 | 0.012 | 0.386 | 0.374 |
| `attn.full` | 0.503 | 0.431 | 0.027 | 0.003 | 0.027 | 0.277 | 0.250 |
| `qkv` | 3.501 | 2.059 | 0.188 | 0.014 | 0.188 | 0.363 | 0.175 |
| `rope` | 0 | 1.979 | 0 | 0.014 | 0.014 | 0.184 | 0.170 |
| `oproj` | 2.001 | 1.356 | 0.108 | 0.009 | 0.108 | 0.217 | 0.110 |
| `tail` | 0 | 1.256 | 0 | 0.009 | 0.009 | 0.044 | 0.035 |

- 実測: `attn` 以外は level 1 の中央値 (`moe` 1.102、`shared` 0.570、`tail` 0.044、各 n = 3)。`attn` の中の段は、level 1 の `attn` 中央値 2.250 s を level 2 の中央の run の割合で割った (level 2 は膨らむので秒を直接使わない)。表に無い `embed` 0.000、`norm` 0.007、`kvcopy` 0.007 s は下限を出していない。
- 下限の式 (導出): max(FLOP ÷ 18.61 TFLOP/s, バイト ÷ 144.4 GB/s)。T = 2478、manifest の `arch` (hidden 2816、30 層のうち full 5 層、sliding は q 16 × 256・kv 8 × 256、full は q 16 × 512・kv 2 × 512 で `attentionKEqV` により v 射影なし、shared 2112、expert 704 × top-8 / 128、window 1024)。
  バイトは段が最低限動かす量。int4 g32 + bf16 scale の重み (0.5625 B/param) を 1 回読み、fp16 の活性を 1 回読んで 1 回書く。`moe` の 13.684 GB の大半は全 128 expert の重み 12.8 GB。
- 演算量は [17 §1](17-NAX-NOT-USED.md) と一致する (full 層の k = v を 1 本に数えた分だけ `qkv` が小さい)。

### 1.2 ホスト側 (`b1-host`、run 順)

| | run 1 | run 2 | run 3 |
| --- | ---: | ---: | ---: |
| prefill 壁時計 s | 5.368 | 4.512 | 4.486 |
| `gpuq busy` s | 3.620 | 3.563 | 3.574 |
| `gpuq idle` s | 1.650 | 0.838 | 0.806 |
| `wait.front` s | 2.356 | 2.367 | 2.369 |
| `fetch` s | 1.884 | 1.075 | 1.047 |
| `wait.tile` s | 0.878 | 0.822 | 0.818 |
| prefill io s | 1.082 | 0.862 | 0.838 |

- `gpuq idle` は、GPU のキューが空いていた秒。GPU 側の段の実測にも含まれない。
- [14](14-G5-VERIFY-AHEAD.md) の `g5-host-par-ahead2` (3c45f83、`TF_` で同じ経路を開いた候補) では `fetch` 0.638 s、`gpuq idle` 0.412 s だった。今回はどちらも大きい。[23 §2](23-GPU-CANDIDATES-DEFAULT.md) の「prefill io が 0.9〜1.1 s に増えた」と同じ向き。原因は調べていない。

### 1.3 分けた行 (`b1s-*`、61619a9)

```
python3 bench/m6/floor_table.py bench/m6/results/2026-09-24-Mac18-5-b1s-profile bench/m6/results/2026-09-24-Mac18-5-b1s-profile-detail
```

**B2 に渡す順** (分けた行で置き換えた後の、差の大きい順。`moe` と `post` は分けた行で数える):

| 順 | 段 | 下限 s | 実測 s | 差 s |
| ---: | --- | ---: | ---: | ---: |
| 1 | `attn.swa` | 0.044 | 0.813 | 0.769 |
| 2 | `moe.gateup` | 0.253 | 0.738 | 0.485 |
| 3 | `shared` | 0.143 | 0.570 | 0.427 |
| 4 | `post.router` | 0.003 | 0.366 | 0.363 |
| 5 | `attn.full` | 0.027 | 0.280 | 0.253 |
| 6 | `moe.down` | 0.127 | 0.346 | 0.219 |
| 7 | `qkv` | 0.188 | 0.364 | 0.176 |
| 8 | `rope` | 0.014 | 0.177 | 0.163 |
| 9 | `oproj` | 0.108 | 0.219 | 0.111 |
| 10 | `tail` | 0.009 | 0.044 | 0.035 |
| — | `post.norm` | 0.017 | 0.017 | −0.001 |
| — | `moe.gelu` | 0.017 | 0.010 | −0.007 |

- 実測の出し方は §1.1 と同じ (level 1 の中央値を、level 2 の中央の run の割合で割る)。`moe` の level 1 は 1.083〜1.144 s、level 2 は 1.086〜1.152 s で、切っても `moe` の GPU 秒は膨らんでいない。
- level 2 の生の秒 (run 順): `post.router` 0.393、0.396、0.395 s、`post.norm` 0.017、0.018、0.018 s。
- `moe.gelu` と `post.norm` は、実測が DRAM の帯域の下限より小さい。tile ごとの scratch は小さく、GPU のキャッシュに残っていると読める (未確認)。この 2 段の下限は当てにならないが、どちらも差は 0 前後なので順位には効かない。
- `post.router` の演算量は 0.054 TFLOP。0.366 s は実効 0.15 TFLOP/s (導出)。
- `moe.gateup` と `moe.down` は実効 6.4、6.8 TFLOP/s (導出)。

## 2. 分かったこと / 外れたこと

- **下限から最も遠いのは `attn.swa` と `moe` で、ほぼ同じ秒 (0.76 と 0.72 s)。**`attn.swa` は sgmm を既定にした後でも、下限 0.044 s に対して 0.809 s。
- **`post` は演算量がほぼ無いのに 0.386 s。**下限は帯域側の 0.012 s。[18 §1](18-RESET-ABSOLUTE-FLOOR.md) の読み (router の重みをトークンの数だけ読み直す、1 トークン 1 threadgroup) と矛盾しない。分けると router が 0.366 s、norm が 0.017 s (§1.3)。
- `rope` は 0.184 s、下限 0.014 s。
- `moe` の実測は tile stream の待ちを含む、と 04 B1 と 18 は書いていたが、GPU の秒には含まれていない (§1.3、tile の残り 0.000 s)。待ちはホスト側の `wait.tile` と `gpuq idle` に出る。
- 下限表は GPU の秒の和で、壁時計ではない。壁時計 4.5 s には、`gpuq idle` と、層ファイルの検証・expert の読み込みの重なりが入る。

**逸脱**: 数字は (b) でなく (c)。§1.1 と §1.3 は commit が違う (6791e58 と 61619a9、既定の経路は同じ)。帯域の天井は読みだけのカーネルの中央値で、GEMM の読み書きの混ざり方は反映していない。`moe` の下限は重みが GPU から読めることを前提にし、SSD やページキャッシュからの読み込み (prefill io) は含まない。

## 3. 次に渡すもの

- [04 B2](04-GATES.md) は §1.3 の順に 1 段ずつ。1 番目は `attn.swa` (差 0.769 s)。
- 4 番目の `post.router` は演算量が小さく、差のほぼ全部が作り (1 トークンに 1 threadgroup、router の重みをトークンの数だけ読み直す、[18 §1](18-RESET-ABSOLUTE-FLOOR.md))。
