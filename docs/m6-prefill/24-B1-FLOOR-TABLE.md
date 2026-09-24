# 24. B1 の下限表 — 既定 (6791e58) の上の段の秒と、M6 の帯域の実測 (2026-09-24)

**候補ではなく既定の測定。台帳 §1 は動かない** (コードを変えていない。進捗の行は [06 §1](06-LEDGER.md) の `ledger-gpu3-default` 4.584 s のまま)。
[04 B1](04-GATES.md) の材料を揃えた。**B1 はまだ閉じない** (§3)。数字は (c) (新しいプロセスの 1 回目) で取った。B0 の (b) の口が無いため。

証拠 (M6、この順に取った):
[`b1-profile-detail`](../../bench/m6/results/2026-09-24-Mac18-5-b1-profile-detail/) (`TF_PREFILL_GPU_PROFILE=2`)、
[`b1-profile`](../../bench/m6/results/2026-09-24-Mac18-5-b1-profile/) (`=1`)、
[`b1-host`](../../bench/m6/results/2026-09-24-Mac18-5-b1-host/) (`TF_PREFILL_HOST_PROFILE=1`)、各 n = 3。
[`gpu_bw_probe`](../../bench/m6/results/2026-09-24-Mac18-5-gpu_bw_probe/) (`swift bench/m6/gpu_bw_probe.swift 1024`)。

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 6791e58 / minos 15.0
(帯域プローブだけ commit 1695b92。プローブを足した commit で、ランタイムのソースは同じ)
```

コマンドは台帳と同じ固定形 (`prefill_bench.sh`: `--expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 --max-context 4096 --verification full-sha256`) に、上の `TF_` を 1 つずつ付けた。
9 run すべて exit 0、Swapouts 0。profile 付きなので、どの壁時計も台帳には載らない (§2 の候補欄に出る)。

## 0. 結論

1. **M6 の GPU の帯域は、読み 144.4 GB/s、書き 143.0 GB/s、コピー (読み + 書き) 136.2 GB/s** (中央値、各 7 本、1 GiB)。公称 153 GB/s の代わりに、下限表の天井には読みの 144.4 を使う。
2. **下限の合計は 0.924 s (導出)、同じ段の実測の合計は 3.966 s。**差の大きい順は `attn.swa` 0.764 → `moe` 0.722 → `shared` 0.427 → `post` 0.374 → `attn.full` 0.250 → `qkv` 0.175 → `rope` 0.170 → `oproj` 0.110 → `tail` 0.035 (§1)。
3. **GPU の外の空き (`gpuq idle`) は 2 本目以降 0.806、0.838 s** (1 本目 1.650 s)。下限は 0 s なので、差としては `moe` と `shared` の間に入る。ただし GPU の段の表とは別の量で、足せない (§1.2)。

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

## 2. 分かったこと / 外れたこと

- **下限から最も遠いのは `attn.swa` と `moe` で、ほぼ同じ秒 (0.76 と 0.72 s)。**`attn.swa` は sgmm を既定にした後でも、下限 0.044 s に対して 0.809 s。
- **`post` は演算量がほぼ無いのに 0.386 s。**下限は帯域側の 0.012 s。[18 §1](18-RESET-ABSOLUTE-FLOOR.md) の読み (router の重みをトークンの数だけ読み直す、1 トークン 1 threadgroup) と矛盾しないが、norm と router の内訳は取れていない (§3)。
- `rope` は 0.184 s、下限 0.014 s。
- `moe` の実測は tile stream の待ちを含む。GEMM だけの秒は分からないので、`moe` の差 0.722 s のうちどれだけが GEMM かは言えない。
- 下限表は GPU の秒の和で、壁時計ではない。壁時計 4.5 s には、`gpuq idle` と、層ファイルの検証・expert の読み込みの重なりが入る。

**逸脱**: 数字は (b) でなく (c)。帯域の天井は読みだけのカーネルの中央値で、GEMM の読み書きの混ざり方は反映していない。`moe` の下限は重みが GPU から読めることを前提にし、SSD やページキャッシュからの読み込み (prefill io) は含まない。

## 3. B1 を閉じるのに足りないもの

[04 B1](04-GATES.md) の合格は「全段に下限 / 実測 / 差が並ぶ」。残りは 2 行で、どちらも **Sources の profile を変えないと取れない**。
AGENTS.md の範囲 (ソースを変えない) を越えるので、この文書では手を付けていない。

1. `post` の中の norm と router の分離。
2. `moe` の中の GEMM / gelu / tile stream の待ちの分離。

これが無くても、差の大きい順の上位 (`attn.swa`、`moe`、`shared`、`post`) は上の表で決まる。
