# 27. 段の実測の取り直し — 既定 (cd8c26b) の上の下限表と B2 の順 (2026-09-24)

**候補ではなく既定の測定。台帳 §1 は動かない** (profile 付きの 9 run だけ。進捗の行は [06 §1](06-LEDGER.md) の `ledger-swa-mpp-default` 3.496 s のまま)。
[26 §3](26-SWA-ATTN-MPP-DEFAULT.md) の「段の実測は、この既定の上で取り直す」をやった。[24](24-B1-FLOOR-TABLE.md) と同じ手順で、数字は (c) (新しいプロセスの 1 回目)。

証拠 (M6、この順に取った):
[`b2-profile-detail`](../../bench/m6/results/2026-09-24-Mac18-5-b2-profile-detail/) (`TF_PREFILL_GPU_PROFILE=2`)、
[`b2-profile`](../../bench/m6/results/2026-09-24-Mac18-5-b2-profile/) (`=1`)、
[`b2-host`](../../bench/m6/results/2026-09-24-Mac18-5-b2-host/) (`TF_PREFILL_HOST_PROFILE=1`)、各 n = 3。

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit a122dd3 / minos 15.0
(a122dd3 は docs と証拠だけを足した commit で、ソースは cd8c26b と同じ)
```

コマンドは台帳の固定形 (`prefill_bench.sh`: `--expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 --max-context 4096 --verification full-sha256`) に、上の `TF_` を 1 つずつ付けた。
9 run すべて exit 0、Swapouts 0。天井は 24 と同じ (18.61 TFLOP/s、144.4 GB/s)。

## 0. 結論

1. **`attn.swa` は実測 0.813 → 0.258 s (差 0.769 → 0.214 s) で、順位は 1 番から 6 番に下がった。**
2. **B2 の次は `moe.gateup` (下限 0.253、実測 0.741、差 0.487 s)。**2 番以下は `shared` 0.427 → `post.router` 0.353 → `attn.full` 0.240 → `moe.down` 0.220 s。`attn.swa` 以外の段の実測は、24 と 0.02 s 以内で同じ。
3. **GPU の段の実測の合計は 3.966 → 3.354 s** (分けた行の無い段の和、導出)。下限の合計は 0.924 s のまま。
4. **`gpuq idle` は 0.576、0.395、0.424 s** (run 順)。24 の 2 本目以降 (0.806、0.838 s) より小さい。

## 1. 数字

### 1.1 GPU の段 (`bench/m6/floor_table.py` の出力を差の大きい順に。`moe` と `post` は分けた行で数える)

```
python3 bench/m6/floor_table.py bench/m6/results/2026-09-24-Mac18-5-b2-profile bench/m6/results/2026-09-24-Mac18-5-b2-profile-detail
```

| 順 | 段 | 下限 s | 実測 s | 差 s | 24 §1.3 の実測 s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `moe.gateup` | 0.253 | 0.741 | 0.487 | 0.738 |
| 2 | `shared` | 0.143 | 0.570 | 0.427 | 0.570 |
| 3 | `post.router` | 0.003 | 0.356 | 0.353 | 0.366 |
| 4 | `attn.full` | 0.027 | 0.267 | 0.240 | 0.280 |
| 5 | `moe.down` | 0.127 | 0.346 | 0.220 | 0.346 |
| 6 | `attn.swa` | 0.044 | 0.258 | 0.214 | 0.813 |
| 7 | `rope` | 0.014 | 0.180 | 0.166 | 0.177 |
| 8 | `qkv` | 0.188 | 0.354 | 0.165 | 0.364 |
| 9 | `oproj` | 0.108 | 0.210 | 0.103 | 0.219 |
| 10 | `tail` | 0.009 | 0.044 | 0.035 | 0.044 |
| — | `post.norm` | 0.017 | 0.016 | −0.001 | 0.017 |
| — | `moe.gelu` | 0.017 | 0.010 | −0.008 | 0.010 |

- 実測の出し方は 24 と同じ (level 1 の中央値。`attn` の中の段は level 1 の `attn` 中央値を level 2 の中央の run の割合で割る)。表に無い `embed` 0.000、`norm` 0.007、`kvcopy` 0.007、`moe.other` 0.000 s。
- level 1 の生の秒 (run 順): `attn` 1.656、1.662、1.655 s、`moe` 1.174、1.098、1.098 s、`shared` 0.570 s × 3。
- level 2 の生の秒 (run 順): `moe.gateup` 0.791、0.753、0.734 s、`moe.down` 0.366、0.352、0.345 s、`attn.swa` 0.286、0.287、0.288 s、`post.router` 0.399、0.398、0.397 s。
- `moe.gateup` は実効 6.4 TFLOP/s (4.716 TFLOP ÷ 0.741 s、導出)。24 と同じ。

### 1.2 ホスト側 (`b2-host`、run 順)

| | run 1 | run 2 | run 3 |
| --- | ---: | ---: | ---: |
| prefill 壁時計 s | 3.681 | 3.474 | 3.507 |
| `gpuq busy` s | 2.999 | 2.973 | 2.976 |
| `gpuq idle` s | 0.576 | 0.395 | 0.424 |
| `wait.front` s | 1.775 | 1.770 | 1.772 |
| `fetch` s | 0.796 | 0.600 | 0.629 |
| `wait.tile` s | 0.865 | 0.852 | 0.860 |
| prefill io s | 0.447 | 0.368 | 0.377 |

- `wait.front` は 24 の 2.36 s から 1.77 s に減った (`attn.swa` の縮みと同じ向き、約 0.59 s)。`wait.tile` は 0.82 → 0.86 s でほぼ同じ。
- 1 本目の layerVerify は 9.468〜9.495 s (level 1・2)。host の 1 本目は 6.309 s で、これだけ 1 本目の壁時計が伸びていない (3.681 s)。直前の level 1 から間を空けずに取ったため、と読める (未確認、[22](22-GPU-CANDIDATES-GREEDY-AND-FIRST-RUN.md) の性質)。

## 2. 分かったこと / 外れたこと

- **`attn.swa` の縮み (level 2 で 0.873 → 0.288 s、[25](25-SWA-ATTN-MPP.md)) は、既定の上でも同じ量で出ている。**他の段は動いていない。
- `moe.gateup` は 24 と同じく最大の差で、今は単独の 1 番。`matmul2d` 版 (5decec4 で既定) の上で実効 6.4 TFLOP/s。
- ホストの `fetch` と `gpuq idle` は 24 より小さい。24 §1.2 の「prefill io が 0.9〜1.1 s」は今回 0.37〜0.45 s。M6 の状態の差で、26 §0-2 と同じ向き (原因は調べていない)。

**逸脱**: 数字は (b) でなく (c)。24 と commit は違う (6791e58・61619a9 と a122dd3) が、違いは既定の経路の `attn.swa` だけ。

## 3. 次に渡すもの

- [04 B2](04-GATES.md) の 2 段目は `moe.gateup`。変更前は level 2 の 0.734〜0.791 s (本文書 §1.1)、下限 0.253 s。合格の閾値は 04 B2 の表に置いた (計測の前)。
