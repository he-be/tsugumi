# 39. QKV エピローグを 1 ディスパッチに — GPU busy は変わらず、既定に入れない (実測、2026-09-25)

**候補の測定。台帳 §1 は動かない。**
[38 §3](38-ROUTER-NAX.md) の段の表で `rope` は 0.101 s (MLX の RoPE だけで 0.023 s、[31](31-MLX-KERNELS-PER-STAGE.md))。
この段は q・k・v の per-head RMS norm と q・k の NeoX RoPE を、6 回のディスパッチで q・k・v を読み書きして行っている。
1 つのカーネル `prefill_qkv_norm_rope_block` にまとめた (44b3c4f、`TF_PREFILL_QKV_EPILOGUE=fused`)。
norm は 256 スレッドで同じ順に和を取り、RoPE の前に half に丸めるので、出力は分けた版とビット一致する (MBP のテスト `fusedEpilogueMatchesSeparatePassesBitForBit`、両方の層の形)。

証拠 (M6、`bench/m6/results/2026-09-25-Mac18-5-<ラベル>/`): `e1-{default,fused}-{a,b}`、`e1-host-fused`。

```
測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 44b3c4f〜5f22a7c (ソースは 44b3c4f と同じ) / minos 15.0
```

コマンドは台帳の固定形 (`prefill_bench.sh`)。全 run exit 0、Swapouts 0。MBP の `Scripts/test.sh` は 44b3c4f のソースで 1630 本通過。

## 0. 結論

1. **GPU busy は変わらない。**融合版 1.874・1.864 s (`e1-host-fused` の 2・3 本目)、38 の既定 1.871・1.870 s (`r1-host-nax`)。
2. 壁時計 (c) は、交互で既定 2.326 s・融合 2.290 s (各 n = 6)。6 組すべてで融合側が短いが、差 0.036 s は同じ構成で出る揺れ (0.05 s 程度、[36 §2](36-FULL-ATTN-UNFUSED.md)) の中で、GPU の仕事も減っていない。**縮んだとは言わず、既定に入れない。**
3. 段の表の `rope` 0.101 s は、段ごとに command buffer を切る計測 (level 1) で膨らんだ分と読む。GPU の仕事としては小さかった。

## 1. 数字 (壁時計 (c)、run 順、秒)

| 組 | 既定 | 融合 | 中央値 n = 6 (導出) |
| --- | --- | --- | --- |
| `e1` a / b | 2.464、2.330、2.322 / 2.434、2.318、2.253 | 2.448、2.297、2.264 / 2.389、2.283、2.248 | 2.326 / 2.290 |

## 2. 次に渡すもの

- 段の表 (level 1) で小さい段の秒は、GPU の仕事の大きさを言わない。手を入れる段は GPU busy の変化で確かめてから選ぶ。
- 候補の口 (`TF_PREFILL_QKV_EPILOGUE=fused`) とテストは残す。
