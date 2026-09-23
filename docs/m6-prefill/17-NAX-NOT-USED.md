> **§0-4 と §3 の方針は [18](18-RESET-ABSOLUTE-FLOOR.md) で取り消した。**§1 の演算量は [04 B1](04-GATES.md) の下限表に引き継ぐ。

# 17. M6 の prefill は NAX をほぼ使っていない — 実効 2.3 TFLOP/s。単体プローブの関門を外し、本番カーネルを `matmul2d` で書き直す (導出と方針、2026-09-24)

G0 ([10](10-G0-BASELINE.md)) と G1 ([16](16-G1-MPP-GROUP32.md)) を終えた時点で、**M6 の prefill は M3 Pro の 1.11 倍しか速くない。**
M6 の GPU には各コアに Neural Accelerator (NAX。ANE とは別物、[05 §10](05-RISKS.md)) があるのに、今のコードはそれを使う形になっていない。
この文書は、その事実を数字で示し、計画 ([04](04-GATES.md)) の順序を変える理由を書く。**新しい実測は無い。**数字はすべて 10 の実測と model の manifest からの**導出**。

## 0. 結論

1. **prefill の GPU 時間 7.12 s は、ほぼすべてを従来の simdgroup カーネルが使っている。**routed MoE・shared MLP・attention 本体は tensor ops を 1 本も使っていない。
   G1 で MPP に通せるようにしたのは q/k/v/o の射影だけで、既定では off。
2. **実効は 2.3 TFLOP/s (導出)。**同じ M6 で、Apple 製の fp16 GEMM は 18.61 TFLOP/s 出ている ([03 §1](03-FACTS.md))。
   2478 トークンの prefill の演算量は約 16.7 TFLOP なので、その速さで回れば GPU 側は 0.9 s になる (導出。天井であって目標値ではない)。
3. **G1 の MPP カーネルは NAX の速さを引き出す作りになっていない** (§2)。attn が 0.3 s 縮んだだけなのはそのため。
4. **方針**: G2 (単体プローブで合否を決めてから G3) の関門を外し、**本番カーネルを `matmul2d` で書き直すことから始める。**順序は routed MoE → 射影と shared → attention 本体。判定は毎回 M6 の本番ループで取る。

## 1. 段ごとの実効 TFLOP/s (**導出**)

演算量は manifest の `arch` から数えた (hidden 2816、30 層のうち full 5 層、q 16 頭、sliding の kv 8 頭 × 256、full の kv 2 頭 × 512、
shared 2112、expert 704 × top-8、sliding window 1024)。T = 2478、1 積和 = 2 FLOP。k と v は別々の射影として数えた (full 層の `attentionKEqV` の分は上限側に倒れる)。
秒は [10 §1.2](10-G0-BASELINE.md) の level 1 の中央値。`attn` の中の割合は [10 §1.3](10-G0-BASELINE.md) の level 2 の割合を、level 1 の 3.534 s に掛けた。

| 段 | 演算量 TFLOP | GPU 秒 | 実効 TFLOP/s |
| --- | ---: | ---: | ---: |
| routed MoE (`moe`、tile stream の待ちを含む) | 7.07 | 2.678 | 2.6 |
| 射影 q/k/v/o (`attn` の 35%) | 5.57 | 1.24 | 4.5 |
| shared MLP (`shared`) | 2.65 | 0.866 | 3.1 |
| attention 本体、sliding (`attn` の 36%、平均文脈 813) | 0.82 | 1.27 | 0.6 |
| attention 本体、full (`attn` の 8%、平均文脈 1240) | 0.50 | 0.28 | 1.8 |
| **合計** (router 0.05 を含む) | **16.68** | **7.123** | **2.3** |

- 比べる先の 18.61 TFLOP/s は、MPS が 4096³ の fp16 GEMM で出した値。int4 の展開も、expert ごとに行数がばらつく形も含んでいない。**天井であって、届くと約束する数字ではない。**
- **sliding window の attention 本体が 0.6 TFLOP/s と最も低い。**演算量は小さいが、秒数は routed MoE の半分ある。

## 2. G1 の MPP カーネルが速くない理由 (コードを読んだもの、**未計測**)

`Metal/TensorCore/tensorops.metal` の `mpp_prefill_affine_threadgroup_f16`:

- 出力タイルが 64 × 32 と小さく、4 simdgroup で 1 回の `matmul2d` を回す。
- K 方向 64 要素ごとに、int4 を fp16 に展開して threadgroup メモリへ書く → barrier → `matmul2d` 1 回 → 結果を手のループで累積器に足す → barrier。展開と同期の間は NAX が止まっている。
- 同じ重みタイルを、M 方向の別の threadgroup がそれぞれ展開し直している。
- 使っているのは射影だけ (GPU 合計の 17%、[10 §3](10-G0-BASELINE.md))。

## 3. 計画の変更

| 変える前 ([04](04-GATES.md)) | 変えた後 |
| --- | --- |
| G2 で `matmul2d` の単体プローブを作り、B ≥ 1.5 × A かつ B ≥ 0.5 × C を満たしたら G3 へ。満たさなければ tensor ops の線を閉じる | **G2 を関門にしない。**単体の数字で線を閉じない。本番カーネルを書いて本番ループで判断する (単体と本番で符号が反転する、[05 §2](05-RISKS.md)) |
| G3 (routed MoE) → G4 (attention と射影) | **1. routed MoE の expert GEMM** (2.68 s、最大) → **2. 射影と shared** (G1 の MPP カーネルを作り直し、shared にも使う) → **3. attention 本体** (sliding を先に) |
| G1 の (b) greedy 不一致で止まる | [16](16-G1-MPP-GROUP32.md) に記録したまま、今は追わない。**新しいカーネルごとに greedy の差は記録する**が、それを理由に止めない |

カーネル 1 本ごとの進め方:

1. MBP で書く (MBP では tensor 経路が存在しないので、ビルドと既存テストだけ)。環境変数で新旧を切り替え、同じバイナリで比べる。
2. M6 で `bench/m6/kernel_check.sh` → CPU 参照との一致。
3. M6 で `prefill_bench.sh` を新旧交互に 6 本ずつ + `TF_PREFILL_GPU_PROFILE=1` を 3 本。**その段の GPU 秒と壁時計**を 10 の分母 (壁時計 9.507 s、routed MoE 2.678 s、`attn` 3.534 s) と比べる。
4. 結果文書を 1 本書く (routed MoE は 12 番。04 で G3 に割り当て済みの番号)。

**未確認のまま進む前提**: 自前カーネルの `matmul2d` が NAX に届くこと (公称、[03 §3](03-FACTS.md))。
M6 には Xcode が無いのでキャプチャでは示せない ([05 §5](05-RISKS.md))。示せるのは本番ループの GPU 秒が縮んだかどうかだけ。
