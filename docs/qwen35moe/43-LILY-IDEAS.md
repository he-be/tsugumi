# 43. Perplexity Lily からの持ち帰り — GQA packing と tensor ops の MoE prefill、写さないもの 4 つ (検討、2026-09-13)

[Optimizing On-Device Inference for Apple Silicon](https://www.perplexity.ai/hub/blog/optimizing-on-device-inference-for-apple-silicon)
(Perplexity Engineering、2026-09-01。デモは [pplx-garden/lily](https://github.com/perplexityai/pplx-garden/tree/main/lily)) は、
Qwen3.6-35B-A3B 専用の Rust + Metal 推論エンジン Lily で MLX-LM 比 prefill 1.23 倍 / decode 1.35 倍を出した記事である。
読む理由は家族の近さ: **モデル構造が本線と同じ** (256 expert top-8 + shared 1、full attention 10 層 GQA 16Q/2KV + Gated DeltaNet 30 層、
Q4 group-64 affine) で、Metal で同じ 3 つの形 (不均一な expert、伸びる KV、固定長の再帰) を相手にしている。

**表記: 本書の「公表値(Lily)」は記事の記載である。ソースは読んでいない。**
そして土俵が違う: **記事は M5 Max 40 コア GPU / 128 GB で 19.4 GB の重みを全部常駐**させた batch-1 の値で、
本線は 16〜18 GB でページキャッシュに預ける腕 ([27 §9](27-PHASE6-THROUGHPUT.md))。
待ちの通貨が「GPU の帯域」と「ページの写像」で違うので、**写せるのは機構の形だけで、数字は 1 つも写せない。**
本書は実測を 1 つも足していない。既定は 1 つも変えない。M6 Mac mini (16 GB / 256 GB、2026-09-22 着) を前提にした
機種側の見立ては [investigations/M6_MAC_MINI_TARGETING.md](../investigations/M6_MAC_MINI_TARGETING.md) (復元済み、§10 が補遺)。

---

## 0. 結論を先に

| # | 論点 | 結論 |
| --- | --- | --- |
| 1 | 記事の 14 項目のうち | **8 つは既に入っている** (§1)。GEMM 内 Q4 展開、GDN のレジスタ常駐 scan、chunked prefill、row-parallel GEMV、融合 3 本、coalesced KV、投機デコード |
| 2 | 写せる未実装は 2 つ | **GQA packing** (§2、M6 に依らず今すぐ) と **routed MoE prefill GEMM の `matmul2d` 化** (§3、M6 待ち)。それ以外の未実装 4 つは常駐前提の最適化で、ページキャッシュの腕では的が小さい (§4) |
| 3 | 記事の否定的結果 | **「投機デコードは 18% 遅くなった」は本線に当てはまらない。**記事は常駐で verify 行 2〜5 本の GEMM 形状が損、本線は expert フェッチを verify 行で償却できる側で、実測 ×1.2〜1.33 ([38](38-MTP-VERIFY-PATH.md) / [39 §0](39-RESIDENCY-COMMIT.md) #5)。MTP は残す |
| 4 | M6 で記事の数字は出るか | **出ない。**記事の prefill 1.23 倍は「GPU が 97% 忙しい常駐」での値。本線の Qwen prefill は GPU 3.05 ms/tok に対し io 4.48 ms/tok ([27 §2](27-PHASE6-THROUGHPUT.md)) で **I/O 側が重い**。tensor ops は GPU 側しか縮めない。`F_RDADVISE` 後 ([mtp/52](../mtp/52-D-P7-PREFILL-QUEUE-DEPTH.md)) の io を測ってから §3 に着手する |
| 5 | 順番 | (1) §2 GQA packing を M3 Pro で。(2) 9/22 に M6 文書 §7 の 5 つの実測。(3) §3 を M6 で。(4) §4 はやらない |

---

## 1. 対応表 — 記事の 14 項目と本線の現状

**済** = 本線に同等の機構がある。**部分** = 形はあるが記事と違う。**未** = 無い。根拠は file:line と文書。

### prefill

| # | 記事 (公表値(Lily)) | 本線 | 根拠 |
| --- | --- | --- | --- |
| 1 | Q4 を grouped GEMM 内で bf16 tile に展開、fp32 累積、展開済み配列を作らない (ablation +77.4% @512) | **済** | `prefill_moe_gemm_int4` が nibble を threadgroup の B tile に `fma(nibble, scale, bias)` で直接展開、`simdgroup_float8x8` で累積 (`Metal/Prefill/prefill.metal:831`、展開 `:925-940`、累積 `:903`)。bf16 展開配列はどこにも無い |
| 2 | routing (histogram → prefix scan → scatter → block map) を 1 command buffer 内で GPU 完結、層内 CPU 同期ゼロ (ablation +89% @512) | **未 (意図的)** | router GEMV + top-8 は GPU (`QwenForwardRunner.swift:1470-1498`) だが top-8 は毎層ホストに読み戻す (`:1206`、`:1543`)。prefill は `[T, topK]` 表を読み戻して histogram / sort / block map を Swift で組む (`QwenPrefill.swift:1020-1042`、`PrefillMoEGrouping.swift:91`、`PrefillGroupedRoutedMoE.swift:45-77`)。**ホストが top-8 を見てスロットと residency を組む設計**なので外せない。§4-1 |
| 3 | expert の負荷から tile 幅と simdgroup 数を選ぶ (16 行 → 32 行 × 4 simdgroups で +13.2% @2K) | **部分** | 選ぶのは幾何でなく変種: 最大 expert の行数で rows 経路を切替 (`usesExpertRowsPath`、`PrefillGroupedRoutedMoE.swift:701`)、function constant の行上限 (`:537-548`)、down を占有率の崖で分割 (`:806-836`、[mtp/31 §4](../mtp/31-M8-A-ROWS-BENCH.md) / [32 §3](../mtp/32-M8-A-ROWS-SPLIT.md))。tile は 64×64×32 / 4 simdgroups 固定 (`prefill.metal:825-828`) |
| 4 | GDN prefill scan: 状態を simdgroup のレジスタに常駐、fp32 state/gate、bf16 q/k、barrier 無し (+5.6% @2K) | **済** | `float4 st[4]` fp32、gate fp32、還元は `simd_shuffle` のみ、ホットループに `threadgroup_barrier` 無し (`Metal/Qwen/gdn.metal:22-27`、`:60-165`)。違いは q/k/v が bf16 でなく **fp16** な点だけ。設計根拠は [03 §2-6](03-DESIGN.md) |
| 5 | prompt chunking で一時メモリを上限化 | **済** | 運用 2048 (`RuntimeConfiguration.swift:84`)、スクラッチは最大幅で 1 回確保して再利用 (`QwenPrefill.swift:52-115`)。[35](35-PREFILL-CHUNK-WIDTH.md) |
| 6 | prefill GEMM を Metal 4 tensor ops (Neural Accelerator) に載せる | **未 (Qwen 側)** | `mpp_prefill_affine_threadgroup_f16` (`Metal/TensorCore/tensorops.metal`) は **Gemma の `RealForwardRunner` からしか呼ばれない** (`:194`、`:436`)。Qwen 経路は MPP を一切参照しない。[03 §4](03-DESIGN.md) `:283`、`:302-308` に「full 層は tensor 版 attention の head_dim=256 兄弟が要る」と計画だけある。§3 |

### decode

| # | 記事 (公表値(Lily)) | 本線 | 根拠 |
| --- | --- | --- | --- |
| 7 | batch-1 は row-parallel GEMV、simdgroup が 1 出力に協調 | **済** | `dequant_int4_gemv_simd` (`Metal/Quant/dequant_int4.metal:217`)、`moe_phase1/2` (`Metal/MoE/moe.metal:612`、`:651`) |
| 8 | argmax を次ステップの入力スロットに GPU が直接書く、command buffer 2 本の交互 | **未** | head の argmax は GPU だがホストが読んで次ステップの embed にスカラー定数で渡す (`QwenForwardRunner.swift:1339`、`:1131-1138`、`runSync`)。§4-2 |
| 9 | 依存を記録した concurrent pass、独立カーネルを同時実行 (795 kernel / 555 段) | **未** | `MTLDispatchTypeConcurrent` はリポジトリに 0 件。段ごとに serial encoder。重なりは command buffer 粒度 (deferred join、shared 分岐と expert I/O、[27 §3-1](27-PHASE6-THROUGHPUT.md))。§4-3 |
| 10a | expert の gate+up 射影と活性化の融合 | **済** | `moe_phase1_gate_up_act_u16load` (`moe.metal:612`) |
| 10b | down 射影 + routing 重み + shared expert の合流 | **済** | `moe_phase2_down_reduce_k8` が 8 expert の down GEMV、routing 重み、残差 (shared 込み) を 1 本で (`moe.metal:651-687`) |
| 10c | attention 前の q/k 準備の融合 | **済** | `qwen_qkv_epilogue` (q/k RMSNorm + RoPE、`QwenForwardRunner.swift:1406`)、GDN 側 `qwen_delta_qkv_prepare` / `qwen_delta_gates`。q 16 行を抜く blit 1 本が残る (`:1427-1435`) |
| 10d | 再帰更新と正規化の融合 | **部分** | 再帰は単独カーネル、norm + z-gate はその後ろで融合 (`encodeDeltaNormGate`、`:1384`) |
| 11 | KV 読みの coalescing (+2.1% @3,840) | **済** | `attention_decode_partial` が head-dim 連続で歩く (`Metal/Attention/attention.metal:191-210`) |
| 12 | **GQA packing**: Q 4 head を 1 threadgroup に束ねて KV 行を 1 回読む (+23.8% @32K) | **未 (Qwen 側)** | packed カーネル `attention_decode_gqa_swa_partial` はあるが `qPerKV <= 2` **かつ** SWA にゲート (`Attention.swift:172`、`:218`) = Gemma 専用。Qwen の 16Q/2KV は `encodeFull` → `preferGQASWA: false` (`:248`、呼び出し `QwenForwardRunner.swift:1440`) で **8 head が同じ KV 行を読み直す**。`kAttnMaxFullQPerKV = 8` / `kAttnFullQPerThreadgroup = 2` (`attention.metal:36-37`) は宣言だけ。§2 |
| 13 | 32K 以上で fixed-block レイアウトに切替 (+7.7% @32K、+40.2% @128K) | **未** | 分割数は `min(16, effLen)` 固定 (`Attention.swift:63-66`)。長さで切り替えるのは MTP verify の行 attention だけ ([38 §2-2](38-MTP-VERIFY-PATH.md))。§4-4 |
| 14 | 投機デコード: **18% 遅くなった**ので不採用 | **済で、逆の結果** | `--qwen-mtp` (503 MB sidecar、幅 2 verify、[36](36-MTP-DECODE.md))。a1 ×1.207 / t4 ×1.110 ([38](38-MTP-VERIFY-PATH.md) `:185-189`)、async residency 込みで a1 ×1.333 / t4 ×1.190 ([39 §0](39-RESIDENCY-COMMIT.md) #5)。§0 #3 |

### 記事が言う「もう伸びない」と本線の律速

記事は MoE GEMM / GEMV が重み読み出し速度の 97.9% / 90.3% に達し、演算を抜いても 0.2% しか動かない (= 帯域律速) と言う。
本線の decode は **gpu 28.50 / io 15.75 / host 5.73 ms/tok** ([27 §2](27-PHASE6-THROUGHPUT.md)) で、io の 85% が residency set の保守、
その 94% が `MTLResidencySet.commit()` (13.6 ms/tok)、disk0 は 0.6〜0.85 GB/s しか動いていない ([27 §9-1, 9-3](27-PHASE6-THROUGHPUT.md))。
**律速は帯域でも launch でもなく写像**である。gpu の 28.5 ms は 5 通りの腕で動かなかった ([27 §9-2](27-PHASE6-THROUGHPUT.md))。
この 1 点が §4 の「写さない」判定の根拠になる。

---

## 2. GQA packing — M6 を待たず、今すぐ (優先: 高)

**何を**: Qwen の full attention 10 層 (16Q/2KV、head_dim 256) の decode で、同じ KV head を共有する 8 つの Q head を
1 threadgroup に 4 つ束ね、KV 行の読み出しを 8 回 → 2 回にする。記事は算術も出力バイトも同一で +23.8% (@32K)。

**なぜ本線に写せるか**:
- 機構が純粋に GPU 内で完結し、ページキャッシュの腕と直交する。記事の中で**唯一まるごと移せる**未実装項目。
- 形はもうある: `attention_decode_gqa_swa_partial` の `qPerKV <= 2` / SWA ゲート (`Attention.swift:172`、`:218`) を外し、
  full 用に `qPerKV = 8` / threadgroup あたり 4 head の兄弟を書く。定数 `kAttnMaxFullQPerKV = 8` は既に置いてある。
- 検証は既存の attention 検査 ([17](17-PHASE2-KERNELS.md)) と greedy 一致で足りる (出力バイトは同一のはず)。

**見込み (導出、誤差大)**: 記事の +23.8% は 32K の値で、attention が decode の大半を占める文脈長。本線の運用点 (数 K) では
attention 10 層の KV 読みが gpu 28.5 ms のうち小さい割合なので、**数 % 止まり**と見るのが妥当。
それでも t4 (2,698 トークン要約) のように文脈が伸びる腕では効きが上がる。**取る前に attention 10 層の decode 時間を
[27 §3](27-PHASE6-THROUGHPUT.md) の stage profile で確かめ、上限を導出してから書く。**

---

## 3. routed MoE prefill GEMM の `matmul2d` 化 — M6 待ち (優先: 中、条件付き)

**何を**: `prefill_moe_gemm_int4` の内側を Metal 4 `mpp::tensor_ops::matmul2d` に置き換え、M6 の GPU 各コアの
Neural Accelerator に乗せる。記事の構造 (tile 単位で Q4 → bf16 に展開し threadgroup memory に置いて tensor op に渡す、
fp32 累積、bf16 出力) は本線の `prefill_moe_gemm_int4` と同じで、置き換え先が明確。
M6 文書の M6-2 に当たる。記事は prefill 時間の約 9 割が expert GEMM と言う。

**なぜ M6 待ちか**: M3 Pro (apple9 / macOS 15) では `__HAVE_TENSOR__` が立たず経路が丸ごと落ちる
(`MetalContext.swift:251-256`)。M6 + macOS 27 で初めて生きる。

**なぜ条件付きか**:
1. **本線の Qwen prefill は I/O 側が重い**: gpu 3.05 ms/tok 対 io 4.48 ms/tok ([27 §2](27-PHASE6-THROUGHPUT.md))。
   tensor ops は 3.05 の側しか縮めない。`F_RDADVISE` で io が 5.56 → 11.99 GB/s に上がった腕 ([mtp/52 §0](../mtp/52-D-P7-PREFILL-QUEUE-DEPTH.md))
   で io が gpu を下回るかを **M6 で先に測る**。下回らなければ着手しても壁時計は動かない。
2. M6 の GPU は 12 コアで M3 Pro の 18 コアより少ない。tensor ops を使わないままの pp が下がる可能性がある
   (M6 文書 §2-2、§7-4)。**比の分母を先に取る。**
3. 前提作業 2 つ: M6-0 (`supportsFamily(.apple10)` ゲートをパイプライン生成の成否に置換、`PrefillAttention.swift:74`) と、
   Q4 g64 の scale/bias をタイル番号でなくグローバル K 位置から索く (M6-1)。本線は g64 なので M6-1 は Qwen 側では不要のはず (**未確認**)。

**見込み (導出)**: M6 文書 §2-4 は Gemma で MoE 8.59 → 2 s と見積もった。Qwen でも比率は同程度と見るが、
壁時計は io に上限され、**記事の 1.23 倍は出ない**。

---

## 4. 写さないもの 4 つ

### 4-1. routing の GPU 完結 (記事 #2)

記事はホストが routing 中間を覗く ablation より +89% と言うが、それは常駐で「覗く理由が無い」から。
本線は **ホストが top-8 を見てスロットを割り、residency set を組む** ([27 §9](27-PHASE6-THROUGHPUT.md)、[39](39-RESIDENCY-COMMIT.md)) ので、
読み戻しは設計の芯であり外せない。1 層 2 command buffer (`QwenForwardRunner.swift:1204-1206`) もここから来る。
histogram / scan / scatter だけ GPU に移す部分適用は可能だが、読み戻しが残る以上、同期点は減らない。

### 4-2. トークン受け渡しの GPU 内完結 (記事 #8)

節約できるのは join 1 回あたり 0.2 ms 程度。写像の 15 ms/tok の前では誤差。deferred commit で join は既に 81 → 41 本/tok に減っている。

### 4-3. concurrent dispatch (記事 #9)

本線も 1 トークン約 800 dispatch で記事の 795 と同規模だが、gpu 28.5 ms は重みストリーミングに縛られ 5 通りの腕で動かなかった
([27 §9-2](27-PHASE6-THROUGHPUT.md))。空き GPU 資源を埋める余地が無い。M6 でも 16 GB 構成は 153 GB/s で M3 Pro と同じ。

### 4-4. 長文での attention レイアウト切替 (記事 #13)

効き始めるのが 32K から。本線の運用点は数 K で、prompt cache ([41](41-PROMPT-CACHE.md)) も長文を prefill し直さない方向。
文脈長が常用で 32K を超える日が来たら §2 の packed カーネルに分割数の可変化を足す。

---

## 5. 次の一手

| # | いつ | 何を | 合否 |
| --- | --- | --- | --- |
| 1 | 今 (M3 Pro) | attention 10 層の decode 時間を stage profile で取り、§2 の上限を導出 | 上限が 3% 未満なら §2 は保留 |
| 2 | 今 (M3 Pro) | §2 GQA packing (full 用 packed カーネル) | greedy 一致、t1〜t4 で退行なし、t4 で改善 |
| 3 | 9/22 | M6 文書 §7 の 5 つ (apple10、SSD `F_NOCACHE`、fp16 ピーク、そのままビルドの pp、deployment target) | 「M6 で N 倍」を書く前提 |
| 4 | 9/22 | Qwen prefill の gpu / io ms/tok を M6 + `F_RDADVISE` で取る | io < gpu なら §3 に着手 |
| 5 | M6 | M6-0 → §3 | GPU 合計が下がり、壁時計も下がる。片方だけなら記録して止める |

---

## 出典

- 記事: [Optimizing On-Device Inference for Apple Silicon](https://www.perplexity.ai/hub/blog/optimizing-on-device-inference-for-apple-silicon) (Perplexity Engineering、2026-09-01)
- デモ: [pplx-garden/lily](https://github.com/perplexityai/pplx-garden/tree/main/lily) (**ソース未読**)
- 機種側: [investigations/M6_MAC_MINI_TARGETING.md](../investigations/M6_MAC_MINI_TARGETING.md)
- 本線の実測: [27](27-PHASE6-THROUGHPUT.md)、[38](38-MTP-VERIFY-PATH.md)、[39](39-RESIDENCY-COMMIT.md)、[mtp/52](../mtp/52-D-P7-PREFILL-QUEUE-DEPTH.md)
