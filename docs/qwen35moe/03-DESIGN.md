# 03. 設計 — 変換・カーネル・ランタイム

---

## 1. 変換と repack (残作業)

どちらの候補も MLX 量子化済み・3 ロール分割済みなので ([02](02-CHECKPOINTS.md))、
残るのは「`.gturbo` に書く前の焼き込み」「名前寄せ」「repack を通すこと」だけ。
MLX 側は焼き込みをしないもの (`q_norm` の 1/16) があるので、本リポジトリ側で必ずやる。

### 1-1. 焼き込みと読み込み時の注意

| 処理 | 要否 | 根拠 |
| --- | --- | --- |
| RMSNorm 系 (`input_layernorm` / `post_attention_layernorm` / `q_norm` / `k_norm` / `norm` / `mtp.*_norm`) への `+1` | **公式 MLX-4bit: 不要 (MLX 変換側が焼き済み)。足すと二重になる** | 上流 bf16 との差がちょうど +1.000001 ([10 §4](10-MLX4BIT-AUDIT.md)) |
| 同上 (oQ4e-g64) | **不要 (こちらも焼き済み)。**差は +0.99999x、MTP 側の norm も同じ規約 | [12 §2](12-OQ4E-G64-AUDIT.md) |
| `linear_attn.norm` への `+1` | **しない (どちらでも)。**`RMSNormGated` は `1+w` 規約ではない | 上流 bf16 とビット一致 ([10 §4](10-MLX4BIT-AUDIT.md)) |
| `q_norm.weight` × `head_dim ** -0.5 = 1/16` | **要る (どちらでも)。実施済み — `Scripts/qwen35/bake_snapshot.py`、無損失** ([12 §5](12-OQ4E-G64-AUDIT.md))。q_norm は RoPE の直前で、RoPE は回転 (ノルム保存) なので順序を入れ替えてよい。attention の `scale` を **1.0** に固定でき、`scale==1.0` ゲートの tensorops 経路 (§4-3) の余地が残る | |
| `gate_up_proj` の gate / up 分割 | **不要。**MLX が `switch_mlp.gate_proj` / `up_proj` に分割済み。`RepackPlanner.planLayerFile` の「ロール 3 本」ループが無改造で動く | [10 §3](10-MLX4BIT-AUDIT.md) |
| `conv1d.weight` の軸順 | **MLX は `[8192, 4, 1]`** (上流 bf16 は `[8192, 1, 4]`)。squeeze 後の値はビット一致。読み込み側で軸を必ず確認 | [10 §4](10-MLX4BIT-AUDIT.md) |

### 1-2. 名前寄せ (実施済み — [13 §3](13-PHASE1-REPACK.md))

`RepackPlanner.classify` は `language_model.` 接頭辞を要求する (実測 = ソース)。
oQ 系の名前との対応:

| チェックポイント側 | 本リポジトリ |
| --- | --- |
| `language_model.model.layers.{i}.…` | `classify` の `language_model.` 接頭辞に**そのまま合致する** |
| `…mlp.switch_mlp.{gate,up,down}_proj` | `routedExpertRole` が見る `.experts.switch_glu.` を **`.mlp.switch_mlp.` にも当てる** (1 文字列) |
| `vision_tower.…` | `isMultimodalTensorName` の既存接頭辞と**一致する** |
| `language_model.mtp.…` | draft セクションへ (§6-4)。**本体より先に外す。**同梱ドラフターの `layers.0` は本体の `layers.0` ではないので、`switch_mlp` の文字列だけ足すと層 0 の routed expert が二重になって planner が落ちる ([13 §3](13-PHASE1-REPACK.md)) |
| `language_model.lm_head` | tie しないので常駐に入れる |

### 1-3. ディスク

候補はローカルに揃っているのでダウンロードは無い。repack の出力
(19.5〜21.9 GB) ぶんの空きが要る。`DiskSpaceChecker` の予約と合わせて
**空き 26 GB を install の門にする** (bf16 変換案時代の「30 GB / ピーク 25 GB」は消滅)。
実測: テキストのみの install は **20.49 GB** ([13 §2](13-PHASE1-REPACK.md))。

### 1-4. Phase 1 の出口で導出が実測に変わる

`--verify-install` が緑、ファイルサイズが [01 §5-3](01-MODEL.md) の表と一致、
`expertStride == 1_769_472`。ここが「導出を実測に変える最初の点」
(ファイル読みのレベルでは 3 回バイト一致済み — [10 §2](10-MLX4BIT-AUDIT.md))。

---

## 2. 新規に書くカーネル

**方針: 新規は全部 `Sources/TurboFieldfare/Metal/Qwen/qwen.metal` に置く。**
既存 `.metal` を触らない (Gemma の実測値を凍結したままにするため)。
SiLU だけは共有ヘッダに置きたくなるが、`gelu_pytorch_tanh` が 4 箇所に重複定義されている
現状に合わせて **`qwen.metal` にローカル定義する**。

### 2-1. `qwen_silu_mul` — SiLU

`grep -rni silu Sources/TurboFieldfare/Metal` は **0 件** (実測 = ソース)。
`gelu_mul_fp16` (`utility.metal`) の SiLU 版を書く: `y = x * sigmoid(x) * up`。
shared expert (int4) と routed expert (phase1) の両方から呼ぶ。

### 2-2. `qwen_rope_partial` — Qwen 規約の partial RoPE

```
rotary_dim = 64,  half = 32
pair < half:  x0 = h[pair], x1 = h[half + pair]
              freq = pow(theta, -2*pair / rotary_dim)      // 分母は 64
              回転
pair >= half: そのまま (h[64..255] は無変更)
```

既存 `fused_rope_neox_pair` との違いは [01 §3-2](01-MODEL.md)。**流用禁止。**
`q_norm` / `k_norm` と融合して `qwen_qkv_epilogue` にまとめる (Gemma の
`fused_qkv_epilogue` と同じ構造: q 16 head + k 2 head + v 2 head = 20 threadgroup)。
**v には norm も RoPE も掛からない** (Gemma は v に no-scale norm を掛けていたので違う)。

### 2-3. `qwen_attn_output_gate` — 出力ゲート

`o[h,d] *= sigmoid(gate[h,d])`。gate は `q_proj` 出力の後半 4096 次元。
`o_proj` の直前に 1 dispatch。decode では `fused` に畳んでよい (4096 要素の elementwise)。

### 2-4. `qwen_moe_shared_gate` — shared expert の sigmoid ゲート

`shared_out *= sigmoid(dot(shared_expert_gate, x))`。1 行 GEMV + スカラー乗算。
`SharedExpertInt4` の後に追加する。

### 2-5. `qwen_delta_conv_step` / `qwen_delta_conv_chunk` — 因果 depthwise conv

チャネル 8192、カーネル 4、groups=8192、bias 無し、直後に SiLU。
decode は状態 `[8192, 3]` を持ち回して 1 step 更新。prefill は §2-6 の中に畳む。

### 2-6. `qwen_delta_rule` — 本命 (omlx の blocked-sequential を写す)

方式の選定: **chunkwise (WY / UT 変換) ではなく厳密な逐次漸化式で書く。**
prefill 2048 トークンの線形注意の総計算量は 193 GFLOP (**導出**、[01 §5-6](01-MODEL.md)) で
MoE の 4.63 TFLOP の 4% しかなく、omlx のソースコメントも
「chunked 再定式化なしの厳密漸化式 = WY 経路の半分の FLOP」と結論している。
幾何は `omlx/custom_kernels/qwen35_prefill/gdn.py` の `gated_delta_blocked_seq` を写す
(**実測(上流)** = ソース):

```
TB = 32 (時間ブロック),  DB = 32 (threadgroup が持つ dv 行数),  256 スレッド
grid = (Dv/DB = 4, Hv = 32, B = 1)  →  128 threadgroup
thread → dv = tid/8 (0..31),  seg = tid%8,  d0 = seg*16
状態は【レジスタ】: float4 st[4] = 16 float/thread  (256 × 16 = 32 × 128 ✓)
threadgroup memory: k_s[32][Dk+8], q_s[32][Dk+8], v_s[32][DB+8], g_s[32], b_s[32]
                    ≒ 20 KB (上限 32 KiB に収まる。+8 はバンク衝突回避)
```

肝は 3 つ:

1. **状態をレジスタに置く。**`float4 st[4]` は 16 レジスタで、占有率をほとんど落とさない
   (threadgroup memory に 16 KiB 置く案より良い)
2. **縮約は `simd_shuffle_down(4→2→1)` + `simd_shuffle` の同報。
   ホットループに `threadgroup_barrier` が 1 個も無い。**barrier は時間ブロックの境界だけ
3. **k/q/v を時間ブロック単位で threadgroup memory に協調ロードする。**
   素の mlx_lm カーネルは dv を Dv/4 で切るので同じ k/q 行を 32 回読み直し、
   16k トークンの 1 層で約 13 GB の冗長トラフィックになる。Dv/16 で切って staging
   すると 8 分の 1。**[01 §5-5](01-MODEL.md) は「状態往復 126 MiB/token」しか
   数えていないが、切り方を間違えると k/q の読み直しがそれを桁で超える**

小技: 減衰 `st *= gt` を `kv_mem` の縮約と同じループで掛ける (2 パスにしない)。
出力書き出しは `seg == 0` のスレッドだけ。時間ブロック TB は `(16, 32, 48)` が
可変なので、本機では 3 通り測る (Phase 4)。

decode は同じカーネルに `T=1` で入る。1 dispatch = 1 層ぶんのチャンク全部で、
`S` はチャンク内で 1 度もメモリに出ず、最後に状態バッファへ書き戻すだけ。

**中止線: 逐次形が 30 層合計で 150 ms を超えたら再設計** ([05 §2](05-RISKS.md) #2)。
ただし omlx の記述どおりなら chunkwise は FLOP が 2 倍で逃げ道としての魅力は薄い —
先に「TB を振る」「Dv の切り方を変える」を試すこと。

### 2-7. `qwen_delta_norm_gate` — RMSNormGated

head_v_dim=128 の RMSNorm (**`+1` しない**) → `* silu(z)`。
`z` は `in_proj_z` の出力 4096 を 32×128 に見たもの。

### 2-8. `qwen_lm_head_greedy` — LM head

`LMHeadChainInt4` は `realDecodeD=2816` / `realDecodeVocab=262144` を関数定数で焼いている
(実測 = ソース)。**`D=2048` / `vocab=248320` の specialization を足す。**
`vocab_size=248,320` に対し実際の語彙は BPE 248,044 + added 33 = 248,077 なので、
**末尾 248,077..248,319 の 243 行は学習されていない。argmax / sampling で -inf にマスクする**
(コストはほぼゼロ、事故は防げる。[10 §3](10-MLX4BIT-AUDIT.md))。

### 2-9. 要らないもの

- **logit softcap** — Qwen には無い。`LogitOutput` / `Sampler` の無条件適用を切る
- **埋め込みの `sqrt(hidden)` スケール** — Gemma だけの規約。`embedInt4` が融合しているので
  Qwen 経路では 1.0 を渡す (または専用パスを足す)
- **mRoPE** — テキストのみなら恒等 ([01 §3-2](01-MODEL.md))。Vision まで不要

---

## 3. ランタイムの構造変更

### 3-1. アーキテクチャの選択を実体化する

現状は `Model.load(… expecting: ArchConfig = .gemma4_26B_A4B)` の**デフォルト引数 1 個**が
唯一の選択点で、CLI / Server / App はどれも引数を渡していない (実測 = ソース)。

```swift
public enum ModelFamily: String, Sendable { case gemma4, qwen35moe }
```

1. `manifest.arch.family` を新設 (無ければ `gemma4`)。`versionMinor` を **3** に上げる
2. `Model.load` は `expecting:` を捨て、**manifest の family でベースラインを選ぶ**
3. `ArchConfig.qwen35_35B_A3B` を `ModelTypes.swift` に追加
4. `fullAttentionLayerMask: [UInt8]` を **`layerKinds: [UInt8]` (0=sliding, 1=full, 2=linear)** に
   改名。gemma4 の manifest を読むときは旧フィールドから写す

### 3-2. forward runner は分ける

**`RealForwardRunner` (3,300 行) は触らない。`QwenForwardRunner` を新設する。**

理由: `RealForwardRunner` の decode ループは Gemma 専用に手で畳んだ
コマンドバッファ 3 本のパイプライン (CB1 → shared/routed を跨いで次の層と重ねる) で、
そこに第 3 の層種別を差し込むと **Gemma 側の実測値が動く。**この
リポジトリの資産は「測った数字」なので、動かさないほうが安い。

共有するもの: `Model` / 各 streamer / `ExpertCacheBudget` / `MoE` / `SharedExpertInt4` /
`Sampler` / `LMHeadChainInt4` / prefill の GEMM 一式 / `Attention` の汎用 PSO。

`QwenForwardRunner` の 1 層 (線形注意層、decode):

```
h_in = rmsnorm_bf16w(h, input_layernorm)
qkv  = int4_gemv(in_proj_qkv, h_in)                 # 8192
z    = int4_gemv(in_proj_z,  h_in)                  # 4096
a,b  = fp16_gemv(in_proj_a/b, h_in)                 # 32 ずつ
qkv  = qwen_delta_conv_step(qkv, convState)         # + silu
o    = qwen_delta_rule(qkv, a, b, A_log, dt_bias, S)
o    = qwen_delta_norm_gate(o, z, norm)
h   += int4_gemv(out_proj, o)
h_r  = rmsnorm_bf16w(h, post_attention_layernorm)
… MoE (既存の router / shared / routed をそのまま) …
h   += moe_out
```

full attention 層は `qkv` が `q_proj/k_proj/v_proj` になり、
`qwen_qkv_epilogue` → `Attention.encodeFull` → `qwen_attn_output_gate` → `o_proj` に変わる。

**MoE 部分は Gemma と同一の呼び出し**にできる (§4-1)。ここが本計画最大の再利用点。

### 3-3. 状態の置き場所 — `RecurrentStateManager` 新設

`KVCacheManager.LayerKind` は `{swa, full}` の 2 値で、
`kSlot` / `vSlot` / `stride` / `capacity` がすべて「トークン添字のキャッシュ」を前提にしている
(実測 = ソース)。**線形注意層には居場所が無い。**

```swift
final class RecurrentStateManager {
    // 層 → S:[32,128,128] fp32 と conv:[8192,3] fp32
    // 62.8 MiB を 1 本の MTLBuffer に連結、層ごとにオフセット
    func reset()                       // 会話境界。ゼロ埋め
    func snapshot(into: Handle)        // 62.8 MiB のコピー
    func restore(from: Handle)
}
```

`KVCacheManager` 側は `layerKinds[l] == .linear` の層に**バッファを 1 バイトも確保しない**。
`ExpertCacheBudget.kvCacheByteEstimate` も同じ規則で数え直す
(**線形層を勘定に入れると 18 GB 機で false negative が出る**)。

### 3-4. 巻き戻せないという性質 (重要)

再帰状態は**途中のトークンを捨てられないし、巻き戻せない。**
影響するのは 3 箇所:

| 場所 | 現状 | Qwen での扱い |
| --- | --- | --- |
| `KVCacheManager.maximumSafeRewind` | SWA リングの容量から巻き戻し可能量を出す | 線形層は **0**。巻き戻すならスナップショットが要る |
| 投機デコードの verify ブロック | k 行流して j 行だけ採用、KV は j でカーソルを戻す | **状態は k 行ぶん進んでしまう。**§6-3 に対策 |
| サーバーの prompt cache (LCP 再利用) | 共通接頭辞ぶんの KV をそのまま使う | **接頭辞末尾の状態スナップショット (62.8 MiB) を slot ごとに持つ**。§5 |

---

## 4. 既存資産の再利用可否

### 4-1. そのまま使える (差分ゼロ or ほぼゼロ)

| 資産 | 根拠 |
| --- | --- |
| `router_gemv_gemma4_r4` / `_bf16_r4` | `D` / `num_experts` は関数定数。公式 MLX-4bit の router は 8-bit g64 なので int8 経路 (`_r4`) が既定で当たる。oQ4e (打ち直し前) は BF16 なので `_bf16_r4`。**oQ4e-g64 側は未確認** ([02 §1](02-CHECKPOINTS.md)) |
| `router_topk_select_k8` | top-8 固定、Qwen も top-8。softmax の式も等価 ([01 §3-3](01-MODEL.md)) |
| `moe_phase1_gate_up_act_u16load` / `_subset` | `D` / `F` / `top_k` は関数定数。**活性化だけ SiLU に差し替え** |
| `moe_phase2_down_reduce_k8` | top-8 = 8 simdgroup 固定。一致 |
| 期待値の事前確保 | `maxRouterRows * 256 * Float` と `kPrefillRouterMaxExperts=256` が**既に 256** |
| `PreadExpertStreamer` / `MmapExpertMapping` / `ExpertCacheBudget` | `expertStride` / `expertsPerLayer` を layout.json から読む。次元のハードコード無し |
| `GTurboPackedExpertsLayoutV1` | 16 KiB 揃え。Ornith は 108 ページちょうど |
| `rmsnorm_bf16w` / `rmsnorm_no_scale` | `1+w` 規約が一致 (焼き込み状態は候補ごとに確認 — §1-1) |
| `attention_prefill_causal_qblock_d256` | **head_dim だけで選ばれる。**Qwen の full 層 (256) がそのまま当たる |
| `MPPPrefillInt4QMM` (Metal 4 tensor ops) | `K % 64 == 0` と group 64 のみが条件。2048 / 4096 / 512 すべて満たす |
| `dequant_int4_gemv_simd` 一式 | 次元は実行時引数 |
| `Attention` の汎用 PSO | `(256,16,2)` は専用化が無いので汎用に落ちる。**正しさは保たれる** |

### 4-2. 手を入れる

| 資産 | 内容 |
| --- | --- |
| `MoE.swift` の `realDecode*` 定数 | `(2048, 512, 8, 256)` の specialization を追加。無くても汎用 PSO で動く (**性能だけの話**) |
| `FusedQKVGEMV` | Gemma の `(4096,2048,2816)` / `(8192,1024,2816)` に対し Qwen は `(8192,512,2048)`。specialization 追加 |
| `Attention` | `(256,16,2)` の専用 PSO を足す。`kAttnMaxQPerKV=2` は SWA 用 GQA カーネル専用で、full 経路 (`attention_decode_partial`) は通らないので**触らない** |
| `LMHeadChainInt4` | `D=2048 / vocab=248320` の specialization (§2-8) |
| `LogitOutput` / `Sampler` | softcap を条件化 (Qwen は 0.0 = 無効) |
| `KVCacheManager` / `ExpertCacheBudget` | 層種別 3 値、線形層は KV を確保しない (§3-3) |
| `ManifestReader` | `family` で期待ベースラインを選ぶ |
| `RDAdvice` のポリシー定数 | 1.69 MiB 単位で再調律 ([05 §1-3](05-RISKS.md)) |

### 4-3. 将来の伸びしろ (Phase 6 以降)

`attention_prefill_full_tensorops_2d_validity_v2` は
`kPrefillTensorOpsHeadDim=512` / `kPrefillTensorOpsOutputs=8` / `kPrefillTensorOpsKeys=64` を
焼いていて、Swift 側が `headDim==512 && numQHeads==16 && numKVHeads==2 && scale==1.0` で門を閉めている。

**Qwen の full 層は `16/2 = 8` で `kPrefillTensorOpsOutputs` が一致する。**
`scale` も §1-1 の q_norm 焼き込みで 1.0 にできる。
**`kPrefillTensorOpsHeadDim=256` の兄弟カーネルを作るだけで Metal 4 tensor op 経路が開く**
可能性がある。差分が小さいわりに効く候補として記録しておく (**未確認**)。

---

## 5. サーバーへの波及 (SPEC / CONFORMANCE との関係)

**本計画はサーバー仕様を書き換えない。**[docs/serving/SPEC.md](../serving/SPEC.md) が
唯一の規範である。ここに書くのは「Qwen を載せると SPEC のどの不変条件が前提を失うか」だけ。

| 論点 | 現状 | Qwen |
| --- | --- | --- |
| prompt cache の LCP 再利用 | 共通接頭辞ぶんの KV をそのまま残す | **線形層の状態は「接頭辞の終端でのスナップショット」でしか復元できない。**slot ごとに 62.8 MiB を持つか、共通接頭辞から再計算するかの二択 |
| 文脈シフト (古いトークンを捨てる) | SWA リングが自然に回る | **不可能。**線形層の状態から過去を引き算できない。溢れたら状態リセット + 再 prefill |
| `maximumSafeRewind` | リング容量 − window | 線形層は 0 |
| reasoning 分離 | `<\|channel\|>thought` の 3 状態機械 | `<think>` / `</think>` の**トークン ID 248068 / 248069** で切る別実装 ([04](04-PHASES.md) Phase 5) |

**推奨:** slot あたり 62.8 MiB のスナップショットを持つ。32 slot なら 2.0 GB。
18 GB 機では重い。**Phase 8 で「slot 数 × 62.8 MiB」を `ExpertCacheBudget` の勘定に入れ、
入らない構成は起動時に断る**という、既存と同じ作法にする。

---

## 6. MTP — 本モデル最大の追い風

### 6-1. ドラフターが同梱されている

Gemma 側は `mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit` を別途 pin して
取ってきていた (実測 = ソース)。**Ornith は `mtp.*` として同じチェックポイントに入っている。**
`mtp_use_dedicated_embeddings: false` なので `embed_tokens` と `lm_head` は本体と共用。
実物の供給源は oQ4e(-g64) の `language_model.mtp.*` (503 MB、switch_mlp は 4-bit g64 で
積層済み — [02 §5](02-CHECKPOINTS.md))。**公式 MLX-4bit には入っていない。**

### 6-2. 全エキスパート常駐という手

MTP 層の routed expert は 256 本 × 1.6875 MiB = **453 MB**。
**これを丸ごと常駐にすれば、ドラフト 1 トークンの SSD I/O はゼロになる。**

Gemma のドラフターは 236 MB を常駐させていた (`DraftModelSource.expectedPayloadBytes`、
実測 = ソース) ので、453 MB は同じ桁の判断で通る。**推奨: `draft/draft_weights.bin` に
エキスパートも含めて全部入れ、resident index で持つ。**

### 6-3. verify ブロックと再帰状態 (§3-4 の続き)

k 行の verify ブロックを流して j 行だけ採用するとき、線形層の状態は k 行ぶん進んでいる。

| 案 | コスト | 判断 |
| --- | --- | --- |
| ブロック前にスナップショット、j < k なら復元して j 行を replay | 復元 126 MiB + **replay 1 トークンにつき線形層の重み 570 MB 再読み** | 却下。replay が高すぎる |
| **行ごとに状態を書き出し、j が決まったら j 番目を採用** | 追加書き出し `(k-1) × 62.8 MiB`。k=4 なら **188 MiB / ブロック = 導出 1.25 ms** | **採用** |
| 状態を fp16 で保存 | バイト半減 | 却下。採用された状態が丸め済みになり誤差が蓄積する |

ブロック 4 で平均 3 トークン採用なら **オーバーヘッドは 0.42 ms/token** (**導出**)。
1 トークン 15〜20 ms の予算に対して 2〜3%。**許容範囲だが、`DraftAcceptanceProbe` の
計器に「状態書き出しのナノ秒」を足して実測できるようにする。**

### 6-4. `DraftForward` は書き直し

Gemma のドラフターは (a) 4 層 (b) 本体の K/V を共有 (c) `k_proj`/`v_proj` を持たない
(d) sandwich norm (実測 = ソース)。**Ornith の MTP は全部違う** — 1 層、自前の
`k_proj`/`v_proj` (= 自前の KV キャッシュ 2 KiB/token が要る)、norm 2 本、MoE 付き。

```
h_mtp = mtp.fc( concat( pre_fc_norm_embedding(embed(x_{t+1})),
                        pre_fc_norm_hidden(h_t) ) )
h_mtp = decoder_layer(h_mtp)          # full attention + MoE-256
logits = lm_head(mtp.norm(h_mtp))     # lm_head は本体と共用
```

`GTurboManifestDraftV1` は `tieWordEmbeddings == true` を**無条件に要求**し、
`sharedSlidingKVLayer` / `sharedFullKVLayer` を必須にしている (実測 = ソース)。
**Qwen の draft セクションは別スキーマにする** (`draftFamily` で分岐)。
配線の第三者資料として `Shiftedx/…-mtplx` の `mtp_contract` が読める
([02 §8](02-CHECKPOINTS.md))。
