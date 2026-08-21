# 04. 段階計画 Phase 0〜9

各 Phase は「出口条件」を満たすまで次へ行かない。**GPU を使う Phase は 2 以降**
(Phase 2 は GPU を使うがモデルは載せない)。

## 進捗 (2026-08-21 夜)

| Phase | 状態 |
| --- | --- |
| Phase 0 事実確定 | **大半済み。**`gate_up` の順序・RMSNorm の `+1`・`conv1d` の軸順・`in_proj_a` の感度が確定 ([10](10-MLX4BIT-AUDIT.md))。**`oQ4e-g64` 側の照合も完了** ([12](12-OQ4E-G64-AUDIT.md))。**残: 実活性での再測 (合成入力でしか測っていない)、fixtures の作成** |
| Phase 1 変換 | **完了** ([13](13-PHASE1-REPACK.md))。`oQ4e-g64-baked` を repack し `--verify-install` が緑。20.49 GB / `expertStride 1,769,472` / 上流とバイト一致 |
| Phase 2 以降 | 未着手。**GPU はまだ 0 回。**GPU 不要の作業はここで尽きた |

---

## Phase 0 — 事実確定 (GPU 不要)

1. `Scripts/qwen35/dump_reference.py`: CPU / float32 の `transformers` で 8 トークンの
   プロンプトを 1 回流し、**層ごとの入出力・線形注意の状態・router の top-8・logits** を
   `.npy` で落とす (`Scripts/vision/dump_vision_fixtures.py` と同じ立て付け)
2. `gate_up_proj` の連結順 — **解決済み** (MLX が分割済み)。ただし fixtures で 1 度だけ突き合わせる
3. `A_log` / `dt_bias` / `in_proj_a` の値域 — **出した** ([10 §5](10-MLX4BIT-AUDIT.md))。
   残: **実活性での再測** (合成入力 `x ~ N(0,1)` でしか測っていない)
4. 4-bit RTN を通したときの品質: **層ごとの相対誤差と最終 logits の KL** を出す。
   imatrix 版 (oQ4e-g64) との対照もここで取る

**出口:** fixtures がリポジトリに入り、「4-bit で行けるか」に数字が付いている。

**中止線:** 4-bit affine で logits の top-1 一致率が参照に対し 95% を切ったら、
attention と `in_proj_*` を int8 に上げる案 (+約 580 MB) を先に評価する。

## Phase 1 — 変換 (GPU 不要)

入力はローカルの候補 ([02 §2](02-CHECKPOINTS.md))。ダウンロード不要。
焼き込みと名前寄せ ([03 §1](03-DESIGN.md)) → `TurboFieldfareRepack --source-snapshot` →
`.gturbo`。`ArchInfo.load` に Qwen の `config.json` パーサを足し、
`GTurboManifestArchV1` に `family` / `layerKinds` / `linearAttention` セクションを足す。

**出口:** `--verify-install` が緑。ファイルサイズが [01 §5-3](01-MODEL.md) と一致。
`expertStride == 1_769_472`。

## Phase 2 — カーネル (GPU を使うが、モデルは載せない)

[03 §2](03-DESIGN.md) の各カーネルを `TurboFieldfareKernelCheck` で Phase 0 の fixtures に
対して検証。FP16 の誤差床は `Scripts/vision/fp16_error_floor.py` と同じ手続きで先に測る
(**カーネルのバグと丸め誤差を分離できる検証系を先に作る** — PLAN_VISION §6 の教訓)。

**出口:** 全カーネルが誤差床の中。特に `qwen_delta_rule` は
**2048 トークン流した後の状態**が参照と一致すること (1 トークンだけでは足りない)。

## Phase 3 — decode 結線

`QwenForwardRunner` の decode 経路のみ。prefill は off、投機も off、
`--temp 0` の greedy。

**出口:** 固定プロンプトから **64 トークン、参照 (CPU float32) と完全一致**。
一致しないなら層ごとの hidden を突き合わせて発散点を特定する。

**中止線:** 発散点が `qwen_delta_rule` の数値的な蓄積 (fp32 でも合わない) なら、
状態を fp64 相当 (2×fp32 の compensated summation) にする案を検討。それでも
合わなければ chunkwise 形へ (誤差の出方が変わる)。

## Phase 4 — prefill

チャンク幅 512 / 1024 / 2048。`qwen_delta_rule` の T>1 経路。時間ブロック TB を
`(16, 32, 48)` の 3 通り測る ([03 §2-6](03-DESIGN.md))。

**出口:** prefill 経由の greedy 64 トークンが Phase 3 と一致。
**線形注意の 30 層合計が 150 ms 以内** ([05 §2](05-RISKS.md) #2)。

## Phase 5 — トークナイザ / テンプレート / CLI

- `verifyDecoderConfiguration` に **ByteLevel** を許す分岐 (現状は metaspace +
  ByteFallback + Fuse の 3 段固定を要求しており、**確実に弾かれる** — [10 §3](10-MLX4BIT-AUDIT.md))
- `GemmaDecoding` の兄弟として `ByteLevelDecoding` (GPT-2 の byte↔unicode 表)
- `GrammarVocabulary` の piece 復元をデコーダ種別で切り替える
- `vocabSize = 262_144` のリテラルを tokenizer から読むように
- 特殊トークン: `<|im_start|>` 248045 / `<|im_end|>` 248046 / `<|endoftext|>` 248044 /
  `<think>` 248068 / `</think>` 248069 / `<tool_call>` 248058。
  **停止トークンは `[248046, 248044]`** (`generation_config.json`、実測(上流))
- チャットテンプレートは**上流の `chat_template.jinja` をそのまま同梱する**。
  Gemma 側は「上流に chat_template が無い」から Swift で書いていた (実測 = ソース) が、
  Ornith は持っている。サーバーの redraw 不変条件用の兄弟 jinja は Phase 8 で
- ツール呼び出しは `<tool_call><function=名前><parameter=名前>値</parameter></function></tool_call>`
  という **XML 形** (実測(上流))。Gemma の `call:name{...}` とは別物なので
  パーサと GBNF ビルダを新設

**出口:** `TurboFieldfareCLI --model … --prompt …` が日本語と英語で通る。
`<think>` ブロックが `reasoning_content` に分離される。

## Phase 6 — 計測と運用点

`bench.sh` の作法をそのまま踏襲する。**temp 1.0 のまま、クールダウン 20 秒**
(採点は temp 0 / 10 秒)。GPU は 1 個だけ。

1. `ExpertTelemetry.startTrace` の TSV を 1 回だけ取り、**オフラインで**
   8/16/24/32/48 スロット × LFU/LRU のヒット率を再計算する (モデル再実行不要)
2. その結果を持って、運用点の候補をユーザーに出す。**既定は変えない**
3. チャンク幅 512/1024/2048 (と 4096 を評価するなら候補追加の可否を含めて) の A/B
4. `RDAdvice` のバイト上限を 1.69 MiB 単位で調律
5. `RouterPreviewProbe` を 256-way の基準線 3.13% に対して取り直す
6. (必要になったら) 案「oQ を自分で回す」([02 §3](02-CHECKPOINTS.md)) の評価もここから

**注意:** 反復 3 未満のセルには解釈を書かない。数字だけ置く。

## Phase 7 — MTP

[03 §6](03-DESIGN.md)。全エキスパート常駐 + 行ごと状態書き出し。

**出口:** `RESULTS_MTP.md` と同じ様式で tok/s / TTFT / peak の 3 点。
**受入は「非投機と greedy でバイト一致」** (Gemma 側の D5 不変条件と同じ)。

## Phase 8 — サーバー

[03 §5](03-DESIGN.md)。prompt cache のスナップショット方針を決め、
`ExpertCacheBudget` に勘定を足す。[docs/serving/SPEC.md](../serving/SPEC.md) に
Qwen 固有の行を足すかは、そこで別途判断する。

## Phase 9 — Vision

下記。

---

## Vision (後回しの根拠と、やるときの中身)

**tower の形は Gemma と偶然ほぼ同じ** (どちらも SigLIP2-so400m-patch16 系):

| | Gemma 4 vision | Ornith vision |
| --- | ---: | ---: |
| hidden / 層 / head / FFN | 1152 / 27 / 16 / 4304 | **1152 / 27 / 16 / 4304** |
| patch | 16 | 16 |
| head_dim | 72 | 72 |

**しかし中身は別物である:**

| 項目 | Gemma 4 | Ornith |
| --- | --- | --- |
| norm | RMSNorm (bias 無し) | **LayerNorm (bias 有り)** — `norm1.bias` 等が実在 (実測(上流)) |
| qkv | 3 本に分離、bias 無し | **融合 1 本 `[3456,1152]` + bias** |
| 位置 | 加算テーブル `[2,10240,1152]` **かつ** 2D RoPE (θ=100) | **`pos_embed [2304,1152]` の補間 + mRoPE** |
| マージ | **3×3 平均プーリング** + 標準化 | **2×2 patch merger MLP** (`merger.linear_fc1 [4608,4608]` → `fc2 [2048,4608]`) |
| リサイズ | アスペクト比保存、48 の倍数 | **面積境界** (`shortest_edge 65536` = 256², `longest_edge 16777216` = 4096²) の smart resize |
| 時間軸 | 無し | `temporal_patch_size 2` (静止画は同じフレームを 2 枚) |
| LM 側の位置 | 1 次元 | **mRoPE (t,h,w)。ここで初めて [01 §3-2](01-MODEL.md) の interleaved mrope が要る** |

`VisionImagePreprocessor` には時間軸の概念が無く、`vision_pool_std_block` は 3×3 固定、
`vision_qk_norm_rope2d_block` は加算テーブル + 2D RoPE を前提にしている (実測 = ソース)。
**カーネルもパイプラインも書き直しになる。**再利用できるのは
`vision_bf16_qmm_f16` と `vision_attention_full_seg_d72` くらい。

**画像 1 枚のトークン数の上限に注意:** `longest_edge 16777216` px を素直に取ると
`16777216 / 16² / 2² = 65,536` トークンになる。**上限を切る** (Gemma は 280 だった)。

---

## 次の一手 (2026-08-21 夜)

**GPU 不要 (全部終わった):**

1. ~~`oQ4e-g64` の norm 規約の照合~~ → **完了。焼き済み / `linear_attn.norm` は素**
   ([12 §2](12-OQ4E-G64-AUDIT.md))
2. ~~`conv1d.weight` の軸順~~ → **完了。`[8192,4,1]`、squeeze 後 30/30 ビット一致**
   ([12 §1](12-OQ4E-G64-AUDIT.md))
3. ~~router のビット幅~~ → **完了。BF16。公式版 (int8) と違い、MTP 経路に効く**
   ([12 §4](12-OQ4E-G64-AUDIT.md))
4. ~~`q_norm` への `1/16` の焼き込み~~ → **完了。**差分シャード 1 枚を足す形で、
   **無損失** (2 のべき乗なので指数だけが動く) を検査つきで
   ([12 §5](12-OQ4E-G64-AUDIT.md))
5. **名前寄せ** ([03 §1-2](03-DESIGN.md)) — `.mlp.switch_mlp.` を `routedExpertRole` に当てる 1 文字列
6. `.gturbo` への repack を通す

道具: `Scripts/qwen35/audit_checkpoint.py` / `bake_snapshot.py` (numpy だけ、GPU 不要)。
repack 済みモデルは `scratch/ornith-oq4e-g64.gturbo`。
`--bf16` を付けると上流 bf16 の抽出と突き合わせる。
repack の入力は焼き込み済みの `~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-baked`。

**GPU が要る (ここから先は指示待ち。GPU 不要の作業はもう無い):**

7. **生成スモーク。`oQ4e-g64` はまだ 1 度も推論を通していない。**
   相対 L2 誤差を測っただけで、文が出るかは未確認
8. `in_proj_a` の感度 ([10 §5](10-MLX4BIT-AUDIT.md)) を**合成入力ではなく実活性**で
   測り直す (Phase 0 の宿題)
9. 2 候補の品質差を測る。**ここで初めて「imatrix に 2.35 GB の価値があるか」が決まる**

**まだ手を付けていない前提:**

- **tokenizer は確実に弾かれる** ([10 §3](10-MLX4BIT-AUDIT.md))。`decoder.type = ByteLevel` が
  `verifyDecoderConfiguration` の要求と合わない。Phase 5 の作業として据え置き
- Gated DeltaNet カーネル ([03 §2-6](03-DESIGN.md)) は 1 行も書いていない
- **ランタイムはまだこの `.gturbo` を開けない。**最初の障害はカーネルではなく
  **attention のビット幅が層ごとに違うこと** ([13 §4-2](13-PHASE1-REPACK.md))。
  `Model.swift` は `quant.attention` のスロット 1 個に対して全層を検証している
- 運用点 (スロット数・チャンク幅) は Gemma 4 の値のままで、Ornith 用には何も測っていない
