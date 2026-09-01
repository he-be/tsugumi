# 13. Phase 1 — `.gturbo` への repack (実測(手元)、2026-08-21)

[04-PHASES.md](04-PHASES.md) Phase 1 の出口を満たした。**GPU は使っていない。**
入力は [12 §5](12-OQ4E-G64-AUDIT.md) の焼き込み済みスナップショット。

```
.build/release/TurboFieldfareRepack --output scratch/ornith-oq4e-g64.gturbo \
    --source-snapshot ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-baked
.build/release/TurboFieldfareRepack --verify-install \
    --input-gturbo scratch/ornith-oq4e-g64.gturbo
```

---

## 1. 出口条件

| 条件 | 結果 |
| --- | --- |
| `--verify-install` が緑 | **緑。**47 ファイル / **20,494,912,547 B (20.49 GB)** |
| `expertStride == 1_769_472` | **一致** (16 KiB × 108、余りなし) |
| ファイルサイズが [01 §5-3](01-MODEL.md) と一致 | **§2 参照。**内訳は上流とバイト一致。合計が §5-3 の「約 19.5 GB」より大きいのは、そこが**公式 MLX-4bit (int4 一律) の導出**だから — oQ4e-g64 は embed / lm_head / attention / shared expert が 8-bit ([12 §3](12-OQ4E-G64-AUDIT.md)) |

## 2. バイトの内訳 — 上流とバイト一致

| 区画 | install | 上流 (oQ4e-g64) | 差 |
| --- | ---: | ---: | --- |
| `packed_experts/layer_NN.bin` × 40 | 18,119,393,280 | 18,119,393,280 | **0** |
| `model_weights.bin` | 2,340,141,312 | 2,340,059,392 (core 1,259,370,752 + embed/lm_head 1,080,688,640) | **+81,920 = 索引 5 ページ** |
| `packed_experts/layout.json` | 22,493,846 | — | 生成物 |
| `tokenizer/` (tokenizer.json / config / chat_template.jinja ほか) | 12,873,491 | — | 転記 |

**テキストの重みは 1 バイトも作り変えていない。**repack は並べ替えとページ揃えだけをする
という前提 ([02 §3](02-CHECKPOINTS.md)) が、実物で確認できた。

外したもの: **vision 333 本 (893 MB)** と **MTP 42 本 (503 MB)**。
どちらも `.gturbo` の別セクション ([03 §6](03-DESIGN.md)) に入るもので、Phase 7 / 9 の作業。
audit に `tensors_dropped_multimodal` / `tensors_dropped_inline_draft` として本数が残る。

## 3. 実装したこと

**形式 (`TurboFieldfareFormat`)** — `arch` に 3 つ足した:

| 追加 | 中身 |
| --- | --- |
| `family` | `qwen3_5_moe`。**Gemma の manifest には書かない** (optional、`JSONEncoder` は nil を書かない) ので、既存の `.gturbo` のバイトは動かない |
| `layerKinds` | 層ごとの `full_attention` / `linear_attention`。既存の `fullAttentionLayerMask` は互換面として残し、**両者が食い違う manifest は弾く** — マスクしか知らない読み手が別のモデルを見ることになるため |
| `linearAttention` | `numKeyHeads 16 / numValueHeads 32 / keyHeadDim 128 / valueHeadDim 128 / convKernelDim 4 / layerCount 30` |

`flags.linearAttention` + `versionMinor 3` は**塔・ドラフターと同じ互換ゲート**。
再帰層を知らないランタイムは、`slidingWindow: 0` を SWA と読んで静かに壊れるのではなく、
未知フラグとしてモデルごと拒む。`slidingWindow == 0` は `linearAttention` のある族にだけ許す。

**repack:**

- `ArchInfo` に `qwen3_5_moe` のパーサ。`sliding_window` / `final_logit_softcapping` /
  `num_global_key_value_heads` / `top_k_experts` の無い config を、Gemma のパーサに
  「キーが無い」と投げさせずに読む。`shared_expert_intermediate_size` を dense FFN の幅として使う
- `routedExpertRole` が `.mlp.switch_mlp.` も見る (計画どおり 1 文字列)
- **`language_model.mtp.*` を先に外す。**同梱ドラフターの `layers.0` は本体の `layers.0` では
  ないので、文字列を足しただけだと**層 0 の routed expert が二重**になって planner が落ちる。
  計画の「名前寄せは 1 文字列」は**ここだけ足りなかった**
- resident の並びに Qwen のスロット順を追加、router / shared expert のビット幅を両族の綴りから拾う
- 焼き込み済みスナップショットの index digest を `SourceFingerprint` に pin

**テスト:** 形式の検証 9 本 (`GTurboLinearAttentionManifestTests`、Gemma manifest に
3 キーが出ないことの検査を含む) と、**1/10 スケールの合成 Qwen スナップショット**
(`SyntheticQwenSnapshot`: 4 層のうち 3 層が線形、experts 2、MTP と vision 同梱) を
planner に通す 5 本。パッケージテスト全体 **1265 件が緑**。

## 4. 途中で分かったこと

### 4-1. `layout.json` は 16 MB に収まらない

`packed_experts/layout.json` は層 × エキスパートに比例する
(エキスパート 1 個あたり約 2.2 KB)。**Gemma 30×128 = 8.5 MB、Ornith 40×256 = 22.5 MB。**
書き手・検証器・ランタイム読み手の 3 箇所にあった 16 MB の上限を
`GTurboFormatV1.packedExpertsLayoutMaxBytes = 64 MB` に集約して上げた。
上限そのものは残す (信用しない入力の境界なので)。

### 4-2. ★ attention のビット幅が層ごとに違う

`oQ4e-g64` の attention は **4-bit と 8-bit が混ざっている** (**実測(手元)**):

| ロール | 4-bit | 8-bit |
| --- | ---: | ---: |
| `self_attn.{q,k,v}_proj` | 5 層 | 6 本 (5 層 + MTP) |
| `self_attn.o_proj` | 6 | 5 |
| `linear_attn.in_proj_qkv` | 25 | 5 |
| `linear_attn.{in_proj_z, out_proj, in_proj_a, in_proj_b}` | 0 | 30 ずつ |

imatrix が層ごとに配ったビット ([02 §4](02-CHECKPOINTS.md)) がそのまま出ている。

**ここが Phase 3 に効く。**manifest の `quant.attention` は**スロット 1 個で 1 つの
ビット幅しか書けず**、`Model.swift` は全 attention テンソルをその 1 個に対して検証する
(実測 = ソース)。**今のランタイムはこのモデルを受け付けない。**
`ManifestReader.validateQuant` の許容表も `embedding: [4]` / `attention: [4]` 固定で、
oQ4e-g64 の 8-bit を弾く。

**逃げ道はある:** resident 索引は各テンソルの `sizeBytes` / `shape` / `scaleSize` を
持っているので、**ビット幅と group はテンソルごとに導出できる** (形式変更は要らない)。
**→ 決着した ([18](18-MIXED-BITS.md))。索引から導くことにし、manifest のスロットは
幅の上限を述べるだけになった。本線は `Model.load` を通る。**
**公式 MLX-4bit は attention 一律 4-bit なので、この論点は oQ4e-g64 固有。**

### 4-3. 4-bit の対称性はあと 1 グループで成立しなかった

`SymmetricProbe` の結果 (**実測(手元)**):

```
affine: 1 of 7,946,240 groups break bias == -8 * scale
        (first in language_model.model.embed_tokens)
```

**794 万グループのうち 1 個**だけが `sym` 版のバイト節約 (4.5 → 4.0 bit/weight、
`docs/mtp/44-W1-WEIGHT-DIET.md`) を止めている。破れているのは routed expert ではなく
**embed_tokens** で、これは 8-bit の区画。判定が 4-bit の格子だけを見るように
なっているかは Phase 3 で確認する余地がある (今回は既定のまま affine で通した)。

## 5. この Phase が動かした結論

| 対象 | 更新 |
| --- | --- |
| [04](04-PHASES.md) Phase 1 | **完了。**GPU 不要の作業は全部終わった |
| [03 §1-2](03-DESIGN.md) 名前寄せ | 「1 文字列」+ **MTP の切り出し**が要った (§3) |
| [03 §1-3](03-DESIGN.md) ディスク | 実測 20.49 GB (テキストのみ)。install の門 26 GB は妥当 |
| Phase 3 の宿題 | **混在ビット幅の受け入れ** (§4-2) → **片づいた** ([18](18-MIXED-BITS.md)) |
| 形式 v1 | minor 3 / `flags.linearAttention` / `layout.json` 64 MB |
