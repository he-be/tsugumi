# 30. 出荷版 MTP ヘッドは学習されていない — 差し替え候補と形式適合 (実測、2026-08-22)

上流 [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
の [discussion #10](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/discussions/10)
「`mtp.*` tensors look like random init, not trained weights」を受けて、
**手元のウェイトで検算した**。指摘は本物で、しかも量子化の副作用ではなく
**上流の公式 bf16 の時点でそうなっている**。

本書は (1) その検算、(2) 差し替え候補の選別と**形式適合**、(3) mlx-lm の
MTP 実装 PR を参考資料としてどう読むか、を持つ。**受理率は 1 つも測っていない** —
引用する受理率はすべて外部の公表値で、本ランタイム上の値ではない。
既定は変えない。判断はユーザー。

---

## 0. 結論を先に

| # | 論点 | 結論 |
| --- | --- | --- |
| 1 | 出荷版 MTP ヘッドは学習済みか | **されていない。**上流 bf16 の `mtp.*` は全テンソルが std ≈ 0.0200・尖度 ≈ 3.00 の純ガウス。学習済みの本体は同じ種類で std 0.0066〜0.0166・尖度 3.1〜68.8 と散る (**実測(上流)** §1-1、**実測(手元)** §1-2) |
| 2 | 量子化のせいではないか | **違う。**上流 raw bf16 と oQ4e 逆量子化の std が小数第 5 位まで一致する。同じ 4bit/g64 で本体 layer-0 の switch_mlp は尖度 5.65 に残るので、量子化がガウスに均したのではない (§1-2) |
| 3 | 一番強い構造証拠 | **256 エキスパートが統計的に見分けられない。**expert 間の std のばらつきが **CV 0.69%** (本体は 4.07% / 14.07%)、`o_proj` の行ノルム CV **1.58%** (本体は 15.1%)。i.i.d. ドローの指紋である (§1-2) |
| 4 | norm が ~1.0 に見える件 | **規約の違い。**上流は **zero-centered gamma** (`w−1` を保存、実効スケール `1+w`) で、oQ4e は変換時に +1 済み。`max\|oQ4e−(上流+1)\| = 3.9e-03` = bf16 の 1 ULP。README #11 (c) の「Qwen も `1+w` 規約」と同じ話 (§1-3) |
| 5 | 差し替え候補 | **実質 2 つ。**`shisa-ai/…-MTP-ONLY` (KL 蒸留、19 本 BF16、1.689 GB) と Qwen3.6 のドナーヘッド。`EryriLabs/…-BigBang-MTP` は **`mtp.fc.weight` の sha256 がドナーと完全一致**するドナーの再梱包で、第 3 の選択肢ではない (§2-1) |
| 6 | どれを採るか | **shisa の 12K 蒸留。**公表値で native 32.19% → ドナー直グラフト 48.34% → **60.51%** (コードは 69.27%)。同じ形式・同じサイズで全項目上回る。学習済みであることは手元でも確認した (`mtp.fc` 尖度 **504.9**、§2-2) |
| 7 | 形式は乗るか | **乗る。19 本 → oQ4e-g64 に既にある 42 本ちょうどに 1:1。**平文 9 + 量子化 11×3 = 42。量子化対象の入力次元はすべて 64 の倍数で、g64 ビルドの override 314 本は全部 8bit/g64 = `supportedGroupSizes = [32,64]` の内側 (§3-1) |
| 8 | 語彙互換は論点か | **ならない。**ヘッドに語彙サイズのテンソルが 1 本も無い (`mtp_use_dedicated_embeddings: false`、embed/lm_head は本体と共有) (§3-1) |
| 9 | 推測で置いた所はあるか | **無い。**融合 `gate_up_proj` の分割順・norm の +1・量子化の代償の 3 点はすべて実測で潰した。分割順は本体層が同じ融合形なので**手元の oQ4e と突き合わせて相関 0.995** で確定 (§3-2) |
| 10 | 量子化の代償 | **無い。**8b/g64 で相対 L2 0.0055〜0.0062 (既存の 8b/g128 0.0086 より良い)、4b/g64 で 0.093 — 対照に取った**上流本体の学習済み expert** の 0.092〜0.098 と同じ (§3-2) |
| 11 | mlx-lm の PR は使えるか | **配線とテンソル変換はほぼそのまま、状態巻き戻しは設計案として使える。**ただし [#1740](https://github.com/ml-explore/mlx-lm/pull/1740) は **2026-08-21 に未マージで close**、親の [#990](https://github.com/ml-explore/mlx-lm/pull/990) も CHANGES_REQUESTED。却下理由は数学ではなく構造の結合 (§4) |
| 12 | PR から拾った一番大きい未確認 | **本体 hidden を MTP に渡すとき pre-norm か post-norm か。**#1740 は pre-norm、下流の同一成果物 A/B は **post_norm 80.30% / pre_norm 75.24%**。**受理率 5 ポイントの分岐**で、[03 §6-4](03-DESIGN.md) は定数ではなくスイッチとして持つべき (§4-3) |
| 13 | Phase 7 への波及 | **[29 §0 #7](29-MTP-PREFETCH-OUTLOOK.md) の「一番大きい未知 = 受理長 a」は、測らずに向きが出た。**乱数初期化のヘッドの受理は当てずっぽうで、出荷ヘッドのまま Phase 7 に進んでも取り分は無い。**ヘッドを差し替えてから測る**のが順序 (§5) |
| 14 | 差し替えれば動くのか | **動かない。**移植はウェイトを正しくするだけで、**Qwen 側に MTP 推論経路がまだ無い** (`DraftRepackPlanner` は Gemma の密 MLP ドラフター用)。Phase 7 の作業量は変わらない (§6) |

---

## 1. 出荷版 MTP ヘッドは乱数初期化である

### 1-1. 上流 bf16 での直接確認 (**実測(上流)**)

`~/LLM/Ornith-1.5-35B-A3B-bf16-partial/` ([11](11-OQ4E-G64-REBUILD.md) で
HTTP Range 抽出した置き場) に上流の `mtp.*` が 8 本入っていた。
**量子化を 1 度も経ていない原本**なので、これが一番強い証拠になる。

| 上流 raw bf16 | std | 尖度 |
| --- | ---: | ---: |
| `mtp.layers.0.mlp.shared_expert.down_proj` | 0.01995 | 2.99 |
| `mtp.layers.0.mlp.shared_expert.gate_proj` | 0.01979 | 3.00 |
| `mtp.layers.0.mlp.shared_expert.up_proj` | 0.01999 | 3.01 |
| `mtp.layers.0.mlp.shared_expert_gate` | 0.01907 | 3.02 |
| **対照** `model.…layers.22.mlp.shared_expert.gate_proj` | 0.00792 | 13.90 |
| **対照** `model.…layers.32.mlp.shared_expert.gate_proj` | 0.00973 | 15.50 |
| **対照** `model.…layers.37.mlp.shared_expert.gate_proj` | 0.01432 | 5.38 |

尖度 3.00 ちょうどは外れ値ゼロの純ガウス。学習を通った同じ種類のテンソルは
5.4〜15.5 に散る。

### 1-2. oQ4e の MTP 42 本すべて (**実測(手元)**)

`oQ4e-mtp` の `mtp.*` を全数逆量子化した。**形も役割も違う 11 本の重みが
std 0.01905〜0.02009 に収まり、尖度は 2.91〜3.05。**

| oQ4e の MTP | bits/g | std | 尖度 |
| --- | --- | ---: | ---: |
| `switch_mlp.gate_proj` (256,512,2048) | 4/64 | 0.02009 | 2.94 |
| `switch_mlp.up_proj` (256,512,2048) | 4/64 | 0.02008 | 2.94 |
| `switch_mlp.down_proj` (256,2048,512) | 4/64 | 0.01994 | 2.91 |
| `self_attn.q_proj` (8192,2048) | 8/64 | 0.01993 | 3.00 |
| `self_attn.k_proj` / `v_proj` (512,2048) | 8/64 | 0.01989 / 0.02004 | 3.00 / 3.00 |
| `self_attn.o_proj` (2048,4096) | 8/64 | 0.01996 | 3.00 |
| `shared_expert.{down,gate,up}_proj` | 8/128 | 0.01995 / 0.01979 / 0.01999 | 2.99 / 2.99 / 3.01 |
| `shared_expert_gate` (1,2048) | 8/64 | 0.01905 | 3.02 |
| `mtp.fc.weight` (2048,4096) | BF16 | 0.0195 | 3.05 |
| `mtp.layers.0.mlp.gate.weight` (256,2048) | BF16 | 0.0194 | 2.93 |

対照 (同じチェックポイントの本体) は **std 0.0066〜0.0166・尖度 3.1〜68.8**。
本体 router (`mlp.gate.weight`) は layer 0/1/24 で std 0.0232/0.0240/0.0168、
尖度 6.72/7.29/19.95。

**量子化の交絡は無い。**上流 raw bf16 (§1-1) と逆量子化値が
`0.01995 / 0.01979 / 0.01999` と小数第 5 位まで一致する。さらに**同じ 4bit/g64**
で本体 layer-0 の `switch_mlp.gate_proj` は尖度 5.65 に残るので、
「4bit がガウスに均した」は成立しない。

構造の指標も同じ向きを指す:

| 指標 | MTP | 本体 (学習済み) |
| --- | ---: | ---: |
| `switch_mlp.gate_proj` の 256 エキスパート間 std の CV | **0.69%** (0.01993〜0.02045) | layer 0: 14.07% / layer 24: 4.07% |
| `o_proj` 出力行ノルムの CV | **1.58%** | layer 3: 15.15% / layer 7: 15.09% |

256 個のエキスパートが統計的に見分けられない = 同一分布からの独立ドロー。
行ノルムの集中も乱数行列の理論どおりで、学習済みの本体とは 1 桁違う。

### 1-3. norm の規約 — zero-centered gamma (**実測(上流)** + **実測(手元)**)

上流と oQ4e で norm の見え方が違う。**規約の差であって、結論は動かない。**

- **上流は `w−1` を保存する** (zero-centered gamma、実効スケール `1+w`)。
  mlx-lm の変換が +1 して絶対値形にする。検算: 3 本の MTP norm で
  `max|oQ4e − (上流+1)| = 3.9e-03` (= bf16 の 1 ULP)、`max|oQ4e − 上流| = 1.004`。
  README #11 (c) の「RMSNorm は Qwen も `1+w` 規約」と同じ話。
- **その規約で読むと、MTP の norm は初期値からほとんど動いていない。**

| zero-centered 形 | mean | std |
| --- | ---: | ---: |
| MTP `input_layernorm` | 0.00006 | 0.00302 |
| MTP `post_attention_layernorm` | 0.01483 | 0.00356 |
| MTP `q_norm` / `k_norm` | 0.00390 / 0.00392 | 0.00680 / 0.00681 |
| **対照** 本体 `layers.9.input_layernorm` | −0.00029 | 0.12890 |
| **対照** 本体 `layers.9.post_attention_layernorm` | 0.18072 | 0.18551 |
| **対照** 本体 `layers.7.input_layernorm` | −0.11503 | 0.11986 |

`mtp.norm.weight` は oQ4e で **2048 要素すべてが定数 1.0234 (std 0.0000)** =
上流 0.0234。要素ごとの勾配を 1 度も受けていない。

discussion #10 と shisa の README が言う「norm が 0.02 付近」はこの
zero-centered 形の値で、比較対象の Qwen3.6 の 0.87〜1.93 も同じ形。
**比較自体は成立している** — 「本来 ~1.0 のはず」という言い方が緩いだけである。

---

## 2. 差し替え候補

### 2-1. 4 リポジトリの選別 — 実質 2 つ (**実測(上流)**)

| リポジトリ | 中身 | 判定 |
| --- | --- | --- |
| **`shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY`** | `model-mtp.safetensors` 単体。19 本すべて BF16、844,640,768 param / 1.689 GB (payload 1,689,281,536 B) | **本命** |
| `Qwen/Qwen3.6-35B-A3B` | 本体 26 シャード内に `mtp.*` 19 本 | ドナー (無学習グラフト) |
| `EryriLabs/Ornith-1.5-35B-A3B-BigBang-MTP` | `mtp.fc.weight` の sha256 が**ドナーと完全一致** (`5015bb3b80ffa82d`)。norm 7 本も小数 4〜5 桁まで一致 | **ドナーの再梱包。**第 3 の選択肢ではない |
| `shisa-ai/…-MTP` | 71.9 GB のマージ済み本体。`mtp-merge-manifest.json` が `native_mtp_removed: 785` / `mtp_count: 19` を記録 | 不要 (ヘッドは MTP-ONLY と同一) |
| `shisa-ai/…-MTP-FP8` | FP8 | Mac 不可 |

公表値 (vLLM、MTP3、temp 0、concurrency 1、10 プロンプト × 3 反復、≤384 トークン):

| ヘッド | コード受理 | 受理長 | P1/P2/P3 | 全体 |
| --- | ---: | ---: | --- | ---: |
| Ornith 純正 (無学習) | 37.20% | 2.116 | 85.2 / 22.9 / 3.5% | 32.19% |
| Qwen3.6 直グラフト | 50.19% | 2.506 | 80.8 / 44.7 / 25.1% | 48.34% |
| 5K KL 蒸留 | 66.99% | 3.010 | 89.2 / 67.3 / 44.5% | 58.62% |
| **12K KL 蒸留 (MTP-ONLY)** | **69.27%** | **3.078** | 92.2 / 71.1 / 44.5% | **60.51%** |

shisa 側は純正ヘッドの ShareGPT 受理を BF16 21.5% / 公式 FP8 21.7% と並べて
**FP8 由来ではない**ことも示している。**いずれも本ランタイム上の値ではない。**

shisa のヘッドは「Qwen3.6 のヘッドで初期化し、Ornith-1.5 の隠れ状態に
全語彙 KL 蒸留で再整合」させたもの。ドナー直グラフトは
「1.5 タワーは Qwen3.6 と cos で 0.2〜1% しか離れていない」に乗っているだけで、
Ornith への整合は入っていない。

### 2-2. shisa ヘッドが学習済みであることの確認 (**実測(手元)**)

Range でヘッダとテンソルを取り、§1 と同じ物差しを当てた。

| shisa MTP-ONLY | mean | std | 尖度 |
| --- | ---: | ---: | ---: |
| `mtp.fc.weight` | 0.00008 | 0.00892 | **504.89** |
| `mtp.layers.0.mlp.gate.weight` | 0.00007 | 0.00974 | 6.99 |
| `shared_expert.{down,gate,up}_proj` | ≈0 | 0.00857 / 0.00916 / 0.01369 | 15.87 / 5.35 / 5.91 |
| `self_attn.{k,v,o}_proj` | ≈0 | 0.01854 / 0.02131 / 0.01888 | 5.57 / 4.33 / 24.59 |
| routed `experts` gate / up / down (4 expert 標本) | ≈0 | 0.01342 / 0.01419 / 0.01327 | 3.77 / 3.40 / 4.78 |
| `mtp.norm.weight` (zero-centered) | 1.92405 | 0.29671 | 6.71 |
| `pre_fc_norm_embedding` / `pre_fc_norm_hidden` | −0.73409 / −0.50256 | 0.05790 / 0.09536 | 3.32 / 11.28 |
| `input_layernorm` / `post_attention_layernorm` | −0.09426 / 0.86825 | 0.15668 / 0.21286 | 6.94 / 8.35 |

出荷版の「全部 0.0200 / 3.00」とは別物である。テンソルごとに std が分化し、
norm に実質的なばらつきがある。

**ドナーとの差分も取れた。**Qwen3.6 の同名 norm は
−0.09501 / 0.86859 / 0.74179 / 0.76725 / 1.92513 / −0.72665 / −0.50630 で、
shisa 側と 1e-3 程度しか違わない — KL 蒸留は主に射影を動かし、norm はほぼ据え置き。
`mtp.fc` の sha256 は両者で異なる (蒸留が入っている証拠)。

**Ornith が Qwen3.6 派生であることの傍証**も出た: `layers.9.input_layernorm` の
mean が Qwen3.6 −0.00039 / Ornith −0.00029、std は両方 0.12890。

---

## 3. 形式適合 — 19 本 → 42 本

### 3-1. 対応表

供給側 19 本は、`oQ4e-g64` に**既にある 42 本**へちょうど 1:1 で写る。

| 供給側 (BF16) | 変換先 | bits/g |
| --- | --- | --- |
| `mtp.fc.weight` (2048,4096) | そのまま | BF16 |
| `mtp.layers.0.mlp.gate.weight` (256,2048) | そのまま (router) | BF16 |
| norm 7 本 | **+1 して**そのまま | BF16 |
| `self_attn.{q,k,v,o}_proj` | 同名 | 8/64 affine |
| `shared_expert.{gate,up,down}_proj`、`shared_expert_gate` | 同名 | 8/64 affine |
| `experts.gate_up_proj` (256,1024,2048) | **分割** → `switch_mlp.gate_proj` / `up_proj` | 4/64 affine |
| `experts.down_proj` (256,2048,512) | `switch_mlp.down_proj` | 4/64 affine |

平文 9 + 量子化 11 × 3 (weight/scales/biases) = **42** ✓

- 量子化対象の入力次元 (2048 / 512 / 4096) はすべて 64 の倍数。
- `oQ4e-g64` の per-tensor override は **314 本すべて 8bit/g64**、global は 4bit/g64 affine。
  `Quantization.supportedGroupSizes = [32,64]` の内側 ([18](18-MIXED-BITS.md))。
- **ヘッドに語彙サイズのテンソルが 1 本も無い** (`mtp_use_dedicated_embeddings: false`、
  lm_head も持たない) ので、語彙互換は論点にならない。

### 3-2. 推測せず実測で潰した 3 点 (**実測(手元)**)

**(a) 融合 `gate_up_proj` の分割順。**上流 Ornith は**本体 40 層も同じ融合形**なので
(MTP だけが per-expert 785 本)、mlx-lm が既に分割済みの手元の oQ4e と突き合わせられる。
上流 layer-0 expert-0 を Range で取って 3 仮説を当てた:

| 仮説 | 相関 |
| --- | ---: |
| **連続 `[gate(0:512); up(512:1024)]`** | **0.99501 / 0.99554** |
| 連続・入替 `[up; gate]` | −0.01018 / −0.01010 |
| interleave (stride 2) | 0.02619 / −0.00036 |

残差は 4bit 量子化ノイズ。mlx-lm PR #1740 の `_unfuse_experts`
(`mid = shape[-2] // 2`、`[..., :mid, :]` = gate) がコード側で同じことを言っている (§4-2)。

**(b) norm の +1。**§1-3 のとおり上流は zero-centered。shisa のヘッドは
Qwen3.6 のヘッドで初期化されており、そのドナーの本体 norm が near-zero mean
(§2-2) なので**同じ規約**。移植時に +1 が要る。

**(c) 量子化の代償。**

| 変換 | 相対 L2 |
| --- | ---: |
| 8b/g64: `k_proj` / `q_proj` / `o_proj` | 0.00581 / 0.00555 / 0.00613 |
| 8b/g64: `shared_expert.gate_proj` / `down_proj` / `shared_expert_gate` | 0.00560 / 0.00622 / 0.00567 |
| 4b/g64: routed gate / up / down (4 expert 標本) | 0.09371 / 0.09273 / 0.09324 |
| **対照** 4b/g64: 上流本体 layer-0 expert-0 の gate / up | 0.09833 / 0.09226 |

8bit 側は既存の 8b/g128 (`shared_expert.down_proj` 0.0086、[02 §5](02-CHECKPOINTS.md))
より良い。4bit 側は**本体の学習済み expert と区別がつかない**。形式起因の劣化は無い。

### 3-3. 配線側

repack の経路は既に MTP を知っている:

- `RepackPlanner.swift:183` — `isInlineDraftTensorName` が
  `language_model.mtp.` を「本体に同梱されたドラフター」として分類する。
- 同 `:171-176` — `.mlp.switch_mlp.` の gate/up/down 対応。

---

## 4. mlx-lm の MTP 実装 PR をどう読むか

[#1740](https://github.com/ml-explore/mlx-lm/pull/1740)
"Native Qwen MTP speculative decoding with transactional prompt-cache reuse"
(PhilipJohnBasile、2026-08-16 起票、7 ファイル +1576/−132)。
親は [#990](https://github.com/ml-explore/mlx-lm/pull/990) (AirRunner、2026-03-13)。

### 4-1. 状態: どちらもマージされていない

**#1740 は 2026-08-21 に zcbenz が close**、#990 は open のまま
同日 CHANGES_REQUESTED。**仕様ではなく参考実装として読む。**

却下理由は数学ではなく**構造の結合**である:

> There are 2 major problems with this implementation: sampler is coupled with
> generation step, and module implementation is coupled with speculative decoding.
> … the first step would be making `ArraysCache` trimmable.

あるべき形として `make_draft_model` + 既存の `speculative_generate_step`、
そのために `ArraysCache` を trim 可能にせよ、と指定されている。
**本ランタイムは自前の forward runner を書くのでこの結合問題自体は当たらない**が、
「draft フラグを全層に通すのではなく状態側を trim 可能にする」という設計上の
指針は取っておく価値がある (#1740 は `n_confirmed` を全層に通している)。

### 4-2. そのまま効く所

**MTP ブロックの配線** — [03 §6-4](03-DESIGN.md) の第三者資料が 1 本増える。

```python
e = self.pre_fc_norm_embedding(embed_tokens(next_token_ids))
h = self.pre_fc_norm_hidden(hidden_states)
fused = self.fc(mx.concatenate([e, h], axis=-1))
```

concat 順は **[embedding, hidden]** で、Shiftedx の `mtp_contract`
(`concat_order: embedding_hidden`、[02](02-CHECKPOINTS.md)) と一致。
MTP 層 (`MTPDecoderLayer`) は **full attention のみ (GatedDeltaNet なし) +
`SparseMoeBlock`**、最後に `norm`、lm_head と embed_tokens は本体と共有 —
§3-1 で読んだ 42 本の在庫 (`self_attn.*` はあるが `linear_attn.*` は無い、
語彙テンソルゼロ) とぴったり合う。

**テンソル変換** — `qwen3_5_moe.py` の `sanitize` に両レイアウトがある:

- `_unfuse_experts` — `mid = gate_up.shape[-2] // 2`、`[..., :mid, :]` = gate、
  `[..., mid:, :]` = up。**§3-2 (a) の実測と同じ。**融合形 (Qwen3.6 / shisa) 用。
- `_stack_per_expert` — per-expert を stack。**Ornith 純正の 785 本**用。

**GDN 状態の巻き戻し** — Ornith の 30/40 層の急所。confirmed 区間と draft 区間を
`_process_chunk` で分けて処理し、境目で `(conv_state, ssm_state)` を
`cache.rollback_state` にスナップショット、棄却時に復元する。KVCache 側は単に trim。
**「線形注意はスナップショット、full attention は trim」という非対称**が要点。

**`MTPPromptCacheState`** — MTP ヘッド側のキャッシュは本体より**意図的に 1 位置
遅れる** (次の更新に後続の先頭トークンが要るため)。落とし穴として明示されている。

**`tests/test_mtp.py`** (504 行) — 挙動仕様として読める。

### 4-3. 割り引く所 / 拾った未確認

**pre-norm か post-norm かが割れている (一番大きい)。**#1740 は本体の
**正規化前** hidden を MTP に渡す (`return_hidden` が `normed` ではなく `hidden`)。
一方 waybarrios が下流 (vllm-mlx #660) の Qwen3.6-27B 変換で取った同一成果物 A/B は:

| MTP 入力 | 受理 | tok/s |
| --- | ---: | ---: |
| post-norm | 80.30% | 36.2 |
| pre-norm | 75.24% | 34.3 |
| post-norm 再実行 | 80.30% | 36.2 |

再実行がカウンタまで完全再現。彼らは `mtp_hidden_state_mode` を明示設定にした。
Shiftedx の `mtp_contract` も `base_hidden_variant: post_norm`。
**[03 §6-4](03-DESIGN.md) は定数ではなくスイッチとして持ち、当方で符号を測る。**

**検証数値は自己申告。**21/21・232/232・受理 90.64% はいずれも著者自身の
fork CI。唯一の第三者データ点 (gltanaka: 1.84x@1K / 2.18x@4K / 2.19x@16K、
40 タスク中 39 本がバイト一致) は **oMLX 0.6.1 + fcmeyer 変換 / Qwen3.8-27B /
M4 Max 128GB** という**別スタック**で、本人も「このブランチは動かしていない」と
断っている。

**本書の §1 には触れていない。**PR は「Qwen3.5/3.6/3.8 系は学習済み MTP ヘッドを
同梱している」を前提に書かれている。ヘッドの中身と配線は独立なので、
参考にする分に支障は無い。

---

## 5. 既存文書への波及

| 文書 | 現在の記述 | 本書との関係 |
| --- | --- | --- |
| [README](README.md) #12 | 「専用ドラフターを別リポジトリから取ってくる必要が無い」 | **崩れた。**同梱ヘッドは使えない。差し替えるなら外部リポジトリが要る (§2) |
| [02 §5](02-CHECKPOINTS.md) | oQ4e-mtp の採用理由 = MTP + vision | **MTP 側の根拠が崩れた。**vision は無傷なので採用自体は立つ |
| [29 §0 #7](29-MTP-PREFETCH-OUTLOOK.md) | 「一番大きい未知 = 受理長 a。1 つも測っていない。a が 1.0 に張り付くなら Phase 7 の期待値そのものが消える」 | **測らずに向きが出た。**出荷ヘッドでは a ≈ 1。§3-4 の CPU 先行測定は**ヘッドを差し替えてから**行う |
| [04 Phase 7](04-PHASES.md) | 全エキスパート常駐 + 行ごと状態書き出し | 前提にヘッド差し替えが 1 段増える (§6) |
| [03 §6-4](03-DESIGN.md) | hidden 契約 | **スイッチにする** (§4-3) |

運用ルールどおり、食い違いは番号の大きい本書が正。

---

## 6. 残る未確認と、次の一手

**未確認:**

1. **本ランタイム上の受理率。**§2-1 の公表値は BF16/FP8 ターゲット上の
   vLLM/CUDA 実測。**routed expert 4bit の当方ターゲットでの値は誰も測っていない。**
2. **pre-norm / post-norm の符号** (§4-3)。当方で 5 ポイント動くかは未確認。
3. **shisa ヘッドの実体は未取得。**Range でヘッダとスライスを読んだだけで、
   1.689 GB 本体は落としていない。
4. **Qwen 側の MTP 推論経路が無い。**`DraftRepackPlanner` は Gemma の密 MLP
   ドラフター用で、この MoE ブロックの形ではない。**移植はウェイトを正しくする
   だけで、ドラフトが動くようにはならない。**

**次の一手 (どれもユーザー判断):**

- **M0': ヘッドの差し替え。**`MTP-ONLY` を取得 → §3-1 の対応表で 42 本に変換
  (+1 と分割は §3-2) → `oQ4e-g64` に差した派生を作る。§3 の検算どおりなら
  ここに未知は無い。**カーネルは 1 本も要らない。**
- **M0: 受理長 a の CPU 先行測定** ([29 §3-4](29-MTP-PREFETCH-OUTLOOK.md))。
  float32 参照器 ([14](14-REFERENCE.md)) に `mtp.*` を足し、実生成のトレース上で
  draft top-1 と本体 top-1 の一致率を取る。**差し替え後のヘッドで**行い、
  pre/post-norm の両方で引けば §4-3 も同時に片づく。
- a の桁が出てから [29](29-MTP-PREFETCH-OUTLOOK.md) の在庫を再判定する。

---

## 7. コードと文書の根拠 (2026-08-22 に確認した現物)

| 事実 | 場所 |
| --- | --- |
| 上流 `mtp.*` 8 本の raw bf16 統計 | `~/LLM/Ornith-1.5-35B-A3B-bf16-partial/model.safetensors` (**実測(上流)**、抽出経緯は [11](11-OQ4E-G64-REBUILD.md)) |
| oQ4e の MTP 42 本の逆量子化統計・expert 間 CV・行ノルム CV | `~/LLM/Ornith-1.5-35B-A3B-oQ4e-mtp` (**実測(手元)**) |
| `max\|oQ4e − (上流+1)\| = 3.9e-03` | 同上 3 本の norm で検算 (**実測(手元)**) |
| `oQ4e-g64` の override 314 本が全部 8bit/g64、MTP は shared_expert 3 本のみ打ち直し・norm 未改変 | `~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64/config.json` + index (**実測(手元)**) |
| `MTP-ONLY` のヘッダ 19 本 BF16 / payload 1,689,281,536 B、各テンソル統計 | `shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY` を HTTP Range (**実測(上流)**) |
| 受理率・throughput の公表値、純正ヘッドの BF16/FP8 対照 | 同 README |
| `mtp.fc.weight` sha256 が Qwen3.6 と EryriLabs で一致 (`5015bb3b80ffa82d`)、shisa は別 (`6c78add4323c817e`) | 3 リポジトリを Range で照合 (**実測(上流)**) |
| Qwen3.6 の MTP norm 7 本と本体 norm、Ornith との一致 | `Qwen/Qwen3.6-35B-A3B` を Range (**実測(上流)**) |
| 融合 `gate_up_proj` の分割順 (相関 0.995) | 上流 layer-0 expert-0 を Range 取得 → 手元 oQ4e と突き合わせ (**実測(手元)**) |
| 8b/g64 と 4b/g64 の往復誤差、上流本体 expert の対照 | `mx.quantize` / `mx.dequantize` (**実測(手元)**) |
| `supportedGroupSizes = [32,64]` | `Sources/TurboFieldfare/Infrastructure/ModelIO/Quantization.swift:16` |
| `language_model.mtp.` を同梱ドラフターとして分類 / `switch_mlp` の gate-up-down 対応 | `Sources/TurboFieldfareRepack/Core/Planning/RepackPlanner.swift:183` / `:171-176` |
| PR の状態・却下レビュー・`MTPModule` / `_unfuse_experts` / `rollback_state` / `MTPPromptCacheState` | [mlx-lm#1740](https://github.com/ml-explore/mlx-lm/pull/1740) (close 2026-08-21) / [#990](https://github.com/ml-explore/mlx-lm/pull/990) |
| post-norm 80.30% / pre-norm 75.24% の A/B | #1740 の waybarrios コメント (下流 vllm-mlx#660) |
| 元の指摘 | [ornith-ai/Ornith-1.5-35B-A3B discussion #10](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/discussions/10) |
