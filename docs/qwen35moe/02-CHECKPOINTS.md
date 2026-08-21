# 02. チェックポイント — 候補と持ち物

上流の実体確認は 2026-08-21 (**実測(上流)**)。手元での検証・打ち直しの数字は
[10-MLX4BIT-AUDIT.md](10-MLX4BIT-AUDIT.md) と [11-OQ4E-G64-REBUILD.md](11-OQ4E-G64-REBUILD.md) が持つ。

---

## 1. 結論: 候補は 2 本、本線は未決定

| | **公式 MLX-4bit** (+ 打ち直し) | **oQ4e-g64** ([11](11-OQ4E-G64-REBUILD.md)) |
| --- | --- | --- |
| 実効サイズ | 19.51 GB (打ち直し込みで約 19.87 GB) | **21.86 GB** |
| MTP | **無い。**oQ4e から移植が要る | **入っている** (42 本、`switch_mlp` 積み済み 503 MB) |
| vision | **無い。**oQ4e から移植が要る | **入っている** (333 本 bf16 893 MB) |
| routed expert の校正 | **素の RTN** | **imatrix つき** (`oqe_code_multilingual` 128×512) |
| `in_proj_a`/`b` | 4-bit → **8-bit 打ち直しを決定済み** ([10 §5](10-MLX4BIT-AUDIT.md)) | **8-bit g64** (打ち直し後) |
| `embed`/`lm_head` | 4-bit (品質懸念、[10 §5](10-MLX4BIT-AUDIT.md)) | **8-bit g64** |
| shared expert | 4-bit (同上) | **8-bit g64** (打ち直し後) |
| RMSNorm の `+1` | **MLX が焼き済み** ([10 §4](10-MLX4BIT-AUDIT.md)) | **未照合。次の一手の #1** ([04](04-PHASES.md)) |
| 赤リスト (`group ∉ {32,64}` or `bits ∉ {4,8}`) | 0 本 | 0 本 |

**差は 2.35 GB で、その対価が「imatrix + MTP + vision + 高ビットの保護箇所」である。**
どちらを本線にするかは**まだ決めていない**。MTP と vision は本プロジェクトの主題であり
供給源は oQ4e(-g64) しか無いので、公式 MLX-4bit を本線にする場合も移植は oQ4e から行う。
決める前に必要な照合 (norm 規約・conv1d 軸順・router のビット幅) は
[04](04-PHASES.md) の「次の一手」1〜3。

> **注意:** 公式 MLX-4bit では `mlp.gate` (router) が **8-bit g64** で、§6 の表が
> oQ4e (打ち直し前) について言う「router は BF16」は当てはまらなかった。
> **oQ4e-g64 側の router がどちらなのかは未確認。**変換器が違えば結論も違う —
> 一方の照合結果をもう一方に流用してはいけない。

## 2. 手元にあるもの (2026-08-21 夜)

| 置き場 | 中身 | 実効サイズ | 状態 |
| --- | --- | ---: | --- |
| `~/LLM/Ornith-1.5-35B-A3B-MLX-4bit` | 公式 MLX 4-bit。テキスト専用 (vision / MTP なし)、打ち直し 0 本 | 19.51 GB | [10](10-MLX4BIT-AUDIT.md) で検証済み |
| `~/LLM/Ornith-1.5-35B-A3B-oQ4e-mtp` | oQ4e。**MTP 42 本 + vision 333 本つき**、imatrix つき。非互換 248 本あり | 21.61 GB | 取得・検証済み ([11 §1](11-OQ4E-G64-REBUILD.md)) |
| `~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64` | 上の非互換 248 本を 8-bit g64 で打ち直したもの。**非互換 0 本** | 21.86 GB | 作成済み ([11 §2](11-OQ4E-G64-REBUILD.md))。**推論は 1 度も通していない** |
| `~/LLM/Ornith-1.5-35B-A3B-bf16-partial` | 上流 bf16 の部分抽出 (HTTP range)。**518 本 3.50 GB**。打ち直し対象 248 本の bf16 原本は 248/248 揃っている | 3.50 GB | [10 §6](10-MLX4BIT-AUDIT.md) / [11 §2](11-OQ4E-G64-REBUILD.md) |

## 3. なぜ変換器を書かずに済んだか (経緯の要約)

1. **本リポジトリの repack は量子化しない** (実測 = ソース)。
   `SupportedModelSource.repoID = "mlx-community/gemma-4-26b-a4b-it-4bit"` —
   既存の `RepackPlanner` は「dtype が `u32` で名前が `.weight` で終わるものは
   4-bit パック済み、`.scales` / `.biases` が対」という MLX の規約を前提に、
   **並べ替えとページ揃えだけ**をする。`bits / group_size / mode` は上流
   `config.json` の `quantization` から読む。ストリーミング install
   (range copy + checkpoint + fingerprint) は「バイトをそのまま運ぶ」ことに
   依存していて、途中で値を作る余地が無い。**量子化器を足すなら Swift 側ではなく
   Python 側** (`Scripts/` の fetch 系と同じ立て付け) — これは当初から決めていた
2. 当初は「Ornith は bf16 しか無い」という前提で、**bf16 → MLX affine 4-bit の
   Python 変換器 (素の RTN)** を Phase 1 の主題にしていた
3. **oQ4e-mtp の発見**で 3 案になった:

   | 案 | 中身 | ダウンロード | 状態 |
   | --- | --- | ---: | --- |
   | 案「素の RTN を書く」 | numpy で bf16 → MLX 規約の変換器を書く | 71.9 GB | **不要になった** |
   | 案「公開版を打ち直す」 | oQ4e を食い、非互換 248 本だけ 8-bit g64 で再量子化 | 21.6 GB | **実行済み** ([11](11-OQ4E-G64-REBUILD.md))。入力は逆量子化ではなく**上流 bf16 原本の range 取得** |
   | 案「oQ を自分で回す」 | omlx の oQ を `gs()` 64 固定・bits {4,8} に制約して imatrix つきでローカル実行 | **71.9 GB** | 保留。校正データを差し替えたい / GPTQ も掛けたいとなったときの道。**上流 bf16 全体を要求する唯一の用途** (§7)。GPU を使うので Phase 6 以降 |

4. さらに**公式 MLX-4bit の発見**で「打ち直し 0 本」の候補が加わり ([10](10-MLX4BIT-AUDIT.md))、
   現在の 2 候補 (§1) になった

当初の変換器仕様から生き残ったもの: 量子化の式 (下記) は打ち直し器として実装済み、
焼き込み (`q_norm` × 1/16 など) は**どの候補でも要る** ([03 §1](03-DESIGN.md))。

**量子化の式 (affine 4/8-bit, group 64, 入力次元方向にグループ):**

```
行 r の入力次元 K を 64 ずつに切る。各グループで
  lo, hi = min(w), max(w);  scale = (hi-lo)/(2^bits - 1);  bias = lo
  q = clip(round((w - bias)/scale), 0, 2^bits - 1)
uint32 にリトルエンディアンで詰める (MLX の並び)。scales / biases は bf16。
```

`K % 64 == 0` は全対象で成立 (2048, 4096, 512, 1024 のいずれか、**実測(上流)** の shape で確認済み)。

## 4. oQ とは何か

`docs/oQ_Quantization.md` と `omlx/oq.py` (8,389 行) から (**実測(上流)** = ソース):

1. **校正推論で感度を測る (imatrix)。**
   `sensitivity = MSE(float_output, quantized_output) / mean(float_output²)`。
   出力の大きさで正規化するので、残差が積み上がる後段の層が不当に敏感に見えない。
   今回のアップロードは `oqe_code_multilingual` を 128 サンプル × 512 トークンで収集
   (`oq_imatrix_report.json`)。`Linear` 511 本 + `SwitchLinear` 123 本を計装している
2. **感度で bit を配る。**最大感度比 ≥50% → base+4 bit、≥20% → base+2、それ未満 → base+1。
   bpw 予算に入らなければ 8 → 6 → 5 と落とす貪欲割り当て
3. **固定の保護。**`lm_head` 8-bit / MoE router / vision / SSM パラメータ
4. **routed expert は base のまま。**「バイトあたりの品質改善効率が悪い」から昇格させない
5. **oQ+ は GPTQ を掛ける。**列ごとに丸め、逆ヘッセ行列で残り列を補正する。
   MoE 向けに**層内の全エキスパートで Hessian を共有したバッチ GPTQ** を持ち、
   Qwen3.5-35B-A3B で 90 分 → 6 分と書いてある

**決定的なのは 1 と 5 の組み合わせ:**「重要度で重みを付けた scale/bias 探索」も
「GPTQ 補正」も、**書き出すバイトの値を変えるだけでフォーマットを変えない。**
4-bit g64 affine は 4-bit g64 affine のままである。だから品質だけをランタイム無改造で貰える。

## 5. oQ4e-mtp の実体 (実測(上流))

`total_size = 21,613,054,816 B (21.61 GB)`、テンソル 2,052 本、5 シャード。

| 区画 | バイト | 割合 | 量子化 | 本ランタイム |
| --- | ---: | ---: | --- | --- |
| routed experts (`mlp.switch_mlp.{gate,up,down}_proj`) | **18,119,393,280** | 83.8% | **4-bit affine g64** | **そのまま乗る** |
| embed_tokens + lm_head | 1,080,688,640 | 5.0% | 8-bit affine g64 | 乗る (int8 経路あり) |
| core (attn / linear_attn / shared / router / norm) | 1,016,785,152 | 4.7% | **混在 (下記)** | **一部乗らない → 打ち直し済み** |
| vision_tower | 893,142,496 | 4.1% | BF16 | 乗る (現行 tower も bf16) |
| MTP (`language_model.mtp.*`) | 503,045,248 | 2.3% | switch_mlp は 4-bit g64 | 乗る |

**routed expert 1 本のバイト配置** (ヘッダから):

```
gate_proj  weight U32  [256, 512, 256]   scales/biases BF16 [256, 512, 32]
up_proj    weight U32  [256, 512, 256]   scales/biases BF16 [256, 512, 32]
down_proj  weight U32  [256, 2048, 64]   scales/biases BF16 [256, 2048, 8]
→ 1 エキスパート = 589,824 × 3 = 1,769,472 B = 16 KiB × 108
```

**[01 §5-2](01-MODEL.md) の導出と完全一致** (40 層合計 18,119,393,280 B も一致)。

**そのままは乗らなかった 248 本** (`config.json → quantization` の per-tensor override と
ヘッダの shape から確定。**打ち直し済み** — [11](11-OQ4E-G64-REBUILD.md)):

| テンソル | 本数 | oQ の指定 |
| --- | ---: | --- |
| `linear_attn.out_proj` | 30 | 5-bit **g128** |
| `linear_attn.in_proj_z` | 29 | 5-bit **g128** |
| `linear_attn.in_proj_a` / `in_proj_b` | 21+21 / 8+8 | 5-bit **g128** / 6-bit g64 |
| `linear_attn.in_proj_qkv` | 3 / 1 | 5-bit **g128** / 6-bit g64 |
| `mlp.shared_expert.{gate,up,down}_proj` | 120 | 8-bit **g128** |
| `self_attn.{q,k,v}_proj` | 3 | 6-bit g64 / 6-bit **g128** |
| **合計** | **248** | **730,464,256 param / 533,434,368 B** |

なぜ g128 が混ざるか: `oq.py` の `gs()` が **`num_experts >= 150` なら 128 を返す**。
Ornith は 256 なので、routed expert 以外は自動的に g128 側に倒れている。
routed expert だけは予算計画側の経路が base (`group_size 64`) を保っている。
本ランタイムは (a) アフィン group size がシェーダライブラリ全体の**コンパイル時定数**
(32 か 64) で**モデル内混在が原理的に不可能** (b) 5-bit / 6-bit のカーネルが無い。

## 6. omlx から他に確認できた細部

| 事実 (実測(上流)) | 本計画への影響 |
| --- | --- |
| Gated DeltaNet prefill は「chunkwise (WY) より blocked-sequential が速い」が omlx の実装上の結論 | **最大の設計上の収穫。**カーネルの幾何ごと写す ([03 §2-6](03-DESIGN.md)) |
| oQ4e の `mlp.gate` (router) は **BF16 のまま** | `router_gemv_gemma4_bf16_r4` 経路 (QAT 用) が当たる。**ただし公式 MLX-4bit は 8-bit g64、oQ4e-g64 は未確認** (§1 の注意) |
| `A_log` / `dt_bias` / `conv1d.weight` / 全 norm は BF16 | 小テンソルは量子化しない、の割り当てと一致 |
| oQ4e の `in_proj_a` / `in_proj_b` は 5〜8 bit | 打ち直しで 8-bit g64 に統一済み ([11 §2](11-OQ4E-G64-REBUILD.md)) |
| `mtp.fc.weight` は BF16 (16.8 MB) | 量子化しない |
| `conv1d.weight` の shape が `[8192, 4, 1]` (上流 bf16 は `[8192, 1, 4]`) | **MLX 変換で軸が入れ替わっている。**読み込み側で必ず確認する (squeeze 後の値はビット一致 — [10 §4](10-MLX4BIT-AUDIT.md)) |
| imatrix レポートの `expert_coverage`: 123 モジュール × 256 = 31,488 エキスパート、**全部が校正で発火**、`min_count 32` / `median 1678` / `max 29349` | **ルーターの偏りの一次資料。**median の 17 倍が max。「256-way でも相関があるか」([05 §1-4](05-RISKS.md)) を測る前の事前分布 |
| omlx は ANE prefill 経路 (`qwen35_ane.metal` 123 KB) を持つ | 扱わない ([05 §4](05-RISKS.md))。存在だけ記録 |
| omlx は quantized KV (`turboquant_kv.py`) を持つ | 本ランタイムは TurboQuant KV を削除済みで manifest 側で拒否している。**逆方向。追わない** |

**取り入れないもの:**

- **mxfp4 / mxfp8 モード** — `_bytes_per_group` にあるが、今回のチェックポイントは
  全部 `affine`。int4/int8 affine 以外のカーネルは持たない
- **oQ2〜oQ3 の低ビット帯** — 18 GB を 3.5 bpw に落とせば 14 GB になるが、
  2/3 bit のカーネルが無い。**候補としてだけ記録**
- **oMLX の連続バッチ / 階層 KV / マルチモデル配信** — 本リポジトリの設計思想と別物
- **per-expert の bit 混在** — `expertStride` が manifest 単一値で、
  `expertOffset = layer*expertsPerLayer*stride + expert*stride` の等間隔前提が壊れる。
  **原理的に不可能。**oQ 側も routed expert は一律 base bit なので衝突しない

## 7. 上流 bf16 (71.9 GB) を全部引く必要はあるか — 無い

| 用途 | 供給源 | 状態 |
| --- | --- | --- |
| Phase 0〜4 の本体 (テキスト) | `MLX-4bit` 19.51 GB または `oQ4e-g64` 21.86 GB | **取得済み** |
| 攻めすぎ箇所の打ち直し (in_proj_a/b, shared expert, embed/lm_head) | **bf16 部分抽出 3.50 GB** | **取得済み** |
| カーネル検証の参照 | 4-bit を脱量子化して fp32 で回す | 手元で足りる |
| Vision (Phase 9) | `oQ4e-mtp` の `vision_tower.*` (bf16 893 MB) | **取得済み** |
| MTP (Phase 7) | `oQ4e-mtp` の `language_model.mtp.*` (503 MB) | **取得済み** |
| imatrix の効きの対照 | `oQ4e-mtp` | **取得済み** |
| **oQ / GPTQ を自分で回す (案「oQ を自分で回す」)** | **bf16 全体** | **これだけが 71.9 GB を要求する** |

**Phase 0〜9 は上流 bf16 の全体を必要としない。**残る唯一の用途は、18 GB の routed
expert を**自分で** imatrix + GPTQ で量子化し直す案「oQ を自分で回す」であり、
やると決めてから引けばよい。

## 8. その他の公開量子化 (選別済み、2026-08-21)

HF の Ornith-1.5-35B-A3B 系リポジトリを `config.json` / `index.json` で選別した結果
(**実測(上流)**)。足切りの根拠は `Quantization.supportedGroupSizes = [32,64]` と
「routed expert は int4 GEMV しか無い」:

- **公式 `ornith-ai/…-MLX-{4,6,8}bit` と MLX-bf16** はいずれも `language_model.*` だけで、
  **vision_tower も MTP も入っていない** (1,757 本、`pipeline_tag: text-generation`)。
  4bit 版だけが形式互換で、候補 2 本の片方になっている (§1)
- **6bit / 8bit 系** (公式 MLX-6/8bit、oQ6e / oQ8e ほか) は routed expert が int4 でないので全滅
- **`OsaurusAI/…-JANG_4M`** (21.5 GB) — 全部 g64 affine でメニュー外が 5-bit 44 本だけ。
  MTP + vision あり。ただし MTP が per-expert 展開 2,304 本で積み直しが要り、
  **AWQ を RMSNorm に吸収済みなので norm 重みが上流と一致しない** (Phase 0 の
  突き合わせで引っかかる)。未取得。対照候補としてだけ記録
- **`Shiftedx/…-mtplx`** — MTP を index の外の `mtp/weights.safetensors` (785 本 BF16
  1.69 GB) に置いている (index だけ見ると「MTP 無し」と誤判定する)。本体は g32 ベースで
  router だけ g64 なので混在して乗らないが、**`mtplx_runtime.json` の `mtp_contract`
  (`base_hidden_variant: post_norm` / `concat_order: embedding_hidden` /
  `mtp_position_mode: cache` / `depth_max 3`) は MTP 配線 ([03 §6](03-DESIGN.md)) の
  第三者資料として読む価値がある**
- **`…-oQ4e-fp16-mtp`** — oQ4e-mtp と量子化設定が同一で scales/biases が F16。
  本ランタイムは BF16 ビットパターン前提なので劣化版
