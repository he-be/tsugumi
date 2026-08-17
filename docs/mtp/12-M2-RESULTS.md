# 12. M2 の結果 — ドラフター forward 単体

測定: 2026-08-17、M3 Pro 18GB / macOS 15.7.5 / Swift 6.2。
対象モデル `scratch/gemma4-qat.gturbo` (M1 でドラフター追記済み)。

表記は PLAN 系と同じ: **実測** / **導出** / **未確認**。

---

## 0. 結論

| # | 出口条件 (04-PHASES M2) | 結果 |
| --- | --- | --- |
| 1 | 参照実装との突き合わせ (合成入力) | **成立 (実測)。**3 ケース × 8 ステージ、`TurboFieldfareKernelCheck --draft` が 74/74 PASS (§2) |
| 2 | 閾値は測って決める | **決定 (実測)。**参照自身の FP16 床を `Scripts/mtp/fp16_error_floor.py` で測り、その内側に収めている (§2) |
| 3 | 検出力 3 件 (壊した参照が閾値を超える) | **成立 (実測)。**ropeOffByOne / qNormDropped / attentionScaleClassic が 6/6 超え (§3) |
| 4 | (副次) argmax の一致 | **3/3 完全一致** (§2) |

**decode 経路は無変更。**入れたのは `Model.draftWeights()` の遅延ロードと
`DraftForward` (独立的な forward 1 本) だけで、`RealForwardRunner` /
`RawCompletion` には 1 行も触れていない。テキスト生成の経路が M2 の影響を受ける
箇所は存在しない (**導出**: 呼び出しグラフ上 `DraftForward` を参照するのは
`TurboFieldfareKernelCheck --draft` のみ)。

## 1. 参照実装の確定 — transformers 5.10.4

`scratch/mtp-ref/` の vendored transformers 5.6.2 と venv の 5.6.2 は、本ドラフターの
config (`num_kv_shared_layers == num_hidden_layers` = 全層共有) を
**構築できない** (`Gemma4TextAttention.__init__` が `prev_layers` 空で壊れる、**実測**)。
上流 config は `transformers_version: 5.10.0.dev0` を名乗るため stable を走査し、
**5.10.4 で確定**した (**実測**):

- 全層 `is_kv_shared_layer = True`、k_proj / v_proj は 1 本もない。
- shared KV は**層 type 文字列** (`"sliding_attention"` / `"full_attention"`) でキー ing。
- head_dim は 256/256/256/512、`attention_k_eq_v` で full 層の V==K。
- vendored `modeling_gemma4_assistant.py` との差分はこの shared-KV 受け渡し周りのみ
  (算法は同一)。

venv は `scratch/mtp-venv` (vision-venv とは別、5.10.4 + torch 2.13 + mlx)。
int4 デ量子化は numpy 自作 → `mx.dequantize` (f32 スケール) と**bit-exact** を確認済み
(**実測**): LE U32、偶数要素が下ニブル、group 64 の BF16 scale/bias。

## 2. 突き合わせと閾値 (**実測**)

fixture は `Scripts/mtp/dump_draft_fixtures.py` が transformers 5.10.4 を **float32** で
回した段階別出力 (`scratch/mtp-fixtures/`)。入力は乱数の
`(ターゲット埋め込み 2816, ターゲット hidden 2816, 共有 KV [1,H,len,dim])` で、
26B ターゲットは両側とも一切動かさない (04-PHASES §5 の設計どおり)。

ケースは 3 本: `short-37` (窓内)、`window-1024` (窓の境界)、`past-window-1500`
(クエリが窓を超え、双方向 SWA マスクが生きる)。

`KernelCheck --draft` の結果 (要約、全文は実行ログ):

| ケース | argmax | 最悪 max rel | 最悪 rms rel |
| --- | --- | ---: | ---: |
| short-37 | **一致** (236787) | 9.7e-3 (logits) | 1.7e-3 (h-norm) |
| window-1024 | **一致** (239627) | 1.0e-2 (h-norm) | 1.9e-3 (logits) |
| past-window-1500 | **一致** (1606) | 9.5e-2 (logits) | 2.7e-2 (logits) |

**kv ≥ 1024 で誤差が伸びるのは FP16 自体であって実装バグではない**ことを、
vision と同じ手法 (参照自身を fp16 で回して床を測る) で示した
(`Scripts/mtp/fp16_error_floor.py`、**実測**):

| ケース | 上流 fp16 last_hidden (max / rms) | 本実装 (max / rms) |
| --- | --- | --- |
| short-37 | 4.3e-3 / 7.4e-4 | 7.3e-3 / 1.2e-3 |
| window-1024 | 7.7e-2 / 1.2e-2 | 5.0e-3 / 1.1e-3 |
| past-window-1500 | **1.9e-1 / 3.2e-2** | 6.0e-2 / 1.3e-2 |

本実装は上流 fp16 と同等以下の誤差で、argmax も上流 fp16 と同じ選択をする。
長い KV で誤差が伸びるメカニズム (**導出**): softmax の分母が √kvLen 程度で
増大するため、attention 出力の要素が小さくなり、FP16 の相対刻みが効く。

閾値は「測った最悪値の ~2 倍」で固定 (`DraftTolerance`): h-pre 1e-3/2e-4、
layer 8e-2/2.5e-2、h-norm 1.5e-1/3e-2、last-hidden 1.2e-1/2.5e-2、
logits 2e-1/5e-2。

## 3. 検出力 (**実測**)

同じ fixture を `DraftForward.Fault` で壊して比較 (floor: max 1.3e-1 / rms 3.5e-2 —
正常最悪値と故障最小値の間に独立に設定):

| 故障 | max rel | rms rel | 判定 |
| --- | ---: | ---: | --- |
| ropeOffByOne (position+1) | 5.2e-1 | 1.2e-1 | 超える |
| qNormDropped (q_norm スキップ) | 2.2e-1 | 5.7e-2 | 超える |
| attentionScaleClassic (1/√d) | 6.2e-1 | 1.6e-1 | 超える |

## 4. 入れたもの

| 面 | 内容 |
| --- | --- |
| `Scripts/mtp/fetch_draft_weights.py` | 236 MB 取得 + 由来テンソル 3 本の SHA-256 照合 (vision 写像) |
| `Scripts/mtp/dump_draft_fixtures.py` | float32 参照の段階別 fixture 生成 (3 ケース × 15 ファイル) |
| `Scripts/mtp/fp16_error_floor.py` | 参照自身の FP16 床測定 (§2) |
| `ManifestReader` | `manifest.draft` を `ManifestDraft` として公開 + `files` 収まり検査 |
| `Sources/TurboFieldfare/MTP/DraftWeights.swift` | 遅延ロード・スキーマ検査 (量子化エントリ込み)・アクセサ。`VisionWeights` の int4 対応写像 |
| `Sources/TurboFieldfare/MTP/DraftForward.swift` | forward 1 ステップ。既存 decode カーネルのみで組成 (新規 Metal は `scale_inplace_fp16` 1 本のみ) |
| `Model.swift` | `hasDraft` / `draftWeights()` (vision と同型の遅延ロード) |
| `TurboFieldfareKernelCheck --draft <model>` | fixture 突合 + 検出力 (`--vision-tower` の写像) |

レイヤ tail は **non-MMoE 版のサンドイッチ** (`h1 = h + rmsnorm(attn)`、
`out = (h1 + rmsnorm(mlp)) × layer_scalar` — norm 2 回) で、ターゲットの MoE tail
(norm 4 回) と本数が違う。そのため `FusedLayerTail` ではなく vision の
`VisionNormResidualAdd` 2 回 + `scale_inplace_fp16` 1 回で構成した (**実測**:
上記突合がこの構成の正しさの検査)。

## 5. 確定した仕様 (01 §5 の残り)

| # | 質問 | 決着 |
| --- | --- | --- |
| Q4 | 共有 KV は層 28/29 の K/V をそのまま使うか | **そのまま (実測)。**追加の norm/RoPE なし。fixture は K/V に何の前処理もせずに一致 |
| Q6 | 埋め込みスケール √2816 はどちらに掛かるか | **ターゲットの埋め込み (2816 幅) に掛かる (実測)。**`pre_projection` の入力は `concat(target_embed(tok) × √2816, target_hidden)` の 5632 次元で、ドラフター自身の 1024 幅 `embed_tokens` は tied lm head 専用。5.6.2 を 1024 幅で走らせると形状エラーで即死するので、実装が間違っていれば fixture 生成の時点で落ちる |

これで Q1〜Q6 すべて決着 (Q1〜Q3 は M0、Q4/Q6 は M2)。

## 6. テスト (**実測**)

- `Scripts/test.sh`: **826 テスト / 140 スイート / 11 issue — 既知の陳腐化スイートのみ、新規失敗ゼロ** (M1 と同一構成。M2 はテストを足していない: 突合は `swift test` で走らない KernelCheck 側に置いた。vision の layer B と同じ構成)
- `TurboFieldfareKernelCheck` 全部盛り (INT4 両 group + vision + draft): **100 cases PASS**、exit 0

## 7. 次 (M3) に渡すもの

- `DraftForward.encode` が 1 ステップぶんの部品としてそのまま使える。M3 の受理率測定
  (04-PHASES §2) は「各 decode ステップでドラフターを回して提案列を記録するだけ」
  なので、必要なのは `targetEmbed` (decode 経路の埋め込み lookup に outScale を付けた
  もの) と `lastHidden` (N3 の post-norm hidden 取り出し) の接続だけ。
- post-norm hidden の取り出し (02 §N3) は M2 では不要だった (fixture が直接渡すため)。
  M3/M4 で初めて必要になる部品。
- fp16 誤差の知見: 長い KV で logits rms が ~3e-2 まで伸びる。受理判定はターゲット側
  logits で行うので影響しないが、ドラフター argmax の信頼度は kv 長に依存して
  ゆっくり劣化する (M3 の受理率データで実測する)。
