# 11. M1 の結果 — フォーマット拡張と repacker

測定: 2026-08-17、M3 Pro 18GB / macOS 15.7.5 / Swift 6.2。
対象モデル `scratch/gemma4-qat.moepack` (QAT lattice-aligned + vision tower)。

表記は PLAN 系と同じ: **実測** / **導出** / **未確認**。

---

## 0. 結論

| # | 出口条件 (04-PHASES M1) | 結果 |
| --- | --- | --- |
| 1 | `--include-draft` / `--add-draft` | **実装。**`TsugumiRepack` の 2 モード (§2) |
| 2 | ドラフターなしの `.moepack` が現行とバイト一致 | **成立 (実測)。**凍結フィクスチャ `productionWritersMatchPreRefactorV1Fixtures` が通る。`draft` は optional なのでキー自体が現れない (§3) |
| 3 | 旧ランタイムが `flags.mtpDraft` を名指しで拒否 | **成立 (実測)。**`manifest.flags contains unknown key "mtpDraft"` で exit 1 (§5) |
| 4 | 対照 (フラグなし) は exit 0 | **成立 (実測)。**同じバイナリが `gemma4.moepack` を生成して exit 0 (§5) |
| 5 | 上流ピンの再確認 (05-RISKS U6) | **一致 (実測)。**revision・テンソル全数・payload・BF16 3 本の SHA-256 が 01 の記録どおり (§1) |
| 6 | 実機の `scratch/gemma4-qat.moepack` | **ドラフター追記済み。**236 MB のみ取得、テキスト側 inode 不変、再検証 exit 0 (§4)。受入ゲート 7 を M1 の時点で満たしている |

**ランタイムはドラフターをまだ 1 バイトも読まない。**M1 が触るのは
`MoEPackFormat` / `TsugumiRepack` と、`MoEPackFormatV1.knownFlags` に
`mtpDraft` を足したことによる**受理**だけ。decode 経路は無変更。

## 1. 上流ピンの再確認 (**実測**、2026-08-17)

`01-CHECKPOINT.md` の記録は 2026-08-17 の調査時のもので、上流で変わりうる。
M1 の取得前に HF API と Range 取得で読み直した。

| 項目 | 01 の記録 | M1 で読んだ値 | 判定 |
| --- | --- | --- | --- |
| revision | `bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c` | 同じ | 一致 |
| `model.safetensors` テンソル数 | 94 | 94 | 一致 |
| payload 合計 | 236,124,704 B (ファイル) | header 10,264 + payload **236,114,440** | 一致 |
| `model.norm.weight` の SHA-256 | `3bf68317e6d4e33e…` | `3bf68317e6d4e33e29a3d019eb744d52d5fb3ebf5dca52513e878e1f845f9047` | 一致 |
| `layers.0.input_layernorm.weight` | `fbd6be5ad58d336c…` | `fbd6be5ad58d336c6bd398bd161db25bcc13bf2e07e50db8c6b55d8f400959eb` | 一致 |
| `layers.3.self_attn.q_norm.weight` | `07101aaa0ac6e6df…` | `07101aaa0ac6e6df4b28cc3b209f04edaa9c277fc5037b50525478e3e60df15b` | 一致 |
| `model.safetensors.index.json` の SHA-256 | (未記録) | `ab54b0e481714d358d800ad10366f585841e678f982be3274ea6660e9bedd3eb` | **新規にピン**。インストールの入口の指紋 |

3 本は **google の bf16 assistant と MLX 版で同じ値**であることも取り直した
(01 §1-3 の再現)。この 3 本が `DraftModelSource.pin.provenanceTensors` で、
インストール時に MLX 版から取り直して照合する。**「4bit のどれか」ではなく
「Google の QAT assistant の変換物である」ことの証拠**になる。

`config.json` から確定した追加事実 (**実測**):

- `text_config.final_logit_softcapping = null` → **M0 Q1 の答えが上流 config にも一致**。
  softcap を持つ config はインストール時に拒否する。
- `use_ordered_embeddings = false` / `num_kv_shared_layers = 4` / `enable_moe_block = false`。
  いずれも本ランタイムが表現できない構成を弾く検査に使った。

## 2. 入れたもの

### 2-1. フォーマット (`MoEPackFormat`)

| 追加 | 内容 |
| --- | --- |
| `flags.mtpDraft` | `knownFlags` に追加。**旧ビルドの拒否はこれが担う** |
| `versionMinorDraft = 2` | vision の 1 と同じく記述的 |
| `draft/draft_weights.bin` | `MoEPackFormatV1.draftWeightsPath` |
| `manifest.draft` | optional セクション (`MoEPackManifestDraftV1`) |
| `validateDraftSection` | フラグ/セクションの対、minor ゲート、`files` 宣言、payload の収まり、**ターゲット arch との一致** |

vision との差は最後の 1 行にある。ドラフターは**自前の K/V を持たず**ターゲットの
層 28/29 の K/V を読む (01 §2) ので、head 構成・窓・RoPE 定数・語彙は自由変数ではなく
**ターゲットの値の再掲**である。したがって codec が `arch` と突き合わせ、食い違う
manifest は encode も decode も通さない:

```
backboneHiddenSize == arch.hiddenSize      vocabSize    == arch.vocabSize
slidingWindow      == arch.slidingWindow   headDim      == arch.headDim
fullHeadDim        == arch.fullHeadDim     numKVHeads   == arch.numKVHeads
numFullKVHeads     == arch.numFullKVHeads  attentionKEqV== arch.attentionKEqV
ropeTheta / fullRopeTheta / partialRotaryFactor == arch の同名
sharedSlidingKVLayer は sliding 層、sharedFullKVLayer は full 層を指すこと
```

`sharedSlidingKVLayer` / `sharedFullKVLayer` は**ターゲットの層マスクから導く**
(最後の sliding = 28、最後の full = 29)。ピン側に書かないので、層構成の違う
ターゲットには自分の層番号が入る。

### 2-2. repacker (`TsugumiRepack`)

| ファイル | 役割 | 写像元 |
| --- | --- | --- |
| `DraftModelSource.swift` | ピン (repo / revision / index digest / 由来テンソル / config) | `VisionModelSource.swift` |
| `DraftSourceLoader.swift` | commit 固定・index digest 照合・config 照合・由来テンソルのハッシュ | `VisionSourceLoader.swift` |
| `DraftRepackPlanner.swift` | config から導く 48 エントリの棚卸しと resident レイアウト | `VisionRepackPlanner.swift` |
| `DraftAppendInstaller.swift` | `--add-draft` | `VisionAppendInstaller.swift` |

vision との実質的な差は **int4 であること**。tower は BF16 一枚ぶんで済むが、
ドラフターは 23 本の量子化テンソル (weight U32 + scales/biases BF16) と
25 本の BF16 で、`ResidentEntry` の scale/bias 欄を使う。棚卸しは
`config` から生成し、**94 本の source テンソルが過不足なく 48 エントリに畳まれること**を
検査する (余りが 1 本でもあれば拒否)。

`--include-draft` と `--add-draft` は同じ manifest バイト列を書く。
両者が一致することはテストで直接比べている (§3)。

### 2-3. CLI

```
TsugumiRepack --output <model.moepack> --include-draft
TsugumiRepack --add-draft --input-moepack <model.moepack>
```

`--add-vision` と `--add-draft` は排他 (どちらもモデルディレクトリを占有するため)。

## 3. テスト (**実測**)

`Scripts/test.sh`: **826 テスト / 140 スイート、11 issue**。
issue の内訳は `RESULTS_VISION.md` §7 と同じ陳腐化スイート
(QAT ピン / prefill 2048 / 48 スロット / causalQBlock) で、**新しい失敗はゼロ**
(803 → 826 は今回の +23)。

新規 23 件の内訳:

| スイート | 件数 | 何を守るか |
| --- | ---: | --- |
| `DraftInstallTests` | 6 | 中身がソースのバイトであること、フラグなしで何も動かないこと、ピン違反・由来違反・寸法違反・欠損の拒否 |
| `DraftAppendInstallTests` | 6 | 追記が新規インストールと同一、tower との共存、二重追記の拒否、失敗時に何も残さないこと、テキスト側 inode/mtime 不変 |
| `MoEPackFormatCodecTests` (追加) | 11 | セクション/フラグの対、minor ゲート、arch との一致、共有 KV 層の種類、tie されていない lm head の拒否 |

**バイト一致 (出口条件 2)**: `MoEPackFormatCompatibilityTests` の
`productionWritersMatchPreRefactorV1Fixtures` が通る。これは manifest / layout /
resident index を凍結フィクスチャと**バイト比較**する検査で、`draft` を optional に
したことでキー自体が現れないことを示す (`textOnlyManifestOmitsTheDraftSectionEntirely`
が JSON レベルでも同じことを確認する)。

`DraftInstallTests.addingTheFlagChangesNothingOnTheTextSide` は同じスナップショットを
2 回インストールし、`model_weights.bin` / `packed_experts/*` の**バイト一致**と、
manifest の差が「ドラフター 1 件ぶんだけ」であることを直接比べる。

## 4. 実機での `--add-draft` (**実測**)

`scratch/gemma4-qat.moepack` (QAT + vision tower、16 GB) に追記した。

```
$ time ./.build/release/TsugumiRepack --add-draft --input-moepack scratch/gemma4-qat.moepack
Added the MTP drafter to /Users/mh/LLM/tsugumi/scratch/gemma4-qat.moepack
Drafter: 48 tensors, 236114440 bytes
Source: mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit @ bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c
Downloaded 236114440 bytes
Re-verified 39 files (17216936158 bytes)
./.build/release/TsugumiRepack --add-draft --input-moepack   7.64s user 3.53s system 15% cpu 1:14.20 total
                                                                             (exit 0)
```

**取得は 236,114,440 B ちょうど** = payload と同じ。ギャップの余分な取得はゼロ
(ドラフターの 94 本は 1 ファイルを埋め尽くしているので、coalesce が全域を 4 リクエストに畳む)。

受入ゲート 7 (テキスト側不変) の実測:

| ファイル | inode | mtime | サイズ | 判定 |
| --- | ---: | ---: | ---: | --- |
| `model_weights.bin` | 35170185 → 35170185 | 1786853712 → 同じ | 1,512,886,332 | **不変** |
| `packed_experts/layer_00.bin` | 35170186 → 35170186 | 1786853694 → 同じ | 476,053,504 | **不変** |
| `vision/vision_weights.bin` | 35766994 → 35766994 | 1786920392 → 同じ | 1,145,637,984 | **不変** |
| `draft/draft_weights.bin` | (新規) 35922760 | 1786958359 | 236,130,824 | 追加 |
| `manifest.json` | 35767390 → 35922926 | 更新 | 8,650 → 9,894 | 書き換え |

`draft_weights.bin` = index ページ 16,384 + payload 236,114,440。

書かれた manifest の `draft` セクション (抜粋):

```json
"sharedSlidingKVLayer": 28, "sharedFullKVLayer": 29,
"backboneHiddenSize": 2816, "hiddenSize": 1024, "numLayers": 4,
"headDim": 256, "fullHeadDim": 512, "numKVHeads": 8, "numFullKVHeads": 2,
"fullAttentionLayerMask": [0,0,0,1], "tensorCount": 48, "payloadBytes": 236114440,
"quant": { "weightBits": 4, "scheme": "affine", "groupSize": 64, ... }
```

共有 KV の相手 28 / 29 は**ターゲットの層マスクから導かれた値**で、01 §2 の
「最終 sliding = 28、最終 full = 29」と一致する。`flags` は
`{aneSharedExpert, streamingPresent, turboQuantKV, visionTower, mtpDraft}`、
`versionMinor` は 1 → **2**。

書かれたファイルの中身の検査 (**実測**):

| 検査 | 結果 |
| --- | --- |
| resident index | 48 エントリ、payload 合計 236,114,440 |
| 論理 shape | `embed_tokens.weight` [262144,1024] / `pre_projection.weight` [1024,5632] / `post_projection.weight` [2816,1024] / `layers.3.self_attn.q_proj.weight` [8192,1024] — すべて 01 §2 の形 |
| バイト一致 | `norm.weight` (2 KB)・`layers.3.self_attn.q_norm.weight` (1 KB)・`post_projection.weight` (1.44 MB、U32 量子化本体) を HF から Range 取得して比較 → **3 本とも完全一致** |

追記後の状態:

```
$ ./.build/release/TsugumiRepack --add-draft --input-moepack scratch/gemma4-qat.moepack
add-draft failed: configuration invalid: …/scratch/gemma4-qat.moepack already has an MTP
drafter; reinstall the model to change it                                    (exit 1)

$ ./.build/release/TsugumiRepack --verify-install --input-moepack scratch/gemma4-qat.moepack
Verified 39 files (17216936158 bytes)                                        (exit 0)
```

> **運用上の注意:** `scratch/gemma4-qat.moepack` は `mtpDraft` フラグを持つようになった。
> これは §5 の拒否がまさに働くということで、**M1 以前のビルド (古い CLI / Server /
> Mac アプリのバイナリ) はこのモデルを読み込めない。**該当するバイナリは再ビルドが要る。
> `scratch/gemma4.moepack` (旧 4bit インストール) はフラグを持たないので影響を受けない。

## 5. 旧ランタイムの拒否と対照 (**実測**)

M1 の変更を 1 行も含まないビルド (この作業の直前に `swift build -c release` して
退避したバイナリ) に、ドラフター付きの `.moepack` を食わせた。

```
$ <pre-M1>/TsugumiCLI --model scratch/gemma4-qat.moepack \
      --prompt "The capital of France is" --max-new 8
error: manifest.flags contains unknown key "mtpDraft"
exit 1
```

**対照 (この検査の検出力そのもの):** 同じバイナリで、`mtpDraft` を持たない
`scratch/gemma4.moepack` は正常に生成して exit 0。

| バイナリ | モデル | flags | 結果 |
| --- | --- | --- | --- |
| pre-M1 | `gemma4-qat.moepack` (ドラフターあり) | `mtpDraft: true` | **exit 1**、`unknown key "mtpDraft"` |
| pre-M1 | `gemma4.moepack` (ドラフターなし) | フラグなし | exit 0、`tok/s=13.734` |
| M1 | `gemma4-qat.moepack` (ドラフターあり) | `mtpDraft: true` | exit 0、`tok/s=11.567`、`load=0.763s` |

拒否はフラグに対するものであって、「古いバイナリが何にでも失敗する」のではない
(vision の `RESULTS_VISION.md` §8 と同じ形)。

**ドラフターは読まれていない (導出、根拠は実測):** `Model.load` が SHA-256 を検証して
開くのは `model_weights.bin` / `packed_experts/layout.json` / 使う層ファイルだけで、
`draft/` も `vision/` も load 経路に現れない。`load=0.763s` は追記前の水準
(10-M0 §4 の 0.708 s) と同じ桁で、236 MB を読んだ痕跡はない。
tok/s の差は同一セッション内 A/B ではないので**比較に使えない**
(`RESULTS_VISION.md` §3 の警告)。

## 6. 次 (M2) に渡すもの

- `draft/draft_weights.bin` の中身は **MLX の量子化レイアウトそのまま** (U32 packed +
  BF16 scales/biases、group 64)。M2 のドラフター forward はこの並びを読む。
- resident index のエントリ名は上流の `model.` を落とした形
  (`embed_tokens.weight` / `layers.0.self_attn.q_proj.weight` / `pre_projection.weight` /
  `post_projection.weight` / `norm.weight`)。
- 共有 KV の相手は manifest が名指しする (`sharedSlidingKVLayer` / `sharedFullKVLayer`)。
- `Scripts/mtp/fetch_draft_weights.py` と `dump_draft_fixtures.py` (04-PHASES §5) は
  **M2 の作業**。M1 では要らなかった (重みは Swift 側の `--add-draft` が取る)。
