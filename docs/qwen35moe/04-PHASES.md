# 04. 段階計画 Phase 0〜9

各 Phase は「出口条件」を満たすまで次へ行かない。**GPU を使う Phase は 2 以降**
(Phase 2 は GPU を使うがモデルは載せない)。

## 進捗 (2026-08-22)

| Phase | 状態 |
| --- | --- |
| Phase 0 事実確定 | **ほぼ済み。**`gate_up` の順序・RMSNorm の `+1`・`conv1d` の軸順・`in_proj_a` の感度が確定 ([10](10-MLX4BIT-AUDIT.md))。**`oQ4e-g64` 側の照合も完了** ([12](12-OQ4E-G64-AUDIT.md))。**float32 参照器が動き、算式が上流実装と一致** ([14](14-REFERENCE.md))。**残: fixtures の絞り込みと 2048 トークン後の状態** |
| Phase 1 変換 | **完了** ([13](13-PHASE1-REPACK.md))。`oQ4e-g64-baked` を repack し `--verify-install` が緑。20.49 GB / `expertStride 1,769,472` / 上流とバイト一致 |
| Phase 2 カーネル | **完了。**`qwen_delta_rule` ([15](15-PHASE2-GDN.md)、prefill 30 層 **125.7 ms**)、周辺 7 本 ([17](17-PHASE2-KERNELS.md))、**INT8 の LM head chain** ([19](19-LM-HEAD-INT8.md)、1 トークン 4.0 ms / 134 GB/s)。`--qwen` の検査は **39 本すべて緑**、うち 10 本は負例 |
| Phase 3 decode 結線 | **完了** ([20](20-PHASE3-DECODE.md) / [21 §1](21-PHASE4-PREFILL.md))。`QwenForwardRunner` / `RecurrentStateManager` / `LayerKind.linear` が入り、**参照の生成 55 本すべてと一致**。負例 5 本も落ちる。64 本に届かないのは**モデルが `<\|im_end\|>` を出して止まった**ため — その生成に 56 本目は無い |
| Phase 4 prefill | **一致の条件が通った** ([21](21-PHASE4-PREFILL.md))。`QwenPrefill.swift` と **INT8 の QMM**が入り、**チャンク経由でも 55 本すべてが一致** (幅 512 と 8 の 2 通り)。時間の条件は #16 と一緒 |
| Phase 5 tokenizer/CLI | **完了** ([22](22-PHASE5-TOKENIZER.md) / [23](23-PHASE5-TOOLS.md))。`QwenTokenizer` (ByteLevel) と上流 jinja が入り、**CLI が日本語と英語で答える** (`--qwen-tokenizer` 223 本)。**XML 形のツール呼び出しと GBNF も入った** — `--qwen-tools` が 36 本、うち 6 本は負例。中心は**テンプレートが描いた呼び出しを文法が受理しパーサが同じ引数に戻す**こと。残るのはサーバー結線 (Phase 8) と CLI の `--tools` |
| Phase 6 以降 | 計測はこれから。エキスパート計数の phase stamp は入った ([22 §6](22-PHASE5-TOKENIZER.md)) |

---

## Phase 0 — 事実確定 (GPU 不要)

1. ~~`Scripts/qwen35/dump_reference.py`~~ → **`reference_forward.py` として完了**
   ([14](14-REFERENCE.md))。ただし計画とは形が違う: **上流 bf16 は手元に無く、
   18 GB の RAM に 21.9 GB は載らない**ので、`transformers` をランタイムとして
   使うのではなく、**層を 1 つずつ開いて捨てる float32 参照器**を書いた
   (`transformers` は**算式の出典と検証相手**として使う)。層ごとの入出力・
   線形注意の状態・router の top-8・logits が `.npz` で落ちる
2. `gate_up_proj` の連結順 — **解決済み** (MLX が分割済み)。ただし fixtures で 1 度だけ突き合わせる
3. `A_log` / `dt_bias` / `in_proj_a` の値域 — **出した** ([10 §5](10-MLX4BIT-AUDIT.md))。
   残: **実活性での再測** (合成入力 `x ~ N(0,1)` でしか測っていない)。
   参照器が実活性を持つようになったので、**測る道具は揃った**
4. 4-bit RTN を通したときの品質: **層ごとの相対誤差と最終 logits の KL** を出す。
   imatrix 版 (oQ4e-g64) との対照もここで取る

**出口:** fixtures がリポジトリに入り、「4-bit で行けるか」に数字が付いている。
**注意:** 参照器が展開するのは**4-bit チェックポイントを float32 に戻したもの**で、
bf16 そのものではない。したがって「4-bit で行けるか」を bf16 との差として測ることは
**この機械ではできない** (全 70 GB の取得が要る)。候補どうしの比較 (NLL) は測れる
([14 §7](14-REFERENCE.md))。

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

**`qwen_delta_rule` は済んだ** ([15](15-PHASE2-GDN.md))。検証は fixtures ではなく
**実物と同じ形の合成入力 + CPU 参照 2 本** (float32 の床 / double の真値) で立てた
— チェックポイントを開かずに済み、2048 トークンの検査長を自由に取れるため。
`--gdn` で 15 本。

**周辺 7 本も同じ形で済んだ** ([17](17-PHASE2-KERNELS.md))。`--qwen` で 29 本、
うち 6 本は「このモデルで静かに壊れる道」を参照側に作った**負例**
(conv の軸順 / l2norm の eps / `1+w` / Gemma の RoPE の組 など)。
**LM head も書けた** ([19](19-LM-HEAD-INT8.md)): INT4 の specialization ではなく
INT8 の chain で、`--qwen` は 39 本になった。**Phase 2 はこれで閉じている。**

## Phase 3 — decode 結線

`QwenForwardRunner` の decode 経路のみ。prefill は off、投機も off、
`--temp 0` の greedy。

**出口:** 固定プロンプトから **64 トークン、参照 (CPU float32) と完全一致**。
一致しないなら層ごとの hidden を突き合わせて発散点を特定する。

**中止線:** 発散点が `qwen_delta_rule` の数値的な蓄積 (fp32 でも合わない) なら、
状態を fp64 相当 (2×fp32 の compensated summation) にする案を検討。それでも
合わなければ chunkwise 形へ (誤差の出方が変わる)。

**通った** ([20](20-PHASE3-DECODE.md))。`--qwen-decode` が **55 トークン全一致**、
負例 5 本も落ちる。中止線は引かれていない — 発散点が無い。
**64 本に届かないのは参照の側の事情ではなくなった** ([21 §1](21-PHASE4-PREFILL.md)):
取り直した参照は 55 本目に `<|im_end|>` を出して止まっており、**56 本目が存在しない。**
実物で分かった食い違いは 3 つ
(埋め込みと shared ゲートが 8-bit、routed の活性化が SiLU) で、いずれも
**落ちずにそれらしく間違う**形だった ([20 §2](20-PHASE3-DECODE.md))。

## Phase 4 — prefill

チャンク幅 512 / 1024 / 2048。`qwen_delta_rule` の T>1 経路。時間ブロック TB を
`(16, 32, 48)` の 3 通り測る ([03 §2-6](03-DESIGN.md))。

**出口:** prefill 経由の greedy 64 トークンが Phase 3 と一致。
**線形注意の 30 層合計が 150 ms 以内** ([05 §2](05-RISKS.md) #2)。

**通った** ([21](21-PHASE4-PREFILL.md))。`--qwen-prefill` が **55 トークン全一致**、
チャンク幅 512 (1 チャンク) と 8 (3 チャンク) の両方で同じ。負例 5 本も落ちる。
**チャンク幅がモデルから見えない**ことがこの 2 通りの主張である ([21 §4](21-PHASE4-PREFILL.md))。
新しく要ったのは **INT8 の QMM** (INT4 版は 8-bit を読めない) と、T 行版の小物 3 本。
`prefill.metal` の gelu も関数定数で分けた — **Gemma の PSO は同じコードを吐く。**

**時間の条件 (150 ms) はまだ閉じていない。**[17 §4-2](17-PHASE2-KERNELS.md) の
締め方の判断 (#16) と同じ話で、そこはユーザー判断のまま。合成入力での prefill 全体は
**チャンク 2048 で 5.5 ms/トークン** ([21 §5](21-PHASE4-PREFILL.md)、**運用値ではない**)。

**TB の 3 通りは合成入力で済んだ: TB=32 が最良** (125.7 / 128.4 / 144.8 ms、
[15 §4](15-PHASE2-GDN.md))。実物の活性での再測は Phase 6。

**周辺のカーネルも合成入力で測ってある** ([17 §4](17-PHASE2-KERNELS.md))。
チャンク 2048 で `qwen_delta_qkv_prepare` 21.9 ms / `qwen_delta_norm_gate` 11.5 ms /
`qwen_delta_gates` 0.3 ms (いずれも 30 層ぶん)、`qwen_qkv_epilogue` 10.0 ms (10 層)。
**`qwen_delta_rule` を含めた線形注意 30 層の合計は 159.4 ms** で、上の出口条件の
読み方だと外れる。再帰カーネル単体は 125.7 ms で [05 §2](05-RISKS.md) #2 の線の内側
なので**再設計の引き金は引いていない**。締め方の判断はユーザーに出す
([17 §4-2](17-PHASE2-KERNELS.md))。

## Phase 5 — トークナイザ / テンプレート / CLI

**通った** ([22](22-PHASE5-TOKENIZER.md))。ただし**計画とは形が違う**:

- ~~`verifyDecoderConfiguration` に ByteLevel を許す分岐~~ → **やめた。**
  Gemma の検査は「よそのトークナイザを弾く」ために在るので、そこに ByteLevel を
  通す穴を開けない。代わりに **`QwenTokenizer` という別の型**を書き、
  両方向 (Ornith の宣言は Gemma の検査に落ち、逆も落ちる) を検査に入れた
  ([22 §1](22-PHASE5-TOKENIZER.md))
- `GemmaDecoding` の兄弟 → **`ByteLevelDecoding` / `ByteLevelRun`** が入った。
  **落ちた特殊トークンをまたいで run は融合する**という上流の規則が
  実装の分かれ目で、そこが負例になっている ([22 §2](22-PHASE5-TOKENIZER.md))
- `vocabSize = 262_144` のリテラル → Qwen 経路は tokenizer から読む (248,070)。
  **LM head の幅とは別の数**である ([19](19-LM-HEAD-INT8.md))
- チャットテンプレート → **上流の `chat_template.jinja` をそのまま**
  swift-jinja が描画する。`--thinking` が `enable_thinking` に直結する
- 停止トークンは `[248046, 248044]` (実測(上流)) で、`QwenTokenizer` が持つ

**出口条件は満たした:** `TurboFieldfareCLI --model … --messages-file …` が
日本語でも英語でも答え、`<think>` ブロックは stderr、答えは stdout に分かれる
([22 §5](22-PHASE5-TOKENIZER.md))。

**ツール呼び出しも入った** ([23](23-PHASE5-TOOLS.md)):

- パーサは `QwenToolCallParser`。**値の綴りは宣言された型で決まる** —
  テンプレートが文字列だけを生で書くので、`Set<String>` ではなく
  ツールのスキーマを要る。駆動は `QwenStructuredAssistantDecoder`
- GBNF は `QwenToolCallGrammar` (`TurboFieldfare` 側) と
  `QwenChatGrammarBuilder` (サーバー側の薄い口)。生の文字列値は
  **`\n</parameter>` を含まない任意のテキスト**を 13 状態で綴る —
  `[^<]*` にするとマークアップを引数に取れない
- `GrammarVocabulary` の piece は **ByteLevel 用の入口を足した**。Gemma の
  規則で作ると落ちずに日本語が 1 文字も通らなくなるので、負例で押さえた
- `tools` ブロックの食い違い ([22 §4-2](22-PHASE5-TOKENIZER.md)) は再訪し、
  **記述を訂正した** ([23 §5-1](23-PHASE5-TOOLS.md)): swift-jinja のキー順は
  不定ではなく**昇順**で、両方決定的。だから引数を昇順で綴る設計が成り立つ

**残っているもの:** サーバー結線 (Phase 8)、CLI の `--tools`、
入れ子 JSON の往復 2 件 ([23 §5-2](23-PHASE5-TOOLS.md))。

## Phase 6 — 計測と運用点

`bench.sh` の作法をそのまま踏襲する。**temp 1.0 のまま、クールダウンは 4 秒**
(`COOLDOWN=4`。2026-08-22 のユーザー指定。Gemma 側の既定 20 秒・採点 10 秒とは
別の値で、この Phase にだけかかる)。GPU は 1 個だけ。熱ドリフトの検定
(先頭と末尾で base を 2 回、`|head/tail| > 5%` の run は捨てる) は**残す** —
間隔を詰めた分、判定はこちらが担う。

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

## 次の一手 (2026-08-22)

**済んだもの:**

1〜4. ~~`oQ4e-g64` の norm 規約 / `conv1d` の軸順 / router のビット幅 / `q_norm` の
   焼き込み~~ → **完了** ([12](12-OQ4E-G64-AUDIT.md))
5〜6. ~~名前寄せ~~ / ~~`.gturbo` への repack~~ → **完了** ([13](13-PHASE1-REPACK.md))
7. ~~生成スモーク~~ → **完了。文が出た** ([14 §6](14-REFERENCE.md))
9. ~~2 候補の品質差 (NLL)~~ → **完了。4 本とも `oQ4e-g64` が低い。本線を
   `oQ4e-g64` に決めた** ([16 §1](16-QUALITY.md))
12. ~~Gated DeltaNet カーネル~~ → **完了。GPU が初めて回った**
   ([15](15-PHASE2-GDN.md))。2048 トークン後の状態が CPU float32 の床と一致、
   prefill 30 層 125.7 ms で中止線の内側
13. ~~Phase 2 の残りのカーネル~~ → **LM head を除いて完了** ([17](17-PHASE2-KERNELS.md))。
   `qwen.metal` に 7 本、`--qwen` の検査 29 本すべて緑。**LM head は
   「int4 の specialization」ではなく「INT8 の chain」だった**ので #11 の後ろに回る
11. ~~混在ビット幅の受け入れ~~ → **完了。本線が `Model.load` を通る**
   ([18](18-MIXED-BITS.md))。**索引から導く**に決め、manifest のスロットは幅の
   上限を述べるだけにした。`.gturbo` の形式は変えていない。Qwen 用の常駐
   スキーマ (`linear_attn` 9 本 / 倍幅の `q_proj` / 別テンソルの `lm_head`) と
   `--qwen-open` も入った
15. ~~INT8 の LM head chain~~ → **完了。Phase 2 が閉じた** ([19](19-LM-HEAD-INT8.md))。
   実物の語彙で 1 トークン **4.0 ms / 134 GB/s**。`vocab` に 248,077 を渡すだけで
   未学習の 243 行は採点されない (マスクのコードは要らなかった)
17. ~~64 トークンの参照の取り直し~~ → **完了。55 本で、そこが終わりだった**
   ([21 §1](21-PHASE4-PREFILL.md))。`<|im_end|>` を出して止まったので **56 本目が無い**。
   `--qwen-decode` は 55 本すべて一致し、**Phase 3 が閉じた**
18. ~~Phase 4 の prefill 結線~~ → **完了。チャンク経由でも同じトークンが出た**
   ([21](21-PHASE4-PREFILL.md))。`QwenPrefill.swift` は decode と同じく直列。
   **INT8 の QMM** が要った (INT4 版は 8-bit を読めない) ので `--qwen` は 57 本に。
   `prefill.metal` の SiLU も関数定数で分けた
19. ~~Phase 5 の tokenizer / テンプレート / CLI~~ → **完了。CLI が日本語と
   英語で答えた** ([22](22-PHASE5-TOKENIZER.md))。`GFTokenizer` に分岐を足すのでは
   なく **`QwenTokenizer` という兄弟**にした (Gemma の検査を緩めないため)。
   `--qwen-tokenizer` の 223 本は上流 (`tokenizers` / `transformers`) の実行
   結果との突き合わせで、うち 4 本は負例。**XML 形のツール呼び出しと GBNF は
   残っている** (#22)
22. ~~XML 形のツール呼び出しと GBNF~~ → **完了。テンプレートの描画が
   文法を通り、パーサが同じ引数に戻した** ([23](23-PHASE5-TOOLS.md))。
   `QwenToolCallParser` / `QwenStructuredAssistantDecoder` /
   `QwenToolCallGrammar` / `QwenChatGrammarBuilder` と、`GrammarVocabulary` の
   ByteLevel 入口。`--qwen-tools` は 36 本 (負例 6)、`swift test` は 1,329 件。
   [22 §4-2](22-PHASE5-TOKENIZER.md) の「キー順は不定」は**訂正した**
14. ~~Phase 3 の結線~~ → **完了。decode が参照と一致した** ([20](20-PHASE3-DECODE.md))。
   `QwenForwardRunner` は直列 (1 層 = コマンドバッファ 2 本)、`RealForwardRunner` は無変更。
   `RecurrentStateManager` (62.8 MiB、文脈長に依らない) と `LayerKind.linear` が入り、
   `ExpertCacheBudget` は再帰層に K/V を数えなくなった

**次にやること:**

| # | やること | 要るもの |
| --- | --- | --- |
| 16 | **線形注意 30 層の締め方の判断** ([17 §4-2](17-PHASE2-KERNELS.md))。周辺まで数えると 159.4 ms で Phase 4 の出口条件を外れる。再帰カーネル単体は 125.7 ms で [05 §2](05-RISKS.md) #2 の内側。**prefill が通ったので、実物の壁時計も出せるようになった** ([21 §5](21-PHASE4-PREFILL.md)) | ユーザー判断 |
| 20 | **routed expert のタイル版を prefill に通す** ([21 §3-2](21-PHASE4-PREFILL.md))。いま通しているのは per-pair GEMV の方で、[05 §1-2](05-RISKS.md) の占有率の話はまだ始まっていない | GPU |
| 21 | **2048 トークン / チャンク 512 の逆転**を説明する ([21 §5](21-PHASE4-PREFILL.md))。他の 3 行と違い 1 回目より 2・3 回目が遅い | GPU |
| 8 | `in_proj_a` の実活性再測を 200 トークン級の `--dump` でやり直す ([16 §2](16-QUALITY.md))。**本線は 8-bit なので、これは本線を止めない** | CPU |
| 10 | fixtures を Phase 3 が要る層だけに絞る ([14 §5](14-REFERENCE.md))。**2048 トークン後の状態は 15 が合成入力で見たので、fixtures 側の宿題ではなくなった** | CPU |

道具: `Scripts/qwen35/audit_checkpoint.py` / `bake_snapshot.py` / `mlx_quant.py` /
`reference_forward.py` (numpy だけ)。`test_reference_forward.py` だけ torch を使う。
カーネルの検査は `TurboFieldfareKernelCheck --gdn` と `--qwen`
(どちらもモデルもチェックポイントも要らない)。時間は `--gdn-bench` / `--qwen-bench`。
**repack 済みの実物を開くのは `--qwen-open <path>`** ([18](18-MIXED-BITS.md))、
**実物を走らせて参照と突き合わせるのは `--qwen-decode <path>`** ([20](20-PHASE3-DECODE.md);
`--qwen-decode-fixture` / `--qwen-decode-new` / `--qwen-decode-fault-tokens`)、
**プロンプトを T 行の経路に通すのは `--qwen-prefill <path>`**
([21](21-PHASE4-PREFILL.md); `--qwen-prefill-chunks`)。時間は `--qwen-prefill-bench`。
**ツール呼び出しを実物の語彙で見るのは `--qwen-tools <path>`**
([23](23-PHASE5-TOOLS.md); fixture 不要)。
**トークナイザを上流と突き合わせるのは `--qwen-tokenizer <path>`**
([22](22-PHASE5-TOKENIZER.md); `--qwen-tokenizer-fixture`)。その fixture は
`Scripts/qwen35/tokenizer_fixture.py` が上流を 1 度回して作る
(`~/LLM/venv/bin/python3` に `tokenizers` と `transformers` が入っている)。
**CLI から実物に話しかけるのは `TurboFieldfareCLI --model … --messages-file …`**
(family は `manifest.json` から読む)。
参照の fixture は `scratch/qwen35/decode-fixture-55.json`、
トークナイザの fixture は `scratch/qwen35/tokenizer-fixture.json`。
repack 済みモデルは `scratch/ornith-oq4e-g64.gturbo`、repack の入力は
`~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-baked`。**参照器には焼き込み前の
`~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64` を渡す** (`q_norm` の 1/16 は本ランタイム
専用の細工なので、参照の算式には入れない)。

**まだ手を付けていない前提:**

- ~~**tokenizer は確実に弾かれる**~~ → 片づいた ([22 §1](22-PHASE5-TOKENIZER.md))。
  弾いていたのは **Gemma のローダ**で、それは正しい。Ornith は `QwenTokenizer` が開く
- **ツール呼び出しは文法とパーサまで** ([23](23-PHASE5-TOOLS.md))。
  **サーバーも CLI もまだ呼んでいない** — `QwenChatGrammarBuilder` に
  呼び出し側が無い。サーバー (Phase 8) と vision (Phase 9) は手つかず
- 運用点 (スロット数・チャンク幅) は Gemma 4 の値のままで、Ornith 用には何も測っていない
- **参照器では bf16 との差は測れない** ([14 §7](14-REFERENCE.md))
- **パッケージテストは `--no-parallel` で回す。**素の `swift test` は remote install 系が
  20 本前後落ちるが、これは変更前の tree でも同じ ([18 §7](18-MIXED-BITS.md))
