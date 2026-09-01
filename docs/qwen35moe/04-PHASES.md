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
| Phase 4 prefill | **一致の条件が通った** ([21](21-PHASE4-PREFILL.md))。`QwenPrefill.swift` と **INT8 の QMM**が入り、**チャンク経由でも 55 本すべてが一致** (幅 512 と 8 の 2 通り)。**routed expert のタイル版も通り、同じ 55 本を出した** ([24](24-PREFILL-MOE-PATH.md))。時間の条件は #16 と一緒 |
| Phase 5 tokenizer/CLI | **完了** ([22](22-PHASE5-TOKENIZER.md) / [23](23-PHASE5-TOOLS.md) / [25](25-CLI-TOOLS.md))。`QwenTokenizer` (ByteLevel) と上流 jinja が入り、**CLI が日本語と英語で答える** (`--qwen-tokenizer` 223 本)。**XML 形のツール呼び出しと GBNF も入った** — `--qwen-tools` が 36 本、うち 6 本は負例。**CLI の `--tools` も入り、実物が呼び出しを書いた** ([25](25-CLI-TOOLS.md))。残るのはサーバー結線 (Phase 8) |
| Phase 8 サーバー | **完了** ([26](26-PHASE8-SERVER.md))。`QwenServerSession` (`ServerModelSession` の兄弟) と `QwenGenerationPlan` が入り、**HTTP から実物が答える** — 日本語・ストリーミング・ツール呼び出しとその応答の往復まで。**HTTP 層は 1 行も変えていない。**prompt cache は持たない (`cache_n` は常に 0) と決めたので、`ExpertCacheBudget` に足す勘定は無かった |
| Phase 6 計測 | **1 本目が済んだ** ([27](27-PHASE6-THROUGHPUT.md))。運用点の数字を実タスク 4 本で取り、**そこで見えた直列を 2 つ外した** — 生成 **+31〜41%**、prefill **−25〜45%** (短いプロンプトの TTFT 2.4 → 1.5 秒)。footer が GPU / 取得 / ホストの 3 分割を出すようになり、`bench/qwen35.sh` が家族専用の駆動になった。残るのは層をまたぐ先読み (§8) |

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

**routed expert の 2 つの経路を A/B した** ([24](24-PREFILL-MOE-PATH.md))。
`prefillRoutedPath` で per-pair GEMV とタイル版を選べるようになり、
**タイル版が 4 通りとも速い** (GPU 時間で −22〜40%)。**既定はタイル版に切り替えた**
(2026-08-22 ユーザー確定)。同じ測定で [21 §5](21-PHASE4-PREFILL.md) の
**説明の付いていなかった行も片づき、あの表は 1 行訂正された**
(クールダウン無しの GPU クロック低下。2048/512 は 17.80 ではなく **9.43 ms**)。

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

**CLI の `--tools` も入った** ([25](25-CLI-TOOLS.md)):

- 融合ヘッドは logit をどこにも書かないので、Gemma のように
  `forceLogitsHead: true` で語彙幅を書き出して `Sampler` に棄却再抽選させる
  手は使えない。代わりに**マスクつきでもう一度畳む**
  (`qwen_lm_head_greedy_int8_rows_chunk_masked` / `encodeMaskedRescore`)。
  常の手はマスクを 1 bit も読まず、**拒まれたトークン 1 個につきヘッドを
  もう 1 回** — 実測 4.086 ms で、素の 4.084 ms と差が測れない
- `--tools` / `--tool-choice` / `--parallel-tool-calls`。宣言は
  `--messages-file` だけに付き、Gemma の install には**断る**
- 検査は `--qwen` に 6 本 (負例 2)、実物に当てる **`--qwen-constrain` が
  9 本** (負例 1)。後者は文法ではなく**スタブ制約**を当てる — 文法の判定は
  モデルが書くテキストの性質なので、1 回も棄却しない run は棄却経路について
  何も言わない
- 実物は宣言したツールを呼んだ。**素の argmax がすでに整形式だった検体では
  文法が 1 回も効かなかった** (`--tool-choice none` と `required` が同じ 27
  トークンを出した)。外した検体では**停止トークンが 1 回だけ拒まれ**、
  モデルは自分で辻褄を合わせてから呼び出しを書いた

**残っているもの:** サーバー結線 (Phase 8)、
入れ子 JSON の往復 2 件 ([23 §5-2](23-PHASE5-TOOLS.md))、
実物が並列で呼び出しを書くかどうか。

## Phase 6 — 計測と運用点

**1 本目は済んだ** ([27](27-PHASE6-THROUGHPUT.md))。下の 6 項目のうち
**1・3・4 が片づき、2 は材料が出た**:

| # | 状態 |
| --- | --- |
| 1 トレースとヒット率カーブ | **済み** ([27 §6-1](27-PHASE6-THROUGHPUT.md))。机上の値は実機とヒット数まで一致する。lfu が lru に 3〜3.5 ポイント勝つ |
| 2 運用点の候補 | **材料が出た** ([27 §6-2](27-PHASE6-THROUGHPUT.md))。ただし **mmap の腕ではヒット率が壁時計の代理にならない** — 12.7 ポイントの差が tok/s では 4.8% にしかならないので、48 スロットの押しは弱い。**既定は変えていない** |
| 3 チャンク幅 | **済み。既定の 2048 が最良** (2,378 トークンで 14.53 / 10.57 / **8.63** 秒)。4096 の候補追加はユーザー判断 ([27 §6-3](27-PHASE6-THROUGHPUT.md)) |
| 4 `RDAdvice` の調律 | **不要だった。**4 通りが振れの中で、**既定の mmap の腕は `F_RDADVISE` を出さない** ([27 §6-3](27-PHASE6-THROUGHPUT.md)) |
| 5 `RouterPreviewProbe` の 256-way | **済み** ([27 §9-4](27-PHASE6-THROUGHPUT.md))。**miss の 64.5% を 1 層前に名指せる** (Gemma の 128-way が 70%)。投げると **+1.9〜13.2%**、select を出さない経路なら **+8.4〜22.3%** ([31 §4](31-PREFETCH-CHEAPER.md))。**既定は off** — 入れるかはユーザー判断 (#29) |
| 6 oQ を自分で回す案 | 手つかず |

`bench.sh` の作法をそのまま踏襲する。**クールダウンは飾りではない** —
空けずに回すと GPU 時間が 2.6 倍になる条件が実在する
([24 §4-2](24-PREFILL-MOE-PATH.md))。**temp 1.0 のまま、クールダウンは 4 秒**
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

**駆動は `bench/qwen35.sh`** ([27](27-PHASE6-THROUGHPUT.md))。`bench.sh` の兄弟で、
違いは 3 つだけ (サンプリングが無い / クールダウン 4 秒 / 熱ドリフトの判定を
こちらが持つ)。`tasksab` が実タスク 4 本の A/B、`pipeline` が m.json での同じ A/B、
`slots` / `chunk` / `rdadvise` / `arm` が条件のスイープ、`trace` が机上のカーブ。

## Phase 7 — MTP

[03 §6](03-DESIGN.md)。全エキスパート常駐 + 行ごと状態書き出し。
**先読みとの合流と着手前の机上ゲート (union 分布・受理長の CPU 測定) は
[29](29-MTP-PREFETCH-OUTLOOK.md)** — 特に受理長 a はまだ 1 つも測っておらず、
期待値のすべてを握る ([29 §1-3](29-MTP-PREFETCH-OUTLOOK.md))。
**判定線が 1 本増えた** ([32 §1-4](32-NVMAI-ADOPT.md)、実測(NVMAI)): sparse MoE では幅 2 の検証パスがエキスパートの**和集合**で 1.585 倍になり、**受理率 p > 約 0.585** を切ると decode 高速化としての取り分が無い。`mtp_acceptance.py` はこの線に対して読む。draft へ渡す hidden が **pre-final-norm** であることも同じ文書が決着させた ([32 §1-1](32-NVMAI-ADOPT.md))。

**先に 1 段増えていた MTP ヘッドの差し替えは済んだ**
([30 §6](30-MTP-HEAD-GRAFT.md))。上流の `mtp.*` は乱数初期化のままで
**出荷ヘッドで a を測っても意味が無い**ので、`shisa-ai/…-MTP-ONLY` の 19 本を
`oQ4e-g64` の 42 本に写した `~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-shisa-baked` を作った
(**カーネルは 1 本も書いていない**。expert 間 std の CV 0.69% → 9.46%、
`expertStride` と参照バイトは不変)。**以降 MTP はこれだけを使う。**
**そのヘッドで受理長も測った** ([33](33-MTP-ACCEPTANCE.md)): P1 = 78.70% /
a = 2.344 (平均)。**運用幅は k=2 (ドラフト 1 本)** — 検証費用がエキスパートの
和集合で伸びるので (幅 2 で 1.554 倍、幅 4 で 2.430 倍)、k=4 は利得より費用が勝つ
([33 §3-2](33-MTP-ACCEPTANCE.md))。取り分の机上は **+8.5〜11.5%** で、
**符号を握るのはパスあたりの固定費** (未測定)。pre/post-norm は深さ 1 では
区別がつかないので、運用点では論点にならない。

**出口:** `RESULTS_MTP.md` と同じ様式で tok/s / TTFT / peak の 3 点。
**受入は「非投機と greedy でバイト一致」** (Gemma 側の D5 不変条件と同じ)。

**2026-08-22: 実機で通し、残っていた受入条件も閉じた** ([36](36-MTP-DECODE.md) →
[38](38-MTP-VERIFY-PATH.md) → [39](39-RESIDENCY-COMMIT.md) → [40](40-MTP-GRAMMAR.md))。
**文法・ツール呼び出しとの併用が入り** (行を名指せる再畳み込み、新カーネル 0 本)、
**サーバーが `--draft-block-size 2` を受ける**。バイト一致の条件は
**投機の中立性 (強制棄却の対照と一致) では満たしているが、素の decode とは
答えが変わる** — T 行カーネルと decode カーネルの加算順の差
([36 §5-3](36-MTP-DECODE.md))。既定は off のまま。

**回した** ([36](36-MTP-DECODE.md)、2026-08-22)。ヘッドは 503 MB の sidecar で
GPU に載り (`.gturbo` の repack は 0 回)、幅 2 の検証パスと巻き戻しが回り、
実タスク 5 本を 192 トークン生成した。**取り分はタスクで符号が変わる** —
エージェント形 ×1.146 / ×1.199、コード ×0.959、散文 ×0.941、
2,698 トークンの要約 ×0.843。**受入条件のうち 1 つが未達である**:
「非投機とバイト一致」は**投機については満たす** (強制棄却の対照とトークン列が
完全一致) が、**素の decode とは一致しない** — まさに下の「先に確かめること」が
警告していた T 行経路と decode 経路の演算差で、192 トークンのうち一致は
34〜192 本。行 GEMV は decode とビット一致まで持っていったが、
prefill attention / block router / per-pair MoE の加算順が残っている
([36 §7](36-MTP-DECODE.md) #1)。**既定は off。**

**先に確かめること:** prefill の routed expert が**タイル版 (FP16 staging)** に
なり、decode の GEMV (FP32) との演算差が広がっている
([24 §3-3](24-PREFILL-MOE-PATH.md) #1)。**バイト一致をこの経路で 1 回取る**
まで、この受入条件は成り立つと仮定しない。

## Phase 8 — サーバー

**通った** ([26](26-PHASE8-SERVER.md))。ただし計画とは形が違う:

- ~~prompt cache のスナップショット方針を決める~~ → **持たないことに決めた。**
  [03 §5](03-DESIGN.md) の推奨 (slot あたり 62.8 MiB) は採らず、要求ごとに
  ランナーを `reset()` してプロンプトを全部計算する。`cache_n` は常に 0
- ~~`ExpertCacheBudget` に勘定を足す~~ → **足す勘定が無かった。**スナップショットを
  持たないので、生きている再帰状態 1 本 (62.8 MiB、`QwenForwardRunner` の中) だけが
  残る。DEV-3 の「生成スロットは 1 本」がそれを許している
- [docs/serving/SPEC.md](../serving/SPEC.md) は**書き換えていない。**家族ごとの
  差は SPEC の規範ではなく、既存の語彙 (R3 の「受理して無視」、DEV-16 の
  `approximations`、EP-4 の `modalities`) で表現できた

**入ったもの:** `QwenServerSession` (`ServerInferenceBackend` の実装、
`ServerModelSession` の兄弟)、`QwenGenerationPlan` (純粋型、C0 で 12 本)、
`QwenForwardRunner.runGreedyCompletion` (停止・キャンセル・RSP-3 の内訳)、
`QwenReasoningSplitter` を CLI からライブラリへ、`main.swift` の家族分岐。
**HTTP 層・要求検証・待ち行列・timings は 1 行も変えていない。**

**できないこと 4 つ** ([26 §4](26-PHASE8-SERVER.md)): 画像は 400、投機は起動時に
断る、prompt cache は無い、サンプラは**受理して無視** (R3 — 既定の要求の
`temperature` は 1.0 なので、断ると既定の客が全部落ちる)。

**残った判断が 1 つ** ([26 §6-1](26-PHASE8-SERVER.md)): `tool_choice: required` に
ツールと無関係な質問を与えると、**呼び出しを書かないまま `max_tokens` まで
散文が続く**。非遅延文法の前置きが「セクション開始でない任意のトークン」なので、
止まれはしないが出てもこない。締める案は既存の検査 2 本と
[23](23-PHASE5-TOOLS.md) の判断に触るので、**変えずにユーザーに出す**。

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
20. ~~routed expert のタイル版を prefill に通す~~ → **完了。同じ 55 本が出て、
   4 通りとも速かった** ([24](24-PREFILL-MOE-PATH.md))。`prefillRoutedPath` の
   分岐 1 か所で、`PrefillGroupedRoutedMoE` も `RealForwardRunner` も無改造。
   **[05 §1-2](05-RISKS.md) の占有率は実在した**が、per-pair に留まる理由には
   ならなかった。**既定はタイル版に切り替えた** (#25 も同日に閉じた) —
   `TF_PREFILL_MOE=scalar` を立てると既定も自動的に `.perPair` に戻る
21. ~~2048 トークン / チャンク 512 の逆転~~ → **完了。GPU 時間で、4 秒空けると
   消える** ([24 §4](24-PREFILL-MOE-PATH.md))。bench に GPU / 取得 / ホストの
   3 列とプロセス内クールダウンを足して割った。**[21 §5](21-PHASE4-PREFILL.md)
   の表は 1 行訂正した**
14. ~~Phase 3 の結線~~ → **完了。decode が参照と一致した** ([20](20-PHASE3-DECODE.md))。
   `QwenForwardRunner` は直列 (1 層 = コマンドバッファ 2 本)、`RealForwardRunner` は無変更。
   `RecurrentStateManager` (62.8 MiB、文脈長に依らない) と `LayerKind.linear` が入り、
   `ExpertCacheBudget` は再帰層に K/V を数えなくなった
24. ~~Phase 8 のサーバー結線~~ → **完了。HTTP から実物が答えた**
   ([26](26-PHASE8-SERVER.md))。`QwenServerSession` は `ServerModelSession` の
   **兄弟**で、HTTP 層は 1 行も変えていない。prompt cache は**持たない**と決め、
   `ExpertCacheBudget` に足す勘定は無かった。`swift test` は **1,350 件**。
   `tool_choice: required` の前置きに 1 件、ユーザー判断が残った (#26)
23. ~~CLI の `--tools`~~ → **完了。実物が呼び出しを書いた** ([25](25-CLI-TOOLS.md))。
   融合ヘッドのまま文法をかけるために **マスクつきの再畳み込み**を足した
   (`qwen_lm_head_greedy_int8_rows_chunk_masked`)。`--qwen` は **80 本**
   (2026-08-22 に +17: logit を書くヘッドとサンプラ、[42](42-SAMPLING.md))、
   実物に当てる `--qwen-constrain` が **9 本**、`swift test` は **1,338 件**。
   **Phase 5 の箇条書きはこれで全部片づいた**

27. ~~層をまたぐ先読みの的中率 (256-way)~~ → **完了。64.5% の miss を 1 層前に
   名指せた** ([27 §9-4](27-PHASE6-THROUGHPUT.md))。ついでに「取得 15.8 ms/tok」の
   正体も割れた: **ホスト側のページ写像 (1 トークン 9,200 ページ)** で、
   `commit()` を消しても**カーネル内フォールトに移るだけ**。非投機的な 3 つの手
   (落とさない / set を使わない / commit を間引く) は**どれも引き分け**だった
   ([27 §9-2](27-PHASE6-THROUGHPUT.md))

30. ~~先読みの固定費と N の実験 (#29 の材料)~~ → **完了。select を出さない経路が全 26 ペアで
   +2.4〜3.1%** ([31](31-PREFETCH-CHEAPER.md))。計器 (`declined` / wait) で
   **d=2 と半減リトライは落ちた**。**N>8 は負ける** — 名指しは +23 ポイント
   積むのに、増えた写像が陰 (層あたり GPU 0.7 ms) に入りきらない。
   router 融合カーネルは引き分けで既定 off。**既定は変えていない** (#29)。
   **適応スキップ ([28 §3-4 (c)](28-PREFETCH-IDEAS.md)) も机上で否定した**
   ([31 §7](31-PREFETCH-CHEAPER.md)): 層別 miss 分布に崖が無く、どこを切っても負。
   ついでに **preview は 40 本/tok ではなく 39 本**と分かった

25. ~~Phase 6 の 1 本目 (実タスクの prefill と生成)~~ → **完了。生成 +31〜41%、
   prefill −25〜45%** ([27](27-PHASE6-THROUGHPUT.md))。速くしたのは**待ちを外した**
   ことだけで、カーネルは 1 本も書いていない: 遅延 join (1 トークンの join が
   81 → 41)、shared 分岐を routed の読みに重ねる、prefill のタイル読み先行。
   `TF_QWEN_PIPELINE=0` で元の直列に戻る。`--qwen-decode` / `--qwen-prefill` は
   **両腕で全一致**、`swift test` は 1,350 件緑

32. ~~文法・ツール呼び出しと MTP の併用~~ / ~~サーバーへの投機の口~~ →
   **完了。エージェント形のターンが投機で回る** ([40](40-MTP-GRAMMAR.md))。
   排他の正体は**再畳み込みが読む行が固定**だったことで、
   `encodeMaskedRescore` に `hiddenNormed` を足しただけ (**新カーネル 0 本**)。
   3 腕が 55/55・63/63 でトークン一致 (文法が実際に効いた検体を含む)、
   サーバーは `--draft-block-size 2`。`--qwen-constrain` は **15 本** (負例 3)、
   `swift test` は 1,350 件緑。**残るのは prompt cache** (#31 (b)) —
   2,935 トークンのターンで prefill 11.3 秒を**毎ターン**払う

33. ~~prompt cache~~ → **完了。エージェント経路の TTFT が 10.33 → 0.72 秒**
   ([41](41-PROMPT-CACHE.md))。形は**その場保持の 1 エントリ・厳密な延長のみ**で、
   追加メモリ 0 バイト・新カーネル 0 本・Gemma 0 行。
   [34 §4-3](34-PROMPT-CACHE-ESTIMATE.md) が一番恐れた再レンダの継ぎ目は
   **実測ではずれず**、tools + thinking + ツール応答の往復 + ストリーミングの
   どれでも 2 ターン目以降が当たる。34 の式で 1 つ外れていたのは
   「状態は最後の 1 トークンを食っていない」で、**幅 2 の受理で終わったパスは
   食っている**。`--qwen-resume` 7 本 + C0 9 本、`swift test` **1,359 件**

**次にやること:**

| # | やること | 要るもの |
| --- | --- | --- |
| 29 | **層をまたぐ先読みを既定にするか** ([27 §9](27-PHASE6-THROUGHPUT.md))。**[31](31-PREFETCH-CHEAPER.md) で安くなった** — preview の select を出さない広い経路 (`TF_QWEN_EXPERT_PREFETCH=8 TF_QWEN_PREVIEW_WIDE=1`) で **+8.4% (56 tok) / +11.2% (60 tok) / +22.3% (48 tok) / +10.5% (498 tok) / +15.8% (2,698 tok)**、出力もヒット率もメモリも n8 と同一。**N は 8 のまま** (9〜16 位は当たるが写像が陰に入りきらない、[31 §2](31-PREFETCH-CHEAPER.md))、**d=2 とリトライは計器で落ちた** ([31 §1](31-PREFETCH-CHEAPER.md))。残る対価は予測 GEMV だけ | ユーザー判断 |
| 31 | **NVMAI からの移植** ([32](32-NVMAI-ADOPT.md))。**(b) prompt cache は [41](41-PROMPT-CACHE.md) で入った** (その場保持・厳密な延長のみ)。残るのは (c) と、2 本目の会話を持つ形 ([41 §8](41-PROMPT-CACHE.md) #1)。(a) ~~判定線~~ → **済んだ。**P1 78.70%、そして **[33 §3-6](33-MTP-ACCEPTANCE.md) / [§3-7](33-MTP-ACCEPTANCE.md) の実測 2 本も済んだ** — GDN の snapshot/restore は**コピー 0 回**にでき (0.28%)、幅 2 の 1 パスは **1.27〜1.30 倍**で和集合の予測より軽い。**取り分は +15〜29%**。残るのは幅 2 の頭 (測定 3)、(b) **prompt cache の机上が出た** ([34](34-PROMPT-CACHE-ESTIMATE.md): 取り分 最大 9.2 秒、追加 0 バイト、Swift 約 200 行・カーネル 0 本、最大の危険は再レンダの継ぎ目)、(c) decode の hit-fixup は未着手、(d) 測定は interleaved A/B を既定に | ユーザー判断 |
| 34 | **FreeToken の在庫から着手するか** ([42](42-FREETOKEN-IDEAS.md)、検討・公表値のみ)。候補は独立に 4 つ: (a) **意味的境界の anchor** — [41 §4-2](41-PROMPT-CACHE.md) の全ミス形 (thinking を送り返さないクライアント) を境界までのヒットに変える。会話内は再帰状態の写真 ≈ 62 MiB/個で、**その形のクライアントを実際に使うかの運用判断が先** (b) global expert cache — `expert_sim.py` 流儀のオフライン判定だけなら GPU 0 分・委譲可 ([31 §7](31-PREFETCH-CHEAPER.md) の「崖が無い」は逆風) (c) cold expert の CPU fixup — #31 (c) の変種として同じ A/B に並べる (d) KV ↔ スロットの弾力予算 — [41 §8](41-PROMPT-CACHE.md) #1 の写真プールを持つ日に | ユーザー判断 |
| 28 | **候補追加の可否 2 件** (どちらも既定は変えない): `allowedExpertCacheSlots` に 48、`allowedPrefillChunkTokens` に 4096。48 の押しは弱い ([27 §6-2](27-PHASE6-THROUGHPUT.md))、4096 は prefill が素直に伸びる見込み ([27 §6-3](27-PHASE6-THROUGHPUT.md)) | ユーザー判断 |
| 16 | ~~**線形注意 30 層の締め方の判断**~~ → **2026-08-22 のユーザー判断で保留。**周辺まで数えると 159.4 ms で Phase 4 の出口条件を外れ、再帰カーネル単体は 125.7 ms で [05 §2](05-RISKS.md) #2 の内側 ([17 §4-2](17-PHASE2-KERNELS.md))。**放置してよい理由が 1 つ増えた**: prefill 全体の GPU 時間はチャンク 2048 で 5,036 ms ([24 §3](24-PREFILL-MOE-PATH.md)) なので、**159.4 ms はその約 3%** (別々に測った 2 つの比なので **導出**)。どちらの定義で締めても prefill 全体はほとんど動かない — chunkwise 形の FLOP 2 倍を払う理由はこの比率では出てこない | 保留 |
| 26 | **`tool_choice: required` の前置きの締め方の判断** ([26 §6-1](26-PHASE8-SERVER.md))。案 A は前置きを `responseFormatGrammar` と同じ `(!</think>* </think> [ \t\n]{0,20})?` に揃える (thinking off なら 1 手目から呼び出しが強制される) が、チェックポイント自身のシステムプロンプトと食い違い、既存の検査 2 本が落ちる。案 B は現状維持 | ユーザー判断 |
| 8 | `in_proj_a` の実活性再測を 200 トークン級の `--dump` でやり直す ([16 §2](16-QUALITY.md))。**本線は 8-bit なので、これは本線を止めない** | CPU |
| 10 | fixtures を Phase 3 が要る層だけに絞る ([14 §5](14-REFERENCE.md))。**2048 トークン後の状態は 15 が合成入力で見たので、fixtures 側の宿題ではなくなった** | CPU |

道具: `Scripts/qwen35/audit_checkpoint.py` / `bake_snapshot.py` / `mlx_quant.py` /
`reference_forward.py` (numpy だけ)。`test_reference_forward.py` だけ torch を使う。
カーネルの検査は `TurboFieldfareKernelCheck --gdn` と `--qwen`
(どちらもモデルもチェックポイントも要らない)。時間は `--gdn-bench` / `--qwen-bench`。
**幅 2 の投機が再帰状態に払う費用は `--qwen-state-bench`**
(`--qwen-state-bench-iterations` / `--qwen-state-bench-cooldown`、腕は交互。
モデル不要、[33 §3-6](33-MTP-ACCEPTANCE.md))。
**実物の速度は `bench/qwen35.sh`** ([27](27-PHASE6-THROUGHPUT.md);
`tasksab` / `pipeline` / `slots` / `chunk` / `rdadvise` / `arm` / `trace` /
`summarize` / `prefetch` / `prefetchw` / `prefetchn`。結果は
`bench/qwen35-results.tsv`、先読みの footer の証拠は `bench/qwen35-prefetch/`)。
**repack 済みの実物を開くのは `--qwen-open <path>`** ([18](18-MIXED-BITS.md))、
**実物を走らせて参照と突き合わせるのは `--qwen-decode <path>`** ([20](20-PHASE3-DECODE.md);
`--qwen-decode-fixture` / `--qwen-decode-new` / `--qwen-decode-fault-tokens`)、
**プロンプトを T 行の経路に通すのは `--qwen-prefill <path>`**
([21](21-PHASE4-PREFILL.md); `--qwen-prefill-chunks`)。時間は `--qwen-prefill-bench`
(`--qwen-prefill-bench-moe per-pair|tiled` / `--qwen-prefill-bench-cooldown <秒>`、
[24](24-PREFILL-MOE-PATH.md))。**幅を腕にして交互に測るのは
`--qwen-prefill-bench-chunk 1,2`、本物のトークン列を食わせるのは
`--qwen-prefill-bench-token-file <--dump-tokens の出力>`**
([33 §3-7](33-MTP-ACCEPTANCE.md))。腕ごとに**エキスパート要求数**が出るので、
両腕で同じ値なら幅が効いていない ([35](35-PREFILL-CHUNK-WIDTH.md))。
**ツール呼び出しを実物の語彙で見るのは `--qwen-tools <path>`**
([23](23-PHASE5-TOOLS.md); fixture 不要)。
**制約つき貪欲の棄却経路を実物に当てるのは `--qwen-constrain <path>`**
([25](25-CLI-TOOLS.md); `--qwen-constrain-new N`)。
**CLI からツールを宣言するのは `--tools <path>` と `--tool-choice`**
([25 §3](25-CLI-TOOLS.md); 例は `scratch/qwen35/tools.json`)。
**トークナイザを上流と突き合わせるのは `--qwen-tokenizer <path>`**
([22](22-PHASE5-TOKENIZER.md); `--qwen-tokenizer-fixture`)。その fixture は
`Scripts/qwen35/tokenizer_fixture.py` が上流を 1 度回して作る
(`~/LLM/venv/bin/python3` に `tokenizers` と `transformers` が入っている)。
**CLI から実物に話しかけるのは `TurboFieldfareCLI --model … --messages-file …`**
(family は `manifest.json` から読む)。
**HTTP から話しかけるのは `TurboFieldfareServer --model …`** — 家族は同じく
`manifest.json` から読み、起動ログの `family=` が名乗る ([26](26-PHASE8-SERVER.md))。
参照の fixture は `scratch/qwen35/decode-fixture-55.json`、
トークナイザの fixture は `scratch/qwen35/tokenizer-fixture.json`。
repack 済みモデルは `scratch/ornith-oq4e-g64.gturbo`、repack の入力は
`~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-baked`。**参照器には焼き込み前の
`~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64` を渡す** (`q_norm` の 1/16 は本ランタイム
専用の細工なので、参照の算式には入れない)。

**まだ手を付けていない前提:**

- ~~**tokenizer は確実に弾かれる**~~ → 片づいた ([22 §1](22-PHASE5-TOKENIZER.md))。
  弾いていたのは **Gemma のローダ**で、それは正しい。Ornith は `QwenTokenizer` が開く
- ~~**ツール呼び出しはサーバーがまだ呼んでいない**~~ → 片づいた
  ([26](26-PHASE8-SERVER.md))。`QwenChatGrammarBuilder` に読み手ができ、
  実物が HTTP 経由でツールを呼び、その応答を読んで答えた。**vision (Phase 9)
  は手つかず**で、画像は 400 で断る
- **Ornith の経路は貪欲のまま。**マスクつき argmax は分布を作らないので、
  温度を入れるには語彙幅の logit ヘッドが要る ([25 §1](25-CLI-TOOLS.md))。
  **サーバーはこれを断らず、受理して無視する** (R3) — 既定の要求の
  `temperature` は 1.0 なので、断ると既定の客が全部 400 になる
  ([26 §4-1](26-PHASE8-SERVER.md))
- 運用点 (スロット数・チャンク幅) は Gemma 4 の値のままで、Ornith 用には何も測っていない
- **参照器では bf16 との差は測れない** ([14 §7](14-REFERENCE.md))
- **パッケージテストは `--no-parallel` で回す。**素の `swift test` は remote install 系が
  20 本前後落ちるが、これは変更前の tree でも同じ ([18 §7](18-MIXED-BITS.md))
