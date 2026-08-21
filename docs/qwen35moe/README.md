# Qwen3.5-MoE (Ornith-1.5-35B-A3B) 対応 — 計画

QAT・Vision・MTP に続く 4 つ目の大改修。
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
(`architectures: ["Qwen3_5MoeForConditionalGeneration"]`, `model_type: qwen3_5_moe`) を
本ランタイムに載せる。M3 Pro 18GB / macOS 15.7.5 / ブランチ `macos15-support`。
2026-08-21 起草。旧 `PLAN_QWEN35.md` (単一ファイル) を分割したもの。
プラン内で積み重なっていた追記の層 (§15 → §16 → §17) は畳んであり、
**各文書は現在の結論だけを書く。**実測の経緯と数字は 10 番台の結果文書が持つ。

**現在地 (2026-08-22): Phase 5 が閉じた — CLI が答え、ツール呼び出しも通る。**
Gated DeltaNet (`qwen_delta_rule`) は 2048 トークン後の状態が **CPU float32 の床と
3 桁一致**、prefill 30 層 **125.7 ms** ([15](15-PHASE2-GDN.md))。その周辺 7 本
(因果 `conv1d` + l2norm / 減衰ゲート / `RMSNormGated` / partial RoPE / 出力ゲート /
shared ゲート / SiLU) も通り、**検査は 29 本すべて緑、うち 6 本は負例**
([17](17-PHASE2-KERNELS.md))。突き合わせ先は [14](14-REFERENCE.md) の float32 参照器
(上流実装と相対 6.4e-07 / top-1 一致 100%)。`.gturbo` への repack は
`--verify-install` が緑 (20.49 GB、[13](13-PHASE1-REPACK.md))。
**本線は `oQ4e-g64`** — 同じ文章 4 本の平均 NLL が公式 MLX-4bit より 4 本とも低く、
MTP と vision の実物も入っている ([16 §1](16-QUALITY.md))。
**混在ビット幅は片づいた** ([18](18-MIXED-BITS.md)): 幅は manifest のスロットではなく
**常駐索引からテンソルごとに導く**ことにし、本線の `oQ4e-g64` が `Model.load` を
1,294 ms で通るようになった (`--qwen-open`)。形式は 1 バイトも変えていない。
**最後の 1 本だった LM head も書けた** ([19](19-LM-HEAD-INT8.md)): 本線の `lm_head` が
8-bit なので INT4 の specialization ではなく **INT8 の chain**で、実物の語彙で
1 トークン **4.0 ms / 134 GB/s**。`--qwen` の検査は 39 本すべて緑。
**Phase 3 の結線が通った** ([20](20-PHASE3-DECODE.md)): `QwenForwardRunner` (直列) と
`RecurrentStateManager` を新設し、固定プロンプトから参照と全一致、負例 5 本も落ちる。
実物で見えた食い違いは 3 つ (埋め込みと shared ゲートが 8-bit、routed の活性化が SiLU) で、
いずれも**落ちずにそれらしく間違う**形だった。`RealForwardRunner` は 1 行も動かしていない。
**Phase 3 は閉じ、Phase 4 の prefill が通った** ([21](21-PHASE4-PREFILL.md)):
参照を取り直したら **55 本目に `<|im_end|>` を出して止まっていた**ので、
「64 トークン」の出口条件は**その生成の全体と一致**という形で満たされた。
その 55 本が、**プロンプトを T 行の経路に通しても全部一致する** — チャンク幅
512 (1 チャンク) と 8 (3 チャンク) の両方で。要ったのは **INT8 の QMM**
(INT4 版は 8-bit を読めない) と T 行版の小物 3 本で、`--qwen` は **57 本**になった。
**Phase 5 が通った** ([22](22-PHASE5-TOKENIZER.md)): `QwenTokenizer` は
`GFTokenizer` の分岐ではなく**兄弟**で (Gemma の検査を緩めないため)、decode は
**ByteLevel** の streaming、framing は**上流の `chat_template.jinja` そのもの**。
`--qwen-tokenizer` の **223 本**は上流 `tokenizers` / `transformers` との
突き合わせで、うち 4 本は負例。CLI は `RunQwen.swift` という別経路で、
**推論 (`<think>`) は stderr、答えは stdout** に分かれる。
**Phase 5 の残りも片づいた** ([23](23-PHASE5-TOOLS.md)): XML 形の
ツール呼び出しに `QwenToolCallParser` と `QwenToolCallGrammar` が入り、
**テンプレート自身が描いた呼び出しを文法が受理し、パーサが同じ引数に戻す**
(往復 7 検体)。生の文字列値は **`\n</parameter>` を含まない任意のテキスト**を
13 状態で綴るので、`<b>bold</b>` のような引数が通る。`GrammarVocabulary` の
piece も ByteLevel 用の入口を足した — Gemma の規則だと**落ちずに日本語が
1 文字も通らなくなる**ので、負例で押さえてある。`--qwen-tools` は **36 本**、
うち 6 本は負例。**[22 §4-2](22-PHASE5-TOKENIZER.md) の「swift-jinja の
キー順は不定」は訂正した** — 昇順で、両方決定的である
([23 §5-1](23-PHASE5-TOOLS.md))。
**残るのはサーバー結線 (Phase 8) と CLI の `--tools`** ([04](04-PHASES.md))。
**Gemma の既定 69 本と `swift test` の 1,329 件 (旧 1,297 + 新規 32) は緑のまま。**
**速度・ヒット率・TTFT の運用数字は依然 導出 か 未確認** — 計測 (Phase 6) はこれから。
運用点 (スロット数・チャンク幅) は Gemma 4 の値をそのまま持ち越せない —
理由は [05 §1](05-RISKS.md)、測り直しの手順は [04](04-PHASES.md) Phase 6。

方針は既存 PLAN と同じ: **汎用性を捨てる。**この 1 台 (M3 Pro / 18GB /
macOS 15.7.5) で速いことだけを目的にし、互換性・移植性・他アーキテクチャへの
一般化は最初から狙わない。

---

## 結論を先に

| # | 論点 | 結論 |
| --- | --- | --- |
| 1 | これは何か | **Qwen3.5-MoE。ただの「Qwen 版 Gemma」ではない。**40 層のうち **30 層が線形注意 (Gated DeltaNet)**、10 層だけが full attention。SWA は 1 層も無い (**実測(上流)**、[01 §1](01-MODEL.md)) |
| 2 | 一番大きい実装 | **Gated DeltaNet カーネル → 書けた** ([15](15-PHASE2-GDN.md))。omlx の blocked-sequential の幾何を写し、状態はレジスタ、ホットループに barrier 無し。検証 15 本が緑、prefill 30 層 125.7 ms。**周辺 7 本も書けた** (検査 29 本、[17](17-PHASE2-KERNELS.md))。**KV キャッシュではなく固定サイズの再帰状態を持つ層**という構造変更 ([03 §3-3](03-DESIGN.md)) も**入った** — `RecurrentStateManager` は 62.8 MiB で文脈長に依らない ([20 §1-1](20-PHASE3-DECODE.md)) |
| 3 | 重み変換 | **当初の「bf16 → MLX 4-bit 変換器を新規に書く」は消えた。**MLX 4-bit 量子化済みの候補が 2 本手元にある ([02](02-CHECKPOINTS.md))。残るのは焼き込み (`q_norm` × 1/16)・名前寄せ・repack ([03 §1](03-DESIGN.md)) |
| 4 | チェックポイントの選定 | **`oQ4e-g64` に決めた** (21.86 GB、imatrix + MTP + vision、router は BF16)。同じ文章 4 本の平均 NLL が公式 MLX-4bit より 4 本とも低い ([16 §1](16-QUALITY.md))。対価だった**混在ビット幅は片づいた** — 幅は索引から導く ([18](18-MIXED-BITS.md)) |
| 5 | MoE の形は乗るか | **乗る。ただし `numExperts <= 256` の precondition にちょうど乗る (余裕ゼロ)。**top-8 は一致、`D=2048` は 64 の倍数、prefill router のスクラッチは既に 256 で確保済み。**decode/prefill の MoE カーネルは無改造で正しく動く**見込み (専用化 PSO から汎用 PSO に落ちるだけ) ([03 §4](03-DESIGN.md)) |
| 6 | エキスパート 1 個のバイト数 | **1,769,472 B = 16 KiB × 108 ちょうど。パディング 0 バイト** ([01 §3-2](01-MODEL.md))。導出がのちに実物と 3 回バイト一致し、**実測に格上げ済み** ([10 §2](10-MLX4BIT-AUDIT.md))。Gemma は 205 ページ中 13,312 B が捨て札なので、そこは改善 |
| 7 | 1 トークンあたりのバイト | **導出で Gemma の 0.78 倍** (4K 文脈、全ヒット時 2.41 GB → 1.89 GB)。**decode は Gemma より速くなり得る**。ただしヒット率が落ちる要因が別にある (#8) ([01 §3-5](01-MODEL.md)) |
| 8 | 一番大きい性能リスク | **エキスパート母集団が 3,840 → 10,240 に増える。**32 スロットが覆う割合は 1 層あたり 25% → 12.5% に半減する。スロット 1 本の値段は 96.1 → 67.5 MiB に下がるので**バイト等価なら 45 本**だが、`allowedExpertCacheSlots` は `[8,16,24,32]` で頭打ち ([05 §1-1](05-RISKS.md)) |
| 9 | 二番目に大きい性能リスク | **prefill の GEMM 占有率が半減する。**チャンク 2048 でエキスパート 1 個あたり平均 128 行 → 64 行。64 行ブロックがちょうど 1 個しか埋まらない ([05 §1-2](05-RISKS.md)) |
| 10 | 一番大きい正しさリスク | **partial RoPE のペアの取り方が Gemma と違う。**本ランタイムは `(i, HD/2+i)` を回し周波数の分母に `HD` を使う。Qwen は `(i, 32+i)` を回し分母は `rotary_dim=64`。**既存カーネルを流用すると静かに間違う** ([03 §2-2](03-DESIGN.md)) |
| 11 | ただで貰えるもの | (a) `attention_prefill_causal_qblock_d256` は head_dim だけで選ばれるので**そのまま当たる** (b) RMSNorm は Qwen も `1+w` 規約なので既存カーネルのまま (c) 文脈長は 10 層ぶんしか KV が要らず、線形層の状態は**文脈長に依らず 62.8 MiB 固定** ([01 §3-4](01-MODEL.md)) |
| 12 | MTP | **本モデルは MTP ヘッドを同梱している** (`mtp.*`、1 層、805M のエキスパートつき)。**専用ドラフターを別リポジトリから取ってくる必要が無い。**実物 (一式 503 MB、うちエキスパート 453 MB) は oQ4e(-g64) に入っており、**全 256 エキスパートを常駐にできる** = ドラフトの I/O がゼロになる ([03 §6](03-DESIGN.md)) |
| 13 | サーバーへの波及 | **prompt cache と投機デコードの前提が壊れる。**再帰状態は「途中を捨てる」「巻き戻す」ができない。SPEC の LCP 再利用と MTP の verify ブロックの両方に設計変更が要る ([03 §5](03-DESIGN.md)) |
| 14 | Vision | 後回しでよい。**tower の形は Gemma と偶然ほぼ同じ** (1152 / 27 層 / 16 head / 4304) だが、**位置符号化とマージが別物**。カーネルは書き直し ([04 §11](04-PHASES.md))。bf16 の tower 実物 (893 MB) は oQ4e に入っている |

---

## 読む順

| # | 文書 | 役割 |
| --- | --- | --- |
| 1 | [01-MODEL.md](01-MODEL.md) | 上流の事実 (config・テンソル・算式)、Gemma 4 26B-A4B との差分、数量 (パラメータ・ページ・バイト予算) |
| 2 | [02-CHECKPOINTS.md](02-CHECKPOINTS.md) | チェックポイント候補 2 本と手元の持ち物、供給源選定の経緯、oQ / omlx から取り入れるもの・取り入れないもの |
| 3 | [03-DESIGN.md](03-DESIGN.md) | 変換と repack の残作業、新規カーネル 8 本、ランタイムの構造変更、既存資産の再利用可否、サーバー波及、MTP |
| 4 | [04-PHASES.md](04-PHASES.md) | Phase 0〜9 の分解と出口条件、Vision の中身、次の一手 |
| 5 | [05-RISKS.md](05-RISKS.md) | 性能リスク、中止線、残る未確認、やらないこと |
| 6 | [10-MLX4BIT-AUDIT.md](10-MLX4BIT-AUDIT.md) | **実測(手元)。**公式 MLX-4bit の検証、上流 bf16 との規約照合、減衰ゲート (`in_proj_a`) の感度測定 |
| 7 | [11-OQ4E-G64-REBUILD.md](11-OQ4E-G64-REBUILD.md) | **実測(手元)。**oQ4e-mtp の取得と、非互換 248 本の 8-bit g64 打ち直し (`oQ4e-g64` の作成) |
| 8 | [12-OQ4E-G64-AUDIT.md](12-OQ4E-G64-AUDIT.md) | **実測(手元)。**`oQ4e-g64` の規約照合 (norm / `conv1d` / router) と、2 候補を同じ物差しで並べた表、`q_norm` の焼き込み |
| 9 | [13-PHASE1-REPACK.md](13-PHASE1-REPACK.md) | **実測(手元)。**`.gturbo` への repack と `--verify-install`、形式に足した 3 つのセクション、混在ビット幅という Phase 3 の宿題 |
| 10 | [14-REFERENCE.md](14-REFERENCE.md) | **実測(手元)。**float32 の層ストリーミング参照器、逆量子化と算式の検証、実物の初回 forward と生成スモーク、fixtures |
| 11 | [15-PHASE2-GDN.md](15-PHASE2-GDN.md) | **実測(手元)。**Gated DeltaNet カーネル (`qwen_delta_rule`)、3 精度での検証 15 本、TB の 3 通りと 30 層の時間 |
| 12 | [16-QUALITY.md](16-QUALITY.md) | **実測(手元)。**2 候補の平均 NLL (本線の決定) と、`in_proj_a` の実活性再測 (未決着) |
| 13 | [17-PHASE2-KERNELS.md](17-PHASE2-KERNELS.md) | **実測(手元)。**`qwen.metal` の 7 本、負例 6 本を含む検査 29 本、fast math と減衰ゲート、conv のトークン分割 (50.0 → 21.9 ms)、LM head が INT8 だと分かった件 |
| 14 | [18-MIXED-BITS.md](18-MIXED-BITS.md) | **実測(手元)。**混在ビット幅を索引から導く決定、Qwen 用の常駐スキーマ、負例 5 本、本線が開いた記録 |
| 15 | [19-LM-HEAD-INT8.md](19-LM-HEAD-INT8.md) | **実測(手元)。**INT8 の LM head chain、負例 4 本、1 トークン 4.0 ms / 134 GB/s、天井が 79 → 71 tok/s になる話 |
| 16 | [20-PHASE3-DECODE.md](20-PHASE3-DECODE.md) | **実測(手元)。**decode の結線、トークンの一致、負例 5 本、実物で見えた 3 つの食い違い、再帰状態の置き場所 |
| 17 | [21-PHASE4-PREFILL.md](21-PHASE4-PREFILL.md) | **実測(手元)。**55 トークンの参照 (Phase 3 が閉じた)、INT8 の QMM と検査 18 本、prefill の結線、チャンクをまたぐもの 3 つ、チャンク幅の時間 |
| 18 | [22-PHASE5-TOKENIZER.md](22-PHASE5-TOKENIZER.md) | **実測(手元)。**ByteLevel のトークナイザと上流 jinja、上流との突き合わせ 223 本 (負例 4)、CLI の Ornith 経路、tools の JSON の食い違い (キー順の記述は [23 §5-1](23-PHASE5-TOOLS.md) が訂正) |
| 19 | [23-PHASE5-TOOLS.md](23-PHASE5-TOOLS.md) | **実測(手元)。**XML 形のツール呼び出しのパーサと GBNF、生の値の 13 状態、ByteLevel の piece 表、検査 36 本 (負例 6)、テンプレートとの往復で閉じたもの・残ったもの |

## 表記

PLAN.md / PLAN_QAT.md / PLAN_VISION.md と同じ **実測** / **導出** / **未確認**。
ただし本計画の **実測** は 2 種類ある。混ぜないために区別する:

| 記号 | 意味 |
| --- | --- |
| **実測(上流)** | 上流リポジトリの実体を取得して確認した事実。`config.json`、`model.safetensors.index.json`、各シャードの safetensors ヘッダ (HTTP range で先頭のみ取得)、`tokenizer_config.json`、`chat_template.jinja`、`transformers` の `modeling_qwen3_5_moe.py`、omlx のソース |
| **実測(手元)** | この機械で数字を取ったもの。**CPU のもの** (ファイルを読む / 逆量子化する / float32 で流す) と、**GPU のもの** ([15](15-PHASE2-GDN.md) 以降のカーネル検査とマイクロベンチ) がある。**モデルを載せて動かしたのは [20](20-PHASE3-DECODE.md) が最初**で、そこの時間は直列・prefill 無しの n=1 なので運用値ではない |

## 運用ルール

- 各文書は**現在の結論**を書く。実測の経緯と数字は 10 番台の結果文書が持ち、
  **測定の結論を二重に書かない** (docs/mtp と同じ作法)。食い違ったら結果文書
  (番号の大きいもの) が正
- 反復 3 未満のセルには解釈を書かない。数字だけ置く
- 運用点 (スロット数・チャンク幅) の**既定は変えない**。候補追加の提案に留め、
  判断はユーザーが行う (Gemma の `[8,16,24,32]` は 2026-08-20 のユーザー確定値)
- Gemma 4 経路の実測値を動かさない。共有コードに手を入れるときは Gemma のベンチを取り直す

## 参照

- 上流 (bf16): [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
- 公式 MLX 4-bit: [`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit)
- oQ 4-bit (MTP / vision つき): [`scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp`](https://huggingface.co/scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp)
- 量子化ツール: [`jundot/omlx`](https://github.com/jundot/omlx) (Apache-2.0) —
  `docs/oQ_Quantization.md` / `omlx/oq.py` / `omlx/custom_kernels/qwen35_prefill/gdn.py`
- `transformers` `models/qwen3_5_moe/modeling_qwen3_5_moe.py`
- 本リポジトリ: [PLAN.md](../../PLAN.md) / [PLAN_QAT.md](../../PLAN_QAT.md) /
  [PLAN_VISION.md](../../PLAN_VISION.md) / [docs/mtp/README.md](../mtp/README.md) /
  [docs/SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) / [docs/serving/SPEC.md](../serving/SPEC.md)
