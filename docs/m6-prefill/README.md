# M6 で prefill を速くする — 計画と証拠の置き場

2026-09-22 起草。**MBP** (M3 Pro 18 GB / macOS 15 / GUI) と **M6 mini** (16 GB / macOS 27 / ヘッドレス) の
2 機で prefill (pp / TTFT) を伸ばす仕事の文書群。2 機の取り決めそのものは [TWO_MACHINE_DEV.md](../TWO_MACHINE_DEV.md)。
機種側の旧い見立て [investigations/M6_MAC_MINI_TARGETING.md](../investigations/M6_MAC_MINI_TARGETING.md) は
2026-09-01〜22 の追記が層になって読めなくなったので**凍結**する (参照のみ、追記しない)。以降はここに書く。

**この文書群の約束は 1 つ。M6 について書く実測の数字は、すべて `bench/m6/results/` の証拠ディレクトリを指す。**
指せない数字は「公称」「導出」「未確認」の札を付けるか、書かない。規則は [02-EVIDENCE.md](02-EVIDENCE.md)。

## 現在地

**進捗は [06-LEDGER.md](06-LEDGER.md) の既定行の秒だけで測る** (2026-09-24、[18](18-RESET-ABSOLUTE-FLOOR.md))。
既定のバイナリ・固定コマンドの prefill 壁時計を、スクリプトが証拠ディレクトリから生成した表に載せる。
G0 からの比、段の秒、tensor ops に載せたか、「下限の何倍か」は、どれも進捗ではない。

| | 値 (秒) | 出所 |
| --- | --- | --- |
| **既定の prefill 壁時計、自然文** (2026-09-25、`w1-def-*`、各 n = 4、feca5d5 と同じソース) | `ja-twomachine` 2.595、`ja-m6readme` 3.441、`en-readme` 4.407、`en-server` 4.568 s。1 run で SSD から 5.9〜12.0 GiB 読み、expert io 1.4〜3.3 s。台帳の形 (`ledger-*`) ではまだ取っていない | [43](43-LEDGER-PROMPT-REPEATED-SENTENCE.md) |
| 台帳の最新行 (繰り返し文。**進捗の物差しから取り下げ**) | 1.768 (1.747〜1.922)、commit 619c515、n = 6。本文 `bench/l.json` は同じ 1 文の 50 回繰り返しで、SSD の読みが 3 GiB に収まる本文だった。最初の行 (ef29110) は 2.897 | [06 §1](06-LEDGER.md)、[43](43-LEDGER-PROMPT-REPEATED-SENTENCE.md) |
| 同じ M6・繰り返し文の外部ランタイム (校正、進捗ではない) | llama.cpp IQ3_XXS 2.393 s (tensor API を切ると 4.494 s)、mlx_lm 3-bit 1.823 s (peak 13.3 GB、スワップあり)。重みを全部載せる実装なので本文の足あとで秒が変わらない。自然文では取り直していない | [29](29-REFERENCE-RUNTIMES-ON-M6.md) |
| **既定: 外付け SSD の写しから 32% を読む** (9d66daa、`expert-mirror.json`) | 自然文 5 本を腕ごとにまとめて流して、写しなし 3.79〜4.49 → 既定 3.32〜3.72 s (trusted-install、各 n = 4)。full-sha256 でも 4.26〜4.63 → 3.62〜3.94 s。内蔵 3.3 + 外付け 1.57 GB/s を同時に読む | [44](44-EXPERT-MIRROR-EXTERNAL-SSD.md) |
| 到達目標 (ユーザー指定) | mlx_lm の 1.823 s。**「下回った」(42) は取り下げた。**繰り返し文でだけ 1.768 s で、自然文では 2.6〜4.6 s。比較は自然文の同じ組で取り直す | [43](43-LEDGER-PROMPT-REPEATED-SENTENCE.md) |

**台帳の本文が繰り返し文だったと分かった** ([43](43-LEDGER-PROMPT-REPEATED-SENTENCE.md)、2026-09-25)。36〜42 の縮みは自然文でも出ている (`old` → 既定で 3.859 → 2.595 s など) が、自然文の残り時間の大半は SSD の読み (expert io) で、カーネルの変更では縮まない。台帳の本文は自然文の組に替える (43 §3、スクリプトは未変更)。その前は **routing が分かった層のエキスパートを一度に先読みするようにした** (1.914 → 1.768 s、619c515、[42](42-ADVISE-LAYER-AT-ROUTING.md)。読む量は同じで、頼むのが早くなった)。その前は **routed MoE の GEMM を MLX 写しの 16×16 断片にした** (2.267 → 1.914 s、0564855、[40](40-MOE-GEMM-NAX.md)。mpp とビット一致)。QKV エピローグの 1 ディスパッチ化は GPU busy が変わらず既定に入れていない ([39](39-QKV-EPILOGUE-FUSED.md))。SSD の側 (1 run 約 3 GiB) は、層の先読みで読みが 8 GiB に増え、1 周で済ませても読みは減らない ([41](41-SSD-READ-AHEAD.md))。その前は **router を tensor の 16×16 断片にした** (2.482 → 2.267 s、5122f36、[38](38-ROUTER-NAX.md))。その前は **SWA の attention も帯の GEMM → softmax → GEMM にした** (2.626 → 2.482 s、a9d4ce6、[37](37-SWA-ATTN-UNFUSED.md)。greedy の本文が 5 トークン目で変わった)。その前は **全注意 (headDim 512) を MLX と同じ GEMM → softmax → GEMM にした** (2.897 → 2.740 → 2.626 s、102fe93 と f4075ac、[36](36-FULL-ATTN-UNFUSED.md))。その前に **5769e1f の検証の先回り深さ 8 は取り下げた** (縮んだのは検証の待ちで、prefill の計算ではない。b0b4de2 で戻した、[34](34-VERIFY-AHEAD-DEPTH.md))。その前は **6f9ae16 で router の GEMM と shared の投入順を既定にした** (同じバイナリの交互で 3.185 → 3.150 s、台帳は 3.134 → 3.142 s で縮んでいない、[33](33-ROUTER-GEMM-SHARED-LATE.md))。その前は **3.496 → 3.134 s** (883df28、射影と shared MLP の QMM を MLX の `affine_qmm_t_nax` から写したものに、[32](32-QMM-NAX-DEFAULT.md))。その前は 4.584 → 3.496 s (cd8c26b、[26](26-SWA-ATTN-MPP-DEFAULT.md))、その前は 7.184 → 4.584 s (5decec4、[23](23-GPU-CANDIDATES-DEFAULT.md))、 8.682 → 7.184 s (79eea8f、[20](20-VERIFY-AHEAD-ALONE.md))。基準線 9.506 s → 8.682 s がどの commit で動いたかは未確認。

| Gate | 問い | 状態 |
| --- | --- | --- |
| **B0** 測る量を分ける | 起動と初回検証 (a) / 同じプロセスでの 2 回目以降 (b) / 新しいプロセスでの 1 回目 (c、台帳の量) | (c) は ef29110 から検証なし (`trusted-install`) で測る ([35](35-PREFILL-WITHOUT-VERIFY.md))。(b) の口は無い |
| **B1** 段ごとの下限表 | 各段の (実測 − 下限) の秒。手を入れる段を選ぶ道具で、進捗の物差しではない | **済み** ([24](24-B1-FLOOR-TABLE.md)、(c) で取った)。今の既定の上の差の大きい順は `moe.gateup` 0.49 → `shared` 0.43 → `post.router` 0.35 → `attn.full` 0.24 s ([27](27-B1-FLOOR-TABLE-ON-SWA-MPP.md)) |
| **B2** 下限から遠い段から直す | 1 段ずつ。段の秒と、台帳の行 | 1 段目 `attn.swa` は既定に入った (GPU 0.873 → 0.288 s、[25](25-SWA-ATTN-MPP.md)・[26](26-SWA-ATTN-MPP-DEFAULT.md))。2 段目 `moe.gateup` は候補で 0.753 → 0.567 s、閾値 0.50 s に届かず不合格 ([28](28-MOE-GATEUP-PREFETCH.md))。3 段目 `shared` (+ `qkv`・`oproj`) は MLX から写したカーネルで `shared` 0.570 → 0.417 s、閾値 0.25 s に届かず不合格だが、壁時計が縮んだので既定に入れた ([32](32-QMM-NAX-DEFAULT.md))。4 段目 `post.router` は GEMM 化で閾値 0.05 s に対して 0.346 s、不合格。区間が shared と重なっていた ([33](33-ROUTER-GEMM-SHARED-LATE.md)) |
| **B3** 候補を既定に入れる | 検証の並行先読み (79eea8f) と `mpp` / `mpp64` / `sgmm` (5decec4) | 23 までの候補は**済み** ([23](23-GPU-CANDIDATES-DEFAULT.md))。`TF_PREFILL_ATTN=mpp` も既定に入った ([26](26-SWA-ATTN-MPP-DEFAULT.md)、中央値は合格範囲の速い側に 0.05 s 外れ、対照も同じ向き)。`TF_PREFILL_QMM=nax` も既定に入った ([32](32-QMM-NAX-DEFAULT.md))。router の GEMM と shared の投入順も既定に入った ([33](33-ROUTER-GEMM-SHARED-LATE.md))。検証の先回りの深さ 8 は取り下げた ([34](34-VERIFY-AHEAD-DEPTH.md)) |

以前の Gate (G0〜G5) の結果と読み替えは [04 の末尾](04-GATES.md) の索引にある。

**律速の移り変わり (2026-09-24、[20](20-VERIFY-AHEAD-ALONE.md)〜[23](23-GPU-CANDIDATES-DEFAULT.md))**: 既定は層ファイルの SHA-256 の直列で律速していた。検証を裏に回した後は GPU 側が律速になり、候補 3 つを既定にして 4.584 s。

**既定の性質 ([22](22-GPU-CANDIDATES-GREEDY-AND-FIRST-RUN.md))**: greedy の本文は、旧既定と 3 行目で分かれる。直前のモデルプロセスから空きがあると、1 本目の layerVerify が 9.4〜10.1 s に伸び、壁時計が 0.9 s 前後遅くなる。原因は未調査。

## 読む順

| 文書 | 役割 |
| --- | --- |
| [01-ROLES.md](01-ROLES.md) | 2 機の分担と、MBP → M6 の 1 サイクル。ヘッドレス固有の注意 |
| [02-EVIDENCE.md](02-EVIDENCE.md) | 証拠の規則。過去にこの repo で起きた「でっちあげ」の例と、LLM への禁止事項 |
| [03-FACTS.md](03-FACTS.md) | M6 の実測台帳 (値 / 日付 / 証拠のパス)。証拠が無い行は無いと書いてある |
| [04-GATES.md](04-GATES.md) | G0〜G6。問い・測り方・合否・不合格なら何をやめるか |
| [05-RISKS.md](05-RISKS.md) | 罠。16 GB のページキャッシュ、単体ベンチの符号反転、Xcode 無し、ほか |
| [06-LEDGER.md](06-LEDGER.md) | **台帳。進捗の数字はここの既定行だけ。**`bench/m6/ledger.sh` の出力を貼る |
| 10 番台〜 | 結果文書。Gate ごとに 1 本、[00-TEMPLATE-RESULT.md](00-TEMPLATE-RESULT.md) の形。[18](18-RESET-ABSOLUTE-FLOOR.md) と [43](43-LEDGER-PROMPT-REPEATED-SENTENCE.md) は方向の取り直し |
| [46](46-WIRED-DURING-RUN.md) | prefill の外。要求を処理している間の wired (10.4〜10.7 GiB、終われば戻る)。48 §7 の見込みの訂正 |
| [47](47-KAJIMACHINE-REPLAY-TSUGUMI-VS-LLAMACPP.md) | prefill の外。kajimachine の鑑定の要求を再生し、Tsugumi (QAT) と llama.cpp (IQ3_XXS) を並べた。読み直しの量、1 回の秒、IQ3_XXS の英名の崩れ |
| [48](48-KAJIMACHINE-EXPERT-SLICE.md) | prefill の外。鑑定で使うエキスパートだけを残した Q4_0 (層ごと 80 個、8.77 GiB) を llama.cpp で M6 に載せた。1 回 7.54 s・Swapouts 0。文脈長は wired を動かさない。英名の崩れの率 |
| [49](49-KAJIMACHINE-SLICE-CHECKS.md) | prefill の外。48 の 80 個版の確かめ。ログイン + Chrome + Web カメラでも 1 回 8.09 s (wired 上限は既定)。残すリストは 460 枚で選んでも変わらない。英名の崩れはスキーマの `maxLength` で 0 |
| `bench/m6/prompts/` | 台帳用の自然文の本文 5 本 (43 §3)。`bench/l.json` は繰り返し文で、進捗の判定には使わない |
| `bench/m6/` | プローブと計測スクリプト。[results/README.md](../../bench/m6/results/README.md) が証拠の置き方 |

## 運用ルール — この文書群を長大化させないために

1. **README は現在地の表と索引だけ。**結論の本文を書かない。表の各セルは 2 文まで。
2. **01〜05 は現在の結論だけ。**経緯と数字は結果文書 (10 番台〜) に置く。**同じ数字を 2 か所に書かない。**食い違ったら結果文書が正。
3. **結果文書は Gate ごとに 1 本、150 行まで。**超えるなら次の番号で分ける。
4. **「YYYY-MM-DD 追記」の層を作らない。**古くなった記述は書き換える。経緯は git log が持つ。
5. 実測の数字は証拠ディレクトリへのリンクを付けて書く。文書内で計算した値は「導出」と書き、元の実測を指す。
   **進捗は [06](06-LEDGER.md) の既定行の秒だけ。**倍率・%・pp・「下限の何倍か」を進捗として書かない。変更の効果は秒の差で書く。
6. 他プロジェクト (mflux 等) の数字を引かない。この repo のプローブで取り直して置く。
7. 運用点の既定 (スロット数・チャンク幅・`allowedExpertCacheSlots`) は変えない。提案に留め、判断はユーザー。
