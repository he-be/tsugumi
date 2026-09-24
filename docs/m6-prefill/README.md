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
| **既定の prefill 壁時計** (台帳の最新行) | 2.626 (2.617〜3.030)、commit 084de6b、2026-09-25、n = 6。検証を含まない量 (`trusted-install`) で、最初の行 (ef29110) は 2.897 | [06 §1](06-LEDGER.md)、[36](36-FULL-ATTN-UNFUSED.md) |
| 同じ M6・同じ本文の外部ランタイム (校正、進捗ではない) | llama.cpp IQ3_XXS 2.393 s (tensor API を切ると 4.494 s)、mlx_lm 3-bit 1.823 s (peak 13.3 GB、スワップあり)。暖まったプロセスの値なので (c) ではなく GPU busy 2.97〜3.00 s と並べる | [29](29-REFERENCE-RUNTIMES-ON-M6.md) |
| 到達目標 (ユーザー指定) | 同じ本文で mlx_lm の 1.823 s。GPU busy は 2.20 s。31 の差の残りは `rope` → `moe.down` → `moe.gateup`。`attn.full` は MLX と同じ GEMM 形で 0.042 s (MLX 0.073 s) になった | [31](31-MLX-KERNELS-PER-STAGE.md)、[36](36-FULL-ATTN-UNFUSED.md) |

**全注意 (headDim 512) を MLX と同じ GEMM → softmax → GEMM にした** (2.897 → 2.740 → 2.626 s、102fe93 と f4075ac、[36](36-FULL-ATTN-UNFUSED.md))。その前に **5769e1f の検証の先回り深さ 8 は取り下げた** (縮んだのは検証の待ちで、prefill の計算ではない。b0b4de2 で戻した、[34](34-VERIFY-AHEAD-DEPTH.md))。その前は **6f9ae16 で router の GEMM と shared の投入順を既定にした** (同じバイナリの交互で 3.185 → 3.150 s、台帳は 3.134 → 3.142 s で縮んでいない、[33](33-ROUTER-GEMM-SHARED-LATE.md))。その前は **3.496 → 3.134 s** (883df28、射影と shared MLP の QMM を MLX の `affine_qmm_t_nax` から写したものに、[32](32-QMM-NAX-DEFAULT.md))。その前は 4.584 → 3.496 s (cd8c26b、[26](26-SWA-ATTN-MPP-DEFAULT.md))、その前は 7.184 → 4.584 s (5decec4、[23](23-GPU-CANDIDATES-DEFAULT.md))、 8.682 → 7.184 s (79eea8f、[20](20-VERIFY-AHEAD-ALONE.md))。基準線 9.506 s → 8.682 s がどの commit で動いたかは未確認。

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
| 10 番台〜 | 結果文書。Gate ごとに 1 本、[00-TEMPLATE-RESULT.md](00-TEMPLATE-RESULT.md) の形。[18](18-RESET-ABSOLUTE-FLOOR.md) は方向の取り直し |
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
