# M6 で prefill を速くする — 計画と証拠の置き場

2026-09-22 起草。**MBP** (M3 Pro 18 GB / macOS 15 / GUI) と **M6 mini** (16 GB / macOS 27 / ヘッドレス) の
2 機で prefill (pp / TTFT) を伸ばす仕事の文書群。2 機の取り決めそのものは [TWO_MACHINE_DEV.md](../TWO_MACHINE_DEV.md)。
機種側の旧い見立て [investigations/M6_MAC_MINI_TARGETING.md](../investigations/M6_MAC_MINI_TARGETING.md) は
2026-09-01〜22 の追記が層になって読めなくなったので**凍結**する (参照のみ、追記しない)。以降はここに書く。

**この文書群の約束は 1 つ。M6 について書く実測の数字は、すべて `bench/m6/results/` の証拠ディレクトリを指す。**
指せない数字は「公称」「導出」「未確認」の札を付けるか、書かない。規則は [02-EVIDENCE.md](02-EVIDENCE.md)。

## 現在地

| Gate | 問い | 状態 |
| --- | --- | --- |
| **G0** 基準線 | M6 は何を返し、素の pp はいくつか | **G0-1 / G0-2 は両機で証拠あり** ([03](03-FACTS.md))。**G0-3 (pp) は証拠ディレクトリ無し** — 旧文書の数字しかないので取り直す |
| G1 tensor 経路を開ける | group 32 のモデルで MPP 経路が本当にディスパッチされるか | 未着手 |
| G2 `matmul2d` の素の実効値 | 自前カーネルから tensor ops で何 TFLOP/s 出るか | 未着手。プローブは未作成 |
| G3 routed MoE GEMM の tensor 化 | 本番ループで GPU 時間と壁時計が縮むか | 未着手 (G2 合格が条件) |
| G4 attention / 射影の tensor 化 | 同上 | 未着手 (G3 の後) |
| G5 チャンク境界の io 直列の解消 | io を GPU の陰に隠せるか | 未着手 (G0 の後、G1〜G4 と独立) |
| G6 Qwen 経路への波及 | Qwen prefill は M6 で io と gpu のどちらが重いか | 未着手 (モデルを M6 へ運ぶところから) |

**次の一手**: G0-3 を両機で取る。MBP は `bench/m6/prefill_bench.sh g0-baseline 6`、
M6 は `bench/m6/cycle.sh prefill_bench.sh g0-baseline 6`。結果は `10-G0-BASELINE.md` に書く (まだ無い)。

## 読む順

| 文書 | 役割 |
| --- | --- |
| [01-ROLES.md](01-ROLES.md) | 2 機の分担と、MBP → M6 の 1 サイクル。ヘッドレス固有の注意 |
| [02-EVIDENCE.md](02-EVIDENCE.md) | 証拠の規則。過去にこの repo で起きた「でっちあげ」の例と、LLM への禁止事項 |
| [03-FACTS.md](03-FACTS.md) | M6 の実測台帳 (値 / 日付 / 証拠のパス)。証拠が無い行は無いと書いてある |
| [04-GATES.md](04-GATES.md) | G0〜G6。問い・測り方・合否・不合格なら何をやめるか |
| [05-RISKS.md](05-RISKS.md) | 罠。16 GB のページキャッシュ、単体ベンチの符号反転、Xcode 無し、ほか |
| 10 番台〜 | 結果文書。Gate ごとに 1 本、[00-TEMPLATE-RESULT.md](00-TEMPLATE-RESULT.md) の形 |
| `bench/m6/` | プローブと計測スクリプト。[results/README.md](../../bench/m6/results/README.md) が証拠の置き方 |

## 運用ルール — この文書群を長大化させないために

1. **README は現在地の表と索引だけ。**結論の本文を書かない。表の各セルは 2 文まで。
2. **01〜05 は現在の結論だけ。**経緯と数字は結果文書 (10 番台〜) に置く。**同じ数字を 2 か所に書かない。**食い違ったら結果文書が正。
3. **結果文書は Gate ごとに 1 本、150 行まで。**超えるなら次の番号で分ける。
4. **「YYYY-MM-DD 追記」の層を作らない。**古くなった記述は書き換える。経緯は git log が持つ。
5. 実測の数字は証拠ディレクトリへのリンクを付けて書く。文書内で計算した値は「導出」と書き、元の実測を指す。
6. 他プロジェクト (mflux 等) の数字を引かない。この repo のプローブで取り直して置く。
7. 運用点の既定 (スロット数・チャンク幅・`allowedExpertCacheSlots`) は変えない。提案に留め、判断はユーザー。
