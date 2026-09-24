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
| **既定の prefill 壁時計** (台帳の最新行) | 8.682 (8.633〜8.801)、commit 22d7629、2026-09-24、n = 6 | [06 §1](06-LEDGER.md) |
| 既定でない最良 (候補、`TF_` 4 つ) | 4.407〜4.443、commit 3c45f83 | [06 §2](06-LEDGER.md) |

**既定は基準線 9.506 s から 8.682 s に動いていた** (差 −0.824 s)。どの commit で動いたかは未確認。

| Gate | 問い | 状態 |
| --- | --- | --- |
| **B0** 測る量を分ける | 起動と初回検証 (a) / 同じプロセスでの 2 回目以降 (b) / 新しいプロセスでの 1 回目 (c、台帳の量) | (b) を測る口が無い。CLI の変更が要るので判断待ち |
| **B1** 段ごとの下限表 | 各段の (実測 − 下限) の秒。手を入れる段を選ぶ道具で、進捗の物差しではない | 未着手 |
| **B2** 下限から遠い段から直す | 1 段ずつ。段の秒と、台帳の行 | B1 の後 |
| **B3** 候補を既定に入れる | `TF_PREFILL_MOE=mpp` / `TF_PREFILL_QMM=mpp64` / `TF_PREFILL_VERIFY_AHEAD[_PARALLEL]` を既定にするか | 判断待ち。入れるまで台帳は動かない |

以前の Gate (G0〜G5) の結果と読み替えは [04 の末尾](04-GATES.md) の索引にある。

**中断中の作業 (2026-09-24)**: attn.swa (既定構成の GPU 内訳で 1.43 s、[`h0-profile-detail`](../../bench/m6/results/2026-09-24-Mac18-5-h0-profile-detail/)) を
simdgroup_matrix の flash カーネル `attention_prefill_causal_sgmm_d256` (71786c5、`TF_PREFILL_ATTN=sgmm` の候補) に置き換える試み。
- CPU 参照との一致: `Scripts/test.sh --filter PrefillAttentionTests` が MBP で通る。
- M6 の本番ループ ([`sgmm-profile-detail`](../../bench/m6/results/2026-09-24-Mac18-5-sgmm-profile-detail/)、n = 3、profile): attn.swa 1.64 s。**既定の qblock の 1.43 s より遅い。**
- カーネル単体 (`bench/m6/attn_bench.swift`、MBP、記録値ではない): qblock 1.73 s、sgmm 1.12 s。MBP と M6 で順位が逆。

**再開手順**:
1. M6 で単体ベンチを回し、M6 でも sgmm が遅いかを見る (本番ループの外で順位が逆になるかの切り分け):
   `scp bench/m6/attn_bench.swift Sources/Tsugumi/Metal/Prefill/prefill.metal m6:/tmp/ && ssh m6 'cd /tmp && swiftc -O attn_bench.swift -o attn_bench && ./attn_bench prefill.metal attention_prefill_causal_qblock_d256:256:32 attention_prefill_causal_sgmm_d256:128:32'`
2. M6 でも遅ければ、カーネルを直す。候補: threadgroup メモリからの読みが MMA 1 回あたり 1〜1.5 回と多い (QK は Q と Kᵀ の 2 本で 2 回の MMA)。
   S を 2×2 の simdgroup で割る / PV を head dim で割って V の断片を 4 行ブロックで使い回す / GQA の 2 ヘッドを 1 threadgroup にまとめて K/V の読みを半分に。
3. 単体で qblock を下回ったら `TF_PREFILL_ATTN=sgmm TF_PREFILL_GPU_PROFILE=2 bench/m6/cycle.sh prefill_bench.sh sgmm-profile-detail-2 3` → 壁時計 `TF_PREFILL_ATTN=sgmm bench/m6/cycle.sh prefill_bench.sh sgmm-a 6`。
4. 候補のままなので台帳 §1 は動かない。既定に入れるかはユーザーの判断 ([04 B3](04-GATES.md))。結果文書は 19 番 (番号は `ls` で確かめてから)。

その後は B3 の判断と、B0 (b) の口。

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
