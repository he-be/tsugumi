# QFN の prefill を速くする — MBP + M6 の 2 機で回す。実験で分け方を決めてから作る

2026-09-25 起草。**QFN** = Qwen3.8-Flash-Next (DS4-IQ2、`docs/qwen38/` の本線)。
Gemma の M6 prefill ([m6-prefill](../m6-prefill/README.md)) に区切りが付いたので、本題の QFN に戻る。

**前提は 1 つ。運用の形は「MBP 単体で QFN が動き、M6 mini を TB4 で直結すると速くなる」である。**
MBP (Mac15,6 / M3 Pro 18 GB / 1 TB) がアプリとサーバーを持ち、M6 mini (Mac18,5 / 16 GB / 256 GB + 外付け SSD256) は TB4 直結の先で仕事の一部を引き受ける。
この特定の 2 機に特化した分散推論として作る。**分け方は 2 つ (層で分ける A / expert で分ける B、[08](08-CANDIDATES.md)) あり、どちらを作るかは
M6 で QFN を動かした秒と link の秒を測ってから式で決める ([07](07-EXPERIMENTS.md))。**2026-09-25 の Q1 で **B** に決めた ([10](10-Q1-SPLIT-DECISION.md))。

2 機の取り決めそのものは [TWO_MACHINE_DEV.md](../TWO_MACHINE_DEV.md) だが、**§1 (性能の記録値は M6 でしか取らない) と §3 (TB4 直結は当てにしない) は
QFN では当てはまらない** ([01 §1](01-ROLES.md)、判断待ち)。

**約束は m6-prefill と同じ。実測の数字は `bench/qfn/results/` の証拠ディレクトリを指す。**指せない数字は「公称」「導出」「未確認」の
札を付けるか、書かない。規則は [02](02-EVIDENCE.md)。

## 現在地

**進捗は [06-LEDGER.md](06-LEDGER.md) の既定行の秒だけ。**単体 (MBP だけ) と 2 機 (MBP + M6) の 2 列を持ち、どちらも秒。倍率・%・tok/s は進捗ではない。

| | 値 | 出所 |
| --- | --- | --- |
| **台帳** | まだ行が無い (`bench/qfn/` のスクリプトが未作成) | [06](06-LEDGER.md) |
| MBP 単体の prefill (旧文書、`bench/qfn/results/` の証拠なし) | ベンチ 32K で 336.5 s (n = 1)。アプリの 1 ラウンドは prefill 秒 ≈ 6.20 + 新規 ÷ 89.6 (83 ラウンドの回帰) | [qwen38/23](../qwen38/23-KV-Q8-32K.md)、[qwen38/42 §1-1](../qwen38/42-KVA-SPEED-ON-APP-ROUND-PLAN.md)、[03 §3](03-FACTS.md) |
| チャンク 2048 の 1 本の中身 (MBP、8K ベンチ) | 20〜22 s = pre-router 6.4〜7.2 + route 6.3〜6.5 (pread 約 5、top-10 約 1.3) + routed 6.8〜6.9 + PLE 0.1〜2.2。**GPU 11.5〜12.2 s と SSD 約 5 s が直列** | [qwen38/08 §4](../qwen38/08-ROUTED-MIXED.md)、[03 §3](03-FACTS.md) |
| M6 で QFN を動かした実測 | 今のコードで 4K を Swapouts なしで通した。GPU の秒は MBP とほぼ同じで、SSD と PLE が長い | [10 §1.2](10-Q1-SPLIT-DECISION.md) |
| TB4 直結の生 TCP | 両方向とも 1 GiB 4.7 GB/s、84 MB 片道と 21 MB 往復 × 48 の秒は 10 §1.1 | [10 §1.1](10-Q1-SPLIT-DECISION.md) |
| 分け方 A / B の秒 | **B に決めた** (導出、式に host の残り o を足した)。記録 12 ターンの prefill 合計は 10 §0 | [10](10-Q1-SPLIT-DECISION.md) |

| Gate | 問い | 状態 |
| --- | --- | --- |
| **Q0** 基準線 | 今のコードの MBP 単体を `bench/qfn/` の形で取る (自然文 5 本 + 記録の再生 1 ターン)。揺れの幅を決める | 未着手 |
| **Q1** 分け方を決める実験 | E0 link、E1 M6 で QFN の段の秒、E2 M6 の SSD、E3 B の 1 層の模擬、E4 M6 の expert GEMM (fp32 / fp16 tensor ops)。式で A か B を選ぶ | **閉じた: B** (σ 0.40、写しあり 0.45)。E4-(b) は精度の規則で使わず、Q8 は作らない ([10](10-Q1-SPLIT-DECISION.md)) |
| **Q2** PLE の行を先に読む | 行番号はトークンだけで決まるので、要求の頭で全チャンク分を並列に読む。A / B どちらでも効く | 未着手 |
| **Q3** host の top-10 | チャンクごとに約 1.3 s の Swift ループを縮める。どちらでも効く | 未着手 |
| **Q4** M6 の外付けに expert の写し | Gemma の 44 と同じ形。内蔵 3.34 + 外付け 1.57 GB/s を同時に読む。どちらでも効く | 未着手 (E2 で写しを先に作る) |
| **Q5** 機内で読みと GPU を重ねる | 次の層の expert を固定の置き場 (2 層分) に `F_NOCACHE` で読む。**RAM の条件付き** | 未着手 (判断待ち) |
| **Q6** 選んだ分け方を作り、正しさと秒 | B を作る。単体と logits が一致。台帳の 2 機の列 | 未着手 (Q1 済み) |
| **Q7** decode と会話の機能 | 2 機で decode が単体より遅くならない。MTP・チェックポイント・prompt cache・ツール呼び出しが通る | Q6 の後 |
| **Q8** 小さいラウンドの固定分 | 1 ラウンド 6.2 s の固定分と 8 トークンのチャンク 2.5 s | 未着手 |

Gate の測り方と合否は [04](04-GATES.md)。順は Q0 → Q1 → (Q2、Q3、Q4 は分け方に依らないので Q1 と並行可) → Q6 → Q7。Q5 は判断待ち。

## 判断待ち (ユーザー)

1. **記録値を取る機械**: QFN の記録値は「MBP が駆動する 2 機」と「MBP 単体」の 2 列で取る。TWO_MACHINE_DEV §1 の「M6 でしか取らない」は Gemma の規則として残し、QFN はこの README を正にしてよいか。
2. **TB4 直結を常設にする**か。運用の形が 2 機なら、ケーブルは繋ぎっぱなしが前提になる (IP は link-local で変わる、[01 §3](01-ROLES.md))。
3. **Q5 の RAM**: 2 層分 (約 1.4 GiB) の固定の置き場を取る。先読み・常駐は decode で「RAM を使えば速いというだけ」として閉じた経緯がある ([qwen38/13](../qwen38/13-KERNEL-TO-GPU-WAIT.md)、[15 §0-2](../qwen38/15-DECODE-PLAN.md))。prefill でも同じ判断なら Q5 は外す。
4. **M6 の PLE BF16 (95 GiB)**: Q1 で B に決まったので要らない ([10 §4](10-Q1-SPLIT-DECISION.md))。消すかどうか。

## 読む順

| 文書 | 役割 |
| --- | --- |
| [01-ROLES.md](01-ROLES.md) | 2 機の分担 (QFN 用)。TB 直結の繋ぎ方、1 サイクル、ヘッドレスの注意 |
| [02-EVIDENCE.md](02-EVIDENCE.md) | 証拠の規則。m6-prefill/02 を継ぎ、2 機ぶんのホスト札と link の札を足す |
| [03-FACTS.md](03-FACTS.md) | 事実の台帳: モデルの寸法、1 チャンクの読み、2 機の SSD / GPU / RAM、TB の実測、重みの置き場 |
| [04-GATES.md](04-GATES.md) | Q0〜Q8。問い・測り方・合否 |
| [05-RISKS.md](05-RISKS.md) | 罠。link-local の IP、2 機の OS 差、16 GB、外付けの手動マウント、ほか |
| [06-LEDGER.md](06-LEDGER.md) | **台帳。進捗の数字はここの既定行だけ。** |
| [07-EXPERIMENTS.md](07-EXPERIMENTS.md) | **分け方を決める実験 E0〜E4 と、決め方** |
| [08-CANDIDATES.md](08-CANDIDATES.md) | 分け方の候補 A / B の形と式。数字は書かない |
| 10 番台〜 | 結果文書。Gate ごとに 1 本、[00-TEMPLATE-RESULT.md](00-TEMPLATE-RESULT.md) の形 |
| [10-Q1-SPLIT-DECISION.md](10-Q1-SPLIT-DECISION.md) | Q1: E0〜E4 と式。B に決めた |
| `bench/qfn/` | 計測スクリプトと証拠。E0〜E4 の道具と `formula.py` はある。Q0 の `pair_bench.sh`・`ledger.sh` は未作成。本文は `bench/m6/prompts/` の自然文 5 本を共用する |

## 運用ルール (m6-prefill と同じ)

1. README は現在地の表と索引だけ。結論の本文を書かない。表の各セルは 2 文まで。
2. 01〜08 は現在の結論だけ。経緯と数字は結果文書 (10 番台〜) に置く。同じ数字を 2 か所に書かない。食い違ったら結果文書が正。
3. 結果文書は Gate ごとに 1 本、150 行まで。
4. 「YYYY-MM-DD 追記」の層を作らない。古くなった記述は書き換える。経緯は git log が持つ。
5. 実測の数字は証拠ディレクトリへのリンクを付けて書く。文書内で計算した値は「導出」と書く。**進捗は台帳の既定行の秒だけ。**倍率・%・tok/s を進捗として書かない。
6. 2 機の数字と単体の数字を 1 つの行に混ぜない。2 機の行には両方のホスト札と link の札が付く。
7. 運用点の既定 (32K、KV Q8_0、チャンク 2048、thinking off、公式のサンプリング値) は変えない。提案に留め、判断はユーザー。
8. 挙動 (答えの良し悪し) は測らない。Mac で測るのは秒だけ ([qwen38/37](../qwen38/37-RESTART-ON-PC-Q4KXL.md))。
9. **設計を選ぶ根拠は実測だけ。**測っていない量で A / B を選び直さない。選び直すなら [07](07-EXPERIMENTS.md) に実験を足してから。
