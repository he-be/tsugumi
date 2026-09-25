# QFN の prefill を速くする — MBP + M6 の 2 機で回す。実験で分け方を決めてから作る

2026-09-25 起草。**QFN** = Qwen3.8-Flash-Next (DS4-IQ2、`docs/qwen38/` の本線)。
Gemma の M6 prefill ([m6-prefill](../m6-prefill/README.md)) に区切りが付いたので、本題の QFN に戻る。

**前提は 1 つ。運用の形は「MBP 単体で QFN が動き、M6 mini を TB4 で直結すると速くなる」である。**
MBP (Mac15,6 / M3 Pro 18 GB / 1 TB) がアプリとサーバーを持ち、M6 mini (Mac18,5 / 16 GB / 256 GB + 外付け SSD256) は TB4 直結の先で仕事の一部を引き受ける。
この特定の 2 機に特化した分散推論として作る。**分け方は 2 つ (層で分ける A / expert で分ける B、[08](08-CANDIDATES.md)) あり、どちらを作るかは
A・B の模擬の壁時計で決める ([07](07-EXPERIMENTS.md)、[12](12-Q1-REDESIGN.md))。**10 で B としたが取り下げ ([11](11-Q1-REOPEN-TENSOR-OPS.md))、前提 P1〜P5 を先に固定して設計し直した ([12](12-Q1-REDESIGN.md))。

2 機の取り決めそのものは [TWO_MACHINE_DEV.md](../TWO_MACHINE_DEV.md) だが、**§1 (性能の記録値は M6 でしか取らない) と §3 (TB4 直結は当てにしない) は
QFN では当てはまらない** ([01 §1](01-ROLES.md)、判断待ち)。

**約束は m6-prefill と同じ。実測の数字は `bench/qfn/results/` の証拠ディレクトリを指す。**指せない数字は「公称」「導出」「未確認」の
札を付けるか、書かない。規則は [02](02-EVIDENCE.md)。

## 現在地

**進捗は [06-LEDGER.md](06-LEDGER.md) の既定行の秒だけ。**単体 (MBP だけ) と 2 機 (MBP + M6) の 2 列を持ち、どちらも秒。倍率・%・tok/s は進捗ではない。

| | 値 | 出所 |
| --- | --- | --- |
| **台帳** | 単体: 本文 1 (自然文 5 本) 28.384〜30.540 s、本文 2 (記録の再生 1 ターン) 156.233 s (Q2 の後)。閾値 3.06 s | [06](06-LEDGER.md)、[13](13-Q0-BASELINE.md)、[14](14-Q2-PLE-PREFETCH.md) |
| MBP 単体の prefill (旧文書、`bench/qfn/results/` の証拠なし) | ベンチ 32K で 336.5 s (n = 1)。アプリの 1 ラウンドは prefill 秒 ≈ 6.20 + 新規 ÷ 89.6 (83 ラウンドの回帰) | [qwen38/23](../qwen38/23-KV-Q8-32K.md)、[qwen38/42 §1-1](../qwen38/42-KVA-SPEED-ON-APP-ROUND-PLAN.md)、[03 §3](03-FACTS.md) |
| チャンク 2048 の 1 本の中身 (MBP、8K ベンチ) | 20〜22 s = pre-router 6.4〜7.2 + route 6.3〜6.5 (pread 約 5、top-10 約 1.3) + routed 6.8〜6.9 + PLE 0.1〜2.2。**GPU 11.5〜12.2 s と SSD 約 5 s が直列** | [qwen38/08 §4](../qwen38/08-ROUTED-MIXED.md)、[03 §3](03-FACTS.md) |
| M6 で QFN を動かした実測 | 今のコードで 4K を Swapouts なしで通した。GPU の秒は MBP とほぼ同じで、SSD と PLE が長い | [10 §1.2](10-Q1-SPLIT-DECISION.md) |
| TB4 直結の生 TCP | 両方向とも 1 GiB 4.7 GB/s、84 MB 片道と 21 MB 往復 × 48 の秒は 10 §1.1 | [10 §1.1](10-Q1-SPLIT-DECISION.md) |
| 分け方 A / B の秒 | 10 の B は取り下げた。式の単体 1922 s はアプリ実測 1618 s と A / B の差と同じだけ外れていて、式では選ばない。前提 P1〜P5 の下で F0〜F4 を取って模擬の壁時計で決める | [12](12-Q1-REDESIGN.md) |

| Gate | 問い | 状態 |
| --- | --- | --- |
| **Q0** 基準線 | 今のコードの MBP 単体を `bench/qfn/` の形で取る (自然文 5 本 + 記録の再生 1 ターン)。揺れの幅を決める | **済み ([13](13-Q0-BASELINE.md))**。本文 2 の揺れ幅 1.529 s、閾値 3.06 s。段の秒は 14 の `q2-b2-off` が兼ねる |
| **Q1** 分け方を決める実験 | 前提 P1〜P5 ([12 §1](12-Q1-REDESIGN.md))。F1 M6 の 1 層を tensor ops で、F2 A の模擬、F3 B の模擬、F4 常駐の上限。模擬の壁時計で A か B を選ぶ | **済み、A** ([15](15-Q1-F1-LAYER.md)〜[18](18-Q1-F2-F3-PAIR.md))。記録の 1 ターンは A 837.5 s (k 40 常駐、格子の端)・B 978.3 s (σ 0.55 常駐)・模擬の単体 1365.6 s。M6 の 1 層の GEMM は常駐 59.7〜61.3 ms、常駐は自前のバッファで 11.10 GB まで ([17](17-Q1-F4-OWNED.md)) |
| **Q2** PLE の行を先に読む | 行番号はトークンだけで決まるので、要求の頭で全チャンク分を並列に読む。A / B どちらでも効く | **既定に入れた ([14](14-Q2-PLE-PREFETCH.md))**。本文 2 は 168.702 → 156.233 s。短いチャンクの `ple` 0.24〜0.52 s は未確認 |
| **Q3** host の top-10 | チャンクごとに約 1.3 s の Swift ループを縮める。どちらでも効く | 未着手 |
| **Q4** M6 の第二の読み口 | 腕 a: 外付けに expert の写し (内蔵 3.34 + 外付け 1.57 GB/s)。**腕 b: MBP の内蔵 SSD から TB4 で送る (内蔵 3.34 + link 4.73 GB/s)**。A の非常駐層の読みを縮める | **腕 b は模擬で通った ([20](20-Q4B-EXPERT-FEED.md))**。A k 40 常駐の 2 チャンク 18.509 → 12.916 s (比 0.59)、M6 のチャンク −3.1 s。TB ポートは別バスで競合なし。MBP の Pageins は増えた (壁時計の中) |
| **Q5** 機内で読みと GPU を重ねる | 次の層の expert を固定の置き場 (2 層分) に `F_NOCACHE` で読む。**RAM の条件付き** | **M6 側は A に入れる ([21](21-Q5-M6-OVERLAP.md)、ユーザーの判断)**。k 48・送り 0.59 の M6 のチャンク 6.301 → 3.829 s、1 ターン (導出) 529.9 → 322.5 s。Swapouts 0、wired 14.34 GB。残りは非常駐層の読みの待ち。MBP 側は判断待ち 3 (k 48 なら対象なし) |
| **Q6** A を作り、正しさと秒 | M6 が層 0〜k−1、常駐 13 層。正しさは ds4-metal と同じ基準 (04)。k は 1 つから (模擬では切り替えが閾値に届かない、[20 §3](20-Q4B-EXPERT-FEED.md)) | **decode は MBP 単体で確定 ([24](24-DECODE-SPLIT-RESULT.md))**: MTP on で層で分けた 2 機 (j 16・24・32)・M6 が幹の全層はどれも遅い (s/token 中央値 S0 0.1265、S16 0.1421)。P6 ([22](22-Q6-P6-DECODE-PLACEMENT.md)) のまま、**次は k の格子 (44・46・48、SLAB 2)**。分けた腕で I/O は減っていて (S0 111〜114 → S24 82〜84 ms)、遅いのは M6 の T = 2 の I/O 以外 (層あたり MBP の約 2 倍、T = 1 では同じ)。その理由は単体では出ず、split-decode の serve 側 ([26](26-M6-T2-TRUNK-RESULT.md)。計画 [25](25-M6-T2-TRUNK-PLAN.md))。送り (比 0.59) の模擬では k 48 (M6 が全層、MBP は送るだけ) が最小で 1 ターン 529.9 s (導出、[20 §1.5](20-Q4B-EXPERT-FEED.md))。k 44・46・48 から |
| **Q7** decode と会話の機能 | 2 機で decode が単体より遅くならない。MTP・チェックポイント・prompt cache・ツール呼び出しが通る | Q6 の後 |
| **Q8** 小さいラウンドの固定分 | 1 ラウンド 6.2 s の固定分と 8 トークンのチャンク 2.5 s | 未着手 |

Gate の測り方と合否は [04](04-GATES.md)。順は Q0 → Q2 → Q1 (済み) → Q4 の腕 b (模擬は済み、[20](20-Q4B-EXPERT-FEED.md)) → Q5 の M6 側 (A に入れる、[21](21-Q5-M6-OVERLAP.md)) → **Q6 (P6 を先に書く) → Q7** ([19](19-AFTER-Q1-PLAN.md)、20 §3)。Q3 は Q6 と並行可。Q5 の MBP 側は判断待ち 3 で、k 48 が通れば対象が無い。

## 判断待ち (ユーザー)

1. **記録値を取る機械**: QFN の記録値は「MBP が駆動する 2 機」と「MBP 単体」の 2 列で取る。TWO_MACHINE_DEV §1 の「M6 でしか取らない」は Gemma の規則として残し、QFN はこの README を正にしてよいか。
2. **TB4 直結を常設にする**か。運用の形が 2 機なら、ケーブルは繋ぎっぱなしが前提になる (IP は link-local で変わる、[01 §3](01-ROLES.md))。
3. **Q5 の RAM (MBP 側)**: 2 層分 (約 1.4 GiB) の固定の置き場を取る。先読み・常駐は decode で「RAM を使えば速いというだけ」として閉じた経緯がある ([qwen38/13](../qwen38/13-KERNEL-TO-GPU-WAIT.md)、[15 §0-2](../qwen38/15-DECODE-PLAN.md))。prefill でも同じ判断なら Q5 の MBP 側は外す。M6 側の常駐と wired limit は P2 で決まった ([12 §1](12-Q1-REDESIGN.md)) ので、M6 側はこの判断を待たずに測る (04 Q5)。k 48 が Q6 で通れば MBP に prefill の層は無く、この判断は要らなくなる。
4. (閉じた) M6 の PLE BF16 は外付けにしか置けず、置いてある ([03 §5](03-FACTS.md))。A では Q2 の先読みが外付けから読む。決めることは無い ([19 §3](19-AFTER-Q1-PLAN.md))。

## 読む順

| 文書 | 役割 |
| --- | --- |
| [01-ROLES.md](01-ROLES.md) | 2 機の分担 (QFN 用)。TB 直結の繋ぎ方、1 サイクル、ヘッドレスの注意 |
| [02-EVIDENCE.md](02-EVIDENCE.md) | 証拠の規則。m6-prefill/02 を継ぎ、2 機ぶんのホスト札と link の札を足す |
| [03-FACTS.md](03-FACTS.md) | 事実の台帳: モデルの寸法、1 チャンクの読み、2 機の SSD / GPU / RAM、TB の実測、重みの置き場 |
| [04-GATES.md](04-GATES.md) | Q0〜Q8。問い・測り方・合否 |
| [05-RISKS.md](05-RISKS.md) | 罠。link-local の IP、2 機の OS 差、16 GB、外付けの手動マウント、ほか |
| [06-LEDGER.md](06-LEDGER.md) | **台帳。進捗の数字はここの既定行だけ。** |
| [07-EXPERIMENTS.md](07-EXPERIMENTS.md) | **分け方を決める実験 (E0〜E2 済み、F0〜F4) と、決め方** |
| [08-CANDIDATES.md](08-CANDIDATES.md) | 分け方の候補 A / B の形と式。数字は書かない |
| 10 番台〜 | 結果文書。Gate ごとに 1 本、[00-TEMPLATE-RESULT.md](00-TEMPLATE-RESULT.md) の形 |
| [10-Q1-SPLIT-DECISION.md](10-Q1-SPLIT-DECISION.md) | Q1: E0〜E4 の実測と式 (結論は取り下げ) |
| [11-Q1-REOPEN-TENSOR-OPS.md](11-Q1-REOPEN-TENSOR-OPS.md) | Q1 の開け直し: 理由、ds4-metal の手本、**再開手順** (計画は 12 へ) |
| [12-Q1-REDESIGN.md](12-Q1-REDESIGN.md) | **Q1 の設計し直し: 揺れた理由、前提 P1〜P5、F0〜F4、決め方** |
| [13-Q0-BASELINE.md](13-Q0-BASELINE.md) | Q0: 単体の基準線と閾値 |
| [14-Q2-PLE-PREFETCH.md](14-Q2-PLE-PREFETCH.md) | Q2: PLE の先読み (既定) |
| [15-Q1-F1-LAYER.md](15-Q1-F1-LAYER.md) | Q1 F1: M6 の 1 層を tensor ops で (P1 の経路と精度、冷 / 常駐) |
| [16-Q1-F4-RESIDENCY.md](16-Q1-F4-RESIDENCY.md) | Q1 F4: M6 の常駐の上限 (page cache の形) |
| [17-Q1-F4-OWNED.md](17-Q1-F4-OWNED.md) | Q1 F4 の取り直し: 自前のバッファに持つ形 (F2・F3 の前提) |
| [18-Q1-F2-F3-PAIR.md](18-Q1-F2-F3-PAIR.md) | Q1 F2・F3: A と B の模擬の壁時計、12 §3 の判定 |
| [19-AFTER-Q1-PLAN.md](19-AFTER-Q1-PLAN.md) | **Q1 の後の手順**: 待ちとチャンク幅の答え、Q4 の腕 b (MBP の SSD から TB4 で送る)、TB ポートの対称性、Q6 の k の規則 |
| [20-Q4B-EXPERT-FEED.md](20-Q4B-EXPERT-FEED.md) | Q4 の腕 b: TB ポートの確認と、MBP の内蔵 SSD から TB4 で送る模擬 (比 0.5 / 0.59) |
| [26-M6-T2-TRUNK-RESULT.md](26-M6-T2-TRUNK-RESULT.md) | 25 の結果: 単体・MTP on で M6 の verify の I/O 以外 96.8 ms は MBP 103.4 ms より短い。24 の S48 の 229 ms は split-decode の serve 側。step の差 (286 対 258 ms) は I/O (SSD) |
| [25-M6-T2-TRUNK-PLAN.md](25-M6-T2-TRUNK-PLAN.md) | **計画**: M6 の T = 2 の幹の I/O 以外が MBP の層あたり 2 倍な理由 (GPU / host / フォルト / 道具) を、単体・MTP on の検証 1 行の内訳で判別する |
| [24-DECODE-SPLIT-RESULT.md](24-DECODE-SPLIT-RESULT.md) | decode の置き場の判別の結果: MTP on で MBP 単体が最短、P6 確定 |
| [23-DECODE-SPLIT-PLAN.md](23-DECODE-SPLIT-PLAN.md) | decode の置き場 (MBP 単体 / 層で分けた 2 機 / M6 が幹の全層) を MTP on で判別する実験の計画 |
| [22-Q6-P6-DECODE-PLACEMENT.md](22-Q6-P6-DECODE-PLACEMENT.md) | Q6 の前提 P6: decode は MBP、状態の持ち主と受け渡し、両機単体の decode の実測 |
| [21-Q5-M6-OVERLAP.md](21-Q5-M6-OVERLAP.md) | Q5 の M6 側: 非常駐層の読みを前の層の GPU の裏に回す模擬 (staging 1 / 2 層分、k 48・送り 0.59) |
| `bench/qfn/` | 計測スクリプトと証拠。E0〜E4 の道具と `formula.py`、Q0 の `pair_bench.sh --solo`・`ledger.sh` (13) がある。2 機の形と `results/README.md` は未作成。本文 1 は `bench/m6/prompts/` の自然文 5 本を共用する |

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
