# 02. 証拠の規則 — m6-prefill/02 を継ぎ、2 機ぶんを足す

札 (実測 / 導出 / 公称 / 未確認)、LLM への禁止事項、過去の「でっちあげ」の例は [m6-prefill/02](../m6-prefill/02-EVIDENCE.md) の §1・§3・§4 がそのまま効く。
ここには QFN と 2 機で変わるところだけ書く。

## 1. 証拠ディレクトリ

`bench/qfn/results/YYYY-MM-DD-<形>-<label>/`。`<形>` は `pair` (MBP + M6) か `Mac15-6` (MBP 単体)。スクリプトが自動で作る。

| ファイル | 中身 | 作る側 |
| --- | --- | --- |
| `hostinfo.mbp.txt` | `bench/hostinfo.sh` の 1 行 (MBP、commit と minos 入り) | 自動 |
| `hostinfo.m6.txt` | 同じく M6。`pair` のときだけ | 自動 |
| `link.txt` | `bridge0` の両端の IP、`ping -c 5` の rtt 中央値、1 GiB の生 TCP の秒、分け方 (A の k か B の σ)。`pair` のときだけ | 自動 |
| `command.txt` | 実行した完全なコマンド (MBP 側と、M6 側の起動コマンド) | 自動 |
| `env.txt` | `Q38_*` / `TF_*` の環境変数 (A/B の腕の証明)。両機ぶん | 自動 |
| `run-N.log` | MBP 側の生の stdout + stderr。削らない | 自動 |
| `run-N.m6.log` | M6 側のプロセスの生ログ。`pair` のときだけ | 自動 |
| `run-N.pre.txt` / `run-N.post.txt` | 両機の `vm_stat` / `memory_pressure -Q` / `iostat -d -I` (disk ごとの累積 MB)。差が run 中の Pageins / Swapouts / 読んだ量 | 自動 |
| `round.jsonl` | `Q38_ROUND_LOG` のチャンクごとの StepProfile (両機) | 自動 |
| `summary.tsv` | footer から抜いた表。文書はここから中央値を引く | 自動 |

- 解釈を書くセルは **n ≥ 3**。中央値とレンジ。外れ値は捨てずに残して `pre.txt` で説明する。
- 証拠ディレクトリは結果文書と同じ commit に入れる。**両機の `hostinfo` の commit は同じでなければならない** (`cycle.sh` が push してからビルドするので揃う)。
- 文書からのリンクは `../../bench/qfn/results/<dir>/summary.tsv` の形。

## 2. 2 機の数字の読み方

1. **壁時計は MBP の時計で 1 つ。**M6 の秒は `run-N.m6.log` の「層 0〜k−1 の秒」で、壁時計の成分であって別の壁時計ではない。2 機の秒を足して壁時計にしない (重なっている)。
2. **単体の行と 2 機の行は別の列。**同じ commit・同じ本文・同じスクリプトで、M6 を起動するかどうかだけが違う。
3. **link の待ち** (MBP が M6 の返事を待った秒) はログに出し、`summary.tsv` の列にする。link が律速なら E0 / E3 ([07](07-EXPERIMENTS.md)) の値が本番で再現していない。
4. 読んだ量は disk ごと (MBP 内蔵、M6 内蔵、M6 外付け) に書く。縮んだのが読む量か読む速さかを分ける (m6-prefill/44 の読み方)。
5. **本文は自然文で、run ごとに替える** (m6-prefill/43)。同じ本文を続けて流すとその本文のページがキャッシュに残る。QFN は 1 チャンクで約 32 GiB 読むので、18 GB / 16 GB のキャッシュに残る量は小さいが、dense (約 5 GB) と PLE の行は残る。**冷えている run と暖かい run を混ぜない。**

## 3. 測る量 (m6-prefill/04 B0 の QFN 版)

| 量 | 中身 | 台帳 |
| --- | --- | --- |
| (a) 起動 | GGUF の mmap、M6 側のプロセスの起動、link の確立 | 入れない。別に書く |
| (b) 同じプロセスでの 2 回目以降 | アプリの 2 ターン目以降の形。dense と PLE の一部がキャッシュに残る | **記録の再生 (1 ターン、複数ラウンド) がこれ**。台帳の本文 2 |
| (c) 新しいプロセスでの 1 回目 | 自然文 1 本、空のキャッシュ | 台帳の本文 1 |

QFN には層ファイルの SHA-256 検証が無い (GGUF を mmap するだけ) ので、m6-prefill/35 の「検証なし」の区別は要らない。

## 4. QFN で特に禁じること

1. **tok/s を進捗として書かない。**「60 t/s → N t/s」は分母 (新規トークン数) を選べる。秒と秒の差で書く。
2. PC (RTX 5080) や llm-server の数字を Mac の期待値にしない ([investigations/M6_QWEN38_FLASH_NEXT](../investigations/M6_QWEN38_FLASH_NEXT.md) は全部導出で、SSD の前提が既に外れている)。
3. 挙動 (答えの中身) を Mac で評価しない ([qwen38/37](../qwen38/37-RESTART-ON-PC-Q4KXL.md))。greedy の本文は**記録する**が、それで止めない。
4. [08 §3](08-CANDIDATES.md) の式の値は導出で、目標でも進捗でもない。合否の閾値は [04](04-GATES.md) に計測前に置き、後から動かさない。
5. **測っていない量で分け方 (A / B) を選ばない。**選ぶのは [07](07-EXPERIMENTS.md) の E0〜E4 の後。
