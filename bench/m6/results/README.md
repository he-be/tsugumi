# bench/m6/results — 証拠ディレクトリ

`docs/m6-prefill/02-EVIDENCE.md` §2 の置き場。**ここに無い M6 の数字は文書に書かない。**

- 命名: `YYYY-MM-DD-<hw.model>-<label>/`。`Mac15-6` = MBP (M3 Pro)、`Mac18-5` = M6 mini。
- 作るのはスクリプト (`../run_probe.sh`、`../prefill_bench.sh`) で、手で作らない。
- 中身: `hostinfo.txt` (commit 入り) / `command.txt` / `env.txt` / `run-N.log` / `run-N.pre.txt` / `summary.tsv` / `output.txt`。
- `run-N.log` は削らない。結果文書は `summary.tsv` から中央値を引く。
- 結果文書と同じ commit に入れる。
