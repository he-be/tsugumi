# 01. 2 機の分担と 1 サイクル

前提は [TWO_MACHINE_DEV.md](../TWO_MACHINE_DEV.md) (役割・同期・ホスト札・ストレージ)。ここは prefill の仕事に絞った運用だけを書く。

## 1. 分担

| | MBP (Mac15,6 / M3 Pro / macOS 15 / Xcode 26.6) | M6 mini (Mac18,5 / 16 GB / macOS 27 / CLT のみ) |
| --- | --- | --- |
| 編集 | **ここだけ** | しない (`receive.denyCurrentBranch updateInstead` が守る) |
| 正しさ | `swift test`、`TsugumiKernelCheck`、fixtures、macOS 15 の回帰 | tensor 経路の CPU 参照との一致 (**MBP では tensor 経路が存在しないので、ここでしか取れない**) |
| 性能の記録値 | 取らない (比較の対照としてだけ取る) | **すべて** |
| GUI が要る作業 | Instruments、GPU キャプチャ、Mac アプリ | できない |
| tensor ops (MSL 4.0) | **コンパイルすら通らない** ([証拠](../../bench/m6/results/2026-09-22-Mac15-6-gpu_family_probe/output.txt)) | 通り、PSO も作れる ([証拠](../../bench/m6/results/2026-09-22-Mac18-5-gpu_family_probe/output.txt)) |

**MBP で tensor 経路の正しさは一切検証できない。**`swift test` が緑でも tensor カーネルは 1 行も走っていない。
G1 以降の正しさは M6 で `TsugumiKernelCheck` と greedy 一致を取る ([04 G1](04-GATES.md))。

## 2. 1 サイクル (MBP から)

```bash
# 1. MBP で編集して commit (cycle.sh は dirty tree を拒む — ホスト札の commit を正直に保つため)
git commit -am '...'

# 2. push → M6 でビルド → 計測 → 証拠を引き戻す
bench/m6/cycle.sh prefill_bench.sh <label> 6          # prefill の記録値 (6 本)
TF_PREFILL_GPU_PROFILE=1 bench/m6/cycle.sh prefill_bench.sh <label>-profile 3   # GPU 内訳
bench/m6/cycle.sh run_probe.sh gpu_family_probe        # 単発プローブ

# 3. 引き戻した bench/m6/results/<date>-Mac18-5-<label>/ を結果文書と一緒に commit
```

MBP 側の対照は同じスクリプトを直接叩く (`bench/m6/prefill_bench.sh <label> 6`)。
ディレクトリ名の `Mac15-6` / `Mac18-5` が機械の札で、`hostinfo.txt` に commit と minos が入る。

`cycle.sh` は `TF_*` / `PROMPT_TOKENS` / `MODEL` の環境変数を M6 に運ぶ。A/B の腕は**環境変数で切り替え、同じバイナリで取る**。

## 3. ヘッドレス固有の注意

- **モデルは M6 の内蔵 SSD** (`~/dev/tsugumi/scratch/gemma4-qat-sym.gturbo`) から読む。外付け・ネットワーク越しでは測らない。
- 外付け `SSD256` は GUI ログインが無いので自動マウントされない。`ssh m6 'diskutil mount disk5s1'`。
- `sudo purge` が使えないので**ページキャッシュを冷やせない**。`prefill_bench.sh` が各 run の前に `vm_stat` と `memory_pressure -Q` を `run-N.pre.txt` に残す。外れ値はそれで説明する。
- M6 に Xcode は無い。Metal System Trace も GPU フレームキャプチャも M6 では取れない ([05 §5](05-RISKS.md))。
- 別のモデルプロセスが動いていると `prefill_bench.sh` は測らずに止まる (`pgrep`)。
- 2026-09-22 時点で M6 内蔵の空きは 129 GB。Qwen (G6) の 21.9 GB は入るが、**測り終えたら消す**。

## 4. MBP で並行できること

M6 が計測している間、MBP でできるのは tensor に依らない仕事だけ:

- G5 (チャンク境界の io 直列) の実装と正しさ。数字は M6 で取る。
- G1 の Swift 側の書き換え (`swift build` と既存テストは通る。経路が死んでいるだけ)。
- 結果文書と台帳の更新。
