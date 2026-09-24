# 01. 2 機の分担 (QFN 用) と 1 サイクル

前提は [TWO_MACHINE_DEV.md](../TWO_MACHINE_DEV.md) (同期・ホスト札・ストレージ)。ここは QFN で違うところだけ書く。

## 1. 分担 — Gemma と違うところ

| | MBP (Mac15,6 / M3 Pro 18 GB / macOS 15 / Xcode 26.6) | M6 mini (Mac18,5 / 16 GB / macOS 27 / CLT のみ、ヘッドレス) |
| --- | --- | --- |
| 運用での役割 | **アプリ・サーバー・トークナイザ・サンプラ・MTP・prompt cache・LM head。**駆動側 | TB4 直結の先で MBP から呼ばれる。持つものは分け方で違う (A: 層 0〜k−1 と PLE / B: 担当分の expert だけ、[08](08-CANDIDATES.md)) |
| 編集 | ここだけ | しない (`receive.denyCurrentBranch updateInstead`) |
| 正しさ | `swift test`、`TsugumiKernelCheck`、CPU 参照器 (`Scripts/qwen38/reference_forward.py`) との一致 | 2 機の出力が単体と一致すること (Q1) は**両機が要る** |
| 性能の記録値 | **2 機の行 (MBP が駆動) と、単体の行の両方をここから取る** | 単独では取らない (M6 単体は運用の形ではない。16 GB / 3.34 GB/s では QFN の単体運用が成り立たない、[03 §4](03-FACTS.md)) |
| GUI が要る作業 | Instruments、GPU キャプチャ、Mac アプリ | できない |

**TWO_MACHINE_DEV §1 の「性能の記録値は M6 でしか取らない」は Gemma (M6 が対象機) の規則。**QFN では MBP が駆動側で、
記録値は MBP の壁時計で取る。M6 の秒は M6 側のログから「層 0〜k−1 の秒」として添える。この読み替えは README の判断待ち 1。

## 2. 1 サイクル (MBP から)

```bash
# 1. MBP で編集して commit (cycle.sh は dirty tree を拒む)
git commit -am '...'

# 2. push → M6 でビルド → M6 側のプロセスを ssh で起動 → MBP で計測 → 両機のログを引き戻す  (bench/qfn/cycle.sh、未作成)
bench/qfn/cycle.sh pair_bench.sh <label> 6            # 2 機の記録値 (6 本)
bench/qfn/pair_bench.sh <label> 6 --solo              # 単体の対照 (MBP だけ。M6 を起動しない)

# 3. 引き戻した bench/qfn/results/<date>-pair-<label>/ を結果文書と一緒に commit
```

- `pair_bench.sh` は 2 機の形と単体の形を**同じスクリプト・同じ本文**で取る。違いは M6 を起動するかどうかだけ。
- A/B の腕は環境変数 (`Q38_*`) で切り替え、同じバイナリで取る。`ledger-*` のラベルでは `Q38_` と追加引数を拒む (m6-prefill と同じ)。
- M6 側のビルドは `swift build -c release --product TsugumiCLI` (CLT で通る、TWO_MACHINE_DEV §6)。M6 側のプロセスは CLI の 1 サブコマンドにする。[07](07-EXPERIMENTS.md) の実験は `TsugumiKernelCheck` と `Scripts/qwen38/` の道具で、ランタイムに手を入れずに取る。

## 3. TB4 直結の繋ぎ方 (2026-09-25 の実測、証拠ディレクトリなし)

- 直結は IP over Thunderbolt (`bridge0`) で、**両機とも link-local (169.254.x.x) の自動割り当て**。再接続で変わる。
  MBP 側は `ifconfig bridge0`、M6 側は `ssh m6 'ifconfig bridge0'` (Tailscale 経由) で読む。`~/.ssh/config` の `m6-tb` の `HostName` は古くなる (この日 169.254.24.129 → 169.254.13.135)。
- ssh は `Ciphers aes128-gcm@openssh.com` を指定する (既定の chacha20 は 0.39 GB/s、TWO_MACHINE_DEV §3)。この日 2 GiB を 1.43 s (起動込み)。
- M6 側のプロセスは **`bridge0` の link-local アドレスにだけ bind する** (Tailscale・LAN には出さない。認証も TLS も無いので、サーバーの 127.0.0.1 と同じ扱い)。
- MBP のアプリから M6 側のプロセスを ssh で起動する形は後回し。当面は `cycle.sh` / 手動で起動する。
- 常設にするかどうかは README の判断待ち 2。常設でないなら「M6 が居なければ単体で動く」が既定の形になる (A / B どちらでも、[08](08-CANDIDATES.md))。

## 4. ヘッドレス固有の注意 (m6-prefill/01 §3 と同じ、QFN で効くもの)

- 外付け `SSD256` は再起動のたびに `ssh m6 'diskutil mount disk5s1'`。**PLE BF16 は外付けにしか無い**ので、未マウントなら E1 と A の M6 側は起動できない。
- `iogpu.wired_limit_mb` は再起動で 0 に戻る (m6-prefill/29)。E1 は既定 (0) のまま取る。
- `sudo purge` が使えず、ページキャッシュを冷やせない。run の前後の `vm_stat` を残し、外れ値は `pre.txt` で説明する。
- 他のモデルプロセス (`TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|...`) が動いていれば計測を拒む。**2 機では「MBP に 1 つ、M6 に 1 つ」が正常**で、同じ機械に 2 つは無い。

## 5. Gemma の資産で使い回すもの

| もの | 使い方 |
| --- | --- |
| `bench/m6/prompts/` の自然文 5 本 (2345〜2499 トークン) | QFN の台帳の本文 1 (単発)。中身の 1 行は m6-prefill/43 §3 |
| `bench/hostinfo.sh` | 両機のホスト札 |
| `bench/m6/dual_ssd_probe.sh`、`rec_probe.c` | M6 の外付けの帯域 (Q4 の前提) |
| `Scripts/make-expert-mirror.sh` の考え方 (`evenAssignment`、share 0.32、manifest の SHA-256) | Q4 の写し。QFN は 1 本の GGUF なので、写す範囲と振り分けは QFN 用に書く ([08 §4](08-CANDIDATES.md)、E2 で先に作る) |
| `ExpertMirrorReceipt` の運用 (未マウントなら警告して内蔵だけ、`TF_EXPERT_MIRROR=0` で 1 run だけ外す) | 同じ形を `Q38_` 側に |
| `Q38_ROUND_LOG` (qwen38/42 §3-2) と `Scripts/qwen38/round_breakdown.py` | 段の秒。2 機では M6 側の行も同じ形で出す |
