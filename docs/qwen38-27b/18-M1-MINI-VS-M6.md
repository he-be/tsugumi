# 18. M1 mini と M6 で ishizuki の Bonsai 2 27B を同じ条件で測る (2026-09-28、実測)

目的は記事「M6 は M1 の何倍か」の数字。対象は Ternary Bonsai 2 27B の MLX 2-bit パックを ishizuki で走らせたときの prefill と decode。
MTP は [16 §14〜§16](16-M6-ISHIZUKI-AGENT-PLAN.md) の改修が終わった版 (`dec0826`) を使う。
**両機の数字は `bench/m6/results/2026-09-28-{Mac18-5,Macmini9-1}-final-*` の証拠ディレクトリを指す。**

言葉: **prefill** = prompt の全トークンを 1 回通して状態を作るまでの時間 (ishizuki の `Generator.promptSeconds`、[15 §7-5](15-M6-PREFILL-PLAN.md) と同じ量)。
**decode** = prompt の後に 1 トークンずつ生成する速さ (tok/s)。**MTP** = パックに足した MTP ヘッドで 1 トークン下書きし、本体が 2 トークンをまとめて確かめる生成 ([16 §14](16-M6-ISHIZUKI-AGENT-PLAN.md))。

## 0. 結論

| | M6 | M1 | 比 |
| --- | ---: | ---: | ---: |
| prefill en-readme (2,427 トークン、中央値 n = 6) | 9.207 s (263.6 tok/s) | 68.006 s (35.7 tok/s) | **7.39 倍** |
| prefill ja-twomachine (2,336 トークン、中央値 n = 6) | 8.882 s (263.0 tok/s) | 65.141 s (35.9 tok/s) | **7.33 倍** |
| decode、MTP なし (中央値 n = 3) | 15.96 tok/s | 6.47 tok/s | **2.47 倍** |
| decode、MTP あり (中央値 n = 3) | 21.73 tok/s | 6.24 tok/s | 3.48 倍 |
| decode、各機の速い方 (M6 は MTP あり、M1 は MTP なし) | 21.73 tok/s | 6.47 tok/s | 3.36 倍 |

- 比は中央値どうし。prefill は M1 の秒 ÷ M6 の秒、decode は M6 の tok/s ÷ M1 の tok/s。
- **M1 では MTP が MTP なしを下回る。**2 トークンの forward が 1 トークンの 1.46 倍 (254 対 174 ms) かかり、M6 の 1.13 倍 (71 対 63 ms) より重い。受理は両機ほぼ同じ (M6 381/520、M1 380/521)。
- 両機とも全 run で Swapouts +0、peak 9.43 GB。

## 1. 条件

| | M6 | M1 mini |
| --- | --- | --- |
| 機械 | Mac18,5 Apple M6 16 GB | Macmini9,1 Apple M1 16 GB |
| OS | macOS 27.0 (26A428) | macOS 27.0 (26A428) |
| SSD | APPLE SSD AP0256Z | APPLE SSD AP0256Q |
| `iogpu.wired_limit_mb` | 0 (既定) | 0 (既定) |
| パック本体の置き場 | 外付け SSD256 (`~/LLM/ref` からのリンク) | 内蔵 |
| 測る前 | MTP の作業のサーバを止めた直後 | 再起動から 3 時間 (この測定の前に同じ条件の試し run のみ) |

- **バイナリ**: `dec0826` (作業ツリーに変更なし) を MBP で `ISHIZUKI_BIN_DIR=ishizuki-bin-final bench/m6/ishizuki_ship.sh m6-lan` で組み、同じ `.build/ship` を mini に rsync。両機で sha256 一致 (`ishizuki-prefill` bd19433f…、`ishizuki-decode-probe` 182beea8…、`default.metallib` 24d4cfcd…)。
- **パック**: M6 の `Ternary-Bonsai-2-27B-mlx-2bit` を MBP 経由のパイプ (`ssh m6-lan tar | ssh mini-lan tar`) で mini へ。約 40 MB/s、3 分 39 秒。`model.safetensors` の sha256 は同梱の `files.json` の値 (130de592…) と一致。`-mtp` ディレクトリ (MTP ヘッド 8-bit / group 32、[16 §14](16-M6-ISHIZUKI-AGENT-PLAN.md)) も同じ絶対パスのリンクのまま運び、`mtp.safetensors`・`config.json` の sha256 が M6 と一致。
- **本文**: prefill は `bench/m6/prompts/en-readme.json`・`ja-twomachine.json` (`ref_ishizuki_bonsai.sh` が同じ形で包む)。decode は `bench/m6/prompts/ja-twomachine.rendered.txt` (2,316 トークン、sha256 d09595b8…、16 §16-4 の probe と同じファイル)。
- **コマンド** (両機同じ、置き場 `~/dev/tsugumi-bench` は bench スクリプトとプロンプトの rsync):

```sh
export ISHIZUKI_BIN=$HOME/LLM/ref/ishizuki-bin-final
sh bench/m6/ref_ishizuki_bonsai.sh final-prefill-en-readme 6; sleep 20
PROMPT=bench/m6/prompts/ja-twomachine.json sh bench/m6/ref_ishizuki_bonsai.sh final-prefill-ja-twomachine 6; sleep 20
sh bench/m6/ref_ishizuki_decode.sh final-decode-ja-twomachine    # reps 10、gen 300
```

`ref_ishizuki_decode.sh` はこの測定で足した。`ishizuki-decode-probe` を 1 プロセス流して証拠ディレクトリに置くだけ。
probe の生成は公式値 (thinking off: 0.7 / 20 / 0.8 / 0.0 / presence 1.5)、300 トークン、seed 1 / 2 / 3、同じプロセス ([16 §15-3](16-M6-ISHIZUKI-AGENT-PLAN.md) と同じ)。

## 2. prefill

| 本文 | 機械 | run 1〜6 (s) | 中央値 | tok/s | 読み込み |
| --- | --- | --- | ---: | ---: | ---: |
| en-readme | M6 | 9.504 / 9.209 / 9.206 / 9.207 / 9.205 / 9.191 | 9.207 | 263.6 | 5.7〜6.6 s |
| en-readme | M1 | 68.395 / 67.656 / 67.911 / 68.021 / 68.078 / 67.990 | 68.006 | 35.7 | 3.3〜6.0 s |
| ja-twomachine | M6 | 8.899 / 8.879 / 8.866 / 8.874 / 8.941 / 8.884 | 8.882 | 263.0 | 5.7〜5.8 s |
| ja-twomachine | M1 | 65.214 / 65.125 / 65.078 / 64.906 / 65.157 / 65.191 | 65.141 | 35.9 | 3.4 s |

- 各 run の全部の値は `summary.tsv`。M6 の en-readme は前日の 459ee00 の 9.205 s ([15 §7-5](15-M6-PREFILL-PLAN.md)) と同じ水準。mtp.patch は 1 クエリと prefill の経路を変えていない (16 §16-3)。
- **読み込みの秒は比べない。**M6 はパック本体を外付けから、M1 は内蔵から読んでいる。

## 3. decode (`ishizuki-decode-probe`)

| | M6 | M1 |
| --- | --- | --- |
| forward 1 トークン (logits 全位置、中央値) | 63.3〜63.6 ms | 174.0〜175.1 ms |
| forward 2 トークン (まとめた形) | 71.0〜71.4 ms | 253.9〜254.5 ms |
| 2 トークン ÷ 1 トークン | 1.13 | 1.46 |
| 生成、MTP なし (tok/s) | 15.70 / 15.96 / 15.96 | 6.37 / 6.47 / 6.47 |
| 生成、MTP (まとめた形、既定) | 20.82 / 21.73 / 22.54、受理 381/520 | 5.94 / 6.24 / 6.46、受理 380/521 |
| 生成、MTP (`fold off`、16 §16 の修正前の形。参考) | 19.56 / 18.88 / 18.73、受理 335/565 | 5.43 / 5.54 / 5.01、受理 323/577 |

- 2 トークンの検証の 2 行目の誤り (16 §16-2) は M1 でも同じに出て (fold off で TV 0.4047)、まとめた形で消える (TV 0.0013)。1 行目は TV 0.0105。
- M6 の値は 16 §16-4 の probe-5 (20.83 / 21.72 / 22.23、受理 381/520) と同じ水準。

## 4. 手順から外れたこと

- **hostinfo の commit 欄が `?+dirty`**: 両機とも git ではない置き場 (`~/dev/tsugumi-bench`) から流したため。バイナリの commit は `build-info.txt` の `dec0826`。各証拠ディレクトリに `NOTE.txt` を置いた。
- **M6 のサーバを止めた**: MTP の作業のセッションが立てた `ishizuki-serve` (`ishizuki-bin-mtp5`) をユーザーの了承で SIGTERM し、測定後に同じ引数 (`ISHIZUKI_SPEC_TRACE=1`、`--slots 2 --ctx 32768`) で立て直した。`/props` の応答まで確かめた。
- **M1 の試し run (証拠に入れない)**: 転送とハッシュの直後は空きが 18 MB で、459ee00 の試し run が Swapouts +64,708 / +11,732 ページ (69.13 / 67.73 s)。再起動後は同じ条件 3 回で Swapouts 0、68.32 / 67.60 / 67.76 s。秒はスワップの有無で変わらなかった。本測定はこの再起動の後。
- M6 と M1 は同じ時間帯に並べて流した (機械は別、ネットワークは使わない)。
