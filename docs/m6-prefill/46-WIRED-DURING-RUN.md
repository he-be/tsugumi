# 46. 要求を処理している間、wired は 10.4〜10.7 GiB まで上がり、5 秒以内に 1.25 GiB 前後へ戻る — 48 §7 の「約 2.5 GiB」は M6 では外れる (実測、2026-11-14)

証拠 (M6): [`wired-server-mmap`](../../bench/m6/results/2026-11-14-Mac18-5-wired-server-mmap/summary.tsv)、
[`wired-server-pread`](../../bench/m6/results/2026-11-14-Mac18-5-wired-server-pread/summary.tsv)、
[`wired-cli-mmap`](../../bench/m6/results/2026-11-14-Mac18-5-wired-cli-mmap/summary.tsv)。
ホスト札 (3 つとも同じ):

```
測定: 2026-11-14 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / Xcode 27.0 / APPLE SSD AP0256Z / commit 8642573c / minos 15.0
```

スクリプトは `bench/m6/wired_probe.sh` と `bench/m6/wired_client.py` (この文書と同じ commit で入れた。測ったときは M6 の作業ツリーに未追跡で置いていたので、札の commit は Sources の版を指す)。コマンド (`command.txt`):

```
server: .build/release/TsugumiServer --model scratch/gemma4-qat-sym.gturbo --port 8081 --model-id gemma4-26b-a4b-lattice --reasoning-budget 0 --draft-block-size 4 & python3 -I bench/m6/wired_client.py <IMAGE> 3
cli:    .build/release/TsugumiCLI --model scratch/gemma4-qat-sym.gturbo --prompt この写真に写っているものを、日本語で 3 文で説明してください。 --max-new 128 --expert-cache-slots 32 --draft-block-size 4 --max-context 16384
```

- server の要求: 写真 1 枚 (`05_m.jpg`、sha256 は `env.txt`) の説明を流しで 128 トークン、続けて同じ会話に短い問い 3 つ (各 32 トークン)。これを 3 往復。展示の読みと同じ形で、prompt cache も効く。
- 腕: `server-mmap` は既定 (エキスパートを `MAP_SHARED` の層ファイルから GPU に直接読ませる)、`server-pread` は `TF_EXPERT_MMAP=0` (私有スロットに `pread`)、`cli-mmap` は CLI の既定。
- 各 run は新しいプロセスで、`vm_stat` を 0.25 s おきに取る (`run-N.wired.tsv`)。内蔵 SSD の読みは `iostat -d -I disk0` の要求前後の差。反復 n = 3。
- `iogpu.wired_limit_mb` は 0 (既定)。**外付けの写し (`SSD256`) はマウントされていない** (`env.txt`) ので、読みはすべて内蔵 SSD から。
- 台帳 ([06](06-LEDGER.md)) は動かない (prefill の秒ではない)。

## 0. 結論

1. **要求を処理している間、wired は server の既定で 10.64 GiB (10.42〜10.69) まで上がる。**起動直後の待機中は 1.25〜1.27 GiB で、要求が終わって 5 s 後には 1.25〜1.28 GiB に戻っている。
2. **サーバー固有ではない。**CLI の既定でも 9.25 GiB (8.98〜9.60) まで上がる。旧経路 (`TF_EXPERT_MMAP=0`) でも 7.16 GiB (7.16〜7.20) まで上がる。
3. **既定の mmap の経路のほうが速く、SSD の読みも少ない。**同じ要求で 13.26 s (12.82〜13.40)・11.78 GB (11.71〜12.66) に対し、旧経路は 15.21 s (15.01〜15.31)・15.95 GB (15.29〜16.55)。9 run すべてで Swapouts 0。

## 1. 数字 (各 n = 3 の中央値 [最小〜最大]、`summary.tsv`)

| 腕 | 待機中の wired GiB | 処理中の wired 最大 GiB | 処理中のファイルのページ 最小 GiB | 5 s 後の wired GiB | 内蔵 SSD の読み GB | 要求の壁時計 s |
| --- | --- | --- | --- | --- | --- | --- |
| server-mmap (既定) | 1.26 [1.25〜1.27] | 10.64 [10.42〜10.69] | 2.77 [2.72〜2.97] | 1.26 [1.25〜1.28] | 11.78 [11.71〜12.66] | 13.26 [12.82〜13.40] |
| server-pread | 1.24 [1.24〜1.25] | 7.16 [7.16〜7.20] | 6.06 [6.01〜6.11] | 1.25 [1.25〜1.30] | 15.95 [15.29〜16.55] | 15.21 [15.01〜15.31] |
| cli-mmap | 1.25 [1.24〜1.25] | 9.25 [8.98〜9.60] | 5.21 [4.81〜5.44] | 1.25 | 13.18 [9.72〜13.56] | 6.22 [5.05〜6.93] |

壁時計は server では 12 要求の合計 (クライアントの開始から終了まで)、cli ではプロセス 1 本 (起動とロードを含む) で、腕の間でしか比べられない。cli と server の秒は並べない。

## 2. 分かったこと / 外れたこと

- **[docs/mtp/48](../mtp/48-D-P1-P4-MMAP-RESIDENCY.md) §7 の「D が wire するのは約 2.5 GiB、既定の wired limit 8 GiB に収まる」は M6 では外れる。**あれは 1 ステップで名指すバッファ (26.9 MB × 30 層) に常駐コアと KV を足した**導出**で、MBP (macOS 15) でも system 全体の wire_count がほとんど動かず、実測では確かめられていなかった (48 §1a)。M6 (macOS 27) の既定では処理中に 10.4〜10.7 GiB まで上がる。それでも `recommendedMaxWorkingSetSize` 12,124 MiB ([03 §1](03-FACTS.md)) の下にある。
- **増えた分は要求の間だけで、待機中には残らない。**どの腕も 5 s 後には起動直後と同じ値に戻る。README の「要る RAM 1.3 GB」は待機中の値としては合っている。
- **既定と旧経路の差 (導出、§1 の中央値)**: 処理中の wired 最大が +3.48 GiB、ファイルのページの最小が −3.29 GiB。旧経路が持つ私有スロット (3.22 GB、48 §2) は既定には無いので、既定は私有スロットの代わりに層ファイルのページを wired として抱えている、と読める。**どのページが wired になっているかは数えていない (未確認)。**
- **wired + ファイルのページは一定ではない。**処理中に 7.48〜14.47 GiB の幅で動く (`summary.tsv` の 2 列)。増えた wired がすべてファイルのページだとは、この計器では言えない。
- 1 往復目は SSD から重みをほぼ丸ごと読み直す。これは [43](43-LEDGER-PROMPT-REPEATED-SENTENCE.md) の「1 run で SSD から 6〜12 GiB 読む」と同じ性質で、新しい不具合ではない。

## 3. 開いたまま残るもの

- wired になっているページの内訳 (層ファイル / 常駐コア / vision / KV)。`vmmap` か IOKit の計器で取る。
- 写し (`SSD256`) を戻した既定での同じ数字。[44](44-EXPERT-MIRROR-EXTERNAL-SSD.md) の既定はマウントが前提。
- MBP (18 GB / macOS 15) で同じスクリプトを回したときの値。両機の数字は並べない ([02 §4](02-EVIDENCE.md))。
