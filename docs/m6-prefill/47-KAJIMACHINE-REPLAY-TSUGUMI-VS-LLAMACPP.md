# 47. 鑑定の要求を M6 で再生した — Tsugumi (QAT) は 1 回 9.73 s で SSD を 11.7 GB 読み、llama.cpp (IQ3_XXS) は 8.74 s で 0.09 GB。ただし IQ3_XXS は英名が崩れる (実測、2026-11-14)

prefill の台帳 ([06](06-LEDGER.md)) は動かない。M6 を kajimachine (jenga の `~/dev/kajimachine`) の VLM 専用機として使う前提で、
**本線が実際に送る要求だけ**を使い、いまの Tsugumi と「メモリに載り切る量子化 + 普通のエンジン」を並べた記録。

証拠 (M6、この順に取った):
[`kj-tsugumi-qat-realclient`](../../bench/m6/results/2026-11-14-Mac18-5-kj-tsugumi-qat-realclient/rows.jsonl) (jenga の本線から)、
[`kj-tsugumi-qat-preboot`](../../bench/m6/results/2026-11-14-Mac18-5-kj-tsugumi-qat-preboot/summary.tsv) (再生、再起動前)、
[`kj-llamacpp-iq3xxs-wired0`](../../bench/m6/results/2026-11-14-Mac18-5-kj-llamacpp-iq3xxs-wired0/server.log) (失敗)、
[`kj-tsugumi-qat`](../../bench/m6/results/2026-11-14-Mac18-5-kj-tsugumi-qat/summary.tsv) (再生、再起動後)、
[`kj-llamacpp-iq3xxs`](../../bench/m6/results/2026-11-14-Mac18-5-kj-llamacpp-iq3xxs/summary.tsv) (再生、再起動後)。

```
測定: 2026-11-14 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / Xcode 27.0 / APPLE SSD AP0256Z / commit 8642573c
llama.cpp 61dbf49 / gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf 11416548832 bytes / mmproj-BF16.gguf 1194828256 bytes (unsloth)
```

スクリプトは `bench/m6/kj_record.py` (jenga で本線の `Session` を回して要求を記録)、`bench/m6/kj_replay.py` (記録を再生)、
`bench/m6/kj_replay_probe.sh` (サーバーを起こし、再生し、`vm_stat` を 0.5 s おきに取り、証拠に落とす)。コマンド (`command.txt`):

```
tsugumi:   .build/release/TsugumiServer --model scratch/gemma4-qat-sym.gturbo --port 8081 --model-id gemma4-26b-a4b-lattice --reasoning-budget 0 --draft-block-size 4
llama.cpp: llama-server -m gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf --mmproj mmproj-BF16.gguf --port 8082 --host 127.0.0.1 -ngl 99 -fit off -fa on
           -b 1024 -ub 512 -np 1 -cram 0 -c 8192 --reasoning off --image-max-tokens 280 --temp 1.0 --top-k 64 --min-p 0
           (LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"enable_thinking":false}。jenga の llama-swap の gemma4-26b-a4b-lattice-fast と同じ引数から -md / --spec を除いたもの)
```

- **要求**: 本線の `Session.begin()`・`draw()` が送る 4 問 (読み = 写真つき・流し、英名、選択、動作。どれも `response_format` の JSON スキーマつき)。
  `out/plays` を等間隔に 12 枚、計 48 要求。後続の問いは前の答えを messages に入れない作りなので、記録をそのまま再生すれば本線と同じ prompt になる。
- **記録と答えはリポジトリに入れない。**来場者の写真を含むので M6 の `~/diet-probe/` に置いた (`kj_payloads.jsonl` の sha256 は `env.txt`)。証拠に入れたのは秒とカウンタだけ。
- 1 回 = 写真 1 枚ぶんの 4 問。再生は 12 回を 2 周、間を空けない。SSD の読みは `iostat -d -I disk0` の 1 回ごとの差。
- M6 は `kj-llamacpp-iq3xxs-wired0` のあとで再起動した。再起動後の 2 本だけを比べる。Tsugumi は `iogpu.wired_limit_mb` 0 (既定)、llama.cpp は 14336。

## 0. 結論

1. **Tsugumi は鑑定 1 回ごとに SSD から 11.7 GB (再起動前は 13.3 GB) を読む。**本線から叩いても再生しても同じで、重み全体をほぼ毎回読み直している。
   jenga の `docs/m6_split.md` §2-3 の「1 回の前後で 12.7 GB」と同じもの。
2. **載り切る llama.cpp (IQ3_XXS) は 1 回 8.74 s で、SSD を読まない (0.09 GB)。**Tsugumi は 9.73 s。読みの最初の文字が出るまでは 0.89 s 対 2.37 s。
3. **llama.cpp は `iogpu.wired_limit_mb` を上げないと動かない。**既定 (0) では `kIOGPUCommandBufferCallbackErrorOutOfMemory` で 1 件も通らない。
   14336 (sudo、再起動で戻る) で通るが、wired は最大 13.56 GiB、走っている間の空きは 5%、Swapouts が 2 周目でも 31,363 ページ出る。**載ってはいるが余裕は無い。**
4. **IQ3_XXS は英名が崩れる。**同じ写真 1 枚 (`20260905-113707`) の英名が 2 周とも繰り返しに落ちて上限まで書き、JSON として閉じなかった (1 / 48 が 2 周とも)。
   もう 1 枚でも英名が 113 トークンまで伸びた。Tsugumi の QAT は 0 / 48 が 2 周とも。**速さはあるが、このままでは本線に出せない。**

## 1. 数字 (再起動後、各 12 回 × 2 周の中央値、`summary.tsv`)

| | Tsugumi QAT 1 周目 | Tsugumi QAT 2 周目 | llama.cpp IQ3_XXS 1 周目 | llama.cpp IQ3_XXS 2 周目 |
| --- | ---: | ---: | ---: | ---: |
| 1 回の秒 [最小〜最大] | 10.00 [9.04〜14.62] | 9.73 [8.95〜10.75] | 8.77 [8.57〜11.42] | 8.74 [8.46〜11.29] |
| 読み / 英名 / 選択 / 動作 s | 4.59 / 1.18 / 2.36 / 1.79 | 4.53 / 1.19 / 2.33 / 1.70 | 3.67 / 1.72 / 2.13 / 1.20 | 3.65 / 1.77 / 2.19 / 1.21 |
| 読みの最初の文字まで s | 2.39 | 2.37 | 0.90 | 0.89 |
| 内蔵 SSD の読み GB / 回 | 12.20 | 11.71 | 0.14 | 0.09 |
| prefill tok/s (100 トークン以上の問い) | 289 | 305 | 640 | 644 |
| decode tok/s (16 トークン以上の問い) | 46.6 (下書き 4) | 47.0 (下書き 4) | 43.7 | 43.8 |
| JSON として読めない答え | 0 / 48 | 0 / 48 | 1 / 48 | 1 / 48 |
| Swapouts (周の合計、ページ) | 13,464 | 0 | 61,216 | 31,363 |
| wired の最大 GiB | 12.81 | 12.81 | 13.56 | 13.56 |

- 再起動前の Tsugumi (`kj-tsugumi-qat-preboot`、起動から 26 日) は 2 周目で 10.52 s・13.30 GB / 回・Swapouts 9,816。本線から叩いた 12 回 (`realclient`) は
  10.29〜11.90 s・12.55〜15.57 GB / 回 (1 枚目は 14.94 s・26.69 GB)。再生は本線と同じ負荷になっている。
- llama.cpp の英名が遅い (1.77 対 1.19 s) のは答えが長いため (26〜40 トークン、崩れた回は 113〜120。Tsugumi の本線は 22 前後)。
- llama.cpp の 11.2〜11.4 s の回は英名が崩れた 2 枚。
- llama.cpp のサーバーの RSS は 12.67 GiB。Tsugumi は 0.09 GiB (重みはページキャッシュ側)。

## 2. 分かったこと

- **読み直しは「載り切らない」ことの帰結で、載り切るエンジンでは起きない。**Tsugumi の 1 回の秒のうちどれだけが読みに食われているかは、この計器では分けていない (未確認)。
- **M6 の GPU は速い。足を引いているのは置き場である。**同じ 4 問で prefill が 305 対 644 tok/s。[29](29-REFERENCE-RUNTIMES-ON-M6.md) の「llama.cpp 2.39 s」と同じ向きの結果が、本線の要求でも出た。
- **「3bit 級を載せる」は、16 GB では上限いっぱいの運用になる。**wired 上限の変更が起動のたびに要り、ほかに何か動けばスワップが増える。
- **品質は量子化で決まり、エンジンでは決まらない。**崩れたのは英名だけで、読み・選択・動作は読んだ範囲では日本語も内容も自然だった。枚数が 12 なので率は言えない。

## 3. 発端と、やらなかったこと

きっかけは DeepSeek-V4.1-Flash の XQ 量子化 (IndrasMirror / he-be/Xolotl) で、「使うデータで測ってエキスパートの精度を落とし、16 GB に収める」ことが Gemma 4 でもできるかという問いだった。

- **XQ はエキスパートの使用頻度で精度を決めていない。**各エキスパートを各 tier で GPTQ 量子化したときの held-out 上の出力誤差をコストにして、バイト予算の下で配っている
  (`tools/v41/v41_alloc.py`)。頻度の順位は置き場 (VRAM / RAM / NVMe) の決定に使われている。校正データはコーディングとエージェント寄りで、日本語は入っていない。以上は他プロジェクトの文書とコードの読みで、ここでは測っていない。
- **Gemma 4 26B-A4B の形** (`gemma4-qat-sym.gturbo` の manifest): 30 層 × 128 エキスパート、top-8、1 エキスパート 3,358,720 バイト、routed expert の合計 12.9 GB。
- **Tsugumi の側でエキスパート単位にビット幅を混ぜる実験はしていない。**2 / 3 bit の GEMV と `expertStride` の等間隔の前提 ([QUANTIZATION.md](../QUANTIZATION.md) §2) を動かす話で、
  載り切るなら普通のエンジンで足りる、という方向に切り替えたため。
- **「足跡が小さければ Tsugumi の読み直しが消えるか」も測っていない** (後ろの層を前の層へのハードリンクにした模擬モデルを考えたが、作っていない)。
- 雑多な質問 (日本語・英語・コード・翻訳ほか) での計測は、用途と違うので途中で捨てた。証拠は残していない。
- mlx は測っていない。最新の venv (`mlx 0.32.4 / mlx-lm 0.32.2 / mlx-vlm 0.7.12`) を作っただけ。[29](29-REFERENCE-RUNTIMES-ON-M6.md) では 3-bit がテキストだけで peak 13.3 GB・スワップあり。

## 4. 開いたまま残るもの

1. **載り切って、英名が崩れない量子化があるか。**候補は (a) 鑑定の要求を校正データにした imatrix で GGUF を作り直す (GGUF はエキスパートが層ごとに 1 テンソルなので、配れるのは層・射影の単位まで。
   未量子化の重みが要る)、(b) 既製の IQ3_S (10.51 GiB)・Q2_K_XL (9.82 GiB)、(c) mmproj を Q8 にしコンテキストを 4096 に下げて、浮いた分で Q3_K_M (11.85 GiB) を載せる。どれも未測定。
2. **崩れの率。**jenga の `experiments/m6_vlm/m6_compare.py` と `m6_sheet.py` で 46 枚を回し、人が読む。今回の 12 枚では率を言えない。
3. **wired 上限を上げた運用を展示で続けられるか。**再起動のたびに sudo が要る。Swapouts が出たままの長時間運転は見ていない。
4. Tsugumi のまま読み直しを無くす道 (足跡を減らす、wired とページキャッシュの取り合いを直す) は手つかず。[46](46-WIRED-DURING-RUN.md) §3 と同じ場所にある。

## 5. M6 と jenga に残したもの

- M6: `~/diet-probe/` (記録した要求と答え。来場者の写真を含む)、`~/LLM/ref/llama.cpp-latest/` (61dbf49、`llama-server` と `llama-mtmd-cli`)、`~/LLM/ref/mmproj-BF16.gguf`、
  `~/LLM/ref/mlx-venv-latest/`。9 月の `~/LLM/ref/llama.cpp` (4c5957c) と `mlx-venv` は触っていない。
- M6 の `iogpu.wired_limit_mb` は 14336 のまま (次の再起動で 0 に戻る)。**kajimachine 用の TsugumiServer は止まっている。**起こすのは jenga の `experiments/m6_vlm/m6_restart.sh`。
- jenga: `/tmp/kj_record.py`・`/tmp/kj_payloads.jsonl`・`/tmp/kj_record_rows.jsonl`。リポジトリは変えていない。
