# 15. M6 で Ternary Bonsai 2 27B の prefill を測る — 調査と計画、最初の行 (2026-09-27)

対象は Ternary Bonsai 2 27B ([05](05-BONSAI-PQ2-0.md) の PQ2_0 と、同じ重みの MLX 2-bit パック)。
問いは「M6 (Mac18,5、16 GB、macOS 27、CLT のみ) でこのモデルの prefill が何 tok/s 出るか」。
表記は [qwen38/01](../qwen38/01-Q2-FIRST-LIGHT.md) と同じ (実測 / 導出 / 公称 / 未確認)。
**M6 の実測は [m6-prefill/02](../m6-prefill/02-EVIDENCE.md) の規則に従い、`bench/m6/results/` の証拠ディレクトリを指す。**
指せない数字は札を付ける。他機 (MBP、ishizuki の M1 Max) の数字は M6 の期待値にしない。
[m6-prefill/06](../m6-prefill/06-LEDGER.md) の台帳は Gemma の既定行だけなので、この文書の数字で台帳は動かない。

言葉: **prefill** = prompt の全トークンを 1 回通して KV / GDN 状態を作るまでの時間。**pp** = prompt のトークン数 ÷ prefill 秒。
**経路** = 同じ重みを走らせる実装 (llama.cpp の prism フォーク、パック同梱の MLX ランタイム、ishizuki、Tsugumi)。

## 0. 結論

1. **M6 で今すぐ測れる経路は llama.cpp の prism フォークだけで、最初の行を取った (§7)。**
   自然文 2,408 トークン (`bench/m6/prompts/en-readme.json`) の prompt eval が **13.50 s (13.32〜13.83)、178.4 tok/s** (実測、n = 6、§7-1)。
   `llama-bench` の pp512 / pp2048 は 231.2 / 173.2 tok/s (§7-2)。tensor API を切ると 30.20 s、79.7 tok/s で、PQ2_0 の行列積が M6 の tensor 経路に乗っていることが A/B で分かった (§7-3)。
   同じ本文を MBP (M3 Pro) の同じ経路で流した記録は無い。MBP の 92.6 tok/s ([07 §3](07-PQ2-0-SMALL-BATCH-KERNEL.md)) は `llama-batched-bench -npp 64` の別の形なので並べない。
2. **Tsugumi 自身に Bonsai を走らせる経路は無い。**[05 §10](05-BONSAI-PQ2-0.md) の通りランナーの結線が無く、[14](14-PQ2-0-OWN-GEMV.md) で書いたのは GEMV 1 本だけ。
   アプリは子サーバの `llama-server` を立てる形 ([11](11-GUI-WIRING.md)) で、prefill はその llama.cpp の数字である。したがって「Tsugumi の Bonsai の prefill」は当面 **llama.cpp の prism フォークの prefill** を指す。
3. **ishizuki は M6 では組めない** (§3)。mlx-swift が Metal シェーダをビルド時に `metal` でコンパイルし、CLT にはそれが無い (実測、エラー行を §3 に写した)。
   組む道は 2 つ: MBP の Xcode.app を M6 に置いて `DEVELOPER_DIR` で組む、または MBP で組んで (SDK 26.5、`.macOS(.v26)`) 実行ファイルと metallib を M6 へ運ぶ。
   どちらも ishizuki には CLI が無いので、ベンチを呼ぶ小さな実行ファイルを 1 本書く必要がある (§3-2)。
4. **パック同梱の MLX ランタイム (Python) は M6 の既存 venv で動く見込み**で、重み 8.6 GB は M6 へ取得中 (§4)。動いたらこれが 2 本目の経路になる。
   prefill の秒を出す口が `quickstart.py` に無いので、`artifact.load_model` を呼んで prompt を 1 回通す短いスクリプトが要る (§5-2)。
5. **上限の目安** (導出): 1 トークン 54 GFLOP ([08 §1](08-PP-TG-CEILING.md) の 27B × 2) を M6 の fp16 GEMM 18.61 TFLOP/s ([証拠](../../bench/m6/results/2026-09-22-Mac18-5-gpu_peak_probe/output.txt)、MPS) で割ると 345 tok/s。
   これは演算だけの床で、期待値ではない。MBP では `mul_mm` が GEMM の上限の 8 割で止まった ([08 §1](08-PP-TG-CEILING.md)) が、その比を M6 に持ち込まない ([m6-prefill/02 §4](../m6-prefill/02-EVIDENCE.md))。

## 1. 対象の形 (実測は [05 §1](05-BONSAI-PQ2-0.md)、パックは HF の `config.json`)

| | PQ2_0 GGUF | MLX 2-bit パック |
| --- | --- | --- |
| 置き場 | `prism-ml/Ternary-Bonsai-2-27B-gguf` | `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (2026-09-22 更新) |
| 大きさ | 7,206,168,928 B | `model.safetensors` 8,595 MB |
| 量子化 | 128 重みごとに fp16 scale + 2 bit の三値 ([05 §2](05-BONSAI-PQ2-0.md)) | affine 2-bit / group 128 (`quantization`)。三値を affine の格子に載せた形 (未確認: ビット一致かどうか) |
| Hadamard | GGUF ヘッダの `prism.hadamard.*`、runtime が活性を回す ([05 §3](05-BONSAI-PQ2-0.md)) | `hadamard.json` (prism contract v1)、`model_type = prism_hadamard_qwen35` |
| 読める実装 | prism フォークの llama.cpp (upstream には型 142 が無い。M6 の `~/LLM/ref/llama.cpp` 4c5957c で `grep pq2_0` は空、実測) | 同梱 `runtime/artifact.py` (mlx 0.32.0 / mlx-lm 0.31.3 固定)、ishizuki (`PrismHadamardConfiguration`) |
| MTP | 無し ([05 §0-1](05-BONSAI-PQ2-0.md))。有志の MTP 付き GGUF は [06](06-BONSAI-MTP-ON-MAC.md) | 無し (`components.mtp = false`) |
| Vision | mmproj 別ファイル | `components.vision = true` (mlx-vlm 0.6.3 が要る。text だけなら不要) |

構造は Qwen3.8-27B: 64 層 (GDN 48 + 全注意 16、`full_attention_interval` 4)、hidden 5,120、MLP 17,408、語彙 248,320。
prefill の演算は 1 トークン約 54 GFLOP (導出、[08 §1](08-PP-TG-CEILING.md))。ビット幅で演算量は変わらない。

## 2. これまでの実測 (すべて MBP、M3 Pro 18 GB、macOS 15。M6 の期待値にはしない)

| 経路 | 形 | pp | 出所 |
| --- | --- | ---: | --- |
| llama.cpp prism フォーク (b10709) | `llama-batched-bench -c 8192 -npp 64 -ntg 64`、B = 1 | 92.6 tok/s | [07 §3](07-PQ2-0-SMALL-BATCH-KERNEL.md)、[08 §2-2](08-PP-TG-CEILING.md) |
| 同上、アプリの 1 ターン (子サーバ、14 ターン × 2 セット) | 読んだトークン ÷ prefill 秒 | 86.6 / 87.7 tok/s | [12 §1](12-NATIVE-PORT-GAIN.md) |
| 同上、`mul_mm` の天井 | `test-backend-ops perf`、m 4096 / n 512 / k 14336 | 5.15 TFLOPS (型によらず 5.1〜5.2) | [08 §1](08-PP-TG-CEILING.md) |
| Tsugumi の自前ランナー (Bonsai ではなく ISTA IQ3_S、同じ 27B 構造) | 4K の最初の 512 トークン、KV q8_0、1 回 | 54.0 tok/s (GPU 時間だけなら 73) | [04 §3-1](04-RUNNER-REFERENCE-MATCH.md) |
| PTQ1_0 (同じフォーク) | 上と同じ batched-bench | 79.3 tok/s (PQ2_0 より遅い) | [08 §2-2](08-PP-TG-CEILING.md) |

MBP の結論は「prefill は `mul_mm` の行列積そのもので律速し、逆量子化を速くしても動かない」([08 §0-1](08-PP-TG-CEILING.md))。
M6 は GPU に Neural Accelerator (tensor API) があり、llama.cpp は Gemma でそれを使って 4.49 → 2.39 s になった ([m6-prefill/29](../m6-prefill/29-REFERENCE-RUNTIMES-ON-M6.md))。
Bonsai の `mul_mm` (PQ2_0) が tensor 経路に乗るかは、同じ A/B で確かめる (§7-2)。

## 3. ishizuki (struffl/ishizuki、459ee00、2026-09-27 に読んだ)

MLX Swift の上に書かれた macOS 26+ のエンジン。もともと Ternary Bonsai 2 27B 向けで、MLX パック (Hadamard 付き) を `config.json` / `hadamard.json` から読む。
GGUF も 11 型読むが **PQ2_0 は含まない** (`Sources/IshizukiKit/Core/GGMLDequant.swift` の `supported` に無い、`grep -ri pq2` が空)。
Bonsai を走らせるなら MLX パックの方である。

公開値 (公称、**M1 Max 64 GB**、`BENCHMARKS.md`。機体が違うので M6 の見込みには使わない):
Bonsai 2-bit の pp512 / pp2048 が ishizuki 139.3 / 132.0、mlx-lm (パック同梱ランタイム) 111.0 / 119.6 tok/s。
512 トークンで 138.9、3,622 で 95.4、7,218 で 69.9 tok/s。decode は 22.5、DFlash 2 で 45 (code / math)。

### 3-1. prefill の作り (読んだだけ)

- prefill のチャンクは既定 512 (`Generator.prefillChunkSize`、`Politeness.prefillChunk`)。行列積は MLX の affine 量子化 matmul。
  「Kernels written for the verify」の自前カーネルは投機の検証 (5〜8 列) 用で、prefill の大きい M には効かない。
- **Neural Engine への切り出し** (`ANEOffload`、`PrefillBench.Options.ane`): MLP の gate / up を ANE 用に切ったスライスをパックに同梱し、GPU と分担する形。
  パックに `ane` のスライスが無ければ拒否する (`Refusal.noSlices`)。HF のパックにスライスは無い (`files.json` に無い、未確認: 作る手順は `Scripts/ane-export.py`)。
- ベンチの口は 3 つ: `ContextBench` (1K〜32K の prefill / decode / KV、**同じ 1 文の繰り返し**で本文を作る)、`PrefillBench` (射影ごとの秒、512 / 2048 チャンク × 6 回)、
  テスト `EngineBenchProbe` (`ISHIZUKI_BENCH_PACK=<pack> swift test`、pp512 / pp2048 を **同じ段落の繰り返し**で 2 回取って良い方)。
  どれも [m6-prefill/43](../m6-prefill/43-LEDGER-PROMPT-REPEATED-SENTENCE.md) の「繰り返し文」で、自然文ではない。
- **CLI が無い。**`Package.swift` の product はライブラリ 3 つだけで、ベンチはメニューバーアプリの Tools タブ (`App/Ishizuki/BenchController.swift`) か `swift test` から呼ぶ。

### 3-2. M6 で組めない (実測、2026-09-27)

`~/LLM/ref/ishizuki` に clone し `swift build -c release --target IshizukiKit` を流した (M6、Swift 6.4、CLT)。依存の取得は通り、mlx-swift の Metal のコンパイルで止まった
(ログ `~/LLM/ref/ishizuki-build.log`、M6 のみ):

```
error: unable to spawn process 'metal' (No such file or directory)
error: CompileMetalFile .../mlx-swift/Source/Cmlx/mlx-generated/metal/conv.metal failed with a nonzero exit code
```

`xcrun -find metal` も M6 では見つからない。[TWO_MACHINE_DEV §6](../TWO_MACHINE_DEV.md) の「CLT に `metal` が無い」がそのまま当たった。
Tsugumi は実行時にシェーダをコンパイルするので困らないが、mlx-swift はビルド時にコンパイルする。

組む道 (どちらも未実施):

| 道 | やること | 代償 |
| --- | --- | --- |
| A. M6 に Xcode | MBP の `/Applications/Xcode.app` (Xcode 26.6、SDK 26.5) を外付け `SSD256` へ写し、`DEVELOPER_DIR=/Volumes/SSD256/Xcode.app swift build`。`xcode-select` は CLT のまま | 転送と初回起動の同意。SDK が CLT の 27.0 より古いが、ishizuki の要件は macOS 26 なので足りる (未確認: 実際に通るか) |
| B. MBP で組んで運ぶ | MBP (Xcode 26.6) で `.macOS(.v26)` のまま組み、実行ファイルと `mlx-swift_Cmlx.bundle` (metallib) を M6 へ scp | MBP は macOS 15 なので手元で動かせず、失敗の切り分けが M6 でしかできない |

どちらでも **ベンチを呼ぶ実行ファイルが要る** (`ISHIZUKI_BENCH_PACK` のテストは `swift test` のランナーが Xcode 由来なので A の道でだけ使える)。
`EngineBenchProbe.swift` の pp の部分 (`Generator(model:)` → `generate(promptTokens:options:.greedy,maxTokens:1)` → `stats.promptSeconds`) を写して、
自然文の messages json を読む 60 行ほどの `main.swift` を、ishizuki を依存に持つ別パッケージとして置く。**書くのは判断待ち** (§8)。

## 4. M6 に置いたもの (2026-09-27、実測)

| もの | 場所 (M6) | 状態 |
| --- | --- | --- |
| prism フォークの配布バイナリ b10743 (adfffbe、2026-09-25、`macos-arm64`) | `~/LLM/ref/prism-b10743/` | `llama-completion --version` が `build 10743 ... AppleClang 21.0.0.21000101 for Darwin arm64` を返す。`llama-bench` / `llama-batched-bench` あり。MBP の b10709 ([07](07-PQ2-0-SMALL-BATCH-KERNEL.md) の修正入り) より新しい配布物で、[Scripts/qwen38_27b/prism-b10709-mac.patch](../../Scripts/qwen38_27b/prism-b10709-mac.patch) は当たっていない (prefill には関係しない。07 の修正は 2〜4 列の mat-vec) |
| PQ2_0 GGUF | `~/LLM/ref/Ternary-Bonsai-2-27B-PQ2_0.gguf` | MBP から `m6-lan` 経由で rsync、7,206,168,928 B (HF と同じ大きさ) |
| MLX 2-bit パックの付属物 | `~/LLM/ref/Ternary-Bonsai-2-27B-mlx-2bit/` | `config.json` / `hadamard.json` / `tokenizer.json` / `runtime/*.py` は取得済み。`model.safetensors` 8.6 GB は `curl -C -` で取得中 (`~/LLM/ref/download-bonsai-mlx.log`) |
| ishizuki の checkout | `~/LLM/ref/ishizuki` (459ee00) | 組めない (§3-2)。`.build` は残してある |
| 既存の venv | `~/LLM/ref/mlx-venv` | Python 3.12.14、mlx 0.32.2、mlx_lm 0.31.3、transformers 5.17.0、tokenizers 0.23.2、numpy 2.5.3。パックの固定は mlx 0.32.0 / transformers 5.5.0 で **版が違う** (未確認: 動くか)。mlx-vlm は無い (text だけなら不要) |
| `iogpu.wired_limit_mb` | — | 14336 (前回の sudo が効いたまま。再起動で 0 に戻る) |
| 内蔵の空き | — | 38 GB (GGUF と取得中のパックを置いた後) |

「重みを M6 に常駐させない」([TWO_MACHINE_DEV §5](../TWO_MACHINE_DEV.md)) に従い、測り終えたら GGUF とパックは消す。

## 5. 経路と、それぞれの測り方

| 経路 | 状態 | 測る道具 | 何が出るか |
| --- | --- | --- | --- |
| R1. llama.cpp prism フォーク (PQ2_0) | **測れる** | [`bench/m6/ref_llamacpp_bonsai.sh`](../../bench/m6/ref_llamacpp_bonsai.sh) (自然文、新プロセス × n、`prompt eval time`)、[`bench/m6/ref_llamabench_bonsai.sh`](../../bench/m6/ref_llamabench_bonsai.sh) (`llama-bench -p 512,2048 -n 0`、外の表と同じ形) | prefill 秒と pp。tensor API の A/B は `GGML_METAL_TENSOR_DISABLE=1` |
| R2. パック同梱の MLX ランタイム (Python) | 重み取得中 | 未作成: `artifact.load_model` で prompt を 1 回通し、`mx.eval` までの秒を出す 30 行 (§5-2) | 同じ重みの MLX 側の pp。ishizuki の表の「mlx-lm」列に当たる |
| R3. ishizuki | 組めない (§3-2) | 未作成: 道 A か B + driver | MLX Swift の pp |
| R4. Tsugumi | 経路が無い ([05 §10](05-BONSAI-PQ2-0.md)) | — | — |

### 5-1. R1 の固定形

- 本文は `bench/m6/prompts/*.json` の自然文 (Bonsai の tokenizer で 2,083〜2,430 トークン、下表)。`bench/l.json` は同じ 1 文の繰り返しなので使わない ([m6-prefill/43](../m6-prefill/43-LEDGER-PROMPT-REPEATED-SENTENCE.md))。
  Qwen の chat turn で包み、thinking を切った空の think ブロックを付ける (パックの `chat_template.jinja` が `enable_thinking=false` で出す形)。
- `-c 4096 -b 2048 -ub 2048 -fa on --temp 0 -n 1 -fit off -lv 4`。`-fit off` は Gemma のときと同じ理由 ([m6-prefill/29](../m6-prefill/29-REFERENCE-RUNTIMES-ON-M6.md))。
- n ≥ 3。1 本目はページキャッシュが冷たいので load が伸びる。中央値とレンジを書き、外れ値は捨てない。
- 出るのは llama.cpp 自身が出す `prompt eval time` (暖まったプロセス、検証なし) なので、Tsugumi の台帳 (c) とは量が違う。

| 本文 (M6 の tokenizer.json で数えた、実測) | トークン | 文字 |
| --- | ---: | ---: |
| `code-runner.json` | 2,430 | 8,545 |
| `en-readme.json` | 2,396 | 5,158 |
| `en-server.json` | 2,400 | 8,590 |
| `ja-m6readme.json` | 2,335 | 4,070 |
| `ja-twomachine.json` | 2,305 | 4,561 |
| `wu.json` | 2,083 | 3,620 |
| `l.json` (繰り返し文、比較用) | 2,366 | 5,928 |

chat turn で包むと 2,408 (en-readme) のように十数トークン増える。

### 5-2. R2 の作り (未作成)

```python
import sys, time, json, mlx.core as mx
sys.path.insert(0, PACK + "/runtime"); from artifact import load_model
from tokenizers import Tokenizer
model, _ = load_model(PACK)                      # 重みの読み込み (秒は別に取る)
ids = Tokenizer.from_file(PACK + "/tokenizer.json").encode(render(prompt), add_special_tokens=False).ids
cache = model.make_cache()
t0 = time.perf_counter()
logits = model.lm_head(model.model(mx.array([ids]), cache=cache)[:, -1:, :]); mx.eval(logits)
print(len(ids), time.perf_counter() - t0)         # prefill 秒 (チャンク分割なし、1 回で全部)
```

`quickstart.py` は prompt を 1 回で通す (チャンク無し) ので、この形がパックの素の prefill。`peak memory` は `mx.get_peak_memory()` で添える。
16 GB で 8.6 GB の重み + 2.4K の活性は入る見込みだが、Gemma の 3-bit MLX (13.3 GB) はスワップした ([m6-prefill/29 §0-3](../m6-prefill/29-REFERENCE-RUNTIMES-ON-M6.md)) ので Swapouts を添える。

## 6. 測る順

1. **R1 の最初の行** (済み、§7): en-readme × 6、tensor off × 3、llama-bench pp512 / pp2048。
2. R1 の残りの本文 5 本 × 3 以上。`PROMPT=` で替える。
3. R2: パックが届いたら §5-2 のスクリプトで同じ 6 本。動かなければ venv を別に切って `runtime/requirements.txt` の版で試す (mlx 0.32.0)。
4. R3 は判断待ち (§8)。
5. 1〜3 が揃ったら、同じ本文の秒を経路ごとに並べた表を 1 つ書く (M6 だけ、同じ commit、同じ本文)。

## 7. 最初の行 (実測、M6、2026-09-27、R1)

証拠: [`bonsai-llamacpp-en-readme`](../../bench/m6/results/2026-09-27-Mac18-5-bonsai-llamacpp-en-readme/) (n = 6)、
[`bonsai-llamacpp-en-readme-tensor-off`](../../bench/m6/results/2026-09-27-Mac18-5-bonsai-llamacpp-en-readme-tensor-off/) (`GGML_METAL_TENSOR_DISABLE=1`、n = 3)、
[`bonsai-llamabench`](../../bench/m6/results/2026-09-27-Mac18-5-bonsai-llamabench/) (`llama-bench -r 5`、1 プロセス)。この順に続けて取った。

```
測定: 2026-09-27 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 5027e5d
version: 0.2.0-dev (build 10743, commit adfffbe41) / Ternary-Bonsai-2-27B-PQ2_0.gguf 7206168928 bytes
```

```
llama-completion -m Ternary-Bonsai-2-27B-PQ2_0.gguf -f prompt.txt -n 1 -c 4096 -b 2048 -ub 2048 -fa on --temp 0 -no-cnv --no-display-prompt -fit off -lv 4
llama-bench -m Ternary-Bonsai-2-27B-PQ2_0.gguf -p 512,2048 -n 0 -b 2048 -ub 2048 -fa 1 -r 5 -o md
```

`iogpu.wired_limit_mb=14336`。Swapouts は 9 run すべて +0 ページ (`run-N.pre.txt` / `post.txt` の差)。

### 7-1. 自然文 `en-readme` (chat turn 込みで 2,408 トークン)

| | prompt eval 中央値 (レンジ) | pp 中央値 (レンジ) | load (1 本目 / 2 本目以降) |
| --- | ---: | ---: | ---: |
| tensor API あり (既定、n = 6) | **13.50 s** (13.32〜13.83) | **178.4 tok/s** (174.1〜180.8) | 2.81 s / 0.42〜0.43 s |
| tensor API なし (`GGML_METAL_TENSOR_DISABLE=1`、n = 3) | **30.20 s** (30.19〜30.29) | **79.7 tok/s** (79.5〜79.8) | 0.43〜0.44 s |

run 順 (tensor あり): 13.666 / 13.420 / 13.581 / 13.835 / 13.379 / 13.322 s。1 本目はページキャッシュが冷たい分 load が 2.4 s 長いが、prompt eval は他と同じ幅にある。

### 7-2. `llama-bench` (合成トークン、1 プロセスで 5 回、平均 ± 標準偏差)

| test | tok/s |
| --- | ---: |
| pp512 | 231.21 ± 2.18 |
| pp2048 | 173.19 ± 1.84 |

`llama-bench` が出したモデルの行: `qwen35 27B PQ2_0 - 2.13 bpw (group 128) | 6.70 GiB | 26.90 B | MTL,BLAS`。

### 7-3. 分かったこと

1. **PQ2_0 の `mul_mm` は M6 の tensor 経路に乗っている。**同じバイナリ・同じ本文で tensor API を切ると 13.50 → 30.20 s (差 16.7 s、導出)。
   Gemma の IQ3_XXS で見た向き ([m6-prefill/29](../m6-prefill/29-REFERENCE-RUNTIMES-ON-M6.md)) と同じ。
2. 2,408 トークンの自然文の pp (178.4) と `llama-bench` の pp2048 (173.2) は近い。pp512 は 231.2 で、短いほど速い (この経路の秒に prompt の長さに比例しない分がある。内訳は未測定)。
3. §0-5 の演算だけの床 345 tok/s (導出) に対し、実測は 178 tok/s。差がどこにあるか (行列積の効率、GDN のスキャン、注意、Hadamard の変換) はまだ測っていない。
   MBP のように `test-backend-ops perf` で PQ2_0 の `mul_mm` 単体を取ると切り分けられる (配布物に `test-backend-ops` が無いので、フォークを M6 で組む必要がある。未実施)。

## 8. 判断待ち / 未確認

- ishizuki を組む道 (A: Xcode を M6 に置く、B: MBP で組んで運ぶ) とベンチ用の driver を書くか。書くならどこに置くか (この repo の `bench/m6/` か、外)。
- R2 が版違いの venv で動くか (未確認)。動かないときに venv をもう 1 つ作るか。
- MLX パックの 2-bit affine と PQ2_0 の三値が同じ重みか (未確認。`reload-validation.json` は直列化の検査で、ランタイム間の一致は検査していないと `PACK-RUNTIME.md` にある)。
  prefill の秒を比べるだけなら影響しないが、出力を比べるなら要る。
- 実測 178 tok/s と演算だけの床 345 tok/s (導出) の差の内訳。`test-backend-ops perf` を M6 で回すにはフォークを組む (cmake は `/opt/homebrew/bin` にある。未実施)。
