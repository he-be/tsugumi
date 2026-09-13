# 05. routed expert の読み待ち — F_RDADVISE をやめて pread、8K で 176 → 131 s (チャンク 4096 で 119 s)

実測: 2026-09-13、M3 Pro 18 GB、macOS 15。対象は [04](04-QSA-GPU.md) と同じ GGUF・ランナー・プロンプト。
表記は 01 と同じ (実測 / 導出 / 未確認)。**速度の行はどれも n=1 で、数字だけを書く。**

## 0. 結論

1. **読み待ちの正体は「mmap のページフォルトで読むと遅い」だった。**同じ expert の範囲を冷えた状態から読むと、
   `pread` は 1 スレッドでも **6.2 GB/s**、mmap を触って読むと 1 スレッド 0.73・8 スレッド 1.48 GB/s、`F_RDADVISE` の後に触ると 0.97 GB/s (§1)。
   `pread` で読んだページは mmap 側でもページキャッシュに載っている (mincore で確認)。
2. **T ≥ 32 のバッチは、選んだ expert を GPU の前に 4 スレッドの `pread` で読む** (`GGUFFile.preadRanges`、`Q38_PREAD_MIN_T`)。decode は advise のまま。
3. **速度** (8,192 トークン、§2):

   | | チャンク 2048 | チャンク 4096 |
   | --- | ---: | ---: |
   | 04 (advise) | 176.1 s (46.5 tok/s) | 140.1 s (58.5 tok/s) |
   | 05 (pread) | **130.8 s (62.6 tok/s)** | **119.2 s (68.7 tok/s)** |
   | peak footprint | 3.37 GB | 5.03 GB |

   routed の wall は GPU 時間とほぼ同じになった (チャンク 4096 の 2 つ目で 28.4 / 27.6 s、advise では 45.9 / 27.6 s)。代わりに読みが 1 チャンク 5.0〜5.6 s。
4. 残る 1 チャンク (4096) 58 s の内訳: **routed の GPU 27.6 s**、pre-router の GPU 16.4 s (GDN 10.2 s・注意 3.3 s)、読み 5.6 s、PLE 3.5 s、top-10 1.4 s (§3)。

## 1. 読み方ごとの速さ (`Scripts/qwen38/readbench.c`)

GGUF の expert テンソル (1 層 = gate 216 MB + up 216 MB + down 324 MB) を 2 層ずつ、毎回まだ読んでいない層で。読む前の resident は 0.03 GB。

| 方法 | スレッド | ブロック | 1.53 GB の時間 | 冷えた分の速さ |
| --- | ---: | ---: | ---: | ---: |
| pread | 1 | 16 MB | 0.24 s | 6.22 GB/s |
| pread | 4 | 16 MB | 0.23 s | 6.50 GB/s |
| pread | 8 | 4 MB | 0.23 s | 6.58 GB/s |
| pread | 16 | 1 MB | 0.23 s | 6.55 GB/s |
| pread | 32 | 1 MB | 0.23 s | 6.60 GB/s |
| mmap を 16K ページごとに 1 バイト触る | 1 | — | 2.04 s | 0.73 GB/s |
| 同 | 8 | — | 1.01 s | 1.48 GB/s |
| F_RDADVISE (0.21 s) の後に触る | 1 | — | 1.54 s | 0.97 GB/s |

pread の直後に同じ範囲を触ると resident 1.51 GB・0.09 s。スレッドを増やしても pread は 6.2 → 6.6 GB/s で頭打ち (SSD 側と見ている、未確認)。

ランナーの 1 チャンク (4096) は 91 % の expert を選ぶので約 33 GB を読む (導出)。advise では読み待ち 18.3 s + advise 4.3 s だった。

## 2. ランナーでの結果 (`Q38_SPLIT_PRE=1 --qwen38-prefill-bench`、8,192 トークン)

チャンク 4096:

| | 合計 | pre wall / GPU | route (うち読み) | routed wall / GPU | PLE |
| --- | ---: | ---: | ---: | ---: | ---: |
| advise、0..<4096 | 65.5 s | 19.0 / 14.7 s | 5.6 s (advise 4.0) | 33.8 / 27.6 s | 5.1 s |
| advise、4096..<8192 | 74.6 s | 19.5 / 16.3 s | 5.7 s (advise 4.3) | 45.9 / 27.6 s | 3.5 s |
| pread、0..<4096 | 61.1 s | 18.9 / 14.7 s | 6.6 s (読み 5.0) | 28.2 / 27.6 s | 5.4 s |
| pread、4096..<8192 | 58.1 s | 19.2 / 16.4 s | 7.0 s (読み 5.6) | 28.4 / 27.6 s | 3.5 s |

チャンク 2048 (pread): 34.9 / 31.7 / 32.0 / 32.3 s、routed wall 14.3〜14.5 s / GPU 13.7〜13.8 s、読み 4.7〜5.3 s。
1 トークンあたりの読みはチャンク 2048 で 2.5 ms、4096 で 1.4 ms (導出)。Swapouts 増分はどれも 0。

decode (8,192 の直後 1 トークン、advise のまま) 0.44 s。decode にも pread を使うと route が 56 → 246 ms、全体 0.60 → 0.71 s だった (各 n=1)。
resident の範囲を mincore で飛ばしても 268 ms で変わらなかった (decode の expert はそもそも冷えている)。

### 2-1. やめたもの: 層の expert を丸ごと裏で先読み

pre-router の GPU と並行して、その層の expert テンソル 3 本 (750 MB) を別スレッドで pread する形 (チャンク 4096、T ≥ 1024 で)。
**走行中に Swapouts が +10,536 ページ (約 165 MB) 増え、見張りが止めた。**原因は特定していない。得られるのは最大でも読み 5.6 s ぶんなので、再試行はしていない。

## 3. 次

チャンク 4096 の 1 チャンク 58 s の順:

1. **routed expert の GPU** (27.6 s、47 %)。組ごとに IQ2_XXS / Q2_K の行を展開し直している。expert ごとにトークンを束ねて 1 回だけ float32 に展開して sgemm にする形を、1 層の A/B で見積もってから (03 §6-3 の 3 番)。
2. GDN (10.2 s: step 6.3・入力 GEMM 2.6・出力 1.3)。
3. 読み (5.6 s)、PLE (3.5〜5.4 s)、top-10 (1.4 s)。
4. チャンク 4096 の footprint 5.03 GB の中身 (1 チャンク目の `attentionSgemm` のスコア 805 MB と、`ensureAttnScratch` が `maxBatch × 24 × 2051` で確保する 1.58 GB など)。32K で使うチャンクを決める前に。

## 4. 足したもの

| 種類 | パス |
| --- | --- |
| モデル I/O | `GGUFFile.preadRanges(_:threads:blockBytes:)` |
| ランナー | `preadMinTokens` (`Q38_PREAD_MIN_T`、既定 32、0 で従来の advise)、`readThreads` (`Q38_READ_THREADS`、既定 4) |
| スクリプト | `Scripts/qwen38/readbench.c`、`expert_ranges.py`、`guarded.sh` (Swapouts +64 MB でコマンドを止める) |

再現:

```bash
PYTHONPATH=~/LLM/llama.cpp/gguf-py ~/LLM/venv/bin/python Scripts/qwen38/expert_ranges.py $G > scratch/qwen38/expert-ranges.txt
clang -O2 -o scratch/qwen38/readbench Scripts/qwen38/readbench.c
scratch/qwen38/readbench $G scratch/qwen38/expert-ranges.txt 0 2 pread 4 16    # 層 0..1、まだ読んでいない層で
Scripts/qwen38/guarded.sh out.txt env Q38_SPLIT_PRE=1 .build/release/TsugumiKernelCheck --qwen38-prefill-bench scratch/qwen38/prompt-code.tokens --q38-tokens 8192 --q38-chunk 4096
```

## 5. 落とし穴

- **ページキャッシュに入れるだけなら mmap を触るより pread。**macOS の mmap のフォルトは 1 GB/s に届かない。`F_RDADVISE` も読みを速くはしていなかった (04 までの advise は「フォルトを GPU の中で 1 つずつ待たない」効果だけ)。
- **ページキャッシュへの読みでも Swapouts は動く。**大きな先読みを GPU と並行させた走行で +165 MB。長い走行は `guarded.sh` で回す。
- readbench の測定は「まだ読んでいない層」を毎回使う。同じ層を 2 回目に読むとキャッシュの速さになる。

## 6. 再開手順 (新しいセッションで続けるとき)

**最新は [06 §8](06-ROUTED-GEMM.md)。**以下は 05 時点のもの。routed expert の GPU (6-3 の 1 番) は 06 で済んだ。
**6-2 の `guarded.sh … /usr/bin/time -l env …` は、直す前の guarded.sh ではベンチ本体を殺せなかった** (06 §4)。

03 §6 (状態・検査・落とし穴) の後継。03 §6 の落とし穴はそのまま有効なので、先に一度読む。

### 6-1. 状態

- コードはこの文書と同じコミットまで入っている。ランナーは `Sources/Tsugumi/Runtime/Qwen38/Qwen38Runner.swift` 1 本。
  - `forward(tokens:startPos:allLogits:)` が T ≥ 1 (decode は `step` = T=1)。
  - 注意: T ≥ 32 で選択なしは `attentionSgemm`、選択ありは `attentionSelected` (04)、T < 32 は `attentionHost`。
  - expert の読み: T ≥ 32 は `GGUFFile.preadRanges` (4 スレッド)、T < 32 は `F_RDADVISE` (05)。
- 速度の現在地 (8,192 トークン、n=1): チャンク 2048 で 130.8 s (62.6 tok/s、footprint 3.37 GB)、チャンク 4096 で 119.2 s (68.7 tok/s、5.03 GB)。decode は短文脈 0.09〜0.11 s/トークン、8K の直後 0.44 s。
- **32K はまだ流していない。**導出ではチャンク 4096 で約 8 分 (注意が n とともに伸びる分は入っていない)。
- ウェイト・トークナイザの在処は [01 §7-1](01-Q2-FIRST-LIGHT.md)。`G=~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf`。
- `scratch/qwen38/` (git 管理外): 03 §6-1 のもの + `expert-ranges.txt` (§4 の `expert_ranges.py` で作り直せる)、`guarded.sh` (`Scripts/qwen38/guarded.sh` と同じ)。
- 運用点は thinking 無効・32K・エージェント主体・英語主体 (メモリ `qwen38-operating-point`)。

### 6-2. 検査を一通り (どれも数分以内、GPU は 1 本ずつ、間に 20 秒)

```bash
swift build -c release --product TsugumiKernelCheck
B=.build/release/TsugumiKernelCheck
$B --q38-select-check
$B --q2-expert scratch/qwen38/expert-fixture-l24
$B --ggml-dense $G
$B --qwen38-decode scratch/qwen38/ref-france-dump.log --q38-ref-logits scratch/qwen38/ref-france.logits
$B --qwen38-decode scratch/qwen38/ref-fuji-idx16.log --q38-ref-logits scratch/qwen38/ref-fuji-idx16.logits --q38-indexer-top-k 16
$B --qwen38-prefill scratch/qwen38/ref-france-dump.log --q38-ref-logits scratch/qwen38/ref-france.logits --q38-chunk 4
$B --qwen38-prefill scratch/qwen38/ref-fuji-idx16.log --q38-ref-logits scratch/qwen38/ref-fuji-idx16.logits --q38-indexer-top-k 16 --q38-chunk 16
Q38_MPS_MIN_T=16 Q38_ATTN_MPS_MIN_T=16 $B --qwen38-prefill scratch/qwen38/ref-fuji-idx16.log --q38-ref-logits scratch/qwen38/ref-fuji-idx16.logits --q38-indexer-top-k 16 --q38-chunk 16
$B --qwen38-prefill scratch/qwen38/ref-fuji-idx16.log --q38-ref-logits scratch/qwen38/ref-fuji-idx16.logits --q38-indexer-top-k 16 --q38-chunk 53
Q38_ATTN_MPS_MIN_T=0 $B --qwen38-prefill scratch/qwen38/ref-fuji-idx16.log --q38-ref-logits scratch/qwen38/ref-fuji-idx16.logits --q38-indexer-top-k 16 --q38-chunk 53
$B --qwen38-prefill scratch/qwen38/ref-fuji-ple.log --q38-chunk 53
```

期待値: 全部 PASS。logits の最悪は 8.5e-7 (France)、6.95e-7 (Fuji チャンク 16)、1.78e-6 (同 MIN_T 16)、2.26e-6 (チャンク 53)、2.48e-6 (従来経路)。

速度 (約 2 分、`guarded.sh` 越しに):

```bash
Scripts/qwen38/guarded.sh out.txt /usr/bin/time -l env Q38_SPLIT_PRE=1 $B --qwen38-prefill-bench scratch/qwen38/prompt-code.tokens --q38-tokens 8192 --q38-chunk 4096
grep -E "^\s+\[|GPU ms|prefill|decode|footprint|GUARD" out.txt
```

§2 の表 (pread の 2 行) と比べる。長文脈で注意を変えたら `Q38_ATTN_SEL_COMPARE=1` (04 §2-2) で同じ入力の `ao` を比べる。端から端の logits 比較は長文脈の検査にならない。

### 6-3. 次にやること

チャンク 4096 の 1 チャンク 58 s (routed GPU 27.6・pre GPU 16.4・読み 5.6・PLE 3.5・top-10 1.4 s) に対して:

1. **routed expert の GPU** (27.6 s / 4096 トークン)。場所は `moeRouted` (host の top-10・スロット割り当て・読み・encode) と
   `Sources/Tsugumi/Metal/MoE/moe_ggml.metal` の `moe_iq2xxs_phase1_gate_up_act` (IQ2_XXS の gate/up + SiLU) と `moe_q2k_phase2_down_reduce` (Q2_K の down + 重み付き和)。
   どちらも (トークン, k) の組ごとに量子化行を読み直している。
   - 案: expert ごとに組を集め (1 層 40,960 組を約 470 expert に)、その expert の gate/up/down を 1 回だけ float32 に展開して sgemm、結果を組に散らす。
   - **伸びしろの見積もり (導出)**: 1 層の routed は 40,960 組 × 約 5.25M 積和 = 215G FLOPs、48 層で 10.3T を 27.6 s = 実効 374 GFLOPS。
     04 の注意の sgemm は約 600 GFLOPS だったので、FLOPs の効率だけなら上限 1.6 倍程度。展開 (1 層 約 470 expert × 5.2M 重み) と集め・散らしの分がそこから引かれる。
   - **先に 1 層の A/B で見積もる。**`--q8-gemm-bench` (`GGMLDenseCheck.swift` の `runQ8GemmBench`) と同じ作りで、実際の選択分布 (1 expert あたり平均 87 組、偏りあり) を使う。
     正しさは `--q2-expert` の fixture と、組み込んだ後に §6-2 の一式。IQ2_XXS / Q2_K の展開カーネルはまだ無い (ds4 の行演算を `moe_ggml.metal` から移す)。
   - 1.6 倍が見込めないなら 2 に回る。
2. **GDN** (10.2 s: `q38_gdn_step` 6.3・入力 GEMM 2.6・出力 1.3)。step は時間方向に直列。チャンク内の並列形 (delta rule の WY/chunked 形) は大きい。
3. **32K を 1 回流す** (チャンク 4096、`guarded.sh` 越し、約 8 分の見込み)。注意の伸び (04 §4 の 4) と footprint を見る。KV・インデクサ鍵が増えるぶんページキャッシュが減るので、Swapouts に注意。
4. 読み 5.6 s、PLE 3.5〜5.4 s、top-10 1.4 s (host の 512 要素 × 10 回の選択)、チャンク 4096 の footprint 5.03 GB の内訳 (§3 の 4)。

### 6-4. 落とし穴 (04・05 で踏んだもの。03 §6-4 に追加)

- **長いベンチは `Scripts/qwen38/guarded.sh` 越しに回す。**ページキャッシュへの読みだけでも Swapouts が動いた (§2-1)。
- **ページキャッシュを埋めるなら pread。**mmap のフォルトも `F_RDADVISE` も 1 GB/s 前後 (§1)。
- **読みの測定は冷えた範囲で。**`readbench` は毎回まだ読んでいない層を指定する。
- **in place で書き換えるバッファ** (`q38_attn_prep` の KV 行、`q38_idx_q_prep` の `iq`) は、新旧を同じ入力で比べるときに戻す (04 §6)。
- MSL に `bitcast` は無い (`reinterpret_cast<device const uint*>`)。
- `.metal` を変えたら `swift build` でリソースを更新してから走らせる (古いソースのまま実行時コンパイルされ、直したはずのエラーが出続ける)。
