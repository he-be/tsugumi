# 02. 重みの取得と CPU 参照器

実測: 2026-09-17、M3 Pro 18 GB、macOS 15。[01](01-FEASIBILITY.md) §5 の 1・2 段。
表記は [qwen38/01](../qwen38/01-Q2-FIRST-LIGHT.md) と同じ (実測 / 導出 / 未確認)。どの数字も 1 回ずつの測定 (n=1)。

## 0. 結論

1. **`Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` を取得した。**12,120,016,960 B、sha256 は HF の LFS 値と一致。所要 19 分 40 秒 (約 10 MB/s)。
   curl の RSS は 8.5 MB で一定、Swapouts の増分は 0 (§1)。
2. **qwen35 dense の CPU 参照器ができた** (`Scripts/qwen38_27b/reference_forward.py`)。
   "The capital of France is" → greedy で ` Paris.\nThe capital of` (§2-1)。
3. **負例で算式を確かめた。**53 トークンの英文の平均 NLL: 正例 0.900、GDN のヘッドをブロック対応で組む負例 8.13、ゲート付きノルムを sigmoid にする負例 11.37 (§2-2)。
4. 1 回の通しは約 40 秒 (64 層 + 出力)。footprint はワーカー込みで最大 3.62 GB、Swapouts の増分は 0 (§2-3)。

## 1. 取得

| もの | 場所 |
| --- | --- |
| 本体 GGUF | `~/LLM/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (sha256 `58fd8267…c12f`) |
| 取得スクリプトとログ | 同じディレクトリの `download.sh`・`download.log` (10 秒ごとのサイズ・RSS・Swapouts) |
| トークナイザ | `~/LLM/Qwen3.8-27B-GSQ-RCO-GGUF/tokenizer/` (`Qwen/Qwen3.8-27B` の `tokenizer.json`・`tokenizer_config.json`)。`tokenizer.json` は Flash-Next のものとバイト単位で同一 |

`curl -sSL --fail --retry 20 -C -` で 1 本だけ取得した。見張りは curl の RSS 5 GB、Swapouts の 60 秒で +512 MiB または合計で +1 GiB で止める設定 (どれも発動しなかった)。

## 2. CPU 参照器

算式は手元の llama.cpp (fe8156f78、**読むだけ**) の `src/models/qwen35.cpp`・`src/models/delta-net-base.cpp` と `conversion/qwen.py` に従う。
Flash-Next の参照器 (`Scripts/qwen38/reference_forward.py`) との違い:

| | Flash-Next (qwen4exp) | 27B (qwen35) |
| --- | --- | --- |
| 残差 | hc 4 本 + mixer | 1 本。`x += mixer(rms(x, attn_norm))`、`x += ffn(rms(x, post_attention_norm))` |
| FFN | MoE (routed 10 + shared) | dense SwiGLU (SiLU) |
| GDN の出力ノルムのゲート | sigmoid | **SiLU** (`build_norm_gated`) |
| GDN のヘッドの組み方 | value j ↔ key j % 16 | 同じ (変換器 `_LinearAttentionVReorderBase` が V を巡回順に並べ替え、`ggml_repeat_4d` で広げる) |
| KV ヘッド | 2 | 4 (q head h → kv head h // 6) |
| PLE・indexer | あり | なし |
| 最終ノルム | output_hc の mixer | `output_norm` |

共通点: ノルムの gamma は `ssm_norm` 以外 1+w を焼き込み済み (`conversion/qwen.py` 394 行)、`ssm_a` は −exp(A_log)、q_proj は head ごとに [Q 256, gate 256]、部分 rope 64 次元 (neox 形)。
rope の sections [11, 11, 10, 0] (IMRoPE) は、テキストだけなら 3 本の位置が同じなので neox と同じになる (導出)。ランナーとの一致で確かめる。
l2 ノルムは ggml と同じ `x / max(‖x‖, eps)` にした (Flash-Next の参照器は `sqrt(Σx² + eps)`)。

27B の f32 は 100 GB を超えるので常駐させない。行列積は約 400 万要素ずつ行を区切り、spawn したワーカー 8 本が gguf-py で逆量子化して掛ける。
逆量子化が通しの大半を占めるので、プロンプトは全位置を層ごとにまとめて流す。生成は 1 トークンずつ全層を通す。
gguf-py は 10 型 (Q4_K・IQ2_S・IQ4_XS・IQ2_XS・IQ2_XXS・IQ3_XXS・IQ3_S・Q2_K・IQ1_M・Q6_K) をすべて逆量子化できた (1 コアで 107〜570 M 要素/秒)。

### 2-1. 生成 (実測)

"The capital of France is" (5 トークン `[760, 6511, 314, 9338, 369]`) → greedy ` Paris.\nThe capital of`。

| 位置 | トークン | logit |
| ---: | --- | ---: |
| 4 | ` Paris` (11751) | 17.361 |
| 5 | `.` (13) | 19.224 |
| 6 | `\n` (198) | 14.783 |
| 7 | `The` (760) | 12.655 |
| 8 | ` capital` (6511) | 14.878 |
| 9 | ` of` (314) | 20.421 |

### 2-2. 負例 (実測)

qwen38/01 §3-2 と同じ 53 トークンの英文 (富士山、`scratch/qwen38_27b/fuji.tokens`) で、52 位置の平均 NLL を取った。`--variant` で算式をわざと変える。

| 算式 | 平均 NLL |
| --- | ---: |
| 正例 | **0.900** |
| `gdn-block`: value head j を key head j // 3 と組む (HF の `repeat_interleave` をそのまま写した形) | 8.132 |
| `gate-sigmoid`: GDN の出力ノルムのゲートを sigmoid にする (Flash-Next の形) | 11.370 |

正例の top-1 が次のトークンと一致したのは 52 位置中 35。

### 2-3. 時間とメモリ (実測)

| | 値 |
| --- | ---: |
| 1 回の通し (64 層 + 出力、1 トークン) | 40.1〜40.3 s |
| 5 トークンのプロンプト (ワーカー起動込み) | 48.8 s |
| 53 トークンのプロンプト (起動込み、全体) | 56.5 s |
| footprint (親 + ワーカー 8 本の合計、最大) | 3.62 GB |
| Swapouts の増分 | 0 |

footprint 5 GB 超か Swapouts +64 MB で自分で止まる。

## 3. 参照の在処と作り直し

`scratch/` は git 管理外。消えたら次で作り直す (それぞれ 1〜5 分)。

| ファイル | 中身 |
| --- | --- |
| `scratch/qwen38_27b/ref-france.log` + `ref-france.logits` | France、10 位置 × 248,320 の float32 |
| `scratch/qwen38_27b/ref-fuji-ok.log` + `ref-fuji.logits` | 富士山、52 位置 |
| `scratch/qwen38_27b/ref-fuji-{gdn-block,gate-sigmoid}.log` | 負例 |

```bash
P=~/LLM/venv/bin/python; R=Scripts/qwen38_27b/reference_forward.py; S=scratch/qwen38_27b
$P $R --text "The capital of France is" --new 6 --dump-logits $S/ref-france.logits > $S/ref-france.log
$P $R --tokens $(cat $S/fuji.tokens) --new 0 --dump-logits $S/ref-fuji.logits > $S/ref-fuji-ok.log
$P $R --tokens $(cat $S/fuji.tokens) --new 0 --variant gdn-block > $S/ref-fuji-gdn-block.log
```

`fuji.tokens` は `scratch/qwen38/ref-fuji-ple.log` の 1 行目のトークン列。KV Q8_0 の参照は `--kv-type q8_0` で作る (まだ作っていない)。

## 4. 次

01 §5 の 3 段: IQ3_S の GEMV 1 本 (正解は gguf-py の float64 逆量子化、`TsugumiKernelCheck`)。一致したら残りの型、次に prefill 用。
