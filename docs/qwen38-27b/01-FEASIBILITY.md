# 01. Qwen3.8-27B GSQ-RCO IQ3_S をこの Mac で動かす検討

書いた日: 2026-09-17。対象は `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` の `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (12,120,016,960 B)。
表記は [qwen38/01](../qwen38/01-Q2-FIRST-LIGHT.md) と同じ (実測 / 導出 / 未確認)。
**この文書は検討と計画。**別のタスクがマシンを使っていたため、読んだのは HF の README・テンソル割り当て表・GGUF の先頭 24 MB (Range 取得) と、
Metal の上限値 (`MTLDevice`) だけ。ダウンロード・ビルド・推論はしていない。§1 と §2 の上限値以外の数字は導出。

## 0. 結論

1. **dense 27B なので、重みを全部 GPU に載せる形にする。**Flash-Next のようなエキスパートのストリーミングは要らない。
   構造は Qwen3.5 系のハイブリッド (GDN 48 層 + gated attention 16 層) で、`Qwen38Runner` の GDN・注意・KV Q8_0・MTP・チェックポイントがそのまま土台になる (§1)。
2. **32K・KV Q8_0・MTP ありで GPU に載せる量は約 12.18 GiB + 作業域。この機体の `recommendedMaxWorkingSetSize` は 12.0 GiB (実測) なので、
   上限を上げる。`sudo sysctl iogpu.wired_limit_mb` で上げる (ユーザー決定、2026-09-17)** (§2)。
3. ダイエットで削れるのは `token_embd` を CPU に置く 0.38 GiB だけ。dense なので脂肪はほとんど無い (§2-2)。
4. **いちばん大きい作業は dense の行列積カーネル。**混在型 10 種 (IQ3_S 144 本・IQ4_XS 96・IQ3_XXS 78・Q4_K 39・IQ2_S 17・Q2_K 13・IQ2_XS 9・Q6_K 8 + MTP・IQ2_XXS 5・IQ1_M 1) の
   decode 用と prefill 用。今ある dense のカーネルは Q8_0 / F16 / F32 / BF16 だけ (§3)。
5. 運用点は Flash-Next と同じ (thinking 無効が主体・32K・MTP は使えるなら使う・即答してツールを呼ぶエージェント主体・英語主体)。
   **サンプリングは thinking 無効 / 有効それぞれの公式値に固定** (§4)。
6. 順序: ダウンロード → CPU 参照器 → IQ3_S カーネル 1 本 → ランナーの参照一致 → sysctl を上げて 4K / 16K / 32K の wired・Swapouts・tok/s → MTP の受理率 (§5)。

## 1. モデルの形 (実測、GGUF ヘッダ)

`Scripts/qwen38_27b/gguf_header.py` で先頭 24 MB を読んだ (ヘッダの終わりは 10,995,772 B)。

| キー | 値 |
| --- | --- |
| `general.architecture` | `qwen35` |
| `block_count` / `nextn_predict_layers` | 65 / 1 (blk.0..63 が本体、blk.64 が MTP) |
| `full_attention_interval` | 4 (本体 64 層のうち 16 層が全注意、48 層が GDN) |
| `embedding_length` / `feed_forward_length` | 5,120 / 17,408 (dense SwiGLU) |
| `attention.head_count` / `head_count_kv` / `key_length` / `value_length` | 24 / 4 / 256 / 256 |
| `rope.dimension_count` / `rope.dimension_sections` / `rope.freq_base` | 64 / [11, 11, 10, 0] / 1e7 |
| `ssm.inner_size` / `state_size` / `group_count` / `time_step_rank` / `conv_kernel` | 6,144 / 128 / 16 / 48 / 4 |
| `context_length` | 262,144 |
| 語彙 | 248,320 (eos 248046、add_bos false) |
| `general.sampling.*` | temp 1.0 / top_p 0.95 / top_k 20 / min_p 0 (thinking 有効の値だけが入っている、§4) |

テンソル名と形 (blk.0 = GDN、blk.3 = 全注意):

- GDN: `attn_qkv` [5120 → 10240]、`attn_gate` [5120 → 6144]、`ssm_alpha` / `ssm_beta` [5120 → 48] (BF16)、`ssm_a`・`ssm_dt.bias` [48]、`ssm_conv1d` [4, 10240]、`ssm_norm` [128]、`ssm_out` [6144 → 5120]
- 全注意: `attn_q` [5120 → 12288] (Q 6,144 + ゲート 6,144)、`attn_k` / `attn_v` [5120 → 1024]、`attn_q_norm` / `attn_k_norm` [256]、`attn_output` [6144 → 5120]
- 各層: `attn_norm`、`post_attention_norm`、`ffn_gate` / `ffn_up` [5120 → 17408]、`ffn_down` [17408 → 5120]
- MTP (blk.64): 全注意 1 層 + dense FFN + `nextn.eh_proj` [10240 → 5120]・`enorm`・`hnorm`・`shared_head_norm`。すべて Q6_K / F32

**Flash-Next (`qwen4exp`) との違い:** hc (hyper-connection)・PLE・QSA indexer・MoE が無い。`post_attention_norm` と dense FFN がある。KV ヘッドが 2 → 4。
GDN・gated attention・`nextn` はテンソル名が同じ。indexer が無いので、32K の注意は密 (§2 の KV が全部効く)。
rope の sections はテキストだけなら 3 本とも同じ位置になるので、ふつうの部分 rope (64 次元) と同じになるはず (未確認、参照器で確かめる)。

HF のモデルカード (`Qwen/Qwen3.8-27B`) の記述: 64 層、「16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))」、MTP で学習、
文脈 262,144 (1M まで拡張可)、thinking は `chat_template_kwargs` の `enable_thinking: False` で無効。

### 1-1. 量子化の割り当て (実測、HF の `tensor-allocation/…IQ3_S-mtp.rco-allocation.txt`)

全体 3.50 bpw (MTP ヘッド込みで 3.5457)。866 テンソル: BF16 96、F32 360、IQ1_M 1、IQ2_S 17、IQ2_XS 9、IQ2_XXS 5、IQ3_S 144、IQ3_XXS 78、IQ4_XS 96、Q2_K 13、Q4_K 39、Q6_K 8。
型はテンソルごとに違う (例: blk.0 は `ffn_down` IQ2_S・`ffn_gate` IQ2_XS・`ffn_up` IQ2_XXS、`attn_qkv` / `attn_gate` IQ4_XS)。
`token_embd` IQ2_S、`output` Q4_K。MTP 版と MTP なし版は、blk.64 の 15 本以外の割り当てが同一。

HF の品質表 (BF16 基準、ISTA の測定): IQ3_S は AIME25 100.00 / GPQA-D 89.39 / LCB v6 85.71 (BF16 は 100.00 / 89.90 / 85.71)。
MTP の図 (llama.cpp、3 トークン先読み、機材の記載なし): 受理率の平均 54.2%、IQ3_S の decode 104〜116 t/s。**機材が違うので、この機体の見込みには使わない。**

## 2. メモリ

### 2-1. 上限 (実測、2026-09-17、`iogpu.wired_limit_mb` = 0 のとき)

| | 値 |
| --- | ---: |
| 物理メモリ (`hw.memsize`) | 18.0 GiB |
| `recommendedMaxWorkingSetSize` | 12.0 GiB |
| `maxBufferLength` | 9.0 GiB |

`maxBufferLength` があるので、重みは 1 本のバッファにせず、テンソルごとの mmap no-copy ビューで持つ (`Qwen38Runner` と同じ)。

### 2-2. 載せる量 (導出)

| 中身 | GiB | 備考 |
| --- | ---: | --- |
| ファイル全体 (`-mtp`) | 11.29 | MTP なし版は 10.96、MTP ヘッドは 0.32 |
| − `token_embd` (IQ2_S、82 B / 256 重み) | −0.38 | 1 トークン 1 行を引くだけなので CPU (mmap) に置く |
| うち `output` (Q4_K) | (0.67) | 毎トークン全語彙に掛けるので GPU に要る |
| KV (全注意 16 層 + MTP 1 層、Q8_0、1 トークン 36,992 B) | 32K: 1.13 / 16K: 0.56 / 4K: 0.14 | f16 なら 32K で 2.13 |
| GDN の状態 (f32、48 層 × 48 ヘッド × 128 × 128 + conv 3 × 10,240) | 0.15 | 文脈に依らない |
| **合計 (32K)** | **12.18 + 作業域** | 作業域 (prefill の活性・logits・ドライバ分) は未測定 |
| 合計 (16K) | 11.62 + 作業域 | |

- **dense は毎トークンすべての重みを読む。**「常駐しない分を SSD から読む」逃げ道が無く、載せる量がそのまま使用量になる。
- ダイエットで削れるのは `token_embd` だけ。ほかに読まずに済むのは mmproj (0.9 GB、読み込まない) と imatrix (配布物のみ)。`ssm_alpha` / `ssm_beta` の BF16 は合計 47 MB で、手を入れる理由が無い。
- 参考: Flash-Next の 12K の運用点でも wired の最大は 14.4〜15.0 GB ([qwen38/16 §0](../qwen38/16-WEIGHT-DIET.md))。wired は GPU 以外の分も含むので、この表とは直接比べない。

### 2-3. 上限を上げる (ユーザー決定、2026-09-17)

32K を通すため、`iogpu.wired_limit_mb` を上げる。手順と注意は [SERVER_RUNBOOK](../SERVER_RUNBOOK.md) と同じ:

```bash
sysctl iogpu.wired_limit_mb                  # 0 (既定) か確認
sudo sysctl iogpu.wired_limit_mb=14336       # 再起動で戻る
```

- 値は 14,336 MB (14.0 GiB) から始める。12.18 GiB + 作業域に対して約 1.8 GiB の余り。OS とアプリに残るのは約 4 GiB。
- **未確認:** この機体で sysctl を上げたとき `recommendedMaxWorkingSetSize` が追従するか。上げた後に `MTLDevice` の値を読み直してから計測する。
- **sudo なので、夜間の自律実行では設定しない。**ユーザーが立てる。
- 再起動で戻るので、アプリで 27B を選ぶときは値を確認し、足りなければ知らせる必要がある (配布の扱いは別途)。

## 3. コードで足りないもの

### 3-1. dense の行列積カーネル (最大の作業)

| 型 (ggml id) | テンソル数 | 今あるもの |
| --- | ---: | --- |
| IQ3_S (21) | 144 | なし |
| IQ4_XS (23) | 96 | なし |
| IQ3_XXS (18) | 78 | なし |
| Q4_K (12) | 39 | MoE エキスパート専用 (`moe_ggml.metal`) |
| IQ2_S (22) | 17 | なし |
| Q2_K (10) | 13 | MoE エキスパート専用 |
| IQ2_XS (17) | 9 | なし |
| Q6_K (14) | 8 + MTP 10 | なし |
| IQ2_XXS (16) | 5 | MoE エキスパート専用 |
| IQ1_M (29) | 1 | なし |
| BF16 / F32 | 96 / 360 | `ggml_dense.metal` にあり |

- decode (1 行) 用の GEMV と、prefill (T 行) 用の両方が要る。今の dense は `ggml_dense.metal` の Q8_0 / F16 / F32 / BF16 だけ。
- 算式の出典は手元の llama.cpp (`~/LLM/llama.cpp`、fe8156f78) の `ggml/src/ggml-metal/ggml-metal.metal` (MIT)。**推論には使わず、読む資料としてだけ使う** ([qwen38/01 §1](../qwen38/01-Q2-FIRST-LIGHT.md))。
- 正解は gguf-py の逆量子化 (float64) で作り、`TsugumiKernelCheck` で相対誤差を見る (qwen38/01 §2-1 と同じ形)。
- `GGUFFile.GGMLType` に iq3_s / iq3_xxs / iq4_xs / iq2_s / iq2_xs / q6_K / iq1_m を足す。

### 3-2. ランナー

`Qwen38Runner` を削る: hc の mix / combine、PLE、indexer、MoE (ルータ・shared・routed) を外し、`post_attention_norm` と dense SwiGLU を足す。
arch キーは `qwen35.*`。KV ヘッド 4。`token_embd` は CPU で行を逆量子化して渡す。
Flash-Next の実装と共通にするか別ファイルにするかは、参照一致が取れてから決める。

### 3-3. CPU 参照器

`Scripts/qwen38/reference_forward.py` を qwen35 dense に直す。重みは層ごとに gguf-py で逆量子化し、使い終わったら捨てる (27B の f32 は 100 GB 超なので全部は持たない)。
最初の検査は qwen38/01 と同じく "The capital of France is" と 53 トークンの英文で、top-1 と logits の相対誤差を見る。

## 4. 運用点とサンプリング (ユーザー指定、2026-09-17)

運用点は Flash-Next と同じ ([qwen38/01 §1-1](../qwen38/01-Q2-FIRST-LIGHT.md)、[qwen38/23](../qwen38/23-KV-Q8-32K.md)):

- thinking 無効が主体
- 文脈 32K、KV Q8_0
- MTP は使えるなら使う
- 即答してツールを呼ぶエージェント動作が主体
- 英語主体

**サンプリングは公式値に固定し、変更しない** (`Qwen/Qwen3.8-27B` のモデルカード):

| | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| thinking 有効 | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| thinking 無効 (instruct) | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

- GGUF の `general.sampling.*` には thinking 有効の値しか入っていない。**メタデータから読まず、thinking の有無で表から選ぶ。**
- thinking 無効の値は、Flash-Next の `Qwen38Sampler` (presence_penalty 1.5 を生成トークンに掛ける、HF の順序) と同じ。
- 評価のときだけ greedy を使う。

## 5. 進め方

各段で測れる最小の形を作り、落ちたら止めて報告する。

1. **ダウンロード。**別タスクが終わってから、`IQ3_S-mtp.gguf` (12.1 GB、100 Mbps で約 17 分) を `curl -C -` で 1 本取得する。
   RSS・実書き込み・Swapouts を最初の数分見る。`hf download` (xet) は使わない。
2. **CPU 参照器** (§3-3)。
3. **IQ3_S の GEMV 1 本** (§3-1)。一致したら残りの型、次に prefill 用。
4. **ランナーの参照一致** (§3-2)。全位置の top-1 と logits。
5. **メモリと速度。**sysctl を上げて (§2-3) 4K / 16K / 32K で wired・Swapouts・decode tok/s・prefill を測る。反復 3 回未満のセルには解釈を書かない。
6. **MTP。**英語のエージェント課題で受理率と n_max を測る (thinking 無効・公式サンプリング)。
7. server / アプリへの結線は、5・6 の後に別文書で。

## 6. 再現

```bash
R=https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main
curl -sL -r 0-25165823 -o head.bin $R/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
python3 Scripts/qwen38_27b/gguf_header.py head.bin
curl -sL -O $R/tensor-allocation/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.rco-allocation.txt
```

`recommendedMaxWorkingSetSize` / `maxBufferLength` は `MTLCreateSystemDefaultDevice()` の値を `swift` で 1 行印字した。
