# 検証計画: Qwen3.8-Flash-Next を Tsugumi に載せる前に、M3 Pro と PC で決着させること

作成: 2026-09-13
目的: [M6_QWEN38_FLASH_NEXT.md](M6_QWEN38_FLASH_NEXT.md) の試算 (32 GB で 15〜18 tok/s) は導出だけで組んである。
M6 (9/22 着) を待たずに **M3 Pro 18 GB と PC (RTX 5080) で確定できる数字を先に取り、24/32 GB への変更を締切前に判断する**。
表記は他の調査文書と同じ (実測 / 導出 / 公称 / 未確認)。

> **2026-09-13 夜の改訂**: 本線は DS4-IQ2 (Q2) に替わり、参照機はこの MBP だけ (DS4 / llama.cpp は推論に使わない) になった。
> 以下の「oQ4e を引いて affine で打ち直す」「DS4-IQ2 は取らない」は置き換わっている。
> 進捗と再開手順は [../qwen38/01-Q2-FIRST-LIGHT.md](../qwen38/01-Q2-FIRST-LIGHT.md) (Metal ランナーが CPU 参照と logits まで一致、QSA indexer 込み)。

---

## 0. 結論を 3 行で

1. **ウェイトは一から作らなくてよい。Ornith と同じ手順で oQ を引いてくる。**
   `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` (2026-08-27、約 106 GB、imatrix つき、MTP 込み) が
   Ornith の `scottlowry/…-oQ4e-mtp` にあたる。メニュー外 (5/6-bit、g128) の打ち直しも [qwen35moe/11](../qwen35moe/11-OQ4E-G64-REBUILD.md) と同じ
   (上流 bf16 の range 取得 → 8-bit g64)。PLE 表は 4-bit **g32** なので GPU に載せず CPU で gather する (§2-3)。
2. **ただし oQ4e は Q4_K_XL 級 (expert 2.76 MB) であって IQ3_XXS 級ではない。**
   本ランタイムは affine int4/int8 しか持たず、IQ2_S / IQ4_NL / Q6_K は乗らない。
   前回の「IQ3_XXS 前提で 32 GB 15〜18」に相当する数字を出すには **2-bit または 3-bit affine の expert カーネルを新規に書く**必要がある (§1)。
   その品質の裏付けは、PC で GGUF の KLD を取ること (§4 D2) と、`ivanfioravanti/…-DS4-IQ2` の公表値 (2-bit imatrix expert で bf16 に対し top-1 90.3%、§2-2) から始める。
3. **9/22 前に決着させるのは 3 つ (§5)**: (D1) expert キャッシュ hit の外挿を PC のトレースで机上確定 (Mac 不要)、
   (D2) 2〜3 bpw expert の品質を PC の KLD で判定、(D3) M3 Pro の SSD で expert 大のランダム読みを実測し、D1 の hit と合わせて decode を合成する。
   この 3 つで「32 GB に変えるか」は決まる。Tsugumi の移植 (D4〜D8) はその後で、M6 の到着とは独立に進む。
   **既製の ds4-metal で decode の実物を取る案は捨てた** (§2-4): Qwen 経路に SSD ストリーミングが無く、41.7 GiB を全部 GPU 常駐させるので 18 GB では動かない。

---

## 1. 前提の訂正: 「IQ3_XXS 前提」は本ランタイムの形式に直すと何になるか

| 経路 | routed expert の形式 | expert 1 個 | 48 層合計 | 1 トークンの expert バイト | [M6 文書](M6_QWEN38_FLASH_NEXT.md) §3 の対応列 | カーネル |
| --- | --- | ---: | ---: | ---: | --- | --- |
| **(a) oQ4e-g64** (Ornith と同じ) | affine 4-bit g64 (4.5 bpw) | **2.76 MB** | 68 GB | 1.42 GB | Q4_K_XL 列 (32 GB で 10〜13) | **有る** |
| (b) affine 3-bit g64 | 3.5 bpw | 2.15 MB | 53 GB | 1.10 GB | Q4 と IQ3 の間 | **無い** |
| (c) affine 2-bit g32 (Vontra oQ2 の既定) | 3.0 bpw | 1.84 MB | 45 GB | 0.95 GB | IQ3_XXS 列 (32 GB で 15〜18) | **無い** |
| (d) affine 2-bit g64 | 2.5 bpw | 1.54 MB | 38 GB | 0.79 GB | IQ3_XXS より軽い | **無い** |
| 参考: GGUF UD-IQ3_XXS | IQ2_S gate/up + IQ4_NL down | 1.97 MB | 48.6 GB | 0.945 GB | IQ3_XXS 列 | 無い (llama.cpp / ds4-metal 用) |
| 参考: GGUF DS4-IQ2 | IQ2_XXS gate/up + Q2_K down (768 に pad) | **1.49 MB** | 36.6 GB | 0.72 GB | — | 無い (ds4-metal 用) |

expert 1 個 = 3 × 2560 × 640 = 4.92 M param。affine の bpw = bits + 32/group (scale と bias が bf16)。
DS4-IQ2 の行は `.gguf.json` の tensor バイト数から (**実測(上流)**): 1 層の routed expert 762,839,040 B ÷ 512 = 1,489,920 B (gate/up 各 422,400 B + down 645,120 B)。

**dense は expert より重い。** 1 トークンに読む dense は Q8_0 で約 3.0〜3.5 GB、expert は上の列。
M6 の帯域 (170 GB/s) では dense 3 GB だけで 23 ms を食うので、**expert を 2 bpw に落とすより dense を 8 → 4-bit に落とす方が効く** ([M6 文書 §3-1](M6_QWEN38_FLASH_NEXT.md))。
oQ4e は dense (attention / linear_attn / shared expert / lm_head) を 5〜8 bit に守っており、打ち直しでこれを 8-bit g64 に**上げる**のが Ornith の流儀だった。
Flash-Next では逆に **dense を 4-bit に落とす案**を品質つきで検討する価値がある (§4 D2 の派生)。

---

## 2. ウェイトの供給源 (HF、2026-09-13 時点。取得は 1 つもしていない)

### 2-1. 候補

| リポジトリ | 形式 | サイズ | routed expert | 本ランタイム | 用途 |
| --- | --- | ---: | --- | --- | --- |
| **`Jundot/Qwen3.8-Flash-Next-oQ4e-mtp`** | MLX affine、oQ (imatrix)、MTP 込み | 約 106 GB (21 シャードの和) | 4-bit g64 | メニュー外の打ち直し後に乗る (§2-2) | **本線 (a)** |
| `Vontra/Qwen3.8-Flash-Next-MLX-oQ2-MTP` | MLX affine、oQ、MTP 込み | 未確認 (推定 55〜60 GB) | **2-bit g32** (既定)、dense は 3〜8 bit | 2-bit カーネルが要る。dense の 3/5/6-bit は打ち直し | **(c) の供給源。品質の対照** |
| `Vontra/…-MLX-oQ4-MTP` / `-oQ8-MTP` | 同上 | 未確認 | 4-bit / 8-bit | Jundot と同種 / 8-bit は対象外 | 予備 |
| `mlx-community/Qwen3.8-Flash-Next-4bit` | MLX affine、素の RTN | 112 GB | 4-bit **g32** (全部) | g32 で統一されているので打ち直し不要だが、imatrix 無し、MTP 無し | 対照 (Ornith の公式 MLX-4bit にあたる) |
| `GBP-DE/…-oQ5e-mtp` | MLX affine | 未確認 | 5-bit | 5-bit カーネルは無い | 対象外 |
| **`ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2`** | GGUF (ds4-metal 専用)。**PLE サイドカー Q4_1 (32 GB) は同梱されず `…-DS4-Q4` リポジトリにある** | 44.8 GB (+ 別リポの PLE 32 GB) | IQ2_XXS gate/up、Q2_K down | 乗らない | **2-bit 品質の公表値だけ使う**。M3 Pro での実行は不可 (§2-4) |
| `unsloth/…-GGUF` UD-Q4_K_XL / IQ4_XS / IQ3_XXS | GGUF | 104 / 87 / 76 GiB | Q4_K+Q5_1 / IQ3_S / IQ2_S | 乗らない | PC / llm-server に有る。**§4 D2 の KLD 基準** |
| `Qwen/Qwen3.8-Flash-Next` (上流 bf16) | safetensors bf16 | 約 360 GB | — | 打ち直しの原本。**全体は引かない**、range 取得のみ | §2-2 |

**oQ を自分で回す案は成立しない。** omlx の oQ は MLX でモデルを全載せして校正推論をするので、
177B の 4-bit (106 GB) でも 128 GB 以上の Mac が要る。M3 Pro 18 GB でも M6 32 GB でも不可。
**低ビット版が要るなら Vontra の oQ2 を引くか、bf16 から素の RTN で自分で切る** (CPU、range 取得、校正なし。品質は落ちる)。

### 2-2. Jundot oQ4e-mtp の中身 (config.json を読んだ。**実測(上流)** = ヘッダとメタデータ)

- `architectures: ["Qwen4ExpForConditionalGeneration"]`、`model_type: qwen4_exp`。既定 `4-bit / g64 / affine`。
- text_config: 48 層、512 expert top-10、`moe_intermediate_size 640`、hidden 2560、GDN **16 QK head / 48 V head × 128**、
  full attention **24 Q / 2 KV × 256**、vocab 248,320、`ngram_size 3`、`ngram_vocab_size_base 20,000,000`、`ple_layer_ids [2]`、MTP 1 層。
- メニュー外 (打ち直し対象、概数):

| bits / group | 本数 | 何か |
| --- | ---: | --- |
| 4 / **g32** | 128 | **PLE 表** `layers.1.ple.ple_embedding.ngram_embedding.shards.*` (次元 160 は 64 で割れない) |
| 5 / g64 | 約 180 | `attn_hyper_connection.block_inject_weight` など (**hyper-connection、Qwen3.5 に無い部品**)、MTP の同種 |
| 5 / g128 | 約 60 | `linear_attn.out_proj` / `in_proj_z` の一部 |
| 6 / g64 | 約 80 | `linear_attn.in_proj_a/b/qkv`、`self_attn.q/k_proj`、**indexer** |
| 8 / g128 | 約 150 | `mlp.shared_expert.{gate,up,down}_proj` 全層 |
| 8 / g64 | 約 40 | `lm_head`、`embed_tokens`、`shared_expert_gate` |

g128 が出るのは Ornith と同じ理由 (`oq.py` の `gs()` が `num_experts >= 150` で 128 を返す)。
routed expert 以外を **8-bit g64 に打ち直す**手順は [11 §2](../qwen35moe/11-OQ4E-G64-REBUILD.md) そのまま:
`config.json` の override から `group ≠ 64 or bits ∉ {4,8}` を機械的に拾い、上流 bf16 を HTTP Range で取って `mx.quantize(bits=8, group_size=64)`。
対象は Ornith の 248 本より多く **約 500 本、bf16 換算 6〜8 GB** の取得で済む見込み (**未確認**)。

- **DS4-IQ2 の公表品質 (公称)**: bf16 参照 99 本 2,376 トークンで **top-token 一致 91.3% (down MXFP4) / 90.3% (down Q2_K)**、NLL 0.291 / 0.304。
  gate/up が IQ2_XXS (2.06 bpw) でこの値なら、このモデルは routed expert の低ビット化に強い (ds4-metal も「aggressive routed-expert quantization を許容する」と書く)。
  **affine 2-bit RTN は IQ2_XXS (格子コードブック + imatrix) より確実に悪い**ので、この数字は上限として読む。

### 2-3. PLE 表の扱い

- 16 head × 約 2000 万行 × 160 次元、4-bit g32 で約 29 GB (mlx-community は「PLE を量子化するには g32 が要る」と明記)。
- 本ランタイムは affine group size がシェーダライブラリ全体のコンパイル時定数で、モデル内の g32 / g64 混在は不可 ([02 §5](../qwen35moe/02-CHECKPOINTS.md))。
- **解: PLE は GPU に載せない。**1 トークンに要るのは 16 行 × 160 = 2,560 値だけなので、ホストが pread で 16 行 (各約 100 B) を読み、
  CPU で逆量子化して 2,560 次元の bf16 ベクトルを第 2 層の入力に足す。ハッシュ (`layer_multipliers` 3 本、`head_offsets`、`head_vocab_sizes` の素数)
  は llama.cpp `qwen4exp.cpp` から写す。読みはトークン確定時に発行して第 1 層に隠す ([M6 文書 §2](M6_QWEN38_FLASH_NEXT.md))。
- prefill は 2048 × 16 本の pread をチャンク先頭でまとめて発行する。

### 2-4. ds4-metal で DS4-IQ2 を M3 Pro 18 GB で動かせるか → 動かない (コードを読んだ。実行はしていない)

調べたのは `ivanfioravanti/ds4-metal` の `origin/qwen3.8-flash-next` = `3030554` (2026-09-12)。

- **ブランチの状態**: Qwen 対応は 9/12 に main (`bd66c40`) の上へ 1 コミットにまとめて載せ直されている。
  HF のモデルカードが要求する `5bd8796` (padded Q2_K down) はこのブランチの祖先ではなくなったが、変更内容は `metal/qwen4.metal` に入っている。
  ローカルに `qwen3.8-flash-next` を作っただけだと main と同じで Qwen のコードは無い (`git checkout -B qwen3.8-flash-next origin/qwen3.8-flash-next` が要る)。
- **Qwen 経路に SSD ストリーミングは無い。**`docs/SSD_STREAMING.md` は「Metal supports streaming for DeepSeek and GLM」。
  `ds4.c:58505` で `generate_qwen4_metal_argmax()` に分岐し、streaming 系の引数は渡らない。`--ssd-streaming` はエラーにならず無視される。
- **GGUF 全体を GPU に常駐させる。**mmap した tensor 領域を no-copy の `MTLBuffer` view (1 本 ≤ `maxBufferLength`) に切り、
  全 view を 1 つの residency set に入れて `requestResidency` する (`ds4_metal.m:2124`、macOS 15 以降)。
  M3 Pro 18 GB の値は **`maxBufferLength` 9.0 GiB、`recommendedMaxWorkingSetSize` 12.0 GiB** (**実測**) なので、
  41.72 GiB を約 5 本の view で常駐させる形になり、GPU の out-of-memory か実用にならないページングになる (**導出**。起動は試していない)。
- **公称の下限**: モデルカードと `docs/QWEN38_FLASH_NEXT.md` は「64 GB なら ctx 8192 / prefill chunk 1024 から」。計測は M3 Ultra 512 GiB で、64 GB に収まることも確認していないと明記。
- **PLE サイドカー**は CPU 専用の private mmap で demand-paged (1 トークン ≈ head ごとに 1 ページ)。これ単体は 18 GB でも問題にならない。
- **GGUF の内訳** (`.gguf.json`、**実測(上流)**): 本体 41.72 GiB = routed expert (48 層) 34.10 + MTP ブロック 1.40 + token_embd (BF16) 1.18 + それ以外 (dense Q8_0、HC、norm など) 約 5.04。
- **Qwen 用ストリーミングを自作しても M3 Pro は代わりの測定台にならない**: `recommendedMaxWorkingSetSize` 12 GiB から常駐の非 expert 約 6.2 GiB と実行時バッファを引くと、
  expert キャッシュは 3〜4 GiB ≈ 2,000〜2,500 個 ≈ **1 層 40〜50 スロット** (**導出**)。D1 が見る 150 / 240 / 390 スロットと hit の効き方が違い、M6 32 GB への換算の土台にならない。

**結論**: D3 を「ds4-metal の実行」で決着させる案は捨てる。DS4-IQ2 (77 GB) の取得もしない。decode の実物は D6 以降の Tsugumi で取り、
9/22 前の判断は D1 (hit) + D3 (SSD の実測) + 既知の DRAM 帯域の合成で行う (§4、§5)。

---

## 3. Tsugumi 移植の差分 (Qwen3.5-35B-A3B → Flash-Next)

| 部品 | Ornith (今) | Flash-Next | 影響 |
| --- | --- | --- | --- |
| hidden / expert ff | 2048 / 512 | 2560 / 640 | `ArchConfig` の値。640 = 64 × 10 で g64 は割れる |
| routed expert | 256、top-8 | **512、top-10** | `moe_phase2_down_reduce_k8` が k=8 固定 → k=10 版。`TsugumiKernelCheck` の `topK = 8` も。`expertStride` 1,769,472 → **2,764,800** |
| GDN | 16 QK / 32 V × 128 | 16 QK / **48 V** × 128 | `qwen_delta_rule` の head 数はパラメータのはず (**未確認**)。状態は 48 × 128 × 128 × fp32 = 3 MB/層 × 36 層 |
| full attention | 16 Q / 2 KV × 256 | **24 Q** / 2 KV × 256 | GQA 比 8 → 12。[43 §2](../qwen35moe/43-LILY-IDEAS.md) の packed カーネルの動機が強まる |
| **hyper-connection** | 無し | `attn_hyper_connection` / `mlp_hyper_connection` (`block_inject_weight` ほか) | **新規。**残差の代わりに層入出力を混ぜる小さな行列。算式は上流 `modeling_qwen4_exp` から取る |
| **indexer (疎注意)** | 無し | 4 head × 128、予算 512 ブロック / 2048 トークン | **新規だが後回し可**: 文脈が 2048 以下なら全トークンが選ばれ dense attention と等価。参照器・fixtures はまず 2048 以下で作る |
| **PLE** | 無し | 第 2 層に 16 行の n-gram 埋め込みを加算 | §2-3。GPU カーネルは「2,560 次元を足す」だけ |
| MTP | 1 層 (sidecar 503 MB) | 1 層 (oQ4e 内蔵、約 2.5 GB) | [36](../qwen35moe/36-MTP-DECODE.md) の器がそのまま。効かない見込みなので後回し ([M6 文書 §1-6](M6_QWEN38_FLASH_NEXT.md)) |
| vision | 有り | 有り | 後回し |
| 量子化 | affine 4/8 g64 | 同じ + (PLE は CPU) | 2/3-bit は §4 D8 |

---

## 4. 決着リスト — 何を、どこで、いつまでに

**D1〜D3 は Tsugumi のコードを 1 行も書かずにできる。**D4 以降は移植で、M6 の到着と独立に進む。

| # | 問い | 方法 | 機材 | 所要 | 決まること / 判定線 |
| --- | --- | --- | --- | ---: | --- |
| **D1** | expert キャッシュの hit は 150 / 240 / 390 スロットでいくつか | PC の `qfn-bench/e2/trace1.tsv` (E2 の形式は `bench/expert_sim.py` が読める) を `expert_sim.py` / `cache_sim.py` で回す。admission つき LRU と LFU、幅 2 の和集合、層先読みの上限も同時に | **どこでも (Mac 不要)** | 半日 | [M6 文書 §1-5](M6_QWEN38_FLASH_NEXT.md) の外挿 (`1 − 0.45 (32/s)^0.6`) を実値に置換。**240 で hit ≥ 0.85 なら試算維持、< 0.80 なら 32 GB 案は落ちる** |
| **D2** | 2〜3 bpw の routed expert で品質は許容か | PC で `llama-perplexity --kl-divergence`: UD-IQ2_XXS (bartowski / unsloth) と UD-IQ3_XXS を **UD-Q4_K_XL 基準** (`kld_q4kxl.bin` と同じ 16 chunk) で。既知の錨: IQ4_XS は KLD 0.096 / top-1 90.3%。加えて DS4-IQ2 の公表値 (bf16 基準 top-1 90.3%) | PC (GGUF は E: に有る / 追加 DL 45〜76 GB) | 1 本 30 分、DL 込み 1 日 | **IQ3_XXS の top-1 が Q4 基準で 85% 以上、実タスク (`PLAN-quant-eval-pi.md`) で完走率が落ちなければ「3 bpw 級で行く」**。affine RTN はこれより悪いので、通っても §4 D8 で再測 |
| **D3** | M3 Pro の SSD で expert 1 個分のランダム読みは何 GB/s・何 ms か | 手元の大きなファイル (Jundot のシャードなど、D5 で取るもの) に対し、`F_NOCACHE` で **1.49 MB / 1.84 MB / 2.76 MB** のレコードをランダム位置から pread。QD 1 / 10 / 20、各 60 秒以上。p50 / p99 レイテンシと GB/s、`fs_usage` で実 I/O を確認。~~ds4-metal で DS4-IQ2 を流す~~ は不可 (§2-4) | **M3 Pro** | 半日 (測る対象のファイルが有れば) | 試算の **SSD ミス項を実値に置換**。D1 の hit と合わせて「1 トークンのミス数 × QD 10 の 1 回あたり時間」を出し、既知の DRAM 項 (dense 3.0〜3.5 GB / 134 GB/s) と足して M3 Pro の decode を合成する。M6 は帯域比 (170/150) で換算、SSD 比は未確認なので 1 |
| **D4** | 参照器 (float32、層ストリーミング) を qwen4exp にできるか | `Scripts/qwen35/reference_forward.py` を拡張: hyper-connection、PLE (ハッシュ + 16 行 gather)、GDN 48 V head、512 top-10、MTP。突き合わせ先は **llama.cpp の CPU 実行 (`-ngl 0`、決定的)** を PC で回して logits を落とす (GPU 版は top_k が非決定的、`RESULTS-pc-qsa-gather.md` §4)。文脈 ≤ 2048 で indexer を回避 | M3 Pro (CPU) + PC | 3〜5 日 | 算式の一致 (相対 1e-6、top-1 100%)。fixtures が出る。**PLE ハッシュの行番号が llama.cpp と一致**することが最初の関門 |
| **D5** | oQ4e-mtp を本ランタイムの形式に落とせるか | Jundot を `hf download` (hf_transfer、約 30 分)。`audit_checkpoint.py` でメニュー外を数える → 上流 bf16 を range 取得 → 8-bit g64 打ち直し (`checkpoint_io.py` / `mlx_quant.py` 流) → `TsugumiRepack` (PLE 128 shard は repack の対象外にして別ファイルに) → `--verify-install` | M3 Pro (CPU) | 2〜3 日 | install サイズ (見込み 78 GB = expert 68 + dense 8 + MTP 2.5、PLE 29 GB は別)、`expertStride == 2,764,800`、赤リスト 0 本 |
| **D6** | カーネルと結線 | k=10 の phase2、hidden 2560 / ff 640 の幾何、GDN 48 head、hyper-connection、PLE 加算 → `TsugumiKernelCheck` で D4 の fixtures に対し検証 → `QwenForwardRunner` の Flash-Next 分岐 → greedy が参照と全一致 | M3 Pro | 1〜2 週 | Ornith の Phase 2〜3 と同じ出口 ([04](../qwen35moe/04-PHASES.md)) |
| **D7** | 試算の DRAM / host 項の実値 | D6 の後、M3 Pro で stage profile: dense 読みの ms/tok (3.0〜3.5 GB を 134 GB/s で 22〜26 ms のはず)、residency commit (480 タッチ/tok)、SSD ミスの GB/s (2.76 MB × QD 10)、decode tok/s、prefill の ms/tok と 1 チャンクの全走査バイト | M3 Pro | D6 の後 1〜2 日 | [M6 文書 §3](M6_QWEN38_FLASH_NEXT.md) の DRAM / host 列を実値に置換。M6 は帯域比で換算 |
| **D8** | 2-bit / 3-bit affine の expert カーネル (D2 が通った場合だけ) | `moe_phase1` / `phase2` の int2 (g32) または int3 (g64) 版。ウェイトは Vontra oQ2 の routed expert (dense は D5 と同じ打ち直し) か、bf16 から自分で RTN。品質は D4 の参照器で NLL (oQ4e-g64 との差) | M3 Pro | 1 週 | **これが通って初めて「32 GB で 15〜18」の列が本ランタイムのものになる** |
| D9 | dense を 4-bit に落とす案 | D5 の打ち直しで dense を 8 → 4-bit g64 にした版を作り、D4 の参照器で NLL 比較 | M3 Pro (CPU) | 1〜2 日 | 通れば M6 の天井が 25 → 34 tok/s ([M6 文書 §3-1](M6_QWEN38_FLASH_NEXT.md)) |

**M3 Pro / PC では決まらないもの**: M6 の SSD 実速度 (2 MB ランダム pread、256 / 512 GB / 1 TB で違いうる)、M6 12 コア GPU の絶対性能と tensor ops、170 GB/s の実効、`apple10` の family 判定。すべて [M6_MAC_MINI_TARGETING.md §7](M6_MAC_MINI_TARGETING.md) の到着後リストにある。

---

## 5. 9/22 までの順序と判断

| 日 | やること | 並列 |
| --- | --- | --- |
| 1 | **D1** (シミュレーション、Mac 不要) | PC で **D2** の GGUF を DL して KLD を回し始める。M3 Pro で Jundot (106 GB) の DL を始める (空き 368 GiB)。DS4-IQ2 は取らない (§2-4) |
| 2 | **D3** M3 Pro で SSD のランダム読みを計測 (Jundot のシャードを対象に) | D2 の残り |
| 3 | **判断** (下) | D5 の監査と range 取得を始める |
| 4〜 | D4 → D5 → D6 → D7 (→ D8 / D9) | M6 到着後は M6 文書 §7 の 5 つと §5 の SSD プローブ |

**32 GB に変える条件 (3 つとも)**:

1. D1: 240 スロット/層で hit ≥ 0.85 (LRU + admission)。
2. D2: 3 bpw 級 (IQ3_XXS) の top-1 が Q4_K_XL 基準で 85% 以上、かつ実タスクで完走率が落ちない。
   → これが通らなければ本線は (a) oQ4e-g64 で、32 GB でも 10〜13 tok/s。変更する理由が無い。
3. D3: D1 の hit (240 スロット) と D3 の SSD 実測、既知の DRAM 項から合成した **M6 32 GB の decode が 15 tok/s 以上** (短文脈)。
   合成は「DRAM 項 (dense 3.0〜3.5 GB + expert ヒット分) × 150/170 + SSD ミス項 (1 トークンのミス数 = 10 × 48 × (1 − hit)、QD 10 の実測時間) × (M3 Pro の SSD / M6 の SSD、未確認なので 1 と置く)」。
   15 に届かなければ変更しない。
   ~~ds4-metal が M3 Pro 18 GB で 8 tok/s 以上~~ は測れないので条件から外した (§2-4)。実機の合成値との照合は D7 に回る。

3 つ揃わなければ **16 GB のまま**、Flash-Next は PC (23 tok/s) の仕事にしておく。
移植 (D4〜D7) は Ornith 系の資産 (GDN、MoE、MTP の器) がそのまま効く範囲なので、32 GB を見送っても M5 Pro 64 GB や将来機のために価値は残る。ただし優先度は下がる。

---

## 6. ディスクと機材

| 置き場 | 必要量 | 状態 |
| --- | ---: | --- |
| M3 Pro 内蔵 (空き 368 GiB) | Jundot 106 + bf16 range 取得 6〜8 + moepack 78 + PLE 29 = **約 220 GB** | 入る。余裕 140 GB。DS4-IQ2 (77 GB) は取らない (§2-4) |
| PC C: (空き 509 GB) / E: (17 TB) | GGUF IQ2_XXS 45〜50 GB、IQ3_XXS 76 GiB (E: に無ければ DL) | 余裕あり |
| llm-server | `kld_q4kxl.bin` (Q4_K_XL 基準 logits) | 有る (`/root/qfn-bench/logs/`)。PC で取り直してもよい (30 分) |

---

## 7. 未確認リスト

1. Jundot oQ4e-mtp の imatrix 校正セット (`oq_imatrix_report.json` は同梱、中身未読) と、打ち直し対象の正確な本数・バイト
2. Vontra oQ2-MTP の実サイズと dense 側の bit 配分の詳細 (config は読んだが本数は概数)
3. ~~ds4-metal の Flash-Next 対応コミットと、18 GB での動作可否~~ → **解決** (§2-4): `origin/qwen3.8-flash-next` = `3030554` に入っている。Qwen 経路は SSD ストリーミング無しで全常駐のため 18 GB では動かない (コードからの導出、起動は試していない)
4. `qwen_delta_rule` の V head 数がパラメータ化されているか (48 head)
5. hyper-connection の正確な算式 (上流 `modeling_qwen4_exp` 未読)
6. 上流 bf16 の総サイズ (約 360 GB と推定) と range 取得の速度 (Ornith のときは単一ストリーム 11.3 MB/s、`hf download` で 60 MB/s)
7. affine 2-bit RTN と IQ2_XXS (imatrix) の品質差の大きさ

---

## 出典

- HF: `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` (config.json、ファイル一覧)、`Vontra/Qwen3.8-Flash-Next-MLX-oQ2-MTP` (config.json)、
  `mlx-community/Qwen3.8-Flash-Next-4bit` (モデルカード: mlx-vlm `d1bd74ed`、g32)、`ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2` (モデルカード、`qwen38-q2down.md`、`…-MTP.gguf.json` の tensor バイト数、revision `b8b20398`)、
  `Qwen/Qwen3.8-Flash-Next` (モデルカード)
- GitHub: `ivanfioravanti/ds4-metal` (README)。`origin/qwen3.8-flash-next` `3030554` の `docs/QWEN38_FLASH_NEXT.md`、`docs/SSD_STREAMING.md`、
  `ds4.c` (`generate_metal_graph_raw_swa` → `generate_qwen4_metal_argmax`、`--ple` の mmap)、`ds4_metal.m` (`ds4_gpu_add_model_view_range`、`ds4_gpu_model_residency_request_views`)
- 手元 (M3 Pro): `MTLDevice.maxBufferLength` / `recommendedMaxWorkingSetSize` を Swift で読んだ値 (2026-09-13)
- 手元: [qwen35moe/02](../qwen35moe/02-CHECKPOINTS.md) (oQ の仕組み、Ornith の供給源)、[11](../qwen35moe/11-OQ4E-G64-REBUILD.md) (打ち直し手順)、
  [14](../qwen35moe/14-REFERENCE.md) (参照器)、[18](../qwen35moe/18-MIXED-BITS.md) (混在ビット幅)、`Sources/Tsugumi/Infrastructure/ModelIO/ModelTypes.swift` (`ArchConfig.ornith1_5_35B_A3B`)、
  `Sources/Tsugumi/Kernels/MoE/MoE.swift` (`moe_phase2_down_reduce_k8`)、`omlx/omlx/oq.py:377` (`gs()`)
- リモート (`192.168.0.199` `C:\LLM`): `qfn-bench/RESULTS-e2.md` (トレースとシミュレータ)、`PLAN-article-and-3rd-gpu.md` §2-1 (IQ3_XXS 内訳)、`PLAN-quant-eval-pi.md`、`RESULTS-pc-qsa-gather.md` §4 (GPU top_k の非決定性)
