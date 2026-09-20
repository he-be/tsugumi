# 05. Ternary Bonsai 2 27B (PQ2_0) を読む — 形・CPU 参照・llama.cpp でのエージェント 1 巡

実測: 2026-09-20、M3 Pro 18 GB、macOS 15、`iogpu.wired_limit_mb` = 14336。
表記は [qwen38/01](../qwen38/01-Q2-FIRST-LIGHT.md) と同じ (実測 / 導出 / 未確認)。
**速度とメモリの数字は断りがなければ 1 回の実行**で、解釈は書かない。

対象は `prism-ml/Ternary-Bonsai-2-27B-gguf` (2026-09-17)。土台は [01](01-FEASIBILITY.md)〜[04](04-RUNNER-REFERENCE-MATCH.md) と同じ
Qwen3.8-27B (GGUF の `qwen35`) だが、量子化と重みの基底が違う。

## 0. 結論

1. **MTP ヘッドが無い。**PQ2_0・Q2_0 (dev)・F16 の 3 本とも `qwen35.block_count = 64` で、`nextn_predict_layers` キーも
   `blk.64` (`nextn.eh_proj`・`enorm`・`hnorm`・`shared_head_norm`) も無い。01 §1 で数えた MTP の 15 本が落ちている (§1)。
   Prism 側の投機は MTP ではなく別ファイルの drafter (dspark) で、公開されているのは前世代向け。
   彼らの記載では Apple Silicon は非推奨 (M5 Max で code / math だけ約 1.2 倍、chat / reasoning は遅くなる)。
   **ただしこれは prism-ml の 3 本についての記述**で、有志が Qwen3.8-27B のヘッドを足した一本が別にある
   ([06](06-BONSAI-MTP-ON-MAC.md))。Mac では受理率は同じで速度は 0.74 倍だった。
2. **PQ2_0 は 128 重みごとに fp16 の scale 1 個 + 2 bit の三値。逆量子化は `(q - 1) * d` の 1 行**で、03 の混在 10 型より簡単 (§2)。
   自前の復号は、同じリポジトリの F16 (折り込み済みの原本) と **相対誤差 0 で一致**した (§2-1)。
3. **重みは Hadamard で回った基底に折り込んである。**runtime は行列積ごとに活性を `x' = H · (s ⊙ x)` に変換する必要がある
   (H は 1024 次元の正規直交 Walsh-Hadamard、s は入力幅ごとの ±1)。`token_embd` だけは逆で、行を引いた後に戻す (§3)。
4. **CPU 参照器が動いた。**「The capital of France is」の位置 1 で ` of`、位置 3 で ` is` を当て、4 位置の平均 NLL は 2.95
   (同じ文・同じ参照器で ISTA IQ3_S は 3.376) (§4)。
5. **prism のフォークの llama.cpp (macOS arm64 のビルド済み) でそのまま動く。**decode 13.3 tok/s、重みだけで RSS 9.12 GB (§5)。
6. **アプリのツールループ (Offline 5 問) が 2 セットとも全ラウンド通った。**checks は 10 / 10 ok。
   正解 note との一致は set1 が 5 / 5、set2 が 4 / 5 (N4 が逆の結論) (§6)。
   **Online は 6 会話 × 3 ターンのうち 5 ターンで打ち切った** (1 会話に約 8 分かかり、6 会話で 1 時間弱の見込みだったため)。
   Serper は 0 回。5 ターンとも answer と error は通り、`online` は続きの 2 ターンで不成立 (§6-1)。
7. **Vision も動く。**`logicool™`、SK-II のボトルの `SK-II` / `FACIAL TREATMENT ESSENCE` / `WORLDWIDE PARTNER` / 五輪マークを読んだ。
   画像 1 枚で Swapouts +19,024 ページ (297 MB)、システムの空きは 11% まで下がった (§7)。
8. **根拠のない日本語の知識は落ちている。**「高市早苗」の読みが「たかしまさえ」「たかみ はやね」、「青森県」の振り仮名が「あおしま」。
   公開されている 14 本のベンチマークは英語なので、この面は覆われていない (§8)。
9. **一方、[qwen38/36](../qwen38/36-Q2_0-JAPANESE-BROKEN.md) が ISTA の Q2_0 を打ち切った理由 (日本語の表記の崩れ) は出ていない。**
   §6・§6-1 の回答 15 本に、日本語に無い漢字 0・中国語コンマ 0 (§8-1)。
10. **この機体では遅い。**Offline 1 問が 40〜107 秒、Online 1 ターンが 54〜237 秒。15K 文脈で decode 11.07 tok/s (§5、§6-1)。

## 1. 形 (実測、GGUF ヘッダ)

`Scripts/qwen38_27b/gguf_header.py` に先頭 24 MB の Range 取得を食わせた。

| ファイル | 大きさ | `block_count` | `nextn_predict_layers` | `blk.64` | テンソル数 |
| --- | ---: | ---: | --- | --- | ---: |
| `…-PQ2_0.gguf` | 7,206,168,928 | 64 | 無し | 無し | 851 |
| `…-PTQ1_0.gguf` | 5,946,648,928 | (未確認) | — | — | — |
| `…-F16.gguf` | 53,808,408,928 | 64 | 無し | 無し | — |
| `…-Q2_0-prism-fork-required.gguf` (別リポジトリ `-dev`) | 7,626,008,928 | 64 | 無し | 無し | — |
| `…-mmproj-Q8_0.gguf` | 629,246,976 | — | — | — | 334 |

- hparams は 01 §1 と同じ (`embedding_length` 5,120 / `feed_forward_length` 17,408 / `full_attention_interval` 4 /
  head 24・KV 4・key 256 / `ssm.*` は inner 6,144・state 128・group 16・dt_rank 48・conv 4 / rope 64 次元・1e7 / 語彙 248,320)。
  違うのは `block_count` が 65 → 64 (MTP の分) と、`general.sampling.*` に thinking 有効の値だけが入っている点 (01 §4 と同じ)。
- PQ2_0 の型は 402 本が 142 (PQ2_0)、96 本が 30 (BF16、`ssm_alpha` / `ssm_beta`)、353 本が 0 (F32)。
  `token_embd` と `output` も 142。「低ビットの外に逃がしたテンソル」は `ssm_*` とノルムだけ。
- mmproj は `general.architecture = clip`、`clip.projector_type = qwen3vl_merger`、27 層・hidden 1,152・FFN 4,304・
  patch 16・image_size 768・`spatial_merge_size` 2。`is_deepstack_layers` は 27 本とも false。
  `mm.0` [4608, 4608] と `mm.2` [4608, 5120] (4,608 = 1,152 × 2 × 2)。
  **既存の `Sources/Tsugumi/Vision/` は Gemma 4 の塔 (`vision_tower.patch_embedder.*`) なので、これとは別物。**

## 2. PQ2_0 の形 (出どころ: `PrismML-Eng/llama.cpp` の `prism-v7`、読むだけ)

`ggml-common.h`:

```c
#define QK_PQ2_0 128
typedef struct {
    ggml_half d;                 // scale
    uint8_t   qs[QK_PQ2_0 / 4];  // 2 bit x 128
} block_pq2_0;                   // 34 B / 128 重み = 2.125 bpw
```

`ggml-quants.c` の `dequantize_row_pq2_0` は、重み j について `byte = qs[j / 4]`、`q = (byte >> ((j % 4) * 2)) & 3`、
値は `(q - 1) * d` (00 = -1 / 01 = 0 / 10 = +1)。上位の 11 は使わない。

同じ 2 bit の符号で group 64 のものが upstream 互換の `Q2_0` (型 42、`-dev` リポジトリの 7.63 GB)、
base-3 で詰めたものが `PTQ1_0` (128 重みで 24 + 2 + 2 B)。**Prism の README は、素の llama.cpp が `Q2_0` を警告なしに読んで
出力が壊れる (活性側の変換が無いため) と書いている。**

### 2-1. 復号の確かめ (実測)

`Scripts/qwen38_27b/prism_gguf.py` の `dequant_pq2_0` を、F16 ファイルの同じテンソルと突き合わせた。
F16 は Range 取得で 1 本だけ (`blk.0.attn_qkv.weight`、5,120 × 10,240、104,857,600 B、offset 5,159,670,112)。

| 見たもの | 値 |
| --- | --- |
| 相関 | 1.0 |
| `|q - ref| / |ref|` の中央値 | 0.0 |
| 行ごとの余弦 (最初の 5 行) | 1.0 × 5 |

F16 ファイルは三値を fp16 で書いたもので、連続値の原本ではない。したがってこれは **復号のビット一致の確認**であって、
量子化誤差の測定ではない。

## 3. Hadamard (出どころ: 同じフォーク、読むだけ)

メタデータ (PQ2_0 の実測):

| キー | 値 |
| --- | --- |
| `prism.hadamard.version` | 1 |
| `prism.hadamard.block_size` | 1024 |
| `prism.hadamard.transform` | `normalized-sylvester-walsh-hadamard` |
| `prism.hadamard.axis` | `input-last-dimension` |
| `prism.hadamard.sign_mode` | `explicit` |
| `prism.hadamard.weight_names` | 401 本 |
| `prism.hadamard.sign_widths` / `sign_values` | [5120, 6144, 17408] / 28,672 個の ±1 (幅の順に連結) |
| `prism.hadamard.inverse_weight_names` | `token_embd.weight` |
| `prism.hadamard.gdn_v_grouped` | true |

- 行列は `llama-model.cpp` が作る `H[r][c] = (-1)^popcount(r & c) / sqrt(1024)`。対称で正規直交なので、逆変換も同じ行列。
  Metal 側は `misc.metal` の `kernel_fwht`(simd_shuffle_xor の蝶形) に置き換わる。
- 適用の順序は `llama-graph.cpp` の `build_lora_mm`: 折り込み済みの重み w について
  `x' = H · (s_幅 ⊙ perm(x))` を作ってから `w · x'`。同じ活性に複数の重みがかかる場合 (attn_qkv と attn_gate、ffn_gate と ffn_up、
  attn_q / k / v) は 1 回で済ませている。
- `token_embd` は逆向き: 行を引いた後に `h = s ⊙ (H z)` で元の基底に戻す。
- `gdn_v_grouped` は `ssm_out` の入力だけに効く並べ替え。`ne[0]` = 6,144、`n_v` = `ssm.time_step_rank` = 48、
  `n_k` = `ssm.group_count` = 16 なので `(hd, nk, rep) = (128, 16, 3)`、ggml の `[hd, nk, rep]` → `[hd, rep, nk]`。
- 折り込みの対象 (401 本): `output`、各層の `attn_qkv` / `attn_gate` / `ssm_out` (GDN 層)、
  `attn_q` / `attn_k` / `attn_v` / `attn_output` (全注意層)、`ffn_gate` / `ffn_up` / `ffn_down`。

1 トークンの decode で回す量は 64 層 × (5,120 + 6,144 + 5,120 + 17,408) ≒ 2.16M 要素 (導出)。

## 4. CPU 参照器 (実測、`Scripts/qwen38_27b/reference_forward.py --bonsai`)

02 の参照器をそのまま使う。変えたのは 3 点だけで、算式 (GDN・全注意・SwiGLU・残差) は 02 と同一:

1. GGUF の読み手を `prism_gguf.Reader` にした (gguf-py は型 142 を知らずヘッダで落ちる)。既知の型は gguf-py の表と
   逆量子化に回すので、ISTA の IQ3_S-mtp も同じ経路で読める。
2. `nextn_predict_layers` が無い場合を 0 にした。
3. 折り込み済みの重みの行列積の前と、`token_embd` を引いた後に §3 の変換を入れた。

| 文 | 位置 | 平均 NLL | 備考 |
| --- | ---: | ---: | --- |
| The capital of France is (Bonsai PQ2_0) | 4 | 2.951 | 位置 1 が ` of`、位置 3 が ` is` |
| 同 (ISTA IQ3_S-mtp、既存の `scratch/qwen38_27b/ref-france.log`) | 4 | 3.376 | 同じ位置で同じ top-1 |

1 回の通しは 19.2 秒 (64 層、5 位置、workers 8)、footprint の最大 0.55 GB、Swapouts +0。

読み手を差し替えても ISTA 側は変わっていない (実測): 同じ文を IQ3_S-mtp で流し直すと、位置ごとの NLL
(8.650 / 0.595 / 3.723 / 0.536)・top-1・平均 3.3760 が既存の `scratch/qwen38_27b/ref-france.log` と一致した。
既存の IQ / K のカーネルも壊していない (`TsugumiKernelCheck --q27-dense scratch/qwen38_27b/dense-fixture` の 52 本が PASS。
§10 でマクロに 1 ブロックの重み数を渡す形に変えたため)。

置いたもの (`scratch/bonsai27b/`): `ref-france.log` / `.logits` (5 + 6 トークン、11 位置)、
`ref-fuji-ok.log` / `ref-fuji.logits` (53 トークン、52 位置)、`ref-fuji-kvq8.*` (KV Q8_0)、
`ref-fuji-gdn-block.log`・`ref-fuji-gate-sigmoid.log` (負例)。

## 5. prism のフォークの llama.cpp (実測)

ビルド済みの `llama-prism-b10709-9a9394a-bin-macos-arm64.tar.gz` (11.5 MB) を展開して使った。**自前でビルドしていない。**

```
llama-server -m …/Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 -fa on -c 32768 -np 1 --jinja \
  --host 127.0.0.1 --port 8080
```

| 見たもの | 値 |
| --- | --- |
| 読み込み | 17 秒 |
| decode (36 トークンの prompt、116 トークン生成) | 13.3 tok/s |
| RSS (重みだけ) | 9.12 GB |
| decode (§6-1 の ja-news turn 2、文脈 16,837 トークン、1,492 トークン生成) | 11.07 tok/s |

Bonsai の README が載せている Apple の数字は M5 Max 47.0 / M5 Pro 28.7 / M4 Pro 18.0 tok/s (機材が違うので見込みには使わない)。

## 6. アプリのツールループ (実測、Offline 5 問 × 2 セット)

`TsugumiToolLoopCheck` は llama-swap を前提に `upstream/<id>/…` を叩くので、`/upstream/<id>/<path>` を
`/<path>` に送り直すだけの中継を 127.0.0.1:8081 に挟んだ (`Scripts` には入れていない。session の scratch 置き)。

```
.build/release/TsugumiToolLoopCheck --out DIR --model ~/LLM/Qwen3.8-Flash-Next-DS4-IQ2 \
  --conversations docs/experiments/knowledge-sources/questions-v2-stage2.json \
  --network offline --max-rounds 6 --thinking off --context 32768 --repeats 1 \
  --endpoint http://127.0.0.1:8081 --remote-model bonsai
```

| 問 | set1 ラウンド / 秒 / checks | set2 ラウンド / 秒 / checks | 正解 note との一致 |
| --- | --- | --- | --- |
| N1-pm | 3 / 43 / ok | 4 / 40 / ok | set1 ○ (第104代) ・ set2 ○ (第2次内閣まで) |
| N2-easternmost | 6 / 107 / ok | 4 / 52 / ok | ○ / ○ |
| N3-gasoline | 4 / 67 / ok | 4 / 73 / ok | ○ / ○ |
| N4-hiroshima | 4 / 74 / ok | 4 / 70 / ok | ○ (「現在は登れません」) / **×** (「基本的に登れます」) |
| N5-passport | 6 / 105 / ok | 3 / 45 / ok | ○ / ○ |

- checks (live / error / answer / progress) は 10 ターンとも ok。ツール呼び出しの構文が壊れた回は無い。
- `wikipedia_search` → `wikipedia_page` → 節を指定して読み直す、という順で進んでいる。
- N1 の set1 の回答は読みを「たかみ はやね」と書いた (§8)。

### 6-1. Online (実測、固定検索、5 ターンで打ち切り)

固定済みの検索結果が残っているのは [qwen38/48](../qwen38/48-ONLINE-EXTRACT-BY-E4B.md) の 6 会話分
(`Scripts/qwen38/tool_loop_conversations.json`、各 3 ターン) だけだった。
37 §4 が使う `first-fetch/questions.json` の 40 問分は、セッションの一時ディレクトリにあったので消えている。
store は `scratch/qwen38/extract48/web` を `scratch/bonsai27b/web` に複製して使った (元の実験の記録を書き足さないため)。

```
.build/release/TsugumiToolLoopCheck --out DIR --model ~/LLM/Qwen3.8-Flash-Next-DS4-IQ2 \
  --conversations Scripts/qwen38/tool_loop_conversations.json \
  --network online --max-rounds 6 --thinking off --context 32768 --repeats 1 \
  --web-store scratch/bonsai27b/web --pin-search --search-budget 0 \
  --endpoint http://127.0.0.1:8081 --remote-model bonsai
```

| 会話 | ターン | ラウンド | 秒 | checks | web |
| --- | ---: | ---: | ---: | --- | --- |
| ja-news | 1 | 7 | 190 | ok | replayed 4 / recorded 0 / pinned 1 |
| ja-news | 2 | 7 | 237 | ok | replayed 2 / recorded 0 / pinned 3 |
| ja-news | 3 | 4 | 64 | `online` 不成立 | replayed 2 / recorded 0 / pinned 1 |
| en-swift | 1 | 7 | 153 | ok | replayed 2 / recorded 0 / pinned 1 |
| en-swift | 2 | 3 | 54 | `online` 不成立 | replayed 1 / recorded 0 / pinned 1 |

- **`search budget 0/0`、`recorded 0`。Serper もページ取得も 1 回も外に出ていない。**
- ラウンドの形は 36 §1 と同じ: 強制 `web_search` → 強制 `fetch_page` → auto → `none` で回答。
  `structured_output_failure` は出ていない。
- `online` が不成立の 2 ターンは、続きの問い (「ここまでの内容を 5 行で要約して」「その機能のコード例を出して」) に対して
  新しく検索せず手元の文脈から答えた回。`answer` と `error` は通っている。**この検査は毎ターンの検索を求めるので、
  続きのターンでは元々通りにくい。1 セットずつなので、ここから腕の差は読まない。**
- 残り 4 会話 (ja-howto・en-url・ja-price・en-facts) と 2 会話の 3 ターン目は流していない。

## 7. Vision (実測)

`--mmproj …-mmproj-Q8_0.gguf --image-max-tokens 1024` を足して起動し、`/v1/chat/completions` に `image_url` で投げた。
画像は `~/Pictures/sample_imgs` (リポジトリ外)。

| 画像 | 聞いたこと | 結果 | 秒 / prompt トークン |
| --- | --- | --- | --- |
| `05_m.jpg` (Logicool のロゴ) | 何が写っているか・文字 | `logicool™` を正しく読んだ | 10.7 / 96 |
| `NO_FUSION_0804_001.png` (SK-II 2 本) | 商品名・読める文字 | `SK-II`・`FACIAL TREATMENT ESSENCE`・`WORLDWIDE PARTNER`・五輪マーク・`White Rabbit` を読んだ。右のボトルにも五輪マークがあると書いたが写っていない | 29.7 / 1042 |
| `Shiba_inu_taiki.jpg` (柴犬) | 犬種 | 柴犬と答えたが表記が「シュibaイ」に壊れ、「日本チワワ」を足した | 9.4 / 94 |

| 見たもの | 値 |
| --- | --- |
| RSS (mmproj 込み) | 9.73〜10.72 GB |
| 画像 1 枚での Swapouts 差分 | 19,024 ページ (297 MB) |
| システムの空き | 11% |

## 8. 日本語 (実測、5 問、素の chat)

| 聞いたこと | 答え | 正否 |
| --- | --- | --- |
| カタカナの外来語を 10 個 | コーヒー / テレビ / コンピュータ / レストラン / ホテル / **バース** / バス / ビル / ガソリン / アイスクリーム | 1 個がおかしい |
| 「高市早苗」の読み | たかしまさえ | × (たかいちさなえ) |
| 都道府県を北から 5 つ、振り仮名付き | 北海道 (ほっかいどう) / 青森県 (**あおしま**県) / 岩手県 / 宮城県 / 秋田県 | 1 個が × |
| メモリ階層を 3 行で | L1/L2/L3 キャッシュ → メインメモリ → 二次記憶 | ○ |
| 英訳 | The meeting tomorrow has been postponed. | ○ |

文としての日本語は崩れていない。外れるのは固有名詞の読みと振り仮名。
§6 のように資料が文脈にあるときは、set1・set2 の 10 ターンとも日本語の本文は素直だった。

### 8-1. [qwen38/36](../qwen38/36-Q2_0-JAPANESE-BROKEN.md) §2-2 の物差しで数える (実測)

36 は ISTA の Q2_0 を「日本語が壊れている」として打ち切った。同じ見方 (日本語の地の文に混ざったもの、URL とコードは除く) で
§6 の回答を数えた。**36 の数は手で数えた語、ここは機械で数えた文字**なので、列の意味が同じではない。

| | 中国語の語・簡体字 | 中国語コンマ | 地の文の英単語 | 回答の本数 |
| --- | ---: | ---: | --- | ---: |
| Bonsai 2 27B PQ2_0 (ここ、Offline set1) | 0 | 0 | 2 (`Wikipedia` のみ) | 5 |
| Bonsai 2 27B PQ2_0 (ここ、Offline set2) | 0 | 0 | 1 (`Wikipedia` のみ) | 5 |
| Bonsai 2 27B PQ2_0 (ここ、Online ja-news) | 0 | 0 | 小文字始まり 18 (内訳は URL の断片と製品名) | 3 |
| ISTA Q2_0 (36 §2-2) | 6 | 1 | 5 (taken / temporary / finance / Insurance / published) | 10 |
| unsloth UD-Q4_K_XL (36 §2-2) | 5 | 0 | 0 | 10 |

- 数え方: `Scripts` には入れていない (session の scratch 置き)。CJK 統合漢字のうち JIS X 0213 (`euc_jis_2004`) で
  符号化できない字を「日本語の字種に無い」とした。36 の実例のうち `观`・`时`・`电`・`进` はこれで拾えるが、
  `构`・`从`・`随` は JIS にあるので拾えない。**取りこぼす方向の数え方**で、0 は「1 つも無い」ことの上限ではない。
- 36 が挙げた崩れ方 (読点が全部 `，`、「地球上的生命を支える」のような差し替え) は、§6・§6-1 の 15 本には出ていない。
- 36 の「地の文の英単語」は taken / temporary / finance のような普通名詞の差し替えだった。Online の ja-news に出る
  `Flash` / `Claude` / `Gemini` / `OpenAI` は読んだ記事の固有名詞で、種類が違う。
- 数え方は `Scripts/qwen38_27b/script_mix.py`。
- **36 §2-1 (ツールを使わない 5 問) は、原文が残っていないのでここでは同じ問いを流せていない。**§8 の 5 問は別の問い。

## 9. 手元に置いたもの

| 置き場 | 中身 |
| --- | --- |
| `~/LLM/Ternary-Bonsai-2-27B-gguf/` | PQ2_0 (7.21 GB)、mmproj Q8_0 (0.63 GB)、mmproj BF16 (0.93 GB) |
| `~/LLM/prism-llamacpp/llama-prism-b10709-9a9394a/` | フォークのビルド済み macOS arm64 |
| `scratch/bonsai27b/` | §4 の参照ログと logits、`runs/offline-set1`・`offline-set2`・`runs/online-set1` (5 ターン)、`web/` (§6-1 の store の複製) |

F16 (53.8 GB) と PTQ1_0 (5.95 GB) は落としていない。

## 10. コード

- `Scripts/qwen38_27b/prism_gguf.py` (新規): 型 142 を読む GGUF リーダ、`dequant_pq2_0`、`fwht`、`Hadamard`、`gdn_v_perm`。
- `Scripts/qwen38_27b/reference_forward.py`: §4 の 3 点。`--bonsai` で GGUF を切り替える。
- `Scripts/qwen38_27b/script_mix.py` (新規): §8-1 の数え方。`turns.jsonl` を受け取る。
- `Sources/Tsugumi/Metal/Quant/ggml_iq.metal`: `block_pq2_0` と `ggml_deq_pq2_0`、`ggml_pq2_0_gemv` /
  `ggml_pq2_0_dequant_f32`。マクロに 1 ブロックの重み数を渡せるようにした (既存の 10 型は 256 のまま)。
- `Sources/Tsugumi/Infrastructure/ModelIO/GGUFFile.swift`: `pq2_0 = 142` (34 B / 128 重み)。
- `Sources/Tsugumi/Kernels/Qwen38/GGMLDenseGEMV.swift`: PQ2_0 の登録と、行幅の条件を 1 ブロックの重み数に合わせた。

**まだ無いもの:** Metal の Hadamard (FWHT と符号)、ランナーへの結線と参照一致、Qwen3-VL の塔、速度とメモリの測定。
`TsugumiKernelCheck` の PQ2_0 の検査も書いていない。`llama-swap` の経路を真似る中継も session の scratch 置きのまま。

## 11. 次に決めること

1. **移植を続けるか。**§6・§6-1 の通り道具としては立っており、36 が Q2_0 を落とした理由 (§8-1) も出ていない。
   一方で §8 の固有名詞と §10 の速さ (この機体で Offline 1 問 40〜107 秒) が残る。
2. 続ける場合の順序は 01 §5 と同じ: Metal の Hadamard → PQ2_0 の検査 → ランナーの参照一致 → wired と tok/s → Vision。
3. Online をきちんと測るなら、40 問分の固定検索を取り直す (Serper 40 回) か、48 の 6 会話で揃える (Serper 0 回) かを先に決める。
