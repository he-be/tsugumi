# 06. Bonsai 2 27B の MTP を Mac で測る — 受理率は同じ、速度は 0.74 倍

実測: 2026-09-20、M3 Pro 18 GB、macOS 15、`iogpu.wired_limit_mb` = 14336。
表記は [qwen38/01](../qwen38/01-Q2-FIRST-LIGHT.md) と同じ (実測 / 導出 / 未確認)。
**速度とメモリの数字は断りがなければ 1 回の実行**で、解釈は書かない。

[05](05-BONSAI-PQ2-0.md) §0-1 に「MTP ヘッドが無い」と書いたが、これは
`prism-ml/Ternary-Bonsai-2-27B-gguf` の 3 本についての記述で、Bonsai 2 27B 全体には当てはまらない。
有志が MTP を足した一本があり、その報告 (CUDA) が
[prism-ml の discussion #23](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/discussions/23) に出ている。

**この後 [07](07-PQ2-0-SMALL-BATCH-KERNEL.md) で §6 の原因 (行列積のカーネル選択) が分かり、
§0-5・§0-6 と §7-4 は変わった。速度の結論は 07 を見ること。受理率 (§4) はそのまま。**

## 0. 結論

1. **MTP 入りの一本は存在する。**`ProCreations/Ternary-Bonsai-2-27B-MTP` (2026-09-19) が、公式 PQ2_0 の
   851 テンソルをバイト不変のまま `blk.64.*` (15 本、Q8_0) を足して 1 ファイルにしている。
   ヘッドの出所は **Qwen3.8-27B 公式の MTP 15 テンソル**で、配布元の `reports/donor-provenance.json` は
   公式リビジョンと 15 本とも sha256 一致と書いている。それを Bonsai の本体に合わせて蒸留し直したもの (§1)。
2. **取得は 467 MB で足りる。**本体は手元にあるので、遠隔からはヘッダ 11.1 MB と末尾の MTP 451.3 MB だけ
   Range で取って結合した。結果は公開物と **sha256 一致** (`3cb3f005…`)。
   本体 851 本が公式 PQ2_0 とバイト同一であることも、これで同時に確かめたことになる (§2)。
3. **ビルド済みの公式 prism は Mac でも MTP を拒否する。**
   `Hadamard-latent table 'token_embd.weight' is read without the inverse transform` (b10709、§3)。
   幹の `build_inp_embd` と同じ逆変換を `graph_mtp` に足す 12 行を当てて Metal でビルドすると動く。
4. **受理率は CUDA の報告とほぼ同じ。**12 本の prompt で 67.8% (1890/2786)。
   報告値は 68.1% (803/1180) で、prompt ごとの並びも同じ (自由文 47〜57%、コードと定型 89〜91%) (§4)。
5. **それでも Mac では遅い。**decode は **0.74 倍** (中央値 13.4 → 9.8 tok/s、`--spec-draft-n-max 2`)。
   n_max 1 でも 0.77 倍。文脈を 32K から 8K に下げてスワップを消しても比は変わらない (§4、§5)。
   → [07](07-PQ2-0-SMALL-BATCH-KERNEL.md) §4 で 1.30 倍になった (n_max 1)。**この行はもう成り立たない。**
6. **理由は受理率ではなくバッチが伸びないこと。**`llama-batched-bench` で B=1 13.67、B=2 11.21、
   B=3 12.34、B=4 12.21 tok/s (合計)。**k トークンをまとめて検証しても k 倍の時間がかかる**ので、
   受理率がいくら高くても投機は勝てない。prefill も 92〜95 tok/s しか出ていない (§6)。
   → 伸びなかったのは 2〜8 列のとき遅いカーネル (`mul_mv_ext`) が選ばれていたためで、
   計算律速ではなかった ([07](07-PQ2-0-SMALL-BATCH-KERNEL.md) §1・§3)。prefill の方はそのまま。
7. したがって MTP の採否はヘッドの質の問題ではなく、**この Mac の PQ2_0 カーネルの問題**。
   先に直すのは投機の設定ではなくカーネル側 (§7)。

## 1. MTP 入りの一本の中身 (公開物の記述)

| 項目 | 値 |
| --- | --- |
| ファイル | `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` 7,657,489,728 B (公式 PQ2_0 + 451,320,800 B) |
| 本体 | 公式 `prism-ml/…-PQ2_0.gguf` の 851 テンソル (改変なしと明記、§2 で確認) |
| MTP | `blk.64.*` 15 本。行列は Q8_0、norm は F32。約 4.247 億パラメータ |
| ヘッドの出所 | `Qwen/Qwen3.8-27B` の `mtp.*` 15 本 (BF16)。配布元は公式リビジョンと sha256 一致と記載 |
| 合わせ直し | 本体の最終隠れ状態を教師に forward KL + 0.1 × CE。Stage 1 = UltraChat 256 会話、Stage 2 = 本体が生成した 48 本の続きを 4 倍に重み付け |
| 配布元の比較 | 受理率は公式ヘッドそのままが 0.559、蒸留後が 0.605 (RTX PRO 6000、temp 1 / top_p 0.95 / top_k 20) |

**別の本体に他所のヘッドを移して合わせ直す**手で、Ornith で本線のヘッドをドナーにした形
([qwen35moe/42](../qwen35moe/42-FREETOKEN-IDEAS.md)) と同じ。

## 2. Range で 467 MB だけ取る (実測)

`Scripts/qwen38_27b/bonsai_mtp_fetch.py`。手順は 3 つ:

1. 遠隔のヘッダ (先頭 16 MB) を Range で取り、GGUF の KV とテンソル情報を読む。
   **本体 851 本は名前・形・型・オフセットまで手元の PQ2_0 と完全一致**、KV の差は
   `qwen35.block_count` 64 → 65、`qwen35.nextn_predict_layers` 1 の追加、`general.name` だけ。
2. MTP 15 本は末尾の連続領域 (データ領域の 7,195,047,936 B 以降 = 451,319,808 B) にある。ここだけ取る。
3. 遠隔のヘッダ 11,121,984 B + 手元の本体のデータ領域 + 取った末尾、の順に書き出す。

| 見たもの | 値 |
| --- | --- |
| ヘッダ 16 MB の取得 | 3.4 秒 |
| 末尾 451 MB の取得 | 45 秒 (約 10 MB/s) |
| 結合 (手元 7.2 GB の読み直しを含む) | 5 秒 |
| 組み上がった 7.66 GB の sha256 | `3cb3f0056d2e34ee44245a64396004a21f8492573d6ce1266ec4b7222c131dd4` = 公開の `SHA256SUMS` と一致 |

## 3. ランタイム (実測)

ビルド済みの `llama-prism-b10709-9a9394a-bin-macos-arm64` (05 §5 で使ったもの) は、このファイルを
MTP 付きで起動すると落ちる:

```
I common_speculative_init_result: creating MTP draft context against the target model '…-MTP-Q8_0.gguf'
W llama_verify_hadamard_graph: latent lookup 'mtp_tok_embd-64' consumed by op=RMS_NORM name='norm-64' src0 hint=0
E llama_init_from_model: failed to initialize the context: Hadamard-latent table 'token_embd.weight' is read without the inverse transform
E common_speculative_init_result: failed to create MTP context
```

`src/models/qwen35.cpp::graph_mtp` が `ggml_get_rows` した埋め込みを回った基底のまま使っているため。
幹の `llm_graph_context::build_inp_embd` (`src/llama-graph.cpp:2424-2436`) は同じ場所で
`h = s ⊙ (H z)` に戻しているので、それと同じ 12 行を `graph_mtp` に足した (05 §3 に書いた `token_embd` の扱いと同じ)。
これは上流に出ている [PrismML-Eng/llama.cpp#205](https://github.com/PrismML-Eng/llama.cpp/pull/205) と同じ内容で、
配布元の `runtime/bonsai-mtp-embedding.patch` とも同じ。**当てたのは公式の `prism-b10709-9a9394a` を clone した木**で、
配布元のソース一式は使っていない。

```
git clone --depth 1 --branch prism-b10709-9a9394a https://github.com/PrismML-Eng/llama.cpp src-b10709
# src/models/qwen35.cpp の ggml_get_rows 直後に 12 行
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server -j 6
```

起動すると `creating MTP draft context against the target model` が出て、`-md` は要らない
(draft が本体の `token_embd` / `output` を共有する。別ファイルの drafter にすると語彙の複製で割に合わない)。

## 4. A/B (実測、12 + 2 本の prompt を 2 回ずつ)

`Scripts/qwen38_27b/bonsai_mtp_ab.sh`。`-ngl 99 -fa on -c 32768 -np 1`、
`POST /completion` に `temperature 0, top_k 1, seed 7, n_predict 128, cache_prompt false`。
順番は none → mtp → mtp → none で、ブロックの間に 20 秒空けた。
prompt は discussion #23 の 12 本をそのまま使い (比較のため)、日本語を 2 本足した。

| prompt | none 1 回目 | none 2 回目 | MTP 1 回目 | MTP 2 回目 | 比 | 受理率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 reasoning | 13.41 | 13.49 | 9.65 | 9.68 | 0.719 | 66.1% (144/218) |
| R2 reasoning | 13.08 | 13.19 | 8.12 | 8.12 | 0.618 | 46.9% (122/260) |
| R3 reasoning | 12.52 | 13.45 | 8.94 | 8.93 | 0.688 | 56.8% (134/236) |
| C1 code | 12.85 | 13.54 | 10.18 | 10.17 | 0.771 | 71.8% (148/206) |
| C2 code | 12.90 | 13.50 | 11.78 | 11.77 | 0.892 | 91.1% (164/180) |
| C3 code | 12.53 | 13.42 | 9.38 | 9.41 | 0.724 | 62.5% (140/224) |
| M1 math | 12.48 | 13.42 | 10.15 | 10.19 | 0.785 | 72.1% (150/208) |
| M2 math | 12.61 | 13.38 | 9.41 | 9.42 | 0.724 | 62.5% (140/224) |
| F1 format | 12.84 | 13.41 | 11.69 | 11.71 | 0.892 | 90.0% (162/180) |
| F2 format | 13.24 | 13.40 | 11.58 | 11.59 | 0.870 | 89.0% (162/182) |
| Z1 chinese | 12.95 | 13.40 | 9.73 | 9.77 | 0.740 | 66.7% (144/216) |
| J1 日本語の文 | 13.40 | 13.27 | 8.30 | 8.32 | 0.623 | 49.6% (126/254) |
| J2 日本語の定型 | 13.40 | 13.39 | 10.62 | 10.65 | 0.794 | 77.8% (154/198) |
| **中央値** | | | | | **0.740** | **67.8% (1890/2786)** |

- Z2 (中国語) は両方の腕で 1 トークンで停止したので除外した (`/completion` を chat template 無しで叩く形の
  ためで、discussion #23 でも同じ 1 本が同じ理由で外されている)。
- **受理率は CUDA の報告とほぼ一致**: 報告 68.1% に対し 67.8%。prompt ごとに見ても R1 66.1% / 66.1%、
  R2 46.9% / 46.9%、C2 91.1% / 94.3%、F1 90.0% / 90.0% と並びが同じ。**ヘッドの当たり方は機材に依らない。**
- 速度だけが逆で、報告は 1.34 倍、ここでは 0.74 倍。
- スワップは MTP の腕で増えた (none +69,220 / +62,520 ページ、MTP +192,208 / +121,696)。
  RSS は none 9.10 GB、MTP 10.00〜10.05 GB。これが原因かどうかは §5 で切り分けた。

## 5. n_max と文脈長で切り分ける (実測、各 1 回)

`-c 8192` に下げるとスワップはほぼ消える (none +0、n_max 1 +25,576、n_max 2 +0 ページ)。
それでも比は変わらない。

| prompt | none | n_max 1 | 比 | 受理率 | n_max 2 | 比 | 受理率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 reasoning | 13.53 | 10.42 | 0.770 | 81.4% | 9.67 | 0.714 | 66.1% |
| R2 reasoning | 13.51 | 9.55 | 0.707 | 65.8% | 8.11 | 0.600 | 46.9% |
| R3 reasoning | 13.49 | 9.97 | 0.739 | 74.0% | 8.93 | 0.662 | 56.8% |
| C1 code | 13.42 | 10.50 | 0.782 | 82.6% | 10.17 | 0.758 | 71.8% |
| C2 code | 13.40 | 11.34 | 0.846 | 96.9% | 11.76 | 0.877 | 91.1% |
| C3 code | 13.42 | 10.37 | 0.772 | 80.0% | 9.41 | 0.701 | 62.5% |
| M1 math | 13.46 | 10.57 | 0.785 | 84.1% | 10.18 | 0.757 | 72.1% |
| M2 math | 13.45 | 10.14 | 0.754 | 77.5% | 9.41 | 0.700 | 62.5% |
| F1 format | 13.46 | 11.10 | 0.825 | 93.8% | 11.69 | 0.869 | 90.0% |
| F2 format | 13.46 | 11.05 | 0.821 | 93.8% | 11.57 | 0.860 | 89.0% |
| Z1 chinese | 13.46 | 10.39 | 0.772 | 82.6% | 9.75 | 0.725 | 66.7% |
| J1 日本語の文 | 13.44 | 9.34 | 0.694 | 63.6% | 8.31 | 0.618 | 49.6% |
| J2 日本語の定型 | 13.42 | 10.82 | 0.807 | 88.1% | 10.64 | 0.793 | 77.8% |
| **中央値** | 13.46 | 10.42 | **0.772** | 81.3% | 9.75 | **0.725** | 67.8% |

1 ステップ (草稿 + 検証を 1 回ずつ) にかかる時間を、受理率から求めた 1 ステップあたりの
トークン数で割り戻すと (導出):

| | 1 ステップ | 進むトークン |
| --- | ---: | ---: |
| 投機なし | 74 ms | 1 |
| n_max 1 (草稿 1 + 検証 2) | 174 ms | 約 1.8 |
| n_max 2 (草稿 2 + 検証 3) | 240 ms | 約 2.4 |

**1 トークン増えるごとにほぼ 66〜74 ms ずつ増えている。**MTP ヘッド自体 (1 層 + 共有の出力射影) は
重みで 0.77 GB 程度なので、この増え方はヘッドの費用では説明できない。

## 6. バッチが伸びない (実測、1 回)

`llama-batched-bench -m …-PQ2_0.gguf -ngl 99 -fa on -c 8192 -npp 64 -ntg 64 -npl 1,2,3,4` (MTP 無しの本体):

| 並列 B | prefill tok/s | decode tok/s (合計) | decode tok/s (1 本あたり) |
| ---: | ---: | ---: | ---: |
| 1 | 92.74 | 13.67 | 13.67 |
| 2 | 91.25 | 11.21 | 5.61 |
| 3 | 95.08 | 12.34 | 4.11 |
| 4 | 95.52 | 12.21 | 3.05 |

**合計スループットが B で増えない。**帯域で律速しているなら B=4 は 4 倍近くになるはずで、実際は 0.9 倍。
つまり decode は**計算で律速**しており、k トークンの検証は k 回の decode とほぼ同じ時間がかかる。
投機の得は「検証がまとめてできる」ことに依っているので、この状態では受理率が 100% に近くても勝てない。
prefill が 92〜95 tok/s (decode の 7 倍) しか出ていないのも同じ側の話。

カーネルは在る (`kernel_mul_mv_pq2_0_f32`、`kernel_mul_mm_pq2_0_f32`、小バッチ用の
`kernel_mul_mv_ext_pq2_0_f32_r1_{2..5}`)。小バッチ経路は `ne11` が 2〜8 のときに選ばれる条件に
`GGML_TYPE_PQ2_0` も入っている (`ggml-metal-ops.cpp:2802`)。**選ばれた上でこの伸びない**ということになる。
選択の実際と、`N_R0_PQ2_0 = 8` / `N_SG_PQ2_0 = 2` (`ggml-metal-impl.h:35`) が M3 Pro で妥当かは未確認。

## 7. 次にやること

1. **Tsugumi の結線と一緒に測る。**ここまでは `/completion` を直接叩いた数字で、アプリの 1 ターン
   (ツールループ) では測っていない。05 §6 の `TsugumiToolLoopCheck` の経路で、MTP 有無を同じ会話で比べる。
2. **PTQ1_0 を測る。**ユーザーの言によれば **PQ2_0 と品質は同じで、パッキングだけが違う**。
   discussion #23 の CUDA では「PQ2_0 が prefill、PTQ1_0 が decode で勝つ」と書かれている。
   §6 が計算律速を示しているので、Mac ではパッキングの違いがそのまま効く可能性がある。まだ落としていない。
3. **Metal カーネルをこの機械に合わせる。**§6 の伸びない小バッチ経路が先。
   → [07](07-PQ2-0-SMALL-BATCH-KERNEL.md)。`N_R0_PQ2_0` / `N_SG_PQ2_0` と tuning の表はまだ手つかず。
   `N_R0_PQ2_0` / `N_SG_PQ2_0` の値、`mul_mv_ext` が実際に選ばれているか、
   このフォークが持っている `ggml-metal-tuning` の表を M3 Pro で作れるか。
   **ここが直るまで、投機 (MTP・ngram・dspark) の可否は判断できない。**
4. MTP そのものは**既定 off のまま**にする。§6 が直るまで測り直さない。
   → §6 は [07](07-PQ2-0-SMALL-BATCH-KERNEL.md) §2 で直り、測り直した (07 §4・§5)。
