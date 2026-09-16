# 28. LLKVApprox (後半層の KV/状態近似) を Qwen3.8 の prefill に入れる検討 — 公開されている証拠、うちのランナーへの形、最初に測る段

調査: 2026-09-16。対象は [PixelML/Qwen3.8-Flash-Next-KVA-Projector](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-KVA-Projector) と、その元の
[きしださんの LLKVApprox](https://nowokay.hatenablog.com/entry/2026/09/11/120001) ([デモ](https://kishida.github.io/webdemos/llkvapprox/)、
[engine.js](https://github.com/kishida/webdemos/blob/main/llkvapprox/engine.js))。実装はまだ何もしていない。**この文書の速度の数字は全部 導出 で、実測は 0 本。**

## 0. 結論

1. **仕組み**: prefill で層 0..23 だけを全トークンに流し、層 24..47 の「prefill が残すもの」(GDN 18 層の再帰状態、注意 6 層の K・V・インデクサ鍵) を
   境界の残差状態 (hc 4 本 × 2560 = 10240 次元) から小さな投影器で埋める。末尾 N トークン (推奨 256) だけ層 24..47 を正確に流し、decode は全層正確。
2. **投影器の中身は小さい**: `projector.safetensors` (385 MB bf16、192.5M パラメータ) は GDN 層の LoRA (r128) と注意層の MLP (隠れ 512) だけ。
   土台は「その層自身の in_proj を 4 本の残差の平均に掛ける」(own-weights block-average) で、これは重みを何も足さずに (zero-shot) うちのランナーで作れる。
3. **エンジンのコードは公開されていない**。PixelML の [PR #45](https://github.com/PixelML/club-170hx/pull/45) にはノートブックと receipts だけで、
   本体は `~/WIP/llkvapprox-flash-next/` (非公開 issue pixelml/pixelml#139)。transformers `qwen4_exp` + HF eager が前提で、どのみち Metal のランナーには移植できない。
   **組み込むなら自前実装**。きしださんの JS は密な Qwen3-8B 用 (hc・GDN・MoE・インデクサなし) で、形の参考にしかならない。
4. **品質の証拠は薄い** (§2)。Flash-Next 向けの投影器の品質は **6〜15 トークンのプロンプト 7 本 × 48 トークン**の greedy 一致 75.9% (suffix 1) しかなく、
   長いプロンプト (うちの運用点 16K/32K) の測定は無い。学習データも 16 本 × 2,048 トークン。作者自身が「診断であって task accuracy ではない」と書いている。
   27B 版では近似が実際に効いたプロンプト 8 本で 13〜16%。一方 Gemma 4 12B で 4〜6M トークン学習した別の人の実験は 600 問中 589 (素 597)。
   **品質は学習量で決まり、その学習が一番高い**。
5. **速度は 1.7 倍前後の話** (導出、§3)。PixelML の 2.2 倍は 2 枚の GPU に前半/後半を分けた HF eager (53 tok/s) の比で、うちの土俵には移せない
   (23 の ratio-vs-base の教訓)。うちの 32K は 336 s → 約 200 s、ツールの 1 ラウンド (2〜8K) は 20〜85 s → 12〜50 s の見込み。
6. **最初に測る段** (§5): 重みを何も落とさず、zero-shot の埋めと oracle (正確な値を同じ経路で埋める) の 2 モードをランナーに作り、
   既存の `--qwen38-prefill` 参照検査で配管を確かめてから 8K/32K の速度と、22 §4・25 の探針で品質を測る。HF の LoRA/MLP はその後に F16 で読む (§4-3)。

## 1. 公開されている事実

| 出所 | 何 | 数字 |
| --- | --- | --- |
| きしださん (Qwen3-8B、デモ) | 層 18 で切り、投影器は per-layer の norm + 線形 (k/v + bias) か小さな ctx transformer。最後の位置だけ後半層 | 341 トークンで prefill 半分。散文は通る、コードは目に見えて劣化 (作者) |
| PixelML 27B v1 (PR #42、1 枚) | 48 GDN + 16 FA、層 32 で切る、suffix 256 | 6,603 トークンで 1.71 倍 (oracle 2.01)。23 本の greedy 128 で 69.8% (zero-shot) / 70.8% (学習後)、**近似が効いた 8 本は 13.3 / 16.0%、全一致 0**。suffix 1 は 6.8 / 7.4% |
| PixelML Flash-Next v2 (PR #45、2 枚) | 層 24 で切る。GDN 18 層に qkv/b/a、FA 6 層に K (pre-RoPE)・V・index_k。学習 2,000 歩・0.88 h、16 × 2,048 トークン | oracle 2.19〜2.27 倍 (1K〜6.6K)。**学習後 suffix 1: 7 本 (6〜15 トークン) × 48 で 255/336 = 75.9%**。suffix 256 は全本が 256 未満なので正確経路 = 100% (情報なし) |
| 同、ridge の上限 | 24.5K トークンで閉形式の線形 | GDN qkv 0.949 / b 0.981 / a 0.976、FA k 0.62 / v 0.71 / index_k 0.63 (FA は own-weights + MLP が要る) |
| 同、oracle 検査 | 正確な値を埋めても最後の位置の logits は mean 0.68 / max 6.8 ずれ、top-5 重なり 3、教師強制 32 歩で argmax 28/32 | M=1 と M=T の GEMM の ulp 差でルータが反転 (作者の説明) |
| jun76 (Gemma 4 12B) | 層 36 で切る、線形と注意つき、学習 3.99〜5.99M トークン | 16K の時間比 0.73、600 問で 589 (素 597)、PPL 比 0.90、コード 1.05 |

投影器の構成 (`config.json` と safetensors のヘッダから):

| 層 | 土台 (重みなし) | 足すもの (HF のファイル) | 出力 |
| --- | --- | --- | --- |
| GDN 24,25,26,28,…,46 (18) | 自層の `attn_qkv` / `ssm_beta` / `ssm_alpha` を 4 本平均の 2560 に | LoRA down [128, 10240] → up [10240, 128] (qkv)、[48, 128] (b, a)、zero-init | qkv 10240、b 48、a 48 の「再帰の入力」。conv・ゲート・再帰は本物を回す |
| FA 27,31,35,39,43,47 (6) | 自層の `attn_k` / `attn_v` / `indexer.k_proj` を同じく | MLP down [512, 10240] → up [512, 512] (k, v)、[128, 512] (ik)、W2 zero-init | K 512 (Hkv 2 × D 256、pre-RoPE)、V 512、インデクサ生鍵 128 |

土台の own-weights の bf16 コピーは PixelML では 3.95 GB あるが、線形なので**うちでは量子化済みの自層の重みをそのまま使えば 0 バイト**。
HF の投影器は AWQ-INT4 の bf16 化した土台に対して学習されている。うちは IQ2_XXS / Q2_K + KV Q8_0 なので、LoRA/MLP の補正はずれる可能性がある (zero-shot の土台は自分の重みなのでずれない)。

## 2. 品質について分かっていること・いないこと

- 分かっている: 短いプロンプトの散文で 7 割前後の greedy 一致 (それも自由走行の一致で、作者は「教師強制のドリフトで見るべき」と言っている)。
- 分かっていない: **長いプロンプト**、**ツール呼び出しの JSON**、**コード**、**日本語**。うちの運用点 (16K/32K、ツールループ、英語主体) はどれも未測定。
- Q2 の重みの上に近似が乗る二重の損失も未測定。
- 学習し直すなら教師の目標は 1 トークン約 386 KB (bf16)。6M トークン級は 2 TB でオンザフライ生成が要り、うちのランナーで教師 1 本 95 tok/s (6M トークンで 17 時間の prefill) の上に、
  wired 13〜14 GB のランナーと 192M パラメータの学習を 18 GB で同居させられない。**学習は別の機械の話になる** (PC の GPU の有無は未確認)。

## 3. 速度の見積もり (導出、実測なし)

[08 §4](08-ROUTED-MIXED.md) のチャンク 2048 の内訳 (1 チャンク ≈ 21.5 s = pre wall ≈ 7.0 + route ≈ 6.3 + routed ≈ 6.8 + PLE 0.1〜2.2、8K の各 n=1):

| 項 | 今 | 近似後 | 根拠 |
| --- | ---: | ---: | --- |
| route + routed (48 層) | 13.1 s | ≈ 6.6 s | 層 24..47 の expert の読み・top-10・GEMM が末尾 N トークン分だけになる |
| pre-router (48 層) | ≈ 7.0 s | ≈ 3.5 + 1.2 s | 後半 24 層の hc・注意・GDN・共有 expert が消え、代わりに埋め (GDN 18 層の入力 GEMM ≈ 0.65 s、chunked 再帰 18 × 40 ms ≈ 0.4 s、FA の GEMM 3 本 × 6 層と LoRA/MLP ≈ 0.2 s) |
| PLE | 同じ | 同じ | 層 0 の入口 (01 §2) |
| 合計 | 21.5 s | **≈ 12.5〜13 s (1.65〜1.7 倍)** | |

32K は 336.5 s → 約 200 s。32K のチャンクは後半ほど遅い (18.8〜23.3 s、選択つき注意が伸びる) ので、FA 6 層が消える分は少し上乗せ。
ページキャッシュを叩く expert の読みが半分になるので Swapouts にも効く方向 (未測定)。
**それでも 32K の初回は 3 分台**で、「prefill が遅い」を根本から解く話ではなく、1.7 倍の話。prompt cache (17・18) と併用が前提。

## 4. うちのランナーへの形

### 4-1. 経路

`Qwen38Runner.forwardBody` の層ループ (`for il in 0..<nTrunk`) を 3 段に分ける。チャンク T トークン、開始位置 p、末尾 N (最後のチャンクだけ、N = min(N, T))。

1. **層 0..23**: 今のまま全 T 行。終わりの `R` ([T][4][2560]) が境界状態。
2. **埋め (層 24..47、行 0..<T−N)**:
   - x̄ = R の 4 本平均 (2560)。zero-shot の土台。代案は自層の `hcMix` を境界状態に掛ける形 (層 24 は境界状態がそのまま入力なので正確になる) — どちらが良いかは測る。
   - GDN 層: `linear()` の前半 (`attn_qkv` / `ssm_beta` / `ssm_alpha` の GEMM → conv (`linHist` 更新) → qk norm → gates → `Qwen38GDNChunk`) を x̄ で回し、`linState` を進める。`attn_gate`・`ssm_norm`・`ssm_out` は要らない。
   - FA 層: `attn_k` / `attn_v` / `indexer.k_proj` の GEMM を x̄ で回し、既存の norm・RoPE・Q8 量子化 (`attnPrep` / `psoKVQuantize`)、生鍵 F16 とブロック鍵 (`psoIdxBlockKey`) をそのまま。`attn_q`・注意・`attn_output` は要らない。
   - LoRA / MLP (§4-3) は各 GEMM の後に足す。
3. **末尾 N 行 (層 24..47)**: `R` の末尾 N 行を先頭に詰め (logits の経路と同じ memmove)、今の層ループを il = 24..<48、T = N、startPos = p + T − N で回す。
   GDN の状態は 2 の続き、FA は近似 KV の後ろに正確な行が並ぶ。MoE も N 行分だけ。
   最後でないチャンクは 2 を全行に掛け、3 は無し。

チェックポイント (18・25) は状態 + 履歴 + KV の位置なので形は変わらない。`onLayer` の進捗は 24 + 埋め + 末尾で数え直す。

### 4-2. うちにだけある問題

- **MTP のドラフト**: `Qwen38Completion` は prefill の全トークンに `runner.hidden(row:)` (幹の最終残差) を渡して MTP の KV を作る。近似した行にはそれが無い。
  候補は (a) 境界状態を代わりに渡す、(b) 末尾 N 行だけ渡す。どちらも受理率で測るしかない (PixelML は MTP 未使用)。
- **チャンクの継ぎ目**: 埋めは conv 履歴と再帰状態を層 24..46 で進めるので継ぎ目は今と同じ形。PR #45 の「255+1 で状態 10% ずれ」は oracle 検査の話で、運用の経路ではない。
- **文法つき生成・ツールループ**: 近似の影響が一番出やすい所。25 の分岐点探針 (P(<tool_call>)) がそのまま使える。

### 4-3. HF の重みを使うとき

- LoRA (18 層) と MLP (6 層) を F16 の Metal バッファに (385 MB、Q8 なら ≈ 200 MB)。wired の余裕は 32K spec で 14.08 GB (23) なので +0.4 GB は入る見込み (未測定)。
- 土台を PixelML と同じ「4 本平均に自層の in_proj、norm なし」にしないと補正がずれる。config の記述はそれだが、K が k-norm の前か後か (27B 版は「post k-norm」、Flash-Next 版は「pre-RoPE」のみ) は receipts に無い。両方試すか、zero-shot で足りるなら使わない。
- AWQ-INT4 の土台向けの補正を IQ2 の土台に載せるずれは、zero-shot との差で測る。

## 5. 最初に測る段 (提案)

1. **配管**: §4-1 を `Q38_LLKV_SPLIT=24`・`Q38_LLKV_SUFFIX=N`・`Q38_LLKV_FILL=oracle|zeroshot` で作る。oracle は層 24..47 も全行流して同じ値を同じ経路で書く (= 今の結果とビット一致するはず)。
   `--qwen38-prefill` の France / Fuji 参照で oracle が PASS すれば配管は正しい。
2. **速度**: `--qwen38-prefill-bench` 8,192 と 32,000 (チャンク 2048、memlog 越し、Swapouts と wired を並べる) を zeroshot / suffix 256 で。§3 の見積もりと比べる。
3. **品質**: 22 §4 の greedy 200 (code / explain / tool / long) を近似ありなしで、25 の分岐点探針で P(<tool_call>) を、21 の記録済み会話 (13/14) をツールループ検査で。数字だけ書く (n=1)。
4. その後に HF の LoRA/MLP を読んで 3 を繰り返す。学習し直すかはここまでの数字を見てから。

量: 1 は既存の関数の並べ替えが中心 (新しいカーネルは要らない見込み)、ローダは 2 のあと。

実施: 1〜3 の一部は [29](29-LLKVAPPROX-ZEROSHOT.md) (8K で 1.44〜1.50 倍、zero-shot は k6 の分岐点で P(`<tool_call>`) が動く)。
