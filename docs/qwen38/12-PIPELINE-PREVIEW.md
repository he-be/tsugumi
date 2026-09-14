# 12. decode を Tsugumi の形に: 層あたり同期 1 回・shared expert を読みの裏へ・層またぎの router 予測で先に advise

実測: 2026-09-14、M3 Pro 18 GB、macOS 15。対象は [10](10-MTP-SHADOW-SPEC.md)・[11](11-VERIFY-PASS-SPLIT.md) と同じ GGUF・プロンプト。
表記は 01 と同じ (実測 / 導出 / 未確認)。**どのセルも 2 パスなので、数字だけを書く。**

## 0. 結論

1. **11 §5 の「層 L の読みは隠せない」は、直列形のランナーの上で問いを閉じていた。**Gemma (`RealForwardRunner`) と
   Ornith (`QwenForwardRunner`) は、層あたり同期 1 回・shared expert を読みの裏・層またぎの先読み (Ornith) を持っていたが、
   `Qwen38Runner` は層あたり 2 回待ち、shared expert を id の前のバッファに置き、先読みも無かった (§1)。
2. 3 つを入れた。**すべての腕・12 組でトークン列が一致** (§2)。検査 24 PASS、MTP 参照 PASS、素 = 全部棄却 = 投機 (code、greedy、変更前の出力とも一致)。
3. steady tok/s (instruct、200 トークン、腕を順・逆で交互、§2):
   - パイプライン + shared 後置 (`late`、既定): 直列比 素 1.04〜1.12 倍・投機 1.04〜1.13 倍。
   - それに**層 L+1 の router を層 L の FFN 入力に当て、各行の上位 10 本を advise** (`prev10`): **素 1.34〜1.48 倍 (8.12〜8.97 tok/s)、投機 1.21〜1.31 倍 (9.34〜10.37 tok/s)**。
4. 予測の的中 (実 expert のうち予測に入っていた割合): 上位 10 本で素 0.620〜0.656、検証 0.639〜0.669。上位 20 本で 0.842 (code 1 本、§3)。
5. 残る露出は routed の kernel→GPU 開始の待ちで、prev10 でも素 25〜34 ms・検証 47〜65 ms (直列 32〜39 / 51〜69)。
   prev10 で縮んだのは advise の syscall (15〜21 → 7〜8 ms) と、host の予測 20〜23 ms (検証 39〜44 ms) が GPU の裏に入った分。
6. **予測は既定 off のまま。**12K (Swapouts の縁) で先読みがページキャッシュを押す量を見ていない。

## 1. 形 (`Qwen38Runner.forwardBody`)

```
直列 (11):  cb1(L) [hc_attn, 混合器, hc_ffn, shared, router] wait → top-10 → advise → cb2(L) [routed] wait → cb1(L+1) …
late:       cb1(L) [hc_attn, 混合器, hc_ffn, router (+ 層 L+1 の router 予測)] wait (cb2(L-1) もここで合流)
            → top-10 → shared(L) commit → views・advise → cb2(L) commit (待たない)
            → [prev10] 層 L+1 の上位 10 本/行 を advise → cb1(L+1) encode → commit → wait
```

- host が層の途中で GPU の結果を読むのは router の logits と、長文脈の indexer スコア (その同期で前の cb2 も合流する) だけ。
  `nq`・`selection`・`pairSlot`・`routedArg` などは host が書くだけで、書く時点で前の cb2 は終わっている。最終層の cb2 だけは `exportHidden` の前に待つ。
- 予測は `ffn_gate_inp` (層 L+1) を層 L の `mixed` (hc_ffn の出力) に当てた 512 logits。正確な routing には一切使わない。T < 32 のバッチだけ。
- 環境変数: `Q38_PIPELINE` (既定 1)、`Q38_SHARED_LATE` (既定 1)、`Q38_PREVIEW_N` (既定 0 = off)、`Q38_PREVIEW_ADVISE` (既定 0 = 的中を数えるだけ)。

## 2. 腕の比較 (`scratch/qwen38/pipe12.sh`)

腕: `serial` (`Q38_PIPELINE=0`)、`pipe` (`Q38_SHARED_LATE=0`)、`late` (既定)、`prev10` (`Q38_PREVIEW_N=10 Q38_PREVIEW_ADVISE=1`)。
プロンプトごとにパス 1 は serial → pipe → late → prev10 (各腕 off → spec)、パス 2 はその逆順。instruct、seed = パス、200 トークン、チャンク 512、間 10 秒。
48 本すべて Swapouts 0 (late の 1 本だけ 8 ページ)。

steady tok/s (素の括弧は steady 中央値 ms、投機の括弧は検証の中央値 ms):

| プロンプト | パス | mtp | serial | pipe | late | prev10 | late / serial | prev10 / serial |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code | 1 | off | 6.55 (147) | 7.11 (134) | 7.30 (130) | 8.93 (112) | 1.115 | 1.363 |
| code | 2 | off | 6.64 (147) | 7.39 (132) | 6.91 (142) | 8.97 (112) | 1.041 | 1.351 |
| tool | 1 | off | 5.73 (163) | 6.22 (155) | 6.20 (156) | 8.46 (115) | 1.082 | 1.476 |
| tool | 2 | off | 5.68 (175) | 5.98 (159) | 6.31 (153) | 8.12 (120) | 1.111 | 1.430 |
| explain | 1 | off | 6.41 (149) | 7.24 (130) | 6.85 (136) | 8.97 (108) | 1.069 | 1.399 |
| explain | 2 | off | 6.54 (144) | 7.03 (134) | 6.91 (136) | 8.74 (110) | 1.057 | 1.336 |
| code | 1 | spec | 8.44 (194) | 9.27 (174) | 9.51 (166) | 10.37 (156) | 1.127 | 1.229 |
| code | 2 | spec | 8.37 (191) | 8.99 (180) | 8.79 (181) | 10.11 (157) | 1.050 | 1.208 |
| tool | 1 | spec | 7.52 (221) | 8.57 (199) | 8.27 (206) | 9.84 (175) | 1.100 | 1.309 |
| tool | 2 | spec | 7.74 (226) | 7.89 (225) | 8.10 (216) | 9.82 (173) | 1.047 | 1.269 |
| explain | 1 | spec | 7.60 (190) | 8.33 (176) | 7.92 (175) | 9.63 (151) | 1.042 | 1.267 |
| explain | 2 | spec | 7.45 (189) | 7.97 (181) | 7.99 (173) | 9.34 (151) | 1.072 | 1.254 |

- 12 組すべてで 4 腕のトークン列が一致。
- 受理率は腕の間で同じ (同じ seed で同じトークン列) なので載せない。

### 2-1. 内訳 (steady の中央値 ms、最初のステップを除く)

パイプラインの腕の `pre` は前の層の routed の待ち (kernel→GPU・GPU) を含み、`routed` は commit だけになる。

| プロンプト | パス | mtp | 腕 | wall | pre [GPU] | route (advise) | kernel→GPU | routed GPU | 予測の host | 的中 |
| --- | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| code | 1 | off | serial | 147 | 68 [57] | 17 (14.9) | 33.0 | 11.6 | — | — |
| code | 1 | off | late | 130 | 104 [54] | 19 (15.6) | 32.3 | 11.5 | — | — |
| code | 1 | off | prev10 | 112 | 74 [50] | 9 (6.9) | 28.0 | 8.5 | 20.3 | 0.647 |
| tool | 1 | off | serial | 162 | 80 [68] | 21 (18.0) | 38.8 | 13.8 | — | — |
| tool | 1 | off | late | 156 | 119 [62] | 23 (19.8) | 36.0 | 14.2 | — | — |
| tool | 1 | off | prev10 | 115 | 74 [51] | 10 (7.8) | 31.0 | 8.6 | 23.0 | 0.620 |
| explain | 1 | off | serial | 149 | 71 [60] | 19 (16.1) | 33.0 | 12.5 | — | — |
| explain | 1 | off | late | 136 | 109 [57] | 18 (14.9) | 33.2 | 12.8 | — | — |
| explain | 1 | off | prev10 | 108 | 71 [50] | 9 (6.8) | 25.0 | 8.5 | 20.3 | 0.647 |
| code | 1 | spec (検証) | serial | 194 | 66 [55] | 30 (26.4) | 55.7 | 16.5 | — | — |
| code | 1 | spec (検証) | late | 164 | 125 [51] | 31 (27.3) | 52.8 | 15.5 | — | — |
| code | 1 | spec (検証) | prev10 | 156 | 88 [53] | 16 (13.2) | 52.5 | 15.3 | 40.6 | 0.658 |
| tool | 1 | spec (検証) | serial | 219 | 75 [63] | 38 (34.0) | 68.3 | 21.0 | — | — |
| tool | 1 | spec (検証) | late | 196 | 151 [55] | 35 (31.1) | 70.3 | 17.4 | — | — |
| tool | 1 | spec (検証) | prev10 | 170 | 94 [55] | 19 (15.9) | 65.2 | 15.4 | 43.3 | 0.639 |
| explain | 1 | spec (検証) | serial | 190 | 66 [55] | 32 (28.7) | 52.0 | 16.8 | — | — |
| explain | 1 | spec (検証) | late | 174 | 130 [52] | 33 (29.7) | 54.0 | 16.6 | — | — |
| explain | 1 | spec (検証) | prev10 | 150 | 81 [53] | 15 (12.5) | 46.9 | 15.3 | 40.7 | 0.659 |

パス 2 と pipe の腕の行は `scratch/qwen38/pipe12/` のログから同じ抽出で出る。

prefill (短文脈、trunk s、パス 1 / 2): code serial 5.0 / 5.1・late 4.8 / 5.1、tool serial 12.8 / 12.5・late 11.3 / 10.5、explain serial 4.1 / 4.0・late 3.8 / 3.8。
(この走行の prev10 は予測が prefill のバッチにも掛かっていた。走行後に T < 32 へ絞った。)

## 3. 予測の的中 (計器のみ、`Q38_PREVIEW_N=10|20`、code、投機、60 トークン、n=1)

| 上位 N / 行 | 的中 (実 expert の和) | 予測した expert / 実 expert |
| ---: | --- | ---: |
| 10 | 15,931 / 23,893 = 0.667 | 1.00 倍 |
| 20 | 20,109 / 23,893 = 0.842 | 1.97 倍 |

比べ先: Gemma は層 L の hidden に層 L+1 の router で実ミスの 66% (上位 1 本で 70%)、壁時計 −1.5〜−3.5% ([mtp/30](../mtp/30-M8-B-PREFETCH.md)、32 スロットの奪い合い)。
Ornith は要取得の 64.5%、+8.4〜22.3% ([qwen35moe/27](../qwen35moe/27-PHASE6-THROUGHPUT.md)・[31](../qwen35moe/31-PREFETCH-CHEAPER.md))。Qwen3.8 はスロットを持たずページキャッシュがそのままキャッシュなので、外れは帯域とキャッシュの押し出しで払う。

## 4. 未測定・次

- **12K での prev10** (wired・file-backed・Swapouts)。既定を変える前に要る。
- 上位 N の掃引 (5 / 20)、検証の行ごとの N。
- **kernel→GPU 開始の待ち (素 25〜34 ms・検証 47〜65 ms) の中身。**11 §3 で全部 pread 済みでも T=1 で 12〜13 ms 残ったので、SSD でない常駐の費用がある。
  予測した expert を別スレッドで residency set に入れて `requestResidency` する腕 (Gemma P-5 [mtp/49](../mtp/49-D-P5-RESIDENCY-SET.md)、Ornith の commit 非同期 [qwen35moe/39](../qwen35moe/39-RESIDENCY-COMMIT.md)) は、予測があって初めて「前もって」出せる。
- **MTP 固有の形。**検証の行 y はドラフトに依存しないので、ドラフト中に幹の層 0 を進められる。
  行 d を y の半層後ろに流せば、y の層 L の読みを d の層 L の計算で隠せる (いまは T=2 の和集合を同じ瞬間に読んでいる)。どちらも作っていない。
- 常駐 expert を先に流す (Gemma の hit-first) は、キャッシュ状態を持たないので未着手。

## 5. 足したもの・変えたもの

| 種類 | パス |
| --- | --- |
| ランナー | `pipeline`・`sharedLate`・`previewTopN`・`previewAdvise`、`commitShared`・`previewExperts`・`advisePreview`、`StepProfile.preview*` |
| 検査 | `routeDetail` に `preview hit a/b named n host ms` |
| スクリプト (git 管理外) | `scratch/qwen38/pipe12.sh` と `pipe12/`、`preview12/`、`checks12.out` |
