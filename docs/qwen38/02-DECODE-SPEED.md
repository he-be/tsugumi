# 02. Q2 ランナーの decode 速度 — 1 トークン 0.22〜0.38 s → 0.12〜0.15 s (短文脈、キャッシュ温)

実測: 2026-09-13、M3 Pro 18 GB、macOS 15。対象は [01](01-Q2-FIRST-LIGHT.md) の `Qwen38Runner` と同じ GGUF。
表記は 01 と同じ (実測 / 導出 / 未確認)。**反復 3 未満の数字には解釈を書かない。**

## 0. 結論

1. **pre-router の GPU 時間の 2/3 は行列積ではなかった。**hyper-connection の RMS (4 スレッドが 2560 要素ずつ直列ループ) と、
   F16/F32 GEMV の「1 レーン 1 要素飛び」のアクセス形が主因 (§1、§2)。
2. 直した 3 つで pre-router の GPU は **152〜160 ms → 50〜59 ms** (France 10 位置、n=2 の走行)。
   - F16/F32 GEMV を 32 要素チャンク読みに (行幅 ≥ 1024 のみ): hc down 14.3 → 1.67 ms / 48 回。
   - RMS を「SIMD レーンで二乗和 → 要素ごとに掛ける」の 2 段に: 16.4 → 0.28 ms / 48 回。
   - 注意を 5 パス (score → max → sum → weight → mix) に: **n=2048 で 5,783 → 19 ms / 48 回** (§2-3)。
     旧形のままだと 32K の運用点で注意だけ 1 トークン約 1.4 s だった (導出)。
3. **Q8_0 GEMV は既に帯域の天井**で、別の形 2 つはどちらも遅い (§1-1)。
4. **routed expert の待ちは SSD 読み。**53 トークンで選ばれた expert のうち約 200 MB/トークンがページキャッシュに無い (18 GB では載り切らない)。
   選ばれた 30 範囲に `F_RDADVISE` を出すと中央値 **240〜325 ms → 160〜185 ms** (§3)。residency set を重ねても変わらない。
5. 1 トークン (France、位置 1〜9、キャッシュ温): **0.22〜0.38 s → 0.12〜0.15 s**。logits は参照と ≤ 8.5e-7、Fuji 予算 16 は 53/53・≤ 5.4e-7。
6. 残り: PLE (host 17〜28 ms)、routed の SSD 待ち、pre の host 10 ms と route 10〜13 ms (advise の syscall)、prefill、MTP (§5)。

## 1. 内訳を割る

`StepProfile` に `preGPU` / `routedGPU` (`gpuEndTime − gpuStartTime`) を足し、`Q38_SPLIT_PRE=1` で pre-router を 4 本のコマンドバッファ
(hc_attn / 混合器 / combine + hc_ffn / 共有 expert + router) に割って GPU 時間を帰属させた。

変更前 (France、位置 1〜9、n=3):

| 段 | wall | GPU |
| --- | ---: | ---: |
| pre-router | 162〜172 ms | 152〜160 ms |
| routed | 25〜203 ms | 8〜9 ms |

pre は GPU 待ちで、host の encode は約 10 ms。routed は GPU が 8 ms で、残りは commit から GPU 開始までの待ち (§3)。

割った内訳 (変更前、位置 6〜9、n=3): **hc_attn 53〜59、hc_ffn 53〜59**、gdn 29、attn 12〜15、shexp+router 8 ms。
Q8_0 の形から見積もると dense 行列積は 1 トークン約 25 ms、hc の F16 行列積は 1.3 GB で帯域なら約 9 ms (導出) なので、hc の 108 ms は読むバイト数では説明できない。

エンコーダ数 (1 トークン約 1,440 個) は原因ではない: 空カーネル 1,440 エンコーダの GPU 時間は 1.6〜2.2 ms (実測、15 回の中央値)。

### 1-1. Q8_0 GEMV の形 (実測、`--ggml-dense-bench`、48 回 / サンプル、15 回の中央値、GPU ms)

| テンソル | 現行 (64 スレッド/TG、4 行/群) | lane256 (int8 本線の形) | rows (1 スレッド 1 行) |
| --- | ---: | ---: | ---: |
| attn_qkv 10240×2560 | 10.06 | 10.59 | 10.27 |
| ssm_out 2560×6144 | 6.13 | 7.17 | 10.00 |
| ffn_up_shexp 640×2560 | 0.50 | 0.88 | 1.92 |

attn_qkv × 48 は 1.26 GB を 10 ms で読んでいて、M3 Pro の帯域 (150 GB/s) に近い (導出)。現行のまま。

## 2. 直したもの

### 2-1. F16 / F32 GEMV (`ggml_f16_gemv_chunk` / `ggml_f32_gemv_chunk`)

同じディスパッチのまま、レーン l が読むのを「要素 l, l+32, …」から「32 要素のチャンク l, l+32, … を丸ごと」に変えた (Q8_0 と同じアクセス形)。

| テンソル | stride | chunk |
| --- | ---: | ---: |
| hc_attn_down F16 320×10240 | 14.30 | 1.67 |
| hc_attn_up F16 10240×320 | 1.60 | 1.83 |
| hc_attn_inject F16 4×10240 | 7.12 | 0.86 |
| ffn_gate_inp F32 512×2560 | 3.74 | 1.04 |
| ssm_alpha F32 48×2560 | 1.97 | 0.47 |

(実測、48 回 / サンプル、15 回の中央値、交互順。出力差 ≤ 7e-7。) 行幅 320 はチャンクが 10 個でレーンが余るので stride のまま (`GGMLDenseGEMV` で N ≥ 1024 のときだけ chunk)。

### 2-2. RMS (`q38_group_rms_scale` + `q38_rms_apply`)

旧 `q38_grouped_rms` は群ごとに 1 スレッドで 2560 要素を 2 回なめる。4 群だと 1 SIMD グループに収まり、並列度がほぼ無い。
二乗和を 1 群 = 1 SIMD グループ (32 レーン、チャンク読み) で `simd_sum` し、掛け算は要素ごとのスレッドにした。

| | GPU ms / 48 回 |
| --- | ---: |
| 旧 grouped_rms 4×2560 | 16.35 |
| rms_scale + rms_apply | 0.13 + 0.15 |

hc セクションは 1 本 53〜59 → 25〜30 (§2-1 後、1 位置だけ 41) → **7.6〜9.6 ms** (実測、位置 6〜9、各 n=3)。

### 2-3. 注意 (`q38_attn_score` / `_stat` / `_weight` / `_mix`)

旧 `q38_attn_decode` はヘッドごとに 1 スレッドで n × 256 を 2 回なめる。

| | n=10 | n=2048 |
| --- | ---: | ---: |
| 旧 attn_decode (GPU ms / 48 回) | 25.6 | **5,783** |
| 5 パス | 1.96 | **19.1** |

(実測、bench はランダム入力、n=2048 は 3〜5 回の中央値。) 5 パス: score (TG = (トークン, ヘッド)、レーンは次元)、max と Σexp (TG = ヘッド、レーンはトークン)、
重み (要素ごと)、mix (TG = (次元, ヘッド)、レーンはトークン)。選択 (`sel`) の間接参照は score と mix の中。

QSA の選択を通す Fuji 予算 16 で 53/53、logits ≤ 5.4e-7 (変更前 1.41e-6)。

### 2-4. 残っている小さいカーネル (実測、48 回 / サンプル、15 回の中央値、GPU ms)

gdn_step 3.40、gdn_norm_gate 1.13、gdn_qk_norm 1.00、attn_prep 7.06、gdn_conv 0.15、hc_mix 0.11、hc_combine 0.13。
1 トークンに直すと GDN 3 つで約 4 ms、attn_prep 約 1.8 ms (導出)。後回し。

## 3. routed expert の待ち — `F_RDADVISE`

53 トークンの Fuji (`ref-fuji-ple.log`、logits 無し) で、選ばれた expert の非常駐バイトを `mincore` で数えると **52 トークンで 8.6〜11.3 GB**。
同じプロンプトを続けて回しても減らない (1 トークン 715 MB の選択に対し、ページキャッシュに残る分が足りない)。

腕 (各 n=2、順番は ABBA、20 秒クールダウン、Swapouts 増分 0。上 4 行は数えるための `mincore` が route に入っていて、それだけで 15〜18 ms):

| 腕 | 1 トークンの中央値 | routed wall − GPU (中央値) | route (中央値) |
| --- | ---: | ---: | ---: |
| A: 何もしない | 325 / 280 ms | 135 / 114 ms | 22 / 23 ms |
| R: residency set 2 GB (LRU) | 260 / 240 ms | 17 / 16 ms | 124 / 114 ms |
| V: 選んだ 30 範囲に `F_RDADVISE` (mincore で欠けだけ) | 170 / 160〜170 ms | 28〜29 ms | 38〜41 ms |
| VR: V + R | 180 / 170 ms | 11 ms | 60〜61 ms |
| V (mincore 無し、全範囲) | 185 / 160 ms、170 / 170 ms | 39〜44 ms | 17〜20 ms |
| V を `concurrentPerform` で並列に出す | 170 / 170 ms | 43〜44 ms | 17〜18 ms |
| advise 無し (n=1) | 245 ms | 112 ms | 4 ms |

residency set は単独では A より速い (待ちが routed から route の `requestResidency` に移り、合計も縮む) が、advise と重ねると V と変わらず、裾 (最大) は悪い。**advise (直列、mincore 無し) を既定にした** (`Q38_ADVISE=0` で外す、`Q38_COUNT_MISS=1` で数える)。
route の 10〜20 ms は advise の syscall (1 トークン 1,440 回)。並列にしても縮まない。

## 4. 変更後の 1 トークン (France、位置 1〜9、実測)

| 段 | 変更前 (n=3) | 変更後 (n=2) |
| --- | ---: | ---: |
| 合計 | 0.22〜0.38 s | **0.12〜0.15 s** |
| ple (host) | 17〜26 ms | 17〜27 ms |
| pre (wall / GPU) | 162〜172 / 152〜160 ms | 60〜69 / 50〜59 ms |
| route | 4〜7 ms | 10〜13 ms |
| routed (wall / GPU) | 25〜203 / 8〜9 ms | 21〜40 / 8〜12 ms |
| head | 6 ms | 5 ms |

最後の 3 本のうち 1 本は pre GPU 64〜117 ms・routed GPU 12〜32 ms と全体に倍近く、原因は特定していない (その 1 本は表に入れていない)。
同じ日の途中に、ユーザーが止めた別プロセスのテストが GPU を使っていた走行も 1 本ある (これも除外)。

## 5. 次

1. **PLE を GPU に** (host 17〜28 ms、Q4_1 16 行と 2 本の Q8_0 行列積と conv)。
2. **routed の SSD 待ち**: 1 層先の router を今の層の入力で回して先に advise する (本線の `ExpertPrefetch` の考え方)。当たり率を先に測る。
3. **prefill** (32K の運用点でツール結果を読む速度)。今は 1 トークンずつしか流せない。
4. MTP (blk.48)、品質 (server に繋いで英語のツール呼び出し) は 01 §7-4 のまま。

## 6. 足したもの

| 種類 | パス |
| --- | --- |
| カーネル | `ggml_dense.metal`: `ggml_f16_gemv_chunk`・`ggml_f32_gemv_chunk` (採用)、`ggml_q8_0_gemv_lane`・`_rows` (bench 用、不採用) |
| | `qwen38.metal`: `q38_group_rms_scale`・`q38_rms_apply`・`q38_attn_score`・`_stat`・`_weight`・`_mix` (旧 `q38_grouped_rms`・`q38_attn_decode` は削除) |
| ランナー | `Qwen38Runner`: `StepProfile.preGPU`・`routedGPU`・`sections`・`missBytes`、`splitPreRouter`、`adviseExperts`、`countMisses` |
| GGUF | `GGUFFile.nonResidentBytes`・`adviseRead` |
| 検査 | `--ggml-dense-bench <gguf> [iters]`、`--q38-small-bench [iters]`、環境変数 `Q38_SPLIT_PRE=1`・`Q38_ADVISE=0`・`Q38_COUNT_MISS=1` |

再現: 01 §7-2 の 4 本 (期待値は変わらず PASS) に加えて

```bash
B=.build/release/TsugumiKernelCheck; G=~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf
$B --ggml-dense-bench $G 15
$B --q38-small-bench 15
Q38_SPLIT_PRE=1 $B --qwen38-decode scratch/qwen38/ref-france-dump.log --q38-ref-logits scratch/qwen38/ref-france.logits
Q38_COUNT_MISS=1 $B --qwen38-decode scratch/qwen38/ref-fuji-ple.log
```

**zsh では `env $VARS cmd` が単語分割されない** (`${=VARS}` が要る)。§3 の最初の VR 腕はこれで両方 off のまま走り、A と同じ数字になった。
