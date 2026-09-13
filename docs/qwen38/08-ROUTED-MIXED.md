# 08. routed expert の混成 — 組の少ない expert は per-pair カーネルに、1 チャンク 4096 の routed GPU 11.7 → 9.4 s・8K チャンク 2048 で 107.4 → 87.4 s

実測: 2026-09-14、M3 Pro 18 GB、macOS 15。対象は [07](07-GDN-CHUNK.md) と同じ GGUF・ランナー・プロンプト。
表記は 01 と同じ (実測 / 導出 / 未確認)。**ランナーの速度の行はどれも n=1 (§2 の 24 だけ n=2) で、数字だけを書く。**

## 0. 結論

1. **T ≥ 1024 の routed で、組が `gemmMinPairs` (既定 24) 未満の expert は展開 + sgemm をやめ、集めた行の上で従来の per-pair カーネルを回す** (`Qwen38Runner.routedGemm`)。
   `moeRouted` が top-10 の直後に slot を「組の多い expert → 少ない expert」の順に振り直すので、展開のグループは連続した slot のまま、少ない側の組は `gemmRows` の末尾に並ぶ。
   カーネルは足していない (§1)。`Q38_GEMM_MIN_PAIRS=0` で 07 と同じ経路。
2. **見積もり (導出)**: 06 §2-2 の表から 1 expert の固定費 0.35 ms・1 組 2.1 µs (sgemm 側)、per-pair 側 13.9 µs/組。48 層の実際の選択 (`routes/c4096-l*.bin`) に当てると 11.7 s → しきい値 24〜32 で 9.3 s。
3. **実測 (4,096 トークン 1 チャンク、§2)**: routed GPU は 0 で 11.71 s、16〜48 で 9.35〜9.59 s、既定 24 で 9.47 / 9.36 s。
4. **正しさ**: `Q38_GEMM_MIN_T=1` で「全部 per-pair 側 (1000)・混ぜる (3)・全部 sgemm 側 (0)」を参照と照合し全部 PASS、top-1 のずれ 0 (§3)。07 §3 の 19 本も PASS。
5. **8K** (§4):

   | | チャンク 2048 | チャンク 4096 |
   | --- | ---: | ---: |
   | 07 | 107.4 s (76.3 tok/s) | 79.9 s (102.6 tok/s) |
   | 08 | **87.4 s (93.8 tok/s)** | **2 回とも Swapouts で見張りが kill** |
   | routed GPU / チャンク | 8.8〜9.6 → 6.1〜6.3 s | (1 チャンク目だけ 11.9 → 9.4 s) |
   | peak footprint | 3.72 → 3.70 GB | (1 チャンクで 5.24〜5.32 GB) |

6. **チャンク 4096 の 8K は 2 本とも 2 チャンク目で Swapouts** (+10,752 / +5,960 ページ)。どちらも 2 チャンク目に入った瞬間に wired が 8.3 → 13 GB に跳ね、その十数秒後に出た。
   07 の走行も同じ跳ね (最大 12.69 GB) で、そのときは出なかった。**この変更のせいかは確かめていない** (§4-1)。32K の前にこの跳ねの中身を見る (§5 の 1)。

## 1. 形

`moeRouted` (T ≥ `gemmMinTokens` のとき):

1. top-10 の後、slot ごとの組の数を数え、組 ≥ `gemmMinPairs` の slot を先 (`gemmBigSlots` 個)、残りを後ろに並べ直して `pairSlot` を書き換える (各側の中の順は保つ)。
   expert のビュー・読み (pread)・`routedArg` はその後なので、並べ直した slot 番号のまま従来どおり。

`routedGemm` (06 §1 の流れに足したもの):

- 展開 + sgemm のグループは `0..<SB` だけ。`gemmGateUp` は `SB = 0` でも 1 行ぶん確保する (0 バイトだと `ensureBuffer` が作らず落ちる)。
- 組の少ない側は位置 `p0 = start[SB] ..< P` に並んでいる。`gemmPosSlot[i] = pairSlot[order[p0 + i]]` を作り:
  1. `moe_iq2xxs_phase1_gate_up_act` を `x = gemmRows + p0`、`top_k = 1`、`tokens = nSmall`、`pair_slot = gemmPosSlot` で。`acts[i]` に SiLU(gate)·up。
  2. blit の `fill` で `gemmRows[p0..<P]` を 0 に (phase 2 は residual を足すので、読み終わった行を 0 にして residual と出力の両方に使う)。
  3. `moe_q2k_phase2_down_reduce` を `routing_w = gemmOnes` (全部 1)、`residual = y = gemmRows + p0`、`top_k = 1` で。重みを掛けない down の出力が行に戻る。
- 散らし (`moe_scatter_weighted`) は 06 と同じ。重みはここで掛かる。

`acts` の pad 列 (640..767) は一度も書かないので 0 のまま (phase 2 はそこも読む)。

## 2. しきい値 (`Q38_SPLIT_PRE=1 --qwen38-prefill-bench --q38-tokens 4096 --q38-chunk 4096`、memlog 越し、間に 20 秒)

走った順に 24・0・16・32・48・24。

| `Q38_GEMM_MIN_PAIRS` | 合計 | routed wall / GPU | PLE | footprint | wired の最大 | file-backed の最小 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 35.73 s | 12.30 / 11.71 s | 0.19 s | 5.30 GB | 12.94 GB | 2.16 GB |
| 16 | 35.44 | 10.09 / 9.50 | 2.63 | 5.32 | 10.92 | 2.12 |
| 24 (1 本目) | 39.11 | 10.11 / 9.47 | 5.57 | 5.26 | 7.54 | 5.24 |
| 24 (2 本目) | 34.13 | 9.94 / 9.36 | 1.38 | 5.24 | 10.33 | 3.17 |
| 32 | 34.53 | 9.96 / 9.35 | 1.84 | 5.24 | 8.27 | 3.27 |
| 48 | 32.74 | 10.15 / 9.59 | 0.28 | 5.24 | 8.15 | 4.22 |

07 の 8K 走行の 1 チャンク目は routed 12.47 / 11.87 s。pre-router GPU はどの行も 9.49〜9.66 s。合計は PLE の冷えた読みで上下する。
**既定を 24 にした**のは §0 の 2 の見積もりの谷と、16〜48 の実測の幅から。wired の最大は同じ 1 チャンクでも 7.5〜12.9 GB と揺れた。

## 3. 参照との照合

`Q38_GEMM_MIN_T=1` (decode も含めて routed を全部 gemm 経路に) で `Q38_GEMM_MIN_PAIRS` を変えたもの:

| 検査 | 1000 (全部 per-pair 側) | 3 (混ぜる) | 0 (全部 sgemm 側) |
| --- | ---: | ---: | ---: |
| `--qwen38-decode` France | 8.70e-7 | 8.70e-7 | 7.42e-7 |
| Fuji 予算 16、チャンク 16 | 6.11e-7 | 7.92e-7 | 6.59e-7 |
| Fuji 予算 16、チャンク 53 | 2.95e-6 | 2.27e-6 | 2.52e-6 |
| Fuji PLE チャンク 53 | | PASS | |

top-1 のずれはどれも 0。全部 per-pair 側の France 8.70e-7 は、従来経路 (8.50e-7) と重みを掛ける場所 (phase 2 の和の中か、散らしか) が違う分。
France decode は T = 1 で全 expert が 1 組なので、3 でも全部 per-pair 側。

07 §3 の 19 本 (`checks07.sh`) は 24 PASS。変わったのは `Q38_GEMM_MIN_T=1` の 3 行だけ (既定 24 の混成になる): France decode 7.42e-7 → 8.70e-7、Fuji チャンク 16 6.59e-7 → 6.11e-7、チャンク 53 2.52e-6 → 2.51e-6。
`--q38-wy-bench 4096 32 3` は A 173.1 ms / B GPU 29.0 ms + encode 9.4 ms (変更前に測った、07 と同じ)。

## 4. ランナーでの結果 (`Q38_SPLIT_PRE=1 --qwen38-prefill-bench`、8,192 トークン)

チャンク 2048 (memlog 越し):

| チャンク | 合計 | pre wall / GPU | route | routed wall / GPU | PLE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 07、0..<2048 | 28.35 s | | 6.38 s | 9.48 / 8.85 s | |
| 07、2048..<4096 | 25.85 | | 6.87 | 9.97 / 9.23 | |
| 07、4096..<6144 | 26.56 | | 7.15 | 10.29 / 9.56 | |
| 07、6144..<8192 | 26.61 | | 7.16 | 10.25 / 9.51 | |
| 08、0..<2048 | 21.83 | 7.37 / 5.10 | 5.89 | 6.66 / 6.13 | 0.11 |
| 08、2048..<4096 | 20.36 | 6.40 / 5.20 | 6.27 | 6.80 / 6.21 | 0.89 |
| 08、4096..<6144 | 22.56 | 6.96 / 5.60 | 6.47 | 6.92 / 6.33 | 2.21 |
| 08、6144..<8192 | 22.61 | 7.22 / 5.85 | 6.39 | 6.78 / 6.19 | 2.21 |

合計 107.4 → **87.4 s (93.8 tok/s)**。decode (8,192 の直後 1 トークン) 0.44 s (07 は 0.43 s)。peak footprint 3.70 GB、wired の最大 12.78 GB、file-backed の最小 1.80 GB、Swapouts の増分 0。
チャンク 2048 は 1 expert あたりの組が少ないので、per-pair 側に回る expert が 4096 より多い (未測定)。

### 4-1. チャンク 4096 の 8K: 2 本とも kill

| 走行 | 出た時刻 | 直前の wired / file-backed | Swapouts |
| --- | --- | --- | ---: |
| 1 本目 (sweep の最初、既定 24) | t = 68〜70 s | 12.51 / 1.20 GB | +10,752 ページ |
| 2 本目 (§4 のあと、既定 24) | t = 56〜58 s | 12.62 / 1.86 GB | +5,960 ページ |

- どちらも `guarded.sh` が子孫ごと kill し、`pgrep -x TsugumiKernelCheck` は空。出力はバッファされていて、1 チャンク目の行も残らなかった。
- wired は 1 チャンク目 6.3〜7.5 GB、2 チャンク目に入る t ≈ 42〜44 s で 8.3 → 12.8〜13.1 GB。同じ瞬間に bench の RSS が 1.5 → 0.3 GB に落ちる。**07 の走行 (`scratch/qwen38/mem07-4096.log`) も同じ形** (t = 44〜48 s で wired 9.6 → 12.7 GB、RSS 1.6 → 0.6 GB) で、file-backed の最小が 1.77 GB、Swapouts 0 だった。
- 始める前の file-backed は 07 が 12.2 GB、今回が 9.7 / 12.3 GB (2 本目 / 1 本目)。この変更で増えた常駐は `acts` の触ったページ (最大 126 MB、導出) だけのはず。
- **原因は未確認。** 変更前の経路 (`Q38_GEMM_MIN_PAIRS=0`) の 8K チャンク 4096 は今日は流していない (3 本目の Swapouts を避けた)。

## 5. 次

1. **2 チャンク目で wired が約 4.5 GB 跳ねる中身を見る** (32K に行く前に)。n = 8192 列の注意の一時領域 (`attnScores` など、03 §6-3 の 2 の 790 MB の見積もり)・インデクサ鍵・KV の伸びのどれか。
   `ensureBuffer` の確保を大きさ付きで出し、wired の跳ねと時刻を合わせる。1 チャンク目で済む形 (行を分けて一時領域を n に比例させない) にできるかを見る。
2. routed の残り (チャンク 4096 で 9.4 s): 展開と sgemm。組の多い expert は展開 0.2 ms が組数で割られるので、残りはほぼ sgemm。
3. route 7〜8 s (読み 5 s・top-10 1.3 s)、PLE、pre-router の GPU 9.5〜11.3 s (07 §5 の 3)。
4. 32K を 1 回 (1 が片付いてから)。

## 6. 足したもの

| 種類 | パス |
| --- | --- |
| ランナー | `moeRouted` の slot の並べ直し、`routedGemm` の per-pair 側 (`bigSlots`・`rowBytes` 引数)、`gemmMinPairs` (`Q38_GEMM_MIN_PAIRS`、既定 24、0 で 07 と同じ)、バッファ `gemmPosSlot`・`gemmOnes` |
| カーネル | なし (`moe_iq2xxs_phase1_gate_up_act`・`moe_q2k_phase2_down_reduce` を top_k 1 で流用) |

## 7. 落とし穴

- **`ensureBuffer(bytes: 0)` はバッファを作らない** (`nil < 0` が偽)。後で `!` を付けると release ビルドでは何も言わずに exit 133 (SIGTRAP) で落ちる。検査スクリプトの grep 越しだと出力が空になるだけに見える。
- 流用した phase 2 は residual を足すので、出力先を 0 にしてから流す (§1)。
- `memlog.sh` 越しのベンチは標準出力がバッファされ、kill されると 1 チャンク目の行も残らない。
- zsh の `Monitor` スクリプトで `$D/out*.txt(N)` は `no matches found` で落ちる (`ls ... 2>/dev/null` にする)。

## 8. 再開手順 (新しいセッションで続けるとき)

07 §8 の後継。**07 §7・06 §7・05 §6-4・03 §6-4 の落とし穴はそのまま有効**なので、先に一度読む。

### 8-1. 状態

- コードはこの文書と同じ変更まで入っている。ランナーは `Sources/Tsugumi/Runtime/Qwen38/Qwen38Runner.swift`。
  - 注意: T ≥ 32 で選択なしは `attentionSgemm`、選択ありは `attentionSelected` (04)、T < 32 は `attentionHost`。
  - expert の読み: T ≥ 32 は `GGUFFile.preadRanges` (4 スレッド)、T < 32 は `F_RDADVISE` (05)。
  - routed expert: T ≥ 1024 は `routedGemm` (06)、その中で組 < 24 の expert は per-pair カーネル (08)。T < 1024 は per-pair カーネル。
  - GDN step: T ≥ 16 は `Qwen38GDNChunk` (チャンク 32)、それ未満は `q38_gdn_step` (07)。
- 速度の現在地 (8,192 トークン、n=1): チャンク 2048 で 87.4 s (93.8 tok/s、footprint 3.70 GB)。**チャンク 4096 は Swapouts で完走していない** (§4-1)。4,096 トークン 1 チャンクは 34〜39 s。
  decode は短文脈 0.09〜0.11 s/トークン、8K の直後 0.44 s。
- **32K はまだ流していない。**
- ウェイト・トークナイザの在処は [01 §7-1](01-Q2-FIRST-LIGHT.md)。`G=~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf`。
- `scratch/qwen38/` (git 管理外): 07 §8-1 のもの。
- 運用点は thinking 無効・32K・エージェント主体・英語主体 (メモリ `qwen38-operating-point`)。

### 8-2. 検査を一通り

```bash
swift build -c release --product TsugumiKernelCheck
B=.build/release/TsugumiKernelCheck; S=scratch/qwen38
Scripts/qwen38/guarded.sh checks.out $S/checks07.sh
grep -c PASS checks.out; grep -i fail checks.out  # 24 と空
for mp in 1000 3 0; do
  Q38_GEMM_MIN_T=1 Q38_GEMM_MIN_PAIRS=$mp $B --qwen38-prefill $S/ref-fuji-idx16.log --q38-ref-logits $S/ref-fuji-idx16.logits --q38-indexer-top-k 16 --q38-chunk 53 | tail -2
  sleep 20
done
```

期待値は §3 の表。

速度 (約 1.5 分、memlog 越しに、間に 20 秒)。**チャンク 4096 の 8K は §5 の 1 が片付くまで流さない。**

```bash
scratch/qwen38/memlog.sh mem.log out.txt /usr/bin/time -l env Q38_SPLIT_PRE=1 $B --qwen38-prefill-bench $S/prompt-code.tokens --q38-tokens 8192 --q38-chunk 2048
grep -E "\.\.<|prefill|decode|footprint|GUARD" out.txt; pgrep -x TsugumiKernelCheck || echo none
```

§4 の表と比べる。

### 8-3. 次にやること

§5 の順 (2 チャンク目の wired の跳ね → routed の残り → route / PLE / pre-router → 32K)。

### 8-4. 落とし穴

§7 と、07 §7・06 §7・05 §6-4・03 §6-4。
