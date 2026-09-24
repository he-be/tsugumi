# 44. 層ファイルの写しを外付け SSD に置き、エキスパートの 32% をそちらから読む — 既定にした。自然文の prefill が 0.5〜0.9 s 縮む (実測、2026-09-25)

**既定に入れた (§4)。**§1〜§3 は候補 (`TF_EXPERT_MIRROR=<dir>`、5eb5238) のときの測定。[43](43-LEDGER-PROMPT-REPEATED-SENTENCE.md) で、自然文の prefill の大半は SSD の読み (1 run 9〜13 GiB) だと分かった。M6 の内蔵 SSD は 3.3 GB/s で頭打ち ([M6_SSD_BANDWIDTH](../investigations/M6_SSD_BANDWIDTH.md))。
外付け (TB4、`SSD256`) を**内蔵と同時に読んで、帯域を足す**。

- 外付けに `packed_experts/` の写しを置く (`/Volumes/SSD256/tsugumi-mirror/packed_experts`、31 本すべて manifest の SHA-256 と一致を確かめた)。
- 各層で、エキスパート番号に均等に散らした 32% (128 個中 40 個、`ExpertMirror.evenAssignment`) を写しのマッピングから、残りを内蔵から読む。どのエキスパートも常に同じ側から読むので、同じバイトがページキャッシュに 2 度載ることは無い。`F_RDADVISE` も同じ振り分けで 2 本の fd に交互に出す。
- 既定の形 (§4): `Scripts/make-expert-mirror.sh <model> <mirror dir> [share]` が写しを作って原本と `cmp` で照合し、モデルのディレクトリに `expert-mirror.json` (置き場・割合・manifest の SHA-256) を書く。読み込みはそれがあれば写しを使い、manifest が違う・置き場が無い (未マウント) なら警告して内蔵だけで読む。`TF_EXPERT_MIRROR=0` で外す。mmap の腕 (既定) と pread の腕の両方に入っている。

証拠 (M6、`bench/m6/results/2026-09-25-Mac18-5-<ラベル>/`): `dual-ssd-probe` (帯域)、`mirror-a` (交互の並び)、`mirror-block`・`mirror-block-s25`・`mirror-block-s40` (腕ごとのまとまりの並び)。

```
測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit f6cfb76 (dual-ssd-probe)
測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 5eb5238 / minos 15.0 (mirror-a)
測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 3b79fde / minos 15.0 (mirror-block、Sources は 5eb5238 と同じ)
```

```
def: TsugumiCLI --model scratch/gemma4-qat-sym.gturbo --messages-file bench/m6/prompts/<本文>.json --expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 --max-context 4096 --verification trusted-install
mir: TF_EXPERT_MIRROR=/Volumes/SSD256/tsugumi-mirror/packed_experts TF_EXPERT_MIRROR_SHARE=0.32 <def>   (候補のときの形。既定では expert-mirror.json)
```

全 run exit 0、Swapouts 0。本文は 43 §3 の自然文 5 本 (`bench/m6/prompts/`)。各 run で `iostat -d -I` の累積 MB を前後に取り、disk0 (内蔵) と disk4 (外付け) が何を読んだかを差で出した。

## 0. 結論

1. **同時に読むと帯域は足し算になる。**3.36 MB のランダム読み (`F_NOCACHE`) で、内蔵だけ 3.30〜3.38 GB/s、外付けだけ 1.57 GB/s、同時には内蔵 3.26〜3.35 + 外付け 1.57 GB/s で、互いに落とさない (§1.1)。
2. **腕ごとにまとめて流すと (1 人の利用者の形)、5 本すべてで 0.9〜1.2 s 縮む** (§1.2、各 n = 4)。1 run の Pageins は変わらず (9〜13 GiB)、そのうち 2.5〜3.7 GiB を外付けが読んでいる。縮んだのは読む量ではなく、読む速さ。
3. **割合は 0.32 がよい。**0.25 は縮みが 0.66〜1.49 s で 5 本中 4 本が 0.32 より小さく、0.40 は縮まない (外付けが律速になり、Pageins も 14 GiB に増える、§1.3)。
4. 計算は変わらない。greedy 64 トークン (`ja-twomachine`、`--temperature 0`) の本文は両腕で同じ (`mirror-a/greedy-{def,mir}.log`)。

## 1. 数字

### 1.1 帯域 (`dual-ssd-probe`、`rec_probe` 3,358,720 B、GB/s、2 巡)

| QD (各 SSD) | 内蔵だけ | 外付けだけ | 同時: 内蔵 + 外付け |
| ---: | --- | --- | --- |
| 4 | 3.379、3.305 | 1.566、1.572 | 3.351 + 1.573、3.257 + 1.572 |
| 8 | 2.992、3.299 | 1.572、1.570 | 3.277 + 1.569、3.298 + 1.571 |

外付けの p50 は QD4 で 8.5 ms、QD8 で 17.1 ms (QD 2〜4 で飽和)。割合 0.32 は 1.57 / (3.3 + 1.57) から置いた (導出)。

### 1.2 prefill 壁時計 (c)、`mirror-block` (share 0.32、各 n = 4 の中央値 [最小〜最大])

| 本文 | 既定 | 写し 0.32 | 差 s (導出) | Pageins GiB 既定 / 写し | 外付けの読み GiB |
| --- | ---: | ---: | ---: | --- | ---: |
| ja-twomachine | 4.670 [3.667〜5.090] | 3.550 [2.630〜4.441] | −1.120 | 12.9 / 11.3 | 3.7 |
| ja-m6readme | 3.996 [3.683〜4.214] | 2.978 [2.599〜3.616] | −1.018 | 10.7 / 9.4 | 2.5 |
| en-readme | 4.429 [4.210〜5.681] | 3.339 [3.188〜3.656] | −1.090 | 11.1 / 11.1 | 3.1 |
| en-server | 4.762 [4.576〜4.983] | 3.576 [3.527〜3.653] | −1.186 | 12.6 / 12.5 | 3.6 |
| code-runner | 4.489 [4.407〜4.827] | 3.587 [3.521〜3.602] | −0.902 | 12.0 / 12.2 | 3.5 |

並びは 1 巡 = 片方の腕で 5 本、もう片方で 5 本 (先の腕を巡ごとに入れ替え)。腕が替わった直後の 1 本目 (`ja-twomachine`) は前の腕のキャッシュが使えず、両腕とも遅い側に出る (既定 5.06・5.09、写し 4.42・4.44)。
交互の並び (`mirror-a`、run ごとに腕を替える、各 n = 3) でも写しが 5 本すべてで短い (既定 5.18〜6.10 → 写し 4.21〜4.72) が、どちらの腕も 1 run 15〜16 GiB 読んでいて、1 人の利用者の形ではない。

**43 §1 の既定 (`ja-twomachine` 2.595 s など) とは並びが違うので比べない。**43 は同じ本文を 4 回続けて流していて、2 回目以降はその本文のページがキャッシュに残る。ここでは run ごとに本文を替えている (43 §3 の規則 1)。

### 1.3 割合 (`mirror-block-s25`・`-s40`、同じ並び、各 n = 4 の中央値)

| 本文 | 0.25: 既定 → 写し | 0.32: 既定 → 写し | 0.40: 既定 → 写し (写しの Pageins GiB) |
| --- | --- | --- | --- |
| ja-twomachine | 4.612 → 3.798 | 4.670 → 3.550 | 4.970 → 4.762 (14.0) |
| ja-m6readme | 4.011 → 3.353 | 3.996 → 2.978 | 3.952 → 4.323 (13.4) |
| en-readme | 5.104 → 3.612 | 4.429 → 3.339 | 4.216 → 4.440 (13.9) |
| en-server | 4.611 → 3.871 | 4.762 → 3.576 | 4.551 → 4.518 (14.2) |
| code-runner | 4.581 → 3.867 | 4.489 → 3.587 | 4.489 → 4.489 (14.1) |

0.40 で写しの Pageins が既定より 2〜3 GiB 増える理由は調べていない。

## 2. 分かったこと / 外れたこと

- [M6_SSD_BANDWIDTH](../investigations/M6_SSD_BANDWIDTH.md) §0-5「外付けで逃げられない」は、外付け**だけ**で読む場合の話だった。外付けは内蔵より遅い (1.57 GB/s) が、別の経路なので**足す**ことはできる。
- [TWO_MACHINE_DEV §5](../TWO_MACHINE_DEV.md) の「測定は必ず内蔵から走らせる」は、既定にした時点で「既定は内蔵 + 写し、写しなしは `TF_EXPERT_MIRROR=0`」に書き換えた。

## 3. 候補のときに残した問い (§4 で答えた)

- 既定の入れ方 → モデルのディレクトリの `expert-mirror.json` (§4)。引数は足していない。CLI・サーバ・アプリが同じ `Model.load` を通るので、どれも写しを使う。
- 写しは 12 GB の複製である (AGENTS の「.moepack を複製しない」はテストのための複製の話で、これは構成の一部)。
- 割合は 3 点しか測っていない。0.28〜0.36 の間はまだ測っていない。

## 4. 既定にした (9d66daa)

証拠: `default-mirror-block` (trusted-install)、`default-mirror-block-sha` (full-sha256)、`ledger-expert-mirror`。腕は `def` (既定のバイナリ、`TF_` なし) と `off` (`TF_EXPERT_MIRROR=0`)。並びは §1.2 と同じ、各 n = 4。全 run exit 0、Swapouts 0。

```
測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 9d66daa / minos 15.0
bench/m6/mirror_bench.sh default-mirror-block 4 block                (--verification trusted-install)
bench/m6/mirror_bench.sh default-mirror-block-sha 4 block full-sha256
```

| 本文 | trusted: 写しなし → 既定 (差) | full-sha256: 写しなし → 既定 (差) |
| --- | --- | --- |
| ja-twomachine | 4.316 → 3.720 (−0.596) | 4.479 → 3.856 (−0.623) |
| ja-m6readme | 3.790 → 3.316 (−0.474) | 4.261 → 3.622 (−0.639) |
| en-readme | 4.111 → 3.497 (−0.614) | 4.373 → 3.875 (−0.498) |
| en-server | 4.492 → 3.574 (−0.918) | 4.633 → 3.941 (−0.692) |
| code-runner | 4.440 → 3.570 (−0.870) | 4.510 → 3.931 (−0.579) |

- trusted の縮みは §1.2 (0.9〜1.2 s) より小さい。§1.2 の既定 (写しなし) の腕が 3.996〜4.762 s で、今回の 3.790〜4.492 s より遅かった。同じ構成の別の時間帯で 0.2〜0.3 s 動くので、差の大きさは読まない。向きは 5 本とも両方の測定で同じ。
- greedy 64 トークンの本文は 4 本 (両腕 × 両検証) で同じ (`greedy-{def,off}.log`)。
- **full-sha256 の検証は、各エキスパートを読む側から繋いだバイト列を manifest の SHA-256 と照合する**。写しの壊れは、そのバイトを読むときにだけ検出される (読まない側の壊れは使われない)。最初の実装 (2 本の fd を交互に 1 MB ずつ `pread`) は先読みが効かず、full-sha256 の prefill が 3.9〜5.3 → 7.2 s に伸びた (n = 1)。16 MB の窓を 2 台同時に読み、前の窓のハッシュと重ねる形にして上の表になった。peak は 1.27 → 1.34〜1.35 GB (窓 64 MB では 1.60 GB)。
- 台帳 (繰り返し文) の行は 1.768 → 1.716 s (`ledger-expert-mirror`、1 本目 3.747 s を含む n = 6)。繰り返し文なので進捗の判定には使わない ([43](43-LEDGER-PROMPT-REPEATED-SENTENCE.md))。

**運用**: M6 は再起動のたびに `diskutil mount disk5s1` が要る ([TWO_MACHINE_DEV §5](../TWO_MACHINE_DEV.md))。マウントし忘れると警告を出して内蔵だけで読む (遅いほうに戻るだけで止まらない)。写しなしの数字は `TF_EXPERT_MIRROR=0` を付けて取る。decode は測っていない。
