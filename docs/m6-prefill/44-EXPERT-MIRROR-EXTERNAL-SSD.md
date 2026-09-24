# 44. 層ファイルの写しを外付け SSD に置き、エキスパートの 32% をそちらから読む — 自然文の prefill が 0.9〜1.2 s 縮む (実測、2026-09-25)

**候補の測定 (`TF_EXPERT_MIRROR`)。台帳 §1 は動かない。**[43](43-LEDGER-PROMPT-REPEATED-SENTENCE.md) で、自然文の prefill の大半は SSD の読み (1 run 9〜13 GiB) だと分かった。M6 の内蔵 SSD は 3.3 GB/s で頭打ち ([M6_SSD_BANDWIDTH](../investigations/M6_SSD_BANDWIDTH.md))。
外付け (TB4、`SSD256`) を**内蔵と同時に読んで、帯域を足す**。

- 外付けに `packed_experts/` の写しを置く (`/Volumes/SSD256/tsugumi-mirror/packed_experts`、31 本すべて manifest の SHA-256 と一致を確かめた)。
- 各層で、エキスパート番号に均等に散らした 32% (128 個中 40 個、`ExpertMirror.evenAssignment`) を写しのマッピングから、残りを内蔵から読む。どのエキスパートも常に同じ側から読むので、同じバイトがページキャッシュに 2 度載ることは無い。`F_RDADVISE` も同じ振り分けで 2 本の fd に交互に出す。
- 開き方は `TF_EXPERT_MIRROR=<dir>` と `TF_EXPERT_MIRROR_SHARE` (既定 0.32)。写しのファイルは manifest のサイズと照合し、`full-sha256` なら SHA-256 も検証する。mmap の腕 (既定) と pread の腕の両方に入れた。

証拠 (M6、`bench/m6/results/2026-09-25-Mac18-5-<ラベル>/`): `dual-ssd-probe` (帯域)、`mirror-a` (交互の並び)、`mirror-block`・`mirror-block-s25`・`mirror-block-s40` (腕ごとのまとまりの並び)。

```
測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit f6cfb76 (dual-ssd-probe)
測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 5eb5238 / minos 15.0 (mirror-a)
測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 3b79fde / minos 15.0 (mirror-block、Sources は 5eb5238 と同じ)
```

```
def: TsugumiCLI --model scratch/gemma4-qat-sym.gturbo --messages-file bench/m6/prompts/<本文>.json --expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 --max-context 4096 --verification trusted-install
mir: TF_EXPERT_MIRROR=/Volumes/SSD256/tsugumi-mirror/packed_experts TF_EXPERT_MIRROR_SHARE=0.32 <def>
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
- [TWO_MACHINE_DEV §5](../TWO_MACHINE_DEV.md) の「測定は必ず内蔵から走らせる」は、既定の数字を外付けで汚さないための規則で、変えない。写しの腕はそれと別の構成として、`TF_EXPERT_MIRROR` を `env` に残して測る。

## 3. 次に渡すもの

- **既定に入れるかは未決。**外付けがある機械でしか効かないので、`TF_` ではなく CLI / サーバの引数 (たとえば `--expert-mirror <dir>`) と、写しを作って検証するコマンドが要る。入れるなら台帳の行は「写しあり」の構成として別に取る。
- 写しは 12 GB の複製である (AGENTS の「.moepack を複製しない」はテストのための複製の話で、これは構成の一部)。
- decode は測っていない。greedy 64 トークンの decode は 33.5 → 32.2 tok/s (n = 1、解釈しない)。
- 割合は 3 点しか測っていない。0.28〜0.36 の間はまだ測っていない。
