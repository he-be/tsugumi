# 19. sliding window の attention 本体を simdgroup_matrix + GQA 2 ヘッド共有に — GPU の attn.swa 1.433 → 0.866 s、既定構成の壁時計は縮まず、候補 4 つの構成では 4.565 → 4.290 s (実測、2026-09-24)

証拠 (M6):
[`sgmm-profile-detail-2`](../../bench/m6/results/2026-09-24-Mac18-5-sgmm-profile-detail-2/)、
[`sgmm-a`](../../bench/m6/results/2026-09-24-Mac18-5-sgmm-a/)、
[`best4-a`](../../bench/m6/results/2026-09-24-Mac18-5-best4-a/)、
[`best4-b`](../../bench/m6/results/2026-09-24-Mac18-5-best4-b/)、
[`best4-sgmm-a`](../../bench/m6/results/2026-09-24-Mac18-5-best4-sgmm-a/)、
[`best4-sgmm-b`](../../bench/m6/results/2026-09-24-Mac18-5-best4-sgmm-b/)。
比べる先: [`h0-profile-detail`](../../bench/m6/results/2026-09-24-Mac18-5-h0-profile-detail/)、
[`ledger-head`](../../bench/m6/results/2026-09-24-Mac18-5-ledger-head/)、
[`sgmm-profile-detail`](../../bench/m6/results/2026-09-24-Mac18-5-sgmm-profile-detail/) (旧 sgmm)。
ホスト札 (今回の 6 ディレクトリ共通):

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 3f042e7 / minos 15.0
```

比べる先の `h0-profile-detail` と `ledger-head` は commit 22d7629、`sgmm-profile-detail` は 71786c5 (ホスト札の他の欄は同じ)。

コマンド (全ディレクトリ共通、差は `env.txt` の `TF_` だけ):

```
/Users/mh/dev/tsugumi/.build/release/TsugumiCLI --model /Users/mh/dev/tsugumi/scratch/gemma4-qat-sym.gturbo --messages-file /Users/mh/dev/tsugumi/bench/l.json --expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 --max-context 4096 --verification full-sha256
```

「候補 4 つ」は `TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=2` ([06 §2](06-LEDGER.md) の最良の行と同じ組)。

**台帳** ([06](06-LEDGER.md)): この変更は候補 (`TF_PREFILL_ATTN=sgmm` で開く)。台帳 §1 は動かない。

## 0. 結論

1. **GPU の attn.swa は 1.433 s → 0.866 s** (profile level 2、各 n = 3、差 −0.567 s **導出**)。
2. **既定構成 + `TF_PREFILL_ATTN=sgmm` の prefill 壁時計は 8.829 s** (n = 6)。台帳の既定 8.682 s (別 commit 22d7629) より縮んでいない。
3. **候補 4 つの構成では、同じ commit で交互に取って 4.565 s → 4.290 s** (各 n = 6、差 −0.275 s **導出**)。

## 1. 数字

prefill 壁時計は `summary.tsv` の `prefill_s`。測った量は新しいプロセスでの 1 回目 ([04 B0](04-GATES.md) の (c))。

| | 変更前 | 変更後 | 差 s (**導出**) |
| --- | ---: | ---: | ---: |
| attn.swa GPU 秒 (profile level 2、n = 3 ずつ) | 1.433 (1.432〜1.434) `h0-profile-detail` | 0.866 (0.865〜0.868) `sgmm-profile-detail-2` | −0.567 |
| GPU 合計 (同上) | 7.499 (7.489〜7.531) | 7.002 (6.973〜7.021) | −0.497 |
| 既定構成の prefill 壁時計 (n = 6) | 8.682 (8.633〜8.801) `ledger-head` | 8.829 (8.626〜8.957) `sgmm-a` | +0.147 |
| 同 prefill io | 0.780 | 0.959 | +0.179 |
| 同 layerVerify | 4.589 | 4.646 | +0.057 |
| 候補 4 つの prefill 壁時計 (a+b、n = 6) | 4.565 (4.473〜5.181) `best4-a/b` | 4.290 (4.110〜5.203) `best4-sgmm-a/b` | −0.275 |
| peak_gb (各構成の中央値) | 1.21〜1.22 | 1.43〜1.44 | +0.22 |

- 旧 sgmm (71786c5) の attn.swa は 1.639 s (1.638〜1.648、`sgmm-profile-detail`)。
- 候補 4 つの 2 セット (a、b) では、どちらも 1 本目が遅い (best4-a 5.181 s、best4-sgmm-a 5.203 s、どちらも layerVerify 9.5 s)。中央値にはそのまま入れた。
- 既定構成の比較 (8.682 と 8.829) は commit が違い、交互にも取っていない。候補 4 つの比較は同じ commit で交互に取った。

## 2. カーネルの変更 (3f042e7)

`attention_prefill_causal_sgmm_gqa_d256` を追加した。旧 sgmm (71786c5) との違いは次のとおり。

- KV ヘッドを共有する 2 つの query ヘッドを 1 threadgroup (8 simdgroup、256 スレッド) にまとめ、K/V タイルを共有する。
- Q は threadgroup メモリに置かず、各レーンが自分の断片の要素を device から直接読んでレジスタに持つ。threadgroup メモリは 25 KB から K と V の 2 枚 (計 16.9 KB) に減った。
- K と V を 1 回のフェーズで読み込み、barrier はキータイル 1 枚あたり 4 回から 2 回になった。
- ホスト側では、headDim 256 かつ `numQHeads == 2 × numKVHeads` のときだけ `TF_PREFILL_ATTN=sgmm` がこれを使う。それ以外のヘッド比は旧 sgmm のまま。

一致の確認:
- MBP で `Scripts/test.sh --filter PrefillAttentionTests` の 13 テストが通った。16/16 ヘッドで旧 sgmm を通すテストを 1 本足した。
- M6 ではテストを組めない (`TestingMacros` の plugin が無い、[05 §5](05-RISKS.md))。M6 では単体ベンチの出力比較 (qblock との最大差 3.05e-5) だけ。

単体ベンチ (`bench/m6/attn_bench.swift`、25 層 × 2 チャンク、**記録値ではない**) の中央値:

| | qblock | 旧 sgmm | GQA 版 |
| --- | ---: | ---: | ---: |
| M6 | 1.434 | 1.627 | 0.853 |
| MBP | 1.732 | 1.118 | 0.525 |

## 3. 分かったこと / 外れたこと

- 再開手順 2 の見立てどおり、M6 で旧 sgmm が qblock より遅かったのはカーネルの作りによるもので、直すと M6 でも qblock を下回った。どの変更がどれだけ効いたかは分けて測っていない。
- **GPU で 0.567 s 縮んでも、既定構成の壁時計には出なかった。**候補 4 つの構成 (層の検証を先行・並列にする) では −0.275 s 出た。既定構成で何が律速かは測っていない (未確認)。
- 最初の版には、チャンク末尾の範囲外の行があるレーンだけが行列演算の前で `continue` する不具合があった。末尾 6 行が最大 0.04 ずれ、単体ベンチの出力比較で見つかった (テストはこの版では回していない)。
- peak_gb が約 0.22 GB 増えた (1.21〜1.22 → 1.43〜1.44)。旧 sgmm の `sgmm-profile-detail` は 1.25。原因は調べていない。

## 4. 次の Gate に渡すもの

- 既定にするかはユーザーの判断 ([04 B3](04-GATES.md))。入れるなら `ledger-*` の行を取り直す。
- 既定構成で GPU の短縮が壁時計に出ない理由 (未確認)。
- peak_gb の +0.22 GB (未確認)。
- greedy の差 (`--max-new 64 --temperature 0`) は取っていない。
