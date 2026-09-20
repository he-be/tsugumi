# 09. Tsugumi の Online を最後まで流す — 14 ターンとも回答まで届く。会話の切り替えでスワップが出る

実測: 2026-09-20、M3 Pro 18 GB、macOS 15、`iogpu.wired_limit_mb` = 14336。
表記は [qwen38/01](../qwen38/01-Q2-FIRST-LIGHT.md) と同じ (実測 / 導出 / 未確認)。
**どの数字も 1 セット (各ターン 1 回)** で、解釈は書かない。

[08](08-PP-TG-CEILING.md) §6-1 の続き。[05](05-BONSAI-PQ2-0.md) §6-1 は Online を 5 ターンで打ち切っていた。
ここでは [07](07-PQ2-0-SMALL-BATCH-KERNEL.md) のパッチ入りの `llama-server` に MTP 同梱版 (n_max 1) を載せ、
`TsugumiToolLoopCheck --network online` で 48 の 6 会話 (14 ターン) を最後まで流した。

## 0. 結論

1. **14 ターンとも回答まで届いた。**`answer` と `error` は 14 / 14、`structured_output_failure` は 0、
   ラウンド上限で止まった回 (`stopped`) も 0。`online` は 12 / 14 (§2)。
2. **`online` が不成立の 2 ターンは、検索せずに答えた回。**ja-news turn 3 (「ここまでの内容を 5 行で要約して」、
   05 §6-1 と同じ) と en-url turn 1 (URL 付きの問いに 1 ラウンドで回答) (§2)。
3. **14 ターンの合計は 1,465 秒 (1 ターン 37〜162 秒)。**05 §6-1 と同じ 4 ターンは
   190 → 162、237 → 148、64 → 53、153 → 132 秒 (§2)。
4. **decode は 15.2〜17.9 tok/s、MTP の受理率は 83.5%** (4,875 / 5,836) (§3)。
5. **prefill は文脈が短いところで 91.5 tok/s、16K を超えると 80.8 tok/s** (§3)。
   1 ターンの壁時計のうち prefill が 23〜107 秒を占める。
6. **会話を切り替えた直後にスワップが出る。**同じサーバで次の会話に入ると、2 回とも 1 分以内に
   見張りの閾値 (60 秒で +512 MiB) を超えた: +121,616 ページ (1.86 GiB) と +39,080 ページ (0.60 GiB)。
   会話の中では 14 ターンとも +0。**原因は調べていない** (§4)。
7. このため残り 4 会話は 1 会話ごとにサーバを立て直して流した。**同じサーバで 6 会話を続けて流す形は通っていない。**

## 1. 流し方

```
Scripts/qwen38_27b/bonsai_online_run.sh          # OUT= / SPEC= / ONLY= で切り替え
```

- サーバ: `~/LLM/prism-llamacpp/src-b10709/build/bin/llama-server` (07 のパッチ入り)、
  `-ngl 99 -fa on -c 32768 -np 1 --jinja --spec-type draft-mtp --spec-draft-n-max 1`、
  重みは `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` (7.66 GB)。
- 中継: `Scripts/qwen38_27b/upstream_relay.py`。`TsugumiToolLoopCheck --endpoint` は llama-swap の
  `/upstream/<id>/<path>` を叩くので `/<path>` に送り直す (05 §6 では session の scratch 置きで、消えていた)。
- 検査: 05 §6-1 と同じ引数 (`--network online --max-rounds 6 --thinking off --context 32768
  --web-store scratch/bonsai27b/web --pin-search --search-budget 0`)。**Serper は 0 回** (`search budget 0/0`、
  `fetched 0`)。store に無いページの取得は 9 回あった (`recorded` の合計)。
- **サンプリングは公式値とずれていた** ([10](10-GUI-PLAN.md) §1-3、あとから分かった)。遠隔クライアントが送るのは
  temperature 0.7 / top_p 0.80 / top_k 20 だけで、`presence_penalty` と `min_p` はサーバの既定 (0.0 / 0.05) で走った。
  公式は 1.5 / 0.0。05 §6 の実行も同じ。
- 見張り: Swapouts が 60 秒で +512 MiB か合計 +1 GiB でサーバも検査も止める。1 分ごとに RSS・空き率・
  メモリ上位 6 プロセスを `watch.log` に残す。

## 2. ターンごとの結果 (実測)

| 会話 | ターン | ラウンド | 秒 | うち prefill | decode | 最大 prompt | checks | 05 §6-1 (MTP なし) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ja-news | 1 | 7 | 162 | 107 s | 921 tok / 17.0 tok/s | 9,984 | ok | 190 秒 |
| ja-news | 2 | 7 | 148 | 75 s | 1,190 / 16.3 | 16,537 | ok | 237 秒 |
| ja-news | 3 | 3 | 53 | 32 s | 318 / 15.2 | 19,162 | `online` 不成立 | 64 秒 (同じく不成立) |
| en-swift | 1 | 7 | 132 | 82 s | 843 / 17.2 | 7,573 | ok | 153 秒 |
| en-swift | 2 | 3 | 67 | 27 s | 632 / 16.0 | 10,515 | ok | 54 秒 (不成立) |
| en-swift | 3 | 7 | 89 | 59 s | 502 / 16.8 | 16,160 | ok | — |
| ja-howto | 1 | 5 | 137 | 56 s | 1,413 / 17.4 | 5,314 | ok | — |
| ja-howto | 2 | 7 | 116 | 65 s | 849 / 16.6 | 11,124 | ok | — |
| en-url | 1 | 1 | 37 | 25 s | 218 / 17.9 | 2,338 | `online` 不成立 | — |
| en-url | 2 | 7 | 53 | 23 s | 515 / 17.2 | 4,735 | ok | — |
| ja-price | 1 | 7 | 148 | 92 s | 928 / 16.9 | 8,508 | ok | — |
| ja-price | 2 | 7 | 137 | 48 s | 1,298 / 15.3 | 12,816 | ok | — |
| en-facts | 1 | 6 | 89 | 66 s | 399 / 17.7 | 6,247 | ok | — |
| en-facts | 2 | 7 | 98 | 64 s | 527 / 17.1 | 12,132 | ok | — |

- 合計 1,465 秒。`error` は 14 ターンとも無し。
- 05 §6-1 の列は別の実行 (MTP なし、ビルド済みバイナリ) で、サンプリングは temperature 付きなので生成長も違う。
  **同じ会話を並べただけで、比は取らない。**
- 回答の中身 (事実の当否、表記) はここでは採点していない。

## 3. 速度 (実測、サーバログ)

prefill は、1 回に 300 トークン以上を処理した 48 回を処理後の文脈長で分けた。

| 処理後の文脈長 | 回数 | prefill 中央値 | 幅 |
| --- | ---: | ---: | --- |
| 〜4K | 12 | 91.5 tok/s | 86.5〜94.3 |
| 4K〜8K | 11 | 89.9 | 87.0〜91.9 |
| 8K〜12K | 13 | 86.1 | 83.5〜90.2 |
| 12K〜16K | 7 | 83.5 | 82.2〜84.4 |
| 16K〜24K | 5 | 80.8 | 77.4〜82.8 |

- どれも prompt cache が効いた状態で 1 回に 700〜1,500 トークンを足した値。
  **長い prompt を頭から 1 本で入れた値は測っていない。**
- 最初のラウンド (cache なし 1,831 トークン) の prefill は 19.6 秒、以降のラウンドは 7〜23 秒。
- MTP の受理は 4,875 / 5,836 = 83.5% (全実行の合計)。
- サーバの RSS は 9.95〜11.22 GB、空き率は会話の中で 19〜25%。

## 4. 会話の切り替えでスワップが出る (実測、2 回)

| 実行 | 切り替え | 直前の文脈 | サーバの `f_keep` | 次の 1 分の Swapouts | 空き率 |
| --- | --- | ---: | ---: | ---: | --- |
| `online-mtp-set1` | ja-news → en-swift | 19,410 | 0.093 | +121,616 ページ (1.86 GiB) | — |
| `online-mtp-set1b` | en-swift → ja-howto | 16,400 | 0.110 | +39,080 ページ (0.60 GiB、切り替えの 11 秒後の値) | 25% → 10% |

- 2 回とも見張りが止めた。止めた時点のメモリ上位は `llama-server` 9.95 GB だけで、次は 0.26 GB。
- 会話の中 (切り替えなし) の Swapouts は、1 会話 1 サーバの 4 実行を含めて全部 +0。
- **原因は調べていない。**サーバログには切り替えの前後にキャッシュの保存や確保を示す行が無い。
  `llama-server` の既定は `--cache-ram 8192`・`--ctx-checkpoints 32` のまま。
- アプリではチャットを切り替えるたびにこの場面が来る。

## 5. 手元に置いたもの

| 置き場 | 中身 |
| --- | --- |
| `scratch/bonsai27b/runs/online-mtp-set1` | ja-news 3 ターン + en-swift の途中 (見張りが停止) |
| `scratch/bonsai27b/runs/online-mtp-set1b` | en-swift 3 ターン + ja-howto の途中 (見張りが停止) |
| `scratch/bonsai27b/runs/online-mtp-set1-{ja-howto,en-url,ja-price,en-facts}` | 1 会話 1 サーバの 4 実行 |

§2 の表は en-swift を `set1b`、ja-news を `set1`、残りを会話ごとの実行から取った。

## 6. 次に決めること

1. **§4 のスワップをどうするか。**原因を調べるか、この形 (llama-server + 32K + MTP) を運用の候補から外すか。
2. 2 セット目を流すか。ここは 1 セットで、サーバは非決定的。
3. → [10](10-GUI-PLAN.md) に GUI へ入れる計画。Mac アプリ本体からこのサーバへつなぐ経路は無い (アプリの推論クライアントは `RealInferenceClient` と
   `DecodeServiceInferenceClient` のローカル 2 つで、遠隔の `RemoteInferenceClient` は `TsugumiToolLoopCheck` にしか無い)。
   ここで通したのは、アプリと同じ Online のツールループを `TsugumiToolLoopCheck` から回した形。
