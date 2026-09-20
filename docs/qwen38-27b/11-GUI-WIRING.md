# 11. Bonsai 2 27B を Mac アプリから選べるようにする — 子サーバの起動・停止と種別の振り分け

実測: 2026-09-20、M3 Pro 18 GB、macOS 15、`iogpu.wired_limit_mb` = 14336。
表記は [qwen38/01](../qwen38/01-Q2-FIRST-LIGHT.md) と同じ (実測 / 導出 / 未確認)。
回数は節ごとに書く (1 回か 2 回)。解釈は書かない。言葉は [10](10-GUI-PLAN.md) の冒頭と同じ
(遠隔クライアント = `RemoteInferenceClient`、子サーバ = アプリが自分で起動して止める `llama-server`)。

[10](10-GUI-PLAN.md) の S1〜S5 を実装し、S6 の検査を流した。**GUI そのものはまだ触っていない** (§5)。

## 0. 結論

1. **S6 の検査は 3 つとも通った。**14 ターン × 2 セットを、6 会話とも**同じ子サーバで続けて**流しきった
   (`answer`・`error` 28/28、Swapouts 合計 +0、§6)。会話の切り替えのスワップは `--cache-ram 0` で 2 回とも +0 になり、
   `manifest.json` に入れた (§5)。[10](10-GUI-PLAN.md) §6-1 の「切り替えのたびに子サーバを立て直す」形は要らなくなった。
2. **S1〜S5 は入った。**`.bonsai27b` のディレクトリをロードすると `llama-server` が子サーバとして立ち、
   別のモデルに替えるか unload すると止まる (§1、§2、検査ツールで実測)。アプリのモデルのメニューは
   `AppModelKind.allCases` を並べるので「Bonsai 2 27B PQ2_0 (MTP)」が出るはずだが、GUI では見ていない (未確認)。
3. **サンプリングのずれは Mac の Bonsai だけだった** (§3、実測)。PC の llama-swap は Qwen3.8 の項目が
   サーバ引数で `--min-p 0 --presence-penalty 1.5` を持っていた。直した後は、Mac のサーバの `/slots` に
   `min_p 0.0`・`presence_penalty 1.5` が出る。
4. **合否** (§2)。S1 の PC 向け 1 会話だけ未実施 (PC の llama-swap が止まっていて、GPU を別のプロセスが使っていた)。
5. **S3 の見落としが 1 つ、検査で出た。**インストール済み判定 (`AppModelInstallationProbe`) が新しい種別を知らず、
   ロードが始まらなかった。検査は 600 秒待つだけだったので、ロード不可を即座に報告するようにした (§4)。

## 1. 入れたもの

```
AppModel ── KindRoutingInferenceClient ─┬─ DecodeServiceInferenceClient   (Gemma / Ornith / Qwen3.8、今のまま)
                                        └─ LlamaServerInferenceClient     (.bonsai27b)
                                              ├─ 子サーバの起動・/health 待ち・停止・pid ファイル
                                              └─ RemoteInferenceClient (Routing.direct、127.0.0.1)
```

| 段 | 場所 | 中身 |
| --- | --- | --- |
| S1 | `Sources/TsugumiApp/Core/Inference/RemoteInferenceClient.swift` | `TsugumiToolLoopCheck` から移した。`Routing` (`.llamaSwap` = `/upstream/<id>/`、`.direct` = 前置きなし)。検査は `--remote-direct` |
| S2 | `AppGenerationRequest.minP` / `.presencePenalty`、`AppModelKind.officialSampling(thinking:)` | `AppModel.makeRequest` が埋め、遠隔の 3 経路 (chat・強制呼び出し・`constrained`) が同じ値を送る。nil なら送らない (Gemma・Ornith) |
| S3 | `AppModelKind.bonsai27b` | family `qwen3_8_dense_llamacpp`、thinking の有無で [01](01-FEASIBILITY.md) §4 の 2 行、4K〜32K、Vision なし、ダウンロード元なし、`runsOnLlamaServer` |
| S4 | `LlamaServerInferenceClient.swift` | `manifest.json` の `llama_server`・`gguf`・`server_args` に `-m`・`-c`・`--host`・`--port` (空き) を足して起動。`-c` などを manifest に書くと拒む |
| S5 | `KindRoutingInferenceClient.swift`、`TsugumiMacApp.swift` | ロードするディレクトリの種別で内側を選び、先に反対側を unload。`applicationWillTerminate` で子サーバを止める |

- 置き場は `~/LLM/Ternary-Bonsai-2-27B-MTP/manifest.json` (10 §2 の形) と、`scratch/Ternary-Bonsai-2-27B-MTP` のリンク。
- 子サーバのログと pid ファイルは `~/Library/Application Support/Tsugumi/llama-server/` (検査では `--out` の下)。
- 表示は decode service と同じメールボックスを読むので、子サーバ側のトークンも振り分けの中で同じメールボックスに写す。
- `TsugumiToolLoopCheck` は `--endpoint` なしで `.bonsai27b` のディレクトリを渡すと、アプリと同じ振り分けを通る
  (エンジン側は同一プロセスの `RealInferenceClient`。decode service ではない)。流し方は
  `Scripts/qwen38_27b/bonsai_child_server_run.sh` (`OUT=` / `ONLY=` / `EXTRA=` / `SERVER_ARGS=`)。

## 2. 合否 (実測、各 1 回。Swapouts はどれも +0)

| 段 | 合否の条件 (10 §3) | 結果 | 実行 (`scratch/bonsai27b/runs/`) |
| --- | --- | --- | --- |
| S1 | 中継なしで 1 会話 | 通った。ja-howto 2 ターン、7 ラウンド × 2、149 s・119 s | `s1-direct-howto` |
| S1 | PC の llama-swap 向けに 1 会話 | **未実施**。経路の文字列は単体テストだけ | — |
| S2 | サーバ側に 1.5 / 0.0 が出る | `/slots` に temperature 0.7・top_k 20・top_p 0.8・min_p 0.0・presence_penalty 1.5 | `s1-direct-howto/slot-sampling.txt` |
| S3 | build・既存テスト・`probe` のテスト | `TsugumiAppCoreTests` 358 本。§4 の 1 本を除き緑 | — |
| S4 | 起動 → 1 問 → unload の後に残らない | en-url 2 ターンの後、子サーバの pid は消え、終了後の `pgrep` は空 | `s4-child-url` |
| S4 | アプリを kill した次の起動で残りが止まる | `kill -9` の後サーバ 2958 は生存 → 次の起動で 2958 は消え、2980 だけ | `s4-kill` |
| S5 | Gemma → Bonsai → Gemma で 1 問ずつ | 3 つとも回答。Bonsai のロード 20.4 s、Gemma に戻ると子サーバは消える | `s5-switch` |
| S6-3 | Stop | 40 トークンで取り消し → 0.21 s でアイドル、サーバログに `cancel task`、次のターンは回答 | `s6-stop` |

- en-url turn 1 は `online` 不成立 (1 ラウンドで回答、37 s)。[09](09-ONLINE-FULL-RUN.md) §2 と同じ回。
- Stop を確かめたのは、文字が流れている chat の回だけ。強制呼び出しの回 (`/v1/completions`、ストリームなし) の途中では試していない。

## 3. サンプリングのずれはどこにあったか (実測)

| 環境 | 09 までに効いていた min_p / presence_penalty | 出所 |
| --- | --- | --- |
| Mac の Bonsai (`bonsai_online_run.sh`) | **0.05 / 0.0** | 要求にもサーバ引数にも無く、`llama-server` の既定 |
| PC の llama-swap、Qwen3.8 の 3 項目 | 0.0 / 1.5 | `C:\LLM\config.yaml` の `--min-p 0 --presence-penalty 1.5` |
| Mac のローカル (Flash-Next) | 0.0 / 1.5 | `Qwen38Sampler` |

今は要求が値を持つので、サーバ引数に依らない。[05](05-BONSAI-PQ2-0.md) §6 と 09 の数字は 0.05 / 0.0 のもの。

## 4. 検査が見つけたもの

1. **インストール済み判定が `.bonsai27b` を知らなかった** (S3 の見落とし)。`AppModelInstallationProbe` は
   `archConfig == nil` を Qwen3.8 のディレクトリとして読むので `partial` になり、`canLoadModel` が偽のまま
   ロードが始まらなかった。`runsOnLlamaServer` の種別は `LlamaServerModelDirectory` (サーバと GGUF が在るか) で判定する。
   `TsugumiToolLoopCheck` は 600 秒待つだけだったので、`canLoadModel` が偽なら判定の中身を出してすぐ終わるようにした。
2. **取り消しのテストが今の機体では落ちる。今回の変更が原因ではない** (実測)。`TsugumiAppCoreTests` を並列で回すと
   `AppModelTests.cancelDuringPrefillKeepsPromptSnapshotUntilClear` (3 回中 3 回) と `cancelAfterPartialOutputCanBeCleared`
   (3 回中 1 回) が落ちる。**変更前の 263dabe を別のワークツリーでビルドして同じ条件で回しても 3 回中 3 回落ち**、
   そちらは `AppModelToolLoopTests.cancellingDuringAToolStopsTheLoop` も 3 回とも落ちた。`AppModelTests` だけを回すと
   8 回中 5 回通る。モックの生成は全部で約 25 ms、検査は 5 ms 刻みで待ってから取り消す形で、取り消しの前に生成が終わる回がある。
   回している間、`mediaanalysisd` か `XProtectRemediator` が CPU 96〜97% を使っていた。S1〜S3 を入れた直後の 1 回は 352 本が全部通っている。
   **直していない。**
3. **`swift build` が古いオブジェクトを残した。**`LlamaServerInferenceClient.init` に引数を足した後、`TsugumiMacApp.swift` が
   再コンパイルされず、デバッグビルドのリンクが旧シグネチャの未定義シンボルで落ちた。ファイルを touch して解消。

## 5. 会話の切り替えのスワップ (S6-2、実測、各設定 2 回)

同じ子サーバで ja-news (3 ターン) → en-swift と続けて流し、5 秒刻みで Swapouts・RSS・空き率を残した。
見張りは 60 秒で +512 MiB か合計 +1 GiB。切り替えは en-swift の最初の要求で、サーバログの `f_keep` が 0.08〜0.11 に落ちる回。
表の「後」は切り替えから 120 秒まで (見張りが止めた回は止まるまで)。

| 子サーバの引数 | 回 | 直前の文脈 | `f_keep` | 空き率 (前 60 s) | RSS 前 → 後の最大 | Swapouts (後) | 見張り | en-swift turn 1 |
| --- | --- | ---: | ---: | --- | --- | ---: | --- | --- |
| 既定 (`--cache-ram 8192`・`--ctx-checkpoints 32`) | 1 | 18,480 | 0.095 | 18〜20% | 10.34 → 10.73 GB | +58,156 ページ (0.89 GiB)、5 秒後 | 止めた | — |
| | 2 | 16,792 | 0.105 | 23〜24% | 10.62 → 11.04 GB | +76,788 ページ (1.17 GiB)、8 秒後から | 止めた (58 秒後) | — |
| `--cache-ram 0` | 1 | 22,208 | 0.080 | 24〜25% | 10.93 → 10.95 GB | +0 | 通過 | 7 ラウンド、155 s、ok |
| | 2 | 19,651 | 0.090 | 16〜18% | 10.41 → 10.74 GB | +0 | 通過 | 7 ラウンド、159 s、ok |
| `--ctx-checkpoints 0` | 1 | 16,143 | 0.109 | 25〜26% | 10.23 → 11.21 GB | +0 | 通過 | 7 ラウンド、119 s、ok |
| | 2 | 19,615 | 0.090 | 21〜23% | 9.98 → 11.40 GB | +0 | 通過 | 7 ラウンド、217 s、ok |

- 既定は [09](09-ONLINE-FULL-RUN.md) §4 の 2 回と合わせて 4 回とも見張りが止めた。止めた時点のメモリ上位は `llama-server` だけ。
- `--cache-ram 0` の 1 回目は en-swift の 3 ターンまで同じサーバで流しきった (6 ターン、Swapouts 合計 +0)。
- **`--ctx-checkpoints 0` の 1 回目は、ja-news の 2・3 ターン目の最初のラウンドが `cached=0`** だった
  (5,939 と 12,851 トークンを読み直し、ターンは 214 s・213 s)。2 回目と他の 4 実行は、同じ位置で
  prompt 6,103〜17,050 に対して cached が 5,414〜17,018。
- サーバログには、どの設定でも切り替えの前後にキャッシュの保存・確保を示す行が無い (09 §4 と同じ)。

**`manifest.json` の `server_args` に `--cache-ram 0` を足した。**足した後に戻ってきた会話の prefill がどうなるかは測っていない (未確認)。
流し方は `Scripts/qwen38_27b/bonsai_switch_sets.sh`。

## 6. 14 ターン × 2 セット (S6-1、実測)

`manifest.json` に `--cache-ram 0` を入れた後、アプリと同じ経路 (振り分け → 子サーバ) で 6 会話 14 ターンを
**1 つの子サーバで続けて**流した。[09](09-ONLINE-FULL-RUN.md) は 1 会話ごとにサーバを立て直していた。
検査の引数は 09 §1 と同じ (`--network online --max-rounds 6 --thinking off --context 32768 --pin-search --search-budget 0`)、
サンプリングは S2 の後の値 (§3)。表は `Scripts/qwen38_27b/turn_table.py`。

| | セット 1 | セット 2 | 09 (参考、min_p 0.05 / presence 0.0、1 会話 1 サーバ) |
| --- | ---: | ---: | ---: |
| `answer`・`error` | 14 / 14 | 14 / 14 | 14 / 14 |
| `online` | 11 / 14 | 10 / 14 | 12 / 14 |
| 合計の秒 (1 ターンの幅) | 1,551 (22〜188) | 1,317 (24〜130) | 1,465 (37〜162) |
| うち prefill | 870 s | 692 s | — |
| decode | 10,865 tok / 16.3 tok/s | 9,500 tok / 16.4 tok/s | 15.2〜17.9 tok/s |
| MTP の受理 | 4,857 / 5,960 = 81.5% | 4,197 / 5,255 = 79.9% | 83.5% |
| Swapouts の合計 | +0 | +0 | 会話の中は +0 |
| サーバの RSS | 9.95〜10.88 GB | 9.96〜10.73 GB | 9.95〜11.22 GB |
| 空き率 (サーバが居る間の最小) | 16% | 19% | 19% |
| 終了後の `llama-server` | なし | なし | — |

| 会話 | ターン | セット 1: ラウンド / 秒 / prefill / decode / 最大 prompt / checks | セット 2 |
| --- | ---: | --- | --- |
| ja-news | 1 | 7 / 188 / 115 s / 1,210 tok 16.7 / 10,760 / ok | 6 / 127 / 76 s / 852 tok 16.7 / 7,222 / ok |
| ja-news | 2 | 7 / 169 / 70 s / 1,585 tok 16.1 / 17,741 / ok | 7 / 118 / 72 s / 725 tok 15.7 / 13,660 / `online` 不成立 |
| ja-news | 3 | 7 / 139 / 95 s / 661 tok 15.1 / 25,560 / `online` 不成立 | 3 / 44 / 20 s / 377 tok 16.0 / 15,817 / `online` 不成立 |
| en-swift | 1 | 7 / 126 / 81 s / 767 tok 17.3 / 8,925 / ok | 7 / 114 / 46 s / 810 tok 17.3 / 5,748 / ok |
| en-swift | 2 | 3 / 45 / 15 s / 472 tok 15.9 / 10,761 / `online` 不成立 | 7 / 119 / 78 s / 678 tok 16.4 / 13,260 / ok |
| en-swift | 3 | 7 / 67 / 44 s / 379 tok 16.5 / 15,035 / ok | 7 / 96 / 36 s / 612 tok 15.9 / 16,785 / ok |
| ja-howto | 1 | 7 / 158 / 86 s / 1,196 tok 16.6 / 9,350 / ok | 7 / 126 / 60 s / 1,091 tok 16.8 / 7,005 / ok |
| ja-howto | 2 | 7 / 98 / 45 s / 774 tok 15.3 / 14,140 / ok | 7 / 106 / 48 s / 892 tok 15.6 / 12,105 / ok |
| en-url | 1 | 1 / 22 / 11 s / 189 tok 17.6 / 2,338 / `online` 不成立 | 1 / 24 / 11 s / 227 tok 17.5 / 2,338 / `online` 不成立 |
| en-url | 2 | 7 / 57 / 25 s / 498 tok 17.1 / 4,833 / ok | 7 / 56 / 29 s / 472 tok 17.4 / 5,285 / ok |
| ja-price | 1 | 7 / 120 / 75 s / 753 tok 16.9 / 8,454 / ok | 7 / 130 / 76 s / 896 tok 16.7 / 8,467 / ok |
| ja-price | 2 | 7 / 169 / 83 s / 1,294 tok 15.3 / 16,290 / ok | 3 / 72 / 15 s / 875 tok 15.6 / 10,396 / `online` 不成立 |
| en-facts | 1 | 7 / 127 / 89 s / 582 tok 17.3 / 9,655 / ok | 4 / 69 / 48 s / 366 tok 17.0 / 5,757 / ok |
| en-facts | 2 | 4 / 65 / 34 s / 505 tok 16.5 / 12,919 / ok | 7 / 116 / 78 s / 627 tok 16.6 / 13,022 / ok |

- `online` 不成立は 2 セットとも en-url turn 1 (1 ラウンドで回答、09 と同じ)。他はセットで入れ替わる。
- 09 との差にはサンプリング (§3) と「1 サーバで続けて流した」の両方が入っているので、どちらの効果とも書かない。

## 7. 残っているもの

1. **GUI そのもの。**モデルの選択・ロード中の表示・Online の 1 問・チャットの切り替え・Stop・終了後にプロセスが残らないこと
   (10 §3-1 の末尾) は、実際のアプリではまだ誰も触っていない。検査が通したのは同じ `AppModel` と同じ振り分けだが、
   エンジン側は `RealInferenceClient` で、GUI の `DecodeServiceInferenceClient` (launchd のジョブ) との切り替えは未確認。
2. S1 の PC 向け 1 会話 (§2)。
3. 強制呼び出しの回の途中での Stop (§2)。
4. ~~`--cache-ram 0` で、前の会話に戻ったときの prefill (§5)。~~ **[13](13-KVQ8-AND-SLOT-SAVE.md) §2・§3 で流した**: 全部読み直し (16K で約 175 s)。スロットの save / restore は厳密な延長のときだけ効き、アプリには入れない (ユーザー決定)。
5. 表示名は案のまま「Bonsai 2 27B PQ2_0 (MTP)」(10 §6-2)。
6. 速度計の針を Bonsai で出さない処理 (10 §4) は入れていない。今どう表示されるかは未確認。
