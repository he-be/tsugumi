# 35. R を Windows の Q2_0 で測る (33 §3 の書き直し)

書いた日: 2026-09-18。表記は 01 と同じ (実測 / 導出 / 未確認)。

> **状態 (2026-09-18)**: W0 で打ち切り (36、日本語が壊れている)。§0-1・§1 (Mac の GGUF が llama.cpp で通らない理由、Q2_0 と Mac の重みの差) は記録として残す。段取りは土俵を Q4_K_XL に移して 37 §3。

33 の R (往復を減らす) の**測り方だけ**を差し替える。P (33 §4) と K (33 §5) は 33 のまま。

## 0. 何を変えたか

- **土俵**: 192.168.0.199 (Windows、llama.cpp、llama-swap) で、ISTA-DASLab の `Qwen3.8-Flash-Next-GSQ-RCO-GGUF` の **Q2_0** を動かす (ユーザーが準備中)。
  これまでリモートで使っていたのは unsloth の UD-Q4_K_XL (32 §4-1、§5)。
- **33 の R0 (記録を数えて形を決める) は取り下げる。** 記録に反復が無く、1 走行の数はラウンド数の揺れの中に入るため (34)。代わりに W0・W1 を置く。
- Mac でモデルは動かさない。動かすのは最後の確認 (§6) だけ。

### 0-1. なぜ Mac の重みをそのまま持っていかないか (実測)

`~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/…-Q2KDownPad768-MTP.gguf` を gguf-py で読んだ結果。

- `general.architecture = qwen4exp`、ハイパーパラメータも `qwen4exp.*`。llama.cpp が読み方を決める名前がこれでは通らない。
- `blk.0.ffn_down_exps.weight` は `[768, 2560, 512]`。640 列を 768 に詰めてある (`ds4.qwen4.down.logical_input=640` / `physical_input=768`)。
- PLE 表は外付けの別 GGUF (`ple-bf16/…`、102 GB)。llama.cpp は PLE を本体に持つ形で読む。
- モデルの README も「padded-down 対応の ds4-metal 5bd8796 以降が要る。他のランナーとの互換性は主張しない」と書いている。

ISTA の GGUF は標準 GGUF で、llama.cpp でそのまま動く (README の記述、W0 で確かめる)。

## 1. Q2_0 と Mac の重みの違い (実測)

ISTA の `tensor-allocation/*.rco-allocation.txt` の型と、Mac の GGUF から読んだ要素数で計算した。Q2_0 の bpw は shard 1 の実サイズから 2.25 (計算 37.61 GB 対 実際 37.62 GB)。

| クラス | 重み | Mac | Q2_0 | (参考) IQ2_XS |
| --- | ---: | ---: | ---: | ---: |
| routed gate/up | 80.5G | 2.06 (IQ2_XXS 一様) | 2.25 (Q2_0 一様) | 2.35 (IQ2_S / IQ2_XXS / IQ1_M の混成) |
| routed down | 48.3G | 2.62 (Q2_K、768 に詰め。実重みあたり 3.15) | 2.25 (Q2_0) | 2.25 (Q2_0) |
| dense/other | 3.4G | 10.41 | 6.56 | 6.75 |
| token_embd | 0.6G | 16.00 | 3.44 | 4.25 |
| output | 0.6G | 8.58 | 5.61 | 4.37 |
| shared expert | 0.2G | 8.51 | 3.61 | 4.58 |
| 合計 (PLE 表を除く) | 133.8G | 2.59 | 2.39 | 2.47 |

- **MTP が無い。** Mac の GGUF は `blk.48` に MTP の 32 テンソルを持つ。ISTA の 3 変種はどれも 1,223 本で、この 32 本が無い。
- **PLE 表は IQ4_NL の 4.5 bpw 固定** (shard 2、28.8 GB、3 変種で同一)。Mac の既定は BF16 表。表を変えると出力が動くことは 31 で測ってある。
- Q2_0 はルックアップ型を避けた形式で、Mac の gate/up (IQ2_XXS、コードブック型) とは系統が違う。ISTA の測定では、IQ2_XS に対して prefill 3.4 倍・end-to-end 1.9 倍で、課題平均は 89.07 対 89.16。

## 2. 段取り

| 段 | 何 | どこで | 止める条件 |
| --- | --- | --- | --- |
| W0 | 土俵を立て、道具が最後まで通ることを確かめる | PC (llama-swap) | どれか 1 つでも通らなければ報告して止める |
| W1 | 既定の腕 (今の形) のベースラインと、揺れの床 | PC | 2 セットの差が大きすぎて床が引けなければ報告 |
| R1 | 腕 (ツールの形) を 1 つずつ実装して比べる | 実装は Mac、測定は PC | 床を超えない腕は、そこで打ち切って報告 |
| M | 床を超えた腕だけ Mac で確認 | Mac | — |

- 段の結果を書いてから次に進む。進むかどうかはユーザーが決める。
- 道具 (`TsugumiToolLoopCheck`) は Mac で動き、**推論だけ** PC に出す。ローカル Wikipedia と Web ツールは今までどおり Mac 側で実行される。

## 3. W0: 土俵を立てる (PC)

llama-swap のエントリは、既存の `qwen3.8-flash-next-q4kxl-32k-instruct-*` を雛形にする。変えるのは本体と MTP。

```
-m …\ISTA-DASLab\…\Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
（--mmproj は同リポジトリの mmproj-…-BF16.gguf、使わないなら省く）
-ngl 99 -lm none -fa on -fit off -ncmoe <空き VRAM に合わせて> -t 10
-c 32768 -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 -np 1 --jinja
--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5
env: LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"enable_thinking":false}
```

- **MTP は指定しない。** ISTA の GGUF に MTP ブロックが無い。unsloth の MTP draft と組むかは未確認で、W0 の目的ではない。
- サンプラの値は公式のまま。変えない。
- KV は Mac の運用点に合わせて q8_0、文脈 32K。

確かめること。**どれか 1 つでも落ちたら、そこで止めて報告する。**

1. `/v1/chat/completions` が、ツール宣言つきの要求で通り、Qwen の記法で呼び出しが返る (`RemoteInferenceClient` の auto のラウンド)。
2. `/apply-template` が使える (強制ラウンドは、描き直した prompt に `<tool_call>` の頭を足して `/v1/completions` で続ける作り)。
3. `<tool_call>` の token id が取れ、`logit_bias` で `none` のラウンドが呼び出しを書かない。
4. Offline の 5 問 (`docs/experiments/knowledge-sources`) を 1 セット流し、最後まで回る。
5. 記録すること: `ncmoe` の値、pp / tg、1 問の壁時計、VRAM と RAM。**ここで測った 1 問の時間で、W1 の反復回数が決まる。**

## 4. W1: 既定の腕のベースラインと床 (PC)

- **Offline**: knowledge-sources の 5 問 + 正解のある 2 記事の課題から 15 問 = 20 問 (33 のまま)。15 問は未作成なので、W1 の前に作る。
- **Online**: 32 §5 の 40 問 (`docs/experiments/first-fetch/questions.json`) と `--pin-search`。固定済みの検索結果は残っている (`…/7f01f2ec-…/scratchpad/firstfetch/web/pinned`、40 本)。**Serper は 0 回**。
  - 32 §5 の測定は `--stop-after-round 2` で止めてあるので、ラウンド数は入っていない。W1 では最後まで流す。
- 各腕、**独立した 2 セット**。1 セットは 1 問 N 回で、N は W0 の壁時計から決める (下限 5)。
- 出す数字: ラウンド数の分布、ラウンドの種類ごとの本数 (`Scripts/qwen38/round_breakdown.py` が `rounds.jsonl` を読む)、生成トークン、正答、壁時計。
- **床**: 同じ腕の 2 セットの差。以後、腕の差はこの床を超えたものだけ扱う。
- 参考: UD-Q4_K_XL では、iPhone 18 の 1 問を 20 回流して 4〜7 ラウンド (平均 4.35 / 4.40、32 §4-1)。Q2_0 では取り直す。

## 5. R1: 腕 (実装は Mac、測定は PC)

多い種類から 1 つずつ。どれも宣言は会話の途中で変えない。プロンプトの文言で確率を下げる案は入れない (26)。

- **R1-a 強制検索の結果に、上位 k 件の目次を添える** (k = 1〜3 を腕にする)。強制 fetch と、その後の「同じページの節」を 1 往復に畳むのが狙い。
- **R1-c Offline の検索結果に、冒頭の数文と目次を添える。** R1-a の Offline 版。
- **R1-b は実装の確認だけ。** アプリは 1 ラウンドの呼び出しをすべて実行してから次に進み、結果は 1 つの継続に積まれる (`AppModel` の `toolTask`)。24 以前の 2 本同時の 8 ラウンドでは、次のラウンドの継続が 3 件 (呼び出し 1 + 結果 2) 増えていた (34 §3)。実行は直列で、その待ち時間は記録に無い。

各腕、W1 と同じ数字を 2 セット。床を超えない腕は、そこで止めて報告する。

## 6. M: Mac での確認

- 床を超えた腕だけ、Mac で Offline 5 問 × 2 回。`turn-metrics.jsonl` の秒・pp・tg・読み込み・wired の最大・swapout。
- ここで初めて「Mac で速くなったか」を言う。

## 7. 土俵の差として、受け入れること・受け入れないこと

受け入れる (腕どうしは同じ土俵で比べるため)。

- 本体の型 (Q2_0 対 Mac の IQ2_XXS + 詰めた Q2_K)、PLE 表 (IQ4_NL 対 BF16)、dense の精度 (6.6 対 10.4 bpw)。
- PC 側に MTP が無いこと。KV の型と文脈長は Mac に合わせる。

受け入れない。

- PC の壁時計を Mac の秒の代わりにすること。
- PC で出た腕の差を、Mac で確かめずに採ること。32 §4-1 では、PLE 表を変えただけで開くページが変わった (apple.com を開いたのが Q4_1 9/20、BF16 0/20)。転移は保証されない。
- 1 セットだけの差を腕の差として読むこと。

## 8. K (33 §5) への影響

- KVA の正解データは、**Mac のランナーと同じ値**でなければならない (33 §5)。型も PLE も違う Q2_0 は、その用途には使えない。K0・K1 は 33 のまま、上流の BF16 から PyTorch (CUDA) で作る。
- PC は R の測定 (llama-swap) と K の学習で取り合う。同時には動かさない (33 §2 のまま)。

## 9. 未確認

- ISTA の GGUF が、PC の llama.cpp のビルドでそのまま読めるか (README は標準 GGUF と書いている。W0-1)。
- GGUF に入っている chat template が、アプリの描画と同じ呼び出し記法になるか (W0-1・W0-2)。
- 66.4 GB (本体 37.6 + n-gram 表 28.8) を VRAM 16 GB + RAM 126 GB で回したときの速さ (W0-5)。
- MTP の有無でラウンド数の分布が動くか。同じ土俵で腕を比べる限りは効かないが、Mac は MTP on で動く。
