# 50. Inkling 2 Mini を鑑定機に当てた — 1 回 19.9 s・Swapouts あり、80 個版 (7.9 s) から乗り換えない (実測、2026-11-21)

prefill の台帳 ([06](06-LEDGER.md)) は動かない。2026-11-20 に出た Thinking Machines の Inkling 2 Mini (26B looped MoE、3.4B active) を、
[48](48-KAJIMACHINE-EXPERT-SLICE.md)・[49](49-KAJIMACHINE-SLICE-CHECKS.md) の 80 個版 (`lattice-keep_80.gguf`、Gemma 4 26B-A4B) の置き換え候補として M6 で測った。
**決定: 鑑定機は Gemma 4 (80 個版) のまま。Inkling には乗り換えない** (ユーザー、2026-11-21)。

証拠 (M6、この順):
[`inkling2-llama-bench`](../../bench/m6/results/2026-11-21-Mac18-5-inkling2-llama-bench/bench.txt)・
[`kj-llamacpp-lattice-k80-vsink`](../../bench/m6/results/2026-11-21-Mac18-5-kj-llamacpp-lattice-k80-vsink/summary.tsv)・
[`kj-inkling2-iq3xxs-prefill`](../../bench/m6/results/2026-11-21-Mac18-5-kj-inkling2-iq3xxs-prefill/summary.tsv)。

```
2026-11-21 / Mac18,5 Apple M6 16GB / macOS 27.2.1 (26B131) / APPLE SSD AP0256Z / ヘッドレス / iogpu.wired_limit_mb 13312
Inkling: unsloth llama.cpp b12588-mix-c0c3de8 (883286076) / Inkling-2-Mini-UD-IQ3_XXS.gguf 11663309504 bytes / mmproj-F16.gguf 639348160 bytes
80 個版: mainline llama.cpp 61dbf49 / lattice-keep_80.gguf 9415431584 bytes / mmproj-lattice-Q8_0.gguf 806408032 bytes (48 と同じ引数)
```

## 0. 結論

1. **鑑定の再生 (12 枚 × 2 周) で、1 回 19.9 s 対 7.9 s。**読み 6.6 対 3.3 s、選択 6.3 対 2.0 s。prefill 212 対 757 tok/s、decode 27.5 対 48.5 tok/s。
2. **Inkling は M6 でスワップする。**wired 最大 13.88 GiB、Swapouts 1 周目 144,244・2 周目 42,528 ページ。80 個版は 11.87 GiB・0 / 0。
3. **Inkling は思考を切れない。**`reasoning_effort` none でも思考を書き、鑑定の `max_tokens` (読み 560・英名 120) を使い切って本文が空になる。
   assistant の書き出しに `{` を置くと思考を飛ばす。この形で 96 件すべて JSON として読めた (`{` の補い方は §3)。
4. **答えは鑑定の用途で上回らない。**写っている物の特定は具体的 (型番・固有名まで言う) だが、取り違え (白い紙を別の物と読む、犬を猫と書く) と、
   動作文の規則破り (主語の指定を無視、動作の後に飾りの句を足す) がある。12 枚を読んだ印象で、数えた比較ではない。
5. **M6 に載るのは UD-IQ3_XXS (11.66 GB) が上限。**次の UD-Q3_K_XL は 13.63 GB で、利用可能の下限 13.3 GB を越える (導出)。

## 1. 載せ方 (M6 に Inkling を載せるときの前提)

- unsloth の GGUF の `ffn_down_exps` は 24 層中 20 層が **NVFP4 のまま** (IQ3_XXS で 3.82 GB、ヘッダを読んで数えた)。
  mainline の Metal は NVFP4 の mul_mat / get_rows を持たない (`ggml-metal-device.m` が非対応を返す。Metal 化の PR #20456 は 2026-06 から止まっていてコンフリクト)。その部分は CPU に落ちる。
- **unsloth の prebuilt (b12588-mix-c0c3de8) は Metal の NVFP4 カーネル (unslothai/llama.cpp#348) を持つ**ので、それを使った。読めるのはリポジトリ直下の Unsloth レイアウトの GGUF。
  `Mainline_Support/` の `.gguf_file` は同じテンソルを mainline の `inkling2_loop` 名で並べたもので、ggml-org/llama.cpp #32694 (2026-11-20 マージ) に加えて #32704 (未マージ、ループの 2・3 周目の scale) が要る。
- 11.66 GB は既定の GPU 上限 (約 10.7 GB) を越えるので、`sudo sysctl iogpu.wired_limit_mb=13312` が要る (再起動で 0 に戻る)。
- DSpark (同梱の投機デコーダ、0.40 GB) は `-c 8192` で Metal が OOM。`-c 4096 -ub 256` では動くが、受理率 46% で decode 28.4 → 23.3 tok/s に**下がる**。使わない (この数字は証拠ディレクトリを作っていない。未確認扱い)。

## 2. 速度 (`llama-bench -ngl 99 -p 512 -n 128 -r 3`、[`bench.txt`](../../bench/m6/results/2026-11-21-Mac18-5-inkling2-llama-bench/bench.txt))

| | ファイル GB | GiB (llama-bench) | params | pp512 tok/s | tg128 tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 80 個版 Q4_0 (unsloth / mainline) | 9.42 | 8.75 | 16.66 B | 1446 / 1444 | 54.9 / 54.8 |
| Gemma 4 UD-IQ3_XXS、全 128 個 (unsloth / mainline) | 11.42 | 10.62 | 25.23 B | 1222 / 1222 | 46.8 / 46.9 |
| Inkling 2 Mini UD-IQ3_XXS (unsloth) | 11.66 | 10.85 | 26.07 B | 307 | 29.3 |

- Gemma は unsloth と mainline で差がない。unsloth の prebuilt を Gemma に使っても損はしない。
- Inkling の decode が 1/1.9 なのは、中の 12 層を 3 周して 1 トークンあたり 48 層ぶん回る (公称の実効 active 5.4B) ことでおおむね説明がつく。
  prefill の 1/4.7 は説明がついていない (NVFP4 カーネル、相対位置バイアスの注意の実装の候補。未調査)。

## 3. 鑑定の再生 (各 12 枚 × 2 周、2 周目の中央値、`summary.tsv`・`mem.tsv`)

要求は [47](47-KAJIMACHINE-REPLAY-TSUGUMI-VS-LLAMACPP.md) と同じ 12 枚 × 4 問 (`~/diet-probe/kj_payloads.jsonl`、sha256 は各 `env.txt`)。
Inkling 側は [`bench/kj_slice/ink_prefill.py`](../../bench/kj_slice/ink_prefill.py) `make` で各要求の最後に assistant `{` を足した版。
両腕は [`bench/kj_slice/ink_vs_k80.sh`](../../bench/kj_slice/ink_vs_k80.sh) で続けて流した (80 個版 → Inkling)。Inkling のサンプリングは公式推奨の temperature 1.0・top-p 1.0・top-k 0・min-p 0。

| | 80 個版 | Inkling |
| --- | ---: | ---: |
| 1 回の秒 [最小〜最大] | 7.90 [7.15〜10.25] | 19.86 [16.69〜21.87] |
| 読み / 英名 / 選択 / 動作 s | 3.29 / 1.57 / 1.96 / 1.06 | 6.58 / 3.09 / 6.29 / 3.72 |
| 読みの最初の文字まで s | 0.79 | 1.23 |
| prefill / decode tok/s | 757 / 48.5 | 212 / 27.5 |
| wired の最大 GiB | 11.87 | 13.88 |
| Swapouts 1 周目 / 2 周目 (ページ) | 0 / 0 | 144,244 / 42,528 |
| server の RSS GiB | 11.02 | 12.02 |
| JSON として読めない答え (96 件) | 1 (英名が 120 トークンで切れた、48 §5 の崩れ。`maxLength` 未適用の要求) | 0 (下の補正後) |

- `summary.tsv` の Inkling の `invalid_json` 12 / 周は、ハーネスが置いた `{` を知らないための数え違い。ストリーミングの本文には置いた `{` が入らない。
  一方で読み以外の 3 問 (72 件すべて) はモデルが自分でも `{` を書き直す。**`{` は先頭が `{` でないときだけ補う** (`ink_prefill.py score`) と 96 / 96 が読める。
  本線で使うなら、kajimachine のクライアントにこの 2 つ (書き出しの `{`、条件つきの補い) が要る。
- 思考させたまま `max_tokens` を 2,000 足した試し (1 件ずつ) は、読み 72.8 s (1,834 トークン)・英名 51.6 s で、使えない (証拠ディレクトリなし)。

## 4. 答え (12 枚 × 4 問を人が並べて読んだ、答えはリポジトリに入れない)

- **良い所**: 正体が具体的。80 個版が「紙」「円形の物体」で済ませた物を、品目や商品名の水準で言い当てる。英名も形の特徴を多く書く。
- **悪い所**:
  - 取り違え。12 枚中 2 枚で正体や見立てが写真と食い違う (白い紙を別の物と読む、犬の写真の見立て・選んだわけに「猫」が出る)。
  - 連想する語に同じ語 (鋼・深夜・傷) が写真をまたいで繰り返し出る。誤字 (旧字の混入、「させせず」) がある。
  - 動作文が規則を守らない。主語の指定 (she / he) を無視した主語、「動作で文を終える」に反して後ろに比喩の句を足す、が複数ある。選んだわけは長く、見立てに無いことを足す。
- 80 個版は見立て・選んだわけが短く、指示の形を守る。正体は大づかみ。

## 5. やっていないこと

- Inkling の思考を切ったときの品質の数え上げ (48 §4 のような類似度、英名の崩れの率)。決定に要らないので取らない。
- Q2_K_XL など小さい量子化でスワップが止まるか。速度で決着しているので取らない。
- unsloth の prebuilt を 80 個版の本番に使うこと。速度は同じだが、本番は mainline のまま。

## 6. 各機に残したもの

- M6: `~/LLM/ref/inkling2/` (Inkling の GGUF・mmproj-F16・dspark Q8_0、`bench.sh`、`dl.sh`)、`~/LLM/ref/unsloth-llama-b12588/` (prebuilt)、
  `~/LLM/ref/llama.cpp-latest/build/bin/llama-bench` (足した)。`~/diet-probe/kj_payloads_ink_prefill.jsonl`・`ink_vs_k80.sh`・`answers/kj-*-vsink.jsonl`・`answers/kj-inkling2-iq3xxs-prefill.jsonl`。
  計 12.7 GB。乗り換えないので、Inkling の GGUF は消してよい (未削除)。
- `iogpu.wired_limit_mb` は 13312 (再起動で 0 に戻る)。M6 で動いているモデルのプロセスはない。
