# 15. Phase 2 — Gated DeltaNet カーネル (実測(手元)、2026-08-21)

**この計画で最初に GPU が回った。**[04-PHASES.md](04-PHASES.md) Phase 2 が本命と
呼んでいた `qwen_delta_rule` ([03 §2-6](03-DESIGN.md)) を書き、CPU の参照 2 本に
対して検証し、時間を測った。モデルは載せていない — 入力は**実物と同じ形の合成値**で、
チェックポイントも `.moepack` も要らない。

```bash
swift run -c release TsugumiKernelCheck --gdn              # 検証 15 本
swift run -c release TsugumiKernelCheck --gdn --gdn-bench  # 時間
```

---

## 1. 出口条件

| 条件 ([04](04-PHASES.md) Phase 2) | 結果 |
| --- | --- |
| 全カーネルが誤差床の中 | **`qwen_delta_rule` は床の上に乗った** (§3)。状態の相対誤差 1.145e-07 に対し CPU float32 の床が 1.14e-07 |
| **2048 トークン流した後の状態**が参照と一致 | **一致** (§3)。1 トークンでは蓄積の間違いが見えないので、既定の検査長を 2048 にした |
| 線形注意 30 層が 150 ms 以内 ([05 §2](05-RISKS.md) #2) | **125.7 ms** (TB=32、§4)。**中止線の内側** |

**Phase 2 の残り**は `qwen_delta_norm_gate` ([03 §2-7](03-DESIGN.md))・`conv1d`・
partial RoPE・LM head の specialization で、本カーネルはそのうちの 1 本目。

## 2. 何を書いたか

| ファイル | 中身 |
| --- | --- |
| `Sources/Tsugumi/Metal/Qwen/gdn.metal` | `qwen_delta_rule_tb{16,32,48}`。幾何は omlx の `gated_delta_blocked_seq` を写した (Apache-2.0、参照のみ) |
| `Sources/Tsugumi/Kernels/Qwen/GatedDeltaNet.swift` | 包み。`stateIn` と `stateOut` に同じバッファを渡してよい (各スレッドが自分の断片を書く前に読む) |
| `Sources/TsugumiKernelCheck/GatedDeltaNetCheck.swift` | 検証 15 本とマイクロベンチ |

**Gemma 4 経路には 1 バイトも足していない。**`gdn` は `tensorops` と同じ扱いで、
共通ライブラリ (`shaderModules`) には入れず `moduleLibrary` で単独に組む。
`MetalContext` への変更は置き場の 1 行だけ。

幾何 (`Dk = 128` はコンパイル時定数、threadgroup 配列の寸法なので):

```
grid   = (Dv/32 = 4, Hv = 32) threadgroup、256 スレッド → 128 threadgroup
thread → dv = tid/8、seg = tid%8 → d0 = seg*16
状態   = レジスタ float4 st[4] (16 float/thread、256×16 = 32×128 ✓)
縮約   = simd_shuffle_down(4→2→1) + simd_shuffle。ホットループに barrier 無し
staging= k/q/v を TB トークンぶん threadgroup memory へ協調ロード
```

threadgroup memory は 1 行 632 B なので **TB=16: 10,112 B / TB=32: 20,224 B /
TB=48: 30,336 B** (上限 32,768 B)。TB は関数定数ではなくマクロで 3 本の
エントリポイントを出している — threadgroup 配列の寸法は kernel スコープの
コンパイル時定数でなければならないため。

## 3. 検証 — 3 つの精度で走らせる

**カーネルのバグと丸め誤差を分離する** (PLAN_VISION §6 の教訓) ために、同じ入力を
3 通り流す: GPU (fp16 の q/k/v、fp32 の状態) / **CPU float32** (床) / **CPU double** (真値)。
入力は実物と同じ規約で作る — `k` は L2 正規化、`q` は L2 正規化 × `Dk^-0.5`
([14 §3](14-REFERENCE.md) の非対称な倍率)、`beta = sigmoid(·)`、
`g = exp(-exp(A_log)·softplus(·))`、状態は 0 から始めない。

**実測(手元)** (M3 Pro / macOS 15.7.5、15 本すべて PASS):

| 見るもの | GPU の相対誤差 | CPU float32 の床 | 許容 |
| --- | ---: | ---: | ---: |
| `y` decode (T=1) | 4.164e-04 | 4.20e-07 | 4e-3 |
| 状態 decode (T=1) | **8.143e-08** | **8.14e-08** | 20×床 |
| `y` prefill (T=2048) | 2.870e-04 | 4.28e-07 | 4e-3 |
| 状態 prefill (T=2048) | **1.145e-07** | **1.14e-07** | 20×床 |

**状態は CPU float32 の床と 3 桁一致している** — 2048 ステップ積み上げても、
GPU の答えは「同じ精度で CPU がやったのと同じだけしか」ずれていない。
`y` の 2.9e-04 は fp16 で書き戻す刻み (2^-11 = 4.9e-04) が支配していて、
カーネルの誤差ではない。TB=16/32/48 の 3 本は**同じ数字を出す**。

残り 2 本:

| 検査 | 結果 |
| --- | --- |
| **状態の持ち越し** — 1 回で 2048 流したものと、32 ずつ 64 回に切って状態を渡したもの | **ビット一致** (state 0 / y 0 本が不一致)。時間ブロックは算術の順序を変えない。[14 §3](14-REFERENCE.md) の 3 本目 (conv・再帰・KV の繰り越し) の GPU 版にあたる |
| **検出力** — 減衰を縮約の**あと**に掛ける「もっともらしい間違い」に対して外れること | **8.4e-02**。正しい参照が通った物差し (4e-3) の **21 倍**離れている。1 度も落ちたことのない検査は証拠にならない |

## 4. 時間 — 中止線の内側

**実測(手元)**、20 回の中央値。GPU の `gpuEndTime - gpuStartTime`。
**これはカーネル 1 本のマイクロベンチで、モデルの速度ではない** (`bench.sh` の
作法の対象外)。

| T | TB | 1 層 (ms) | **30 層 (ms)** | 状態の往復 GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 0.054 | 1.6 | 77.4 |
| 1 | **32** | **0.049** | **1.5** | 85.4 |
| 1 | 48 | 0.053 | 1.6 | 78.4 |
| 2048 | 16 | 4.281 | 128.4 | — |
| 2048 | **32** | **4.191** | **125.7** | — |
| 2048 | 48 | 4.827 | 144.8 | — |

- **prefill 2048 は 125.7 ms で、中止線 150 ms の内側。**逐次形のまま進んでよい。
  chunkwise (WY/UT) は FLOP が 2 倍なので、[03 §2-6](03-DESIGN.md) の判断は変えない
- **TB=32 が 3 本とも最良。**TB=48 は threadgroup memory を 30 KB 使って占有率を
  落とすぶんだけ遅い (+15%)。[04](04-PHASES.md) Phase 4 が「3 通り測る」と
  言っていた宿題はここで**済んだ** — ただし実物の活性ではなく合成入力での話
- **decode は 30 層で 1.5 ms/token。**状態の往復は 1 層 4.2 MB
  (fp32 の `[32,128,128]` を読んで書く) で、30 層 126 MB/token。
  [01 §5-5](01-MODEL.md) の導出「状態往復 126 MiB/token」と一致し、
  85 GB/s 出ている。**decode 予算 (約 31 ms/token) の 5%**

> 注: ベンチは状態バッファを使い回すので、反復のあいだに状態は減衰して 0 に近づく。
> 命令数は値に依らないので時間には効かない。

## 5. この文書が動かした結論

| 対象 | 更新 |
| --- | --- |
| [04](04-PHASES.md) Phase 2 | **本命カーネルは通った。**残りは norm gate / conv1d / RoPE / LM head |
| [04](04-PHASES.md) Phase 4 の TB 掃引 | **合成入力では済んだ** (TB=32)。実物での再測は結線後 |
| [05](05-RISKS.md) §2 #2 の中止線 | **抵触しない** (125.7 ms < 150 ms) |
| [04](04-PHASES.md)「次の一手」12 | **完了** |
| [03 §2-6](03-DESIGN.md) の設計 | 変更なし。omlx の幾何がそのまま乗った |
