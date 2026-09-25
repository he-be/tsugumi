# 11. Q1 を開け直す — M6 は tensor ops を使う前提。10 の「B」は fp32 のコードで M6 を測った結論 (2026-09-25)

**状態: Q1 は開いている。**[10](10-Q1-SPLIT-DECISION.md) の数字 (E0〜E4) は実測としてそのまま使えるが、結論 (B、σ 0.40 / 0.45) と「E4-(b) を使わない」は取り下げる。
この文書は、取り下げた理由・ds4-metal の手本・再開手順を置く。**§2 の計画は [12](12-Q1-REDESIGN.md) で置き換えた** (前提 P1〜P5 を先に固定し、A・B とも模擬の壁時計で決める)。新しい実測はまだ無い。

## 0. 取り下げた理由

ユーザーの前提 (2026-09-25): **M6 では必ず tensor ops を使う。使わないなら M6 を使う意味が無い。MPS で同じ速さなら MPS でよい。**

10 はこの前提を満たしていなかった。

1. **E1 は M6 で今の fp32 のコードを動かしただけ。**QFN の dense も routed も fp32 の sgemm で tensor ops を使わない。
   そのため M6 の GPU の秒は MBP とほぼ同じに出た (pre-router GPU 4.55 対 4.97 s/チャンク)。M6 の強み (fp16 の tensor ops、MPS で 18.61 TFLOP/s。MBP は 5.94、m6-prefill/03) を捨てた条件で分け方を選んでいた。
2. **B は「状態をすべて MBP が持つ」という実装の都合で dense を MBP に残した。**08 §5 で「dense を 2 機で分ける」を外したのも同じ都合で、速さから決めた形ではない。
   B では M6 が引き受けるのは routed の読みと GEMM だけで、MBP に残る dense・top-10・host の残りが 1 チャンク約 9 s ある (10 §2 の d + t + o、導出)。M6 の tensor ops はここに効かない。
3. A (層ごと M6 に渡す = M6 が dense も持つ) も、M6 の層を fp32 のコードで測ったので 1 層 0.44〜0.49 s (MBP 0.41 s) となり、式で負けた。
4. 07 の E4-(b) の規則「相対差が qwen38/01 の幅 (1.7e-6) を超えたら使わない」は、fp32 のまま計算したときの幅で、**fp16 の行列積は原理的に満たせない** (§1-3)。この規則は外す。

参考 (導出、`bench/qfn/formula.py --turns`): r_M6 だけを MPS fp16 の腕 C の 0.0759 s/層に置き換えても、記録 12 ターンの B は 1564 → 1524 s (写しあり 1476 s)。r だけでは足りず、d_M6 (dense) を tensor ops で測っていないことが効いている。

## 1. 手本: ds4-metal の Q2 tensor tiles

`~/LLM/ds4-metal` の `origin/qwen2-nax-tiles` (dddb1ba)。`main` は 2026-09-25 に origin の 8db1d1d へ fast-forward した。作業中の枝 `qwen3.8-flash-next` は触っていない。QFN と同じ Q2 の形 (gate/up IQ2_XXS、down Q2_K を 768 に詰めたもの) を M5 の tensor ops で回している。

| 項目 | 中身 | 在処 |
| --- | --- | --- |
| カーネル | mid (gate/up + SiLU) と down を各 1 本。expert の行 64 × トークン 64 (端数は 32 のカーネル)、K は 32 刻み、`matmul2d` の積み上げは fp32。重みはスレッドグループのメモリに half で展開し、次の K の元データはレジスタに先読み。活性化は呼び出しの頭で 1 回 half に丸める。SiLU はレジスタの中で、down 用の half の写し `midh` も書く。down は K = 640 のまま回し、768 の詰めは読まない | `metal/qwen4.metal` L2740-2831 (展開)、L3066-3084 (f32→f16)、L3085-3421 (`kernel_qwen4_moe_mm_mid_nax_t`・`_down_nax_t`)、`ds4_metal.m` L48610-48860 (切り替えと dispatch) |
| 速さ (M5 Max、ds4-metal の実測) | Q2 の mid + down の単体 19.33 → 7.38 ms。prefill 8K〜40K で 1036 → 1333 tok/s。1 タイルの内訳は tensor ops 41%・重みの展開 35%・活性化の準備 18% | commit `9f0c99a`・`22d0461`、`speed-bench/qwen38-m5-round5.md` L95-136 |
| 精度 | ビット一致は原理的に無理 (tensor ops の中で 16 要素ずつ丸められ、順序は制御できない)。合否はカーネル単体で simdgroup 版との差 ≤ 2e-3 × scale、モデルで BF16 参照に対する NLL・top-1 (Q2: NLL 0.30418 対 0.30346、top-1 90.28% 対 90.32%、decode はバイト一致) | `a4e8e49`、`tests/test_qwen4_kernels.c` の `test_moe_mm_tiles_iq2`、`QA_BEFORE_RELEASES.md` L689-735 |
| 注意 | IQ2_XXS / Q2_K の 32 トークンの端数カーネルは、ds4-metal では M3 Ultra でしか有効にしていない (`qwen4_moe_mm_tails()`) | 同上 |

10 の腕 C (展開と MPS の行列積を別パス) は M6 で展開 34〜41 ms + 行列積 37〜42 ms/層。ds4-metal は展開をタイルの中に入れているので、腕 C より速いはず (未確認)。

## 2. やり直しの計画 → [12](12-Q1-REDESIGN.md)

ここに置いていた計画 (E4-(c)・E5 を別々に測り、式に入れて A / B を比べる) は 12 で置き換えた。残るのは精度の基準の置き換えだけ:
**カーネル単体は今の per-pair 経路 (腕 A) との差 ≤ 2e-3 × scale、モデルは logits の top-1 一致と差の分布を記録する** (P1、04 Q6)。
E4-(c) の移植は 12 の F1 (1 層まとめてのベンチ) の中で行い、E5 は F1 に吸収した。式で A / B を比べることはしない (P5)。

## 3. 再開手順

```bash
# 0. 前提の確認 (両機)。モデルプロセスが無いこと、外付けがマウントされていること
cd ~/LLM/turbo-fieldfare && git log --oneline -1          # この文書の commit 以降
pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiCLI|TsugumiKernelCheck'; ssh m6 'pgrep -fl TsugumiKernelCheck; df -h /Volumes/SSD256 || diskutil mount disk5s1'
ifconfig bridge0 | grep 'inet '; ssh m6 "ifconfig bridge0 | grep 'inet '"   # link-local は繋ぎ直すと変わる

# 1. M6 を同期してビルド (M6 は編集しない。証拠は bench/qfn/results/ に出る)
git push m6 HEAD && ssh m6 'cd dev/tsugumi && swift build -c release --product TsugumiKernelCheck 2>&1 | tail -1'
swift build -c release --product TsugumiKernelCheck

# 2. 今の腕 A/B/C を 1 層で確かめる (MBP / M6)
bench/qfn/e4_gemm.sh e4-gemm-check 5 24
ssh m6 'cd dev/tsugumi && QFN_MODEL=$HOME/dev/tsugumi/scratch/Qwen3.8-Flash-Next-DS4-IQ2 zsh bench/qfn/e4_gemm.sh e4-gemm-check 5 24'

# 3. 手本を読む (checkout しない)
git -C ~/LLM/ds4-metal show origin/qwen2-nax-tiles:metal/qwen4.metal | sed -n 2740,2831p   # 展開
git -C ~/LLM/ds4-metal show origin/qwen2-nax-tiles:metal/qwen4.metal | sed -n 3085,3421p   # mid / down の tensor ops
git -C ~/LLM/ds4-metal show origin/qwen2-nax-tiles:speed-bench/qwen38-m5-round5.md | less

# 4. 式 (E0〜E4 の証拠から、導出。k / σ の格子を絞るだけに使う、P5)
python3 bench/qfn/formula.py --turns
```

- 次にやるのは 12 §2 の順: F0 (Q0) → Q2 → F1 → F4 → F2 / F3。
- F1 の腕 D は `Sources/TsugumiKernelCheck/Q2GemmBench.swift` の腕 C と同じ作り (このチェックアウトの `moe_ggml.metal` に足して KernelCheck の中でコンパイルし、ランタイムには入れない) で足す。
  tensor ops には MSL 4.0 (`MTLCompileOptions.languageVersion = .version4_0`) と `MetalPerformancePrimitives` が要る。MBP (macOS 15) では組めないので、腕 D は M6 でだけ走らせる。
- 置いてあるもの:
  - route の dump `scratch/qfn/routes/c2048-l<L>.bin` (両機、48 層、prompt-code の先頭 2048 トークン)。
  - トークン列 `scratch/qwen38/prompt-code.tokens` (M6 にも写した)。
  - M6 の外付けの写し `/Volumes/SSD256/qfn/mirror/layer0〜23.bin` (17 GB、E2 用)。
  - M6 で追い出しに使う Gemma の重み `~/dev/tsugumi/scratch/gemma4-qat-sym.gturbo` (15 GB)。
- 道具の一覧は `bench/qfn/` (README は未作成): `link_probe.c`・`e0_link.sh` (E0)、`e1_stages.sh` (E1)、`range_read.c`・`e2_*.py`・`e2_ssd.sh` (E2)、`e3_split.sh` + `--qfn-split-probe` (E3)、`e4_gemm.sh` + `--q2-gemm-bench` の腕 A/B/C (E4)、`formula.py`。
- 両機で KernelCheck は同時に 1 本まで。見張りは `Scripts/qwen38/guarded.sh`。10 の最初の E3 は、層ごとに作業領域を取って MBP で Swapouts +91,992 ページを出した (見張りが止めた)。層ごとの大きな確保は避ける。
