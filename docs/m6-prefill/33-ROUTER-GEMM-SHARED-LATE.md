# 33. router の GEMM 化と shared expert の投入順 — 同じバイナリの交互で 3.185 → 3.150 s、台帳は 3.134 → 3.142 s (実測、2026-09-24)

[32 §3](32-QMM-NAX-DEFAULT.md) の続き。まず 32 の既定の上で段の秒を取り直し、[31](31-MLX-KERNELS-PER-STAGE.md) の差の 1 番だった `post.router` に手を入れた。
router のカーネルを GEMM にしても壁時計は動かなかった。調べると、`post.router` と `shared` の段の秒は、並行に投入された 2 本の command buffer が重なった分を含んでいた。shared expert の投入を front の後ろへずらす順序と組み合わせると縮んだので、両方を既定に入れた ([04 原則 2](04-GATES.md))。
戻すのは `TF_PREFILL_ROUTER=row` と `TF_PREFILL_SHARED_LATE=0`。speculative block (MTP の検証) の経路は変えていない。

証拠 (M6、この順に取った。どれも `bench/m6/results/2026-09-24-Mac18-5-<ラベル>/`):
`c1-profile-detail`・`c1-profile`・`c1-host` (32 の既定、1f84c76)、
`router-gemm-kernel-check`、`c2-profile-detail-gemm`、`router-bench`、
`c2-default-a` → `c2-gemm-a` → `c2-default-b` → `c2-gemm-b`、`c2-host-gemm`、
`c3-host-late`・`c3-host-late-gemm`・`c3-profile-late`、`c4-host-late-gemm`・`c4-host-late`、
`c4-default-a` → `c4-lategemm-a` → `c4-default-b` → `c4-lategemm-b`、`c4-greedy-default`・`c4-greedy-lategemm`、
`ledger-router-shared-late` (n = 6)、`router-shared-late-kernel-check`。

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 6f9ae16 / minos 15.0
(候補の run は f04ed69〜cfdb5d2。ソースは候補を TF_ の裏に置いた版で、既定の経路は 883df28 と同じ)
```

コマンドは台帳の固定形 (`prefill_bench.sh`)。単体は `TsugumiKernelCheck --group-size 32 --router-bench` (`bench/m6/router_bench.sh`)。
全 run exit 0、Swapouts 0。MBP の `Scripts/test.sh` は 6f9ae16 のソースで 1627 本通過。

**台帳** ([06](06-LEDGER.md)): 既定を変えた。`ledger-router-shared-late` は **3.142 s (3.119〜4.446)**。前の行 (883df28) の 3.134 s より 0.008 s 長く、**台帳の行は縮んでいない。**

## 0. 結論

1. **同じバイナリで既定と交互に取った壁時計 (c) は 3.185 → 3.150 s** (各 n = 6、差 −0.035 s)。台帳の行 3.142 s は候補の範囲 (3.137〜4.388 s) に入る (B3)。ただし同じ日の既定の run は 3.134 s (32 の台帳) から 3.17〜3.19 s にずれていて、台帳の上では前の行と区別がつかない。
2. **router を GEMM にしただけでは、壁時計は変わらない** (交互で 3.174 → 3.175 s)。カーネル単体は 1 チャンクで 8.13 → 4.65 ms (§1.2)。
3. **`post.router` と `shared` の段の秒には、2 本の command buffer の重なりが入っていた。**shared を front と重ねずに流すと、level 1 の `shared` は 0.417 → 0.162 s、`attn` は 1.37 → 1.30 s (§1.3)。32 §2 の「`shared` の中の GEMM 以外 約 0.27 s」は、この重なりの分だった。shared の GEMM だけの秒は MLX の 0.159 s と同じ程度。
4. **B2 の閾値 (level 2 の `post.router` 0.05 s 以下) は不合格。**候補の `post.router` は 0.346 s。区間の終わりを shared の command buffer が決めていて、router のカーネルの秒を測れていない。
5. 正しさ: KernelCheck の router の新しい検査 (row 1 件・gemm 3 件、2 つの group size と 2 つの方式で計 16 件) を含め、group 64 の 105 件・group 32 の 135 件がすべて PASS。greedy 64 トークンの本文は既定・候補とも `c0707c48e5`。

## 1. 数字

### 1.1 32 の既定の上の段 (`c1-*`、`floor_table.py`、差の大きい順)

| 段 | 下限 s | 実測 s | 差 s | 27 の実測 s |
| --- | ---: | ---: | ---: | ---: |
| `moe.gateup` | 0.253 | 0.752 | 0.499 | 0.741 |
| `post.router` | 0.003 | 0.341 | 0.338 | 0.356 |
| `shared` | 0.143 | 0.417 | 0.274 | 0.570 |
| `attn.full` | 0.027 | 0.281 | 0.254 | 0.267 |
| `moe.down` | 0.127 | 0.354 | 0.227 | 0.346 |
| `attn.swa` | 0.044 | 0.271 | 0.226 | 0.258 |
| `rope` | 0.014 | 0.106 | 0.092 | 0.180 |
| `oproj` | 0.108 | 0.138 | 0.030 | 0.210 |
| `qkv` | 0.188 | 0.204 | 0.016 | 0.354 |

- `post.router` と `shared` の行は §1.3 の重なりを含む。`rope` が 0.180 → 0.106 s になったのは、`attn` の中の段を level 2 の割合で割る出し方で、`qkv` と `oproj` が縮んだ分の割合が移ったため (読み、未確認)。
- host (`c1-host`、2・3 本目): `wait.front` 1.471〜1.482 s、`fetch` 0.612〜0.664 s、`wait.tile` 0.907〜0.911 s、`gpuq idle` 0.439〜0.505 s。

### 1.2 router 単体 (`--router-bench`、M 2048、D 2816、128 エキスパート、top-8、20 回の中央値、GPU µs)

| | M6 | MBP (参考、記録値ではない) |
| --- | ---: | ---: |
| row (旧既定、1 トークンに 1 threadgroup) | 8126.1 | 11188.0 |
| gemm (32 トークン × 128 エキスパート、simdgroup の float 8×8、top-k を融合) | 4652.9 | 2637.5 |

30 層・2478 トークンで row 約 0.30 s、gemm 約 0.17 s (導出)。gemm は 0.32 TFLOP/s で、1 チャンク 64 threadgroup しか無い。

### 1.3 shared を front の後ろに投入したとき (level 1、run 順)

| | 既定 (`c1-profile`) | `TF_PREFILL_SHARED_LATE=1` (`c3-profile-late`) |
| --- | --- | --- |
| `attn` | 1.372、1.366、1.377 | 1.302、1.303、1.309 |
| `shared` | 0.417 × 3 | 0.163、0.162、0.161 |
| `moe` | 1.234、1.114、1.116 | 1.115、1.116、1.224 |

### 1.4 host の内訳 (2・3 本目、秒)

| 構成 | `wait.front` | `wait.shared` | `fetch` | `wait.tile` | call |
| --- | --- | --- | --- | --- | --- |
| 既定 (`c1-host`) | 1.471、1.482 | 0.016、0.017 | 0.664、0.612 | 0.911、0.907 | 3.177、3.132 |
| gemm (`c2-host-gemm`) | 1.465、1.465 | 0.017、0.017 | 0.647、0.653 | 0.901、0.902 | 3.144、3.151 |
| late、shared を fetch の前で待つ (`c3-host-late`) | 1.419、1.411 | 0.134、0.134 | 0.693、0.636 | 0.871、0.899 | 3.230、3.194 |
| late + gemm、同上 (`c3-host-late-gemm`) | 1.309、1.314 | 0.136、0.135 | 0.695、0.713 | 0.910、0.893 | 3.165、3.170 |
| late + gemm、shared を tail の後で待つ (`c4-host-late-gemm`) | 1.311、1.311 | 0.000、0.000 | 0.753、0.740 | 0.948、0.948 | 3.126、3.114 |

- router の GEMM だけでは `wait.front` は 0.01 s しか縮まない。shared を後ろにずらすと 0.16 s 縮む。ただしその分は、shared の待ちか `fetch`・`wait.tile` に移る。
- 最後の行の順序 (front → 待つ → shared を投入 → expert の読み → tail の後で shared を確かめる) が既定になった。

### 1.5 壁時計 (c)、既定と交互 (run 順)

| 組 | 既定 | 候補 |
| --- | --- | --- |
| gemm だけ a / b (`c2-*`) | 4.474、3.168、3.140 / 3.275、3.179、3.145 | 4.409、3.186、3.163 / 3.298、3.161、3.155 |
| 中央値 (n = 6) | 3.174 | 3.175 |
| late + gemm a / b (`c4-*`) | 4.404、3.187、3.161 / 4.489、3.182、3.158 | 4.388、3.150、3.149 / 3.258、3.137、3.146 |
| 中央値 (n = 6) | 3.185 | 3.150 |

- 台帳 `ledger-router-shared-late` (6f9ae16): 4.446、3.146、3.137、3.119、3.138、3.153 s。

## 2. 分かったこと / 外れたこと

- **31 の `post.router` の差 0.342 s は、中身が揃っていなかった。**Tsugumi 側の区間は、front の command buffer の最後から完了まで。その間 shared の command buffer が並行に走っていて、区間の終わりは router でなく両者の取り合いで決まる。`shared` の段 (sharedCB の開始から終了まで) も同じ重なりを含む。
- 層ごとの直列の鎖は `wait.front` (front の GPU) → `fetch` (32 スロットへの preads) → `wait.tile` (routed MoE の GPU)。front の GPU を削っても、削った分は `fetch` と `wait.tile` にかなり移る。壁時計に届いたのは 0.16 s のうち 0.035 s。
- `fetch` が shared を裏に回すと 0.06〜0.1 s 伸びる理由は調べていない。

**逸脱**: 数字は (b) でなく (c)。候補の run の commit は証拠だけ・候補の口だけの commit が混ざる (§ヘッダ)。

## 3. 次に渡すもの

- 到達目標 (mlx_lm 1.823 s) に対して、台帳は 3.142 s。
- 段の秒の出し方に重なりが入るので、並行に走る 2 本の command buffer の段 (`post.router`・`shared`) は level 2 の差を読まない。router のカーネルの秒は `--router-bench` で読む。
- 31 の残りの差の順は `attn.full` 0.194 → `rope` → `moe.down` → `moe.gateup`。ただし front の GPU 秒の縮みは、壁時計には一部しか届かない (§2)。`fetch` と `wait.tile` の鎖を先に見るか、`attn.full` に進むかは、次の段を選ぶときに決める。
