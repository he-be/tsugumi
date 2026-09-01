# 35. prefill のチャンク幅が 2 本目から効いていなかった (実測(手元)、2026-08-22)

[33 §3-7](33-MTP-ACCEPTANCE.md) の測定 1 (幅 2 の 1 パス費用) を
`--qwen-prefill-bench` の腕として組んだところ、**幅 1 と幅 2 の腕が
まったく同じ数字を出した** — 総時間 15,874 ms 対 15,884 ms、
エキスパート要求 62,358 対 62,358。**チャンク幅が効いていなかった。**

原因は 1 行、影響は測定だけでなく **Phase 4 の検査の半分**に及ぶ。

---

## 0. 結論を先に

| # | 論点 | 結論 |
| --- | --- | --- |
| 1 | 何が起きていたか | `QwenPrefill.prefill` が**要求された幅ではなく scratch の幅**でプロンプトを切っていた。scratch は「前の呼び出しが作ったもので十分大きければ使い回す」ので、**1 プロセスの中で幅を変えると 2 本目以降は最初の幅で走る** (§1) |
| 2 | 出力は間違っていたか | **いいえ。**幅はモデルから見えないので、どの幅で走っても答えは同じ。**壊れていたのは「別の幅を試した」という主張の方**である (§1-2) |
| 3 | 一番効く影響 | **[21 §4](21-PHASE4-PREFILL.md) の「チャンク 8 (3 チャンク)」は 1 度も走っていなかった。**fixture のプロンプトは 19 トークンなので、先に走る幅 512 の腕が 19 幅の scratch を作り、続く幅 8 の腕はそれを使い回して**また 1 チャンク**で終わっていた。21 自身が「チャンク 8 を回すのが本題の半分」「片方だけでは主張になっていない」と書いている、その半分が空だった (§2) |
| 4 | 直した後どうなったか | **`PASS chunk 8 (3 chunks)`** — 引き継ぎ (conv の窓・再帰状態・K/V のカーソル) が**初めて本当にチャンク境界を越え、通った** (§3)。**隠れていた不一致は無い。**空振りだっただけである |
| 5 | 直し方 | 1 行。切るのは要求された幅、scratch は「それを載せられる大きさ」でしかない。広い scratch の使い回しはそのまま残る (`prefillChunk` は `T = tokens.count` で回るので、広い scratch に狭いチャンクを載せること自体は元から正しい) (§1-3) |
| 6 | 他に空だった主張 | 同じ検査の **routed 経路 × 幅** の 4 通り ([24 §2](24-PREFILL-MOE-PATH.md)) も、幅の側は 1 通りしか走っていなかった。**経路の側 (per-pair / tiled) は本物** — あれは runner の property で scratch と無関係 (§2-2) |
| 7 | 速度の測定への影響 | [24 §3](24-PREFILL-MOE-PATH.md) と [21 §5](21-PHASE4-PREFILL.md) の**幅ごとの時間は無事**。あれは幅ごとに**プロセスを分けて**測っている (`--qwen-prefill-bench` は 1 プロセス 1 幅だった)。**壊れるのは 1 プロセスで複数の幅を回したときだけ** (§4) |

---

## 1. 中身

### 1-1. 使い回しの規則と、切り方の規則が混ざっていた

```swift
// QwenPrefill.swift (直す前)
let ctxScratch = try prefillContext(width: min(chunkWidth, tokens.count))
while offset < tokens.count {
    let count = min(ctxScratch.width, tokens.count - offset)   // ← ここ
```

`prefillContext` は `if let existing = prefillScratch, existing.width >= width { return existing }`
で、**要求より広い scratch があればそれを返す。**返ってきた scratch の `width` で
プロンプトを切っていたので、要求 `chunkWidth` は**上限としてしか働かない**。

`QwenForwardRunner.reset()` は `kv.reset()` と `state.reset()` だけで、
**`prefillScratch` は残る**。したがって同じ runner を使い回す限り、
1 度広い幅を通したら以降は狭い幅を頼んでも広い幅で走る。

### 1-2. 答えは合っていた

チャンク幅はモデルから見えない ([21](21-PHASE4-PREFILL.md))。広い幅で走っても
出るトークンは同じなので、**この不具合は 1 度も間違った答えを出していない。**
壊れていたのは検査の**主張**の方である — 「幅 8 でも同じだった」は
「幅 8 では走っていない」だった。

### 1-3. 直し

```swift
let width = min(chunkWidth, tokens.count)
let ctxScratch = try prefillContext(width: width)
while offset < tokens.count {
    let count = min(width, tokens.count - offset)
```

`prefillChunk` は行数を `T = tokens.count` から取る (scratch の `width` は
バッファの大きさにしか使わない) ので、**広い scratch に狭いチャンクを載せるのは
元から正しい。**使い回しの最適化はそのまま残してある。

---

## 2. 何が空だったか

### 2-1. `--qwen-prefill` の「チャンク 8」

既定は `--qwen-prefill-chunks 512,8` で、fixture のプロンプトは **19 トークン**。

| 腕 | 意図 | 実際 (直す前) |
| --- | --- | --- |
| 幅 512 | 1 チャンク (19 を丸ごと) | そのとおり。scratch 幅 19 ができる |
| 幅 8 | **3 チャンク**。conv の窓・再帰状態・K/V カーソルが境界を越える | **19 幅の scratch を使い回して 1 チャンク。**境界が 1 つも無い |

[21 §4](21-PHASE4-PREFILL.md) の「**チャンク 8 を回すのが本題の半分である**…
片方だけでは主張になっていない」は正しい認識で、**その半分が実行されていなかった。**

### 2-2. routed 経路 × 幅 の 4 通り

[24 §2](24-PREFILL-MOE-PATH.md) の「4 通り一致」は
`{per-pair, tiled} × {512, 8}`。**経路の側は本物** (`runner.prefillRoutedPath` は
scratch と無関係な property)。**幅の側が 1 通りだった**ので、実質 2 通りだった。
24 §2 の「チャンク 8 はタイル版の batch planner が一番小さい群を見る形」という
狙いも、そこでは実現していない。

---

## 3. 直したあと (**実測**)

```
$ ./.build/release/TsugumiKernelCheck --qwen-prefill scratch/ornith-oq4e-g64.moepack
PASS  41 tokens, every one equal to the float32 reference
PASS  chunk 8 (3 chunks) — the same 41 tokens
PASS  routed experts on the per-pair path, chunk 512 (1 chunks) — the same 41 tokens
PASS  routed experts on the per-pair path, chunk 8 (3 chunks) — the same 41 tokens
  negative controls — each must disagree within 16 tokens:  5 本すべて PASS
```

**`(3 chunks)` が初めて本物である。**引き継ぎは通った — 隠れていた不一致は無く、
[21](21-PHASE4-PREFILL.md) の結論そのものは動かない。**動くのは「確かめた」の
中身**で、Phase 4 の出口条件は**今日はじめて本当に満たされた**。

### 3-1. 検査

| 何 | 結果 |
| --- | --- |
| `--qwen-prefill` (既定 `512,8`、負例 5 本つき) | **全 PASS。**`chunk 8 (3 chunks)` が**初めて本物** |
| `swift test --no-parallel` | **1,350 件緑** (206 suite、119 秒) |

`QwenPrefill.swift` の変更は 3 行 (幅を要求どおりに取る) で、scratch の
使い回しも `prefillChunk` も触っていない。

---

## 4. 影響しないもの

- **[24 §3](24-PREFILL-MOE-PATH.md) / [21 §5](21-PHASE4-PREFILL.md) の幅ごとの時間。**
  `--qwen-prefill-bench` は 1 プロセスにつき 1 幅しか受け取らなかったので、
  幅を変えるたびにプロセスが変わり、scratch も作り直されていた。**測定値は無事。**
- **decode 経路。**`prefillScratch` は prefill にしか無い。
- **サーバーと CLI。**1 つの要求は 1 つの幅で通る。ただし**同じ runner で幅を
  変える将来の経路 (幅 2 の検証パスなど) は、直す前ならここで黙って幅を失っていた** —
  [33 §3-7](33-MTP-ACCEPTANCE.md) の測定がまさにそれを踏んだ。

---

## 5. どうやって見つかったか (作法の話)

腕を**同じプロセスの中で交互に**回したから見つかった
([32 §5-1](32-NVMAI-ADOPT.md) の interleaved A/B)。逐次スイープ —
幅ごとにプロセスを分ける — なら、両方それらしい数字を出して**何も起きなかった**。
`--qwen-prefill` が 1 プロセスで 2 つの幅を回していたのに空振りしていたのは、
**検査が数字ではなく PASS/FAIL しか見ていなかった**ためである。

計器を 1 つ足してある: `--qwen-prefill-bench` は腕ごとに**エキスパート要求数**を
出す。幅 1 で 248×40×8 = 79,360、幅 2 で 62,358 (= 12.572 experts/層/パス)。
**この 2 つが同じ値なら、幅は効いていない。**
