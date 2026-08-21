# -*- coding: utf-8 -*-
"""`in_proj_a` の量子化が減衰ゲート g = -exp(A_log)·softplus(a+dt_bias) に与える害を、
**実活性**で測り直す (docs/qwen35moe/04「次の一手」#8。10 §5 は合成入力 x~N(0,1))。

参照は上流 bf16 の抽出。比べるのは公式 MLX-4bit (in_proj_a は 4-bit) と
oQ4e-g64 (8-bit)。活性は参照器の `--dump` が落とした層入力 (RMSNorm の**前**)。

α = exp(g) は再帰状態の減衰係数で、head ごとに T トークンで Πα として積み上がる。
**実測できるのは手元の T トークンぶんだけ**なので、1000 トークンの欄は導出として
分けて出す。
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path("Scripts/qwen35").resolve()))
from checkpoint_io import Checkpoint          # noqa: E402
from mlx_quant import Dequantizer             # noqa: E402

EPS = 1e-6
BF = Path.home() / "LLM/Ornith-1.5-35B-A3B-bf16-partial"
SOURCES = {
    "mlx-4bit": Path.home() / "LLM/Ornith-1.5-35B-A3B-MLX-4bit",
    "oq4e-g64": Path.home() / "LLM/Ornith-1.5-35B-A3B-oQ4e-g64",
}


def rms_norm(x, w, eps=EPS):
    return (x / np.sqrt(np.mean(np.square(x), -1, keepdims=True) + eps)) * w


def softplus(x):
    return np.logaddexp(x, 0.0)


def main(fixtures: str, layers: list[int]) -> int:
    fx = np.load(fixtures)
    T = fx["layer0.input"].shape[0]
    bf = Checkpoint(BF)
    deq = {k: Dequantizer(Checkpoint(v)) for k, v in SOURCES.items()}
    print(f"# fixtures {fixtures}   T = {T} トークン   層 {layers}")
    rows_a, rows_b = [], []
    for L in layers:
        p_bf = f"model.language_model.layers.{L}"
        p_mlx = f"language_model.model.layers.{L}"
        x = fx[f"layer{L}.input"].astype(np.float32)
        w_norm = bf.f32(f"{p_bf}.input_layernorm.weight") + 1.0
        baked = deq["oq4e-g64"].vector(f"{p_mlx}.input_layernorm.weight")
        gap = float(np.abs(baked - w_norm).max())
        if gap > 5e-3:
            print(f"  !! 層 {L} の norm 規約が合わない (max |差| {gap:.3g})")
        h = rms_norm(x, w_norm)
        A = np.exp(bf.f32(f"{p_bf}.linear_attn.A_log"))
        dt = bf.f32(f"{p_bf}.linear_attn.dt_bias")
        W_ref = bf.f32(f"{p_bf}.linear_attn.in_proj_a.weight")
        g_ref = -A * softplus(h @ W_ref.T + dt)          # log α、[T, Hv]
        a_ref = np.exp(g_ref)
        mask = a_ref > 0.5
        share = float(mask.mean())
        for name, d in deq.items():
            prefix = f"{p_mlx}.linear_attn.in_proj_a"
            W = d.matrix(prefix)
            bits = int(d.spec(prefix)["bits"]) if d.is_quantized(prefix) else 16
            dw = float(np.linalg.norm(W - W_ref) / np.linalg.norm(W_ref))
            diff = -A * softplus(h @ W.T + dt) - g_ref   # Δ log α、[T, Hv]
            rel = np.abs(np.expm1(diff))
            sel = rel[mask] if mask.any() else rel.ravel()
            per_head = diff.sum(axis=0)                  # head ごとの Δ log Πα
            ratio = np.exp(per_head)                     # **実測** T トークンの Πα 比
            mean_d = float(diff.mean())
            se = float(diff.mean(axis=1).std(ddof=1) / np.sqrt(T))
            rows_a.append((L, name, bits, dw, np.median(sel), np.percentile(sel, 95),
                           sel.max(), share, float(np.median(a_ref))))
            rows_b.append((L, name, mean_d, se, float(np.median(ratio)),
                           float(ratio.min()), float(ratio.max())))
    print("\n## A. 重みと 1 ステップの誤差")
    print(f"{'層':>3} {'源':9} {'bits':>4} {'|dW|/|W|':>9} {'中央値':>9} {'p95':>8} "
          f"{'max':>8}   (α>0.5 の |Δα/α|)   {'α>0.5 の割合':>11} {'α 中央値':>8}")
    for L, n, b, dw, med, p95, mx, share, amed in rows_a:
        print(f"{L:3d} {n:9} {b:4d} {dw*100:8.2f}% {med*100:8.3f}% {p95*100:7.3f}% "
              f"{mx*100:7.2f}%                       {share*100:10.1f}% {amed:8.3f}")
    print("\n## B. 偏りと蓄積")
    print(f"{'層':>3} {'源':9} {'mean Δg':>10} {'±SE':>9}  "
          f"{'実測 Πα 比 (T tok) 中央値':>24} {'最小':>8} {'最大':>8}  {'導出 1000 tok':>12}")
    for L, n, m, se, med, lo, hi in rows_b:
        print(f"{L:3d} {n:9} {m:10.2e} {se:9.2e}  {med:24.3f} {lo:8.3f} {hi:8.3f}  "
              f"{np.exp(1000*m):12.3f}")
    return 0


if __name__ == "__main__":
    fixtures = sys.argv[1] if len(sys.argv) > 1 else "scratch/qwen35/fixtures-oq4e-g64.npz"
    layers = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 4, 8, 16, 28]
    sys.exit(main(fixtures, layers))
