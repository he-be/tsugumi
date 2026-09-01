#!/usr/bin/env python3
"""MTP 自身の注意履歴をどこから始めるかで、深さ 1 の受理率がどれだけ動くか。

[33](../../docs/qwen35moe/33-MTP-ACCEPTANCE.md) の測定は教師強制なので、MTP の
K/V は**プロンプトを含む全位置**が埋まった因果キャッシュだった。実機の decode
ループでそれを再現するには、プロンプト全体に MTP 層を通す prefill が要る
(fc の 2048x4096 を T 行、k/v 射影を T 行)。**それを書く前に、履歴を捨てたら
受理がどれだけ落ちるか**を、既に取ってある hidden の npz だけで引く。

    scratch/mtp-venv/bin/python Scripts/qwen35/mtp_history_ablation.py \\
        ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-shisa \\
        --tokens scratch/qwen35/mtp-a/t1-ja-explain.tokens.json \\
        --hidden-cache scratch/qwen35/mtp-a/t1-ja-explain.hidden.npz

`--start full` はプロンプトから (33 と同じ)、`--start gen` は生成開始位置から
(実機の decode-only)、`--start last` は履歴なし (自分だけを見る)。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference_forward import ReferenceModel, rms_norm, sigmoid  # noqa: E402
from mtp_acceptance import MTP, MTPHead  # noqa: E402


def depth1_drafts(head: MTPHead, base: np.ndarray, next_ids: np.ndarray,
                  history_start: np.ndarray) -> np.ndarray:
    """深さ 1 の draft top-1。`history_start[t]` より前の K/V は見せない。"""
    cfg = head.cfg
    hd = cfg.head_dim
    eps = cfg.rms_norm_eps
    n = base.shape[0]
    positions = np.arange(n) + 1

    h = head.project(base, next_ids)
    resid = h
    x = rms_norm(h, head._vec("layers.0.input_layernorm.weight"), eps)
    q, k, v, gate = head.qkv(x, positions)
    groups = cfg.num_heads // cfg.num_kv_heads
    keys = np.repeat(k, groups, axis=1)
    values = np.repeat(v, groups, axis=1)

    scores = np.einsum("thd,shd->hts", q, keys) * (hd ** -0.5)
    s_idx = np.arange(n)[None, :]
    t_idx = np.arange(n)[:, None]
    visible = (s_idx <= t_idx) & (s_idx >= history_start[:, None])
    scores = np.where(visible[None, :, :], scores, -np.inf)
    probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    o = np.einsum("hts,shd->thd", probs, values)

    o = o * sigmoid(gate)
    h = resid + head.m._dense(o.reshape(n, cfg.num_heads * hd),
                              f"{MTP}.layers.0.self_attn.o_proj")
    resid = h
    x = rms_norm(h, head._vec("layers.0.post_attention_layernorm.weight"), eps)
    h = resid + head.m._moe(x, "mtp", None, prefix=f"{MTP}.layers.0.mlp")
    return head.m.lm_head_argmax(rms_norm(h, head._vec("norm.weight"), eps))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--hidden-cache", required=True)
    ap.add_argument("--variant", default="post_norm", choices=["post_norm", "pre_norm"])
    ap.add_argument("--json")
    args = ap.parse_args()

    root = Path(args.checkpoint).expanduser()
    dumped = json.loads(Path(args.tokens).read_text(encoding="utf-8"))
    prompt, generated = list(dumped["prompt"]), list(dumped["generated"])
    tokens = np.asarray(prompt + generated, dtype=np.int64)
    prompt_len = len(prompt)

    blob = np.load(Path(args.hidden_cache))
    if not np.array_equal(blob["tokens"], tokens):
        raise SystemExit("npz は別のトークン列のもの")
    base_all = blob["pre" if args.variant == "pre_norm" else "post"]
    body_top1 = blob["body_top1"]

    model = ReferenceModel(root)
    head = MTPHead(model)
    n = len(tokens) - 1
    base = base_all[:n]
    nxt = tokens[1:n + 1]
    pos = np.arange(prompt_len - 1, n - 1)

    name = Path(args.tokens).stem
    print(f"## {name}  プロンプト {prompt_len} + 生成 {len(generated)}  "
          f"({args.variant}, 深さ 1, 判定位置 {len(pos)})", flush=True)

    arms = {
        "full": np.zeros(n, dtype=np.int64),
        "gen": np.full(n, prompt_len - 1, dtype=np.int64),
        "last": np.arange(n, dtype=np.int64),
    }
    out = {"task": name, "variant": args.variant, "positions": int(len(pos)), "arms": {}}
    for label, start in arms.items():
        t0 = time.time()
        draft = depth1_drafts(head, base, nxt, np.minimum(start, np.arange(n)))
        p1 = float(np.mean(draft[pos] == body_top1[pos + 1]))
        out["arms"][label] = {"P1": p1, "seconds": time.time() - t0}
        print(f"  {label:5s} P1 = {p1 * 100:6.2f}%   ({time.time() - t0:.1f}s)", flush=True)

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
