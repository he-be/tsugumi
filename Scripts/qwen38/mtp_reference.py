#!/usr/bin/env python3
"""Qwen3.8 の MTP ヘッド (blk.48) の CPU 参照。`--qwen38-mtp-dump` の入力で logits を計算して突き合わせる (docs/qwen38/10)。

算式は llama.cpp unsloth-mtp (`a9e9c3c`) の `src/models/qwen4exp.cpp` 489〜660 行 `graph_mtp`:
次トークンの埋め込み → enorm を hc 本に複製、h は stream ごとに RMS → hnorm、stream ごとに [e, h] を連結して
eh_proj → hc_attn ミックス → 密な注意 (QSA なし、自前の KV) → combine → hc_ffn ミックス → MoE → combine →
nextn.hc_head ミックス → output.weight。ブロックの部品 (hc_mix・注意・MoE) は reference_forward.py をそのまま使う。

    ~/LLM/venv/bin/python Scripts/qwen38/mtp_reference.py scratch/qwen38/mtp-code.dump [--positions 0,1,47,48,...]

幹の隠れ状態は Swift の書き出しをそのまま入力に使う (幹は reference_forward.py と既に一致している、01〜09)。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_forward as rf  # noqa: E402


def mtp_token(model, st, token, h, pos, want_logits):
    w = model.w
    pre = f"blk.{model.n_trunk}."
    e = rf.rms(w.row("token_embd.weight", token), w.f32(pre + "nextn.enorm.weight"))
    hn = rf.grouped_rms(h, w.f32(pre + "nextn.hnorm.weight"), rf.HC, rf.E).reshape(rf.HC, rf.E)
    R = np.concatenate([w.matvec(pre + "nextn.eh_proj.weight", np.concatenate([e, hn[s]])) for s in range(rf.HC)])
    R = R.astype(np.float32)
    il = model.n_trunk
    mixed, inj = model.hc_mix(R, pre + "hc_attn")
    blk = model.attention(il, st, mixed, pos)   # indexer_top_k is set huge: every token attends its whole prefix
    model.hc_combine(R, blk, inj)
    mixed, inj = model.hc_mix(R, pre + "hc_ffn")
    blk, sel = model.moe(il, mixed)
    model.hc_combine(R, blk, inj)
    if not want_logits:
        return None
    mixed, _ = model.hc_mix(R, pre + "nextn.hc_head", with_inject=False)
    return w.matvec("output.weight", mixed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("--positions", default=None, help="logits を比べる位置 (カンマ区切り、既定は 0,1,47〜54 と最後)")
    ap.add_argument("--gguf", type=Path, default=rf.DEFAULT_GGUF)
    ap.add_argument("--ple", type=Path, default=rf.DEFAULT_PLE)
    args = ap.parse_args()

    raw = args.dump.read_bytes()
    T, V = np.frombuffer(raw[:8], np.int32)
    off = 8
    tokens = np.frombuffer(raw[off:off + 4 * T], np.int32); off += 4 * T
    W = rf.HC * rf.E
    hidden = np.frombuffer(raw[off:off + 4 * T * W], np.float32).reshape(T, W); off += 4 * T * W
    logits = np.frombuffer(raw[off:off + 4 * T * V], np.float32).reshape(T, V)
    if args.positions:
        positions = {int(p) for p in args.positions.split(",")}
    else:
        positions = {0, 1, T - 1} | set(range(47, min(55, T)))

    model = rf.Model(args.gguf, args.ple)
    model.indexer_top_k = 1 << 30
    st = rf.State(model.n_layer, cap=T + 1)
    worst = 0.0
    mismatches = 0
    for p in range(T):
        t0 = time.time()
        ref = mtp_token(model, st, int(tokens[p]), hidden[p], p, p in positions)
        if ref is None:
            continue
        got = logits[p]
        rel = float(np.abs(got.astype(np.float64) - ref).max() / np.abs(ref).max())
        same = int(np.argmax(got)) == int(np.argmax(ref))
        mismatches += 0 if same else 1
        worst = max(worst, rel)
        print(f"[{p:3d}] token {tokens[p]:7d} top1 metal {int(np.argmax(got)):7d} ref {int(np.argmax(ref)):7d} "
              f"{'ok' if same else 'DIFF'}  logit rel err {rel:.2e}  ({time.time() - t0:.1f}s, footprint {rf.footprint_gb():.2f} GB)",
              flush=True)
        if rf.footprint_gb() > rf.RSS_LIMIT_GB:
            print("footprint over limit, stopping")
            return 2
    ok = mismatches == 0 and worst < 1e-3
    print(f"top-1 mismatches {mismatches}, worst logit rel err {worst:.2e}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
