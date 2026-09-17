#!/usr/bin/env python3
"""Recorded tool-loop runs → branch-probe manifests, one per run × conversation (docs/qwen38/32).

Reads the prompts `TsugumiToolLoopCheck --replay` wrote (REPLAY/<arm>-<rep>/prompts.jsonl). The trunk is the
conversation's last round; every round is a variant (its prompt is a prefix of the trunk), the app's resume positions
are the extra chunk cuts. Rounds named in --draw get SAMPLES draws; a `none` round (the app's last round after the
limit) draws with `<tool_call>` forbidden, as the app does.

    python3 Scripts/qwen38/replay_branches.py scratch/qwen38/branch33 --draw N2-easternmost:1,N3-gasoline:2,N1-pm:*
"""
import argparse, json
from pathlib import Path

TOOL_CALL = 248058

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--draw", default="")
ap.add_argument("--samples", type=int, default=16)
ap.add_argument("--sample-tokens", type=int, default=48)
ap.add_argument("--top", type=int, default=20)
a = ap.parse_args()
root = Path(a.root)
draw = {}
for item in filter(None, a.draw.split(",")):
    conv, rounds = item.split(":")
    draw[conv] = None if rounds == "*" else {int(r) for r in rounds.split("/")}
runs = Path("docs/experiments/knowledge-sources/ple-ab")
seen = set()   # identical prompts are drawn once
total = 0
for replay in sorted((root / "replay").iterdir()):
    arm, rep = replay.name.split("-")
    choices = {(r["conversation"], r["round"]): r["choice"]
               for r in map(json.loads, open(runs / arm / "offline" / rep / "rounds.jsonl"))}
    prompts = [json.loads(l) for l in open(replay / "prompts.jsonl")]
    for conv in sorted({p["conversation"] for p in prompts}):
        ps = [p for p in prompts if p["conversation"] == conv and p["rendered"] > 0 and p["recorded"] > 0]
        if not ps:
            continue
        variants = []
        for p in ps:
            tokens = open(replay / p["file"]).read()
            none = choices.get((conv, p["round"])) == "none"
            wanted = conv in draw and (draw[conv] is None or p["round"] in draw[conv])
            v = {"label": f"{replay.name}/{conv}/r{p['round']}", "tokens": str((replay / p["file"]).resolve()),
                 "samples": a.samples if wanted and tokens not in seen else 0}
            if wanted:
                seen.add(tokens)
            if none:
                v["forbid"] = [TOOL_CALL]
                v["label"] += "-none"
            variants.append(v)
        name = f"{replay.name}-{conv}"
        manifest = {"trunk": variants[-1]["tokens"], "sampleTokens": a.sample_tokens, "top": a.top,
                    "speculative": True, "samples": 0, "cuts": [p["rendered"] for p in ps[:-1]], "variants": variants}
        out = root / "manifests" / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(manifest, open(out, "w"), indent=1)
        total += ps[-1]["rendered"]
        print(name, ps[-1]["rendered"], sum(v["samples"] > 0 for v in variants))
print("trunk tokens", total)
