#!/usr/bin/env python3
"""Print a run's answers and tool trace: show.py scratch/knowledge-sources/qwen/offline [chars]"""
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
for line in open(d / "turns.jsonl"):
    t = json.loads(line)
    print(f"=== {t['conversation']}  ({t['rounds']} rounds, {t['wall_s']}s) : {t['question']}")
    for step in t.get("trace") or []:
        if isinstance(step, dict):
            name = step.get("name") or step.get("tool") or "?"
            print(f"  · {name}: {step.get('subject','')}  [{step.get('summary','')}]"[:200])
        else:
            print(f"  · {str(step)[:160]}")
    for c in (t.get("cites") if isinstance(t.get("cites"), list) else []):
        print(f"  ↳ cite {str(c)[:160]}")
    print((t.get("answer") or "").strip()[:n])
    print()
