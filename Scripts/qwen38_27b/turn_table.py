#!/usr/bin/env python3
"""TsugumiToolLoopCheck の出力 (turns.jsonl + turn-metrics.jsonl) から docs/qwen38-27b/09 §2 と同じ形の表を作る。
    turn_table.py RUN_DIR [RUN_DIR ...]
turn-metrics.jsonl の行はラウンド順なので、turns.jsonl の rounds の数だけ順に割り当てる。"""
import json, sys

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

for run in sys.argv[1:]:
    turns = load(f"{run}/turns.jsonl")
    metrics = load(f"{run}/turn-metrics.jsonl")
    print(f"\n{run}\n")
    print("| 会話 | ターン | ラウンド | 秒 | うち prefill | decode | 最大 prompt | checks |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    at = 0
    total = dict(wall=0.0, prefill=0.0, tokens=0, decode=0.0, proposed=0, accepted=0)
    for t in turns:
        rows = metrics[at:at + t["rounds"]]
        at += t["rounds"]
        prefill = sum(r.get("prefillSeconds") or 0 for r in rows)
        tokens = sum(r.get("generatedTokens") or 0 for r in rows)
        decode = sum(r.get("decodeSeconds") or 0 for r in rows)
        failed = sorted(k for k, v in t["checks"].items() if not v)
        print(f"| {t['conversation']} | {t['turn']} | {t['rounds']} | {t['wall_s']:.0f} | {prefill:.0f} s | "
              f"{tokens:,} tok / {tokens / decode if decode else 0:.1f} tok/s | "
              f"{max((r.get('promptTokens') or 0) for r in rows):,} | "
              f"{'ok' if not failed else ', '.join('`%s` 不成立' % f for f in failed)} |")
        total["wall"] += t["wall_s"]; total["prefill"] += prefill; total["tokens"] += tokens; total["decode"] += decode
        total["proposed"] += sum(r.get("draftProposed") or 0 for r in rows)
        total["accepted"] += sum(r.get("draftAccepted") or 0 for r in rows)
    assert at == len(metrics), (at, len(metrics))
    walls = [t["wall_s"] for t in turns]
    print(f"\n合計 {total['wall']:.0f} 秒 (1 ターン {min(walls):.0f}〜{max(walls):.0f} 秒)、prefill {total['prefill']:.0f} 秒、"
          f"decode {total['tokens']:,} tok / {total['tokens'] / total['decode']:.1f} tok/s、"
          f"MTP の受理 {total['accepted']:,} / {total['proposed']:,} = {100 * total['accepted'] / max(total['proposed'], 1):.1f}%")
