"""記録した鑑定の要求を 1 件ずつ流し、答えと (あれば) ルーター記録ファイルの区切りを残す。
    python run_arm.py <base-url> <payloads.jsonl> <out.jsonl> [--greedy] [--trace <file>] [--plays every2:0|every2:1|all]
--greedy は temperature 0 を要求に足す (モデル間の一致を見る用)。"""
import json, os, sys, time, urllib.request
a = sys.argv
base, payloads, outp = a[1].rstrip("/"), a[2], a[3]
greedy = "--greedy" in a
trace = a[a.index("--trace") + 1] if "--trace" in a else None
sel = a[a.index("--plays") + 1] if "--plays" in a else "all"
only = a[a.index("--only") + 1] if "--only" in a else None
rounds = int(a[a.index("--rounds") + 1]) if "--rounds" in a else 1
plays = {}
for line in open(payloads, encoding="utf-8"):
    r = json.loads(line); plays.setdefault(r["play"], []).append(r)
names = list(plays)
if sel.startswith("every2:"):
    names = names[int(sel[-1])::2]
def ask(p):
    body = dict(p); body["stream"] = False
    if greedy: body["temperature"] = 0; body["top_k"] = 1
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        got = json.loads(resp.read())
    return got["choices"][0]["message"]["content"], got["choices"][0].get("finish_reason"), got.get("timings") or {}
size = lambda: os.path.getsize(trace) if trace and os.path.exists(trace) else 0
with open(outp, "w", encoding="utf-8") as out:
    for name in [n for _ in range(rounds) for n in names]:
        for r in sorted(plays[name], key=lambda r: r["k"]):
            kind = r["payload"]["response_format"]["json_schema"]["name"]
            if only and kind != only: continue
            o0 = size(); t0 = time.time()
            try: text, fin, tm = ask(r["payload"]); err = ""
            except Exception as e: text, fin, tm, err = "", None, {}, repr(e)  # noqa: BLE001
            try: json.loads(text); valid = True
            except ValueError: valid = False
            out.write(json.dumps({"play": name, "name": kind, "text": text, "finish": fin, "valid": valid, "s": round(time.time() - t0, 3),
                                  "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"), "predicted_n": tm.get("predicted_n"),
                                  "prompt_tps": tm.get("prompt_per_second"), "predicted_tps": tm.get("predicted_per_second"),
                                  "trace": [o0, size()], "error": err}, ensure_ascii=False) + "\n"); out.flush()
print("ARM_DONE", outp)
