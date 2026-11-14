"""Replay recorded kajimachine requests against an OpenAI-compatible server on this machine.

The payloads are the ones kajimachine's Session (read -> item -> pick -> deed) sent for real
photographs, recorded verbatim (they carry visitors' pictures, so they and the answers stay
outside the repository). One play = the 4 requests of one photograph, in order. Per play, one
JSON line: wall seconds, per-request seconds and the server's own timings, internal-SSD
gigabytes read (iostat), Pageins and Swapouts (vm_stat), and whether each answer parsed as
JSON. Answers go to <answers> for a person to read.

    python3 -I bench/m6/kj_replay.py <base-url> <payloads.jsonl> <rounds> <answers.jsonl>
"""
import json
import subprocess
import sys
import time
import urllib.request

base, payloads, rounds, answers = sys.argv[1].rstrip("/"), sys.argv[2], int(sys.argv[3]), sys.argv[4]


def counters():
    disk = subprocess.run(["iostat", "-d", "-I", "disk0"], capture_output=True, text=True).stdout.split()[-1]
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pick = lambda key: int(next(l for l in vm.splitlines() if l.startswith(key)).split()[-1].rstrip("."))
    return float(disk), pick("Pageins"), pick("Swapouts")


def ask(payload, stream):
    body = json.dumps({**payload, "stream": True} if stream else payload).encode()
    request = urllib.request.Request(f"{base}/v1/chat/completions", body, {"Content-Type": "application/json"})
    started = time.time()
    first, text, timings, usage = None, "", None, None
    with urllib.request.urlopen(request, timeout=900) as response:
        if not stream:
            got = json.loads(response.read())
            text, timings, usage = got["choices"][0]["message"]["content"], got.get("timings"), got.get("usage")
        else:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line[6:] == "[DONE]":
                    continue
                chunk = json.loads(line[6:])
                timings, usage = chunk.get("timings") or timings, chunk.get("usage") or usage
                piece = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                if piece:
                    first = first or time.time() - started
                    text += piece
    return time.time() - started, first, text, timings, usage


plays = {}
for line in open(payloads, encoding="utf-8"):
    row = json.loads(line)
    plays.setdefault(row["play"], []).append(row)

with open(answers, "w", encoding="utf-8") as out:
    for n in range(rounds):
        for name, requests in plays.items():
            disk0, pagein0, swap0 = counters()
            started = time.time()
            calls = []
            for r in sorted(requests, key=lambda r: r["k"]):
                kind = r["payload"]["response_format"]["json_schema"]["name"]
                try:
                    wall, first, text, timings, usage = ask(r["payload"], r["stream"])
                    error = ""
                except Exception as e:  # noqa: BLE001
                    wall, first, text, timings, usage, error = time.time() - started, None, "", None, None, repr(e)
                try:
                    json.loads(text)
                    valid = True
                except ValueError:
                    valid = False
                t = timings or {}
                calls.append({"name": kind, "s": round(wall, 3), "first_s": first and round(first, 3), "valid": valid,
                              "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"), "cache_n": t.get("cache_n"),
                              "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"), "error": error})
                out.write(json.dumps({"round": n + 1, "play": name, "name": kind, "text": text}, ensure_ascii=False) + "\n")
                out.flush()
            wall = time.time() - started
            disk1, pagein1, swap1 = counters()
            print(json.dumps({"round": n + 1, "play": name, "wall_s": round(wall, 2),
                              "disk_gb": round((disk1 - disk0) / 1024, 2),
                              "pageins_gib": round((pagein1 - pagein0) * 16384 / 2**30, 2),
                              "swapouts": swap1 - swap0, "calls": calls}, ensure_ascii=False), flush=True)
