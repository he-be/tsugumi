"""本線の Session を M6 に当て、送った要求をそのまま JSONL に記録する（tsugumi のダイエット計測用）。
写真は out/plays を等間隔に N 枚。1 回ごとに M6 の disk0 読みと Pageins も取る。リポジトリは変更しない。"""
import json, random, subprocess, sys, time
from pathlib import Path
from kajimachine import vlm
from kajimachine.summon import agent

out_payloads, out_rows, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
cur = {"play": "", "k": 0}
fp = out_payloads.open("w")
_post, _stream = vlm.post, vlm.stream

def post(payload, timeout=vlm.TIMEOUT_S, base=""):
    fp.write(json.dumps({"play": cur["play"], "k": cur["k"], "stream": False, "payload": payload}, ensure_ascii=False) + "\n"); cur["k"] += 1
    return _post(payload, timeout=timeout, base=base)

def stream(payload, timeout=vlm.TIMEOUT_S, base=""):
    fp.write(json.dumps({"play": cur["play"], "k": cur["k"], "stream": True, "payload": payload}, ensure_ascii=False) + "\n"); cur["k"] += 1
    return _stream(payload, timeout=timeout, base=base)

vlm.post, vlm.stream = post, stream

def m6():
    o = subprocess.run(["ssh", "-o", "BatchMode=yes", "mh@macmini-m6",
        "iostat -d -I disk0 | tail -1; vm_stat | grep -E \"Pageins|Swapouts\""], capture_output=True, text=True).stdout.split()
    return float(o[2]), int(o[4].rstrip(".")), int(o[6].rstrip("."))

plays = [p for p in sorted(Path("out/plays").iterdir()) if (p / "photo.jpg").exists()]
plays = plays[:: max(1, len(plays) // n)][:n]
with out_rows.open("w") as rows:
    for p in plays:
        cur["play"], cur["k"] = p.name, 0
        uri = vlm.data_uri(p / "photo.jpg")
        s = agent.Session(uri, rng=random.Random(p.name))
        d0, pi0, so0 = m6(); t0 = time.perf_counter(); err = ""
        try:
            s.begin(on_field=lambda k, v: None); s.draw(); s.choose()
        except Exception as e:  # noqa: BLE001
            err = repr(e)
        wall = time.perf_counter() - t0; d1, pi1, so1 = m6()
        row = {"play": p.name, "wall_s": round(wall, 2), "secs": [round(x, 3) for x in s.secs], "requests": cur["k"],
               "disk_gb": round((d1 - d0) / 1024, 2), "pageins_gib": round((pi1 - pi0) * 16384 / 2**30, 2),
               "swapouts": so1 - so0, "err": err}
        rows.write(json.dumps(row, ensure_ascii=False) + "\n"); rows.flush()
        print(json.dumps(row, ensure_ascii=False), flush=True)
fp.close()
