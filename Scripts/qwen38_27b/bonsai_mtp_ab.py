#!/usr/bin/env python3
"""Bonsai 2 27B の MTP A/B (Mac)。discussion #23 と同じ 12 本の prompt・同じ body。"""
import json, sys, time, urllib.request, subprocess

PORT, TAG, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
URL = f"http://127.0.0.1:{PORT}/completion"

P = {
 "R1 reasoning": "Quantum computing exploits superposition and entanglement to perform computations that are intractable for classical machines. The principal obstacle in practice is decoherence, which",
 "R2 reasoning": "The reason ternary weights appeal to hardware designers is not only the storage density. A ternary multiply accumulator can be implemented as",
 "R3 reasoning": "When a large language model is quantized below two bits per weight, the failure mode that appears first is usually not factual recall but rather",
 "C1 code":      "def merge_intervals(intervals):\n    \"\"\"Merge overlapping intervals.\"\"\"\n    if not intervals:\n        return []\n    intervals = sorted(intervals)\n    merged = [intervals[0]]\n    for start, end in intervals[1:]:\n",
 "C2 code":      "class RingBuffer:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.buf = [None] * capacity\n        self.head = 0\n        self.size = 0\n\n    def push(self, item):\n",
 "C3 code":      "async def fetch_all(session, urls, concurrency=8):\n    sem = asyncio.Semaphore(concurrency)\n    async def one(u):\n        async with sem:\n            async with session.get(u) as r:\n",
 "M1 math":      "A train leaves at 09:15 and travels 240 km at an average speed of 96 km/h. It then waits 20 minutes and returns at 80 km/h. Step by step, the arrival time back is",
 "M2 math":      "We flip a fair coin until we see two heads in a row. Let E be the expected number of flips. Setting up the recursion gives",
 "F1 format":    "Output the JSON object for a user record with fields name, age, city, tags (array), then output it again, then again, then again:",
 "F2 format":    "Item 1: alpha\nItem 2: beta\nItem 3: gamma\nItem 4: delta\nItem 5:",
 "Z1 chinese":   "把下面这段技术说明改写成更通俗的中文，保持信息不丢：折叠旋转基把正交变换折进权重，运行时只需对激活做一次同构变换即可。改写结果：",
 "Z2 chinese":   "请用中文解释为什么三值权重（-1、0、+1）在存储上比 int4 更省，并给一个具体数字例子。回答：",
 "J1 ja prose":  "日本語で説明します。三値の重み (-1, 0, +1) を使う量子化が省メモリになる理由は、",
 "J2 ja format": "次の形式で 4 件続けて出力してください。\n{\"名前\": \"田中\", \"年齢\": 34, \"市\": \"横浜\"}\n{\"名前\": \"佐藤\", \"年齢\": 41, \"市\": \"札幌\"}\n",
}

def swapouts():
    o = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    for line in o.splitlines():
        if 'Swapouts' in line: return int(line.split(':')[1].strip().rstrip('.'))
    return -1

def run(name, prompt):
    body = json.dumps({"prompt": prompt, "n_predict": 128, "temperature": 0.0,
                       "top_k": 1, "seed": 7, "cache_prompt": False}).encode()
    t0 = time.time()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    wall = time.time() - t0
    t = r.get("timings", {})
    dn, da = t.get("draft_n") or 0, t.get("draft_n_accepted") or 0
    row = {"name": name, "tps": t.get("predicted_per_second", 0.0), "n": t.get("predicted_n"),
           "prompt_n": t.get("prompt_n"), "wall": wall, "dn": dn, "da": da,
           "content_head": r.get("content", "")[:80]}
    print(f"  {name:14} {row['tps']:6.2f} t/s  {row['n']:4} tok  {wall:6.1f}s" +
          (f"  acc {da/dn*100:5.1f}% ({da}/{dn})" if dn else "  no draft"), flush=True)
    return row

sw0 = swapouts()
res = [run(k, v) for k, v in P.items()]
sw1 = swapouts()
json.dump({"tag": TAG, "swapouts_delta": sw1 - sw0, "results": res}, open(OUT, "w"), indent=1)
print(f"  ---- {TAG}: swapouts +{sw1-sw0}", flush=True)
