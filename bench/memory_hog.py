#!/usr/bin/env python3
"""Hold N GiB of incompressible anonymous memory and keep it warm.

Stands in for "other apps": random bytes so the compressor cannot shrink
them, touched every second so the kernel keeps them active and evicts the
page cache (the model's weights) instead."""
import mmap, os, sys, time
gib = float(sys.argv[1])
n = int(gib * (1 << 30))
page = os.sysconf("SC_PAGESIZE")
mm = mmap.mmap(-1, n)
chunk = 64 << 20
off = 0
while off < n:
    k = min(chunk, n - off)
    mm[off:off + k] = os.urandom(k)
    off += k
print("ready", flush=True)
while True:
    for o in range(0, n, page):
        mm[o]
    time.sleep(1)
