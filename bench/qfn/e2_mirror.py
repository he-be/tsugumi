"""Write the E2 mirror (whole layers, gate/up/down per layer file) with F_NOCACHE on both ends, so neither the
source layers nor the mirror are left warm in the page cache.

    python3 bench/qfn/e2_mirror.py <model dir> <mirror dir> <first> <layers>
"""
import fcntl
import os
import subprocess
import sys
import time

F_NOCACHE = 48
here = os.path.dirname(os.path.abspath(__file__))
model, mirror, first, layers = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(mirror, exist_ok=True)
env = dict(os.environ, E2_MAIN_RANGES=os.path.join(here, "expert-ranges.txt"), E2_DOWN_RANGES=os.path.join(here, "down-ranges.txt"))
plan = subprocess.run([sys.executable, os.path.join(here, "e2_lists.py"), "--mirror-plan", model, mirror, first, layers],
                      env=env, check=True, capture_output=True, text=True).stdout.split("\n")
fds = {}
def fd(path, write):
    if (path, write) not in fds:
        f = os.open(path, (os.O_WRONLY | os.O_CREAT) if write else os.O_RDONLY, 0o644)
        fcntl.fcntl(f, F_NOCACHE, 1)
        fds[(path, write)] = f
    return fds[(path, write)]
t0 = time.time(); total = 0
for line in plan:
    if not line:
        continue
    src, off, n, dst, doff = line.split()
    off, n, doff = int(off), int(n), int(doff)
    s, d = fd(src, False), fd(dst, True)
    blk = 16 << 20
    for p in range(0, n, blk):
        b = os.pread(s, min(blk, n - p), off + p)
        os.pwrite(d, b, doff + p)
    total += n
    print(f"{dst} +{n} ({total / 1e9:.2f} GB, {total / 1e9 / (time.time() - t0):.2f} GB/s)", flush=True)
for f in fds.values():
    os.close(f)
