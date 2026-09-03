#!/usr/bin/env python3
"""Hold N GiB of incompressible anonymous memory and keep it warm — but
never past the line where macOS starts swapping.

Stands in for "other apps": random bytes so the compressor cannot shrink
them, touched every second so the kernel keeps them active and evicts the
page cache (the model's weights) instead.

**Swap guard (2026-09-04).** A hog larger than what is actually free makes
macOS swap the hog itself — random bytes, so every swapped page is written
in full — and re-touching it every second turns that into a steady write
stream. Two experiments of that kind put on the order of a terabyte of
writes on this Mac's SSD (Swapouts 3,260 万ページ since boot). So:

  * the request is capped at (physical − used − MARGIN); more is refused,
    not trimmed, so a driver cannot silently run a different cell;
  * a watchdog reads `vm_stat` every 0.5 s and exits the moment Swapouts
    grows by more than SWAP_ABORT_PAGES, releasing the memory. The driver
    sees the hog gone (`poll()`) and must abort the experiment.

    python3 bench/memory_hog.py 4            # 4 GiB, prints "ready"
    HOG_MARGIN_GIB=2 python3 bench/memory_hog.py 4
"""
import mmap, os, re, subprocess, sys, threading, time

PAGE = os.sysconf("SC_PAGESIZE")
PHYS = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
MARGIN = float(os.environ.get("HOG_MARGIN_GIB", "2.5")) * (1 << 30)
SWAP_ABORT_PAGES = int(os.environ.get("HOG_SWAP_ABORT_PAGES", "4096"))  # 64 MiB at 16 KiB pages


def vm_counters():
    out = subprocess.check_output(["vm_stat"], text=True)
    def g(label):
        return int(re.search(label + r":\s+(\d+)\.", out).group(1))
    anon, purg = g("Anonymous pages"), g("Pages purgeable")
    used = ((anon - purg) + g("Pages wired down") + g("Pages occupied by compressor")) * PAGE
    return dict(used=used, swapouts=g("Swapouts"))


def main():
    gib = float(sys.argv[1])
    n = int(gib * (1 << 30))
    start = vm_counters()
    room = PHYS - start["used"] - MARGIN
    if n > room:
        print(f"refused: {gib:.1f} GiB asked, {room / 2**30:.1f} GiB is the most this Mac can "
              f"lend without swapping (physical {PHYS / 2**30:.0f} − used {start['used'] / 2**30:.1f} "
              f"− margin {MARGIN / 2**30:.1f})", flush=True)
        sys.exit(3)

    mm = mmap.mmap(-1, n)
    chunk = 64 << 20
    off = 0
    while off < n:
        k = min(chunk, n - off)
        mm[off:off + k] = os.urandom(k)
        off += k
        if vm_counters()["swapouts"] - start["swapouts"] > SWAP_ABORT_PAGES:
            print(f"abort: macOS began swapping while filling ({off / 2**30:.1f} GiB in)", flush=True)
            os._exit(4)

    def watchdog():
        while True:
            grown = vm_counters()["swapouts"] - start["swapouts"]
            if grown > SWAP_ABORT_PAGES:
                print(f"abort: Swapouts +{grown} pages ({grown * PAGE / 2**20:.0f} MiB written); "
                      f"releasing {gib:.1f} GiB", flush=True)
                os._exit(4)
            time.sleep(0.5)
    threading.Thread(target=watchdog, daemon=True).start()

    print("ready", flush=True)
    while True:
        for o in range(0, n, PAGE):
            mm[o]
        time.sleep(1)


main()
