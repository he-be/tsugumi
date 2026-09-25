"""The M6's MB read from disk0 (internal) and disk4 (external) and its Swapouts between two bench/qfn/pair_serve.sh snaps.

    python3 bench/qfn/m6_cols.py <pre.m6.txt> <post.m6.txt>   # prints: internal_MB external_MB swapouts (tab separated)
"""
import re
import sys


def read(path):
    text = open(path).read()
    swapouts = int(re.search(r"Swapouts:\s+(\d+)", text).group(1))
    io = text.strip().splitlines()[-1].split()   # iostat -d -I disk0 disk4: KB/t xfrs MB  KB/t xfrs MB
    return float(io[2]), float(io[5]), swapouts


a, b = read(sys.argv[1]), read(sys.argv[2])
print(f"{b[0] - a[0]:.0f}\t{b[1] - a[1]:.0f}\t{b[2] - a[2]}")
