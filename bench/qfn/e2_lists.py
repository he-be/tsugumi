"""Range lists for E2 (docs/qfn-prefill/07): one layer's routed experts as the runtime reads them
(gate/up from the main GGUF, down from the down/ sidecar), split by expert index between the internal file and an
external mirror.

    python3 bench/qfn/e2_lists.py <model dir> <mirror dir> <layer> <share> > list.txt   # share 0 = internal only, 1 = mirror only
    python3 bench/qfn/e2_lists.py --mirror-plan <model dir> <mirror dir> <first> <layers> # "src offset len dst dstoff" for dd

Ranges files (from Scripts/qwen38/expert_ranges.py) are expected at <model dir>/../expert-ranges.txt and down-ranges.txt
next to this list's caller; see e2_ssd.sh. The mirror keeps each layer's three tensors, whole, in one file
<mirror dir>/layer<L>.bin (gate, up, down in that order), so the split is by expert index inside each tensor.
Experts per tensor: 512, laid out expert-major (each expert contiguous).
"""
import os
import sys

E = 512

def load(path):
    rows = {}
    for line in open(path):
        off, n, name = line.split()
        rows[name] = (int(off), int(n))
    return rows

def layer_tensors(main_ranges, down_ranges, main, down, L):
    return [(main, *main_ranges[f"blk.{L}.ffn_gate_exps.weight"]),
            (main, *main_ranges[f"blk.{L}.ffn_up_exps.weight"]),
            (down, *down_ranges[f"blk.{L}.ffn_down_exps.weight"])]

def main_():
    args = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    plan = args[0] == "--mirror-plan"
    if plan:
        args = args[1:]
    model, mirror = args[0], args[1]
    mr = load(os.environ.get("E2_MAIN_RANGES", "expert-ranges.txt"))
    dr = load(os.environ.get("E2_DOWN_RANGES", "down-ranges.txt"))
    main = os.path.join(model, "Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf")
    down = os.path.join(model, "down", "Qwen3.8-Flash-Next-Q2KDown640.gguf")
    if plan:
        first, layers = int(args[2]), int(args[3])
        for L in range(first, first + layers):
            dst = 0
            for f, off, n in layer_tensors(mr, dr, main, down, L):
                print(f, off, n, os.path.join(mirror, f"layer{L}.bin"), dst)
                dst += n
        return
    L, share = int(args[2]), float(args[3])
    cut = round(E * (1 - share))   # experts [0, cut) internal, [cut, E) mirror
    dst = 0
    for f, off, n in layer_tensors(mr, dr, main, down, L):
        per = n // E
        if cut > 0:
            print(0, f, off, cut * per)
        if cut < E:
            print(1 if cut > 0 else 0, os.path.join(mirror, f"layer{L}.bin"), dst + cut * per, (E - cut) * per)
        dst += n

main_()
