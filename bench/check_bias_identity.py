#!/usr/bin/env python3
"""`.gturbo` の affine 量子化フィールドに何が入っているかを検定する。

使い方: python3 check_bias_identity.py <model.gturbo> [layers]

2 つ測る (docs/mtp/44-W1-WEIGHT-DIET.md §1 / §2):

1. **bias == -8*scale** が bf16 のビットパターンとして成り立つか。
   `-8` は 2 の冪なので bf16 で丸め無しに表現でき、成り立つなら
   bias は scale から完全に導出できる = 保存する必要がない。
   格子整合の QAT チェックポイントでのみ成り立つ (通常の affine 量子化は
   3 分の 1 が外れる)。
2. **行内の exponent span**。scale は全て正で、行内の指数の幅が
   小さければ、行ごとのアンカーに対する短い符号で可逆に格納できる。
   span <= 3 なら 2 ビットの指数差 + bf16 の仮数 7 ビット = 9 ビットで
   ビット一致する。
"""
import json, os, struct, sys
import numpy as np

def bf16(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << 16).view(np.float32)

def expected_bias_bits(s: np.ndarray) -> np.ndarray:
    # -8*s は 2 の冪倍なので bf16 では丸め無しに表現できる
    return ((-8.0 * bf16(s)).view(np.uint32) >> 16).astype(np.uint16)

SPANS = np.zeros(16, dtype=np.int64)

def note_spans(scale_words, rows):
    """行ごとの bf16 exponent の幅を数える。rows で割り切れなければ諦める。"""
    if rows <= 0 or scale_words.size % rows:
        return
    e = ((scale_words.reshape(rows, -1) >> 7) & 0xFF).astype(np.int32)
    span = np.clip(e.max(1) - e.min(1), 0, 15)
    SPANS[:] += np.bincount(span, minlength=16)

def check_resident(path, label):
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        idx_size, _res_size, n = struct.unpack("<QQQ", f.read(24))
        tbl = f.read(72 * n)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    bad = tot = w = sc = bi = 0
    for i in range(n):
        e = tbl[i * 72:(i + 1) * 72]
        _fo, sz = struct.unpack_from("<QQ", e, 8)
        shape = struct.unpack_from("<4I", e, 24)
        so, ss, bo, bs = struct.unpack_from("<4Q", e, 40)
        w += sz; sc += ss; bi += bs
        if ss:
            note_spans(mm[so:so + ss].view(np.uint16), int(shape[0]))
        if bs == 0:
            continue
        s = mm[so:so + ss].view(np.uint16)
        b = mm[bo:bo + bs].view(np.uint16)
        bad += int((expected_bias_bits(s) != b).sum()); tot += b.size
    report(label, tot, bad, bi, w + sc + bi)

def check_experts(root, label, layers=None):
    pe = os.path.join(root, "packed_experts")
    if not os.path.isdir(pe):
        return
    layout = json.load(open(os.path.join(pe, "layout.json")))
    bad = tot = bi = all_bytes = 0
    for L in layout["layers"]:
        if layers is not None and L["layer"] not in layers:
            continue
        mm = np.memmap(os.path.join(pe, L["file"]), dtype=np.uint8, mode="r")
        for ex in L["experts"]:
            for name, t in ex["tensors"].items():
                all_bytes += t["size"]
                if not name.endswith("_biases"):
                    continue
                bi += t["size"]
                st = ex["tensors"][name[:-7] + "_scales"]
                s = mm[ex["offset"] + st["offset"]:][:st["size"]].view(np.uint16)
                b = mm[ex["offset"] + t["offset"]:][:t["size"]].view(np.uint16)
                note_spans(s, int(st["shape"][0]))
                bad += int((expected_bias_bits(s) != b).sum()); tot += b.size
    report(label, tot, bad, bi, all_bytes)

def report(label, tot, bad, bias_bytes, all_bytes):
    pct = 100 * bad / tot if tot else 0.0
    share = 100 * bias_bytes / all_bytes if all_bytes else 0.0
    print(f"{label:34s} groups={tot:>11,d} mismatch={bad:>11,d} ({pct:6.2f}%)  bias share={share:5.2f}%")

if __name__ == "__main__":
    root = sys.argv[1]
    only = None if len(sys.argv) < 3 else {int(x) for x in sys.argv[2].split(",")}
    check_resident(os.path.join(root, "model_weights.bin"), "resident")
    check_resident(os.path.join(root, "draft", "draft_weights.bin"), "draft")
    check_experts(root, "packed experts", only)
    total = SPANS.sum()
    if total:
        print(f"  行内 exponent span ({total:,d} 行):", end="")
        for k, v in enumerate(SPANS):
            if v:
                print(f"  {k}:{100 * v / total:.4f}%", end="")
        print(f"   9 ビット符号で可逆になる行 {100 * SPANS[:4].sum() / total:.4f}%")
