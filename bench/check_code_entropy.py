#!/usr/bin/env python3
"""`.moepack` の INT4 コードそのものの分布を数える (docs/mtp/46 の §1)。

使い方: python3 check_code_entropy.py <model.moepack> [layers]
        (layers は "0" や "0,15,29"。省略で全 30 層)

28-M8-PROPOSAL §5 は「エキスパートの zstd/LZ4 圧縮」を
**「int4 QAT 重みは高エントロピーで 1 割も縮まない見込み」**という理由で却下した。
これは repo の中で数少ない「測っていない却下」である。本スクリプトはそれを実測に
格上げする。測るのは 3 つ:

1. **order-0 エントロピー** `H(q)`。コード q は 0..15 で、格子上の値は c = q-8。
   静的テーブルの rANS が到達できる下限が `H` bits/weight である。
   現行の `sym` は 4.000 bits/weight + scale 0.500 = **4.500 bpw**。
   テーブルの粒度を 2 通りで出す:
     - **per-role**: テンソルの役割 (gate/up/down) ごとに 1 つの表
     - **per-blob**: (層, エキスパート, 役割) ごとに 1 つの表。
       表は 16 シンボルなので数十バイトで、991 KB のテンソルに対して無視できる。
       エキスパート blob 単位で独立にデコードできる要件とも一致する
2. **バイト対の order-0 エントロピー** `H(byte)/2`。1 バイトは k 方向に隣り合う
   2 つのコードなので、`H(byte)/2 < H(q)` なら隣接相関が残っている
3. **群 (32 コード = 16 バイト) ごとの振幅**。適応ビット幅 (「中間案」) の上限を出す。
   固定窓 (|c| <= 3 なら 3bit) と min オフセット付きの 2 通り

**読まないもの**: scale/bias のページには触らない (44 §1/§2 が既に測った)。
本スクリプトが読むのは 4bit スロットの重みページだけである。
"""
import json, math, os, struct, sys
import numpy as np

GROUP = 32                      # production の affine 群サイズ (ArchConfig)
BYTES_PER_GROUP = GROUP // 2    # 16
SCALE_BITS_PER_WEIGHT = 16.0 / GROUP   # bf16 scale 1 個 / 32 コード = 0.500


def entropy_bits(hist: np.ndarray) -> float:
    """order-0 エントロピー (bits/symbol)。空なら 0。"""
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


class Census:
    """役割ごとに、バイト 256 ビンとコード 16 ビンと群の振幅を貯める。"""

    def __init__(self):
        self.byte_hist = {}        # role -> (256,) int64
        self.blob_bits = {}        # role -> [(H_blob, weights), ...]
        self.amp_hist = {}         # role -> (16,) int64   max|c| の分布
        self.span_hist = {}        # role -> (16,) int64   (max q - min q) の分布
        self.weights = {}          # role -> int

    def _slot(self, role):
        if role not in self.byte_hist:
            self.byte_hist[role] = np.zeros(256, dtype=np.int64)
            self.blob_bits[role] = []
            self.amp_hist[role] = np.zeros(16, dtype=np.int64)
            self.span_hist[role] = np.zeros(16, dtype=np.int64)
            self.weights[role] = 0
        return role

    def add(self, role, buf: np.ndarray):
        """buf は 4bit パックされた重みの生バイト列。"""
        self._slot(role)
        bh = np.bincount(buf, minlength=256).astype(np.int64)
        self.byte_hist[role] += bh
        self.weights[role] += 2 * buf.size

        # このブロブ単独の表で符号化したときのビット数
        nib = np.zeros(16, dtype=np.int64)
        idx = np.arange(256)
        np.add.at(nib, idx & 0x0F, bh)
        np.add.at(nib, idx >> 4, bh)
        self.blob_bits[role].append((entropy_bits(nib), 2 * buf.size))

        # 群ごとの振幅。末端が半端なら切り捨てる (production は必ず割り切れる)
        n_groups = buf.size // BYTES_PER_GROUP
        if n_groups:
            g = buf[: n_groups * BYTES_PER_GROUP].reshape(n_groups, BYTES_PER_GROUP)
            lo = g & 0x0F
            hi = g >> 4
            qmax = np.maximum(lo.max(1), hi.max(1)).astype(np.int32)
            qmin = np.minimum(lo.min(1), hi.min(1)).astype(np.int32)
            amp = np.maximum(np.abs(qmax - 8), np.abs(qmin - 8))
            self.amp_hist[role] += np.bincount(np.clip(amp, 0, 15), minlength=16)
            self.span_hist[role] += np.bincount(np.clip(qmax - qmin, 0, 15),
                                                minlength=16)


def nibble_hist(byte_hist: np.ndarray) -> np.ndarray:
    nib = np.zeros(16, dtype=np.int64)
    idx = np.arange(256)
    np.add.at(nib, idx & 0x0F, byte_hist)
    np.add.at(nib, idx >> 4, byte_hist)
    return nib


def scan_experts(root, census, layers=None):
    pe = os.path.join(root, "packed_experts")
    if not os.path.isdir(pe):
        return
    layout = json.load(open(os.path.join(pe, "layout.json")))
    for L in layout["layers"]:
        if layers is not None and L["layer"] not in layers:
            continue
        mm = np.memmap(os.path.join(pe, L["file"]), dtype=np.uint8, mode="r")
        for ex in L["experts"]:
            base = ex["offset"]
            for name, t in ex["tensors"].items():
                if t.get("bits") != 4:
                    continue
                buf = np.asarray(mm[base + t["offset"]:][: t["size"]])
                census.add(name, buf)
        del mm
        print(f"  layer {L['layer']:02d} done", file=sys.stderr, flush=True)


def scan_resident(path, census, label):
    """resident の 4bit スロット。役割名は語彙側とそれ以外に粗く分ける。"""
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        idx_size, _res, n = struct.unpack("<QQQ", f.read(24))
        tbl = f.read(72 * n)
        f.seek(0)
        head = f.read(idx_size)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    for i in range(n):
        e = tbl[i * 72:(i + 1) * 72]
        no, nl = struct.unpack_from("<IH", e, 0)
        name = head[no:no + nl].decode("utf-8")
        fo, sz = struct.unpack_from("<QQ", e, 8)
        shape = struct.unpack_from("<4I", e, 24)
        ss = struct.unpack_from("<Q", e, 48)[0]
        numel = 1
        for d in shape:
            if d:
                numel *= d
        if ss == 0 or numel == 0 or sz * 8 // numel != 4:
            continue          # 4bit スロットだけを見る
        role = f"{label}:embed" if "embed_tokens" in name else f"{label}:dense"
        census.add(role, np.asarray(mm[fo:fo + sz]))
    del mm


def report(census):
    print()
    print("== order-0 エントロピー (bits/weight) ==")
    print(f"{'role':16s} {'weights':>15s} {'H(q) per-role':>14s} "
          f"{'H(q) per-blob':>14s} {'H(byte)/2':>11s} {'bpw*':>8s} {'対 4.500':>9s}")
    grand = np.zeros(256, dtype=np.int64)
    grand_bits = 0.0
    grand_w = 0
    for role in sorted(census.byte_hist):
        bh = census.byte_hist[role]
        nib = nibble_hist(bh)
        h_role = entropy_bits(nib)
        blob = census.blob_bits[role]
        w = census.weights[role]
        h_blob = sum(h * n for h, n in blob) / w
        h_byte = entropy_bits(bh) / 2.0
        bpw = h_blob + SCALE_BITS_PER_WEIGHT
        print(f"{role:16s} {w:>15,d} {h_role:>14.4f} {h_blob:>14.4f} "
              f"{h_byte:>11.4f} {bpw:>8.3f} {100 * (bpw / 4.5 - 1):>8.2f}%")
        grand += bh
        grand_bits += h_blob * w
        grand_w += w
    if grand_w:
        nib = nibble_hist(grand)
        h_all = entropy_bits(nib)
        h_blob = grand_bits / grand_w
        bpw = h_blob + SCALE_BITS_PER_WEIGHT
        print(f"{'ALL':16s} {grand_w:>15,d} {h_all:>14.4f} {h_blob:>14.4f} "
              f"{entropy_bits(grand) / 2.0:>11.4f} {bpw:>8.3f} "
              f"{100 * (bpw / 4.5 - 1):>8.2f}%")
        print("  * bpw = H(q) per-blob + scale 0.500 bits/weight。"
              "現行 sym は 4.000 + 0.500 = 4.500")

        print()
        print("== コードの分布 (全体) ==")
        tot = nib.sum()
        for q in range(16):
            bar = "#" * int(round(60 * nib[q] / nib.max()))
            print(f"  q={q:>2d} (c={q - 8:>3d}) {100 * nib[q] / tot:6.3f}%  {bar}")

    print()
    print("== 群 (32 コード) の振幅 ==")
    for role in sorted(census.amp_hist):
        amp = census.amp_hist[role]
        span = census.span_hist[role]
        n = amp.sum()
        if n == 0:
            continue
        # 固定窓: max|c| <= 3 なら 3bit ([-4,3])、<= 1 なら 2bit ([-2,1])
        w3 = 100 * amp[:4].sum() / n
        w2 = 100 * amp[:2].sum() / n
        # min オフセット付き: span <= 7 で 3bit、<= 3 で 2bit
        s3 = 100 * span[:8].sum() / n
        s2 = 100 * span[:4].sum() / n
        print(f"  {role:14s} 群 {n:>12,d}  固定窓 3bit {w3:6.2f}% / 2bit {w2:5.2f}%"
              f"   オフセット付き 3bit {s3:6.2f}% / 2bit {s2:5.2f}%")


if __name__ == "__main__":
    root = sys.argv[1]
    only = None if len(sys.argv) < 3 else {int(x) for x in sys.argv[2].split(",")}
    census = Census()
    scan_resident(os.path.join(root, "model_weights.bin"), census, "resident")
    scan_experts(root, census, only)
    report(census)
