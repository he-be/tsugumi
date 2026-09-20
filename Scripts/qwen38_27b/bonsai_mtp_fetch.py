#!/usr/bin/env python3
"""MTP 同梱版の GGUF を、手元の PQ2_0 + Range 取得した末尾だけで組み立てる。

`ProCreations/Ternary-Bonsai-2-27B-MTP` は公式 `prism-ml/…-PQ2_0.gguf` の 851 テンソルを
バイト不変のまま残し、`blk.64.*` (MTP 15 本、Q8_0) を末尾に足しただけのファイル。
本体は手元にあるので、遠隔からはヘッダと末尾だけ取れば足りる (7.66 GB -> 467 MB)。

  python3 Scripts/qwen38_27b/bonsai_mtp_fetch.py \
      ~/LLM/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf \
      ~/LLM/Ternary-Bonsai-2-27B-MTP/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf

最後に SHA256SUMS (公開物) と突き合わせるので、本体がバイト同一であることの確認も兼ねる。
"""
from __future__ import annotations

import hashlib
import struct
import sys
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP/resolve/main"
BUNDLE = "Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf"
SHA256 = "3cb3f0056d2e34ee44245a64396004a21f8492573d6ce1266ec4b7222c131dd4"

SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
          6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


class _NeedMore(Exception):
    pass


class _R:
    def __init__(self, b): self.b, self.p = b, 0

    def take(self, n):
        if self.p + n > len(self.b): raise _NeedMore
        v = self.b[self.p:self.p + n]; self.p += n; return v

    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def s(self): return self.take(self.u64()).decode("utf-8", "replace")


def _value(r, t):
    if t == 8:
        return r.s()
    if t == 9:
        et, n = r.u32(), r.u64()
        if et == 8: return [r.s() for _ in range(n)]
        if et == 9: return [_value(r, 9) for _ in range(n)]
        f, sz = SCALAR[et]
        return list(struct.unpack("<" + f[1] * n, r.take(sz * n)))
    f, sz = SCALAR[t]
    return struct.unpack(f, r.take(sz))[0]


def read_header(buf: bytes) -> dict:
    """GGUF のヘッダ (KV + テンソル情報) を読む。バイトが足りなければ _NeedMore。"""
    r = _R(buf)
    assert r.take(4) == b"GGUF"
    r.u32()
    n_tensors, n_kv = r.u64(), r.u64()
    kv = {}
    for _ in range(n_kv):
        k = r.s(); kv[k] = _value(r, r.u32())
    tensors = []
    for _ in range(n_tensors):
        name, nd = r.s(), r.u32()
        dims = [r.u64() for _ in range(nd)]
        tensors.append({"name": name, "dims": dims, "type": r.u32(), "offset": r.u64()})
    align = kv.get("general.alignment", 32)
    return {"kv": kv, "tensors": tensors, "data_start": (r.p + align - 1) // align * align}


def _get(url: str, first: int, last: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def main(base_path: str, out_path: str) -> int:
    url = f"{REPO}/{BUNDLE}"
    base = Path(base_path)

    n = 1 << 20
    while True:                      # 遠隔のヘッダ (本体より 15 本分だけ長い)
        head = _get(url, 0, n - 1)
        try:
            rh = read_header(head); break
        except _NeedMore:
            n *= 2
            if n > (1 << 28): raise

    n = 1 << 20
    while True:
        try:
            lh = read_header(base.open("rb").read(n)); break
        except _NeedMore:
            n *= 2

    local = {t["name"]: t for t in lh["tensors"]}
    for t in rh["tensors"]:          # 本体側は名前・形・型・オフセットまで一致しているはず
        o = local.get(t["name"])
        if o is None:
            continue
        assert o["dims"] == t["dims"] and o["type"] == t["type"] and o["offset"] == t["offset"], t["name"]
    mtp = [t for t in rh["tensors"] if t["name"] not in local]
    base_bytes = base.stat().st_size - lh["data_start"]
    assert min(t["offset"] for t in mtp) == base_bytes, "MTP が末尾の連続領域に無い"
    first = rh["data_start"] + base_bytes
    print(f"MTP {len(mtp)} 本。遠隔から取るのは ヘッダ {rh['data_start']:,} B + 末尾 "
          f"{first:,} B 以降")

    h = hashlib.sha256()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as w:
        head = head[:rh["data_start"]]
        w.write(head); h.update(head)
        with base.open("rb") as f:   # 手元の本体のデータ領域
            f.seek(lh["data_start"])
            while (b := f.read(1 << 22)):
                w.write(b); h.update(b)
        step = 1 << 26               # 遠隔の末尾 (MTP)
        pos = first
        while True:
            b = _get(url, pos, pos + step - 1)
            if not b: break
            w.write(b); h.update(b); pos += len(b)
            if len(b) < step: break
    ok = h.hexdigest() == SHA256
    print(f"{out} {out.stat().st_size:,} B  sha256 {'一致' if ok else '不一致 ' + h.hexdigest()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
