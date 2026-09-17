#!/usr/bin/env python3
"""Print the metadata and tensor table of a GGUF from its first bytes only.

The header of Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf ends at 10,995,772 B (docs/qwen38-27b/01 §1),
so a 24 MB Range fetch is enough:

    curl -sL -r 0-25165823 -o head.bin \
      https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
    python3 Scripts/qwen38_27b/gguf_header.py head.bin [--all-tensors]

Without --all-tensors only the non-block tensors and blocks 0, 3 and the last one are listed.
"""
import struct
import sys

SCALARS = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}


class Reader:
    def __init__(self, data):
        self.d = data
        self.o = 0

    def scalar(self, fmt):
        v = struct.unpack_from(fmt, self.d, self.o)[0]
        self.o += struct.calcsize(fmt)
        return v

    def string(self):
        n = self.scalar('<Q')
        v = self.d[self.o:self.o + n].decode(errors='replace')
        self.o += n
        return v

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et = self.scalar('<I')
            n = self.scalar('<Q')
            a = [self.value(et) for _ in range(n)]
            return a if n <= 8 else f'[{n} x type {et}] {a[:3]}'
        return self.scalar(SCALARS[t])


def main():
    path = sys.argv[1]
    all_tensors = '--all-tensors' in sys.argv
    r = Reader(open(path, 'rb').read())
    assert r.d[:4] == b'GGUF', 'not a GGUF'
    r.o = 4
    print('version', r.scalar('<I'))
    n_tensors = r.scalar('<Q')
    n_kv = r.scalar('<Q')
    for _ in range(n_kv):
        key = r.string()
        v = r.value(r.scalar('<I'))
        print(key, '=', str(v)[:120])
    rows = []
    for _ in range(n_tensors):
        name = r.string()
        shape = [r.scalar('<Q') for _ in range(r.scalar('<I'))]
        ggml_type = r.scalar('<I')
        r.scalar('<Q')  # offset
        rows.append((name, shape, ggml_type))
    last = max(int(n.split('.')[1]) for n, _, _ in rows if n.startswith('blk.'))
    for name, shape, ggml_type in rows:
        parts = name.split('.')
        if all_tensors or parts[0] != 'blk' or parts[1] in ('0', '3', str(last)):
            print(name, shape, ggml_type)
    print('header end', r.o)


if __name__ == '__main__':
    main()
