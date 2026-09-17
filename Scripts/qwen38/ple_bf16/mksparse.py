"""Q4_1 PLE サイドカーの KV とメタテンソルを写し、ple.weight を BF16 [160, 320001536] にした疎な GGUF を作る (行データは未書き込み = 穴)。"""
import struct, sys, os
src, dst = sys.argv[1], sys.argv[2]
full = len(sys.argv) > 3 and sys.argv[3] == 'full'   # 全行を後から流し込む版 (fetch_full.sh)
b = open(src, 'rb').read(8 << 20)
o = 4
def u(fmt):
    global o
    v = struct.unpack_from(fmt, b, o); o += struct.calcsize(fmt); return v[0]
def skip_val(t):
    global o
    SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t in SZ: o += SZ[t]; return
    if t == 8: n = u('<Q'); o += n; return
    if t == 9:
        et = u('<I'); n = u('<Q')
        for _ in range(n): skip_val(et)
        return
    raise ValueError(t)
ver = u('<I'); nt = u('<Q'); nkv = u('<Q')
kvs = []
for _ in range(nkv):
    s0 = o; n = u('<Q'); k = b[o:o+n].decode(); o += n; t = u('<I'); skip_val(t)
    kvs.append((k, b[s0:o]))
tinfo = []
for _ in range(nt):
    n = u('<Q'); name = b[o:o+n].decode(); o += n
    nd = u('<I'); dims = [u('<Q') for _ in range(nd)]; ty = u('<I'); off = u('<Q')
    tinfo.append([name, dims, ty, off])
align = 32
data0 = (o + align - 1) // align * align
fsz = os.path.getsize(src)
# 元のメタテンソルの中身
meta = {}
for i, (name, dims, ty, off) in enumerate(tinfo):
    if name == 'ple.weight': continue
    end = tinfo[i+1][3] if i + 1 < len(tinfo) else fsz - data0
    with open(src, 'rb') as f:
        f.seek(data0 + off); meta[name] = f.read(end - off)
def kvstr(k, v):
    kb = k.encode(); vb = v.encode()
    return struct.pack('<Q', len(kb)) + kb + struct.pack('<I', 8) + struct.pack('<Q', len(vb)) + vb
out_kv = []
for k, raw in kvs:
    if k == 'ds4.pack.quant.ple': out_kv.append(kvstr(k, 'BF16'))
    elif k == 'general.name': out_kv.append(kvstr(k, 'Qwen3.8-Flash-Next PLE BF16' + ('' if full else ' (sparse rows') + ' from antirez/qwen3.8-flash-next-gguf Q2' + ('' if full else ')')))
    else: out_kv.append(raw)
out_kv.append(kvstr('tsugumi.ple.source' if full else 'tsugumi.ple.sparse_source', 'antirez/qwen3.8-flash-next-gguf/Qwen3.8-Flash-Next-Q2.gguf per_layer_token_embd.weight'))
ROWS = 320001536
new_t = []; cur = 0
def pad(x): return (x + align - 1) // align * align
for name, dims, ty, off in tinfo:
    if name == 'ple.weight':
        new_t.append((name, dims, 30, cur)); cur = pad(cur + ROWS * 320)
    else:
        new_t.append((name, dims, ty, cur)); cur = pad(cur + len(meta[name]))
hdr = b'GGUF' + struct.pack('<IQQ', ver, len(new_t), len(out_kv)) + b''.join(out_kv)
for name, dims, ty, off in new_t:
    nb = name.encode()
    hdr += struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<IQ', ty, off)
d0 = pad(len(hdr))
with open(dst, 'wb') as f:
    f.write(hdr + b'\0' * (d0 - len(hdr)))
    for name, dims, ty, off in new_t:
        if name != 'ple.weight':
            f.seek(d0 + off); f.write(meta[name])
    f.truncate(d0 + cur)
print('data_start', d0, 'ple.weight offset', [t[3] for t in new_t if t[0]=='ple.weight'][0], 'size', d0 + cur)
