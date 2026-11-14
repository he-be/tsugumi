"""Gemma 4 26B-A4B の GGUF から、層ごとに残すエキスパートだけを抜いた GGUF を書く (再量子化しない、バイトの切り出しだけ)。
    python slice_experts.py <in.gguf> <keep.json> <out.gguf>
keep.json = {"0": [expert ids...], ...}。全層同じ個数。ルーターの行・gate_up・down・down の scale を同じ順で切る。"""
import json, sys
sys.path.insert(0, "/root/llama.cpp/gguf-py")
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

src, keepf, dst = sys.argv[1:4]
keep = {int(k): sorted(v) for k, v in json.load(open(keepf)).items()}
K = len(next(iter(keep.values())))
assert all(len(v) == K and len(set(v)) == K for v in keep.values())
r = GGUFReader(src)
arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
w = GGUFWriter(dst, arch=arch)
for name, f in r.fields.items():
    if name == "general.architecture" or name.startswith("GGUF."):
        continue
    vt = f.types[0]
    if name == f"{arch}.expert_count":
        w.add_uint32(name, K); continue
    if vt == GGUFValueType.ARRAY:
        w.add_array(name, f.contents()) if len(f.types) < 2 else w.add_key_value(name, f.contents(), vt, sub_type=f.types[-1])
    else:
        w.add_key_value(name, f.contents(), vt)
SLICED = ("ffn_gate_up_exps.weight", "ffn_down_exps.weight", "ffn_down_exps.scale", "ffn_gate_inp.weight",
          "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_up_exps.scale", "ffn_gate_exps.scale")
out = []
for t in r.tensors:
    d = t.data
    if t.name.startswith("blk.") and t.name.split(".", 2)[2] in SLICED:
        il = int(t.name.split(".")[1])
        assert d.shape[0] == 128, (t.name, d.shape)
        d = np.ascontiguousarray(d[keep[il]])
    out.append((t, d))
    w.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t, d in out:
    w.write_tensor_data(d)
w.close()
print("wrote", dst, "K", K)
