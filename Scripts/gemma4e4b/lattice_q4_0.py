#!/usr/bin/env python3
"""QAT (q4_0) の原本を、格子どおりの Q4_0 に詰めた GGUF にする。

使い方: lattice_q4_0.py IN_BF16.gguf OUT.gguf [REPORT.jsonl] [--ple MODEL.safetensors]

--ple: PLE の表 (embed_tokens_per_layer) を原本の safetensors から群ごとに読んで per_layer_token_embd.weight として
足す (convert_text_without_ple.py で表を除いて変換したとき。表は bf16 で 5.6 GB あり、convert の中で丸ごと
展開すると 18 GB の Mac がスワップする)。

IN は convert_hf_to_gguf.py --outtype bf16 で原本 (google/gemma-4-E4B-it-qat-q4_0-unquantized など) から作ったもの。
BF16 のテンソルごとに、32 個の群が q4_0 の格子に乗るかを調べる (格子チェック):

    d > 0、k ∈ [-8, 7] の整数、x == bf16(f32(d) * k)   (群の 32 個すべて)

群の端は -8d のことも ±7d のこともあるので、d の見積もりは max|x|/8 と max|x|/7 の両方を試す。
全群が格子に乗ったテンソルは Q4_0 (d は fp16、q = k + 8) で書く。d は、原本にビット一致で戻る fp16 を
近いものから探す。見つからない群 (fp16 の d では戻らない) は数えて報告し、区間の中央に最も近い fp16 を使う。
1 群でも格子に乗らないテンソルは BF16 のまま書く。BF16 以外 (F32 のノルムなど) はそのまま。

最後に OUT を読み直し、Q4_0 にしたテンソルを bf16(d*k) に戻して IN とビットで比べる (往復の検査)。
"""
import json
import os
import shutil
import struct
import sys

import numpy as np

sys.path.insert(0, "/Users/mh/LLM/llama.cpp/gguf-py")
import gguf  # noqa: E402

GROUP = 32
BLOCK_BYTES = 18
CHUNK_GROUPS = 1 << 18  # 1 回に扱う群の数 (bf16 で 16 MiB、float64 の作業配列で約 64 MiB ずつ)


def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(f):
    u = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + ((u >> 16) & 1) + 0x7FFF) >> 16).astype(np.uint16)


def _interval(x, g, d0):
    """d0 から k を決め、bf16(d*k) == x を満たす d の区間 [L, H] を返す。"""
    k = np.rint(x / d0[:, None])
    ag = g.astype(np.int32)
    up = np.abs(bf16_to_f32(((ag + 1) & 0xFFFF).astype(np.uint16)).astype(np.float64))
    dn = np.abs(bf16_to_f32(((ag - 1) & 0xFFFF).astype(np.uint16)).astype(np.float64))
    ax = np.abs(x)
    hi_abs = (ax + np.maximum(up, dn)) / 2
    lo_abs = (ax + np.minimum(up, dn)) / 2
    ak = np.abs(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.where(ak > 0, lo_abs / ak, 0.0)
        hi = np.where(ak > 0, hi_abs / ak, np.inf)
    L = lo.max(1)
    H = hi.min(1)
    zero_ok = np.where(k == 0, (g & 0x7FFF) == 0, True).all(1)
    sign_ok = np.where(k != 0, np.sign(x) == np.sign(k), True).all(1)
    inr = ((k >= -8) & (k <= 7)).all(1)
    return inr & sign_ok & zero_ok & (L <= H), L, H, k


def lattice(g):
    """g: (G, 32) の bf16 ビット。→ ok (格子に乗る), L, H, k (int8)。全部 0 の群は d = 0, k = 0。"""
    x = bf16_to_f32(g).astype(np.float64)
    amax = np.abs(x).max(1)
    zero = amax == 0
    a = np.where(zero, 1.0, amax)
    ok8, L8, H8, k8 = _interval(x, g, a / 8)
    ok7, L7, H7, k7 = _interval(x, g, a / 7)
    use7 = ~ok8 & ok7
    ok = ok8 | ok7 | zero
    L = np.where(use7, L7, L8)
    H = np.where(use7, H7, H8)
    k = np.where(use7[:, None], k7, k8)
    k[zero] = 0
    L[zero] = 0
    H[zero] = 0
    return ok, L, H, k.astype(np.int8), zero


def pick_d(g, L, H, k, zero):
    """原本にビット一致で戻る fp16 の d を探す。→ d (fp16), exact (bool)。"""
    mid = ((L + H) / 2).astype(np.float16)
    d = mid.copy()
    exact = np.zeros(len(g), bool)
    kf = k.astype(np.float32)
    cand = mid
    order = [0, 1, -1, 2, -2, 3, -3]
    bits = mid.view(np.uint16).astype(np.int32)
    for step in order:
        c = ((bits + step) & 0xFFFF).astype(np.uint16).view(np.float16)
        rec = f32_to_bf16_bits(c.astype(np.float32)[:, None] * kf)
        hit = ~exact & (rec == g).all(1) & (c.astype(np.float32) > 0)
        d[hit] = c[hit]
        exact |= hit
    del cand
    d[zero] = 0
    exact |= zero
    return d, exact


def m12_exact(g, L, H, k, zero):
    """仮数 12 ビット (符号なし・指数は別) の d でビット一致に戻る群か。→ exact (bool), d の 2 進指数。"""
    mid = (L + H) / 2
    m, e = np.frexp(np.where(zero, 1.0, mid))  # mid = m * 2^e, m ∈ [0.5, 1)
    kf = k.astype(np.float32)
    exact = zero.copy()
    base = np.rint(m * 8192)  # 暗黙の 1 + 12 ビット = 13 ビット
    for step in (0, 1, -1, 2, -2):
        c = np.ldexp((base + step) / 8192, e).astype(np.float32)
        rec = f32_to_bf16_bits(c[:, None] * kf)
        exact |= (rec == g).all(1)
    return exact, e[~zero]


def quantize(bits2d, path):
    """bits2d: (rows, cols) の bf16 ビット (cols は 32 の倍数)。→ path に書いた Q4_0 のブロック (G, 18) の memmap
    (格子に乗らない群があれば None) と統計。ブロックは RAM に溜めない。"""
    flat = bits2d.reshape(-1, GROUP)
    n = len(flat)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(n, BLOCK_BYTES))
    stats = {"groups": n, "off_lattice": 0, "not_fp16_exact": 0, "not_m12_exact": 0, "zero_groups": 0, "edge7": 0,
             "d_exp_min": 999, "d_exp_max": -999}
    for s in range(0, n, CHUNK_GROUPS):
        g = np.ascontiguousarray(flat[s:s + CHUNK_GROUPS])
        ok, L, H, k, zero = lattice(g)
        stats["off_lattice"] += int((~ok).sum())
        if stats["off_lattice"]:
            del out
            os.remove(path)
            return None, stats
        d, exact = pick_d(g, L, H, k, zero)
        stats["not_fp16_exact"] += int((~exact).sum())
        ex12, dexp = m12_exact(g, L, H, k, zero)
        stats["not_m12_exact"] += int((~ex12).sum())
        if dexp.size:
            stats["d_exp_min"] = min(stats["d_exp_min"], int(dexp.min()))
            stats["d_exp_max"] = max(stats["d_exp_max"], int(dexp.max()))
        stats["zero_groups"] += int(zero.sum())
        stats["edge7"] += int((np.abs(k).max(1) == 7).sum())
        q = (k + 8).astype(np.uint8)
        out[s:s + len(g), :2] = d.view(np.uint8).reshape(-1, 2)
        out[s:s + len(g), 2:] = q[:, :16] | (q[:, 16:] << 4)
    out.flush()
    return out, stats


def mismatches(blocks, orig):
    """Q4_0 のブロックを bf16(d*k) に戻して、原本の bf16 ビットと違う要素の数。"""
    n = 0
    for s in range(0, len(blocks), CHUNK_GROUPS):
        n += int((dequant_bf16(np.asarray(blocks[s:s + CHUNK_GROUPS])) != orig[s:s + CHUNK_GROUPS]).sum())
    return n


def safetensors_bits(path, name):
    """safetensors の 1 テンソルを、bf16 ビットの memmap で (shape) として返す。"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    info = header[name]
    assert info["dtype"] == "BF16", info["dtype"]
    start, end = info["data_offsets"]
    return np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + start, shape=tuple(info["shape"]))


def dequant_bf16(blocks):
    d = blocks[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)
    nib = blocks[:, 2:]
    q = np.concatenate([nib & 0x0F, nib >> 4], axis=1).astype(np.float32) - 8
    return f32_to_bf16_bits(d[:, None] * q)


def main():
    args = sys.argv[1:]
    ple = None
    if "--ple" in args:
        i = args.index("--ple")
        ple = args[i + 1]
        del args[i:i + 2]
    src, dst = args[0], args[1]
    report = open(args[2], "w") if len(args) > 2 else None
    reader = gguf.GGUFReader(src)
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(dst, arch)
    for field in reader.fields.values():
        if field.name in (gguf.Keys.General.ARCHITECTURE, gguf.Keys.General.FILE_TYPE) or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_0)
    spill = dst + ".blocks"
    os.makedirs(spill, exist_ok=True)

    plan = []  # (tensor, kind)
    for t in reader.tensors:
        is_bf16 = t.tensor_type == gguf.GGMLQuantizationType.BF16
        if is_bf16 and len(t.shape) >= 2 and int(t.shape[0]) % GROUP == 0:
            plan.append((t, "try"))
        else:
            plan.append((t, "copy"))

    # 格子チェックを先に全部やって、Q4_0 にできるかを決める (tensor info を先に書く必要がある)
    results = {}
    total = {"q4_0": 0, "kept": 0}
    sources = [(t.name, [int(x) for x in t.shape], np.asarray(t.data).view(np.uint16).reshape(-1, int(t.shape[0])))
               for t, kind in plan if kind == "try"]
    PLE_NAME = "per_layer_token_embd.weight"
    if ple:
        bits = safetensors_bits(ple, "model.language_model.embed_tokens_per_layer.weight")
        sources.append((PLE_NAME, [int(bits.shape[1]), int(bits.shape[0])], bits))  # GGUF の ne は (列, 行)
    for name, shape, bits in sources:
        path = os.path.join(spill, name + ".npy")
        blocks, stats = quantize(bits, path)
        rec = {"name": name, "shape": shape, **stats}
        if blocks is None:
            rec["result"] = "BF16 のまま (格子に乗らない群あり)"
            results[name] = None
            total["kept"] += 1
            if name == PLE_NAME:
                raise SystemExit("PLE の表が格子に乗らない: この形では書けない")
        else:
            rec["result"] = "Q4_0"
            results[name] = path
            total["q4_0"] += 1
            # 往復の検査はここで先にやる (書き出し後の読み直しでもう一度やる)
            rec["roundtrip_mismatch_elems"] = mismatches(blocks, bits.reshape(-1, GROUP))
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        if report:
            report.write(json.dumps(rec, ensure_ascii=False) + "\n")
            report.flush()
        del blocks

    names = [t.name for t, _ in plan] + ([PLE_NAME] if ple else [])
    tensors = {t.name: t for t, _ in plan}
    ple_rows = sources[-1][2].shape if ple else None
    for name in names:
        path = results.get(name)
        if path is not None:
            blocks = np.load(path, mmap_mode="r")
            rows = ple_rows[0] if name == PLE_NAME else int(np.prod([int(x) for x in tensors[name].shape[1:]]))
            writer.add_tensor_info(name, (rows, blocks.shape[0] // rows * BLOCK_BYTES), np.dtype(np.uint8),
                                   blocks.nbytes, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
        else:
            t = tensors[name]
            writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for name in names:
        path = results.get(name)
        if path is not None:
            blocks = np.load(path, mmap_mode="r")
            rows = ple_rows[0] if name == PLE_NAME else int(np.prod([int(x) for x in tensors[name].shape[1:]]))
            data = blocks.reshape(rows, -1)
        else:
            data = tensors[name].data
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()
    shutil.rmtree(spill)
    print(json.dumps({"summary": total}), flush=True)

    # 読み直して、もう一度往復の検査
    out = gguf.GGUFReader(dst)
    src_t = {name: bits for name, _, bits in sources}
    bad = 0
    n_q = 0
    for t in out.tensors:
        if t.tensor_type != gguf.GGMLQuantizationType.Q4_0:
            same = np.array_equal(np.asarray(t.data).view(np.uint8), np.asarray(tensors[t.name].data).view(np.uint8))
            if not same:
                print("copy mismatch", t.name)
                bad += 1
            continue
        n_q += 1
        blocks = np.asarray(t.data).view(np.uint8).reshape(-1, BLOCK_BYTES)
        mism = mismatches(blocks, src_t[t.name].reshape(-1, GROUP))
        rec = {"reread": t.name, "mismatch_elems": mism}
        if report:
            report.write(json.dumps(rec) + "\n")
        if mism:
            print("roundtrip mismatch", t.name, mism)
            bad += 1
    print(json.dumps({"reread_q4_0_tensors": n_q, "tensors_with_mismatch": bad}), flush=True)


if __name__ == "__main__":
    main()
