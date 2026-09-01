#!/usr/bin/env python3
"""45 の A/B ログから、エントロピー符号化の「変換率」を出す (docs/mtp/46 §4)。

入力はすべて `bench/mtp45/ab_tps_32slots.log` にある実測値である。
やっているのは割り算だけで、新しい測定はしていない (**導出**)。

問いは 1 つ: **圧縮した blob を展開する側は、どれだけの帯域を出す必要があるか。**
展開が io 区間に完全に隠れて初めて、削ったバイトが t/s になる。
"""

STRIDE = {"affine": 3_719_168, "sym": 3_358_720}   # 45 §4 / layout.json
IO_MS = {"affine": 16.10, "sym": 14.21}            # decode/tok io、3 round の中央値
REQUESTS, HITS, STEPS = 61200, 51394, 255          # 45 §4 (両スキームで同一)

# エキスパート blob の内訳 (packed_experts/layout.json、1 エキスパート)
CODES = 3 * 991_232        # gate/up/down の 4bit コード
SCALES = 3 * 123_904       # bf16 scale
ALIGN = 16_384             # MoEPackFormatV1.alignmentBytes

H_CODE = 3.4509            # bench/mtp46/code_entropy_full.log、per-blob 表

misses = REQUESTS - HITS
per_tok = misses / STEPS
print(f"decode ミス {misses:,} / {STEPS} 歩 = {per_tok:.2f} ミス/tok "
      f"({per_tok / 30:.2f} ミス/層)")
print()
print(f"{'':8s} {'MB/tok':>9s} {'io ms/tok':>10s} {'実効 GB/s':>11s}")
for k in ("affine", "sym"):
    b = per_tok * STRIDE[k]
    print(f"{k:8s} {b / 1e6:9.1f} {IO_MS[k]:10.2f} {b / 1e6 / IO_MS[k]:11.2f}")

blob = CODES + SCALES
ideal = CODES * H_CODE / 4.0 + SCALES
stride_new = -(-ideal // ALIGN) * ALIGN
print()
print(f"blob {blob:,} B (codes {CODES:,} / scales {SCALES:,} / "
      f"pad {STRIDE['sym'] - blob:,} = {100 * (STRIDE['sym'] - blob) / STRIDE['sym']:.2f}%)")
print(f"理想 rANS  {ideal:,.0f} B ({100 * (ideal / blob - 1):+.2f}%)  "
      f"16KB 整列後 stride {stride_new:,.0f} "
      f"({100 * (stride_new / STRIDE['sym'] - 1):+.2f}%)")
print(f"experts 合計 {30 * 128 * STRIDE['sym'] / 2**30:.2f} GiB -> "
      f"{30 * 128 * stride_new / 2**30:.2f} GiB")

gain_ms = IO_MS["sym"] * (1 - stride_new / STRIDE["sym"])
window = IO_MS["sym"] - gain_ms
comp_mb = per_tok * stride_new / 1e6
raw_mb = per_tok * STRIDE["sym"] / 1e6
print()
print(f"取り分   io {gain_ms:.2f} ms/tok (圧縮後も同じ実効帯域が出ると仮定)")
print(f"窓       {window:.2f} ms/tok。この中に展開を完全に隠して初めて取り分が残る")
print(f"必要帯域 入力 {comp_mb:.1f} MB/tok / {window:.2f} ms = "
      f"{comp_mb / window:.2f} GB/s (圧縮ストリームの消費)")
print(f"         出力 {raw_mb:.1f} MB/tok / {window:.2f} ms = "
      f"{raw_mb / window:.2f} GB/s (展開後コードの生成)")
