import Foundation
import Metal
import TurboFieldfare

// MARK: - Weight-format probe: what does the routed MoE rows kernel cost per byte?
//
// `docs/mtp/43-P2-MOE-ROWS-TAIL.md` §2 established two things about these two
// kernels at the production shape. First, `gate/up` sits on the 135 GB/s floor
// 20-M4.8 §3 measured (133-137 GB/s across every shape in the sweep), so its
// time is bytes. Second, `down` does not (89.2 GB/s = 66% of the floor); its
// time is the six-group tail's per-row cost.
//
// The bytes those numbers count are 5.0 bits per weight: at affine group 32 one
// group is 16 bytes of packed nibbles, a BF16 scale and a BF16 bias. Two of
// those twenty bytes are the bias, and on the lattice-aligned QAT checkpoint
// the bias is not data -- `bias == -8 * scale` holds bit-for-bit over all
// 788,175,872 groups of `scratch/gemma4-qat.gturbo` (44 §1). A third and a bit
// of the scale field is not data either: every scale is positive, and the
// per-row exponent span is <= 1 for 97.9% of rows and <= 3 for all of them, so
// nine of the sixteen bits carry the information (44 §2).
//
// This probe puts the three formats side by side on the same shapes:
//
//   `affine`  q * s + b, BF16 s and b per group                   5.000 bpw
//   `sym`     s * (q - 8), BF16 s per group                       4.500 bpw
//   `sym8`    s * (q - 8), s an 8-bit code against a per-row      4.256 bpw
//             exponent anchor (2 bits exponent offset, 6 bits
//             mantissa, assembled straight into an fp32 bit pattern)
//
// The arithmetic is deliberately held constant: `affine` accumulates
// `fma(s, dot, acc); fma(b, sum, acc)` and the two symmetric forms accumulate
// `fma(s, fma(-8, sum, dot), acc)`. Two FMAs either way, the same `sum` of
// eight activations, the same loads of x. **Only the weight-side bytes move.**
// If `gate/up` is on the floor its time must follow them; if it does not, the
// floor reading in 43 §2 was not what it looked like.
//
// The kernels here are a self-contained copy of `prefill_moe_rows_gate_up_act`
// and `prefill_moe_rows_down` compiled into their own MTLLibrary. Nothing in
// `Sources/TurboFieldfare/` is touched, and §1 of the probe's output checks the
// copy against 43 §2's numbers before anything else is read off it.

private let bpwProbeSource = #"""
#include <metal_stdlib>
using namespace metal;

#ifndef TURBO_AFFINE_GROUP_SIZE
#define TURBO_AFFINE_GROUP_SIZE 32
#endif
constant constexpr uint kGroupSize        = TURBO_AFFINE_GROUP_SIZE;
constant constexpr uint kGroupsPerBlock   = 256u / kGroupSize;
constant constexpr uint kLanesPerGroup    = kGroupSize / 8u;
constant constexpr uint kTailLanes        = kGroupSize / 2u;
constant constexpr uint kRowsMax          = 8;
constant constexpr uint kRowsPerTG        = 8;
constant constexpr uint kMaxTileExperts   = 16;

// Cap, as in production (`FC_MOE_ROWS_CAP`). Weight format: 0 affine, 1 sym,
// 2 sym8.
constant uint FC_CAP [[function_constant(16)]];
constant uint FC_FMT [[function_constant(17)]];

struct BpwPair  { uint token; uint expert; uint rank; uint reserved; };
struct BpwBlock { uint local_slot; uint pair_start; uint row_count; uint local_row; };
struct BpwBlobs { device const uint8_t* blob[kMaxTileExperts]; };
struct BpwParams {
    uint D; uint F; uint top_k; uint hidden_stride;
    uint gate_w; uint gate_s; uint gate_a;
    uint up_w;   uint up_s;   uint up_a;
    uint down_w; uint down_s; uint down_a;
};

static inline float bpw_gelu(float x) {
    const float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

/// One group's scale, and -- for `affine` only -- its bias.
///
/// `sym8` rebuilds an fp32 from the row anchor and the code: the anchor is the
/// biased fp32 exponent of the row's largest scale, the code's top two bits
/// step the exponent down and its low six bits are the mantissa's top six.
/// Four integer ops, and they are outside the row loop -- the same place the
/// BF16 converts already were.
struct BpwScale { float s; float b; };
static inline BpwScale bpw_scale(device const bfloat* s16,
                                 device const bfloat* b16,
                                 device const uint8_t* s8,
                                 uint anchor, uint g) {
    BpwScale out;
    if (FC_FMT == 0u) {
        out.s = float(s16[g]);
        out.b = float(b16[g]);
        return out;
    }
    if (FC_FMT == 1u) {
        out.s = float(s16[g]);
    } else if (FC_FMT == 2u) {
        const uint code = uint(s8[g]);
        out.s = as_type<float>(((anchor - (code >> 6)) << 23) | ((code & 63u) << 17));
    } else {
        // sym9: two bits of exponent step and BF16's seven mantissa bits, so
        // the reconstruction is the stored BF16 scale exactly. Nine bits start
        // at bit 9*g of the row, which spans at most two bytes.
        const uint bit = g * 9u;
        const uint byte = bit >> 3;
        const uint code = ((uint(s8[byte]) | (uint(s8[byte + 1u]) << 8)) >> (bit & 7u))
                          & 0x1FFu;
        out.s = as_type<float>(((anchor - (code >> 7)) << 23) | ((code & 0x7Fu) << 16));
    }
    out.b = -8.0f * out.s;
    return out;
}

/// Bytes one row of sym9 scale codes occupies, padded so rows stay 4-aligned.
static inline uint bpw_sym9_row_stride(uint n_groups) {
    // One spare byte: the last group's nine bits are read as two bytes.
    return ((((n_groups * 9u + 7u) >> 3) + 1u) + 3u) & ~3u;
}

/// `acc += s * dot + b * sum` for affine, `acc += s * (dot - 8 * sum)` for the
/// symmetric forms. Two FMAs in both branches.
/// The same two FMAs in the same order for every scheme -- which is what makes
/// the symmetric library bit-identical to the affine one on weights that
/// satisfy the identity, since `-8 * scale` is exact in BF16. `sb.b` is the
/// loaded bias for `affine` and the derived `-8 * s` for the others.
static inline float bpw_accumulate(float acc, BpwScale sb, float dot, float sum) {
    acc = fma(sb.s, dot, acc);
    return fma(sb.b, sum, acc);
}

kernel void bpw_rows_gate_up(
    device const half*     hidden       [[buffer(0)]],
    device const BpwPair*  sorted_pairs [[buffer(1)]],
    device const BpwBlock* blocks       [[buffer(2)]],
    device half*           act          [[buffer(7)]],
    device const BpwBlobs& routed       [[buffer(9)]],
    constant BpwParams&    p            [[buffer(10)]],
    uint2                  tgid         [[threadgroup_position_in_grid]],
    uint                   sg_idx       [[simdgroup_index_in_threadgroup]],
    uint                   lane         [[thread_index_in_simdgroup]]
) {
    const BpwBlock blk = blocks[tgid.y];
    const uint f = tgid.x * kRowsPerTG + sg_idx;
    if (f >= p.F || blk.row_count == 0u) return;
    const uint cap = FC_CAP;
    const uint rows = min(blk.row_count, cap);

    device const uint8_t* base = routed.blob[blk.local_slot];
    const uint n_groups = p.D / kGroupSize;
    const uint row_bytes = p.D / 2u;
    device const uint8_t* gW_row = base + p.gate_w + f * row_bytes;
    device const uint8_t* uW_row = base + p.up_w + f * row_bytes;
    // Scale rows. `affine` and `sym` index BF16, `sym8` indexes bytes; the
    // unused pointer is never dereferenced because FC_FMT specializes the load.
    device const bfloat* gS16 = (device const bfloat*)(base + p.gate_s) + f * n_groups;
    device const bfloat* uS16 = (device const bfloat*)(base + p.up_s) + f * n_groups;
    device const bfloat* gB16 = (device const bfloat*)(base + p.gate_a) + f * n_groups;
    device const bfloat* uB16 = (device const bfloat*)(base + p.up_a) + f * n_groups;
    const uint s8_stride = (FC_FMT == 3u) ? bpw_sym9_row_stride(n_groups) : n_groups;
    device const uint8_t* gS8 = base + p.gate_s + f * s8_stride;
    device const uint8_t* uS8 = base + p.up_s + f * s8_stride;
    uint gAnchor = 0u, uAnchor = 0u;
    if (FC_FMT >= 2u) {
        gAnchor = uint((base + p.gate_a)[f]);
        uAnchor = uint((base + p.up_a)[f]);
    }

    device const half* x_row[kRowsMax];
    for (uint r = 0; r < cap; ++r) {
        const uint pick = min(r, rows - 1u);
        const BpwPair pr = sorted_pairs[blk.pair_start + pick];
        x_row[r] = hidden + pr.token * p.hidden_stride;
    }

    float g_acc[kRowsMax];
    float u_acc[kRowsMax];
    for (uint r = 0; r < cap; ++r) { g_acc[r] = 0.0f; u_acc[r] = 0.0f; }

    const uint full_blocks = n_groups / kGroupsPerBlock;
    for (uint b = 0; b < full_blocks; ++b) {
        const uint byte_base = b * 128u + lane * 4u;
        device const ushort* gp = (device const ushort*)(gW_row + byte_base);
        device const ushort* up = (device const ushort*)(uW_row + byte_base);
        const uint gw4 = uint(gp[0]) | (uint(gp[1]) << 16);
        const uint uw4 = uint(up[0]) | (uint(up[1]) << 16);
        const uint g = b * kGroupsPerBlock + lane / kLanesPerGroup;
        const BpwScale gsb = bpw_scale(gS16, gB16, gS8, gAnchor, g);
        const BpwScale usb = bpw_scale(uS16, uB16, uS8, uAnchor, g);
        const uint elem = byte_base * 2u;
        const uint gb0 = gw4 & 0xFFu, gb1 = (gw4 >> 8) & 0xFFu;
        const uint gb2 = (gw4 >> 16) & 0xFFu, gb3 = (gw4 >> 24) & 0xFFu;
        const uint ub0 = uw4 & 0xFFu, ub1 = (uw4 >> 8) & 0xFFu;
        const uint ub2 = (uw4 >> 16) & 0xFFu, ub3 = (uw4 >> 24) & 0xFFu;
        for (uint r = 0; r < cap; ++r) {
            if (r >= rows) break;
            const half4 xa = *((device const half4*)(x_row[r] + elem));
            const half4 xb = *((device const half4*)(x_row[r] + elem + 4u));
            const float e0 = float(xa.x), e1 = float(xa.y);
            const float e2 = float(xa.z), e3 = float(xa.w);
            const float e4 = float(xb.x), e5 = float(xb.y);
            const float e6 = float(xb.z), e7 = float(xb.w);
            const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
            float g_dot = 0.0f;
            g_dot = fma(float(gb0 & 0x0Fu), e0, g_dot); g_dot = fma(float(gb0 >> 4), e1, g_dot);
            g_dot = fma(float(gb1 & 0x0Fu), e2, g_dot); g_dot = fma(float(gb1 >> 4), e3, g_dot);
            g_dot = fma(float(gb2 & 0x0Fu), e4, g_dot); g_dot = fma(float(gb2 >> 4), e5, g_dot);
            g_dot = fma(float(gb3 & 0x0Fu), e6, g_dot); g_dot = fma(float(gb3 >> 4), e7, g_dot);
            float u_dot = 0.0f;
            u_dot = fma(float(ub0 & 0x0Fu), e0, u_dot); u_dot = fma(float(ub0 >> 4), e1, u_dot);
            u_dot = fma(float(ub1 & 0x0Fu), e2, u_dot); u_dot = fma(float(ub1 >> 4), e3, u_dot);
            u_dot = fma(float(ub2 & 0x0Fu), e4, u_dot); u_dot = fma(float(ub2 >> 4), e5, u_dot);
            u_dot = fma(float(ub3 & 0x0Fu), e6, u_dot); u_dot = fma(float(ub3 >> 4), e7, u_dot);
            g_acc[r] = bpw_accumulate(g_acc[r], gsb, g_dot, sum);
            u_acc[r] = bpw_accumulate(u_acc[r], usb, u_dot, sum);
        }
    }
    for (uint g = full_blocks * kGroupsPerBlock; g < n_groups; ++g) {
        if (lane >= kTailLanes) break;
        const BpwScale gsb = bpw_scale(gS16, gB16, gS8, gAnchor, g);
        const BpwScale usb = bpw_scale(uS16, uB16, uS8, uAnchor, g);
        const uint8_t gbv = gW_row[g * (kGroupSize / 2u) + lane];
        const uint8_t ubv = uW_row[g * (kGroupSize / 2u) + lane];
        for (uint r = 0; r < cap; ++r) {
            if (r >= rows) break;
            const float x0 = float(x_row[r][g * kGroupSize + lane * 2u]);
            const float x1 = float(x_row[r][g * kGroupSize + lane * 2u + 1u]);
            const float sum = x0 + x1;
            float g_dot = fma(float(uint(gbv & 0x0Fu)), x0, 0.0f);
            g_dot = fma(float(uint(gbv >> 4)), x1, g_dot);
            float u_dot = fma(float(uint(ubv & 0x0Fu)), x0, 0.0f);
            u_dot = fma(float(uint(ubv >> 4)), x1, u_dot);
            g_acc[r] = bpw_accumulate(g_acc[r], gsb, g_dot, sum);
            u_acc[r] = bpw_accumulate(u_acc[r], usb, u_dot, sum);
        }
    }

    for (uint r = 0; r < cap; ++r) {
        if (r >= rows) break;
        const float gate = simd_sum(g_acc[r]);
        const float up_value = simd_sum(u_acc[r]);
        if (lane == 0) {
            act[(blk.local_row + r) * p.F + f] = half(bpw_gelu(gate) * up_value);
        }
    }
}

kernel void bpw_rows_down(
    device const BpwPair*  sorted_pairs   [[buffer(1)]],
    device const BpwBlock* blocks         [[buffer(2)]],
    device half*           route_partials [[buffer(5)]],
    device const half*     act            [[buffer(7)]],
    device const BpwBlobs& routed         [[buffer(9)]],
    constant BpwParams&    p              [[buffer(10)]],
    uint2                  tgid           [[threadgroup_position_in_grid]],
    uint                   sg_idx         [[simdgroup_index_in_threadgroup]],
    uint                   lane           [[thread_index_in_simdgroup]]
) {
    const BpwBlock blk = blocks[tgid.y];
    const uint d = tgid.x * kRowsPerTG + sg_idx;
    if (d >= p.D || blk.row_count == 0u) return;
    const uint cap = FC_CAP;
    const uint rows = min(blk.row_count, cap);

    device const uint8_t* base = routed.blob[blk.local_slot];
    const uint n_groups = p.F / kGroupSize;
    const uint row_bytes = p.F / 2u;
    device const uint8_t* W_row = base + p.down_w + d * row_bytes;
    device const bfloat* S16 = (device const bfloat*)(base + p.down_s) + d * n_groups;
    device const bfloat* B16 = (device const bfloat*)(base + p.down_a) + d * n_groups;
    const uint s8_stride = (FC_FMT == 3u) ? bpw_sym9_row_stride(n_groups) : n_groups;
    device const uint8_t* S8 = base + p.down_s + d * s8_stride;
    uint anchor = 0u;
    if (FC_FMT >= 2u) { anchor = uint((base + p.down_a)[d]); }

    device const half* x_row[kRowsMax];
    uint dst[kRowsMax];
    for (uint r = 0; r < cap; ++r) {
        const uint pick = min(r, rows - 1u);
        const BpwPair pr = sorted_pairs[blk.pair_start + pick];
        x_row[r] = act + (blk.local_row + pick) * p.F;
        dst[r] = (pr.token * p.top_k + pr.rank) * p.D;
    }

    float acc[kRowsMax];
    for (uint r = 0; r < cap; ++r) { acc[r] = 0.0f; }

    const uint full_blocks = n_groups / kGroupsPerBlock;
    for (uint b = 0; b < full_blocks; ++b) {
        const uint byte_base = b * 128u + lane * 4u;
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g = b * kGroupsPerBlock + lane / kLanesPerGroup;
        const BpwScale sb = bpw_scale(S16, B16, S8, anchor, g);
        const uint elem = byte_base * 2u;
        const uint b0 = w4 & 0xFFu, b1 = (w4 >> 8) & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu, b3 = (w4 >> 24) & 0xFFu;
        for (uint r = 0; r < cap; ++r) {
            if (r >= rows) break;
            const half4 xa = *((device const half4*)(x_row[r] + elem));
            const half4 xb = *((device const half4*)(x_row[r] + elem + 4u));
            const float e0 = float(xa.x), e1 = float(xa.y);
            const float e2 = float(xa.z), e3 = float(xa.w);
            const float e4 = float(xb.x), e5 = float(xb.y);
            const float e6 = float(xb.z), e7 = float(xb.w);
            float dot = 0.0f;
            dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
            dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
            dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
            dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
            const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
            acc[r] = bpw_accumulate(acc[r], sb, dot, sum);
        }
    }
    for (uint g = full_blocks * kGroupsPerBlock; g < n_groups; ++g) {
        if (lane >= kTailLanes) break;
        const BpwScale sb = bpw_scale(S16, B16, S8, anchor, g);
        const uint8_t byte = W_row[g * (kGroupSize / 2u) + lane];
        for (uint r = 0; r < cap; ++r) {
            if (r >= rows) break;
            const float x0 = float(x_row[r][g * kGroupSize + lane * 2u]);
            const float x1 = float(x_row[r][g * kGroupSize + lane * 2u + 1u]);
            float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
            dot = fma(float(uint(byte >> 4)), x1, dot);
            acc[r] = bpw_accumulate(acc[r], sb, dot, x0 + x1);
        }
    }

    for (uint r = 0; r < cap; ++r) {
        if (r >= rows) break;
        const float total = simd_sum(acc[r]);
        if (lane == 0) { route_partials[dst[r] + d] = half(total); }
    }
}
"""#

// MARK: - Host side

/// The three weight formats, in the order the report prints them.
enum BpwFormat: UInt32, CaseIterable {
    case affine = 0   // BF16 scale + BF16 bias  -- production
    case sym = 1      // BF16 scale, w = s*(q-8)
    case sym8 = 2     // 8-bit scale code (2 exp + 6 mantissa) + per-row anchor
    case sym9 = 3     // 9-bit scale code (2 exp + 7 mantissa) + per-row anchor

    var name: String { ["affine", "sym", "sym8", "sym9"][Int(rawValue)] }
    /// Bytes the row's anchor costs, per row.
    var anchorBytesPerRow: Int { self == .affine || self == .sym ? 0 : 1 }

    /// Bytes one row's scale metadata occupies. sym9 packs nine bits per group
    /// and pads the row to a 4-byte boundary; the others are whole bytes.
    func scaleBytesPerRow(groups: Int) -> Int {
        switch self {
        case .affine: return groups * 4
        case .sym: return groups * 2
        case .sym8: return groups
        // One spare byte: the last group's nine bits span two.
        case .sym9: return ((((groups * 9 + 7) / 8) + 1) + 3) & ~3
        }
    }

    func bytesPerRow(groupSize: Int, reductionLength: Int) -> Int {
        reductionLength / 2 + scaleBytesPerRow(groups: reductionLength / groupSize)
            + anchorBytesPerRow
    }

    func bitsPerWeight(groupSize: Int, reductionLength: Int) -> Double {
        Double(bytesPerRow(groupSize: groupSize, reductionLength: reductionLength))
            * 8 / Double(reductionLength)
    }
}

/// Byte offsets inside one expert blob: gate, then up, then down, each as
/// (weights, scales, anchors-or-biases). `affine` puts BF16 biases where the
/// other two put nothing (`sym`) or a byte of exponent per row (`sym8`), which
/// is why the third offset is named for its role and not its content.
struct BpwLayout {
    var gateW = 0, gateS = 0, gateA = 0
    var upW = 0, upS = 0, upA = 0
    var downW = 0, downS = 0, downA = 0
    var stride = 0

    init(d: Int, f: Int, groupSize: Int, format: BpwFormat) {
        func advance(_ cursor: inout Int, rows: Int, cols: Int) -> (Int, Int, Int) {
            func align4(_ value: inout Int) { value = (value + 3) & ~3 }
            align4(&cursor)
            let w = cursor
            cursor += rows * cols / 2
            align4(&cursor)
            let s = cursor
            let groups = cols / groupSize
            // The affine bias lives in its own region of the same size, so the
            // scale region is half of what `scaleBytesPerRow` accounts for.
            cursor += rows * (format == .affine ? groups * 2
                                                : format.scaleBytesPerRow(groups: groups))
            align4(&cursor)
            switch format {
            case .affine:
                let a = cursor
                cursor += rows * groups * 2
                return (w, s, a)
            case .sym:
                return (w, s, s)   // no third region; the pointer is never read
            case .sym8, .sym9:
                let a = cursor
                cursor += rows
                return (w, s, a)
            }
        }
        var cursor = 0
        (gateW, gateS, gateA) = advance(&cursor, rows: f, cols: d)
        (upW, upS, upA) = advance(&cursor, rows: f, cols: d)
        (downW, downS, downA) = advance(&cursor, rows: d, cols: f)
        cursor = (cursor + 15) & ~15
        stride = cursor
    }
}

/// `BpwParams` in the probe's MSL.
struct BpwParamsHost {
    var d: UInt32 = 0, f: UInt32 = 0, topK: UInt32 = 0, hiddenStride: UInt32 = 0
    var gateW: UInt32 = 0, gateS: UInt32 = 0, gateA: UInt32 = 0
    var upW: UInt32 = 0, upS: UInt32 = 0, upA: UInt32 = 0
    var downW: UInt32 = 0, downS: UInt32 = 0, downA: UInt32 = 0
}

struct BpwPairHost { var token: UInt32; var expert: UInt32; var rank: UInt32; var reserved: UInt32 }
struct BpwBlockHost { var localSlot: UInt32; var pairStart: UInt32; var rowCount: UInt32; var localRow: UInt32 }

enum BpwStage: String, CaseIterable {
    case gateUp = "gate/up"
    case down = "down"
    var function: String { self == .gateUp ? "bpw_rows_gate_up" : "bpw_rows_down" }
}

// MARK: - The 8-bit scale code
//
// `anchor` is the biased fp32 exponent of the row's largest scale. A code is
// two bits of exponent step below that anchor and six bits of mantissa, which
// reassemble into an fp32 bit pattern with no arithmetic:
//
//     s = as_type<float>(((anchor - (code >> 6)) << 23) | ((code & 63) << 17))
//
// Two exponent bits cover a 8x span; 44 §2 measures the per-row span of the
// real checkpoint at <= 2x for 97.9% of rows and <= 8x for all of them, so no
// row underflows. Six mantissa bits is one less than BF16 carries, which caps
// the relative error at 2^-7 = 0.78%.

private func bpwAnchor(_ maxScale: Float) -> UInt8 {
    UInt8((maxScale.bitPattern >> 23) & 0xFF)
}

private func bpwEncodeScale(_ s: Float, anchor: UInt8) -> UInt8 {
    let bits = s.bitPattern
    var e = Int(anchor) - Int((bits >> 23) & 0xFF)
    if e < 0 { e = 0 }
    if e > 3 { return UInt8(3 << 6) }          // underflows the anchor: smallest code
    var m = Int((bits >> 17) & 0x3F) + Int((bits >> 16) & 1)   // round to nearest
    if m == 64 {
        m = 0
        e -= 1
        if e < 0 { e = 0; m = 63 }             // only the row max can land here
    }
    return UInt8((e << 6) | m)
}

private func bpwDecodeScale(_ code: UInt8, anchor: UInt8) -> Float {
    let e = UInt32(code >> 6)
    let m = UInt32(code & 63)
    return Float(bitPattern: ((UInt32(anchor) - e) << 23) | (m << 17))
}

/// sym9: two bits of exponent step below the row anchor and BF16's seven
/// mantissa bits verbatim, so the decode reproduces the stored BF16 scale
/// **exactly** whenever the row's exponent span is at most 3. Measured over
/// every routed expert of `scratch/gemma4-qat.gturbo` -- 16,220,160 rows -- one
/// row has a span of 4 and every other row fits (44 §3). `clamped` counts the
/// groups that do not, which the packer would have to report.
private func bpwEncodeScale9(_ s: Float, anchor: UInt8, clamped: inout Int) -> UInt16 {
    let bits = s.bitPattern
    var e = Int(anchor) - Int((bits >> 23) & 0xFF)
    if e < 0 { e = 0 }
    if e > 3 { e = 3; clamped += 1; return UInt16(3 << 7) }
    return UInt16((e << 7) | Int((bits >> 16) & 0x7F))
}

private func bpwDecodeScale9(_ code: UInt16, anchor: UInt8) -> Float {
    let e = UInt32(code >> 7)
    let m = UInt32(code & 0x7F)
    return Float(bitPattern: ((UInt32(anchor) - e) << 23) | (m << 16))
}

/// BF16 round-to-nearest-even, as the packer would write it.
private func bpwBF16(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let rounding = UInt32(0x7FFF) &+ ((bits >> 16) & 1)
    return UInt16(truncatingIfNeeded: (bits &+ rounding) >> 16)
}

// MARK: - Fixture

let bpwTileExperts = 16
let bpwMaxRows = 8
let bpwTopK = 8

/// One expert's weights, generated once and written into every format.
///
/// The three blobs describe *the same* quantized weights: identical nibbles,
/// identical scales, and `affine`'s bias is exactly `-8 * scale` -- the identity
/// 44 §1 verified holds bit-for-bit on the shipped checkpoint. So `affine` and
/// `sym` must agree to floating-point noise, and whatever `sym8` differs by is
/// the 6-bit mantissa and nothing else.
struct BpwWeights {
    let d: Int
    let f: Int
    let groupSize: Int
    /// Packed nibbles per matrix, in gate/up/down order.
    var packed: [[UInt8]] = []
    /// Per-row-major scales per matrix.
    var scales: [[Float]] = []

    init(d: Int, f: Int, groupSize: Int) {
        self.d = d
        self.f = f
        self.groupSize = groupSize
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state >> 33
        }
        for (rows, cols) in [(f, d), (f, d), (d, f)] {
            var bytes = [UInt8](repeating: 0, count: rows * cols / 2)
            for index in bytes.indices { bytes[index] = UInt8(truncatingIfNeeded: next()) }
            packed.append(bytes)

            // A row's scales sit inside a ~2x band around a row base, which is
            // the spread 44 §2 measures on the checkpoint (median 1.72x, p99
            // 2.55x). Drawing them this way is what makes the sym8 error in §3
            // comparable to what the real weights would see.
            let groups = cols / groupSize
            var rowScales = [Float](repeating: 0, count: rows * groups)
            for row in 0..<rows {
                let base = 0.004 + 0.008 * Float(next() % 1000) / 1000.0
                for g in 0..<groups {
                    let spread = exp2(-1.35 * Float(next() % 1000) / 1000.0)
                    // Round-trip through BF16: the packer stores BF16 scales, so
                    // that is the value all three formats start from.
                    let raw = base * spread
                    rowScales[row * groups + g] =
                        Float(bitPattern: UInt32(bpwBF16(raw)) << 16)
                }
            }
            scales.append(rowScales)
        }
    }

    /// Fill one blob of `layout.stride` bytes in `format`.
    func write(into pointer: UnsafeMutableRawPointer, layout: BpwLayout,
               format: BpwFormat) {
        let regions = [(layout.gateW, layout.gateS, layout.gateA, f, d),
                       (layout.upW, layout.upS, layout.upA, f, d),
                       (layout.downW, layout.downS, layout.downA, d, f)]
        for (index, region) in regions.enumerated() {
            let (wOff, sOff, aOff, rows, cols) = region
            let groups = cols / groupSize
            packed[index].withUnsafeBytes { src in
                pointer.advanced(by: wOff).copyMemory(from: src.baseAddress!,
                                                      byteCount: src.count)
            }
            let rowScales = scales[index]
            switch format {
            case .affine:
                let s = pointer.advanced(by: sOff).bindMemory(to: UInt16.self, capacity: rows * groups)
                let b = pointer.advanced(by: aOff).bindMemory(to: UInt16.self, capacity: rows * groups)
                for i in 0..<(rows * groups) {
                    s[i] = bpwBF16(rowScales[i])
                    b[i] = bpwBF16(-8 * rowScales[i])
                }
            case .sym:
                let s = pointer.advanced(by: sOff).bindMemory(to: UInt16.self, capacity: rows * groups)
                for i in 0..<(rows * groups) { s[i] = bpwBF16(rowScales[i]) }
            case .sym8:
                let s = pointer.advanced(by: sOff).bindMemory(to: UInt8.self, capacity: rows * groups)
                let a = pointer.advanced(by: aOff).bindMemory(to: UInt8.self, capacity: rows)
                for row in 0..<rows {
                    let slice = rowScales[(row * groups)..<((row + 1) * groups)]
                    let anchor = bpwAnchor(slice.max() ?? 1)
                    a[row] = anchor
                    for g in 0..<groups {
                        s[row * groups + g] =
                            bpwEncodeScale(rowScales[row * groups + g], anchor: anchor)
                    }
                }
            case .sym9:
                let stride = format.scaleBytesPerRow(groups: groups)
                let s = pointer.advanced(by: sOff).bindMemory(to: UInt8.self,
                                                              capacity: rows * stride)
                let a = pointer.advanced(by: aOff).bindMemory(to: UInt8.self, capacity: rows)
                var clamped = 0
                for row in 0..<rows {
                    let slice = rowScales[(row * groups)..<((row + 1) * groups)]
                    let anchor = bpwAnchor(slice.max() ?? 1)
                    a[row] = anchor
                    for byte in 0..<stride { s[row * stride + byte] = 0 }
                    for g in 0..<groups {
                        let code = bpwEncodeScale9(rowScales[row * groups + g],
                                                   anchor: anchor, clamped: &clamped)
                        let bit = g * 9
                        let byte = row * stride + (bit >> 3)
                        let shift = bit & 7
                        let packed = UInt32(code) << UInt32(shift)
                        s[byte] |= UInt8(truncatingIfNeeded: packed)
                        s[byte + 1] |= UInt8(truncatingIfNeeded: packed >> 8)
                    }
                }
            }
        }
    }

    /// Worst and RMS relative error a coded format introduces on the scales
    /// themselves, before any kernel runs. `exact` counts the groups the code
    /// reproduces bit-for-bit.
    func scaleError(_ format: BpwFormat) -> (maxRel: Double, rmsRel: Double, exact: Double) {
        var worst = 0.0
        var sumSq = 0.0
        var count = 0
        var exact = 0
        var clamped = 0
        for (index, rowScales) in scales.enumerated() {
            let cols = index == 2 ? f : d
            let rows = index == 2 ? d : f
            let groups = cols / groupSize
            for row in 0..<rows {
                let slice = rowScales[(row * groups)..<((row + 1) * groups)]
                let anchor = bpwAnchor(slice.max() ?? 1)
                for g in 0..<groups {
                    let want = rowScales[row * groups + g]
                    let got: Float
                    if format == .sym9 {
                        got = bpwDecodeScale9(
                            bpwEncodeScale9(want, anchor: anchor, clamped: &clamped),
                            anchor: anchor)
                    } else {
                        got = bpwDecodeScale(bpwEncodeScale(want, anchor: anchor),
                                             anchor: anchor)
                    }
                    if got.bitPattern == want.bitPattern { exact += 1 }
                    let rel = Double(abs(got - want) / want)
                    worst = max(worst, rel)
                    sumSq += rel * rel
                    count += 1
                }
            }
        }
        return (worst, (sumSq / Double(count)).squareRoot(),
                Double(exact) / Double(count))
    }
}

struct BpwFixture {
    let format: BpwFormat
    let layout: BpwLayout
    let blobs: [MTLBuffer]
    let argumentBuffer: MTLBuffer
    let params: BpwParamsHost
}

final class BpwProbe {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    let groupSize: Int
    var d: Int
    var f: Int

    let hidden: MTLBuffer
    let act: MTLBuffer
    let partials: MTLBuffer
    let pairs: MTLBuffer

    init(context: MetalContext, groupSize: Int, d: Int, f: Int) throws {
        self.device = context.device
        self.queue = context.queue
        self.groupSize = groupSize
        self.d = d
        self.f = f

        let options = MTLCompileOptions()
        options.languageVersion = .version3_2
        options.preprocessorMacros = ["TURBO_AFFINE_GROUP_SIZE": NSNumber(value: groupSize)]
        self.library = try context.device.makeLibrary(source: bpwProbeSource,
                                                      options: options)

        let allocator = context.device
        func halves(_ count: Int) -> MTLBuffer {
            guard let buffer = allocator.makeBuffer(length: count * 2,
                                                    options: .storageModeShared)
            else { fatalError("fp16 allocation failed (\(count))") }
            return buffer
        }
        self.hidden = halves(bpwMaxRows * d)
        self.act = halves(bpwTileExperts * bpwMaxRows * f)
        self.partials = halves(bpwMaxRows * bpwTopK * d)

        var pairList: [BpwPairHost] = []
        for expert in 0..<bpwTileExperts {
            for row in 0..<bpwMaxRows {
                pairList.append(BpwPairHost(token: UInt32(row), expert: UInt32(expert),
                                            rank: UInt32(row), reserved: 0))
            }
        }
        guard let pairBuffer = allocator.makeBuffer(
            bytes: pairList,
            length: pairList.count * MemoryLayout<BpwPairHost>.stride,
            options: .storageModeShared) else { fatalError("pair allocation failed") }
        self.pairs = pairBuffer
    }

    func pipeline(_ stage: BpwStage, cap: Int, format: BpwFormat) throws
        -> MTLComputePipelineState {
        let values = MTLFunctionConstantValues()
        var capValue = UInt32(cap)
        var fmtValue = format.rawValue
        values.setConstantValue(&capValue, type: .uint, index: 16)
        values.setConstantValue(&fmtValue, type: .uint, index: 17)
        let function = try library.makeFunction(name: stage.function, constantValues: values)
        return try device.makeComputePipelineState(function: function)
    }

    func makeFixture(_ format: BpwFormat, weights: BpwWeights) -> BpwFixture {
        let layout = BpwLayout(d: d, f: f, groupSize: groupSize, format: format)
        guard let master = device.makeBuffer(length: layout.stride, options: .storageModeShared)
        else { fatalError("blob allocation failed (\(layout.stride) bytes)") }
        weights.write(into: master.contents(), layout: layout, format: format)
        // Every expert in the tile carries the same bytes at a different
        // address. The DRAM traffic is identical to sixteen distinct experts --
        // 59 MB per dispatch at the production shape, far past any cache -- and
        // generating one is what keeps the fixture rebuild off the timings.
        var blobs: [MTLBuffer] = [master]
        for _ in 1..<bpwTileExperts {
            guard let copy = device.makeBuffer(length: layout.stride, options: .storageModeShared)
            else { fatalError("blob allocation failed") }
            copy.contents().copyMemory(from: master.contents(), byteCount: layout.stride)
            blobs.append(copy)
        }

        guard let function = try? library.makeFunction(name: "bpw_rows_gate_up",
                                                       constantValues: {
            let values = MTLFunctionConstantValues()
            var capValue = UInt32(1)
            var fmtValue = format.rawValue
            values.setConstantValue(&capValue, type: .uint, index: 16)
            values.setConstantValue(&fmtValue, type: .uint, index: 17)
            return values
        }()) else { fatalError("bpw_rows_gate_up missing") }
        let encoder = function.makeArgumentEncoder(bufferIndex: 9)
        guard let argumentBuffer = device.makeBuffer(length: encoder.encodedLength,
                                                     options: .storageModeShared)
        else { fatalError("argument buffer allocation failed") }
        encoder.setArgumentBuffer(argumentBuffer, offset: 0)
        for (index, blob) in blobs.enumerated() { encoder.setBuffer(blob, offset: 0, index: index) }

        var params = BpwParamsHost()
        params.d = UInt32(d); params.f = UInt32(f)
        params.topK = UInt32(bpwTopK); params.hiddenStride = UInt32(d)
        params.gateW = UInt32(layout.gateW); params.gateS = UInt32(layout.gateS)
        params.gateA = UInt32(layout.gateA)
        params.upW = UInt32(layout.upW); params.upS = UInt32(layout.upS)
        params.upA = UInt32(layout.upA)
        params.downW = UInt32(layout.downW); params.downS = UInt32(layout.downS)
        params.downA = UInt32(layout.downA)

        return BpwFixture(format: format, layout: layout, blobs: blobs,
                          argumentBuffer: argumentBuffer, params: params)
    }

    /// The same fixture, but with the expert blobs supplied from outside
    /// instead of allocated and filled here. `slots` names, per tile slot, the
    /// buffer to bind and the byte offset inside it -- so one `mmap`-backed
    /// buffer per expert (offset 0) and a single buffer spanning a whole layer
    /// file (offset = rank * stride) both fit, which is exactly the pair
    /// `docs/mtp/47-D-MMAP-RESIDENCY-PROPOSAL.md` §6 P-1 contrasts.
    ///
    /// `blobs` is the *distinct* resource list: `encode` only calls
    /// `useResource` on what is passed here, so the wired-page delta counts the
    /// experts a command buffer actually names and nothing else.
    func makeFixture(format: BpwFormat, slots: [(buffer: MTLBuffer, offset: Int)],
                     blobs: [MTLBuffer]) -> BpwFixture {
        precondition(slots.count == bpwTileExperts, "need \(bpwTileExperts) tile slots")
        let layout = BpwLayout(d: d, f: f, groupSize: groupSize, format: format)

        guard let function = try? library.makeFunction(name: "bpw_rows_gate_up",
                                                       constantValues: {
            let values = MTLFunctionConstantValues()
            var capValue = UInt32(1)
            var fmtValue = format.rawValue
            values.setConstantValue(&capValue, type: .uint, index: 16)
            values.setConstantValue(&fmtValue, type: .uint, index: 17)
            return values
        }()) else { fatalError("bpw_rows_gate_up missing") }
        let encoder = function.makeArgumentEncoder(bufferIndex: 9)
        guard let argumentBuffer = device.makeBuffer(length: encoder.encodedLength,
                                                     options: .storageModeShared)
        else { fatalError("argument buffer allocation failed") }
        encoder.setArgumentBuffer(argumentBuffer, offset: 0)
        for (index, slot) in slots.enumerated() {
            encoder.setBuffer(slot.buffer, offset: slot.offset, index: index)
        }

        var params = BpwParamsHost()
        params.d = UInt32(d); params.f = UInt32(f)
        params.topK = UInt32(bpwTopK); params.hiddenStride = UInt32(d)
        params.gateW = UInt32(layout.gateW); params.gateS = UInt32(layout.gateS)
        params.gateA = UInt32(layout.gateA)
        params.upW = UInt32(layout.upW); params.upS = UInt32(layout.upS)
        params.upA = UInt32(layout.upA)
        params.downW = UInt32(layout.downW); params.downS = UInt32(layout.downS)
        params.downA = UInt32(layout.downA)

        return BpwFixture(format: format, layout: layout, blobs: blobs,
                          argumentBuffer: argumentBuffer, params: params)
    }

    func encode(_ stage: BpwStage, pso: MTLComputePipelineState,
                commandBuffer: MTLCommandBuffer, fixture: BpwFixture,
                rowCounts: [Int]) {
        var blocks: [BpwBlockHost] = []
        var localRow = 0
        for expert in rowCounts.indices {
            let count = rowCounts[expert]
            if count > 0 {
                blocks.append(BpwBlockHost(localSlot: UInt32(expert),
                                           pairStart: UInt32(expert * bpwMaxRows),
                                           rowCount: UInt32(count),
                                           localRow: UInt32(localRow)))
            }
            localRow += count
        }
        guard !blocks.isEmpty, let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        var params = fixture.params
        enc.setComputePipelineState(pso)
        if stage == .gateUp { enc.setBuffer(hidden, offset: 0, index: 0) }
        enc.setBuffer(pairs, offset: 0, index: 1)
        enc.setBytes(&blocks, length: blocks.count * MemoryLayout<BpwBlockHost>.stride, index: 2)
        if stage == .down { enc.setBuffer(partials, offset: 0, index: 5) }
        enc.setBuffer(act, offset: 0, index: 7)
        enc.setBuffer(fixture.argumentBuffer, offset: 0, index: 9)
        enc.setBytes(&params, length: MemoryLayout<BpwParamsHost>.stride, index: 10)
        for blob in fixture.blobs { enc.useResource(blob, usage: .read) }
        let outputRows = stage == .gateUp ? f : d
        enc.dispatchThreadgroups(
            MTLSize(width: (outputRows + 7) / 8, height: blocks.count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * 8, height: 1, depth: 1))
        enc.endEncoding()
    }

    func seconds(_ stage: BpwStage, pso: MTLComputePipelineState, fixture: BpwFixture,
                 rowCounts: [Int], iterations: Int, label: String) -> Double {
        guard let warmup = queue.makeCommandBuffer() else { fatalError("command buffer failed") }
        encode(stage, pso: pso, commandBuffer: warmup, fixture: fixture, rowCounts: rowCounts)
        waitAndCheck(warmup, "\(label) warmup")
        guard let cmd = queue.makeCommandBuffer() else { fatalError("command buffer failed") }
        for _ in 0..<iterations {
            encode(stage, pso: pso, commandBuffer: cmd, fixture: fixture, rowCounts: rowCounts)
        }
        waitAndCheck(cmd, label)
        return (cmd.gpuEndTime - cmd.gpuStartTime) / Double(iterations)
    }

    func fillHidden() {
        let halves = hidden.contents().bindMemory(to: Float16.self, capacity: bpwMaxRows * d)
        for index in 0..<(bpwMaxRows * d) {
            halves[index] = Float16(Float((index * 37) % 211) * 0.0009 - 0.095)
        }
    }

    func fillAct() {
        let count = bpwTileExperts * bpwMaxRows * f
        let halves = act.contents().bindMemory(to: Float16.self, capacity: count)
        for index in 0..<count {
            halves[index] = Float16(Float((index * 53) % 197) * 0.0011 - 0.108)
        }
    }

    func readAct(rows: Int) -> [Float] {
        let halves = act.contents().bindMemory(to: Float16.self, capacity: bpwTileExperts * bpwMaxRows * f)
        return (0..<(rows * f)).map { Float(halves[$0]) }
    }

    func readPartials(rows: Int) -> [Float] {
        let halves = partials.contents().bindMemory(to: Float16.self, capacity: bpwMaxRows * bpwTopK * d)
        var out: [Float] = []
        for row in 0..<rows {
            let base = (row * bpwTopK + row) * d
            out.append(contentsOf: (0..<d).map { Float(halves[base + $0]) })
        }
        return out
    }
}

/// Both figures are scaled by the RMS of the reference, not element-wise: the
/// outputs of these kernels cross zero (gelu on gate/up, a signed dot product on
/// down), and a pointwise relative error on a near-zero element says nothing
/// about the format. `rms` is the error energy against the signal energy;
/// `max` is the single worst deviation in units of that signal.
private func bpwRelative(_ a: [Float], _ b: [Float]) -> (maxRel: Double, rmsRel: Double) {
    var worstAbs = 0.0
    var sumSq = 0.0
    var sumRef = 0.0
    for index in a.indices {
        let ref = Double(b[index])
        let diff = Double(a[index]) - ref
        sumSq += diff * diff
        sumRef += ref * ref
        worstAbs = max(worstAbs, abs(diff))
    }
    let rms = (sumRef / Double(a.count)).squareRoot()
    return (worstAbs / max(rms, 1e-30), (sumSq / max(sumRef, 1e-30)).squareRoot())
}

// MARK: - `--bpw-probe`

/// The row mixture a production verify block hands these kernels
/// (`docs/mtp/32-M8-A-ROWS-SPLIT.md` §5: 1:4036 2:1257 3:818 4:449).
private func bpwRowMixture(width: Int) -> [Int] {
    let share = [1: 0.615, 2: 0.192, 3: 0.125, 4: 0.068]
    var rows: [Int] = []
    for r in [4, 3, 2] {
        rows.append(contentsOf: [Int](repeating: r,
                                      count: Int((Double(width) * share[r]!).rounded())))
    }
    rows.append(contentsOf: [Int](repeating: 1, count: max(0, width - rows.count)))
    return rows.sorted()
}

func runBpwProbe(groupSize: Int, iterations: Int) throws {
    let context = try makeContext(groupSize: groupSize)
    let d = 2816, f = 704
    let probe = try BpwProbe(context: context, groupSize: groupSize, d: d, f: f)
    let weights = BpwWeights(d: d, f: f, groupSize: groupSize)
    let fixtures = Dictionary(uniqueKeysWithValues:
        BpwFormat.allCases.map { ($0, probe.makeFixture($0, weights: weights)) })

    print("=== routed MoE rows kernels, weight format swept "
            + "(D=\(d) F=\(f) group \(groupSize), \(bpwTileExperts) experts/tile, "
            + "\(iterations) iterations) ===")
    print("Same arithmetic in all three: two FMAs per group per row, the same")
    print("`sum` of eight activations, the same activation loads. Only the")
    print("weight-side bytes move. GB/s is weight traffic against the 135 GB/s")
    print("floor 20-M4.8 §3 measured (461 MB in 3.41 ms).")
    print("")
    print("  format   scale/group   B/row (gate/up, down)      bpw   expert blob")
    for format in BpwFormat.allCases {
        let layout = BpwLayout(d: d, f: f, groupSize: groupSize, format: format)
        print(String(format: "  %-8@ %8.2f b   %6d  %6d          %5.3f   %9d B",
                     format.name as NSString,
                     Double(format.scaleBytesPerRow(groups: d / groupSize) * 8)
                        / Double(d / groupSize),
                     format.bytesPerRow(groupSize: groupSize, reductionLength: d),
                     format.bytesPerRow(groupSize: groupSize, reductionLength: f),
                     format.bitsPerWeight(groupSize: groupSize, reductionLength: d),
                     layout.stride))
    }

    // ---- §1/§2: the sweep -------------------------------------------------
    print("")
    print("us per dispatch of one 16-expert tile, uniform r rows on every expert.")
    print("`vs affine` is the change against production's format at the same r.")
    var baseline: [String: Double] = [:]
    for stage in BpwStage.allCases {
        let outputRows = stage == .gateUp ? f : d
        let reduction = stage == .gateUp ? d : f
        let matrices = stage == .gateUp ? 2 : 1
        print("")
        print("  \(stage.rawValue)  (reduces over \(stage == .gateUp ? "D" : "F") "
                + "= \(reduction), \(reduction / groupSize) groups, output "
                + "\(outputRows) rows)")
        print("    format       r=1      r=2      r=4   |  GB/s r=2  |  vs affine r=1/2/4")
        for format in BpwFormat.allCases {
            guard let fixture = fixtures[format] else { continue }
            var line = String(format: "    %-8@", format.name as NSString)
            var deltas: [String] = []
            var atTwo = 0.0
            for r in [1, 2, 4] {
                let pso = try probe.pipeline(stage, cap: r, format: format)
                let seconds = probe.seconds(stage, pso: pso, fixture: fixture,
                                            rowCounts: [Int](repeating: r, count: bpwTileExperts),
                                            iterations: iterations,
                                            label: "\(stage.rawValue) \(format.name) r=\(r)")
                line += String(format: " %8.1f", seconds * 1e6)
                let key = "\(stage.rawValue)/\(r)"
                if format == .affine {
                    baseline[key] = seconds
                    deltas.append("   .")
                } else if let base = baseline[key] {
                    deltas.append(String(format: "%+5.1f%%", (seconds / base - 1) * 100))
                }
                if r == 2 { atTwo = seconds }
            }
            let bytes = Double(bpwTileExperts * outputRows * matrices
                                * format.bytesPerRow(groupSize: groupSize,
                                                     reductionLength: reduction))
            line += String(format: "   |   %7.1f  |  %@", bytes / atTwo / 1e9,
                           deltas.joined(separator: " ") as NSString)
            print(line)
        }
    }

    // ---- §3: does the format change the answer? ---------------------------
    //
    // One block, so the sixteen experts do not race for the same partials row.
    print("")
    print("  numerical agreement (one expert, r=2; `act` for gate/up, "
            + "`route_partials` for down)")
    for format in [BpwFormat.sym8, .sym9] {
        let error = weights.scaleError(format)
        print(String(format: "    %@ scale codes alone:   max %.4f%%   rms %.4f%%   "
                        + "bit-exact %.2f%% of groups",
                     format.name as NSString, error.maxRel * 100, error.rmsRel * 100,
                     error.exact * 100))
    }
    let single = [2] + [Int](repeating: 0, count: bpwTileExperts - 1)
    var gateUpOut: [BpwFormat: [Float]] = [:]
    var downOut: [BpwFormat: [Float]] = [:]
    for format in BpwFormat.allCases {
        guard let fixture = fixtures[format] else { continue }
        probe.fillHidden()
        guard let cmd = context.queue.makeCommandBuffer() else { fatalError("cmd") }
        probe.encode(.gateUp, pso: try probe.pipeline(.gateUp, cap: 2, format: format),
                     commandBuffer: cmd, fixture: fixture, rowCounts: single)
        waitAndCheck(cmd, "bpw check gate/up \(format.name)")
        gateUpOut[format] = probe.readAct(rows: 2)

        probe.fillAct()
        guard let cmd2 = context.queue.makeCommandBuffer() else { fatalError("cmd") }
        probe.encode(.down, pso: try probe.pipeline(.down, cap: 2, format: format),
                     commandBuffer: cmd2, fixture: fixture, rowCounts: single)
        waitAndCheck(cmd2, "bpw check down \(format.name)")
        downOut[format] = probe.readPartials(rows: 2)
    }
    for (label, lhs, rhs) in [("sym   vs affine", BpwFormat.sym, BpwFormat.affine),
                              ("sym8  vs sym", BpwFormat.sym8, BpwFormat.sym),
                              ("sym9  vs sym", BpwFormat.sym9, BpwFormat.sym)] {
        let gate = bpwRelative(gateUpOut[lhs]!, gateUpOut[rhs]!)
        let down = bpwRelative(downOut[lhs]!, downOut[rhs]!)
        print(String(format: "    %@   gate/up  max %.4f%%  rms %.4f%%   |   "
                        + "down  max %.4f%%  rms %.4f%%",
                     label as NSString, gate.maxRel * 100, gate.rmsRel * 100,
                     down.maxRel * 100, down.rmsRel * 100))
    }

    // ---- §4: production's tile ---------------------------------------------
    print("")
    print("  production row mixture (32 §5: 5.7 experts/tile, rows "
            + "1:4036 2:1257 3:818 4:449; one dispatch, cap = widest block)")
    print("    width  mixture         format    gate/up      down      both   vs affine")
    for width in [6, 16] {
        let mix = bpwRowMixture(width: width)
        let cap = mix.max() ?? 1
        var both: [BpwFormat: Double] = [:]
        for format in BpwFormat.allCases {
            guard let fixture = fixtures[format] else { continue }
            var perStage: [BpwStage: Double] = [:]
            for stage in BpwStage.allCases {
                let pso = try probe.pipeline(stage, cap: cap, format: format)
                for pass in 0..<2 {
                    let seconds = probe.seconds(stage, pso: pso, fixture: fixture,
                                                rowCounts: mix, iterations: iterations,
                                                label: "mix w=\(width) \(format.name) \(stage.rawValue)")
                    if pass == 1 { perStage[stage] = seconds * 1e6 }
                }
            }
            let sum = (perStage[.gateUp] ?? 0) + (perStage[.down] ?? 0)
            both[format] = sum
            let delta = format == .affine ? "      ." : String(
                format: "%+7.1f%%", (sum / (both[.affine] ?? sum) - 1) * 100)
            print(String(format: "    %5d  %-14@  %-8@ %9.1f %9.1f %9.1f  %@",
                         width, mix.map(String.init).joined() as NSString,
                         format.name as NSString, perStage[.gateUp] ?? 0,
                         perStage[.down] ?? 0, sum, delta as NSString))
        }
    }
}
