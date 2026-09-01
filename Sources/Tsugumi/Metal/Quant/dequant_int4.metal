#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_int4 — MLX `affine` 4-bit dequant.
//
// Layout (per row of length N):
//   W       : N/2 bytes. Low nibble of byte k = component 2k (unsigned 0..15),
//             high nibble = component 2k+1.
//   scales  : N/64 BF16, one per group of 64.
//   biases  : N/64 BF16, one per group of 64.
//   value   : w[i] = float(nibble[i]) * scale[i/64] + bias[i/64].
//
// Affine factoring for GEMV (sum over a group of 64):
//   sum_k (q_k * s + b) * x_k = s * sum_k(q_k * x_k) + b * sum_k x_k
// so scale and bias each cost one mul + one FMA per group instead of per
// element; the per-element inner loop keeps the scalar path's FMA count.
// ============================================================================

#ifndef MOEPACK_AFFINE_GROUP_SIZE
#define MOEPACK_AFFINE_GROUP_SIZE 64
#endif
constant constexpr uint kGroupSize = MOEPACK_AFFINE_GROUP_SIZE;

// ---------------------------------------------------------------------------
// Affine zero point
//
// `MOEPACK_AFFINE_SYMMETRIC` is the second whole-model compile-time constant
// (`MetalContext.affineScheme`), alongside the group size. When it is set the
// packer has dropped the bias arrays because the checkpoint satisfies
// `bias == -8 * scale` as a BF16 bit pattern in every group -- exactly, not
// approximately (`docs/mtp/44-W1-WEIGHT-DIET.md` §1). The bias *bindings* stay:
// callers alias them onto the scale buffer, so no kernel signature changes and
// the pointer below is simply never dereferenced.
// ---------------------------------------------------------------------------
#ifndef MOEPACK_AFFINE_SYMMETRIC
#define MOEPACK_AFFINE_SYMMETRIC 0
#endif

static inline float dq4_int4_bias(device const bfloat* biases, uint index,
                                     float scale) {
#if MOEPACK_AFFINE_SYMMETRIC
    return -8.0f * scale;
#else
    return float(biases[index]);
#endif
}


// Geometry of the vectorized INT4 GEMV block, derived from the group size.
//
// A SIMD group consumes a fixed 128 bytes per block (32 lanes x 4 bytes = 256
// nibbles). How many affine groups that spans depends on the group size, and
// so does how many lanes cover one group. At group 64 a group is 32 bytes, so
// a block is 4 groups and 8 lanes cover one group; at group 32 a group is 16
// bytes, so a block is 8 groups and 4 lanes cover one group. Both of these
// were previously hardcoded as `4` and `lane >> 3`, which is why a bare group
// size change reads the wrong scales and runs off the end of the row.
constant constexpr uint kGroupsPerBlock = 256u / kGroupSize;
constant constexpr uint kLanesPerGroup  = kGroupSize / 8u;
// Bytes in one group == the number of lanes with real work in the scalar tail.
constant constexpr uint kTailLanes      = kGroupSize / 2u;
constant uint FC_INT4_M [[function_constant(20)]];
constant uint FC_INT4_N [[function_constant(21)]];
constant bool FC_INT4_USE_FC [[function_constant(22)]];
constant uint FC_INT4_QKV_MQ [[function_constant(23)]];
constant uint FC_INT4_QKV_MKV [[function_constant(24)]];
constant uint FC_INT4_QKV_N [[function_constant(25)]];
constant bool FC_INT4_QKV_USE_FC [[function_constant(26)]];

static inline uint int4_fc_m(constant uint& M) {
    return (is_function_constant_defined(FC_INT4_USE_FC) &&
            FC_INT4_USE_FC &&
            is_function_constant_defined(FC_INT4_M)) ? FC_INT4_M : M;
}

static inline uint int4_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT4_USE_FC) &&
            FC_INT4_USE_FC &&
            is_function_constant_defined(FC_INT4_N)) ? FC_INT4_N : N;
}

static inline uint int4_qkv_fc_mq(constant uint& Mq) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_MQ)) ? FC_INT4_QKV_MQ : Mq;
}

static inline uint int4_qkv_fc_mkv(constant uint& Mkv) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_MKV)) ? FC_INT4_QKV_MKV : Mkv;
}

static inline uint int4_qkv_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_N)) ? FC_INT4_QKV_N : N;
}

inline uint nib_lo(uint8_t b) { return uint(b & 0x0F); }
inline uint nib_hi(uint8_t b) { return uint(b >> 4); }


kernel void embed_lookup_int4(
    device const uint8_t* table     [[buffer(0)]],   // [V, D/2] nibbles
    device const bfloat*  scales    [[buffer(1)]],   // [V, D/64] BF16
    device const bfloat*  biases    [[buffer(2)]],   // [V, D/64] BF16
    device half*          out       [[buffer(3)]],   // [D] FP16
    constant uint&        token_id  [[buffer(4)]],
    constant uint&        D         [[buffer(5)]],
    constant float&       out_scale [[buffer(6)]],   // pass 1.0 to disable
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    const uint groups_per_row = D / kGroupSize;
    device const uint8_t* row_q = table  + uint(token_id) * (D / 2u);
    device const bfloat*  row_s = scales + uint(token_id) * groups_per_row;
    device const bfloat*  row_b = biases + uint(token_id) * groups_per_row;
    uint8_t byte = row_q[gid >> 1];
    uint    q    = (gid & 1u) ? uint(byte >> 4) : uint(byte & 0xFu);
    float   s    = float(row_s[gid / kGroupSize]);
    float   b    = dq4_int4_bias(row_b, gid / kGroupSize, s);
    out[gid] = half((float(q) * s + b) * out_scale);
}

// y[m] = sum_{n} W[m, n] * x[n]. One-SIMD-per-row variant: 32 threads
// cooperate on a single output row, each handling 2 elements per group of 64
// (one byte → two nibbles). simd_sum reduces across the group; lane 0 writes.
//
// Requires N % 64 == 0 (per group of 64). Validated at the wrapper.
// Each threadgroup handles eight consecutive rows, one SIMD per row. The
// larger work unit gives the scheduler enough independent rows while sharing
// the L1-cached input-vector reads.
static inline void dequant_int4_gemv_simd_body(
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device const half*    x,
    device half*          y,
    uint                  M,
    uint                  N,
    uint                  rows_per_tg,
    uint                  tg_idx,
    uint                  sg_idx,
    uint                  lane
) {
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint n_groups  = N / kGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    // The vectorized row path reads weights as a uint (4 bytes = 8 nibbles) and
    // x as half4 in 128-byte blocks, with a scalar byte-per-lane remainder.
    // Within a block the 32 lanes split `kLanesPerGroup` per group, each
    // handling 8 contiguous elements of one group, so the affine factoring
    // s·Σqx + b·Σx is preserved (simd_sum aggregates; s/b are constant within a
    // group). Aligned: row stride N/2 and weightsOffset are multiples of 4; x
    // is half4-aligned (lane*8 elements). The remainder loop covers any group
    // count that is not a whole number of blocks.
    const uint full_blocks = n_groups / kGroupsPerBlock;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        // Read the 4-byte weight chunk as two ushorts. The resident weight
        // tensors are 2-byte aligned but NOT 4-byte aligned (BF16 scale/bias
        // regions leave a 2-aligned weightsOffset), so a `uint*` load would be
        // misaligned (undefined → garbage); a `ushort*` load is safe (row stride
        // N/2, weightsOffset, and byte_base are all even) and halves the loads
        // vs byte-by-byte.
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * kGroupsPerBlock + lane / kLanesPerGroup;
        const float s = float(s_row[g]);
        const float b = dq4_int4_bias(b_row, g, s);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * kGroupsPerBlock; g < n_groups; ++g) {
        // Only the first kTailLanes lanes hold a byte of this group.
        if (lane >= kTailLanes) break;
        const float s = float(s_row[g]);
        const float b = dq4_int4_bias(b_row, g, s);
        const uint8_t byte = W_row[g * (kGroupSize / 2) + lane];
        const float x0 = float(x[g * kGroupSize + lane * 2u]);
        const float x1 = float(x[g * kGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

kernel void dequant_int4_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint MM = int4_fc_m(M);
    const uint NN = int4_fc_n(N);
    dequant_int4_gemv_simd_body(W, scales, biases, x, y, MM, NN,
                                rows_per_tg, tg_idx, sg_idx, lane);
}

// y[t, m] = sum_n W[m, n] * x[t, n] for t < T, T <= kInt4MaxRows.
//
// The body above with the activation loop moved inside the weight block: one
// SIMD group still owns one weight row, but it spends that row on every one of
// the T activations before moving on, so a weight byte is read once per block
// instead of once per row. That is the whole of what a speculative verify block
// needs from the dense projections — the k rows differ only in the activation
// (docs/mtp/16-M4.5-PLAN.md §1) — and running them as T separate GEMVs reads
// the model's dense half T times over for no arithmetic reason.
//
// The reduction for a given (t, m) is the one-row kernel's, element for element
// and in the same order, so T=1 is bit-identical to `dequant_int4_gemv_simd`
// and a wider block keeps decode's numerics rather than a tiled kernel's.
constant constexpr uint kInt4MaxRows = 8;

static inline void dequant_int4_gemv_rows_simd_body(
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device const half*    x,
    device half*          y,
    uint                  M,
    uint                  N,
    uint                  T,
    uint                  x_stride,
    uint                  y_stride,
    uint                  rows_per_tg,
    uint                  tg_idx,
    uint                  sg_idx,
    uint                  lane
) {
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint n_groups  = N / kGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc[kInt4MaxRows];
    for (uint t = 0; t < kInt4MaxRows; ++t) { acc[t] = 0.0f; }

    const uint full_blocks = n_groups / kGroupsPerBlock;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * kGroupsPerBlock + lane / kLanesPerGroup;
        const float s = float(s_row[g]);
        const float b = dq4_int4_bias(b_row, g, s);
        const uint elem = byte_base * 2u;
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        for (uint t = 0; t < kInt4MaxRows; ++t) {
            if (t >= T) break;
            device const half* x_row = x + t * x_stride;
            const half4 xa = *((device const half4*)(x_row + elem));
            const half4 xb = *((device const half4*)(x_row + elem + 4u));
            const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
            const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
            float dot = 0.0f;
            dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
            dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
            dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
            dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
            const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
            acc[t] = fma(s, dot, acc[t]);
            acc[t] = fma(b, sum, acc[t]);
        }
    }
    for (uint g = full_blocks * kGroupsPerBlock; g < n_groups; ++g) {
        // Only the first kTailLanes lanes hold a byte of this group.
        if (lane >= kTailLanes) break;
        const float s = float(s_row[g]);
        const float b = dq4_int4_bias(b_row, g, s);
        const uint8_t byte = W_row[g * (kGroupSize / 2) + lane];
        for (uint t = 0; t < kInt4MaxRows; ++t) {
            if (t >= T) break;
            device const half* x_row = x + t * x_stride;
            const float x0 = float(x_row[g * kGroupSize + lane * 2u]);
            const float x1 = float(x_row[g * kGroupSize + lane * 2u + 1u]);
            float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
            dot = fma(float(uint(byte >> 4)), x1, dot);
            const float sum = x0 + x1;
            acc[t] = fma(s, dot, acc[t]);
            acc[t] = fma(b, sum, acc[t]);
        }
    }
    // `T` is uniform across the SIMD group, so every lane reaches each of these
    // reductions together.
    for (uint t = 0; t < kInt4MaxRows; ++t) {
        if (t >= T) break;
        const float total = simd_sum(acc[t]);
        if (lane == 0) {
            y[t * y_stride + row] = half(total);
        }
    }
}

kernel void dequant_int4_gemv_rows_simd(
    device const uint8_t* W        [[buffer(0)]],
    device const bfloat*  scales   [[buffer(1)]],
    device const bfloat*  biases   [[buffer(2)]],
    device const half*    x        [[buffer(3)]],
    device half*          y        [[buffer(4)]],
    constant uint&        M        [[buffer(5)]],
    constant uint&        N        [[buffer(6)]],
    constant uint&        T        [[buffer(7)]],
    constant uint&        x_stride [[buffer(8)]],
    constant uint&        y_stride [[buffer(9)]],
    uint                  tg_idx   [[threadgroup_position_in_grid]],
    uint                  sg_idx   [[simdgroup_index_in_threadgroup]],
    uint                  lane     [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint MM = int4_fc_m(M);
    const uint NN = int4_fc_n(N);
    dequant_int4_gemv_rows_simd_body(W, scales, biases, x, y, MM, NN, T,
                                     x_stride, y_stride,
                                     rows_per_tg, tg_idx, sg_idx, lane);
}


kernel void dequant_int4_qkv_gemv_simd(
    device const uint8_t* qW      [[buffer(0)]],
    device const bfloat*  qScales [[buffer(1)]],
    device const bfloat*  qBiases [[buffer(2)]],
    device const uint8_t* kW      [[buffer(3)]],
    device const bfloat*  kScales [[buffer(4)]],
    device const bfloat*  kBiases [[buffer(5)]],
    device const uint8_t* vW      [[buffer(6)]],
    device const bfloat*  vScales [[buffer(7)]],
    device const bfloat*  vBiases [[buffer(8)]],
    device const half*    x       [[buffer(9)]],
    device half*          qY      [[buffer(10)]],
    device half*          kY      [[buffer(11)]],
    device half*          vY      [[buffer(12)]],
    constant uint&        Mq      [[buffer(13)]],
    constant uint&        Mkv     [[buffer(14)]],
    constant uint&        N       [[buffer(15)]],
    uint                  tg_idx  [[threadgroup_position_in_grid]],
    uint                  sg_idx  [[simdgroup_index_in_threadgroup]],
    uint                  lane    [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint QQ = int4_qkv_fc_mq(Mq);
    const uint KK = int4_qkv_fc_mkv(Mkv);
    const uint NN = int4_qkv_fc_n(N);
    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    const uint total_rows = QQ + 2u * KK;
    if (global_row >= total_rows) { return; }

    device const uint8_t* W;
    device const bfloat* scales;
    device const bfloat* biases;
    device half* y;
    uint local_row;
    uint M;
    if (global_row < QQ) {
        W = qW; scales = qScales; biases = qBiases; y = qY;
        local_row = global_row;
        M = QQ;
    } else if (global_row < QQ + KK) {
        W = kW; scales = kScales; biases = kBiases; y = kY;
        local_row = global_row - QQ;
        M = KK;
    } else {
        W = vW; scales = vScales; biases = vBiases; y = vY;
        local_row = global_row - QQ - KK;
        M = KK;
    }
    dequant_int4_gemv_simd_body(W, scales, biases, x, y, M, NN,
                                1u, local_row, 0u, lane);
}

// ============================================================================
// y[t, m] = sum_n W[m, n] * x[t, n], with the per-t work cut down.
//
// `dequant_int4_gemv_rows_simd` above already reads W once for the whole
// block, which is why the verify block's dense bytes do not grow with k. What
// it does grow is arithmetic: `docs/mtp/19-M4.7-RESULTS.md` §5 measures the LM
// head at 4.1 ms for t=1 (a hair off this machine's DRAM roof) and 11.7 ms for
// t=8, and the shared expert the same shape. Two things in the inner loop are
// charged per activation row that need not be:
//
//   a. `sum = e0 + ... + e7`, the group's activation sum for the affine bias
//      term, does not depend on the weight row — yet every one of the 262144
//      head rows recomputes it. `xsum` carries it in, precomputed once per
//      block by `int4_rows_group_sums` in the same order, so the value is
//      bit-identical and seven adds per eight elements per row disappear.
//   b. The activation load and its half->float conversion are likewise shared
//      by every weight row. `FC_INT4_ROWS_PER_SG` gives one SIMD group R
//      consecutive weight rows instead of one, so a loaded `x` pays for R rows
//      of output rather than 1.
//
// `FC_INT4_ROWS_T` pins t at pipeline-build time so the row loop unrolls with
// no trip-count test and `acc` stays in registers.
//
// Per output element the arithmetic is `dequant_int4_gemv_simd`'s, in the same
// order, so this stays bit-identical to decode (see `--rows-bench --verify`).
// ============================================================================

constant uint FC_INT4_ROWS_PER_SG [[function_constant(27)]];
constant uint FC_INT4_ROWS_T      [[function_constant(28)]];

/// Weight rows one SIMD group may carry, and activation rows one dispatch may
/// carry. Both bound register arrays, and `acc` is their product, so they are
/// the kernel's occupancy knob: `docs/mtp/20-M4.8-RESULTS.md` §2 measures the
/// cliff at `T = 5`, which is why the wrapper splits a wider block into
/// dispatches of four rather than growing this.
constant constexpr uint kInt4MaxRowsPerSG = 4;
constant constexpr uint kInt4WideMaxT     = 4;

static inline uint int4_rows_per_sg() {
    return is_function_constant_defined(FC_INT4_ROWS_PER_SG) ? FC_INT4_ROWS_PER_SG : 1u;
}

/// `FC_INT4_ROWS_T` unset falls back to the bound `T`, which keeps one
/// pipeline for every block width at the cost of the trip-count test.
static inline uint int4_rows_t(constant uint& T) {
    return is_function_constant_defined(FC_INT4_ROWS_T) ? FC_INT4_ROWS_T : T;
}

// xsum[t, j] = sum of the eight activations lane j of the vectorized block
// covers, in the block kernel's order. `j` runs over `xsum_stride =
// (N / 256) * 32` entries, which is exactly the elements the full-block loop
// reads; the scalar tail keeps computing its own two-element sum.
kernel void int4_rows_group_sums(
    device const half*  x           [[buffer(0)]],
    device float*       xsum        [[buffer(1)]],
    constant uint&      x_stride    [[buffer(2)]],
    constant uint&      xsum_stride [[buffer(3)]],
    uint2               gid         [[thread_position_in_grid]]
) {
    if (gid.x >= xsum_stride) return;
    device const half* x_row = x + gid.y * x_stride;
    const uint elem = gid.x * 8u;
    const half4 xa = *((device const half4*)(x_row + elem));
    const half4 xb = *((device const half4*)(x_row + elem + 4u));
    const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
    const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
    xsum[gid.y * xsum_stride + gid.x] = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
}

kernel void dequant_int4_gemv_rows_wide_simd(
    device const uint8_t* W           [[buffer(0)]],
    device const bfloat*  scales      [[buffer(1)]],
    device const bfloat*  biases      [[buffer(2)]],
    device const half*    x           [[buffer(3)]],
    device half*          y           [[buffer(4)]],
    constant uint&        M           [[buffer(5)]],
    constant uint&        N           [[buffer(6)]],
    constant uint&        T_in        [[buffer(7)]],
    constant uint&        x_stride    [[buffer(8)]],
    constant uint&        y_stride    [[buffer(9)]],
    device const float*   xsum        [[buffer(10)]],
    constant uint&        xsum_stride [[buffer(11)]],
    uint                  tg_idx      [[threadgroup_position_in_grid]],
    uint                  sg_idx      [[simdgroup_index_in_threadgroup]],
    uint                  lane        [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint R = int4_rows_per_sg();
    const uint T = int4_rows_t(T_in);
    const uint MM = int4_fc_m(M);
    const uint NN = int4_fc_n(N);

    const uint row0 = (tg_idx * rows_per_tg + sg_idx) * R;
    if (row0 >= MM) return;

    const uint n_groups  = NN / kGroupSize;
    const uint row_bytes = NN / 2;
    const uint full_blocks = n_groups / kGroupsPerBlock;

    float acc[kInt4WideMaxT * kInt4MaxRowsPerSG];
    for (uint i = 0; i < kInt4WideMaxT * kInt4MaxRowsPerSG; ++i) { acc[i] = 0.0f; }

    // A row past the end reads the last row's bytes and is dropped at the
    // store; every production shape divides evenly, so this is a guard rather
    // than a path.
    uint row_of[kInt4MaxRowsPerSG];
    for (uint r = 0; r < R; ++r) { row_of[r] = min(row0 + r, MM - 1u); }

    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint g    = blk * kGroupsPerBlock + lane / kLanesPerGroup;
        const uint elem = byte_base * 2u;

        // Weight-side work: once per weight row, outside the activation loop.
        float q[kInt4MaxRowsPerSG * 8];
        float sv[kInt4MaxRowsPerSG];
        float bv[kInt4MaxRowsPerSG];
        for (uint r = 0; r < R; ++r) {
            const uint row = row_of[r];
            device const ushort* wp =
                (device const ushort*)(W + row * row_bytes + byte_base);
            const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
            sv[r] = float(scales[row * n_groups + g]);
            bv[r] = dq4_int4_bias(biases, row * n_groups + g, sv[r]);
            const uint b0 =  w4        & 0xFFu;
            const uint b1 = (w4 >> 8)  & 0xFFu;
            const uint b2 = (w4 >> 16) & 0xFFu;
            const uint b3 = (w4 >> 24) & 0xFFu;
            q[r * 8 + 0] = float(b0 & 0x0Fu); q[r * 8 + 1] = float(b0 >> 4);
            q[r * 8 + 2] = float(b1 & 0x0Fu); q[r * 8 + 3] = float(b1 >> 4);
            q[r * 8 + 4] = float(b2 & 0x0Fu); q[r * 8 + 5] = float(b2 >> 4);
            q[r * 8 + 6] = float(b3 & 0x0Fu); q[r * 8 + 7] = float(b3 >> 4);
        }

        for (uint t = 0; t < kInt4WideMaxT; ++t) {
            if (t >= T) break;
            device const half* x_row = x + t * x_stride;
            const half4 xa = *((device const half4*)(x_row + elem));
            const half4 xb = *((device const half4*)(x_row + elem + 4u));
            const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
            const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
            const float sum = xsum[t * xsum_stride + blk * 32u + lane];
            for (uint r = 0; r < R; ++r) {
                float dot = 0.0f;
                dot = fma(q[r * 8 + 0], e0, dot); dot = fma(q[r * 8 + 1], e1, dot);
                dot = fma(q[r * 8 + 2], e2, dot); dot = fma(q[r * 8 + 3], e3, dot);
                dot = fma(q[r * 8 + 4], e4, dot); dot = fma(q[r * 8 + 5], e5, dot);
                dot = fma(q[r * 8 + 6], e6, dot); dot = fma(q[r * 8 + 7], e7, dot);
                acc[r * kInt4WideMaxT + t] = fma(sv[r], dot, acc[r * kInt4WideMaxT + t]);
                acc[r * kInt4WideMaxT + t] = fma(bv[r], sum, acc[r * kInt4WideMaxT + t]);
            }
        }
    }

    for (uint gg = full_blocks * kGroupsPerBlock; gg < n_groups; ++gg) {
        if (lane >= kTailLanes) break;
        for (uint t = 0; t < kInt4WideMaxT; ++t) {
            if (t >= T) break;
            device const half* x_row = x + t * x_stride;
            const float x0 = float(x_row[gg * kGroupSize + lane * 2u]);
            const float x1 = float(x_row[gg * kGroupSize + lane * 2u + 1u]);
            const float sum = x0 + x1;
            for (uint r = 0; r < R; ++r) {
                const uint row = row_of[r];
                const float s = float(scales[row * n_groups + gg]);
                const float b = dq4_int4_bias(biases, row * n_groups + gg, s);
                const uint8_t byte = W[row * row_bytes + gg * (kGroupSize / 2) + lane];
                float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
                dot = fma(float(uint(byte >> 4)), x1, dot);
                acc[r * kInt4WideMaxT + t] = fma(s, dot, acc[r * kInt4WideMaxT + t]);
                acc[r * kInt4WideMaxT + t] = fma(b, sum, acc[r * kInt4WideMaxT + t]);
            }
        }
    }

    // `R` and `T` are compile-time, so every lane reaches each reduction.
    for (uint r = 0; r < R; ++r) {
        for (uint t = 0; t < kInt4WideMaxT; ++t) {
            if (t >= T) break;
            const float total = simd_sum(acc[r * kInt4WideMaxT + t]);
            if (lane == 0 && row0 + r < MM) {
                y[t * y_stride + row0 + r] = half(total);
            }
        }
    }
}
