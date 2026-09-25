// F1 (docs/qfn-prefill/12 §2): one layer's GEMMs on the Metal 4 tensor ops, for `TsugumiKernelCheck --qfn-layer-bench`.
// Not part of the runtime. The bench compiles this after Sources/Tsugumi/Metal/MoE/moe_ggml.metal (its IQ2_XXS tables,
// block structs and GgmlRoutedBlobs) with MSL 4.0, on a device with the tensor API only.
//
// Routed gate/up (+ SiLU) and down: ds4-metal origin/qwen2-nax-tiles (dddb1ba), metal/qwen4.metal L2740-2831 (raw loads
// and dequant) and kernel_qwen4_moe_mm_{mid,down}_nax_t<64, half, false> (L3085-3421), with two changes:
// the expert bases come from the runner's per-expert views (GgmlRoutedBlobs + part_off, one slot per selected expert),
// and the Q2_K down rows are the 212 B sidecar rows whose last block is the 44 B half block (scales[8] | qs[32] | d | dmin).
// The mid tiles write only the half copy of mid (the down tiles' operand).
// Dense: the same tile (64 weight rows x 64 tokens, K steps of 32, fp32 accumulation) over Q8_0 / F16 rows, tokens
// contiguous.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

/// Raw words of the 16 weights [32 kb + 16 hf, +16) of one row, loaded one K step ahead of their dequant.
struct qfn_raw { uint4 h; uint4 q; };

static inline qfn_raw qfn_load(device const char *row, uint kb, uint hf, uint type, uint row_bytes) {
    qfn_raw r;
    if (type == 8) {            // Q8_0: d | qs[32], 34 B, 2-byte aligned
        device const ushort *p = (device const ushort *)(row + kb * 34);
        r.h.x = p[0];
        device const ushort *q = p + 1 + hf * 8;
        r.q = uint4(uint(q[0]) | (uint(q[1]) << 16), uint(q[2]) | (uint(q[3]) << 16),
                    uint(q[4]) | (uint(q[5]) << 16), uint(q[6]) | (uint(q[7]) << 16));
    } else if (type == 1) {     // F16
        device const uint4 *p = (device const uint4 *)(row + kb * 64 + hf * 32);
        r.h = p[0];
        r.q = p[1];
    } else if (type == 10) {    // Q2_K: scales[16] @0 | qs[64] @16 | d @80 | dmin @82; half block: scales[8] @0 | qs[32] @8 | d @40
        const uint sb = kb / 8, g = (kb % 8) * 2 + hf;
        device const uchar *blk = (device const uchar *)(row + sb * 84);
        const bool tail = (sb + 1) * 84 > row_bytes;
        r.h.x = *(device const uint *)(blk + (tail ? 40 : 80));
        r.h.y = blk[g];
        device const uint *qs = (device const uint *)(blk + (tail ? 8 : 16) + 32u * (g / 8u) + 16u * (g & 1u));
        r.q = uint4(qs[0], qs[1], qs[2], qs[3]);
    } else {                    // 16, IQ2_XXS: d | qs[32 ushort], 66 B
        const uint sb = kb / 8, ib32 = kb % 8;
        device const uchar *blk = (device const uchar *)(row + sb * 66);
        device const ushort *q2 = (device const ushort *)(blk + 2) + 4 * ib32;
        const uint ag = (uint)q2[0] | ((uint)q2[1] << 16);
        const uint auxs = (uint)q2[2] | ((uint)q2[3] << 16);
        r.h.x = ((uint)*(device const ushort *)blk) | (ag << 16);
        r.h.y = (ag >> 16) | (auxs << 16);
        r.h.z = auxs >> 16;
    }
    return r;
}

static inline void qfn_dequant(qfn_raw r, uint kb, uint hf, uint type, threadgroup half *dst) {
    if (type == 8) {
        const float d = (float)as_type<half>((ushort)(r.h.x & 0xFFFFu));
        for (uint i = 0; i < 16; i++) dst[i] = (half)(d * (float)as_type<char>((uchar)((r.q[i >> 2] >> (8u * (i & 3u))) & 0xFFu)));
        return;
    }
    if (type == 1) {
        for (uint i = 0; i < 8; i++) dst[i] = as_type<half>((ushort)((r.h[i >> 1] >> (16u * (i & 1u))) & 0xFFFFu));
        for (uint i = 0; i < 8; i++) dst[8 + i] = as_type<half>((ushort)((r.q[i >> 1] >> (16u * (i & 1u))) & 0xFFFFu));
        return;
    }
    if (type == 10) {
        const uint g = (kb % 8) * 2 + hf;
        const float d = (float)as_type<half>((ushort)(r.h.x & 0xFFFFu));
        const float dmin = (float)as_type<half>((ushort)(r.h.x >> 16));
        const float ds = d * (float)(r.h.y & 0xFu), dm = dmin * (float)(r.h.y >> 4);
        const uint shift = ((g / 2u) & 3u) * 2u;
        for (uint i = 0; i < 16; i++) {
            const uint byte = (r.q[i >> 2] >> (8u * (i & 3u))) & 0xFFu;
            dst[i] = (half)(ds * (float)((byte >> shift) & 3u) - dm);
        }
        return;
    }
    const uint ag = (r.h.x >> 16) | (r.h.y << 16);
    const uint auxs = (r.h.y >> 16) | (r.h.z << 16);
    const float d = (float)as_type<half>((ushort)(r.h.x & 0xFFFFu));
    const float dl = d * (0.5f + (float)(auxs >> 28)) * 0.25f;
    for (uint h = 0; h < 2; h++) {
        const uint qq = hf * 2 + h;
        constant const uchar *grid = (constant const uchar *)(ds4_metal_iq2xxs_grid + ((ag >> (8u * qq)) & 0xFFu));
        const uint signs = ds4_metal_ksigns_iq2xs[(auxs >> (7u * qq)) & 127u];
        for (uint i = 0; i < 8; i++) dst[h * 8 + i] = (half)(dl * (float)grid[i] * ((signs >> i) & 1u ? -1.0f : 1.0f));
    }
}

static inline void qfn_stage8(threadgroup half *dst, device const half *src) {
    *(threadgroup uint4 *)dst = src ? *(device const uint4 *)src : uint4(0u);
}

struct qfn_dense_args { uint T, K, M, row_bytes, type; };

/// y[t][m] = W[m] . x[t] for 64 rows x 64 tokens per threadgroup (128 threads), x in half, y in float.
kernel void qfn_dense_nax(constant qfn_dense_args &a [[buffer(0)]],
                          device const char *W [[buffer(1)]],
                          device const half *x [[buffer(2)]],
                          device float *y [[buffer(3)]],
                          threadgroup char *shmem [[threadgroup(0)]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]],
                          ushort tiisg [[thread_index_in_simdgroup]],
                          ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64, NR1 = 64, NK = 32, NB = NR1 * 4 / 128;
    const uint row0 = tg.x * NR0, t0 = tg.y * NR1;
    const uint n_tile = min((uint)NR1, a.T - t0);
    threadgroup half *As = (threadgroup half *)shmem;             // [64 row][32 k]
    threadgroup half *Bs = (threadgroup half *)(shmem + 4096);    // [64 tok][32 k]
    threadgroup float *Cs = (threadgroup float *)shmem;           // [64 tok][64 row] after the K loop
    auto tA = tensor(As, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(Bs, dextents<int32_t, 2>(NK, NR1));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0.0f; }
    device const half *xr[NB];
    threadgroup half *bdst[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) {
        const uint item = (uint)tid + (uint)b * 128u, tok = item / 4u, kq = item % 4u;
        xr[b] = tok < n_tile ? x + (ulong)(t0 + tok) * a.K + kq * 8u : (device const half *)0;
        bdst[b] = Bs + tok * NK + kq * 8u;
    }
    const uint ar = tid / 2, aq = tid % 2;
    const bool a_row = row0 + ar < a.M;
    device const char *wrow = W + (ulong)min(row0 + ar, a.M - 1u) * a.row_bytes;
    const uint nk = a.K / NK;
    qfn_raw rw = qfn_load(wrow, 0, aq, a.type, a.row_bytes);
    for (uint kb = 0; kb < nk; kb++) {
        threadgroup half *dst = As + ar * NK + aq * 16;
        if (a_row) qfn_dequant(rw, kb, aq, a.type, dst);
        else for (uint i = 0; i < 16; i++) dst[i] = 0.0h;
        if (kb + 1 < nk) rw = qfn_load(wrow, kb + 1, aq, a.type, a.row_bytes);
#pragma unroll
        for (int b = 0; b < NB; b++) qfn_stage8(bdst[b], xr[b] ? xr[b] + kb * NK : (device const half *)0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto mB = tB.slice(0, 0);
            auto mA = tA.slice(0, 0);
            mm.run(mB, mA, cT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    {
        auto tC = tensor(Cs, dextents<int32_t, 2>(NR0, NR1));
        auto mC = tC.slice(0, 0);
        cT.store(mC);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = sgitg; j < n_tile; j += 4) {
        device float *out = y + (ulong)(t0 + j) * a.M + row0;
        for (uint i = tiisg; i < (uint)NR0 && row0 + i < a.M; i += 32) out[i] = Cs[j * NR0 + i];
    }
}

struct qfn_moe_args {
    uint n_slots;       // top-k: pair = token * n_slots + k
    uint in_dim;
    uint out_rows;
    uint row_bytes;
    uint list_cap;
    uint n_expert;      // selected experts (slots of the argument buffer)
    uint tiles_per_launch;
    uint type;
    uint part;          // 0 gate/up, 2 down: which part_off / view of the slot
};

static inline float qfn_silu(float x) { return x / (1.0f + exp(-x)); }

/// midh[pair][r] = silu(gate[r] . x[t]) * (up[r] . x[t]) for the pairs of slot e, 64 rows x 64 pairs per tile.
kernel void qfn_moe_mid_nax(constant qfn_moe_args &args [[buffer(0)]],
                            device const GgmlRoutedBlobs &routed [[buffer(1)]],
                            device const uint *part_off [[buffer(2)]],
                            device const int *lists [[buffer(3)]],
                            device const int *counts [[buffer(4)]],
                            device const half *x [[buffer(5)]],        // [T][in_dim]
                            device half *midh [[buffer(6)]],           // [T * n_slots][out_rows]
                            threadgroup char *shmem [[threadgroup(0)]],
                            uint3 tgpig [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort tiisg [[thread_index_in_simdgroup]],
                            ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64, NR1 = 64, NK = 32, NB = NR1 * 4 / 128;
    const uint n_rb = (args.out_rows + NR0 - 1u) / NR0;
    const uint rb = tgpig.x % n_rb, tile0 = tgpig.x / n_rb;   // expert-major
    const uint e = tgpig.y;
    if (e >= args.n_expert) return;
    const uint count = (uint)counts[e];
    threadgroup half *Ag = (threadgroup half *)shmem;
    threadgroup half *Au = (threadgroup half *)(shmem + 4096);
    threadgroup half *Bs = (threadgroup half *)(shmem + 8192);
    threadgroup float *Cs = (threadgroup float *)shmem;
    device const char *gbase = (device const char *)routed.gate[e] + part_off[3 * e];
    device const char *ubase = (device const char *)routed.up[e] + part_off[3 * e + 1];
    device const int *list = lists + (ulong)e * args.list_cap;
    const uint row0 = rb * NR0;
    const uint nk = args.in_dim / NK;
    auto tA_g = tensor(Ag, dextents<int32_t, 2>(NK, NR0));
    auto tA_u = tensor(Au, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(Bs, dextents<int32_t, 2>(NK, NR1));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    const uint ar = tid / 2, aq = tid % 2;
    for (uint tile = tile0; tile * NR1 < count; tile += args.tiles_per_launch) {
        const uint t0 = tile * NR1;
        const uint n_tile = min((uint)NR1, count - t0);
        auto cG = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA_g), float>();
        auto cU = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA_u), float>();
#pragma unroll
        for (uint16_t i = 0; i < cG.get_capacity(); ++i) { if (cG.is_valid_element(i)) cG[i] = 0.0f; }
#pragma unroll
        for (uint16_t i = 0; i < cU.get_capacity(); ++i) { if (cU.is_valid_element(i)) cU[i] = 0.0f; }
        device const half *xr[NB];
        threadgroup half *bdst[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) {
            const uint item = (uint)tid + (uint)b * 128u, tok = item / 4u, kq = item % 4u;
            const int pair = tok < n_tile ? list[t0 + tok] : -1;
            xr[b] = pair >= 0 ? x + (ulong)((uint)pair / args.n_slots) * args.in_dim + kq * 8u : (device const half *)0;
            bdst[b] = Bs + tok * NK + kq * 8u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // previous tile's C consumed
        const bool a_row = row0 + ar < args.out_rows;
        device const char *grow = gbase + (ulong)(row0 + min(ar, args.out_rows - 1u)) * args.row_bytes;
        device const char *urow = ubase + (ulong)(row0 + min(ar, args.out_rows - 1u)) * args.row_bytes;
        qfn_raw rg = qfn_load(grow, 0, aq, args.type, args.row_bytes), ru = qfn_load(urow, 0, aq, args.type, args.row_bytes);
        for (uint kb = 0; kb < nk; kb++) {
            {
                threadgroup half *dg = Ag + ar * NK + aq * 16;
                threadgroup half *du = Au + ar * NK + aq * 16;
                if (a_row) {
                    qfn_dequant(rg, kb, aq, args.type, dg);
                    qfn_dequant(ru, kb, aq, args.type, du);
                } else {
                    for (uint i = 0; i < 16; i++) { dg[i] = 0.0h; du[i] = 0.0h; }
                }
                if (kb + 1 < nk) {
                    rg = qfn_load(grow, kb + 1, aq, args.type, args.row_bytes);
                    ru = qfn_load(urow, kb + 1, aq, args.type, args.row_bytes);
                }
            }
#pragma unroll
            for (int b = 0; b < NB; b++) qfn_stage8(bdst[b], xr[b] ? xr[b] + kb * NK : (device const half *)0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                auto mB = tB.slice(0, 0);
                auto mAg = tA_g.slice(0, 0);
                auto mAu = tA_u.slice(0, 0);
                mm.run(mB, mAg, cG);
                mm.run(mB, mAu, cU);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
#pragma unroll
        for (uint16_t i = 0; i < cG.get_capacity(); ++i) { if (cG.is_valid_element(i)) cG[i] = qfn_silu(cG[i]) * cU[i]; }
        {
            auto tC = tensor(Cs, dextents<int32_t, 2>(NR0, NR1));
            auto mC = tC.slice(0, 0);
            cG.store(mC);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = sgitg; j < n_tile; j += 4) {
            device half *out = midh + (ulong)list[t0 + j] * args.out_rows + row0;
            for (uint i = tiisg; i < (uint)NR0 && row0 + i < args.out_rows; i += 32) out[i] = (half)Cs[j * NR0 + i];
        }
    }
}

/// part[pair][r] = down[r] . midh[pair] for the pairs of slot e (K = in_dim = 640; the sidecar's pad is never read).
kernel void qfn_moe_down_nax(constant qfn_moe_args &args [[buffer(0)]],
                             device const GgmlRoutedBlobs &routed [[buffer(1)]],
                             device const uint *part_off [[buffer(2)]],
                             device const int *lists [[buffer(3)]],
                             device const int *counts [[buffer(4)]],
                             device const half *midv [[buffer(5)]],    // [T * n_slots][in_dim]
                             device float *part [[buffer(6)]],         // [T * n_slots][out_rows]
                             threadgroup char *shmem [[threadgroup(0)]],
                             uint3 tgpig [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]],
                             ushort tiisg [[thread_index_in_simdgroup]],
                             ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64, NR1 = 64, NK = 32, NB = NR1 * 4 / 128;
    const uint n_rb = (args.out_rows + NR0 - 1u) / NR0;
    const uint rb = tgpig.x % n_rb, tile0 = tgpig.x / n_rb;
    const uint e = tgpig.y;
    if (e >= args.n_expert) return;
    const uint count = (uint)counts[e];
    threadgroup half *As = (threadgroup half *)shmem;
    threadgroup half *Bs = (threadgroup half *)(shmem + 4096);
    threadgroup float *Cs = (threadgroup float *)shmem;
    device const char *dbase = (device const char *)routed.down[e] + part_off[3 * e + 2];
    device const int *list = lists + (ulong)e * args.list_cap;
    const uint row0 = rb * NR0;
    const uint nk = args.in_dim / NK;
    auto tA = tensor(As, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(Bs, dextents<int32_t, 2>(NK, NR1));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    const uint ar = tid / 2, aq = tid % 2;
    for (uint tile = tile0; tile * NR1 < count; tile += args.tiles_per_launch) {
        const uint t0 = tile * NR1;
        const uint n_tile = min((uint)NR1, count - t0);
        auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
#pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0.0f; }
        device const half *mr[NB];
        threadgroup half *bdst[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) {
            const uint item = (uint)tid + (uint)b * 128u, tok = item / 4u, kq = item % 4u;
            const int pair = tok < n_tile ? list[t0 + tok] : -1;
            mr[b] = pair >= 0 ? midv + (ulong)pair * args.in_dim + kq * 8u : (device const half *)0;
            bdst[b] = Bs + tok * NK + kq * 8u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool a_row = row0 + ar < args.out_rows;
        device const char *drow = dbase + (ulong)(row0 + min(ar, args.out_rows - 1u)) * args.row_bytes;
        qfn_raw rd = qfn_load(drow, 0, aq, args.type, args.row_bytes);
        for (uint kb = 0; kb < nk; kb++) {
            {
                threadgroup half *dd = As + ar * NK + aq * 16;
                if (a_row) qfn_dequant(rd, kb, aq, args.type, dd);
                else for (uint i = 0; i < 16; i++) dd[i] = 0.0h;
                if (kb + 1 < nk) rd = qfn_load(drow, kb + 1, aq, args.type, args.row_bytes);
            }
#pragma unroll
            for (int b = 0; b < NB; b++) qfn_stage8(bdst[b], mr[b] ? mr[b] + kb * NK : (device const half *)0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                auto mB = tB.slice(0, 0);
                auto mA = tA.slice(0, 0);
                mm.run(mB, mA, cT);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        {
            auto tC = tensor(Cs, dextents<int32_t, 2>(NR0, NR1));
            auto mC = tC.slice(0, 0);
            cT.store(mC);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = sgitg; j < n_tile; j += 4) {
            device float *out = part + (ulong)list[t0 + j] * args.out_rows + row0;
            for (uint i = tiisg; i < (uint)NR0 && row0 + i < args.out_rows; i += 32) out[i] = Cs[j * NR0 + i];
        }
    }
}

/// y[t] = residual[t] + sum_k w[t][k] part[t * top_k + k], four columns per thread.
kernel void qfn_moe_reduce(device const float4 *part [[buffer(0)]], device const float *w [[buffer(1)]],
                           device const float4 *residual [[buffer(2)]], device float4 *y [[buffer(3)]],
                           constant uint &D4 [[buffer(4)]], constant uint &top_k [[buffer(5)]],
                           uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= D4) return;
    float4 acc = residual[(ulong)pos.y * D4 + pos.x];
    for (uint k = 0; k < top_k; k++) {
        const uint pair = pos.y * top_k + k;
        acc += w[pair] * part[(ulong)pair * D4 + pos.x];
    }
    y[(ulong)pos.y * D4 + pos.x] = acc;
}
