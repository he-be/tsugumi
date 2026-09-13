#include <metal_stdlib>
using namespace metal;

// Routed-expert kernels for GGML block formats: IQ2_XXS gate/up and Q2_K down,
// the layout of `ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2`
// (`docs/investigations/QWEN38_FLASH_NEXT_VERIFY_PLAN.md`).
//
// The row arithmetic is ported from ds4-metal (MIT, Copyright (c) 2026 The
// ds4.c authors), `metal/moe.metal` at `3030554`:
// `kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` and `kernel_glm_q2_K_addr_down_f32`,
// which descend from llama.cpp's `ggml-metal.metal` (MIT). The lookup tables
// below are copied verbatim.
//
// What differs from ds4 is the calling convention, which is Tsugumi's decode
// one (`moe.metal`) widened to a batch: one argument buffer of per-slot expert
// blobs (a slot per distinct expert of the batch, up to 512), T tokens x top_k
// (token, expert) pairs mapped to slots by `pair_slot`, and phase 2 folding the
// routing weights and the residual in. Activations are float32, as in the
// Qwen3.8-Flash-Next verification runner (`qwen38.metal`).
//
// Lane geometry (both kernels, 2 SIMD groups x 32 lanes per threadgroup):
//   IQ2_XXS  lane l walks sub-blocks l, l+32, l+64, ... (32 weights each), so
//            the 32 lanes cover a row of `nb * 8` sub-blocks; 4 rows per group.
//   Q2_K     lane l = 8*ix + 4*iq + ir reads 8 weights at
//            ix*256 + 128*iq + 8*ir within blocks ix, ix+4, ...; 4 rows per group.
// `simd_sum` over the group's lanes assembles each row; lane 0 writes it.
// The IQ2_XXS path copies its codebook into threadgroup memory
// (256 x uint64 + 128 x uint8, set by the host), four entries per thread, which
// is why the threadgroup must be exactly 64 threads.

#define QK_K 256
#define GGML_ROWS_PER_GROUP 4
constant constexpr uint kGgmlGroupsPerTG = 2;
constant constexpr uint kGgmlMaxExperts = 512;

struct block_iq2_xxs {
    half d;
    ushort qs[QK_K/8];
};

struct block_q2_K {
    uchar scales[QK_K/16];
    uchar qs[QK_K/4];
    half d;
    half dmin;
};

/// Per-row byte counts of the two formats.
struct GgmlExpertOffsets {
    uint gate_row_bytes;   // IQ2_XXS row over D inputs
    uint down_row_bytes;   // Q2_K row over `act_stride` inputs
};

/// Per slot, the buffers holding that expert's gate, up and down rows. They may be
/// one contiguous blob (all three the same buffer) or three no-copy views of the
/// mapped GGUF; `part_off[3 * slot + {0,1,2}]` is where each part starts in its buffer.
struct GgmlRoutedBlobs {
    device const uint8_t* gate[kGgmlMaxExperts];
    device const uint8_t* up[kGgmlMaxExperts];
    device const uint8_t* down[kGgmlMaxExperts];
};

static constant uchar ds4_metal_kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static constant uchar ds4_metal_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static constant ulong ds4_metal_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

/// acts[pair * act_stride + f] = silu(gate_f . x[pair / top_k]) * (up_f . x[pair / top_k]),
/// for f < F and pair < tokens * top_k, with the expert of pair in slot pair_slot[pair].
/// `act_stride` is the Q2_K down input width (768 for a 640-wide expert); the
/// host zeroes the pad once and this kernel never writes it.
kernel void moe_iq2xxs_phase1_gate_up_act(
    device const GgmlRoutedBlobs& routed [[buffer(0)]],
    constant GgmlExpertOffsets& off [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    constant uint& act_stride [[buffer(7)]],
    device const uint* part_off [[buffer(8)]],
    constant uint& tokens [[buffer(9)]],
    device const uint* pair_slot [[buffer(10)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint64_t* svalues = (threadgroup uint64_t*)(shmem);
    threadgroup uint8_t* ssigns = (threadgroup uint8_t*)(svalues + 256);
    {
        int pos = (32 * sgitg + tiisg) * 4;
        for (int i = 0; i < 4; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        pos = (32 * sgitg + tiisg) * 2;
        for (int i = 0; i < 2; ++i) ssigns[pos + i] = ds4_metal_ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint first_row = (tg * kGgmlGroupsPerTG + sgitg) * GGML_ROWS_PER_GROUP;
    float sumg[GGML_ROWS_PER_GROUP] = {0.f};
    float sumu[GGML_ROWS_PER_GROUP] = {0.f};
    const bool live = first_row < tokens * top_k * F;
    const uint pair = live ? first_row / F : 0;
    const uint slot = pair_slot[pair];
    const uint row0 = live ? first_row % F : 0;

    if (live) {
        device const block_iq2_xxs* xg = (device const block_iq2_xxs*)(
            routed.gate[slot] + part_off[3 * slot] + row0 * off.gate_row_bytes);
        device const block_iq2_xxs* xu = (device const block_iq2_xxs*)(
            routed.up[slot] + part_off[3 * slot + 1] + row0 * off.gate_row_bytes);
        const int nb32 = int(D / QK_K) * (QK_K / 32);
        const uint step = off.gate_row_bytes / 2;

        float yl[32];
        const int ix = tiisg;
        device const float* y4 = x + (pair / top_k) * D + 32 * ix;
        for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
            for (short i = 0; i < 32; ++i) yl[i] = y4[i];
            const int ibl = ib32 / (QK_K / 32);
            const int ib = ib32 % (QK_K / 32);
            device const uint16_t* qg = (xg + ibl)->qs + 4 * ib;
            device const uint16_t* qu = (xu + ibl)->qs + 4 * ib;
            device const half* dhg = &(xg + ibl)->d;
            device const half* dhu = &(xu + ibl)->d;
            for (short row = 0; row < GGML_ROWS_PER_GROUP; row++) {
                device const uint8_t* aux8g = (device const uint8_t*)qg;
                device const uint8_t* aux8u = (device const uint8_t*)qu;
                const uint32_t aux32g = qg[2] | (qg[3] << 16);
                const uint32_t aux32u = qu[2] | (qu[3] << 16);
                const float dg = float(dhg[0]) * (0.5f + (aux32g >> 28));
                const float du = float(dhu[0]) * (0.5f + (aux32u >> 28));
                float sg = 0;
                float su = 0;
                for (short l = 0; l < 4; ++l) {
                    const threadgroup uint8_t* gridg = (const threadgroup uint8_t*)(svalues + aux8g[l]);
                    const threadgroup uint8_t* gridu = (const threadgroup uint8_t*)(svalues + aux8u[l]);
                    const uint8_t signg = ssigns[(aux32g >> 7 * l) & 127];
                    const uint8_t signu = ssigns[(aux32u >> 7 * l) & 127];
                    for (short j = 0; j < 8; ++j) {
                        const float v = yl[8 * l + j];
                        sg += v * gridg[j] * (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                        su += v * gridu[j] * (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    }
                }
                sumg[row] += dg * sg;
                sumu[row] += du * su;
                dhg += step; dhu += step;
                qg += step;  qu += step;
            }
            y4 += 32 * 32;
        }
    }

    // Every lane of the group reaches this reduction, live or not.
    float4 rg, ru;
    for (int row = 0; row < GGML_ROWS_PER_GROUP; ++row) {
        rg[row] = simd_sum(sumg[row]);
        ru[row] = simd_sum(sumu[row]);
    }
    if (live && tiisg == 0) {
        for (uint row = 0; row < GGML_ROWS_PER_GROUP && row0 + row < F; ++row) {
            const float g = rg[row] * 0.25f;
            const float u = ru[row] * 0.25f;
            acts[pair * act_stride + row0 + row] = g / (1.0f + exp(-g)) * u;
        }
    }
}

/// y[t][r] = residual[t][r] + sum_k routing_w[t * top_k + k] * (down row r . acts[t * top_k + k]),
/// the down rows of the pair's slot.
kernel void moe_q2k_phase2_down_reduce(
    device const GgmlRoutedBlobs& routed [[buffer(0)]],
    constant GgmlExpertOffsets& off [[buffer(1)]],
    device const float* acts [[buffer(2)]],
    device const float* routing_w [[buffer(3)]],
    device const float* residual [[buffer(4)]],
    device float* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& act_stride [[buffer(7)]],
    constant uint& top_k [[buffer(8)]],
    device const uint* part_off [[buffer(9)]],
    constant uint& tokens [[buffer(10)]],
    device const uint* pair_slot [[buffer(11)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint first_row = (tg * kGgmlGroupsPerTG + sgitg) * GGML_ROWS_PER_GROUP;
    float sumf[GGML_ROWS_PER_GROUP] = {0.f};
    const bool live = first_row < tokens * D;
    const uint tok = live ? first_row / D : 0;
    const uint row0 = live ? first_row % D : 0;

    if (live) {
        const short ix = tiisg / 8;
        const short it = tiisg % 8;
        const short iq = it / 4;
        const short ir = it % 4;
        const short is = (8 * ir) / 16;
        const int nb = int(act_stride / QK_K);
        const uint rb = off.down_row_bytes;
        for (uint k = 0; k < top_k; k++) {
            const uint pair = tok * top_k + k;
            const uint slot = pair_slot[pair];
            const float w = routing_w[pair];
            device const block_q2_K* xb =
                (device const block_q2_K*)(routed.down[slot] + part_off[3 * slot + 2] + row0 * rb);
            device const float* y4 = acts + pair * act_stride + ix * QK_K + 128 * iq + 8 * ir;
            for (int ib = ix; ib < nb; ib += 4) {
                float yl[32];
                float4 sumy = {0.f, 0.f, 0.f, 0.f};
                for (short i = 0; i < 8; ++i) {
                    yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                    yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                    yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                    yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
                }
                device const uint8_t* sc = (device const uint8_t*)xb[ib].scales + 8 * iq + is;
                device const uint16_t* qs = (device const uint16_t*)xb[ib].qs + 16 * iq + 4 * ir;
                device const half* dh = &xb[ib].d;
                for (short row = 0; row < GGML_ROWS_PER_GROUP && row0 + uint(row) < D; row++) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                        acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                        acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                        acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                        acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                        acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                        acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                        acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                    }
                    const float d = dh[0];
                    const float m = dh[1] * 1.f / 16.f;
                    sumf[row] += w * (d * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                           (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                           (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                           (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                                      m * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                           sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0)));
                    qs += rb / 2;
                    sc += rb;
                    dh += rb / 2;
                }
                y4 += 4 * QK_K;
            }
        }
    }

    float4 reduced;
    for (int row = 0; row < GGML_ROWS_PER_GROUP; ++row) reduced[row] = simd_sum(sumf[row]);
    if (live && tiisg == 0) {
        for (uint row = 0; row < GGML_ROWS_PER_GROUP && row0 + row < D; ++row) {
            y[tok * D + row0 + row] = residual[tok * D + row0 + row] + reduced[row];
        }
    }
}

// ---------------------------------------------------------------------------
// Dequantize + sgemm form of the two kernels above (`--q2-gemm-bench`, docs/qwen38/06):
// the pairs are gathered by expert, each expert's rows are expanded to float32 once
// and multiplied with MPS, and the results are scattered back with the routing weights.

/// Gate and up rows of the G experts in slots first_slot.. -> float32 out[g][row][col], rows 0..<F gate
/// then F..<2F up, reading the same per-slot views as `moe_iq2xxs_phase1_gate_up_act`.
/// Thread (32-weight sub-block, row, g). Same weights as that kernel (d * (0.5 + scale) * grid byte * sign * 0.25).
kernel void moe_iq2xxs_dequant_gate_up_f32(
    device const GgmlRoutedBlobs& routed [[buffer(0)]],
    device const uint* part_off [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    constant uint& F [[buffer(4)]],
    constant uint& first_slot [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    const uint row_bytes = (D / QK_K) * sizeof(block_iq2_xxs);
    const uint slot = first_slot + pos.z;
    const bool is_up = pos.y >= F;
    const uint r = is_up ? pos.y - F : pos.y;
    device const uint8_t* src = (is_up ? routed.up[slot] : routed.gate[slot]) + part_off[3 * slot + (is_up ? 1 : 0)] + r * row_bytes;
    device const block_iq2_xxs* b = (device const block_iq2_xxs*)(src + (pos.x / 8) * sizeof(block_iq2_xxs));
    device const uint16_t* q = b->qs + 4 * (pos.x % 8);
    const uint32_t aux32 = q[2] | (uint32_t(q[3]) << 16);
    const float db = float(b->d) * (0.5f + (aux32 >> 28)) * 0.25f;
    device const uint8_t* aux8 = (device const uint8_t*)q;
    device float* o = out + (ulong(pos.z) * 2 * F + pos.y) * D + pos.x * 32;
    for (uint l = 0; l < 4; ++l) {
        const ulong grid = ds4_metal_iq2xxs_grid[aux8[l]];
        const uchar sign = ds4_metal_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
        for (uint j = 0; j < 8; ++j) {
            const float v = float((grid >> (8 * j)) & 0xff);
            o[8 * l + j] = db * ((sign & ds4_metal_kmask_iq2xs[j]) ? -v : v);
        }
    }
}

/// Down rows of the G experts in slots first_slot.., first `cols` (<= stride) columns -> float32
/// out[g][row][col]. Thread (16-weight sub-block, row, g). Q2_K as in `dequantize_row_q2_K`:
/// sub-block s reads byte 32 * (s / 8) + 16 * (s % 2) + l at shift 2 * ((s % 8) / 2).
kernel void moe_q2k_dequant_down_f32(
    device const GgmlRoutedBlobs& routed [[buffer(0)]],
    device const uint* part_off [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    constant uint& stride [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& first_slot [[buffer(6)]],
    uint3 pos [[thread_position_in_grid]]
) {
    const uint row_bytes = (stride / QK_K) * sizeof(block_q2_K);
    const uint slot = first_slot + pos.z;
    const uint p = pos.x * 16;
    device const block_q2_K* b = (device const block_q2_K*)(
        routed.down[slot] + part_off[3 * slot + 2] + pos.y * row_bytes + (p / QK_K) * sizeof(block_q2_K));
    const uint s = (p % QK_K) / 16;
    const uchar sc = b->scales[s];
    const float dl = float(b->d) * float(sc & 0xF);
    const float ml = float(b->dmin) * float(sc >> 4);
    device const uint8_t* q = b->qs + 32 * (s / 8) + 16 * (s % 2);
    const uint shift = 2 * ((s % 8) / 2);
    device float* o = out + (ulong(pos.z) * D + pos.y) * cols + p;
    const uint n = min(16u, cols - p);
    for (uint l = 0; l < n; ++l) o[l] = dl * float((q[l] >> shift) & 3) - ml;
}

/// out[i][c] = x[src[i] / top_k][c]: the token rows of the pairs, in the order `src` lists them.
/// Thread (32-column chunk, i).
kernel void moe_gather_pair_rows(
    device const float* x [[buffer(0)]],
    device const uint* src [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& W [[buffer(3)]],
    constant uint& top_k [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]
) {
    device const float* xi = x + (src[pos.y] / top_k) * W + pos.x * 32;
    device float* o = out + ulong(pos.y) * W + pos.x * 32;
    const uint n = min(32u, W - pos.x * 32);
    for (uint c = 0; c < n; ++c) o[c] = xi[c];
}

/// gu[i][f] = silu(gu[i][f]) * gu[i][F + f] for rows first..<first + count. Thread (f, i).
kernel void moe_silu_mul_halves(
    device float* gu [[buffer(0)]],
    constant uint& F [[buffer(1)]],
    constant uint& first [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const ulong base = ulong(first + pos.y) * 2 * F;
    const float g = gu[base + pos.x];
    gu[base + pos.x] = g / (1.0f + exp(-g)) * gu[base + F + pos.x];
}

/// y[t][r] = residual[t][r] + sum_k routing_w[t * top_k + k] * rows[at[t * top_k + k]][r].
/// Thread (32-column chunk, t).
kernel void moe_scatter_weighted(
    device const float* rows [[buffer(0)]],
    device const uint* at [[buffer(1)]],
    device const float* routing_w [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint c0 = pos.x * 32;
    const uint n = min(32u, D - c0);
    float acc[32] = {0.f};
    for (uint k = 0; k < top_k; ++k) {
        const uint pair = pos.y * top_k + k;
        const float w = routing_w[pair];
        device const float* r = rows + ulong(at[pair]) * D + c0;
        for (uint c = 0; c < n; ++c) acc[c] += w * r[c];
    }
    device const float* res = residual + ulong(pos.y) * D + c0;
    device float* o = y + ulong(pos.y) * D + c0;
    for (uint c = 0; c < n; ++c) o[c] = res[c] + acc[c];
}
