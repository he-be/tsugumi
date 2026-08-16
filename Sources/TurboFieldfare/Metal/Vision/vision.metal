#include <metal_stdlib>
using namespace metal;

// Gemma 4 vision tower kernels.
//
// The tower is a 27-layer SigLIP-shaped encoder whose weights stay BF16: it is
// the one part of the model that is not quantized, so none of the INT4 kernels
// apply and none of the affine group machinery is needed here. Where a prefill
// kernel already does the same arithmetic on the same layout it is reused
// instead of copied (`prefill_rmsnorm_bf16w_block` for the [P, 1152] norms,
// `prefill_rmsnorm_no_scale_perhead_block` for the projector's pre-norm); this
// file holds only what the text path has no equivalent of.
//
// Shapes, for the S = 280 soft-token maximum: P = 9S = 2520 patches,
// hidden 1152, 16 heads of 72, MLP intermediate 4304.

constant constexpr uint kVisionQMMTileM = 64;
constant constexpr uint kVisionQMMTileN = 64;
constant constexpr uint kVisionQMMTileK = 32;
constant constexpr uint kVisionQMMThreads = 128;
// Three elements per lane over a 32-lane simdgroup: the Q/K/V epilogue covers
// head dimensions up to 96, and Gemma 4's tower is 72.
constant constexpr uint kVisionMaxHeadDim = 96;
constant constexpr float kVisionGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kVisionGeluCubicCoeff = 0.044715f;

static inline float vision_gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = kVisionGeluSqrt2OverPi * (x + kVisionGeluCubicCoeff * x3);
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

/// `Y[T, N] = X[T, K] * W[N, K]^T` with BF16 weights and 8x8 `simdgroup_matrix`
/// tiles.
///
/// This is `prefill_int4_qmm_simdgroup_f16` with the dequant removed: the
/// weight tile is filled by a BF16 -> half conversion instead of an unpack and
/// an affine reconstruction, so the scale/bias rows and the group indexing are
/// gone with it. Two differences remain against that kernel:
///
///  * K is not constrained. The INT4 kernel could assume K was a multiple of
///    the affine group and therefore of the 32-wide K tile; here the down
///    projection has K = 4304, which is 16 short of a whole tile, so both
///    staging loops mask the K tail to zero.
///  * The weights are BF16 in memory and half in the tile, which costs one
///    rounding per weight — the same trade the INT4 path already makes, and the
///    reason the tower is not bit-identical to a float32 reference.
///
/// The accumulators stay float, so the K reduction has the precision of the
/// scalar reference. One threadgroup (4 simdgroups, 128 threads) owns a 64x64
/// output tile; each simdgroup owns a 32x32 quadrant as a 4x4 array of 8x8
/// accumulators. `stage` is aliased as the two half tiles during the K loop and
/// reused as the float epilogue tile afterwards.
kernel void vision_bf16_qmm_f16(
    device const bfloat* W    [[buffer(0)]],
    device const half*   X    [[buffer(1)]],
    device half*         Y    [[buffer(2)]],
    constant uint&       T    [[buffer(3)]],
    constant uint&       N    [[buffer(4)]],
    constant uint&       K    [[buffer(5)]],
    uint3                lid3 [[thread_position_in_threadgroup]],
    uint3                tgid3 [[threadgroup_position_in_grid]],
    uint                 sgid [[simdgroup_index_in_threadgroup]]
) {
    const uint lid = lid3.x;
    const uint2 tgid = tgid3.xy;
    threadgroup float stage[kVisionQMMTileM * kVisionQMMTileN];
    threadgroup half* As = (threadgroup half*)stage;
    threadgroup half* Bs = As + kVisionQMMTileM * kVisionQMMTileK;

    const uint tBase = tgid.y * kVisionQMMTileM;
    const uint nBase = tgid.x * kVisionQMMTileN;

    // Quadrant of the 64x64 tile this simdgroup accumulates.
    const uint sg_m = (sgid / 2u) * 32u;
    const uint sg_n = (sgid % 2u) * 32u;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4u; ++i) {
        for (uint j = 0; j < 4u; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    // Weight row and 8-wide chunk this thread refills every K tile. Two threads
    // share a row and take two of the four chunks each.
    const uint w_n_local = lid & (kVisionQMMTileN - 1u);
    const uint w_chunk_base = lid / kVisionQMMTileN;
    const uint w_n = nBase + w_n_local;
    device const bfloat* w_row = W + w_n * K;

    for (uint k0 = 0; k0 < K; k0 += kVisionQMMTileK) {
        // Activations: 64x32 halves, 16 per thread, K-contiguous per row.
        for (uint i = 0; i < 16u; ++i) {
            const uint idx = i * kVisionQMMThreads + lid;
            const uint m = idx / kVisionQMMTileK;
            const uint kk = idx % kVisionQMMTileK;
            const uint t = tBase + m;
            const uint k = k0 + kk;
            As[idx] = (t < T && k < K) ? X[t * K + k] : half(0.0f);
        }

        // Weights: 32x64 halves, stored K-major so an 8x8 load is already the
        // [k, n] fragment the matrix unit wants.
        for (uint c = 0; c < 2u; ++c) {
            const uint kk = (w_chunk_base + c * 2u) * 8u;
            for (uint p = 0; p < 8u; ++p) {
                const uint k = k0 + kk + p;
                const bool live = (w_n < N) && (k < K);
                Bs[(kk + p) * kVisionQMMTileN + w_n_local] =
                    live ? half(float(w_row[k])) : half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 a[4];
        simdgroup_half8x8 b[4];
        for (uint kk = 0; kk < kVisionQMMTileK; kk += 8u) {
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_load(a[i],
                               As + (sg_m + i * 8u) * kVisionQMMTileK + kk,
                               kVisionQMMTileK);
            }
            for (uint j = 0; j < 4u; ++j) {
                simdgroup_load(b[j],
                               Bs + kk * kVisionQMMTileN + sg_n + j * 8u,
                               kVisionQMMTileN);
            }
            for (uint i = 0; i < 4u; ++i) {
                for (uint j = 0; j < 4u; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 4u; ++i) {
        for (uint j = 0; j < 4u; ++j) {
            simdgroup_store(acc[i][j],
                            stage + (sg_m + i * 8u) * kVisionQMMTileN + sg_n + j * 8u,
                            kVisionQMMTileN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < 32u; ++i) {
        const uint idx = i * kVisionQMMThreads + lid;
        const uint t = tBase + idx / kVisionQMMTileN;
        const uint n = nBase + idx % kVisionQMMTileN;
        if (t < T && n < N) {
            Y[t * N + n] = half(stage[idx]);
        }
    }
}

/// `out = 2 * (x - 0.5)`, the [0,1] -> [-1,1] rescale the patch embedder does
/// before its projection. Upstream calls no normalization on the pixels
/// (`do_normalize: false`); this is the whole of it.
kernel void vision_patch_prescale_block(
    device const half* x     [[buffer(0)]],
    device half*       out   [[buffer(1)]],
    constant uint&     count [[buffer(2)]],
    uint               gid   [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    out[gid] = half(2.0f * (float(x[gid]) - 0.5f));
}

/// Adds the two position embeddings of patch `t` to its projected row.
///
/// Upstream builds a one-hot matrix and multiplies it by the table; with no
/// padding that is exactly `table[0, x] + table[1, y]`, so this is two table
/// reads. Positions are not stored: patches are emitted row-major with no
/// padding, so `x = t % pw` and `y = t / pw`.
kernel void vision_patch_pos_add_block(
    device half*         h        [[buffer(0)]],
    device const bfloat* table    [[buffer(1)]],
    constant uint&       P        [[buffer(2)]],
    constant uint&       D        [[buffer(3)]],
    constant uint&       pw       [[buffer(4)]],
    constant uint&       tableLen [[buffer(5)]],
    uint2                gid      [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (t >= P || d >= D) return;

    const uint x = t % pw;
    const uint y = t / pw;
    // A patch grid wider or taller than the table would silently wrap; clamp so
    // it reads the last row instead, and let the host reject the shape.
    const uint xi = min(x, tableLen - 1u);
    const uint yi = min(y, tableLen - 1u);
    const float ex = float(table[xi * D + d]);
    const float ey = float(table[(tableLen + yi) * D + d]);
    h[t * D + d] = half(float(h[t * D + d]) + ex + ey);
}

/// Per-head Q/K RMSNorm, scale-less V RMSNorm, and the 2D RoPE, in one pass.
///
/// One simdgroup owns one (patch, head) triple of head rows. Lane `l` holds
/// elements `l, l + 32, l + 64` of a row, which is three elements at head
/// dim 72 with the last one live on lanes 0..7 only; the masked lanes
/// contribute zero to the `simd_sum` and store nothing.
///
/// The RoPE is the tower's one structural difference from the text path. The
/// head dimension splits in half: the first 36 channels rotate by the patch's x
/// position, the second 36 by its y position, and both halves use the same
/// frequencies (`inv_freq[j] = theta^(-2j/36)`, j = 0..17) — upstream computes
/// them from `head_dim / 2` per dimension rather than splitting one table. Each
/// half is a NeoX rotation of pairs (i, i + 18), so the partner element lives
/// in another lane and the normed row goes through threadgroup memory first.
kernel void vision_qk_norm_rope2d_block(
    device half*         q         [[buffer(0)]],
    device half*         k         [[buffer(1)]],
    device half*         v         [[buffer(2)]],
    device const bfloat* q_weight  [[buffer(3)]],
    device const bfloat* k_weight  [[buffer(4)]],
    constant uint&       P         [[buffer(5)]],
    constant uint&       head_dim  [[buffer(6)]],
    constant uint&       num_heads [[buffer(7)]],
    constant uint&       pw        [[buffer(8)]],
    constant float&      theta     [[buffer(9)]],
    constant float&      eps       [[buffer(10)]],
    uint3                tg        [[threadgroup_position_in_grid]],
    uint                 lane      [[thread_index_in_simdgroup]]
) {
    const uint h = tg.x;
    const uint t = tg.y;
    if (t >= P || h >= num_heads || head_dim > kVisionMaxHeadDim) return;

    threadgroup float row[kVisionMaxHeadDim];

    const uint stride = num_heads * head_dim;
    const uint base = t * stride + h * head_dim;
    const uint half_dim = head_dim / 2u;   // 36: channels per spatial dimension
    const uint pairs = half_dim / 2u;      // 18: NeoX pairs within a half
    const float pos_x = float(t % pw);
    const float pos_y = float(t / pw);

    // Q and K: scaled RMSNorm then RoPE. V: scale-less RMSNorm, no RoPE.
    for (uint which = 0u; which < 3u; ++which) {
        device half* data = which == 0u ? q : (which == 1u ? k : v);
        device const bfloat* weight = which == 0u ? q_weight : k_weight;
        const bool scaled = which < 2u;
        const bool rope = which < 2u;

        float acc = 0.0f;
        for (uint i = 0; i < 3u; ++i) {
            const uint idx = lane + i * 32u;
            const float value = idx < head_dim ? float(data[base + idx]) : 0.0f;
            acc = fma(value, value, acc);
        }
        const float inv = rsqrt(simd_sum(acc) / float(head_dim) + eps);

        for (uint i = 0; i < 3u; ++i) {
            const uint idx = lane + i * 32u;
            if (idx >= head_dim) continue;
            const float scale = scaled ? float(weight[idx]) : 1.0f;
            row[idx] = float(data[base + idx]) * inv * scale;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (rope) {
            for (uint p = lane; p < 2u * pairs; p += 32u) {
                const uint dim = p / pairs;          // 0 = x, 1 = y
                const uint j = p % pairs;
                const uint i0 = dim * half_dim + j;
                const uint i1 = i0 + pairs;
                const float freq = pow(theta, -float(2u * j) / float(half_dim));
                const float angle = (dim == 0u ? pos_x : pos_y) * freq;
                const float c = cos(angle);
                const float s = sin(angle);
                const float x0 = row[i0];
                const float x1 = row[i1];
                row[i0] = x0 * c - x1 * s;
                row[i1] = x1 * c + x0 * s;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint i = 0; i < 3u; ++i) {
            const uint idx = lane + i * 32u;
            if (idx >= head_dim) continue;
            data[base + idx] = half(row[idx]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct VisionAttentionParams {
    uint tokenCount;
    uint headDim;
    uint numHeads;
    uint qTokenStrideElements;
    uint kvTokenStrideElements;
    uint oTokenStrideElements;
    float scale;
};

/// Full (non-causal) attention over one image's patches.
///
/// Shaped after `attention_prefill_causal_qblock_impl`: one simdgroup owns
/// `kQBlock` consecutive queries of one head, each key row is read into
/// registers once and reused across them, and the per-key reduction is a
/// `simd_sum` with no threadgroup barrier. The differences are that every query
/// sees every key (`is_causal = False` upstream, and the whole image is one
/// bidirectional group), the scale is 1.0 rather than 1/sqrt(head_dim), and the
/// head dimension is 72 — not a multiple of 32, so unlike the text kernels the
/// per-lane slice is masked rather than exact. Lanes 8..31 own two elements
/// instead of three; the masked slot contributes zero to the dot product and is
/// not stored.
///
/// `kQBlock = 8` at three elements per lane costs 48 registers of `q_reg` plus
/// `acc`, in the same envelope as the d256 text kernel's 4x8, and it cuts the
/// K/V re-reads over a 2520-key image by 8.
template <uint kElemsPerLane, uint kQBlock>
static inline void vision_attention_full_qblock_impl(
    device const half* Q,
    device const half* K,
    device const half* V,
    device half* O,
    constant VisionAttentionParams& p,
    uint3 tg,
    uint lane,
    uint simd_group,
    uint simdgroups
) {
    const uint qh = tg.y;
    if (qh >= p.numHeads || p.headDim > kElemsPerLane * 32u) return;

    const uint queries_per_group = kQBlock * simdgroups;
    const uint q_first = tg.x * queries_per_group + simd_group * kQBlock;
    if (q_first >= p.tokenCount) return;
    // Uniform across the simdgroup, so the branches below keep it converged and
    // `simd_sum` stays well defined.
    const uint q_count = min(kQBlock, p.tokenCount - q_first);

    const uint head_offset = qh * p.headDim;

    float q_reg[kQBlock][kElemsPerLane];
    float acc[kQBlock][kElemsPerLane];
    float row_max[kQBlock];
    float row_sum[kQBlock];

    for (uint j = 0u; j < kQBlock; ++j) {
        const bool active = j < q_count;
        const uint t = active ? (q_first + j) : q_first;
        device const half* q_row = Q + t * p.qTokenStrideElements + head_offset;
        for (uint i = 0u; i < kElemsPerLane; ++i) {
            const uint idx = lane + i * 32u;
            q_reg[j][i] = (active && idx < p.headDim) ? float(q_row[idx]) : 0.0f;
            acc[j][i] = 0.0f;
        }
        row_max[j] = -INFINITY;
        row_sum[j] = 0.0f;
    }

    for (uint key = 0u; key < p.tokenCount; ++key) {
        device const half* k_row = K + key * p.kvTokenStrideElements + head_offset;
        device const half* v_row = V + key * p.kvTokenStrideElements + head_offset;

        float k_reg[kElemsPerLane];
        float v_reg[kElemsPerLane];
        for (uint i = 0u; i < kElemsPerLane; ++i) {
            const uint idx = lane + i * 32u;
            const bool live = idx < p.headDim;
            k_reg[i] = live ? float(k_row[idx]) : 0.0f;
            v_reg[i] = live ? float(v_row[idx]) : 0.0f;
        }

        for (uint j = 0u; j < kQBlock; ++j) {
            float partial = 0.0f;
            for (uint i = 0u; i < kElemsPerLane; ++i) {
                partial = fma(q_reg[j][i], k_reg[i], partial);
            }
            const float score = simd_sum(partial) * p.scale;

            const float new_max = max(row_max[j], score);
            const float old_scale = row_sum[j] > 0.0f ? fast::exp(row_max[j] - new_max) : 0.0f;
            const float new_scale = fast::exp(score - new_max);
            for (uint i = 0u; i < kElemsPerLane; ++i) {
                acc[j][i] = fma(new_scale, v_reg[i], acc[j][i] * old_scale);
            }
            row_sum[j] = row_sum[j] * old_scale + new_scale;
            row_max[j] = new_max;
        }
    }

    for (uint j = 0u; j < kQBlock; ++j) {
        if (j >= q_count) {
            continue;
        }
        device half* out_row = O + (q_first + j) * p.oTokenStrideElements + head_offset;
        const float inv = row_sum[j] > 0.0f ? 1.0f / row_sum[j] : 0.0f;
        for (uint i = 0u; i < kElemsPerLane; ++i) {
            const uint idx = lane + i * 32u;
            if (idx < p.headDim) {
                out_row[idx] = half(acc[j][i] * inv);
            }
        }
    }
}

/// The same attention with the simdgroup split into four 8-lane segments.
///
/// The kernel above inherits the text path's layout: all 32 lanes hold one
/// query's head row and every score costs a full `simd_sum`, which is five
/// shuffles. At head dim 256 that buys eight FMAs of useful work per lane; at
/// the tower's 72 it buys three, and the measured result was 254 GFLOP/s — a
/// twelfth of what the same device gives the BF16 QMM, and 78% of the tower's
/// runtime (`RESULTS_VISION.md`).
///
/// 72 is 8 x 9, so eight lanes can hold a whole head row with nine elements
/// each and the reduction becomes three `simd_shuffle_xor` steps within the
/// segment rather than five across the simdgroup. Four segments work on four
/// different queries against the same key row, so per key a lane now does 18
/// FMAs and 3 shuffles for four queries where before it did 48 FMAs, 40
/// shuffles and 8 redundant exponentials for eight. Lane `l` of a segment owns
/// elements `l, l + 8, ...`, so the eight lanes of a segment still cover 72
/// contiguous halfs per step and the loads stay coalesced.
template <uint kElemsPerLane, uint kQPerSegment>
static inline void vision_attention_full_seg_impl(
    device const half* Q,
    device const half* K,
    device const half* V,
    device half* O,
    constant VisionAttentionParams& p,
    uint3 tg,
    uint lane,
    uint simd_group,
    uint simdgroups
) {
    constexpr uint kSegments = 4u;
    constexpr uint kLanesPerSegment = 8u;
    const uint head_dim = kElemsPerLane * kLanesPerSegment;
    const uint qh = tg.y;
    if (qh >= p.numHeads || p.headDim != head_dim) return;

    const uint segment = lane / kLanesPerSegment;
    const uint slot = lane % kLanesPerSegment;

    const uint queries_per_simdgroup = kSegments * kQPerSegment;
    const uint queries_per_group = queries_per_simdgroup * simdgroups;
    const uint q_first = tg.x * queries_per_group + simd_group * queries_per_simdgroup
        + segment * kQPerSegment;
    // Not `q_first >= p.tokenCount`: the segments of one simdgroup straddle the
    // tail, and every lane has to reach the shuffles below.
    if (tg.x * queries_per_group >= p.tokenCount) return;

    const uint head_offset = qh * head_dim;

    float q_reg[kQPerSegment][kElemsPerLane];
    float acc[kQPerSegment][kElemsPerLane];
    float row_max[kQPerSegment];
    float row_sum[kQPerSegment];

    for (uint j = 0u; j < kQPerSegment; ++j) {
        const uint t = q_first + j;
        const bool active = t < p.tokenCount;
        device const half* q_row = Q + (active ? t : 0u) * p.qTokenStrideElements
            + head_offset;
        for (uint i = 0u; i < kElemsPerLane; ++i) {
            q_reg[j][i] = active ? float(q_row[slot + i * kLanesPerSegment]) : 0.0f;
            acc[j][i] = 0.0f;
        }
        row_max[j] = -INFINITY;
        row_sum[j] = 0.0f;
    }

    for (uint key = 0u; key < p.tokenCount; ++key) {
        device const half* k_row = K + key * p.kvTokenStrideElements + head_offset;
        device const half* v_row = V + key * p.kvTokenStrideElements + head_offset;

        float k_reg[kElemsPerLane];
        float v_reg[kElemsPerLane];
        for (uint i = 0u; i < kElemsPerLane; ++i) {
            k_reg[i] = float(k_row[slot + i * kLanesPerSegment]);
            v_reg[i] = float(v_row[slot + i * kLanesPerSegment]);
        }

        for (uint j = 0u; j < kQPerSegment; ++j) {
            float partial = 0.0f;
            for (uint i = 0u; i < kElemsPerLane; ++i) {
                partial = fma(q_reg[j][i], k_reg[i], partial);
            }
            // Reduce within the 8-lane segment only.
            partial += simd_shuffle_xor(partial, 1u);
            partial += simd_shuffle_xor(partial, 2u);
            partial += simd_shuffle_xor(partial, 4u);
            const float score = partial * p.scale;

            const float new_max = max(row_max[j], score);
            const float old_scale = row_sum[j] > 0.0f ? fast::exp(row_max[j] - new_max) : 0.0f;
            const float new_scale = fast::exp(score - new_max);
            for (uint i = 0u; i < kElemsPerLane; ++i) {
                acc[j][i] = fma(new_scale, v_reg[i], acc[j][i] * old_scale);
            }
            row_sum[j] = row_sum[j] * old_scale + new_scale;
            row_max[j] = new_max;
        }
    }

    for (uint j = 0u; j < kQPerSegment; ++j) {
        const uint t = q_first + j;
        if (t >= p.tokenCount) {
            continue;
        }
        device half* out_row = O + t * p.oTokenStrideElements + head_offset;
        const float inv = row_sum[j] > 0.0f ? 1.0f / row_sum[j] : 0.0f;
        for (uint i = 0u; i < kElemsPerLane; ++i) {
            out_row[slot + i * kLanesPerSegment] = half(acc[j][i] * inv);
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void vision_attention_full_seg_d72(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant VisionAttentionParams& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]
) {
    vision_attention_full_seg_impl<9u, 2u>(
        Q, K, V, O, p, tg, lane, simd_group, simdgroups);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void vision_attention_full_qblock_d96(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant VisionAttentionParams& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]
) {
    vision_attention_full_qblock_impl<3u, 8u>(
        Q, K, V, O, p, tg, lane, simd_group, simdgroups);
}

/// `out = gelu_tanh(gate) * up`, the tower MLP's activation. Same expression as
/// the text path's `prefill_gelu_pytorch_tanh`, on separate gate/up buffers
/// rather than one packed activation block.
kernel void vision_mlp_act_block(
    device const half* gate  [[buffer(0)]],
    device const half* up    [[buffer(1)]],
    device half*       out   [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint               gid   [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    out[gid] = half(vision_gelu_pytorch_tanh(float(gate[gid])) * float(up[gid]));
}

/// 3x3 average pool over the patch grid, then `* sqrt(hidden)`, then the
/// tower's output standardization `(x - std_bias) * std_scale`.
///
/// The pooled grid is `(pw/3) x (ph/3)` in row-major order, and cell (cx, cy)
/// averages the nine patches at `(3cx + dx, 3cy + dy)` — with no padding the
/// one-hot matmul upstream is exactly this. The mean accumulates in float, as
/// it does upstream (`hidden_states.float()`), and the sqrt(hidden) scaling is
/// applied after the mean rather than folded into it, so the rounding sequence
/// matches the reference.
kernel void vision_pool_std_block(
    device const half*   h            [[buffer(0)]],
    device half*         out          [[buffer(1)]],
    device const bfloat* std_scale    [[buffer(2)]],
    device const bfloat* std_bias     [[buffer(3)]],
    constant uint&       D            [[buffer(4)]],
    constant uint&       pw           [[buffer(5)]],
    constant uint&       pooled_w     [[buffer(6)]],
    constant uint&       pooled_h     [[buffer(7)]],
    constant uint&       kernel_size  [[buffer(8)]],
    constant float&      root_hidden  [[buffer(9)]],
    constant uint&       standardize  [[buffer(10)]],
    uint2                gid          [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint cell = gid.y;
    if (d >= D || cell >= pooled_w * pooled_h) return;

    const uint cx = cell % pooled_w;
    const uint cy = cell / pooled_w;

    float acc = 0.0f;
    for (uint dy = 0u; dy < kernel_size; ++dy) {
        const uint y = cy * kernel_size + dy;
        for (uint dx = 0u; dx < kernel_size; ++dx) {
            const uint x = cx * kernel_size + dx;
            acc += float(h[(y * pw + x) * D + d]);
        }
    }
    float value = (acc / float(kernel_size * kernel_size)) * root_hidden;
    if (standardize != 0u) {
        value = (value - float(std_bias[d])) * float(std_scale[d]);
    }
    out[cell * D + d] = half(value);
}
