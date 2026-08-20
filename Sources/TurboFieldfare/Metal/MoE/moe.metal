#include <metal_stdlib>
using namespace metal;

#ifndef TURBO_AFFINE_GROUP_SIZE
#define TURBO_AFFINE_GROUP_SIZE 64
#endif
constant constexpr uint kMoEGroupSize = TURBO_AFFINE_GROUP_SIZE;

// Affine zero point -- see the note in dequant_int4.metal. `TURBO_AFFINE_SYMMETRIC`
// is a whole-model compile-time constant (`MetalContext.affineScheme`); when it
// is set the bias arrays do not exist and the bindings alias the scales.
#ifndef TURBO_AFFINE_SYMMETRIC
#define TURBO_AFFINE_SYMMETRIC 0
#endif

static inline float moe_int4_bias(device const bfloat* biases, uint index,
                                     float scale) {
#if TURBO_AFFINE_SYMMETRIC
    return -8.0f * scale;
#else
    return float(biases[index]);
#endif
}

/// `acc + scale * dot + bias * sum` for one group, where `dot` is the group's
/// integer-weighted activation sum and `sum` its plain activation sum.
///
/// The two schemes run the *same* two FMAs in the same order, and `sym`'s
/// `bias` is `-8 * scale` computed in FP32. That product is exact -- BF16
/// stores `-8 * scale` without rounding, so converting it back gives the same
/// float the affine model loaded -- which makes the whole pipeline **bit
/// identical** to the affine one on a checkpoint that satisfies the identity.
/// Folding the zero point into `dot` instead would save nothing (`gate/up` is
/// on the bandwidth floor, so the FMA is free) and would cost that property.
static inline float moe_int4_accumulate(float acc, float scale, float bias,
                                           float dot, float sum) {
    acc = fma(scale, dot, acc);
    return fma(bias, sum, acc);
}
// Vectorized INT4 block geometry — see the note in dequant_int4.metal.
// A block is a fixed 128 bytes (32 lanes x 4 bytes); the group size decides how
// many affine groups that spans and how many lanes cover one group.
constant constexpr uint kMoEGroupsPerBlock = 256u / kMoEGroupSize;
constant constexpr uint kMoELanesPerGroup  = kMoEGroupSize / 8u;
constant constexpr uint kMoETailLanes      = kMoEGroupSize / 2u;

constant constexpr uint kMaxStreamedExperts = 8;
constant constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kGeluCubicCoeff = 0.044715f;

constant uint FC_ROUTER_NUM_EXPERTS [[function_constant(40)]];
constant uint FC_ROUTER_D [[function_constant(41)]];
constant uint FC_ROUTER_TOP_K [[function_constant(42)]];
constant bool FC_ROUTER_USE_FC [[function_constant(43)]];

constant uint FC_MOE_D [[function_constant(0)]];
constant uint FC_MOE_F [[function_constant(1)]];
constant uint FC_MOE_TOP_K [[function_constant(2)]];
constant bool FC_MOE_USE_FC [[function_constant(3)]];

static inline uint router_fc_num_experts(constant uint& num_experts) {
    return (is_function_constant_defined(FC_ROUTER_USE_FC) &&
            FC_ROUTER_USE_FC &&
            is_function_constant_defined(FC_ROUTER_NUM_EXPERTS))
        ? FC_ROUTER_NUM_EXPERTS
        : num_experts;
}

static inline uint router_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_ROUTER_USE_FC) &&
            FC_ROUTER_USE_FC &&
            is_function_constant_defined(FC_ROUTER_D))
        ? FC_ROUTER_D
        : D;
}

static inline uint moe_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_D)) ? FC_MOE_D : D;
}

static inline uint moe_fc_f(constant uint& F) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_F)) ? FC_MOE_F : F;
}

static inline uint moe_fc_top_k(constant uint& top_k) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_TOP_K)) ? FC_MOE_TOP_K : top_k;
}

static inline float gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = kGeluSqrt2OverPi * (x + kGeluCubicCoeff * x3);
    // Clamping avoids Metal tanh producing NaN at large magnitudes while being
    // equivalent to the saturated result at FP32 precision.
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

struct ExpertOffsets {
    uint gate_W_off;
    uint gate_s_off;
    uint gate_b_off;
    uint up_W_off;
    uint up_s_off;
    uint up_b_off;
    uint down_W_off;
    uint down_s_off;
    uint down_b_off;
};

struct RoutedBlobs {
    device const uint8_t* blob[kMaxStreamedExperts];
};

static inline void router_gemv_gemma4_body(
    device const uint8_t* W,
    device const bfloat* scales,
    device const bfloat* biases,
    device const half* hidden,
    device const bfloat* effective_scale,
    device float* out_logits,
    constant uint& num_experts,
    constant uint& D,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint NE = router_fc_num_experts(num_experts);
    const uint DD = router_fc_d(D);
    const uint e = tg_idx * rows_per_tg + sg_idx;
    if (e >= NE) return;

    const uint n_groups = DD / kMoEGroupSize;
    device const uint8_t* W_row = W + uint(e) * DD;
    device const bfloat* s_row = scales + uint(e) * n_groups;
    device const bfloat* b_row = biases + uint(e) * n_groups;

    // One step is a fixed 64 elements (32 lanes x 2 INT8 weights), so the group
    // size decides how many groups a step spans and how many lanes share one
    // scale. At group 64 this is one group per step and `g == st`, which is the
    // geometry this kernel was originally written for.
    const uint groups_per_step = 64u / kMoEGroupSize;
    const uint lanes_per_group = 32u / groups_per_step;
    const uint steps = DD / 64u;

    float acc = 0.0f;
    for (uint st = 0; st < steps; ++st) {
        const uint g = st * groups_per_step + lane / lanes_per_group;
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint idx = st * 64u + lane * 2u;
        const float q0 = float(uint(W_row[idx]));
        const float q1 = float(uint(W_row[idx + 1u]));
        const float x0 = float(hidden[idx]) * float(effective_scale[idx]);
        const float x1 = float(hidden[idx + 1u]) * float(effective_scale[idx + 1u]);
        acc = fma(s, q0 * x0 + q1 * x1, acc);
        acc = fma(b, x0 + x1, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) out_logits[e] = acc;
}

// BF16 router (QAT checkpoints leave `router.proj.weight` unquantized). There
// is no group structure, so this body is independent of the affine group size.
static inline void router_gemv_gemma4_bf16_body(
    device const bfloat* W,
    device const half* hidden,
    device const bfloat* effective_scale,
    device float* out_logits,
    constant uint& num_experts,
    constant uint& D,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint NE = router_fc_num_experts(num_experts);
    const uint DD = router_fc_d(D);
    const uint e = tg_idx * rows_per_tg + sg_idx;
    if (e >= NE) return;

    device const bfloat* W_row = W + uint(e) * DD;
    float acc = 0.0f;
    for (uint i = lane; i < DD; i += 32u) {
        acc = fma(float(W_row[i]),
                  float(hidden[i]) * float(effective_scale[i]),
                  acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) out_logits[e] = acc;
}

kernel void router_gemv_gemma4_r4(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* hidden [[buffer(3)]],
    device const bfloat* effective_scale [[buffer(4)]],
    device float* out_logits [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    router_gemv_gemma4_body(W, scales, biases, hidden, effective_scale,
                            out_logits, num_experts, D, 4, tg_idx, sg_idx, lane);
}

kernel void router_gemv_gemma4_bf16_r4(
    device const bfloat* W [[buffer(0)]],
    device const half* hidden [[buffer(1)]],
    device const bfloat* effective_scale [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    router_gemv_gemma4_bf16_body(W, hidden, effective_scale, out_logits,
                                 num_experts, D, 4, tg_idx, sg_idx, lane);
}

static inline void router_topk_select_k8_body(
    device const float* logits,
    device const bfloat* per_expert_scale,
    device uint* out_indices,
    device half* out_weights,
    constant uint& num_experts
) {
    const uint NE = router_fc_num_experts(num_experts);
    uint top_idx[8];
    float top_score[8];
    for (uint i = 0; i < 8; ++i) {
        top_idx[i] = 0u;
        top_score[i] = -INFINITY;
    }

    for (uint e = 0; e < NE; ++e) {
        const float s = logits[e];
        if (s <= top_score[7]) continue;
        uint pos = 8u;
        for (uint i = 0; i < 8; ++i) {
            if (s > top_score[i] || (s == top_score[i] && e < top_idx[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= 8u) continue;
        for (uint i = 7; i > pos; --i) {
            top_idx[i] = top_idx[i - 1];
            top_score[i] = top_score[i - 1];
        }
        top_idx[pos] = e;
        top_score[pos] = s;
    }

    const float max_s = top_score[0];
    float sum_exp = 0.0f;
    float exps[8];
    for (uint i = 0; i < 8; ++i) {
        const float ex = fast::exp(top_score[i] - max_s);
        exps[i] = ex;
        sum_exp += ex;
    }
    for (uint i = 0; i < 8; ++i) {
        const uint expert_idx = top_idx[i];
        const float weight = exps[i] / sum_exp;
        out_indices[i] = expert_idx;
        out_weights[i] = half(weight * float(per_expert_scale[expert_idx]));
    }
}

kernel void router_topk_select_k8(
    device const float* logits [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;
    router_topk_select_k8_body(logits, per_expert_scale, out_indices,
                               out_weights, num_experts);
}

// The same two kernels with the block's rows in the grid's second dimension.
//
// A speculative verify block used to dispatch the decode router once per row,
// which is 8 encoders a layer and 240 a block — nearly all of the `post` stage
// was the dispatches, not the 720 KB of router weights
// (docs/mtp/27-M7-RESULTS.md §5). Each row keeps its own threadgroup and its
// own lane layout, so a row's dot product is summed in exactly the order the
// per-row dispatch summed it and the expert choice cannot move.
kernel void router_gemv_gemma4_bf16_rows(
    device const bfloat* W [[buffer(0)]],
    device const half* hidden [[buffer(1)]],
    device const bfloat* effective_scale [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant uint& hidden_stride_elements [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint NE = router_fc_num_experts(num_experts);
    router_gemv_gemma4_bf16_body(W,
                                 hidden + tgid.y * hidden_stride_elements,
                                 effective_scale,
                                 out_logits + tgid.y * NE,
                                 num_experts, D, 4, tgid.x, sg_idx, lane);
}

kernel void router_topk_select_k8_rows(
    device const float* logits [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    if (tid.x != 0 || tid.y != 0) return;
    const uint NE = router_fc_num_experts(num_experts);
    const uint top_k = 8u;
    router_topk_select_k8_body(logits + tgid.y * NE,
                               per_expert_scale,
                               out_indices + tgid.y * top_k,
                               out_weights + tgid.y * top_k,
                               num_experts);
}


// Each SIMD computes one affine INT4 row. Four adjacent groups are loaded as
// aligned 32-bit chunks; remaining groups use one byte per lane.
static inline float moe_int4_gemv_row_simd_dev_vec(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W + uint(row) * row_bytes;
    device const bfloat* s_row = S + uint(row) * n_groups;
    device const bfloat* b_row = B + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / kMoEGroupsPerBlock;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint w4 = *((device const uint*)(W_row + byte_base));
        const uint g = blk * kMoEGroupsPerBlock + lane / kMoELanesPerGroup;
        const float s = float(s_row[g]);
        const float b = moe_int4_bias(b_row, g, s);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 = w4 & 0xFFu;
        const uint b1 = (w4 >> 8) & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
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
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * kMoEGroupsPerBlock; g < n_groups; ++g) {
        // Only the first kMoETailLanes lanes hold a byte of this group.
        if (lane >= kMoETailLanes) break;
        const float s = float(s_row[g]);
        const float b = moe_int4_bias(b_row, g, s);
        const uint8_t byte = W_row[g * (kMoEGroupSize / 2) + lane];
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        acc = fma(s, dot, acc);
        acc = fma(b, x0 + x1, acc);
    }
    return simd_sum(acc);
}

// Gate and up rows share activation loads. Two 16-bit loads assemble each
// 4-byte weight chunk because packed sub-tensor offsets need only be 2-byte aligned.
static inline float2 moe_int4_gate_up_rows_simd_dev_vec_u16load(
    device const uint8_t* gateW,
    device const bfloat* gateS,
    device const bfloat* gateB,
    device const uint8_t* upW,
    device const bfloat* upS,
    device const bfloat* upB,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* gW_row = gateW + uint(row) * row_bytes;
    device const uint8_t* uW_row = upW + uint(row) * row_bytes;
    device const bfloat* gS_row = gateS + uint(row) * n_groups;
    device const bfloat* gB_row = gateB + uint(row) * n_groups;
    device const bfloat* uS_row = upS + uint(row) * n_groups;
    device const bfloat* uB_row = upB + uint(row) * n_groups;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    const uint full_blocks = n_groups / kMoEGroupsPerBlock;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* gp = (device const ushort*)(gW_row + byte_base);
        device const ushort* up = (device const ushort*)(uW_row + byte_base);
        const uint gw4 = uint(gp[0]) | (uint(gp[1]) << 16);
        const uint uw4 = uint(up[0]) | (uint(up[1]) << 16);
        const uint g = blk * kMoEGroupsPerBlock + lane / kMoELanesPerGroup;
        const float gs = float(gS_row[g]);
        const float gb = moe_int4_bias(gB_row, g, gs);
        const float us = float(uS_row[g]);
        const float ub = moe_int4_bias(uB_row, g, us);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;

        const uint gb0 = gw4 & 0xFFu;
        const uint gb1 = (gw4 >> 8) & 0xFFu;
        const uint gb2 = (gw4 >> 16) & 0xFFu;
        const uint gb3 = (gw4 >> 24) & 0xFFu;
        float g_dot = 0.0f;
        g_dot = fma(float(gb0 & 0x0Fu), e0, g_dot); g_dot = fma(float(gb0 >> 4), e1, g_dot);
        g_dot = fma(float(gb1 & 0x0Fu), e2, g_dot); g_dot = fma(float(gb1 >> 4), e3, g_dot);
        g_dot = fma(float(gb2 & 0x0Fu), e4, g_dot); g_dot = fma(float(gb2 >> 4), e5, g_dot);
        g_dot = fma(float(gb3 & 0x0Fu), e6, g_dot); g_dot = fma(float(gb3 >> 4), e7, g_dot);

        const uint ub0 = uw4 & 0xFFu;
        const uint ub1 = (uw4 >> 8) & 0xFFu;
        const uint ub2 = (uw4 >> 16) & 0xFFu;
        const uint ub3 = (uw4 >> 24) & 0xFFu;
        float u_dot = 0.0f;
        u_dot = fma(float(ub0 & 0x0Fu), e0, u_dot); u_dot = fma(float(ub0 >> 4), e1, u_dot);
        u_dot = fma(float(ub1 & 0x0Fu), e2, u_dot); u_dot = fma(float(ub1 >> 4), e3, u_dot);
        u_dot = fma(float(ub2 & 0x0Fu), e4, u_dot); u_dot = fma(float(ub2 >> 4), e5, u_dot);
        u_dot = fma(float(ub3 & 0x0Fu), e6, u_dot); u_dot = fma(float(ub3 >> 4), e7, u_dot);

        g_acc = moe_int4_accumulate(g_acc, gs, gb, g_dot, sum);
        u_acc = moe_int4_accumulate(u_acc, us, ub, u_dot, sum);
    }
    for (uint g = full_blocks * kMoEGroupsPerBlock; g < n_groups; ++g) {
        // Only the first kMoETailLanes lanes hold a byte of this group.
        if (lane >= kMoETailLanes) break;
        const float gs = float(gS_row[g]);
        const float gb = moe_int4_bias(gB_row, g, gs);
        const float us = float(uS_row[g]);
        const float ub = moe_int4_bias(uB_row, g, us);
        const uint8_t gbv = gW_row[g * (kMoEGroupSize / 2) + lane];
        const uint8_t ubv = uW_row[g * (kMoEGroupSize / 2) + lane];
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        const float sum = x0 + x1;
        float g_dot = fma(float(uint(gbv & 0x0Fu)), x0, 0.0f);
        g_dot = fma(float(uint(gbv >> 4)), x1, g_dot);
        float u_dot = fma(float(uint(ubv & 0x0Fu)), x0, 0.0f);
        u_dot = fma(float(uint(ubv >> 4)), x1, u_dot);
        g_acc = moe_int4_accumulate(g_acc, gs, gb, g_dot, sum);
        u_acc = moe_int4_accumulate(u_acc, us, ub, u_dot, sum);
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

static inline void moe_phase1_gate_up_act_u16load_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    uint D,
    uint F,
    uint top_k,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= top_k * F) return;
    const uint slot = rowg / F;
    const uint f = rowg % F;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        gW, gS, gB, uW, uS, uB, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(gelu_pytorch_tanh(gu.x) * gu.y);
}

static inline void moe_phase1_gate_up_act_subset_u16load_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    device const uint* active_slots,
    uint active_count,
    uint D,
    uint F,
    uint top_k,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= active_count * F) return;
    const uint active_idx = rowg / F;
    const uint slot = active_slots[active_idx];
    if (slot >= top_k) return;
    const uint f = rowg % F;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        gW, gS, gB, uW, uS, uB, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(gelu_pytorch_tanh(gu.x) * gu.y);
}

kernel void moe_phase1_gate_up_act_u16load(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    moe_phase1_gate_up_act_u16load_body(
        routed, routed_offsets, x, acts, moe_fc_d(D), moe_fc_f(F),
        moe_fc_top_k(top_k), rows_per_tg, tg_idx, sg_idx, lane);
}

kernel void moe_phase1_gate_up_act_subset_u16load(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* active_slots [[buffer(7)]],
    constant uint& active_count [[buffer(8)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    moe_phase1_gate_up_act_subset_u16load_body(
        routed, routed_offsets, x, acts, active_slots, active_count,
        moe_fc_d(D), moe_fc_f(F), moe_fc_top_k(top_k), rows_per_tg,
        tg_idx, sg_idx, lane);
}

kernel void moe_phase2_down_reduce_k8(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[8];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;

    device const uint8_t* base = routed.blob[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* dW = base + re.down_W_off;
    device const bfloat* dS = (device const bfloat*)(base + re.down_s_off);
    device const bfloat* dB = (device const bfloat*)(base + re.down_b_off);
    device const half* act_slot = acts + sg_idx * FF;

    const float value = moe_int4_gemv_row_simd_dev_vec(
        dW, dS, dB, act_slot, d, FF, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];
        y[d] = half(acc);
    }
}
