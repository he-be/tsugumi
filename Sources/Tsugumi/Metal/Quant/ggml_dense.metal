#include <metal_stdlib>
using namespace metal;

// Dense GEMV straight off GGML tensors: y[r] = W[r] . x for r < M, W [M, N]
// row-major as stored in the GGUF. Float32 in, float32 out -- the
// Qwen3.8-Flash-Next verification runner keeps activations in FP32 so it can
// be compared against the CPU reference without a half-precision floor.
//
// Lane geometry, shared by the three kernels (2 SIMD groups x 32 lanes per
// threadgroup, 4 rows per group):
//   Q8_0      lane l reads blocks l, l+32, l+64, ... (34 B: f16 d + 32 int8)
//   F16/F32   lane l reads elements l, l+32, l+64, ...
// `simd_sum` over the group's lanes assembles each row; lane 0 writes it.

constant constexpr uint kGgmlDenseGroupsPerTG = 2;
constant constexpr uint kGgmlDenseRowsPerGroup = 4;

struct block_q8_0 {
    half d;
    int8_t qs[32];
};

kernel void ggml_q8_0_gemv(
    device const block_q8_0* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint row0 = (tg * kGgmlDenseGroupsPerTG + sgitg) * kGgmlDenseRowsPerGroup;
    const uint nb = N / 32;
    float sum[kGgmlDenseRowsPerGroup] = {0.f};
    if (row0 < M) {
        const uint rows = min(kGgmlDenseRowsPerGroup, M - row0);
        for (uint ib = tiisg; ib < nb; ib += 32) {
            device const float* xb = x + ib * 32;
            for (uint row = 0; row < rows; ++row) {
                device const block_q8_0* b = W + (row0 + row) * nb + ib;
                float facc = 0.f;
                for (uint j = 0; j < 32; ++j) facc += float(b->qs[j]) * xb[j];
                sum[row] += float(b->d) * facc;
            }
        }
    }
    float4 reduced;
    for (uint row = 0; row < kGgmlDenseRowsPerGroup; ++row) reduced[row] = simd_sum(sum[row]);
    if (row0 < M && tiisg == 0) {
        for (uint row = 0; row < kGgmlDenseRowsPerGroup && row0 + row < M; ++row) {
            y[row0 + row] = reduced[row];
        }
    }
}

#define GGML_FLOAT_GEMV(NAME, WTYPE)                                                     \
kernel void NAME(                                                                        \
    device const WTYPE* W [[buffer(0)]],                                                 \
    device const float* x [[buffer(1)]],                                                 \
    device float* y [[buffer(2)]],                                                       \
    constant uint& M [[buffer(3)]],                                                      \
    constant uint& N [[buffer(4)]],                                                      \
    uint tg [[threadgroup_position_in_grid]],                                            \
    ushort tiisg [[thread_index_in_simdgroup]],                                          \
    ushort sgitg [[simdgroup_index_in_threadgroup]]                                      \
) {                                                                                      \
    const uint row0 = (tg * kGgmlDenseGroupsPerTG + sgitg) * kGgmlDenseRowsPerGroup;     \
    float sum[kGgmlDenseRowsPerGroup] = {0.f};                                           \
    if (row0 < M) {                                                                      \
        const uint rows = min(kGgmlDenseRowsPerGroup, M - row0);                         \
        for (uint row = 0; row < rows; ++row) {                                          \
            device const WTYPE* w = W + (row0 + row) * N;                                \
            float acc = 0.f;                                                             \
            for (uint i = tiisg; i < N; i += 32) acc += float(w[i]) * x[i];              \
            sum[row] = acc;                                                              \
        }                                                                                \
    }                                                                                    \
    float4 reduced;                                                                      \
    for (uint row = 0; row < kGgmlDenseRowsPerGroup; ++row) reduced[row] = simd_sum(sum[row]); \
    if (row0 < M && tiisg == 0) {                                                        \
        for (uint row = 0; row < kGgmlDenseRowsPerGroup && row0 + row < M; ++row) {      \
            y[row0 + row] = reduced[row];                                                \
        }                                                                                \
    }                                                                                    \
}

GGML_FLOAT_GEMV(ggml_f16_gemv, half)
GGML_FLOAT_GEMV(ggml_f32_gemv, float)
