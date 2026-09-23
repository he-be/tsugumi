#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

#ifndef MOEPACK_AFFINE_GROUP_SIZE
#define MOEPACK_AFFINE_GROUP_SIZE 64
#endif
constant constexpr uint kW4A8GroupSize = MOEPACK_AFFINE_GROUP_SIZE;

// Affine zero point -- see the note in dequant_int4.metal. `MOEPACK_AFFINE_SYMMETRIC`
// is a whole-model compile-time constant (`MetalContext.affineScheme`); when it
// is set the bias arrays do not exist and the bindings alias the scales.
#ifndef MOEPACK_AFFINE_SYMMETRIC
#define MOEPACK_AFFINE_SYMMETRIC 0
#endif

static inline float w4a8_int4_bias(device const bfloat* biases, uint index,
                                     float scale) {
#if MOEPACK_AFFINE_SYMMETRIC
    return -8.0f * scale;
#else
    return float(biases[index]);
#endif
}

constant constexpr int kMPPAffineTileM = 64;
constant constexpr int kMPPAffineTileN = 32;
constant constexpr int kMPPAffineTileK = 64;

kernel void mpp_prefill_affine_threadgroup_f16(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device half* activations            [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint3 lid3                          [[thread_position_in_threadgroup]],
    uint3 threads3                      [[threads_per_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffineTileM, kMPPAffineTileN, kMPPAffineTileK,
        false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half weightTile[kMPPAffineTileN * kMPPAffineTileK];
    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineTileK, kMPPAffineTileN),
        array<int32_t, 2>({1, kMPPAffineTileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineTileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(
        0,
        int32_t(tgid.y) * kMPPAffineTileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    // The tile walks K in fixed 64-element steps; the affine group may be 32
    // or 64, so the scale/bias index comes from the global K position rather
    // than the tile counter (a group-32 tile spans two scale/bias pairs).
    const uint rowBytes = K / 2u;
    const uint groupsPerRow = K / kW4A8GroupSize;
    const uint tilesPerRow = K / uint(kMPPAffineTileK);
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint kTile = 0; kTile < tilesPerRow; ++kTile) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kMPPAffineTileN * kMPPAffineTileK);
             linear += threads) {
            const uint localN = linear / uint(kMPPAffineTileK);
            const uint localK = linear % uint(kMPPAffineTileK);
            const uint globalN = tgid.x * uint(kMPPAffineTileN) + localN;
            if (globalN < N) {
                const uint globalK = kTile * uint(kMPPAffineTileK) + localK;
                const uint8_t packed = packedWeights[globalN * rowBytes + (globalK >> 1)];
                const uint q = (globalK & 1u) == 0u
                    ? uint(packed & 0x0fu)
                    : uint(packed >> 4);
                const uint group = globalN * groupsPerRow + globalK / kW4A8GroupSize;
                const float scale = float(scales[group]);
                const float bias = w4a8_int4_bias(biases, group, scale);
                weightTile[linear] = half(fma(float(q), scale, bias));
            } else {
                weightTile[linear] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_half_tensor groupA(
            activations + kTile * uint(kMPPAffineTileK),
            dextents<int32_t, 2>(kMPPAffineTileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(
            0,
            int32_t(tgid.y) * kMPPAffineTileM);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kMPPAffineTileN) + uint(position[0]);
        const uint globalM = tgid.y * uint(kMPPAffineTileM) + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[globalM * N + globalN] = half(accumulator[element]);
        }
    }
}

// The dense twin of `prefill_moe_gemm_int4_mpp` (prefill.metal,
// docs/m6-prefill/12-G3-MOE-MATMUL2D.md): a 64x64 output tile, A and the
// dequantized B both staged in threadgroup memory as K-contiguous rows, and
// one `multiply_accumulate` destination across the whole K loop — no
// per-step zeroing and no hand-written accumulate. The weight values staged
// are the same FP16 roundings the kernel above and the simdgroup QMM stage.
//
// A is loaded as half4, so K must be a multiple of 64 (the K step) and the
// activation base 8-byte aligned; the Swift side checks both.
constant constexpr int kMPPTile64M = 64;
constant constexpr int kMPPTile64N = 64;
constant constexpr int kMPPTile64K = 64;
constant constexpr uint kMPPTile64Threads = 128;

kernel void mpp_prefill_affine_tile64_f16(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device const half* activations      [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint3 lid3                          [[thread_position_in_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPTile64M, kMPPTile64N, kMPPTile64K,
        false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half As[kMPPTile64M * kMPPTile64K];
    threadgroup half Bs[kMPPTile64N * kMPPTile64K];
    threadgroup_half_tensor tileA(
        As,
        dextents<int32_t, 2>(kMPPTile64K, kMPPTile64M),
        array<int32_t, 2>({1, kMPPTile64K}));
    threadgroup_half_tensor tileB(
        Bs,
        dextents<int32_t, 2>(kMPPTile64K, kMPPTile64N),
        array<int32_t, 2>({1, kMPPTile64K}));
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(tileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint lid = lid3.x;
    const uint mBase = tgid.y * uint(kMPPTile64M);
    const uint nBase = tgid.x * uint(kMPPTile64N);
    // Weights: each thread owns one output column and one 32-wide half of the
    // K step — 16 packed bytes inside a single affine group (32 or 64).
    const uint groupsPerRow = K / kW4A8GroupSize;
    const uint rowBytes = K / 2u;
    const uint wNLocal = lid & uint(kMPPTile64N - 1);
    const uint wKHalf = (lid / uint(kMPPTile64N)) * 32u;
    const uint wN = nBase + wNLocal;
    device const uint8_t* wRow = packedWeights + wN * rowBytes;
    threadgroup half* bDst = Bs + wNLocal * uint(kMPPTile64K) + wKHalf;

    for (uint k0 = 0; k0 < K; k0 += uint(kMPPTile64K)) {
        // Activations: 64x64 halves as 1024 half4, 8 per thread.
        for (uint i = 0; i < 8u; ++i) {
            const uint unit = i * kMPPTile64Threads + lid;
            const uint m = unit / 16u;
            const uint kk = (unit % 16u) * 4u;
            const half4 v = (mBase + m < M)
                ? *((device const half4*)(activations + (mBase + m) * K + k0 + kk))
                : half4(0.0h);
            *((threadgroup half4*)(As + m * uint(kMPPTile64K) + kk)) = v;
        }

        if (wN < N) {
            const uint kb = k0 + wKHalf;
            const uint group = wN * groupsPerRow + kb / kW4A8GroupSize;
            const float scale = float(scales[group]);
            const float bias = w4a8_int4_bias(biases, group, scale);
            // Packed sub-tensor offsets are only 2-byte aligned.
            device const ushort* wPtr = (device const ushort*)(wRow + (kb >> 1));
            for (uint q = 0; q < 8u; ++q) {
                const uint packed = uint(wPtr[q]);
                half4 v;
                v.x = half(fma(float(packed & 0x0Fu), scale, bias));
                v.y = half(fma(float((packed >> 4) & 0x0Fu), scale, bias));
                v.z = half(fma(float((packed >> 8) & 0x0Fu), scale, bias));
                v.w = half(fma(float((packed >> 12) & 0x0Fu), scale, bias));
                *((threadgroup half4*)(bDst + q * 4u)) = v;
            }
        } else {
            for (uint q = 0; q < 8u; ++q) {
                *((threadgroup half4*)(bDst + q * 4u)) = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        operation.run(tileA, tileB, accumulator);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = nBase + uint(position[0]);
        const uint globalM = mBase + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[globalM * N + globalN] = half(accumulator[element]);
        }
    }
}

#endif
