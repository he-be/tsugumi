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

// --- MLX's Neural Accelerator QMM, ported (docs/m6-prefill/31) ---
// From MLX 0.32.2 (MIT, Copyright (c) 2023-2024 Apple Inc.):
// mlx/backend/metal/kernels/quantized_nax.h (affine_qmm_t_nax / qmm_t_nax_tgp_impl
// at BM 64, BN 64, BK 64, WM 2, WN 2, and the group-32 QuantizedBlockLoader) and
// steel/gemm/nax.h (BaseNAXFrag::get_coord / load / store / mma, tile_matmad_nax).
// Kept from MLX: 2x2 simdgroups each owning a 32x32 block of C in registers;
// A read from device memory straight into 16x16 fragments; only the
// dequantized B staged in threadgroup memory (row stride BK + 8); matmul2d
// 16x32x16 on a single simdgroup with relaxed precision, one run per
// (16-row A fragment, 32-column B pair, 16-wide K step).
// Changed: Tsugumi's bf16 scales and scheme bias (w4a8_int4_bias), dequant in
// f32 as tile64 does, fp16 activations and output, plain M/N/K arguments.
constant constexpr uint kNAXTileM = 64;
constant constexpr uint kNAXTileN = 64;
constant constexpr uint kNAXTileK = 64;
constant constexpr uint kNAXLd = kNAXTileK + 8;

typedef vec<half, 8> nax_frag_h;
typedef vec<float, 8> nax_frag_f;

// BaseNAXFrag::get_coord: this lane's (column, row) in a 16x16 fragment. The
// lane holds rows r and r + 8, columns c .. c + 3, as elements i * 4 + j.
static inline short2 nax_coord(ushort lane) {
    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);
    return short2(fn, fm);
}

// BaseNAXFrag::mma, the (C n, C n+1) += A * (B n, B n+1)^T form.
static inline void nax_mma(thread nax_frag_f& c0, thread nax_frag_f& c1,
                           thread const nax_frag_h& a,
                           thread const nax_frag_h& b0, thread const nax_frag_h& b1) {
    constexpr auto desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto ct_a = op.get_left_input_cooperative_tensor<half, half, float>();
    auto ct_b = op.get_right_input_cooperative_tensor<half, half, float>();
    auto ct_c = op.get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();
    for (short i = 0; i < 8; ++i) {
        ct_a[i] = a[i];
        ct_b[i] = b0[i];
        ct_b[8 + i] = b1[i];
        ct_c[i] = c0[i];
        ct_c[8 + i] = c1[i];
    }
    op.run(ct_a, ct_b, ct_c);
    for (short i = 0; i < 8; ++i) {
        c0[i] = ct_c[i];
        c1[i] = ct_c[8 + i];
    }
}

kernel void mpp_prefill_affine_nax_f16(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device const half* activations      [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint lid                            [[thread_index_in_threadgroup]],
    ushort sgid                         [[simdgroup_index_in_threadgroup]],
    ushort lane                         [[thread_index_in_simdgroup]]) {
    threadgroup half Ws[kNAXTileN * kNAXLd];

    const uint mBase = tgid.y * kNAXTileM;
    const uint nBase = tgid.x * kNAXTileN;
    const uint tm = 32u * uint(sgid / 2);
    const uint tn = 32u * uint(sgid % 2);
    const short2 sc = nax_coord(lane);

    // Loader: thread t dequantizes row t / 2, columns (t % 2) * 32 .. +31 of the
    // 64x64 B tile -- one affine group at g32, half of one at g64.
    const uint groupsPerRow = K / kW4A8GroupSize;
    const uint bRow = lid >> 1;
    const uint bCol = (lid & 1u) * 32u;
    const uint wN = nBase + bRow;
    device const uint8_t* wRow = packedWeights + wN * (K / 2u);
    threadgroup half* bDst = Ws + bRow * kNAXLd + bCol;

    nax_frag_f C[2][2];
    for (short a = 0; a < 2; ++a) {
        for (short b = 0; b < 2; ++b) {
            C[a][b] = nax_frag_f(0.0f);
        }
    }

    for (uint k0 = 0; k0 < K; k0 += kNAXTileK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (wN < N) {
            const uint kb = k0 + bCol;
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

        for (uint kk1 = 0; kk1 < kNAXTileK; kk1 += 32u) {
            nax_frag_h A[2][2];   // [m fragment][k fragment]
            nax_frag_h B[2][2];   // [n fragment][k fragment]
            for (short mm = 0; mm < 2; ++mm) {
                for (short kk = 0; kk < 2; ++kk) {
                    const uint col = k0 + kk1 + uint(kk) * 16u + uint(sc.x);
                    for (short i = 0; i < 2; ++i) {
                        const uint row = mBase + tm + uint(mm) * 16u + uint(sc.y) + uint(i) * 8u;
                        const half4 v = row < M
                            ? *((device const half4*)(activations + row * K + col))
                            : half4(0.0h);
                        A[mm][kk][i * 4 + 0] = v.x;
                        A[mm][kk][i * 4 + 1] = v.y;
                        A[mm][kk][i * 4 + 2] = v.z;
                        A[mm][kk][i * 4 + 3] = v.w;
                    }
                }
            }
            for (short nn = 0; nn < 2; ++nn) {
                for (short kk = 0; kk < 2; ++kk) {
                    const uint col = kk1 + uint(kk) * 16u + uint(sc.x);
                    for (short i = 0; i < 2; ++i) {
                        const uint row = tn + uint(nn) * 16u + uint(sc.y) + uint(i) * 8u;
                        const half4 v = *((threadgroup half4*)(Ws + row * kNAXLd + col));
                        B[nn][kk][i * 4 + 0] = v.x;
                        B[nn][kk][i * 4 + 1] = v.y;
                        B[nn][kk][i * 4 + 2] = v.z;
                        B[nn][kk][i * 4 + 3] = v.w;
                    }
                }
            }
            // tile_matmad_nax, the TN % 2 == 0 branch.
            for (short mm = 0; mm < 2; ++mm) {
                for (short kk = 0; kk < 2; ++kk) {
                    nax_mma(C[mm][0], C[mm][1], A[mm][kk], B[0][kk], B[1][kk]);
                }
            }
        }
    }

    for (short mm = 0; mm < 2; ++mm) {
        for (short nn = 0; nn < 2; ++nn) {
            for (short i = 0; i < 2; ++i) {
                const uint m = mBase + tm + uint(mm) * 16u + uint(sc.y) + uint(i) * 8u;
                if (m >= M) continue;
                for (short j = 0; j < 4; ++j) {
                    const uint n = nBase + tn + uint(nn) * 16u + uint(sc.x) + uint(j);
                    if (n < N) {
                        output[m * N + n] = half(C[mm][nn][i * 4 + j]);
                    }
                }
            }
        }
    }
}


#endif
