#include <metal_stdlib>
using namespace metal;

// ============================================================================
// logit.metal — output-head kernels (#12, #13 in the inventory).
//
//   logit_softcap_softmax  Apply 30 * tanh(z / 30) to raw logits and produce a
//                          numerically-stable softmax over V=262144.
//   sample                 Sample a single token id from softmaxed probs.
//                          Supports greedy (temperature=0), temperature, top-k,
//                          top-p with a seeded PRNG.
//
// Both kernels run in a single threadgroup of 256 threads. V is far too large
// to fit in threadgroup memory (262144 * 4 B = 1 MB), so reductions are done
// in two stages: each thread strides over V, then a SIMD-group + cross-SIMD
// merge collapses partials. Threadgroup memory stores only one float per
// SIMD-group (up to 8 lanes for 256-thread groups on Apple silicon).
// ============================================================================

constant constexpr uint kLogitMaxSimdGroups = 8;
constant constexpr float kSampleTopMaxK     = 256.0f;  // cap for top-k mask scan

// ----------------------------------------------------------------------------
// K8: logit_softcap_softmax
//
// Online safe softmax (Milakov & Gimelshein, arXiv:1805.02867). Each thread
// walks its stride of V and maintains a running (m, d) pair: m is the max
// softcap'd logit seen so far, d is sum(exp(logit_i - m)). On every new logit
// the rule is:
//
//   m_new = max(m, z')
//   d_new = d * exp(m - m_new) + exp(z' - m_new)
//
// Softcap is applied inline before the max update — going through DRAM with a
// separate softcap pass costs 1 MB of round-trip traffic.
//
// After the per-thread loop the (m, d) pairs are merged across the threadgroup
// using the same rule, broadcast, and a second pass writes normalized FP16
// probabilities.
// ----------------------------------------------------------------------------

inline float softcap_value(float z, float softcap) {
    // A cap of zero or less means the family has no softcap and the logit
    // passes through unchanged. Qwen 3.5-MoE is that family
    // (`docs/qwen35moe/03-DESIGN.md` §2: "logit softcap — Qwen には無い"), and
    // applying Gemma's 30*tanh(z/30) to it would not be a small error: the cap
    // flattens everything above ~60 into the same value, which is exactly the
    // region a confident model draws from. The host mirror of this function
    // (`Sampler.softcapped`) already spells the same rule.
    if (!(softcap > 0.0f)) { return z; }
    // tanh saturates well before |z/softcap|=10, so values like +1e3 collapse
    // cleanly to softcap=30 without exp overflow downstream.
    return softcap * precise::tanh(z / softcap);
}

inline float logit_softmax_exp(float x) {
    return fast::exp(x);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void logit_softcap_softmax(
    device const half*  logits   [[buffer(0)]],   // [V] FP16
    device       half*  probs    [[buffer(1)]],   // [V] FP16
    constant     uint&  V        [[buffer(2)]],
    constant     float& softcap  [[buffer(3)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    // Per-SIMD-group partials for both the running max and the running sum.
    // Sized for the worst case of 8 SIMD-groups (256 threads / 32-lane SIMD).
    threadgroup float partial_m[kLogitMaxSimdGroups];
    threadgroup float partial_d[kLogitMaxSimdGroups];
    threadgroup float final_m;
    threadgroup float final_inv_d;

    // -log-of-zero sentinel: any real logit beats this on the first compare.
    float m = -INFINITY;
    float d = 0.0f;

    for (uint i = lid; i < V; i += lsize) {
        float z  = softcap_value(float(logits[i]), softcap);
        float mn = max(m, z);
        // Guard against the (-inf, -inf) → (-inf, NaN) case on the first iter.
        float scale = (m == -INFINITY) ? 0.0f : logit_softmax_exp(m - mn);
        d = d * scale + logit_softmax_exp(z - mn);
        m = mn;
    }

    // -- SIMD-group reduce.
    // simd_max gives the SIMD-wide max in one instruction; we then rescale our
    // own d to that common max so all 32 lanes share a comparable d term.
    float m_simd = simd_max(m);
    // Lanes that saw no element (m == -inf) must contribute 0, not NaN: when an
    // entire SIMD group is empty (V < threads_per_threadgroup, e.g. small-vocab
    // toy models) m_simd is also -inf, and exp(-inf - -inf) = exp(NaN) = NaN
    // would poison the sum. 0 * NaN is NaN, so guard before the multiply.
    float d_simd = simd_sum((m == -INFINITY) ? 0.0f : d * logit_softmax_exp(m - m_simd));

    if (simd_lane_id == 0) {
        partial_m[simd_group_id] = m_simd;
        partial_d[simd_group_id] = d_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Cross-SIMD merge in SIMD-group 0. Up to kLogitMaxSimdGroups partials.
    if (simd_group_id == 0) {
        float mp = (simd_lane_id < simdgroups) ? partial_m[simd_lane_id] : -INFINITY;
        float dp = (simd_lane_id < simdgroups) ? partial_d[simd_lane_id] : 0.0f;

        float m_all = simd_max(mp);
        // Same rescale trick, same empty-lane guard as above: an empty partial
        // (mp == -inf) contributes 0, never NaN.
        float d_all = simd_sum((mp == -INFINITY) ? 0.0f : dp * logit_softmax_exp(mp - m_all));

        if (simd_lane_id == 0) {
            final_m     = m_all;
            // Reciprocal once so the normalize loop is a single multiply.
            final_inv_d = 1.0f / d_all;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float m_final     = final_m;
    const float inv_d_final = final_inv_d;

    for (uint i = lid; i < V; i += lsize) {
        float z = softcap_value(float(logits[i]), softcap);
        probs[i] = half(logit_softmax_exp(z - m_final) * inv_d_final);
    }
}

// ----------------------------------------------------------------------------
// K9: sample
//
// Reads softmaxed probabilities and writes one token id. Behaviour:
//
//   temperature == 0   greedy argmax (the top-k / top-p / rng inputs are
//                      ignored; the argmax of probs is also the argmax of
//                      logits because softmax is monotonic).
//   temperature  > 0   top-p filters against the full normalized probability
//                      distribution, top-k caps the surviving set, then
//                      temperature reweights only that final categorical
//                      draw. This matches mlx-lm's sampler-chain order.
//
// xorshift64 is chosen for two reasons: it has a single 64-bit state (cheap
// to seed and to broadcast through threadgroup memory) and the output is bit-
// reproducible across hardware and compilers, which is what the tests assert.
// ----------------------------------------------------------------------------

// Thomas's "xorshift64*" variant — period 2^64-1, bias is negligible for
// sampling and the multiplier scrambles low-bit correlation. We only call it
// from one thread per dispatch, so contention is a non-issue.
inline uint64_t xorshift64(thread uint64_t& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s * 2685821657736338717ULL;
}

inline float uniform01(thread uint64_t& s) {
    // Use the top 24 bits as a [0,1) float — matches IEEE-754 mantissa width
    // so the result is exactly representable.
    uint32_t bits = uint32_t(xorshift64(s) >> 40);
    return float(bits) * (1.0f / 16777216.0f);
}

// Counter-based SplitMix64 keeps Gumbel draws deterministic across replays at
// a fixed seed without a precomputed random buffer.
inline uint64_t lmhead_splitmix64(uint64_t x) {
    x = x + 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline float lmhead_gumbel_for(uint64_t seed, uint position, uint row) {
    // Domain separation: mix (seed, position, row) by xors before the
    // splitmix scramble so adjacent rows yield independent draws.
    uint64_t key = seed ^ (uint64_t(position) * 0xD2B74407B1CE6E93ull) ^ uint64_t(row);
    uint64_t r   = lmhead_splitmix64(key);
    // [0, 1) with 24 mantissa bits; clamp away from zero/one so the double-log
    // can never overflow. 1/16777216 maps to ~5.96e-8, well above FP32 ULP.
    uint32_t bits = uint32_t(r >> 40);
    float u = (float(bits) + 0.5f) * (1.0f / 16777216.0f);
    return -log(-log(u));
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sample(
    device const half*    probs        [[buffer(0)]],   // [V] FP16
    device       uint*    out_token    [[buffer(1)]],   // [1] UInt32
    constant     uint&    V            [[buffer(2)]],
    constant     float&   temperature  [[buffer(3)]],
    constant     uint&    top_k        [[buffer(4)]],
    constant     float&   top_p        [[buffer(5)]],
    constant     uint64_t& seed        [[buffer(6)]],
    constant     uint&    position     [[buffer(7)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial_val[kLogitMaxSimdGroups];
    threadgroup uint  partial_idx[kLogitMaxSimdGroups];
    threadgroup uint  chosen;

    // ------------------------------------------------------------------
    // Greedy path (temperature == 0). Pure argmax(probs).
    // Softmax is monotonic, so we can argmax over probs directly without
    // looking back at the raw logits.
    // ------------------------------------------------------------------
    if (temperature <= 0.0f) {
        float best_v = -INFINITY;
        uint  best_i = 0;

        for (uint i = lid; i < V; i += lsize) {
            float p = float(probs[i]);
            if (p > best_v) {
                best_v = p;
                best_i = i;
            }
        }

        // SIMD-group argmax: simd_max gives the value; we need the matching
        // index, which simd_max can't return. Roll it manually: every lane
        // broadcasts (val, idx) via simd_shuffle and keeps the best pair.
        // 32-lane fan-in is one log2(32)=5-step tree but the lane-by-lane
        // scan is equally fast at this size and simpler to read.
        float m_simd = simd_max(best_v);
        uint  i_simd = best_i;
        // Tie-break: lowest index wins. Lanes whose value isn't the SIMD max
        // bow out with idx = UINT_MAX so simd_min picks the surviving smallest.
        i_simd = (best_v == m_simd) ? best_i : 0xFFFFFFFFu;
        i_simd = simd_min(i_simd);

        if (simd_lane_id == 0) {
            partial_val[simd_group_id] = m_simd;
            partial_idx[simd_group_id] = i_simd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
            float v = (simd_lane_id < simdgroups) ? partial_val[simd_lane_id] : -INFINITY;
            uint  i = (simd_lane_id < simdgroups) ? partial_idx[simd_lane_id] : 0xFFFFFFFFu;
            float m_all = simd_max(v);
            uint  i_all = (v == m_all) ? i : 0xFFFFFFFFu;
            i_all = simd_min(i_all);
            if (simd_lane_id == 0) {
                out_token[0] = i_all;
            }
        }
        return;
    }

    // ------------------------------------------------------------------
    // Fast stochastic path: no top-k, no top-p. Sampling from p^(1/T)/Z is
    // exactly argmax_i(log(p_i)/T + g_i) with iid Gumbel noise g_i — one
    // parallel V-pass, same shape as the greedy argmax above. The top-k
    // extraction loop below costs k full V-passes (k clamps to 256 when
    // top_k is "disabled"), which measured at ~2.6 s/token at V=262144;
    // this path replaces it for the common plain-temperature case.
    // ------------------------------------------------------------------
    if (top_k == 0u && (top_p <= 0.0f || top_p >= 1.0f)) {
        float inv_t = 1.0f / temperature;
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;

        for (uint i = lid; i < V; i += lsize) {
            float p = float(probs[i]);
            if (!(p > 0.0f)) continue;
            float s = inv_t * log(p) + lmhead_gumbel_for(seed, position, i);
            if (s > best_v) {
                best_v = s;
                best_i = i;
            }
        }

        float m_simd = simd_max(best_v);
        uint  i_simd = (best_v == m_simd) ? best_i : 0xFFFFFFFFu;
        i_simd = simd_min(i_simd);

        if (simd_lane_id == 0) {
            partial_val[simd_group_id] = m_simd;
            partial_idx[simd_group_id] = i_simd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
            float v = (simd_lane_id < simdgroups) ? partial_val[simd_lane_id] : -INFINITY;
            uint  i = (simd_lane_id < simdgroups) ? partial_idx[simd_lane_id] : 0xFFFFFFFFu;
            float m_all = simd_max(v);
            uint  i_all = (v == m_all) ? i : 0xFFFFFFFFu;
            i_all = simd_min(i_all);
            if (simd_lane_id == 0) {
                // All-zero probs cannot occur post-softmax, but a fully
                // skipped scan would leave UINT_MAX; fall back to index 0.
                out_token[0] = (i_all == 0xFFFFFFFFu) ? 0u : i_all;
            }
        }
        return;
    }

    // ------------------------------------------------------------------
    // Truncating stochastic path (top-k and/or top-p set).
    //
    // 1. Find the top-k threshold by repeatedly extracting the max and
    //    storing it in a small threadgroup buffer (k <= kSampleTopMaxK).
    // 2. Top-p truncates against the full-vocabulary probability mass.
    // 3. Temperature reweights the survivors as p^(1/T).
    // 4. CDF inverse-transform sample with the seeded PRNG.
    //
    // We use a *single* thread for the top-k / top-p / sample logic. Each
    // top-k slot costs one full V-pass, so explicit small top_k stays
    // tolerable; the k=256 worst case is exactly what the fast path above
    // removes from the default configuration.
    // ------------------------------------------------------------------

    // -- Threadgroup buffers for the sampler state. 256 floats + 256 indices
    // = 3 KB, comfortably inside the 32 KB threadgroup limit on Apple9+.
    threadgroup float topk_val[256];
    threadgroup uint  topk_idx[256];
    threadgroup uint  topk_count;

    // Effective k. top_k == 0 disables top-k entirely; we represent that as
    // k = V (subject to the kSampleTopMaxK working-buffer cap).
    uint k_req = (top_k == 0u) ? V : top_k;
    uint k     = min(k_req, uint(kSampleTopMaxK));
    k          = min(k, V);
    float inv_temp = 1.0f / temperature;

    if (lid == 0) {
        topk_count = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Extract top-k via k repeated argmax passes.
    // For each slot: every thread scans its V-stride for the argmax that
    // hasn't already been claimed, then we reduce across the threadgroup.
    // Claimed slots are tracked by index, not by zeroing the input — leaves
    // the probs buffer untouched (it's `device const`).
    for (uint slot = 0; slot < k; ++slot) {
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;

        for (uint i = lid; i < V; i += lsize) {
            float p = float(probs[i]);
            // Skip already-claimed indices.
            bool claimed = false;
            for (uint c = 0; c < slot; ++c) {
                if (topk_idx[c] == i) { claimed = true; break; }
            }
            if (claimed) continue;
            if (p > best_v) {
                best_v = p;
                best_i = i;
            }
        }

        float m_simd = simd_max(best_v);
        uint  i_simd = (best_v == m_simd) ? best_i : 0xFFFFFFFFu;
        i_simd = simd_min(i_simd);

        if (simd_lane_id == 0) {
            partial_val[simd_group_id] = m_simd;
            partial_idx[simd_group_id] = i_simd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
            float v = (simd_lane_id < simdgroups) ? partial_val[simd_lane_id] : -INFINITY;
            uint  i = (simd_lane_id < simdgroups) ? partial_idx[simd_lane_id] : 0xFFFFFFFFu;
            float m_all = simd_max(v);
            uint  i_all = (v == m_all) ? i : 0xFFFFFFFFu;
            i_all = simd_min(i_all);
            if (simd_lane_id == 0) {
                topk_val[slot] = m_all;
                topk_idx[slot] = i_all;
                topk_count     = slot + 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -- Top-p truncation and sample, single-threaded (k <= 256 elements).
    if (lid == 0) {
        // Drop trailing invalid slots. When the working set k exceeds the
        // number of rankable entries (high temperature flattens ties; the tail
        // can exhaust before k slots fill), a slot's reduction yields a -INF
        // value / UINT_MAX index. Those sort last (descending by mass), so the
        // first invalid slot bounds the real count. Slot 0 is the global argmax
        // and is always valid, so kept >= 1 holds whenever V > 0.
        uint kept = 0;
        while (kept < topk_count
               && topk_idx[kept] < V
               && isfinite(topk_val[kept])) {
            kept += 1;
        }
        // mlx-lm applies Top-P before Top-K. The probabilities still carry
        // their full-vocabulary normalization here, so comparing the raw
        // descending cumulative mass to top_p produces the same intersection
        // without sorting the entire vocabulary. If Top-64 holds less than
        // top_p mass, all 64 survive and Top-K is the limiting filter.
        if (top_p > 0.0f && top_p < 1.0f) {
            float cum = 0.0f;
            uint  cut = kept;
            for (uint i = 0; i < kept; ++i) {
                cum += topk_val[i];
                if (cum >= top_p) { cut = i + 1; break; }
            }
            kept = cut;
        }

        // MLX applies temperature in categorical_sampling after both filters.
        for (uint i = 0; i < kept; ++i) {
            if (temperature != 1.0f) {
                topk_val[i] = pow(topk_val[i], inv_temp);
            }
        }

        // Sum the reweighted surviving mass for inverse-CDF sampling.
        float surviving = 0.0f;
        for (uint i = 0; i < kept; ++i) surviving += topk_val[i];

        uint64_t rng = seed;
        // One xorshift step before sampling so identical seeds across calls
        // don't produce the same first draw when the seed is small (a few
        // bits short of full entropy on xorshift's first output).
        (void)xorshift64(rng);
        float u = uniform01(rng) * surviving;

        // Pick by walking the CDF until we cross u. With kept <= 256 this is
        // a tight loop and stays inside L1.
        // Default to slot 0 (the global argmax — always a valid index) so a
        // CDF that never crosses u (FP rounding leaves run slightly below the
        // surviving mass) still returns a real token, never a tail sentinel.
        float run = 0.0f;
        uint  picked = topk_idx[0];
        for (uint i = 0; i < kept; ++i) {
            run += topk_val[i];
            if (u <= run) { picked = topk_idx[i]; break; }
        }

        chosen = picked;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        out_token[0] = chosen;
    }
}

// Specialized Top-64 selection. Each stage reduces sorted
// 1,024-entry tiles to 64 pairs, so the full 262,144-entry vocabulary reaches
// one final tile in three dispatches without a full-vocabulary FP32 copy.
inline bool sample_topk64_better(float lhs_value, uint lhs_index,
                                 float rhs_value, uint rhs_index) {
    return lhs_value > rhs_value
        || (lhs_value == rhs_value && lhs_index < rhs_index);
}

inline void sample_topk64_sort_tile(threadgroup float* values,
                                    threadgroup uint* indices,
                                    uint lid) {
    for (uint width = 2; width <= 1024; width <<= 1) {
        for (uint stride = width >> 1; stride > 0; stride >>= 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lid; i < 1024; i += 256) {
                uint partner = i ^ stride;
                if (partner <= i) continue;

                float lhs_value = values[i];
                uint lhs_index = indices[i];
                float rhs_value = values[partner];
                uint rhs_index = indices[partner];
                bool descending = (i & width) == 0;
                bool swap = descending
                    ? sample_topk64_better(rhs_value, rhs_index,
                                           lhs_value, lhs_index)
                    : sample_topk64_better(lhs_value, lhs_index,
                                           rhs_value, rhs_index);
                if (swap) {
                    values[i] = rhs_value;
                    indices[i] = rhs_index;
                    values[partner] = lhs_value;
                    indices[partner] = lhs_index;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sample_topk64_stage1(
    device const half* probs [[buffer(0)]],
    device float* out_values [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    constant uint& V [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float values[1024];
    threadgroup uint indices[1024];
    uint base = group * 1024;

    for (uint i = lid; i < 1024; i += 256) {
        uint source = base + i;
        bool valid = source < V;
        values[i] = valid ? float(probs[source]) : -INFINITY;
        indices[i] = valid ? source : 0xFFFFFFFFu;
    }
    sample_topk64_sort_tile(values, indices, lid);

    if (lid < 64) {
        uint destination = group * 64 + lid;
        out_values[destination] = values[lid];
        out_indices[destination] = indices[lid];
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sample_topk64_reduce(
    device const float* in_values [[buffer(0)]],
    device const uint* in_indices [[buffer(1)]],
    device float* out_values [[buffer(2)]],
    device uint* out_indices [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float values[1024];
    threadgroup uint indices[1024];
    uint base = group * 1024;

    for (uint i = lid; i < 1024; i += 256) {
        uint source = base + i;
        bool valid = source < count;
        values[i] = valid ? in_values[source] : -INFINITY;
        indices[i] = valid ? in_indices[source] : 0xFFFFFFFFu;
    }
    sample_topk64_sort_tile(values, indices, lid);

    if (lid < 64) {
        uint destination = group * 64 + lid;
        out_values[destination] = values[lid];
        out_indices[destination] = indices[lid];
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sample_topk64_final(
    device const float* in_values [[buffer(0)]],
    device const uint* in_indices [[buffer(1)]],
    device uint* out_token [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& temperature [[buffer(4)]],
    constant float& top_p [[buffer(5)]],
    constant uint64_t& seed [[buffer(6)]],
    uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float values[1024];
    threadgroup uint indices[1024];

    for (uint i = lid; i < 1024; i += 256) {
        bool valid = i < count;
        values[i] = valid ? in_values[i] : -INFINITY;
        indices[i] = valid ? in_indices[i] : 0xFFFFFFFFu;
    }
    sample_topk64_sort_tile(values, indices, lid);

    if (lid == 0) {
        uint kept = 0;
        while (kept < 64
               && indices[kept] != 0xFFFFFFFFu
               && isfinite(values[kept])) {
            kept += 1;
        }

        if (top_p > 0.0f && top_p < 1.0f) {
            float cumulative = 0.0f;
            uint cut = kept;
            for (uint i = 0; i < kept; ++i) {
                cumulative += values[i];
                if (cumulative >= top_p) {
                    cut = i + 1;
                    break;
                }
            }
            kept = cut;
        }

        float inv_temperature = 1.0f / temperature;
        for (uint i = 0; i < kept; ++i) {
            if (temperature != 1.0f) {
                values[i] = pow(values[i], inv_temperature);
            }
        }

        float surviving = 0.0f;
        for (uint i = 0; i < kept; ++i) surviving += values[i];
        uint64_t rng = seed;
        (void)xorshift64(rng);
        float u = uniform01(rng) * surviving;
        float running = 0.0f;
        uint picked = indices[0];
        for (uint i = 0; i < kept; ++i) {
            running += values[i];
            if (u <= running) {
                picked = indices[i];
                break;
            }
        }
        out_token[0] = picked;
    }
}


// Fused greedy lm-head path. Eight SIMD groups each evaluate one INT4 row;
// a second dispatch reduces the per-threadgroup argmax summaries.
constant constexpr uint kLMHeadRowsPerTG = 8;
#ifndef TURBO_AFFINE_GROUP_SIZE
#define TURBO_AFFINE_GROUP_SIZE 64
#endif
constant constexpr uint kLMHeadGroupSize = TURBO_AFFINE_GROUP_SIZE;

// Affine zero point -- see the note in dequant_int4.metal. `TURBO_AFFINE_SYMMETRIC`
// is a whole-model compile-time constant (`MetalContext.affineScheme`); when it
// is set the bias arrays do not exist and the bindings alias the scales.
#ifndef TURBO_AFFINE_SYMMETRIC
#define TURBO_AFFINE_SYMMETRIC 0
#endif

static inline float lmhead_int4_bias(device const bfloat* biases, uint index,
                                     float scale) {
#if TURBO_AFFINE_SYMMETRIC
    return -8.0f * scale;
#else
    return float(biases[index]);
#endif
}

// Vectorized INT4 block geometry — see the note in dequant_int4.metal.
// A block is a fixed 128 bytes (32 lanes x 4 bytes); the group size decides how
// many affine groups that spans and how many lanes cover one group.
constant constexpr uint kLMHeadGroupsPerBlock = 256u / kLMHeadGroupSize;
constant constexpr uint kLMHeadLanesPerGroup  = kLMHeadGroupSize / 8u;
constant constexpr uint kLMHeadTailLanes      = kLMHeadGroupSize / 2u;

constant constexpr uint kLMHeadRowSummaryStride = 2;
constant uint FC_HEAD_D [[function_constant(10)]];
constant uint FC_HEAD_V [[function_constant(11)]];
constant bool FC_HEAD_USE_FC [[function_constant(13)]];

static inline uint lmhead_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_HEAD_USE_FC) &&
            FC_HEAD_USE_FC &&
            is_function_constant_defined(FC_HEAD_D)) ? FC_HEAD_D : D;
}

static inline uint lmhead_fc_v(constant uint& V) {
    return (is_function_constant_defined(FC_HEAD_USE_FC) &&
            FC_HEAD_USE_FC &&
            is_function_constant_defined(FC_HEAD_V)) ? FC_HEAD_V : V;
}

inline float lmhead_int4_gemv_row_simd_dev(device const uint8_t*    W,
                                           device const bfloat*     scales,
                                           device const bfloat*     biases,
                                           device const half*       x,
                                           uint row,
                                           uint D,
                                           uint lane) {
    const uint n_groups  = D / kLMHeadGroupSize;
    const uint row_bytes = D / 2u;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / kLMHeadGroupsPerBlock;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * kLMHeadGroupsPerBlock + lane / kLMHeadLanesPerGroup;
        const float s = float(s_row[g]);
        const float b = lmhead_int4_bias(b_row, g, s);
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
    for (uint g = full_blocks * kLMHeadGroupsPerBlock; g < n_groups; ++g) {
        // Only the first kLMHeadTailLanes lanes hold a byte of this group.
        if (lane >= kLMHeadTailLanes) break;
        const float s = float(s_row[g]);
        const float b = lmhead_int4_bias(b_row, g, s);
        const uint8_t byte = W_row[g * (kLMHeadGroupSize / 2) + lane];
        const float x0 = float(x[g * kLMHeadGroupSize + lane * 2u]);
        const float x1 = float(x[g * kLMHeadGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    return simd_sum(acc);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void lm_head_greedy_int4_rows_chunk_raw(
    device const half*    x_normed     [[buffer(0)]],
    device const uint8_t* W            [[buffer(1)]],
    device const bfloat*  scales       [[buffer(2)]],
    device const bfloat*  biases       [[buffer(3)]],
    device       float*   summaries    [[buffer(4)]],
    constant     uint&    D            [[buffer(5)]],
    constant     uint&    V            [[buffer(6)]],
    uint  tg_idx         [[threadgroup_position_in_grid]],
    uint  simd_lane_id   [[thread_index_in_simdgroup]],
    uint  simd_group_id  [[simdgroup_index_in_threadgroup]],
    uint  simdgroups     [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial_v[kLogitMaxSimdGroups];
    threadgroup uint partial_i[kLogitMaxSimdGroups];
    const uint DD = lmhead_fc_d(D);
    const uint VV = lmhead_fc_v(V);

    const uint row = tg_idx * kLMHeadRowsPerTG + simd_group_id;
    float best_v = -INFINITY;
    uint best_i = 0xFFFFFFFFu;

    if (row < VV) {
        float z = lmhead_int4_gemv_row_simd_dev(W, scales, biases,
                                                x_normed, row, DD, simd_lane_id);
        if (simd_lane_id == 0 && isfinite(z)) {
            best_v = z;
            best_i = row;
        }
    }

    if (simd_lane_id == 0) {
        partial_v[simd_group_id] = best_v;
        partial_i[simd_group_id] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        const bool active = simd_lane_id < simdgroups;
        const float v = active ? partial_v[simd_lane_id] : -INFINITY;
        const uint idx = active ? partial_i[simd_lane_id] : 0xFFFFFFFFu;
        const float v_all = simd_max(v);
        uint i_all = (v == v_all) ? idx : 0xFFFFFFFFu;
        i_all = simd_min(i_all);
        if (simd_lane_id == 0) {
            device float* slot = summaries + tg_idx * kLMHeadRowSummaryStride;
            slot[0] = v_all;
            slot[1] = as_type<float>(i_all);
        }
    }
}

// The head for a block of at most eight rows.
//
// `lm_head_greedy_int4_rows_chunk_raw` above scores one row against the whole
// vocabulary, so a k-row speculative verify block encoded k of them and read
// the 461 MB lm-head table k times. In bytes that is the second largest item in
// the verify bill — at k=4 it is 1.38 GB, more than half a decode step
// (docs/mtp/16-M4.5-PLAN.md §2). This kernel keeps one SIMD group on one
// vocabulary row and spends that row's weights on all T activations, so the
// table is read once per block. Per (row, vocabulary row) the reduction is the
// one-row kernel's, in the same order.
//
// `summaries` is `[T, row_groups, 2]`: the reducer runs once per row over its
// own slice.
constant constexpr uint kLMHeadMaxBlockRows = 8;

[[kernel, max_total_threads_per_threadgroup(256)]]
void lm_head_greedy_int4_rows_chunk_block(
    device const half*    x_normed     [[buffer(0)]],
    device const uint8_t* W            [[buffer(1)]],
    device const bfloat*  scales       [[buffer(2)]],
    device const bfloat*  biases       [[buffer(3)]],
    device       float*   summaries    [[buffer(4)]],
    constant     uint&    D            [[buffer(5)]],
    constant     uint&    V            [[buffer(6)]],
    constant     uint&    T            [[buffer(7)]],
    constant     uint&    row_groups   [[buffer(8)]],
    uint  tg_idx         [[threadgroup_position_in_grid]],
    uint  simd_lane_id   [[thread_index_in_simdgroup]],
    uint  simd_group_id  [[simdgroup_index_in_threadgroup]],
    uint  simdgroups     [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial_v[kLogitMaxSimdGroups * kLMHeadMaxBlockRows];
    threadgroup uint partial_i[kLogitMaxSimdGroups * kLMHeadMaxBlockRows];
    const uint DD = lmhead_fc_d(D);
    const uint VV = lmhead_fc_v(V);
    const uint row = tg_idx * kLMHeadRowsPerTG + simd_group_id;

    float z[kLMHeadMaxBlockRows];
    for (uint t = 0; t < kLMHeadMaxBlockRows; ++t) { z[t] = 0.0f; }

    if (row < VV) {
        const uint n_groups = DD / kLMHeadGroupSize;
        const uint row_bytes = DD / 2u;
        device const uint8_t* W_row = W + uint(row) * row_bytes;
        device const bfloat* s_row = scales + uint(row) * n_groups;
        device const bfloat* b_row = biases + uint(row) * n_groups;

        float acc[kLMHeadMaxBlockRows];
        for (uint t = 0; t < kLMHeadMaxBlockRows; ++t) { acc[t] = 0.0f; }

        const uint full_blocks = n_groups / kLMHeadGroupsPerBlock;
        for (uint blk = 0; blk < full_blocks; ++blk) {
            const uint byte_base = blk * 128u + simd_lane_id * 4u;
            device const ushort* wp = (device const ushort*)(W_row + byte_base);
            const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
            const uint g = blk * kLMHeadGroupsPerBlock
                + simd_lane_id / kLMHeadLanesPerGroup;
            const float s = float(s_row[g]);
            const float b = lmhead_int4_bias(b_row, g, s);
            const uint elem = byte_base * 2u;
            const uint b0 = w4 & 0xFFu, b1 = (w4 >> 8) & 0xFFu;
            const uint b2 = (w4 >> 16) & 0xFFu, b3 = (w4 >> 24) & 0xFFu;
            for (uint t = 0; t < kLMHeadMaxBlockRows; ++t) {
                if (t >= T) break;
                device const half* x = x_normed + t * DD;
                const half4 xa = *((device const half4*)(x + elem));
                const half4 xb = *((device const half4*)(x + elem + 4u));
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
                acc[t] = fma(s, dot, acc[t]);
                acc[t] = fma(b, sum, acc[t]);
            }
        }
        for (uint g = full_blocks * kLMHeadGroupsPerBlock; g < n_groups; ++g) {
            // Only the first kLMHeadTailLanes lanes hold a byte of this group.
            if (simd_lane_id >= kLMHeadTailLanes) break;
            const float s = float(s_row[g]);
            const float b = lmhead_int4_bias(b_row, g, s);
            const uint8_t byte = W_row[g * (kLMHeadGroupSize / 2) + simd_lane_id];
            for (uint t = 0; t < kLMHeadMaxBlockRows; ++t) {
                if (t >= T) break;
                device const half* x = x_normed + t * DD;
                const float x0 = float(x[g * kLMHeadGroupSize + simd_lane_id * 2u]);
                const float x1 = float(x[g * kLMHeadGroupSize + simd_lane_id * 2u + 1u]);
                float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
                dot = fma(float(uint(byte >> 4)), x1, dot);
                acc[t] = fma(s, dot, acc[t]);
                acc[t] = fma(b, x0 + x1, acc[t]);
            }
        }
        // `T` is uniform, so every lane reaches each reduction.
        for (uint t = 0; t < kLMHeadMaxBlockRows; ++t) {
            if (t >= T) break;
            z[t] = simd_sum(acc[t]);
        }
    }

    if (simd_lane_id == 0) {
        for (uint t = 0; t < kLMHeadMaxBlockRows; ++t) {
            if (t >= T) break;
            const uint slot = simd_group_id * kLMHeadMaxBlockRows + t;
            const bool live = row < VV && isfinite(z[t]);
            partial_v[slot] = live ? z[t] : -INFINITY;
            partial_i[slot] = live ? row : 0xFFFFFFFFu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        for (uint t = 0; t < kLMHeadMaxBlockRows; ++t) {
            if (t >= T) break;
            const bool active = simd_lane_id < simdgroups;
            const uint slot = simd_lane_id * kLMHeadMaxBlockRows + t;
            const float v = active ? partial_v[slot] : -INFINITY;
            const uint idx = active ? partial_i[slot] : 0xFFFFFFFFu;
            const float v_all = simd_max(v);
            uint i_all = (v == v_all) ? idx : 0xFFFFFFFFu;
            i_all = simd_min(i_all);
            if (simd_lane_id == 0) {
                device float* out = summaries
                    + (t * row_groups + tg_idx) * kLMHeadRowSummaryStride;
                out[0] = v_all;
                out[1] = as_type<float>(i_all);
            }
        }
    }
}

// The block head with the per-activation-row work taken out of the weight loop.
//
// `lm_head_greedy_int4_rows_chunk_block` above reads the 461 MB table once per
// block, which is why the block's bytes do not grow with k. Its *time* still
// did: at k=4 the stage cost 7 ms against a 3.4 ms bandwidth floor, because
// three things inside the weight loop were charged per activation row that need
// not be — the group's activation sum (which does not depend on the vocabulary
// row at all, yet all 262144 of them recomputed it), the activation load, and
// its half->float conversion (`docs/mtp/20-M4.8-RESULTS.md` §2).
//
// So: `xsum` carries the activation sums in, precomputed once per block by
// `int4_rows_group_sums` in the same order; `FC_LMHEAD_ROWS_PER_SG` gives one
// SIMD group R vocabulary rows so a loaded activation pays for R of them; and
// `FC_LMHEAD_T` pins the block width at build time so the row loop unrolls and
// `acc` stays in registers. Past four rows it stops fitting, so the wrapper
// splits a wider block into dispatches of four.
//
// Per (activation row, vocabulary row) the reduction is unchanged, element for
// element and in the same order. The argmax is unchanged too: a SIMD group now
// picks between its R rows first, keeping "greater value wins, and on a tie the
// smaller vocabulary index wins", which is what the cross-SIMD merge below
// already did across the threadgroup.
constant uint FC_LMHEAD_ROWS_PER_SG [[function_constant(14)]];
constant uint FC_LMHEAD_T           [[function_constant(15)]];

constant constexpr uint kLMHeadMaxRowsPerSG = 2;
constant constexpr uint kLMHeadWideMaxT     = 4;

static inline uint lmhead_rows_per_sg() {
    return is_function_constant_defined(FC_LMHEAD_ROWS_PER_SG) ? FC_LMHEAD_ROWS_PER_SG : 1u;
}

static inline uint lmhead_wide_t(constant uint& T) {
    return is_function_constant_defined(FC_LMHEAD_T) ? FC_LMHEAD_T : T;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void lm_head_greedy_int4_rows_chunk_block_wide(
    device const half*    x_normed     [[buffer(0)]],
    device const uint8_t* W            [[buffer(1)]],
    device const bfloat*  scales       [[buffer(2)]],
    device const bfloat*  biases       [[buffer(3)]],
    device       float*   summaries    [[buffer(4)]],
    constant     uint&    D            [[buffer(5)]],
    constant     uint&    V            [[buffer(6)]],
    constant     uint&    T_in         [[buffer(7)]],
    constant     uint&    row_groups   [[buffer(8)]],
    device const float*   xsum         [[buffer(9)]],
    constant     uint&    xsum_stride  [[buffer(10)]],
    uint  tg_idx         [[threadgroup_position_in_grid]],
    uint  lane           [[thread_index_in_simdgroup]],
    uint  simd_group_id  [[simdgroup_index_in_threadgroup]],
    uint  simdgroups     [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial_v[kLogitMaxSimdGroups * kLMHeadWideMaxT];
    threadgroup uint partial_i[kLogitMaxSimdGroups * kLMHeadWideMaxT];
    const uint R = lmhead_rows_per_sg();
    const uint T = lmhead_wide_t(T_in);
    const uint DD = lmhead_fc_d(D);
    const uint VV = lmhead_fc_v(V);
    const uint row0 = (tg_idx * kLMHeadRowsPerTG + simd_group_id) * R;

    // Best (value, vocabulary row) this SIMD group holds, per activation row.
    float best_v[kLMHeadWideMaxT];
    uint best_i[kLMHeadWideMaxT];
    for (uint t = 0; t < kLMHeadWideMaxT; ++t) {
        best_v[t] = -INFINITY;
        best_i[t] = 0xFFFFFFFFu;
    }

    if (row0 < VV) {
        const uint n_groups = DD / kLMHeadGroupSize;
        const uint row_bytes = DD / 2u;
        const uint full_blocks = n_groups / kLMHeadGroupsPerBlock;

        uint row_of[kLMHeadMaxRowsPerSG];
        for (uint r = 0; r < R; ++r) { row_of[r] = min(row0 + r, VV - 1u); }

        float acc[kLMHeadWideMaxT * kLMHeadMaxRowsPerSG];
        for (uint i = 0; i < kLMHeadWideMaxT * kLMHeadMaxRowsPerSG; ++i) { acc[i] = 0.0f; }

        for (uint blk = 0; blk < full_blocks; ++blk) {
            const uint byte_base = blk * 128u + lane * 4u;
            const uint g = blk * kLMHeadGroupsPerBlock + lane / kLMHeadLanesPerGroup;
            const uint elem = byte_base * 2u;

            float q[kLMHeadMaxRowsPerSG * 8];
            float sv[kLMHeadMaxRowsPerSG];
            float bv[kLMHeadMaxRowsPerSG];
            for (uint r = 0; r < R; ++r) {
                const uint row = row_of[r];
                device const ushort* wp =
                    (device const ushort*)(W + row * row_bytes + byte_base);
                const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
                sv[r] = float(scales[row * n_groups + g]);
                bv[r] = lmhead_int4_bias(biases, row * n_groups + g, sv[r]);
                const uint b0 =  w4        & 0xFFu;
                const uint b1 = (w4 >> 8)  & 0xFFu;
                const uint b2 = (w4 >> 16) & 0xFFu;
                const uint b3 = (w4 >> 24) & 0xFFu;
                q[r * 8 + 0] = float(b0 & 0x0Fu); q[r * 8 + 1] = float(b0 >> 4);
                q[r * 8 + 2] = float(b1 & 0x0Fu); q[r * 8 + 3] = float(b1 >> 4);
                q[r * 8 + 4] = float(b2 & 0x0Fu); q[r * 8 + 5] = float(b2 >> 4);
                q[r * 8 + 6] = float(b3 & 0x0Fu); q[r * 8 + 7] = float(b3 >> 4);
            }

            for (uint t = 0; t < kLMHeadWideMaxT; ++t) {
                if (t >= T) break;
                device const half* x = x_normed + t * DD;
                const half4 xa = *((device const half4*)(x + elem));
                const half4 xb = *((device const half4*)(x + elem + 4u));
                const float e0 = float(xa.x), e1 = float(xa.y);
                const float e2 = float(xa.z), e3 = float(xa.w);
                const float e4 = float(xb.x), e5 = float(xb.y);
                const float e6 = float(xb.z), e7 = float(xb.w);
                const float sum = xsum[t * xsum_stride + blk * 32u + lane];
                for (uint r = 0; r < R; ++r) {
                    float dot = 0.0f;
                    dot = fma(q[r * 8 + 0], e0, dot); dot = fma(q[r * 8 + 1], e1, dot);
                    dot = fma(q[r * 8 + 2], e2, dot); dot = fma(q[r * 8 + 3], e3, dot);
                    dot = fma(q[r * 8 + 4], e4, dot); dot = fma(q[r * 8 + 5], e5, dot);
                    dot = fma(q[r * 8 + 6], e6, dot); dot = fma(q[r * 8 + 7], e7, dot);
                    acc[r * kLMHeadWideMaxT + t] =
                        fma(sv[r], dot, acc[r * kLMHeadWideMaxT + t]);
                    acc[r * kLMHeadWideMaxT + t] =
                        fma(bv[r], sum, acc[r * kLMHeadWideMaxT + t]);
                }
            }
        }
        for (uint gg = full_blocks * kLMHeadGroupsPerBlock; gg < n_groups; ++gg) {
            // Only the first kLMHeadTailLanes lanes hold a byte of this group.
            if (lane >= kLMHeadTailLanes) break;
            for (uint t = 0; t < kLMHeadWideMaxT; ++t) {
                if (t >= T) break;
                device const half* x = x_normed + t * DD;
                const float x0 = float(x[gg * kLMHeadGroupSize + lane * 2u]);
                const float x1 = float(x[gg * kLMHeadGroupSize + lane * 2u + 1u]);
                for (uint r = 0; r < R; ++r) {
                    const uint row = row_of[r];
                    const float s = float(scales[row * n_groups + gg]);
                    const float b = lmhead_int4_bias(biases, row * n_groups + gg, s);
                    const uint8_t byte =
                        W[row * row_bytes + gg * (kLMHeadGroupSize / 2) + lane];
                    float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
                    dot = fma(float(uint(byte >> 4)), x1, dot);
                    acc[r * kLMHeadWideMaxT + t] =
                        fma(s, dot, acc[r * kLMHeadWideMaxT + t]);
                    acc[r * kLMHeadWideMaxT + t] =
                        fma(b, x0 + x1, acc[r * kLMHeadWideMaxT + t]);
                }
            }
        }
        // `R` and `T` are compile-time, so every lane reaches each reduction.
        // Rows are visited in increasing order and only a strictly greater
        // value displaces the incumbent, so a tie keeps the smaller row.
        for (uint r = 0; r < R; ++r) {
            for (uint t = 0; t < kLMHeadWideMaxT; ++t) {
                if (t >= T) break;
                const float z = simd_sum(acc[r * kLMHeadWideMaxT + t]);
                if (lane == 0 && row0 + r < VV && isfinite(z) && z > best_v[t]) {
                    best_v[t] = z;
                    best_i[t] = row0 + r;
                }
            }
        }
    }

    if (lane == 0) {
        for (uint t = 0; t < kLMHeadWideMaxT; ++t) {
            if (t >= T) break;
            partial_v[simd_group_id * kLMHeadWideMaxT + t] = best_v[t];
            partial_i[simd_group_id * kLMHeadWideMaxT + t] = best_i[t];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        for (uint t = 0; t < kLMHeadWideMaxT; ++t) {
            if (t >= T) break;
            const bool active = lane < simdgroups;
            const uint slot = lane * kLMHeadWideMaxT + t;
            const float v = active ? partial_v[slot] : -INFINITY;
            const uint idx = active ? partial_i[slot] : 0xFFFFFFFFu;
            const float v_all = simd_max(v);
            uint i_all = (v == v_all) ? idx : 0xFFFFFFFFu;
            i_all = simd_min(i_all);
            if (lane == 0) {
                device float* out = summaries
                    + (t * row_groups + tg_idx) * kLMHeadRowSummaryStride;
                out[0] = v_all;
                out[1] = as_type<float>(i_all);
            }
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void lm_head_greedy_int4_rows_reduce(
    device const float*   summaries    [[buffer(0)]],
    device       uint*    out_token    [[buffer(1)]],
    constant     uint&    row_groups   [[buffer(2)]],
    uint  lid            [[thread_position_in_threadgroup]],
    uint  lsize          [[threads_per_threadgroup]],
    uint  simd_lane_id   [[thread_index_in_simdgroup]],
    uint  simd_group_id  [[simdgroup_index_in_threadgroup]],
    uint  simdgroups     [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial_v[kLogitMaxSimdGroups];
    threadgroup uint  partial_i[kLogitMaxSimdGroups];

    float best_v = -INFINITY;
    uint best_i = 0xFFFFFFFFu;
    for (uint i = lid; i < row_groups; i += lsize) {
        device const float* slot = summaries + i * kLMHeadRowSummaryStride;
        const float v = slot[0];
        const uint idx = as_type<uint>(slot[1]);
        if (v > best_v || (v == best_v && idx < best_i)) {
            best_v = v;
            best_i = idx;
        }
    }

    float v_simd = simd_max(best_v);
    uint i_simd = (best_v == v_simd) ? best_i : 0xFFFFFFFFu;
    i_simd = simd_min(i_simd);

    if (simd_lane_id == 0) {
        partial_v[simd_group_id] = v_simd;
        partial_i[simd_group_id] = i_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        const bool active = simd_lane_id < simdgroups;
        const float v = active ? partial_v[simd_lane_id] : -INFINITY;
        const uint idx = active ? partial_i[simd_lane_id] : 0xFFFFFFFFu;
        const float v_all = simd_max(v);
        uint i_all = (v == v_all) ? idx : 0xFFFFFFFFu;
        i_all = simd_min(i_all);
        if (simd_lane_id == 0) {
            out_token[0] = (i_all == 0xFFFFFFFFu) ? 0u : i_all;
        }
    }
}
