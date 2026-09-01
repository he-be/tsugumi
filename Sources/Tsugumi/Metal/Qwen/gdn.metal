#include <metal_stdlib>
using namespace metal;

// ============================================================================
// qwen_delta_rule — Gated DeltaNet (Qwen3.5-MoE の線形注意 30 層)
//
// 漸化式 (docs/qwen35moe/01-MODEL.md §3、上流実装と参照器で検証済み):
//
//   S      = g[t] * S                       // 減衰。S: [Dv, Dk] を head ごとに
//   kv_mem = S . k[t]                       // [Dv]
//   delta  = (v[t] - kv_mem) * beta[t]      // [Dv]
//   S     += outer(delta, k[t])
//   y[t]   = S . q[t]                       // [Dv]
//
// **chunkwise (WY/UT) ではなく厳密な逐次形**。線形注意の総 FLOP は prefill 2048
// トークンで 193 GFLOP しかなく MoE の 4% で、chunkwise は FLOP が 2 倍になる
// (docs/qwen35moe/03-DESIGN.md §2-6)。
//
// 幾何は omlx の `gated_delta_blocked_seq` を写したもの (Apache-2.0、参照のみ):
//
//   grid        = (Dv/DB, Hv, 1) threadgroup、256 スレッド
//   thread      → dv = tid/8 (DB 行のどれか)、seg = tid%8 → d0 = seg*16
//   状態        = **レジスタ** float4 st[4] (16 float/thread、256×16 = 32×128)
//   縮約        = simd_shuffle_down(4→2→1) + simd_shuffle の同報。
//                 **ホットループに threadgroup_barrier が 1 個も無い**
//   k/q/v       = TB トークンぶんを threadgroup memory に協調ロード。
//                 dv を Dv/4 で切ると同じ k/q 行を 32 回読み直すことになる
//
// TB (時間ブロック) は 16 / 32 / 48 の 3 本を用意する (Phase 4 で測る)。
// threadgroup memory: TB×(Dk+8)×2 本 + TB×(DB+8) の half と TB×2 の float。
//   TB=16: 10,112 B   TB=32: 20,224 B   TB=48: 30,336 B  (上限 32,768 B)
// ============================================================================

// 鍵の次元は Qwen3.5-MoE では 128 固定 (`linear_key_head_dim`)。threadgroup 配列の
// 寸法なのでコンパイル時定数である必要がある。呼び手が別の値を渡したら弾く。
constant constexpr int kQwenGDNDk = 128;
// +8 half = バンク衝突の回避 (omlx と同じ)。
constant constexpr int kQwenGDNKStride = kQwenGDNDk + 8;
// 1 threadgroup が持つ dv 行数。Dv=128 なので 4 threadgroup で 1 head を覆う。
constant constexpr int kQwenGDNDB = 32;
constant constexpr int kQwenGDNVStride = kQwenGDNDB + 8;

struct QwenDeltaRuleParams {
    uint  seqLen;      // T
    uint  numKHeads;   // Hk = 16
    uint  numVHeads;   // Hv = 32
    uint  valueDim;    // Dv = 128
};

// 8 スレッド (seg=0..7) の部分和を畳んで、その dv 行の全スレッドに配る。
// simdgroup 内で lane = tid%32、dv 行は 8 lane 連続なので shuffle だけで閉じる。
static inline float qwen_gdn_row_reduce(float part, uint tid) {
    part += simd_shuffle_down(part, 4);
    part += simd_shuffle_down(part, 2);
    part += simd_shuffle_down(part, 1);
    return simd_shuffle(part, (tid % 32) / 8 * 8);
}

// TB は呼び手の kernel が持つ threadgroup 配列の行数。ポインタで渡すので
// ここでは実行時の値でよい (threadgroup 配列は kernel スコープでしか置けない)。
static inline void qwen_delta_rule_body(
    device const half*  q,
    device const half*  k,
    device const half*  v,
    device const float* g,
    device const float* beta,
    device const float* stateIn,
    device half*        y,
    device float*       stateOut,
    constant QwenDeltaRuleParams& p,
    threadgroup half*   k_s,      // [TB][kQwenGDNKStride]
    threadgroup half*   q_s,      // [TB][kQwenGDNKStride]
    threadgroup half*   v_s,      // [TB][kQwenGDNVStride]
    threadgroup float*  g_s,      // [TB]
    threadgroup float*  b_s,      // [TB]
    int   TB,
    uint  tid,
    uint3 group)
{
    const int Dk  = kQwenGDNDk;
    const int Dv  = int(p.valueDim);
    const int Hv  = int(p.numVHeads);
    const int Hk  = int(p.numKHeads);
    const int T   = int(p.seqLen);

    const int blk = int(group.x);              // Dv/DB のどのブロックか
    const int hv  = int(group.y);              // v head
    const int hk  = hv / (Hv / Hk);            // k/q head (GQA の共有)
    const int dv0 = blk * kQwenGDNDB;

    const int dv  = int(tid) / 8;              // 0..DB-1
    const int seg = int(tid) % 8;              // 0..7
    const int d0  = seg * 16;

    device const half* k_base = k + (size_t)hk * Dk;
    device const half* q_base = q + (size_t)hk * Dk;
    device const half* v_base = v + (size_t)hv * Dv + dv0;
    const size_t krow = (size_t)Hk * Dk;       // トークン 1 個ぶんの q/k の幅
    const size_t vrow = (size_t)Hv * Dv;

    // 状態の断片をレジスタに読む: st[.] = S[hv][dv0+dv][d0 .. d0+16]
    float4 st[4];
    {
        device const float4* sIn = (device const float4*)(
            stateIn + (((size_t)hv * Dv) + dv0 + dv) * Dk + d0);
        for (int i = 0; i < 4; ++i) { st[i] = sIn[i]; }
    }

    device half* y_base = y + (size_t)hv * Dv + dv0;

    for (int t0 = 0; t0 < T; t0 += TB) {
        const int tt = min(TB, T - t0);
        // 協調ロード (coalesced)。1 threadgroup が同じ k/q 行を 1 度だけ読む。
        for (int i = int(tid); i < tt * Dk; i += 256) {
            const int r = i / Dk, d = i % Dk;
            k_s[r * kQwenGDNKStride + d] = k_base[(size_t)(t0 + r) * krow + d];
            q_s[r * kQwenGDNKStride + d] = q_base[(size_t)(t0 + r) * krow + d];
        }
        for (int i = int(tid); i < tt * kQwenGDNDB; i += 256) {
            const int r = i / kQwenGDNDB, d = i % kQwenGDNDB;
            v_s[r * kQwenGDNVStride + d] = v_base[(size_t)(t0 + r) * vrow + d];
        }
        for (int i = int(tid); i < tt; i += 256) {
            g_s[i] = g[(size_t)(t0 + i) * Hv + hv];
            b_s[i] = beta[(size_t)(t0 + i) * Hv + hv];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < tt; ++t) {
            const float gt = g_s[t];
            const float bt = b_s[t];
            const threadgroup half4* k4 =
                (const threadgroup half4*)(k_s + t * kQwenGDNKStride + d0);
            const threadgroup half4* q4 =
                (const threadgroup half4*)(q_s + t * kQwenGDNKStride + d0);

            float4 kf[4];
            for (int i = 0; i < 4; ++i) { kf[i] = float4(k4[i]); }

            // 減衰と kv_mem の縮約を 1 パスで (2 パスにしない)
            float4 acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                st[i] *= gt;
                acc += st[i] * kf[i];
            }
            const float kv_mem =
                qwen_gdn_row_reduce(acc.x + acc.y + acc.z + acc.w, tid);
            const float delta =
                (float(v_s[t * kQwenGDNVStride + dv]) - kv_mem) * bt;

            float4 out4 = 0.0f;
            for (int i = 0; i < 4; ++i) {
                st[i] += kf[i] * delta;
                out4 += st[i] * float4(q4[i]);
            }
            const float out =
                qwen_gdn_row_reduce(out4.x + out4.y + out4.z + out4.w, tid);
            if (seg == 0) {
                y_base[(size_t)(t0 + t) * vrow + dv] = half(out);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float4* sOut = (device float4*)(
        stateOut + (((size_t)hv * Dv) + dv0 + dv) * Dk + d0);
    for (int i = 0; i < 4; ++i) { sOut[i] = st[i]; }
}

#define QWEN_DELTA_RULE_KERNEL(NAME, TBV)                                      \
kernel void NAME(                                                              \
    device const half*  q         [[buffer(0)]],                               \
    device const half*  k         [[buffer(1)]],                               \
    device const half*  v         [[buffer(2)]],                               \
    device const float* g         [[buffer(3)]],                               \
    device const float* beta      [[buffer(4)]],                               \
    device const float* stateIn   [[buffer(5)]],                               \
    device half*        y         [[buffer(6)]],                               \
    device float*       stateOut  [[buffer(7)]],                               \
    constant QwenDeltaRuleParams& p [[buffer(8)]],                             \
    uint3 tid   [[thread_position_in_threadgroup]],                            \
    uint3 group [[threadgroup_position_in_grid]])                              \
{                                                                              \
    threadgroup half  k_s[(TBV) * kQwenGDNKStride];                            \
    threadgroup half  q_s[(TBV) * kQwenGDNKStride];                            \
    threadgroup half  v_s[(TBV) * kQwenGDNVStride];                            \
    threadgroup float g_s[(TBV)];                                              \
    threadgroup float b_s[(TBV)];                                              \
    qwen_delta_rule_body(q, k, v, g, beta, stateIn, y, stateOut, p,            \
                         k_s, q_s, v_s, g_s, b_s, (TBV), tid.x, group);        \
}

QWEN_DELTA_RULE_KERNEL(qwen_delta_rule_tb16, 16)
QWEN_DELTA_RULE_KERNEL(qwen_delta_rule_tb32, 32)
QWEN_DELTA_RULE_KERNEL(qwen_delta_rule_tb48, 48)
