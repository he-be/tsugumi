#include <metal_stdlib>
using namespace metal;

// ============================================================================
// qwen.metal — Qwen3.5-MoE (Ornith-1.5-35B-A3B) の新規カーネル
//
// `docs/qwen35moe/03-DESIGN.md` §2。本命の `qwen_delta_rule` だけは
// `gdn.metal` に別置き (threadgroup 配列の寸法が TB ごとに違うため)。
//
// **既存 `.metal` を触らない**のがこのファイルの存在理由。Gemma 4 の実測値を
// 凍結したまま第 2 のアーキテクチャを足す (`docs/qwen35moe/README.md` 運用ルール)。
// SiLU も共有ヘッダに出さずここにローカル定義する (`gelu_pytorch_tanh` が 4 箇所に
// 重複定義されている現状に合わせる)。
//
// 中身:
//   qwen_delta_qkv_prepare  線形注意 30 層の因果 depthwise conv1d + SiLU +
//                           q/k の l2norm + q の 1/sqrt(Dk)  (§2-5 と §3-6)
//   qwen_delta_gates        g = exp(-exp(A_log)·softplus(a+dt_bias)) と
//                           beta = sigmoid(b)                          (§3-6)
//   qwen_delta_norm_gate    RMSNormGated (**`+1` しない**) × silu(z)    (§2-7)
//   qwen_qkv_epilogue       full attention 10 層の q_norm/k_norm +
//                           **Qwen 規約の partial RoPE**               (§2-2)
//   qwen_attn_output_gate   o *= sigmoid(gate)                         (§2-3)
//   qwen_moe_shared_gate    shared *= sigmoid(w·x)                     (§2-4)
//   qwen_silu_mul           y = silu(gate) * up                        (§2-1)
// ============================================================================

constant constexpr uint kQwenMaxSimdGroups = 8;      // 256 スレッド / 32
constant constexpr uint kQwenMaxHeadDim    = 256;    // full attention の head_dim

// SiLU / sigmoid / softplus。参照器 (`Scripts/qwen35/reference_forward.py`) と
// 同じ算式にする。softplus は |x| が大きいときに exp が溢れないほうの形。
static inline float qwen_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float qwen_silu(float x)    { return x * qwen_sigmoid(x); }
// **MSL に log1p が無いので自前で持つ。**素朴な `log(1 + u)` は u が小さいとき
// 1 + u の丸めで有効数字を落とす: x = -14 なら u = exp(-14) = 8.3e-7 で、float32 の
// 1.0 近傍の刻み 6e-8 に対して 4 bit しか残らない。softplus はまさにその領域
// (減衰ゲートが 1 に貼り付く側) で使うので、Kahan の補正を掛ける。
// 実測: これを入れる前は g の誤差が float32 の床の 19 倍 (1.16e-06)、入れた後は床。
//
// `precise::` を付けるのは、softplus と exp が
// `g = exp(-exp(A_log)·softplus(·))` の二重指数の中にいて、g が再帰状態に毎
// トークン掛かるため (誤差はトークン数だけ積まれる)。
static inline float qwen_log1p(float u) {
    const float y = 1.0f + u;
    return (y == 1.0f) ? u : precise::log(y) * (u / (y - 1.0f));
}

static inline float qwen_softplus(float x) {
    return max(x, 0.0f) + qwen_log1p(precise::exp(-abs(x)));
}

// 256 スレッド以下の threadgroup の総和。`partial` は呼び手が持つ
// [kQwenMaxSimdGroups] の threadgroup 配列。戻り値は全スレッドに配られる。
static inline float qwen_block_sum(float value,
                                   threadgroup float* partial,
                                   uint lid,
                                   uint simd_lane_id,
                                   uint simd_group_id,
                                   uint simdgroups) {
    float acc = simd_sum(value);
    if (simd_lane_id == 0) { partial[simd_group_id] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) { partial[0] = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// ============================================================================
// qwen_delta_qkv_prepare — `in_proj_qkv` の出力から `qwen_delta_rule` の入力まで
//
//   conv = causal_depthwise_conv1d(qkv, w)   // C=8192, K=4, groups=C, bias 無し
//   conv = silu(conv)
//   q,k,v = split(conv, [2048, 2048, 4096])
//   q = l2norm(q) * Dk^-0.5,  k = l2norm(k),  v はそのまま
//
// **1 スレッド = 1 チャネル、1 threadgroup = 1 ヘッド × R トークン。**
// grid = (2Hk+Hv, ceil(T/R))。因果窓 4 のうち過去 3 個は【レジスタ】に持ち回る。
//
// **トークン方向にも切るのが肝。**depthwise conv の因果依存は窓 4 ぶんしかないので、
// トークンは本当は並列に出せる。切らずに 1 threadgroup で T 全部を回すと
// threadgroup が 64 個しか立たず、prefill 2048 で実測 40 GB/s (M3 Pro の帯域の
// 4 分の 1) しか出なかった。R トークンずつに切ると threadgroup 数は
// 64 × ceil(T/R) になる。読み直しは窓のぶんだけ増える ((R+3)/R、R=32 で +9%)。
//
// l2norm の縮約は 128 チャネル = 1 ヘッドが 1 threadgroup に収まる (Dk=Dv=128) から
// そこで閉じる。barrier はトークンあたり 2 個だが、threadgroup が十分あれば隠れる。
//
// 状態 `convState` は `[K-1, C]` の時系列順 (state[0] が最も古い)。参照器の
// `padded = concat(state, qkv)` と同じ並び。**`stateIn` と `stateOut` を分けてある**:
// 状態を書くのは最後のトークンブロックを持つ threadgroup、読むのは先頭ブロックの
// threadgroup で、両者の間に順序の保証が無い。同じバッファを渡してよいのは
// **トークンブロックが 1 個のとき (T <= R、decode は常にそう)** だけ。
//
// 分割の境界はブロック番号で決まる: 0..Hk-1 が q、Hk..2Hk-1 が k、以降が v。
// **`.gturbo` の `conv1d.weight` は `[C, K]`** (MLX の `[8192, 4, 1]` を squeeze
// したもの。上流 bf16 の `[8192, 1, 4]` とは軸順が違う — docs/qwen35moe/10 §4)。
// ============================================================================

struct QwenDeltaQKVParams {
    uint  seqLen;         // T
    uint  numKHeads;      // Hk = 16
    uint  numVHeads;      // Hv = 32
    uint  headDim;        // Dk = Dv = 128
    uint  convKernel;     // K = 4
    uint  tokensPerGroup; // R
    float l2Eps;          // 1e-6 (l2norm は mean ではなく和に足す)
    float qScale;         // Dk^-0.5
};

[[kernel, max_total_threads_per_threadgroup(128)]]
void qwen_delta_qkv_prepare(
    device const half*   qkv        [[buffer(0)]],   // [T, C] FP16
    device const bfloat* convWeight [[buffer(1)]],   // [C, K] BF16
    device const half*   stateIn    [[buffer(2)]],   // [K-1, C] FP16
    device       half*   stateOut   [[buffer(3)]],   // [K-1, C] FP16
    device       half*   q          [[buffer(4)]],   // [T, Hk, Dk] FP16
    device       half*   k          [[buffer(5)]],   // [T, Hk, Dk] FP16
    device       half*   v          [[buffer(6)]],   // [T, Hv, Dv] FP16
    constant QwenDeltaQKVParams& p  [[buffer(7)]],
    uint3 lid3          [[thread_position_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simdgroups    [[simdgroups_per_threadgroup]],
    uint3 group3        [[threadgroup_position_in_grid]]
) {
    threadgroup float partial[kQwenMaxSimdGroups];

    const uint lid = lid3.x;
    const uint2 group = group3.xy;
    const uint HD = p.headDim;
    const uint Hk = p.numKHeads;
    const uint Hv = p.numVHeads;
    const uint C  = (2u * Hk + Hv) * HD;
    const uint K  = p.convKernel;
    const uint T  = p.seqLen;
    // 1 threadgroup = HD スレッドちょうどで呼ぶ (呼び手が保証する)。早期 return を
    // 書くと l2norm の barrier に来ないスレッドができてしまうので、ここで返せるのは
    // **threadgroup 単位で揃う条件だけ**。
    const uint t0 = group.y * p.tokensPerGroup;
    if (t0 >= T) { return; }
    const uint tt = min(p.tokensPerGroup, T - t0);

    const uint block = group.x;
    const uint channel = block * HD + lid;
    const bool isQ = block < Hk;
    const bool isK = !isQ && block < 2u * Hk;
    const uint head = isQ ? block : (isK ? block - Hk : block - 2u * Hk);

    float w[4];
    for (uint j = 0; j < K; ++j) { w[j] = float(convWeight[channel * K + j]); }

    // 因果窓の初期値。自分のブロックより前のトークンは qkv から直接読み、
    // ブロック 0 の手前だけが `stateIn` を見る。
    float history[4];
    for (uint j = 0; j + 1 < K; ++j) {
        const int src = int(t0) - int(K - 1) + int(j);
        history[j] = src < 0
            ? float(stateIn[(uint)(src + int(K - 1)) * C + channel])
            : float(qkv[(size_t)src * C + channel]);
    }

    for (uint r = 0; r < tt; ++r) {
        const uint t = t0 + r;
        const float x = float(qkv[(size_t)t * C + channel]);
        float acc = x * w[K - 1];
        for (uint j = 0; j + 1 < K; ++j) { acc = fma(history[j], w[j], acc); }
        for (uint j = 0; j + 2 < K; ++j) { history[j] = history[j + 1]; }
        if (K >= 2) { history[K - 2] = x; }

        const float value = qwen_silu(acc);

        if (isQ || isK) {
            // l2norm は **平均ではなく和** に eps を足す (参照器 `l2_norm`)。
            const float sum = qwen_block_sum(value * value, partial, lid,
                                             simd_lane_id, simd_group_id, simdgroups);
            const float inv = rsqrt(sum + p.l2Eps);
            const float scaled = isQ ? value * inv * p.qScale : value * inv;
            device half* dst = isQ ? q : k;
            dst[((size_t)t * Hk + head) * HD + lid] = half(scaled);
        } else {
            v[((size_t)t * Hv + head) * HD + lid] = half(value);
        }
    }

    // 最後のトークンブロックを持つ threadgroup だけが状態を進める。history には
    // その時点で「直近 K-1 個の入力」がそのまま入っている (T < K-1 のときは
    // `stateIn` から引き継いだ値が残る) ので、書き戻すだけでよい。
    if (t0 + tt == T) {
        for (uint j = 0; j + 1 < K; ++j) {
            stateOut[j * C + channel] = half(history[j]);
        }
    }
}

// ============================================================================
// qwen_delta_gates — 減衰ゲートと beta
//
//   g[t,h]    = exp(-exp(A_log[h]) * softplus(a[t,h] + dt_bias[h]))
//   beta[t,h] = sigmoid(b[t,h])
//
// `qwen_delta_rule` は g を**掛ける値そのもの**として受け取る (指数を取った後)。
// 出力が FP32 なのは、g が 1 に貼り付く領域 (実測の α 中央値 0.78〜0.99、
// docs/qwen35moe/10 §5) で FP16 の刻み 2^-11 が効いてしまうため。
// ============================================================================

struct QwenDeltaGateParams {
    uint seqLen;
    uint numVHeads;
};

kernel void qwen_delta_gates(
    device const half*   a       [[buffer(0)]],   // [T, Hv] FP16 (in_proj_a)
    device const half*   b       [[buffer(1)]],   // [T, Hv] FP16 (in_proj_b)
    device const bfloat* aLog    [[buffer(2)]],   // [Hv] BF16
    device const bfloat* dtBias  [[buffer(3)]],   // [Hv] BF16
    device       float*  g       [[buffer(4)]],   // [T, Hv] FP32
    device       float*  beta    [[buffer(5)]],   // [T, Hv] FP32
    constant QwenDeltaGateParams& p [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.seqLen * p.numVHeads) { return; }
    const uint h = gid % p.numVHeads;
    const float dt = qwen_softplus(float(a[gid]) + float(dtBias[h]));
    g[gid] = precise::exp(-precise::exp(float(aLog[h])) * dt);
    beta[gid] = qwen_sigmoid(float(b[gid]));
}

// ============================================================================
// qwen_delta_norm_gate — `Qwen3_5MoeRMSNormGated`
//
//   h = o * rsqrt(mean(o^2) + eps) * weight        // **`1 + w` ではない**
//   h = h * silu(z)
//
// 30 層ぶんの `linear_attn.norm` はここだけが `+1` しない norm で、取り違えると
// 静かに壊れる (docs/qwen35moe/01-MODEL.md §3-1)。weight は head 方向に共有の
// `[Dv]` 1 本。z は `in_proj_z` の出力 `[T, Hv*Dv]` を head で切ったもの。
//
// 1 threadgroup = 1 (トークン, head)。grid = (Hv, T)。
// ============================================================================

struct QwenDeltaNormGateParams {
    uint  seqLen;
    uint  numVHeads;
    uint  headDim;
    float eps;
};

[[kernel, max_total_threads_per_threadgroup(128)]]
void qwen_delta_norm_gate(
    device const half*   o      [[buffer(0)]],   // [T, Hv, Dv] FP16
    device const half*   z      [[buffer(1)]],   // [T, Hv*Dv]  FP16
    device const bfloat* weight [[buffer(2)]],   // [Dv] BF16
    device       half*   out    [[buffer(3)]],   // [T, Hv*Dv]  FP16
    constant QwenDeltaNormGateParams& p [[buffer(4)]],
    uint3 lid3          [[thread_position_in_threadgroup]],
    uint3 lsize3        [[threads_per_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simdgroups    [[simdgroups_per_threadgroup]],
    uint3 group         [[threadgroup_position_in_grid]]
) {
    threadgroup float partial[kQwenMaxSimdGroups];

    const uint lid = lid3.x;
    const uint lsize = lsize3.x;
    const uint HD = p.headDim;
    const uint head = group.x;
    const uint token = group.y;
    if (head >= p.numVHeads || token >= p.seqLen) { return; }

    device const half* src = o + ((size_t)token * p.numVHeads + head) * HD;
    const size_t zbase = ((size_t)token * p.numVHeads + head) * HD;

    float acc = 0.0f;
    for (uint i = lid; i < HD; i += lsize) {
        const float value = float(src[i]);
        acc = fma(value, value, acc);
    }
    const float sum = qwen_block_sum(acc, partial, lid,
                                     simd_lane_id, simd_group_id, simdgroups);
    const float inv = rsqrt(sum / float(HD) + p.eps);

    for (uint i = lid; i < HD; i += lsize) {
        const float normed = float(src[i]) * inv * float(weight[i]);
        out[zbase + i] = half(normed * qwen_silu(float(z[zbase + i])));
    }
}

// ============================================================================
// qwen_qkv_epilogue — full attention 10 層の q_norm / k_norm + partial RoPE
//
// **Gemma の `fused_qkv_epilogue` を流用してはいけない** (docs/qwen35moe/03 §2-2):
//
//   Gemma : 組は (i, HD/2+i)、周波数の分母は HD、v にも no-scale norm
//   Qwen  : 回すのは先頭 rotary_dim=64 だけ、組は **(i, 32+i)**、分母は
//           **rotary_dim**、64..255 は無変更、**v には何も掛からない**
//
// q_proj は 2 倍幅 `[T, NQ, 2*HD]` で、ヘッドごとに前半が q、後半が
// `attn_output_gate` 用の gate (docs/qwen35moe/01-MODEL.md §3-2)。gate はここでは
// 触らない (`qwen_attn_output_gate` が attention の後で使う)。
//
// norm の weight は `1+w` 焼き込み済み、さらに q 側は `× head_dim^-0.5` を
// 焼いてある (`Scripts/qwen35/bake_snapshot.py`)。したがって attention の
// `scale` は 1.0 で呼ぶ。RoPE は回転なのでノルムを変えず、順序を入れ替えてよい。
//
// grid = (NQ + NKV, T)。正規化した値を **一度 half に丸めてから** RoPE を掛ける
// (Gemma と同じ。単体カーネルに割ったときと同じ数になるようにするため)。
// ============================================================================

struct QwenQKVEpilogueParams {
    uint  seqLen;
    uint  numQHeads;    // 16
    uint  numKVHeads;   // 2
    uint  headDim;      // 256
    uint  rotaryDim;    // 64 = head_dim * partial_rotary_factor
    uint  position;     // 先頭トークンの絶対位置
    float theta;        // 10,000,000
    float eps;          // 1e-6
};

[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_qkv_epilogue(
    device       half*   q       [[buffer(0)]],   // [T, NQ, 2*HD] FP16 (in-place)
    device       half*   k       [[buffer(1)]],   // [T, NKV, HD]  FP16 (in-place)
    device const bfloat* qWeight [[buffer(2)]],   // [HD] BF16
    device const bfloat* kWeight [[buffer(3)]],   // [HD] BF16
    constant QwenQKVEpilogueParams& p [[buffer(4)]],
    uint3 lid3          [[thread_position_in_threadgroup]],
    uint3 lsize3        [[threads_per_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simdgroups    [[simdgroups_per_threadgroup]],
    uint3 group         [[threadgroup_position_in_grid]]
) {
    threadgroup float partial[kQwenMaxSimdGroups];
    threadgroup half  staged[kQwenMaxHeadDim];

    const uint lid = lid3.x;
    const uint lsize = lsize3.x;
    const uint HD = p.headDim;
    const uint token = group.y;
    const uint slot = group.x;
    if (token >= p.seqLen || slot >= p.numQHeads + p.numKVHeads) { return; }

    const bool isQ = slot < p.numQHeads;
    device half* dst = isQ
        ? q + ((size_t)token * p.numQHeads + slot) * 2u * HD
        : k + ((size_t)token * p.numKVHeads + (slot - p.numQHeads)) * HD;
    device const bfloat* weight = isQ ? qWeight : kWeight;

    float acc = 0.0f;
    for (uint i = lid; i < HD; i += lsize) {
        const float value = float(dst[i]);
        acc = fma(value, value, acc);
    }
    const float sum = qwen_block_sum(acc, partial, lid,
                                     simd_lane_id, simd_group_id, simdgroups);
    const float inv = rsqrt(sum / float(HD) + p.eps);

    for (uint i = lid; i < HD; i += lsize) {
        staged[i] = half(float(dst[i]) * inv * float(weight[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rd = p.rotaryDim;
    const uint halfRotary = rd / 2u;
    const float position = float(p.position + token);
    for (uint i = lid; i < HD; i += lsize) {
        if (i < halfRotary) {
            const float exponent = -float(2u * i) / float(rd);
            const float angle = position * pow(p.theta, exponent);
            const float c = cos(angle);
            const float s = sin(angle);
            const float x0 = float(staged[i]);
            const float x1 = float(staged[halfRotary + i]);
            dst[i] = half(x0 * c - x1 * s);
            dst[halfRotary + i] = half(x1 * c + x0 * s);
        } else if (i >= rd) {
            dst[i] = staged[i];   // 回さない残り (64..255)
        }
    }
}

// ============================================================================
// qwen_attn_output_gate — `attn_output_gate`
//
//   o[t,h,d] *= sigmoid(gate[t,h,d]),  gate は q_proj 出力の後半 HD 次元
//
// Gemma に対応物が無い 1 行 (docs/qwen35moe/01-MODEL.md §3-2)。decode では
// elementwise 4096 個なので、結線のときに前後に畳んでよい。
// ============================================================================

struct QwenAttnGateParams {
    uint seqLen;
    uint numQHeads;
    uint headDim;
};

kernel void qwen_attn_output_gate(
    device       half* o     [[buffer(0)]],   // [T, NQ, HD] FP16 (in-place)
    device const half* qGate [[buffer(1)]],   // [T, NQ, 2*HD] FP16 (gate は後半)
    constant QwenAttnGateParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = p.seqLen * p.numQHeads * p.headDim;
    if (gid >= total) { return; }
    const uint d = gid % p.headDim;
    const uint head = (gid / p.headDim) % p.numQHeads;
    const uint token = gid / (p.headDim * p.numQHeads);
    const size_t gateIndex =
        ((size_t)token * p.numQHeads + head) * 2u * p.headDim + p.headDim + d;
    o[gid] = half(float(o[gid]) * qwen_sigmoid(float(qGate[gateIndex])));
}

// ============================================================================
// qwen_moe_shared_gate — shared expert の sigmoid ゲート
//
//   y *= sigmoid(dot(shared_expert_gate, x))       // x は MoE ブロックの入力
//
// `shared_expert_gate.weight` は `[1, 2048]`。1 threadgroup で内積を取って
// そのまま y を書き換える (docs/qwen35moe/01-MODEL.md §3-4)。
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_moe_shared_gate(
    device       half*   y      [[buffer(0)]],   // [D] FP16 (in-place)
    device const half*   x      [[buffer(1)]],   // [D] FP16
    device const bfloat* weight [[buffer(2)]],   // [D] BF16
    constant     uint&   D      [[buffer(3)]],
    uint  lid           [[thread_position_in_threadgroup]],
    uint  lsize         [[threads_per_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simdgroups    [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kQwenMaxSimdGroups];

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        acc = fma(float(x[i]), float(weight[i]), acc);
    }
    const float logit = qwen_block_sum(acc, partial, lid,
                                       simd_lane_id, simd_group_id, simdgroups);
    const float scale = qwen_sigmoid(logit);
    for (uint i = lid; i < D; i += lsize) {
        y[i] = half(float(y[i]) * scale);
    }
}

// ============================================================================
// qwen_silu_mul — SiLU ゲート付き MLP の中身
//
//   y = silu(gate) * up
//
// `grep -rni silu Sources/TurboFieldfare/Metal` は 0 件だった (Gemma は
// `gelu_pytorch_tanh`)。shared expert と routed expert の両方から呼ぶ。
// ============================================================================

kernel void qwen_silu_mul(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device       half* out  [[buffer(2)]],
    constant     uint& n    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    out[gid] = half(qwen_silu(float(gate[gid])) * float(up[gid]));
}
