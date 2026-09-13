#include <metal_stdlib>
using namespace metal;

// Qwen3.8-Flash-Next (qwen4exp) decode kernels, float32 activations.
//
// Transcribed from ds4-metal's CPU reference `qwen4_ref_*` (the same math as
// `Scripts/qwen38/reference_forward.py`), one token at a time, for the Q2
// verification runner. Mostly plain per-thread loops; the exceptions are the
// RMS scale and attention, where a thread looping over thousands of elements
// was the cost (a 4-thread RMS over 4x2560 took 340 us, attention at 2048
// tokens 120 ms per layer) and the loop is split into 32 SIMD lanes
// (`docs/qwen38/02-DECODE-SPEED.md` §2). The weights they read come straight from the
// DS4-IQ2 GGUF (norm gammas already folded to 1+w, ssm_a = -exp(A_log), GDN
// value head j paired with key head j % Hk).

static inline float q38_silu(float x) { return x / (1.0f + exp(-x)); }
static inline float q38_sigmoid(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + exp(-x)) : exp(x) / (1.0f + exp(x));
}
// MSL has no log1p; below -15 the float32 log(1 + e^x) rounds to 0, so take e^x there.
static inline float q38_softplus(float x) { return x > 20.0f ? x : (x < -15.0f ? exp(x) : log(1.0f + exp(x))); }

struct Q38RmsParams {
    uint groups;
    uint n;
    float eps;
};

struct Q38UnaryParams {
    uint op;        // 0: silu, 1: sigmoid
    float inScale;  // applied before the function
    float outScale; // applied after
};

/// scale[g] = rsqrt(mean(x[g]^2) + eps), one SIMD group (32 lanes) per group g.
/// Lane l sums the 32-element chunks l, l+32, ... of the group (so a group of n
/// elements needs no per-thread loop longer than n/32); lane 0 also takes the
/// tail past the last whole chunk. Dispatch: threadgroups (groups, 1, 1), 32 threads each.
kernel void q38_group_rms_scale(
    device const float* x [[buffer(0)]],
    device float* scale [[buffer(1)]],
    constant Q38RmsParams& p [[buffer(2)]],
    uint g [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint base = g * p.n;
    const uint nc = p.n / 32;
    float ss = 0.0f;
    for (uint c = lane; c < nc; c += 32) {
        device const float* xc = x + base + c * 32;
        for (uint j = 0; j < 32; ++j) ss += xc[j] * xc[j];
    }
    if (lane == 0) {
        for (uint i = nc * 32; i < p.n; ++i) ss += x[base + i] * x[base + i];
    }
    ss = simd_sum(ss);
    if (lane == 0) scale[g] = 1.0f / sqrt(ss / float(p.n) + p.eps);
}

/// out[i] = x[i] * scale[i / n] * gamma[i] (or gamma[i % n] if !wide). Thread per element.
kernel void q38_rms_apply(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant Q38RmsParams& p [[buffer(3)]],
    constant uint& gammaWide [[buffer(4)]],
    device const float* scale [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    out[i] = x[i] * scale[i / p.n] * gamma[gammaWide != 0 ? i : i % p.n];
}

kernel void q38_unary(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant Q38UnaryParams& p [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    const float v = x[i] * p.inScale;
    out[i] = p.outScale * (p.op == 0 ? q38_silu(v) : q38_sigmoid(v));
}

struct Q38HCParams {
    uint hc;
    uint e;
};

/// mixed[d] = mean_s sigmoid(gate[s*e + d]) * xn[s*e + d]
kernel void q38_hc_mix(
    device const float* xn [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device float* mixed [[buffer(2)]],
    constant Q38HCParams& p [[buffer(3)]],
    uint d [[thread_position_in_grid]]
) {
    float acc = 0.0f;
    for (uint s = 0; s < p.hc; ++s) acc += q38_sigmoid(gate[s * p.e + d]) * xn[s * p.e + d];
    mixed[d] = acc / float(p.hc);
}

/// R[s*e + d] += inj[s] * blk[d]
kernel void q38_hc_combine(
    device float* R [[buffer(0)]],
    device const float* blk [[buffer(1)]],
    device const float* inj [[buffer(2)]],
    constant Q38HCParams& p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    R[i] += inj[i / p.e] * blk[i % p.e];
}

/// a[i] += scale * b[i]; `scale` comes from `scaleBuf[0]` through `scaleOp`
/// (0: as is, 1: sigmoid) so the shared-expert gate never leaves the GPU.
kernel void q38_add_scaled(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* scaleBuf [[buffer(2)]],
    constant uint& scaleOp [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    const float s = scaleOp == 1 ? q38_sigmoid(scaleBuf[0]) : scaleBuf[0];
    a[i] += s * b[i];
}

/// g[i] = silu(g[i]) * u[i]
kernel void q38_silu_mul(
    device float* g [[buffer(0)]],
    device const float* u [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    g[i] = q38_silu(g[i]) * u[i];
}

// ---------------------------------------------------------------------------
// Gated DeltaNet, one token

/// conv[c] = silu(cw[c*K + K-1] * qkv[c] + sum_{k<K-1} cw[c*K + k] * hist[k*C + c]);
/// then hist shifts left by one row and takes qkv. `hist` is [K-1][C], oldest first.
kernel void q38_gdn_conv(
    device const float* qkv [[buffer(0)]],
    device float* hist [[buffer(1)]],
    device const float* cw [[buffer(2)]],
    device float* conv [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint c [[thread_position_in_grid]]
) {
    float acc = cw[c * K + (K - 1)] * qkv[c];
    for (uint k = 0; k + 1 < K; ++k) acc += cw[c * K + k] * hist[k * C + c];
    conv[c] = q38_silu(acc);
    for (uint k = 0; k + 2 < K; ++k) hist[k * C + c] = hist[(k + 1) * C + c];
    hist[(K - 2) * C + c] = qkv[c];
}

struct Q38GDNParams {
    uint hk;
    uint hv;
    uint d;
};

/// L2-normalize q (heads 0..<hk) and k (heads hk..<2hk) in place; q also scaled by 1/sqrt(d).
kernel void q38_gdn_qk_norm(
    device float* conv [[buffer(0)]],
    constant Q38GDNParams& p [[buffer(1)]],
    uint h [[thread_position_in_grid]]
) {
    const uint base = h * p.d;
    float ss = 0.0f;
    for (uint i = 0; i < p.d; ++i) ss += conv[base + i] * conv[base + i];
    float scale = 1.0f / sqrt(ss + 1e-6f);
    if (h < p.hk) scale *= 1.0f / sqrt(float(p.d));
    for (uint i = 0; i < p.d; ++i) conv[base + i] *= scale;
}

/// State S[hv][dv][dk]. Thread (dv, hv).
kernel void q38_gdn_step(
    device const float* conv [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device const float* A [[buffer(3)]],
    device const float* dt [[buffer(4)]],
    device float* S [[buffer(5)]],
    device float* o [[buffer(6)]],
    constant Q38GDNParams& p [[buffer(7)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint dv = pos.x, hv = pos.y, D = p.d;
    const uint kh = hv % p.hk;
    const float g = exp(A[hv] * q38_softplus(a[hv] + dt[hv]));
    const float beta = q38_sigmoid(b[hv]);
    device const float* q = conv + kh * D;
    device const float* k = conv + p.hk * D + kh * D;
    const float v = conv[2 * p.hk * D + hv * D + dv];
    device float* s = S + (hv * D + dv) * D;
    float kv = 0.0f;
    for (uint i = 0; i < D; ++i) {
        s[i] *= g;
        kv += s[i] * k[i];
    }
    const float delta = (v - kv) * beta;
    float acc = 0.0f;
    for (uint i = 0; i < D; ++i) {
        s[i] += k[i] * delta;
        acc += s[i] * q[i];
    }
    o[hv * D + dv] = acc;
}

/// o[h] = rms(o[h]) * nw * sigmoid(z[h]), per value head.
kernel void q38_gdn_norm_gate(
    device float* o [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const float* nw [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint h [[thread_position_in_grid]]
) {
    const uint base = h * D;
    float ss = 0.0f;
    for (uint i = 0; i < D; ++i) ss += o[base + i] * o[base + i];
    const float scale = 1.0f / sqrt(ss / float(D) + eps);
    for (uint i = 0; i < D; ++i) o[base + i] = o[base + i] * scale * nw[i] * q38_sigmoid(z[base + i]);
}

// ---------------------------------------------------------------------------
// Gated full attention, one token (QSA reduces to dense below 2049 visible tokens)

struct Q38AttnParams {
    uint h;
    uint hkv;
    uint d;
    uint nrot;
    uint pos;     // rope position / index of this token's cache row
    float eps;
};

static inline void q38_rms_rope(thread float* t, device const float* gamma, constant Q38AttnParams& p,
                                device const float* freq) {
    float ss = 0.0f;
    for (uint i = 0; i < p.d; ++i) ss += t[i] * t[i];
    const float scale = 1.0f / sqrt(ss / float(p.d) + p.eps);
    for (uint i = 0; i < p.d; ++i) t[i] = t[i] * scale * gamma[i];
    const uint half_ = p.nrot / 2;
    for (uint i = 0; i < half_; ++i) {
        const float theta = float(p.pos) * freq[i];
        const float c = cos(theta), s = sin(theta);
        const float x0 = t[i], x1 = t[i + half_];
        t[i] = x0 * c - x1 * s;
        t[i + half_] = x0 * s + x1 * c;
    }
}

/// Thread h < H: q[h] = rope(rms(qg[h][:D])), gate[h] = qg[h][D:].
/// Thread h >= H: k-cache row (pos, h-H) normed and roped in place.
kernel void q38_attn_prep(
    device const float* qg [[buffer(0)]],
    device float* kcache [[buffer(1)]],
    device const float* qnw [[buffer(2)]],
    device const float* knw [[buffer(3)]],
    device const float* freq [[buffer(4)]],
    device float* q [[buffer(5)]],
    device float* gate [[buffer(6)]],
    constant Q38AttnParams& p [[buffer(7)]],
    uint h [[thread_position_in_grid]]
) {
    float t[512];
    if (h < p.h) {
        for (uint i = 0; i < p.d; ++i) {
            t[i] = qg[h * 2 * p.d + i];
            gate[h * p.d + i] = qg[h * 2 * p.d + p.d + i];
        }
        q38_rms_rope(t, qnw, p, freq);
        for (uint i = 0; i < p.d; ++i) q[h * p.d + i] = t[i];
    } else {
        const uint base = (p.pos * p.hkv + (h - p.h)) * p.d;
        for (uint i = 0; i < p.d; ++i) t[i] = kcache[base + i];
        q38_rms_rope(t, knw, p, freq);
        for (uint i = 0; i < p.d; ++i) kcache[base + i] = t[i];
    }
}

// Attention over n tokens in five passes with no per-thread loop over n x D:
// score -> max -> sum -> weight -> mix. Token i of the pass is t = sel[i] when
// `useSel`, else i. scores[h * n + i] is scratch (score, then weight).

struct Q38AttnPassParams {
    uint h;
    uint hkv;
    uint d;
    uint n;
    uint useSel;
};

static inline uint q38_tok(device const uint* sel, constant Q38AttnPassParams& p, uint i) {
    return p.useSel != 0 ? sel[i] : i;
}

/// scores[h * n + i] = q[h] . k[t_i] / sqrt(D). Threadgroups (n, H), 32 threads; lane l reads dims l, l+32, ...
kernel void q38_attn_score(
    device const float* q [[buffer(0)]],
    device const float* kcache [[buffer(1)]],
    device float* scores [[buffer(2)]],
    device const uint* sel [[buffer(3)]],
    constant Q38AttnPassParams& p [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint i = tg.x, h = tg.y;
    if (i >= p.n || h >= p.h) return;
    const uint kvh = h / (p.h / p.hkv);
    device const float* qh = q + h * p.d;
    device const float* kt = kcache + (q38_tok(sel, p, i) * p.hkv + kvh) * p.d;
    float dot = 0.0f;
    for (uint j = lane; j < p.d; j += 32) dot += qh[j] * kt[j];
    dot = simd_sum(dot);
    if (lane == 0) scores[h * p.n + i] = dot / sqrt(float(p.d));
}

/// stat[h] = max_i scores[h * n + i] (op 0), or sum_i exp(scores - stat[h]) into stat2[h] (op 1).
/// Threadgroups (H, 1), 32 threads; lane l reads tokens l, l+32, ...
kernel void q38_attn_stat(
    device const float* scores [[buffer(0)]],
    device float* mx [[buffer(1)]],
    device float* sum [[buffer(2)]],
    constant Q38AttnPassParams& p [[buffer(3)]],
    constant uint& op [[buffer(4)]],
    uint h [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    device const float* s = scores + h * p.n;
    if (op == 0) {
        float m = -FLT_MAX;
        for (uint i = lane; i < p.n; i += 32) m = max(m, s[i]);
        m = simd_max(m);
        if (lane == 0) mx[h] = m;
    } else {
        const float m = mx[h];
        float acc = 0.0f;
        for (uint i = lane; i < p.n; i += 32) acc += exp(s[i] - m);
        acc = simd_sum(acc);
        if (lane == 0) sum[h] = acc;
    }
}

/// scores[h * n + i] = exp(scores - mx[h]) / sum[h]. Thread (i, h).
kernel void q38_attn_weight(
    device float* scores [[buffer(0)]],
    device const float* mx [[buffer(1)]],
    device const float* sum [[buffer(2)]],
    constant Q38AttnPassParams& p [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint k = pos.y * p.n + pos.x;
    scores[k] = exp(scores[k] - mx[pos.y]) / sum[pos.y];
}

/// o[h * D + j] = sigmoid(gate[h * D + j]) * sum_i w[h, i] v[t_i][j].
/// Threadgroups (D, H), 32 threads; lane l reads tokens l, l+32, ...
kernel void q38_attn_mix(
    device const float* w [[buffer(0)]],
    device const float* vcache [[buffer(1)]],
    device const float* gate [[buffer(2)]],
    device float* o [[buffer(3)]],
    device const uint* sel [[buffer(4)]],
    constant Q38AttnPassParams& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint j = tg.x, h = tg.y;
    if (j >= p.d || h >= p.h) return;
    const uint kvh = h / (p.h / p.hkv);
    device const float* wh = w + h * p.n;
    float acc = 0.0f;
    for (uint i = lane; i < p.n; i += 32) {
        acc += wh[i] * vcache[(q38_tok(sel, p, i) * p.hkv + kvh) * p.d + j];
    }
    acc = simd_sum(acc);
    if (lane == 0) o[h * p.d + j] = acc * q38_sigmoid(gate[h * p.d + j]);
}

// ---------------------------------------------------------------------------
// QSA indexer (ds4 `qwen4_ref_select`): blocks of 4 tokens, pooled raw keys, normed and roped at the
// block's first position; score = sum_h relu(q_h . key_b). Selection itself is on the host.

/// blockKeys[block] = rope(rms(mean(rawKeys[block*4 .. block*4+3])), block*4). One thread.
kernel void q38_idx_block_key(
    device const float* rawKeys [[buffer(0)]],
    device const float* gk [[buffer(1)]],
    device const float* freq [[buffer(2)]],
    device float* blockKeys [[buffer(3)]],
    constant Q38AttnParams& p [[buffer(4)]],   // d = 128, pos = block * 4
    uint unused [[thread_position_in_grid]]
) {
    float t[512];
    const uint base = p.pos;
    for (uint i = 0; i < p.d; ++i) {
        t[i] = (rawKeys[base * p.d + i] + rawKeys[(base + 1) * p.d + i] +
                rawKeys[(base + 2) * p.d + i] + rawKeys[(base + 3) * p.d + i]) / 4.0f;
    }
    q38_rms_rope(t, gk, p, freq);
    for (uint i = 0; i < p.d; ++i) blockKeys[(base / 4) * p.d + i] = t[i];
}

/// iq[h] = rope(rms(iq[h]), pos), in place. Thread per indexer head.
kernel void q38_idx_q_prep(
    device float* iq [[buffer(0)]],
    device const float* gq [[buffer(1)]],
    device const float* freq [[buffer(2)]],
    constant Q38AttnParams& p [[buffer(3)]],   // d = 128
    uint h [[thread_position_in_grid]]
) {
    float t[512];
    for (uint i = 0; i < p.d; ++i) t[i] = iq[h * p.d + i];
    q38_rms_rope(t, gq, p, freq);
    for (uint i = 0; i < p.d; ++i) iq[h * p.d + i] = t[i];
}

/// score[b] = sum_h relu(iq[h] . blockKeys[b]). Thread per block.
kernel void q38_idx_score(
    device const float* iq [[buffer(0)]],
    device const float* blockKeys [[buffer(1)]],
    device float* score [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    uint b [[thread_position_in_grid]]
) {
    float acc = 0.0f;
    for (uint h = 0; h < heads; ++h) {
        float dot = 0.0f;
        for (uint i = 0; i < d; ++i) dot += iq[h * d + i] * blockKeys[b * d + i];
        acc += max(dot, 0.0f);
    }
    score[b] = acc;
}
