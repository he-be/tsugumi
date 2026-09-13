#include <metal_stdlib>
using namespace metal;

// Qwen3.8-Flash-Next (qwen4exp) kernels, float32 activations, T >= 1 tokens.
//
// Transcribed from ds4-metal's CPU reference `qwen4_ref_*` (the same math as
// `Scripts/qwen38/reference_forward.py`) for the Q2 verification runner. Every
// per-token buffer holds T rows one after another; decode is T = 1 and prefill
// runs a chunk at once. The token axis is the last axis of each dispatch grid.
// The weights come straight from the DS4-IQ2 GGUF (norm gammas already folded
// to 1+w, ssm_a = -exp(A_log), GDN value head j paired with key head j % Hk).
//
// Loops over thousands of elements inside one thread are what made the first
// forms slow (a 4-thread RMS over 4x2560 took 340 us, attention at 2048 tokens
// 120 ms per layer), so sums over long axes run as 32 SIMD lanes reduced with
// `simd_sum` (`docs/qwen38/02-DECODE-SPEED.md` §2).

static inline float q38_silu(float x) { return x / (1.0f + exp(-x)); }
static inline float q38_sigmoid(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + exp(-x)) : exp(x) / (1.0f + exp(x));
}
// MSL has no log1p; below -15 the float32 log(1 + e^x) rounds to 0, so take e^x there.
static inline float q38_softplus(float x) { return x > 20.0f ? x : (x < -15.0f ? exp(x) : log(1.0f + exp(x))); }

struct Q38RmsParams {
    uint groups;    // all groups in the batch (tokens x groups per token)
    uint n;
    float eps;
    uint gammaLen;  // n, or groups-per-token x n for a per-group gamma
};

/// scale[g] = rsqrt(mean(x[g]^2) + eps), one SIMD group (32 lanes) per group g.
/// Lane l sums the 32-element chunks l, l+32, ... of the group; lane 0 also takes the
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

/// out[i] = x[i] * scale[i / n] * gamma[i % gammaLen]. Thread per element.
kernel void q38_rms_apply(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant Q38RmsParams& p [[buffer(3)]],
    device const float* scale [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    out[i] = x[i] * scale[i / p.n] * gamma[i % p.gammaLen];
}

struct Q38UnaryParams {
    uint op;        // 0: silu, 1: sigmoid
    float inScale;  // applied before the function
    float outScale; // applied after
};

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

/// mixed[t][d] = mean_s sigmoid(gate[t][s][d]) * xn[t][s][d]. Thread (d, t).
kernel void q38_hc_mix(
    device const float* xn [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device float* mixed [[buffer(2)]],
    constant Q38HCParams& p [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint d = pos.x, base = pos.y * p.hc * p.e;
    float acc = 0.0f;
    for (uint s = 0; s < p.hc; ++s) acc += q38_sigmoid(gate[base + s * p.e + d]) * xn[base + s * p.e + d];
    mixed[pos.y * p.e + d] = acc / float(p.hc);
}

/// R[t][s][d] += inj[t][s] * blk[t][d]. Thread per element of R.
kernel void q38_hc_combine(
    device float* R [[buffer(0)]],
    device const float* blk [[buffer(1)]],
    device const float* inj [[buffer(2)]],
    constant Q38HCParams& p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    const uint w = p.hc * p.e;
    R[i] += inj[i / p.e] * blk[(i / w) * p.e + i % p.e];
}

/// a[i] = (op 2: 0, else a[i]) + s * b[i], s = scale[i / n] (op 1, 2: sigmoid of it).
kernel void q38_add_scaled(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* scaleBuf [[buffer(2)]],
    constant uint& scaleOp [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    const float raw = scaleBuf[i / n];
    const float s = scaleOp == 0 ? raw : q38_sigmoid(raw);
    a[i] = (scaleOp == 2 ? 0.0f : a[i]) + s * b[i];
}

kernel void q38_silu_mul(
    device float* g [[buffer(0)]],
    device const float* u [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    g[i] = q38_silu(g[i]) * u[i];
}

// ---------------------------------------------------------------------------
// Gated DeltaNet

struct Q38ConvParams {
    uint C;   // channels per token
    uint K;   // kernel width; hist holds the K-1 previous tokens, oldest first
    uint T;
};

/// conv[t][c] = silu(sum_k cw[c][k] * x[t - (K-1-k)][c]), x before the batch read from hist.
/// Thread (c, t).
kernel void q38_gdn_conv(
    device const float* qkv [[buffer(0)]],
    device const float* hist [[buffer(1)]],
    device const float* cw [[buffer(2)]],
    device float* conv [[buffer(3)]],
    constant Q38ConvParams& p [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint c = pos.x, t = pos.y;
    float acc = 0.0f;
    for (uint k = 0; k < p.K; ++k) {
        const uint back = p.K - 1 - k;
        const float v = back <= t ? qkv[(t - back) * p.C + c] : hist[(p.K - 1 - (back - t)) * p.C + c];
        acc += cw[c * p.K + k] * v;
    }
    conv[t * p.C + c] = q38_silu(acc);
}

/// hist <- the last K-1 of (hist, qkv[0..<T]). Thread per channel; rows rewritten oldest
/// first, each reading either qkv or a later (not yet rewritten) hist row.
kernel void q38_gdn_conv_hist(
    device const float* qkv [[buffer(0)]],
    device float* hist [[buffer(1)]],
    constant Q38ConvParams& p [[buffer(2)]],
    uint c [[thread_position_in_grid]]
) {
    for (uint r = 0; r + 1 < p.K; ++r) {
        const uint j = p.K - 1 - r;  // tokens back from the end of the batch
        hist[r * p.C + c] = j <= p.T ? qkv[(p.T - j) * p.C + c] : hist[(r + p.T) * p.C + c];
    }
}

struct Q38GDNParams {
    uint hk;
    uint hv;
    uint d;
    uint T;
};

/// L2-normalize q (heads 0..<hk) and k (heads hk..<2hk) of each token in place;
/// q also scaled by 1/sqrt(d). Thread (h, t); a token's row starts with q then k.
kernel void q38_gdn_qk_norm(
    device float* conv [[buffer(0)]],
    constant Q38GDNParams& p [[buffer(1)]],
    constant uint& C [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint h = pos.x;
    const uint base = pos.y * C + h * p.d;
    float ss = 0.0f;
    for (uint i = 0; i < p.d; ++i) ss += conv[base + i] * conv[base + i];
    float scale = 1.0f / sqrt(ss + 1e-6f);
    if (h < p.hk) scale *= 1.0f / sqrt(float(p.d));
    for (uint i = 0; i < p.d; ++i) conv[base + i] *= scale;
}

/// In place: a[t][hv] <- exp(A[hv] * softplus(a + dt[hv])) (decay), b[t][hv] <- sigmoid(b) (beta).
/// Thread (hv, t).
kernel void q38_gdn_gates(
    device float* a [[buffer(0)]],
    device float* b [[buffer(1)]],
    device const float* A [[buffer(2)]],
    device const float* dt [[buffer(3)]],
    constant Q38GDNParams& p [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint hv = pos.x, k = pos.y * p.hv + hv;
    a[k] = exp(A[hv] * q38_softplus(a[k] + dt[hv]));
    b[k] = q38_sigmoid(b[k]);
}

/// State S[hv][dv][dk], stepped through the T tokens in order (decay and beta from
/// `q38_gdn_gates`). One SIMD group per (dv, hv):
/// lane l owns key dims l, l+32, l+64, l+96, and the k.S and q.S sums of each step are
/// `simd_sum`s across the lanes, so no thread loops over T x 128.
/// Threadgroups (Dl, Hv), 32 threads.
kernel void q38_gdn_step(
    device const float* conv [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* S [[buffer(5)]],
    device float* o [[buffer(6)]],
    constant Q38GDNParams& p [[buffer(7)]],
    constant uint& C [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint dv = tg.x, hv = tg.y, D = p.d;
    const uint kh = hv % p.hk;
    device float* s = S + (hv * D + dv) * D;
    for (uint t = 0; t < p.T; ++t) {
        const float g = a[t * p.hv + hv];
        const float beta = b[t * p.hv + hv];
        device const float* row = conv + t * C;
        device const float* q = row + kh * D;
        device const float* k = row + p.hk * D + kh * D;
        const float v = row[2 * p.hk * D + hv * D + dv];
        float kv = 0.0f;
        for (uint i = lane; i < D; i += 32) {
            s[i] *= g;
            kv += s[i] * k[i];
        }
        kv = simd_sum(kv);
        const float delta = (v - kv) * beta;
        float acc = 0.0f;
        for (uint i = lane; i < D; i += 32) {
            s[i] += k[i] * delta;
            acc += s[i] * q[i];
        }
        acc = simd_sum(acc);
        if (lane == 0) o[(t * p.hv + hv) * D + dv] = acc;
    }
}

/// o[t][h] = rms(o[t][h]) * nw * sigmoid(z[t][h]), per value head. Thread (h, t).
kernel void q38_gdn_norm_gate(
    device float* o [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const float* nw [[buffer(2)]],
    constant Q38GDNParams& p [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint D = p.d;
    const uint base = (pos.y * p.hv + pos.x) * D;
    float ss = 0.0f;
    for (uint i = 0; i < D; ++i) ss += o[base + i] * o[base + i];
    const float scale = 1.0f / sqrt(ss / float(D) + eps);
    for (uint i = 0; i < D; ++i) o[base + i] = o[base + i] * scale * nw[i] * q38_sigmoid(z[base + i]);
}

// ---------------------------------------------------------------------------
// Gated full attention with QSA

struct Q38AttnParams {
    uint h;
    uint hkv;
    uint d;
    uint nrot;
    uint pos;     // position of token 0 of the batch (rope position / cache row)
    float eps;
};

static inline void q38_rms_rope(thread float* t, device const float* gamma, uint d, uint nrot, uint pos,
                                float eps, device const float* freq) {
    float ss = 0.0f;
    for (uint i = 0; i < d; ++i) ss += t[i] * t[i];
    const float scale = 1.0f / sqrt(ss / float(d) + eps);
    for (uint i = 0; i < d; ++i) t[i] = t[i] * scale * gamma[i];
    const uint half_ = nrot / 2;
    for (uint i = 0; i < half_; ++i) {
        const float theta = float(pos) * freq[i];
        const float c = cos(theta), s = sin(theta);
        const float x0 = t[i], x1 = t[i + half_];
        t[i] = x0 * c - x1 * s;
        t[i + half_] = x0 * s + x1 * c;
    }
}

/// Thread (h, t), h < H: q[t][h] = rope(rms(qg[t][h][:D])), gate[t][h] = qg[t][h][D:].
/// h >= H: k-cache row (pos + t, h - H) normed and roped in place.
kernel void q38_attn_prep(
    device const float* qg [[buffer(0)]],
    device float* kcache [[buffer(1)]],
    device const float* qnw [[buffer(2)]],
    device const float* knw [[buffer(3)]],
    device const float* freq [[buffer(4)]],
    device float* q [[buffer(5)]],
    device float* gate [[buffer(6)]],
    constant Q38AttnParams& p [[buffer(7)]],
    uint2 hp [[thread_position_in_grid]]
) {
    const uint h = hp.x, tok = hp.y, D = p.d;
    const uint at = p.pos + tok;
    float t[512];
    if (h < p.h) {
        const uint src = (tok * p.h + h) * 2 * D;
        const uint dst = (tok * p.h + h) * D;
        for (uint i = 0; i < D; ++i) {
            t[i] = qg[src + i];
            gate[dst + i] = qg[src + D + i];
        }
        q38_rms_rope(t, qnw, D, p.nrot, at, p.eps, freq);
        for (uint i = 0; i < D; ++i) q[dst + i] = t[i];
    } else {
        const uint base = (at * p.hkv + (h - p.h)) * D;
        for (uint i = 0; i < D; ++i) t[i] = kcache[base + i];
        q38_rms_rope(t, knw, D, p.nrot, at, p.eps, freq);
        for (uint i = 0; i < D; ++i) kcache[base + i] = t[i];
    }
}

// Attention over n_t tokens for query t in five passes with no per-thread loop over n x D:
// score -> max -> sum -> weight -> mix. Token i of query t is sel[t * nCap + i] when
// useSel[t], else i. scores[(t * H + h) * nCap + i] is scratch (score, then weight).

struct Q38AttnPassParams {
    uint h;
    uint hkv;
    uint d;
    uint nCap;
    uint T;
};

static inline uint q38_tok(device const uint* sel, device const uint* useSel, constant Q38AttnPassParams& p,
                           uint t, uint i) {
    return useSel[t] != 0 ? sel[t * p.nCap + i] : i;
}

/// scores = q[t][h] . k[tok] / sqrt(D). Threadgroups (nMax, H, T), 32 threads; lane l reads dims l, l+32, ...
kernel void q38_attn_score(
    device const float* q [[buffer(0)]],
    device const float* kcache [[buffer(1)]],
    device float* scores [[buffer(2)]],
    device const uint* sel [[buffer(3)]],
    constant Q38AttnPassParams& p [[buffer(4)]],
    device const uint* nq [[buffer(5)]],
    device const uint* useSel [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint i = tg.x, h = tg.y, t = tg.z;
    if (h >= p.h || t >= p.T || i >= nq[t]) return;
    const uint kvh = h / (p.h / p.hkv);
    device const float* qh = q + (t * p.h + h) * p.d;
    device const float* kt = kcache + (q38_tok(sel, useSel, p, t, i) * p.hkv + kvh) * p.d;
    float dot = 0.0f;
    for (uint j = lane; j < p.d; j += 32) dot += qh[j] * kt[j];
    dot = simd_sum(dot);
    if (lane == 0) scores[(t * p.h + h) * p.nCap + i] = dot / sqrt(float(p.d));
}

/// op 0: mx[t][h] = max_i scores; op 1: sum[t][h] = sum_i exp(scores - mx).
/// Threadgroups (H, T), 32 threads; lane l reads tokens l, l+32, ...
kernel void q38_attn_stat(
    device const float* scores [[buffer(0)]],
    device float* mx [[buffer(1)]],
    device float* sum [[buffer(2)]],
    constant Q38AttnPassParams& p [[buffer(3)]],
    constant uint& op [[buffer(4)]],
    device const uint* nq [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint h = tg.x, t = tg.y;
    if (h >= p.h || t >= p.T) return;
    const uint n = nq[t];
    device const float* s = scores + (t * p.h + h) * p.nCap;
    if (op == 0) {
        float m = -FLT_MAX;
        for (uint i = lane; i < n; i += 32) m = max(m, s[i]);
        m = simd_max(m);
        if (lane == 0) mx[t * p.h + h] = m;
    } else {
        const float m = mx[t * p.h + h];
        float acc = 0.0f;
        for (uint i = lane; i < n; i += 32) acc += exp(s[i] - m);
        acc = simd_sum(acc);
        if (lane == 0) sum[t * p.h + h] = acc;
    }
}

/// scores = exp(scores - mx) / sum, and 0 past the query's n. Thread (i, t * H + h).
kernel void q38_attn_weight(
    device float* scores [[buffer(0)]],
    device const float* mx [[buffer(1)]],
    device const float* sum [[buffer(2)]],
    constant Q38AttnPassParams& p [[buffer(3)]],
    device const uint* nq [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint th = pos.y;
    const uint k = th * p.nCap + pos.x;
    scores[k] = pos.x < nq[th / p.h] ? exp(scores[k] - mx[th]) / sum[th] : 0.0f;
}

// The matrix form of the same attention for a batch whose queries all see the plain
// causal prefix (no QSA selection): per KV group g, Q_g = [T x H/Hkv][D] and K_g, V_g =
// [n][D] are gathered into contiguous rows, scores = Q_g K_g^T / sqrt(D) and out = W V_g^T
// run as MPS sgemm, and the weight pass above zeroes the causal tail.

/// qg[(t * G + h) * D + j] = q[(t * H + g * G + h) * D + j], G = H / Hkv. Thread per element.
kernel void q38_attn_gather_q(
    device const float* q [[buffer(0)]],
    device float* qg [[buffer(1)]],
    constant Q38AttnPassParams& p [[buffer(2)]],
    constant uint& g [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    const uint G = p.h / p.hkv;
    const uint row = i / p.d, j = i % p.d;
    const uint t = row / G, h = row % G;
    qg[i] = q[(t * p.h + g * G + h) * p.d + j];
}

/// kg[i * D + j] = cache[(i * Hkv + g) * D + j]. Thread per element.
kernel void q38_attn_gather_kv(
    device const float* cache [[buffer(0)]],
    device float* kg [[buffer(1)]],
    constant Q38AttnPassParams& p [[buffer(2)]],
    constant uint& g [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    kg[i] = cache[((i / p.d) * p.hkv + g) * p.d + i % p.d];
}

/// o[(t * H + g * G + h) * D + j] = og[(t * G + h) * D + j] * sigmoid(gate[same]). Thread per element of og.
kernel void q38_attn_scatter_out(
    device const float* og [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device float* o [[buffer(2)]],
    constant Q38AttnPassParams& p [[buffer(3)]],
    constant uint& g [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    const uint G = p.h / p.hkv;
    const uint row = i / p.d, j = i % p.d;
    const uint k = ((row / G) * p.h + g * G + row % G) * p.d + j;
    o[k] = og[i] * q38_sigmoid(gate[k]);
}

/// o[t][h][j] = sigmoid(gate[t][h][j]) * sum_i w[t][h][i] v[tok][j].
/// Threadgroups (D, H, T), 32 threads; lane l reads tokens l, l+32, ...
kernel void q38_attn_mix(
    device const float* w [[buffer(0)]],
    device const float* vcache [[buffer(1)]],
    device const float* gate [[buffer(2)]],
    device float* o [[buffer(3)]],
    device const uint* sel [[buffer(4)]],
    constant Q38AttnPassParams& p [[buffer(5)]],
    device const uint* nq [[buffer(6)]],
    device const uint* useSel [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint j = tg.x, h = tg.y, t = tg.z;
    if (j >= p.d || h >= p.h || t >= p.T) return;
    const uint n = nq[t];
    const uint kvh = h / (p.h / p.hkv);
    device const float* wh = w + (t * p.h + h) * p.nCap;
    float acc = 0.0f;
    for (uint i = lane; i < n; i += 32) {
        acc += wh[i] * vcache[(q38_tok(sel, useSel, p, t, i) * p.hkv + kvh) * p.d + j];
    }
    acc = simd_sum(acc);
    const uint k = (t * p.h + h) * p.d + j;
    if (lane == 0) o[k] = acc * q38_sigmoid(gate[k]);
}

// ---------------------------------------------------------------------------
// QSA indexer (ds4 `qwen4_ref_select`): blocks of 4 tokens, pooled raw keys, normed and roped at the
// block's first position; score = sum_h relu(q_h . key_b). Selection itself is on the host.

/// blockKeys[b] = rope(rms(mean(rawKeys[4b .. 4b+3])), 4b) for b = pos + u. Thread u.
kernel void q38_idx_block_key(
    device const float* rawKeys [[buffer(0)]],
    device const float* gk [[buffer(1)]],
    device const float* freq [[buffer(2)]],
    device float* blockKeys [[buffer(3)]],
    constant Q38AttnParams& p [[buffer(4)]],   // d = 128, pos = first block
    uint u [[thread_position_in_grid]]
) {
    float t[512];
    const uint b = p.pos + u, base = 4 * b;
    for (uint i = 0; i < p.d; ++i) {
        t[i] = (rawKeys[base * p.d + i] + rawKeys[(base + 1) * p.d + i] +
                rawKeys[(base + 2) * p.d + i] + rawKeys[(base + 3) * p.d + i]) / 4.0f;
    }
    q38_rms_rope(t, gk, p.d, p.nrot, base, p.eps, freq);
    for (uint i = 0; i < p.d; ++i) blockKeys[b * p.d + i] = t[i];
}

/// iq[t][h] = rope(rms(iq[t][h]), pos + t), in place. Thread (h, t).
kernel void q38_idx_q_prep(
    device float* iq [[buffer(0)]],
    device const float* gq [[buffer(1)]],
    device const float* freq [[buffer(2)]],
    constant Q38AttnParams& p [[buffer(3)]],   // d = 128, h = indexer heads
    uint2 hp [[thread_position_in_grid]]
) {
    float t[512];
    const uint base = (hp.y * p.h + hp.x) * p.d;
    for (uint i = 0; i < p.d; ++i) t[i] = iq[base + i];
    q38_rms_rope(t, gq, p.d, p.nrot, p.pos + hp.y, p.eps, freq);
    for (uint i = 0; i < p.d; ++i) iq[base + i] = t[i];
}

/// score[t][b] = sum_h relu(iq[t][h] . blockKeys[b]). Thread (b, t).
kernel void q38_idx_score(
    device const float* iq [[buffer(0)]],
    device const float* blockKeys [[buffer(1)]],
    device float* score [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    constant uint& nBlocks [[buffer(5)]],   // row width of `score`
    uint2 pos [[thread_position_in_grid]]
) {
    const uint b = pos.x, t = pos.y;
    float acc = 0.0f;
    for (uint h = 0; h < heads; ++h) {
        float dot = 0.0f;
        for (uint i = 0; i < d; ++i) dot += iq[(t * heads + h) * d + i] * blockKeys[b * d + i];
        acc += max(dot, 0.0f);
    }
    score[t * nBlocks + b] = acc;
}

// ---------------------------------------------------------------------------
// PLE (layer-1 entry): the per-token n-gram embedding rows come from the host; the rest is here.

/// g[t][s] = sigmoid(sign(z) sqrt(max(|z|, 1e-6))), z = keyn[t][s] . query[t][s] / sqrt(e).
/// Threadgroups (hc * T, 1), 32 threads; lane l reads 32-element chunks l, l+32, ...
kernel void q38_ple_gate(
    device const float* keyn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device float* gate [[buffer(2)]],
    constant uint& e [[buffer(3)]],
    uint g [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint base = g * e, nc = e / 32;
    float dot = 0.0f;
    for (uint c = lane; c < nc; c += 32) {
        for (uint j = 0; j < 32; ++j) dot += keyn[base + c * 32 + j] * query[base + c * 32 + j];
    }
    dot = simd_sum(dot);
    if (lane == 0) {
        const float z = dot / sqrt(float(e));
        const float mag = sqrt(max(abs(z), 1e-6f));
        gate[g] = q38_sigmoid(z > 0.0f ? mag : (z < 0.0f ? -mag : 0.0f));
    }
}

/// gated[t][s][d] = gate[t][s] * value[t][d]. Thread per element.
kernel void q38_ple_gated(
    device const float* gate [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* gated [[buffer(2)]],
    constant Q38HCParams& p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    gated[i] = gate[i / p.e] * value[(i / (p.hc * p.e)) * p.e + i % p.e];
}

struct Q38PleConvParams {
    uint W;         // hc * e
    uint dilation;  // n-gram size
    uint histRows;  // (K - 1) * dilation previous tokens, oldest first
    uint T;
};

/// R[t][c] += gated[t][c] + silu(sum_k cw[c][k] * normed[t - (3-k) * dilation][c]),
/// rows before the batch read from hist. Thread (c, t).
kernel void q38_ple_conv_add(
    device float* R [[buffer(0)]],
    device const float* gated [[buffer(1)]],
    device const float* normed [[buffer(2)]],
    device const float* hist [[buffer(3)]],
    device const float* cw [[buffer(4)]],
    constant Q38PleConvParams& p [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint c = pos.x, t = pos.y;
    float acc = 0.0f;
    for (uint k = 0; k < 4; ++k) {
        const uint back = (3 - k) * p.dilation;
        const float v = back <= t ? normed[(t - back) * p.W + c] : hist[(p.histRows - (back - t)) * p.W + c];
        acc += cw[c * 4 + k] * v;
    }
    R[t * p.W + c] += gated[t * p.W + c] + q38_silu(acc);
}

/// hist <- the last histRows of (hist, normed[0..<T]). Thread per channel.
kernel void q38_ple_hist(
    device const float* normed [[buffer(0)]],
    device float* hist [[buffer(1)]],
    constant Q38PleConvParams& p [[buffer(2)]],
    uint c [[thread_position_in_grid]]
) {
    for (uint r = 0; r < p.histRows; ++r) {
        const uint j = p.histRows - r;
        hist[r * p.W + c] = j <= p.T ? normed[(p.T - j) * p.W + c] : hist[(r + p.T) * p.W + c];
    }
}
