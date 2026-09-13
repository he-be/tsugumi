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
/// `snap_t` < T also copies the state after token snap_t into `snap` (same layout): the speculative
/// rollback's state after the first verified token (docs/qwen38/10 §5-4). UINT_MAX: no copy.
kernel void q38_gdn_step(
    device const float* conv [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* S [[buffer(5)]],
    device float* o [[buffer(6)]],
    constant Q38GDNParams& p [[buffer(7)]],
    constant uint& C [[buffer(8)]],
    device float* snap [[buffer(9)]],
    constant uint& snap_t [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint dv = tg.x, hv = tg.y, D = p.d;
    const uint kh = hv % p.hk;
    device float* s = S + (hv * D + dv) * D;
    device float* sn = snap + (hv * D + dv) * D;
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
        if (t == snap_t) {
            for (uint i = lane; i < D; i += 32) sn[i] = s[i];
        }
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
// GDN step, chunked (WY) form for prefill (`Qwen38GDNChunk`, docs/qwen38/07)
//
// Per value head, a chunk of L tokens from state S0 (Dv x Dk), with g_t the decay and
// r(t, s) = g_{s+1} ... g_t, gamma_t = r(t, -1):
//   u_t = beta_t (v_t - gamma_t S0 k_t) - sum_{s<t} beta_t r(t, s) (k_s.k_t) u_s   -> (I + A) U = R
//   o_t = gamma_t S0 q_t + sum_{s<=t} r(t, s) (k_s.q_t) u_s                          -> O = gamma Y + M U
//   S_L = gamma_{L-1} S0 + sum_s r(L-1, s) u_s k_s^T
// Chunk buffers are laid out [hv][rows][cols] with the chunk's L as the row stride.

struct Q38WYParams {
    uint hk;
    uint hv;
    uint d;
    uint L;
    uint t0;   // first token of the chunk in the batch
    uint C;    // conv row width
};

/// kq[hv] = (k_0..k_{L-1}, q_0..q_{L-1}), k[hv] = k rows, value head hv paired with key head hv % hk.
/// Thread (dk, t, hv).
kernel void q38_wy_gather(
    device const float* conv [[buffer(0)]],
    device float* kq [[buffer(1)]],
    device float* k [[buffer(2)]],
    constant Q38WYParams& p [[buffer(3)]],
    uint3 pos [[thread_position_in_grid]]
) {
    const uint dk = pos.x, t = pos.y, hv = pos.z, D = p.d, L = p.L;
    const uint kh = hv % p.hk;
    device const float* row = conv + (p.t0 + t) * p.C;
    const float kv = row[(p.hk + kh) * D + dk];
    kq[(hv * 2 * L + t) * D + dk] = kv;
    kq[(hv * 2 * L + L + t) * D + dk] = row[kh * D + dk];
    k[(hv * L + t) * D + dk] = kv;
}

/// From G[hv] = kq k^T ([2L][L]): row t of A = I + (beta_t r(t, s) k_s.k_t)_{s<t}, of
/// M = (r(t, s) k_s.q_t)_{s<=t}, gamma_t, and for t = L-1 the state weights w_s = r(L-1, s).
/// Thread (t, hv); the loop over s is at most L.
kernel void q38_wy_tri(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* G [[buffer(2)]],
    device float* A [[buffer(3)]],
    device float* M [[buffer(4)]],
    device float* gam [[buffer(5)]],
    device float* w [[buffer(6)]],
    constant Q38WYParams& p [[buffer(7)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint t = pos.x, hv = pos.y, L = p.L;
    const uint row = (hv * L + t) * L;
    device const float* gk = G + (hv * 2 * L + t) * L;
    device const float* gq = G + (hv * 2 * L + L + t) * L;
    const float beta = b[(p.t0 + t) * p.hv + hv];
    for (uint s = t + 1; s < L; ++s) {
        A[row + s] = 0.0f;
        M[row + s] = 0.0f;
    }
    float r = 1.0f;
    for (int s = int(t); s >= 0; --s) {
        M[row + s] = r * gq[s];
        A[row + s] = uint(s) == t ? 1.0f : beta * r * gk[s];
        if (t == L - 1) w[hv * L + s] = r;
        r *= a[(p.t0 + s) * p.hv + hv];
    }
    gam[hv * L + t] = r;
}

/// X[hv] = A[hv]^-1 for the unit lower-triangular A from `q38_wy_tri`, one column per thread (c, hv):
/// x[i][c] = -sum_{s=c..i-1} A[i][s] x[s][c]. (MPSMatrixSolveTriangular solves only the first matrix of a
/// batch, and 48 separate solves per chunk were 165 ms of a 190 ms layer at T = 4096.) L <= 512.
kernel void q38_wy_tinv(
    device const float* A [[buffer(0)]],
    device float* X [[buffer(1)]],
    constant Q38WYParams& p [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint c = pos.x, hv = pos.y, L = p.L;
    const uint base = hv * L * L;
    float col[512];
    for (uint i = 0; i < c; ++i) X[base + i * L + c] = 0.0f;
    col[c] = 1.0f;
    X[base + c * L + c] = 1.0f;
    for (uint i = c + 1; i < L; ++i) {
        device const float* ai = A + base + i * L;
        float acc = 0.0f;
        for (uint s = c; s < i; ++s) acc += ai[s] * col[s];
        col[i] = -acc;
        X[base + i * L + c] = -acc;
    }
}

/// R[hv][t] = beta_t (v_t - gamma_t S0 k_t), with S0 k_t in XY[hv][t]. Thread (dv, t, hv).
kernel void q38_wy_rhs(
    device const float* conv [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* gam [[buffer(2)]],
    device const float* XY [[buffer(3)]],
    device float* R [[buffer(4)]],
    constant Q38WYParams& p [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]
) {
    const uint dv = pos.x, t = pos.y, hv = pos.z, D = p.d, L = p.L;
    const float v = conv[(p.t0 + t) * p.C + (2 * p.hk + hv) * D + dv];
    const float beta = b[(p.t0 + t) * p.hv + hv];
    R[(hv * L + t) * D + dv] = beta * (v - gam[hv * L + t] * XY[(hv * 2 * L + t) * D + dv]);
}

/// o[t0 + t][hv] = (M U)[hv][t] + gamma_t S0 q_t (XY[hv][L + t]). Thread (dv, t, hv).
kernel void q38_wy_out(
    device const float* MU [[buffer(0)]],
    device const float* XY [[buffer(1)]],
    device const float* gam [[buffer(2)]],
    device float* o [[buffer(3)]],
    constant Q38WYParams& p [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]]
) {
    const uint dv = pos.x, t = pos.y, hv = pos.z, D = p.d, L = p.L;
    o[((p.t0 + t) * p.hv + hv) * D + dv] = MU[(hv * L + t) * D + dv]
        + gam[hv * L + t] * XY[(hv * 2 * L + L + t) * D + dv];
}

/// k[hv][s] *= w_s (the chunk's k rows turned into the state update's right factor). Thread (dk, s, hv).
kernel void q38_wy_kw(
    device float* k [[buffer(0)]],
    device const float* w [[buffer(1)]],
    constant Q38WYParams& p [[buffer(2)]],
    uint3 pos [[thread_position_in_grid]]
) {
    const uint dk = pos.x, s = pos.y, hv = pos.z;
    k[(hv * p.L + s) * p.d + dk] *= w[hv * p.L + s];
}

/// S0[hv] *= gamma_{L-1}. Thread (dk, dv, hv).
kernel void q38_wy_decay_state(
    device float* S [[buffer(0)]],
    device const float* gam [[buffer(1)]],
    constant Q38WYParams& p [[buffer(2)]],
    uint3 pos [[thread_position_in_grid]]
) {
    const uint dk = pos.x, dv = pos.y, hv = pos.z;
    S[(hv * p.d + dv) * p.d + dk] *= gam[hv * p.L + p.L - 1];
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
// block's first position; score = sum_h relu(q_h . key_b). Small batches select on the host, large ones below.

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

// The GPU form of the selection for a batch (no host sort): the per-head dots come from one
// sgemm (iq [rows x heads][d] x blockKeys [nb][d]^T), top-k is a bitwise search per query, and
// the attention runs over the union of a sub-batch's selected tokens with the rest masked.

/// score[t][b] = sum_h relu(dots[(t * heads + h) * nb + b]). Thread (b, t), t local to `dots`.
kernel void q38_idx_relu_sum(
    device const float* dots [[buffer(0)]],
    device float* score [[buffer(1)]],
    constant uint& heads [[buffer(2)]],
    constant uint& nb [[buffer(3)]],        // row width of both
    uint2 pos [[thread_position_in_grid]]
) {
    const uint b = pos.x, t = pos.y;
    float acc = 0.0f;
    for (uint h = 0; h < heads; ++h) acc += max(dots[(t * heads + h) * nb + b], 0.0f);
    score[t * nb + b] = acc;
}

struct Q38TopKParams {
    uint nb;    // row width of the scores
    uint k;     // blocks to keep
    uint pos;   // position of query 0
    uint T;
};

/// Query t (position pos + t) has n = (pos + t + 1) / 4 complete blocks. Block b < n is selected when
/// bits(score) > thr[t], or == thr[t] and b <= cut[t]: the k largest, lower block first on ties
/// (scores are sums of relus, so >= 0 and ordered as their bits). n <= k keeps every block.
/// Threadgroups (T, 1), 32 threads; lane l reads blocks l, l+32, ...
kernel void q38_idx_topk(
    device const float* score [[buffer(0)]],
    device uint* thr [[buffer(1)]],
    device uint* cut [[buffer(2)]],
    constant Q38TopKParams& p [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint t = tg.x;
    if (t >= p.T) return;
    const uint n = (p.pos + t + 1) / 4;
    if (n <= p.k) {
        if (lane == 0) { thr[t] = 0; cut[t] = n; }
        return;
    }
    device const uint* s = reinterpret_cast<device const uint*>(score + t * p.nb);
    // Largest v with #(bits >= v) >= k: the k-th largest score.
    uint v = 0;
    for (int bit = 30; bit >= 0; --bit) {
        const uint cand = v | (1u << uint(bit));
        int c = 0;
        for (uint b = lane; b < n; b += 32) c += s[b] >= cand ? 1 : 0;
        c = simd_sum(c);
        if (uint(c) >= p.k) v = cand;
    }
    int gt = 0, eq = 0;
    for (uint b = lane; b < n; b += 32) {
        const uint u = s[b];
        gt += u > v ? 1 : 0;
        eq += u == v ? 1 : 0;
    }
    gt = simd_sum(gt);
    eq = simd_sum(eq);
    const uint need = p.k - uint(gt);
    uint last = n - 1;
    if (uint(eq) > need) {
        // Largest c with #(== v, b < c) < need: block c is the need-th tie.
        uint c = 0;
        for (int bit = 20; bit >= 0; --bit) {
            const uint cand = c | (1u << uint(bit));
            if (cand > n) continue;
            int m = 0;
            for (uint b = lane; b < cand; b += 32) m += s[b] == v ? 1 : 0;
            m = simd_sum(m);
            if (uint(m) < need) c = cand;
        }
        last = c;
    }
    if (lane == 0) { thr[t] = v; cut[t] = last; }
}

static inline bool q38_block_selected(device const float* score, uint thr, uint cut, uint b) {
    const uint u = reinterpret_cast<device const uint*>(score)[b];
    return u > thr || (u == thr && b <= cut);
}

/// any[b] = 1 when some query of the sub-batch (local t < rows, positions pos + t) selects block b.
/// Thread b; score / thr / cut bound at the sub-batch's first query.
kernel void q38_idx_union(
    device const float* score [[buffer(0)]],
    device const uint* thr [[buffer(1)]],
    device const uint* cut [[buffer(2)]],
    device uchar* any [[buffer(3)]],
    constant Q38TopKParams& p [[buffer(4)]],   // T = rows of the sub-batch
    uint b [[thread_position_in_grid]]
) {
    uchar acc = 0;
    for (uint t = 0; t < p.T; ++t) {
        if (b < (p.pos + t + 1) / 4 && q38_block_selected(score + t * p.nb, thr[t], cut[t], b)) { acc = 1; break; }
    }
    any[b] = acc;
}

struct Q38MaskParams {
    uint g;     // score rows per query
    uint u;     // columns (union tokens, padded)
    uint nb;    // row width of the block scores
    uint pos;   // position of local query 0
};

/// scores[(t * G + h) * U + i] = -FLT_MAX unless query t attends token list[i]: at or before it and in
/// its incomplete block or a selected block. Thread (i, t * G + h), everything bound at the sub-batch.
kernel void q38_attn_mask_sel(
    device float* scores [[buffer(0)]],
    device const uint* list [[buffer(1)]],
    device const float* score [[buffer(2)]],
    device const uint* thr [[buffer(3)]],
    device const uint* cut [[buffer(4)]],
    constant Q38MaskParams& p [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint i = pos.x, t = pos.y / p.g;
    const uint at = p.pos + t, j = list[i], b = j / 4;
    const bool keep = j <= at && (b >= (at + 1) / 4 || q38_block_selected(score + t * p.nb, thr[t], cut[t], b));
    if (!keep) scores[pos.y * p.u + i] = -FLT_MAX;
}

/// kg[i * D + j] = cache[(list[i] * Hkv + g) * D + j], 0 for padding (list[i] = ~0). Thread per element.
kernel void q38_attn_gather_kv_list(
    device const float* cache [[buffer(0)]],
    device float* kg [[buffer(1)]],
    device const uint* list [[buffer(2)]],
    constant Q38AttnPassParams& p [[buffer(3)]],
    constant uint& g [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    const uint tok = list[i / p.d];
    kg[i] = tok == 0xFFFFFFFFu ? 0.0f : cache[(tok * p.hkv + g) * p.d + i % p.d];
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
