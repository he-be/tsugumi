#!/usr/bin/env python3
"""Qwen3.8-Flash-Next (qwen4exp) の CPU float32 参照器。DS4-IQ2 GGUF と PLE Q4_1 サイドカーを 1 トークンずつ読む。

Q2 検証 (`docs/investigations/QWEN38_FLASH_NEXT_VERIFY_PLAN.md`) で Tsugumi の Metal 実装を突き合わせる相手。
算式は ds4-metal `ds4.c` の `qwen4_ref_*` (「Metal graph の parity oracle」、`3030554` の 66149 行〜) を
そのまま写した。重みの慣習もそこに従う:

- ノルムの gamma は ssm_norm 以外 1+w を焼き込み済み (GGUF の値をそのまま掛ける)
- ssm_a は -exp(A_log)
- GDN の value head j は key head j % 16 と組む
- 残差は 2560 次元 x 4 本 (hyper-connection)。PLE は第 1 層 (0 起点) の入口
- 最終ノルムは output_hc の mixer が兼ねる

全体を常駐させない。dense は Q8_0 のブロックのまま行列積、routed expert は選ばれた 10 個だけ gguf-py で
逆量子化する。QSA indexer (ds4 `qwen4_ref_select`) も持つ。予算 (既定 2048 トークン) は `--indexer-top-k` で
小さく上書きでき、短い文で選択を発動させて Metal 側と突き合わせるのに使う。

    ~/LLM/venv/bin/python Scripts/qwen38/reference_forward.py \\
        --text "The capital of France is" --new 16

ログは `Scripts/qwen35/reference_forward.py` と同じ形 (`入力 N トークン: [...]`、`[ i] token= X`) で、
`--dump-logits` を付けると [n, vocab] の float32 を書く。
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "LLM/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

DEFAULT_GGUF = Path.home() / "LLM/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf"
DEFAULT_PLE = Path.home() / "LLM/Qwen3.8-Flash-Next-DS4-IQ2/ple/Qwen3.8-Flash-Next-PLE-Q4_1.gguf"
DEFAULT_TOKENIZER = Path.home() / "LLM/Qwen3.8-Flash-Next-tokenizer/tokenizer.json"

E = 2560          # hidden
HC = 4            # hyper-connection streams
EPS = 1e-6
RSS_LIMIT_GB = 5.0  # bench-hygiene: この参照器が 5 GB を超えたら止める


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def rms(x, g, eps=EPS):
    x = x.astype(np.float32)
    scale = 1.0 / np.sqrt(np.float32((x.astype(np.float64) ** 2).mean()) + np.float32(eps))
    out = x * scale
    return out * g if g is not None else out


def q8_0_roundtrip(x):
    """ggml `quantize_row_q8_0_ref` → dequant の往復 (KV の Q8_0、docs/qwen38/22)。32 要素のブロックごとに
    d = max|x| / 127 (fp16 に丸めて持つ)、q = roundf(x / d) (0.5 は 0 から遠い側、np.rint は偶数丸めで合わない)。"""
    x = np.asarray(x, dtype=np.float32)
    b = x.reshape(-1, 32)
    d = (np.abs(b).max(axis=1) / np.float32(127)).astype(np.float32)
    inv = np.where(d != 0, np.float32(1) / np.where(d != 0, d, np.float32(1)), np.float32(0)).astype(np.float32)
    v = b * inv[:, None]
    a = np.abs(v)
    r = np.floor(a)
    r = r + (a - r >= np.float32(0.5))
    q = np.clip(np.copysign(r, v), -128, 127).astype(np.int8).astype(np.float32)  # int8 を経由 (-0.0 を作らない)
    return (d.astype(np.float16).astype(np.float32)[:, None] * q).reshape(x.shape)


def f16_roundtrip(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def grouped_rms(x, g, groups, n):
    xs = x.reshape(groups, n).astype(np.float64)
    scale = 1.0 / np.sqrt((xs ** 2).mean(axis=1, keepdims=True).astype(np.float32) + np.float32(EPS))
    out = (x.reshape(groups, n) * scale).reshape(-1)
    return out * g


class Weights:
    """GGUF のテンソルを名前で引き、型ごとに行列積と行の取り出しを持つ。"""

    def __init__(self, path: Path):
        self.reader = GGUFReader(path, "r")
        self.t = {t.name: t for t in self.reader.tensors}
        self.fields = self.reader.fields

    def field(self, key):
        return self.fields[key].contents()

    def has(self, name):
        return name in self.t

    def f32(self, name):
        t = self.t[name]
        assert t.tensor_type.name == "F32", (name, t.tensor_type.name)
        return np.asarray(t.data, dtype=np.float32)

    def row(self, name, index):
        """dim[0] 次元の行 index を float32 で。"""
        t = self.t[name]
        ty = t.tensor_type.name
        data = t.data
        if ty == "BF16":
            raw = np.ascontiguousarray(data[index]).view(np.uint16).astype(np.uint32) << 16
            return raw.view(np.float32)
        if ty == "F32":
            return np.asarray(data[index], dtype=np.float32)
        if ty == "F16":
            return np.asarray(data[index], dtype=np.float32)
        return dequantize(np.ascontiguousarray(data[index]), t.tensor_type).reshape(-1).astype(np.float32)

    def matvec(self, name, x, row0=0, rows=None):
        """W[row0:row0+rows] @ x。W の行は dim[0] 次元。"""
        t = self.t[name]
        ty = t.tensor_type.name
        data = t.data
        x = np.asarray(x, dtype=np.float32)
        if data.ndim == 1:  # [n] は 1 行
            data = data.reshape(1, -1)
        elif data.ndim == 3:  # routed expert [experts, rows, bytes/row] は行を通し番号で引く
            data = data.reshape(-1, data.shape[-1])
        if rows is None:
            rows = data.shape[0] - row0
        if ty == "F32":
            return np.asarray(data[row0:row0 + rows], dtype=np.float32) @ x
        if ty == "F16":
            return np.asarray(data[row0:row0 + rows], dtype=np.float32) @ x
        if ty == "Q8_0":
            n = x.shape[0]
            nb = n // 32
            out = np.empty(rows, np.float32)
            xb = x.reshape(nb, 32)
            chunk = 8192
            for s in range(0, rows, chunk):
                blk = np.asarray(data[row0 + s:row0 + min(rows, s + chunk)]).reshape(-1, nb, 34)
                d = np.ascontiguousarray(blk[:, :, 0:2]).view(np.float16).reshape(-1, nb).astype(np.float32)
                q = np.ascontiguousarray(blk[:, :, 2:34]).view(np.int8).astype(np.float32)
                out[s:s + blk.shape[0]] = (np.einsum("rbj,bj->rb", q, xb) * d).sum(axis=1)
            return out
        w = dequantize(np.ascontiguousarray(data[row0:row0 + rows]), t.tensor_type)
        w = w.reshape(rows, -1).astype(np.float32)
        if w.shape[1] != x.shape[0]:  # padded down (768 列) に 640 の act を渡した場合
            xp = np.zeros(w.shape[1], np.float32)
            xp[: x.shape[0]] = x
            x = xp
        return w @ x


class State:
    def __init__(self, n_layers, cap):
        self.cap = cap
        self.lin_state = {}   # il -> [48, 128, 128]
        self.lin_hist = {}    # il -> [3, 10240] oldest first
        self.attn_k = {}      # il -> [cap, 2, 256]
        self.attn_v = {}
        self.idx_k = {}       # il -> [cap, 128] raw indexer keys
        self.ple_hist = np.zeros(((4 - 1) * 3, E * HC), np.float32)  # oldest first, normed
        self.ple_prev = [248044, 248044]  # newest first


class Model:
    def __init__(self, gguf: Path, ple: Path):
        self.w = Weights(gguf)
        self.ablate_ple = False  # 負例: PLE を抜く (--ablate-ple)
        self.ple = Weights(ple)
        w = self.w
        self.n_layer = int(w.field("qwen4exp.block_count"))
        self.n_trunk = self.n_layer - int(w.field("qwen4exp.nextn_predict_layers"))
        self.full_interval = int(w.field("qwen4exp.full_attention_interval"))
        self.ple_layer = int(list(w.field("qwen4exp.ple.layers"))[0])
        self.eos = int(w.field("qwen4exp.ple.eos_token_id"))
        self.n_expert_used = int(w.field("qwen4exp.expert_used_count"))
        self.ple_mult = [int(v) for v in w.field("qwen4exp.ple.layer_multipliers")]
        self.ple_offsets = [int(v) for v in w.field("qwen4exp.ple.head_offsets")]
        self.ple_vocab = [int(v) for v in w.field("qwen4exp.ple.head_vocab_sizes")]
        self.ple_ngram = int(w.field("qwen4exp.ple.ngram_size"))
        self.ple_per = int(w.field("qwen4exp.ple.heads_per_ngram"))
        self.rope_base = float(w.field("qwen4exp.rope.freq_base"))
        self.n_rot = int(w.field("qwen4exp.rope.dimension_count"))
        half = self.n_rot // 2
        self.rope_freq = np.array([self.rope_base ** (-2.0 * i / self.n_rot) for i in range(half)], np.float64)
        self.indexer_top_k = int(w.field("qwen4exp.attention.indexer.top_k"))  # トークン数。--indexer-top-k で上書き
        self.last_selection = {}  # il -> 選ばれたトークン番号 (検査用)
        self.kv_q8 = False  # --kv-type q8_0: RoPE 後の K と V を Q8_0、インデクサ鍵を F16 に往復 (Metal の保持と同じ)

    def is_linear(self, il):
        return (il + 1) % self.full_interval != 0

    # --- blocks -----------------------------------------------------------------

    def hc_mix(self, R, prefix, with_inject=True):
        w = self.w
        xn = grouped_rms(R, w.f32(prefix + "_norm.weight"), HC, E)
        lo = silu(w.matvec(prefix + "_down.weight", xn) / np.float32(HC))
        gate = w.matvec(prefix + "_up.weight", lo)
        mixed = (sigmoid(gate.astype(np.float64)) * xn.astype(np.float64)).reshape(HC, E).sum(axis=0) / HC
        inj = None
        if with_inject:
            inj = 2.0 * sigmoid(w.matvec(prefix + "_inject.weight", xn) / np.float32(HC))
        return mixed.astype(np.float32), inj

    @staticmethod
    def hc_combine(R, out, inj):
        R.reshape(HC, E)[:] += inj.astype(np.float32)[:, None] * out[None, :]

    def ple_rows(self, token, st):
        ctx = [token]
        cut = False
        for s in range(1, self.ple_ngram):
            t = self.eos if cut else st.ple_prev[s - 1]
            cut = cut or t == self.eos
            ctx.append(self.eos if cut else t)
        rows = []
        for n in range(2, self.ple_ngram + 1):
            mixed = (ctx[0] * self.ple_mult[0]) & 0xFFFFFFFFFFFFFFFF
            for j in range(1, n):
                mixed ^= (ctx[j] * self.ple_mult[j]) & 0xFFFFFFFFFFFFFFFF
            for g in range(self.ple_per):
                h = (n - 2) * self.ple_per + g
                rows.append(mixed % self.ple_vocab[h] + self.ple_offsets[h])
        st.ple_prev = [token] + st.ple_prev[:-1]
        return rows

    def ple_block(self, il, st, token, R):
        w = self.w
        pre = f"blk.{il}."
        rows = self.ple_rows(token, st)
        emb = np.concatenate([self.ple.row("ple.weight", r) for r in rows]).astype(np.float32)
        key = w.matvec(pre + "ple_key.weight", emb)
        keyn = grouped_rms(key, w.f32(pre + "ple_norm_key.weight"), HC, E)
        query = grouped_rms(R, w.f32(pre + "ple_norm_query.weight"), HC, E)
        value = w.matvec(pre + "ple_value.weight", emb)
        dots = (keyn.reshape(HC, E).astype(np.float64) * query.reshape(HC, E)).sum(axis=1) / math.sqrt(E)
        gated = np.empty((HC, E), np.float32)
        for s in range(HC):
            g = np.float32(dots[s])
            mag = np.sqrt(max(abs(g), np.float32(1e-6)))
            g = sigmoid(mag if g > 0 else (-mag if g < 0 else np.float32(0)))
            gated[s] = g * value
        gated = gated.reshape(-1)
        normed = grouped_rms(gated, w.f32(pre + "ple_norm_conv.weight"), HC, E)
        cw = w.f32(pre + "ple_conv1d.weight")  # [hc_dim, K]
        K = cw.shape[1]
        hist_rows = st.ple_hist.shape[0]
        acc = np.zeros(E * HC, np.float64)
        for k in range(K):
            back = (K - 1 - k) * self.ple_ngram
            xk = normed if back == 0 else st.ple_hist[hist_rows - back]
            acc += cw[:, k].astype(np.float64) * xk
        R += gated + silu(acc.astype(np.float32))
        st.ple_hist[:-1] = st.ple_hist[1:]
        st.ple_hist[-1] = normed

    def linear(self, il, st, x):
        w = self.w
        pre = f"blk.{il}."
        Hk, Hv, D = 16, 48, 128
        k_dim, v_dim = Hk * D, Hv * D
        conv_dim = 2 * k_dim + v_dim
        qkv = w.matvec(pre + "attn_qkv.weight", x)
        z = w.matvec(pre + "attn_gate.weight", x)
        b = w.matvec(pre + "ssm_beta.weight", x)
        a = w.matvec(pre + "ssm_alpha.weight", x)
        cw = w.f32(pre + "ssm_conv1d.weight")  # [conv_dim, K]
        K = cw.shape[1]
        hist = st.lin_hist.setdefault(il, np.zeros((K - 1, conv_dim), np.float32))
        acc = cw[:, K - 1].astype(np.float64) * qkv
        for k in range(K - 1):
            acc += cw[:, k].astype(np.float64) * hist[k]
        conv = silu(acc.astype(np.float32))
        hist[:-1] = hist[1:]
        hist[-1] = qkv

        q = conv[:k_dim].reshape(Hk, D).copy()
        kk = conv[k_dim:2 * k_dim].reshape(Hk, D).copy()
        v = conv[2 * k_dim:].reshape(Hv, D)
        q /= np.sqrt((q.astype(np.float64) ** 2).sum(axis=1, keepdims=True).astype(np.float32) + np.float32(1e-6))
        kk /= np.sqrt((kk.astype(np.float64) ** 2).sum(axis=1, keepdims=True).astype(np.float32) + np.float32(1e-6))
        q *= np.float32(1.0 / math.sqrt(D))
        A = w.f32(pre + "ssm_a")
        dt = w.f32(pre + "ssm_dt.bias")
        nw = w.f32(pre + "ssm_norm.weight")
        g = np.exp(A * softplus(a + dt)).astype(np.float32)        # [Hv]
        beta = sigmoid(b).astype(np.float32)
        S = st.lin_state.setdefault(il, np.zeros((Hv, D, D), np.float32))  # [Hv][dk][dv]
        kh = np.arange(Hv) % Hk
        qj, kj = q[kh], kk[kh]                                     # [Hv, D]
        S *= g[:, None, None]
        kv = np.einsum("hkv,hk->hv", S, kj)
        delta = (v - kv) * beta[:, None]
        S += kj[:, :, None] * delta[:, None, :]
        o = np.einsum("hkv,hk->hv", S, qj)                         # [Hv, D]
        on = np.stack([rms(o[j], nw) for j in range(Hv)])
        o = (on * sigmoid(z.reshape(Hv, D))).reshape(-1)
        return w.matvec(pre + "ssm_out.weight", o)

    def rope(self, x, pos):
        half = self.n_rot // 2
        theta = pos * self.rope_freq
        c, s = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
        x0, x1 = x[..., :half].copy(), x[..., half:self.n_rot].copy()
        x[..., :half] = x0 * c - x1 * s
        x[..., half:self.n_rot] = x0 * s + x1 * c
        return x

    def rope_at(self, x, positions):
        """x [n, d] を行ごとの位置 positions [n] で回す。"""
        half = self.n_rot // 2
        theta = positions[:, None].astype(np.float64) * self.rope_freq[None, :]
        c, s = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
        x0, x1 = x[:, :half].copy(), x[:, half:self.n_rot].copy()
        x[:, :half] = x0 * c - x1 * s
        x[:, half:self.n_rot] = x0 * s + x1 * c
        return x

    def select(self, il, st, x, pos):
        """QSA のトークン選択 (ds4 `qwen4_ref_select`): 4 トークンのブロックを indexer のスコアで上位 k、残りの端数は全部。"""
        w = self.w
        pre = f"blk.{il}."
        ratio, Di, Hi = 4, 128, 4
        k_blocks = self.indexer_top_k // ratio
        qi = w.matvec(pre + "indexer.q_proj.weight", x).reshape(Hi, Di)
        ik = st.idx_k.setdefault(il, np.zeros((st.cap, Di), np.float32))
        ik[pos] = w.matvec(pre + "indexer.k_proj.weight", x)
        if self.kv_q8:
            ik[pos] = f16_roundtrip(ik[pos])
        n_vis = pos + 1
        n_blocks = n_vis // ratio
        if n_blocks <= k_blocks:
            return np.arange(n_vis)
        gq = w.f32(pre + "indexer.q_norm.weight")
        gk = w.f32(pre + "indexer.k_norm.weight")
        q = np.stack([rms(qi[h], gq) for h in range(Hi)])
        self.rope_at(q, np.full(Hi, pos))
        pooled = ik[: n_blocks * ratio].reshape(n_blocks, ratio, Di).astype(np.float64).mean(axis=1).astype(np.float32)
        keys = np.stack([rms(pooled[b], gk) for b in range(n_blocks)])
        self.rope_at(keys, np.arange(n_blocks) * ratio)
        if self.kv_q8:
            keys = f16_roundtrip(keys)
        dots = q.astype(np.float64) @ keys.astype(np.float64).T      # [Hi, n_blocks]
        score = np.maximum(dots, 0.0).sum(axis=0).astype(np.float32)
        taken = np.sort(np.argsort(-score, kind="stable")[:k_blocks])  # 同点は小さい番号
        sel = (taken[:, None] * ratio + np.arange(ratio)[None, :]).reshape(-1)
        return np.concatenate([sel, np.arange(n_blocks * ratio, n_vis)])

    def attention(self, il, st, x, pos):
        w = self.w
        pre = f"blk.{il}."
        H, Hkv, D = 24, 2, 256
        qg = w.matvec(pre + "attn_q.weight", x).reshape(H, 2 * D)
        k = w.matvec(pre + "attn_k.weight", x).reshape(Hkv, D)
        v = w.matvec(pre + "attn_v.weight", x).reshape(Hkv, D)
        gqn = w.f32(pre + "attn_q_norm.weight")
        gkn = w.f32(pre + "attn_k_norm.weight")
        q = np.stack([rms(qg[h, :D], gqn) for h in range(H)])
        gate = qg[:, D:]
        k = np.stack([rms(k[h], gkn) for h in range(Hkv)])
        self.rope(q, pos)
        self.rope(k, pos)
        kc = st.attn_k.setdefault(il, np.zeros((st.cap, Hkv, D), np.float32))
        vc = st.attn_v.setdefault(il, np.zeros((st.cap, Hkv, D), np.float32))
        if self.kv_q8:
            k, v = q8_0_roundtrip(k), q8_0_roundtrip(v)
        kc[pos], vc[pos] = k, v
        sel = self.select(il, st, x, pos)
        self.last_selection[il] = sel
        ks, vs = kc[sel], vc[sel]                                  # [n_sel, Hkv, D]
        rep = H // Hkv
        kvh = np.arange(H) // rep
        scores = np.einsum("hd,nhd->hn", q, ks[:, kvh, :]) / np.float32(math.sqrt(D))
        scores = scores - scores.max(axis=1, keepdims=True)
        p = np.exp(scores.astype(np.float64))
        p /= p.sum(axis=1, keepdims=True)
        o = np.einsum("hn,nhd->hd", p, vs[:, kvh, :].astype(np.float64)).astype(np.float32)
        o = (o * sigmoid(gate)).reshape(-1)
        return w.matvec(pre + "attn_output.weight", o)

    def moe(self, il, x):
        w = self.w
        pre = f"blk.{il}."
        logits = w.matvec(pre + "ffn_gate_inp.weight", x).astype(np.float64)
        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        sel = []
        for _ in range(self.n_expert_used):  # 同点は小さい番号 (ds4 の選び方)
            masked = prob.copy()
            masked[sel] = -1.0
            sel.append(int(np.argmax(masked)))
        wsum = prob[sel].sum()
        F = 640
        out = np.zeros(E, np.float64)
        for e in sel:
            g = w.matvec(pre + "ffn_gate_exps.weight", x, row0=e * F, rows=F)
            u = w.matvec(pre + "ffn_up_exps.weight", x, row0=e * F, rows=F)
            act = silu(g) * u
            y = w.matvec(pre + "ffn_down_exps.weight", act, row0=e * E, rows=E)
            out += (prob[e] / wsum) * y
        g = w.matvec(pre + "ffn_gate_shexp.weight", x)
        u = w.matvec(pre + "ffn_up_shexp.weight", x)
        y = w.matvec(pre + "ffn_down_shexp.weight", silu(g) * u)
        sg = sigmoid(w.matvec(pre + "ffn_gate_inp_shexp.weight", x)[0])
        out += sg * y
        return out.astype(np.float32), sel

    def layer(self, il, st, token, pos, R):
        if il == self.ple_layer and not self.ablate_ple:
            self.ple_block(il, st, token, R)
        mixed, inj = self.hc_mix(R, f"blk.{il}.hc_attn")
        blk = self.linear(il, st, mixed) if self.is_linear(il) else self.attention(il, st, mixed, pos)
        self.hc_combine(R, blk, inj)
        mixed, inj = self.hc_mix(R, f"blk.{il}.hc_ffn")
        blk, sel = self.moe(il, mixed)
        self.hc_combine(R, blk, inj)
        return sel

    def forward_token(self, st, token, pos):
        R = np.tile(self.w.row("token_embd.weight", token), HC).astype(np.float32)
        routes = []
        for il in range(self.n_trunk):
            routes.append(self.layer(il, st, token, pos, R))
        mixed, _ = self.hc_mix(R, "output_hc", with_inject=False)
        logits = self.w.matvec("output.weight", mixed)
        return logits, routes


def footprint_gb():
    """phys_footprint (匿名メモリ + 圧縮分)。mmap した GGUF のページは RSS に入るが捨てられるので、こちらで見る。"""
    import ctypes
    import os
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    buf = (ctypes.c_uint64 * 40)()
    libproc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(buf))  # RUSAGE_INFO_V2
    # rusage_info_v2: uuid (16 B = 2 x u64) の後、user, system, pkg_idle, interrupt, pageins, wired, resident, phys_footprint
    return buf[2 + 7] / 2**30


def swapouts():
    import subprocess
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Swapouts"):
            return int(line.split(":")[1].strip().rstrip("."))
    return 0


def rss_gb():
    return footprint_gb()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--ple", type=Path, default=DEFAULT_PLE)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--text", default=None)
    ap.add_argument("--tokens", default=None, help="comma-separated ids (--text の代わり)")
    ap.add_argument("--new", type=int, default=8)
    ap.add_argument("--dump-logits", type=Path, default=None)
    ap.add_argument("--indexer-top-k", type=int, default=None,
                    help="QSA の予算 (トークン) を上書きする。短い文で選択を発動させて Metal と突き合わせる検査用")
    ap.add_argument("--ablate-ple", action="store_true",
                    help="負例: PLE を足さない。正しい PLE より NLL が悪くなることを見る")
    ap.add_argument("--kv-type", choices=["f32", "q8_0"], default="f32",
                    help="q8_0: K (RoPE 後) と V を Q8_0、インデクサの raw 鍵と block 鍵を F16 に往復する (Q38_KV_TYPE=q8_0 の参照)")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    if args.tokens:
        prompt = [int(t) for t in args.tokens.split(",")]
    else:
        prompt = tok.encode(args.text).ids

    model = Model(args.gguf, args.ple)
    model.ablate_ple = args.ablate_ple
    model.kv_q8 = args.kv_type == "q8_0"
    if model.kv_q8:
        print("KV: K/V Q8_0、インデクサ鍵 F16")
    if args.indexer_top_k is not None:
        model.indexer_top_k = args.indexer_top_k
        print(f"indexer top_k 上書き: {args.indexer_top_k} トークン ({args.indexer_top_k // 4} ブロック)")
    st = State(model.n_layer, cap=len(prompt) + args.new + 1)
    print(f"入力 {len(prompt)} トークン: {prompt}", flush=True)
    seq = list(prompt)
    all_logits = []
    nll = []
    t0 = time.time()
    swap0 = swapouts()
    for pos in range(len(prompt) + args.new - 1):
        token = seq[pos]
        ts = time.time()
        logits, _ = model.forward_token(st, token, pos)
        if args.dump_logits:
            all_logits.append(logits)
        lp = logits.astype(np.float64) - logits.max()
        lse = math.log(np.exp(lp).sum())
        nxt = int(np.argmax(logits))
        if pos + 1 < len(prompt):
            target = prompt[pos + 1]
            nll.append(-(lp[target] - lse))
            print(f"[{pos:3d}] prefill token={token:7d} next={target:7d} nll={nll[-1]:.3f} "
                  f"top1={nxt:7d} {tok.decode([nxt])!r} ({time.time() - ts:.1f}s, footprint {rss_gb():.2f} GB)", flush=True)
        else:
            seq.append(nxt)
            top = np.argsort(-logits)[:5]
            print(f"[{pos:3d}] token={nxt:7d} {tok.decode([nxt])!r} logit={logits[nxt]:.3f} "
                  f"top5={[int(t) for t in top]} ({time.time() - ts:.1f}s, footprint {rss_gb():.2f} GB)", flush=True)
        if rss_gb() > RSS_LIMIT_GB:
            print(f"footprint {rss_gb():.2f} GB > {RSS_LIMIT_GB} GB, stopping", flush=True)
            return 2
        if swapouts() - swap0 > 4096:  # 64 MB (16 KiB ページ)
            print(f"Swapouts +{swapouts() - swap0} pages, stopping", flush=True)
            return 2
    if nll:
        print(f"prompt NLL mean {np.mean(nll):.4f} over {len(nll)} tokens")
    print(f"生成: {tok.decode(seq[len(prompt):])!r}")
    print(f"合計 {time.time() - t0:.1f}s")
    if args.dump_logits:
        np.stack(all_logits).astype(np.float32).tofile(args.dump_logits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
