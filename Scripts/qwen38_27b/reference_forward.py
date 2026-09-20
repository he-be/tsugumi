#!/usr/bin/env python3
"""Qwen3.8-27B (GGUF arch `qwen35`、dense) の CPU float32 参照器。ISTA GSQ-RCO IQ3_S-mtp を読む。

docs/qwen38-27b/01 §3-3。Flash-Next の `Scripts/qwen38/reference_forward.py` から hc・PLE・indexer・MoE を外し、
`post_attention_norm` と dense SwiGLU を足した。算式は手元の llama.cpp (fe8156f78、読むだけ) の
`src/models/qwen35.cpp`・`delta-net-base.cpp` と `conversion/qwen.py` に従う:

- ノルムの gamma は ssm_norm 以外 1+w を焼き込み済み、ssm_a は -exp(A_log)
- GDN の value head j は key head j % 16 と組む (変換器が V を巡回順に並べ替え、ggml_repeat で広げる)
- GDN の出力ノルムのゲートは SiLU (Flash-Next の sigmoid と違う)
- 全注意は q_proj が head ごとに [Q 256, gate 256]、部分 rope 64 次元 (IMRoPE はテキストなら位置が揃うので neox と同じ)
- 残差: x += mixer(rms(x, attn_norm)); x += ffn(rms(x, post_attention_norm))

27B を f32 で持つと 100 GB を超えるので常駐させない。行列積は行を区切って gguf-py で逆量子化し、ワーカーで並列に掛ける。
逆量子化が 1 回の通しの大半を占めるので、プロンプトは全位置を層ごとにまとめて流す (状態は 1 トークンずつ流したのと同じ)。

    ~/LLM/venv/bin/python Scripts/qwen38_27b/reference_forward.py \\
        --text "The capital of France is" --new 6 --dump-logits ref.logits

ログは Flash-Next の参照器と同じ形 (`入力 N トークン: [...]`、`[ i] prefill ...`、`[ i] token= X`)。
`--dump-logits` は [n, vocab] の float32。
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prism_gguf import Hadamard, Reader, gdn_v_perm  # noqa: E402

DEFAULT_GGUF = Path.home() / "LLM/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
BONSAI_GGUF = Path.home() / "LLM/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf"
DEFAULT_TOKENIZER = Path.home() / "LLM/Qwen3.8-27B-GSQ-RCO-GGUF/tokenizer/tokenizer.json"

ARCH = "qwen35"
FOOTPRINT_LIMIT_GB = 5.0   # bench-hygiene: 親とワーカーの合計
SWAPOUT_LIMIT_PAGES = 4096  # 64 MB (16 KiB ページ)
CHUNK_ELEMENTS = 4_000_000  # 1 タスクで逆量子化する重みの要素数


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def rms_rows(x, g, eps):
    """最後の軸で RMSNorm (ggml_rms_norm: 1 / sqrt(mean + eps))。"""
    x = np.asarray(x, dtype=np.float32)
    ms = (x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True).astype(np.float32)
    out = x * (np.float32(1) / np.sqrt(ms + np.float32(eps)))
    return out * g if g is not None else out


def l2_rows(x, eps):
    """ggml_l2_norm: x / max(||x||, eps)。"""
    n = np.sqrt((x.astype(np.float64) ** 2).sum(axis=-1, keepdims=True)).astype(np.float32)
    return x / np.maximum(n, np.float32(eps))


def q8_0_roundtrip(x):
    """ggml `quantize_row_q8_0_ref` → dequant の往復 (KV の Q8_0、docs/qwen38/22)。Flash-Next の参照器と同じ。"""
    x = np.asarray(x, dtype=np.float32)
    b = x.reshape(-1, 32)
    d = (np.abs(b).max(axis=1) / np.float32(127)).astype(np.float32)
    inv = np.where(d != 0, np.float32(1) / np.where(d != 0, d, np.float32(1)), np.float32(0)).astype(np.float32)
    v = b * inv[:, None]
    a = np.abs(v)
    r = np.floor(a)
    r = r + (a - r >= np.float32(0.5))
    q = np.clip(np.copysign(r, v), -128, 127).astype(np.int8).astype(np.float32)
    return (d.astype(np.float16).astype(np.float32)[:, None] * q).reshape(x.shape)


# --- ワーカー (spawn) ---------------------------------------------------------------

_W = None


def _worker_init(path):
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    global _W
    _W = Reader(path).tensors


def _worker_rows(t, r0, r1):
    return t.rows(r0, r1)


def _worker_matmul(args):
    name, r0, r1, X = args
    W = _worker_rows(_W[name], r0, r1)
    return r0, (X @ W.T)  # [T, rows]


# --- 監視 ---------------------------------------------------------------------------

def footprint_gb(pid=None):
    """phys_footprint (匿名メモリ + 圧縮分)。mmap した GGUF のページは数えない。"""
    import ctypes
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    buf = (ctypes.c_uint64 * 40)()
    libproc.proc_pid_rusage(pid if pid is not None else os.getpid(), 2, ctypes.byref(buf))  # RUSAGE_INFO_V2
    return buf[2 + 7] / 2**30


def swapouts():
    import subprocess
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Swapouts"):
            return int(line.split(":")[1].strip().rstrip("."))
    return 0


class Guard(Exception):
    pass


# --- 重み ---------------------------------------------------------------------------

class Weights:
    def __init__(self, path: Path, workers: int):
        self.path = str(path)
        self.reader = Reader(path)
        self.t = self.reader.tensors
        self.fields = self.reader.fields
        self.h = Hadamard(self.fields)   # Bonsai 以外では空 (`bool(h)` が偽)
        self.gdn_perm = {}               # 重み名 -> ssm_out の並べ替え
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(workers, initializer=_worker_init, initargs=(self.path,))
        self.worker_pids = [p.pid for p in self.pool._pool]
        self.swap0 = swapouts()
        self.peak_gb = 0.0

    def field(self, key, default=None):
        v = self.fields.get(key, default)
        return v.item() if isinstance(v, np.ndarray) and v.size == 1 else v

    def check(self):
        total = footprint_gb() + sum(footprint_gb(p) for p in self.worker_pids)
        self.peak_gb = max(self.peak_gb, total)
        if total > FOOTPRINT_LIMIT_GB:
            raise Guard(f"footprint {total:.2f} GB > {FOOTPRINT_LIMIT_GB} GB")
        ds = swapouts() - self.swap0
        if ds > SWAPOUT_LIMIT_PAGES:
            raise Guard(f"Swapouts +{ds} pages")
        return total

    def f32(self, name):
        t = self.t[name]
        assert t.ggml_type == 0, (name, t.ggml_type)
        a = t.rows(0, t.n_rows)
        return a.reshape(-1) if t.n_rows == 1 else a

    def row(self, name, index):
        return _worker_rows(self.t[name], index, index + 1)[0]

    def matmul(self, name, X):
        """X [T, in] → X @ W.T [T, out]。W の行は GGUF の dim[1] (出力) 方向。"""
        t = self.t[name]
        X = np.ascontiguousarray(X, dtype=np.float32)
        if name in self.h.forward:
            # 折り込み済みの重み: 活性を回してから掛ける (llama-graph.cpp `build_lora_mm`)
            X = self.h.rotate(X, t.n_cols, self.gdn_perm.get(name))
        rows = t.n_rows
        if t.ggml_type in (0, 1, 30):
            return X @ _worker_rows(t, 0, rows).T
        cols = X.shape[1]
        step = max(1, CHUNK_ELEMENTS // cols)
        tasks = [(name, r0, min(rows, r0 + step), X) for r0 in range(0, rows, step)]
        out = np.empty((X.shape[0], rows), np.float32)
        for r0, part in self.pool.imap_unordered(_worker_matmul, tasks):
            out[:, r0:r0 + part.shape[1]] = part
        self.check()
        return out

    def close(self):
        self.pool.terminate()


# --- モデル -------------------------------------------------------------------------

class State:
    def __init__(self, cap):
        self.cap = cap
        self.conv = {}   # il -> [K-1, conv_dim] oldest first
        self.ssm = {}    # il -> [Hv, Dk, Dv]
        self.k = {}      # il -> [cap, Hkv, D]
        self.v = {}


class Model:
    def __init__(self, w: Weights):
        self.w = w
        f = lambda k: w.field(f"{ARCH}.{k}")  # noqa: E731
        self.n_layer = int(f("block_count"))
        # Bonsai には MTP ヘッド (blk.64) が無く、このキーも無い
        self.n_trunk = self.n_layer - int(w.field(f"{ARCH}.nextn_predict_layers", 0) or 0)
        self.full_interval = int(f("full_attention_interval"))
        self.E = int(f("embedding_length"))
        self.eps = float(f("attention.layer_norm_rms_epsilon"))
        self.H = int(f("attention.head_count"))
        self.Hkv = int(f("attention.head_count_kv"))
        self.D = int(f("attention.key_length"))
        self.n_rot = int(f("rope.dimension_count"))
        base = float(f("rope.freq_base"))
        self.rope_freq = np.array([base ** (-2.0 * i / self.n_rot) for i in range(self.n_rot // 2)], np.float64)
        self.Dk = int(f("ssm.state_size"))
        self.Hk = int(f("ssm.group_count"))
        self.Hv = int(f("ssm.time_step_rank"))
        self.Dv = int(f("ssm.inner_size")) // self.Hv
        if w.h.gdn_v_grouped:
            perm = gdn_v_perm(self.Hv * self.Dv, self.Hv, self.Hk)
            w.gdn_perm = {f"blk.{il}.ssm_out.weight": perm for il in range(self.n_layer)}
        self.kv_q8 = False
        self.variant = "ok"  # 負例: gdn-block (value head j を key head j // 3 と組む)、gate-sigmoid
        self.layers = self.n_trunk  # --layers で短くする (動作確認用)

    def is_linear(self, il):
        return (il + 1) % self.full_interval != 0

    def rope(self, x, positions):
        """x [T, heads, D] の先頭 n_rot 次元を neox 形で回す。"""
        half = self.n_rot // 2
        theta = positions[:, None].astype(np.float64) * self.rope_freq[None, :]
        c = np.cos(theta).astype(np.float32)[:, None, :]
        s = np.sin(theta).astype(np.float32)[:, None, :]
        x0, x1 = x[..., :half].copy(), x[..., half:self.n_rot].copy()
        x[..., :half] = x0 * c - x1 * s
        x[..., half:self.n_rot] = x0 * s + x1 * c
        return x

    def linear(self, il, st, x):
        w, pre = self.w, f"blk.{il}."
        T = x.shape[0]
        Hk, Hv, Dk, Dv = self.Hk, self.Hv, self.Dk, self.Dv
        k_dim, v_dim = Hk * Dk, Hv * Dv
        conv_dim = 2 * k_dim + v_dim
        qkv = w.matmul(pre + "attn_qkv.weight", x)
        z = w.matmul(pre + "attn_gate.weight", x)
        b = w.matmul(pre + "ssm_beta.weight", x)
        a = w.matmul(pre + "ssm_alpha.weight", x)
        cw = w.f32(pre + "ssm_conv1d.weight")  # [conv_dim, K]
        K = cw.shape[1]
        A = w.f32(pre + "ssm_a")
        dt = w.f32(pre + "ssm_dt.bias")
        nw = w.f32(pre + "ssm_norm.weight")
        hist = st.conv.setdefault(il, np.zeros((K - 1, conv_dim), np.float32))
        S = st.ssm.setdefault(il, np.zeros((Hv, Dk, Dv), np.float32))
        kh = np.arange(Hv) // (Hv // Hk) if self.variant == "gdn-block" else np.arange(Hv) % Hk
        g_all = np.exp(A[None, :] * softplus(a + dt[None, :])).astype(np.float32)  # [T, Hv]
        beta_all = sigmoid(b).astype(np.float32)
        out = np.empty((T, v_dim), np.float32)
        for t in range(T):
            acc = cw[:, K - 1].astype(np.float64) * qkv[t]
            for k in range(K - 1):
                acc += cw[:, k].astype(np.float64) * hist[k]
            conv = silu(acc.astype(np.float32))
            hist[:-1] = hist[1:]
            hist[-1] = qkv[t]
            q = l2_rows(conv[:k_dim].reshape(Hk, Dk), self.eps) * np.float32(1.0 / math.sqrt(Dk))
            kk = l2_rows(conv[k_dim:2 * k_dim].reshape(Hk, Dk), self.eps)
            v = conv[2 * k_dim:].reshape(Hv, Dv)
            qj, kj = q[kh], kk[kh]
            S *= g_all[t][:, None, None]
            kv = np.einsum("hkv,hk->hv", S, kj)
            delta = (v - kv) * beta_all[t][:, None]
            S += kj[:, :, None] * delta[:, None, :]
            o = np.einsum("hkv,hk->hv", S, qj)                       # [Hv, Dv]
            zt = z[t].reshape(Hv, Dv)
            o = rms_rows(o, nw, self.eps) * (sigmoid(zt) if self.variant == "gate-sigmoid" else silu(zt))
            out[t] = o.reshape(-1)
        return w.matmul(pre + "ssm_out.weight", out)

    def attention(self, il, st, x, pos0):
        w, pre = self.w, f"blk.{il}."
        T = x.shape[0]
        H, Hkv, D = self.H, self.Hkv, self.D
        qg = w.matmul(pre + "attn_q.weight", x).reshape(T, H, 2 * D)
        k = w.matmul(pre + "attn_k.weight", x).reshape(T, Hkv, D)
        v = w.matmul(pre + "attn_v.weight", x).reshape(T, Hkv, D)
        q = rms_rows(qg[..., :D], w.f32(pre + "attn_q_norm.weight"), self.eps)
        gate = qg[..., D:].reshape(T, H * D)
        k = rms_rows(k, w.f32(pre + "attn_k_norm.weight"), self.eps)
        positions = np.arange(pos0, pos0 + T)
        self.rope(q, positions)
        self.rope(k, positions)
        if self.kv_q8:
            k, v = q8_0_roundtrip(k), q8_0_roundtrip(v)
        kc = st.k.setdefault(il, np.zeros((st.cap, Hkv, D), np.float32))
        vc = st.v.setdefault(il, np.zeros((st.cap, Hkv, D), np.float32))
        kc[pos0:pos0 + T], vc[pos0:pos0 + T] = k, v
        P = pos0 + T
        kvh = np.arange(H) // (H // Hkv)
        ks, vs = kc[:P][:, kvh, :], vc[:P][:, kvh, :]                 # [P, H, D]
        scores = np.einsum("thd,phd->thp", q, ks).astype(np.float64) / math.sqrt(D)
        mask = np.arange(P)[None, :] > positions[:, None]              # [T, P]
        scores = np.where(mask[:, None, :], -np.inf, scores)
        scores -= scores.max(axis=2, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=2, keepdims=True)
        o = np.einsum("thp,phd->thd", p, vs.astype(np.float64)).astype(np.float32).reshape(T, H * D)
        o = o * sigmoid(gate)
        return w.matmul(pre + "attn_output.weight", o)

    def ffn(self, il, x):
        w, pre = self.w, f"blk.{il}."
        g = w.matmul(pre + "ffn_gate.weight", x)
        u = w.matmul(pre + "ffn_up.weight", x)
        return w.matmul(pre + "ffn_down.weight", silu(g) * u)

    def forward(self, st, tokens, pos0, log=None):
        """tokens [T] を位置 pos0.. に流し、logits [T, vocab] を返す。"""
        w = self.w
        x = np.stack([w.row("token_embd.weight", t) for t in tokens]).astype(np.float32)
        if "token_embd.weight" in w.h.inverse:
            # 回った基底で格納された表。引いた直後に戻す (llama-graph.cpp、h = s ⊙ (H z))
            x = w.h.unrotate(x, self.E)
        for il in range(self.layers):
            ts = time.time()
            pre = f"blk.{il}."
            h = rms_rows(x, w.f32(pre + "attn_norm.weight"), self.eps)
            x = x + (self.linear(il, st, h) if self.is_linear(il) else self.attention(il, st, h, pos0))
            h = rms_rows(x, w.f32(pre + "post_attention_norm.weight"), self.eps)
            x = x + self.ffn(il, h)
            if log:
                log(f"  layer {il:2d} {'gdn ' if self.is_linear(il) else 'attn'} {time.time() - ts:.1f}s "
                    f"footprint {w.check():.2f} GB |x| {float(np.abs(x).mean()):.4g}")
        h = rms_rows(x, w.f32("output_norm.weight"), self.eps)
        return w.matmul("output.weight", h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--bonsai", action="store_true", help=f"--gguf {BONSAI_GGUF} の短縮")
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--text", default=None)
    ap.add_argument("--tokens", default=None, help="comma-separated ids (--text の代わり)")
    ap.add_argument("--new", type=int, default=8)
    ap.add_argument("--dump-logits", type=Path, default=None)
    ap.add_argument("--kv-type", choices=["f32", "q8_0"], default="f32",
                    help="q8_0: RoPE 後の K と V を Q8_0 に往復する (KV Q8_0 の参照)")
    ap.add_argument("--variant", choices=["ok", "gdn-block", "gate-sigmoid"], default="ok",
                    help="負例: 算式をわざと変え、NLL が悪くなることを見る")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--layers", type=int, default=None, help="先頭 N 層だけ通す (動作確認用、参照には使わない)")
    ap.add_argument("--verbose", action="store_true", help="層ごとの時間と footprint を出す")
    args = ap.parse_args()
    if args.bonsai:
        args.gguf = BONSAI_GGUF

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    prompt = [int(t) for t in args.tokens.split(",")] if args.tokens else tok.encode(args.text).ids

    w = Weights(args.gguf, args.workers)
    try:
        model = Model(w)
        model.kv_q8 = args.kv_type == "q8_0"
        model.variant = args.variant
        if args.variant != "ok":
            print(f"負例: {args.variant}")
        if model.kv_q8:
            print("KV: K/V Q8_0")
        if args.layers is not None:
            model.layers = args.layers
            print(f"先頭 {args.layers} 層だけ (動作確認)")
        st = State(cap=len(prompt) + args.new + 1)
        log = (lambda s: print(s, flush=True)) if args.verbose else None
        print(f"入力 {len(prompt)} トークン: {prompt}", flush=True)
        t0 = time.time()
        seq = list(prompt)
        all_logits = []
        nll = []

        ts = time.time()
        logits = model.forward(st, prompt, 0, log)
        dt = time.time() - ts
        for pos in range(len(prompt)):
            lg = logits[pos]
            lp = lg.astype(np.float64) - lg.max()
            lse = math.log(np.exp(lp).sum())
            nxt = int(np.argmax(lg))
            if pos + 1 < len(prompt):
                target = prompt[pos + 1]
                nll.append(-(lp[target] - lse))
                print(f"[{pos:3d}] prefill token={seq[pos]:7d} next={target:7d} nll={nll[-1]:.3f} "
                      f"top1={nxt:7d} {tok.decode([nxt])!r}", flush=True)
        print(f"prefill {len(prompt)} トークン {dt:.1f}s (footprint 最大 {w.peak_gb:.2f} GB)", flush=True)
        if args.new > 0:
            all_logits.append(logits)
        else:
            all_logits.append(logits[:-1])

        last = logits[-1]
        for i in range(args.new):
            pos = len(prompt) + i - 1
            nxt = int(np.argmax(last))
            seq.append(nxt)
            top = np.argsort(-last)[:5]
            print(f"[{pos:3d}] token={nxt:7d} {tok.decode([nxt])!r} logit={last[nxt]:.3f} "
                  f"top5={[int(t) for t in top]} ({dt:.1f}s, footprint 最大 {w.peak_gb:.2f} GB)", flush=True)
            if i + 1 == args.new:
                break
            ts = time.time()
            lg = model.forward(st, [nxt], pos + 1, log)
            dt = time.time() - ts
            all_logits.append(lg)
            last = lg[0]
        if nll:
            print(f"prompt NLL mean {np.mean(nll):.4f} over {len(nll)} tokens")
        print(f"生成: {tok.decode(seq[len(prompt):])!r}")
        print(f"合計 {time.time() - t0:.1f}s、footprint 最大 {w.peak_gb:.2f} GB、Swapouts +{swapouts() - w.swap0} pages")
        if args.dump_logits:
            np.concatenate(all_logits).astype(np.float32).tofile(args.dump_logits)
    except Guard as e:
        print(f"停止: {e}", flush=True)
        return 2
    finally:
        w.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
