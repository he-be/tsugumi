"""Qwen3.5-MoE (Ornith) の float32 参照実装。層をストリーミングする。

**なぜ既存ランタイムを使わないか:** 重みは 19.5〜21.9 GB あり、この機械の RAM は
18 GB しかない。mlx-lm のように「全部載せてから回す」実装は swap 前提の数字しか
出せない。ここは 1 回の forward に必要なものだけを開いて捨てる:

| 区画 | 全体 | 1 回の forward で触る量 |
| --- | ---: | --- |
| routed experts | 18.1 GB | **top-8 だけ** (8 トークンなら 1 層 108 MB) |
| core | 1.26 GB | 層ごとに開いて捨てる |
| embed / lm_head | 1.08 GB | 行だけ / 最後に区切って 1 回 |

ピークは 3 GB 前後。**GPU も mlx も要らない** (numpy だけ)。

算式の出典は 3 つが一致している (docs/qwen35moe/01-MODEL.md §3):
`transformers` の `modeling_qwen3_5_moe.py`、mlx-lm の `qwen3_5.py`、計画側の記述。
実装が正しいことは `test_reference_forward.py` が上流実装そのものに対して確かめる。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_io import Checkpoint  # noqa: E402
from mlx_quant import Dequantizer  # noqa: E402

LM = "language_model."


# --- 算式の部品 -------------------------------------------------------------

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """`Qwen3_5MoeRMSNorm`。**`1+w` は MLX 変換側が焼き済み**なので足さない
    (docs/qwen35moe/10 §4 で上流 bf16 と直接照合済み)。"""
    var = np.mean(np.square(x), axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(var + eps))) * weight


def l2_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """`use_qk_l2norm_in_kernel=True` が中でやっているもの。"""
    return x / np.sqrt(np.sum(np.square(x), axis=-1, keepdims=True) + eps)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(x, 0.0)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# --- config -----------------------------------------------------------------

@dataclass
class Config:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    vocab_size: int
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    conv_kernel: int
    num_k_heads: int
    num_v_heads: int
    key_head_dim: int
    value_head_dim: int
    layer_types: list[str]

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_json(cls, config: dict) -> "Config":
        t = config.get("text_config", config)
        rope = t.get("rope_parameters") or {}
        return cls(
            hidden_size=t["hidden_size"],
            num_layers=t["num_hidden_layers"],
            num_heads=t["num_attention_heads"],
            num_kv_heads=t["num_key_value_heads"],
            head_dim=t["head_dim"],
            rms_norm_eps=t["rms_norm_eps"],
            rope_theta=rope.get("rope_theta", t.get("rope_theta", 10000.0)),
            partial_rotary_factor=rope.get(
                "partial_rotary_factor", t.get("partial_rotary_factor", 1.0)),
            vocab_size=t["vocab_size"],
            num_experts=t["num_experts"],
            top_k=t["num_experts_per_tok"],
            moe_intermediate_size=t["moe_intermediate_size"],
            shared_expert_intermediate_size=t["shared_expert_intermediate_size"],
            conv_kernel=t["linear_conv_kernel_dim"],
            num_k_heads=t["linear_num_key_heads"],
            num_v_heads=t["linear_num_value_heads"],
            key_head_dim=t["linear_key_head_dim"],
            value_head_dim=t["linear_value_head_dim"],
            layer_types=list(t["layer_types"]),
        )


# --- 状態 -------------------------------------------------------------------

class State:
    """層をまたいで持ち越すもの。線形注意は**文脈長に依らない固定サイズ**。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.offset = 0
        conv_dim = cfg.key_head_dim * cfg.num_k_heads * 2 + cfg.value_head_dim * cfg.num_v_heads
        self.conv: dict[int, np.ndarray] = {}
        self.recurrent: dict[int, np.ndarray] = {}
        self.keys: dict[int, np.ndarray] = {}
        self.values: dict[int, np.ndarray] = {}
        for i, kind in enumerate(cfg.layer_types):
            if kind == "linear_attention":
                self.conv[i] = np.zeros((cfg.conv_kernel - 1, conv_dim), np.float32)
                self.recurrent[i] = np.zeros(
                    (cfg.num_v_heads, cfg.value_head_dim, cfg.key_head_dim), np.float32)
            else:
                self.keys[i] = np.zeros((0, cfg.num_kv_heads, cfg.head_dim), np.float32)
                self.values[i] = np.zeros((0, cfg.num_kv_heads, cfg.head_dim), np.float32)

    def nbytes(self) -> int:
        arrays = list(self.conv.values()) + list(self.recurrent.values())
        arrays += list(self.keys.values()) + list(self.values.values())
        return sum(a.nbytes for a in arrays)


# --- モデル -----------------------------------------------------------------

class ReferenceModel:
    def __init__(self, root: str | Path, cache_core: bool = False):
        self.ckpt = Checkpoint(root)
        self.deq = Dequantizer(self.ckpt)
        self.cfg = Config.from_json(self.ckpt.config)
        self.cache_core = cache_core
        self._cache: dict[str, np.ndarray] = {}
        self.bytes_dequantized = 0

    # 重みの取り出し ---------------------------------------------------------

    def _matrix(self, prefix: str) -> np.ndarray:
        got = self._cache.get(prefix)
        if got is None:
            got = self.deq.matrix(prefix)
            self.bytes_dequantized += got.nbytes
            if self.cache_core:
                self._cache[prefix] = got
        return got

    def _vector(self, name: str) -> np.ndarray:
        return self.deq.vector(name)

    def _dense(self, x: np.ndarray, prefix: str) -> np.ndarray:
        return x @ self._matrix(prefix).T

    # 層 ---------------------------------------------------------------------

    def _linear_attention(self, x: np.ndarray, layer: int, state: State,
                          dump: dict | None) -> np.ndarray:
        cfg = self.cfg
        p = f"{LM}model.layers.{layer}.linear_attn"
        T = x.shape[0]
        key_dim = cfg.key_head_dim * cfg.num_k_heads
        value_dim = cfg.value_head_dim * cfg.num_v_heads

        qkv = self._dense(x, f"{p}.in_proj_qkv")
        z = self._dense(x, f"{p}.in_proj_z").reshape(T, cfg.num_v_heads, cfg.value_head_dim)
        b = self._dense(x, f"{p}.in_proj_b")
        a = self._dense(x, f"{p}.in_proj_a")

        # 因果 depthwise conv。**軸順は `[8192, 4, 1]`** (上流 bf16 の `[8192, 1, 4]`
        # から入れ替わっている — docs/qwen35moe/10 §3)。
        w_conv = self._vector(f"{p}.conv1d.weight").reshape(-1, cfg.conv_kernel)
        padded = np.concatenate([state.conv[layer], qkv], axis=0)
        state.conv[layer] = padded[-(cfg.conv_kernel - 1):].copy()
        conv = np.zeros_like(qkv)
        for j in range(cfg.conv_kernel):
            conv += padded[j:j + T] * w_conv[:, j]
        conv = silu(conv)

        q = conv[:, :key_dim].reshape(T, cfg.num_k_heads, cfg.key_head_dim)
        k = conv[:, key_dim:2 * key_dim].reshape(T, cfg.num_k_heads, cfg.key_head_dim)
        v = conv[:, 2 * key_dim:].reshape(T, cfg.num_v_heads, cfg.value_head_dim)

        beta = sigmoid(b)
        A_log = self._vector(f"{p}.A_log")
        dt_bias = self._vector(f"{p}.dt_bias")
        g = np.exp(-np.exp(A_log) * softplus(a + dt_bias))

        repeat = cfg.num_v_heads // cfg.num_k_heads
        q = np.repeat(q, repeat, axis=1)
        k = np.repeat(k, repeat, axis=1)
        q = l2_norm(q) * (cfg.key_head_dim ** -0.5)
        k = l2_norm(k)

        # 再帰。S: [Hv, Dv, Dk]、fp32 固定
        S = state.recurrent[layer]
        out = np.empty((T, cfg.num_v_heads, cfg.value_head_dim), np.float32)
        for t in range(T):
            S *= g[t][:, None, None]
            kv_mem = np.einsum("hvk,hk->hv", S, k[t])
            delta = (v[t] - kv_mem) * beta[t][:, None]
            S += k[t][:, None, :] * delta[:, :, None]
            out[t] = np.einsum("hvk,hk->hv", S, q[t])
        state.recurrent[layer] = S

        norm_w = self._vector(f"{p}.norm.weight")   # **`+1` しない** (RMSNormGated)
        h = rms_norm(out, norm_w, cfg.rms_norm_eps) * silu(z)
        if dump is not None:
            dump[f"layer{layer}.gdn_state"] = S.copy()
            dump[f"layer{layer}.gdn_out"] = h.copy()
        return self._dense(h.reshape(T, value_dim), f"{p}.out_proj")

    def _rope(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """partial RoPE。回すのは先頭 `rotary_dim` だけ、**組は `(i, rd/2+i)`、
        分母は `rotary_dim`** (本ランタイムの既存カーネルとは別物)。"""
        cfg = self.cfg
        rd = cfg.rotary_dim
        half = rd // 2
        inv = cfg.rope_theta ** (-np.arange(0, rd, 2, dtype=np.float64) / rd)
        ang = positions[:, None].astype(np.float64) * inv[None, :]
        cos = np.cos(ang).astype(np.float32)[:, None, :]
        sin = np.sin(ang).astype(np.float32)[:, None, :]
        rot, rest = x[..., :rd], x[..., rd:]
        x1, x2 = rot[..., :half], rot[..., half:]
        out = np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], axis=-1)
        return out

    def _full_attention(self, x: np.ndarray, layer: int, state: State,
                        dump: dict | None) -> np.ndarray:
        cfg = self.cfg
        p = f"{LM}model.layers.{layer}.self_attn"
        T = x.shape[0]
        hd = cfg.head_dim

        # `attn_output_gate`: q_proj は 2 倍幅で、**ヘッドごとに** q と gate に割れる
        qg = self._dense(x, f"{p}.q_proj").reshape(T, cfg.num_heads, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:]
        k = self._dense(x, f"{p}.k_proj").reshape(T, cfg.num_kv_heads, hd)
        v = self._dense(x, f"{p}.v_proj").reshape(T, cfg.num_kv_heads, hd)

        q = rms_norm(q, self._vector(f"{p}.q_norm.weight"), cfg.rms_norm_eps)
        k = rms_norm(k, self._vector(f"{p}.k_norm.weight"), cfg.rms_norm_eps)

        positions = np.arange(state.offset, state.offset + T)
        q = self._rope(q, positions)
        k = self._rope(k, positions)

        state.keys[layer] = np.concatenate([state.keys[layer], k], axis=0)
        state.values[layer] = np.concatenate([state.values[layer], v], axis=0)
        keys, values = state.keys[layer], state.values[layer]

        groups = cfg.num_heads // cfg.num_kv_heads
        keys_r = np.repeat(keys, groups, axis=1)
        values_r = np.repeat(values, groups, axis=1)

        scores = np.einsum("thd,shd->hts", q, keys_r) * (hd ** -0.5)
        total = keys.shape[0]
        causal = np.arange(total)[None, :] > (positions[:, None])
        scores = np.where(causal[None, :, :], -np.inf, scores)
        probs = softmax(scores, axis=-1)
        o = np.einsum("hts,shd->thd", probs, values_r)

        o = o * sigmoid(gate)
        if dump is not None:
            dump[f"layer{layer}.attn_out"] = o.copy()
        return self._dense(o.reshape(T, cfg.num_heads * hd), f"{p}.o_proj")

    def _moe(self, x: np.ndarray, layer: int, dump: dict | None) -> np.ndarray:
        cfg = self.cfg
        p = f"{LM}model.layers.{layer}.mlp"
        T = x.shape[0]

        logits = self._dense(x, f"{p}.gate")
        probs = softmax(logits.astype(np.float32))
        idx = np.argpartition(probs, -cfg.top_k, axis=-1)[:, -cfg.top_k:]
        vals = np.take_along_axis(probs, idx, axis=-1)
        order = np.argsort(-vals, axis=-1)          # 決まった並びで dump するため
        idx = np.take_along_axis(idx, order, axis=-1)
        vals = np.take_along_axis(vals, order, axis=-1)
        weights = vals / np.sum(vals, axis=-1, keepdims=True)
        if dump is not None:
            dump[f"layer{layer}.router_idx"] = idx.copy()
            dump[f"layer{layer}.router_w"] = weights.copy()

        y = np.zeros_like(x)
        for expert in np.unique(idx):
            rows, slots = np.nonzero(idx == expert)
            xe = x[rows]
            wg = self.deq.slice3(f"{p}.switch_mlp.gate_proj", int(expert))
            wu = self.deq.slice3(f"{p}.switch_mlp.up_proj", int(expert))
            wd = self.deq.slice3(f"{p}.switch_mlp.down_proj", int(expert))
            self.bytes_dequantized += wg.nbytes + wu.nbytes + wd.nbytes
            h = silu(xe @ wg.T) * (xe @ wu.T)
            y[rows] += weights[rows, slots][:, None] * (h @ wd.T)

        s = f"{p}.shared_expert"
        shared = silu(self._dense(x, f"{s}.gate_proj")) * self._dense(x, f"{s}.up_proj")
        shared = self._dense(shared, f"{s}.down_proj")
        shared = sigmoid(self._dense(x, f"{p}.shared_expert_gate")) * shared
        return y + shared

    # forward ----------------------------------------------------------------

    def forward(self, tokens: np.ndarray, state: State,
                dump: dict | None = None, logits_for: str = "last") -> np.ndarray:
        cfg = self.cfg
        x = self.deq.matrix(f"{LM}model.embed_tokens", rows=tokens)
        if dump is not None:
            dump["embed"] = x.copy()
        for layer, kind in enumerate(cfg.layer_types):
            prefix = f"{LM}model.layers.{layer}"
            if dump is not None:
                dump[f"layer{layer}.input"] = x.copy()
            h = rms_norm(x, self._vector(f"{prefix}.input_layernorm.weight"),
                         cfg.rms_norm_eps)
            if kind == "linear_attention":
                x = x + self._linear_attention(h, layer, state, dump)
            else:
                x = x + self._full_attention(h, layer, state, dump)
            h = rms_norm(x, self._vector(f"{prefix}.post_attention_layernorm.weight"),
                         cfg.rms_norm_eps)
            x = x + self._moe(h, layer, dump)
            if not self.cache_core:
                self._cache.clear()
        state.offset += tokens.shape[0]

        x = rms_norm(x, self._vector(f"{LM}model.norm.weight"), cfg.rms_norm_eps)
        if dump is not None:
            dump["final_hidden"] = x.copy()
        rows = x if logits_for == "all" else x[-1:]
        return self._lm_head(rows)

    def _lm_head(self, x: np.ndarray, chunk: int = 16384) -> np.ndarray:
        """語彙 248,320 × 2048 を一度に float32 へ広げると 2 GB になるので区切る。"""
        prefix = f"{LM}lm_head"
        out = np.empty((x.shape[0], self.cfg.vocab_size), np.float32)
        for start in range(0, self.cfg.vocab_size, chunk):
            rows = np.arange(start, min(start + chunk, self.cfg.vocab_size))
            w = self.deq.matrix(prefix, rows=rows)
            out[:, start:start + rows.shape[0]] = x @ w.T
        return out

    def generate(self, tokens: list[int], max_new: int, eos: set[int],
                 verbose: bool = True) -> list[int]:
        state = State(self.cfg)
        produced: list[int] = []
        current = np.asarray(tokens, dtype=np.int64)
        for step in range(max_new):
            t0 = time.time()
            logits = self.forward(current, state)
            nxt = int(np.argmax(logits[-1]))
            produced.append(nxt)
            if verbose:
                print(f"  [{step:3d}] token={nxt:6d}  {time.time() - t0:6.1f}s"
                      f"  (文脈 {state.offset})", flush=True)
            if nxt in eos:
                break
            current = np.asarray([nxt], dtype=np.int64)
        return produced


# --- CLI --------------------------------------------------------------------

def load_tokenizer(root: Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(root / "tokenizer.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--text", help="ChatML で 1 ターン包んで流す")
    ap.add_argument("--raw", help="テンプレートを被せずにそのまま流す")
    ap.add_argument("--tokens", help="トークン ID をカンマ区切りで直接")
    ap.add_argument("--max-new", type=int, default=0, help="生成するトークン数")
    ap.add_argument("--dump", help="fixtures を書き出す .npz")
    ap.add_argument("--nll", action="store_true",
                    help="入力そのものの平均 NLL を出す (候補どうしの比較用)")
    ap.add_argument("--cache-core", action="store_true",
                    help="展開した core を持ち続ける (約 5 GB、生成が速くなる)")
    args = ap.parse_args()

    root = Path(args.checkpoint).expanduser()
    model = ReferenceModel(root, cache_core=args.cache_core)
    cfg = model.cfg
    print(f"# {root.name}")
    print(f"  層 {cfg.num_layers} (線形 {cfg.layer_types.count('linear_attention')} / "
          f"full {cfg.layer_types.count('full_attention')})  hidden {cfg.hidden_size}  "
          f"experts {cfg.num_experts} top-{cfg.top_k}  語彙 {cfg.vocab_size}")

    if args.tokens:
        ids = [int(t) for t in args.tokens.replace(",", " ").split()]
    else:
        tok = load_tokenizer(root)
        if args.raw is not None:
            ids = tok.encode(args.raw, add_special_tokens=False).ids
        else:
            text = args.text or "こんにちは。あなたは誰ですか?"
            ids = tok.encode(
                f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n",
                add_special_tokens=False).ids
    print(f"  入力 {len(ids)} トークン: {ids[:16]}{' …' if len(ids) > 16 else ''}")

    generation_config = root / "generation_config.json"
    eos = {248046, 248044}
    if generation_config.exists():
        got = json.loads(generation_config.read_text()).get("eos_token_id")
        if isinstance(got, list):
            eos = set(got)
        elif isinstance(got, int):
            eos = {got}

    if args.max_new > 0:
        t0 = time.time()
        out = model.generate(ids, args.max_new, eos)
        print(f"  生成 {len(out)} トークン / {time.time() - t0:.1f}s")
        try:
            tok = load_tokenizer(root)
            print("--- 出力 ---")
            print(tok.decode(out, skip_special_tokens=False))
        except Exception as exc:  # noqa: BLE001
            print(f"  (復号できず: {exc})")
        return 0

    if args.nll:
        state = State(cfg)
        t0 = time.time()
        logits = model.forward(np.asarray(ids, dtype=np.int64), state,
                               logits_for="all")
        # 位置 t の logits が位置 t+1 のトークンをどれだけ当てているか。
        target = np.asarray(ids[1:], dtype=np.int64)
        row = logits[:-1].astype(np.float64)
        row -= row.max(axis=-1, keepdims=True)
        nll = -(row[np.arange(target.shape[0]), target]
                - np.log(np.exp(row).sum(axis=-1)))
        print(f"  forward {time.time() - t0:.1f}s")
        print(f"  平均 NLL = {nll.mean():.5f}  (perplexity {np.exp(nll.mean()):.4f})"
              f"  位置 {target.shape[0]}")
        return 0

    dump: dict | None = {} if args.dump else None
    state = State(cfg)
    t0 = time.time()
    logits = model.forward(np.asarray(ids, dtype=np.int64), state, dump,
                           logits_for="all")
    print(f"  forward {time.time() - t0:.1f}s  "
          f"展開したバイト {model.bytes_dequantized / 1e9:.2f} GB  "
          f"状態 {state.nbytes() / 1e6:.1f} MB")
    top = np.argsort(-logits[-1])[:5]
    print("  最終位置の top-5:", [(int(i), round(float(logits[-1][i]), 3)) for i in top])
    if dump is not None:
        dump["tokens"] = np.asarray(ids, dtype=np.int64)
        dump["logits"] = logits
        np.savez(args.dump, **dump)
        print(f"  fixtures: {args.dump} ({len(dump)} 本)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
