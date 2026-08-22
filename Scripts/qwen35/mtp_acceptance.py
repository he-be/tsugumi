#!/usr/bin/env python3
"""MTP の**受理長 a** を CPU の float32 参照器で先に測る (Phase 7 の M0)。

[docs/qwen35moe/29 §3-4](../../docs/qwen35moe/29-MTP-PREFETCH-OUTLOOK.md) の測定。
**カーネルは 1 本も要らない。**やることは 2 つ:

1. 実生成のトークン列を本体に**教師強制で 1 回**通し、各位置の hidden と
   本体 top-1 を取る (位置 t の hidden は本体 forward の副産物)。
2. その hidden に MTP ヘッドを当てて draft top-1 を取り、**本体 top-1 と
   一致するか**を数える。greedy では一致 = 受理である。

深さ 3 まで鎖にする。P_i は「先頭 i 個が全部通る確率」で、
**a = 1 + P1 + P2 + P3** ([30 §2-1](../../docs/qwen35moe/30-MTP-HEAD-GRAFT.md) の
公表値と同じ定義)。

**ヘッドは差し替え後のものしか使わない** — 出荷版の `mtp.*` は乱数初期化で、
受理は当てずっぽうになる ([30](../../docs/qwen35moe/30-MTP-HEAD-GRAFT.md))。
`q_norm` を焼く前の派生 (`…-oQ4e-g64-shisa`) を渡すこと。参照器は
`head_dim**-0.5` を自分で掛けるので、焼き込み済みだと二重に掛かる。

**本体 hidden が `model.norm` の前か後かは未決着**なので (§4-3)、既定で
両方引く。

**トークン列はランタイムが書いた id をそのまま使う** (`--dump-tokens`)。
印字された本文を引き直すのでは駄目で、BPE が同じバイト列を別の切り方に
畳み直す (t1 で 192 中 9 位置が食い違った)。`--messages` を添えると、
テンプレートを組み直したプロンプトが同じ id になるかも検算する。

    .build/release/TurboFieldfareCLI --model scratch/ornith-oq4e-g64.gturbo \\
        --messages-file bench/qwen35/t1-ja-explain.json --temperature 0 \\
        --repetition-penalty 1 --thinking off --max-new 192 \\
        --dump-tokens scratch/qwen35/mtp-a/t1.tokens.json …

    Scripts/qwen35/mtp_acceptance.py ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-shisa \\
        --tokens scratch/qwen35/mtp-a/t1.tokens.json \\
        --messages bench/qwen35/t1-ja-explain.json \\
        --json scratch/qwen35/mtp-a/t1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference_forward import LM, ReferenceModel, State, rms_norm, sigmoid  # noqa: E402

MTP = f"{LM}mtp"


# --- プロンプトの復元 -------------------------------------------------------

def build_prompt(root: Path, messages: list, enable_thinking: bool = False) -> str:
    """ランタイムと**同じ** `chat_template.jinja` で組む
    (`QwenTokenizer.applyChatTemplate`: add_generation_prompt=true、
    additionalContext に enable_thinking)。"""
    from jinja2 import Environment
    env = Environment(trim_blocks=False, lstrip_blocks=False)
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    template = env.from_string((root / "chat_template.jinja").read_text(encoding="utf-8"))
    return template.render(messages=messages, add_generation_prompt=True,
                           enable_thinking=enable_thinking, tools=None)


# --- MTP ヘッド -------------------------------------------------------------

class MTPHead:
    """1 層 + `fc` + norm 2 本。算式は docs/qwen35moe/03-DESIGN.md §6-4。

    位置ごとに独立に鎖を伸ばすので、深さ d の問い合わせは
    **深さ 1 の KV を因果で全部** + **自分がそこまでに積んだ d-1 本**を見る。
    実機の cache もそうなる (受理済みの区間だけが残り、棄却ぶんは trim される)。
    """

    def __init__(self, model: ReferenceModel):
        self.m = model
        self.cfg = model.cfg

    def _vec(self, name: str) -> np.ndarray:
        return self.m._vector(f"{MTP}.{name}")

    def project(self, base: np.ndarray, next_ids: np.ndarray) -> np.ndarray:
        """`fc(concat(pre_fc_norm_embedding(embed(x_{t+1})), pre_fc_norm_hidden(h_t)))`"""
        eps = self.cfg.rms_norm_eps
        e = self.m.deq.matrix(f"{LM}model.embed_tokens", rows=next_ids)
        e = rms_norm(e, self._vec("pre_fc_norm_embedding.weight"), eps)
        h = rms_norm(base, self._vec("pre_fc_norm_hidden.weight"), eps)
        fused = np.concatenate([e, h], axis=-1)          # 順は [embedding, hidden]
        return fused @ self.m._matrix(f"{MTP}.fc").T

    def qkv(self, x: np.ndarray, positions: np.ndarray):
        cfg = self.cfg
        p = f"{MTP}.layers.0.self_attn"
        n, hd = x.shape[0], cfg.head_dim
        qg = self.m._dense(x, f"{p}.q_proj").reshape(n, cfg.num_heads, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:]
        k = self.m._dense(x, f"{p}.k_proj").reshape(n, cfg.num_kv_heads, hd)
        v = self.m._dense(x, f"{p}.v_proj").reshape(n, cfg.num_kv_heads, hd)
        q = rms_norm(q, self.m._vector(f"{p}.q_norm.weight"), cfg.rms_norm_eps)
        k = rms_norm(k, self.m._vector(f"{p}.k_norm.weight"), cfg.rms_norm_eps)
        return self.m._rope(q, positions), self.m._rope(k, positions), v, gate

    def step(self, base: np.ndarray, next_ids: np.ndarray, positions: np.ndarray,
             cache: list[tuple[np.ndarray, np.ndarray]]):
        """1 段ぶん。`cache` は (k, v) の列。

        - 先頭の 1 本は**深さ 1 の因果キャッシュ** `[N, kv, hd]` で、位置 t は
          `0..t` を見る。
        - 2 本目以降は**その位置が自分で積んだもの**で、位置 t は自分のぶんだけ。
        """
        cfg = self.cfg
        hd = cfg.head_dim
        eps = cfg.rms_norm_eps
        h = self.project(base, next_ids)
        resid = h
        x = rms_norm(h, self._vec("layers.0.input_layernorm.weight"), eps)
        q, k, v, gate = self.qkv(x, positions)
        n = x.shape[0]
        groups = cfg.num_heads // cfg.num_kv_heads

        base_k, base_v = cache[0]
        keys = np.repeat(base_k, groups, axis=1)         # [S, heads, hd]
        values = np.repeat(base_v, groups, axis=1)
        scores = np.einsum("thd,shd->hts", q, keys) * (hd ** -0.5)
        causal = np.arange(keys.shape[0])[None, :] > np.arange(n)[:, None]
        scores = np.where(causal[None, :, :], -np.inf, scores)

        extras_k = [np.repeat(ck, groups, axis=1) for ck, _ in cache[1:]]
        extras_v = [np.repeat(cv, groups, axis=1) for _, cv in cache[1:]]
        diag = [np.einsum("thd,thd->ht", q, ek)[:, :, None] * (hd ** -0.5)
                for ek in extras_k]
        allscores = np.concatenate([scores] + diag, axis=-1)
        probs = np.exp(allscores - np.max(allscores, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        o = np.einsum("hts,shd->thd", probs[:, :, :keys.shape[0]], values)
        for i, ev in enumerate(extras_v):
            o = o + probs[:, :, keys.shape[0] + i].T[:, :, None] * ev

        o = o * sigmoid(gate)
        h = resid + self.m._dense(o.reshape(n, cfg.num_heads * hd),
                                  f"{MTP}.layers.0.self_attn.o_proj")
        resid = h
        x = rms_norm(h, self._vec("layers.0.post_attention_layernorm.weight"), eps)
        h = resid + self.m._moe(x, "mtp", None, prefix=f"{MTP}.layers.0.mlp")
        logits_in = rms_norm(h, self._vec("norm.weight"), eps)
        return h, self.m.lm_head_argmax(logits_in), (k, v)


# --- 測定 -------------------------------------------------------------------

def measure(model: ReferenceModel, tokens: np.ndarray, prompt_len: int,
            depth: int, variants: list[str], chain: str = "mtp_hidden",
            cache_path: Path | None = None) -> dict:
    # 本体の 1 回 (248 トークンで 280 s) が全体の 9 割なので、hidden と top-1 は
    # 落としておいて鎖の作法を変えるときに使い回す。
    if cache_path is not None and cache_path.exists():
        blob = np.load(cache_path)
        if blob["tokens"].shape != tokens.shape or not np.array_equal(blob["tokens"], tokens):
            raise SystemExit(f"{cache_path} は別のトークン列のもの")
        hidden = {"pre": blob["pre"], "post": blob["post"]}
        body_top1 = blob["body_top1"]
        body_secs = float(blob["body_seconds"])
        print(f"  本体 forward: {cache_path} から再利用 (元は {body_secs:.1f}s)", flush=True)
    else:
        t0 = time.time()
        state = State(model.cfg)
        hidden = model.forward(tokens, state, return_hidden=True)
        body_secs = time.time() - t0
        print(f"  本体 forward {len(tokens)} トークン: {body_secs:.1f}s", flush=True)
        t1 = time.time()
        body_top1 = model.lm_head_argmax(hidden["post"])
        print(f"  本体 top-1: {time.time() - t1:.1f}s", flush=True)
        if cache_path is not None:
            np.savez(cache_path, tokens=tokens, pre=hidden["pre"], post=hidden["post"],
                     body_top1=body_top1, body_seconds=np.float64(body_secs))

    # 生成区間で「本体 top-1 = 次のトークン」になっているか (on-policy の検算)。
    gen = np.arange(prompt_len - 1, len(tokens) - 1)
    hit = body_top1[gen] == tokens[gen + 1]
    on_policy = int(np.sum(hit))
    misses = [{"position": int(t), "runtime": int(tokens[t + 1]),
               "reference": int(body_top1[t])}
              for t in gen[~hit]]
    print(f"  on-policy 検算: {on_policy}/{len(gen)} 位置で本体 top-1 = 実際の次トークン",
          flush=True)
    if misses:
        print(f"    食い違い {len(misses)} 箇所 (先頭 5): "
              + ", ".join(f"t={m['position']} {m['runtime']}≠{m['reference']}"
                          for m in misses[:5]), flush=True)

    out = {"chain": chain,
           "tokens": int(len(tokens)), "prompt_len": int(prompt_len),
           "generated": int(len(tokens) - prompt_len),
           "on_policy_match": on_policy, "on_policy_positions": int(len(gen)),
           "on_policy_misses": misses,
           "body_forward_s": body_secs, "variants": {}}

    head = MTPHead(model)
    for variant in variants:
        t2 = time.time()
        base = hidden["pre" if variant == "pre_norm" else "post"]
        n = len(tokens) - 1                       # 位置 t は x_{t+1} を食べる
        cur_base = base[:n]
        cur_next = tokens[1:n + 1]
        anchor = base[:n]                     # chain="base_hidden" 用
        cache: list[tuple[np.ndarray, np.ndarray]] = []
        drafts: list[np.ndarray] = []
        for d in range(depth):
            positions = np.arange(n) + 1 + d
            if d == 0:
                # 深さ 1 の KV は因果キャッシュそのもの。先に 1 度だけ組む。
                h0 = head.project(cur_base, cur_next)
                x0 = rms_norm(h0, head._vec("layers.0.input_layernorm.weight"),
                              model.cfg.rms_norm_eps)
                _, k0, v0, _ = head.qkv(x0, positions)
                cache = [(k0, v0)]
            out_hidden, draft, kv = head.step(cur_base, cur_next, positions, cache)
            drafts.append(draft)
            # 深さ 2 以降に何を「本体 hidden」として渡すか。既定は MTP 自身の
            # 出力を送り返す (mlx-lm#1740 / DeepSeek の MTP と同じ再帰)。
            # base_hidden は「本体 hidden を据え置き、食べるトークンだけ進める」。
            cur_base = out_hidden if chain == "mtp_hidden" else anchor
            cur_next = draft
            if d == 0:
                cache = [kv]
            else:
                cache = cache + [kv]
        secs = time.time() - t2

        # 位置 t の深さ i の的は本体 top-1[t+i]。深さ i まで全部通ったか。
        limit = len(tokens) - 1 - depth
        pos = np.arange(prompt_len - 1, limit)
        alive = np.ones(len(pos), bool)
        per_depth = []
        for i, draft in enumerate(drafts):
            hit = draft[pos] == body_top1[pos + i + 1]
            alive = alive & hit
            per_depth.append({"depth": i + 1,
                              "accept_here": float(np.mean(hit)),
                              "prefix_accept": float(np.mean(alive))})
        a = 1.0 + sum(d["prefix_accept"] for d in per_depth)
        out["variants"][variant] = {"positions": int(len(pos)),
                                    "per_depth": per_depth,
                                    "acceptance_length": a,
                                    "seconds": secs}
        # `chain` は引数の名前なので使わない — 潰すと 2 本目の variant が
        # 別の作法で回る (2026-08-22 に踏んだ)。
        summary = " / ".join(f"P{d['depth']}={d['prefix_accept'] * 100:.2f}%"
                             for d in per_depth)
        print(f"  {variant:9s} {summary}  →  a = {a:.3f}   ({secs:.1f}s)", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="差し替え後・焼き込み前 (…-oQ4e-g64-shisa)")
    ap.add_argument("--tokens", required=True,
                    help='CLI の --dump-tokens が書いた {"prompt":[…],"generated":[…]}')
    ap.add_argument("--messages", help="添えるとプロンプト id を組み直して検算する")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--chain", default="mtp_hidden",
                    choices=["mtp_hidden", "base_hidden"],
                    help="深さ 2 以降に渡す hidden (既定は MTP 自身の出力)")
    ap.add_argument("--hidden-cache", help="本体 hidden と top-1 の置き場 (npz)")
    ap.add_argument("--variant", default="both",
                    choices=["post_norm", "pre_norm", "both"])
    ap.add_argument("--max-generated", type=int, default=0, help="0 なら全部")
    ap.add_argument("--json")
    args = ap.parse_args()

    root = Path(args.checkpoint).expanduser()
    if not (root / "mtp_graft_manifest.json").exists():
        raise SystemExit("差し替え済みのチェックポイントではない "
                         "(mtp_graft_manifest.json が無い)")
    if (root / "bake_manifest.json").exists():
        raise SystemExit("焼き込み済みは渡さない — 参照器が head_dim**-0.5 を"
                         "自分で掛けるので二重になる")

    dumped = json.loads(Path(args.tokens).read_text(encoding="utf-8"))
    prompt, generated = list(dumped["prompt"]), list(dumped["generated"])
    if args.messages:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        rebuilt = tok.encode(build_prompt(
            root, json.loads(Path(args.messages).read_text(encoding="utf-8"))),
            add_special_tokens=False).ids
        if list(rebuilt) != prompt:
            raise SystemExit(f"テンプレートを組み直したプロンプト {len(rebuilt)} 本が"
                             f"ランタイムの {len(prompt)} 本と一致しない")
        print(f"  プロンプト検算: テンプレート再構成と {len(prompt)} 本すべて一致",
              flush=True)
    if args.max_generated:
        generated = generated[:args.max_generated]

    tokens = np.asarray(prompt + generated, dtype=np.int64)
    name = Path(args.messages or args.tokens).stem
    print(f"## {name}  プロンプト {len(prompt)} + 生成 "
          f"{len(generated)} = {len(tokens)} トークン", flush=True)

    model = ReferenceModel(root)
    variants = ["post_norm", "pre_norm"] if args.variant == "both" else [args.variant]
    result = measure(model, tokens, len(prompt), args.depth, variants,
                     chain=args.chain,
                     cache_path=Path(args.hidden_cache) if args.hidden_cache else None)
    result.update(task=name, checkpoint=str(root), depth=args.depth)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
