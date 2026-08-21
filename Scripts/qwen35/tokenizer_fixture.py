#!/usr/bin/env python3
"""Ornith のトークナイザの上流動作を JSON に落とす (Phase 5 の物差し)。

`--qwen-tokenizer` (TurboFieldfareKernelCheck) が突き合わせる相手。
上流の `tokenizers` / `transformers` を**この機械で 1 度だけ**回して、

- 語彙の数と特殊トークンの ID
- 文字列 → ID (エンコード)
- ID → 文字列 (デコード、special を残す / 落とす の両方)
- `chat_template.jinja` の描画 (`enable_thinking` の on/off、tools 有り)

を書き出す。Swift 側はこの JSON に対して一致を主張するので、**Swift と
上流が同じ場所で間違う**ことは無い (どちらも同じ tokenizer.json を読むが、
実装は別物である)。

    ~/LLM/venv/bin/python3 Scripts/qwen35/tokenizer_fixture.py \\
        --tokenizer scratch/ornith-oq4e-g64.gturbo/tokenizer \\
        --out scratch/qwen35/tokenizer-fixture.json

デコードの検体には**乱数の ID 列**も入れる。ByteLevel の BPE は 1 トークンが
UTF-8 の途中で切れ得るので、境界をまたぐ組み合わせを人手で選んでいては
足りない (seed 固定、既定 64 本 × 長さ 24)。
"""
from __future__ import annotations

import argparse
import json
import random
import unicodedata
from pathlib import Path

# エンコードの検体。日本語・英語・記号・コード・空白・絵文字・
# 追加トークンの文字列そのもの、を意図して混ぜている。
TEXTS = [
    "日本の首都はどこですか。一文で答えてください。",
    "こんにちは。あなたは誰ですか?",
    "The capital of Japan is Tokyo (東京).",
    "he said ' ok ' now",
    "step 1 . done",
    "    . indented",
    "a\tb\nc\r\nd",
    "  leading and trailing  ",
    "絵文字: 👨‍👩‍👧‍👦 🇯🇵 🍣",
    "мороз и солнце; день чудесный",
    "```python\ndef f(x: int) -> int:\n    return x * 2\n```",
    "<think>これは本文である</think>",
    "<|im_start|>user\nこんにちは<|im_end|>",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\n東京\n</parameter>\n</function>\n</tool_call>",
    "1234567890 3.14159 -1e-9",
    "Ω≈ç√∫˜µ≤≥÷ æøå ß∂ƒ©˙∆˚¬",
    "混ざったtextと日本語のmixture、punctuation!",
    "",
]

# チャットの検体。role の並びと reasoning_content の有無を変える。
CHATS: list[dict] = [
    {
        "name": "user-only-thinking",
        "messages": [{"role": "user", "content": "日本の首都はどこですか。一文で答えてください。"}],
        "enable_thinking": True,
    },
    {
        "name": "user-only-no-thinking",
        "messages": [{"role": "user", "content": "日本の首都はどこですか。一文で答えてください。"}],
        "enable_thinking": False,
    },
    {
        "name": "system-user",
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Name three primes."},
        ],
        "enable_thinking": True,
    },
    {
        "name": "multi-turn",
        "messages": [
            {"role": "user", "content": "1 + 1 は?"},
            {"role": "assistant", "content": "2 です。"},
            {"role": "user", "content": "では 2 + 2 は?"},
        ],
        "enable_thinking": False,
    },
    {
        "name": "assistant-with-reasoning",
        "messages": [
            {"role": "user", "content": "1 + 1 は?"},
            {
                "role": "assistant",
                "content": "2 です。",
                "reasoning_content": "足し算をする。",
            },
            {"role": "user", "content": "では 2 + 2 は?"},
        ],
        "enable_thinking": True,
    },
    {
        "name": "tools",
        "messages": [{"role": "user", "content": "東京の天気は?"}],
        "enable_thinking": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    },
]


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 の byte↔unicode 表 (上流 `bytes_to_unicode` そのもの)。"""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_UNICODE_TO_BYTE = {v: k for k, v in _bytes_to_unicode().items()}


def _token_bytes(token: str) -> bytes | None:
    """トークンが表すバイト列。表に無い文字が 1 つでもあれば None。"""
    try:
        return bytes(_UNICODE_TO_BYTE[c] for c in token)
    except KeyError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True,
                    help="tokenizer.json / tokenizer_config.json のあるディレクトリ")
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-samples", type=int, default=64)
    ap.add_argument("--random-length", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260821)
    args = ap.parse_args()

    root = Path(args.tokenizer).expanduser()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    raw = json.loads((root / "tokenizer.json").read_text())
    added = raw["added_tokens"]
    special_ids = {a["id"] for a in added if a["special"]}
    vocab_size = max(max(raw["model"]["vocab"].values()), max(a["id"] for a in added)) + 1

    payload: dict = {
        "source": str(root),
        "vocabSize": vocab_size,
        "baseVocabSize": len(raw["model"]["vocab"]),
        "addedTokens": [
            {"id": a["id"], "content": a["content"], "special": bool(a["special"])}
            for a in added
        ],
        "encode": [],
        "decode": [],
        "chat": [],
    }

    for text in TEXTS:
        # 正規化は tokenizer 側 (NFC) に任せる。ここで NFC にしておくのは
        # 「復号したら元に戻る」を検体の側の性質にするためで、NFD の入力が
        # 通らないという主張ではない。
        text = unicodedata.normalize("NFC", text)
        ids = tok.encode(text, add_special_tokens=False).ids
        payload["encode"].append({
            "text": text,
            "ids": ids,
            "decoded": tok.decode(ids, skip_special_tokens=False),
        })

    def decode_case(ids: list[int]) -> dict:
        return {
            "ids": ids,
            "keep": tok.decode(ids, skip_special_tokens=False),
            "skip": tok.decode(ids, skip_special_tokens=True),
        }

    # 1. 特殊トークンを挟んだ列 (added token が literal で出ることを見る)
    payload["decode"].append(decode_case(
        [248045, 872, 198] + tok.encode("こんにちは", add_special_tokens=False).ids
        + [248046, 198, 248068, 198]))
    # 2. 追加トークンだけの列
    payload["decode"].append(decode_case(sorted(a["id"] for a in added)))
    # 3. 乱数の ID 列 — UTF-8 の途中で切れるトークンの組み合わせを稼ぐ
    rng = random.Random(args.seed)
    hi = len(raw["model"]["vocab"])
    for _ in range(args.random_samples):
        ids = [rng.randrange(hi) for _ in range(args.random_length)]
        payload["decode"].append(decode_case(ids))
    # 4. 特殊トークンを混ぜた乱数列 (skip の側を効かせる)
    for _ in range(8):
        ids = [rng.randrange(hi) for _ in range(args.random_length)]
        for _ in range(3):
            ids[rng.randrange(len(ids))] = rng.choice(sorted(special_ids))
        payload["decode"].append(decode_case(ids))
    # 5. **符号点をまたぐ検体。**先頭バイトで終わるトークンと、継続バイトで
    #    始まるトークンを表から選んで並べる。特殊トークンを間に挟むと
    #    skip の側だけが**融合する** (上流は ID を復号器の前で落とすため)。
    #    追加トークンだが special でないもの (`<think>`) は落ちないので、
    #    どちらの側でも融合しない。乱数列ではまず当たらない組み合わせで、
    #    「特殊トークンで run を閉じる」実装だけがここで外れる。
    lead_tokens = [(t, i) for t, i in raw["model"]["vocab"].items()
                   if _token_bytes(t) and 0xE0 <= _token_bytes(t)[-1] <= 0xEF]
    cont_tokens = [(t, i) for t, i in raw["model"]["vocab"].items()
                   if len(_token_bytes(t) or b"") >= 2
                   and all(0x80 <= b <= 0xBF for b in _token_bytes(t)[:2])]
    lead_tokens.sort(key=lambda p: p[1])
    cont_tokens.sort(key=lambda p: p[1])
    im_end = next(a["id"] for a in added if a["content"] == "<|im_end|>")
    think = next(a["id"] for a in added if a["content"] == "<think>")
    for (_, a), (_, b) in zip(lead_tokens[:4], cont_tokens[:4]):
        payload["decode"].append(decode_case([a, b]))
        payload["decode"].append(decode_case([a, im_end, b]))
        payload["decode"].append(decode_case([a, think, b]))

    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(str(root))
    for case in CHATS:
        rendered = hf.apply_chat_template(
            case["messages"],
            tools=case.get("tools"),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=case["enable_thinking"],
        )
        payload["chat"].append({
            "name": case["name"],
            "messages": case["messages"],
            "tools": case.get("tools"),
            "enableThinking": case["enable_thinking"],
            "text": rendered,
            "ids": tok.encode(rendered, add_special_tokens=False).ids,
        })

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"{out}  encode {len(payload['encode'])} / decode {len(payload['decode'])} / "
          f"chat {len(payload['chat'])}  語彙 {vocab_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
