#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
#     "Pillow>=10.0",
# ]
# ///
"""Generate Japanese captions for images in this directory using the
gemma4-26b-a4b model served by llama-swap on the local network.

Defaults follow Google's Gemma 4 sampling recommendations:
    temperature = 1.0, top_p = 0.95, top_k = 64

Run with:
    uv run caption.py            # uses Google defaults, writes captions_google.json/md
    uv run caption.py --label t02 # override the output label, e.g. for a comparison run

To build a side-by-side comparison from two prior runs:
    uv run caption.py --compare labels.txt   # each line: "<label>\\t<display name>"
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

BASE_URL = "http://100.100.117.10:8080/v1"
MODEL = "gemma4-26b-a4b"
MAX_TOKENS = 2048
TIMEOUT_S = 600.0

IMAGES_DIR = Path(__file__).resolve().parent

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".heic", ".heif"}

# Cap preprocessed image dimensions so the multimodal payload stays small.
MAX_EDGE_PX = 1024
JPEG_QUALITY = 85

# --- Google Gemma 4 default sampling parameters ---
SAMPLING_PRESETS = {
    "google": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "t02": {"temperature": 0.2, "top_p": 0.95, "top_k": 64},
}
DEFAULT_LABEL = "google"

SYSTEM_PROMPT = (
    "あなたは画像キャプショナーです。"
    "ユーザーから渡される画像を見て、日本語で詳細な説明文を書いてください。"
    "箇条書きや見出しは使わず、観察した内容を自然な日本語の文章で述べてください。"
)

USER_PROMPT = (
    "この画像に何が映っているか、被写体・構図・色あい・雰囲気を含めて"
    "日本語で具体的に説明してください。説明文のみを返してください。"
)

# Some models echo the prompt; strip it if it appears at the start of the reply.
PROMPT_HEAD = USER_PROMPT[:32]


def output_paths(label: str) -> tuple[Path, Path]:
    return (
        IMAGES_DIR / f"captions_{label}.json",
        IMAGES_DIR / f"captions_{label}.md",
    )


def parse_spec_file(path: Path) -> list[tuple[str, str]]:
    """Read 'label\\tdisplay_name' lines for --compare."""
    pairs: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) == 1:
            pairs.append((parts[0], parts[0]))
        else:
            pairs.append((parts[0], parts[1]))
    return pairs


def preprocess_image(path: Path) -> tuple[bytes, str]:
    """Read an image, downscale to MAX_EDGE_PX, and re-encode as JPEG."""
    with Image.open(path) as img:
        if img.mode in {"RGBA", "P", "LA"}:
            background = Image.new("RGB", img.size, (255, 255, 255))
            try:
                background.paste(img, mask=img.getchannel("A"))
            except ValueError:
                background.paste(img.convert("RGBA"))
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        scale = min(1.0, MAX_EDGE_PX / max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return buf.getvalue(), "image/jpeg"


def call_caption(
    client: httpx.Client,
    image_bytes: bytes,
    mime: str,
    sampling: dict,
) -> dict:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            },
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "stream": False,
    }
    resp = client.post(f"{BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def collect_images() -> list[Path]:
    return sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def run_caption(label: str) -> list[dict]:
    if label not in SAMPLING_PRESETS:
        print(f"unknown label {label!r}; choose from {sorted(SAMPLING_PRESETS)}", file=sys.stderr)
        sys.exit(2)
    sampling = SAMPLING_PRESETS[label]
    print(f"label={label} sampling={sampling}")

    images = collect_images()
    if not images:
        print("no images found", file=sys.stderr)
        return []
    print(f"found {len(images)} image(s)")

    out_json, out_md = output_paths(label)
    results: list[dict] = []
    with httpx.Client() as client:
        for path in images:
            print(f"\n=== {path.name} ===", flush=True)
            t0 = time.time()
            image_bytes, mime = preprocess_image(path)
            print(f"  preprocessed: {len(image_bytes):,} bytes ({mime})", flush=True)
            data = call_caption(client, image_bytes, mime, sampling)
            elapsed = time.time() - t0

            choice = data["choices"][0]
            text = choice["message"]["content"].strip()
            if text.startswith(PROMPT_HEAD):
                text = text[len(PROMPT_HEAD):].lstrip(" :\n")

            usage = data.get("usage", {})
            results.append({
                "file": path.name,
                "caption": text,
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "elapsed_seconds": round(elapsed, 2),
            })
            print(
                f"  tokens: {usage.get('total_tokens')} "
                f"(in {usage.get('prompt_tokens')} / out {usage.get('completion_tokens')})"
            )
            print(f"  elapsed: {elapsed:.1f}s")
            print(f"  caption: {text[:160]}{'…' if len(text) > 160 else ''}")

    payload = {
        "model": MODEL,
        "label": label,
        "sampling": sampling,
        "max_tokens": MAX_TOKENS,
        "results": results,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# 画像キャプション",
        "",
        f"Model: `{MODEL}`",
        f"Label: `{label}`",
        f"Sampling: `temperature={sampling['temperature']}, top_p={sampling['top_p']}, top_k={sampling['top_k']}`",
        f"max_tokens: `{MAX_TOKENS}`",
        "",
    ]
    for entry in results:
        md_lines.append(f"## {entry['file']}")
        md_lines.append("")
        md_lines.append(entry["caption"])
        md_lines.append("")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nwrote {out_json.name} and {out_md.name}")
    return results


def run_compare(spec: list[tuple[str, str]]) -> int:
    """Build a side-by-side compare markdown from captions_<label>.json files."""
    runs: list[tuple[str, str, dict]] = []
    for label, display in spec:
        json_path, _ = output_paths(label)
        if not json_path.exists():
            print(f"missing {json_path}", file=sys.stderr)
            return 1
        data = json.loads(json_path.read_text(encoding="utf-8"))
        runs.append((label, display, data))

    # Tolerate both the new wrapped format and the legacy bare-list format.
    def results_of(data):
        return data["results"] if isinstance(data, dict) and "results" in data else data

    def sampling_of(data):
        if isinstance(data, dict) and "sampling" in data:
            return data["sampling"]
        return {}

    # Build per-image map: file -> {label: caption}
    files: list[str] = []
    seen: set[str] = set()
    for _, _, data in runs:
        for entry in results_of(data):
            if entry["file"] not in seen:
                seen.add(entry["file"])
                files.append(entry["file"])

    md_lines = [
        "# キャプション比較",
        "",
        f"Model: `{MODEL}`",
        f"max_tokens: `{MAX_TOKENS}`",
        "",
        "パラメータ設定:",
        "",
    ]
    for label, display, data in runs:
        s = sampling_of(data)
        if s:
            md_lines.append(
                f"- **{display}** (`{label}`): temperature={s['temperature']}, "
                f"top_p={s['top_p']}, top_k={s['top_k']}"
            )
        else:
            md_lines.append(f"- **{display}** (`{label}`)")
    md_lines.append("")

    for fname in files:
        md_lines.append(f"## {fname}")
        md_lines.append("")
        for label, display, data in runs:
            match = next((e for e in results_of(data) if e["file"] == fname), None)
            if not match:
                md_lines.append(f"### {display}")
                md_lines.append("")
                md_lines.append("_(no result)_")
                md_lines.append("")
                continue
            md_lines.append(f"### {display}")
            md_lines.append("")
            md_lines.append(match["caption"])
            md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    out = IMAGES_DIR / "captions_compare.md"
    out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"wrote {out.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=f"parameter preset / output label (default: {DEFAULT_LABEL})",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="build captions_compare.md from prior runs described in this file",
    )
    args = parser.parse_args()

    if args.compare:
        spec = parse_spec_file(args.compare)
        if not spec:
            print("compare spec is empty", file=sys.stderr)
            return 1
        return run_compare(spec)

    run_caption(args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
