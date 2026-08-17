#!/usr/bin/env python3
"""Caption the images in sample_imgs/ with the local TurboFieldfare CLI.

The counterpart to `sample_imgs/caption.py`, which asks a llama-swap server on
another machine for the same captions from the same checkpoint. Neither run can
reproduce the other token for token — different kernels, different sampling
RNG, different image resampler — so the comparison is about whether the local
tower is *seeing* the image, not about matching text.

Writes `sample_imgs/captions_<label>.json` and `.md` in the same shape
`caption.py` writes, so its `--compare` mode can put the two side by side:

    python3 Scripts/vision/caption_cli.py --model scratch/gemma4-qat.gturbo --label local-t02
    cd sample_imgs && uv run caption.py --compare compare_spec.tsv
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = REPO_ROOT / "sample_imgs"
CLI = REPO_ROOT / ".build" / "release" / "TurboFieldfareCLI"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".heic", ".heif"}

# Byte-for-byte the prompts sample_imgs/caption.py sends, so the only
# differences left between the two runs are the ones being measured.
SYSTEM_PROMPT = (
    "あなたは画像キャプショナーです。"
    "ユーザーから渡される画像を見て、日本語で詳細な説明文を書いてください。"
    "箇条書きや見出しは使わず、観察した内容を自然な日本語の文章で述べてください。"
)
USER_PROMPT = (
    "この画像に何が映っているか、被写体・構図・色あい・雰囲気を含めて"
    "日本語で具体的に説明してください。説明文のみを返してください。"
)

SAMPLING_PRESETS = {
    "google": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "t02": {"temperature": 0.2, "top_p": 0.95, "top_k": 64},
}

FOOTER_PATTERNS = {
    "prefill_tokens": re.compile(r"prefill=(\d+)tok"),
    "new_tokens": re.compile(r"new=(\d+)tok"),
    "decode_seconds": re.compile(r"decode=([\d.]+)s"),
    "ttft_seconds": re.compile(r"ttft=([\d.]+)s"),
    "tower_seconds": re.compile(r"tower=([\d.]+)s"),
    "soft_tokens": re.compile(r"soft=([\d,]+)"),
    "peak_gb": re.compile(r"peak=([\d.]+)GB"),
}


def collect_images() -> list[Path]:
    return sorted(p for p in IMAGES_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def parse_footer(stderr: str) -> dict:
    out: dict[str, object] = {}
    for name, pattern in FOOTER_PATTERNS.items():
        match = pattern.search(stderr)
        if not match:
            continue
        value = match.group(1)
        if name == "soft_tokens":
            out[name] = [int(v) for v in value.split(",") if v]
        elif name in {"prefill_tokens", "new_tokens"}:
            out[name] = int(value)
        else:
            out[name] = float(value)
    return out


def caption_one(cli: Path, model: Path, image: Path, args, sampling: dict) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8",
                                     delete=False) as handle:
        json.dump(messages, handle, ensure_ascii=False)
        messages_path = Path(handle.name)

    command = [
        str(cli),
        "--model", str(model),
        "--messages-file", str(messages_path),
        "--image", str(image),
        "--image-tokens", str(args.image_tokens),
        "--max-new", str(args.max_new),
        "--max-context", str(args.max_context),
        "--temperature", str(sampling["temperature"]),
        "--top-k", str(sampling["top_k"]),
        "--top-p", str(sampling["top_p"]),
        "--verification", args.verification,
    ]
    started = time.time()
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    finally:
        messages_path.unlink(missing_ok=True)
    elapsed = time.time() - started

    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"{image.name}: CLI exited {result.returncode}")

    footer = parse_footer(result.stderr)
    return {
        "file": image.name,
        "caption": result.stdout.strip(),
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": footer.get("prefill_tokens"),
            "completion_tokens": footer.get("new_tokens"),
            "total_tokens": (footer.get("prefill_tokens") or 0)
            + (footer.get("new_tokens") or 0),
        },
        "elapsed_seconds": round(elapsed, 2),
        "local": footer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to a .gturbo with a vision tower")
    parser.add_argument("--label", default="local-t02",
                        help="output label; captions_<label>.json/.md")
    parser.add_argument("--sampling", default="t02", choices=sorted(SAMPLING_PRESETS))
    parser.add_argument("--image-tokens", type=int, default=280, choices=[70, 140, 280])
    parser.add_argument("--max-new", type=int, default=1024)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--verification", default="trusted-install",
                        choices=["trusted-install", "full-sha256"])
    parser.add_argument("--cli", default=str(CLI))
    args = parser.parse_args()

    cli = Path(args.cli)
    if not cli.exists():
        raise SystemExit(f"{cli} not found; run: swift build -c release")
    model = Path(args.model)
    sampling = SAMPLING_PRESETS[args.sampling]

    images = collect_images()
    if not images:
        raise SystemExit("no images in sample_imgs/")
    print(f"label={args.label} sampling={sampling} image_tokens={args.image_tokens}")
    print(f"found {len(images)} image(s)")

    results = []
    for image in images:
        print(f"\n=== {image.name} ===", flush=True)
        entry = caption_one(cli, model, image, args, sampling)
        results.append(entry)
        local = entry["local"]
        print(f"  soft tokens: {local.get('soft_tokens')}"
              f"  tower: {local.get('tower_seconds')}s"
              f"  ttft: {local.get('ttft_seconds')}s")
        print(f"  tokens: {entry['usage']['total_tokens']}"
              f" (in {entry['usage']['prompt_tokens']}"
              f" / out {entry['usage']['completion_tokens']})")
        print(f"  elapsed: {entry['elapsed_seconds']}s")
        preview = entry["caption"][:160]
        print(f"  caption: {preview}{'…' if len(entry['caption']) > 160 else ''}")

    payload = {
        "model": "turbo-fieldfare/gemma4-26b-a4b-qat",
        "label": args.label,
        "sampling": sampling,
        "max_tokens": args.max_new,
        "image_tokens": args.image_tokens,
        "results": results,
    }
    out_json = IMAGES_DIR / f"captions_{args.label}.json"
    out_md = IMAGES_DIR / f"captions_{args.label}.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 画像キャプション (TurboFieldfare CLI)",
        "",
        f"Label: `{args.label}`",
        f"Sampling: `temperature={sampling['temperature']}, "
        f"top_p={sampling['top_p']}, top_k={sampling['top_k']}`",
        f"image_tokens: `{args.image_tokens}`  max_new: `{args.max_new}`",
        "",
    ]
    for entry in results:
        lines += [f"## {entry['file']}", "", entry["caption"], ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_json.name} and {out_md.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
