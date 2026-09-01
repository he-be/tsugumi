# Command-line interface

`TsugumiCLI` and `TsugumiRepack` for people who would rather not open the app.
Everything here works against the same `.moepack` directory the Mac app
installs. See [the README](../README.md) for what the project is, and
[Runtime controls](RUNTIME_CONTROLS.md) for the defaults these flags override.

The CLI uses an existing `.moepack` installation. The Mac app installs the
adopted packs as `scratch/gemma4-qat-sym.moepack` and
`scratch/ornith-oq4e-g64.moepack`; point `--model` at either. The commands
below build the upstream text-only pack instead, by streaming and repacking a
checkpoint from Hugging Face:

```bash
swift run -c release TsugumiRepack \
  --output scratch/gemma4.moepack \
  --overwrite
```

Continue a cancelled or interrupted download:

```bash
swift run -c release TsugumiRepack \
  --output scratch/gemma4.moepack \
  --overwrite \
  --resume
```

Remove saved download state:

```bash
swift run -c release TsugumiRepack \
  --discard-partial \
  --output scratch/gemma4.moepack
```

The runtime accepts only a completed `.moepack` directory with a final
`manifest.json`.

Repack a checkpoint that is already staged on disk in its distributed form —
the safetensors shards plus `model.safetensors.index.json`, `config.json` and
the tokenizer files — instead of streaming it:

```bash
swift run -c release TsugumiRepack \
  --output scratch/gemma4-qat.moepack \
  --source-snapshot scratch/qat-aligned-snapshot
```

Nothing is downloaded in this mode: the shards are read in place with `pread`.
The snapshot still has to be one this build pins. Its
`model.safetensors.index.json` digest selects the source and supplies the
revision recorded in the manifest and the install receipt, and an unrecognised
digest is rejected.

Verify an existing installation without loading the model:

```bash
swift run -c release TsugumiRepack \
  --verify-install \
  --input-moepack scratch/gemma4.moepack
```

## Instruction chat

Put chat messages in a JSON array and pass it with `--messages-file`:

```json
[
  {"role": "user", "content": "Explain why chunked prefill reduces time to first token while keeping memory bounded."}
]
```

```bash
swift run -c release TsugumiCLI \
  --model scratch/gemma4.moepack \
  --messages-file messages.json
```

This formats messages in the same way as the Mac app. The CLI response limit
is set with `--max-new`, which defaults to 1,024 tokens. The Mac app can
generate until the selected context window is full.

## Images

A model installed with a vision tower (`--include-vision`, or `--add-vision` on
an existing install) accepts images on the last user turn:

```bash
swift run -c release TsugumiCLI \
  --model scratch/gemma4.moepack \
  --messages-file messages.json \
  --image photo.jpg
```

`--image` is repeatable and `--messages-file` also accepts a list of content
parts, which is how an image is placed between paragraphs of a turn:

```json
[
  {"role": "user", "content": [
    {"type": "text", "text": "What is on the second shelf?"},
    {"type": "image", "path": "shelf.jpg"}
  ]}
]
```

Each image costs up to `--image-tokens` (70, 140, or 280; default 280) prompt
tokens — the actual number follows the image's aspect ratio — plus about 1.4 s
of GPU time before the first token. Images need chunked prefill (`--prefill on`,
the default) and a model that carries the tower; both are checked before
anything runs. Writing `<|image|>` into the text is an error rather than a
silently ignored token.

The [local server](OPENAI_SERVER.md#images) takes the same images as
OpenAI-style `image_url` content parts holding a `data:` URI.

## Speculative decoding (MTP)

A model installed with the drafter section (`--include-draft`, or `--add-draft`
on an existing install) can predict several tokens per step:

```bash
swift run -c release TsugumiCLI \
  --model scratch/gemma4.moepack \
  --messages-file messages.json \
  --draft-block-size 4
```

A small drafter proposes the next few tokens, one verify pass checks them all,
and only the tokens the target model itself would have drawn are kept — so the
answer is the answer of the non-speculative run and only the wall clock moves.
The gain follows how predictable the text is: about **1.4x** on code and on
image descriptions, about **1.0x** on Japanese prose, where the drafter is
rarely right. It needs chunked prefill and a repetition penalty of 1.0, and it
is off unless asked for. The [local server](OPENAI_SERVER.md#speculative-decoding-mtp)
takes the same flag. Measurements and the accepted trade-offs are in
[`RESULTS_MTP.md`](../RESULTS_MTP.md).

Common generation options include `--max-context`, `--temperature`, `--top-k`,
`--top-p`, `--repetition-penalty`, `--seed`, and repeatable `--stop` strings.
Runtime options include `--expert-cache-slots`, `--expert-cache-policy`,
`--prefill`, `--prefill-chunk-tokens`, and `--rdadvise`; omitted options use
the [production defaults](RUNTIME_CONTROLS.md). Run the following command
for the complete option list:

```bash
swift run -c release TsugumiCLI --help
```

Generated text goes to standard output. Timing statistics go to standard error;
add `--quiet` to suppress that footer in scripts.

