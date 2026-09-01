<p align="center">
  <img src="docs/assets/tsugumi-logo-rounded.png" alt="Tsugumi logo: a fieldfare inside a segmented cache ring" width="280">
</p>

<h1 align="center">Tsugumi</h1>

<p align="center">
  <strong>Talk to a local AI on an ordinary 16 GB Mac</strong><br>
  16GB の Mac で、ローカルの AI と話す
</p>

<p align="center">
  <img alt="Swift 6.2" src="https://img.shields.io/badge/Swift-6.2-F05138?logo=swift&logoColor=white">
  <img alt="Metal 3.2 or later" src="https://img.shields.io/badge/Metal-3.2%2B-5E5CE6">
  <img alt="macOS 15 or later" src="https://img.shields.io/badge/macOS-15%2B-000000?logo=apple&logoColor=white">
  <a href="LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/License-Apache%202.0-2ea44f"></a>
</p>

<p align="center">
  <a href="#try-it">Quick start</a> ·
  <a href="#upstream">Upstream</a> ·
  <a href="docs/MAC_APP.md">Mac app</a> ·
  <a href="docs/OPENAI_SERVER.md">Local server</a> ·
  <a href="docs/BENCHMARKS.md">Benchmarks</a> ·
  <a href="docs/SYSTEM_DESIGN.md">How it works</a> ·
  <a href="docs/OPTIMIZATION_JOURNEY.md">Experiments</a>
</p>

![Tsugumi generating text with Gemma 4 26B-A4B](docs/assets/tsugumi-app.webp)

A Mac app that talks to a mixture-of-experts model on your own machine, and
sends nothing anywhere. No account, no key, no request leaving the laptop —
the conversation and the weights both stay on the SSD you own.

The trick that makes it fit is that a MoE model does not need all of itself at
once. Tsugumi keeps the shared core and the KV cache resident and streams only
the experts each token actually routes to, straight off the SSD, through a
bounded cache. A 26-billion-parameter checkpoint on disk runs against a working
set a normal Mac has room for.

Two checkpoints are supported, both with speculative decoding on by default:

| | [Gemma 4 26B-A4B QAT](https://ai.google.dev/gemma/docs/core/model_card_4) | [Ornith-1.5 35B-A3B](https://huggingface.co/ornith-ai) |
| --- | --- | --- |
| Download | 15.7 GB | 21.0 GB |
| Images | yes, up to 4 per message | no |
| Thinking | toggle, off by default | toggle, on by default |
| Speculation | 4-token drafter block | MTP head, width 2 |

The runtime, the installer, the CLI, the OpenAI-compatible server and the Mac
app are all Swift and Metal. Nothing here wraps MLX or llama.cpp. See
[the Mac app's design notes](docs/MAC_APP.md) for what the app does with them,
and [Upstream](#upstream) for where this came from.

## Try it

```bash
git clone https://github.com/he-be/tsugumi.git
cd tsugumi
swift build -c release
.build/release/TsugumiMac
```

The release build produces the app and its sibling decode service, which the
app launches to own the model and Metal. On the first run Swift Package Manager
also fetches the tokenizer packages.

When the app opens, choose **Download**. The default is Gemma 4, which arrives
as a finished pack (15.7 GB) whose every file is verified against a SHA-256 pin
before it is accepted. Then choose **Load Model** and start typing. Ornith can
be installed later from the same screen if you want a second model; it is a
larger download and no help on a machine that is short on disk.

## At a glance

| Metric | Value |
| --- | --- |
| Models | Gemma 4 26B-A4B IT (~3.88B active per token) and Ornith-1.5 35B-A3B |
| Weights | affine 4-bit, group 64; 8-bit router; 4-bit shared and routed experts |
| Disk | 15.7 GB for Gemma 4, 21.0 GB for Ornith. Installing both needs ~37 GB free |
| Platform | macOS 15 or newer, Metal 3.2 (Metal 4 tensor kernels on macOS 26), Swift 6.2 |
| M2 measured decode | [5.1-6.3 tok/s](docs/BENCHMARKS.md#m2-measured-decode) on an 8 GB M2 MacBook Air, text-only Gemma pack |
| M5 measured decode | [31-35 tok/s](docs/BENCHMARKS.md#m5-measured-decode) on a 24 GB M5 Pro, text-only Gemma pack |
| Community reports | [Here](docs/COMMUNITY_BENCHMARKS.md#community-results) |

**What actually has to fit in memory** is the resident core plus the expert
cache plus the KV cache, not the file on disk. Measured on an 18 GB M3 Pro with
the vision-capable Gemma pack: 1.51 GB resident + 1.15 GB vision + 0.29 GB
prefill scratch, then about 0.11 GB per expert-cache slot and a KV cache that
depends on the context length ([the arithmetic, with the numbers, is in the
server runbook](docs/SERVER_RUNBOOK.md)). The app refuses to load a
configuration that would exceed what Metal recommends for the device rather
than swapping the machine to death, and says which knob to turn down.

The measured result is a reference point, not a performance ceiling. Prompt
length, generated length, page-cache state, and hardware all affect throughput.
See [community benchmark results](docs/COMMUNITY_BENCHMARKS.md#community-results)
from other Macs, or follow the
[community benchmark guide](docs/COMMUNITY_BENCHMARKS.md) to add your own —
**a 16 GB Mac is exactly the machine this project most wants a report from**,
and none of the numbers above were taken on one.

## Using Tsugumi

Tsugumi provides a native Mac app, a command-line interface, and an
experimental loopback OpenAI-compatible server. They use the same `.moepack`
model directory, but only one model-owning product should run at a time.

The Swift package exposes six products:

| Product | Purpose |
| --- | --- |
| `Tsugumi` | Swift library containing the runtime and Metal kernels |
| `TsugumiMac` | Native Mac app for installation and generation |
| `TsugumiDecodeService` | One-shot local model and Metal owner used by the Mac app |
| `TsugumiCLI` | Command-line instruction chat and raw completion |
| `TsugumiServer` | Loopback OpenAI-compatible Chat Completions server |
| `TsugumiRepack` | Streaming model installer and install verifier |

### Requirements

- An Apple Silicon Mac. The app is aimed at ordinary 16 GB machines; the
  development machine is an 18 GB M3 Pro and upstream validated the text-only
  pack on an 8 GB M2 MacBook Air
- macOS 15 (Sequoia) or newer
- Xcode 26 and Swift 6.2 or newer
- Free storage for the model: 15.7 GB for Gemma 4, 21.0 GB for Ornith
- An internet connection for the first model install

The package is arm64-only. macOS 14 and earlier are not supported.

macOS 26 additionally enables the Metal 4 / MSL 4.0 tensor kernels: the
MetalPerformancePrimitives prefill matmul and the tensor-ops prefill attention
path. On macOS 15 the shader library is compiled as MSL 3.2, those kernels are
compiled out by their `__HAVE_TENSOR__` guard, and the runtime falls back to
the portable prefill kernels. Decode is unaffected; prefill on macOS 15 is
slower than the published macOS 26 numbers.

### Prompting the model

The Mac app treats what you type as an instruction and handles Gemma's chat
formatting automatically. Just describe the task and include any context the
model needs.

Generation defaults to temperature `1.0`, Top-K `64`, and Top-P `0.95`, which
are the values Gemma 4 recommends. Set temperature to `0` for deterministic
greedy output, but expect repetition: below the recommended temperature this
model can fall into a loop and never finish an answer. The model can still
repeat itself or give incorrect answers, so check important results.

The app and CLI support user and model messages plus optional system guidance;
they do not expose or execute tools. The loopback server accepts function-tool
declarations and returns model-produced tool calls for the client to authorize
and execute. The Gemma pack carries a vision tower, so the app accepts up to
four images per message under that model; Ornith is text-only. Audio and video
are not supported by either.

### Mac app

Clone the repository, then run the app from its root:

```bash
swift build -c release
.build/release/TsugumiMac
```

Build the complete package so the app and its sibling decode service are both
available. When launched from this checkout, the app keeps its models under
`scratch/`; a copy launched from anywhere else keeps them in
`~/Library/Application Support/Tsugumi/`.

#### Install a model

On first launch the app checks the available storage and shows the download and
installed sizes. Choose **Download** to begin.

Both checkpoints are downloaded as finished packs rather than repacked on the
machine: neither can come out of a streaming repack (Gemma's symmetric
quantization needs the staged snapshot's bias ranges, and Ornith needs a
q_norm bake and a grafted MTP head that only the Python pipeline produces).
Every file is pinned by SHA-256, written to a `.part` file, resumed with Range
requests if interrupted, and only renamed into place once its hash matches. The
revision in the URL is a convenience, not the integrity anchor. Installation
does not load the model into memory.

#### Load and generate

After installation:

1. Choose **Load Model**.
2. Enter a prompt in the composer.
3. Choose **Generate**, or press <kbd>Command</kbd>+<kbd>Return</kbd>. Use **Settings > Send Message With** to choose Return or Command-Return.
4. Use the stop button or <kbd>Escape</kbd> to end generation early.

Conversations are multi-turn and kept in a sidebar; they are saved to
`~/Library/Application Support/Tsugumi/chats.json` and come back when the app
reopens. Chats are not tied to a model, so a conversation started under one
checkpoint can be continued under the other. One generation runs at a time —
switching to another chat while one is generating is allowed, and the output
keeps landing in the chat that started it.

The status bar shows generation progress, decode speed, and memory use. Use the
right pane to configure sampling, context length, expert-cache slots, and
runtime options. See [Runtime controls](docs/RUNTIME_CONTROLS.md) for the
details and [the Mac app notes](docs/MAC_APP.md) for what differs between the
two models.

### Command-line interface

The CLI uses an existing `.moepack` installation. If you installed the model
through the Mac app, it is already available at `scratch/gemma4.moepack`.
Otherwise, install it from the command line:

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

#### Instruction chat

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

#### Images

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

The [local server](docs/OPENAI_SERVER.md#images) takes the same images as
OpenAI-style `image_url` content parts holding a `data:` URI.

#### Speculative decoding (MTP)

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
is off unless asked for. The [local server](docs/OPENAI_SERVER.md#speculative-decoding-mtp)
takes the same flag. Measurements and the accepted trade-offs are in
[`RESULTS_MTP.md`](RESULTS_MTP.md).

Common generation options include `--max-context`, `--temperature`, `--top-k`,
`--top-p`, `--repetition-penalty`, `--seed`, and repeatable `--stop` strings.
Runtime options include `--expert-cache-slots`, `--expert-cache-policy`,
`--prefill`, `--prefill-chunk-tokens`, and `--rdadvise`; omitted options use
the [production defaults](docs/RUNTIME_CONTROLS.md). Run the following command
for the complete option list:

```bash
swift run -c release TsugumiCLI --help
```

Generated text goes to standard output. Timing statistics go to standard error;
add `--quiet` to suppress that footer in scripts.

### Local OpenAI-compatible server

Build the server and point it at an installed model:

```bash
swift build -c release --product TsugumiServer
.build/release/TsugumiServer \
  --model scratch/gemma4.moepack
```

It listens on `http://127.0.0.1:8080/v1` and supports Chat Completions,
streaming, function tools, `image_url` content parts holding a `data:` URI,
speculative decoding (`--draft-block-size`), and single-prefix prompt reuse for
text-only turns. The client must
authorize and run every tool call. Keep the server on loopback; it has no
remote authentication or TLS.

See [Local server](docs/OPENAI_SERVER.md) for a test request, Python and
OpenCode setup, image limits, prompt reuse, tool handling, and the supported API
subset. A machine-specific runbook for this M3 Pro — the three context/slot
combinations that fit, what to check before starting, and how to read the log —
is in [`docs/SERVER_RUNBOOK.md`](docs/SERVER_RUNBOOK.md) (Japanese).

## Test and contribute

Run the public test suite serially:

```bash
Scripts/test.sh
```

Before starting a model run, close memory-heavy apps and check
`memory_pressure -Q`. If it reports little free memory, postpone the run. Run
only one Tsugumi app, decode service, CLI, server, test, or other
local-model process at a time.

To contribute a comparable performance result, follow the
[community benchmark guide](docs/COMMUNITY_BENCHMARKS.md).

## How the inference engine works

At each transformer layer, Metal computes attention and the router from
resident weights. The CPU uses the router's top-8 expert IDs to plan against
the layer's 16-slot LFU cache, then fills misses with bounded parallel `pread`
calls into Metal-visible buffers. Metal computes the resident shared-expert
branch while those reads run, then combines the shared and routed outputs.

Prompt prefill uses chunks of up to 128 tokens so one fetched expert can serve
multiple rows. Generation repeats the routed layer loop one token at a time.
The installer applies the same bounded-memory rule: it repacks remote ranges
directly into `.moepack` without staging a full shard or tensor.

For a video overview of Tsugumi, see Better Stack's
[Local AI On Apple Silicon uses 7X Less RAM](https://youtu.be/vHhephsP6vU).

For a visual introduction to the model architecture, see Maarten Grootendorst's
[A Visual Guide to Gemma 4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4).

[System design](docs/SYSTEM_DESIGN.md) explains the `.moepack` layout, memory
ownership, prefill, router handoff, `cb1`/`io`/`cb2` phases, Metal kernels, and
correctness invariants.

## Status and scope

Tsugumi currently includes:

- Remote streaming repack into the `.moepack` model format
- Instruction-tuned Gemma 4 26B-A4B with verified text-only chat formatting
- 4-bit MLX affine embedding, attention, shared-expert, and routed-expert
  weights, with an 8-bit router
- Custom Metal kernels for quantized GEMV, attention, MoE, normalization,
  RoPE, sampling, and production fusions
- SSD-backed routed-expert streaming with a bounded expert cache
- Chunked single-prompt prefill and token-by-token generation
- FP16 KV storage with bounded circular storage for 25 sliding-window layers
  and linear storage for 5 full-attention layers
- Exact split-K/V decode attention with distinct normalized K and V paths
- A Swift library, streaming installer, command-line interface, loopback
  OpenAI-compatible server, and native SwiftUI/AppKit Mac app with a one-shot
  local decode service

On top of that, this fork adds a second architecture (Qwen3.5-MoE / Ornith-1.5,
with its linear-attention layers and its own MTP head), speculative decoding for
both families, an OpenAI-compatible server written against llama.cpp's server as
the reference implementation, and a Mac app that carries multi-turn chats,
images, a thinking channel and a prompt cache across two checkpoints.

### Future work

- Measure on an actual 16 GB Mac and make the app pick its own defaults from
  what the device reports, instead of shipping one fixed operating point.
- Ship a signed, notarized `.app` so the project can be used without a
  toolchain.
- Build iPhone and iPad apps, then measure inference speed and memory use on
  mobile hardware.

## Experiments and technical documentation

The [experiments that shaped Tsugumi](docs/OPTIMIZATION_JOURNEY.md)
explain the largest wins, the plausible ideas that failed, and the early
results that reversed under stronger validation. The detailed
[experiment record](docs/experiments/EXPERIMENT_INVENTORY.md) keeps all 103
audited entries as optional evidence.

Useful entry points:

- [Local OpenAI-compatible server](docs/OPENAI_SERVER.md)
- [System design](docs/SYSTEM_DESIGN.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [The experiments that shaped Tsugumi](docs/OPTIMIZATION_JOURNEY.md)
- [Experiment inventory and summaries](docs/experiments/EXPERIMENT_INVENTORY.md)
- [Implementation references](docs/IMPLEMENTATION_REFERENCES.md)

## Upstream

Tsugumi is a fork of **[TurboFieldfare](https://github.com/drumih/turbo-fieldfare)**
by **Andrey Mikhaylov**, released under the Apache License 2.0. The engine's
shape is his: the `.moepack` container (which he wrote as `.gturbo`), the
streaming repack, the bounded expert cache, the Metal kernels for quantized
GEMV, attention, MoE, normalization, RoPE and sampling, and the first Mac app.
None of what follows would exist without that work, and his afterword below is
kept as he wrote it.

The name follows from it rather than away from it. A fieldfare is a thrush —
_Turdus_ — and ツグミ is what that genus is called in Japanese: the same bird,
named again in the language this fork is written in. It is also said to come
from 口をつぐむ, "to hold one's tongue", which is what the app does. Nothing it
is told leaves the Mac.

What this fork changed, at a glance: a second model architecture and its
speculative-decoding head, an OpenAI-compatible server built line by line
against a pinned llama.cpp, a prompt cache, a QAT symmetric weight path, a
vision tower, and a Mac app rebuilt around multi-turn chats over two
checkpoints. The measurements behind each of those are in `docs/`.

## License and model terms

Tsugumi's source and documentation are licensed under the
[Apache License 2.0](LICENSE), as is the upstream project it derives from.

Model weights are not included. The app downloads them separately and they
remain governed by their own terms:

| Weights | Terms |
| --- | --- |
| Gemma 4 26B-A4B | Google publishes Gemma 4 under the [Apache License 2.0](https://ai.google.dev/gemma/apache_2); the drafter used for speculation is a `license:gemma` artifact, so the [Gemma terms of use](https://ai.google.dev/gemma/terms) come with it |
| Ornith-1.5 35B-A3B | MIT, from the upstream checkpoint |
| Ornith's MTP head | Apache 2.0, grafted from shisa.ai's release |

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the full model and
Swift package license review.

Tsugumi is an independent research project. It is not affiliated with,
sponsored by, or endorsed by Google, Alibaba, or any model publisher.

## Upstream's afterword

The section below is Andrey Mikhaylov's, from TurboFieldfare, unchanged.

Thanks for checking out this project!

My name is Andrey Mikhaylov. You can find me on
[LinkedIn](https://www.linkedin.com/in/andrey-mikhaylov-ios-dev/).
I am the author of TurboFieldfare and an iOS and Metal engineer. Most of my
work is with images, video, and on-device AI.

I dedicate this project to my wife, Sasha, the most supportive person I know.
She stands by me even through the hardest times. She loves wildlife, goes
birdwatching, and volunteers with our local birding community. Because of her,
I have also grown closer to birds and nature.

TurboFieldfare is named after the fieldfare, a member of the thrush family and
my favourite bird. It is not the most noticeable or brightly coloured bird, but
it definitely has a character and unique features of its own. I think the same
is true of this project: it may not be the most practical, but I built it with
my favourite tools, especially Metal, in my favourite field, on-device ML
inference. It definitely has its own character and unique features.

Next time you are outside, touch the grass and listen to the birds. Sometimes
it is the most beautiful thing you can do. And if you can, support your local
wildlife community. They do important work.

Thank you!
