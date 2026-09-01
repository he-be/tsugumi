# Tsugumi

Swift and Metal inference for Gemma 4 26B-A4B on Apple Silicon.

## Scope

This checkout is for running and reporting existing behavior. Do not edit source, change runtime defaults, or start optimization work unless the user asks.

## Layout and commands

`Sources/MoEPackFormat/` owns the Foundation-only `.moepack` v1 wire
contract. `Sources/Tsugumi/` is the runtime; `Sources/TsugumiRepack/`,
`Sources/TsugumiCLI/`, `Sources/TsugumiServer/`, and
`Sources/TsugumiApp/` contain the installer, CLI, loopback server, and
Mac app.
`Tests/` contains focused public tests; `docs/` contains design, benchmark, and experiment notes.

```bash
swift run -c release TsugumiRepack --output scratch/gemma4.moepack
swift run -c release TsugumiRepack --output scratch/gemma4.moepack --resume
swift build -c release
.build/release/TsugumiMac
swift run -c release TsugumiCLI \
  --model scratch/gemma4.moepack \
  --prompt "The capital of France is" \
  --max-new 64
```

The installer streams the pinned model without staging the full source checkpoint. Set `HF_TOKEN` only if requested. The download is about 15 GB. Cancellation preserves verified completed ranges; continue them with `--resume` or remove them with `--discard-partial --output scratch/gemma4.moepack`.

## Local server

Follow the [server guide](docs/OPENAI_SERVER.md) for launch commands, health
checks, client setup, prompt reuse, tool loops, and supported API behavior.
Apply the model-process checks below first; never start a second model process
or terminate an existing one.

Keep the server on `127.0.0.1`; it has no remote authentication or TLS, so do
not proxy, tunnel, or expose it. A tool call from the local model never bypasses
the client's normal permission policy. Keep the execution session alive while
the server is needed, and stop only a server you launched.

## Test rules

Before a model run, require macOS 15+, Swift 6.2+, enough disk, acceptable `memory_pressure -Q`, a completed `scratch/gemma4.moepack`, and no process from `pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`. If a check fails, inform the user and stop; do not terminate apps or delete or reinstall the model.

Run package tests through `Scripts/test.sh`. Run only one app, CLI, or model-using test at a time.

For performance results, build release once and follow the [community benchmark guide](docs/COMMUNITY_BENCHMARKS.md) exactly. Do not enable experimental controls or profiling.

Do not download a full checkpoint, duplicate the `.moepack` model, create a worktree, or purge caches just to run tests.

Report the commit, hardware and RAM, macOS, Swift version, exact command, exit code, complete timing footer or error, and every protocol deviation. Treat results as measurements, not performance ceilings.

## App controls

The Mac app runs two checkpoints through the same family sessions the server
uses, so the chat template, the thinking channel, speculative decoding and the
prompt cache are one implementation, not two. It exposes context length,
temperature, Top-K, Top-P, expert-cache slots, prefill, and RDADVISE. Sampling
defaults are each checkpoint's official values — Gemma temperature `1.0` /
Top-K `64` / Top-P `0.95`, editable; Ornith `0.6` / `20` / `0.95`, pinned
rather than editable. Responses can use the context space left after formatting
the prompt, and FP16 is the runtime KV format. The HUD shows generation rate, token count, and
decode-service memory; Last run also shows time to first token and I/O. Build
the app with its sibling `TsugumiDecodeService`; it never loads a second
in-process model. See [README](README.md), [the Mac app notes](docs/MAC_APP.md) and
[Runtime controls](docs/RUNTIME_CONTROLS.md).
