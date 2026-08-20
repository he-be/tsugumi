# Runtime controls

The Mac app, CLI, and local server expose generation and runtime controls. The
app keeps them in its fixed right settings pane. FP16 is the fixed KV format.
Generation settings apply to the next request; app load-time settings require a
reload.

## Generation controls

The Mac app and CLI expose these generation controls:

| Control | Mac values | CLI flag | Default | Effect |
| --- | --- | --- | --- | --- |
| Maximum response | Automatic | `--max-new` | App: remaining context; CLI: 1,024 tokens | The app can use the context space left after formatting the prompt. The CLI uses its explicit or default `--max-new` limit. |
| Maximum context | 4K, 8K, 16K, 32K, 64K | `--max-context` | 4K | Sets prompt plus response capacity. The app shows the FP16 KV-memory delta. The server spells this `-c/--ctx-size`, takes any token count (rounding down to a size this machine can hold), and reaches 128K; the app and its five options do not. |
| Temperature | 0...2 in 0.05 steps | `--temperature` | 1.0 | `0` is greedy; positive values sample. `1.0` is the model-recommended value; lower settings risk repetition loops. |
| Top-K | Off or 1...256 | `--top-k` | 64 | Keeps at most K candidates. CLI `0` turns it off. |
| Top-P | Off or 0.01...1 | `--top-p` | 0.95 | Applies nucleus truncation before Top-K and is effective only while Top-K is enabled. |

With positive temperature, a CLI Top-P below `1` requires Top-K between `1`
and `256`. To disable both truncation controls, pass `--top-k 0 --top-p 1`.
Generation controls apply to the next request and do not require a model
reload. They are interactive product settings, not the fixed community
benchmark protocol.

## Runtime settings

The CLI and the [local server](OPENAI_SERVER.md) accept these flags with the
same names, values, and defaults. The server resolves them before it loads the
model, so an unsupported combination fails immediately with the usage text.

| Control | Mac values | CLI and server flag | Production default | Effect |
| --- | --- | --- | --- | --- |
| Expert-cache slots | 8, 16, 24, 32 | `--expert-cache-slots` | 32 | More slots retain more routed experts and reduce later reads. **32 is both the default and the ceiling** (`docs/mtp/52-D-P7-PREFILL-QUEUE-DEPTH.md` §9a): on the default mmap path a slot is not a private copy, so past the operating point extra slots cost residency and throughput without ever showing up as footprint. Expert reads overlap GPU work, so past the point where they stop poking out from behind it, extra slots buy hit rate and no throughput. Chunked prefill requires at least 16 slots. |
| Expert-cache policy | LFU | `--expert-cache-policy lfu\|lru` | LFU | Chooses which expert is evicted when the cache is full. |
| Prompt prefill | On, off | `--prefill on\|off` | On | On processes known prompt tokens through the chunked prefill path. Off disables that path. |
| Prefill chunk size | 32, 64, 128, 256, 512, 1024, 2048 | `--prefill-chunk-tokens` | 2048 | Sets the number of prompt tokens processed by each chunked-prefill step. A wider chunk routes more tokens through each expert it fetches, so the SSD moves far fewer bytes per prompt token and long prompts prefill faster; it costs KV ring rows (`sliding window + chunk`) and prefill scratch, both counted at load, so a width the device cannot keep resident is rejected there. It has no effect while prefill is off. |
| Speculative block (MTP) | — | `--draft-block-size` | 0 (off) | `0` runs the plain decode loop. `2`...`8` turns on multi-token prediction: the drafter proposes `n - 1` tokens, one verify pass checks them, and only the tokens the target itself would have drawn are kept, so the text is the text of the non-speculative run and only the wall clock moves. Needs a model installed with the drafter section (`--include-draft` / `--add-draft`) and `--prefill on`. The gain follows how predictable the text is: about 1.4x on code and vision answers, about 1.0x on Japanese prose. |
| Reasoning | — | `--reasoning-budget N` | `-1` (unlimited) | Opens the chat template's thought channel, so the model reasons before it answers. `0` closes it; a positive `N` caps the thought and the server forces the closing tag when the cap is reached, so the answer is never empty. The reasoning is generated text: it spends the same token budget as the answer, and `--reasoning-format auto` (the default) returns it separately as `reasoning_content` rather than mixed into the answer. On the server this is the process default, which a request overrides with `chat_template_kwargs.enable_thinking` or `reasoning_effort`. **A request that declares tools still reasons** — see [SPEC](serving/SPEC.md) RSN-5 and MSG-6. The CLI's own `--thinking on\|off` is unrelated and unchanged; it applies to `--messages-file` only. |
| RDADVISE | Off, Default, Bounded, Adaptive | `--rdadvise off\|default\|bounded\|adaptive` | Off | Applies experimental read advice. Its effect depends on the workload; it may help a short decode and slow a long one. |

A request that asks for a repetition penalty other than 1.0 cannot be verified
(the accept rule needs a draw that depends only on position, seed, and
committed history), so the server answers it on the plain decode path instead
of refusing it. The Mac app does not expose the control.

In the app, changing context length, expert-cache slots, or RDADVISE requires a
reload. Some sampling changes also require a reload because greedy and sampled
generation use different output-head paths. Prompt-prefill settings apply to
each request and do not require a reload. Each CLI invocation loads a new model
process, so its selected runtime settings apply immediately. The server fixes
its runtime settings at startup, so changing one means restarting the process.

## Run an experiment

1. Start from 4K context, 16 expert-cache slots, prefill on, and RDADVISE off.
2. Keep the prompt and generation controls fixed.
3. Record a baseline after a warmup.
4. Change one runtime control and reload the app model, or start a new CLI run.
5. Compare prompt prefill, request TTFT, decode rate, peak memory, and I/O per
   token over repeated runs.
6. Restore the production defaults when the experiment ends.

Use the [community benchmark protocol](COMMUNITY_BENCHMARKS.md) for a standard
production result. A run with changed runtime controls is experimental and must
name the changed setting.

## Read the results

- **Decode rate** measures generated tokens per second after prompt prefill.
- **Request TTFT** includes prompt prefill and the wait for the first generated
  token.
- **Peak memory** in Last run is the highest decode-service memory observed
  during the request. The HUD shows the service's current memory instead of the
  much smaller foreground UI process.
- **I/O / token** reports routed-expert read time per generated token.
- **Advanced** shows decode duration and per-token cb1, cb2, and output-head
  time. When RDADVISE runs, it also shows time, calls, data, and skipped advice.

During chunked prefill, the phase label reports exact progress, for example
`Prefill (128/514)`. Errors and unsupported configurations appear only when
they occur. RDADVISE remains experimental and is off by default. A measured
result is a data point, not a performance ceiling.
