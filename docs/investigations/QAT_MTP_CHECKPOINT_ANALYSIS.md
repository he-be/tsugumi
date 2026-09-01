# Investigation: checkpoint candidates — QAT, lattice-aligned, and MTP

Date: 2026-08-16
Scope: investigation only. No source, runtime defaults, or model files were
changed. All remote facts were read directly from the Hugging Face API and
pinned-file downloads on the investigation date; revisions and hashes may
change upstream. Quality numbers quoted from upstream READMEs are
self-reported and have not been reproduced here.

## Candidates

| # | Repository | What it is | Status |
| --- | --- | --- | --- |
| 0 | `mlx-community/gemma-4-26b-a4b-it-4bit` | Current pin (affine, group 64, int8 router) | Production |
| 1 | `mlx-community/gemma-4-26B-A4B-it-qat-4bit` | QAT checkpoint, default MLX affine layout | Investigated |
| 2 | `mlx-community/gemma-4-26B-A4B-it-qat-q4_0-mlx-aligned` | QAT lattice-aligned conversion (group 32, bf16 router) | **Leading candidate** |
| 3 | `mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit` | MTP speculative-decoding drafter | Out of scope |

## Current pinning

The pin lives in `Sources/TsugumiRepack/Core/Remote/SupportedModelSource.swift`
(repo `mlx-community/gemma-4-26b-a4b-it-4bit`, revision `0d77464e…`, index
SHA-256 `bf198c9f…`). The installer runs with `requireKnownSource: true`;
`SourceFingerprint.knownFingerprints` contains exactly one entry. No document
in this repository records a comparison against any alternative — the pin
reflects the single-validated-snapshot policy, not a measured rejection of
QAT. Every candidate swap requires new fingerprint entries, pin/size
constants, and a full install; that is routine for any source change and is
not treated as a differentiator below.

## Candidate 1: QAT 4-bit (`gemma-4-26B-A4B-it-qat-4bit`)

Compatible in structure with the current runtime: identical tensor-name set
(vision tower included, still omittable), byte-identical tokenizer, group-64
affine format, and the runtime already accepts its 8-bit shared experts
(`ManifestReader.validateQuant` allows `[4, 8]`; `SharedExpertInt8.swift`
exists). Differences from the pin: 8-bit shared experts (all 30 layers),
+268 MB source. Adoption here is mostly revalidation work, but it is still a
min/max-re-derived affine conversion of a QAT checkpoint — see candidate 2.

## Candidate 2: lattice-aligned (`qat-q4_0-mlx-aligned`) — leading candidate

Revision `745a97a7…` (2026-07-06). A conversion of Google's
`gemma-4-26B-A4B-it-qat-q4_0-unquantized` that **recovers the original QAT
lattice scales from the snapped weights** (k-sweep + least-squares refit per
32-element block) instead of re-deriving scales from min/max statistics, then
emits standard MLX affine parameters (`scale = s`, `bias = -8·s`).

### Why it is the strongest quality candidate

Upstream self-reported fidelity to the bf16 QAT reference (not reproduced
here, but the methodology — including a matched-noise control — is the most
careful of any candidate):

| | default affine gs=64 (the current pin's conversion class) | aligned |
| --- | --- | --- |
| relRMSE | 7.0–8.6% | 0.18–0.23% (bf16 storage noise floor) |
| mean KL (1600-token teacher-forced) | 0.277 | 0.090 |
| top-1 agreement | 82.7% | 90.3% |

The control (bf16 + σ=0.185% random noise, no quantization) measures
KL 0.151 / 87.7%, i.e. the residual error of this conversion sits at the
intrinsic sensitivity floor of the 128-expert sparse routing, not above it.
Caveats: third-party self-evaluation, low traffic (~850 downloads), and the
pinned repo also carries Gemma-4-specific details this project had to match
carefully (split K/V, per-expert scales), so acceptance tests must decide.

### Structural fit

- Tensor-name set is a **strict subset** of the pin's set — everything except
  `vision_tower.*` and `embed_vision.*`, which the repacker already omits. No
  new tensors to interpret.
- `tokenizer.json` byte-identical; `tokenizer_config.json` /
  `chat_template.jinja` differ only in multimodal tool parts;
  `generation_config.json` and `text_config` effectively identical (same
  eos ids 1/106/50, same defaults).
- Router kept in **bf16** (unquantized `router.proj.weight`, no companions) —
  the upstream rationale matches this runtime's own finding that top-8-of-128
  routing is the most perturbation-sensitive spot.
- **Text-only by construction** — no vision tensors to skip.

### Required retrofit (bounded code work, not a redesign)

Unlike MTP, every difference is reachable from the existing architecture:

1. **Parameterize group size (64 → 32).** `Quantization.groupSize = 64`
   (`Sources/Tsugumi/Infrastructure/ModelIO/Quantization.swift:5`) is a
   global constant; it must become a model-level value carried through the
   manifest (`validateQuant` already compares against it per slot). The quantized
   kernels state `n % groupSize` preconditions (DequantInt4/8GEMV,
   EmbedLookupInt4, LMHeadChainInt4, FusedQKVGEMV, prefill primitives) and
   `moe.metal` hardcodes `kMoEGroupSize = 64`; group 32 halves the per-group
   element count, so tile/threadgroup math shrinks rather than grows. Merging
   two 32-groups into one 64-group is not lossless (per-block scales differ),
   so the format must carry group 32 natively.
2. **Add a bf16 router path.** `validateQuant` allows router bits `[8]` only
   and decode `cb1` uses the int8 affine router GEMV. A bf16 GEMV is a small,
   well-understood kernel addition (or an fp16 GEMV after an offline bf16→fp16
   widen, which is exact); the router is resident and tiny relative to expert
   weights, so a 16-bit router also removes a quantization stage from the most
   routing-sensitive projection.
3. **Pin bookkeeping.** New `SourceFingerprint` entry, `SupportedModelSource`
   constants, duplicated constants in `AppModelInstallDescriptor.default`,
   docs updates, and full install + acceptance + benchmark revalidation under
   the community benchmark policy.

Estimated scope: a focused retrofit of constants-and-kernels, plus the
project's standard validation cycle. No change to the expert-streaming, KV,
or scheduling design.

### Cost on the 8 GB target

Group 32 with MLX affine storage costs ~5.0 bits/weight vs ~4.5 at group 64
(+~11%):

- Routed-expert layer files: 12.90 GB → ~14.3 GB.
- Per-expert slot blob: 3,358,720 B → ~3.73 MB; common file 1.35 GB → ~1.5 GB.
- Source shards total 15.79 GB (all text) vs 14.62 GB downloaded today.

This is the real tradeoff: measurable quality headroom in exchange for more
bytes on a machine whose budget is already tight. Whether the 16-slot cache
and I/O pipeline still fit the 8 GB envelope is a question the acceptance and
benchmark runs would answer.

## Candidate 3: MTP assistant (`qat-assistant-4bit`) — not reachable by retrofit

Not a standalone model: a speculative-decoding drafter (`gemma4_assistant`,
`Gemma4AssistantForCausalLM`) — 4 layers, hidden 1024, no MoE, no K/V
projections, `pre_projection`/`post_projection` bridging to the target hidden
size, centroid machinery. 236 MB. This runtime decodes one token at a time in
a fixed `cb1 → io → cb2` pipeline; speculative decoding never reached a
runtime candidate (`docs/experiments/EXPERIMENT_INVENTORY.md:216`). Using it
means designing a propose-and-verify architecture, not adapting weights. The
drafter was also split from the QAT target, so acceptance rates against any
other target are unknown.

## Summary

| Question | Answer |
| --- | --- |
| Why is the non-QAT pin used? | Single-validated-snapshot policy; no recorded comparison. Not a measured choice. |
| QAT 4-bit (candidate 1)? | Structurally adoptable today (runtime already allows 8-bit shared experts); still a min/max-re-derived affine conversion. |
| Lattice-aligned (candidate 2)? | Best quality evidence (self-reported KL 0.090 vs 0.277), text-only, tokenizer-identical, strict-subset tensors. Needs a bounded retrofit: group-32 support, bf16 router path, pin bookkeeping, revalidation. Costs ~11% more expert storage on the 8 GB target. |
| MTP assistant (candidate 3)? | Requires new speculative-decoding architecture; out of retrofit scope. |

Adopting candidate 2 changes source and kernels; per project scope it should
wait for an explicit commission. This document records the analysis so that
decision can be made with the tradeoffs in view.
