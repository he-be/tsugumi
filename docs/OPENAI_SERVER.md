# Local OpenAI-compatible server

`TurboFieldfareServer` exposes a local Chat Completions API for one Gemma
model. It binds to `127.0.0.1` without authentication or TLS. Do not expose it
through a proxy or tunnel.

## Start the server

First, install the model with the Mac app or `TurboFieldfareRepack`. Then check
that no other TurboFieldfare model process is running:

```bash
pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

If the command prints a match, do not start the server.

```bash
swift build -c release --product TurboFieldfareServer
.build/release/TurboFieldfareServer \
  --model scratch/gemma4.gturbo \
  --port 8080 \
  --max-context 16384
```

The server loads the model before opening the port. Wait for
`TurboFieldfareServer ready`, then keep the process running while clients use
it.

Check the server from another terminal:

```bash
curl --silent --show-error http://127.0.0.1:8080/health
curl --silent --show-error http://127.0.0.1:8080/v1/models
curl --silent --show-error http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-26b-a4b-it",
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "temperature": 0,
    "max_completion_tokens": 16
  }'
```

By default, the server runs one generation and admits up to four additional
requests for preparation or queueing. The limit is enforced before prompt
rendering and tokenization. Use `--queue-limit` to change it. Press Control-C
to stop the server.

## Runtime settings

The server accepts the same runtime flags as the CLI, with the same values and
defaults. See [Runtime controls](RUNTIME_CONTROLS.md) for what each one does.

```bash
.build/release/TurboFieldfareServer \
  --model scratch/gemma4.gturbo \
  --expert-cache-slots 32 \
  --expert-cache-policy lru \
  --prefill on \
  --prefill-chunk-tokens 64 \
  --rdadvise bounded
```

Without these flags the server runs the production defaults: 48 expert-cache
slots, LFU eviction, chunked prefill on with 2048-token chunks, and read advice
off. Values are validated before the model loads, so an unsupported one exits
with the usage text rather than failing partway through startup. Chunked
prefill needs at least 16 expert-cache slots, so `--expert-cache-slots 8`
requires `--prefill off`.

The settings are fixed for the life of the process. Restart the server to
change them.

## Connect a client

The base URL is `http://127.0.0.1:8080/v1`. Some client libraries require an
API key, but the server ignores it.

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
response = client.chat.completions.create(
    model="gemma-4-26b-a4b-it",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

OpenCode:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "turbofieldfare": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "TurboFieldfare",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local"
      },
      "models": {
        "gemma-4-26b-a4b-it": {
          "name": "Gemma 4 26B-A4B IT",
          "limit": {
            "context": 16384,
            "output": 4096
          }
        }
      }
    }
  }
}
```

Select `turbofieldfare/gemma-4-26b-a4b-it` in OpenCode.

Pi uses its `openai-completions` adapter:

```json
{
  "providers": {
    "turbofieldfare": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": {
        "supportsReasoningEffort": false,
        "supportsStrictMode": false,
        "supportsUsageInStreaming": true
      },
      "models": [{
        "id": "gemma-4-26b-a4b-it",
        "name": "Gemma 4 26B-A4B IT",
        "reasoning": false,
        "contextWindow": 16384,
        "maxTokens": 4096
      }]
    }
  }
}
```

Keep the client context setting at or below the server's `--max-context`.

## Images

A model installed with a vision tower accepts images as `image_url` content
parts on user messages. Install the tower with
`TurboFieldfareRepack --add-vision --input-gturbo <model.gturbo>`; a model
without one answers image requests with HTTP 400 and `vision_not_installed`.

```bash
curl --silent --show-error http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-26b-a4b-it",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "What is in this picture?"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]}],
    "temperature": 0.2,
    "max_completion_tokens": 256
  }'
```

**Only `data:` URIs are accepted.** The server never fetches an image URL: an
`http(s)` or `file` URL returns HTTP 400 with `unsupported_image_url`. Fetching
a caller-supplied URL would make the server issue requests on behalf of whoever
can reach it, and a local inference server has no reason to do that.

Limits, all configurable at startup:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--image-tokens` | 280 | Soft-token budget per image: 70, 140, or 280 |
| `--max-images` | 4 | Images per request |
| `--max-image-bytes` | 8388608 | Decoded bytes per image |
| `--max-image-pixels` | 50000000 | Pixels per image, read from the header |

Exceeding a limit returns HTTP 413 (`image_too_large`, `too_many_images`). The
request-body ceiling rises with `--max-images` and `--max-image-bytes`, so
base64 payloads within the limits are not cut off by the transport.

`--image-tokens` is an upper bound, not the count. Each image is resized to a
whole number of 48-pixel cells, so the soft tokens it occupies follow its aspect
ratio: 256 for a square image at the default budget, 266 for a 4:3 one. Those
tokens are part of `usage.prompt_tokens`.

Two conditions are refused rather than approximated: images together with
`tools` (the tool template renders text only), and images while the server runs
`--prefill off` (the unchunked path has nowhere to place a soft token).

Prompt reuse is disabled for any request carrying an image, in both directions:
such a request never resumes a cached prefix and never publishes one. The cache
matches on message text, so two requests with the same words and different
pictures are indistinguishable to it. A conversation that continues after an
image therefore prefills in full each turn.

Audio and video are not supported. Writing `<|image|>`, `<|audio|>`, or
`<|video|>` into message text is an error, not a silent pass-through.

## Prompt reuse

Single-prefix KV reuse is on by default. Send the complete message history with
every request. When a request continues the retained conversation exactly, the
server reuses the verified KV prefix and reports the number of reused tokens in:

```text
usage.prompt_tokens_details.cached_tokens
```

The server retains one prefix. A different or incompatible history replaces
it. Use `--prompt-cache-mode off` to disable reuse.

## Tool calls

The server can return OpenAI-style function calls, but it cannot authorize or
execute them. The client runs the tool loop:

1. Send function schemas in `tools`.
2. When `finish_reason` is `"tool_calls"`, inspect each function name and JSON
   argument object. Apply the client's normal permission checks before running
   the function.
3. Append the assistant message, including its unchanged `tool_calls`.
4. Append each result as a `role: "tool"` message. Its `tool_call_id` must
   match the call it resolves.
5. Send the complete history and tool schemas again.

The server accepts only function tools. Omit `tool_choice` or set it to `auto`
to allow calls. Set it to `none` to disable them. The server does not support
`required`, named tool selection, or `parallel_tool_calls: false`.

Tool schemas need a non-null object at the top level and explicit JSON Schema
types. Nested properties and items may use nullable forms with one concrete
type plus `null`, including equivalent two-branch `anyOf` and disjoint `oneOf`
forms. Unions of string constants are also supported. Overlapping `oneOf`,
mixed-type unions such as `string | object`, and `allOf` return HTTP 400 with
`invalid_tool_schema`; the server does not guess which branch the model should
use.

## Supported API

Endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Chat Completions supports JSON and Server-Sent Events responses. Set
`"stream": true` for streaming. Set
`"stream_options": {"include_usage": true}` to receive a final usage chunk.

Requests may contain system, developer, user, assistant, and tool messages.
Message content may be a string or a list of `text` and `image_url` parts.
Supported options include `temperature`, `top_p`, `top_k`,
`repetition_penalty`, `seed`, `stop`, `max_tokens`,
`max_completion_tokens`, and function-tool fields.

The server supports one model and one choice. Images are supported as described
above; audio and video are not. It does not support the Responses API, legacy
Completions, embeddings, structured output, batching, log probabilities, or
remote model switching.

Context length can be 4K, 8K, 16K, 32K, or 64K. The default is 16K. Larger FP16
KV contexts use more memory. On an 8 GB Mac, run one model process at a time and
watch memory pressure.

For long requests, stderr reports the request lifecycle as prepared, queued,
generating, completed, or failed. It includes token counts and timing, but not
prompt text, tool arguments, headers, or request bodies.
