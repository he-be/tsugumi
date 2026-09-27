#!/bin/sh
# S2 / S3 of docs/qwen38-27b/16-M6-ISHIZUKI-AGENT-PLAN.md: exercise every endpoint and field
# that ishizuki-serve is read to support (§1-2 "ある") and the llama-server routes api.patch adds
# for the Tsugumi client (S3), with curl only, and keep each raw response.
#   bench/m6/ishizuki_api_check.sh [endpoint] [out dir]
# endpoint defaults to http://127.0.0.1:8128 (the ssh forward to the M6). Each case writes
# <case>.req.json and <case>.resp (body, or the SSE stream as received) and prints one line
# with the HTTP status, seconds, and the check. Exit 1 if any check fails.
set -u
ep=${1:-http://127.0.0.1:8128}
out=${2:-$(cd "$(dirname "$0")/../.." && pwd)/scratch/bonsai27b/ishizuki-api/$(date +%Y%m%d-%H%M%S)}
model=ternary-bonsai-2-27b
mkdir -p "$out"
fails=0

post() { # case path json -> writes files, sets code / secs
    printf '%s\n' "$2" > "$out/$1.req.json"
    r=$(curl -s -m 600 -o "$out/$1.resp" -w '%{http_code} %{time_total}' \
        -H 'content-type: application/json' --data-binary @"$out/$1.req.json" "$ep$3")
    code=${r% *}; secs=${r#* }
}
check() { # case description python-expression-over-body
    if python3 - "$out/$1.resp" "$3" <<'EOF'
import json, sys
raw = open(sys.argv[1]).read()
def sse():
    return [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
body = None
try: body = json.loads(raw)
except Exception: pass
msg = body["choices"][0].get("message", {}) if body and "choices" in body else {}
ok = eval(sys.argv[2])
sys.exit(0 if ok else 1)
EOF
    then echo "ok    $1 ($code, ${secs}s) $2"
    else echo "FAIL  $1 ($code, ${secs}s) $2"; fails=$((fails + 1)); fi
}

for p in /health /v1/models; do
    name=get$(echo "$p" | tr '/' '_')
    r=$(curl -s -m 10 -o "$out/$name.resp" -w '%{http_code} %{time_total}' "$ep$p")
    code=${r% *}; secs=${r#* }
    check "$name" "GET $p answers with the model id" "'$model' in raw"
done

user='{"role":"user","content":"日本の首都はどこですか。一文で答えてください。"}'
post chat-plain "{\"model\":\"$model\",\"messages\":[$user],\"max_tokens\":64}" /v1/chat/completions
check chat-plain "non-stream answer, no think markup in content" \
    "'東京' in msg.get('content','') and '<think>' not in msg.get('content','') and body['usage']['completion_tokens'] <= 64"

post chat-stream "{\"model\":\"$model\",\"messages\":[$user],\"max_tokens\":64,\"stream\":true}" /v1/chat/completions
check chat-stream "SSE chunks carry the answer and end with [DONE]" \
    "'東京' in ''.join((c['choices'][0].get('delta') or {}).get('content') or '' for c in sse() if c.get('choices')) and raw.rstrip().endswith('[DONE]')"

post max-tokens "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"1 から 100 まで数字を空白区切りで書いてください。\"}],\"max_tokens\":8}" /v1/chat/completions
check max-tokens "max_tokens 8 stops at 8 with finish_reason length" \
    "body['usage']['completion_tokens'] <= 8 and body['choices'][0]['finish_reason'] == 'length'"

tools='[{"type":"function","function":{"name":"get_weather","description":"指定した都市の今日の天気を返す","parameters":{"type":"object","properties":{"city":{"type":"string","description":"都市名"}},"required":["city"]}}}]'
ask='{"role":"user","content":"大阪の今日の天気を調べてください。"}'
post tools-call "{\"model\":\"$model\",\"messages\":[$ask],\"tools\":$tools,\"max_tokens\":256}" /v1/chat/completions
check tools-call "tools: a get_weather call with a city argument" \
    "msg.get('tool_calls') and msg['tool_calls'][0]['function']['name'] == 'get_weather' and 'city' in json.loads(msg['tool_calls'][0]['function']['arguments'])"

call_id=$(python3 -c 'import json,sys; b=json.load(open(sys.argv[1])); print(b["choices"][0]["message"]["tool_calls"][0]["id"])' "$out/tools-call.resp" 2>/dev/null || echo call_0)
args=$(python3 -c 'import json,sys; b=json.load(open(sys.argv[1])); print(json.dumps(b["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]))' "$out/tools-call.resp" 2>/dev/null || echo '"{\"city\":\"大阪\"}"')
post tools-result "{\"model\":\"$model\",\"messages\":[$ask,{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"$call_id\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":$args}}]},{\"role\":\"tool\",\"tool_call_id\":\"$call_id\",\"content\":\"{\\\"city\\\":\\\"大阪\\\",\\\"weather\\\":\\\"雨のち曇り\\\",\\\"high_c\\\":23}\"}],\"tools\":$tools,\"max_tokens\":256}" /v1/chat/completions
check tools-result "tool role round trip: the answer uses the tool result" \
    "not msg.get('tool_calls') and ('雨' in msg.get('content','') or '23' in msg.get('content',''))"

post tools-stream "{\"model\":\"$model\",\"messages\":[$ask],\"tools\":$tools,\"max_tokens\":256,\"stream\":true}" /v1/chat/completions
check tools-stream "streamed tool call arrives as delta.tool_calls" \
    "any((c['choices'][0].get('delta') or {}).get('tool_calls') for c in sse() if c.get('choices'))"

schema='{"type":"json_schema","json_schema":{"name":"capital","schema":{"type":"object","properties":{"country":{"type":"string"},"capital":{"type":"string"}},"required":["country","capital"],"additionalProperties":false}}}'
post json-schema "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"フランスの首都を JSON で。\"}],\"response_format\":$schema,\"max_tokens\":128}" /v1/chat/completions
check json-schema "response_format json_schema: content parses and has both keys" \
    "set(json.loads(msg['content'])) == {'country','capital'}"

# ishizuki constrains decoding only over finite value sets (enum, bounded integer, boolean);
# a free string is refused with 400 (the case above). This one checks the mechanism itself.
enum='{"type":"json_schema","json_schema":{"name":"capital","schema":{"type":"object","properties":{"capital":{"type":"string","enum":["パリ","リヨン","マルセイユ"]}},"required":["capital"],"additionalProperties":false}}}'
post json-schema-enum "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"フランスの首都を JSON で。\"}],\"response_format\":$enum,\"max_tokens\":128}" /v1/chat/completions
check json-schema-enum "response_format json_schema with an enum: content parses to the capital" \
    "json.loads(msg['content']) == {'capital': 'パリ'}"

post think-off "{\"model\":\"$model\",\"messages\":[$user],\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}" /v1/chat/completions
check think-off "enable_thinking false: no reasoning, no think markup" \
    "not msg.get('reasoning_content') and '<think>' not in msg.get('content','') and '</think>' not in msg.get('content','')"

post think-on "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"17 × 23 はいくつですか。\"}],\"max_tokens\":2048,\"chat_template_kwargs\":{\"enable_thinking\":true}}" /v1/chat/completions
check think-on "enable_thinking true: reasoning kept out of content, answer 391" \
    "'391' in msg.get('content','') and '<think>' not in msg.get('content','') and bool(msg.get('reasoning_content') or msg.get('reasoning'))"

# ---- S3: the llama-server routes and fields the Tsugumi client uses (api.patch) ----
check max-tokens "timings in llama-server's shape" \
    "all(k in body.get('timings', {}) for k in ('cache_n','prompt_n','prompt_ms','predicted_n','predicted_ms','predicted_per_second'))"
check chat-stream "streamed: last chunk carries timings" \
    "any('timings' in c for c in sse())"

r=$(curl -s -m 10 -o "$out/props.resp" -w '%{http_code} %{time_total}' "$ep/props"); code=${r% *}; secs=${r#* }
check props "GET /props reports n_ctx" "body['default_generation_settings']['n_ctx'] > 0"

post tokenize '{"content":"<tool_call>","parse_special":true}' /tokenize
check tokenize "<tool_call> is one token with parse_special" "len(body['tokens']) == 1"
open_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens"][0])' "$out/tokenize.resp" 2>/dev/null || echo 0)

post apply-template "{\"model\":\"$model\",\"messages\":[$ask],\"tools\":$tools,\"chat_template_kwargs\":{\"enable_thinking\":false}}" /apply-template
check apply-template "apply-template renders tools and the generation prompt" \
    "'get_weather' in body['prompt'] and body['prompt'].rstrip().endswith(('<|im_start|>assistant', '</think>'))"

# A forced round as RemoteInferenceClient.forcedRound sends it (qwen dialect).
prompt=$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["prompt"] + "<tool_call>\n<function=get_weather>\n"))' "$out/apply-template.resp")
post forced "{\"model\":\"$model\",\"prompt\":$prompt,\"stop\":[\"</tool_call>\"],\"max_tokens\":1024,\"cache_prompt\":true,\"temperature\":0.7,\"top_k\":20,\"top_p\":0.8,\"min_p\":0,\"presence_penalty\":1.5}" /v1/completions
check forced "/v1/completions continues the forced call to the stop string" \
    "body['choices'][0]['finish_reason'] == 'stop' and '<parameter=city>' in body['choices'][0]['text'] and '</tool_call>' not in body['choices'][0]['text'] and 'timings' in body"

post completion-length "{\"model\":\"$model\",\"prompt\":\"1, 2, 3, 4,\",\"max_tokens\":5}" /v1/completions
check completion-length "/v1/completions stops at max_tokens with length" \
    "body['choices'][0]['finish_reason'] == 'length' and body['usage']['completion_tokens'] == 5"

# A none round: tools declared, the call-opening token banned.
post none-ban "{\"model\":\"$model\",\"messages\":[$ask],\"tools\":$tools,\"max_tokens\":256,\"logit_bias\":[[$open_id,false]],\"chat_template_kwargs\":{\"enable_thinking\":false}}" /v1/chat/completions
check none-ban "logit_bias [[<tool_call>, false]] keeps the round from calling" \
    "not msg.get('tool_calls') and '<tool_call>' not in (msg.get('content') or '') and body['choices'][0]['finish_reason'] in ('stop','length')"

post bias-refused "{\"model\":\"$model\",\"messages\":[$user],\"max_tokens\":8,\"logit_bias\":{\"$open_id\":5}}" /v1/chat/completions
check bias-refused "a logit_bias that is not a ban is refused (400)" "'$code' == '400'"

# The sampler a request ran with comes back as generation_settings (api.patch). Output equality
# is no probe for it: two identical requests part ways where one prefill hit the prefix cache
# and the other did not (16 §10), so the check reads the settings instead.
post sampling "{\"model\":\"$model\",\"messages\":[$user],\"max_tokens\":16,\"temperature\":0.6,\"top_k\":7,\"top_p\":0.9,\"min_p\":0.05,\"presence_penalty\":0.5,\"seed\":7}" /v1/chat/completions
check sampling "per-request temperature / top_k / top_p / min_p / presence_penalty / seed are used" \
    "(lambda g: abs(g['temperature']-0.6)<1e-6 and g['top_k']==7 and abs(g['top_p']-0.9)<1e-6 and abs(g['min_p']-0.05)<1e-6 and abs(g['presence_penalty']-0.5)<1e-6 and g['seed']==7)(body['generation_settings'])"
post sampling-default "{\"model\":\"$model\",\"messages\":[$user],\"max_tokens\":16}" /v1/chat/completions
check sampling-default "without them the server's official set is used" \
    "(lambda g: abs(g['temperature']-0.7)<1e-6 and g['top_k']==20 and abs(g['top_p']-0.8)<1e-6 and g['min_p']==0 and abs(g['presence_penalty']-1.5)<1e-6)(body['generation_settings'])"
check none-ban "the none round ran with the ban" "$open_id in json.load(open('$out/none-ban.resp'))['generation_settings']['banned_tokens']"

# Past n_ctx: the prompt is refused before any stream starts, in llama-server's shape.
python3 -c '
import json, sys
print(json.dumps({"model": sys.argv[1], "stream": True, "max_tokens": 16,
  "messages": [{"role": "user", "content": "りんご " * 40000}]}))' "$model" > "$out/overflow.req.json"
r=$(curl -s -m 600 -o "$out/overflow.resp" -w '%{http_code} %{time_total}' -H 'content-type: application/json' --data-binary @"$out/overflow.req.json" "$ep/v1/chat/completions")
code=${r% *}; secs=${r#* }
check overflow "a prompt past n_ctx is a 400 exceed_context_size_error" \
    "'$code' == '400' and body['error']['type'] == 'exceed_context_size_error' and body['error']['n_prompt_tokens'] > body['error']['n_ctx']"

echo "responses in $out"
[ "$fails" -eq 0 ] || { echo "$fails failed"; exit 1; }
