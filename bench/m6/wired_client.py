"""The request mix bench/m6/wired_probe.sh sends to a TsugumiServer on 127.0.0.1:8081.

One round is the shape of the exhibit's reading: a streamed description of one picture
(128 tokens), then three short follow-ups on the same conversation (32 tokens each), so the
prompt cache is exercised the way a client uses it. Prints one line per request.

    python3 -I bench/m6/wired_client.py <image> <rounds>
"""
import base64
import json
import sys
import time
import urllib.request

image, rounds = sys.argv[1], int(sys.argv[2])
mime = "image/png" if image.endswith(".png") else "image/jpeg"
data = base64.b64encode(open(image, "rb").read()).decode()


def post(messages, max_tokens, stream=False):
    body = json.dumps({"model": "m", "messages": messages,
                       "max_tokens": max_tokens, "stream": stream}).encode()
    request = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=600) as response:
        response.read()
    return time.time() - started


first = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
         {"type": "text", "text": "この写真に写っているものを、日本語で 3 文で説明してください。"}]
for n in range(rounds):
    messages = [{"role": "user", "content": first}]
    print(f"round {n} long {post(messages, 128, stream=True):.2f}s", flush=True)
    for question in ["一言で言うと何ですか。", "英語の名前を 1 つ挙げてください。", "色を 3 つ挙げてください。"]:
        follow = messages + [{"role": "assistant", "content": "了解しました。"},
                             {"role": "user", "content": question}]
        print(f"round {n} short {post(follow, 32):.2f}s", flush=True)
