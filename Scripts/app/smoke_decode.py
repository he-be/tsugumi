#!/usr/bin/env python3
"""Drive TsugumiDecodeService over its unix socket, like the Mac app does."""
import json
import os
import pathlib
import socket
import struct
import subprocess
import sys
import time
import uuid

# 既定はこのチェックアウトの release ビルド。組み上げた .app の中の decode
# service を叩くときは、その実行ファイルを環境変数で指す:
#   TSUGUMI_DECODE_SERVICE=dist/Tsugumi.app/Contents/MacOS/TsugumiDecodeService
# モデルは既定で scratch/ の中を見る (TSUGUMI_MODEL_DIR で差し替え)。名前は
# .moepack を先に、無ければ改名前の .gturbo を使う。
REPO = os.environ.get("TSUGUMI_REPO") or str(pathlib.Path(__file__).resolve().parents[2])
SERVICE = os.environ.get("TSUGUMI_DECODE_SERVICE") or f"{REPO}/.build/release/TsugumiDecodeService"
MODEL_DIR = os.environ.get("TSUGUMI_MODEL_DIR") or f"{REPO}/scratch"


def installed_model(stem):
    for extension in (".moepack", ".gturbo"):
        candidate = f"{MODEL_DIR}/{stem}{extension}"
        if os.path.exists(f"{candidate}/manifest.json"):
            return candidate
    return f"{MODEL_DIR}/{stem}.moepack"


def send(sock, obj):
    payload = json.dumps(obj).encode()
    sock.sendall(struct.pack("<I", len(payload)) + payload)


def recv(sock):
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise EOFError("socket closed")
        header += chunk
    (count,) = struct.unpack("<I", header)
    payload = b""
    while len(payload) < count:
        chunk = sock.recv(count - len(payload))
        if not chunk:
            raise EOFError("socket closed mid-frame")
        payload += chunk
    return json.loads(payload)


def runtime_options(mtp=True):
    return {
        "expertCacheSlots": 32,
        "expertCachePolicy": "lfu",
        "prefillEnabled": True,
        "prefillChunkTokens": 2048,
        "rdadvisePolicy": "off",
        "modelVerification": "trusted-install",
        "mtpEnabled": mtp,
    }


def load(sock, model_path, ctx, force_logits):
    rid = str(uuid.uuid4()).upper()
    send(sock, {"load": {"_0": {
        "modelPath": model_path,
        "maxContextTokens": ctx,
        "runtimeOptions": runtime_options(),
        "forceLogitsHead": force_logits,
        "requestID": rid,
    }}})
    start = time.time()
    while True:
        ev = recv(sock)
        if ev.get("generationID") == rid and ev["kind"] in ("ready", "failed"):
            print(f"[load] {ev['kind']} in {time.time()-start:.1f}s "
                  f"mem={ev.get('currentMemoryBytes', 0)/1e9:.2f}GB "
                  f"err={ev.get('error')}")
            return ev["kind"] == "ready"


def generate(sock, label, prompt, ctx, *, temperature, top_k, top_p,
             thinking, images=(), max_new=96):
    gid = str(uuid.uuid4()).upper()
    send(sock, {"generate": {"_0": {
        "prompt": prompt,
        "maxNewTokens": max_new,
        "maxContextTokens": ctx,
        "temperature": temperature,
        "topK": top_k,
        "topP": top_p,
        "repetitionPenalty": 1,
        "enableThinking": thinking,
        "imagePaths": list(images),
        "runtimeOptions": runtime_options(),
        "generationID": gid,
    }}})
    text, reasoning = [], []
    start = time.time()
    while True:
        ev = recv(sock)
        if ev.get("generationID") != gid:
            continue
        kind = ev["kind"]
        if kind == "prefill":
            continue
        if kind == "snapshot":
            text.append(ev.get("textDelta", ""))
            reasoning.append(ev.get("reasoningDelta") or "")
            continue
        print(f"[{label}] {kind} in {time.time()-start:.1f}s "
              f"tokens={ev.get('tokenCount')} prompt={ev.get('promptTokenCount')} "
              f"cached={ev.get('cachedPromptTokens')} "
              f"tok/s={ev.get('tokensPerSecond', 0):.2f} "
              f"draft={ev.get('draftAccepted')}/{ev.get('draftProposed')} "
              f"stop={ev.get('stopReason')} err={ev.get('error')}")
        joined_reasoning = "".join(reasoning)
        joined = "".join(text)
        if joined_reasoning:
            print(f"[{label}] REASONING ({len(joined_reasoning)} ch): "
                  f"{joined_reasoning[:200]!r}...")
        print(f"[{label}] TEXT: {joined[:400]!r}")
        return kind == "finished"


def session(model_path, ctx, runs):
    proc = subprocess.Popen([SERVICE, "--socket", SOCKET_PATH],
                            stdout=sys.stdout, stderr=sys.stderr)
    try:
        sock = None
        for _ in range(200):
            try:
                sock = socket.socket(socket.AF_UNIX)
                sock.connect(SOCKET_PATH)
                break
            except OSError:
                sock.close()
                sock = None
                time.sleep(0.05)
        if sock is None:
            raise RuntimeError("could not connect")
        ok = load(sock, model_path, ctx, True)
        if ok:
            for run in runs:
                generate(sock, **run, ctx=ctx)
        send(sock, {"shutdown": {}})
        time.sleep(1)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


SOCKET_PATH = f"/tmp/tf-smoke-{uuid.uuid4().hex[:8]}.sock"

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "gemma"
    if which == "gemma":
        runs = [
            dict(label="gemma-plain", prompt="1+1は?ひとことで。",
                 temperature=1.0, top_k=64, top_p=0.95, thinking=False,
                 max_new=48),
            dict(label="gemma-think", prompt="7×8は?ひとことで。",
                 temperature=1.0, top_k=64, top_p=0.95, thinking=True,
                 max_new=128),
        ]
        if len(sys.argv) > 2:
            runs.append(dict(label="gemma-vision",
                             prompt="この画像を一文で説明して。",
                             temperature=1.0, top_k=64, top_p=0.95,
                             thinking=False, images=[sys.argv[2]],
                             max_new=64))
        session(installed_model("gemma4-qat-sym"), 8192, runs)
    else:
        runs = [
            dict(label="ornith-think", prompt="1+1は?ひとことで。",
                 temperature=0.6, top_k=20, top_p=0.95, thinking=True,
                 max_new=256),
            dict(label="ornith-plain", prompt="7×8は?ひとことで。",
                 temperature=0.6, top_k=20, top_p=0.95, thinking=False,
                 max_new=64),
        ]
        session(installed_model("ornith-oq4e-g64"), 8192, runs)
