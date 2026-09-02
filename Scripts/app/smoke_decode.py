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
             thinking, images=(), max_new=96, system_prompt=None,
             continuation=(), tools=(), tool_choice=None, reasoning_budget=None):
    gid = str(uuid.uuid4()).upper()
    request = {
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
    }
    # ツールループの経路 (Mac アプリの Web 検索が通す形): system prompt、
    # 宣言するツール、tool_choice、そして同じ user ターンに続く
    # assistant(tool_calls) + tool の継続ターン。
    if system_prompt:
        request["systemPrompt"] = system_prompt
    if continuation:
        request["continuation"] = list(continuation)
    if tools:
        request["tools"] = list(tools)
    if tool_choice:
        request["toolChoice"] = tool_choice
    if reasoning_budget is not None:
        request["reasoningBudgetTokens"] = reasoning_budget
    send(sock, {"generate": {"_0": request}})
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
            limit = int(os.environ.get("TSUGUMI_SMOKE_REASONING_CHARS", "200"))
            print(f"[{label}] REASONING ({len(joined_reasoning)} ch): "
                  f"{joined_reasoning[:limit]!r}...")
        print(f"[{label}] TEXT: {joined[:400]!r}")
        for call in ev.get("toolCalls") or []:
            print(f"[{label}] TOOL CALL {call['name']} {call['argumentsJSON']}")
        LAST_TOOL_CALLS[:] = ev.get("toolCalls") or []
        return kind == "finished"


LAST_TOOL_CALLS = []

WEB_SEARCH_TOOLS = [
    {"name": "web_search",
     "description": "Web を検索して、上位の結果のタイトル・URL・スニペットを返す。",
     "parametersJSON": json.dumps({"type": "object", "properties": {
         "query": {"type": "string", "description": "検索クエリ"}},
         "required": ["query"]}, ensure_ascii=False)},
    {"name": "fetch_page",
     "description": "URL を開いてページ本文のテキストを返す。",
     "parametersJSON": json.dumps({"type": "object", "properties": {
         "url": {"type": "string", "description": "読むページの URL"}},
         "required": ["url"]}, ensure_ascii=False)},
]

WIKIPEDIA_TOOLS = [
    {"name": "wikipedia_search",
     "description": "この Mac に保存された日本語版 Wikipedia (オフライン) を検索して、該当する記事の題名と導入部を返す。",
     "parametersJSON": json.dumps({"type": "object", "properties": {
         "query": {"type": "string", "description": "検索語"}},
         "required": ["query"]}, ensure_ascii=False)},
    {"name": "wikipedia_page",
     "description": "日本語版 Wikipedia の記事を 1 つ開いて本文を返す。本文が打ち切られたときは from に示された文字位置を渡すと続きが読める。",
     "parametersJSON": json.dumps({"type": "object", "properties": {
         "title": {"type": "string", "description": "記事の題名"},
         "from": {"type": "integer", "description": "本文を読み始める文字位置"}},
         "required": ["title"]}, ensure_ascii=False)},
]


# アプリと同じ system prompt (Sources/TsugumiApp/Core/Resources/ のリソース)。
# 雛形の穴を WebSearchPrompt.system と同じ規則で埋め、日付だけ固定する。
def tool_system_prompt(today="2026年9月2日 (水)", max_rounds=6, web=True, wiki_date=None):
    base = f"{REPO}/Sources/TsugumiApp/Core/Resources/"
    with open(base + "web-search-system-prompt.txt", encoding="utf-8") as handle:
        text = handle.read().strip()
    with open(base + "search-tool-prompts.json", encoding="utf-8") as handle:
        snippets = json.load(handle)
    wikipedia = wiki_date is not None
    web = web or not wikipedia

    def snippet(section, key):
        return snippets[section][key].replace("{wiki_date}", wiki_date or "不明な日付")

    sections = (["wikipedia"] if wikipedia else []) + (["web"] if web else [])
    choice = (snippet("both", "choice") if wikipedia and web
              else snippet("wikipedia", "choice_alone") if wikipedia else "")
    text = (text.replace("{today}", today)
            .replace("{max_rounds}", str(max_rounds))
            .replace("{tool_names}", "、".join(snippet(s, "names") for s in sections))
            .replace("{tool_access}", "\n".join(snippet(s, "access") for s in sections))
            .replace("{tool_choice}\n", choice + "\n" if choice else "")
            .replace("{tool_reading}", "\n".join(snippet(s, "reading") for s in sections))
            .replace("{reference_format}", "や".join(snippet(s, "reference") for s in sections)))
    assert "{" not in text, text
    return text


TOOL_SYSTEM_PROMPT = tool_system_prompt()


def tool_session(model_path, ctx):
    """One forced search round, then the answer from a canned result.

    Checks the whole app path: the tool template renders, the grammar pins a
    call when tool_choice is required, the parser returns it on the terminal
    event, and the continuation turns (assistant tool_calls + tool result)
    redraw so the model answers from the result."""
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
        if load(sock, model_path, ctx, True):
            sampler = dict(temperature=1.0, top_k=64, top_p=0.95, thinking=False)
            prompt = "今日の東京の天気は?"
            # Round 1: required → the grammar forces a call.
            generate(sock, "tools-round1", prompt, ctx, max_new=96,
                     system_prompt=TOOL_SYSTEM_PROMPT, tools=WEB_SEARCH_TOOLS,
                     tool_choice="required", **sampler)
            calls = list(LAST_TOOL_CALLS)
            if calls:
                result = ("検索: 東京 天気 (Serper, 2 件、取得日 2026年9月2日)\n"
                          "[1] 東京都の天気 - 気象庁\n    https://www.jma.go.jp/bosai/forecast/\n"
                          "    2026-09-02 東京地方 今日は晴れ時々曇り、最高気温 31 度。\n"
                          "[2] tenki.jp 東京\n    https://tenki.jp/forecast/3/16/\n"
                          "    晴れ、降水確率 10%。")
                continuation = [
                    {"role": "assistant", "text": "", "toolCalls": calls},
                ] + [
                    {"role": "tool", "text": result,
                     "toolCallID": c["id"], "toolName": c["name"]}
                    for c in calls
                ]
                # Round 2: auto → the model may answer or call again.
                generate(sock, "tools-round2", prompt, ctx, max_new=256,
                         system_prompt=TOOL_SYSTEM_PROMPT, tools=WEB_SEARCH_TOOLS,
                         continuation=continuation, **sampler)
            else:
                print("[tools] round 1 produced no tool call")
            # Auto mode with a question that needs no search: the model
            # should answer directly.
            generate(sock, "tools-auto-plain", "1+1は?ひとことで。", ctx, max_new=48,
                     system_prompt=TOOL_SYSTEM_PROMPT, tools=WEB_SEARCH_TOOLS,
                     **sampler)
        send(sock, {"shutdown": {}})
        time.sleep(1)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def wiki_session(model_path, ctx, index_path, question):
    """The local Wikipedia tools against a real index, one loop.

    Round 1 forces a call; the call is answered by the same search / page
    code the app uses (build_jawiki_index.py's search() and page(), which
    mirror LocalWikipediaIndex); the loop continues until the model answers
    or six rounds are spent. Thinking is on, so this is also the
    simulation-sickness check for the offline tool."""
    sys.path.insert(0, f"{REPO}/Scripts/wiki")
    import build_jawiki_index as wiki  # noqa: E402
    import sqlite3
    conn = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    conn.close()
    date = meta.get("dump_date", "")
    wiki_date = f"{int(date[:4])}年{int(date[4:6])}月{int(date[6:])}日" if len(date) == 8 else "不明な日付"
    system_prompt = tool_system_prompt(web=False, wiki_date=wiki_date)
    print(f"[wiki] {index_path}: {meta.get('articles')} articles, dump {date}")

    def run_tool(call):
        args = json.loads(call.get("argumentsJSON") or call.get("arguments") or "{}")
        if call["name"] == "wikipedia_search":
            hits = wiki.search(index_path, args.get("query", ""), 8)
            if not hits:
                return f"Wikipedia 検索: {args.get('query')} ({wiki_date} 時点) — 該当する記事はありません。"
            lines = [f"Wikipedia 検索: {args.get('query')} ({len(hits)} 件、{wiki_date} 時点の複製)"]
            for i, (pid, title, snip, _) in enumerate(hits, 1):
                lines.append(f"[{i}] {title}\n    {snip}")
            lines.append("\n本文を読むには wikipedia_page に題名を渡します。")
            return "\n".join(lines)
        if call["name"] == "wikipedia_page":
            found = wiki.page(index_path, args.get("title", ""))
            if not found:
                near = wiki.search(index_path, args.get("title", ""), 5)
                return (f"Wikipedia に「{args.get('title')}」という記事はありません。"
                        + ("\n近い題名: " + " / ".join(t for _, t, _, _ in near) if near else ""))
            title, text = found
            start = int(args.get("from") or 0)
            piece = text[start:start + 6000]
            tail = (f"\n…(本文はここで打ち切り。全 {len(text)} 文字。続きは from={start + 6000} で読めます)"
                    if start + 6000 < len(text) else "")
            return f"Wikipedia 記事: {title}\n\n{piece}{tail}"
        return f"error: unknown tool {call['name']}"

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
        if load(sock, model_path, ctx, True):
            sampler = dict(temperature=1.0, top_k=64, top_p=0.95, thinking=True)
            continuation = []
            for round_index in range(6):
                first = round_index == 0
                generate(sock, f"wiki-round{round_index + 1}", question, ctx, max_new=1024,
                         system_prompt=system_prompt, tools=WIKIPEDIA_TOOLS,
                         tool_choice="required" if first else "auto",
                         continuation=continuation,
                         **{**sampler, "thinking": not first})
                calls = list(LAST_TOOL_CALLS)
                if not calls:
                    break
                continuation.append({"role": "assistant", "text": "", "toolCalls": calls})
                for call in calls:
                    result = run_tool(call)
                    print(f"[wiki] {call['name']} {call.get('argumentsJSON')} → {len(result)} chars")
                    print("       " + result[:300].replace("\n", " / "))
                    continuation.append({"role": "tool", "text": result,
                                         "toolCallID": call["id"], "toolName": call["name"]})
        send(sock, {"shutdown": {}})
        time.sleep(1)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


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
    if which == "wiki":
        index_path = os.environ.get("TSUGUMI_WIKIPEDIA_INDEX") or os.path.expanduser(
            "~/Library/Application Support/Tsugumi/wikipedia-ja.sqlite")
        question = sys.argv[2] if len(sys.argv) > 2 else "東京タワーの高さは?"
        wiki_session(installed_model("gemma4-qat-sym"), 8192, index_path, question)
    elif which == "tools":
        tool_session(installed_model("gemma4-qat-sym"), 8192)
    elif which == "think-tools":
        # thinking ON で最初のラウンドだけ: 思考チャネルが何に何トークン
        # 使うかを見る (TSUGUMI_SMOKE_PROMPT / TSUGUMI_SMOKE_SYSTEM で差し替え)。
        prompt = os.environ.get("TSUGUMI_SMOKE_PROMPT", "9/1の生成AIニュースを調べて")
        system = os.environ.get("TSUGUMI_SMOKE_SYSTEM") or TOOL_SYSTEM_PROMPT
        # 既定は Auto と同じ 512 トークンの思考予算。-1 で無制限。
        budget = int(os.environ.get("TSUGUMI_SMOKE_BUDGET", "512"))
        runs = []
        for i in range(int(os.environ.get("TSUGUMI_SMOKE_REPEAT", "2"))):
            runs.append(dict(label=f"think-round1-{i+1}", prompt=prompt,
                             temperature=1.0, top_k=64, top_p=0.95, thinking=True,
                             max_new=2048, system_prompt=system,
                             tools=WEB_SEARCH_TOOLS,
                             reasoning_budget=None if budget < 0 else budget))
        session(installed_model("gemma4-qat-sym"), 8192, runs)
    elif which == "gemma":
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
