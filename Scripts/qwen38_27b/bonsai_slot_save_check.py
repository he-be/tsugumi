#!/usr/bin/env python3
"""llama-server の --slot-save-path で「別の会話に移ってから戻る」を確かめる (docs/qwen38-27b/12 §6-2)。

1 つのサーバで: A (長い会話) → save → A の続き (基準、cache が生きている) → B (別の会話、切り替え)
→ restore → A の続き (復元) → B → A の続き (復元なし、読み直し)。
各要求の prompt_n / cache_n / prompt_ms、続きの 1 トークン目の top-10 logprob、MTP の受理を残す。
サンプリングは公式値 (thinking 無効: 0.7 / 0.8 / 20 / min_p 0 / presence 1.5) のまま。
Swapouts が 60 秒で +512 MiB か合計 +1 GiB でサーバを止める。

  bonsai_slot_save_check.py OUT [--tokens 16000] [--diverge 120] [--server-args "-ctk q8_0 -ctv q8_0"]
"""
import argparse, json, os, signal, subprocess, sys, threading, time, urllib.request

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.expanduser("~/LLM/Ternary-Bonsai-2-27B-MTP")
PORT = 18099
BASE = f"http://127.0.0.1:{PORT}"

def swapouts():
    for line in subprocess.check_output(["vm_stat"], text=True).splitlines():
        if "Swapouts" in line:
            return int(line.split()[-1].rstrip("."))

def http(path, body=None, timeout=1800):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--tokens", type=int, default=16000)
    ap.add_argument("--server-args", default="")
    ap.add_argument("--diverge", type=int, default=0,
                    help="続きの要求で、A の回答の末尾この文字数を別の文に替える (履歴の描き直しが保存した状態の手前で食い違う場合)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slots = os.path.join(a.out, "slots"); os.makedirs(slots, exist_ok=True)
    man = json.load(open(os.path.join(MODEL_DIR, "manifest.json")))
    cmd = [os.path.expanduser(man["llama_server"]), "-m", os.path.join(MODEL_DIR, man["gguf"]), *man["server_args"],
           "-c", "32768", "--host", "127.0.0.1", "--port", str(PORT), "--slot-save-path", slots, *a.server_args.split()]
    log = open(os.path.join(a.out, "llama-server.log"), "w")
    srv = subprocess.Popen(["caffeinate", "-i", *cmd], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    watch = open(os.path.join(a.out, "watch.log"), "w")
    s0 = swapouts(); stop = threading.Event(); phase = ["start"]

    def kill():
        try: os.killpg(srv.pid, signal.SIGTERM)
        except ProcessLookupError: pass

    def watcher():
        hist = []
        while not stop.wait(5):
            now = swapouts(); hist.append(now); hist[:] = hist[-12:]
            rss = subprocess.run("ps -axo rss,comm | awk '/llama-server$/ {printf \"%.2f\", $1/1048576}'", shell=True,
                                 capture_output=True, text=True).stdout
            free = subprocess.run("memory_pressure | awk '/free percentage/ {print $NF}'", shell=True,
                                  capture_output=True, text=True).stdout.strip()
            watch.write(f"{time.strftime('%H:%M:%S')} swapouts +{now - s0} (60s +{now - hist[0]}) rss {rss} GB free {free} {phase[0]}\n"); watch.flush()
            if now - hist[0] > 32768 or now - s0 > 65536:
                watch.write("SWAP LIMIT: stopping\n"); watch.flush(); kill(); os._exit(2)
    threading.Thread(target=watcher, daemon=True).start()

    try:
        for _ in range(600):
            try:
                if http("/health", timeout=2).get("status") == "ok": break
            except Exception: time.sleep(1)
            if srv.poll() is not None: sys.exit("server exited during load")
        else: sys.exit("server did not become healthy")

        def text_of(paths, target):
            t = "\n\n".join(open(os.path.join(R, p)).read() for p in paths)
            n = len(http("/tokenize", {"content": t})["tokens"])
            return t[: int(len(t) * min(1.0, target / n))]
        doc_a = text_of(["docs/qwen38-27b/05-BONSAI-PQ2-0.md", "docs/qwen38-27b/09-ONLINE-FULL-RUN.md", "docs/qwen38-27b/08-PP-TG-CEILING.md",
                         "docs/qwen38-27b/07-PQ2-0-SMALL-BATCH-KERNEL.md", "docs/qwen38-27b/06-BONSAI-MTP-ON-MAC.md"], a.tokens)
        doc_b = text_of(["docs/SYSTEM_DESIGN.md", "docs/CLI.md", "docs/OPENAI_SERVER.md", "docs/RUNTIME_CONTROLS.md"], a.tokens)
        conv_a = [{"role": "user", "content": "次の技術メモを読んで、要点を 5 つにまとめてください。\n\n" + doc_a}]
        conv_b = [{"role": "user", "content": "Read the following documentation and summarize the five most important points.\n\n" + doc_b}]
        results = []

        def chat(name, messages, max_tokens):
            phase[0] = name; t = time.time()
            r = http("/v1/chat/completions", {
                "messages": messages, "max_tokens": max_tokens, "cache_prompt": True,
                "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                "chat_template_kwargs": {"enable_thinking": False}, "logprobs": True, "top_logprobs": 10})
            tm = r.get("timings", {}); lp = r["choices"][0].get("logprobs") or {}
            first = (lp.get("content") or [{}])[0]
            row = {"name": name, "wall_s": round(time.time() - t, 2), "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
                   "prompt_ms": tm.get("prompt_ms"), "predicted_n": tm.get("predicted_n"), "predicted_per_second": tm.get("predicted_per_second"),
                   "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted"),
                   "first_top": [(x.get("token"), round(x.get("logprob"), 4)) for x in first.get("top_logprobs", [])],
                   "text": r["choices"][0]["message"].get("content", "")}
            results.append(row); print(json.dumps({k: v for k, v in row.items() if k != "text"}, ensure_ascii=False), flush=True)
            return row

        def slot(action, filename):
            phase[0] = f"{action} {filename}"; t = time.time()
            r = http(f"/slots/0?action={action}", {"filename": filename})
            row = {"name": f"{action}:{filename}", "wall_s": round(time.time() - t, 2), "response": r}
            p = os.path.join(slots, filename)
            if os.path.exists(p): row["file_bytes"] = os.path.getsize(p)
            results.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)

        follow = {"role": "user", "content": "3 つ目の要点を、数字を挙げてもう少し詳しく説明してください。"}
        a1 = chat("A1", conv_a, 200)
        slot("save", "a.bin")
        reply = a1["text"] if not a.diverge else a1["text"][:-a.diverge] + "(以下略)"
        cont = conv_a + [{"role": "assistant", "content": reply}, follow]
        chat("A2-live", cont, 120)            # cache が生きている基準
        chat("B1", conv_b, 60)                # 切り替え
        slot("restore", "a.bin")
        chat("A2-restored", cont, 120)        # 復元した後の続き
        chat("B2", conv_b, 60)                # もう一度切り替え
        chat("A2-reread", cont, 120)          # 復元なし (読み直し)
        json.dump(results, open(os.path.join(a.out, "results.json"), "w"), ensure_ascii=False, indent=1)
    finally:
        stop.set(); kill()
        try: srv.wait(30)
        except subprocess.TimeoutExpired: os.killpg(srv.pid, signal.SIGKILL)
        watch.write(f"swapouts total +{swapouts() - s0}\n"); watch.close()

if __name__ == "__main__":
    main()
