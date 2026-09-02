#!/usr/bin/env python3
"""Same task, official sampling, under a memory hog of 0/4/8/11 GiB.

Records the gauge (borrowable / streamed weights, the app's formula from
vm_stat) next to tok/s of each run. Cells: 0, 4, 8, 11, then 0 again as
the head/tail drift check."""
import json, os, socket, subprocess, sys, time, uuid, re, signal, pathlib
REPO = os.environ.get("TSUGUMI_REPO") or str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{REPO}/scripts/app")
import smoke_decode as sd
S = pathlib.Path(os.environ.get("OUT_DIR", "."))
MODEL = sd.installed_model("gemma4-qat-sym")
CTX = 32768
PROMPT = os.environ.get("PROMPT", "日本の四季それぞれの特徴を、季節ごとに 3 文ずつ説明してください。")
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))
TAG = os.environ.get("TAG", "seasons")
if os.environ.get("PROMPT_FILE"):
    PROMPT = open(os.environ["PROMPT_FILE"]).read()
PREFIX_REP = os.environ.get("PREFIX_REP") == "1"
REP_COUNTER = [0]
COOL = int(os.environ.get("COOL_S", "10"))
CELLS = [float(x) for x in os.environ.get("CELLS", "0,4,8,11,0").split(",")]
REPS = int(os.environ.get("REPS", "3"))
PAGE = os.sysconf("SC_PAGESIZE")
PHYS = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
WANTED = sum(f.stat().st_size for f in pathlib.Path(MODEL, "packed_experts").iterdir())

def vm():
    out = subprocess.check_output(["vm_stat"], text=True)
    def g(label):
        m = re.search(label + r":\s+(\d+)", out)
        return int(m.group(1)) * PAGE
    anon, purg = g("Anonymous pages"), g("Pages purgeable")
    wired, comp, filebacked = g("Pages wired down"), g("Pages occupied by compressor"), g("File-backed pages")
    used = (anon - purg) + wired + comp
    borrow = max(0, PHYS - used)
    return dict(app_gb=(anon - purg) / 2**30, wired_gb=wired / 2**30, comp_gb=comp / 2**30,
                cached_gb=(filebacked + purg) / 2**30, borrow_gb=borrow / 2**30,
                level=min(1.0, borrow / WANTED))

def generate(sock):
    gid = str(uuid.uuid4()).upper()
    REP_COUNTER[0] += 1
    prompt = (f"({REP_COUNTER[0]} 回目の依頼)\n" + PROMPT) if PREFIX_REP else PROMPT
    sd.send(sock, {"generate": {"_0": {
        "prompt": prompt, "maxNewTokens": MAX_NEW, "maxContextTokens": CTX,
        "temperature": 1.0, "topK": 64, "topP": 0.95, "repetitionPenalty": 1,
        "enableThinking": False, "imagePaths": [], "runtimeOptions": sd.runtime_options(),
        "generationID": gid}}})
    while True:
        ev = sd.recv(sock)
        if ev.get("generationID") != gid or ev["kind"] in ("prefill", "snapshot"):
            continue
        return ev

def main():
    log = open(S / "eval_headroom.jsonl", "a")
    proc = subprocess.Popen([sd.SERVICE, "--socket", sd.SOCKET_PATH],
                            stdout=open(S / "service.log", "a"), stderr=subprocess.STDOUT)
    hog = None
    try:
        sock = None
        for _ in range(200):
            try:
                sock = socket.socket(socket.AF_UNIX); sock.connect(sd.SOCKET_PATH); break
            except OSError:
                sock.close(); sock = None; time.sleep(0.05)
        assert sock and sd.load(sock, MODEL, CTX, True)
        ev = generate(sock)
        print(f"[warmup] tok/s={ev['tokensPerSecond']:.2f}", flush=True)
        time.sleep(COOL)
        for cell in CELLS:
            if cell > 0:
                hog = subprocess.Popen([sys.executable, str(pathlib.Path(__file__).resolve().parent / "memory_hog.py"), str(cell)],
                                       stdout=subprocess.PIPE, text=True)
                assert hog.stdout.readline().strip() == "ready"
                time.sleep(5)
            for rep in range(REPS):
                before = vm()
                ev = generate(sock)
                runner = ev.get("runner") or {}
                row = dict(tag=TAG, hog_gib=cell, rep=rep, ts=time.time(), io_ms_tok=runner.get("ioMillisecondsPerToken"), prefill_s=ev.get("prefillSeconds"), prompt_tokens=ev.get("promptTokenCount"), cached_tokens=ev.get("cachedPromptTokens"), cb1_ms_tok=runner.get("cb1MillisecondsPerToken"), rdadvise_mb_tok=runner.get("rdadviseMegabytesPerToken"),
                           tok_s=ev["tokensPerSecond"], tokens=ev["tokenCount"],
                           decode_s=ev["decodeSeconds"], ttft_s=ev.get("timeToFirstTokenSeconds"),
                           draft=f"{ev.get('draftAccepted')}/{ev.get('draftProposed')}",
                           stop=ev.get("stopReason"), err=ev.get("error"),
                           svc_mem_gb=(ev.get("currentMemoryBytes") or 0) / 2**30, **before)
                log.write(json.dumps(row) + "\n"); log.flush()
                print(f"hog={cell:>4} rep={rep} level={before['level']:.2f} "
                      f"borrow={before['borrow_gb']:.1f}G cached={before['cached_gb']:.1f}G "
                      f"tok/s={row['tok_s']:.2f} tokens={row['tokens']} draft={row['draft']} "
                      f"stop={row['stop']} io={row['io_ms_tok']} cb1={row['cb1_ms_tok']} prefill={row['prefill_s']} prompt={row['prompt_tokens']} cached={row['cached_tokens']} ttft={row['ttft_s']}", flush=True)
                time.sleep(COOL)
            if hog:
                hog.send_signal(signal.SIGTERM); hog.wait(); hog = None
                time.sleep(5)
    finally:
        if hog: hog.kill()
        proc.terminate()
        try: proc.wait(10)
        except Exception: proc.kill()
        try: os.unlink(sd.SOCKET_PATH)
        except OSError: pass

main()
