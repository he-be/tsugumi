#!/usr/bin/env python3
"""Ground truth for the speedometer: what "RAM crowded by other apps" does to
a round, measured where it happens.

Per run: the machine before (vm_stat split), the decode service's own SSD
traffic during the run (`proc_pid_rusage`: bytes read from disk and page-ins,
sampled before, after prefill, and at the end), and the round's cost (tok/s,
prefill seconds, draft stats). Cells: a memory hog of 0 / 6 / 11 GiB standing
in for other apps, three tasks (short chat, eight topics, a ~3K-token document),
three reps each. Official sampling, thinking off, ctx 32K, 32 slots.

**Swap guard.** The hog refuses a size this Mac cannot lend without swapping
and kills itself the moment Swapouts grows (see `memory_hog.py`); this driver
checks the hog is alive before every run, checks Swapouts after every run,
and aborts the whole experiment on either. On the 18 GB M3 Pro the 6, 8 and
11 GiB cells of 2026-09-02/04 swapped the hog itself (swapins 13〜40 万ページ
per run) and wrote on the order of a terabyte to the SSD; they are not cells
any more. With the model loaded (~9 GB used) the line is about 5 GiB.

    OUT_DIR=bench/logs/gauge python3 bench/gauge_eval.py
    CELLS=0,11 TASKS=short,long REPS=2 python3 bench/gauge_eval.py
"""
import ctypes, ctypes.util, json, os, pathlib, re, signal, socket, subprocess, sys, time, uuid
REPO = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{REPO}/scripts/app")
import smoke_decode as sd
OUT = pathlib.Path(os.environ.get("OUT_DIR", "."))
OUT.mkdir(parents=True, exist_ok=True)
MODEL = sd.installed_model("gemma4-qat-sym")
CTX = int(os.environ.get("CTX", "32768"))
COOL = int(os.environ.get("COOL_S", "10"))
CELLS = [float(x) for x in os.environ.get("CELLS", "0,3,5").split(",")]
SWAP_ABORT_PAGES = int(os.environ.get("SWAP_ABORT_PAGES", "4096"))
REPS = int(os.environ.get("REPS", "3"))
TASKS = os.environ.get("TASKS", "short,broad,long").split(",")
PAGE = os.sysconf("SC_PAGESIZE")
PHYS = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
WANTED = sum(f.stat().st_size for f in pathlib.Path(MODEL, "packed_experts").iterdir())

LONG_DOC = open(f"{REPO}/docs/MAC_APP.md", encoding="utf-8").read()[:6000]
PROMPTS = {
    "short": ("日本の四季それぞれの特徴を、季節ごとに 3 文ずつ説明してください。", 256),
    "broad": ("次の 8 つの話題について、それぞれ 3 文ずつ書いてください: 量子コンピュータ、"
              "フランス革命、味噌の作り方、Rust の所有権、ジャズの歴史、腎臓の働き、"
              "囲碁の定石、南極の気候。", 512),
    "long": (LONG_DOC + "\n\n上の文書を 5 文で要約してください。", 64),
}

# --- the decode service's own disk traffic -------------------------------
_libproc = ctypes.CDLL(ctypes.util.find_library("proc"))
class _RusageV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups ri_pageins "
        "ri_wired_size ri_resident_size ri_phys_footprint ri_proc_start_abstime "
        "ri_proc_exit_abstime ri_child_user_time ri_child_system_time ri_child_pkg_idle_wkups "
        "ri_child_interrupt_wkups ri_child_pageins ri_child_elapsed_abstime ri_diskio_bytesread "
        "ri_diskio_byteswritten ri_cpu_time_qos_default ri_cpu_time_qos_maintenance "
        "ri_cpu_time_qos_background ri_cpu_time_qos_utility ri_cpu_time_qos_legacy "
        "ri_cpu_time_qos_user_initiated ri_cpu_time_qos_user_interactive ri_billed_system_time "
        "ri_serviced_system_time ri_logical_writes ri_lifetime_max_phys_footprint ri_instructions "
        "ri_cycles ri_billed_energy ri_serviced_energy ri_interval_max_phys_footprint "
        "ri_runnable_time").split()]

def rusage(pid):
    buf = _RusageV4()
    assert _libproc.proc_pid_rusage(pid, 4, ctypes.byref(buf)) == 0
    return dict(disk_read=buf.ri_diskio_bytesread, pageins=buf.ri_pageins,
                footprint=buf.ri_phys_footprint, t=time.time())

def vm():
    out = subprocess.check_output(["vm_stat"], text=True)
    def g(label):
        return int(re.search(label + r":\s+(\d+)", out).group(1)) * PAGE
    anon, purg = g("Anonymous pages"), g("Pages purgeable")
    wired, comp, filebacked = g("Pages wired down"), g("Pages occupied by compressor"), g("File-backed pages")
    used = (anon - purg) + wired + comp
    return dict(app_gb=(anon - purg) / 2**30, wired_gb=wired / 2**30, comp_gb=comp / 2**30,
                cached_gb=(filebacked + purg) / 2**30, borrow_gb=max(0, PHYS - used) / 2**30,
                level=min(1.0, max(0, PHYS - used) / WANTED),
                sys_pageins=g("Pageins") // PAGE, sys_swapins=g("Swapins") // PAGE,
                sys_swapouts=g("Swapouts") // PAGE)

def generate(sock, pid, prompt, max_new):
    gid = str(uuid.uuid4()).upper()
    sd.send(sock, {"generate": {"_0": {
        "prompt": prompt, "maxNewTokens": max_new, "maxContextTokens": CTX,
        "temperature": 1.0, "topK": 64, "topP": 0.95, "repetitionPenalty": 1,
        "enableThinking": False, "imagePaths": [], "runtimeOptions": sd.runtime_options(),
        "generationID": gid}}})
    after_prefill = None
    while True:
        ev = sd.recv(sock)
        if ev.get("generationID") != gid or ev["kind"] == "prefill":
            continue
        if ev["kind"] == "snapshot":
            if after_prefill is None:
                after_prefill = rusage(pid)
            continue
        return ev, after_prefill or rusage(pid)

def main():
    log = open(OUT / "gauge_eval.jsonl", "a")
    proc = subprocess.Popen([sd.SERVICE, "--socket", sd.SOCKET_PATH],
                            stdout=open(OUT / "service.log", "a"), stderr=subprocess.STDOUT)
    hog = None
    try:
        sock = None
        for _ in range(200):
            try:
                sock = socket.socket(socket.AF_UNIX); sock.connect(sd.SOCKET_PATH); break
            except OSError:
                sock.close(); sock = None; time.sleep(0.05)
        assert sock and sd.load(sock, MODEL, CTX, True)
        ev, _ = generate(sock, proc.pid, PROMPTS["short"][0], 128)
        print(f"[warmup] tok/s={ev['tokensPerSecond']:.2f}", flush=True)
        time.sleep(COOL)
        counter = 0
        for cell in CELLS:
            if cell > 0:
                hog = subprocess.Popen([sys.executable, f"{REPO}/bench/memory_hog.py", str(cell)],
                                       stdout=subprocess.PIPE, text=True)
                first = hog.stdout.readline().strip()
                if first != "ready":
                    print(f"[hog {cell} GiB] {first} — skipping this cell", flush=True)
                    hog.wait(); hog = None
                    continue
                time.sleep(5)
            for task in TASKS:
                prompt, max_new = PROMPTS[task]
                for rep in range(REPS):
                    counter += 1
                    if hog is not None and hog.poll() is not None:
                        raise SystemExit(f"hog exited ({hog.stdout.read().strip()}): macOS was swapping; experiment aborted")
                    # A fresh prefix defeats the prompt cache, so every rep prefills.
                    before_vm, before = vm(), rusage(proc.pid)
                    ev, mid = generate(sock, proc.pid, f"({counter} 回目の依頼)\n" + prompt, max_new)
                    after = rusage(proc.pid)
                    after_vm = vm()
                    tokens = ev["tokenCount"]
                    decode_s = ev["decodeSeconds"]
                    row = dict(
                        task=task, hog_gib=cell, rep=rep, ts=time.time(),
                        tok_s=ev["tokensPerSecond"], tokens=tokens, decode_s=decode_s,
                        prefill_s=ev.get("prefillSeconds"), prompt_tokens=ev.get("promptTokenCount"),
                        cached_tokens=ev.get("cachedPromptTokens"), ttft_s=ev.get("timeToFirstTokenSeconds"),
                        draft=f"{ev.get('draftAccepted')}/{ev.get('draftProposed')}", stop=ev.get("stopReason"),
                        svc_footprint_gb=after["footprint"] / 2**30,
                        prefill_disk_mb=(mid["disk_read"] - before["disk_read"]) / 2**20,
                        prefill_pageins=mid["pageins"] - before["pageins"],
                        decode_disk_mb=(after["disk_read"] - mid["disk_read"]) / 2**20,
                        decode_pageins=after["pageins"] - mid["pageins"],
                        decode_disk_mb_per_tok=(after["disk_read"] - mid["disk_read"]) / 2**20 / max(1, tokens),
                        decode_disk_mb_s=(after["disk_read"] - mid["disk_read"]) / 2**20 / max(1e-3, decode_s),
                        sys_pageins_delta=after_vm["sys_pageins"] - before_vm["sys_pageins"],
                        sys_swapins_delta=after_vm["sys_swapins"] - before_vm["sys_swapins"],
                        sys_swapouts_delta=after_vm["sys_swapouts"] - before_vm["sys_swapouts"],
                        **{f"before_{k}": v for k, v in before_vm.items() if not k.startswith("sys_")})
                    log.write(json.dumps(row) + "\n"); log.flush()
                    swapped = after_vm.get("sys_swapouts", 0) - before_vm.get("sys_swapouts", 0)
                    if swapped > SWAP_ABORT_PAGES:
                        raise SystemExit(f"Swapouts +{swapped} pages during one run; experiment aborted")
                    print(f"hog={cell:>4} {task:5} rep={rep} level={row['before_level']:.2f} "
                          f"borrow={row['before_borrow_gb']:.1f}G wired={row['before_wired_gb']:.1f}G "
                          f"tok/s={row['tok_s']:.1f} pf={row['prefill_s']:.1f}s prompt={row['prompt_tokens']} "
                          f"decode_disk={row['decode_disk_mb']:.0f}MB ({row['decode_disk_mb_per_tok']:.2f}MB/tok, "
                          f"{row['decode_disk_mb_s']:.0f}MB/s) prefill_disk={row['prefill_disk_mb']:.0f}MB "
                          f"draft={row['draft']} swapins={row['sys_swapins_delta']}", flush=True)
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
