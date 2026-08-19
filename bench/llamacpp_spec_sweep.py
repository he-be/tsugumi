#!/usr/bin/env python3
"""llama.cpp (参照実装) の投機デコードを振るドライバ。docs/mtp/35 の測定に使った。

1 構成 = 1 プロセス (llama.cpp は投機パラメータを per-request で受け付けない)。
サンプリングは sample_llamacpp_cmd.txt と同じ temp 1.0 / top-k 64 / min-p 0。
リクエスト間に COOL_S (既定 20 秒) のクールダウンを入れる — 筐体を温めないため。

  python3 bench/llamacpp_spec_sweep.py bench/logs/llamacpp-2026-08-19/cfg-temp1.json out.json
"""
import json, subprocess, time, urllib.request, urllib.error, base64, sys, os

HOME = os.path.expanduser("~")
BIN  = f"{HOME}/LLM/llama.cpp/build/bin/llama-server"
MDIR = f"{HOME}/LLM/gemma-4-26B-A4B"
PROJ = f"{HOME}/LLM/turbo-fieldfare"
SCR  = os.path.dirname(os.path.abspath(__file__))
PORT = 8080
URL  = f"http://127.0.0.1:{PORT}"

BASE = [BIN,
  "-m", f"{MDIR}/gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf",
  "--mmproj", f"{MDIR}/mmproj-BF16.gguf",
  "--port", str(PORT), "--host", "127.0.0.1",
  "-ngl", "99", "--no-mmap", "-fa", "on",
  "-b", "1024", "-ub", "1024", "-np", "1",
  "-ctk", "q8_0", "-ctv", "q8_0", "-c", "16384",
  "--reasoning", "off", "-cram", "0", "-t", "6", "--no-webui"]

MTP = ["-md", f"{MDIR}/mtp-gemma-4-26B-A4B-it.gguf", "--spec-type", "draft-mtp"]

def load_msgs(p):
    return json.load(open(f"{PROJ}/bench/{p}"))

def img_msgs():
    b = base64.b64encode(open(f"{PROJ}/sample_imgs/IMG_2113.JPG","rb").read()).decode()
    sysu = load_msgs("mtp_goal_prompt.json")
    return [sysu[0],
            {"role":"user","content":[
                {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b}},
                {"type":"text","text":sysu[1]["content"]}]}]

PROMPTS = {}
def prompts():
    if not PROMPTS:
        PROMPTS["goal"]  = [{"role":"user","content":"Swift で書かれた推論サーバに、プロンプトキャッシュ（前方一致で KV を再利用する仕組み）を入れる計画を、設計・実装・テストの順に具体的に書いてください。"}]
        PROMPTS["story"] = load_msgs("story.json")
        PROMPTS["math"]  = load_msgs("math.json")
        PROMPTS["image"] = img_msgs()
    return PROMPTS

def post(path, payload, timeout=600):
    req = urllib.request.Request(URL+path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def wait_ready(proc, timeout=600):
    t0 = time.time()
    while time.time()-t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server died")
        try:
            with urllib.request.urlopen(URL+"/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return time.time()-t0
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("timeout waiting for server")

COOL = int(os.environ.get("COOL_S", "20"))

def cool(why=""):
    print(f"# cooldown {COOL}s {why}", flush=True)
    time.sleep(COOL)

def run_cfg(name, extra, tasks, npred=192, temp=0.0, reps=2):
    logf = open(f"{SCR}/srv_{name}.log","w")
    proc = subprocess.Popen(BASE+extra, stdout=logf, stderr=subprocess.STDOUT)
    out = {"name":name, "extra":extra, "runs":[]}
    try:
        out["load_s"] = round(wait_ready(proc),2)
        for t in tasks:
            msgs = prompts()[t]
            for i in range(reps):
                body = {"messages":msgs, "max_tokens":npred, "temperature":temp,
                        "cache_prompt":True, "seed":1234}
                if temp > 0:
                    body.update({"top_k":64,"min_p":0.0})
                else:
                    body["ignore_eos"] = True
                t0=time.time()
                r = post("/v1/chat/completions", body)
                wall = time.time()-t0
                tm = r.get("timings",{})
                rec = {"task":t,"rep":i,"wall":round(wall,3),
                       "prompt_n":tm.get("prompt_n"),"prompt_ms":tm.get("prompt_ms"),
                       "predicted_n":tm.get("predicted_n"),"predicted_ms":tm.get("predicted_ms"),
                       "tg_tps":tm.get("predicted_per_second"),
                       "draft_n":tm.get("draft_n"),"draft_acc":tm.get("draft_n_accepted"),
                       "content_head": (r["choices"][0]["message"].get("content") or "")[:60]}
                out["runs"].append(rec)
                print(json.dumps(rec, ensure_ascii=False), flush=True)
                cool(f"{name}/{t}/{i}")
    finally:
        proc.terminate()
        try: proc.wait(30)
        except Exception: proc.kill()
        logf.close()
    return out

if __name__ == "__main__":
    cfgs = json.load(open(sys.argv[1]))
    res = []
    outp = sys.argv[2]
    for c in cfgs:
        print("=== CFG", c["name"], flush=True)
        try:
            res.append(run_cfg(c["name"], c["extra"], c.get("tasks",["goal","story","math"]),
                               npred=c.get("npred",192), temp=c.get("temp",0.0), reps=c.get("reps",2)))
        except Exception as e:
            print("FAILED", c["name"], e, flush=True)
            res.append({"name":c["name"],"error":str(e)})
        json.dump(res, open(outp,"w"), ensure_ascii=False, indent=1)
        time.sleep(COOL)
