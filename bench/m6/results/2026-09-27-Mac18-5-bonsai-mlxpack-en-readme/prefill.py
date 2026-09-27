import sys, time, json
import mlx.core as mx
pack, prompt_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, pack + "/runtime")
from vision_artifact import load_vl_model
from mlx_vlm import generate
t0 = time.perf_counter()
model, processor, config = load_vl_model(pack)
load_s = time.perf_counter() - t0
prompt = open(prompt_path).read()
mx.reset_peak_memory()
r = generate(model, processor, prompt, max_tokens=1, temperature=0.0, verbose=False)
print(json.dumps({"prompt_tok": r.prompt_tokens, "prompt_tps": round(r.prompt_tps, 1),
                  "prefill_s": round(r.prompt_tokens / r.prompt_tps, 4) if r.prompt_tps else None,
                  "load_s": round(load_s, 3), "peak_gb": round(r.peak_memory, 3), "cached_tokens": r.cached_tokens,
                  "text": r.text}))
