#!/usr/bin/env python3
"""convert_hf_to_gguf.py を、PLE の表 (embed_tokens_per_layer、bf16 で 5.6 GB) を除いて走らせる。
表は convert の中で丸ごと展開されて 18 GB の Mac がスワップする (2026-09-19) ので、lattice_q4_0.py が原本から
群ごとに直接 Q4_0 にする (--ple)。引数は convert_hf_to_gguf.py と同じ。"""
import sys
sys.path.insert(0, "/Users/mh/LLM/llama.cpp")
sys.path.insert(0, "/Users/mh/LLM/llama.cpp/gguf-py")
import conversion.gemma as gemma  # noqa: E402

original = gemma.Gemma4Model.filter_tensors.__func__


def filter_tensors(cls, item):
    if "embed_tokens_per_layer" in item[0]:
        return None
    return original(cls, item)


gemma.Gemma4Model.filter_tensors = classmethod(filter_tensors)
import runpy  # noqa: E402
sys.argv[0] = "/Users/mh/LLM/llama.cpp/convert_hf_to_gguf.py"
runpy.run_path(sys.argv[0], run_name="__main__")
