"""Offsets and sizes of the routed expert tensors of a GGUF, one "offset bytes name" line each (for readbench.c).

    PYTHONPATH=~/LLM/llama.cpp/gguf-py ~/LLM/venv/bin/python Scripts/qwen38/expert_ranges.py <gguf> > scratch/qwen38/expert-ranges.txt
"""
import sys

from gguf import GGUFReader

rows = sorted((int(t.data_offset), int(t.n_bytes), t.name) for t in GGUFReader(sys.argv[1]).tensors if "_exps" in t.name)
for offset, n, name in rows:
    print(offset, n, name)
