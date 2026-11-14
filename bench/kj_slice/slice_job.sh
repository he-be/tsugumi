#!/bin/bash
cd /root/diet; rm -f test64.gguf
Q4=/root/models/gemma-4-26b-a4b-qat-2/Huihui-gemma-4-26B-A4B-it-qat-abliterated-lattice-Q4_0.gguf
for k in keep_96 keep_80 keep_64 keeprand_96 keep_112; do
  [ -f lattice-$k.gguf ] || /root/xolotl/.venv/bin/python slice_experts.py $Q4 $k.json lattice-$k.gguf
done
ls -la lattice-*.gguf
./eval_job.sh full=$Q4 k112=/root/diet/lattice-keep_112.gguf k96=/root/diet/lattice-keep_96.gguf k80=/root/diet/lattice-keep_80.gguf k64=/root/diet/lattice-keep_64.gguf rand96=/root/diet/lattice-keeprand_96.gguf full2=$Q4
