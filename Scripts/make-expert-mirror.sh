#!/bin/sh
# Put a copy of a model's packed_experts/ on a second SSD and register it, so
# the runtime reads part of every layer from each drive at once
# (docs/m6-prefill/44).
#   Scripts/make-expert-mirror.sh <model dir> <mirror dir> [share]
#   Scripts/make-expert-mirror.sh scratch/gemma4-qat-sym.gturbo /Volumes/SSD256/tsugumi-mirror/packed_experts
# share (default 0.32) is the fraction of each layer's experts read from the
# copy; 0.32 fits M6's internal 3.3 GB/s next to a TB4 drive's 1.57 GB/s.
# Every copied file is compared byte for byte with the original before
# expert-mirror.json is written into the model directory. Remove that file to
# stop using the mirror; TF_EXPERT_MIRROR=0 skips it for one run.
set -eu
model=${1:?model dir}; mirror=${2:?mirror dir}; share=${3:-0.32}
model=$(cd "$model" && pwd)
[ -f "$model/manifest.json" ] && [ -d "$model/packed_experts" ] || { echo "not a model directory: $model"; exit 1; }
case "$mirror" in /*) ;; *) echo "mirror dir must be an absolute path"; exit 1 ;; esac
awk -v s="$share" 'BEGIN { exit !(s > 0 && s <= 1) }' || { echo "share must be in (0, 1]"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI' >/dev/null; then
    echo "a model process is running; stop it before changing its mirror"; exit 1
fi
rm -f "$model/expert-mirror.json"
mkdir -p "$mirror"
for f in "$model"/packed_experts/layer_*.bin; do
    b=$(basename "$f")
    if [ -f "$mirror/$b" ] && cmp -s "$f" "$mirror/$b"; then
        echo "$b: already equal"
        continue
    fi
    cp "$f" "$mirror/$b.partial"
    cmp -s "$f" "$mirror/$b.partial" || { echo "$b: copy differs from the original"; exit 1; }
    mv "$mirror/$b.partial" "$mirror/$b"
    echo "$b: copied and compared"
done
sha=$(shasum -a 256 "$model/manifest.json" | cut -d' ' -f1)
printf '{"directory":"%s","share":%s,"manifestSha256":"%s"}\n' "$mirror" "$share" "$sha" \
    > "$model/expert-mirror.json"
echo "wrote $model/expert-mirror.json"
