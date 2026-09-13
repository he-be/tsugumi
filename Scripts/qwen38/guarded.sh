#!/bin/zsh
# guarded.sh <out file> <cmd...>: run cmd, kill it if Swapouts grow by > 4096 pages (64 MB), log free/speculative pages.
out=$1; shift
sw() { vm_stat | awk '/Swapouts/ {gsub(/\./,"",$2); print $2}'; }
s0=$(sw)
"$@" > $out 2>&1 &
pid=$!
peakd=0
while kill -0 $pid 2>/dev/null; do
  sleep 2
  d=$(( $(sw) - s0 )); (( d > peakd )) && peakd=$d
  if (( d > 4096 )); then echo "GUARD: Swapouts +$d pages, killing $pid" >> $out; kill $pid; break; fi
done
wait $pid; rc=$?
echo "GUARD: exit $rc, Swapouts delta peak $peakd pages" >> $out
