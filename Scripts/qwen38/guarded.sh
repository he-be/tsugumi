#!/bin/zsh
# guarded.sh <out file> <cmd...>: run cmd, kill it and every descendant if Swapouts grow by > 4096 pages (64 MB).
# The whole tree: killing only the direct child left `/usr/bin/time -l env ... bench` running (docs/qwen38/06 §5).
out=$1; shift
sw() { vm_stat | awk '/Swapouts/ {gsub(/\./,"",$2); print $2}'; }
killtree() { local c; for c in $(pgrep -P $1); do killtree $c; done; kill $1 2>/dev/null; }
s0=$(sw)
"$@" > $out 2>&1 &
pid=$!
peakd=0
while kill -0 $pid 2>/dev/null; do
  sleep 2
  d=$(( $(sw) - s0 )); (( d > peakd )) && peakd=$d
  if (( d > ${GUARD_PAGES:-4096} )); then echo "GUARD: Swapouts +$d pages, killing $pid and descendants" >> $out; killtree $pid; break; fi
done
wait $pid; rc=$?
sleep 1
if pgrep -P $pid > /dev/null 2>&1; then echo "GUARD: descendants of $pid still alive" >> $out; fi
echo "GUARD: exit $rc, Swapouts delta peak $peakd pages" >> $out
