#!/bin/zsh
# guarded.sh <out file> <cmd...>: run cmd, kill it and every descendant when swap writes are large, not when they are brief.
#
# Stops when Swapouts grow by more than GUARD_WINDOW_PAGES (32768 = 512 MiB) within the last GUARD_WINDOW_S (60) seconds,
# or by more than GUARD_PAGES (65536 = 1 GiB) over the whole run. A round of the tool loop can write 5-8K pages in
# 4-10 s when wired memory swings between prefill and decode and then stay quiet (docs/qwen38/21 §4 E-2); the old
# single 4096-page limit stopped those runs. Thrashing (a hog: 130-400K pages per round) and a steady trickle
# (the 1.6 TB write) still stop, the first within a minute. GUARD_PAGES=-1 fires on the first sample (test).
#
# GUARD_LOG=<file> writes a 2 s sample: epoch, Swapouts / Swapins since start, wired / file-backed / compressor GB.
#
# The whole tree: killing only the direct child left `/usr/bin/time -l env ... bench` running (docs/qwen38/06 §5).
out=$1; shift
total_limit=${GUARD_PAGES:-65536}
window_limit=${GUARD_WINDOW_PAGES:-32768}
window_samples=$(( ${GUARD_WINDOW_S:-60} / 2 ))
counters() { vm_stat | awk '/Swapouts/ {gsub(/\./,"",$2); o=$2} /Swapins/ {gsub(/\./,"",$2); i=$2} END {print o, i}'; }
killtree() { local c; for c in $(pgrep -P $1); do killtree $c; done; kill $1 2>/dev/null; }
read s0 i0 <<< "$(counters)"
[[ -n $GUARD_LOG ]] && echo "epoch swapouts swapins wiredGB fileGB comprGB" > $GUARD_LOG
"$@" > $out 2>&1 &
pid=$!
peakd=0; peakw=0
samples=(0)
while kill -0 $pid 2>/dev/null; do
  sleep 2
  if [[ -n $GUARD_LOG ]]; then
    vm_stat | awk -v s0=$s0 -v i0=$i0 -v e=$(date +%s) '/wired down/ {gsub(/\./,"",$4); w=$4} /File-backed/ {gsub(/\./,"",$3); fb=$3} /Swapouts/ {gsub(/\./,"",$2); o=$2} /Swapins/ {gsub(/\./,"",$2); i=$2} /occupied by compressor/ {gsub(/\./,"",$5); c=$5} END {printf "%d %d %d %.2f %.2f %.2f\n", e, o-s0, i-i0, w*16384/1e9, fb*16384/1e9, c*16384/1e9}' >> $GUARD_LOG
    d=$(tail -1 $GUARD_LOG | awk '{print $2}')
  else
    read o i <<< "$(counters)"; d=$(( o - s0 ))
  fi
  samples+=($d)
  (( ${#samples} > window_samples + 1 )) && samples=(${samples[2,-1]})
  w=$(( d - samples[1] ))
  (( d > peakd )) && peakd=$d
  (( w > peakw )) && peakw=$w
  if (( d > total_limit )); then
    echo "GUARD: Swapouts +$d pages since start (> $total_limit), killing $pid and descendants" >> $out; killtree $pid; break
  fi
  if (( w > window_limit )); then
    echo "GUARD: Swapouts +$w pages in ${GUARD_WINDOW_S:-60} s (> $window_limit), killing $pid and descendants" >> $out; killtree $pid; break
  fi
done
wait $pid; rc=$?
sleep 1
if pgrep -P $pid > /dev/null 2>&1; then echo "GUARD: descendants of $pid still alive"  >> $out; fi
read o i <<< "$(counters)"
echo "GUARD: exit $rc, Swapouts delta peak $peakd pages, window peak $peakw pages, Swapins +$(( i - i0 )) pages" >> $out
