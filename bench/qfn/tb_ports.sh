#!/bin/zsh
# 19 §2-3-1: does the TB4 link (MBP -> M6) slow down while the M6 reads its external SSD, and vice versa?
#   bench/qfn/tb_ports.sh <label> [rounds] [seconds]
# Per round and QD (4, 8): external alone, link alone, both at once. External = rec_probe (3,358,720-byte records,
# random, F_NOCACHE) over /Volumes/SSD256/layer_*.bin (21.5 GB > RAM); link = link_probe stream (1 GiB acked sends).
# Output: bench/qfn/results/<date>-pair-<label>/{hostinfo.*,tb.mbp.txt,tb.m6.txt,link.txt,command.txt,output.tsv,*.link.tsv}
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; rounds=${2:-3}; secs=${3:-15}
out=$repo/bench/qfn/results/$(date +%F)-pair-$label
mkdir -p $out
cc -O2 -o $repo/scratch/link_probe $repo/bench/qfn/link_probe.c
scp -q $repo/bench/qfn/link_probe.c $repo/bench/rec_probe.c m6:/tmp/
ssh m6 'cc -O2 -o /tmp/link_probe /tmp/link_probe.c && clang -O2 -o /tmp/rec_probe /tmp/rec_probe.c -lpthread'
m6_ip=$(ssh m6 "ifconfig bridge0 | awk '/inet / {print \$2}'")
mbp_ip=$(ifconfig bridge0 | awk '/inet / {print $2}')
sh $repo/bench/hostinfo.sh > $out/hostinfo.mbp.txt
ssh m6 'sh dev/tsugumi/bench/hostinfo.sh; echo "iogpu.wired_limit_mb $(sysctl -n iogpu.wired_limit_mb)"' > $out/hostinfo.m6.txt
system_profiler SPThunderboltDataType > $out/tb.mbp.txt
ssh m6 'system_profiler SPThunderboltDataType; diskutil info disk5s1 | grep -E "Device Location|Protocol|Device / Media"' > $out/tb.m6.txt
{ echo "bridge0 MBP $mbp_ip -> M6 $m6_ip"; ping -c 5 -q $m6_ip | tail -1; } > $out/link.txt
ext=/Volumes/SSD256
{ echo "bench/qfn/tb_ports.sh $*"
  echo "external: ssh m6 /tmp/rec_probe $ext 3358720 <qd> $secs"
  echo "link: ssh m6 /tmp/link_probe serve $m6_ip 5401 ; scratch/link_probe stream $m6_ip 5401 $secs"; } > $out/command.txt
ssh m6 "vm_stat | grep -E 'Pageins|Swapouts'" > $out/m6.pre.txt
link() {  # $1 = file
  ssh m6 "/tmp/link_probe serve $m6_ip 5401" > /dev/null 2>&1 &
  local sp=$!
  sleep 1
  $repo/scratch/link_probe stream $m6_ip 5401 $secs > $1
  wait $sp
}
printf 'round\tqd\tarm\tdev\tMB\tqd2\tGBps\tcount\tIOPS\tp10\tp50\tmean\tp99\tp999\n' > $out/output.tsv
for r in $(seq 1 $rounds); do
  for qd in 4 8; do
    line=$(ssh m6 "/tmp/rec_probe $ext 3358720 $qd $secs")
    printf '%s\t%s\talone\texternal\t%s\n' $r $qd "$line" | tee -a $out/output.tsv
    link $out/r$r-q$qd-alone.link.tsv
    printf '%s\t%s\talone\tlink\t%s\n' $r $qd "$(awk '/^total/ {print $3}' $out/r$r-q$qd-alone.link.tsv)" | tee -a $out/output.tsv
    ssh m6 "/tmp/link_probe serve $m6_ip 5401" > /dev/null 2>&1 &
    sp=$!
    sleep 1
    ssh m6 "/tmp/rec_probe $ext 3358720 $qd $secs" > $out/.ext &
    pe=$!
    $repo/scratch/link_probe stream $m6_ip 5401 $secs > $out/r$r-q$qd-both.link.tsv
    wait $pe $sp
    printf '%s\t%s\tboth\texternal\t%s\n' $r $qd "$(cat $out/.ext)" | tee -a $out/output.tsv
    printf '%s\t%s\tboth\tlink\t%s\n' $r $qd "$(awk '/^total/ {print $3}' $out/r$r-q$qd-both.link.tsv)" | tee -a $out/output.tsv
    rm -f $out/.ext
  done
done
ssh m6 "vm_stat | grep -E 'Pageins|Swapouts'" > $out/m6.post.txt
echo "evidence: ${out#$repo/}"
