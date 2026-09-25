#!/bin/zsh
# E0 (docs/qfn-prefill/07): raw TCP over bridge0 in both directions, n reps each, one fresh connection per direction.
#   bench/qfn/e0_link.sh <label> [reps]
# Output: bench/qfn/results/<date>-pair-<label>/{hostinfo.mbp.txt,hostinfo.m6.txt,link.txt,command.txt,mbp-to-m6.tsv,m6-to-mbp.tsv}
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; reps=${2:-5}
out=$repo/bench/qfn/results/$(date +%F)-pair-$label
mkdir -p $out
cc -O2 -o $repo/scratch/link_probe $repo/bench/qfn/link_probe.c
scp -q $repo/bench/qfn/link_probe.c m6:/tmp/link_probe.c
ssh m6 'cc -O2 -o /tmp/link_probe /tmp/link_probe.c'
mbp_ip=$(ifconfig bridge0 | awk '/inet / {print $2}')
m6_ip=$(ssh m6 "ifconfig bridge0 | awk '/inet / {print \$2}'")
sh $repo/bench/hostinfo.sh > $out/hostinfo.mbp.txt
ssh m6 'sh dev/tsugumi/bench/hostinfo.sh' > $out/hostinfo.m6.txt
{ echo "bridge0 MBP $mbp_ip -> M6 $m6_ip"; ping -c 20 -q $m6_ip | tail -1; } > $out/link.txt
echo "MBP->M6: ssh m6 /tmp/link_probe serve $m6_ip 5201 ; scratch/link_probe run $m6_ip 5201 $reps" > $out/command.txt
echo "M6->MBP: scratch/link_probe serve $mbp_ip 5202 ; ssh m6 /tmp/link_probe run $mbp_ip 5202 $reps" >> $out/command.txt
ssh m6 "/tmp/link_probe serve $m6_ip 5201" > $out/m6-server.log 2>&1 &
sleep 1
$repo/scratch/link_probe run $m6_ip 5201 $reps | tee $out/mbp-to-m6.tsv
wait
$repo/scratch/link_probe serve $mbp_ip 5202 > $out/mbp-server.log 2>&1 &
sleep 1
ssh m6 "/tmp/link_probe run $mbp_ip 5202 $reps" | tee $out/m6-to-mbp.tsv
wait
cat $out/link.txt
