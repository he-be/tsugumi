#!/bin/sh
# Generate the progress ledger (docs/m6-prefill/06-LEDGER.md) from the evidence
# directories. The ledger is never written by hand; paste this output.
#   bench/m6/ledger.sh            # default rows only: env.txt has no TF_ line and
#                                 # command.txt is the fixed prefill_bench form
#   bench/m6/ledger.sh --all      # every prefill_bench directory, with its TF_ column
# A row is: date, commit (hostinfo.txt), label, n (runs with exit 0), median /
# min / max of prefill_s from summary.tsv. Runs with exit != 0 are counted
# separately and never enter the median. No ratio column exists on purpose.
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
mode=${1:-default}
host=${LEDGER_HOST:-Mac18-5}
if [ "$mode" = "--all" ]; then
    printf '| 日付 | commit | ラベル | n | prefill 壁時計 中央値 s | 最小 | 最大 | 失敗 | TF_ (無ければ既定) |\n'
    printf '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n'
else
    printf '| 日付 | commit | ラベル | n | prefill 壁時計 中央値 s | 最小 | 最大 | 失敗 |\n'
    printf '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n'
fi
for d in "$repo"/bench/m6/results/*-"$host"-*/; do
    [ -f "$d/summary.tsv" ] && [ -f "$d/env.txt" ] && [ -f "$d/command.txt" ] || continue
    grep -q -- '--messages-file' "$d/command.txt" || continue
    # Only prefill without the integrity check counts (docs/m6-prefill/35):
    # a directory measured under full-sha256 is not listed.
    grep -q -- '--verification trusted-install' "$d/command.txt" || continue
    tf=$(grep '^TF_' "$d/env.txt" | sort | tr '\n' ' ' | sed 's/ *$//' || true)
    extra=$(sed 's/.*--verification trusted-install//' "$d/command.txt" | sed 's/^ *//; s/ *$//')
    if [ "$mode" != "--all" ]; then
        [ -z "$tf" ] && [ -z "$extra" ] || continue
    fi
    name=$(basename "$d")
    date=$(echo "$name" | cut -c1-10)
    label=${name#*-$host-}
    # Section 1 is the ledger rows and the baseline only; a default-binary
    # control taken next to a candidate (n = 3) is not a ledger row.
    if [ "$mode" != "--all" ]; then
        case "$label" in ledger-*) ;; *) continue ;; esac
    fi
    commit=$(sed -n 's/.*commit \([^ /]*\).*/\1/p' "$d/hostinfo.txt")
    failed=$(awk -F'\t' 'NR > 1 && $9 != "0" { n++ } END { print n + 0 }' "$d/summary.tsv")
    stats=$(awk -F'\t' 'NR > 1 && $9 == "0" && $2 != "" { print $2 }' "$d/summary.tsv" | sort -n | awk '
        { v[NR] = $1 }
        END {
            if (NR == 0) { print "0\t-\t-\t-"; exit }
            m = (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2
            printf "%d\t%.3f\t%.3f\t%.3f\n", NR, m, v[1], v[NR]
        }')
    n=$(echo "$stats" | cut -f1); med=$(echo "$stats" | cut -f2)
    lo=$(echo "$stats" | cut -f3); hi=$(echo "$stats" | cut -f4)
    if [ "$mode" = "--all" ]; then
        col="${tf:-既定}"
        [ -n "$extra" ] && col="$col 追加引数: $extra"
        printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' "$date" "$commit" "$label" "$n" "$med" "$lo" "$hi" "$failed" "$col"
    else
        printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' "$date" "$commit" "$label" "$n" "$med" "$lo" "$hi" "$failed"
    fi
done
