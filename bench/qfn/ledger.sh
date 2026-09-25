#!/bin/sh
# Generate the QFN ledger (docs/qfn-prefill/06-LEDGER.md) from the evidence directories of bench/qfn/pair_bench.sh.
# The ledger is never written by hand; paste this output.
#   bench/qfn/ledger.sh          # §1: ledger-* directories whose env.txt has no Q38_/TF_ line
#   bench/qfn/ledger.sh --all    # §2: every pair_bench directory, with its Q38_/TF_ column
# One row per 本文 1 body (median of prefill_s over its runs) and one row for 本文 2 (median of the turn's prefill total).
# Runs with exit != 0 are counted separately and never enter the median. 読み MB is the median Pageins of one run
# (MBP only in the solo column). No ratio column exists on purpose (06 §0-3).
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
mode=${1:-default}
extra_head=""; extra_rule=""
[ "$mode" = "--all" ] && { extra_head=" Q38_ / TF_ |"; extra_rule=" --- |"; }
printf '| 日付 | commit | ラベル | 列 | 分け方 | 本文 | n | prefill 壁時計 中央値 s | 最小 | 最大 | 読み MB (MBP / M6 内 / M6 外) | 失敗 |%s\n' "$extra_head"
printf '| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |%s\n' "$extra_rule"
for d in "$repo"/bench/qfn/results/*/; do
    [ -f "$d/command.txt" ] && [ -f "$d/env.txt" ] && [ -f "$d/hostinfo.mbp.txt" ] || continue
    ls "$d"/summary-b[12].tsv > /dev/null 2>&1 || continue
    name=$(basename "$d"); date=$(echo "$name" | cut -c1-10); rest=${name#??????????-}
    case "$rest" in pair-*) col=2 機; label=${rest#pair-} ;; *) col=単体; label=${rest#*-*-} ;; esac
    knobs=$(grep -E '^(Q38_|TF_)' "$d/env.txt" | sort | tr '\n' ' ' | sed 's/ *$//' || true)
    if [ "$mode" != "--all" ]; then
        case "$label" in ledger-*) ;; *) continue ;; esac
        [ -z "$knobs" ] || continue
    fi
    commit=$(sed -n 's/.*commit \([^ /]*\).*/\1/p' "$d/hostinfo.mbp.txt")
    split=-
    [ -f "$d/link.txt" ] && split=$(sed -n 's/^split //p' "$d/link.txt")
    tail=""
    [ "$mode" = "--all" ] && tail=" ${knobs:-既定} |"
    stats() {  # values on stdin -> n median min max
        sort -n | awk '{ v[NR] = $1 } END {
            if (NR == 0) { print "0\t-\t-\t-"; exit }
            m = (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2
            printf "%d\t%.3f\t%.3f\t%.3f\n", NR, m, v[1], v[NR] }'
    }
    row() {  # body stats reads failed
        printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / - / - | %s |%s\n' "$date" "$commit" "$label" "$col" \
            "$split" "$1" "$(echo "$2" | cut -f1)" "$(echo "$2" | cut -f2)" "$(echo "$2" | cut -f3)" "$(echo "$2" | cut -f4)" \
            "${3%.*}" "$4" "$tail"
    }
    if [ -f "$d/summary-b1.tsv" ]; then
        for body in $(awk -F'\t' 'NR > 1 { print $1 }' "$d/summary-b1.tsv" | sort -u); do
            s=$(awk -F'\t' -v b="$body" 'NR > 1 && $1 == b && $8 == "0" && $5 != "" { print $5 }' "$d/summary-b1.tsv" | stats)
            r=$(awk -F'\t' -v b="$body" 'NR > 1 && $1 == b && $8 == "0" { print $6 }' "$d/summary-b1.tsv" | stats | cut -f2)
            f=$(awk -F'\t' -v b="$body" 'NR > 1 && $1 == b && $8 != "0" { n++ } END { print n + 0 }' "$d/summary-b1.tsv")
            row "本文 1 $body" "$s" "$r" "$f"
        done
    fi
    if [ -f "$d/summary-b2.tsv" ]; then
        s=$(awk -F'\t' 'NR > 1 && $9 == "0" { print $3 }' "$d/summary-b2.tsv" | stats)
        r=$(awk -F'\t' 'NR > 1 && $9 == "0" { print $7 }' "$d/summary-b2.tsv" | stats | cut -f2)
        f=$(awk -F'\t' 'NR > 1 && $9 != "0" { n++ } END { print n + 0 }' "$d/summary-b2.tsv")
        row "本文 2 (1 ターンの合計)" "$s" "$r" "$f"
    fi
done
