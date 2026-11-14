import json, sys, statistics as st
print("arm\tn\tinvalid\tcut at 120\tpredicted_n median / p95 / max\t>=60 tokens")
for a in sys.argv[1:]:
    R = [json.loads(l) for l in open(f"item_{a}.jsonl", encoding="utf-8")]
    n = sorted(r["predicted_n"] for r in R)
    plays = sorted({r["play"] for r in R if not r["valid"]})
    print("%s\t%d\t%d\t%d\t%d / %d / %d\t%d\tplays with a failure: %d" % (a, len(R), sum(not r["valid"] for r in R), sum(r["finish"] == "length" for r in R),
          st.median(n), n[int(len(n) * .95)], n[-1], sum(x >= 60 for x in n), len(plays)))
