"""Inkling 2 Mini を鑑定機の要求に当てるための前後処理 (docs/m6-prefill/50)。

    python3 -I ink_prefill.py make  <kj_payloads.jsonl> <out.jsonl>   # 各要求の最後に assistant "{" を足す
    python3 -I ink_prefill.py score <answers.jsonl>                   # "{" を補って JSON として読めるかを数える

Inkling は思考を切れない (reasoning_effort none でも思考を書き、max_tokens 120〜560 を使い切って本文が空になる)。
assistant の書き出しに "{" を置くと思考を飛ばして本文から書く。ストリーミングの答えには置いた "{" が入らない。
読み以外の 3 問ではモデルが自分でも "{" を書き直すので、補うのは先頭が "{" でないときだけ。
"""
import collections
import json
import sys

mode = sys.argv[1]
if mode == "make":
    with open(sys.argv[3], "w", encoding="utf-8") as out:
        for line in open(sys.argv[2], encoding="utf-8"):
            row = json.loads(line)
            row["payload"]["messages"].append({"role": "assistant", "content": "{"})
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
elif mode == "score":
    bad, again, n = collections.Counter(), collections.Counter(), 0
    for line in open(sys.argv[2], encoding="utf-8"):
        row = json.loads(line); n += 1
        text = row["text"].lstrip()
        if text.startswith("{"): again[row["name"]] += 1
        else: text = "{" + row["text"]
        try: json.loads(text)
        except ValueError: bad[row["name"]] += 1
    print(f"answers {n}  model re-emitted '{{' {dict(again)}  invalid {dict(bad)}")
