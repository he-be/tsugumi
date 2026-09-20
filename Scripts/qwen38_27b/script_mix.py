#!/usr/bin/env python3
"""docs/qwen38/36 §2-2 と同じ数え方。日本語の地の文に混ざったものを数える。

「日本語に無い漢字」は手で並べると取りこぼすので、JIS X 0213 (euc_jis_2004) で符号化できない
CJK 統合漢字を「日本語の字種に無い」とする。36 の実例 (观・时・构・电・从・进・随・们) はこれで拾え、
内・党・会・体・点のような日本語の漢字は拾わない。JIS に無い日本語の異体字は誤検出になりうる。
コードと URL は地の文ではないので除く。
"""
import json, re, sys
from collections import Counter

CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
ENG = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")
CJK_COMMA = "，"


def non_japanese_kanji(text: str):
    out = []
    for c in CJK.findall(text):
        try:
            c.encode("euc_jis_2004")
        except UnicodeEncodeError:
            out.append(c)
    return out


def measure(text: str):
    t = re.sub(r"https?://\S+", " ", text)
    t = re.sub(r"`[^`]*`", " ", t)
    return non_japanese_kanji(t), t.count(CJK_COMMA), ENG.findall(t)


def main(paths):
    for path in paths:
        s_all, c_all, e_all, n = [], 0, [], 0
        for line in open(path):
            rec = json.loads(line)
            a = rec.get("answer", "")
            if not a:
                continue
            s, c, e = measure(a)
            n += 1
            s_all += s; c_all += c; e_all += e
            if s or c:
                print(f"  {rec.get('conversation')}: 日本語に無い漢字 {''.join(s)} / 中国語コンマ {c}")
        print(f"{path}: 回答 {n} 本 | 日本語に無い漢字 {len(s_all)} {Counter(s_all).most_common(6)} | "
              f"中国語コンマ {c_all} | 英単語 {len(e_all)} {Counter(e_all).most_common(8)}")


if __name__ == "__main__":
    main(sys.argv[1:])
