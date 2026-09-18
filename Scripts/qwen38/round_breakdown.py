#!/usr/bin/env python3
"""Count what the Qwen3.8 tool-loop rounds spend, by the kind of round (docs/qwen38/33 R0). Reads records only.

    python3 Scripts/qwen38/round_breakdown.py [--app DIR] [--run LABEL=DIR[,DIR...]] ... [--evidence]

Two kinds of record:

- The app (`chats.json` × `turn-metrics.jsonl`). `chats.json` has no chat or turn ids (session identity is
  per-process), so a metrics turn is matched to a chat turn by shape: the same number of model rounds with the
  same number of calls each, the same answered/unanswered end, and turns of one metrics chat in one chat, in order.
  Among shape matches the one whose "next round's new tokens ÷ this round's result chars" is steadiest wins
  (`--evidence` prints the ratios). Only Qwen3.8 turns with tools (online / offline) are counted.
- `TsugumiToolLoopCheck` runs (`rounds.jsonl`), which carry the calls, the tool choice and the numbers per round.

A round is classified by what it wrote. "結果の読み" is the new tokens of the next round of the same turn, i.e. the
cost of reading what this round's calls returned (plus the call itself and the next turn header).
"""
import argparse, json, statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP_DIR = Path.home() / "Library/Application Support/Tsugumi"
KS = REPO / "docs/experiments/knowledge-sources/ple-ab"
SQ = REPO / "scratch/qwen38"
DEFAULT_RUNS = [
    ("CLI Mac offline 現行 (09-17、knowledge-sources 5 問 × 6)",
     [KS / f"{t}/offline/rep{i}" for t in ("bf16", "q41") for i in (1, 2, 3)]),
    ("CLI Mac online 24 以後 (09-15、7 会話)", [SQ / "outline24/qwen32k-b"]),
    ("CLI Mac online 24 以前 (09-15、23 §5 の 6 会話)", [SQ / "kv22/tools32k-c"]),
]
SEARCHES = ("web_search", "wikipedia_search")
PAGES = ("fetch_page", "wikipedia_page")
APP_MAX_ROUNDS = 6  # WebSearchConfiguration.maxToolRounds (the app's default)

KIND_ORDER = [
    "web_search (強制)", "web_search", "wikipedia_search", "fetch_page (強制)",
    "ページ (sections なし)", "ページ (sections つき)", "ページ (from)", "複数呼び出し", "その他の呼び出し",
    "回答", "回答 (上限)", "回答なしで終わり",
]


def page_key(name, args):
    return (name, args.get("url") or args.get("title"))


def classify(calls, choice, answered_by_limit=False):
    """calls: [(name, args)]; choice: 'auto' / 'none' / 'function:NAME'."""
    if not calls:
        return "回答 (上限)" if (choice == "none" or answered_by_limit) else "回答"
    if len(calls) > 1:
        return "複数呼び出し"
    name, args = calls[0]
    if name == "web_search":
        return "web_search (強制)" if choice == "function:web_search" else "web_search"
    if name == "wikipedia_search":
        return "wikipedia_search"
    if name in PAGES:
        if choice == "function:fetch_page":
            return "fetch_page (強制)"
        if args.get("sections") not in (None, ""):
            return "ページ (sections つき)"
        if args.get("from") not in (None, "", 0):
            return "ページ (from)"
        return "ページ (sections なし)"
    return "その他の呼び出し"


def parse_call(s):
    name, _, rest = s.partition(" ")
    try:
        args = json.loads(rest) if rest else {}
    except json.JSONDecodeError:
        args = {"_raw": rest}
    return name, args


# ---------------------------------------------------------------- CLI runs

def load_run(d):
    """→ list of turns; a turn is a list of round dicts in order."""
    turns = defaultdict(list)
    for line in open(d / "rounds.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        turns[(str(d), r["conversation"], r["repeat"], r["turn"])].append(r)
    out = []
    for key, rs in turns.items():
        rs.sort(key=lambda r: r["round"])
        turn = []
        for r in rs:
            calls = [parse_call(c) for c in r.get("calls", [])]
            if r.get("error"):
                kind = f"失敗 ({r['error']})"
            elif calls or r.get("stop") != "toolCalls":
                kind = classify(calls, r.get("choice", "auto"))
            else:
                kind = "回答なしで終わり"
            # a failed round has no prefill / decode split; its wall time is left out of both
            turn.append(dict(kind=kind, calls=calls, new=r["prompt"] - r.get("cached", 0), gen=r["generated"],
                             prefill=r.get("prefill_s", 0.0), decode=r.get("decode_s", 0.0), read=None))
        out.append(dict(key=key, rounds=turn))
    return out


# ---------------------------------------------------------------- the app

def is_app_call(call):
    return call.get("id", "").startswith("lookup-") or call["name"] == "wikipedia_lookup"


def chat_units(chat):
    """The chat's turns (history, then the current one) as model rounds: [{calls, chars, statuses}], answered."""
    def walk(turns, trace_status, answered_last):
        units, cur = [], None
        results = {}
        for t in turns:
            if t.get("role") == "tool":
                results[t.get("toolCallID")] = len(t.get("text", ""))
        for t in turns:
            role = t.get("role")
            if role == "user":
                cur = dict(pre_chars=0, app_fetch=False, rounds=[], answered=False)
                units.append(cur)
                continue
            if role != "assistant":
                continue
            if cur is None:
                cur = dict(pre_chars=0, app_fetch=False, rounds=[], answered=False)
                units.append(cur)
            calls = t.get("toolCalls") or []
            if not calls:
                cur["answered"] = True
                continue
            app = [c for c in calls if is_app_call(c)]
            model = [c for c in calls if not is_app_call(c)]
            for c in app:
                cur["pre_chars"] += results.get(c.get("id"), 0)
                if c["name"] == "fetch_page":
                    cur["app_fetch"] = True
            if model:
                cur["rounds"].append(dict(
                    calls=[(c["name"], json.loads(c.get("argumentsJSON") or "{}")) for c in model],
                    chars=sum(results.get(c.get("id"), 0) for c in model),
                    statuses=[trace_status.get(c.get("id"), "done") for c in model]))
        if answered_last is not None and units:
            units[-1]["answered"] = answered_last
        return units

    history = walk(chat.get("turns", []), {}, None)
    status = {e["id"]: e["status"] for e in chat.get("outputToolTrace", [])}
    current = walk([{"role": "user"}] + chat.get("outputContinuationTurns", []), status,
                   bool(chat.get("outputText")))
    if not chat.get("outputPromptText") and not current[0]["rounds"]:
        current = []
    return history + current


def metric_turns(app_dir):
    groups = defaultdict(list)
    by_turn = defaultdict(list)
    for line in open(app_dir / "turn-metrics.jsonl"):
        r = json.loads(line)
        if not r["model"].startswith("qwen38") or r["network"] not in ("online", "offline"):
            continue
        by_turn[r["turnID"]].append(r)
    for tid, rs in by_turn.items():
        rs.sort(key=lambda r: r["round"])
        groups[rs[0]["chatID"]].append(dict(
            id=tid, rounds=rs, network=rs[0]["network"], at=rs[0]["recordedAt"],
            sig=[r["toolCalls"] for r in rs if r["toolCalls"] > 0],
            answered=rs[-1]["toolCalls"] == 0 and rs[-1]["outcome"] == "finished"))
    for g in groups.values():
        g.sort(key=lambda t: t["at"])
    return groups


def ratios(mt, unit):
    """new tokens of round k+1 ÷ result chars of model round k."""
    out = []
    rs = mt["rounds"]
    for k, rnd in enumerate(unit["rounds"]):
        if k + 1 < len(rs) and rnd["chars"] > 0:
            new = rs[k + 1]["promptTokens"] - rs[k + 1].get("cachedPromptTokens", 0)
            out.append(new / rnd["chars"])
    return out


def score(mt, unit):
    rs = ratios(mt, unit)
    if len(rs) < 2:
        return 0.5
    import math
    logs = [math.log(max(x, 1e-6)) for x in rs]
    return statistics.pstdev(logs)


def align(group, units):
    """Best in-order placement of the group's turns on shape-matching units → (score, [unit index]) or None."""
    best = None

    def rec(i, start, acc, placed):
        nonlocal best
        if i == len(group):
            s = acc / len(group)
            if best is None or s < best[0]:
                best = (s, list(placed))
            return
        mt = group[i]
        for j in range(start, len(units)):
            u = units[j]
            if [len(r["calls"]) for r in u["rounds"]] == mt["sig"] and u["answered"] == mt["answered"]:
                placed.append(j)
                rec(i + 1, j + 1, acc + score(mt, u), placed)
                placed.pop()

    rec(0, 0, 0.0, [])
    return best


def load_app(app_dir, evidence):
    chats = json.load(open(app_dir / "chats.json"))["chats"]
    units = [chat_units(c) for c in chats]
    groups = metric_turns(app_dir)
    cands = []
    for cid, g in groups.items():
        for ci, us in enumerate(units):
            a = align(g, us)
            if a:
                cands.append((a[0], cid, ci, a[1]))
    cands.sort(key=lambda x: x[0])
    used_g, used_c, matches = set(), set(), {}
    for s, cid, ci, placed in cands:
        if cid in used_g or ci in used_c:
            continue
        used_g.add(cid)
        used_c.add(ci)
        matches[cid] = (ci, placed, s)

    turns, unmatched = [], []
    for cid, g in groups.items():
        if cid not in matches:
            unmatched.extend(g)
            continue
        ci, placed, s = matches[cid]
        for mt, uj in zip(g, placed):
            u = units[ci][uj]
            if evidence:
                rs = ", ".join(f"{x:.2f}" for x in ratios(mt, u))
                print(f"  {mt['at'][:16]} {mt['id'][:8]} {mt['network']:7} → chats[{ci}] 手番 {uj}"
                      f" ({chats[ci]['outputPromptText'][:18]!r})  新規/字: {rs}")
            turns.append(dict(key=(mt["id"],), rounds=app_rounds(mt, u)))
    if evidence:
        for mt in unmatched:
            print(f"  {mt['at'][:16]} {mt['id'][:8]} {mt['network']:7} 対応なし (呼び出し {mt['sig']}、"
                  f"回答 {'あり' if mt['answered'] else 'なし'})")
    return turns, unmatched


def app_rounds(mt, unit):
    online = mt["network"] == "online"
    fetch_tried, searched = unit["app_fetch"], False
    out = []
    ms = mt["rounds"]
    for k, r in enumerate(ms):
        new = r["promptTokens"] - r.get("cachedPromptTokens", 0)
        base = dict(new=new, gen=r["generatedTokens"], prefill=r["prefillSeconds"], decode=r["decodeSeconds"],
                    disk=r.get("prefillDiskReadBytes"), read=None)
        if k < len(unit["rounds"]):
            rnd = unit["rounds"][k]
            if not online or fetch_tried:
                choice = "auto"
            elif searched:
                choice = "function:fetch_page"
            else:
                choice = "function:web_search"
            for (name, _), st in zip(rnd["calls"], rnd["statuses"]):
                if name == "fetch_page":
                    fetch_tried = True
                if name == "web_search" and st == "done":
                    searched = True
            out.append(dict(base, kind=classify(rnd["calls"], choice), calls=rnd["calls"]))
        elif r["toolCalls"] == 0 and r["outcome"] == "finished":
            out.append(dict(base, kind=classify([], "auto", len(unit["rounds"]) >= APP_MAX_ROUNDS), calls=[]))
        else:
            out.append(dict(base, kind="回答なしで終わり", calls=[]))
    return out


# ---------------------------------------------------------------- report

def fill_read(turns):
    for t in turns:
        rs = t["rounds"]
        for k, r in enumerate(rs):
            if r["calls"] and k + 1 < len(rs):
                r["read"] = rs[k + 1]["new"]


def med(xs):
    return statistics.median(xs) if xs else None


def fmt(x, nd=0):
    if x is None:
        return "–"
    return f"{x:,.{nd}f}"


SHORT = {
    "web_search (強制)": "S!", "web_search": "S", "wikipedia_search": "W", "fetch_page (強制)": "F!",
    "ページ (sections なし)": "P", "ページ (sections つき)": "P§", "ページ (from)": "P+", "複数呼び出し": "M",
    "その他の呼び出し": "?", "回答": "A", "回答 (上限)": "A!", "回答なしで終わり": "–",
}


def sequences(turns):
    print("\n記号: S! 強制検索、S web_search、W wikipedia_search、F! 強制 fetch_page、P ページ (sections なし)、"
          "P§ sections つき、P+ from、M 複数呼び出し、A 回答、A! 上限の回答、– 回答なしで終わり、E 失敗、= 直前と同じページ\n")
    for t in turns:
        out, prev = [], None
        for r in t["rounds"]:
            sym = "E" if r["kind"].startswith("失敗") else SHORT.get(r["kind"], "x")
            keys = {page_key(n, a) for n, a in r["calls"] if n in PAGES}
            if prev and keys & prev:
                sym += "="
            prev = keys
            out.append(sym)
        print(f"- `{' '.join(out)}`  {' / '.join(str(k) for k in t['key'][1:]) or t['key'][0][:8]}")


def report(label, turns, remote_note="", show_sequences=False):
    fill_read(turns)
    rounds = [r for t in turns for r in t["rounds"]]
    print(f"\n### {label}\n")
    per_turn = [len(t["rounds"]) for t in turns]
    print(f"ターン {len(turns)}、ラウンド {len(rounds)} (1 ターンのラウンド: {dict(sorted(Counter(per_turn).items()))})"
          f"、prefill 合計 {sum(r['prefill'] for r in rounds):,.0f} s、decode 合計 {sum(r['decode'] for r in rounds):,.0f} s"
          f"、新規 {sum(r['new'] for r in rounds):,}、生成 {sum(r['gen'] for r in rounds):,}{remote_note}\n")
    print("| 種類 | 本数 | 新規 合計 | 新規 中央値 | 生成 合計 | 生成 中央値 | prefill 合計 s | decode 合計 s"
          " | 結果の読み 合計 | 結果の読み 中央値 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    by = defaultdict(list)
    for r in rounds:
        by[r["kind"]].append(r)
    for kind in KIND_ORDER + sorted(set(by) - set(KIND_ORDER)):
        rs = by.get(kind)
        if not rs:
            continue
        reads = [r["read"] for r in rs if r["read"] is not None]
        print(f"| {kind} | {len(rs)} | {sum(r['new'] for r in rs):,} | {fmt(med([r['new'] for r in rs]))}"
              f" | {sum(r['gen'] for r in rs):,} | {fmt(med([r['gen'] for r in rs]))}"
              f" | {sum(r['prefill'] for r in rs):,.0f} | {sum(r['decode'] for r in rs):,.0f}"
              f" | {fmt(sum(reads) if reads else None)} | {fmt(med(reads))} |")
    patterns(turns)
    if show_sequences:
        sequences(turns)


def patterns(turns):
    re_search = re_search_turns = 0
    outline_then_sections = forced_then_sections = forced_fetch = 0
    same_args = multi = multi_calls = 0
    multi_names = Counter()
    page_opens = 0
    after_forced = Counter()
    for t in turns:
        for a, b in zip(t["rounds"], t["rounds"][1:]):
            if a["kind"] == "fetch_page (強制)":
                same = any(page_key(n, x) == page_key(*a["calls"][0]) for n, x in b["calls"])
                after_forced[b["kind"] + (" (同じページ)" if same else "")] += 1
        opened, seen_args, no_section_pages, forced_pages = False, set(), set(), set()
        turn_re = False
        for r in t["rounds"]:
            calls = r["calls"]
            if len(calls) > 1:
                multi += 1
                multi_calls += len(calls)
                multi_names[" + ".join(sorted(n for n, _ in calls))] += 1
            for name, args in calls:
                sig = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                if sig in seen_args:
                    same_args += 1
                seen_args.add(sig)
                if name in SEARCHES and opened:
                    re_search += 1
                    turn_re = True
                if name in PAGES:
                    page_opens += 1
                    key = page_key(name, args)
                    has_sections = args.get("sections") not in (None, "")
                    forced = r["kind"] == "fetch_page (強制)"
                    forced_fetch += forced
                    if has_sections:
                        if key in forced_pages:
                            forced_then_sections += 1
                        elif key in no_section_pages:
                            outline_then_sections += 1
                        forced_pages.discard(key)
                        no_section_pages.discard(key)
                        if forced:
                            forced_pages.add(key)
                    elif forced:
                        forced_pages.add(key)
                    else:
                        no_section_pages.add(key)
                    opened = True
        re_search_turns += turn_re
    print()
    print(f"- ページを開いた後の検索: {re_search} 回 ({re_search_turns} / {len(turns)} ターン)")
    if forced_fetch:
        print(f"- 強制 fetch_page {forced_fetch} 回のうち、後で同じページの sections を読んだ: {forced_then_sections} 回")
        print(f"- 強制 fetch_page の次のラウンド: {dict(after_forced.most_common())}")
    print(f"- 強制でない sections なしの取得の後に、同じページの sections を読んだ: {outline_then_sections} 回")
    print(f"- ページを開いた呼び出し: {page_opens} 回。同じターンで引数まで同じ呼び出しの繰り返し: {same_args} 回")
    print(f"- 複数呼び出しのラウンド: {multi} 本 (呼び出し {multi_calls} 本){': ' + dict(multi_names).__repr__() if multi else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", type=Path, default=APP_DIR)
    ap.add_argument("--no-app", action="store_true")
    ap.add_argument("--run", action="append", default=[], help="LABEL=DIR[,DIR...]")
    ap.add_argument("--evidence", action="store_true", help="print how app turns were matched")
    ap.add_argument("--sequences", action="store_true", help="print each turn's rounds as a row of symbols")
    a = ap.parse_args()

    if not a.no_app:
        if a.evidence:
            print("アプリの突き合わせ:")
        turns, unmatched = load_app(a.app, a.evidence)
        un_rounds = sum(len(t["rounds"]) for t in unmatched)
        note = (f"。対応が取れず除いたターン {len(unmatched)} (ラウンド {un_rounds})" if unmatched else "")
        report(f"アプリ ({a.app / 'turn-metrics.jsonl'}、Qwen3.8、online / offline)", turns, note, a.sequences)

    runs = [(lab, [Path(p) for p in dirs.split(",")]) for lab, _, dirs in (r.partition("=") for r in a.run)]
    for label, dirs in runs or DEFAULT_RUNS:
        turns = [t for d in dirs for t in load_run(d)]
        report(label, turns, show_sequences=a.sequences)


if __name__ == "__main__":
    main()
