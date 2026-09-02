#!/usr/bin/env python3
"""日本語版 Wikipedia を Tsugumi のローカル検索用 SQLite に組む。

入力は Wikimedia の週次検索ダンプ (cirrus_search_index) の jawiki_content
分割ファイル。1 記事が index 行と本文行の対で、本文行の `text` は整形済みの
平文、`opening_text` は導入部、`redirect` は転送元の名前、`incoming_links`
は被リンク数。標準ライブラリだけで動く。

    download  ダンプの分割ファイルを取ってくる (curl、再開可)
    build     分割ファイルから SQLite を組む
    search    組んだ SQLite を検索する (アプリと同じ順位付け)
    page      1 記事の本文を出す

索引の字句はアプリ側 (Sources/TsugumiApp/Core/LocalWikipedia/WikipediaTokenizer.swift)
と同じ規則で切る。`tokenize()` を変えたら Swift も変え、SCHEMA_VERSION を上げる。
"""

import argparse
import bz2
import json
import math
import multiprocessing
import os
import sqlite3
import subprocess
import sys
import time
import unicodedata
import urllib.request
import zlib
from html.parser import HTMLParser

SCHEMA_VERSION = 1
TOKENIZER_VERSION = "bigram-2"
# 組んだ側と読む側で字句の切り方が揃っているかを確かめる標本。両側で
# tokenize() した結果を meta.tokenizer_check と突き合わせる。
TOKENIZER_CHECK_SAMPLE = "東京タワーは2026年9月1日に iPhone 16 Pro で撮った。Ａ１ ｱｲｳ 々 ー A・B M4"

DUMP_BASE = "https://dumps.wikimedia.org/other/cirrus_search_index"


# --- 字句 -----------------------------------------------------------------

def _is_cjk(cp):
    return (0x3040 <= cp <= 0x30FF      # ひらがな・カタカナ (ー を含む)
            or 0x31F0 <= cp <= 0x31FF   # カタカナ拡張
            or 0x3400 <= cp <= 0x4DBF   # 漢字 拡張 A
            or 0x4E00 <= cp <= 0x9FFF   # 漢字
            or 0xF900 <= cp <= 0xFAFF   # 互換漢字
            or 0x20000 <= cp <= 0x2FFFF # 漢字 拡張 B 以降
            or 0xAC00 <= cp <= 0xD7AF   # ハングル
            or cp in (0x3005, 0x3006, 0x3007))  # 々 〆 〇


def _kind(ch):
    """2 = CJK か数字 (バイグラムで切る)、1 = それ以外の英数 (語で切る)、0 = 区切り。
    CJK の範囲でも文字・数字でないもの (中黒「・」など) は区切り。"""
    if not ch.isalnum():
        return 0
    if _is_cjk(ord(ch)) or ch.isdigit():
        return 2
    return 1


def tokenize(text):
    """NFKC・小文字化のあと、CJK と数字の連なりはバイグラム、英数の連なりは
    1 語、それ以外は区切り。1 文字だけの CJK の連なりはその 1 文字。"""
    text = unicodedata.normalize("NFKC", text).lower()
    tokens = []
    run = []
    run_kind = 0

    def flush():
        if not run:
            return
        if run_kind == 1:
            tokens.append("".join(run))
        elif len(run) == 1:
            tokens.append(run[0])
        else:
            for i in range(len(run) - 1):
                tokens.append(run[i] + run[i + 1])
        run.clear()

    for ch in text:
        kind = _kind(ch)
        if kind != run_kind:
            flush()
            run_kind = kind
        if kind:
            run.append(ch)
    flush()
    return tokens


MATCH_ALL_TERMS, MATCH_ANY_TERM, MATCH_ANY_TOKEN = range(3)


def match_expression(query, mode=MATCH_ALL_TERMS):
    """FTS5 の MATCH 式。空白で区切った語ごとにバイグラム列を句にし、全語 AND
    → いずれかの語 OR → いずれかのバイグラム OR の順に緩める。"""
    phrases = []
    for term in query.split():
        toks = tokenize(term)
        if not toks:
            continue
        if mode == MATCH_ANY_TOKEN:
            phrases.extend('"' + t + '"' for t in toks)
        else:
            phrases.append('"' + " ".join(toks) + '"')
    if not phrases:
        return None
    return (" AND " if mode == MATCH_ALL_TERMS else " OR ").join(dict.fromkeys(phrases))


def normalize_title(title):
    return " ".join(unicodedata.normalize("NFKC", title).replace("_", " ").split()).casefold()


# --- スキーマ -------------------------------------------------------------

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE pages (
    id        INTEGER PRIMARY KEY,
    title     TEXT NOT NULL,
    opening   TEXT,
    text      BLOB NOT NULL,      -- 本文 (UTF-8) を raw deflate で圧縮
    text_bytes INTEGER NOT NULL,  -- 圧縮前のバイト数
    incoming  INTEGER NOT NULL DEFAULT 0,
    updated   TEXT
);
CREATE TABLE titles (
    norm     TEXT PRIMARY KEY,    -- normalize_title() した題名か転送元
    page_id  INTEGER NOT NULL,
    redirect INTEGER NOT NULL     -- 1 なら転送元の名前
) WITHOUT ROWID;
CREATE VIRTUAL TABLE search USING fts5(
    title, aliases, opening, body,
    content='', tokenize='unicode61'
);
"""

# bm25 の列の重み: 題名 > 別名 > 導入部 > 本文の頭
BM25_WEIGHTS = (10.0, 6.0, 2.0, 1.0)


def _connect(path):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-262144")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


# --- 読み込み -------------------------------------------------------------

def _open_dump(path):
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def iter_docs(path):
    """index 行を飛ばし、本文行だけを dict で返す。"""
    with _open_dump(path) as handle:
        for line in handle:
            if line.startswith('{"index"'):
                continue
            yield json.loads(line)


def _clean_text(text):
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def make_row(doc, body_chars):
    """1 記事の挿入用タプル。転送ページや空の記事は None。"""
    if doc.get("namespace", 0) != 0 or doc.get("page_type") == "redirect":
        return None
    text = _clean_text(doc.get("text") or "")
    title = doc.get("title") or ""
    if not text or not title:
        return None
    opening = (doc.get("opening_text") or "").strip()
    if not opening:
        opening = text[:300]
    aliases = [r["title"] for r in doc.get("redirect") or []
               if isinstance(r, dict) and r.get("namespace", 0) == 0 and r.get("title")]
    body = text[len(opening):len(opening) + body_chars] if body_chars > 0 else ""
    raw = text.encode("utf-8")
    compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
    packed = compressor.compress(raw) + compressor.flush()
    return (
        int(doc["page_id"]), title, opening, packed, len(raw),
        int(doc.get("incoming_links") or 0), doc.get("timestamp") or "",
        aliases,
        " ".join(tokenize(title)),
        " ".join(" ".join(tokenize(a)) for a in aliases),
        " ".join(tokenize(opening)),
        " ".join(tokenize(body)),
    )


def _shard_worker(args):
    path, body_chars, limit, queue = args
    batch = []
    count = 0
    for doc in iter_docs(path):
        row = make_row(doc, body_chars)
        if row is None:
            continue
        batch.append(row)
        count += 1
        if len(batch) >= 200:
            queue.put(batch)
            batch = []
        if limit and count >= limit:
            break
    if batch:
        queue.put(batch)
    queue.put(("done", path, count))
    return count


def build(out, shards, body_chars, limit, jobs, dump_date):
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; remove it first")
    tmp = out + ".building"
    if os.path.exists(tmp):
        os.remove(tmp)
    conn = _connect(tmp)
    conn.executescript(SCHEMA)
    started = time.time()
    total = 0
    manager = multiprocessing.Manager()
    queue = manager.Queue(maxsize=jobs * 4)
    pool = multiprocessing.Pool(processes=jobs)
    result = pool.map_async(_shard_worker,
                            [(s, body_chars, limit, queue) for s in shards])
    remaining = len(shards)
    last_report = started
    while remaining:
        item = queue.get()
        if isinstance(item, tuple) and item and item[0] == "done":
            remaining -= 1
            print(f"[build] {os.path.basename(item[1])}: {item[2]:,} articles",
                  file=sys.stderr, flush=True)
            continue
        conn.executemany(
            "INSERT INTO pages (id, title, opening, text, text_bytes, incoming, updated)"
            " VALUES (?,?,?,?,?,?,?)",
            [r[:7] for r in item])
        conn.executemany(
            "INSERT INTO search (rowid, title, aliases, opening, body) VALUES (?,?,?,?,?)",
            [(r[0], r[8], r[9], r[10], r[11]) for r in item])
        titles = []
        for r in item:
            titles.append((normalize_title(r[1]), r[0], 0))
            for alias in r[7]:
                titles.append((normalize_title(alias), r[0], 1))
        # 本記事の題名を優先: 別名 (転送) が既存の題名を上書きしない
        conn.executemany(
            "INSERT INTO titles (norm, page_id, redirect) VALUES (?,?,?)"
            " ON CONFLICT(norm) DO UPDATE SET page_id=excluded.page_id, redirect=excluded.redirect"
            " WHERE titles.redirect=1 AND excluded.redirect=0",
            titles)
        total += len(item)
        if time.time() - last_report > 30:
            elapsed = time.time() - started
            print(f"[build] {total:,} articles, {elapsed/60:.1f} min, "
                  f"{total/elapsed:,.0f}/s", file=sys.stderr, flush=True)
            last_report = time.time()
    pool.close()
    pool.join()
    result.get()
    conn.commit()
    print("[build] optimizing the index…", file=sys.stderr, flush=True)
    conn.execute("INSERT INTO search(search) VALUES('optimize')")
    meta = {
        "schema": str(SCHEMA_VERSION),
        "wiki": "jawiki",
        "dump_date": dump_date or "",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "articles": str(total),
        "body_chars": str(body_chars),
        "tokenizer": TOKENIZER_VERSION,
        "tokenizer_check": " ".join(tokenize(TOKENIZER_CHECK_SAMPLE)),
    }
    conn.executemany("INSERT INTO meta (key, value) VALUES (?,?)", meta.items())
    conn.commit()
    conn.close()
    conn = sqlite3.connect(tmp)
    conn.execute("VACUUM")
    conn.close()
    os.rename(tmp, out)
    elapsed = time.time() - started
    size = os.path.getsize(out)
    print(f"[build] {out}: {total:,} articles, {size/1e9:.2f} GB, {elapsed/60:.1f} min",
          file=sys.stderr)


# --- 検索 (アプリと同じ手順) ----------------------------------------------

def search(db, query, limit=10):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = []
    for mode in (MATCH_ALL_TERMS, MATCH_ANY_TERM, MATCH_ANY_TOKEN):
        expr = match_expression(query, mode)
        if expr is None:
            return []
        sql = ("SELECT s.rowid, bm25(search, ?, ?, ?, ?) AS rank"
               " FROM search s WHERE search MATCH ? ORDER BY rank LIMIT 60")
        rows = conn.execute(sql, (*BM25_WEIGHTS, expr)).fetchall()
        if rows:
            break
    # 題名か転送名にそのまま一致する記事は、順位付けに関わらず先頭。
    exact = conn.execute("SELECT page_id FROM titles WHERE norm=?",
                         (normalize_title(query),)).fetchone()
    ids = [r[0] for r in rows]
    if exact and exact[0] not in ids:
        ids.append(exact[0])
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    pages = {pid: (title, opening, incoming) for pid, title, opening, incoming in conn.execute(
        f"SELECT id, title, opening, incoming FROM pages WHERE id IN ({marks})", ids)}
    scored = []
    for pid, rank in rows:
        if exact and pid == exact[0]:
            continue
        title, opening, incoming = pages[pid]
        # 被リンク数で軽く持ち上げる (加算)。新しい記事は 0 なので弱く。
        score = rank - 0.5 * math.log1p(incoming)
        scored.append((score, pid, title, opening, incoming))
    scored.sort()
    if exact:
        title, opening, incoming = pages[exact[0]]
        scored.insert(0, (float("-inf"), exact[0], title, opening, incoming))
    return [(pid, title, snippet(opening, query), incoming) for _, pid, title, opening, incoming
            in scored[:limit]]


def snippet(opening, query, width=120):
    text = " ".join(opening.split())
    lower = unicodedata.normalize("NFKC", text).lower()
    start = 0
    for term in query.split():
        pos = lower.find(unicodedata.normalize("NFKC", term).lower())
        if pos >= 0:
            start = max(0, pos - width // 3)
            break
    piece = text[start:start + width]
    return ("…" if start > 0 else "") + piece + ("…" if start + width < len(text) else "")


def page(db, title):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = conn.execute("SELECT page_id FROM titles WHERE norm=?", (normalize_title(title),)).fetchone()
    if not row:
        return None
    pid, real_title, packed, size = conn.execute(
        "SELECT id, title, text, text_bytes FROM pages WHERE id=?", (row[0],)).fetchone()
    text = zlib.decompress(packed, -15).decode("utf-8")
    assert len(text.encode("utf-8")) == size
    return real_title, text


# --- ダウンロード ---------------------------------------------------------

class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.links.append(value)


def _listing(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        parser = _LinkParser()
        parser.feed(response.read().decode("utf-8", "replace"))
        return parser.links


def download(date, directory, wiki="jawiki"):
    if not date:
        dates = sorted(l.strip("/") for l in _listing(DUMP_BASE + "/") if l.strip("/").isdigit())
        date = dates[-1]
    base = f"{DUMP_BASE}/{date}/index_name%3D{wiki}_content/"
    files = sorted(l for l in _listing(base) if l.endswith(".json.bz2"))
    if not files:
        raise SystemExit(f"no shards under {base}")
    target = os.path.join(directory, f"{wiki}-{date}")
    os.makedirs(target, exist_ok=True)
    print(f"[download] {date}: {len(files)} shards → {target}", file=sys.stderr)
    for name in files:
        dest = os.path.join(target, name)
        subprocess.run(["curl", "-sS", "-C", "-", "--retry", "5", "--retry-delay", "10",
                        "-o", dest, base + name], check=True)
        print(f"[download] {name} {os.path.getsize(dest):,} bytes", file=sys.stderr)
    return target, date


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="ダンプの分割ファイルを取ってくる")
    d.add_argument("--date", default=None, help="YYYYMMDD (省略時は最新)")
    d.add_argument("--dir", default=os.path.expanduser("~/Library/Caches/Tsugumi"))

    b = sub.add_parser("build", help="SQLite を組む")
    b.add_argument("--out", required=True)
    b.add_argument("--body-chars", type=int, default=1000,
                   help="導入部の後、本文の頭を何文字まで索引に入れるか (0 で導入部のみ)")
    b.add_argument("--limit", type=int, default=0, help="分割ファイルごとの記事数の上限 (試し用)")
    b.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    b.add_argument("--dump-date", default=None)
    b.add_argument("shards", nargs="+")

    s = sub.add_parser("search")
    s.add_argument("db")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)

    p = sub.add_parser("page")
    p.add_argument("db")
    p.add_argument("title")
    p.add_argument("--chars", type=int, default=2000)

    t = sub.add_parser("tokenize")
    t.add_argument("text")

    args = parser.parse_args()
    if args.command == "download":
        download(args.date, args.dir)
    elif args.command == "build":
        date = args.dump_date
        if date is None:
            for piece in os.path.basename(args.shards[0]).split("-"):
                if piece.isdigit() and len(piece) == 8:
                    date = piece
        build(args.out, args.shards, args.body_chars, args.limit, args.jobs, date)
    elif args.command == "search":
        for pid, title, snip, incoming in search(args.db, args.query, args.limit):
            print(f"{title}  (id {pid}, links {incoming})\n    {snip}")
    elif args.command == "page":
        found = page(args.db, args.title)
        if found is None:
            raise SystemExit("not found")
        print(found[0])
        print(found[1][:args.chars])
    elif args.command == "tokenize":
        print(" ".join(tokenize(args.text)))


if __name__ == "__main__":
    main()
