#!/usr/bin/env python3
"""Fetch the benchmark's news articles from Common Crawl's CC-NEWS archive.

news_articles.csv (distributed separately, see README) lists every article in the
news index: the day shard it belongs to, its row in that shard, its WARC-Record-ID,
URL, crawl time, published and effective dates, and the CC-NEWS WARC file that
contains it (two candidates, separated by ';', near a file boundary).
This script downloads those WARC files, extracts the listed records with the same
settings used to build the index, and writes one feather file per day shard with
the rows in index order:

    python stage_0_data_collection/fetch_news.py --articles news_articles.csv.gz \
        --out data/news_index [--days 20260101-20260131] [--workers 8] [--keep-warc]

Writes data/news_index/download/<day>/<day>.aggregate.step_7.dedup_filtered.feather;
build_news_index.py then writes data/news_index/embedding/<day>/.

Extraction: trafilatura.bare_extraction(include_comments=False, deduplicate=True,
with_metadata=True, target_language="en"); body = raw_text (else text), summary =
description; then title, summary and body are cleaned (URLs removed, whitespace
collapsed within lines, at most two consecutive blank lines).
Each CC-NEWS WARC is about 1 GB; the full set for the benchmark window is several TB.
"""
import argparse
import csv
import gzip
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.commoncrawl.org/"

URL_PATTERN = re.compile(
    r"""(?ix)
    \b(
        https?://[^\s<>"'()]+
        |
        www\.[^\s<>"'()]+
    )
    """
)


def normalize_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def collapse_inline_spaces(line):
    return re.sub(r"[ \t\f\v]+", " ", line).strip()


def limit_consecutive_blank_lines(text, max_blank_lines=2):
    out, blank_run = [], 0
    for raw in text.split("\n"):
        if raw.strip() == "":
            blank_run += 1
            if blank_run <= max_blank_lines:
                out.append("")
        else:
            blank_run = 0
            out.append(collapse_inline_spaces(raw))
    return "\n".join(out).strip()


def clean(value):
    text = "" if value is None else (value if isinstance(value, str) else str(value))
    text = normalize_newlines(text)
    text = URL_PATTERN.sub("", text)
    return limit_consecutive_blank_lines(text)


def extract(html):
    import trafilatura
    try:
        data = trafilatura.bare_extraction(html, include_comments=False, deduplicate=True,
                                           with_metadata=True, target_language="en")
    except Exception:
        return None
    if not data:
        return None
    if not isinstance(data, dict):
        data = data.as_dict() if hasattr(data, "as_dict") else dict(data.__dict__)
    body = (data.get("raw_text") or data.get("text") or "").strip()
    if not body:
        return None
    return {"title": clean(data.get("title")), "summary": clean(data.get("description")), "text": clean(body),
            "hostname": data.get("hostname")}


def download(path, dest):
    """Resumable download with HTTP range requests. Some networks drop long
    transfers, so keep resuming while data arrives; back off only on no progress."""
    url = BASE + path
    fails, total = 0, None
    while fails < 20:
        have = dest.stat().st_size if dest.exists() else 0
        try:
            with requests.get(url, headers={"Range": f"bytes={have}-"} if have else {}, stream=True, timeout=120) as r:
                if r.status_code == 416:
                    return dest
                r.raise_for_status()
                total = have + int(r.headers.get("Content-Length", 0))
                with open(dest, "ab") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
        except requests.RequestException:
            pass
        now = dest.stat().st_size if dest.exists() else 0
        if total is not None and now >= total:
            return dest
        if now > have:
            fails = 0
        else:
            fails += 1
            time.sleep(min(60, 2 ** fails))
    raise RuntimeError(f"failed to download {url}")


def process_warc(warc_path, wanted, tmp_dir, keep):
    """wanted: {record_id: (day, row)} -> list of (day, row, record_id, url, fields)."""
    from warcio.archiveiterator import ArchiveIterator
    local = download(warc_path, Path(tmp_dir) / Path(warc_path).name)
    out = []
    with open(local, "rb") as fh:
        for rec in ArchiveIterator(fh):
            if rec.rec_type != "response":
                continue
            rid = rec.rec_headers.get_header("WARC-Record-ID")
            if rid not in wanted:
                continue
            mime = rec.http_headers.get_header("Content-Type") if rec.http_headers else None
            if mime and "text/html" not in mime:
                continue
            try:
                html = rec.content_stream().read()
            except Exception:
                continue
            fields = extract(html) if html else None
            day, row = wanted[rid]
            out.append((day, row, rid, rec.rec_headers.get_header("WARC-Target-URI"), fields))
    if not keep:
        local.unlink()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--articles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", default=None, help="YYYYMMDD-YYYYMMDD range of day shards to fetch")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-warc", action="store_true")
    args = ap.parse_args()

    lo, hi = (args.days.split("-") if args.days else ("00000000", "99999999"))
    opener = gzip.open if args.articles.endswith(".gz") else open
    by_warc, rows = defaultdict(dict), defaultdict(dict)
    with opener(args.articles, "rt", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if lo <= r["index_day"] <= hi:
                # a record near the boundary of two WARC files lists both candidates
                for w in r["warc_file"].split(";"):
                    by_warc[w][r["article_id"]] = (r["index_day"], int(r["row"]))
                rows[r["index_day"]][int(r["row"])] = r
    out = Path(args.out); tmp = out / "_warc"; tmp.mkdir(parents=True, exist_ok=True)
    got = defaultdict(dict)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(process_warc, w, ids, str(tmp), args.keep_warc): w for w, ids in by_warc.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            for day, row, rid, url, fields in fut.result():
                got[day][row] = fields
            print(f"[{i}/{len(futs)}] {futs[fut]}", file=sys.stderr)
    for day, want in sorted(rows.items()):
        recs = []
        for row in sorted(want):
            meta, fields = want[row], got[day].get(row) or {}
            recs.append({"id": meta["article_id"], "url": meta["url"], "date": meta["date_published"],
                         "date_crawled": meta["date_crawled"], "effective_date": meta["effective_date"],
                         "title": fields.get("title", ""),
                         "summary": fields.get("summary", ""), "hostname": fields.get("hostname"),
                         "text": fields.get("text", "")})
        missing = sum(1 for r in recs if not r["text"])
        day_dir = out / "download" / day
        day_dir.mkdir(parents=True, exist_ok=True)
        # file name as referenced by the index metadata (and shown to agents as source_file)
        pd.DataFrame(recs).to_feather(day_dir / f"{day}.aggregate.step_7.dedup_filtered.feather")
        print(f"{day}: {len(recs)} articles, {missing} not recovered", file=sys.stderr)

if __name__ == "__main__":
    main()
