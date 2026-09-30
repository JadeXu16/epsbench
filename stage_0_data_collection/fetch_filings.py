#!/usr/bin/env python3
"""Download the 10-K and 10-Q filings in data/filings_index.csv from SEC EDGAR.

Each row names a company, its SEC registrant (CIK), form type and filing date; the
script finds that filing through EDGAR's submissions API and saves its primary HTML document as
    <out>/<TICKER>/<folder>/<primary document>

    SEC_USER_AGENT="Your Name your@email" python stage_0_data_collection/fetch_filings.py --out data/filings_raw

SEC asks automated clients to identify themselves in the User-Agent header and to
stay under 10 requests per second. The benchmark's tools read these filings after
conversion to section-level Markdown; that converted corpus is distributed
separately (see README), and is the version the agents actually read.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
UA = os.environ.get("SEC_USER_AGENT")


def get(url, session):
    for attempt in range(6):
        r = session.get(url, timeout=60)
        if r.status_code == 200:
            time.sleep(0.15)
            return r
        time.sleep(2 ** attempt)
    r.raise_for_status()


def filings_of(cik, session):
    """All filings for a CIK: form, filingDate, accessionNumber, primaryDocument."""
    base = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", session).json()
    blocks = [base["filings"]["recent"]]
    for extra in base["filings"].get("files", []):
        blocks.append(get(f"https://data.sec.gov/submissions/{extra['name']}", session).json())
    out = []
    for b in blocks:
        for form, date, acc, doc in zip(b["form"], b["filingDate"], b["accessionNumber"], b["primaryDocument"]):
            out.append((form, date, acc, doc))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(ROOT / "data" / "filings_index.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "filings_raw"))
    args = ap.parse_args()
    if not UA:
        sys.exit("set SEC_USER_AGENT to 'Your Name your@email' (required by SEC)")
    s = requests.Session(); s.headers["User-Agent"] = UA
    rows = list(csv.DictReader(open(args.index, newline="")))
    cache = {}
    for r in rows:
        t, cik = r["ticker"], int(r["cik"])   # registrant at filing time (XOM and BLK changed CIK)
        if cik not in cache:
            cache[cik] = filings_of(cik, s)
        hit = next(((acc, doc) for form, date, acc, doc in cache[cik] if form == r["form"] and date == r["filing_date"]), None)
        if hit is None:
            print(f"not found: {t} {r['form']} {r['filing_date']}", file=sys.stderr); continue
        acc, doc = hit
        dest = Path(args.out) / t / r["folder"] / doc
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
        dest.write_bytes(get(url, s).content)
        print(f"{t} {r['form']} {r['filing_date']} -> {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
