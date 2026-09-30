"""MCP server exposing the six benchmark tools (filings, transcripts, news search and reading, calculator)."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_2_shared"))
from calculator import calculate_from_prediction

try:
    import faiss as _faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

from mcp.server.fastmcp import FastMCP


import re as _re

TICKER = os.environ.get("TICKER", "").strip().upper()
FISCAL_QUARTER = os.environ.get("FISCAL_QUARTER", "").strip()
CUTOFF_DATE = os.environ.get("CUTOFF_DATE", "").strip()
START_DATE = os.environ.get("START_DATE", "").strip() or None
NEWS_ABLATION = os.environ.get("NEWS_ABLATION", "").strip() == "1"
FILINGS_ABLATION = os.environ.get("FILINGS_ABLATION", "").strip() == "1"

ALLOW_NO_RESEARCH = os.environ.get("ALLOW_NO_RESEARCH", "").strip() == "1"
if NEWS_ABLATION and FILINGS_ABLATION and not ALLOW_NO_RESEARCH:
    raise SystemExit("NEWS_ABLATION and FILINGS_ABLATION are both set, leaving no research "
                     "channel at all. If that is the no-research configuration you intend, also set "
                     "ALLOW_NO_RESEARCH=1 to confirm it's not an accidental double-set.")

_DATE_RE = _re.compile(r"(\d{4}-\d{2}-\d{2})")


def _publication_date(name: str) -> Optional[str]:
    """Extract the publication date embedded in a transcript or filing name."""
    m = _DATE_RE.search(name)
    return m.group(1) if m else None


def _after_cutoff(name: str) -> bool:
    """True if this file or folder was published after the run's information cutoff."""
    if not CUTOFF_DATE:
        return False
    pub = _publication_date(name)
    return pub is not None and pub > CUTOFF_DATE

NEWS_REPORT_ROOT = Path(__file__).resolve().parent.parent / "stage_2_shared" / "filings"
SCHEMA_DIR = Path(__file__).resolve().parent.parent / "stage_1_schema_building" / "ticker_schemas_v4"
_CC_NEWS_ROOT = Path(os.environ.get("CC_NEWS_ROOT", str(Path(__file__).resolve().parent.parent / "data" / "news_index")))
CC_NEWS_INDEX_ROOT = _CC_NEWS_ROOT / "embedding"
CC_NEWS_DATA_ROOT = _CC_NEWS_ROOT / "download"
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-8B"
)
DEFAULT_SEARCH_TOP_K = 10

_index_cache: dict = {}
_index_cache_lock = threading.Lock()
_day_meta_cache: dict = {}
_day_meta_lock = threading.Lock()
_eff_dates_cache: dict = {}
_eff_dates_lock = threading.Lock()
_embed_model = None
_schema_cache: dict = {}


def _day_meta(day_dir) -> dict:
    """Lazily parse and cache one day shard's metadata.json."""
    k = str(day_dir)
    m = _day_meta_cache.get(k)
    if m is not None:
        return m
    with _day_meta_lock:
        m = _day_meta_cache.get(k)
        if m is None:
            try:
                m = json.loads((day_dir / "metadata.json").read_text(encoding="utf-8"))
            except Exception:
                m = {}
            _day_meta_cache[k] = m
        return m

def _eff_dates(day_dir) -> list:
    """Lazily parse (and cache) one day-shard's eff_dates.json sidecar."""
    k = str(day_dir)
    v = _eff_dates_cache.get(k)
    if v is not None:
        return v
    with _eff_dates_lock:
        v = _eff_dates_cache.get(k)
        if v is None:
            try:
                v = json.loads((day_dir / "eff_dates.json").read_text(encoding="utf-8"))
            except Exception:
                v = []
            _eff_dates_cache[k] = v
        return v


def _after_cutoff_idx(day_dir, idx: int) -> bool:
    """True when the article at `idx` in this shard must be withheld."""
    if not CUTOFF_DATE:
        return False
    eff = _eff_dates(day_dir)
    if idx >= len(eff):
        return True
    d = eff[idx]
    return (not d) or (d > CUTOFF_DATE)


mcp = FastMCP("financial-tools")


def _news_tool(fn):
    """Register a news tool unless NEWS_ABLATION=1."""
    return fn if NEWS_ABLATION else mcp.tool()(fn)


def _filing_tool(fn):
    """Register a filing or transcript tool unless FILINGS_ABLATION=1."""
    return fn if FILINGS_ABLATION else mcp.tool()(fn)


def _load_schema() -> dict:
    key = f"{TICKER}_{FISCAL_QUARTER}"
    if key in _schema_cache:
        return _schema_cache[key]
    schema_path = SCHEMA_DIR / f"{key}.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    _schema_cache[key] = schema
    return schema


_embed_model_ready = threading.Event()
_embed_model_error: Optional[BaseException] = None


def _get_embed_model():
    _embed_model_ready.wait()
    if _embed_model_error is not None:
        raise RuntimeError(f"Embedding model failed to load: {_embed_model_error}")
    return _embed_model


def _preload_embed_model() -> None:
    """Load the query embedding model on a background thread at startup."""
    global _embed_model, _embed_model_error
    try:
        from sentence_transformers import SentenceTransformer
        sys.stderr.write(f"Preloading embedding model: {EMBEDDING_MODEL_NAME} ...\n")
        sys.stderr.flush()
        _embed_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, device="cuda", trust_remote_code=True
        )
        sys.stderr.write("Embedding model loaded.\n")
        sys.stderr.flush()
    except BaseException as e:
        _embed_model_error = e
        sys.stderr.write(f"Embedding model FAILED to load: {e!r}\n")
        sys.stderr.flush()
    finally:
        _embed_model_ready.set()


_embed_model_lock = threading.Lock()

EMBEDDING_SERVER_URL = os.environ.get("EMBEDDING_SERVER_URL", "").strip().rstrip("/")

NEWS_SEARCH_SERVER_URL = os.environ.get("NEWS_SEARCH_SERVER_URL", "").strip().rstrip("/")


def _embed_via_server(query: str) -> np.ndarray:
    import time
    import urllib.request
    req = urllib.request.Request(
        f"{EMBEDDING_SERVER_URL}/embed",
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_err = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                payload = json.loads(resp.read())
            return np.asarray(payload["embedding"], dtype=np.float32)
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"embedding server unreachable after 5 tries: {last_err}")


def _embed_query(query: str) -> np.ndarray:
    if EMBEDDING_SERVER_URL:
        return _embed_via_server(query)
    model = _get_embed_model()
    with _embed_model_lock:
        return model.encode([query], normalize_embeddings=True, prompt_name=None).astype(np.float32)


def _search_via_server(qvec: np.ndarray, start_str: str, cutoff_str: str, n_candidates: int):
    """Send the query embedding to news_search_server.py and return its ranked hits."""
    import time
    import urllib.request
    body = json.dumps({
        "query_embedding": np.asarray(qvec, dtype=np.float32).reshape(1, -1).tolist(),
        "start_day": start_str, "cutoff_day": cutoff_str, "n_candidates": int(n_candidates),
    }).encode("utf-8")
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"{NEWS_SEARCH_SERVER_URL}/search", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read())
            return payload["hits"], int(payload.get("total_indexed", 0))
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"news search server unreachable after 3 tries: {last_err}")


def _remote_index_handle(cutoff_str: str, start_str: str):
    """Index handle used when search is delegated to news_search_server.py."""
    if not CC_NEWS_INDEX_ROOT.exists():
        return None
    day_dirs = sorted(
        d for d in CC_NEWS_INDEX_ROOT.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
        and start_str <= d.name <= cutoff_str
        and (d / "faiss.index").exists() and (d / "metadata.json").exists()
    )
    if not day_dirs:
        return None
    _warn_missing_sidecars(day_dirs)
    return {"remote": True, "start_str": start_str, "cutoff_str": cutoff_str,
            "days": [{"day": d.name, "day_dir": d} for d in day_dirs]}


def _warn_missing_sidecars(day_dirs) -> None:
    """Loud stderr warning when an in-window shard has no eff_dates.json."""
    missing = [d.name for d in day_dirs if not (d / "eff_dates.json").exists()]
    if missing:
        sys.stderr.write(
            f"[mcp] FATAL-ish: {len(missing)} in-window news shards have no eff_dates.json "
            f"(e.g. {', '.join(missing[:5])}). Every article in them will be WITHHELD. "
            f"Run stage_2_harness_opencode/scripts/build_effective_dates.py before this batch.\n")
        sys.stderr.flush()


def _load_event_index(cutoff_date: str, start_date: Optional[str] = None, force_local: bool = False):
    """Thread-safe cached loader of the day shards inside the news window."""
    if not _FAISS_AVAILABLE:
        return None

    from datetime import datetime, timedelta

    cutoff_str = cutoff_date.replace("-", "")[:8]
    if start_date:
        start_str = start_date.replace("-", "")[:8]
    else:
        try:
            cutoff_dt = datetime.strptime(cutoff_str, "%Y%m%d")
            start_str = (cutoff_dt - timedelta(days=90)).strftime("%Y%m%d")
        except ValueError:
            return None

    key = (cutoff_str, start_str)
    if NEWS_SEARCH_SERVER_URL and not force_local:
        rkey = ("remote", cutoff_str, start_str)
        handle = _index_cache.get(rkey)
        if handle is None:
            with _index_cache_lock:
                handle = _index_cache.get(rkey) or _remote_index_handle(cutoff_str, start_str)
                if handle is not None:
                    _index_cache[rkey] = handle
        return handle
    if key in _index_cache:
        return _index_cache[key]

    with _index_cache_lock:
        if key in _index_cache:
            return _index_cache[key]
        return _load_event_index_locked(key, cutoff_str, start_str)


def _load_event_index_locked(key: tuple, cutoff_str: str, start_str: str):
    """Load each day's FAISS index separately (memory-mapped)."""
    if not CC_NEWS_INDEX_ROOT.exists():
        return None

    day_dirs = sorted(
        d for d in CC_NEWS_INDEX_ROOT.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
        and start_str <= d.name <= cutoff_str
        and (d / "faiss.index").exists() and (d / "metadata.json").exists()
    )
    if not day_dirs:
        return None
    _warn_missing_sidecars(day_dirs)

    days = []
    for day_dir in day_dirs:
        idx = _faiss.read_index(str(day_dir / "faiss.index"), _faiss.IO_FLAG_MMAP)
        if idx.ntotal == 0:
            continue
        days.append({"index": idx, "day": day_dir.name, "day_dir": day_dir})

    if not days:
        return None

    entry = {"days": days, "start_str": start_str, "cutoff_str": cutoff_str}
    _index_cache[key] = entry
    return entry


def _preload_news_index() -> None:
    """Warm the day-shard cache in the background at startup."""
    try:
        _load_event_index(cutoff_date=CUTOFF_DATE, start_date=START_DATE)
    except Exception as e:
        sys.stderr.write(f"[mcp] news-index warm failed (non-fatal): {e}\n")


@_filing_tool
def read_earnings_call_transcript(quarter: str, offset: int = 0) -> dict:
    """Read the earnings call transcript for the given quarter (e.g. 'Q2_2025').
    Transcripts are returned in chunks of ~15000 chars. If the response has
    "has_more": true, call this tool again with offset=<next_offset> to read
    the next chunk — do not assume you have seen the full transcript until
    has_more is false."""
    transcripts_dir = NEWS_REPORT_ROOT / TICKER / "earnings_call_transcript"
    if not transcripts_dir.exists():
        return {"error": f"Transcript directory not found: {transcripts_dir}"}

    target = f"_{quarter}_"
    matches = [f for f in transcripts_dir.glob("*.txt") if target in f.name]
    if not matches:
        return {"error": f"No transcript found for quarter '{quarter}'"}
    if _after_cutoff(matches[0].name):
        return {"error": f"Transcript for '{quarter}' was published after the "
                         f"information cutoff ({CUTOFF_DATE}) and cannot be read — "
                         f"it is future data relative to this forecast."}

    full_content = matches[0].read_text(encoding="utf-8", errors="ignore").strip()
    total_length = len(full_content)
    offset = max(0, int(offset))
    chunk = full_content[offset:offset + _SECTION_CHUNK_SIZE]
    next_offset = offset + len(chunk)
    has_more = next_offset < total_length

    return {
        "quarter": quarter,
        "content": chunk,
        "offset": offset,
        "total_length": total_length,
        "has_more": has_more,
        "next_offset": next_offset if has_more else None,
    }


@_filing_tool
def list_financial_report_sections(quarter: str) -> dict:
    """List available financial report sections (.md filenames) for the given quarter."""
    reports_root = NEWS_REPORT_ROOT / TICKER / "financial_report"
    if not reports_root.exists():
        return {"error": f"Report root not found: {reports_root}"}

    suffix = f"_{quarter}"
    folder = next((e for e in reports_root.iterdir() if e.is_dir() and e.name.endswith(suffix)), None)
    if folder is None:
        return {"error": f"No report folder found for quarter '{quarter}' under {reports_root}"}
    if _after_cutoff(folder.name):
        return {"error": f"Report for '{quarter}' was filed after the information "
                         f"cutoff ({CUTOFF_DATE}) and cannot be read — future data."}

    sections = [
        {"section": f.stem, "size_kb": round(f.stat().st_size / 1024, 1)}
        for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() == ".md"
    ]
    return {"folder": folder.name, "quarter": quarter, "count": len(sections), "sections": sections}


_SECTION_CHUNK_SIZE = 15000


@_filing_tool
def read_financial_report_section(quarter: str, section_name: str, offset: int = 0) -> dict:
    """Read the content of one financial report section (from list_financial_report_sections).
    Large sections (e.g. raw Financial_Statements tables) are returned in chunks of
    ~15000 chars. If the response has "has_more": true, call this tool again with
    offset=<next_offset> to continue reading the rest of the same section — do not
    assume you have seen the whole section just because the call succeeded."""
    reports_root = NEWS_REPORT_ROOT / TICKER / "financial_report"
    if not reports_root.exists():
        return {"error": f"Report root not found: {reports_root}"}

    suffix = f"_{quarter}"
    folder = next((e for e in reports_root.iterdir() if e.is_dir() and e.name.endswith(suffix)), None)
    if folder is None:
        return {"error": f"No report folder found for quarter '{quarter}'"}
    if _after_cutoff(folder.name):
        return {"error": f"Report for '{quarter}' was filed after the information "
                         f"cutoff ({CUTOFF_DATE}) and cannot be read — future data."}

    file_path = folder / f"{section_name}.md"
    if not file_path.exists():
        return {"error": f"Section not found: {section_name}"}

    full_content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    total_length = len(full_content)
    offset = max(0, int(offset))
    chunk = full_content[offset:offset + _SECTION_CHUNK_SIZE]
    next_offset = offset + len(chunk)
    has_more = next_offset < total_length

    return {
        "folder": folder.name,
        "section": section_name,
        "content": chunk,
        "offset": offset,
        "total_length": total_length,
        "has_more": has_more,
        "next_offset": next_offset if has_more else None,
    }


@_news_tool
def financial_search(query: str, top_k: int = DEFAULT_SEARCH_TOP_K) -> dict:
    """Semantically search recent news. Results are restricted server-side to the
    current run's information window — there is no way to pass a different
    cutoff_date; it is fixed by the environment this process was started with."""
    if not _FAISS_AVAILABLE:
        return {"error": "faiss not installed. Run: pip install faiss-cpu"}

    entry = _load_event_index(cutoff_date=CUTOFF_DATE, start_date=START_DATE)
    if entry is None:
        return {"error": f"No FAISS index found for date range [{START_DATE}, {CUTOFF_DATE}]."}

    remote = bool(entry.get("remote"))
    start_str = entry.get("start_str", "")
    cutoff_str = entry.get("cutoff_str", "")
    days = None if remote else entry["days"]
    if not remote:
        total_indexed = sum(d["index"].ntotal for d in days)
        if total_indexed == 0:
            return {"query": query, "total_news": 0, "results": []}

    try:
        qvec = _embed_query(query)
    except Exception as e:
        return {"error": f"embedding failed: {type(e).__name__}: {e}", "query": query, "results": []}
    per_day_k_factor = 3

    candidates = []
    if remote:
        try:
            hits, total_indexed = _search_via_server(qvec, start_str, cutoff_str, int(top_k) * per_day_k_factor)
            if total_indexed == 0:
                return {"query": query, "total_news": 0, "results": []}
            for h in hits:
                day = str(h["day"])
                candidates.append({"score": float(h["score"]), "idx": int(h["idx"]),
                                   "day": day, "day_dir": CC_NEWS_INDEX_ROOT / day})
        except Exception as e:
            sys.stderr.write(f"[mcp] news search service failed ({type(e).__name__}: {e}); "
                             f"falling back to the local CPU path for this call\n")
            sys.stderr.flush()
            local = _load_event_index(cutoff_date=CUTOFF_DATE, start_date=START_DATE, force_local=True)
            if local is None:
                return {"error": f"news search service unavailable and no local FAISS index for "
                                 f"[{START_DATE}, {CUTOFF_DATE}]."}
            days = local["days"]
            total_indexed = sum(d["index"].ntotal for d in days)
            if total_indexed == 0:
                return {"query": query, "total_news": 0, "results": []}

    if days is not None:
        for d in days:
            n = d["index"].ntotal
            if n == 0:
                continue
            k = min(int(top_k) * per_day_k_factor, n)
            scores, indices = d["index"].search(qvec, k)
            for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
                if idx < 0:
                    continue
                candidates.append({"score": float(score), "idx": idx,
                                   "day": d["day"], "day_dir": d["day_dir"]})

    candidates.sort(key=lambda c: -c["score"])

    results = []
    n_withheld = 0
    for c in candidates:
        if _after_cutoff_idx(c["day_dir"], c["idx"]):
            n_withheld += 1
            continue
        meta = _day_meta(c["day_dir"])
        titles = meta.get("titles", [])
        idx = c["idx"]
        if idx >= len(titles):
            continue
        summaries = meta.get("summaries", [])
        article_ids = meta.get("article_ids", [])
        urls = meta.get("urls", [])
        source_files = meta.get("source_files", [])
        day = (c["day"] or "").strip()
        eff = _eff_dates(c["day_dir"])
        eff_date = eff[idx] if idx < len(eff) and eff[idx] else (
            f"{day[:4]}-{day[4:6]}-{day[6:8]}" if len(day) == 8 and day.isdigit() else day)

        results.append({
            "rank": len(results) + 1,
            "title": titles[idx],
            "similarity_score": round(c["score"], 6),
            "summary": summaries[idx] if idx < len(summaries) else "",
            "article_id": article_ids[idx] if idx < len(article_ids) else "",
            "url": urls[idx] if idx < len(urls) else "",
            "source_file": source_files[idx] if idx < len(source_files) else "",
            "index_date": c["day"],
            "date": eff_date,
            "date_type": "effective",
        })
        if len(results) >= int(top_k):
            break

    return {"query": query, "total_indexed": total_indexed, "returned": len(results),
            "withheld_after_cutoff": n_withheld, "results": results}


@_news_tool
def read_news_article(article_id: str, max_chars: int = 12000) -> dict:
    """Read the full text of a news article returned by financial_search (by article_id)."""
    article_id = str(article_id or "").strip()
    if not article_id:
        return {"error": "read_news_article requires a non-empty article_id"}

    entry = _load_event_index(cutoff_date=CUTOFF_DATE, start_date=START_DATE)
    if entry is None:
        return {"error": f"No FAISS metadata loaded for [{START_DATE}, {CUTOFF_DATE}]. Call financial_search first."}

    days = entry["days"]
    ordered = ([d for d in days if str(d["day_dir"]) in _day_meta_cache]
               + [d for d in days if str(d["day_dir"]) not in _day_meta_cache])
    source_file = None
    date_dir = None
    eff_date_hit = None
    for d in ordered:
        meta = _day_meta(d["day_dir"])
        article_ids = meta.get("article_ids", [])
        source_files = meta.get("source_files", [])
        match_idx = next(
            (i for i, c in enumerate(article_ids) if str(c).strip() == article_id), None
        )
        if match_idx is not None and match_idx < len(source_files):
            if _after_cutoff_idx(d["day_dir"], match_idx):
                return {"error": f"article_id is outside the information window: {article_id}"}
            source_file = str(source_files[match_idx]).strip()
            date_dir = d["day"]
            eff_date_hit = (_eff_dates(d["day_dir"]) or [None] * (match_idx + 1))[match_idx]
            break

    if source_file is None:
        return {"error": f"article_id not found in current news window: {article_id}"}
    if not source_file or not date_dir:
        return {"error": f"Missing source file/date for article_id: {article_id}"}

    root = CC_NEWS_DATA_ROOT.resolve()
    file_path = (root / date_dir / source_file).resolve()
    try:
        file_path.relative_to(root)
    except ValueError:
        return {"error": f"News source path outside data directory: {file_path}"}
    if not file_path.exists():
        return {"error": f"News source file not found: {file_path}"}

    import pandas as pd
    df = pd.read_feather(file_path)
    if "id" not in df.columns:
        return {"error": f"News source file has no id column: {file_path}"}

    matches = df.index[df["id"].astype(str).str.strip() == article_id].tolist()
    if not matches:
        return {"error": f"article_id not found in source file: {article_id}"}

    row = df.iloc[matches[0]]

    def _clean(value) -> str:
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return str(value).strip()

    max_len = max(1000, min(int(max_chars), 50000))
    text = _clean(row.get("text", ""))
    truncated = len(text) > max_len
    if truncated:
        text = text[:max_len].rstrip()

    folder_date = (
        f"{date_dir[:4]}-{date_dir[4:6]}-{date_dir[6:8]}"
        if date_dir and len(date_dir) == 8 and date_dir.isdigit() else date_dir or ""
    )
    return {
        "article_id": article_id,
        "title": _clean(row.get("title", "")),
        "url": _clean(row.get("url", "")),
        "date": eff_date_hit or folder_date,
        "index_folder_date": folder_date,
        "date_published": _clean(row.get("date", "")),
        "date_crawled": _clean(row.get("date_crawled", "")),
        "hostname": _clean(row.get("hostname", "")),
        "excerpt": _clean(row.get("excerpt", "")),
        "content": text,
        "truncated": truncated,
        "source_file": str(file_path),
    }


def _financial_calculate_impl(prediction: dict) -> dict:
    try:
        schema = _load_schema()
        result = calculate_from_prediction(prediction, schema)

        visible_keys = set(schema.get("formulas", {}).keys()) | set(schema.get("variables_baseline", {}).keys())
        full = {k: round(v, 4) for k, v in result.items() if k in visible_keys}

        revenue_field = schema.get("revenue_field")
        if revenue_field and revenue_field != "total_revenue" and revenue_field in full:
            full["total_revenue"] = full[revenue_field]

        pretax_key = next(
            (k for k in schema.get("formulas", {})
             if k.startswith("pretax") or "before_tax" in k or "before_income_tax" in k),
            None,
        )
        promote = ("eps", "total_revenue") + ((pretax_key,) if pretax_key else ())
        key_metrics = {k: full[k] for k in promote if k in full}
        detail = {k: v for k, v in full.items() if k not in key_metrics}

        return {
            "status": "ok",
            "income_statement": {
                "key_metrics": key_metrics,
                "detail": detail,
            },
        }
    except Exception as e:
        return {"error": str(e)}


def _register_financial_calculate_tool() -> None:
    """Register financial_calculate with one typed parameter per driver in this instance's schema."""
    schema = _load_schema()
    pred_schema = schema.get("prediction_schema", {})
    field_names = list(pred_schema.keys())

    bad = [f for f in field_names if not f.isidentifier()]
    if bad:
        raise ValueError(
            f"prediction_schema field(s) {bad} in {TICKER}_{FISCAL_QUARTER} are not valid "
            "Python identifiers, so the financial_calculate signature cannot be built. "
            "Re-run stage_1_schema_building/patch_scripts/build_v4_schemas.py, which "
            "normalises non-identifier field names."
        )

    def _default_for(f: str) -> float:
        meta = pred_schema[f]
        if meta.get("type") == "rate_override":
            return float(meta.get("default", 0.0))
        return 0.0

    params_src = ", ".join(f"{f}: float = {_default_for(f)!r}" for f in field_names)
    body_src = "    return _financial_calculate_impl({" + ", ".join(f'"{f}": {f}' for f in field_names) + "})"
    func_src = f"def financial_calculate({params_src}) -> dict:\n{body_src}\n"

    def _unit_hint(f: str) -> str:
        t = pred_schema[f].get("type")
        if t == "yoy":
            return "decimal YoY, e.g. 0.05 = +5%"
        if t == "rate_override":
            return f"decimal rate, use guidance if found else default {pred_schema[f].get('default')}"
        return "integer bps delta"

    doc_lines = "\n".join(
        f"    {f}: {pred_schema[f].get('description', '')} ({_unit_hint(f)})"
        for f in field_names
    )
    docstring = (
        "Submit predicted driver values for EVERY field below and get back the fully "
        "computed income statement. This is the ONLY way to obtain eps / total_revenue "
        "— do not compute them yourself.\n\n" + doc_lines
    )

    namespace: dict = {"_financial_calculate_impl": _financial_calculate_impl}
    exec(func_src, namespace)
    fn = namespace["financial_calculate"]
    fn.__doc__ = docstring

    mcp.add_tool(fn, name="financial_calculate", description=docstring)


if __name__ == "__main__":
    if not TICKER or not FISCAL_QUARTER or not CUTOFF_DATE:
        sys.stderr.write(
            "WARNING: TICKER/FISCAL_QUARTER/CUTOFF_DATE env vars not fully set. "
            "Tools will fail until a fresh process is started with these set.\n"
        )
    if _FAISS_AVAILABLE and not EMBEDDING_SERVER_URL and not NEWS_ABLATION:
        threading.Thread(target=_preload_embed_model, daemon=True).start()
    else:
        _embed_model_ready.set()

    if _FAISS_AVAILABLE and CUTOFF_DATE and not NEWS_ABLATION:
        threading.Thread(target=_preload_news_index, daemon=True).start()

    if TICKER and FISCAL_QUARTER:
        _register_financial_calculate_tool()

    mcp.run()
