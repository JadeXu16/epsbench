"""news_search_server.py — Long-lived, single-copy GPU news-search service."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

_DEFAULT_CC_NEWS_ROOT = str(Path(__file__).resolve().parent.parent / "data" / "news_index")
INDEX_ROOT = Path(
    os.environ.get("CC_NEWS_INDEX_ROOT")
    or (Path(os.environ.get("CC_NEWS_ROOT", _DEFAULT_CC_NEWS_ROOT)) / "embedding")
)
DTYPE_NAME = os.environ.get("NEWS_SEARCH_DTYPE", "fp16").strip().lower()
DEVICE_ENV = os.environ.get("NEWS_SEARCH_DEVICE", "cuda").strip().lower()
SEARCH_GPU = os.environ.get("SEARCH_GPU", "1").strip()
EMBEDDING_SERVER_URL = os.environ.get("EMBEDDING_SERVER_URL", "").strip().rstrip("/")

_SEARCH_LOCK = threading.Lock()
_STATE: dict = {"ready": False, "error": None, "loading": None}
_MATRIX = None
_DAYS: list = []
_DAY_NAMES: list = []
_TOTAL = 0
_DIM = 0
_DEVICE = None
_TORCH_DTYPE = None


def _log(msg: str) -> None:
    sys.stderr.write(f"[news_search_server] {msg}\n")
    sys.stderr.flush()


def _day_dirs(first_day: str | None, last_day: str | None) -> list:
    """Same predicate mcp_server._load_event_index_locked uses, sorted ascending."""
    if not INDEX_ROOT.exists():
        raise SystemExit(f"index root does not exist: {INDEX_ROOT}")
    dirs = sorted(
        d for d in INDEX_ROOT.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
        and (d / "faiss.index").exists() and (d / "metadata.json").exists()
        and (first_day is None or d.name >= first_day)
        and (last_day is None or d.name <= last_day)
    )
    return dirs


_FLAT_FOURCC = {b"IxF2", b"IxFI", b"IxFl", b"IxFL", b"IxF7"}


def _flat_header(path: Path):
    """(dim, ntotal) from a faiss IndexFlat file header without reading the vectors."""
    try:
        with open(path, "rb") as f:
            hdr = f.read(64)
        fourcc = hdr[:4]
        if fourcc not in _FLAT_FOURCC:
            return None
        d = struct.unpack("<i", hdr[4:8])[0]
        ntotal = struct.unpack("<q", hdr[8:16])[0]
        metric_type = struct.unpack("<i", hdr[33:37])[0]
        off = 37 + (4 if metric_type > 1 else 0)
        count = struct.unpack("<Q", hdr[off:off + 8])[0]
        if d <= 0 or ntotal < 0 or count not in (ntotal * d, ntotal * d * 4):
            return None
        if path.stat().st_size != off + 8 + ntotal * d * 4:
            return None
        return d, ntotal, off + 8
    except Exception:
        return None


def _shard_vectors(idx, n: int, d: int) -> np.ndarray:
    """fp32 [n, d] view/copy of a loaded flat index's vectors."""
    import faiss
    try:
        flat = faiss.downcast_index(idx)
        return faiss.rev_swig_ptr(flat.get_xb(), n * d).reshape(n, d)
    except Exception:
        return idx.reconstruct_n(0, n)


def _pick_device():
    import torch
    if DEVICE_ENV == "cpu":
        _log("NEWS_SEARCH_DEVICE=cpu: TEST MODE, serving from host RAM")
        return torch.device("cpu"), "cpu (test mode)"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available and NEWS_SEARCH_DEVICE != cpu")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd != "":
        dev = torch.device("cuda:0")
        desc = f"cuda:0 = physical GPU {cvd} (CUDA_VISIBLE_DEVICES); GPU 0 is left to the embedding model"
    else:
        dev = torch.device(f"cuda:{SEARCH_GPU}")
        desc = f"cuda:{SEARCH_GPU} (SEARCH_GPU={SEARCH_GPU}; GPU 0 is left to the embedding model)"
    return dev, desc


def _load_corpus(first_day: str | None, last_day: str | None) -> None:
    global _MATRIX, _DAYS, _DAY_NAMES, _TOTAL, _DIM, _DEVICE, _TORCH_DTYPE
    import faiss
    import torch
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    t0 = time.time()
    _DEVICE, dev_desc = _pick_device()
    _TORCH_DTYPE = {"fp16": torch.float16, "fp32": torch.float32}.get(DTYPE_NAME)
    if _TORCH_DTYPE is None:
        raise SystemExit(f"NEWS_SEARCH_DTYPE must be fp16 or fp32, got {DTYPE_NAME!r}")
    np_dtype = np.float16 if _TORCH_DTYPE == torch.float16 else np.float32

    dirs = _day_dirs(first_day, last_day)
    if not dirs:
        raise SystemExit(f"no day shards under {INDEX_ROOT} (range {first_day}..{last_day})")
    _log(f"index root {INDEX_ROOT}: {len(dirs)} day shards {dirs[0].name}..{dirs[-1].name}; device {dev_desc}; dtype {DTYPE_NAME}")

    sizes, offsets = [], []
    dim = None
    for d in dirs:
        hdr = _flat_header(d / "faiss.index")
        if hdr is None:
            idx = faiss.read_index(str(d / "faiss.index"))
            hdr = (idx.d, idx.ntotal)
            del idx
        if dim is None:
            dim = hdr[0]
        elif hdr[0] != dim:
            raise SystemExit(f"dimension mismatch: {d.name} has d={hdr[0]}, expected {dim}")
        sizes.append(hdr[1])
        offsets.append(hdr[2] if len(hdr) > 2 else None)
    total = int(sum(sizes))
    _DIM = int(dim)
    bytes_per = 2 if np_dtype == np.float16 else 4
    _log(f"preallocating [{total}, {_DIM}] {DTYPE_NAME} = {total * _DIM * bytes_per / 1e9:.1f} GB on {_DEVICE}")
    matrix = torch.empty((total, _DIM), dtype=_TORCH_DTYPE, device=_DEVICE)

    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    prefetch = max(1, int(os.environ.get("NEWS_SEARCH_PREFETCH", "16")))

    def _read_shard(i: int):
        d, n, off = dirs[i], sizes[i], offsets[i]
        path = d / "faiss.index"
        if n == 0:
            return None
        if off is not None:
            buf = np.empty((n, _DIM), dtype=np.float32)
            with open(path, "rb") as f:
                f.seek(off)
                got = f.readinto(memoryview(buf).cast("B"))
            if got != buf.nbytes:
                raise RuntimeError(f"{d.name}: short read {got} of {buf.nbytes} bytes")
            out = buf if np_dtype == np.float32 else buf.astype(np_dtype)
            del buf
            return out
        idx = faiss.read_index(str(path))
        if idx.ntotal != n:
            raise RuntimeError(f"{d.name}: header says {n} vectors, index has {idx.ntotal}")
        out = np.ascontiguousarray(_shard_vectors(idx, n, _DIM), dtype=np_dtype)
        del idx
        return out

    days, names, row = [], [], 0
    with ThreadPoolExecutor(max_workers=prefetch, thread_name_prefix="shard-read") as pool:
        pending = deque(pool.submit(_read_shard, i) for i in range(min(prefetch, len(dirs))))
        for i, d in enumerate(dirs):
            _STATE["loading"] = f"{i + 1}/{len(dirs)} {d.name}"
            host = pending.popleft().result()
            if i + prefetch < len(dirs):
                pending.append(pool.submit(_read_shard, i + prefetch))
            if host is None:
                continue
            n = host.shape[0]
            if n != sizes[i]:
                raise SystemExit(f"{d.name}: header says {sizes[i]} vectors, read {n}")
            matrix[row:row + n].copy_(torch.from_numpy(host))
            del host
            days.append({"day": d.name, "row_start": row, "row_end": row + n})
            names.append(d.name)
            row += n
            if (i + 1) % 25 == 0 or i + 1 == len(dirs):
                _log(f"loaded {i + 1}/{len(dirs)} days, {row} vectors, {time.time() - t0:.0f}s")
    if row != total:
        raise SystemExit(f"loaded {row} vectors but the headers announced {total}")
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize(_DEVICE)
    _MATRIX, _DAYS, _DAY_NAMES, _TOTAL = matrix, days, names, row
    gpu_gb = (torch.cuda.memory_allocated(_DEVICE) / 1e9) if _DEVICE.type == "cuda" else 0.0
    _log(f"READY: {len(days)} days, {row} vectors x {_DIM}, {row * _DIM * bytes_per / 1e9:.1f} GB {DTYPE_NAME} "
         f"({gpu_gb:.1f} GB allocated on {_DEVICE}), load time {time.time() - t0:.0f}s")


def _window_rows(start_day: str, cutoff_day: str):
    """[row_start, row_end) covering every loaded day with start_day <= day <= cutoff_day."""
    lo = bisect.bisect_left(_DAY_NAMES, start_day)
    hi = bisect.bisect_right(_DAY_NAMES, cutoff_day)
    if lo >= hi:
        return 0, 0
    return _DAYS[lo]["row_start"], _DAYS[hi - 1]["row_end"]


def _row_to_day_idx(global_row: int):
    pos = bisect.bisect_right([d["row_start"] for d in _DAYS], global_row) - 1
    day = _DAYS[pos]
    return day["day"], global_row - day["row_start"]


def _embed_via_server(query: str) -> np.ndarray:
    import urllib.request
    if not EMBEDDING_SERVER_URL:
        raise RuntimeError("raw 'query' given but EMBEDDING_SERVER_URL is not set")
    req = urllib.request.Request(f"{EMBEDDING_SERVER_URL}/embed", data=json.dumps({"query": query}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=45) as resp:
        return np.asarray(json.load(resp)["embedding"], dtype=np.float32)


def _search(qvec: np.ndarray, start_day: str, cutoff_day: str, n_candidates: int) -> dict:
    import torch
    q = np.asarray(qvec, dtype=np.float32).reshape(-1)
    if q.shape[0] != _DIM:
        raise ValueError(f"query_embedding has dim {q.shape[0]}, index has {_DIM}")
    row_start, row_end = _window_rows(start_day, cutoff_day)
    rows = row_end - row_start
    if rows <= 0 or n_candidates <= 0:
        return {"hits": [], "total_indexed": 0}
    k = min(int(n_candidates), rows)
    with _SEARCH_LOCK:
        block = _MATRIX[row_start:row_end]
        if _DEVICE.type == "cpu":
            scores = torch.mv(block.float(), torch.from_numpy(q))
            top_vals, top_idx = torch.topk(scores, k)
        else:
            qt = torch.from_numpy(q).to(_DEVICE, dtype=_TORCH_DTYPE)
            scores = torch.mv(block, qt)
            if _TORCH_DTYPE == torch.float16:
                kk = min(rows, max(k * 4, k + 64))
                _, cand = torch.topk(scores, kk)
                exact = torch.mv(block[cand].float(), torch.from_numpy(q).to(_DEVICE))
                top_vals, order = torch.topk(exact, k)
                top_idx = cand[order]
            else:
                top_vals, top_idx = torch.topk(scores, k)
        vals = top_vals.float().cpu().tolist()
        idxs = top_idx.cpu().tolist()
    hits = []
    for v, r in zip(vals, idxs):
        day, local = _row_to_day_idx(row_start + int(r))
        hits.append({"score": float(v), "day": day, "idx": int(local)})
    return {"hits": hits, "total_indexed": int(rows)}


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code: int, obj) -> None:
        body = json.dumps(obj).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path == "/health":
            if _STATE["error"]:
                self._reply(500, {"status": "error", "error": _STATE["error"]})
            elif not _STATE["ready"]:
                self._reply(503, {"status": "loading", "progress": _STATE["loading"]})
            else:
                self._reply(200, {"status": "ok", "days": len(_DAYS), "total_vectors": _TOTAL, "dim": _DIM,
                                  "first_day": _DAYS[0]["day"], "last_day": _DAYS[-1]["day"],
                                  "device": str(_DEVICE), "dtype": DTYPE_NAME})
        elif self.path == "/days":
            self._reply(200 if _STATE["ready"] else 503, _DAYS)
        else:
            self._reply(404, {"error": "unknown path"})

    def do_POST(self):
        if self.path != "/search":
            self._reply(404, {"error": "unknown path"})
            return
        if not _STATE["ready"]:
            self._reply(503, {"error": "index still loading", "progress": _STATE["loading"]})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            if payload.get("query_embedding") is not None:
                qvec = np.asarray(payload["query_embedding"], dtype=np.float32)
            else:
                qvec = _embed_via_server(str(payload["query"]))
            start_day = str(payload["start_day"]).replace("-", "")[:8]
            cutoff_day = str(payload["cutoff_day"]).replace("-", "")[:8]
            n_candidates = int(payload.get("n_candidates", 400))
            t0 = time.time()
            out = _search(qvec, start_day, cutoff_day, n_candidates)
            out["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
            self._reply(200, out)
        except Exception as e:
            self._reply(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("NEWS_SEARCH_PORT", "8901")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--first-day", default=None, help="TEST ONLY: load shards >= this YYYYMMDD")
    ap.add_argument("--last-day", default=None, help="TEST ONLY: load shards <= this YYYYMMDD")
    args = ap.parse_args()

    class _Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 128

        def handle_error(self, request, client_address):
            import traceback
            et = sys.exc_info()[0]
            if et not in (BrokenPipeError, ConnectionResetError):
                _log(f"request error from {client_address}: {traceback.format_exc(limit=2).strip().splitlines()[-1]}")
    server = _Server((args.host, args.port), Handler)
    _log(f"bound http://{args.host}:{args.port}; /health answers 503 until the corpus is loaded")
    threading.Thread(target=server.serve_forever, daemon=True, name="http").start()
    try:
        _load_corpus(args.first_day, args.last_day)
        _STATE["ready"] = True
    except BaseException as e:
        _STATE["error"] = f"{type(e).__name__}: {e}"
        _log(f"LOAD FAILED: {_STATE['error']}")
        raise
    _log(f"serving on http://{args.host}:{args.port}")
    threading.Event().wait()


if __name__ == "__main__":
    main()
