"""embedding_server.py — Long-lived, single-copy embedding service."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-8B"
)

_model = None
_ENCODE_LOCK = threading.Lock()


def _load_model():
    global _model
    from sentence_transformers import SentenceTransformer
    sys.stderr.write(f"[embedding_server] loading {MODEL_NAME} ...\n")
    _model = SentenceTransformer(MODEL_NAME, device="cuda", trust_remote_code=True)
    sys.stderr.write("[embedding_server] model ready\n")


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            if _model is None:
                self._reply(503, {"status": "loading"})
            else:
                self._reply(200, {"status": "ok"})
        else:
            self._reply(404, {"error": "unknown path"})

    def do_POST(self):
        if self.path != "/embed":
            self._reply(404, {"error": "unknown path"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            query = payload["query"]
            with _ENCODE_LOCK:
                emb = _model.encode([query], normalize_embeddings=True, prompt_name=None)
            self._reply(200, {"embedding": emb.tolist()})
        except Exception as e:
            self._reply(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    class _Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 128
    server = _Server((args.host, args.port), Handler)
    _load_model()
    sys.stderr.write(f"[embedding_server] serving on http://{args.host}:{args.port}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
