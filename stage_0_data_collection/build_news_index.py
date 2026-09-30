#!/usr/bin/env python3
"""Build the per-day news search index the tool server reads.

Input : data/news_index/download/<day>/<day>.aggregate.step_7.dedup_filtered.feather
        (written by fetch_news.py, rows in index order)
Output: data/news_index/embedding/<day>/
            faiss.index      inner-product index over L2-normalized embeddings
            metadata.json    titles, summaries, article_ids, urls, source_files (row-aligned)
            eff_dates.json   effective date per row (used to enforce the information cutoff)
            texts.parquet    title, summary, text, text_truncated (8,000 cl100k tokens), article_id, url

    python stage_0_data_collection/build_news_index.py --root data/news_index [--days 20260101-20260131]

Embedding model: Qwen/Qwen3-Embedding-8B served with vLLM (pooling runner), documents
truncated to 31,999 tokens; each document is embedded as
    title \n\n - summary \n summary \n\n - content \n body
The tool server embeds queries with the same model through sentence-transformers.
Needs a GPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAX_TOKENS = 32000
SCRAPE_TOKEN_BUDGET = 8000


def column(df, col):
    if col not in df.columns:
        return [""] * len(df)
    s = df[col].fillna("").astype(str).str.strip()
    return s.where(s.str.lower() != "none", "").tolist()


def doc_text(title, summary, content):
    sec = []
    if summary:
        sec.extend(["- summary", summary, ""])
    if content:
        sec.extend(["- content", content])
    sections = "\n".join(sec).strip()
    parts = [p for p in (title, sections) if p]
    return "\n\n".join(parts).strip()


def truncate_for_reading(texts, budget=SCRAPE_TOKEN_BUDGET):
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4")
    toks = enc.encode_batch([t or "" for t in texts], allowed_special="all")
    return [("" if not t else (enc.decode(k[:budget]) + "..." if len(k) > budget else t)) for t, k in zip(texts, toks)]


def embed(llm, texts, batch=256):
    tok = llm.get_tokenizer(); vecs = []
    for i in range(0, len(texts), batch):
        prompts = []
        for t in texts[i:i + batch]:
            ids = tok.encode(t, add_special_tokens=True)
            if len(ids) > MAX_TOKENS - 1:
                ids = tok.encode(t, truncation=True, max_length=MAX_TOKENS - 1, add_special_tokens=True)
            prompts.append({"prompt_token_ids": ids})
        v = np.array([o.outputs.embedding for o in llm.embed(prompts)], dtype=np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        vecs.append(v / np.where(n == 0, 1.0, n))
    return np.concatenate(vecs, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/news_index")
    ap.add_argument("--days", default=None, help="YYYYMMDD-YYYYMMDD")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    args = ap.parse_args()

    import faiss
    from vllm import LLM
    llm = LLM(model=args.model, runner="pooling", trust_remote_code=True, max_model_len=MAX_TOKENS,
              max_num_batched_tokens=131072)
    lo, hi = args.days.split("-") if args.days else ("00000000", "99999999")
    root = Path(args.root)
    for day_dir in sorted((root / "download").iterdir()):
        day = day_dir.name
        if not (lo <= day <= hi):
            continue
        src = day_dir / f"{day}.aggregate.step_7.dedup_filtered.feather"
        df = pd.read_feather(src)
        titles, summaries, contents = column(df, "title"), column(df, "summary"), column(df, "text")
        docs = [doc_text(t, s, c) for t, s, c in zip(titles, summaries, contents)]
        keep = [i for i, d in enumerate(docs) if d]
        vec = embed(llm, [docs[i] for i in keep])
        out = root / "embedding" / day; out.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(vec.shape[1]); index.add(vec)
        faiss.write_index(index, str(out / "faiss.index"))
        pick = lambda xs: [xs[i] for i in keep]
        ids, urls = pick(column(df, "id")), pick(column(df, "url"))
        meta = {"date": day, "n_docs": len(keep), "dim": int(vec.shape[1]), "embedding_model": args.model,
                "scrape_truncation": {"column": "text_truncated", "tokenizer": "cl100k_base", "token_budget": SCRAPE_TOKEN_BUDGET},
                "titles": pick(titles), "summaries": pick(summaries), "article_ids": ids, "urls": urls,
                "source_files": [src.name] * len(keep)}
        (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "eff_dates.json").write_text(json.dumps(pick(column(df, "effective_date"))), encoding="utf-8")
        body = pick(contents)
        pd.DataFrame({"title": pick(titles), "summary": pick(summaries), "text": body,
                      "text_truncated": truncate_for_reading(body), "article_id": ids, "url": urls}
                     ).to_parquet(out / "texts.parquet", compression="snappy", index=False)
        print(f"{day}: {len(keep)} documents")


if __name__ == "__main__":
    main()
