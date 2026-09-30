# EPSBench

Code and data for **EPSBench: Benchmarking LLM Agents for Earnings Forecasting** (anonymous submission).

EPSBench asks an agent to forecast a company's GAAP diluted EPS for one quarter from
filings, earnings-call transcripts and news available the day before the results are
released. The agent sets the drivers of a company-specific accounting schema, a shared
calculator turns them into EPS, and the agent also gives an 80% prediction interval.
The benchmark has 149 firm-quarters from 86 US companies, announced between
December 2025 and June 2026.

## Contents

| Path | What it holds |
|---|---|
| `data/instances.csv` | The 149 instances: release date, information cutoff (τ, the day before release) and news-window start (τ0, the previous earnings call) |
| `data/ground_truth.csv` | Reported GAAP diluted EPS, pretax income and operating income (USD millions), diluted shares, year-ago and last-quarter EPS |
| `data/prices.csv` | Closing price on the last trading day before release (Yahoo Finance) |
| `data/scores.csv` | One row per configuration × instance × run: forecast, PDE, interval coverage, interval score, and win/loss against analyst consensus |
| `data/transcripts_index.csv` | Earnings-call transcripts the tools serve: document ID, company, quarter, call date, expected file name (no text) |
| `data/filings_index.csv` | 10-K and 10-Q filings the tools serve |
| `stage_0_data_collection/` | `fetch_filings.py` (SEC EDGAR), `fetch_news.py` (CC-NEWS), `build_news_index.py` |
| `stage_1_schema_building/` | The 149 accounting schemas (`ticker_schemas_v4/`), per-instance schema verification (`schema_verification_149.csv`) and the script that produces it |
| `stage_2_shared/` | Accounting calculator, prompt and schema loading, system prompts, record writer |
| `stage_2_harness_opencode/` | MCP tool server, news search and embedding services, OpenCode driver |
| `stage_2_harness_claudecode/` | Claude Code driver (`CLAUDE.md` is its system prompt) |
| `stage_2_harness_manual/` | Vanilla harness: a plain OpenAI SDK tool-calling loop |
| `stage_3_eval/` | Scripts that produce every table in the paper |
| `stage_4_experiments/<config>/<run>/` | Forecast records of all 14 configurations × 3 runs, per-run error decomposition (`attribution.csv`), and per-run trajectory statistics (`<config>/run_stats.csv`) |

Distributed separately (anonymous link: https://osf.io/s4nx7/?view_only=08d038325e02473595349aaecf776dfc):
- `trajectories/`: full agent trajectories for every run, with transcript text replaced by document IDs;
- `filings/`: the 10-K and 10-Q filings converted to section-level Markdown, i.e. the text the tools serve;
- `news_articles.csv.gz`: the 11.8M CC-NEWS articles in the news index, with the fields `fetch_news.py` needs.

Not distributed, because of licensing: analyst consensus (IBES), earnings-call transcript
text (Seeking Alpha), and news article text.

## Reproducing the paper's tables

Requirements: Python 3.10+ with `numpy`, `pandas`. From the repository root:

| Paper | Command |
|---|---|
| Table 1, Figures 2 and 3, Table 10 (interval columns), Table 11, Table 13 | `python stage_3_eval/main_tables.py` |
| Table 2 | `python stage_3_eval/table2_departure.py` |
| Table 3, Table 12 | `python stage_3_eval/ablation.py` |
| Table 4 | `python stage_3_eval/harness_effort.py` |
| Table 10 (signed error) | `python stage_3_eval/signed_error.py` |
| Table 14, Section 5.3 | `python stage_3_eval/table14_layers.py` |
| Schema verification (Section 3.2, Appendix B.1) | `python stage_1_schema_building/verify_schema_reconstruction.py` |

`stage_3_eval/attribution.py` regenerates the `attribution.csv` files from the records.
The `stage_3_eval/output_*.md` files are these scripts' outputs, produced with the consensus file in place.

Columns that compare against analyst consensus (the consensus rows, paired ΔPDE against
consensus, surprise direction) need the IBES GAAP EPS consensus at τ. Without it, those
columns print `n/a` and win rates are read from `data/scores.csv`. With WRDS access, put the consensus in `data/consensus.csv` (columns `ticker, fiscal_quarter, eps_gaap_consensus`) and rerun. We used `ibes.statsum_xepsus`, measure `GPS`, quarterly forecast periods (`fpi` 6 to 11), the target quarter's `fpedats`, and the mean estimate from the latest `statpers` on or before the day before the earnings call; per-share values are restated to the basis of reported EPS (NFLX: ×10 for its November 2025 split).

## Running the benchmark

The environment needs, for every instance, the filings, the transcripts and the news index:

1. **Filings.** Download `filings/` from the link above into `stage_2_shared/filings/<TICKER>/financial_report/`.
   `fetch_filings.py` retrieves the original documents from EDGAR.
2. **Transcripts.** Obtain the transcripts listed in `data/transcripts_index.csv` (we used
   Seeking Alpha under license) and save each as plain text under
   `stage_2_shared/filings/<TICKER>/earnings_call_transcript/<filename>`.
3. **News index.** `python stage_0_data_collection/fetch_news.py --articles news_articles.csv.gz --out data/news_index`,
   then `python stage_0_data_collection/build_news_index.py --root data/news_index` (GPU).
   Pin `trafilatura==1.11.0`. Titles, summaries and sources reproduce exactly. trafilatura
   removes text segments already seen earlier in the same process, so a rebuilt article
   body can differ from ours: on a sampled WARC, 58% of bodies were identical and almost
   all others differed only by a few extra short lines (median 4; median length ratio 0.993).

Then install the drivers (`npm ci` in `stage_2_harness_opencode/driver` and
`stage_2_harness_opencode/.opencode`) and the Python packages (`mcp`, `faiss`,
`sentence-transformers`, `openai`, `pandas`, `pyarrow`). The tool server embeds queries
with `Qwen/Qwen3-Embedding-8B`; `embedding_server.py` and `news_search_server.py` share one
model and one index across parallel workers.

```bash
# OpenCode (default harness); provider: openai | azure | anthropic | qwen | qwen_us | zhipu
cd stage_2_harness_opencode
bash driver/run_parallel.sh driver/targets_149.csv 1 4 my_run gpt-5.5_s1 v4 openai gpt-5.5

# Claude Code
bash stage_2_harness_claudecode/driver/run_parallel.sh stage_2_harness_opencode/driver/targets_149.csv 1 4 runs/claude claude-opus-4-6

# Vanilla (OpenAI SDK loop)
bash stage_2_harness_manual/driver/run_parallel.sh stage_2_harness_opencode/driver/targets_149.csv 1 4 runs/vanilla gpt-5.5
```

Reasoning effort and context limits are set per model with `FINBENCH_MODEL_OPTIONS_JSON`,
`FINBENCH_MODEL_LIMIT_JSON` or `stage_2_harness_opencode/driver/lib/finbench_overrides.json`.
The information conditions of Table 3 are selected with `NEWS_ABLATION=1`,
`FILINGS_ABLATION=1`, or both together with `ALLOW_NO_RESEARCH=1`.
Every harness writes records through `stage_2_shared/make_record.py`. To score a new
configuration with the table scripts, place its records under
`stage_4_experiments/<config>/<run>/records/` and add it to the script's configuration list.

Versions used in the paper: OpenCode SDK 1.17.9, Claude Code 2.1.269, Node 22, Python 3.13,
trafilatura 1.11.0, faiss 1.14, vLLM (index build), sentence-transformers 5.5.

## Record format

Each `t-1.record.json` holds the forecast EPS (`eps_gaap_pred`), the 80% interval
(`eps_p10`, `eps_p90`), the agent's driver values (`fundamental_predictions`), the
recomputed revenue, operating and pretax income, and the reported values they are scored
against. PDE = |forecast EPS − reported EPS| / price.

## License

Code: MIT (`LICENSE`). Data we produced (schemas, forecasts, scores, indexes): CC BY 4.0
(`DATA_LICENSE`). Third-party content (SEC filings are public; transcripts, news text and
IBES data are not included) remains under its owners' terms.
