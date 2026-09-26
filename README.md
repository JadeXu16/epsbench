# EPSBench

Code and data release for **EPSBench: Benchmarking LLM Agents for Earnings Forecasting**.

EPSBench evaluates language model agents on forecasting GAAP diluted EPS for 149 firm-quarters from 86 US companies. Agents estimate company-specific income-statement drivers from filings, earnings-call transcripts, and news available before each announcement, and a shared accounting calculator converts the drivers into EPS.

## Planned contents

- Company-specific accounting schemas for all 149 instances
- MCP tool server, harness adapters, and prompts for every information condition
- Evaluation scripts and per-instance forecasts, scores, and share prices
- Agent trajectories (transcript text replaced by document identifiers)
- Fetch scripts for public SEC filings and the filtered CC-News index

IBES analyst consensus values are not redistributed.

The full release is in preparation and will be added to this repository.
