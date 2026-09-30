"""make_record.py: turn one agent result into a prediction record.

Every harness writes its records through this file. It reads the agent's final
JSON on stdin (either the driver's {"ok": ..., "result": ...} wrapper or the bare
result), recomputes the income statement from the agent's driver values with the
shared calculator, and attaches the ground truth from data/ground_truth.csv.

If data/consensus.csv exists (columns: ticker, fiscal_quarter, eps_gaap_consensus;
not distributed, see README), the record also gets eps_gps_consensus.

Usage:
    python stage_2_shared/make_record.py --ticker AAPL --fiscal-quarter 2026Q1 \
        --slice-days 1 --out records/AAPL/2026Q1/t-1.record.json [--eps-source reported] < output.json

--eps-source: "recomputed" (OpenCode) takes EPS from the recomputed statement;
"reported" (Claude Code, vanilla) takes the EPS in the agent's calculator summary.
The two agree except when an agent's last calculator call differs from its final
driver values; each harness keeps the rule it ran with.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / "stage_1_schema_building" / "ticker_schemas_v4"
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "stage_2_shared"))
from calculator import calculate_from_prediction  # noqa: E402

OPERATING_NODES = ("operating_income", "operating_earnings", "income_from_operations",
                   "operating_profit", "earnings_from_operations")


def num(v):
    try:
        return float(v) if v not in (None, "", "NA", "N/A") else None
    except (TypeError, ValueError):
        return None


def pretax_node(formulas):
    return next((k for k in formulas if k.startswith("pretax") or "before_tax" in k or "before_income_tax" in k), None)


def operating_node(formulas):
    return next((k for k in formulas if any(x in k for x in OPERATING_NODES)), None)


def pretax_from_summary(key_metrics, detail):
    for src in (key_metrics, detail):
        for k, v in src.items():
            if k.startswith("pretax") or "before_tax" in k or "before_income_tax" in k:
                return num(v)
    return None


def csv_row(path, ticker, fq):
    if not path.exists():
        return {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if r["ticker"] == ticker and r["fiscal_quarter"] == fq:
            return r
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--fiscal-quarter", required=True)
    ap.add_argument("--slice-days", type=int, default=1)
    ap.add_argument("--eps-source", choices=["recomputed", "reported"], default="recomputed")
    ap.add_argument("--agent-model", default=None)
    ap.add_argument("--out", required=True)
    args, _ = ap.parse_known_args()   # older drivers also pass --ect-date / --agent-provider

    out = json.load(sys.stdin)
    if isinstance(out, dict) and "ok" in out and "result" in out:
        if not out["ok"]:
            print(f"[SKIP] agent run failed: {out.get('error')}", file=sys.stderr)
            sys.exit(1)
        out = out["result"]

    ticker, fq = args.ticker.upper(), args.fiscal_quarter
    schema = json.loads((SCHEMA_DIR / f"{ticker}_{fq}.json").read_text(encoding="utf-8"))
    formulas = schema.get("formulas", {})
    calc = out.get("calculated_results_summary", {}) or {}
    key_metrics = calc.get("key_metrics", {}) or {}
    detail = calc.get("detail", {}) or {}
    predictions = out.get("fundamental_predictions", {}) or {}

    try:
        statement = calculate_from_prediction(predictions, schema) if (formulas and predictions) else {}
    except Exception as e:  # malformed driver values: keep the agent's own summary
        print(f"[recompute] failed: {e}", file=sys.stderr)
        statement = {}

    p_node, o_node = pretax_node(formulas), operating_node(formulas)
    pretax_pred = num(statement.get(p_node)) if p_node else None
    if pretax_pred is None:
        pretax_pred = pretax_from_summary(key_metrics, detail)
    operating_pred = num(statement.get(o_node)) if o_node else None

    if args.eps_source == "recomputed":
        eps_pred = num(statement.get("eps"))
        if eps_pred is None:
            eps_pred = num(key_metrics.get("eps"))
        rev_field = schema.get("revenue_field") or "total_revenue"
        revenue_pred = num(statement.get("total_revenue") or statement.get(rev_field) or key_metrics.get("total_revenue"))
    else:
        eps_pred = num(key_metrics.get("eps"))
        revenue_pred = num(key_metrics.get("total_revenue"))

    gt = csv_row(DATA / "ground_truth.csv", ticker, fq)
    interval = out.get("eps_interval") or {}
    rec = {
        "ticker": ticker,
        "fiscal_quarter": fq,
        "slice_days_before": args.slice_days,
        "eps_gaap_pred": eps_pred,
        "revenue_pred": revenue_pred,
        "operating_pred": operating_pred,
        "pretax_pred": pretax_pred,
        # stored as the agent gave them, not reordered: a flipped interval is a compliance failure
        "eps_p10": num(interval.get("p10")),
        "eps_p90": num(interval.get("p90")),
        "eps_gaap_actual": num(gt.get("eps_gaap_diluted")),
        "operating_actual": num(gt.get("operating_income")),
        "pretax_actual": num(gt.get("pretax_income")),
        "fundamental_predictions": predictions,
        "model": args.agent_model,
    }
    cons = num(csv_row(DATA / "consensus.csv", ticker, fq).get("eps_gaap_consensus"))
    if cons is not None:
        rec["eps_gps_consensus"] = cons

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"OK eps_pred={eps_pred} eps_actual={rec['eps_gaap_actual']}", file=sys.stderr)


if __name__ == "__main__":
    main()
