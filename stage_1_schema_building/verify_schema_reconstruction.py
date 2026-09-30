#!/usr/bin/env python3
"""Schema verification (paper Section 3.2 and Appendix B.1).

Executes each schema's formula DAG on its year-ago (q-4) leaf values and compares
the computed EPS with the EPS reported for that quarter. Discrepancies are in
dollars, since EPS is reported to the cent. Writes schema_verification_149.csv.

Usage: python stage_1_schema_building/verify_schema_reconstruction.py
"""
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage_2_shared"))
from calculator import calculate  # noqa: E402

rows = []
for p in sorted((HERE / "ticker_schemas_v4").glob("*.json")):
    s = json.load(open(p))
    reported = float(s["_validation"]["baseline_fq_eps"])
    computed = float(calculate(s["formulas"], {k: float(v) for k, v in s["variables_baseline"].items()})["eps"])
    rows.append((s["ticker"], s["target_fq"], reported, computed, abs(computed - reported)))

errs = [r[4] for r in rows]; n = len(rows)
print(f"{n} schemas")
for th in (0.005, 0.01, 0.02, 0.04, 0.05):
    print(f"|computed - reported| <= ${th:.3f}: {sum(1 for d in errs if d <= th + 1e-9)}/{n}")
print(f"median ${statistics.median(errs):.4f}, max ${max(errs):.4f}")
for t, q, rep, comp, d in sorted(rows, key=lambda r: -r[4]):
    if d > 0.01 + 1e-9:
        print(f"  {t} {q}: reported {rep:.3f}, computed {comp:.4f}, difference {d:.4f}")

with open(HERE / "schema_verification_149.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["ticker", "fiscal_quarter", "reported_q4_eps", "computed_q4_eps", "abs_diff"])
    for t, q, rep, comp, d in rows:
        w.writerow([t, q, rep, round(comp, 4), round(d, 4)])
