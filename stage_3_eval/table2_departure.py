#!/usr/bin/env python3
"""Table 2: forecast performance by departure from year-ago EPS.

Near and far are the 49 and 50 instances with the smallest and largest
|reported EPS - year-ago EPS| / price. Median PDE is aggregated as in Table 1
(median over runs, then over instances).
  Numeric evidence : share of far-instance runs whose most-adjusted driver has an
                     evidence line citing a specific figure (run_stats.csv; the
                     most-adjusted driver is the one whose reset to its year-ago
                     value changes EPS the most)
  Error halved     : far instances whose median run error is at most half the
                     year-ago EPS error
  Wrong direction  : far instances where at least two of three runs move away
                     from year-ago EPS opposite to the realized change

Usage: python stage_3_eval/table2_departure.py
"""
import math
import statistics as st

from common import CONSENSUS, GRID, GT, MAIN, PR, RUNS, f, load_records, run_stats

ACT = {k: GT[k]['eps_gaap_diluted'] for k in GRID}
YA = {k: GT[k]['eps_year_ago'] for k in GRID}
RECS = {a: {} for a in MAIN}
for a, c in MAIN.items():
    for s in RUNS:
        for k, r in load_records(c, s).items():
            RECS[a].setdefault(k, []).append(f(r['eps_gaap_pred']))

order = sorted(GRID, key=lambda k: abs(ACT[k] - YA[k]) / PR[k])
NEAR, FAR = order[:49], order[-50:]


def run_pde(x, k):
    return abs(x - ACT[k]) / PR[k] if x is not None else math.inf


def numeric_evidence(config):
    rs = [r for (t, q, _), r in run_stats(config).items() if (t, q) in set(FAR) and r['top_driver']]
    return sum(r['top_driver_cites_number'] == '1' for r in rs) / len(rs)


pp = lambda x: f'{x * 100:.3f}'
print('| forecaster | near | far | numeric evidence | error halved | wrong direction |')
print('|---|---|---|---|---|---|')
print(f"| Year-ago EPS | {pp(st.median(run_pde(YA[k], k) for k in NEAR))} | {pp(st.median(run_pde(YA[k], k) for k in FAR))} | -- | -- | -- |")
if CONSENSUS:
    print(f"| Analyst consensus | {pp(st.median(run_pde(CONSENSUS[k], k) for k in NEAR))} | {pp(st.median(run_pde(CONSENSUS[k], k) for k in FAR))} | -- | -- | -- |")
else:
    print('| Analyst consensus | n/a (needs data/consensus.csv) | n/a | -- | -- | -- |')
for a, c in MAIN.items():
    inst = lambda k: st.median(run_pde(x, k) for x in RECS[a][k])
    halved = sum(1 for k in FAR if inst(k) <= 0.5 * abs(YA[k] - ACT[k]) / PR[k])
    wrong = sum(1 for k in FAR
                if st.median([(x - YA[k]) * (ACT[k] - YA[k]) if x is not None else 0 for x in RECS[a][k]]) < 0)
    print(f"| {a} | {pp(st.median(inst(k) for k in NEAR))} | {pp(st.median(inst(k) for k in FAR))} | {numeric_evidence(c):.0%} | {halved} | {wrong} |")
