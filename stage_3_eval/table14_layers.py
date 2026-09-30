#!/usr/bin/env python3
"""Table 14: forecast error decomposed by accounting stage, on the 125 instances
whose schema has an operating-income subtotal.

From each run's attribution.csv: Operating = op_pde; Non-operating = pretax_pde - op_pde;
Tax and shares = pde - pretax_pde (all on the PDE scale, so they add up to PDE).
Instance value = median over the three runs; table value = median over instances.
Largest component counts, per instance, which of the three is largest in absolute value.

Usage: python stage_3_eval/table14_layers.py
"""
import statistics as st

from common import GRID, MAIN, RUNS, attribution, f


def instance_layers(rows):
    rows = [r for r in rows if r[2] is not None]
    if not rows or not all(r[0] is not None and r[1] is not None for r in rows):
        return None
    return (st.median(r[0] for r in rows), st.median(r[1] - r[0] for r in rows), st.median(r[2] - r[1] for r in rows))


pp = lambda x: f'{x * 100:.3f}'
print('| model | n | Operating (pp) | Non-operating (pp) | Tax and shares (pp) | Largest component O / N / T |')
print('|---|---|---|---|---|---|')
for a, c in MAIN.items():
    runs = {}
    for s in RUNS:
        for r in attribution(c, s):
            k = (r['ticker'], r['fq'])
            if k in GRID:
                runs.setdefault(k, []).append((f(r['op_pde']), f(r['pretax_pde']), f(r['pde'])))
    L = [v for v in (instance_layers(x) for x in runs.values()) if v]
    big = {'O': 0, 'N': 0, 'T': 0}
    for v in L:
        big['ONT'[max(range(3), key=lambda i: abs(v[i]))]] += 1
    print(f"| {a} | {len(L)} | {pp(st.median(v[0] for v in L))} | {pp(st.median(v[1] for v in L))} | "
          f"{pp(st.median(v[2] for v in L))} | {big['O']} / {big['N']} / {big['T']} |")
