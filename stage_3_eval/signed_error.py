#!/usr/bin/env python3
"""Signed error column of Table 10: median of (reported EPS - forecast) / price over
all runs; positive means the forecast was too low. The 90% interval resamples
instances (all runs of an instance together), 10,000 draws.

Usage: python stage_3_eval/signed_error.py
"""
import random
import statistics as st

from common import CONSENSUS, MAIN, PR, RUNS, f, load_records


def boot(groups, n=10000):
    ks = sorted(groups); rng = random.Random(0); b = []
    for _ in range(n):
        b.append(st.median([x for k in rng.choices(ks, k=len(ks)) for x in groups[k]]))
    b.sort()
    return b[int(.05 * n)], b[int(.95 * n)]


def show(label, pairs):   # pairs: (forecast, actual, price, instance)
    grp = {}
    for p, a, pr, k in pairs:
        grp.setdefault(k, []).append((a - p) / pr)
    se = [x for v in grp.values() for x in v]
    lo, hi = boot(grp)
    print(f'| {label} | {len(se)} | {st.median(se)*100:+.4f} | [{lo*100:+.4f}, {hi*100:+.4f}] |')


R = {a: {s: load_records(c, s) for s in RUNS} for a, c in MAIN.items()}
print('| forecaster | runs | median signed error (pp) | 90% CI |')
print('|---|---|---|---|')
if CONSENSUS:
    act = {k: f(r['eps_gaap_actual']) for s in RUNS for k, r in R['GPT-5.5'][s].items()}
    show('Analyst consensus', [(CONSENSUS[k], act[k], PR[k], k) for k in act if CONSENSUS.get(k) is not None])
else:
    print('| Analyst consensus | -- | n/a (needs data/consensus.csv) | -- |')
for a in MAIN:
    pairs = []
    for s in RUNS:
        for k, r in R[a][s].items():
            p, y = f(r['eps_gaap_pred']), f(r['eps_gaap_actual'])
            if p is not None and y is not None:
                pairs.append((p, y, PR[k], k))
    show(a, pairs)
