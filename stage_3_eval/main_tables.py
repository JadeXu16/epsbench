#!/usr/bin/env python3
"""Main results: Table 1, Figures 2 and 3, the interval columns of Table 10,
Table 11 (resource use, format compliance, large errors) and Table 13 (pairwise
model comparisons).

Conventions (paper Section 3.4):
  * instance PDE = median over the instance's three runs; a run without a
    parseable forecast has PDE +inf, so it counts as a loss to the consensus
  * paired comparisons: median paired difference with a 90% bootstrap CI over
    instances, wins/losses with ties excluded, sign-test p
  * interval metrics pool all runs that return an interval; the interval score
    is averaged over runs
  * resource use: per-run medians; input counts every input token the model
    processed (uncached + cache read + cache write), summed over requests
  * surprise direction: share of runs with sign(forecast - c) == sign(actual - c)

Usage: python stage_3_eval/main_tables.py
"""
import math
import statistics as st

from common import (ALPHA, CONSENSUS, GRID, GT, MAIN, N_GRID, PR, RUNS, f, load_records,
                    paired, run_stats, scores)

R = {a: {s: load_records(c, s) for s in RUNS} for a, c in MAIN.items()}
T = {a: run_stats(c) for a, c in MAIN.items()}

# reported EPS as stored in the GPT-5.5 records, and the consensus if available
ACT = {k: f(r['eps_gaap_actual']) for s in RUNS for k, r in R['GPT-5.5'][s].items()}
AN = ({k: abs(CONSENSUS[k] - ACT[k]) / PR[k] for k in ACT if CONSENSUS.get(k) is not None}
      if CONSENSUS else None)


def inst_pde(a):
    per = {}
    for s in RUNS:
        for k in GRID:
            r = R[a][s].get(k)
            p, y = (f(r['eps_gaap_pred']), f(r['eps_gaap_actual'])) if r else (None, None)
            per.setdefault(k, []).append(abs(p - y) / PR[k] if (p is not None and y is not None) else math.inf)
    return {k: st.median(v) for k, v in per.items()}


P = {a: inst_pde(a) for a in MAIN}

# reference forecasts from reported EPS (filings, not IBES)
NAIVE_PRED = {
    'copy year-ago quarter (q-4)': {k: GT[k]['eps_year_ago'] for k in GRID},
    'copy last quarter (q-1)': {k: GT[k]['eps_last_quarter'] for k in GRID},
    'predict zero': {k: 0.0 for k in GRID},
}
NAIVE = {nm: {k: abs(p - ACT[k]) / PR[k] for k, p in d.items() if p is not None and ACT.get(k) is not None}
         for nm, d in NAIVE_PRED.items()}


def direction_acc(preds):
    ok = n = 0
    for k, p in preds:
        g, y = CONSENSUS.get(k), ACT.get(k)
        if p is None or g is None or y is None or abs(y - g) < 1e-9 or abs(p - g) < 1e-9:
            continue
        n += 1; ok += ((p > g) == (y > g))
    return ok / n if n else float('nan')


def win_from_scores(config):
    inst = {(r['ticker'], r['fiscal_quarter']): r['instance_vs_consensus'] for r in scores(config)}
    return sum(v == 'win' for v in inst.values()), sum(v == 'loss' for v in inst.values())


NA = 'n/a (needs data/consensus.csv)'
print(f'## Table 1 — Point accuracy vs analyst consensus ({N_GRID} instances, PDE = median of 3 runs)\n')
print('| configuration | median PDE (pp) | Δ vs analyst (pp), 90% CI | head-to-head win | sign p | surprise-direction acc. | compliance |')
print('|---|---|---|---|---|---|---|')
print(f"| analyst (IBES GAAP consensus) | {st.median(AN.values())*100:.3f} | — | — | — | — | — |" if AN else f"| analyst (IBES GAAP consensus) | {NA} | — | — | — | — | — |")
for nm, D in NAIVE.items():
    if AN:
        r = paired(D, AN, GRID)
        print(f"| {nm} | {st.median(D.values())*100:.3f} | {r['med']*100:+.4f} [{r['lo']*100:+.4f}, {r['hi']*100:+.4f}] | {r['w']}/{r['w']+r['l']} ({r['w']/(r['w']+r['l']):.0%}) | {r['p']:.3f} | {direction_acc(NAIVE_PRED[nm].items()):.0%} | — |")
    else:
        print(f"| {nm} | {st.median(D.values())*100:.3f} | {NA} | — | — | — | — |")
for a, c in MAIN.items():
    finite = [v for v in P[a].values() if not math.isinf(v)]
    ok = n_runs = 0; runs = []
    for s in RUNS:
        for k in GRID:
            n_runs += 1; rec = R[a][s].get(k)
            p = f(rec['eps_gaap_pred']) if rec else None
            if p is None: continue
            ok += 1; runs.append((k, p))
    comp = f"{ok}/{n_runs} ({ok/n_runs:.1%})"
    if AN:
        r = paired(P[a], AN, GRID)
        print(f"| {a} | {st.median(finite)*100:.3f} | {r['med']*100:+.4f} [{r['lo']*100:+.4f}, {r['hi']*100:+.4f}] | {r['w']}/{r['w']+r['l']} ({r['w']/(r['w']+r['l']):.0%}) | {r['p']:.3f} | {direction_acc(runs):.0%} | {comp} |")
    else:
        from common import sign_p
        w, l = win_from_scores(c)
        print(f"| {a} | {st.median(finite)*100:.3f} | {NA} | {w}/{w+l} ({w/(w+l):.0%}) | {sign_p(w, l):.3f} | {NA} | {comp} |")

print('\n### Pairwise model comparisons (row minus column, median paired ΔPDE pp [bootstrap 90% CI]; row wins/losses)\n')
names = list(MAIN)
print('| | ' + ' | '.join(names[1:]) + ' |'); print('|---|' + '---|' * (len(names) - 1))
for i, a in enumerate(names[:-1]):
    cells = []
    for b in names[1:]:
        if names.index(b) <= i: cells.append(''); continue
        r = paired(P[a], P[b], GRID); cells.append(f"{r['med']*100:+.3f} [{r['lo']*100:+.3f}, {r['hi']*100:+.3f}] ({r['w']}/{r['l']})")
    print(f"| {a} | " + ' | '.join(cells) + ' |')

print('\n## Table 2 — Calibration of the 80% interval (all runs pooled; IS = mean, pp)\n')
print('| configuration | runs with interval | coverage | below l | above u | median width (pp) | mean IS (pp) | median IS (pp) |')
print('|---|---|---|---|---|---|---|---|')
for a in MAIN:
    n = cov = lo = hi = 0; W = []; IS = []
    for s in RUNS:
        for k, r in R[a][s].items():
            l, u, y, pr = f(r.get('eps_p10')), f(r.get('eps_p90')), f(r['eps_gaap_actual']), PR.get(k)
            if l is None or u is None or y is None or not pr: continue
            n += 1; cov += l <= y <= u; lo += y < l; hi += y > u
            W.append((u - l) / pr); IS.append(((u - l) + (2 / ALPHA) * max(l - y, 0) + (2 / ALPHA) * max(y - u, 0)) / pr)
    print(f"| {a} | {n}/{N_GRID*3} | {cov/n:.1%} | {lo/n:.1%} | {hi/n:.1%} | {st.median(W)*100:.3f} | {st.mean(IS)*100:.3f} | {st.median(IS)*100:.3f} |")

print('\n## Table 3 — Efficiency (per-run medians; tokens, never dollars)\n')
print('| configuration | requests | tool calls | output tokens (K) | reasoning tokens (K) | input processed (K) | peak context (K) | note |')
print('|---|---|---|---|---|---|---|')
for a in MAIN:
    rows = [r for r in T[a].values() if r['requests_all_messages'] != '']
    med = lambda key: st.median(float(r[key]) for r in rows)
    note = 'API reports thinking inside output; reasoning not separated' if a.startswith('Opus') else ''
    print(f"| {a} | {med('requests_all_messages'):.0f} | {med('tool_calls'):.0f} | {med('output_tokens')/1e3:.1f} | {med('reasoning_tokens')/1e3:.1f} | {med('input_tokens')/1e3:.0f} | {med('peak_context_tokens')/1e3:.0f} | {note} |")

print('\n## Table 4 — Reliability\n')
print('| configuration | runs | parseable forecast | with interval | PDE > 1pp (catastrophic) | mid-run compaction |')
print('|---|---|---|---|---|---|')
for a in MAIN:
    n = ok = wi = cat = 0
    for s in RUNS:
        for k in GRID:
            n += 1; r = R[a][s].get(k)
            if not r or f(r['eps_gaap_pred']) is None: continue
            ok += 1; wi += f(r.get('eps_p10')) is not None
            v = abs(f(r['eps_gaap_pred']) - f(r['eps_gaap_actual'])) / PR[k]; cat += v > 0.01
    comp = sum(1 for r in T[a].values() if r['context_compacted'] == '1')
    print(f"| {a} | {n} | {ok} ({ok/n:.1%}) | {wi} ({wi/n:.1%}) | {cat} ({cat/n:.1%}) | {comp} ({comp/n:.1%}) |")
