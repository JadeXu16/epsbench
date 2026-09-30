#!/usr/bin/env python3
"""Table 4: harness and reasoning-effort comparisons.

For each model, paired ΔPDE is relative to OpenCode at the default effort.
Coverage pools runs that return an interval. Tools, input and output are
per-run medians from run_stats.csv; input counts every input token processed
(uncached + cache read + cache write); output includes reasoning tokens where
the API reports them separately.

Usage: python stage_3_eval/harness_effort.py
"""
import math
import statistics as st

from common import CONSENSUS, GRID, N_GRID, PR, RUNS, f, load_records, paired, run_stats

ROWS = [   # (model, harness, effort, config); the first row of each model is its reference
    ('Opus 4.6', 'OpenCode', 'default', 'opus-4.6__opencode__default__full'),
    ('Opus 4.6', 'Claude Code', 'default', 'opus-4.6__claudecode__default__full'),
    ('Opus 4.6', 'OpenCode', 'max', 'opus-4.6__opencode__max__full'),
    ('Opus 4.6', 'Claude Code', 'max', 'opus-4.6__claudecode__max__full'),
    ('GPT-5.5', 'OpenCode', 'default', 'gpt-5.5__opencode__default__full'),
    ('GPT-5.5', 'Vanilla', 'default', 'gpt-5.5__vanilla__default__full'),
    ('GPT-5.5', 'OpenCode', 'max', 'gpt-5.5__opencode__max__full'),
]
R = {c: {s: load_records(c, s) for s in RUNS} for *_, c in ROWS}


def inst(c):
    per = {}
    for s in RUNS:
        for k in GRID:
            r = R[c][s].get(k)
            p, a = (f(r['eps_gaap_pred']), f(r['eps_gaap_actual'])) if r else (None, None)
            per.setdefault(k, []).append(abs(p - a) / PR[k] if (p is not None and a is not None) else math.inf)
    return {k: st.median(v) for k, v in per.items()}


P = {c: inst(c) for *_, c in ROWS}


def coverage(c):
    n = cov = 0
    for s in RUNS:
        for k, r in R[c][s].items():
            l, u, y = f(r.get('eps_p10')), f(r.get('eps_p90')), f(r['eps_gaap_actual'])
            if None in (l, u, y): continue
            n += 1; cov += l <= y <= u
    return cov / n, n


def eff(c):
    rs = [r for r in run_stats(c).values() if r['tool_calls'] != '']
    m = lambda xs: st.median(xs)
    out = m([float(r['output_tokens']) + float(r['reasoning_tokens'] or 0) for r in rs])
    return m([int(r['tool_calls']) for r in rs]), m([int(r['input_tokens']) for r in rs]), out


print('| model | harness | effort | PDE (%) | paired ΔPDE, pp (90% CI) | coverage (%) | tools | input (K) | output (K) |')
print('|---|---|---|---|---|---|---|---|---|')
ref = None
for model, harness, effort, c in ROWS:
    if ref is None or ref[0] != model:
        ref = (model, c); delta = '--'
    else:
        r = paired(P[c], P[ref[1]], GRID)
        delta = f"{r['med']*100:+.3f} [{r['lo']*100:+.3f}, {r['hi']*100:+.3f}]"
    fin = [v for v in P[c].values() if not math.isinf(v)]
    cov, n = coverage(c); tools, inp, out = eff(c)
    print(f'| {model} | {harness} | {effort} | {st.median(fin)*100:.3f} | {delta} | {cov:.1%} ({n}/{3*N_GRID}) | {tools:.0f} | {inp/1e3:.0f} | {out/1e3:.1f} |')

if CONSENSUS:
    print('\nEach configuration against analyst consensus (median paired ΔPDE, pp):')
    act = {k: f(r['eps_gaap_actual']) for s in RUNS for k, r in R['gpt-5.5__opencode__default__full'][s].items()}
    AN = {k: abs(CONSENSUS[k] - act[k]) / PR[k] for k in act if CONSENSUS.get(k) is not None}
    for model, harness, effort, c in ROWS:
        r = paired(P[c], AN, GRID)
        print(f"  {model:9s} {harness:12s} {effort:8s} {r['med']*100:+.4f} [{r['lo']*100:+.4f}, {r['hi']*100:+.4f}]  win {r['w']}/{r['w']+r['l']}")
