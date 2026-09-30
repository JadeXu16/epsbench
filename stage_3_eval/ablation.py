#!/usr/bin/env python3
"""Table 3 (information-source ablation, GPT-5.5 under OpenCode) and Table 12
(instance-level effects of source removal).

Per-run PDE comes from each run's attribution.csv. Instance PDE = median over the
three runs; paired differences are removed-condition minus full.
Threshold for Table 12 = median, over instances and the four conditions, of the
spread (max - min) of PDE across an instance's three runs. A removal is harmful
on an instance when its paired increase exceeds the threshold.

Usage: python stage_3_eval/ablation.py
"""
import math
import statistics as st

from common import CONSENSUS, GRID, PR, RUNS, attribution, boot_ci, f, load_records, run_stats, scores, sign_p

ARMS = {
    'full': ('Full', 'gpt-5.5__opencode__default__full'),
    'news_only': ('News only', 'gpt-5.5__opencode__default__news_only'),
    'filings_only': ('Filings + transcripts', 'gpt-5.5__opencode__default__filings_only'),
    'neither': ('Neither', 'gpt-5.5__opencode__default__neither'),
}


def load_pde(config):
    out = {}
    for s in RUNS:
        for r in attribution(config, s):
            out.setdefault((r['ticker'], r['fq']), []).append(f(r['pde']))
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


data = {key: load_pde(cfg) for key, (_, cfg) in ARMS.items()}
keys = sorted(set.intersection(*(set(d) for d in data.values())) & GRID)
noise = sorted((max(v) - min(v)) * 100 for d in data.values() for k in keys
               if len(v := [x for x in d[k] if x is not None]) >= 2)
floor = noise[len(noise) // 2]
print(f'{len(keys)} instances; run-to-run threshold = {floor:.4f} pp\n')

print('## Paired change in PDE relative to full information (pp)\n')
print('| condition | median ΔPDE | 90% CI | worse on | sign test |')
print('|---|---|---|---|---|')
A = data['full']; deltas = {}
for key in ('news_only', 'filings_only', 'neither'):
    B = data[key]
    d = [(med(A[k]), med(B[k])) for k in keys]
    d = [(a, b) for a, b in d if a is not None and b is not None]
    diffs = [b - a for a, b in d]; deltas[key] = diffs
    lo, hi = boot_ci(diffs)
    w = sum(1 for a, b in d if b > a); l = sum(1 for a, b in d if b < a)
    print(f"| {ARMS[key][0]} | {st.median(diffs)*100:+.4f} | [{lo*100:+.4f}, {hi*100:+.4f}] | {w}/{w+l} ({w/(w+l):.0%}) | p={sign_p(w, l):.3f} |")


def inst_win(config):
    if CONSENSUS is None:     # from the distributed win/loss flags
        inst = {(r['ticker'], r['fiscal_quarter']): r['instance_vs_consensus'] for r in scores(config)}
        return sum(v == 'win' for k, v in inst.items() if k in keys), sum(v == 'loss' for k, v in inst.items() if k in keys)
    per = {}; an = {}
    for s in RUNS:
        for k, r in load_records(config, s).items():
            if k not in keys: continue
            a, y = f(r.get('eps_gaap_actual')), f(r.get('eps_gaap_pred'))
            per.setdefault(k, []).append(abs(y - a) / PR[k] if (y is not None and a is not None) else math.inf)
            if CONSENSUS.get(k) is not None and a is not None:
                an[k] = abs(CONSENSUS[k] - a) / PR[k]
    w = sum(1 for k in per if k in an and st.median(per[k]) < an[k])
    l = sum(1 for k in per if k in an and st.median(per[k]) > an[k])
    return w, l


print('\n## Table 3\n')
print('| information | PDE (%) | win (%) | tools | input (K) |')
print('|---|---|---|---|---|')
for key, (label, cfg) in ARMS.items():
    ag = st.median([x for k in keys if (x := med(data[key][k])) is not None])
    w, l = inst_win(cfg)
    rs = [r for (t, q, _), r in run_stats(cfg).items() if (t, q) in set(keys) and r['tool_calls'] != '']
    tools = st.median(int(r['tool_calls']) for r in rs); inp = st.median(int(r['input_tokens']) for r in rs)
    print(f'| {label} | {ag*100:.3f} | {w/(w+l):.0%} | {tools:.0f} | {inp/1e3:.0f} |')

print('\n## Table 12: instance-level effects of source removal\n')
T = floor / 100
d1, d2, d3 = deltas['news_only'], deltas['filings_only'], deltas['neither']
n = len(d1)
only_filings = [i for i in range(n) if d1[i] > T and d2[i] <= T]   # harmed only by removing filings + transcripts
only_news = [i for i in range(n) if d2[i] > T and d1[i] <= T]      # harmed only by removing news
either = [i for i in range(n) if d1[i] > T and d2[i] > T]
neither_single = [i for i in range(n) if d1[i] <= T and d2[i] <= T]
only_both = [i for i in neither_single if d3[i] > T]
none_harmful = [i for i in neither_single if d3[i] <= T]
single = only_filings + only_news + either
print('| group | instances | share (%) |')
print('|---|---|---|')
for label, g in (('No removal is harmful', none_harmful), ('Only removing both is harmful', only_both),
                 ('A single removal is harmful', single), ('  only removing filings and transcripts', only_filings),
                 ('  only removing news', only_news), ('  either removal', either)):
    print(f'| {label} | {len(g)} | {len(g)/n:.0%} |')
