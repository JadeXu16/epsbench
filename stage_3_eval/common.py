"""Shared loaders and statistics for the evaluation scripts.

Run every script from the repository root, e.g. `python stage_3_eval/main_tables.py`.

Instances, prices and ground truth come from data/. Forecast records live in
stage_4_experiments/<config>/<run>/records/, and per-run trajectory statistics in
stage_4_experiments/<config>/run_stats.csv.

Analyst consensus (IBES) is not distributed. Place it in data/consensus.csv
(columns: ticker, fiscal_quarter, eps_gaap_consensus) to obtain the columns that
need it; without the file those columns print as n/a, and win rates are read from
data/scores.csv.
"""
import csv
import glob
import json
import math
import os
import random
import statistics as st
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
EXP = os.path.join(ROOT, 'stage_4_experiments')
SCHEMAS = os.path.join(ROOT, 'stage_1_schema_building', 'ticker_schemas_v4')
RUNS = ['s1', 's2', 's3']
ALPHA = 0.2          # 80% interval -> interval score with alpha = 0.2

# The six models in the main comparison (OpenCode, default reasoning effort, all sources).
MAIN = {
    'Opus 4.6': 'opus-4.6__opencode__default__full',
    'GPT-5.5': 'gpt-5.5__opencode__default__full',
    'GPT-5.4': 'gpt-5.4__opencode__default__full',
    'GPT-5-mini': 'gpt-5-mini__opencode__default__full',
    'Qwen 3.5': 'qwen3.5__opencode__default__full',
    'GLM 4.5 Air': 'glm-4.5-air__opencode__default__full',
}


def f(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def read_csv(name):
    return list(csv.DictReader(open(os.path.join(DATA, name), newline='', encoding='utf-8')))


INSTANCES = read_csv('instances.csv')
GRID = {(r['ticker'], r['fiscal_quarter']) for r in INSTANCES}
N_GRID = len(GRID)
PR = {(r['ticker'], r['fiscal_quarter']): float(r['close_price']) for r in read_csv('prices.csv')}
GT = {(r['ticker'], r['fiscal_quarter']): {k: f(v) for k, v in r.items() if k not in ('ticker', 'fiscal_quarter')}
      for r in read_csv('ground_truth.csv')}


def load_consensus():
    p = os.path.join(DATA, 'consensus.csv')
    if not os.path.exists(p):
        return None
    return {(r['ticker'], r['fiscal_quarter']): f(r['eps_gaap_consensus']) for r in read_csv('consensus.csv')}


CONSENSUS = load_consensus()


def scores(config):
    """Rows of data/scores.csv for one configuration."""
    return [r for r in read_csv('scores.csv') if r['config'] == config]


def load_records(config, run):
    """instance -> record for one configuration and run."""
    out = {}
    for p in glob.glob(os.path.join(EXP, config, run, 'records', '*', '*', 't-1.record.json')):
        r = json.load(open(p))
        k = (r['ticker'], r['fiscal_quarter'])
        if k in GRID:
            out[k] = r
    return out


def run_stats(config):
    """(ticker, fiscal_quarter, run) -> row of run_stats.csv."""
    p = os.path.join(EXP, config, 'run_stats.csv')
    return {(r['ticker'], r['fiscal_quarter'], r['run']): r for r in csv.DictReader(open(p))}


def attribution(config, run):
    p = os.path.join(EXP, config, run, 'attribution.csv')
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def schema(t, q):
    return json.load(open(os.path.join(SCHEMAS, f'{t}_{q}.json')))


def run_pde(pred, actual, price):
    """Price-deflated error of one run; an unparseable forecast is +inf."""
    return abs(pred - actual) / price if (pred is not None and actual is not None and price) else math.inf


def sign_p(w, l):
    n = w + l; k = min(w, l)
    return min(sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n, 1.0) if n else 1.0


def boot_ci(xs, n=10000, alpha=0.10, seed=0):
    rng = random.Random(seed); k = len(xs)
    b = sorted(st.median(rng.choices(xs, k=k)) for _ in range(n))
    return b[int(alpha / 2 * n)], b[int((1 - alpha / 2) * n)]


def paired(A, B, cells):
    """A, B: instance -> PDE (may be inf). Median paired difference with a 90% bootstrap
    CI over instances, wins/losses (ties excluded), sign-test p. inf-inf pairs are dropped."""
    d = []; w = l = 0
    for k in sorted(cells):
        x, y = A.get(k), B.get(k)
        if x is None or y is None: continue
        if math.isinf(x) and math.isinf(y): continue
        if x < y: w += 1
        elif x > y: l += 1
        if not (math.isinf(x) or math.isinf(y)): d.append(x - y)
    lo, hi = boot_ci(d)
    return dict(med=st.median(d), lo=lo, hi=hi, w=w, l=l, p=sign_p(w, l))
