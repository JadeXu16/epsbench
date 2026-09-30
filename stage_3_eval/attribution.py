#!/usr/bin/env python3
"""Per-run error at three stages of the income statement, written to
stage_4_experiments/<config>/<run>/attribution.csv.

For each run: pde = |EPS forecast - reported EPS| / price, and the operating and
pretax errors converted to the same scale (dollar error / reported diluted shares
/ price). Because all three share one denominator, the increments
op_pde, pretax_pde - op_pde and pde - pretax_pde add up to pde. `layer` names the
stage with the largest increment (banks and energy companies have no operating
subtotal; their first stage is pretax income, labelled driver-miss).

Usage: python stage_3_eval/attribution.py [config ...]     (default: all configurations)
"""
import csv
import os
import sys

from common import EXP, GT, PR, RUNS, f, load_records

COLS = ['ticker', 'fq', 'slice', 'pred', 'actual', 'price', 'pde', 'op_pde', 'pretax_pde', 'layer']
ROUND = {'pred': 3, 'actual': 2, 'price': 2, 'pde': 6, 'op_pde': 6, 'pretax_pde': 6}


def rows_for(config, run):
    out = []
    for k, r in load_records(config, run).items():
        p, a = f(r.get('eps_gaap_pred')), f(r.get('eps_gaap_actual'))
        if p is None or not a:
            continue
        price, sh = PR[k], GT[k]['diluted_shares']
        e_pde = abs(p - a) / price
        op_p, op_a = f(r.get('operating_pred')), f(r.get('operating_actual'))
        pt_p, pt_a = f(r.get('pretax_pred')), f(r.get('pretax_actual'))
        op_pde = abs(op_p - op_a) * 1e6 / sh / price if (sh and op_p is not None and op_a is not None) else None
        pt_pde = abs(pt_p - pt_a) * 1e6 / sh / price if (sh and pt_p is not None and pt_a is not None) else None
        if op_pde is not None and pt_pde is not None:
            jumps = {'operating-miss': op_pde, 'non-op-miss': pt_pde - op_pde, 'tax-share-miss': e_pde - pt_pde}
        elif pt_pde is not None:
            jumps = {'driver-miss': pt_pde, 'tax-share-miss': e_pde - pt_pde}
        else:
            jumps = {}
        row = dict(ticker=k[0], fq=k[1], slice=r.get('slice_days_before', 1), pred=p, actual=a, price=price,
                   pde=e_pde, op_pde=op_pde, pretax_pde=pt_pde, layer=max(jumps, key=jumps.get) if jumps else '')
        out.append({c: (round(v, ROUND[c]) if c in ROUND and v is not None else v) for c, v in row.items()})
    return sorted(out, key=lambda x: (x['ticker'], x['fq']))


if __name__ == '__main__':
    configs = sys.argv[1:] or sorted(d for d in os.listdir(EXP) if os.path.isdir(os.path.join(EXP, d)))
    for c in configs:
        for s in RUNS:
            rows = rows_for(c, s)
            with open(os.path.join(EXP, c, s, 'attribution.csv'), 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)
        print(f'{c}: written')
