"""Recall@GT for Table II's Valentine role cells (role term minus SF with the top-1 key graph), 2026-11-06; EXPLORATORY.

Reuses sf-matcher-study/explore_deproles.py (X46) under SMX_VALENTINE=549: its score_matcher builds the role term
q+ + 5D + gamma R on the stored initial matrices and profiles. In the workers, its metric is replaced so that the stored
value is expected Recall@GT, and its grid is cut to the weights chosen leave-one-family-out in the paper's run
(readable: 0.2 for ChEMBL, OpenData and the other families, 0.5 for Magellan and TPC-DI; recoded: 1.0 everywhere).
SF with the top-1 key graph comes from the assembled gvfull_<regime>549 scores (sf_R_exp / sf_F1_Hr). The J_tf check
uses the stored deproles_scores.csv. Contrast: family macro (families with >= 5 scenarios) of the mean over the
76 matchers; 95% percentile bootstrap over scenarios within family (score_final.macro_contrast).

Run:  SMX_VALENTINE=549 ../../harness/.venv-bdi/Scripts/python.exe recall_roles.py
"""
import csv
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / 'sf-matcher-study'
for p in (ST.parent / 'harness', ST / 'tuning', ST, ST / 'reference'):
    sys.path.insert(0, str(p))

CHOICE = {'plain': {'ChEMBL': 0.2, 'Magellan': 0.5, 'OpenData': 0.2, 'TPC-DI': 0.5, '_other': 0.2},
          'md5': {'_other': 1.0}}


def init(regime):
    os.chdir(ST)
    import explore_deproles as XD
    XD.GRID = sorted({(g, 0.0) for g in CHOICE[regime].values()} | {(0.0, 0.0)})
    orig = XD.all_metrics

    def rmetrics(M, gt, perms):
        r = orig(M, gt, perms)
        return dict(r, F1_Hr=r['R_exp'])          # (R + F1)/2 then stores R_exp
    XD.all_metrics = rmetrics


def work(job):
    import explore_deproles as XD
    return XD.score_matcher(job)


def contrast(rows, x, y):
    from score_final import cell_values, macro_contrast
    inc, means, _ = cell_values(rows, [x, y])
    return macro_contrast(means, x, y, np.random.default_rng(20261076)), len(inc)


def main():
    os.chdir(ST)
    import valentine549 as V
    out = []
    for regime, tag in (('plain', 'Valentine (readable)'), ('md5', 'Valentine (recoded)')):
        reg549 = V.vreg(regime)
        dep = list(csv.DictReader(open(ST / 'results' / reg549 / 'deproles_scores.csv', encoding='utf8')))
        seeds = sorted({r['seed'] for r in dep})
        fam = {(r['dataset'], r['scenario']): r['family'] for r in dep}
        big = {f for f in set(fam.values()) if sum(1 for v in fam.values() if v == f) >= 5}
        g = lambda f: CHOICE[regime].get(f, CHOICE[regime]['_other'])
        sf = {}
        for r in V.assembled('gvfull_%s_scores' % reg549) or []:
            if r['arm'] == 'ucc_top1':
                sf[(r['seed'], r['dataset'], r['scenario'])] = ((float(r['sf_R_exp']) + float(r['sf_F1_Hr'])) / 2,
                                                              float(r['sf_R_exp']))
        # J_tf check from the stored scores
        rows = []
        for r in dep:
            k = (r['seed'], r['dataset'], r['scenario'])
            if k in sf and r['family'] in big:
                rows.append(dict(seed=r['seed'], group=r['family'], dataset=r['dataset'], scenario=r['scenario'],
                                 role=float(r['A_%s_0.0' % g(r['family'])]), key=sf[k][0]))
        (jt, n) = contrast(rows, 'role', 'key')
        # Recall@GT: rescore the role term
        cs = [c for c in V.cases549() if fam.get((c[0], c[1])) in big]
        with Pool(10, initializer=init, initargs=(regime,)) as pool:
            parts = pool.map(work, [(regime, s, cs) for s in seeds])
        rr = []
        for part in parts:
            for r in part:
                k = (r['seed'], r['dataset'], r['scenario'])
                if k in sf:
                    rr.append(dict(seed=r['seed'], group=r['family'], dataset=r['dataset'], scenario=r['scenario'],
                                   role=r['A_%s_0.0' % g(r['family'])], key=sf[k][1]))
        (rt, nr) = contrast(rr, 'role', 'key')
        out.append((tag, 'ro', jt, rt, n, nr))
        print(f'{tag:22s} ro  J {jt[0]:+.4f} [{jt[1]:+.4f}, {jt[2]:+.4f}] ({n})  R {rt[0]:+.4f} [{rt[1]:+.4f}, '
              f'{rt[2]:+.4f}] ({nr})', flush=True)
    os.makedirs(HERE / 'figures' / 'recall', exist_ok=True)
    with open(HERE / 'figures' / 'recall' / 'recall_roles.csv', 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['corpus', 'col', 'J', 'J_lo', 'J_hi', 'R', 'R_lo', 'R_hi', 'matchers'])
        for tag, col, jt, rt, n, nr in out:
            w.writerow([tag, col, *[round(x, 4) for x in jt], *[round(x, 4) for x in rt], nr])


if __name__ == '__main__':
    main()
