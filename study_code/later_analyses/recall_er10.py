"""Recall@GT for Table II's 3-DB class cell (ER10: A_p minus SF with pattern-class nodes) and its italic datatype cell
(A_d minus SF as published), 2026-11-06; EXPLORATORY.

Reproduces sf-matcher-study/reference/explore_er10.py: additive_scores.csv (layer1_indep) for SF, A_p and A_d; the
er10_indep runs rescored with score_layer1.score_seed for SF with class nodes; macro over the three databases; mean
over the matchers with every method on all scenarios; 10,000 scenario draws stratified by database, seed 20261004.
The metric is either J_tf (check) or expected Recall@GT.

Run:  ../../harness/.venv-bdi/Scripts/python.exe recall_er10.py
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


def main():
    os.chdir(ST)
    import explore_er10 as E
    import score_layer1 as sl
    from score_seeds import cases_for
    cases = cases_for('indep')
    fam = {(d, s): f for d, s, f in cases}
    add = list(csv.DictReader((E.RES / 'layer1_indep_scores' / 'additive_scores.csv').open(encoding='utf-8')))
    by_seed = defaultdict(list)
    for d in sorted(p for p in (E.RES / 'er10_indep').iterdir() if (p / 'DONE').is_file()
                    and p.name.split('__', 1)[1] == 'melnik_shape'):
        by_seed[d.name.split('__', 1)[0]].append(str(d))
    with Pool(8) as pool:
        parts = pool.map(sl.score_seed, [('indep', s, dirs, cases) for s, dirs in sorted(by_seed.items())])
    res = {}
    for metric in ('J', 'R'):
        val = defaultdict(dict)
        for r in add:
            k = (r['seed'], r['dataset'], r['scenario'])
            for p, name in (('sf', 'SF'), ('A_p', 'A_p'), ('A_d', 'A_d')):
                dR = float(r[f'{p}_R_exp']) - float(r['q_R_exp'])
                dF = float(r[f'{p}_F1_Hr']) - float(r['q_F1_Hr'])
                val[k][name] = (dR + dF) / 2 if metric == 'J' else dR
        for rows in parts:
            for r in rows:
                k = (r['seed'], r['dataset'], r['scenario'])
                dR = r['sf_R_exp'] - r['seed_R_exp']
                dF = r['sf_F1_Hr'] - r['seed_F1_Hr']
                val[k]['SF+P'] = (dR + dF) / 2 if metric == 'J' else dR
        methods = ['SF', 'SF+P', 'A_p', 'A_d']
        scen = sorted({(d, s) for (_, d, s) in val})
        matchers = sorted({m for (m, _, _) in val
                           if all(all(x in val.get((m, d, s), {}) for x in methods) for d, s in scen)})
        fams = sorted({fam[x] for x in scen})
        idx = {f: np.array([i for i, x in enumerate(scen) if fam[x] == f]) for f in fams}
        M = {x: np.array([[val[(m, d, s)][x] for d, s in scen] for m in matchers]) for x in methods}
        rng = np.random.default_rng(E.SEED)
        draws = [{f: ix[rng.integers(0, len(ix), size=len(ix))] for f, ix in idx.items()} for _ in range(E.B)]
        for a, b, col in (('A_p', 'SF+P', 'cl'), ('A_d', 'SF', 'dt')):
            D = M[a] - M[b]
            est = float(np.mean([D[:, ix].mean(axis=1) for ix in idx.values()], axis=0).mean())
            boot = np.array([np.mean([D[:, dr[f]].mean(axis=1) for f in fams], axis=0).mean() for dr in draws])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            res[(col, metric)] = (est, float(lo), float(hi), len(matchers))
    os.makedirs(HERE / 'figures' / 'recall', exist_ok=True)
    with open(HERE / 'figures' / 'recall' / 'recall_er10.csv', 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['corpus', 'col', 'J', 'J_lo', 'J_hi', 'R', 'R_lo', 'R_hi', 'matchers'])
        for col in ('cl', 'dt'):
            j, r = res[(col, 'J')], res[(col, 'R')]
            w.writerow(['3-DB multi-table', col, *[round(x, 4) for x in j[:3]], *[round(x, 4) for x in r[:3]], r[3]])
            print(f'3-DB {col}: J {j[0]:+.4f} [{j[1]:+.4f}, {j[2]:+.4f}]  R {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}] '
                  f'({r[3]} matchers)')


if __name__ == '__main__':
    main()
