"""Recall@GT for the multi-table cells of Table II (Synthea datatypes; Lahman/MB multi-table datatypes and table
context), 2026-11-06; EXPLORATORY.

Reuses sf-matcher-study/explore_multitable_ext.py unchanged except for its metric: the module's J (= mean of expected
Recall@GT and tie-free Hungarian F1) is replaced by expected Recall@GT. Runs its score_case on the stored initial
matrices and SF outputs (no SF run), sequentially so that the replacement holds, with SMX_VALENTINE=549 as for the
paper's numbers. Contrasts as in its report: per scenario the mean over the matchers that have every arm on every
scenario; within-cell scenario bootstrap (10,000 draws); the multi-table macro averages its cells.

Run:  SMX_VALENTINE=549 ../../harness/.venv-bdi/Scripts/python.exe recall_multitable.py
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / 'sf-matcher-study'
os.chdir(ST)
sys.path.insert(0, str(ST))
import explore_multitable_ext as X  # noqa: E402
from score_screen_metrics import recall_expected  # noqa: E402

X.J = lambda S, gt, perms: recall_expected(S, gt)
NEED = ('q', 'A_d', 'A_dt', 'A_role', 'A_rt', *X.ARMS)


def scores(corpus):
    cs = X.cases(corpus)
    seeds_root = ST / 'reference' / 'seeds' / corpus
    matchers = sorted(p.name for p in seeds_root.iterdir() if p.is_dir() and p.name not in X.CONTROLS)
    dirs = X.run_dirs(corpus)
    rows = []
    for d, s, c in cs:
        rows += X.score_case((corpus, d, s, c, matchers, dirs))
    return rows, cs, matchers


def cell_contrast(rows, cs, matchers, cells, x, y):
    val = {(r['seed'], r['dataset'], r['scenario']): r for r in rows}
    cell_cases = defaultdict(list)
    for d, s, c in cs:
        cell_cases[c].append((d, s))
    rng = np.random.default_rng(X.SEED)
    per = []
    for cell in cells:
        sc = cell_cases[cell]
        used = [m for m in matchers if all((m, d, s) in val and all(k in val[(m, d, s)] for k in NEED) for d, s in sc)]
        diffs = np.array([np.mean([val[(m, d, s)][x] - val[(m, d, s)][y] for m in used]) for d, s in sc])
        per.append(diffs)
    est = float(np.mean([a.mean() for a in per]))
    boots = np.mean([[rng.choice(a, size=a.size).mean() for _ in range(X.B)] for a in per], axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi), len(used)


def main():
    out = []
    rows, cs, ms = scores('synthea')
    out.append(('Synthea$\\to$OMOP', 'dt', *cell_contrast(rows, cs, ms, ['SyntheaOMOP-Pair'], 'A_d', 'melnik')))
    rows, cs, ms = scores('indep2')
    cells = sorted({c for _, _, c in cs})
    out.append(('Lahman/MB multi-table', 'dt', *cell_contrast(rows, cs, ms, cells, 'A_d', 'melnik')))
    out.append(('Lahman/MB multi-table', 'tc', *cell_contrast(rows, cs, ms, cells, 'A_dt', 'melnik')))
    os.makedirs(HERE / 'figures' / 'recall', exist_ok=True)
    with open(HERE / 'figures' / 'recall' / 'recall_multitable.csv', 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['corpus', 'col', 'R', 'R_lo', 'R_hi', 'matchers'])
        for r in out:
            w.writerow([r[0], r[1], round(r[2], 4), round(r[3], 4), round(r[4], 4), r[5]])
    for r in out:
        print(f'{r[0]:24s} {r[1]} R {r[2]:+.4f} [{r[3]:+.4f}, {r[4]:+.4f}] matchers {r[5]}')


if __name__ == '__main__':
    main()
