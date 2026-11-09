"""Review 6, W1: role term minus the key graph with permuted columns, Valentine 549 (2026-11-08; EXPLORATORY).

The paper said the role term does not recover the key edges' ranking information on Valentine "also with permuted
columns", but only key graph minus SF had been permuted. Here, per matcher and scenario:
- role term: Recall@GT from figures/loso/role_recall_<regime>.csv (the per-row rescoring behind Table II's role cells),
  J_tf from results/<reg549>/deproles_scores.csv with the paper's leave-one-family-out weight (recall_roles.CHOICE);
- key graph, original and with permuted columns: figures/recall/perm_keygraph_<reg549>_rows.csv (perm_keygraph549.py).
The role term is column-order invariant (set-based roles plus the datatype term), so it needs no rerun.
Contrast as Table II (family macro over families with >= 5 scenarios, scenario bootstrap within family,
score_final.macro_contrast, seed 20261076); also without ChEMBL and per family.
Writes figures/recall/w1_roles_perm.md.
Run:  ../../harness/.venv-bdi/Scripts/python.exe w1_roles_perm.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / 'sf-matcher-study'
for p in (ST.parent / 'harness', ST / 'tuning', ST, ST / 'reference'):
    sys.path.insert(0, str(p))
from score_final import cell_values, macro_contrast  # noqa: E402

CHOICE = {'plain': {'ChEMBL': 0.2, 'Magellan': 0.5, 'OpenData': 0.2, 'TPC-DI': 0.5, '_other': 0.2},
          'md5': {'_other': 1.0}}
REG = {'plain': 'plain549', 'md5': 'md549'}


def contrast(rows, x, y):
    inc, means, _ = cell_values(rows, [x, y])
    return macro_contrast(means, x, y, np.random.default_rng(20261076)), len(inc), len(means)


def f(t):
    return f'{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]'


def main():
    L = ['# Role term minus key graph with permuted columns (Valentine 549, EXPLORATORY, 2026-11-08)', '',
         '| regime | subset | metric | role - key (original) | role - key (permuted) | matchers / scenarios |',
         '|---|---|---|---|---|---|']
    for regime in ('plain', 'md5'):
        reg = REG[regime]
        perm = {(r['seed'], r['dataset'], r['scenario']): r for r in
                csv.DictReader(open(HERE / 'figures' / 'recall' / f'perm_keygraph_{reg}_rows.csv', encoding='utf8'))}
        roleR = {(r['matcher'], r['dataset'], r['scenario']): float(r['role_R']) for r in
                 csv.DictReader(open(HERE / 'figures' / 'loso' / f'role_recall_{regime}.csv', encoding='utf8'))}
        g = lambda fam: CHOICE[regime].get(fam, CHOICE[regime]['_other'])
        roleJ = {}
        for r in csv.DictReader(open(ST / 'results' / reg / 'deproles_scores.csv', encoding='utf8')):
            roleJ[(r['seed'], r['dataset'], r['scenario'])] = float(r['A_%s_0.0' % g(r['family'])])
        for metric, role, kk, kp in (('R', roleR, 'key_R', 'keyp_R'), ('J', roleJ, 'key', 'keyp')):
            rows = []
            for k, p in perm.items():
                if k in role:
                    rows.append(dict(seed=k[0], group=p['group'], dataset=k[1], scenario=k[2], role=role[k],
                                     key=float(p[kk]), keyp=float(p[kp])))
            for subset, keep in (('all', lambda r: True), ('without ChEMBL', lambda r: r['group'] != 'ChEMBL')):
                sub = [r for r in rows if keep(r)]
                o, n, s = contrast(sub, 'role', 'key')
                pm, _, _ = contrast(sub, 'role', 'keyp')
                L.append(f'| {reg} | {subset} | {metric} | {f(o)} | {f(pm)} | {n} / {s} |')
            fams = sorted({r['group'] for r in rows})
            per = '; '.join(f"{fa}: {np.mean([r['role'] - r['key'] for r in rows if r['group'] == fa]):+.3f} / "
                            f"{np.mean([r['role'] - r['keyp'] for r in rows if r['group'] == fa]):+.3f}" for fa in fams)
            L.append(f'| {reg} | per family (mean, original / permuted) | {metric} | {per} | | |')
    out = HERE / 'figures' / 'recall' / 'w1_roles_perm.md'
    out.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
