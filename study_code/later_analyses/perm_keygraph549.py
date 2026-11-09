"""Key graph vs SF as published under column permutation, all 549 Valentine scenarios (2026-11-06; EXPLORATORY).

Follows the 60-scenario pilot (perm_keygraph.py). The fork session ran the key graph (top-1 UCC edges) on every 549
scenario with source and target columns permuted (tuning/results/xperm_key_<reg>/<m>__ucc_top1, permutations.json and
plan.json next to them). SF as published was checked order-invariant there (2286 matrices bit-identical after
alignment), so its stored outputs serve both orders.

Per matcher and scenario: original key graph and SF as published (valentine549.sf_run, the assembly rules), the
permuted key graph put back into the original order by row and column label (checked against permutations.json), and
the original ground truth (valentine549.corpus_dir). Metrics as recall_keygraph.py: J_tf = (expected Recall@GT +
Hungarian F1 with random ties) / 2 and Recall@GT, on scores clipped at 0. Aggregation as recall_keygraph.py: matchers
of deproles, families with >= 5 scenarios, family macro with scenario bootstrap (recall_cells.contrast).

Run:  SMX_VALENTINE=549 ../../harness/.venv-bdi/Scripts/python.exe perm_keygraph549.py --regime plain
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / 'sf-matcher-study'
TUNING = ST / 'tuning'
for p in (ST.parent / 'harness', TUNING, ST, ST / 'reference', HERE):
    sys.path.insert(0, str(p))
os.environ.setdefault('SMX_VALENTINE', '549')

import recall_cells as RC  # noqa: E402
import valentine549 as V  # noqa: E402
from score_screen_metrics import recall_expected, hungarian_random, permutations  # noqa: E402
from score_seeds import one  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

OUT = 'MatchingStepLine2'


def aligned_back(P, pr, pc, K, kr, kc, perm):
    """P (permuted order) in K's order; by labels when both carry them, else by the stored permutation."""
    back = np.empty_like(P)
    back[np.ix_(perm['source'], perm['target'])] = P
    if pr and pc and kr and kc:
        ri = {l: i for i, l in enumerate(pr)}
        ci = {l: j for j, l in enumerate(pc)}
        lab = P[np.ix_([ri[l] for l in kr], [ci[l] for l in kc])]
        if not np.array_equal(lab, back):
            raise AssertionError('label alignment disagrees with permutations.json')
        return lab
    return back


def score(regime):
    reg = V.vreg(regime)
    res = TUNING / 'results' / f'xperm_key_{reg}'
    perms = json.loads((res / 'permutations.json').read_text(encoding='utf-8'))
    dep = list(csv.DictReader(open(ST / 'results' / reg / 'deproles_scores.csv', encoding='utf8')))
    keep = {r['seed'] for r in dep}
    fam = {(r['dataset'], r['scenario']): r['family'] for r in dep}
    big = {f for f in set(fam.values()) if sum(1 for v in fam.values() if v == f) >= 5}
    done = sorted(d.name.split('__')[0] for d in res.iterdir() if (d / 'DONE').is_file())
    rows, miss, changed = [], 0, []
    for m in done:
        if m not in keep:
            continue
        for key, perm in perms.items():
            ds, sc = key.split('/')
            if fam.get((ds, sc)) not in big:
                continue
            ok, sf = V.sf_run(regime, m, 'ucc_top1', ds, sc), V.sf_run(regime, m, 'melnik', ds, sc)
            fs = [one(d / ds / sc / '_outputs' / OUT) if d else None
                  for d in (ok, res / f'{m}__ucc_top1' / 'merged', sf)]
            g = one(V.corpus_dir(regime, ds, sc) / 'ground_truth')
            if not all(fs) or g is None:
                miss += 1
                continue
            (K, kr, kc), (P, pr, pc), (S, _, _) = (read_matrix(f) for f in fs)
            gt = read_matrix(g)[0] > 0.5
            if not (K.shape == P.shape == S.shape == gt.shape) or not all(np.isfinite(M).all() for M in (K, P, S)):
                miss += 1
                continue
            B = aligned_back(P, pr, pc, K, kr, kc, perm)
            changed.append(float(np.abs(B - K).max() > 1e-6))
            ties = permutations(gt.shape, f'{ds}/{sc}')
            r = dict(seed=m, group=fam[(ds, sc)], dataset=ds, scenario=sc, changed=changed[-1])
            for name, M in (('key', K), ('keyp', B), ('sf', S)):
                Mc = np.clip(M, 0, None)
                re_, f1 = recall_expected(Mc, gt), hungarian_random(Mc, gt, ties)
                r[name], r[name + '_R'] = (re_ + f1) / 2, re_
            rows.append(r)
    lab = {'plain': 'readable', 'md5': 'recoded'}[regime]
    lines = [f'# Key graph vs SF as published under column permutation (Valentine {lab}, 549, EXPLORATORY)', '',
             f'{len(done)} matchers run; {len({r["seed"] for r in rows})} in the deproles panel scored; '
             f'{len({(r["dataset"], r["scenario"]) for r in rows})} scenarios in families with >= 5; {miss} '
             f'matcher-scenario cases missing; key-graph output changed by the permutation in '
             f'{np.mean(changed):.1%} of cases.', '', '| contrast | J_tf [95%] | Recall@GT [95%] |', '|---|---|---|']
    for a, b, name in (('key', 'sf', 'key - SF (original order)'), ('keyp', 'sf', 'key - SF (permuted columns)'),
                       ('keyp', 'key', 'permuted - original key')):
        j, rr, n = RC.contrast(rows, a, b)
        lines.append(f'| {name} | {j[0]:+.4f} [{j[1]:+.4f}, {j[2]:+.4f}] | {rr[0]:+.4f} [{rr[1]:+.4f}, {rr[2]:+.4f}] |')
    fams = sorted({r['group'] for r in rows})
    lines += ['', 'Per family (Recall@GT, key - SF: original / permuted; share of changed outputs):', '']
    for f in fams:
        sub = [r for r in rows if r['group'] == f]
        o = np.mean([r['key_R'] - r['sf_R'] for r in sub])
        p = np.mean([r['keyp_R'] - r['sf_R'] for r in sub])
        lines.append(f'- {f}: {o:+.4f} / {p:+.4f}; changed {np.mean([r["changed"] for r in sub]):.1%} '
                     f'({len({(r["dataset"], r["scenario"]) for r in sub})} scenarios)')
    (HERE / 'figures' / 'recall').mkdir(parents=True, exist_ok=True)
    (HERE / 'figures' / 'recall' / f'perm_keygraph_{reg}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with open(HERE / 'figures' / 'recall' / f'perm_keygraph_{reg}_rows.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', choices=('plain', 'md5'), default='plain')
    score(ap.parse_args().regime)
