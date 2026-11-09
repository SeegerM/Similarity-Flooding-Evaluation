"""SF's gain over its initial matcher under each F1 extraction policy of f1_policies.py (2026-11-06; step 1; EXPLORATORY).

Same jobs as rescore_gain_invariant.py (layer 1, arm melnik = SF as published, Valentine 549 = plain + pnew and
md5 + mnew, duplicate TPC-DI scenario dropped). In the workers, score_layer1.all_metrics also returns F1_<policy>
for every policy, so each row carries all variants for the initial matcher (seed_*) and SF (sf_*).

Writes figures/recall/gain_policies_rows.csv and figures/recall/gain_policies.md.
Run:  ../../harness/.venv-bdi/Scripts/python.exe rescore_f1_policies_gain.py
"""
import csv
import io
import os
import sys
from contextlib import redirect_stdout
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / 'sf-matcher-study'
DUP = ('TPC-DI-Joinable', 'prospect_horizontal_50_ec_av')
CONTROLS = {'constant', 'random', 'oracle', 'oracle_noise025', 'oracle_noise050', 'oracle_erase050',
            'oracle_decoy100', 'embedding_random', 'rema_shuffled', 'semstruct_lm_only', 'datatype_nanfix'}
OUT = HERE / 'figures' / 'recall'


def init():
    os.chdir(ST)
    for p in (str(ST.parent / 'harness'), str(ST / 'tuning'), str(ST), str(ST / 'reference'), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import score_layer1 as L
    import f1_policies as P
    orig = L.all_metrics

    def more(S, gt, perms):
        r = dict(orig(S, gt, perms))
        for p, v in P.all_policies(S, gt, perms).items():
            r['F1_' + p] = v
        A = np.asarray(S, dtype=float)
        m = float(np.max(np.abs(A))) if A.size else 0.0
        r['R_rel'] = M.recall_expected(A, gt, eps=1e-9 * m) if m > 0 else r['R_exp']   # tolerance relative to scale
        r['R_round'] = M.recall_expected(P._scaled(A), gt) if m > 0 else r['R_exp']    # also merges 32-bit noise
        r['S_max'], r['S_min'] = float(np.max(A)), float(np.min(A))
        return r
    import score_screen_metrics as M
    L.all_metrics = more
    extra = ['F1_' + p for p in P.POLICIES] + ['R_rel', 'R_round', 'S_max', 'S_min']
    if extra[0] not in L.METRICS:
        L.METRICS = list(L.METRICS) + extra


def work(job):
    import score_layer1 as L
    return L.score_seed(job)


def main():
    init()
    from score_seeds import cases_for
    jobs = []
    for regime in ('plain', 'pnew', 'md5', 'mnew'):
        cs = cases_for(regime)
        if regime in ('plain', 'md5'):
            cs = [c for c in cs if (c[0], c[1]) != DUP]
        root = ST / 'tuning' / 'results' / f'layer1_{regime}'
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name.endswith('__melnik') and d.name.split('__')[0] not in CONTROLS:
                jobs.append((regime, d.name.split('__')[0], [str(d)], cs))
    print(len(jobs), 'jobs', flush=True)
    with Pool(8, initializer=init) as pool:
        rows = []
        for job, part in zip(jobs, pool.imap(work, jobs)):
            for r in part:
                r['regime'] = {'plain': 'plain', 'pnew': 'plain', 'md5': 'md5', 'mnew': 'md5'}[job[0]]
            rows += part
    keys = sorted({k for r in rows for k in r})
    os.makedirs(OUT, exist_ok=True)
    with open(OUT / 'gain_policies_rows.csv', 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    buf = io.StringIO()
    with redirect_stdout(buf):
        summarise(rows)
    print(buf.getvalue())
    (OUT / 'gain_policies.md').write_text('```\n' + buf.getvalue() + '```\n', encoding='utf8')


def summarise(rows):
    from score_final import cell_values, macro_contrast
    from f1_policies import POLICIES
    for regime in ('plain', 'md5'):
        rr = []
        for r in rows:
            if r['regime'] != regime:
                continue
            d = dict(seed=r['seed'], group=r['family'], dataset=r['dataset'], scenario=r['scenario'],
                     qR=r['seed_R_exp'], sR=r['sf_R_exp'], qRrel=r['seed_R_rel'], sRrel=r['sf_R_rel'],
                     qRround=r['seed_R_round'], sRround=r['sf_R_round'])
            for p in POLICIES:
                d['qF_' + p], d['sF_' + p] = r['seed_F1_' + p], r['sf_F1_' + p]
                d['qJ_' + p], d['sJ_' + p] = (d['qR'] + d['qF_' + p]) / 2, (d['sR'] + d['sF_' + p]) / 2
            rr.append(d)
        fam = {}
        for r in rr:
            fam.setdefault(r['group'], set()).add((r['dataset'], r['scenario']))
        rr = [r for r in rr if len(fam[r['group']]) >= 5]
        keys = ['qR', 'sR', 'qRrel', 'sRrel', 'qRround', 'sRround'] + [f'{a}{m}_{p}' for p in POLICIES for a in 'qs' for m in 'FJ']
        inc, means, by = cell_values(rr, keys)
        print(f'== {regime}: {len(means)} scenarios, {len(inc)} matchers; SF as published minus its initial matcher')
        e = macro_contrast(means, 'sR', 'qR', np.random.default_rng(20261106))
        print(f'  Recall@GT                  {e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}]')
        for p in POLICIES:
            f = macro_contrast(means, 'sF_' + p, 'qF_' + p, np.random.default_rng(20261106))
            j = macro_contrast(means, 'sJ_' + p, 'qJ_' + p, np.random.default_rng(20261106))
            lv = macro_contrast(means, 'qF_' + p, 'qF_old', np.random.default_rng(20261106))
            sv = macro_contrast(means, 'sF_' + p, 'sF_old', np.random.default_rng(20261106))
            print(f'  F1 {p:6s} gain {f[0]:+.4f} [{f[1]:+.4f}, {f[2]:+.4f}]   J_tf gain {j[0]:+.4f}'
                  f'   level vs old: initial {lv[0]:+.4f}, SF {sv[0]:+.4f}')
        # Recall@GT tie tolerance: absolute 1e-9 (as used), relative to max|S|, and after rounding at 1e-6 of max|S|
        raw = [r for r in rows if r['regime'] == regime]
        for side, lab in (('seed', 'initial'), ('sf', 'SF')):
            mx = np.array([r[side + '_S_max'] for r in raw])
            mn = np.array([r[side + '_S_min'] for r in raw])
            print(f'  {lab:7s} score range: max in [{mx.min():.3g}, {mx.max():.3g}], median max {np.median(mx):.3g};'
                  f' min in [{mn.min():.3g}, {mn.max():.3g}]')
            for v in ('R_rel', 'R_round'):
                d = np.array([r[f'{side}_{v}'] - r[side + '_R_exp'] for r in raw])
                print(f'    {v:8s} vs R_exp: changed in {100 * np.mean(np.abs(d) > 1e-9):5.2f}% of cases,'
                      f' mean {d.mean():+.5f}, max |d| {np.abs(d).max():.4f}')
        for v in ('Rrel', 'Rround'):
            e = macro_contrast(means, 's' + v, 'q' + v, np.random.default_rng(20261106))
            print(f'  Recall@GT gain with {v:6s} {e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}]')
        # share of matcher-scenario cases whose F1 changes, per policy step
        for a, b, lab in (('old', 'round', 'storage noise (old -> round)'), ('round', 'inv', 'abstention (round -> inv)'),
                          ('inv', 'full', 'no abstention (inv -> full)')):
            for side in 'qs':
                ch = np.mean([abs(r[f'{side}F_{a}'] - r[f'{side}F_{b}']) > 1e-9 for r in rr])
                print(f'  changed by {lab:32s} {"initial" if side == "q" else "SF":7s} {100 * ch:5.1f}% of cases')
        print()


if __name__ == '__main__':
    main()
