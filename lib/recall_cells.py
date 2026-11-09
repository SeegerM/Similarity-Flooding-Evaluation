# Ported from the study's manuscript folder for this artifact; paths point to results/ and the value folder.
"""Table II on expected Recall@GT instead of J_tf (switch decided 2026-11-06; EXPLORATORY for every cell, including
cells whose J_tf version was registered).

For every cell, the same paired contrast, matcher panel, macro weights and bootstrap design as the paper's J_tf cell are
recomputed twice from the stored scores: once in J_tf (to check that this script reproduces the paper's number) and once
in Recall@GT. The study's own helpers are reused: score_final.cell_values / macro_contrast (within-group means, equal
group weights, 10,000 percentile draws within groups), explore_x81.cluster_ci with the X86 clustering (whole clusters,
>= 5 clusters), and the two-update contrasts of x93 that already carry R_exp. Cells whose sources store only J_tf are
reported as missing.

Writes figures/recall/recall_cells.csv. Run:  ../../harness/.venv-bdi/Scripts/python.exe recall_cells.py
"""
from paths import DERIVED, OUT, STUDY, STUDY_CODE, BASELINES_CODE  # noqa: E402,F401
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ST = STUDY
for p in (BASELINES_CODE, STUDY_CODE / 'tuning', STUDY_CODE, STUDY_CODE / 'reference', STUDY_CODE / 'layer3b'):
    sys.path.insert(0, str(p))
from score_final import cell_values, macro_contrast  # noqa: E402
from explore_x81 import cluster_ci, clusters, group_of  # noqa: E402
from make_results_table import ROWS  # noqa: E402

SEED = 20261076
CONTROLS = {'constant', 'random', 'oracle', 'oracle_noise025', 'oracle_noise050', 'oracle_erase050',
            'oracle_decoy100', 'embedding_random', 'rema_shuffled', 'semstruct_lm_only', 'datatype_nanfix'}


def read(*p):
    with open(ST.joinpath(*p), encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


def num(rows):
    out = []
    for r in rows:
        d = dict(r)
        for k, v in r.items():
            try:
                d[k] = float(v) if v not in ('', None) else None
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def clean(rows, keys):
    return [r for r in rows if r['seed'] not in CONTROLS and not str(r['seed']).startswith('_')
            and all(isinstance(r.get(k), float) and math.isfinite(r[k]) for k in keys)]


def contrast(rows, x, y, design='macro', corpus=None):
    """(J est, lo, hi), (R est, lo, hi) for x - y; rows are wide with x, y, x_R, y_R."""
    keys = [x, y, x + '_R', y + '_R']
    rows = clean(rows, keys)
    inc, means, _ = cell_values(rows, keys)
    if not means:
        return None
    out = []
    for a, b in ((x, y), (x + '_R', y + '_R')):
        if design == 'cluster':
            cl = clusters(corpus, sorted({k[2] for k in means}))
            if len(set(cl.values())) >= 5:
                out.append(cluster_ci(means, a, b, cl, np.random.default_rng(SEED)))
                continue
        out.append(macro_contrast(means, a, b, np.random.default_rng(SEED)))
    return out[0], out[1], len(inc)


def long_to_wide(rows, group_fn, arm_key='arm', j='J', r='R_exp', seed_key='matcher'):
    by = defaultdict(dict)
    for x in rows:
        k = (x[seed_key], group_fn(x), x['dataset'], x['scenario'])
        try:
            by[k][x[arm_key]] = float(x[j])
            by[k][x[arm_key] + '_R'] = float(x[r])
        except (TypeError, ValueError):
            pass
    return [dict(seed=k[0], group=k[1], dataset=k[2], scenario=k[3], **v) for k, v in by.items()]


def layer_long(rows, arms, gkey='family', filt=None):
    """tuning layer files: one row per (seed, arm); the term's metrics sit in sf_*."""
    by = defaultdict(dict)
    for x in rows:
        if x['arm'] not in arms or (filt and not filt(x)):
            continue
        k = (x['seed'], x[gkey], x['dataset'], x['scenario'])
        by[k][x['arm']] = (float(x['sf_R_exp']) + float(x['sf_F1_Hr'])) / 2
        by[k][x['arm'] + '_R'] = float(x['sf_R_exp'])
    return [dict(seed=k[0], group=k[1], dataset=k[2], scenario=k[3], **v) for k, v in by.items()]


# ------------------------------------------------------------------------------------------------ cells
def cells():
    res = {}
    # x81fix: GDC-SM, HDXSM, GOBY, MaDI products, MaDI use cases (datatype and role columns)
    for f, corp in (('gdcsm', 'gdcsm'), ('hdxsm', 'hdxsm'), ('goby', 'goby'), ('madi', None)):
        rows = num(read('results', 'x81fix', 'scores_%s.csv' % f))
        for r in rows:
            r['corpus'], r['group'] = group_of(f, r['scenario']) if f == 'madi' else (f, f)
        for rep, lab in (('gdcsm', 'GDC-SM'), ('hdxsm', 'HDXSM'), ('goby', 'GOBY'), ('madi_products', 'MaDI products'),
                         ('madi_indep', 'MaDI use cases')):
            rr = [r for r in rows if r['corpus'] == rep]
            if not rr:
                continue
            res[(lab, 'dt')] = contrast(rr, 'A_d', 'melnik', 'cluster', rep)
            res[(lab, 'ro')] = contrast(rr, 'A_role', 'ucc_top1', 'cluster', rep)
    # final_B: mixed databases, datatype (F1) and roles against the FD-ranked key graph (F2b)
    fb = num(read('tuning', 'results', 'final_B_scores', 'scores.csv'))
    for regime, lab in (('plain', 'Mixed DBs (readable)'), ('opaque', 'Mixed DBs (recoded)')):
        rr = [r for r in fb if r['regime'] == regime and r['truth'] == 'primary']
        res[(lab, 'dt')] = contrast(rr, 'A_d', 'melnik')
        res[(lab, 'ro')] = contrast(rr, 'A_role', 'ucc1_fd_topk')
    # final_A: independent schema pairs, table-context term (F5: mean over pairs, bootstrap over pairs)
    fa = clean([r for r in num(read('tuning', 'results', 'final_A_scores', 'scores.csv')) if r['truth'] == 'primary'],
               ['A_dt', 'melnik', 'A_dt_R', 'melnik_R'])
    inc, means, _ = cell_values(fa, ['A_dt', 'melnik', 'A_dt_R', 'melnik_R'])
    out = []
    for a, b in (('A_dt', 'melnik'), ('A_dt_R', 'melnik_R')):
        arr = np.array([means[k][a] - means[k][b] for k in sorted(means)])
        rng = np.random.default_rng(SEED)
        boots = [rng.choice(arr, size=arr.size).mean() for _ in range(10_000)]
        out.append((float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))))
    res[('Indep.\\ schema pairs', 'tc')] = (out[0], out[1], len(inc))
    # e2 (Lahman/MusicBrainz) and T2D: datatype term (additive_d) against SF as published, family macro
    e2 = layer_long(read('tuning', 'results', 'e2_ctl_scores', 'scores.csv'), ('additive_d', 'melnik'),
                    filt=lambda x: x['scenario'].endswith('_plain'))
    res[('Lahman/MusicBrainz', 'dt')] = contrast(num_f(e2), 'additive_d', 'melnik')
    t2d = layer_long(read('tuning', 'results', 't2d_exploratory_scores', 'scores.csv'), ('additive_d', 'melnik'))
    res[('T2D web tables', 'dt')] = contrast(num_f(t2d), 'additive_d', 'melnik')
    # classes: X83 term A_p (rows) against X82 SF with shape-class nodes (classfull_dev1), cluster design where possible
    a = num(read('tuning', 'results', 'x83_class_additive', 'rows.csv'))
    b = num(read('tuning', 'results', 'x82_metrics', 'classfull_dev1.csv'))
    both = [r for r in a if r['arm'] == 'A_p'] + [r for r in b if r['arm'] == 'melnik_shape']

    def grp(x):
        if x['corpus'] == 'madi':
            return x['scenario'].split('__')[0]
        if x['corpus'] == 'final_controlled':
            return x['dataset'].split('-')[0]
        return x['corpus']
    wide = long_to_wide(both, grp)
    for w in wide:
        w['corpus'] = next((x['corpus'] for x in ()), None)
    corpus_of = {(x['dataset'], x['scenario']): x['corpus'] for x in both}
    for w in wide:
        w['corpus'] = corpus_of[(w['dataset'], w['scenario'])]
        if w['corpus'] == 'madi':
            w['corpus'] = 'madi_products' if w['group'] == 'products' else 'madi_indep'
        if w['corpus'] == 'final_controlled':
            w['corpus'] += '_opaque' if 'monotone' in w['scenario'] else '_plain'
    for rep, lab, des in (('gdcsm', 'GDC-SM', 'cluster'), ('hdxsm', 'HDXSM', 'cluster'), ('goby', 'GOBY', 'cluster'),
                          ('madi_products', 'MaDI products', 'macro'), ('madi_indep', 'MaDI use cases', 'macro'),
                          ('final_indep', 'Indep.\\ schema pairs', 'macro'),
                          ('final_controlled_plain', 'Mixed DBs (readable)', 'macro')):
        rr = [w for w in wide if w['corpus'] == rep]
        res[(lab, 'cl')] = contrast(rr, 'A_p', 'melnik_shape', des, rep.split('_')[0] if des == 'cluster' else None)
    # Valentine classes (registered classfull): A_p minus SF with shape nodes, family macro
    for regime, lab in (('plain', 'Valentine (readable)'), ('md5', 'Valentine (recoded)')):
        cf = layer_long(read('tuning', 'results', 'classfull_%s_scores' % regime, 'scores.csv'), ('A_p', 'shape'))
        res[(lab, 'cl')] = contrast(num_f(cf), 'A_p', 'shape')
    # Valentine datatype term: deproles A_0.0_0.0 / A_d_R against layer1 SF as published, family macro (families >= 5)
    for regime, lab in (('plain549', 'Valentine (readable)'), ('md549', 'Valentine (recoded)')):
        ad = read('results', regime, 'deproles_scores.csv')
        sf = read('tuning', 'results', 'layer1_%s_scores' % regime, 'scores.csv')
        sfm = {(r['seed'], r['dataset'], r['scenario']): r for r in sf if r['arm'] == 'melnik'}
        fam = defaultdict(set)
        for r in ad:
            fam[r['family']].add((r['dataset'], r['scenario']))
        rows = []
        for r in ad:
            s = sfm.get((r['seed'], r['dataset'], r['scenario']))
            if s is None or len(fam[r['family']]) < 5:
                continue
            rows.append(dict(seed=r['seed'], group=r['family'], dataset=r['dataset'], scenario=r['scenario'],
                             ad=float(r['A_0.0_0.0']), ad_R=float(r['A_d_R']),
                             sf=(float(s['sf_R_exp']) + float(s['sf_F1_Hr'])) / 2, sf_R=float(s['sf_R_exp'])))
        res[(lab, 'dt')] = contrast(rows, 'ad', 'sf')
    # 3-DB table context: registered tablectx term (frozen lambda 0.5) against SF as published, per database
    tc = num(read('tuning', 'results', 'tablectx_indep_scores', 'scores.csv'))
    rows = [dict(seed=r['seed'], group=r['dataset'], dataset=r['dataset'], scenario=r['scenario'],
                 t=r['A_dt@0.5|J'], t_R=r['A_dt@0.5|R_exp'], s=r['sf|J'], s_R=r['sf|R_exp']) for r in tc]
    res[('3-DB multi-table', 'tc')] = contrast(rows, 't', 's')
    # two updates: x93 contrasts carry R_exp directly; x88 rows for the corpora of the held-out test
    x93 = read('tuning', 'results', 'x93_549_scores', 'contrasts.csv')
    name = {'valentine_plain': 'Valentine (readable)', 'valentine_md5': 'Valentine (recoded)', 'gdcsm': 'GDC-SM',
            'hdxsm': 'HDXSM', 'goby': 'GOBY', 'madi_products': 'MaDI products', 'madi_indep': 'MaDI use cases',
            'final_indep': 'Indep.\\ schema pairs'}
    t93 = defaultdict(dict)
    for r in x93:
        if r['contrast'].startswith('SF_t2') and r['corpus'] in name:
            f = lambda s: float(s) if s not in ('', None) else float('nan')
            t93[name[r['corpus']]][r['metric']] = (f(r['estimate']), f(r['lo']), f(r['hi']))
    for lab, v in t93.items():
        if 'J' in v and 'R_exp' in v:
            res[(lab, 'tu')] = (v['J'], v['R_exp'], None)
    # two updates on the mixed databases (x93 rows, per regime, database macro) and on the 3-DB corpus (iter_indep)
    r93 = [r for r in read('tuning', 'results', 'x93_549_scores', 'rows.csv') if r['corpus'] == 'final_controlled'
           and r['arm'] in ('SF_t2', 'SF_conv')]
    for regime, lab in (('plain', 'Mixed DBs (readable)'), ('monotone', 'Mixed DBs (recoded)')):
        rr = [r for r in r93 if (regime in r['scenario']) == (regime == 'monotone') or
              (regime == 'plain' and r['scenario'].endswith('_plain'))]
        rr = [r for r in rr if ('monotone' in r['scenario']) == (regime == 'monotone')]
        w = long_to_wide(rr, lambda x: x['dataset'].split('-')[0])
        res[(lab, 'tu')] = contrast(w, 'SF_t2', 'SF_conv')
    it = layer_long(read('tuning', 'results', 'iter_indep_scores', 'scores.csv'), ('melnik_t2', 'melnik'),
                    gkey='dataset')
    for w in it:
        w['group'] = w['dataset'].split('-')[0].split('_')[0]
    res[('3-DB multi-table', 'tu')] = contrast(num_f(it), 'melnik_t2', 'melnik')
    x88 = num(read('results', 'x88', 'validate_rows.csv'))
    for r in x88:
        r['seed'] = r['matcher']
    for corp, lab in (('Lahman/MusicBrainz single-table', 'Lahman/MusicBrainz'), ('T2D web tables', 'T2D web tables'),
                      ('Synthea-OMOP', 'Synthea$\\to$OMOP'), ('Lahman/MusicBrainz multi-table', 'Lahman/MB multi-table')):
        rr = [dict(r, t2=r['t2_J'], conv=r['conv_J'], t2_R=r['t2_R'], conv_R=r['conv_R']) for r in x88
              if r['corpus'] == corp]
        res[(lab, 'tu')] = contrast(rr, 't2', 'conv')
    return res


def num_f(rows):
    return [dict(r, **{k: float(v) for k, v in r.items() if k not in ('seed', 'group', 'dataset', 'scenario')})
            for r in rows]


def main():
    res = cells()
    os.makedirs(DERIVED / 'recall', exist_ok=True)
    out = []
    cols = ['dt', 'cl', 'ro', 'tc', 'tu']
    for label, g, cs in ROWS:
        for k in cols:
            paper = cs[k]
            pe = paper['e'] if isinstance(paper, dict) else None
            got = res.get((label, k))
            if got is None:
                if pe is not None:
                    out.append(dict(corpus=label, col=k, paper_J=pe, status='missing'))
                continue
            (je, jl, jh), (re_, rl, rh), n = got
            ok = pe is not None and abs(je - pe) <= 0.0015
            out.append(dict(corpus=label, col=k, paper_J=pe, J=round(je, 4), J_lo=round(jl, 4), J_hi=round(jh, 4),
                            R=round(re_, 4), R_lo=round(rl, 4), R_hi=round(rh, 4), n=n,
                            status='ok' if ok else ('no paper value' if pe is None else 'J differs')))
    with open(DERIVED / 'recall' / 'recall_cells.csv', 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=['corpus', 'col', 'paper_J', 'J', 'J_lo', 'J_hi', 'R', 'R_lo', 'R_hi', 'n',
                                          'status'])
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['corpus'][:22]:22s} {r['col']} paper {r['paper_J'] if r['paper_J'] is not None else '':>7} "
              f"J {r.get('J', ''):>8} R {r.get('R', ''):>8} [{r.get('R_lo', '')}, {r.get('R_hi', '')}] {r['status']}")


if __name__ == '__main__':
    main()
