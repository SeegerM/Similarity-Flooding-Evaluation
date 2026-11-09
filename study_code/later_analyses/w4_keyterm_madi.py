"""Review 7, W4 on the MaDI corpora: the key-membership term against the top-1 key graph (2026-11-08; EXPLORATORY).

Same term as w4_keyterm.py (q+ + 5 D + gamma [i in key_s and j in key_t], key reimplemented from Schematch), with gamma
transferred from readable Valentine (its leave-one-family-out choice, read from figures/recall/w4_keyterm.md), as the
role term's weight was transferred. Key graph, SF as published and the role term per matcher and scenario from the
fixed-FD external scores (results/x81fix/scores_madi.csv), initial matrices from reference/seeds/madi, corpus
harness/generated_hyfdfix/madi. Contrasts as Table II's MaDI cells (recall_cells.contrast, cluster design falling back
to the scenario bootstrap). Original column order only (the MaDI key graph was not rerun with permuted columns).
Writes figures/recall/w4_keyterm_madi.md.
Run:  ../../harness/.venv-bdi/Scripts/python.exe w4_keyterm_madi.py
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ST = ROOT / 'sf-matcher-study'
for p in (ROOT / 'harness', ST / 'tuning', ST, ST / 'reference', HERE):
    sys.path.insert(0, str(p))
import w4_keyterm as W  # noqa: E402
import recall_cells as RC  # noqa: E402
from score_screen_metrics import all_metrics, permutations  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

CORPUS = ROOT / 'harness' / 'generated_hyfdfix' / 'madi'
SEEDS = ST / 'reference' / 'seeds' / 'madi'


def gamma():
    t = (HERE / 'figures' / 'recall' / 'w4_keyterm.md').read_text(encoding='utf-8').split('## md549')[0]
    vals = [float(v) for v in re.findall(r'(?:ChEMBL|Magellan|OpenData|TPC-DI) ([0-9.]+)', t)]
    return max(set(vals), key=vals.count)    # the weight chosen for most held-out families on readable Valentine


def main():
    gm = gamma()
    src = []
    for c in ('madi', 'goby', 'hdxsm', 'gdcsm'):        # all external corpora with stored key-graph scores
        for r in csv.DictReader(open(ST / 'results' / 'x81fix' / f'scores_{c}.csv', encoding='utf-8')):
            r['_c'] = c
            src.append(r)
    keys, rows, skipped = {}, [], 0
    for r in src:
        ds, sc, m, c = r['dataset'], r['scenario'], r['seed'], r['_c']
        CORPUS, SEEDS = ROOT / 'harness' / 'generated_hyfdfix' / c, ST / 'reference' / 'seeds' / c
        d = CORPUS / ds / sc
        if (ds, sc) not in keys:
            keys[(ds, sc)] = {s: W.choose_key(d, s) for s in ('source', 'target')}
        k = keys[(ds, sc)]
        f_q, f_d = SEEDS / m / ds / sc / 'seed.csv', SEEDS / 'datatype_nanfix' / ds / sc / 'seed.csv'
        g = next(iter(sorted((d / 'ground_truth').glob('*'))), None)
        if not (f_q.is_file() and f_d.is_file() and g is not None):
            skipped += 1
            continue
        q, rl, cl = read_matrix(f_q)
        D = read_matrix(f_d)[0]
        gt = read_matrix(g)[0] > 0.5
        if not (q.shape == D.shape == gt.shape):
            skipped += 1
            continue
        rl = [x.split('.', 1)[1] if '.' in x else x for x in rl]
        cl = [x.split('.', 1)[1] if '.' in x else x for x in cl]
        K = np.outer([x in set(k['source']) for x in rl], [x in set(k['target']) for x in cl]).astype(float)
        base = np.clip(q, 0, None) + 5.0 * D
        pr = permutations(gt.shape, f'{ds}/{sc}')
        a = all_metrics(base, gt, pr)
        b = all_metrics(base + gm * K, gt, pr)
        sweep = {}
        for g2 in W.G:                                   # weight sweep, as the paper's role-term check
            s = b if g2 == gm else all_metrics(base + g2 * K, gt, pr)
            sweep[f'kt{g2}'], sweep[f'kt{g2}_R'] = (s['R_exp'] + s['F1_Hr']) / 2, s['R_exp']
        f = lambda x: float(x)
        corp, grp = RC.group_of('madi', sc) if c == 'madi' else (c, c)   # as recall_keygraph.py groups them
        rows.append(dict(seed=m, corpus=corp, group=grp, dataset=ds, scenario=sc,
                         kt=(b['R_exp'] + b['F1_Hr']) / 2, kt_R=b['R_exp'],
                         ad_check=(a['R_exp'] + a['F1_Hr']) / 2, ad=f(r['A_d']),
                         key=f(r['ucc_top1']), key_R=f(r['ucc_top1_R']), sf=f(r['melnik']), sf_R=f(r['melnik_R']),
                         role=f(r['A_role']), role_R=f(r['A_role_R']),
                         # review 8 W2: what key membership adds as a term (over the datatype term) and in the graph
                         # (key graph over SF as published); their difference is the difference in differences
                         kinc=(b['R_exp'] + b['F1_Hr']) / 2 - f(r['A_d']), kinc_R=b['R_exp'] - f(r['A_d_R']),
                         ginc=f(r['ucc_top1']) - f(r['melnik']), ginc_R=f(r['ucc_top1_R']) - f(r['melnik_R']),
                         both_keys=float(bool(k['source']) and bool(k['target'])), **sweep))
    chk = max(abs(r['ad_check'] - r['ad']) for r in rows)
    L = ['# Key-membership term on the external corpora (EXPLORATORY, 2026-11-08)', '',
         f'gamma = {gm} (transferred from readable Valentine). {len(rows)} matcher-scenario rows, {skipped} skipped. '
         f'Check: the datatype term recomputed here differs from the stored A_d by at most {chk:.2e}.', '',
         '| corpus | contrast | J_tf [95%] | Recall@GT [95%] | scenarios with keys on both sides |', '|---|---|---|---|---|']
    for rep, lab in (('madi_indep', 'MaDI use cases'), ('madi_products', 'MaDI products'), ('goby', 'GOBY'),
                     ('hdxsm', 'HDXSM'), ('gdcsm', 'GDC-SM')):
        rr = [r for r in rows if r['corpus'] == rep]
        nb = f"{int(sum(r['both_keys'] for r in rr if r['seed'] == rr[0]['seed']))} of " \
             f"{len({(r['dataset'], r['scenario']) for r in rr})}" if rr else ''
        for x, y, name in (('kt', 'key', 'key term - key graph'), ('kt', 'sf', 'key term - SF as published'),
                           ('kt', 'role', 'key term - role term'), ('role', 'key', 'role term - key graph'),
                           ('kinc', 'ginc', 'key increment, term - graph')):
            res = RC.contrast(rr, x, y, 'cluster', rep)
            if res is None:
                continue
            j, rk, n = res
            L.append(f'| {lab} | {name} | {j[0]:+.4f} [{j[1]:+.4f}, {j[2]:+.4f}] | '
                     f'{rk[0]:+.4f} [{rk[1]:+.4f}, {rk[2]:+.4f}] | {nb} ({n} matchers) |')
        sw = []
        for g2 in W.G:
            res = RC.contrast(rr, f'kt{g2}', 'key', 'cluster', rep)
            if res:
                sw.append(f'{g2}: J {res[0][0]:+.4f} [{res[0][1]:+.4f}, {res[0][2]:+.4f}], '
                          f'R {res[1][0]:+.4f} [{res[1][1]:+.4f}, {res[1][2]:+.4f}]')
        L.append(f'| {lab} | key term - key graph, weight sweep | ' + '; '.join(sw) + ' | | |')
    (HERE / 'figures' / 'recall' / 'w4_keyterm_madi.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    keep = ['seed', 'corpus', 'group', 'dataset', 'scenario', 'kt', 'kt_R', 'key', 'key_R', 'sf', 'sf_R', 'role',
            'role_R', 'kinc', 'kinc_R', 'ginc', 'ginc_R', 'both_keys']
    with open(HERE / 'figures' / 'recall' / 'w4_keyterm_ext_rows.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keep, extrasaction='ignore')     # for the artifact (2026-11-08)
        w.writeheader()
        w.writerows(rows)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
