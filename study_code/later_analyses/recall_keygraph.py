"""Key graph (top-1 UCC edges) minus SF as published, in J_tf and expected Recall@GT, per corpus (2026-11-06;
EXPLORATORY). Valentine: assembled 549 rows (gvfull ucc_top1, layer1 melnik), family macro over the families with
>= 5 scenarios and the 76 matchers of deproles; external corpora and mixed DBs: x81fix / final_B rows, as in
recall_cells.py. Writes figures/recall/recall_keygraph.csv.

Run:  SMX_VALENTINE=549 ../../harness/.venv-bdi/Scripts/python.exe recall_keygraph.py
"""
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / 'sf-matcher-study'
sys.path.insert(0, str(HERE))
import recall_cells as RC  # noqa: E402


def valentine(regime):
    os.chdir(ST)
    import valentine549 as V
    reg = V.vreg(regime)
    dep = list(csv.DictReader(open(ST / 'results' / reg / 'deproles_scores.csv', encoding='utf8')))
    keep = {r['seed'] for r in dep}
    fam = {(r['dataset'], r['scenario']): r['family'] for r in dep}
    big = {f for f in set(fam.values()) if sum(1 for v in fam.values() if v == f) >= 5}
    arms = {}
    for folder, arm in (('gvfull_%s_scores' % reg, 'ucc_top1'), ('layer1_%s_scores' % reg, 'melnik')):
        for r in V.assembled(folder) or []:
            if r['arm'] == arm and r['seed'] in keep and fam.get((r['dataset'], r['scenario'])) in big:
                k = (r['seed'], r['dataset'], r['scenario'])
                arms.setdefault(k, {})[arm] = ((float(r['sf_R_exp']) + float(r['sf_F1_Hr'])) / 2, float(r['sf_R_exp']))
    rows = [dict(seed=k[0], group=fam[(k[1], k[2])], dataset=k[1], scenario=k[2], key=v['ucc_top1'][0],
                 key_R=v['ucc_top1'][1], sf=v['melnik'][0], sf_R=v['melnik'][1])
            for k, v in arms.items() if len(v) == 2]
    return RC.contrast(rows, 'key', 'sf')


def main():
    out = []
    for regime, lab in (('plain', 'Valentine (readable)'), ('md5', 'Valentine (recoded)')):
        out.append((lab, valentine(regime)))
    for f in ('gdcsm', 'hdxsm', 'goby', 'madi'):
        rows = RC.num(RC.read('results', 'x81fix', 'scores_%s.csv' % f))
        for r in rows:
            r['corpus'], r['group'] = RC.group_of(f, r['scenario']) if f == 'madi' else (f, f)
        for rep, lab in (('gdcsm', 'GDC-SM'), ('hdxsm', 'HDXSM'), ('goby', 'GOBY'), ('madi_products', 'MaDI products'),
                         ('madi_indep', 'MaDI use cases')):
            rr = [r for r in rows if r['corpus'] == rep]
            if rr:
                out.append((lab, RC.contrast(rr, 'ucc_top1', 'melnik', 'cluster', rep)))
    fb = RC.num(RC.read('tuning', 'results', 'final_B_scores', 'scores.csv'))
    for regime, lab in (('plain', 'Mixed DBs (readable)'), ('opaque', 'Mixed DBs (recoded)')):
        rr = [r for r in fb if r['regime'] == regime and r['truth'] == 'primary']
        out.append((lab, RC.contrast(rr, 'ucc_top1', 'melnik')))
    os.makedirs(HERE / 'figures' / 'recall', exist_ok=True)
    with open(HERE / 'figures' / 'recall' / 'recall_keygraph.csv', 'w', newline='', encoding='utf8') as fh:
        w = csv.writer(fh)
        w.writerow(['corpus', 'J', 'J_lo', 'J_hi', 'R', 'R_lo', 'R_hi', 'matchers'])
        for lab, (j, r, n) in out:
            w.writerow([lab, *[round(x, 4) for x in j], *[round(x, 4) for x in r], n])
            print(f'{lab:22s} key - SF  J {j[0]:+.4f} [{j[1]:+.4f}, {j[2]:+.4f}]   R {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]'
                  f' ({n})')


if __name__ == '__main__':
    main()
