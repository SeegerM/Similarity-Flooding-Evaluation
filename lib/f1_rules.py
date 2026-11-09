"""SF's F1 gain over its initial matcher under two assignment rules, from results/derived/recall/gain_policies_rows.csv.

'threshold': pairs scoring 0 or less stay unassigned (the registered J_tf rule); 'complete': every pair of the
one-to-one assignment is kept. Both on the same matchers and scenarios; family macro over families with at least
five scenarios, scenario bootstrap within family.
"""
import numpy as np

from keygraph import _rows
from score_final import cell_values, macro_contrast


def gains():
    rows = _rows('recall/gain_policies_rows.csv')
    out = {}
    for regime in ('plain', 'md5'):
        rr = []
        for r in rows:
            if r['regime'] != regime:
                continue
            rr.append(dict(seed=r['seed'], group=r['family'], dataset=r['dataset'], scenario=r['scenario'],
                           qR=float(r['seed_R_exp']), sR=float(r['sf_R_exp']),
                           qF_old=float(r['seed_F1_old']), sF_old=float(r['sf_F1_old']),
                           qF_full=float(r['seed_F1_full']), sF_full=float(r['sf_F1_full'])))
        fam = {}
        for r in rr:
            fam.setdefault(r['group'], set()).add((r['dataset'], r['scenario']))
        rr = [r for r in rr if len(fam[r['group']]) >= 5]
        inc, means, _ = cell_values(rr, ['qR', 'sR', 'qF_old', 'sF_old', 'qF_full', 'sF_full'])
        rng = lambda: np.random.default_rng(20261106)
        out[regime] = dict(n=len(inc), recall=macro_contrast(means, 'sR', 'qR', rng()),
                           threshold=macro_contrast(means, 'sF_old', 'qF_old', rng()),
                           complete=macro_contrast(means, 'sF_full', 'qF_full', rng()))
    return out
