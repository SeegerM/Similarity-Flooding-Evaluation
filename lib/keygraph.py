"""Summaries of the key-graph analyses from their cached per-row tables (results/derived/recall, results/derived/loso).

Contrasts follow the paper's design: family macro over families with at least five scenarios, 95% percentile
bootstrap over scenarios within family (Valentine); on the external corpora the cluster bootstrap where a corpus has
at least five clusters, else the scenario bootstrap (recall_cells.contrast).
"""
import csv
import gzip
import io

from paths import DERIVED, STUDY
import recall_cells as RC

CHOICE = {'plain': {'ChEMBL': 0.2, 'Magellan': 0.5, 'OpenData': 0.2, 'TPC-DI': 0.5, '_other': 0.2},
          'md5': {'_other': 1.0}}     # the role term's leave-one-family-out weights (as in Table II)


def _rows(path):
    p = DERIVED / path
    if not p.is_file() and (p.parent / (p.name + '.gz')).is_file():
        with gzip.open(str(p) + '.gz', 'rt', encoding='utf-8') as f:
            return list(csv.DictReader(io.StringIO(f.read())))
    return list(csv.DictReader(open(p, encoding='utf-8')))


def _num(rows, keys):
    for r in rows:
        for k in keys:
            r[k] = float(r[k])
    return rows


def permutation(reg):
    """Key graph - SF as published, original and with permuted columns ({contrast: (J, R)})."""
    rows = _num(_rows(f'recall/perm_keygraph_{reg}_rows.csv'), ['key', 'key_R', 'keyp', 'keyp_R', 'sf', 'sf_R', 'changed'])
    out = {name: RC.contrast(rows, a, b)[:2] for a, b, name in
           (('key', 'sf', 'key graph - SF'), ('keyp', 'sf', 'key graph - SF, columns permuted'),
            ('keyp', 'key', 'permuted - original key graph'))}
    out['_changed'] = sum(r['changed'] for r in rows) / len(rows)
    return out


def roles_vs_key(reg, regime):
    """Role term - key graph, original and permuted, all families and without ChEMBL."""
    perm = {(r['seed'], r['dataset'], r['scenario']): r for r in _rows(f'recall/perm_keygraph_{reg}_rows.csv')}
    role_r = {(r['matcher'], r['dataset'], r['scenario']): float(r['role_R'])
              for r in _rows(f'loso/role_recall_{regime}.csv')}
    role_j = {}
    for r in csv.DictReader(open(STUDY / 'results' / reg / 'deproles_scores.csv', encoding='utf-8')):
        g = CHOICE[regime].get(r['family'], CHOICE[regime]['_other'])
        role_j[(r['seed'], r['dataset'], r['scenario'])] = float(r['A_%s_0.0' % g])
    rows = []
    for k, p in perm.items():
        if k in role_r and k in role_j:
            rows.append(dict(seed=k[0], group=p['group'], dataset=k[1], scenario=k[2], role=role_j[k],
                             role_R=role_r[k], key=float(p['key']), key_R=float(p['key_R']),
                             keyp=float(p['keyp']), keyp_R=float(p['keyp_R'])))
    out = {}
    for sub, keep in (('all', lambda r: True), ('without ChEMBL', lambda r: r['group'] != 'ChEMBL')):
        rr = [r for r in rows if keep(r)]
        out[f'role - key graph ({sub})'] = RC.contrast(rr, 'role', 'key')[:2]
        out[f'role - key graph, columns permuted ({sub})'] = RC.contrast(rr, 'role', 'keyp')[:2]
    return out


def keyterm_valentine(reg):
    """Key-membership term against the key graph, SF as published and the role term."""
    rows = [r for r in _rows('recall/w4_keyterm_rows.csv') if r['regime'] == reg]
    _num(rows, ['kt', 'kt_R', 'ktp', 'ktp_R', 'key', 'key_R', 'keyp', 'keyp_R', 'sf', 'sf_R', 'role', 'role_R'])
    return {name: RC.contrast(rows, a, b)[:2] for a, b, name in
            (('kt', 'key', 'key term - key graph'), ('ktp', 'keyp', 'key term - key graph, columns permuted'),
             ('kt', 'sf', 'key term - SF as published'), ('kt', 'role', 'key term - role term'))}


EXT = (('madi_indep', 'MaDI use cases'), ('madi_products', 'MaDI products'), ('goby', 'GOBY'), ('hdxsm', 'HDXSM'),
       ('gdcsm', 'GDC-SM'))


def _contrast_ext(rows, x, y, corpus):
    """recall_cells.contrast with the 'cluster' design, but with the scenario -> cluster maps cached in
    results/derived/clusters.json (the study derives them from the raw benchmark manifests)."""
    import json
    import numpy as np
    from score_final import cell_values, macro_contrast
    from explore_x81 import cluster_ci
    cl_all = json.loads((DERIVED / 'clusters.json').read_text(encoding='utf-8'))[corpus]
    keys = [x, y, x + '_R', y + '_R']
    rows = RC.clean(rows, keys)
    inc, means, _ = cell_values(rows, keys)
    out = []
    for a, b in ((x, y), (x + '_R', y + '_R')):
        cl = {s: cl_all[s] for s in sorted({k[2] for k in means})}
        if len(set(cl.values())) >= 5:
            out.append(cluster_ci(means, a, b, cl, np.random.default_rng(RC.SEED)))
        else:
            out.append(macro_contrast(means, a, b, np.random.default_rng(RC.SEED)))
    return out[0], out[1]


def keyterm_external():
    """Per external corpus: key term - key graph, and the key lift as a term minus the key lift in the graph."""
    rows = _num(_rows('recall/w4_keyterm_ext_rows.csv'),
                ['kt', 'kt_R', 'key', 'key_R', 'sf', 'sf_R', 'role', 'role_R', 'kinc', 'kinc_R', 'ginc', 'ginc_R',
                 'both_keys'])
    out = {}
    for rep, lab in EXT:
        rr = [r for r in rows if r['corpus'] == rep]
        out[lab] = {'key term - key graph': _contrast_ext(rr, 'kt', 'key', rep),
                    'key lift: term - graph': _contrast_ext(rr, 'kinc', 'ginc', rep),
                    '_both': (int(sum(r['both_keys'] for r in rr if r['seed'] == rr[0]['seed'])),
                              len({(r['dataset'], r['scenario']) for r in rr}))}
    return out


def fmt(t):
    return '%+.3f [%+.3f, %+.3f]' % t
