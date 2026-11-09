"""E06: does SF's key graph add anything that the same key information as a one-step term does not?

The top-1 key graph links each schema's best key (UCC) to a shared key-size node. The experiment checks
(a) whether its lead over SF as published depends on column order (all columns permuted on 549 Valentine scenarios),
(b) how the lossy role term compares with it, and (c) a one-step term with the key graph's key membership:
datatype term + gamma * [both columns in their schema's key], with Schematch's key choice reimplemented.

Paper: Section V-C and the small key-term table. Inputs: results/derived/recall/perm_keygraph_*_rows.csv,
w4_keyterm_rows.csv(.gz), w4_keyterm_ext_rows.csv, results/derived/loso/role_recall_*.csv and the role term's
J_tf scores (results/sf_study/results/<regime>549/deproles_scores.csv). These rows were produced from the stored
similarity matrices by study_code/later_analyses/ (perm_keygraph549.py, w4_keyterm.py, w4_keyterm_madi.py).
Runtime: a few minutes.

    python experiments/E06_key_graph_controls.py
"""
import sys

from _common import V, report

V()
import keygraph as K  # noqa: E402

f = K.fmt
checked, sec = {}, []
for reg, lab in (('plain549', 'readable'), ('md549', 'recoded')):
    p = K.permutation(reg)
    items = []
    for k, (j, r) in ((k, v) for k, v in p.items() if not k.startswith('_')):
        items.append((f'{k}: Recall@GT', f(r)))
        items.append((f'{k}: J_tf', f(j)))
        checked[f'{lab} {k} R'] = f(r)
    items.append(('share of key-graph outputs changed by the permutation', '%.1f%%' % (100 * p['_changed'])))
    sec.append((f'(a) Key graph under column permutation, Valentine {lab}', items))
for reg, regime, lab in (('plain549', 'plain', 'readable'), ('md549', 'md5', 'recoded')):
    items = []
    for k, (j, r) in K.roles_vs_key(reg, regime).items():
        items += [(f'{k}: Recall@GT', f(r)), (f'{k}: J_tf', f(j))]
        checked[f'{lab} {k} R'] = f(r)
    sec.append((f'(b) Role term against the key graph, Valentine {lab}', items))
for reg, lab in (('plain549', 'readable'), ('md549', 'recoded')):
    items = []
    for k, (j, r) in K.keyterm_valentine(reg).items():
        items += [(f'{k}: Recall@GT', f(r)), (f'{k}: J_tf', f(j))]
        checked[f'{lab} {k} R'] = f(r)
    sec.append((f'(c) Key-membership term, Valentine {lab}', items))
ext = K.keyterm_external()
md = ['| corpus | keys on both sides | key term - key graph (Recall@GT) | key lift as term - key lift in graph '
      '(Recall@GT) | same in J_tf |', '|---|---|---|---|---|']
for lab, d in ext.items():
    (j1, r1), (j2, r2) = d['key term - key graph'], d['key lift: term - graph']
    md.append(f"| {lab} | {d['_both'][0]} of {d['_both'][1]} | {f(r1)} | {f(r2)} | {f(j2)} |")
    checked[f'{lab} key lift difference R'] = f(r2)
sec.append(('(c) Key-membership term, external corpora (weight transferred from Valentine)', '\n'.join(md) +
            '\n\nThe last two columns compare what key membership adds as a one-step term (over the datatype term) '
            'with what it adds inside SF\'s graph (key graph over SF as published).'))
sys.exit(report('E06', 'Key graph against the same key information as a one-step term',
                'Is the key graph\'s lead over SF propagation, column order, or key information that a one-step term '
                'can use as well?', sec, checked,
                note='With a key on one side only (every OpenData scenario, most of GOBY) the key graph equals SF.'))
