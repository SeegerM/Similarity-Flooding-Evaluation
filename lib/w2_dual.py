# Ported from the study's manuscript folder for this artifact; paths point to results/ and the value folder.
"""Review 5, W2: every headline claim in both metrics (2026-11-06; EXPLORATORY bookkeeping, no new scoring).

Recall@GT became the primary metric on 2026-11-06, after the registered analyses (FINDINGS §4-5). This script collects
each headline claim of the paper in expected Recall@GT and in J_tf from stored outputs and flags where they disagree:
- Table II cells: shading side (term / SF / none) in Recall@GT (make_results_table_R.ROWS_R) and J_tf
  (make_results_table.ROWS; the two NI-only role cells take the registered test's estimate) -> DIFF set, used by
  make_results_table_R.py to mark cells (superscript j);
- key graph minus SF as published (figures/recall/recall_keygraph.csv), its column-permutation check
  (figures/recall/perm_keygraph_*549_rows.csv), leave-out-ChEMBL (figures/loso/loso_cells.csv), source-group means
  (figures/recall/w1_pooling.md), SF's gain and rho on Valentine (figures/rq1_numbers.tex).
Writes figures/recall/w2_dual.md.

Run:  ../../harness/.venv-bdi/Scripts/python.exe w2_dual.py
"""
from paths import DERIVED, OUT, STUDY, STUDY_CODE, BASELINES_CODE  # noqa: E402,F401
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COLS = ('dt', 'cl', 'ro', 'tc', 'tu')
NAMES = {'dt': 'Datatype', 'cl': 'Class', 'ro': 'Roles', 'tc': 'Table ctx.', 'tu': 'Two upd. - conv.'}


def side(c):
    if not isinstance(c, dict) or c['l'] is None or c.get('nm'):
        return None
    return 'term' if c['l'] > 0 else 'SF' if c['h'] < 0 else 'none'


def table2_diffs():
    from make_results_table import ROWS, NI_R
    from make_results_table_R import ROWS_R, COND_CI
    from w1_pooling import registered_role_J
    reg = registered_role_J()
    # unrounded J_tf bounds where the recomputation has them (ROWS holds macros rounded to 3 decimals, which hid
    # two verdict changes with a lower bound of +0.0003/+0.0004; review 6, W5)
    unr = {}
    for x in csv.DictReader(open(DERIVED / 'recall' / 'recall_cells.csv', encoding='utf8')):
        if x.get('J_lo') not in (None, '', 'nan') and x.get('J_hi') not in (None, '', 'nan'):
            unr[(x['corpus'], x['col'])] = (float(x['J']), float(x['J_lo']), float(x['J_hi']))
    out = []
    for (label, _, cr), (_, _, cj) in zip(ROWS_R, ROWS):
        for k in COLS:
            r, j = cr[k], cj[k]
            if j == NI_R and label in reg:
                j = reg[label]
            if isinstance(j, dict) and (label, k) in unr and (j['l'] is not None or label in COND_CI):
                e, lo, hi = unr[(label, k)]
                j = dict(j, l=lo, h=hi)
            sr, sj = side(r), side(j)
            if sr is None or sj is None:
                continue
            out.append((label, k, r, j, sr, sj))
    return out


def DIFF():
    """{(corpus, col)} whose shading side differs between Recall@GT and J_tf."""
    return {(l, k) for l, k, _, _, sr, sj in table2_diffs() if sr != sj}


def f(x):
    return '%+.3f' % x


def ci(c):
    return f(c['e']) + ('' if c['l'] is None else ' [%s, %s]' % (f(c['l']), f(c['h'])))


def main():
    L = ['# W2: headline claims in both metrics (2026-11-06)', '',
         'Recall@GT = expected Recall@GT (primary since 2026-11-06); J_tf = registered metric.', '',
         '## Table II cells whose shading side differs', '', '| corpus | comparison | Recall@GT | J_tf |', '|---|---|---|---|']
    rows = table2_diffs()
    for l, k, r, j, sr, sj in rows:
        if sr != sj:
            L.append(f'| {l} | {NAMES[k]} | {ci(r)} ({sr}) | {ci(j)} ({sj}) |')
    same = sum(sr == sj for *_, sr, sj in rows)
    L += ['', f'{same} of {len(rows)} cells with an interval in both metrics have the same shading side.', '']

    L += ['## Key graph minus SF as published', '', '| corpus | Recall@GT | J_tf |', '|---|---|---|']
    for r in csv.DictReader(open(DERIVED / 'recall' / 'recall_keygraph.csv', encoding='utf8')):
        L.append(f"| {r['corpus']} | {f(float(r['R']))} [{f(float(r['R_lo']))}, {f(float(r['R_hi']))}] | "
                 f"{f(float(r['J']))} [{f(float(r['J_lo']))}, {f(float(r['J_hi']))}] |")
    for reg in ('plain549', 'md549'):
        t = (DERIVED / 'recall' / f'perm_keygraph_{reg}.md').read_text(encoding='utf8')
        m = re.search(r'\| key - SF \(permuted columns\) \| ([^|]+) \| ([^|]+) \|', t)
        L.append(f'| Valentine {reg}, columns permuted | {m.group(2).strip()} | {m.group(1).strip()} |')
    L.append('')

    L += ['## Valentine readable without ChEMBL (leave-one-source-out)', '', '| comparison | Recall@GT | J_tf |',
          '|---|---|---|']
    lo = {}
    for r in csv.DictReader(open(DERIVED / 'loso' / 'loso_cells.csv', encoding='utf8')):
        if r['corpus'] == 'Valentine (readable)' and r['left_out'].startswith('ChEMBL'):
            lo[(r['col'], r['metric'])] = (float(r['estimate']), float(r['lo']), float(r['hi']))
    for k in ('dt', 'cl', 'ro', 'tu'):
        a, b = lo[(k, 'R')], lo[(k, 'J')]
        L.append(f'| {NAMES[k]} | {f(a[0])} [{f(a[1])}, {f(a[2])}] | {f(b[0])} [{f(b[1])}, {f(b[2])}] |')
    L.append('')

    t = (DERIVED / 'recall' / 'w1_pooling.md').read_text(encoding='utf8')
    L += ['## Source-group means (w1_pooling.md)', '']
    L += [l for l in t.splitlines() if l.startswith('- ') and 'all cells' in l]
    L.append('')

    m = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', (OUT / 'rq1_numbers.tex').read_text(
        encoding='utf8')))
    L += ['## SF vs its initial matcher, readable Valentine', '',
          f"- rho(initial quality, gain): Recall@GT {m['rqOneRhoRexp']}, Hungarian F1 (random ties) {m['rqOneRhoFHr']}",
          '- gain: Table IV gives every metric.', '']
    out = OUT / 'w2_dual.md'
    out.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
