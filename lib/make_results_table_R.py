# Ported from the study's manuscript folder for this artifact; paths point to results/ and the value folder.
"""Table II on expected Recall@GT (2026-11-06), from figures/recall/*.csv (recall_cells.py, recall_multitable.py,
recall_roles.py, recall_er10.py). Keeps the paper's layout and conventions: cells without an interval in the J_tf
table stay point estimates; R/A mark comparisons registered or specified in advance on J_tf; NI is the registered J_tf
non-inferiority verdict; italic: not information-matched. Writes figures/results_table_R.tex and ROWS_R for figures.

Run:  ../../harness/.venv-bdi/Scripts/python.exe make_results_table_R.py
"""
from paths import DERIVED, OUT, STUDY, STUDY_CODE, BASELINES_CODE  # noqa: E402,F401
import csv
import math
import os

from make_results_table import ROWS, NA, NR, NI_R, COLS, SOURCES

HERE = os.path.dirname(os.path.abspath(__file__))
REC = str(DERIVED / 'recall')
COND_CI = ('MaDI products', 'MaDI use cases')


def load():
    vals = {}
    with open(os.path.join(REC, 'recall_cells.csv'), encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r.get('R') not in ('', None):
                vals[(r['corpus'], r['col'])] = tuple(float(r[k]) if r[k] not in ('', 'nan') else math.nan
                                                      for k in ('R', 'R_lo', 'R_hi'))
    for name in ('recall_multitable.csv', 'recall_roles.csv', 'recall_er10.csv'):
        with open(os.path.join(REC, name), encoding='utf8') as f:
            for r in csv.DictReader(f):
                vals[(r['corpus'], r['col'])] = (float(r['R']), float(r['R_lo']), float(r['R_hi']))
    return vals


def ni_cells():
    """Recall@GT of the cells that were only a registered J_tf non-inferiority test (roles on L/MB and T2D), rescored
    on that test's own runs and panel (figures/loso/panel_cells.csv, second agent's Brief A; 2026-11-06, review 5 C1)."""
    p = str(DERIVED / 'loso' / 'panel_cells.csv')
    out = {}
    with open(p, encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['metric'] == 'R' and r['note'].startswith('NI-only'):
                out[(r['corpus'], r['col'])] = dict(e=float(r['estimate']), l=float(r['lo']), h=float(r['hi']),
                                                    st='R', nm=False)
    return out


def tu_panel():
    """Own-panel Recall@GT intervals of two-update cells that Table II printed without one although the panel
    recomputation has one (mixed DBs, 3-DB; review 8 W9). MaDI and the schema pairs have none."""
    p = str(DERIVED / 'loso' / 'panel_cells.csv')
    out = {}
    with open(p, encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['metric'] == 'R' and r['col'] == 'tu' and r['lo'] not in ('', 'nan') and r['hi'] not in ('', 'nan'):
                out[r['corpus']] = (float(r['estimate']), float(r['lo']), float(r['hi']))
    return out


def rows_R():
    vals = load()
    tup = tu_panel()
    ni = ni_cells()
    out = []
    for label, g, cells in ROWS:
        new = {}
        for k in COLS:
            c = cells[k]
            if c == NI_R and (label, k) in ni:
                new[k] = ni[(label, k)]
                continue
            if not isinstance(c, dict):
                new[k] = c
                continue
            v = vals.get((label, k))
            if v is None:
                new[k] = NR
                continue
            # review 7 W3: one interval rule per corpus. The MaDI datatype and class cells have the same conditional
            # scenario-bootstrap interval as their shaded roles cells; the old J_tf table had simply not printed it.
            has_ci = ((c['l'] is not None or (label in COND_CI and k in ('dt', 'cl')))
                      and math.isfinite(v[1]) and math.isfinite(v[2]))
            new[k] = dict(e=v[0], l=v[1] if has_ci else None, h=v[2] if has_ci else None, st=c['st'], nm=c['nm'])
            if k == 'tu' and not has_ci and label in tup:
                e, lo, hi = tup[label]
                new[k] = dict(e=e, l=lo, h=hi, st=c['st'], nm=c['nm'])
        out.append((label, g, new))
    return out


ROWS_R = rows_R()


def show(x):
    if x in (NA, NR):
        return '--'
    if x == NI_R:
        return '$\\dagger$'      # only the registered J_tf non-inferiority test (Table tab:registered)
    s = ('%+.3f' % x['e']).replace('-', '$-$')
    sig = x['l'] is not None and (x['l'] > 0 or x['h'] < 0)
    if x['l'] is None:
        s = '(' + s + ')'
    if x['nm']:
        s = '\\textit{' + s + '}'
    # registration status (x['st']) is shown in Table tab:registered, not here (2026-11-06, plan item 3)
    if sig and not x['nm']:
        s = ('\\cellcolor{oneahead}' if x['e'] > 0 else '\\cellcolor{sfahead}') + s
    return s


def _panel_flips():
    """Cells whose Recall@GT sign changes on the initial matchers common to all corpora of the column
    (figures/loso/panel_cells.csv, second agent's Brief A, 2026-11-06)."""
    p = str(DERIVED / 'loso' / 'panel_cells.csv')
    if not os.path.exists(p):
        return set()
    with open(p, encoding='utf8') as f:
        return {(r['corpus'], r['col']) for r in csv.DictReader(f)
                if r['metric'] == 'R' and r['direction_changed'] == 'True'}


FLIPS = _panel_flips()


_JDIFF = None


def jdiff():
    """Cells whose shading side differs in J_tf (w2_dual.DIFF; review 5, W2)."""
    global _JDIFF
    if _JDIFF is None:
        from w2_dual import DIFF
        _JDIFF = DIFF()
    return _JDIFF


def mark(label, k):
    m = [s for s, on in (('p', (label, k) in FLIPS), ('j', (label, k) in jdiff())) if on]
    return '$^{%s}$' % ','.join(m) if m else ''


def counts(rows=None):
    rows = rows or ROWS_R
    res = {}
    for g in (0, 1):
        t = s = n = 0
        for _, gg, cells in rows:
            if gg != g:
                continue
            for k in ('dt', 'cl', 'ro', 'tc'):
                x = cells[k]
                if isinstance(x, dict) and not x['nm'] and x['l'] is not None:
                    n += 1
                    t += x['l'] > 0
                    s += x['h'] < 0
        res[g] = (t, s, n)
    return res


def table():
    n_int = n_sig = 0
    for _, _, cells in ROWS_R:
        for k in COLS:
            x = cells[k]
            if isinstance(x, dict) and x['l'] is not None and not x['nm']:
                n_int += 1
                n_sig += x['l'] > 0 or x['h'] < 0
    out = [r'\begin{table}[t]',
           r'\caption{Results per corpus in expected Recall@GT: one-step term minus SF with the same information in its'
           r' graph, and SF after two updates minus converged SF.}',
           r'\label{tab:results}', r'\centering\scriptsize\setlength{\tabcolsep}{1.6pt}',
           r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}', r'\toprule',
           r'& & \multicolumn{4}{c}{One-step term $-$ SF} & Two upd.\\', r'\cmidrule(lr){3-6}',
           r'Corpus & Src. & Datatype & Class & Roles & Table ctx. & $-$ conv.\\', r'\midrule',
           r'\multicolumn{7}{@{}l}{\emph{Built from one source}}\\']
    for i, (label, g, cells) in enumerate(ROWS_R):
        if i == 8:
            out.append(r'\addlinespace[2pt]')
            out.append(r'\multicolumn{7}{@{}l}{\emph{Produced or designed independently}}\\')
        out.append(label + ' & %d & ' % SOURCES[i] + ' & '.join(show(cells[k]) + mark(label, k) for k in COLS)
                   + r'\\')
    out += [r'\bottomrule', r'\end{tabular*}\\[2pt]',
            r'\parbox{\linewidth}{\scriptsize Shaded: 95\% interval excludes 0 (information-matched cells); the sign shows which side is ahead'
            r' (\colorbox{oneahead}{term or two updates}, \colorbox{sfahead}{SF}).'
            r' Src.: independent sources (Table~\ref{tab:corpora}); with fewer than five, intervals are conditional on'
            r' them. Parentheses: no interval. All values exploratory; registered and pre-specified comparisons and'
            r' their verdicts: Table~\ref{tab:registered}. Roles:'
            r' against the top-1 key graph (mixed DBs: the best constraint graph; L/MB and T2D: the registered'
            r" test's runs). Italic: not information-matched"
            r' (no table organization). Valentine class cells: earlier copy. $p$: sign changes on the 9 initial'
            r' matchers common to all corpora (supplement). $j$: shading differs in $\Jtf$. --: not run or not applicable.'
            r' Intervals, $\Jtf$ values and the pooling over source groups: supplement.}',
            r'\end{table}']
    return '\n'.join(out)


def supp_table():
    """Supplement: every Table II cell with its Recall@GT interval and the paper's J_tf value and interval."""
    names = {'dt': 'Datatype', 'cl': 'Class', 'ro': 'Roles', 'tc': 'Table ctx.', 'tu': 'Two upd. $-$ conv.'}
    f = lambda v: ('%+.3f' % v).replace('-', '$-$')
    out = ['% Generated by make_results_table_R.py (2026-11-06); EXPLORATORY in Recall@GT; do not edit.',
           r'\begin{table*}[t]', r'\caption{The paper''s results table with intervals: expected Recall@GT (primary) and'
           r' $\Jtf$ (as registered or first reported) per corpus and comparison.}', r'\label{supp:tab-recall}',
           r'\centering\scriptsize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}llrr@{}}', r'\toprule',
           r'Corpus & Comparison & Recall@GT [95\%] & $\Jtf$ [95\%]\\', r'\midrule']
    for (label, g, cr), (_, _, cj) in zip(ROWS_R, ROWS):
        for k in COLS:
            r, j = cr[k], cj[k]
            if not isinstance(j, dict):
                continue
            rs = '--' if not isinstance(r, dict) else (f(r['e']) + ('' if r['l'] is None else
                                                                    ' [%s, %s]' % (f(r['l']), f(r['h']))))
            js = f(j['e']) + ('' if j['l'] is None else ' [%s, %s]' % (f(j['l']), f(j['h'])))
            out.append('%s & %s%s & %s & %s\\\\' % (label, names[k], ' (n.m.)' if j['nm'] else '', rs, js))
    out += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip\raggedright\scriptsize Recall@GT values come from the'
            r' same stored scores, matcher panels, weights and bootstrap designs as the $\Jtf$ values, which the'
            r' recomputation reproduces to within 0.0015 except the Valentine class cells (earlier copy of Valentine)'
            r' and the mixed-database two-update cells (another matcher panel); n.m.: not information-matched.',
            r'\end{table*}']
    return '\n'.join(out)


if __name__ == '__main__':
    t = table()
    c = counts()
    print(t)
    print('counts (term ahead, SF ahead, with interval):', c)
