# Ported from the study's manuscript folder for this artifact; paths point to results/ and the value folder.
"""Main-text table of the registered tests and the two-update rule specified in advance (2026-11-06; plan item 3).

Each row gives the comparison exactly as registered: its metric (J_tf, or Recall@GT where the registration fixed it),
estimate, 95% interval and verdict, all read from the generated macro files figures/*_numbers.tex (no number is typed
here). The last column is the expected Recall@GT value of the same comparison, which is exploratory for J_tf rows:
- from Table II's Recall@GT rows (make_results_table_R.ROWS_R) where Table II has the cell;
- from the registered reports' descriptive gains (gain of the term minus gain of SF, same panel and weights) for the
  3-DB class terms;
- from the stored R part of the registered contrast for the Valentine class test.

Writes figures/registered_table.tex.  Run:  ../../harness/.venv-bdi/Scripts/python.exe make_registered_table.py
"""
from paths import DERIVED, OUT, STUDY, STUDY_CODE, BASELINES_CODE  # noqa: E402,F401
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAT = re.compile(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}')
M = {}
for f in sorted(OUT.glob('*_numbers.tex')):
    M.update(PAT.findall(f.read_text(encoding='utf8')))


def num(k):
    return float(M[k])


def fmt(v, d=3):
    s = ('%+.' + str(d) + 'f') % v
    return s.replace('-', '$-$')


def ci(p, d=3, est='Est', lo='Lo', hi='Hi'):
    return '%s [%s, %s]' % (fmt(num(p + est), d), fmt(num(p + lo), d), fmt(num(p + hi), d))


def ci2(e, lo, hi, d=3):
    return '%s [%s, %s]' % (fmt(num(e), d), fmt(num(lo), d), fmt(num(hi), d))


from make_results_table_R import ROWS_R  # noqa: E402
RR = {lab: cells for lab, _, cells in ROWS_R}


def rec(label, col):
    c = RR[label][col]
    return fmt(c['e']) if isinstance(c, dict) else '--'


VERD = {'confirmed': 'yes', 'CONFIRMED': 'yes', 'not confirmed': 'no', 'not supported': 'no',
        'non-inferior': 'NI', 'superior': 'sup.', 'table context suffices': 'yes'}


def v(k):
    return VERD[M[k]]


# W6 (2026-11-08) and review 6 W3/W4: test type next to the metric (S superiority, NI non-inferiority with margin
# -0.01, 3w three-way); verdicts in parentheses are reported, not decided, by their registration; ^h: confirmed but
# not after Holm over all decided hypotheses with an interval (w6_registry.py). Amendment 7 decides on Lahman/MB only
# (T2D was not profiled before its SF runs), so its T2D rows are exploratory and not here. Amendment 11's T2D rows show
# the registered (frozen-parse) output; ^k: Erratum 1's corrected parse moves each estimate by +0.001 (supplement).
from w6_registry import holm_lookup, registered_t2d_roles, rows as _reg_rows, summary as _reg_summary  # noqa: E402
HOLM = holm_lookup()
RT = registered_t2d_roles()
SUMMARY = _reg_summary(_reg_rows())


def hv(base, text):
    """Registered verdict text, with ^h if a decided confirmation does not survive Holm."""
    return text + ('$^h$' if text in ('yes', 'NI', 'sup.') and base in HOLM and not HOLM[base] else '')


def rep(text):
    return '(' + text + ')'


def rt(h, d=3):
    e, lo, hi, _ = RT[h]
    return '%s [%s, %s]' % (fmt(e, d), fmt(lo, d), fmt(hi, d))


def rtv(h, ni=False):
    ok = RT[h][3] == 'confirmed'
    return ('NI' if ni else 'yes') if ok else 'no'


# (comparison, corpus, metric and type, estimate [95%], verdict, Recall@GT)
REG = [
    ('Datatype $-$ SF', 'Mixed, read.', 'J S', ci('finPlFOne'), hv('finPlFOne', v('finPlFOneVerdict')),
     rec('Mixed DBs (readable)', 'dt')),
    ('', 'Mixed, rec.', 'J S', ci('finOpFOne'), rep(v('finOpFOneVerdict')), rec('Mixed DBs (recoded)', 'dt')),
    # review 8 W5: Amendment 7's P3 is one joint prediction (L/MB and T2D); show its single verdict on both rows
    ('', 'L/MB$^g$', 'J S', ci('etwoPThree'),
     'no' if M['ttdPThreeConsistent'] != 'true' else hv('etwoPThree', v('etwoPThreeVerdict')),
     rec('Lahman/MusicBrainz', 'dt')),
    ('', 'T2D$^g$', 'J S', ci('ttdPThree'), 'yes' if M['ttdPThreeConsistent'] == 'true' else 'no',
     rec('T2D web tables', 'dt')),
    ('', '3-DB$^c$', 'J S', ci2('amfHThree', 'amfHThreeLo', 'amfHThreeHi'), v('amfHThreeVerdict'),
     rec('3-DB multi-table', 'dt')),
    ('Class $-$ SF$^d$', '3-DB', 'J S', ci2('amfHOne', 'amfHOneLo', 'amfHOneHi'), hv('amfHOne', v('amfHOneVerdict')),
     fmt(num('amfApR') - num('amfSfR'))),
    ('Real $-$ shuffled cl.$^e$', 'Valentine', 'J S', ci('cfPlShapeVsShapeShuf'),
     hv('cfPlShapeVsShapeShuf', v('cfPlShapeVsShapeShufVerdict')), fmt(num('cfPlShapeVsShapeShufR'))),
    ('Roles $-$ constr.\\ graph$^f$', 'Mixed, read.', 'J NI', ci('finPlFTwoB'),
     hv('finPlFTwoB', 'NI' if v('finPlFTwoBVerdict') == 'yes' else 'no'), rec('Mixed DBs (readable)', 'ro')),
    ('', 'Mixed, rec.', 'J NI', ci('finOpFTwoB'), rep('NI' if v('finOpFTwoBVerdict') == 'yes' else 'no'),
     rec('Mixed DBs (recoded)', 'ro')),
    ('', 'T2D$^i$', 'J NI', rt('HR4a'), rtv('HR4a', ni=True), rec('T2D web tables', 'ro')),
    ('', 'L/MB', 'J NI', ci('rrCtlPlHFourA'), rep('NI' if v('rrCtlPlHFourAVerdict') == 'yes' else 'no'),
     rec('Lahman/MusicBrainz', 'ro')),
    ('Roles $-$ SF', 'T2D$^i$', 'J S', rt('HR2'), rtv('HR2'), '--'),
    ('Roles $-$ FD graph', 'T2D$^i$', 'J S', rt('HR3'), rtv('HR3'), '--'),
    ('Table ctx.\\ $-$ SF', '3-DB', 'J 3w', fmt(-num('tcHOneEst')) + ' [%s, %s]' % (fmt(-num('tcHOneHi')),
                                                                                    fmt(-num('tcHOneLo'))),
     hv('tcHOne', 'suff.'), rec('3-DB multi-table', 'tc')),
    ('', 'Indep.\\ pairs', 'J S', ci('finAFFive'), v('finAFFiveVerdict'), rec('Indep.\\ schema pairs', 'tc')),
    ('FD graph $-$ SF', 'Mixed, read.', 'R S', ci('finPlFFour'), hv('finPlFFour', v('finPlFFourVerdict')), ''),
    ('', 'L/MB, read.', 'R S', ci('etwoPTwoPl'), hv('etwoPTwoPl', v('etwoPTwoPlVerdict')), ''),
    ('', 'L/MB, rec.', 'J S', ci('etwoPTwoMd'), hv('etwoPTwoMd', v('etwoPTwoMdVerdict')), '--'),
    ('', 'T2D$^g$', 'R S', ci('ttdPTwo'), hv('ttdPTwo', 'yes' if M['ttdPTwoConsistent'] == 'true' else 'no'), ''),
    ('$\\rho$(quality, gain)', 'Mixed, read.', 'R S', ci('finPlFThree', d=2), v('finPlFThreeVerdict'), ''),
    ('', 'T2D$^g$', 'R S', ci('ttdPOne', d=2), hv('ttdPOne', 'yes' if M['ttdPOneConsistent'] == 'true' else 'no'), ''),
    ('', 'L/MB, read.', 'R S', ci('etwoPOnePl', d=2), rep(v('etwoPOnePlVerdict')), ''),
]   # the registered tuning test set is in Table IV (tab:evaluation), which marks it as registered
ADV = [
    ('Two upd.\\ $-$ conv.', 'T2D', 'J NI', ci('svTtd'), v('svTtdVerdict'), rec('T2D web tables', 'tu')),
    ('', 'L/MB', 'J NI', ci('svLmS'), v('svLmSVerdict'), rec('Lahman/MusicBrainz', 'tu')),
    ('', 'Synthea', 'J NI', ci('svSyn'), v('svSynVerdict'), rec('Synthea$\\to$OMOP', 'tu')),
    ('', 'L/MB mt.', 'J NI', ci('svLmM'), v('svLmMVerdict'), rec('Lahman/MB multi-table', 'tu')),
]
assert M['lbxHTwoPos'] and 'confirmed' in (STUDY / 'tuning' / 'results' / 'lockbox_plain' / 'hl_report.md').read_text(encoding='utf8')


def table():
    out = ['% Generated by make_registered_table.py from figures/*_numbers.tex and make_results_table_R.ROWS_R;'
           ' do not edit.',
           r'\begin{table}[t]',
           r'\caption{Selected registered tests (verdict as registered; all registered hypotheses: supplement) and the'
           r' two-update rule specified in advance. R@GT: expected Recall@GT, exploratory for rows registered on'
           r' $\Jtf$.}',
           r'\label{tab:registered}',
           r'\centering\scriptsize\setlength{\tabcolsep}{1.5pt}',
           r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llcrcr@{}}',
           r'\toprule',
           r'Comparison & Corpus & Test$^a$ & Estimate [95\%] & Ver.$^b$ & R@GT\\',
           r'\midrule']
    for rows in (REG, ADV):
        if rows is ADV:
            out.append(r'\addlinespace[2pt]')
            out.append(r'\multicolumn{6}{@{}l}{\emph{Specified in advance} (margin $-0.01$ on $\Jtf$)}\\')
        for r in rows:
            out.append(' & '.join(r) + r'\\')
    out += [r'\bottomrule', r'\end{tabular*}\\[2pt]',
            r'\parbox{\linewidth}{\scriptsize $^a$Registered metric (J: $\Jtf$; R: Recall@GT, then no last column) and'
            r' type (S: superiority; NI: non-inferiority, margin $-0.01$; 3w: three-way, no direction).'
            r' $^b$Confirmed; NI: non-inferior; sup.: superior; suff.: table context suffices; parentheses: reported,'
            r' not decided, by the registration. $^h$Not after Holm over all %d decided hypotheses with an interval'
            r' (supplement). $^g$Amendment~7 decides its datatype prediction jointly on L/MB and T2D; not confirmed,'
            r' as T2D fails though L/MB meets its bound. Its T2D rows were scored by a separate scorer with the'
            r' registered design, on 175 of 178 scenarios (3 without usable profiles), skipping non-finite initial'
            r' scores; before the data, a reading rule and our decision had made T2D exploratory (verdicts under'
            r' both readings: supplement).'
            r' $^i$Registered FD parse; Erratum~1 to Amendment~11 (corrected parse): %s, %s, %s from'
            r' top. Read./rec.: readable/recoded; L/MB: Lahman/MusicBrainz; mt.: multi-table.'
            r' $^c$Not information-matched. $^d$Pattern classes, against SF as published (Table~\ref{tab:results}:'
            r' against SF with class nodes); semantic: %s [%s, %s]. $^e$In SF; earlier copy of Valentine.'
            r' $^f$Mixed: best constraint graph, else top-1 key graph.}'
            % (SUMMARY['n'], fmt(num('rrTTwoDHFourAEst')), fmt(num('rrTTwoDHTwoEst')), fmt(num('rrTTwoDHThreeEst')),
               fmt(num('amfHTwo')), fmt(num('amfHTwoLo')), fmt(num('amfHTwoHi'))),
            r'\end{table}']
    return '\n'.join(out)


if __name__ == '__main__':
    t = table()
    print(t)
