# Ported from the study's manuscript folder for this artifact; paths point to results/ and the value folder.
"""Proposal (2026-11-05): a PRISMA-style results table in place of the forest plots (Figs. 4 and 5).
Reads figures/*_numbers.tex and writes figures/proposal_results_table.tex with plain numbers."""
from paths import DERIVED, OUT, STUDY, STUDY_CODE, BASELINES_CODE  # noqa: E402,F401
import os
from make_new_figures import V, DER, num, fmt

HERE = os.path.dirname(os.path.abspath(__file__))


def c(est, lo=None, hi=None, st='', nm=False):
    if est in DER:
        e, l, h = DER[est]
    else:
        e = num(est)
        l = num(lo) if lo else None
        h = num(hi) if hi else None
    return dict(e=e, l=l, h=h, st=st, nm=nm)


NA, NR = 'na', 'nr'
NI_R = 'niR'
COLS = ['dt', 'cl', 'ro', 'tc', 'tu']
ROWS = [
    ('Valentine (readable)', 0, dict(dt=c('vA'), cl=c('cfPlApVsShapeEst', 'cfPlApVsShapeLo', 'cfPlApVsShapeHi'),
                                     ro=c('drPlRoleVsUccTopEst', 'drPlRoleVsUccTopLo', 'drPlRoleVsUccTopHi'), tc=NA,
                                     tu=c('ldTwoConvJValEst', 'ldTwoConvJValLo', 'ldTwoConvJValHi'))),
    ('Valentine (recoded)', 0, dict(dt=c('vB'), cl=c('cfMdApVsShapeEst', 'cfMdApVsShapeLo', 'cfMdApVsShapeHi'),
                                    ro=c('drMdRoleVsUccTopEst', 'drMdRoleVsUccTopLo', 'drMdRoleVsUccTopHi'), tc=NA,
                                    tu=c('ldTwoConvJVMdEst', 'ldTwoConvJVMdLo', 'ldTwoConvJVMdHi'))),
    ('Mixed DBs (readable)', 0, dict(dt=c('finPlFOneEst', 'finPlFOneLo', 'finPlFOneHi', 'R'), cl=c('cxShapeAddCtl'),
                                     ro=c('finPlFTwoBEst', 'finPlFTwoBLo', 'finPlFTwoBHi', 'R*'), tc=NA,
                                     tu=c('rlxTwoConvCtlPl'))),
    ('Mixed DBs (recoded)', 0, dict(dt=c('finOpFOneEst', 'finOpFOneLo', 'finOpFOneHi', 'R'), cl=NR,
                                    ro=c('finOpFTwoBEst', 'finOpFTwoBLo', 'finOpFTwoBHi', 'R*'), tc=NA,
                                    tu=c('rlxTwoConvCtlOp'))),
    ('Lahman/MusicBrainz', 0, dict(dt=c('etwoPThreeEst', 'etwoPThreeLo', 'etwoPThreeHi', 'R'), cl=NR, ro=NI_R, tc=NA,
                                   tu=c('svLmSEst', 'svLmSLo', 'svLmSHi', 'A'))),
    ('MaDI products', 0, dict(dt=c('xtMadiPAdEst'), cl=c('cxShapeAddMadiP'),
                              ro=c('xtMadiPRoleKeyEst', 'xtMadiPRoleKeyLo', 'xtMadiPRoleKeyHi'), tc=NA,
                              tu=c('ldTwoConvJMadiPEst'))),
    ('3-DB multi-table', 0, dict(dt=c('amfHThree', 'amfHThreeLo', 'amfHThreeHi', 'R', nm=True), cl=c('ertenApVsSfPEst', 'ertenApVsSfPLo', 'ertenApVsSfPHi'), ro=NR,
                                 tc=c('vL', st='R'), tu=c('rlxTwoConvInd'))),
    ('Lahman/MB multi-table', 0, dict(dt=c('mtMacroAdSfEst', 'mtMacroAdSfLo', 'mtMacroAdSfHi', nm=True), cl=NR, ro=NR,
                                      tc=c('mtMacroAdtSfEst', 'mtMacroAdtSfLo', 'mtMacroAdtSfHi'),
                                      tu=c('svLmMEst', 'svLmMLo', 'svLmMHi', 'A'))),
    ('T2D web tables', 1, dict(dt=c('ttdPThreeEst', 'ttdPThreeLo', 'ttdPThreeHi'), cl=NR, ro=NI_R, tc=NA,
                               tu=c('svTtdEst', 'svTtdLo', 'svTtdHi', 'A'))),
    ('Synthea$\\to$OMOP', 1, dict(dt=c('mtSynPairAdSfEst', 'mtSynPairAdSfLo', 'mtSynPairAdSfHi'), cl=NR, ro=NR, tc=NA,
                                  tu=c('svSynEst', 'svSynLo', 'svSynHi', 'A'))),
    ('GDC-SM', 1, dict(dt=c('xtGdcAdEst', 'xtGdcAdClLo', 'xtGdcAdClHi'),
                       cl=c('cxShapeAddGdc', 'cxShapeAddGdcClLo', 'cxShapeAddGdcClHi'),
                       ro=c('xtGdcRoleKeyEst', 'xtGdcRoleKeyClLo', 'xtGdcRoleKeyClHi'), tc=NA,
                       tu=c('ldTwoConvJGdcEst', 'ldTwoConvJGdcLo', 'ldTwoConvJGdcHi'))),
    ('HDXSM', 1, dict(dt=c('xtHdxAdEst', 'xtHdxAdClLo', 'xtHdxAdClHi'),
                      cl=c('cxShapeAddHdx', 'cxShapeAddHdxClLo', 'cxShapeAddHdxClHi'),
                      ro=c('xtHdxRoleKeyEst', 'xtHdxRoleKeyClLo', 'xtHdxRoleKeyClHi'), tc=NA,
                      tu=c('ldTwoConvJHdxEst', 'ldTwoConvJHdxLo', 'ldTwoConvJHdxHi'))),
    ('GOBY', 1, dict(dt=c('xtGobyAdEst', 'xtGobyAdClLo', 'xtGobyAdClHi'),
                     cl=c('cxShapeAddGoby', 'cxShapeAddGobyClLo', 'cxShapeAddGobyClHi'),
                     ro=c('xtGobyRoleKeyEst', 'xtGobyRoleKeyClLo', 'xtGobyRoleKeyClHi'), tc=NA,
                     tu=c('ldTwoConvJGobyEst', 'ldTwoConvJGobyLo', 'ldTwoConvJGobyHi'))),
    ('MaDI use cases', 1, dict(dt=c('xtMadiAdEst'), cl=c('cxShapeAddMadi'),
                               ro=c('xtMadiRoleKeyEst', 'xtMadiRoleKeyLo', 'xtMadiRoleKeyHi'), tc=NA,
                               tu=c('ldTwoConvJMadiEst'))),
    ('Indep.\\ schema pairs', 1, dict(dt=NR, cl=c('cxShapeAddIndep'), ro=NR,
                                      tc=c('finAFFiveEst', 'finAFFiveLo', 'finAFFiveHi', 'R'),
                                      tu=c('ldTwoConvJIndepEst'))),
]


def show(x):
    if x == NA:
        return 'n/a'
    if x == NR:
        return '--'
    if x == 'nm':
        return '\\textit{n.m.}'
    if x == NI_R:
        return 'NI$^{\\mathrm{R}}$'
    s = fmt(x['e'])
    if x['l'] is None:
        s = '(' + s + ')'
    elif (x['l'] > 0 or x['h'] < 0) and not x['nm']:
        s = '\\textbf{' + s + '}'
    if x['nm']:
        s = '\\textit{' + s + '}'
    st = x['st'].replace('*', '\\ast')
    if st:
        s += '$^{\\mathrm{%s}}$' % st if '\\ast' not in st else '$^{\\mathrm{R}\\ast}$'
    return s


def table():
    out = [r'\begin{table}[t]',
           r'\caption{Results per corpus in $\Jtf$: one-step term minus SF with the same information in its graph, and'
           r' SF after two updates minus converged SF (positive: the term, or two updates, is ahead).}',
           r'\label{tab:results}', r'\centering\scriptsize\setlength{\tabcolsep}{2.6pt}',
           r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
           r'& \multicolumn{4}{c}{One-step term $-$ SF} & Two upd.\\', r'\cmidrule(lr){2-5}',
           r'Corpus & Datatype & Class & Roles & Table ctx. & $-$ conv.\\', r'\midrule',
           r'\multicolumn{6}{@{}l}{\emph{Built from one source}}\\']
    for i, (label, g, cells) in enumerate(ROWS):
        if i == 8:
            out.append(r'\addlinespace[2pt]')
            out.append(r'\multicolumn{6}{@{}l}{\emph{Produced or designed independently}}\\')
        out.append(label + ' & ' + ' & '.join(show(cells[k]) for k in COLS) + r'\\')
    out += [r'\bottomrule', r'\end{tabular}\\[2pt]',
            r'\parbox{\linewidth}{\scriptsize Bold: 95\% interval excludes 0 (intervals: Figs.~\ref{fig:forest}'
            r' and~\ref{fig:ladder}, supplement); parentheses: point estimate without interval. R: registered;'
            r' A: specified in advance; others exploratory. NI: non-inferior to the key graph (registered).'
            r' $\ast$: against the best constraint graph. Italic: not information-matched (no table organization).'
            r' n/a: not applicable (single table); --: not run.}',
            r'\end{table}']
    return '\n'.join(out)


# ---------------------------------------------------------------- v2 (fig/table pass, 2026-11-05)
# Leading zeros as in Tables III-IV; a cell whose 95% interval excludes 0 is shaded by which side is ahead
# (blue: the one-step side, i.e. the term or two updates; green: SF / converged SF), matching Fig. 1's colours.
def show_v2(x):
    if x == NA:
        return ''
    if x == NR:
        return '--'
    if x == NI_R:
        return 'NI$^{\\mathrm{R}}$'
    s = ('%+.3f' % x['e']).replace('-', '$-$')
    sig = x['l'] is not None and (x['l'] > 0 or x['h'] < 0)
    if x['l'] is None:
        s = '(' + s + ')'
    if x['nm']:
        s = '\\textit{' + s + '}'
    st = x['st']
    if st == 'R*':
        s += '$^{\\mathrm{R}\\ast}$'
    elif st:
        s += '$^{\\mathrm{%s}}$' % st
    if sig and not x['nm']:
        s = ('\\cellcolor{oneahead}' if x['e'] > 0 else '\\cellcolor{sfahead}') + s
    return s


def table_v2():
    out = [r'\begin{table}[t]',
           r'\caption{Results per corpus in $\Jtf$: one-step term minus SF with the same information in its graph,'
           r' and SF after two updates minus converged SF.}',
           r'\label{tab:results}', r'\centering\scriptsize\setlength{\tabcolsep}{1.9pt}',
           r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrr@{}}', r'\toprule',
           r'& \multicolumn{4}{c}{One-step term $-$ SF} & Two upd.\\', r'\cmidrule(lr){2-5}',
           r'Corpus & Datatype & Class & Roles & Table ctx. & $-$ conv.\\', r'\midrule',
           r'\multicolumn{6}{@{}l}{\emph{Built from one source}}\\']
    for i, (label, g, cells) in enumerate(ROWS):
        if i == 8:
            out.append(r'\addlinespace[2pt]')
            out.append(r'\multicolumn{6}{@{}l}{\emph{Produced or designed independently}}\\')
        out.append(label + ' & ' + ' & '.join(show_v2(cells[k]) for k in COLS) + r'\\')
    out += [r'\bottomrule', r'\end{tabular*}\\[2pt]',
            r'\parbox{\linewidth}{\scriptsize Shaded: 95\% interval excludes 0, with the one-step side'
            r' (\colorbox{oneahead}{term or two updates}) or \colorbox{sfahead}{SF} ahead; intervals:'
            r' Fig.~\ref{fig:paired} and supplement. Parentheses: point estimate without interval.'
            r' R: registered; A: specified in advance; others exploratory. NI: non-inferior to the key graph'
            r' (registered). $\ast$: against the best constraint graph. Italic: not information-matched (no table'
            r' organization). Blank: single table; --: not run. Valentine class cells: earlier copy, fixed with the'
            r' registered class test (\S\ref{sec:methods}).}',
            r'\end{table}']
    return '\n'.join(out)


# Reviewer round (2026-11-05): an independent-source column (Table I), fewer legend symbols (no asterisk, one dash for
# "not run or not applicable"), and the number of shaded cells against what chance alone would shade.
SOURCES = [10, 10, 4, 4, 2, 1, 3, 2, 169, 1, 10, 32, 124, 4, 2]   # Table I, "Src.", in ROWS order


def show_v3(x):
    if x in (NA, NR):
        return '--'
    if x == NI_R:
        return 'NI$^{\\mathrm{R}}$'
    s = ('%+.3f' % x['e']).replace('-', '$-$')
    sig = x['l'] is not None and (x['l'] > 0 or x['h'] < 0)
    if x['l'] is None:
        s = '(' + s + ')'
    if x['nm']:
        s = '\\textit{' + s + '}'
    st = x['st'].replace('*', '')
    if st:
        s += '$^{\\mathrm{%s}}$' % st
    if sig and not x['nm']:
        s = ('\\cellcolor{oneahead}' if x['e'] > 0 else '\\cellcolor{sfahead}') + s
    return s


def shading_counts():
    n_int = n_sig = 0
    for _, _, cells in ROWS:
        for k in COLS:
            x = cells[k]
            if isinstance(x, dict) and x['l'] is not None and not x['nm']:
                n_int += 1
                n_sig += x['l'] > 0 or x['h'] < 0
    return n_int, n_sig


def table_v3():
    assert len(SOURCES) == len(ROWS)
    n_int, n_sig = shading_counts()
    out = [r'\begin{table}[t]',
           r'\caption{Results per corpus in $\Jtf$: one-step term minus SF with the same information in its graph,'
           r' and SF after two updates minus converged SF.}',
           r'\label{tab:results}', r'\centering\scriptsize\setlength{\tabcolsep}{1.6pt}',
           r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}', r'\toprule',
           r'& & \multicolumn{4}{c}{One-step term $-$ SF} & Two upd.\\', r'\cmidrule(lr){3-6}',
           r'Corpus & Src. & Datatype & Class & Roles & Table ctx. & $-$ conv.\\', r'\midrule',
           r'\multicolumn{7}{@{}l}{\emph{Built from one source}}\\']
    for i, (label, g, cells) in enumerate(ROWS):
        if i == 8:
            out.append(r'\addlinespace[2pt]')
            out.append(r'\multicolumn{7}{@{}l}{\emph{Produced or designed independently}}\\')
        out.append(label + ' & %d & ' % SOURCES[i] + ' & '.join(show_v3(cells[k]) for k in COLS) + r'\\')
    out += [r'\bottomrule', r'\end{tabular*}\\[2pt]',
            r'\parbox{\linewidth}{\scriptsize Shaded: 95\%% interval excludes 0; the sign shows which side is ahead'
            r' (\colorbox{oneahead}{term or two updates}, \colorbox{sfahead}{SF}). %d of the %d cells with an'
            r' interval are shaded; if no comparison had an effect and scenarios were independent, about %d would be. Src.: independent sources'
            r' (Table~\ref{tab:corpora}); with fewer than five, intervals are conditional on them. Parentheses: no'
            r' interval. R: registered; A: specified in advance; others exploratory. NI: non-inferior to the key'
            r' graph (registered). Roles: against the top-1 key graph (mixed DBs: the best constraint graph).'
            r' Italic: not information-matched (no table organization). --: not run or not applicable. Intervals:'
            r' supplement. Valentine class cells: earlier copy (\S\ref{sec:methods}).}' % (n_sig, n_int,
                                                                                       round(0.05 * n_int)),
            r'\end{table}']
    return '\n'.join(out)


if __name__ == '__main__':
    t3 = table_v3()
    open(os.path.join(HERE, 'figures', 'results_table_v3.tex'), 'w', encoding='utf8', newline='\n').write(t3 + '\n')
    print(t3)
    raise SystemExit(0)
    t = table()
    open(os.path.join(HERE, 'figures', 'proposal_results_table.tex'), 'w', encoding='utf8', newline='\n').write(t + '\n')
    t2 = table_v2()
    open(os.path.join(HERE, 'figures', 'results_table_v2.tex'), 'w', encoding='utf8', newline='\n').write(t2 + '\n')
    print(t2)
