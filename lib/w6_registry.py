# Ported from the study's manuscript folder for this artifact; paths point to results/ and the value folder.
"""Registry of every registered hypothesis of the study, with hypothesis types, registration times, hashes and a Holm
correction (review 5 W6 on 2026-11-08; extended after review 6 W3/W4 the same day). The verdicts are as registered;
the correction is EXPLORATORY.

Sources (no number is typed here):
- the six amendments behind the paper's Table III (5, 7, 8, 9, 11, 12): generated macro files
  figures/{amf,cf,tc,etwo,ttd,rr,fin}_numbers.tex; Amendment 11 from its registered output
  (tuning/results/rolesreg_t2d_scores/report.md, the frozen FD parse; Erratum 1's corrected parse is in rr_numbers);
- every other registered analysis of the two registration files, parsed from the registered scorers' reports:
  reference/PREREGISTRATION_LAYERS.md (base H1-H5, Amendments 2, 3, 4, 6, 8's H8a, 10) and tuning/PREREGISTRATION.md
  (Amendment 3: tie-breaker; Amendment 4: lockbox).

Types (as registered): S+ / S- superiority (95% interval bound on the predicted side of 0); NI non-inferiority
(lower bound > -0.01); EQ equivalence (interval inside +-0.01); UB upper bound below a stated value (R7a: 0.005);
3w three-way, no direction (Amendment 9); D deterministic (zero order inversions); P point prediction; desc.
descriptive or no interval. p values are recovered from the stored interval (normal on the null side, so that the
uncorrected decision equals the registered interval rule); Holm at a one-sided family-wise 0.025 over every decided
hypothesis with an interval, and separately over the six amendments behind Table III.

Amendment 7 (E2): T2D was not profiled before its SF runs, so by Amendment 7 section 2 ("otherwise" branch, recorded in
the study log at 2026-10-04 21:12 before any E2 output) P1, P3 and P4 are decided on Lahman/MusicBrainz (ctl-plain)
alone; the T2D side is exploratory (explore_e2_t2d.py).

Writes figures/recall/w6_registry.md, figures/recall/w6_holm.csv and supplement/registry.tex.
Run:  ../../harness/.venv-bdi/Scripts/python.exe w6_registry.py
"""
from paths import DERIVED, OUT, STUDY, STUDY_CODE, BASELINES_CODE  # noqa: E402,F401
import csv
import hashlib
import re
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ST = STUDY
RES = ST / 'tuning' / 'results'
ALPHA = 0.025
NUM = re.compile(r'([+-]?\d+\.\d+)\s*(?:\|\s*)?(?:95% CI\s*)?\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]')


def macros():
    m = {}
    for f in OUT.glob('*_numbers.tex'):
        m.update(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', f.read_text(encoding='utf8')))
    return m


M = macros()


def num(k):
    return float(M[k].replace('$', '').replace('+', ''))


def mest(base):
    return num(base + 'Est' if base + 'Est' in M else base), num(base + 'Lo'), num(base + 'Hi')


def report_row(path, pattern):
    """(estimate, lo, hi, verdict text) of the first line of a registered report matching `pattern`."""
    rx = re.compile(pattern)
    for line in Path(path).read_text(encoding='utf8').splitlines():
        if rx.search(line):
            n = NUM.search(line)
            low = line.lower()
            v = ('not met' if 'not met' in low else
                 'not supported' if 'not supported' in low else 'not confirmed' if 'not confirmed' in low else
                 'confirmed' if 'confirmed' in low else '')
            return (float(n.group(1)), float(n.group(2)), float(n.group(3))) if n else None, v
    raise KeyError(f'{pattern} not in {path}')


AM = {  # amendment -> (file, appended, topic)
    'L': ('layers', '2026-10-02 21:00', 'base hypotheses H1-H5'),
    'L2': ('layers', '2026-10-03', 'R7: FD closure and minimal cover'),
    'L3': ('layers', '2026-10-03 14:10', 'R8: restricted propagation scope'),
    'L4': ('layers', '2026-10-03 14:10', 'R8b: mutually best candidates'),
    'L5': ('layers', '2026-10-03 17:40', 'datatype-and-class term, 3-DB'),
    'L6': ('layers', '2026-10-04 ~12:10', 'T1: table membership'),
    'L7': ('layers', '2026-10-04 21:07', 'E2: Lahman/MusicBrainz (T2D not entered)'),
    'L8': ('layers', '2026-10-04 15:27', 'class nodes, Valentine (earlier copy)'),
    'L9': ('layers', '2026-10-04 18:37', 'table-context term, 3-DB'),
    'L10': ('layers', '2026-10-04 20:58', 'Sherlock types as class nodes'),
    'L11': ('layers', '2026-10-05 00:21', 'role term, T2D (erratum 2026-10-06 11:08)'),
    'L12': ('layers', '2026-10-06 18:20', 'final test'),
    'T0': ('tuning', '2026-10-02', 'base rule: adaptive propagation'),
    'T3': ('tuning', '2026-10-02', 'SF as tie-breaker'),
    'T4': ('tuning', '2026-10-04 ~00:45', 'lockbox'),
}
TABLE3 = {'L5', 'L7', 'L8', 'L9', 'L11', 'L12'}

RR_T2D = RES / 'rolesreg_t2d_scores' / 'report.md'


def rows_spec():
    """(amendment, id, contrast, corpus, metric, type, source, decided, note)."""
    S = []
    add = lambda *a: S.append(a)
    # ---- six amendments behind Table III
    add('L5', 'HB1', 'pattern-class term - SF', '3-DB', 'J', 'S+', ('m', 'amfHOne'), True, '')
    add('L5', 'HB2', 'semantic-class term - SF', '3-DB', 'J', 'S+', ('m', 'amfHTwo'), True, '')
    add('L5', 'HB3', 'datatype term - SF', '3-DB', 'J', 'S+', ('m', 'amfHThree'), True, '')
    # Amendment 7 decides P1 on T2D (reported on ctl), P2/P3 on ctl-plain and T2D, P4 on ctl and T2D. T2D was profiled
    # (23:42) before its E2 runs (00:02), so the literal condition held; the user decided at ~23:20 to keep E2 on ctl
    # and run T2D as exploratory. 2026-11-08, the user: "Show them as registered" -> T2D rows decided, marked ^g.
    add('L7', 'P1', 'rho(quality, gain)', 'L/MB read.', 'R', 'S-', ('m', 'etwoPOnePl'), False, '')
    add('L7', 'P1', 'rho(quality, gain)', 'L/MB rec.', 'R', 'S-', ('m', 'etwoPOneMd'), False, '')
    add('L7', 'P2', 'FD graph - SF', 'L/MB read.', 'R', 'S-', ('m', 'etwoPTwoPl'), True, '')
    add('L7', 'P2', 'FD graph - SF', 'L/MB rec.', 'J', 'S+', ('m', 'etwoPTwoMd'), True, '')
    add('L7', 'P3', 'datatype term - SF', 'L/MB read.', 'J', 'S+', ('m', 'etwoPThree'), True, '')
    add('L7', 'P4', 'FD top-1 - FD graph', 'L/MB read.', 'R', 'S+', ('m', 'etwoPFourPl'), True, '')
    add('L7', 'P4', 'FD top-1 - FD graph', 'L/MB rec.', 'R', 'S+', ('m', 'etwoPFourMd'), True, '')
    for hid, con, met, t, base in (('P1', 'rho(quality, gain)', 'R', 'S-', 'ttdPOne'),
                                   ('P2', 'FD graph - SF', 'R', 'S-', 'ttdPTwo'),
                                   ('P3', 'datatype term - SF', 'J', 'S+', 'ttdPThree'),
                                   ('P4', 'FD top-1 - FD graph', 'R', 'S+', 'ttdPFour')):
        add('L7', hid, con, 'T2D', met, t, ('m', base), True, 'g')
    for name, con, t in (('cfPlShapeVsShapeShuf', 'real - shuffled pattern classes', 'S+'),
                         ('cfPlSemVsSemShuf', 'real - shuffled semantic types', 'S+'),
                         ('cfPlShapeVsRef', 'pattern-class nodes - SF', 'S+'),
                         ('cfPlSemVsRef', 'semantic-type nodes - SF', 'S+'),
                         ('cfPlLpVsShape', 'late fusion - pattern nodes', 'S+'),
                         ('cfPlLsVsSem', 'late fusion - semantic nodes', 'S+'),
                         ('cfPlBpVsShape', 'bonus after SF - pattern nodes', 'S+'),
                         ('cfPlBsVsSem', 'bonus after SF - semantic nodes', 'S+')):
        hid = {'ShapeShuf': 'H8b', 'SemShuf': 'H8b', 'VsRef': 'H8c', 'Lp': 'H8d', 'Ls': 'H8d', 'Bp': 'H8e',
               'Bs': 'H8e'}
        h = next(v for k, v in hid.items() if k in name)
        add('L8', h, con, 'Valentine', 'J', t, ('m', name), True, '')
    add('L8', 'H8f', 'semantic-type nodes - SF', 'Valentine rec.', 'J', 'EQ', ('m', 'cfMdSemVsRef'), True, '')
    add('L8', 'H8a', 'within-block inversions (classes)', 'Valentine', '--', 'D',
        ('r', RES / 'classfull_plain_scores', r'H8a \(shape\)'), True, '')
    add('L9', 'HC0', 'table-context term - datatype term', '3-DB', 'J', 'S+', ('m', 'tcHZero'), True, '')
    for h, base, con in (('HC1', 'tcHOne', 'SF - table-context term'), ('HC2', 'tcHTwo', 'SF+P - table-context term'),
                         ('HC3', 'tcHThree', 'SF+S - table-context term')):
        add('L9', h, con, '3-DB', 'J', '3w', ('m', base), True, '')
    for h, con, t in (('HR1', 'role term - datatype term', 'S+'), ('HR2', 'role term - SF', 'S+'),
                      ('HR3', 'role term - FD graph', 'S+'), ('HR4a', 'role term - key graph', 'NI'),
                      ('HR4b', 'role term - key+FD graph', 'NI')):
        add('L11', h, con, 'T2D', 'J', t, ('r', RR_T2D, r'^\| %s\b' % h), True, 'e')
    add('L11', 'HR4a', 'role term - key graph', 'L/MB read.', 'J', 'NI', ('m', 'rrCtlPlHFourA'), False, 'c')
    add('L11', 'HR4a', 'role term - key graph', 'L/MB rec.', 'J', 'NI', ('m', 'rrCtlMdHFourA'), False, 'c')
    add('L12', 'F1', 'datatype term - SF', 'Mixed read.', 'J', 'S+', ('m', 'finPlFOne'), True, '')
    add('L12', 'F2', 'role term - key graph', 'Mixed read.', 'J', 'NI', ('m', 'finPlFTwoA'), True, '')
    add('L12', 'F2', 'role term - key+FD graph', 'Mixed read.', 'J', 'NI', ('m', 'finPlFTwoB'), True, '')
    add('L12', 'F3', 'rho(quality, gain)', 'Mixed read.', 'R', 'S-', ('m', 'finPlFThree'), True, '')
    add('L12', 'F4', 'FD graph - SF', 'Mixed read.', 'R', 'S-', ('m', 'finPlFFour'), True, '')
    add('L12', 'F5', 'table-context term - SF', 'Schema pairs', 'J', 'S+', ('m', 'finAFFive'), True, '')
    add('L12', 'F1', 'datatype term - SF', 'Mixed rec.', 'J', 'S+', ('m', 'finOpFOne'), False, '')
    add('L12', 'F2', 'role term - key+FD graph', 'Mixed rec.', 'J', 'NI', ('m', 'finOpFTwoB'), False, '')
    # ---- other registrations of PREREGISTRATION_LAYERS.md
    add('L', 'H1', 'within-block inversions (Prop. 1)', 'Valentine', '--', 'D', ('fixed', 'confirmed'), True, 'a')
    add('L', 'H2', '|SF - initial| < 0.03 for every matcher', 'Valentine', 'R', 'desc.',
        ('fixed', 'not confirmed'), True, 'b')
    add('L', 'H3', 'constraint arms change ranking more', 'Valentine', 'R', 'desc.', ('fixed', 'held (descriptive)'),
        True, '')
    add('L', 'H4', 'SF gain: 3-DB - Valentine', '3-DB/Val.', 'J', 'S+',
        ('r', RES / 'layer1_indep_scores' / 'h4_report.md', r'^\| H4 \|'), True, 'd')
    add('L', 'H5', 'recoding hurts label matchers', 'Valentine', '--', 'desc.', ('fixed', 'descriptive'), False, '')
    for reg in ('plain', 'md5'):
        cor = 'Valentine' + (' rec.' if reg == 'md5' else '')
        f = RES / f'r7_{reg}_scores'
        add('L2', 'R7a', '|closure effect| < 0.005', cor, 'J', 'UB', ('r', f, r'^\| H-R7a \|'), True, '')
        add('L2', 'R7b', 'cover - closure (QUICK)', cor, 'R', 'S+', ('r', f, r'^\| H-R7b \(QUICK\)'), True, '')
        add('L2', 'R7b', 'cover - closure (reified)', cor, 'R', 'S+', ('r', f, r'^\| H-R7b \(reified\)'), True, '')
        add('L2', 'R7c', 'inversions: cover - FD', cor, '--', 'S+', ('r', f, r'^\| H-R7c \|'), True, '')
        f = RES / f'r8_{reg}_scores'
        add('L3', 'R8a', 'within-block inversions (scope)', cor, '--', 'D', ('r', f, r'^\| H-R8a \|'), True, '')
        for s in ('q050', 'q075', 'q090', 'top1', 'top3'):
            add('L3', 'R8b', f'scope effect on SF gain ({s})', cor, 'R', 'EQ', ('r', f, r'^\| H-R8b %s' % s), True, '')
        for s in ('q090', 'top1'):
            add('L3', 'R8c1', f'FD inversions, scope ({s})', cor, '--', 'S-',
                ('r', f, r'^\| H-R8c\(1\) %s' % s), True, '')
        for s in ('q050', 'q075', 'q090', 'top1', 'top3'):
            add('L3', 'R8c2', f'scope effect: FD - SF ({s})', cor, 'J', 'S+',
                ('r', f, r'^\| H-R8c\(2\) %s' % s), True, '')
        f = RES / f'r8b_{reg}_scores'
        for h, pat, con in (('R8b-1', r'^\| H-R8b-1 \|', 'FD mutual top-1 - FD'),
                            ('R8b-2', r'^\| H-R8b-2 \(fd\)', 'FD mutual top-1 - initial'),
                            ('R8b-2', r'^\| H-R8b-2 \(melnik\)', 'SF mutual top-1 - initial'),
                            ('R8b-3', r'^\| H-R8b-3 \(melnik\)', 'SF mutual top-1 - scoped initial'),
                            ('R8b-3', r'^\| H-R8b-3 \(fd\)', 'FD mutual top-1 - scoped initial'),
                            ('R8b-4', r'^\| H-R8b-4 \|', 'SF mutual top-1 - SF')):
            add('L4', h, con, cor, 'J', 'S+', ('r', f, pat), True, '')
    f = RES / 't1_indep_scores'
    add('L6', 'T1a', 'within-block inversions (collapsed)', '3-DB', '--', 'D', ('r', f, r'T1a \(deterministic\)'), True,
        '')
    for h, pat, con in (('T1b', r'^\| T1b', 'SF - SF on collapsed tables'), ('T1c', r'^\| T1c', 'SF - SF, columns permuted'),
                        ('T1d', r'^\| T1d', 'datatype term - SF collapsed'),
                        ('T1e', r'^\| T1e \(A_p\)', 'pattern term - collapsed nodes'),
                        ('T1e', r'^\| T1e \(A_s\)', 'semantic term - collapsed nodes'),
                        ('T1f', r'^\| T1f \(shape\)', 'pattern nodes - collapsed'),
                        ('T1f', r'^\| T1f \(semtype\)', 'semantic nodes - collapsed')):
        add('L6', h, con, '3-DB', 'J', 'S+', ('r', f, pat), True, '')
    add('L10', 'H10a', 'within-block inversions (Sherlock)', 'Valentine', '--', 'D', ('fixed', 'confirmed'), True, '')
    for h, reg, con in (('H10b', 'plain', 'Sherlock - shuffled'), ('H10c', 'plain', 'Sherlock - SF'),
                        ('H10d', 'md5', 'Sherlock - shuffled'), ('H10e', 'md5', 'Sherlock - SF')):
        add('L10', h, con, 'Valentine' + (' rec.' if reg == 'md5' else ''), 'J', 'S+',
            ('r', RES / f'sherlock_{reg}_scores', r'^\| %s:.*(CONFIRMED|confirmed)' % h), True, '')
    # ---- tuning/PREREGISTRATION.md
    add('T0', 'Adapt.', 'adaptive alpha: predictor rho at least 0.5', 'Val. screen', '--', 'desc.',
        ('r', RES / 'screen_plain' / 'report.md', r'Adaptive-α rule'), True, '')
    add('T0', 'Adapt.', 'adaptive alpha: predictor rho at least 0.5', 'Val. dev', '--', 'desc.',
        ('r', RES / 'dev_plain' / 'report.md', r'Adaptive-α rule'), True, '')
    add('T0', 'Adapt.', 'adaptive alpha: predictor rho at least 0.5', 'Val. lockbox', '--', 'desc.',
        ('r', RES / 'lockbox_plain' / 'report.md', r'Adaptive-α rule'), True, '')
    for reg, path in (('Valentine rec.', ST / 'results' / 'md5' / 'amendment3_primary.txt'),
                      ('Val. lockbox', RES / 'lockbox_plain' / 'tiebreak_primary.txt')):
        add('T3', 'H1', 'tie-breaker - initial', reg, 'J', 'S+', ('r', path, r'^H1 '), True, '')
        add('T3', 'H2', 'tie-breaker - SF', reg, 'J', 'S+', ('r', path, r'^H2 '), True, '')
        add('T3', 'H3', 'tie-breaker - initial', reg, 'R', 'NI', ('r', path, r'^H3 '), True, '')
    for h, con in (('HL1', 'own - production'), ('HL2', 'own - no propagation'), ('HL3', 'own - universal'),
                   ('HL4', 'universal - production')):
        add('T4', h, con, 'Val. lockbox', 'J', 'S+', ('r', RES / 'lockbox_plain' / 'hl_report.md', r'^\| %s \|' % h),
            True, '')
    return S


def first_file(p):
    p = Path(p)
    return p if p.is_file() else sorted(p.glob('*.md'))[0] if len(list(p.glob('*.md'))) == 1 else p


def resolve(src):
    kind = src[0]
    if kind == 'm':
        base = src[1]
        v = M.get(base + 'Verdict') or M.get(base + 'Decision') or ''
        if not v and base + 'Consistent' in M:      # T2D side of Amendment 7: the registered interval rule's outcome
            v = 'confirmed' if M[base + 'Consistent'] == 'true' else 'not confirmed'
        return mest(base), v
    if kind == 'fixed':
        return None, src[1]
    p = Path(src[1])
    files = [p] if p.is_file() else sorted(p.glob('*.md'))
    for f in files:
        try:
            return report_row(f, src[2])
        except KeyError:
            continue
    raise KeyError(f'{src[2]} not found under {p}')


def one_sided(x, b):
    se = abs(x - b) / 1.959964
    if se == 0:
        return 0.0 if x > 0 else 1.0
    return float(stats.norm.sf(x / se))


def pval(t, e, lo, hi):
    if t == 'S+':
        return one_sided(e, lo if e > lo else hi)
    if t == 'S-':
        return one_sided(-e, -hi if -e > -hi else -lo)
    if t == 'NI':
        return one_sided(e + 0.01, lo + 0.01)
    if t == 'EQ':
        return max(one_sided(e + 0.01, lo + 0.01), one_sided(0.01 - e, 0.01 - hi))
    if t == 'UB':
        return one_sided(0.005 - e, 0.005 - hi)
    return one_sided(e, lo) if e > 0 else one_sided(-e, -hi)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= ALPHA / (len(ps) - rank):
            rej[i] = True
        else:
            break
    return rej


def is_conf(v):
    v = v.lower()
    return v.startswith('confirmed') or v.startswith('table context suffices') or v.startswith('held')


def rows():
    out = []
    for am, hid, con, corp, met, t, src, dec, note in rows_spec():
        ci, v = resolve(src)
        e = lo = hi = None
        if ci is not None:
            e, lo, hi = ci
        interval = ci is not None and t not in ('D', 'desc.', 'P')
        out.append(dict(am=am, hid=hid, contrast=con, corpus=corp, metric=met, type=t, decided=dec, note=note,
                        base=src[1] if src[0] == 'm' else '', e=e, lo=lo, hi=hi, registered=v or '',
                        rule=is_conf(v or ''), p=pval(t, e, lo, hi) if interval else None, interval=interval))
    dec = [r for r in out if r['decided'] and r['interval']]
    for r, h in zip(dec, holm([r['p'] for r in dec])):
        r['holm_global'] = h
    t3 = [r for r in dec if r['am'] in TABLE3]
    for r, h in zip(t3, holm([r['p'] for r in t3])):
        r['holm_t3'] = h
    return out


def holm_lookup():
    """{macro base: survives the global Holm correction} for Table III's macro rows."""
    return {r['base']: (r['rule'] and r.get('holm_global', False)) for r in rows() if r['decided'] and r['base']}


def registered_t2d_roles():
    """{HRx: (e, lo, hi, verdict)} of Amendment 11's registered (frozen-parse) T2D output, for Table III."""
    return {h: (*report_row(RR_T2D, r'^\| %s\b' % h)[0], report_row(RR_T2D, r'^\| %s\b' % h)[1])
            for h in ('HR1', 'HR2', 'HR3', 'HR4a', 'HR4b')}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tex(x, d=3):
    return (('%+.' + str(d) + 'f') % x).replace('-', '$-$')


def summary(rr):
    dec = [r for r in rr if r['decided'] and r['interval']]
    conf = [r for r in dec if r['rule']]
    surv = [r for r in conf if r['holm_global']]
    t3 = [r for r in dec if r['am'] in TABLE3]
    t3c = [r for r in t3 if r['rule']]
    return dict(n=len(dec), conf=len(conf), surv=len(surv), lost=[r for r in conf if not r['holm_global']],
                n3=len(t3), conf3=len(t3c), surv3=sum(1 for r in t3c if r['holm_t3']),
                nall=sum(1 for r in rr if r['decided']), failed=[r for r in rr if r['decided'] and not r['rule']])


def amlabel(a):
    return ('L' + a[1:] if a != 'L' else 'L0') if a.startswith('L') else a


def main():
    rr = rows()
    s = summary(rr)
    L = ['# Registry of all registered hypotheses (2026-11-08)', '',
         f"{s['nall']} decided hypotheses; {s['n']} with an interval, of which {s['conf']} confirmed; {s['surv']} survive "
         f"Holm over all {s['n']} (one-sided family-wise {ALPHA}). Behind Table III (Amendments 5, 7, 8, 9, 11, 12): "
         f"{s['n3']} with an interval, {s['conf3']} confirmed, {s['surv3']} survive Holm within that family.",
         'Lost under the global Holm: ' + '; '.join(f"{r['am']} {r['hid']} {r['corpus']} (p {r['p']:.3g})"
                                                    for r in s['lost']), '',
         '| file/am. | id | contrast | corpus | M | type | estimate [95%] | registered | p | Holm all | decided |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rr:
        est = '' if r['e'] is None else f"{r['e']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
        hol = '' if not (r['decided'] and r['interval']) else ('yes' if r['rule'] and r['holm_global'] else 'no')
        L.append(f"| {r['am']} | {r['hid']} | {r['contrast']} | {r['corpus']} | {r['metric']} | {r['type']} | {est} | "
                 f"{r['registered']}{' (' + r['note'] + ')' if r['note'] else ''} | "
                 f"{'' if r['p'] is None else '%.2g' % r['p']} | {hol} | {r['decided']} |")
    L += ['', 'Documents (SHA-256 of the study copies):', '',
          f"- reference/PREREGISTRATION_LAYERS.md: {sha(ROOT / 'registrations' / 'PREREGISTRATION_LAYERS.md')}",
          f"- tuning/PREREGISTRATION.md: {sha(ROOT / 'registrations' / 'PREREGISTRATION.md')}"]
    (OUT / 'w6_registry.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    keys = ['am', 'hid', 'contrast', 'corpus', 'metric', 'type', 'e', 'lo', 'hi', 'registered', 'p', 'decided', 'note',
            'holm_global', 'holm_t3']
    with open(OUT / 'w6_holm.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rr)
    print('\n'.join(L[:5]))
    write_supp(rr, s)


def write_supp(rr, s):
    typ = {'S+': 'S', 'S-': 'S', 'NI': 'NI', 'EQ': 'EQ', '3w': '3w', 'UB': 'UB', 'D': 'D', 'desc.': 'desc.', 'P': 'P'}
    name = {'L': 'Layers', 'T0': 'Tuning', 'T3': 'Tuning Am.~3', 'T4': 'Tuning Am.~4'}
    lost = '; '.join(r'%s %s, %s ($p=%.3f$)' % (name.get(r['am'], 'Am.~' + r['am'][1:]), r['hid'], r['corpus'], r['p'])
                     for r in s['lost'])
    o = ['% Generated by w6_registry.py; do not edit.', r'\section{Registered hypotheses}', r'\label{supp:registry}',
         r'The study has two registration files: one for the configuration search, the tie-breaker analysis and the'
         r' lockbox, and one for the layer-1 and layer-3 hypotheses with twelve dated amendments. Each amendment was'
         r' appended before the data it concerns were run or scored, records the time it was appended and the SHA-256'
         r' hashes of the scorers and inputs it froze, and the study log records the file hash after each append. The'
         r' files were frozen in the study repository, not deposited with an external registry; the artifact copies'
         r' redact three local paths. Tables~\ref{supp:tab-registry}--\ref{supp:tab-registry3} list every'
         r' registered hypothesis with its type: S (superiority: the 95\%% interval excludes 0 on the predicted side;'
         r' some registrations also require the sign in every database), NI (non-inferiority, margin $-0.01$), EQ'
         r' (equivalence: the interval inside $\pm0.01$), UB (upper bound below a stated value), 3w (no direction;'
         r' three-way decision), D (deterministic: zero order inversions) or desc.\ (descriptive). Rows in parentheses'
         r' are reported, not decided, by their registration.',
         '',
         r'As a check that the registrations did not prescribe, we apply Holm' + "'" + r's step-down correction at a'
         r' one-sided family-wise level of 0.025 over all %d decided hypotheses with an interval, with $p$ values'
         r' recovered from each stored interval (normal on the null side, so that the uncorrected decision equals the'
         r' registered interval rule): %d of the %d confirmations survive; lost: %s. Within the six amendments behind'
         r' Table~III of the paper, %d of %d survive.' % (s['n'], s['surv'], s['conf'], lost, s['surv3'], s['conf3']),
         '',
         r'Several registrations deviate from their text. Amendment~7 admits T2D only if T2D is profiled with the'
         r' harness pipeline before any SF run on it. When it was appended (2026-10-04 21:07), T2D was unprofiled, and a'
         r' reading rule written at 21:12 decided P1, P3 and P4 on Lahman/MusicBrainz alone; at about 23:20 we decided to'
         r' run T2D as an exploratory replication. T2D was then profiled (23:42) before the runs in Amendment~7' + "'" +
         r's design were set up (00:02), so the registration' + "'" + r's condition did hold, and we report the T2D'
         r' rows as decided by the registered rule, which makes P1 decided on T2D and P3, decided jointly on T2D and'
         r' Lahman/MusicBrainz, not confirmed. Under the other reading, fixed before the data as well (T2D'
         r' exploratory), P1 and P3 are decided on Lahman/MusicBrainz alone and both confirmed (P1 not after Holm),'
         r' and P4 is confirmed on readable and not on recoded data; under the literal reading, P1, P2 and P4 are also'
         r' confirmed on T2D. Qualifications: the T2D side was scored by a separate scorer that applies the registered'
         r' design to 175 of 178 scenarios (3 had no usable profiles) and skips non-finite initial scores ($^g$), and'
         r' Amendment~7' + "'" + r's'
         r' statement that no SF had run on T2D ignores an earlier harness study of SF on T2D (2026-09-19) outside this'
         r' study' + "'" + r's pipeline. Amendment~11' + "'" + r's registered output'
         r' used a defective FD-file parse; Erratum~1 (appended before the paper used the result) reports the corrected'
         r' parse, which moves every estimate by $+0.001$ and changes no verdict; the table shows the registered values'
         r' ($^e$; corrected: HR2 $-0.008$, HR4a $+0.038$). Base H1 switched its datatype classes to SF' + "'" + r's own'
         r' output by Amendment~1, written after the registered proxy showed spurious inversions ($^a$). H2 predicted'
         r' $\lvert\Delta R\rvert<0.03$ for every initial matcher; 26 of 75 exceed it ($^b$). H4' + "'" + r's decision'
         r' rule (lower bound $>0$) was stated by the scorer' + "'" + r's author before scoring, the registration saying'
         r' only ``larger'' ($^d$). Lahman/MusicBrainz role rows: corrected parse ($^c$).']
    chunks = [rr[i:i + 52] for i in range(0, len(rr), 52)]
    for ci, chunk in enumerate(chunks):
        lab = 'supp:tab-registry' if ci == 0 else 'supp:tab-registry%d' % (ci + 1)
        cap = (r'Registered hypotheses (part %d of %d): registered verdict and Holm correction over all decided'
               r' hypotheses with an interval.' % (ci + 1, len(chunks)))
        o += [r'\begin{table*}[p]', r'\caption{%s}' % cap, r'\label{%s}' % lab,
              r'\centering\scriptsize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}llllllrll@{}}', r'\toprule',
              r'File/Am. & Id & Contrast & Corpus & M. & Type & Estimate [95\%] & Registered & Holm\\', r'\midrule']
        last = None
        for r in chunk:
            if last is not None and r['am'] != last:
                o.append(r'\addlinespace[1pt]')
            last = r['am']
            d = 2 if r['contrast'].startswith('rho') else 3
            est = '--' if r['e'] is None else '%s [%s, %s]' % (tex(r['e'], d), tex(r['lo'], d), tex(r['hi'], d))
            reg = (r['registered'] or '--') + (r'$^{%s}$' % r['note'] if r['note'] else '')
            hol = '--' if not (r['decided'] and r['interval']) else ('yes' if r['rule'] and r['holm_global'] else 'no')
            con = r['contrast'].replace(' - ', ' $-$ ').replace('rho', r'$\rho$').replace('|', r'$\lvert$')
            con = con.replace('< 0.03', r'$<0.03$').replace('< 0.005', r'$<0.005$')
            amn = name.get(r['am'], 'L' + r['am'][1:])
            row = [amn, r['hid'], con, r['corpus'], r['metric'], typ[r['type']], est, reg, hol]
            if not r['decided']:
                row[2] = '(' + row[2] + ')'
            o.append(' & '.join(row) + r'\\')
        o += [r'\bottomrule', r'\end{tabular}\\[2pt]',
              r'\parbox{\textwidth}{\scriptsize File/Am.: L$n$ = Amendment $n$ of the layers registration (L0: its base'
              r' hypotheses); Tuning Am.: amendments of the tuning registration. Appended: ' +
              '; '.join(r'%s %s' % (name.get(k, 'L' + k[1:]), v[1]) for k, v in AM.items()) +
              r'. M.: J $\Jtf$, R Recall@GT. Holm: confirmed and rejected by Holm at a one-sided family-wise 0.025 over'
              r' the %d decided hypotheses with an interval; --: not applicable. SHA-256 of the study copies:'
              r' layers %s, tuning %s.}' % (s['n'], sha(ROOT / 'registrations' / 'PREREGISTRATION_LAYERS.md')[:16] + '\\ldots',
                                            sha(ROOT / 'registrations' / 'PREREGISTRATION.md')[:16] + '\\ldots'),
              r'\end{table*}']
    (OUT / 'registry_table.txt').write_text('\n'.join(o) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
