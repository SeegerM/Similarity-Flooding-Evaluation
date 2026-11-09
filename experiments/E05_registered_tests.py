"""E05: the registered tests, their verdicts, and a Holm correction over all of them.

Lists every hypothesis of the two registration files (registrations/), with its type (superiority, non-inferiority,
equivalence, three-way, deterministic, descriptive), estimate, registered verdict and whether it survives Holm's
correction over all decided hypotheses with an interval, plus the selection the paper shows as Table III.

Paper: Section V-F, Table III and the supplement's registry. Inputs: the registered scorers' reports in
results/sf_study/tuning/results/ and the shared analysis values. Runtime: about a minute.

    python experiments/E05_registered_tests.py
"""
import re
import sys

from _common import V, report

V()
import w6_registry as REG          # noqa: E402
import make_registered_table as T  # noqa: E402
from paths import OUT               # noqa: E402


def plain(s):
    s = s.replace('$-$', '-').replace('\\ ', ' ').replace('$\\rho$', 'rho').replace('$\\to$', ' to ')
    s = re.sub(r'\$\^\{?([a-z,]+)\}?\$', r'^\1', s)
    return s.replace('$', '')


REG.main()
rows = REG.rows()
s = REG.summary(rows)
reg_md = (OUT / 'w6_registry.md').read_text(encoding='utf-8')
t3 = ['| comparison | corpus | test | estimate [95%] | verdict | Recall@GT |', '|---|---|---|---|---|---|']
for part in (T.REG, T.ADV):
    t3 += ['| ' + ' | '.join(plain(x) for x in r) + ' |' for r in part]
checked = {'decided hypotheses': str(s['nall']), 'decided with an interval': str(s['n']),
           'confirmed (with an interval)': str(s['conf']), 'surviving Holm over all': str(s['surv']),
           'within the six amendments behind Table III: decided / confirmed / surviving':
               f"{s['n3']} / {s['conf3']} / {s['surv3']}"}
sections = [
    ('Summary', list(checked.items())),
    ('Selected registered tests (the paper\'s Table III)', '\n'.join(t3) + '\n\nTest: metric (J: J_tf, R: Recall@GT) and '
     'type (S superiority, NI non-inferiority with margin -0.01, 3w three-way). Verdicts in parentheses are reported, '
     'not decided, by their registration; ^h: confirmed but not after Holm; ^g: decided jointly on Lahman/MusicBrainz '
     'and T2D, T2D scored by a separate scorer with the registered design; ^i: registered FD parse (Erratum 1 to '
     'Amendment 11 gives the corrected values).'),
    ('All registered hypotheses', reg_md.split('\n', 2)[2]),
]
sys.exit(report('E05', 'Registered tests and a Holm correction',
                'Which registered predictions held, and which survive a correction for testing many hypotheses?',
                sections, checked,
                note='The registration documents are in registrations/; amendments state when they were appended.'))
