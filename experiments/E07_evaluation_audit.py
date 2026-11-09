"""E07: how much do the metric, the tie rule and the assignment rule change SF's measured gain?

Paper: Section VI-B and Table IV (top). Inputs: the shared analysis values and results/derived/recall/
gain_policies_rows.csv(.gz) (every initial matcher alone and with SF, scored under several assignment rules).
Runtime: about a minute.

    python experiments/E07_evaluation_audit.py
"""
import sys

from _common import V, report, rows

v = V()
import f1_rules  # noqa: E402

c = {}
table4 = rows(v, [
    ("expected Recall@GT (ties filled at random in expectation)", "drmPlSfQRexp~ci"),
    ("pessimistic Recall@GT (ties counted against the matcher)", "drmPlSfQRpess~ci"),
    ("Hungarian F1, ties broken at random", "drmPlSfQFHr~ci"),
    ("Hungarian F1, ties broken by index order", "drmPlSfQFH~ci"),
    ("mutual top-1 F1", "drmPlSfQFM~ci"),
    ("F1, forced one-to-one", "xePlSfQForced~ci"),
    ("F1, assignment policy taken from other families", "xePlSfQVal~ci"),
], c)
g = f1_rules.gains()
fmt = lambda t: '%+.4f [%+.4f, %+.4f]' % t
rule = []
for regime, lab in (('plain', 'readable'), ('md5', 'recoded')):
    th, co = g[regime]['threshold'], g[regime]['complete']
    rule += [(f'{lab}: F1 gain, pairs scoring 0 or less unassigned ({g[regime]["n"]} matchers)', fmt(th)),
             (f'{lab}: F1 gain, complete one-to-one assignment', fmt(co)),
             (f'{lab}: factor between the two rules', '%.1f' % (max(th[0], co[0]) / min(th[0], co[0])))]
    c[f'{lab} F1 gain threshold'], c[f'{lab} F1 gain complete'] = fmt(th), fmt(co)
    c[f'{lab} factor'] = '%.1f' % (max(th[0], co[0]) / min(th[0], co[0]))
const = rows(v, [("Valentine, ties by index order", "xcConstHVal"), ("Valentine, ties at random", "xcConstRVal"),
                 ("MaDI use cases, index order", "xcConstHMadi"), ("HDXSM, index order", "xcConstHHdx")], c)
sections = [
    ("SF as published minus its initial matcher, readable Valentine, by metric and tie rule", table4),
    ("The assignment rule (same matchers and scenarios)", rule),
    ("Hungarian F1 of a constant matrix (no information at all)", const),
]
sys.exit(report("E07", "Evaluation choices",
                "How much of SF's measured benefit is decided by the metric, the tie rule and the assignment rule?",
                sections, c))
