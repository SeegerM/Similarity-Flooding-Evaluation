"""E02: how much of SF's gain depends on its initial matcher (research question Q2).

Paper: Section IV, Fig. 3 and 4. Inputs: results/sf_study/results/plain549/scores_tiefree.csv (every initial matcher
alone and with SF as published on the 549 readable Valentine scenarios) and the external corpora's scores.
Runtime: seconds after the shared analysis run.

    python experiments/E02_gain_vs_initial_matcher.py
"""
import sys

from _common import V, report, rows

v = V()
c = {}
sections = [
    ("SF as published against its initial matcher, readable Valentine", rows(v, [
        ("initial matchers (each scored on all 549 scenarios)", "rqOneN"),
        ("matchers whose expected Recall@GT improves", "rqOneUpRexp"),
        ("mean change in expected Recall@GT", "rqOneMeanRexp"),
        ("smallest change in expected Recall@GT", "rqOneMinRexp"),
        ("largest change in expected Recall@GT", "rqOneMaxRexp"),
        ("matchers whose tie-free Hungarian F1 improves", "rqOneUpFHr"),
        ("mean change in tie-free Hungarian F1", "rqOneMeanFHr"),
    ], c)),
    ("Does the gain fall with initial quality? (Spearman rho over matchers)", rows(v, [
        ("rho(initial Recall@GT, gain in Recall@GT)", "rqOneRhoRexp"),
        ("rho(initial Recall@GT, gain in F1)", "rqOneRhoFHr"),
        ("bounds-only baseline: rho a matcher would show from score bounds alone (Recall@GT)", "rvNullRexp~ci"),
        ("bounds-only baseline (F1)", "rvNullFHr~ci"),
        ("headroom share instead of raw gain: observed rho (Recall@GT)", "rvHeadRexp"),
    ], c)),
]
sys.exit(report("E02", "SF's gain against its initial matcher",
                "How much of SF's gain depends on its initial matcher, and is the negative relation between initial "
                "quality and gain more than what score bounds alone produce?", sections, c,
                note="Leaving out one source family at a time and the external corpora are in the paper's "
                     "supplement; their values are part of the full value list (reports/E02.json covers the values "
                     "above)."))
