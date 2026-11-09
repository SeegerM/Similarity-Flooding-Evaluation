"""E09: does tuning SF's parameters help on data kept apart from tuning? (registered lockbox)

Paper: Section VI-A and Table IV (bottom). Inputs: results/sf_study/tuning/results/lockbox_plain/ (the single
registered evaluation on 94 held-out Valentine scenarios). Runtime: seconds after the shared analysis run.

    python experiments/E09_tuning_lockbox.py
"""
import sys

from _common import V, report, rows

v = V()
c = {}
sections = [
    ("Gain over the initial matcher on the lockbox, J_tf (7 initial matchers, 94 scenarios)", rows(v, [
        ("lockbox scenarios", "lbxScen"),
        ("initial matchers", "lbxN"),
        ("tuned per matcher", "lbxOwn"),
        ("tuned, one configuration for all matchers", "lbxUni"),
        ("production configuration", "lbxProd"),
        ("no propagation (alpha = 0)", "lbxAlphaZero"),
        ("SF as published (reported only)", "lbxRef"),
    ], c)),
    ("Registered hypotheses of the lockbox", rows(v, [
        ("HL1: own - production", "lbxHOne~ci"),
        ("HL2: own - no propagation", "lbxHTwo~ci"),
        ("HL3: own - universal", "lbxHThree~ci"),
        ("HL4: universal - production", "lbxHFour~ci"),
        ("SF as published - tuned per matcher", "rvLbxRefVsOwn~ci"),
    ], c)),
]
sys.exit(report("E09", "Tuning and the registered lockbox",
                "Does a configuration tuned on development data beat its initial matcher and SF as published on "
                "data kept apart from tuning?", sections, c))
