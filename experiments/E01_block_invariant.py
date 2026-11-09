"""E01: SF cannot reorder candidates within a datatype block (Proposition 1, checked on exported runs).

Paper: Section II (Proposition 1). Input: results/sf_study/results/sf_export_check/ (the per-run check of exported
SF graphs and iterates). Runtime: seconds (plus one shared run of the analysis code, about 15 s).

    python experiments/E01_block_invariant.py
"""
import sys

from _common import V, report, rows

v = V()
c = {}
sections = [
    ("Exported runs checked", rows(v, [
        ("runs (14 scenarios x 3 initial matchers x 3 attachments x 2 coefficient policies x 2 literal settings)",
         "gxRuns"),
        ("ordered within-block pairs checked over all iterates, millions", "gxPairsM"),
        ("of these reordered", "gxViol"),
        ("tied pairs within blocks, millions", "gxTiesM"),
        ("ties broken by SF", "gxTiesBroken"),
    ], c)),
]
sys.exit(report("E01", "Datatype-block invariant",
                "Without key or FD edges, is every SF iterate the initial scores, scaled, plus one offset per "
                "datatype pair, so that SF cannot reorder candidates within a block?", sections, c,
                note="The proof is in the paper's supplement; this experiment is the numerical check."))
