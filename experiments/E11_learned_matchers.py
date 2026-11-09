"""E11: learned matchers with message passing (SemStruct reconstruction, OmniMatch) as case studies.

Paper: Section V-G and the supplement. Inputs: results/baselines/ (SemStruct and OmniMatch runs). Runtime: seconds
after the shared analysis run. The SemStruct reconstruction itself is in study_code/semstruct/.

    python experiments/E11_learned_matchers.py
"""
import sys

from _common import V, report, rows

v = V()
c = {}
sections = [
    ("SemStruct reconstruction on readable Valentine, J_tf", rows(v, [
        ("scenarios", "ssaN"),
        ("averaging value embeddings - message passing", "ssaValFull~ci"),
        ("one-hop - full message passing", "ssaOneFull~ci"),
        ("structure permuted - original", "ssaPerm~ci"),
        ("J_tf: full / one-hop / value average", "ssaJFull"),
        ("J_tf, one-hop", "ssaJOne"),
        ("J_tf, value average", "ssaJVal"),
    ], c)),
    ("OmniMatch", rows(v, [
        ("scenarios", "omN"),
        ("graph edges removed - full model", "omEdge~ci"),
        ("random-forest features only - full model", "omRfF~ci"),
    ], c)),
]
sys.exit(report("E11", "Learned matchers with message passing",
                "Does message passing in learned matchers add over simpler use of the same information?", sections, c,
                note="These are exploratory case studies of partial reconstructions; see the paper's supplement."))
