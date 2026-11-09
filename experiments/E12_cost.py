"""E12: what SF and the one-step terms cost (descriptive timing pass).

Paper: Section VII and the supplement. Inputs: the dedicated timing pass on the earlier 382-scenario copy of
readable Valentine and a synthetic sweep. Runtime: seconds after the shared analysis run.

    python experiments/E12_cost.py
"""
import sys

from _common import V, report, rows

v = V()
c = {}
sections = [
    ("Per scenario, end to end (seconds; earlier copy of readable Valentine)", rows(v, [
        ("SF as published", "tpassRefWallScen"),
        ("with FD edges", "tpassFdWallScen"),
        ("with reified n-ary FDs", "tpassReifWallScen"),
        ("memory with reified FDs, GB", "tpassReifRssGB"),
        ("iterating, SF as published, ms", "tpassRefIterMs"),
    ], c)),
    ("Scaling", rows(v, [
        ("exponent of the FD part in |FD_s| x |FD_t| (synthetic sweep)", "swFdProdK"),
        ("one-step term features / SF set-up, small scenarios", "swRatioSmall"),
        ("one-step term features / SF set-up, large scenarios", "swRatioLarge"),
    ], c)),
]
sys.exit(report("E12", "Cost", "What does propagation cost compared with a one-step term?", sections, c,
                note="Timings are descriptive and depend on the machine (servers with 56 hardware threads)."))
