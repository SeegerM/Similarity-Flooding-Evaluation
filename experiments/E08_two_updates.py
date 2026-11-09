"""E08: is SF stopped after two updates as good as converged SF? (rule specified in advance, tested on held-out data)

Paper: Section V-E and Table III (rows "Two upd. - conv."). Inputs: the held-out iteration runs in
results/sf_study/results/x88/. Runtime: seconds after the shared analysis run.

    python experiments/E08_two_updates.py
"""
import sys

from _common import V, report, rows

v = V()
c = {}
sections = [
    ("Two updates minus convergence, J_tf (non-inferiority margin -0.01, fixed in advance)", rows(v, [
        ("T2D web tables", "svTtd~ci"),
        ("T2D verdict", "svTtdVerdict"),
        ("Lahman/MusicBrainz", "svLmS~ci"),
        ("Lahman/MusicBrainz verdict", "svLmSVerdict"),
        ("Synthea to OMOP", "svSyn~ci"),
        ("Synthea verdict", "svSynVerdict"),
        ("Lahman/MusicBrainz multi-table", "svLmM~ci"),
        ("Lahman/MusicBrainz multi-table verdict", "svLmMVerdict"),
        ("corpora non-inferior / superior / tested", "svNonInf"),
        ("superior", "svSup"),
        ("tested", "svN"),
    ], c)),
    ("Closed form of the second update", rows(v, [
        ("single-table scenarios checked", "tuSingleN"),
        ("of these where the one-step offset B is not constant", "tuNonConst"),
    ], c)),
]
sys.exit(report("E08", "Two updates instead of convergence",
                "Does SF stopped after two updates rank as well as converged SF on corpora not used to fix that "
                "rule, and why?", sections, c))
