"""X82 (pick (f), EXPLORATORY): X54/X55's 83-matcher initial panel on the new corpora.

`seeds <c>`: stage 1 via servers/make_x5455_configs.seeds UNCHANGED; this wrapper registers the corpus root and drops the
    matchers already collected for <c> (x5455.seeds forces the whole panel) -> seedstudy_<c> / seedpy_<c>.
`arms <c>`: SF melnik and fd via make_x5455_configs.arms UNCHANGED, with its output suffix set to wide_<c>; afterwards the
    ucc_top1 arms (not requested) and the arms of matchers that already have melnik/fd runs on <c> (final_<c>: E2's
    panel; x81_<c>: X79's panel) are removed. levenshtein__melnik_twin is kept as the gate (= <c>'s own
    levenshtein__melnik cell for cell). Runs: tuning/results/wide_<c>/<m>__<arm>.

    python servers/make_x82_wide.py seeds|arms final_indep|final_controlled|hdxsm|gdcsm|madi|goby|mimic
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "servers"))
import make_x5455_configs as x5455  # noqa: E402

GEN = STUDY.parent / "harness" / "generated"
ROOTS = {"final_indep": STUDY / "data" / "final_indep", "final_controlled": GEN / "final-controlled",
         "hdxsm": GEN / "hdxsm-usable", "gdcsm": GEN / "gdcsm-imported", "madi": GEN / "madi-imported",
         "goby": GEN / "goby-imported", "mimic": STUDY / "data" / "mimic_omop"}
CFG = STUDY / "tuning" / "configs"


def existing_sf(c: str) -> Path:
    return CFG / (c if c.startswith("final_") else f"x81_{c}")


def seeds(c: str) -> int:
    rc = x5455.seeds(c)
    dropped = []
    for suf in (f"seedstudy_{c}", f"seedpy_{c}"):
        for d in sorted((CFG / suf).iterdir()):
            if (STUDY / "results" / c / d.name / "DONE").is_file():
                shutil.rmtree(d)
                dropped.append(d.name)
    print(f"{c}: already collected, not rerun: {dropped}")
    return rc


def arms(c: str) -> int:
    x5455.CORPORA[c] = (ROOTS[c], f"wide_{c}")
    # x5455.arms reads results/<c>/<m> for every panel matcher and crashes on a matcher whose stage 1 never completed
    # (magneto on final_indep, valentine_distribution on gdcsm/hdxsm): such matchers are left out (listed)
    import export_seed_study as ess
    for py in (False, True):
        (STUDY / "tuning" / "results" / ess.suffix_of(c, py)).mkdir(parents=True, exist_ok=True)
        ess.import_cluster_results(c, python=py)
    absent = [m for m in x5455.MATCHERS if not (STUDY / "results" / c / m).is_dir()]
    x5455.MATCHERS = [m for m in x5455.MATCHERS if m not in absent]
    print(f"{c}: no stage-1 results, left out: {absent}")
    rc = x5455.arms(c)
    have = {p.name.split("__")[0] for p in existing_sf(c).iterdir() if p.is_dir() and p.name.endswith("__melnik")}
    out = CFG / f"wide_{c}"
    removed = []
    for d in sorted(out.iterdir()):
        m, arm = d.name.split("__", 1)
        if arm == "ucc_top1" or (m in have and arm in ("melnik", "fd")):
            shutil.rmtree(d)
            removed.append(d.name)
    left = sorted(p.name for p in out.iterdir())
    print(f"wide_{c}: {len(left)} configurations kept ({sum(l.endswith('__melnik') for l in left)} melnik, "
          f"{sum(l.endswith('__fd') for l in left)} fd, twin {'levenshtein__melnik_twin' in left}); removed {len(removed)}")
    return rc


if __name__ == "__main__":
    cmd, c = sys.argv[1], sys.argv[2]
    for mod in (x5455.make_configs, x5455.score_seeds):
        mod.CORPUS[c] = ROOTS[c]
        mod.EXTERNAL.add(c)
    raise SystemExit({"seeds": seeds, "arms": arms}[cmd](c))
