"""X81 (EXPLORATORY; 2026-10-06 21:36-22:30): external corpora with independently produced
schemas. Same panel, arms and mechanics as X79 (servers/make_knobs_configs.py), one corpus per call:

    hdxsm   harness/generated/hdxsm-usable     (HDXSM, 199 of 204 pairs; 5 cannot be written in the legacy profile text)
    gdcsm   harness/generated/gdcsm-imported   (GDC-SM, 10 studies; target UCC profiled with a 24g heap)
    madi    harness/generated/madi-imported    (MaDI-Bench base, 18)
    goby    harness/generated/goby-imported    (GOBY, 200)
    mimic   sf-matcher-study/data/mimic_omop   (MIMIC-IV demo -> OMOP, multi-table, 1)

Stage 1 -> seedstudy_<c>, seedpy_<c>; first-line matrices -> reference/seeds/<c>/<m>/; SF arms -> tuning/configs/x81_<c>,
runs tuning/results/x81_<c>/<m>__<arm>. On the hosts: $SMX_REMOTE_ROOT/smx/corpus/<c>.

    python servers/make_x81_configs.py seeds|arms <c>
"""

from __future__ import annotations

import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "servers"))
import make_knobs_configs as mk  # noqa: E402

GEN = STUDY.parent / "harness" / "generated"
ROOTS = {"hdxsm": GEN / "hdxsm-usable", "gdcsm": GEN / "gdcsm-imported", "madi": GEN / "madi-imported",
         "goby": GEN / "goby-imported", "mimic": STUDY / "data" / "mimic_omop"}


def setup(c: str) -> None:
    mk.CORPUS, mk.SUFFIX, mk.ROOT = c, f"x81_{c}", ROOTS[c]
    mk.mfc.CORPORA[c] = ROOTS[c]
    for mod in (mk.mfc.make_configs, mk.mfc.score_seeds):
        mod.CORPUS[c] = ROOTS[c]
        mod.EXTERNAL.add(c)


if __name__ == "__main__":
    setup(sys.argv[2])
    raise SystemExit({"seeds": mk.seeds, "arms": mk.arms}[sys.argv[1]]())
