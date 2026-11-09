"""EXPLORATORY T2D replication of E2.

Reuses the registered E2 generator (servers/make_e2_configs.py, unchanged) in-process with the regime set to `t2d`
and the corpus set to harness/generated/t2d-usable: the harness-imported and -profiled T2D tree (t2d-imported) without
the 3 scenarios whose dependency metadata the harness refused to materialise. make_configs.py and
export_seed_study.py are not edited either; their corpus table is extended in this process only.

    python servers/make_t2d_configs.py usable   # build harness/generated/t2d-usable (175 scenarios)
    python servers/make_t2d_configs.py seeds    # stage 1: the E2 initial matchers on t2d -> seedstudy_t2d / seedpy_t2d
    python servers/make_t2d_configs.py arms     # stage 2: seed files + tuning/configs/e2_t2d (scored by explore_e2_t2d.py)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY / "servers"))
import make_configs  # noqa: E402

GEN = STUDY.parent / "harness" / "generated"
USABLE = GEN / "t2d-usable"
REFUSED = {"24036779_0_5608105867560183058", "27466715_0_3913547177671701530", "66009064_0_9148652238372261251"}

make_configs.CORPUS["t2d"] = USABLE
make_configs.EXTERNAL.add("t2d")
import make_e2_configs as e2  # noqa: E402

e2.REGIME = "t2d"
e2.CORPUS = USABLE


def usable() -> int:
    src = GEN / "t2d-imported" / "T2D-Unionable"
    if USABLE.exists():
        shutil.rmtree(USABLE)
    dst = USABLE / "T2D-Unionable"
    dst.mkdir(parents=True)
    n = 0
    for s in sorted(p for p in src.iterdir() if p.is_dir()):
        if s.name in REFUSED:
            continue
        assert (s / "metadata" / "source").is_dir() and (s / "metadata" / "target").is_dir(), s.name
        shutil.copytree(s, dst / s.name)
        n += 1
    print(f"t2d-usable: {n} scenarios (of {n + len(REFUSED)}; refused: {sorted(REFUSED)})")
    return 0


def seeds() -> int:
    # make_configs.py is run in this process (not as a subprocess) so that it sees the extended corpus table
    argv = sys.argv
    sys.argv = ["make_configs.py", "--regime", "t2d", "--no-sf", "--seeds", ",".join(e2.MATCHERS + e2.CONTROLS)]
    try:
        make_configs.main()
    finally:
        sys.argv = argv
    import subprocess
    real_run = subprocess.run
    subprocess.run = lambda *a, **k: None          # e2.seeds() would call make_configs.py again as a subprocess
    try:
        return e2.seeds()
    finally:
        subprocess.run = real_run


if __name__ == "__main__":
    raise SystemExit({"usable": usable, "seeds": seeds, "arms": e2.arms}[sys.argv[1]]())
