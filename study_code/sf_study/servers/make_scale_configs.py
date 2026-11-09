"""X51 (EXPLORATORY): configurations for the controlled scaling and memory sweep. Reuses the registered E2 generator (servers/make_e2_configs.py, unchanged) in-process, as
make_t2d_configs.py does, with:
  regime `scale`, corpus harness/generated/scale-imported (93 synthetic scenarios, one per dataset; data/make_scale_corpus.py);
  stage 1: levenshtein (the initial matcher) and datatype (+ datatype_nanfix, E2's D) -> seedstudy_scale;
  stage 2 arms for levenshtein only: melnik, fd (layer1_plain) and fd_reified (layer3_plain) -> tuning/configs/e2_scale
  (the datatype arms e2.arms() also writes are removed; datatype's seed files stay, for the additive term's timing).

    python servers/make_scale_configs.py seeds   # stage 1 configurations (seedstudy_scale)
    python servers/make_scale_configs.py arms    # after stage 1 is collected: seed files + e2_scale
"""

from __future__ import annotations

import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY / "servers"))
import make_configs  # noqa: E402

SCALE = STUDY.parent / "harness" / "generated" / "scale-imported"
make_configs.CORPUS["scale"] = SCALE
make_configs.EXTERNAL.add("scale")
import make_e2_configs as e2  # noqa: E402

e2.REGIME = "scale"
e2.CORPUS = SCALE
e2.CONTROLS = []
e2.ARMS = {"melnik": ("layer1_plain", "melnik", {}), "fd": ("layer1_plain", "fd", {}),
           "fd_reified": ("layer3_plain", "fd_reified", {})}


def seeds() -> int:
    e2.MATCHERS = ["levenshtein", "datatype"]
    argv = sys.argv
    sys.argv = ["make_configs.py", "--regime", "scale", "--no-sf", "--seeds", ",".join(e2.MATCHERS)]
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


def arms() -> int:
    e2.MATCHERS = ["levenshtein", "datatype"]      # seed files for both (datatype: the additive term's D input)
    import shutil
    # e2.arms() also imports the seed-server stage (seedpy_scale), which X51 does not have
    (STUDY / "tuning" / "results" / "seedpy_scale").mkdir(parents=True, exist_ok=True)
    rc = e2.arms()
    out_root = STUDY / "tuning" / "configs" / "e2_scale"
    for d in out_root.glob("datatype__*"):
        shutil.rmtree(d)
    print(f"e2_scale: kept {len(list(out_root.iterdir()))} configurations (levenshtein x {list(e2.ARMS)})")
    return rc


if __name__ == "__main__":
    raise SystemExit({"seeds": seeds, "arms": arms}[sys.argv[1]]())
