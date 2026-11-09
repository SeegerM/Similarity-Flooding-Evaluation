"""EXPLORATORY X30 (user's request): the datatype matcher without its NaN cells.

Cause: a column with no non-null value gets NaN type scores (0/0 in Datatype.calculateScores); the
DatatypeMatcher's string branch, 1 - min(score_a, score_b), passes the NaN on. 86 of 382 scenarios (plain and
MD5) and 27 of 81 (independent corpus) have NaN cells. SF never reads the DatatypeMatcher (its graph takes the
column type from Datatype.determineDatatype), so SF's graphs are unaffected.

Fix (dev copy devjava_b/.../matching/metadata/DatatypeMatcher.java, compiled to devclasses_b, off by
default): with -Dsmx.datatype.nanAsZero=true a NaN cell scores 0, no evidence of compatibility. Every finite
cell is unchanged.

This reruns the seed study's `datatype` configuration (first-line DatatypeMatcher, then SF in the standard
configuration, as configs/<regime>/datatype) with the fix, for plain, md5 and indep, into
results/<regime>/datatype_nanfix, and checks every scenario: finite cells of the old matrix identical, NaN cells
now 0. The fixed initial matrices are exported to reference/seeds/<regime>/datatype_nanfix. The shared source,
target/classes and the old seeds are untouched.

    ../harness/.venv-bdi/Scripts/python.exe run_datatype_nanfix.py --run
    ../harness/.venv-bdi/Scripts/python.exe run_datatype_nanfix.py --check
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE / "tuning"))
sys.path.insert(0, str(HERE / "layer3b"))

from smx.adapters import read_matrix  # noqa: E402
from score_seeds import latest_run, one  # noqa: E402

REGIMES = ("plain", "md5", "indep")
DEV_CLASSES = HERE / "devclasses_b"


def prepare(reg: str) -> Path:
    src, dst = HERE / "configs" / reg / "datatype", HERE / "configs" / reg / "datatype_nanfix"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    g = (dst / "general.yaml").read_text(encoding="utf-8")
    old = f"results/{reg}/datatype\""
    assert old in g, reg
    (dst / "general.yaml").write_text(g.replace(old, f"results/{reg}/datatype_nanfix\""), encoding="utf-8")
    (dst / "jvm.properties").write_text("smx.datatype.nanAsZero=true\n", encoding="utf-8")
    return dst


def run_one(reg: str):
    from r8_smoke import run
    from run_seeds import classpath
    cfg = prepare(reg)
    return run(cfg, HERE / "results" / reg / "datatype_nanfix", classpath(), "12g", str(DEV_CLASSES))


def check(regimes=REGIMES) -> None:
    for reg in regimes:
        old_run, new_run = latest_run(HERE / "results" / reg / "datatype"), latest_run(HERE / "results" / reg / "datatype_nanfix")
        n = changed = nan_left = finite_diff = 0
        sf_ok = 0
        for scen in sorted(new_run.glob("*/*")):
            ds, sc = scen.parent.name, scen.name
            f_new = one(scen / "_outputs" / "MatchingStepLine1")
            f_old = one(old_run / ds / sc / "_outputs" / "MatchingStepLine1")
            if not (f_new and f_old):
                continue
            (N, *lab), O = read_matrix(f_new), read_matrix(f_old)[0]
            n += 1
            fin = np.isfinite(O)
            finite_diff += int((np.abs(N[fin] - O[fin]) > 0).sum())
            nan_left += int((~np.isfinite(N)).sum())
            changed += int((~fin).any())
            assert np.all(N[~fin] == 0), (reg, ds, sc)
            out = HERE / "reference" / "seeds" / reg / "datatype_nanfix" / ds / sc
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f_new, out / "seed.csv")
            s2 = one(scen / "_outputs" / "MatchingStepLine2")
            sf_ok += bool(s2 and np.isfinite(read_matrix(s2)[0]).all())
        print(f"{reg}: {n} scenarios; {changed} had NaN cells; finite cells changed: {finite_diff}; NaN left: {nan_left}; "
              f"SF output finite in {sf_ok} of {n}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
    a = ap.parse_args()
    if a.run:
        with ProcessPoolExecutor(3) as pool:
            for r in pool.map(run_one, REGIMES):
                print(r, flush=True)
    if a.check:
        check(a.regimes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
