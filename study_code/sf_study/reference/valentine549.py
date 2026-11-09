"""Rerun 549 (2026-11-01): what the report scripts need under SMX_VALENTINE=549 to read the assembled 549-scenario inputs
(reference/assemble_549.py) instead of the 382 folders. Unset or SMX_VALENTINE=382: V549 is False and nothing here is
used, so every script keeps its 382 behaviour byte for byte.

  vreg(regime)              plain -> plain549, md5 -> md549 (the assembled folders' suffix)
  cases549()                (dataset, scenario, family) of the 549 scenarios (results/valentine549/cases.csv), in the
                            381 protocol order followed by the 168 new ones
  assembled(folder)         rows of tuning/results/<folder>/scores.csv, or None while that family is not assembled
  arm_status(family, arm)   the arm's status in results/valentine549/assembly.json ("assembled", "pending", ...)
  corpus_dir(regime, ds, sc)  scenario folder with the FIXED FD profiles: harness/generated_hyfdfix/valentine_<r> for the
                            381, harness/generated/valentine-new168(-md5) for the 168
  seed_file(regime, m, ds, sc)  the initial matrix: reference/seeds/<r>/<m> for the 381 (prisma: <r>_fdfix, the fixed-FD
                            recomputation; rema/rema_shuffled: <r>_fill where the old copy lacks the scenario), and
                            reference/seeds/<new>/<m> for the 168
"""

from __future__ import annotations

import csv
import json
import os
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent
RES = STUDY / "tuning" / "results"
V549 = os.environ.get("SMX_VALENTINE", "382") == "549"
NAME = {"plain": "plain549", "md5": "md549"}
NEW = {"plain": "pnew", "md5": "mnew"}
FIXED_TREE = {"plain": ROOT / "harness" / "generated_hyfdfix" / "valentine_plain",
              "md5": ROOT / "harness" / "generated_hyfdfix" / "valentine_md5"}
NEW_TREE = {"plain": ROOT / "harness" / "generated" / "valentine-new168",
            "md5": ROOT / "harness" / "generated" / "valentine-new168-md5"}
SEEDS = STUDY / "reference" / "seeds"


# External FD stages (not Valentine): under SMX_VALENTINE=549 (the rerun state: 549 scenarios AND the fixed HyFD
# profiles everywhere) the FD arms of the external corpora come from the *fix reruns, the non-FD arms from the old
# batches, and FD-derived features (role vectors) from the fixed-FD mirrors in harness/generated_hyfdfix/<corpus>.
HYFDFIX = V549
FIX_PREFIX = (("iter_x81_", "iterx81fix_"), ("iter_x54_", "iterx54fix_"), ("iter_x55_", "iterx55fix_"),
              ("x81_", "x81fix_"), ("wide_", "widefix_"), ("x54_", "x54fix_"), ("x55_", "x55fix_"),
              ("parallel_", "parallelfix_"))
FIXED_MIRROR = ROOT / "harness" / "generated_hyfdfix"


def is_fd_arm(arm: str) -> bool:
    return arm in ("fd", "ucc1_fd_topk", "coef_prod_fd") or arm.startswith("fd_")


def fix_stage(stage: str) -> str | None:
    """The fixed-FD rerun of an external stage (x81_goby -> x81fix_goby), or None if no rerun is named for it."""
    for old, new in FIX_PREFIX:
        if stage.startswith(old):
            return new + stage[len(old):]
    return None


def fix_ready(stage: str) -> bool:
    f = fix_stage(stage)
    return f is not None and (RES / f / "COLLECTED").is_file()


def run_stage(stage: str, arm: str) -> str:
    """The stage folder a run of `arm` is read from: the *fix rerun for FD arms (HYFDFIX), else the stage itself."""
    return fix_stage(stage) if HYFDFIX and is_fd_arm(arm) and fix_stage(stage) else stage


FIX = {"plain": "p382fix", "md5": "m382fix"}


def sf_run(regime: str, m: str, arm: str, ds: str, sc: str) -> Path | None:
    """The merged run folder holding SF's output of (matcher, arm) on a Valentine scenario of the 549, by the assembly
    rules: 168 from the <new> batches (prisma only prismafs_<new>), 381 from the old stage for non-FD arms and from the
    *fix batches for FD arms (prisma only prismafix_<fix>; the rema fill from remafill_<fix>). None if no run has it."""
    fx, nw = FIX[regime], NEW[regime]
    if is_new(ds, sc):
        stages = [f"prismafs_{nw}"] if m == "prisma" else \
            [f"layer1_{nw}", f"layer1fs_{nw}", f"layer1nan_{nw}", f"layer1isr_{nw}", f"iter_{nw}", f"iterfs_{nw}",
             f"gvfull_{nw}", f"gvfullfs_{nw}", f"parallel_{nw}", f"parallelfs_{nw}"]
    elif m == "prisma":
        stages = [f"prismafix_{fx}"]
    elif is_fd_arm(arm):
        stages = [f"remafill_{fx}", f"l1fd_{fx}", f"l1fdisr_{fx}", f"l1nanfix_{fx}", f"l1cpfdisr_{fx}", f"gvfull_{fx}",
                  f"parallelfix_{fx}"]
    else:
        stages = [f"remafill_{fx}", f"layer1_{regime}", f"layer1nan_{regime}", f"layer1isr_{regime}", f"iter_{regime}",
                  f"gvfull_{regime}"]
    for s in stages:
        d = RES / s / f"{m}__{arm}" / "merged"
        if (d / ds / sc).is_dir():
            return d
    return None


def vreg(regime: str) -> str:
    return NAME.get(regime, regime) if V549 else regime


@lru_cache(maxsize=None)
def _cases() -> tuple:
    return tuple((c["dataset"], c["scenario"], c["family"], c["cohort"])
                 for c in csv.DictReader(open(STUDY / "results" / "valentine549" / "cases.csv", encoding="utf-8")))


def cases549() -> list[tuple[str, str, str]]:
    return [c[:3] for c in _cases()]


def is_new(ds: str, sc: str) -> bool:
    return (ds, sc) in {c[:2] for c in _cases() if c[3] == "new168"}


def assembled(folder: str) -> list[dict] | None:
    f = RES / folder / "scores.csv"
    if not (RES / folder / "ASSEMBLED_549").is_file() or not f.is_file():
        return None
    return list(csv.DictReader(open(f, encoding="utf-8")))


def arm_status(family: str, arm: str | None = None) -> str:
    a = json.loads((STUDY / "results" / "valentine549" / "assembly.json").read_text(encoding="utf-8"))
    e = a["families"].get(family, {})
    return e.get("status", "missing") if arm is None else e.get("arms", {}).get(arm, {}).get("status", "missing")


def corpus_dir(regime: str, ds: str, sc: str) -> Path:
    return (NEW_TREE[regime] if is_new(ds, sc) else FIXED_TREE[regime]) / ds / sc


def seed_file(regime: str, m: str, ds: str, sc: str) -> Path:
    if is_new(ds, sc):
        return SEEDS / NEW[regime] / m / ds / sc / "seed.csv"
    if m == "prisma":
        return SEEDS / f"{regime}_fdfix" / m / ds / sc / "seed.csv"
    f = SEEDS / regime / m / ds / sc / "seed.csv"
    if m in ("rema", "rema_shuffled") and not f.is_file():
        return SEEDS / f"{regime}_fill" / m / ds / sc / "seed.csv"
    return f
