"""H1 of PREREGISTRATION_LAYERS.md: on single-table scenarios, SF without dependency edges is affine
in the initial matrix within each datatype-pair block, so it never inverts the seed's order inside
a block.

For every collected layer-1 configuration and protocol scenario: datatype classes are the distinct
rows (source columns) and columns (target columns) of the `datatype` seed's matrix; a within-block
inversion is a pair of cells in one block with q higher by more than 1e-9 and SF lower by more than
1e-6. Arms without dependency edges must have zero inversions; arms with them are expected not to.

Writes tuning/results/layer1_<regime>_scores/prop1.csv (per seed x arm) and prints per arm totals.

    ../harness/.venv-bdi/Scripts/python.exe reference/check_prop1.py --regime plain
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "harness"))
sys.path.insert(0, str(STUDY))
from smx.adapters import read_matrix  # noqa: E402
from score_seeds import cases_for, latest_run, one  # noqa: E402

NO_DEPENDENCIES = {"melnik", "coef_prod", "attach_id", "lit_none"}


def classes(M: np.ndarray) -> np.ndarray:
    """Block of every cell: identical rows define source classes, identical columns target classes."""
    _, ci = np.unique(M.round(7), axis=0, return_inverse=True)
    _, cj = np.unique(M.T.round(7), axis=0, return_inverse=True)
    return (ci.ravel()[:, None] * (cj.max() + 1) + cj.ravel()[None, :]).ravel()


def check(job):
    regime, label_dir, cases, how = job
    label_dir = Path(label_dir)
    seed, arm = label_dir.name.split("__", 1)
    run = label_dir / "merged" if (label_dir / "merged").is_dir() else sorted(
        p for p in label_dir.iterdir() if p.is_dir() and p.name != "_shards")[-1]
    dt_run = latest_run(STUDY / "results" / regime / "datatype")
    inv = pairs = scen = 0
    for ds, sc, _ in cases:
        f_s = one(run / ds / sc / "_outputs" / "MatchingStepLine2")
        f_q = HERE / "seeds" / regime / seed / ds / sc / "seed.csv"
        if how == "constant":
            # Amendment 1: the datatype blocks SF itself uses, read from its output on the constant seed
            # under the reference (Proposition 1: SF(constant) = b(datatype pair); literal seeds separate
            # the datatype names, so this partition is the finest)
            f_d = one(label_dir.parent / "constant__melnik" / "merged" / ds / sc / "_outputs" / "MatchingStepLine2")
        else:
            f_d = one(dt_run / ds / sc / "_outputs" / "MatchingStepLine1")
        if f_s is None or not f_q.is_file() or f_d is None:
            continue
        S, q, D = read_matrix(f_s)[0], read_matrix(f_q)[0], read_matrix(f_d)[0]
        if S.shape != q.shape or D.shape != q.shape:
            continue
        if not np.isfinite(D).all():
            continue
        block = classes(D)
        s, qq = S.ravel(), q.ravel()
        for b in np.unique(block):
            idx = np.where(block == b)[0]
            if len(idx) < 2:
                continue
            dq = qq[idx][:, None] - qq[idx][None, :]
            dS = s[idx][:, None] - s[idx][None, :]
            inv += int(((dq > 1e-9) & (dS < -1e-6)).sum())
            pairs += int((dq > 1e-9).sum())
        scen += 1
    return {"seed": seed, "arm": arm, "scenarios": scen, "ordered_pairs": pairs, "inversions": inv}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="plain")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--classes", choices=("datatype", "constant"), default="constant",
                    help="datatype: the datatype seed's rows/columns (pre-registered; too coarse, see "
                         "Amendment 1); constant: SF's output on the constant seed under melnik")
    a = ap.parse_args()
    root = STUDY / "tuning" / "results" / f"layer1_{a.regime}"
    cases = cases_for(a.regime)
    jobs = [(a.regime, str(d), cases, a.classes) for d in sorted(root.iterdir()) if (d / "DONE").is_file() and "__" in d.name]
    with Pool(a.workers) as pool:
        rows = list(pool.imap_unordered(check, jobs))
    rows.sort(key=lambda r: (r["arm"], r["seed"]))
    out = STUDY / "tuning" / "results" / f"layer1_{a.regime}_scores"
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"prop1_{a.classes}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tot = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        t = tot[r["arm"]]
        t[0] += 1; t[1] += r["scenarios"]; t[2] += r["ordered_pairs"]; t[3] += r["inversions"]
    for arm, (n, sc, p, i) in sorted(tot.items()):
        expect = "0 required" if arm in NO_DEPENDENCIES else "> 0 expected"
        bad = sum(1 for r in rows if r["arm"] == arm and r["inversions"] > 0)
        print(f"{arm:14s} {n:3d} seeds, {sc} scenario runs: {i} inversions of {p} ordered pairs "
              f"({bad} seeds with any; {expect})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
