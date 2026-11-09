"""Rerun 549 check: std_<p|m>382fix_scores against the seed study results/<plain|md5>/scores_tiefree.csv.

The initial-matrix columns must be identical (same matrices, same random orders). The SF columns differ only through
the fixed HyFD profiles; report per configuration the coverage and the mean change of SF's J_tf over the scenarios.

    ../harness/.venv-bdi/Scripts/python.exe -B reference/check_std_fix.py --regime plain
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

STUDY = Path(__file__).resolve().parent.parent
METRICS = ("F1_H", "F1_Hr", "F1_M", "R_pess", "R_exp", "MRR_pess", "MRR_exp")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5"), required=True)
    a = ap.parse_args()
    batch = {"plain": "std_p382fix", "md5": "std_m382fix"}[a.regime]
    old = {(r["seed"], r["dataset"], r["scenario"]): r
           for r in csv.DictReader(open(STUDY / "results" / a.regime / "scores_tiefree.csv", encoding="utf-8"))}
    new = list(csv.DictReader(open(STUDY / "tuning" / "results" / f"{batch}_scores" / "scores.csv", encoding="utf-8")))
    seed_mismatch, missing_old = 0, 0
    per = defaultdict(list)
    for r in new:
        o = old.get((r["seed"], r["dataset"], r["scenario"]))
        if o is None:
            missing_old += 1
            continue
        if any(abs(float(r[f"seed_{m}"]) - float(o[f"seed_{m}"])) > 1e-12 for m in METRICS):
            seed_mismatch += 1
        j_new = (float(r["sf_R_exp"]) + float(r["sf_F1_Hr"])) / 2
        j_old = (float(o["sf_R_exp"]) + float(o["sf_F1_Hr"])) / 2
        per[r["seed"]].append(j_new - j_old)
    old_n = defaultdict(int)
    for s, _, _ in old:
        old_n[s] += 1
    print(f"{batch}: {len(new)} rows; initial-matrix columns differing from the seed study: {seed_mismatch}; "
          f"rows without an old counterpart: {missing_old}")
    print("seed | scenarios new/old | changed scenarios | mean change of SF J_tf | largest |change|")
    for s in sorted(per):
        d = per[s]
        ch = sum(1 for x in d if abs(x) > 1e-12)
        print(f"{s} | {len(d)}/{old_n[s]} | {ch} | {mean(d):+.4f} | {max(abs(x) for x in d):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
