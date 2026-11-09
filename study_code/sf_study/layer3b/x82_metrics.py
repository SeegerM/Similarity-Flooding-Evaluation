"""X82 (EXPLORATORY; EXPLORATORY.md X82, written 2026-10-06 before any X82 output existed): the per-run metric
rows that layer3b/score_x82.py reads, tuning/results/x82_metrics/<family>.csv with the columns
    corpus, dataset, scenario, matcher, arm, J, R_exp, F1_Hr, stage

For every corpus c and every (matcher, arm) of the family, the SF output MatchingStepLine2 of the arm's run folder is scored
against the scenario's full truth matrix (score_seeds.ground_truth_file: the single ground_truth file, or ground_truth_full.csv
for multi-table scenarios) with score_r7.metrics, i.e. J_tf = (R_exp + F1_Hr)/2 over the 50 seeded tie permutations of
score_screen_metrics.permutations(shape, "<dataset>/<scenario>"). The same code scores the arms and their bases.

Arms and where they are read (the base stage is c itself for final_*, else x81_<c>):
    parallel   fd_split (parallel_<c>); base fd (base stage)
    classfull  melnik_shape, melnik_shape_shuffled, melnik_semtype, melnik_semtype_shuffled (classes_<c>); base melnik
    e2x        coef_prod_fd, ucc, fd_top1 (e2rest_<c>), melnik_top1 (r8_<c>); bases ucc_top1, melnik (base stage)
Gate, devnone, twin and check configurations are collection checks and are not scored here. final_controlled rows carry
the database (AdventureWorks, IMDB, SCOPe, Sakila) as `dataset`, so score_x82's macro is over databases; the scenario
column is "<dataset folder>/<scenario>". A matrix whose shape differs from the truth, or with a non-finite cell, is
skipped and counted.

    python layer3b/x82_metrics.py --family parallel|classfull|e2x [--corpora c1,c2] [--workers 10]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STUDY))
from score_r7 import metrics, run_dir  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402
from score_seeds import ground_truth_file, one  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

RES = STUDY / "tuning" / "results"
GEN = STUDY.parent / "harness" / "generated"
ROOTS = {"final_indep": STUDY / "data" / "final_indep", "final_controlled": GEN / "final-controlled",   # as
         "hdxsm": GEN / "hdxsm-usable", "gdcsm": GEN / "gdcsm-imported", "madi": GEN / "madi-imported",   # servers/
         "goby": GEN / "goby-imported", "mimic": STUDY / "data" / "mimic_omop"}                           # make_x82_classes
FAMILIES = {
    "parallel": {"parallel_{c}": ["fd_split"], "BASE": ["fd"]},
    "classfull": {"classes_{c}": ["melnik_shape", "melnik_shape_shuffled", "melnik_semtype", "melnik_semtype_shuffled"],
                  "BASE": ["melnik"]},
    "e2x": {"e2rest_{c}": ["coef_prod_fd", "ucc", "fd_top1"], "r8_{c}": ["melnik_top1"], "BASE": ["ucc_top1", "melnik"]},
}


def base_stage(c: str) -> str:
    return c if c.startswith("final_") else f"x81_{c}"


def arm_dirs(family: str, c: str) -> dict[tuple[str, str], tuple[str, Path]]:
    """(matcher, arm) -> (stage, run folder) for every collected run of the family's arms and bases on corpus c."""
    out = {}
    for stage_t, arms in FAMILIES[family].items():
        stage = base_stage(c) if stage_t == "BASE" else stage_t.format(c=c)
        root = RES / stage
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir() and "__" in p.name):
            m, arm = d.name.split("__", 1)
            if arm in arms:
                r = run_dir(d)
                if r is not None:
                    out[(m, arm)] = (stage, r)
    return out


def job(args):
    c, ds, sc, runs = args
    g = ground_truth_file(ROOTS[c], ds, sc)
    if g is None:
        return [], Counter({"no_truth": 1})
    gt = read_matrix(g)[0] > 0.5
    perms = permutations(gt.shape, f"{ds}/{sc}")
    dataset = ds.split("-")[0] if c == "final_controlled" else ds
    rows, skip = [], Counter()
    for (m, arm), (stage, r) in runs.items():
        o = one(r / ds / sc / "_outputs" / "MatchingStepLine2")
        if o is None:
            continue
        S = read_matrix(o)[0]
        if S.shape != gt.shape or not np.isfinite(S).all():
            skip["shape_or_nonfinite"] += 1
            continue
        x = metrics(S, gt, perms)
        rows.append({"corpus": c, "dataset": dataset, "scenario": f"{ds}/{sc}", "matcher": m, "arm": arm,
                     "J": x["J"], "R_exp": x["R_exp"], "F1_Hr": x["F1_Hr"], "stage": stage})
    return rows, skip


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--corpora", default=",".join(ROOTS))
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    jobs, report = [], []
    for c in a.corpora.split(","):
        runs = arm_dirs(a.family, c)
        new = {k: v for k, v in runs.items() if v[0] != base_stage(c)}
        if not new:
            report.append(f"{c}: no collected {a.family} arm, skipped")
            continue
        root = ROOTS[c]
        scen = [(d.name, s.name) for d in sorted(root.iterdir()) if d.is_dir()
                for s in sorted(d.iterdir()) if s.is_dir()]
        jobs += [(c, ds, sc, runs) for ds, sc in scen]
        report.append(f"{c}: {len(scen)} scenarios, {len(runs)} (matcher, arm) runs "
                      f"({Counter(k[1] for k in runs)})")
    rows, skip = [], Counter()
    with Pool(a.workers) as pool:
        for r, s in pool.imap_unordered(job, jobs, chunksize=4):
            rows += r
            skip += s
    out = RES / "x82_metrics"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["corpus", "scenario", "matcher", "arm"])
    df.to_csv(out / f"{a.family}.csv", index=False)
    print("\n".join(report))
    print(f"{len(df)} rows -> {out / (a.family + '.csv')}; skipped {dict(skip)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
