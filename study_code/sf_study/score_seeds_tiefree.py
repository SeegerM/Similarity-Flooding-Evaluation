"""Score the initial-matcher study on every metric, including the tie-free ones.

Companion to score_seeds.py; it uses the same runs, ground truth and 382-case protocol, and adds
the tie-free variants of Amendment 2 (tuning/PREREGISTRATION.md). For every initial matcher and
scenario, both the initial similarities (MatchingStepLine1, or the FILE seed) and SF's output
(MatchingStepLine2) are scored on:

    F1_H, F1_Hr, F1_M, R_pess, R_exp, MRR_pess, MRR_exp

as defined in tuning/score_screen_metrics.py. F1_Hr uses the same 50 random row and column orders
for both matrices of a scenario, so the difference stays paired.

Writes results/<suffix>/scores_tiefree.csv (one row per initial matcher x scenario) and
results/<suffix>/summary_tiefree.csv (scenario means per initial matcher). score_seeds.py's own
outputs are not touched.

    ../harness/.venv-bdi/Scripts/python.exe score_seeds_tiefree.py --suffix plain
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE / "tuning"))

from smx.adapters import read_matrix  # noqa: E402
from score_screen_metrics import METRICS, all_metrics, permutations  # noqa: E402
from score_seeds import CORPUS, cases_for, ground_truth_file, latest_run, one  # noqa: E402


def score_seed(job: tuple[str, str, str, str]) -> list[dict]:
    seed_dir, corpus, marker, allowed_json = Path(job[0]), Path(job[1]), Path(job[2]), job[3]
    allowed = {tuple(k.split("|")) for k in json.loads(allowed_json)}
    run = latest_run(seed_dir)
    if run is None or not (seed_dir / "DONE").is_file():
        return []
    file_seed = Path(marker.read_text(encoding="utf-8").strip()) if marker.is_file() else None
    out = []
    for scen in sorted(run.glob("*/*")):
        ds, sc = scen.parent.name, scen.name
        if (ds, sc) not in allowed:
            continue
        gt_file = ground_truth_file(corpus, ds, sc)   # multi-table scenarios: ground_truth_full.csv
        if gt_file is None:
            continue
        gt = read_matrix(gt_file)[0] > 0.5
        f_seed = (file_seed / ds / sc / "seed.csv") if file_seed else one(scen / "_outputs" / "MatchingStepLine1")
        f_sf = one(scen / "_outputs" / "MatchingStepLine2")
        if f_seed is None or not Path(f_seed).is_file() or f_sf is None:
            continue
        q, S = read_matrix(f_seed)[0], read_matrix(f_sf)[0]
        if q.shape != gt.shape or S.shape != gt.shape or not (np.isfinite(q).all() and np.isfinite(S).all()):
            continue
        perms = permutations(gt.shape, f"{ds}/{sc}")
        mq, ms = all_metrics(q, gt, perms), all_metrics(S, gt, perms)
        out.append({"seed": seed_dir.name, "dataset": ds, "scenario": sc,
                    **{f"seed_{m}": mq[m] for m in METRICS}, **{f"sf_{m}": ms[m] for m in METRICS}})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    corpus = CORPUS[args.suffix.split("_")[0]]
    root = HERE / "results" / args.suffix
    allowed = json.dumps([f"{c['dataset']}|{c['scenario']}"
                          for c in [{"dataset": d, "scenario": s} for d, s, _ in cases_for(args.suffix.split("_")[0])]])
    jobs = [(str(p), str(corpus), str(HERE / "configs" / args.suffix / p.name / "seed_file_dir.txt"), allowed)
            for p in sorted(root.iterdir()) if p.is_dir()]
    with Pool(args.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_seed, jobs) for r in part]
    rows.sort(key=lambda r: (r["seed"], r["dataset"], r["scenario"]))
    with (root / "scores_tiefree.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for seed in sorted({r["seed"] for r in rows}):
        rs = [r for r in rows if r["seed"] == seed]
        row = {"seed": seed, "n": len(rs)}
        for m in METRICS:
            row[f"seed_{m}"] = mean(r[f"seed_{m}"] for r in rs)
            row[f"sf_{m}"] = mean(r[f"sf_{m}"] for r in rs)
            row[f"delta_{m}"] = row[f"sf_{m}"] - row[f"seed_{m}"]
        summary.append(row)
    with (root / "summary_tiefree.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"{len(rows)} (initial matcher, scenario) pairs, {len(summary)} initial matchers -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
