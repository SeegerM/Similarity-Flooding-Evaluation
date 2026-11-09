"""EXPLORATORY: three post-hoc ways of using SF's output, on the existing initial-matcher runs.

Not part of the pre-registered protocol; every number this writes is exploratory. For every
initial matcher and scenario, the run already holds the initial similarities q (MatchingStepLine1,
or the FILE seed) and SF's output S = SF(q) (MatchingStepLine2). From these two matrices:

    initial     q                                    the initial matcher alone
    sf          S                                    SF as run (standard configuration)
    tiebreak    q's order kept; S orders only pairs that q ties
    mix_<l>     l * q/max(q) + (1-l) * S/max(S)       late mixing, no feedback; l in {0.25, 0.5, 0.75}

"Separate outputs" (q for the ranking, S for the assignment) needs no new matrix: it is q's
ranking metrics with S's assignment metrics, and it is assembled in the summary.

Tie-breaker construction: cells with q <= eps stay 0 (the Hungarian extractor drops them, as for
q). Positive cells become q + delta * S/max(S). Here delta is a quarter of the smallest gap between
distinct positive values of q, divided by min(n, m), so that no sum of tie-break terms over an
assignment can reorder two different values or totals of q. If delta falls below 10 * eps,
the tie-breaker cannot act on that matrix; this is counted and reported.

Metrics are the tie-free ones of tuning/score_screen_metrics.py (F1_Hr, R_exp, MRR_exp, plus
F1_M), with the same 50 random row and column orders per scenario for every variant.

Writes results/<suffix>/posthoc_scores.csv and posthoc_summary.csv.

    ../harness/.venv-bdi/Scripts/python.exe explore_posthoc.py --suffix plain
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
from smx.core import DEFAULT_EPS  # noqa: E402
from score_screen_metrics import all_metrics, permutations  # noqa: E402
from score_seeds import CORPUS, PROTOCOL, latest_run, one  # noqa: E402

LAMBDAS = (0.25, 0.5, 0.75)
METRICS = ("F1_Hr", "F1_M", "R_exp", "MRR_exp")


def scaled(M: np.ndarray) -> np.ndarray:
    m = M.max()
    return M / m if m > DEFAULT_EPS else M.copy()


def tiebreak(q: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, bool]:
    pos = q > DEFAULT_EPS
    vals = np.unique(q[pos])
    if vals.size < 1:
        return q.copy(), False
    gaps = np.diff(vals)
    gaps = gaps[gaps > DEFAULT_EPS]
    gap = gaps.min() if gaps.size else max(vals.min(), 1.0)
    delta = gap / 4.0 / min(q.shape)
    if delta < 10 * DEFAULT_EPS:
        return q.copy(), False
    out = q.copy()
    out[pos] = q[pos] + delta * scaled(S)[pos]
    return out, True


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
        gt_file = one(corpus / ds / sc / "ground_truth")
        f_q = (file_seed / ds / sc / "seed.csv") if file_seed else one(scen / "_outputs" / "MatchingStepLine1")
        f_s = one(scen / "_outputs" / "MatchingStepLine2")
        if gt_file is None or f_q is None or not Path(f_q).is_file() or f_s is None:
            continue
        gt = read_matrix(gt_file)[0] > 0.5
        q, S = read_matrix(f_q)[0], read_matrix(f_s)[0]
        if q.shape != gt.shape or S.shape != gt.shape or not (np.isfinite(q).all() and np.isfinite(S).all()):
            continue
        q = np.clip(q, 0.0, None)  # extractors and the tie-breaker assume non-negative scores
        perms = permutations(gt.shape, f"{ds}/{sc}")
        tb, acted = tiebreak(q, S)
        variants = {"initial": q, "sf": S, "tiebreak": tb,
                    **{f"mix_{l}": l * scaled(q) + (1 - l) * scaled(S) for l in LAMBDAS}}
        row = {"seed": seed_dir.name, "dataset": ds, "scenario": sc, "tiebreak_acted": int(acted),
               "q_ties": int(q.size - np.unique(np.round(q, 9)).size)}
        for name, M in variants.items():
            m = all_metrics(M, gt, perms)
            for k in METRICS:
                row[f"{name}_{k}"] = m[k]
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="plain")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    corpus = CORPUS[args.suffix.split("_")[0]]
    root = HERE / "results" / args.suffix
    allowed = json.dumps([f"{c['dataset']}|{c['scenario']}"
                          for c in json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]])
    jobs = [(str(p), str(corpus), str(HERE / "configs" / args.suffix / p.name / "seed_file_dir.txt"), allowed)
            for p in sorted(root.iterdir()) if p.is_dir()]
    with Pool(args.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_seed, jobs) for r in part]
    rows.sort(key=lambda r: (r["seed"], r["dataset"], r["scenario"]))
    with (root / "posthoc_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    names = ["initial", "sf", "tiebreak", *[f"mix_{l}" for l in LAMBDAS]]
    summary = []
    for seed in sorted({r["seed"] for r in rows}):
        rs = [r for r in rows if r["seed"] == seed]
        row = {"seed": seed, "n": len(rs), "tiebreak_acted": mean(r["tiebreak_acted"] for r in rs)}
        for v in names:
            for k in METRICS:
                row[f"{v}_{k}"] = mean(r[f"{v}_{k}"] for r in rs)
        summary.append(row)
    with (root / "posthoc_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"{len(rows)} (initial matcher, scenario) pairs, {len(summary)} initial matchers -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
