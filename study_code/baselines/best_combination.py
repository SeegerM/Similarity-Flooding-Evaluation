"""What is the best channel combination measured here, and how much is selection?

The record contains channels measured one at a time against different bases:
char3 value overlap (13), the uncertainty-coefficient role and the continuous
order role (21), and correspondence-conditioned FDs (20). This evaluates every
subset of them on top of each available first line, so the best combination is a
measurement rather than an inference from separate experiments.

Fusion is a fixed rule, not a tuned one: channels are folded in successively at
one weight, in a fixed order, with no per-scenario or per-subset choice.

**Selection warning.** The best subset here is chosen by looking at the
evaluation ground truth, which is exactly what the rest of this record refuses to
do. The number is an upper bound on what a tuned pipeline could reach, not an
estimate of held-out performance. The honest quantity is the gap between the best
subset and the one a person would have picked in advance, which is reported
alongside.

Usage:
    python best_combination.py <data-dir> --dataset ChEMBL-Unionable \
        --first-line <run> [--first-line <run> ...]
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from class_comparison_conditioned import read_columns  # noqa: E402
from fd_component_ablation import conditioned  # noqa: E402
from od_role import cosine_similarity, columns_of, scale_align  # noqa: E402
from order_relation import order_matrix, to_ordered  # noqa: E402
from order_role import role_profile  # noqa: E402
from role_vs_conditioned import normalise_base  # noqa: E402
from transferability_classes import parse_fds  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402
from smx.relational_roles import relation_matrices  # noqa: E402
from smx.soft_overlap import fuzzy_matrix, normalise, sample_values  # noqa: E402

NULLS = {"", "null", "none", "nan", "na", "n/a", "-"}
CHANNELS = ("char3", "uncertainty", "order", "fd")
ORDER = {c: i for i, c in enumerate(CHANNELS)}


def char3_channel(s_cols, t_cols, seed):
    raw = [[sorted({v.strip() for v in c if v.strip().lower() not in NULLS})
            for c in side] for side in (s_cols, t_cols)]
    norm = [[[v for v in sorted({normalise(x) for x in c}) if v] for c in side]
            for side in raw]
    samp = [[sample_values(c, 128, seed) for c in side] for side in norm]
    return fuzzy_matrix(*samp, kind="char3")


def order_channel(cols, seed):
    prepared = [to_ordered(c, False)[0] for c in cols]
    n_rows = len(cols[0]) if cols else 0
    rng = np.random.default_rng(seed)
    k = min(60000, max(1000, n_rows * 20))
    pi, pj = rng.integers(0, n_rows, k), rng.integers(0, n_rows, k)
    keep = pi != pj
    return np.nan_to_num(order_matrix(prepared, pi[keep], pj[keep]), nan=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--dataset", default="ChEMBL-Unionable")
    ap.add_argument("--first-line", action="append", required=True)
    ap.add_argument("--step", default="MatchingStepLine1",
                    help="which matching step supplies the base matrix")
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("-o", "--out", default="reports/best_combination.csv")
    args = ap.parse_args()

    root = Path(args.data_dir) / args.dataset
    subsets = [()] + [c for r in range(1, len(CHANNELS) + 1)
                      for c in combinations(CHANNELS, r)]
    rows = []

    for run in args.first_line:
        run_path = Path(run)
        base_of = {}
        for p in sorted(run_path.rglob("*.csv")):
            if args.step in str(p):
                parts = p.relative_to(run_path).parts
                if len(parts) >= 2 and parts[0] == args.dataset:
                    base_of.setdefault(parts[1], p)
        if not base_of:
            continue

        for scenario in sorted(p for p in root.iterdir() if p.is_dir()):
            name = scenario.name
            if name not in base_of:
                continue
            gt_f = sorted((scenario / "ground_truth").glob("*.csv"))
            s_t = sorted((scenario / "source").glob("*.csv"))
            t_t = sorted((scenario / "target").glob("*.csv"))
            if not gt_f or not s_t or not t_t:
                continue
            gt, _, _ = read_matrix(gt_f[0])
            base, _, _ = read_matrix(base_of[name])
            s_names, s_vals = read_columns(s_t[0])
            t_names, t_vals = read_columns(t_t[0])
            if base.shape != gt.shape or len(s_names) != gt.shape[0] \
                    or len(t_names) != gt.shape[1]:
                continue
            truth = gt > 0.5
            prior = normalise_base(base)

            s_rel, _, _ = relation_matrices(s_vals)
            t_rel, _, _ = relation_matrices(t_vals)
            s_fd = parse_fds(scenario / "metadata" / "source" / s_t[0].stem
                             / "FD_results.txt")
            t_fd = parse_fds(scenario / "metadata" / "target" / t_t[0].stem
                             / "FD_results.txt")
            s_cols_n, t_cols_n = columns_of(s_t[0]), columns_of(t_t[0])

            channel = {
                "char3": char3_channel(s_vals, t_vals, args.seed),
                "uncertainty": cosine_similarity(
                    role_profile(s_rel["uncertainty"], len(s_names)),
                    role_profile(t_rel["uncertainty"], len(t_names))),
                "order": cosine_similarity(
                    role_profile(order_channel(s_vals, args.seed), len(s_names)),
                    role_profile(order_channel(t_vals, args.seed), len(t_names))),
                "fd": conditioned(s_fd, t_fd, s_cols_n, t_cols_n, prior, "full")
                if s_fd and t_fd else None,
            }

            for subset in subsets:
                if any(channel[c] is None for c in subset):
                    continue
                fused = base
                for c in sorted(subset, key=ORDER.get):
                    fused = scale_align(fused, channel[c], args.weight)
                for r in evaluate(ScoreMatrix(scores=fused, ground_truth=truth),
                                  policies=["hungarian", "mutual_top1"]):
                    rows.append({"run": run_path.name, "scenario": name,
                                 "subset": "+".join(subset) or "base only",
                                 "extractor": r["extractor"], "f1": r["f1"],
                                 "recall_at_gt": r.get("recall_at_gt_pessimistic",
                                                       float("nan")),
                                 "mrr": r.get("mrr_pessimistic", float("nan"))})

    if not rows:
        print("nothing scored", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for run in sorted({r["run"] for r in rows}):
        sel = [r for r in rows if r["run"] == run]
        n = len({r["scenario"] for r in sel})
        base_f1 = mean(float(r["f1"]) for r in sel if r["subset"] == "base only"
                       and r["extractor"] == "hungarian")
        print(f"\n=== {run}  ({n} scenarios, base F1 {base_f1:.3f}) ===")
        head = (f"{'subset':<34}{'F1 hung':>9}{'F1 mt1':>9}{'R@GT':>8}{'MRR':>8}")
        print(head)
        print("-" * len(head))
        table = []
        for subset in sorted({r["subset"] for r in sel}):
            def agg(field, ex):
                v = [float(r[field]) for r in sel if r["subset"] == subset
                     and r["extractor"] == ex]
                return mean(v) if v else float("nan")
            table.append((agg("f1", "hungarian"), agg("f1", "mutual_top1"),
                          agg("recall_at_gt", "hungarian"),
                          agg("mrr", "hungarian"), subset))
        table.sort(reverse=True)
        for f1h, f1m, rg, mr, subset in table:
            mark = " <-- best F1" if (f1h, f1m, rg, mr, subset) == table[0] else ""
            print(f"{subset:<34}{f1h:>9.3f}{f1m:>9.3f}{rg:>8.3f}{mr:>8.3f}{mark}")
        best_rg = max(table, key=lambda t: t[2])
        print(f"  best Recall@GT: {best_rg[4]} at {best_rg[2]:.3f}")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
