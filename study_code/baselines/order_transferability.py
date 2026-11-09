"""Does the transferability oracle hold for order, on a schema too wide to discover ODs?

FINDINGS 15 could test order dependencies only on `partial050`, the one cohort
narrow enough for exhaustive OD discovery -- and 16 then showed every OD there
is vacuous. The question it left open was whether transferability, which
replicates decisively for FDs, holds for *order* on a cohort with complete
correspondences.

The continuous relation makes that testable. Thresholding mono(a -> b) >= tau
yields a set of unary order dependencies on any schema width, so the identical
four-arm design -- none, all, GT-transferable, random same-size -- runs on
23-column ChEMBL-Unionable, where discovery does not finish at all.

Usage:
    python order_transferability.py <data-dir> --dataset ChEMBL-Unionable \
        --first-line <run> --tau 0.99
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from od_role import cosine_similarity, columns_of, scale_align, od_profile  # noqa: E402
from order_role import read_order_matrix  # noqa: E402
from transferability_classes import correspondence  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402


def threshold_ods(matrix, labels, tau):
    """Unary order dependencies implied by the continuous relation."""
    if matrix is None:
        return []
    out = []
    n = min(matrix.shape[0], len(labels))
    for a in range(n):
        for b in range(n):
            if a != b and np.isfinite(matrix[a, b]) and matrix[a, b] >= tau:
                out.append(((labels[a],), (labels[b],)))
    return out


def transferable(src_ods, tgt_ods, mapping):
    """Source ODs whose image under the correspondence is a target OD.

    Position is preserved, as in FINDINGS 15: an order dependency is not
    symmetric, so mapping the left operand onto the right one would be a
    different dependency rather than the same one.
    """
    tgt = set(tgt_ods)
    keep = []
    for lhs, rhs in src_ods:
        img_l = tuple(mapping.get(c) for c in lhs)
        img_r = tuple(mapping.get(c) for c in rhs)
        if None in img_l or None in img_r:
            continue
        if (img_l, img_r) in tgt:
            keep.append((lhs, rhs))
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--first-line", required=True)
    ap.add_argument("--order-dir", default="order_out")
    ap.add_argument("--tau", type=float, default=0.99)
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("-o", "--out", default="reports/order_transferability.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.data_dir) / args.dataset

    base_of = {}
    for p in sorted(Path(args.first_line).rglob("*.csv")):
        if "MatchingStepLine1" in str(p):
            parts = p.relative_to(args.first_line).parts
            if len(parts) >= 2 and parts[0] == args.dataset:
                base_of.setdefault(parts[1], p)

    rows, shares, sizes = [], [], []
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
        s_cols, t_cols = columns_of(s_t[0]), columns_of(t_t[0])
        if base.shape != gt.shape or len(s_cols) != gt.shape[0] \
                or len(t_cols) != gt.shape[1]:
            continue
        mapping = correspondence(gt, s_cols, t_cols)

        odir = Path(args.order_dir) / args.dataset / name
        s_m, s_lab = read_order_matrix(odir / "order_source.csv")
        t_m, t_lab = read_order_matrix(odir / "order_target.csv")
        if s_m is None or t_m is None:
            continue
        s_all = threshold_ods(s_m, s_lab, args.tau)
        t_all = threshold_ods(t_m, t_lab, args.tau)
        if not s_all:
            continue
        s_keep = transferable(s_all, t_all, mapping)
        shares.append(len(s_keep) / len(s_all))
        sizes.append(len(s_all))
        # Same size as the oracle arm, so the arms differ in which order
        # dependencies are kept and not in how many.
        s_rand = rng.sample(s_all, min(len(s_keep), len(s_all)))

        arms = {"all": s_all, "transferable": s_keep, "random": s_rand}
        for arm, sub in arms.items():
            role = cosine_similarity(od_profile(sub, s_cols),
                                     od_profile(t_all, t_cols))
            fused = scale_align(base, role, args.weight)
            for r in evaluate(ScoreMatrix(scores=fused, ground_truth=gt > 0.5),
                              policies=["hungarian", "mutual_top1"]):
                rows.append({"scenario": name, "arm": arm,
                             "extractor": r["extractor"], "f1": r["f1"],
                             "recall_at_gt": r.get("recall_at_gt_pessimistic",
                                                   float("nan")),
                             "mrr": r.get("mrr_pessimistic", float("nan"))})
        for r in evaluate(ScoreMatrix(scores=base, ground_truth=gt > 0.5),
                          policies=["hungarian", "mutual_top1"]):
            rows.append({"scenario": name, "arm": "none",
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

    n = len({r["scenario"] for r in rows})
    print(f"{args.dataset}: {n} scenarios, tau={args.tau}")
    print(f"  thresholded ODs per table: {mean(sizes):.1f}"
          f"   transferable share: {mean(shares):.3f}\n")
    head = f"{'arm':<14} {'extractor':<13} {'F1':>7} {'R@GT':>7} {'MRR':>7}"
    print(head)
    print("-" * len(head))
    for arm in ("none", "all", "transferable", "random"):
        for ex in ("hungarian", "mutual_top1"):
            sel = [r for r in rows if r["arm"] == arm and r["extractor"] == ex]
            if sel:
                print(f"{arm:<14} {ex:<13} {mean(r['f1'] for r in sel):>7.3f} "
                      f"{mean(r['recall_at_gt'] for r in sel):>7.3f} "
                      f"{mean(r['mrr'] for r in sel):>7.3f}")
    print()
    for ex in ("hungarian", "mutual_top1"):
        by = defaultdict(dict)
        for r in rows:
            if r["extractor"] == ex:
                by[r["scenario"]][r["arm"]] = r["f1"]
        for a, b in (("transferable", "all"), ("transferable", "random"),
                     ("all", "none")):
            pr = [(v[a], v[b]) for v in by.values() if a in v and b in v]
            if not pr:
                continue
            W = sum(1 for x, y in pr if x > y)
            L = sum(1 for x, y in pr if x < y)
            p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
            print(f"{a} vs {b} ({ex}): {mean(x - y for x, y in pr):+.3f} "
                  f"{W}W {L}L {len(pr) - W - L}T  p={p:.4f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
