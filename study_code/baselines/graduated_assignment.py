"""Is PDED's graduated assignment worth reactivating, and is its flat matrix fixable?

FINDINGS 5i measured PDED's soft matrix as near-uniform -- a 4x4 staying inside
0.249 to 0.250 -- and diagnosed the cause in its affinity term:

    edgeAffinity = exp(-(g1.getWeight(i,j) - g2.getWeight(a,b))^2)

summed over *every* neighbour pair (j, b). Where neither graph has an edge both
weights are zero, the affinity is exp(0) = 1, and in a sparse dependency graph
those agreements outnumber the real edges by orders of magnitude. The score is
then dominated by a constant, which is exactly what a flat matrix looks like.

That is the same defect as the column-role descriptor of FINDINGS 20, in a
different place: evidence is being computed from the absence of structure rather
than from where the structure points. The repair is the same one -- count a
neighbour pair only when at least one side actually has an edge.

This replicates graduated assignment in Python so the question can be settled
before rebuilding the Java system. Two arms differ in one line:

    as-is   every (j, b) pair contributes, absent/absent included
    fixed   pairs where both graphs lack the edge are skipped

Matrix flatness is reported alongside accuracy, because the diagnosis in 5i is
about flatness and a fix has to move that first.

Usage:
    python graduated_assignment.py <data-dir> --dataset ChEMBL-Unionable \
        --first-line <run>
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from class_comparison_conditioned import fd_edges, read_columns  # noqa: E402
from od_role import columns_of, scale_align  # noqa: E402
from role_vs_conditioned import normalise_base  # noqa: E402
from transferability_classes import parse_fds  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402
from smx.relational_roles import relation_matrices  # noqa: E402

EPS = 1e-12


def sinkhorn(m, iters=20):
    """Toward a doubly stochastic matrix, as graduated assignment requires."""
    out = np.clip(m, EPS, None)
    for _ in range(iters):
        out = out / np.maximum(out.sum(axis=1, keepdims=True), EPS)
        out = out / np.maximum(out.sum(axis=0, keepdims=True), EPS)
    return out


def graduated_assignment(w1, w2, seed, skip_absent, betas=(0.5, 1.0, 2.0, 4.0),
                         inner=12):
    """PDED's mechanism: iterate soft assignment against neighbour agreement.

    The affinity tensor is materialised once per scenario. It is
    n1 x n1 x n2 x n2, which is 280k entries at 23 columns a side -- small
    enough to hold, and it makes the update a single contraction.
    """
    n1, n2 = w1.shape[0], w2.shape[0]
    affinity = np.exp(-np.square(w1[:, :, None, None] - w2[None, None, :, :]))
    # A node is not its own neighbour.
    affinity[np.arange(n1), np.arange(n1), :, :] = 0.0
    if skip_absent:
        # Two graphs agreeing that an edge is *missing* is not evidence that
        # their endpoints correspond; it is the most common event in a sparse
        # graph and it is what flattens the matrix.
        both_absent = (w1[:, :, None, None] <= EPS) & (w2[None, None, :, :] <= EPS)
        affinity[both_absent] = 0.0

    m = sinkhorn(np.clip(seed, EPS, None))
    for beta in betas:
        for _ in range(inner):
            score = np.einsum("ijab,jb->ia", affinity, m, optimize=True)
            if score.max() > 0:
                score = score / score.max()
            m = sinkhorn(np.exp(beta * score))
    return m


def flatness(m):
    """How much variation the matrix actually carries."""
    finite = m[np.isfinite(m)]
    if finite.size == 0:
        return 0.0, 0.0
    rng = float(finite.max() - finite.min())
    return float(finite.std()), rng


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--dataset", default="ChEMBL-Unionable")
    ap.add_argument("--first-line", required=True)
    ap.add_argument("--graph", choices=["fd", "gcov"], default="fd")
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("-o", "--out", default="reports/graduated_assignment.csv")
    args = ap.parse_args()

    root = Path(args.data_dir) / args.dataset
    base_of = {}
    for p in sorted(Path(args.first_line).rglob("*.csv")):
        if "MatchingStepLine1" in str(p):
            parts = p.relative_to(args.first_line).parts
            if len(parts) >= 2 and parts[0] == args.dataset:
                base_of.setdefault(parts[1], p)

    rows, stats = [], defaultdict(list)
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

        if args.graph == "fd":
            s_fd = parse_fds(scenario / "metadata" / "source" / s_t[0].stem
                             / "FD_results.txt")
            t_fd = parse_fds(scenario / "metadata" / "target" / t_t[0].stem
                             / "FD_results.txt")
            if not s_fd or not t_fd:
                continue
            w1 = fd_edges(s_fd, columns_of(s_t[0]))
            w2 = fd_edges(t_fd, columns_of(t_t[0]))
        else:
            w1 = relation_matrices(s_vals)[0]["gcov"]
            w2 = relation_matrices(t_vals)[0]["gcov"]

        density = ((w1 > EPS).mean() + (w2 > EPS).mean()) / 2
        stats["density"].append(density)

        arms = {"none": base}
        for label, skip in (("as-is", False), ("fixed", True)):
            m = graduated_assignment(w1, w2, prior, skip)
            sd, rng = flatness(m)
            stats[f"{label} std"].append(sd)
            stats[f"{label} range"].append(rng)
            arms[f"GA {label} alone"] = m
            arms[f"GA {label} + base"] = scale_align(base, m, args.weight)

        for arm, m in arms.items():
            for r in evaluate(ScoreMatrix(scores=m, ground_truth=truth),
                              policies=["hungarian", "mutual_top1"]):
                rows.append({"scenario": name, "arm": arm,
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
    print(f"{args.dataset} / {args.graph} graph: {n} scenarios")
    print(f"  mean edge density: {mean(stats['density']):.3f}")
    print(f"  matrix spread (std / max-min):"
          f"  as-is {mean(stats['as-is std']):.5f} / {mean(stats['as-is range']):.5f}"
          f"   fixed {mean(stats['fixed std']):.5f} / {mean(stats['fixed range']):.5f}\n")

    order = ["none", "GA as-is alone", "GA fixed alone",
             "GA as-is + base", "GA fixed + base"]
    head = f"{'arm':<20}{'extractor':<13}{'F1':>7}{'R@GT':>8}{'MRR':>8}"
    print(head)
    print("-" * len(head))
    for arm in order:
        for ex in ("hungarian", "mutual_top1"):
            sel = [r for r in rows if r["arm"] == arm and r["extractor"] == ex]
            if sel:
                print(f"{arm:<20}{ex:<13}{mean(r['f1'] for r in sel):>7.3f}"
                      f"{mean(r['recall_at_gt'] for r in sel):>8.3f}"
                      f"{mean(r['mrr'] for r in sel):>8.3f}")

    for ex in ("hungarian", "mutual_top1"):
        print(f"\n{ex}")
        by = defaultdict(dict)
        for r in rows:
            if r["extractor"] == ex:
                by[r["scenario"]][r["arm"]] = r["f1"]
        for a, b in (("GA fixed alone", "GA as-is alone"),
                     ("GA fixed + base", "none"),
                     ("GA as-is + base", "none")):
            pr = [(v[a], v[b]) for v in by.values() if a in v and b in v]
            if not pr:
                continue
            W = sum(1 for x, y in pr if x > y)
            L = sum(1 for x, y in pr if x < y)
            p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
            print(f"  {a:<18} vs {b:<16}{mean(x - y for x, y in pr):>+8.3f}"
                  f"  {W}W {L}L {len(pr) - W - L}T  p={p:.4f}")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
