"""Which part of the FD scorer produced the FINDINGS 20 effect?

20 measured +0.054 F1 (19W/0L) for correspondence-conditioned scoring over the
role descriptor on identical FDs. 21 attributed that to structure preservation,
because flattening the same FDs to per-column edges halved it. 26 then tested
that attribution directly on two other classes and it failed both times:
preserving DC operators and predicate sets does not beat flattening them, and
neither does preserving composite order left-hand sides.

So the attribution is wrong, or the FD scorer differs from the flattened one in
some way other than "structure". It differs in three ways, and this removes them
one at a time:

    full        determinant set, matched by exact assignment, plus occurrence
                position (determining versus determined)
    no position a source occurrence may match any target occurrence
    singletons  each determinant column becomes its own dependency, so the set
                is gone but the head/body distinction survives
    head only   P depends on the dependant pair alone; the determinant is
                ignored entirely

If `singletons` keeps the effect, the determinant set was never the point. If
`head only` keeps it, none of the FD structure was.

Usage:
    python fd_component_ablation.py <data-dir> --dataset ChEMBL-Unionable \
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

from od_role import cosine_similarity, columns_of, scale_align  # noqa: E402
from role_vs_conditioned import normalise_base, set_agreement  # noqa: E402
from transferability_classes import fd_profile, parse_fds  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402


def conditioned(s_fds, t_fds, s_cols, t_cols, prior, variant):
    n_s, n_t = len(s_cols), len(t_cols)
    out = np.zeros((n_s, n_t))
    if not s_fds or not t_fds:
        return out
    s_index = {c: i for i, c in enumerate(s_cols)}
    t_index = {c: i for i, c in enumerate(t_cols)}

    if variant == "singletons":
        # One dependency per determinant column: the set is dissolved, the
        # head/body distinction is kept.
        s_fds = [([c], dep) for det, dep in s_fds for c in det]
        t_fds = [([c], dep) for det, dep in t_fds for c in det]

    s_det = [tuple(sorted(s_index[c] for c in det if c in s_index)) for det, _ in s_fds]
    t_det = [tuple(sorted(t_index[c] for c in det if c in t_index)) for det, _ in t_fds]
    s_dep = [s_index.get(dep, -1) for _, dep in s_fds]
    t_dep = [t_index.get(dep, -1) for _, dep in t_fds]

    cache = {}
    P = np.zeros((len(s_fds), len(t_fds)))
    for p in range(len(s_fds)):
        if s_dep[p] < 0:
            continue
        for q in range(len(t_fds)):
            if t_dep[q] < 0:
                continue
            head = prior[s_dep[p], t_dep[q]]
            if head <= 0.0:
                continue
            if variant == "head only":
                P[p, q] = head
            else:
                P[p, q] = head * set_agreement(s_det[p], t_det[q], prior, cache)

    s_inc, t_inc = defaultdict(list), defaultdict(list)
    for k, (det, dep) in enumerate(s_fds):
        if dep in s_index:
            s_inc[s_index[dep]].append((k, "dep"))
        for c in det:
            if c in s_index:
                s_inc[s_index[c]].append((k, "det"))
    for k, (det, dep) in enumerate(t_fds):
        if dep in t_index:
            t_inc[t_index[dep]].append((k, "dep"))
        for c in det:
            if c in t_index:
                t_inc[t_index[c]].append((k, "det"))

    ignore_position = variant == "no position"
    for i, si in s_inc.items():
        for j, tj in t_inc.items():
            vals = []
            for k, pos in si:
                rivals = [P[k, m] for m, pos2 in tj
                          if ignore_position or pos2 == pos]
                vals.append(max(rivals) if rivals else 0.0)
            out[i, j] = mean(vals) if vals else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--dataset", default="ChEMBL-Unionable")
    ap.add_argument("--first-line", required=True)
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("-o", "--out", default="reports/fd_component_ablation.csv")
    args = ap.parse_args()

    root = Path(args.data_dir) / args.dataset
    base_of = {}
    for p in sorted(Path(args.first_line).rglob("*.csv")):
        if "MatchingStepLine1" in str(p):
            parts = p.relative_to(args.first_line).parts
            if len(parts) >= 2 and parts[0] == args.dataset:
                base_of.setdefault(parts[1], p)

    variants = ["full", "no position", "singletons", "head only"]
    rows = []
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
        s_meta = scenario / "metadata" / "source" / s_t[0].stem
        t_meta = scenario / "metadata" / "target" / t_t[0].stem
        s_fds = parse_fds(s_meta / "FD_results.txt")
        t_fds = parse_fds(t_meta / "FD_results.txt")
        if not s_fds or not t_fds:
            continue
        prior = normalise_base(base)
        truth = gt > 0.5

        arms = {"none": base,
                "role": scale_align(base, cosine_similarity(
                    fd_profile(s_fds, s_cols), fd_profile(t_fds, t_cols)),
                    args.weight)}
        for v in variants:
            arms[v] = scale_align(
                base, conditioned(s_fds, t_fds, s_cols, t_cols, prior, v),
                args.weight)
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
    print(f"{args.dataset}: {n} scenarios, weight={args.weight}\n")
    head = f"{'arm':<14}{'extractor':<13}{'F1':>7}{'R@GT':>8}{'MRR':>8}"
    print(head)
    print("-" * len(head))
    for arm in ["none", "role"] + variants:
        for ex in ("hungarian", "mutual_top1"):
            sel = [r for r in rows if r["arm"] == arm and r["extractor"] == ex]
            if sel:
                print(f"{arm:<14}{ex:<13}{mean(r['f1'] for r in sel):>7.3f}"
                      f"{mean(r['recall_at_gt'] for r in sel):>8.3f}"
                      f"{mean(r['mrr'] for r in sel):>8.3f}")

    for ex in ("hungarian", "mutual_top1"):
        print(f"\n{ex}: each variant against the role descriptor")
        by = defaultdict(dict)
        for r in rows:
            if r["extractor"] == ex:
                by[r["scenario"]][r["arm"]] = r["f1"]
        for v in variants:
            pr = [(x[v], x["role"]) for x in by.values() if v in x and "role" in x]
            W = sum(1 for a, b in pr if a > b)
            L = sum(1 for a, b in pr if a < b)
            p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
            print(f"  {v:<13}{mean(a - b for a, b in pr):>+8.3f}  {W}W {L}L "
                  f"{len(pr) - W - L}T  p={p:.4f}")
        print(f"  {'-' * 40}")
        for v in variants[1:]:
            pr = [(x[v], x["full"]) for x in by.values() if v in x and "full" in x]
            W = sum(1 for a, b in pr if a > b)
            L = sum(1 for a, b in pr if a < b)
            p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
            print(f"  {v:<13} vs full {mean(a - b for a, b in pr):>+7.3f}  "
                  f"{W}W {L}L {len(pr) - W - L}T  p={p:.4f}")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
