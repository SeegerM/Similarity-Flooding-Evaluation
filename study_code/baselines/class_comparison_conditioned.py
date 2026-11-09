"""Rerun the dependency-class comparison under a representation that is fair to it.

Sections 11, 13 and 16 each substituted a different within-table relation into
the column-role descriptor and each concluded the relations are interchangeable.
FINDINGS 20 then showed that descriptor is not a neutral carrier: on identical
FDs, correspondence-conditioned scoring beats it by +0.054 with 19 wins and no
losses, and the summary is actually *worse than using no dependencies at all* on
Recall@GT. Every one of those comparisons was therefore run through a bottleneck
that discards what distinguishes the relations, which is precisely the thing
being compared.

This reruns the comparison with the bottleneck removed. Each relation is treated
as a weighted directed edge set -- the common form of everything here, FDs and
order included -- and scored two ways:

    role         outgoing/incoming quantiles per column, compared by cosine
                 (the descriptor of 11, 13 and 16, unchanged)

    conditioned  Score(A_s, A_t) = mean over edges (A_s -> x) of
                                     max over edges (A_t -> y) of
                                       prior[x, y] * (1 - |w_x - v_y|)

The prior is the first-line matcher's own normalised scores: no ground truth,
and nothing the base matcher did not already have. Both arms see the identical
relation, so any difference is representation.

The question this settles: does "dependency class does not matter" survive once
the representation stops flattening the classes together?

Usage:
    python class_comparison_conditioned.py <data-dir> --dataset ChEMBL-Unionable \
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

from od_role import cosine_similarity, columns_of, scale_align, parse_dcs  # noqa: E402
from order_relation import order_matrix, to_ordered  # noqa: E402
from order_role import role_profile  # noqa: E402
from role_vs_conditioned import normalise_base  # noqa: E402
from transferability_classes import parse_fds  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402
from smx.relational_roles import relation_matrices  # noqa: E402

EDGE_EPS = 1e-9


def read_columns(path, max_rows=5000):
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        r = csv.reader(fh)
        header = next(r, [])
        cols = [[] for _ in header]
        for n, row in enumerate(r):
            if n >= max_rows:
                break
            for i in range(len(header)):
                cols[i].append(row[i] if i < len(row) else "")
    return header, cols


def fd_edges(fds, columns):
    """FDs as a weighted edge set: determinant column -> dependant column.

    Flattening a determinant set into per-column edges loses the set, which the
    specialised FD scorer in `role_vs_conditioned.py` keeps. It is used here on
    purpose: the point of this comparison is that every class is expressed the
    same way, so no class is advantaged by a bespoke encoding.
    """
    index = {c: i for i, c in enumerate(columns)}
    n = len(columns)
    m = np.zeros((n, n))
    for det, dep in fds:
        if dep not in index:
            continue
        for c in det:
            if c in index:
                m[index[c], index[dep]] += 1.0
    peak = m.max()
    return m / peak if peak > 0 else m


def dc_edges(dcs, columns):
    """DCs as a weighted edge set: co-occurrence of two columns in a constraint.

    A denial constraint is a set of predicates over a tuple pair, so it induces
    an undirected co-occurrence edge rather than a directed one. It is stored
    symmetrically instead of being forced into a direction it does not have.
    """
    index = {c: i for i, c in enumerate(columns)}
    n = len(columns)
    m = np.zeros((n, n))
    for names, _preds, _equalities in dcs:
        present = sorted({index[c] for c in names if c in index})
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                m[present[a], present[b]] += 1.0
                m[present[b], present[a]] += 1.0
    peak = m.max()
    return m / peak if peak > 0 else m


def order_edges(cols, seed=20260903):
    prepared = [to_ordered(c, False)[0] for c in cols]
    n_rows = len(cols[0]) if cols else 0
    if n_rows < 2:
        return np.zeros((len(cols), len(cols)))
    rng = np.random.default_rng(seed)
    k = min(60000, max(1000, n_rows * 20))
    pi, pj = rng.integers(0, n_rows, k), rng.integers(0, n_rows, k)
    keep = pi != pj
    return np.nan_to_num(order_matrix(prepared, pi[keep], pj[keep]), nan=0.0)


def conditioned_relation_scores(rs, rt, prior):
    """Correspondence-conditioned scoring over a weighted directed edge set."""
    n_s, n_t = rs.shape[0], rt.shape[0]
    out = np.zeros((n_s, n_t))
    counted = np.zeros((n_s, n_t))
    for direction in ("out", "in"):
        a = rs if direction == "out" else rs.T
        b = rt if direction == "out" else rt.T
        s_edges = [np.flatnonzero(a[i] > EDGE_EPS) for i in range(n_s)]
        t_edges = [np.flatnonzero(b[j] > EDGE_EPS) for j in range(n_t)]
        for i in range(n_s):
            xs = s_edges[i]
            if xs.size == 0:
                continue
            ws = a[i, xs]
            for j in range(n_t):
                ys = t_edges[j]
                if ys.size == 0:
                    continue
                vs = b[j, ys]
                # prior[x, y] says the two endpoints correspond; the weight term
                # says the two edges are of comparable strength.
                m = prior[np.ix_(xs, ys)] * (1.0 - np.abs(ws[:, None] - vs[None, :]))
                out[i, j] += m.max(axis=1).mean()
                counted[i, j] += 1.0
    return np.divide(out, counted, out=np.zeros_like(out), where=counted > 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--dataset", default="ChEMBL-Unionable")
    ap.add_argument("--first-line", required=True)
    ap.add_argument("--relations", default="fd,gcov,containment,jaccard,uncertainty,order")
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("-o", "--out", default="reports/class_comparison_conditioned.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.data_dir) / args.dataset
    wanted = args.relations.split(",")

    base_of = {}
    for p in sorted(Path(args.first_line).rglob("*.csv")):
        if "MatchingStepLine1" in str(p):
            parts = p.relative_to(args.first_line).parts
            if len(parts) >= 2 and parts[0] == args.dataset:
                base_of.setdefault(parts[1], p)

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
        s_names, s_cols = read_columns(s_t[0])
        t_names, t_cols = read_columns(t_t[0])
        if base.shape != gt.shape or len(s_names) != gt.shape[0] \
                or len(t_names) != gt.shape[1]:
            continue
        truth = gt > 0.5
        prior = normalise_base(base)
        shuffled = prior[rng.permutation(prior.shape[0]), :]

        s_rel, _, _ = relation_matrices(s_cols)
        t_rel, _, _ = relation_matrices(t_cols)
        relations = {k: (s_rel[k], t_rel[k]) for k in
                     ("gcov", "containment", "jaccard", "uncertainty")}
        if "order" in wanted:
            relations["order"] = (order_edges(s_cols, args.seed),
                                  order_edges(t_cols, args.seed))
        s_meta = scenario / "metadata" / "source" / s_t[0].stem
        t_meta = scenario / "metadata" / "target" / t_t[0].stem
        if "fd" in wanted:
            s_fd, t_fd = parse_fds(s_meta / "FD_results.txt"), parse_fds(t_meta / "FD_results.txt")
            if s_fd and t_fd:
                relations["fd"] = (fd_edges(s_fd, s_names), fd_edges(t_fd, t_names))
        if "dc" in wanted:
            s_dc, t_dc = parse_dcs(s_meta / "DC_results.txt"), parse_dcs(t_meta / "DC_results.txt")
            if s_dc and t_dc:
                relations["dc"] = (dc_edges(s_dc, s_names), dc_edges(t_dc, t_names))

        for r in evaluate(ScoreMatrix(scores=base, ground_truth=truth),
                          policies=["hungarian", "mutual_top1"]):
            rows.append({"scenario": name, "relation": "-", "arm": "none",
                         "extractor": r["extractor"], "f1": r["f1"],
                         "recall_at_gt": r.get("recall_at_gt_pessimistic", float("nan")),
                         "mrr": r.get("mrr_pessimistic", float("nan"))})

        for rel in wanted:
            if rel not in relations:
                continue
            a, b = relations[rel]
            arms = {
                "role": cosine_similarity(role_profile(a, a.shape[0]),
                                          role_profile(b, b.shape[0])),
                "conditioned": conditioned_relation_scores(a, b, prior),
                "conditioned, shuffled prior": conditioned_relation_scores(a, b, shuffled),
            }
            for arm, m in arms.items():
                fused = scale_align(base, m, args.weight)
                for r in evaluate(ScoreMatrix(scores=fused, ground_truth=truth),
                                  policies=["hungarian", "mutual_top1"]):
                    rows.append({"scenario": name, "relation": rel, "arm": arm,
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

    present = [r for r in wanted if any(x["relation"] == r for x in rows)]
    n = len({r["scenario"] for r in rows})
    print(f"{args.dataset}: {n} scenarios, weight={args.weight}\n")

    for ex in ("hungarian", "mutual_top1"):
        base_f1 = mean(r["f1"] for r in rows if r["arm"] == "none"
                       and r["extractor"] == ex)
        base_rg = mean(r["recall_at_gt"] for r in rows if r["arm"] == "none"
                       and r["extractor"] == ex)
        print(f"{ex}: no dependency F1={base_f1:.3f} R@GT={base_rg:.3f}")
        head = (f"  {'relation':<13}{'role F1':>9}{'cond F1':>9}{'delta':>8}"
                f"{'role R@GT':>11}{'cond R@GT':>11}")
        print(head)
        print("  " + "-" * (len(head) - 2))
        for rel in present:
            def agg(arm, field):
                sel = [r[field] for r in rows if r["relation"] == rel
                       and r["arm"] == arm and r["extractor"] == ex]
                return mean(sel) if sel else float("nan")
            print(f"  {rel:<13}{agg('role', 'f1'):>9.3f}{agg('conditioned', 'f1'):>9.3f}"
                  f"{agg('conditioned', 'f1') - agg('role', 'f1'):>+8.3f}"
                  f"{agg('role', 'recall_at_gt'):>11.3f}"
                  f"{agg('conditioned', 'recall_at_gt'):>11.3f}")
        print()

    print("per-scenario comparisons (F1)")
    for ex in ("hungarian", "mutual_top1"):
        print(f"\n{ex}")
        by = defaultdict(dict)
        for r in rows:
            if r["extractor"] == ex:
                key = r["scenario"]
                by[key][(r["relation"], r["arm"])] = r["f1"]
        for rel in present:
            for a, b, label in ((("conditioned", rel), ("role", rel), "cond vs role"),
                                (("conditioned", rel), (None, "none"), "cond vs none"),
                                (("conditioned", rel),
                                 ("conditioned, shuffled prior", rel), "cond vs shuffled")):
                ka = (a[1], a[0])
                kb = ("-", "none") if b[0] is None else (b[1], b[0])
                pr = [(v[ka], v[kb]) for v in by.values() if ka in v and kb in v]
                if not pr:
                    continue
                W = sum(1 for x, y in pr if x > y)
                L = sum(1 for x, y in pr if x < y)
                p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
                print(f"  {rel:<13}{label:<18}{mean(x - y for x, y in pr):>+7.3f}"
                      f"  {W}W {L}L {len(pr) - W - L}T  p={p:.4f}")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
