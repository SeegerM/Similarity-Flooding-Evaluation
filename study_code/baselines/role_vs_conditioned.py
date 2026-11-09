"""Is the column-role descriptor the limit, or is the dependency class?

Every dependency result in this record reaches the matcher the same way: the
dependencies of a column are compressed into a per-column summary -- degree,
arity, participation quantiles -- and two columns are compared by the similarity
of those summaries. Sections 11, 13 and 16 all found that *which* relation feeds
that descriptor barely matters, which points at the descriptor rather than at
the relation.

FINDINGS 15 gave the headroom to aim at. On ChEMBL-Unionable an oracle keeping
only ground-truth-transferable FDs is worth +0.135 F1 over keeping all of them,
through this very descriptor. The oracle is not available to a matcher. The
question here is whether a *representation* that keeps what the summary throws
away can reach part of that gap without ground truth.

The alternative asks a different question about a candidate pair. Not "do these
two columns have similar dependency counts" but "do their dependencies connect
them to columns that plausibly correspond":

    Score(A_s, A_t) = mean over d_s containing A_s of
                        max over d_t containing A_t of P(d_s, d_t)

with P built from the first-line matcher's own scores, so no ground truth and no
labels beyond what the base matcher already used. This is the anchor-guided FD
idea moved from dependency *filtering* to candidate *scoring*.

The control the comparison turns on: both arms see the **same dependencies**. If
correspondence-conditioned scoring does not beat the role descriptor on an
identical FD set, the descriptor was not the limit and the loss is elsewhere.

Usage:
    python role_vs_conditioned.py <data-dir> --dataset ChEMBL-Unionable \
        --first-line <run>
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from od_role import cosine_similarity, columns_of, scale_align  # noqa: E402
from transferability_classes import (correspondence, fd_profile, parse_fds,  # noqa: E402
                                     transferable_fds)
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402


def normalise_base(base):
    """Base scores into [0, 1], so P is a product of comparable quantities."""
    finite = base[np.isfinite(base)]
    if finite.size == 0:
        return np.zeros_like(base)
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(base)
    out = (np.nan_to_num(base, nan=lo) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def set_agreement(xs, xt, prior, cache):
    """How well two determinant sets correspond under the prior.

    Best injective assignment between the two sets, averaged, then scaled by the
    size ratio so that a 1-column determinant does not match a 3-column one for
    free. Determinant arity is at most 3 here, so the assignment is enumerated
    exactly rather than approximated.
    """
    key = (xs, xt)
    hit = cache.get(key)
    if hit is not None:
        return hit
    if not xs or not xt:
        cache[key] = 0.0
        return 0.0
    a, b = list(xs), list(xt)
    if len(a) > len(b):
        a, b = b, a
        sub = prior.T
    else:
        sub = prior
    best = 0.0
    for combo in permutations(range(len(b)), len(a)):
        total = 0.0
        for u, v in zip(a, combo):
            total += sub[u, b[v]]
        best = max(best, total / len(a))
    value = best * (len(a) / len(b))
    cache[key] = value
    return value


def incidences(fds, index):
    """(dependency id, position) for every column occurrence.

    Position is kept because a column determining others and a column being
    determined are different structural roles, and matching one to the other
    would be comparing unlike things.
    """
    out = defaultdict(list)
    for k, (det, dep) in enumerate(fds):
        if dep in index:
            out[index[dep]].append((k, "dep"))
        for c in det:
            if c in index:
                out[index[c]].append((k, "det"))
    return out


def conditioned_scores(s_fds, t_fds, s_cols, t_cols, prior, aggregate="mean"):
    """Correspondence-conditioned candidate scores."""
    n_s, n_t = len(s_cols), len(t_cols)
    out = np.zeros((n_s, n_t))
    if not s_fds or not t_fds:
        return out
    s_index = {c: i for i, c in enumerate(s_cols)}
    t_index = {c: i for i, c in enumerate(t_cols)}

    s_det = [tuple(sorted(s_index[c] for c in det if c in s_index)) for det, _ in s_fds]
    t_det = [tuple(sorted(t_index[c] for c in det if c in t_index)) for det, _ in t_fds]
    s_dep = [s_index.get(dep, -1) for _, dep in s_fds]
    t_dep = [t_index.get(dep, -1) for _, dep in t_fds]

    # P(d_s, d_t) once for every dependency pair, then reused by every column
    # pair those dependencies touch.
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
            P[p, q] = head * set_agreement(s_det[p], t_det[q], prior, cache)

    s_inc, t_inc = incidences(s_fds, s_index), incidences(t_fds, t_index)
    for i in range(n_s):
        si = s_inc.get(i)
        if not si:
            continue
        for j in range(n_t):
            tj = t_inc.get(j)
            if not tj:
                continue
            best = []
            for k, pos in si:
                rivals = [P[k, m] for m, other in tj if other == pos]
                # A source incidence with no same-position counterpart scores
                # zero rather than being dropped, so a column whose structural
                # role has no analogue on the other side is penalised.
                best.append(max(rivals) if rivals else 0.0)
            out[i, j] = sum(best) if aggregate == "sum" else mean(best)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--dataset", default="ChEMBL-Unionable")
    ap.add_argument("--first-line", required=True)
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("-o", "--out", default="reports/role_vs_conditioned.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.data_dir) / args.dataset

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
        s_cols, t_cols = columns_of(s_t[0]), columns_of(t_t[0])
        if base.shape != gt.shape or len(s_cols) != gt.shape[0] \
                or len(t_cols) != gt.shape[1]:
            continue

        s_meta = scenario / "metadata" / "source" / s_t[0].stem
        t_meta = scenario / "metadata" / "target" / t_t[0].stem
        s_fds, t_fds = parse_fds(s_meta / "FD_results.txt"), parse_fds(t_meta / "FD_results.txt")
        if not s_fds or not t_fds:
            continue
        mapping = correspondence(gt, s_cols, t_cols)
        s_keep = transferable_fds(s_fds, t_fds, mapping)

        prior = normalise_base(base)
        shuffled = prior[rng.permutation(prior.shape[0]), :]
        uniform = np.ones_like(prior)

        arms = {
            "none": base,
            # Same dependencies, two representations: this pair is the control.
            "role (all FDs)": scale_align(
                base, cosine_similarity(fd_profile(s_fds, s_cols),
                                        fd_profile(t_fds, t_cols)), args.weight),
            "conditioned (all FDs)": scale_align(
                base, conditioned_scores(s_fds, t_fds, s_cols, t_cols, prior),
                args.weight),
            "conditioned sum (all FDs)": scale_align(
                base, conditioned_scores(s_fds, t_fds, s_cols, t_cols, prior,
                                         aggregate="sum"), args.weight),
            # Controls on where the conditioned score gets its information.
            "conditioned, shuffled prior": scale_align(
                base, conditioned_scores(s_fds, t_fds, s_cols, t_cols, shuffled),
                args.weight),
            "conditioned, no prior": scale_align(
                base, conditioned_scores(s_fds, t_fds, s_cols, t_cols, uniform),
                args.weight),
            # Oracle ceiling from FINDINGS 15, for the gap being aimed at.
            "role (GT-transferable)": scale_align(
                base, cosine_similarity(fd_profile(s_keep, s_cols),
                                        fd_profile(t_fds, t_cols)), args.weight),
            "conditioned (GT-transferable)": scale_align(
                base, conditioned_scores(s_keep, t_fds, s_cols, t_cols, prior),
                args.weight),
        }
        for arm, m in arms.items():
            for r in evaluate(ScoreMatrix(scores=m, ground_truth=gt > 0.5),
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

    order = list(dict.fromkeys(r["arm"] for r in rows))
    n = len({r["scenario"] for r in rows})
    print(f"{args.dataset}: {n} scenarios, weight={args.weight}\n")
    head = f"{'arm':<30}{'extractor':<13}{'F1':>7}{'R@GT':>8}{'MRR':>8}"
    print(head)
    print("-" * len(head))
    for arm in order:
        for ex in ("hungarian", "mutual_top1"):
            sel = [r for r in rows if r["arm"] == arm and r["extractor"] == ex]
            if sel:
                print(f"{arm:<30}{ex:<13}{mean(r['f1'] for r in sel):>7.3f}"
                      f"{mean(r['recall_at_gt'] for r in sel):>8.3f}"
                      f"{mean(r['mrr'] for r in sel):>8.3f}")

    comparisons = [
        ("conditioned (all FDs)", "role (all FDs)"),
        ("conditioned (all FDs)", "none"),
        ("conditioned (all FDs)", "conditioned, shuffled prior"),
        ("conditioned (all FDs)", "conditioned, no prior"),
        ("conditioned sum (all FDs)", "conditioned (all FDs)"),
        ("role (GT-transferable)", "role (all FDs)"),
        ("conditioned (GT-transferable)", "conditioned (all FDs)"),
        ("role (GT-transferable)", "conditioned (all FDs)"),
    ]
    for ex in ("hungarian", "mutual_top1"):
        print(f"\n{ex}")
        by = defaultdict(dict)
        for r in rows:
            if r["extractor"] == ex:
                by[r["scenario"]][r["arm"]] = r["f1"]
        for a, b in comparisons:
            pr = [(v[a], v[b]) for v in by.values() if a in v and b in v]
            if not pr:
                continue
            W = sum(1 for x, y in pr if x > y)
            L = sum(1 for x, y in pr if x < y)
            p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
            print(f"  {a} vs {b}: {mean(x - y for x, y in pr):+.3f}  "
                  f"{W}W {L}L {len(pr) - W - L}T  p={p:.4f}")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
