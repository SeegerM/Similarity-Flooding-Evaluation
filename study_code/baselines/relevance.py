"""Unsupervised cross-schema FD relevance.

The oracle experiment showed that selecting FDs which have a counterpart in the
other schema beats both using every FD and a size-matched random subset. That
selection used the ground truth. This computes the same idea without it.

For a source FD d_s: X_s -> A_s and a target FD d_t: X_t -> A_t,

    P(d_s, d_t) = Sim(X_s, X_t) * Sim(A_s, A_t) * g(q_s, q_t)
    r(d_s)      = max over d_t of P(d_s, d_t)

`Sim` comes from the first-line matcher output, which is available before any
dependency is chosen, so nothing here sees the ground truth. Determinants are
compared by the best assignment of their columns, and arity must agree.

Three variants, because they make different assumptions:

  label     Sim from the first-line similarity matrix, times gpdep proximity
  labelonly Sim only, ignoring dependency quality
  blind     no label similarity at all: arity agreement times gpdep proximity,
            which is the variant that survives opaque column names

Usage:
    python relevance.py <first-line-run-dir> <data-dir> <fd_scores.csv> [--variant label]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fd_pairing import scenario_fds  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402


def signature(determinant, dependent) -> str:
    return ",".join(sorted(determinant)) + "->" + dependent


def determinant_similarity(xs, xt, sim, order):
    """Best assignment between two determinants of equal arity.

    Arities must agree: an FD with a two-column determinant is not the same
    dependency as one with three, whatever the columns score.
    """
    if len(xs) != len(xt):
        return 0.0
    xs, xt = list(xs), list(xt)
    n = len(xs)

    # Enumerating permutations is exact but factorial. That was affordable for
    # exactly-discovered FDs, whose determinants average under two columns, but
    # partial discovery raises the mean to nearly four with a tail past seven,
    # where 7! per candidate pair makes the pairing intractable. Maximising a
    # product of similarities is a linear assignment once the similarities are
    # logged, so above a small arity the same optimum is found in polynomial
    # time. The floor keeps a zero similarity finite and heavily penalised, so
    # an all-positive assignment is preferred whenever one exists.
    if n <= 3:
        best = 0.0
        for perm in permutations(xt):
            product = 1.0
            for a, b in zip(xs, perm):
                product *= sim(a, b, order)
                if product <= best:
                    break
            best = max(best, product)
        return best

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    s = np.empty((n, n), dtype=float)
    for i, a in enumerate(xs):
        for j, b in enumerate(xt):
            s[i, j] = sim(a, b, order)
    with np.errstate(divide="ignore"):
        logs = np.where(s > 0.0, np.log(np.maximum(s, 1e-300)), -1e9)
    rows, cols = linear_sum_assignment(logs, maximize=True)
    product = 1.0
    for i, j in zip(rows, cols):
        product *= s[i, j]
    return product


def build_similarity(matrix, row_labels, col_labels):
    """Column-pair similarity lookup, symmetric in the two sides."""
    row_index = {label: i for i, label in enumerate(row_labels)}
    col_index = {label: j for j, label in enumerate(col_labels)}

    def sim(a, b, order):
        if order == "st":
            i, j = row_index.get(a), col_index.get(b)
        else:
            i, j = row_index.get(b), col_index.get(a)
        if i is None or j is None:
            return 0.0
        return float(matrix[i][j])

    return sim


def relevance(fds_own, fds_other, scores_own, scores_other, sim, order, variant):
    """r(d) for every FD of one side, as the best pairing with the other side."""
    out = {}
    other = [(list(x), a, scores_other.get(signature(x, a), 0.0)) for x, a in fds_other]

    for determinant, dependent in fds_own:
        own_sig = signature(determinant, dependent)
        q_own = scores_own.get(own_sig, 0.0)
        best = 0.0
        for xt, at, q_other in other:
            if len(determinant) != len(xt):
                continue
            if variant in ("anchor", "anchorq"):
                # Sim comes from the anchor matrix, so pairing is judged by
                # cross-instance value evidence rather than by label strings.
                # This is the point of the experiment: lexical pairing is
                # circular, because it can only pair dependencies over columns
                # the label matcher already matches.
                structural = determinant_similarity(determinant, xt, sim, order)
                if structural <= 0.0:
                    continue
                structural *= sim(dependent, at, order)
                pairing = structural
                if variant == "anchorq":
                    pairing *= 1.0 - abs(q_own - q_other)
            elif variant == "blind":
                # No labels at all: arity already agrees, so all that is left is
                # whether the two dependencies are of comparable strength.
                pairing = 1.0 - abs(q_own - q_other)
            else:
                structural = determinant_similarity(determinant, xt, sim, order)
                if structural <= 0.0:
                    continue
                structural *= sim(dependent, at, order)
                pairing = structural
                if variant == "label":
                    pairing *= 1.0 - abs(q_own - q_other)
            best = max(best, pairing)
            if best >= 1.0:
                break
        out[own_sig] = best
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("first_line_dir", help="run dir containing first-line sim matrices")
    ap.add_argument("data_dir")
    ap.add_argument("scores")
    ap.add_argument("--variant", default="label",
                    choices=["label", "labelonly", "blind", "anchor", "anchorq"])
    ap.add_argument("--dataset", default=None,
                    help="restrict to one dataset; without it every dataset in "
                         "the run dir is processed, using whatever FD set each "
                         "happens to hold, which is both wrong and slow")
    ap.add_argument("--anchor-dir", default="anchor_out",
                    help="anchor matrices, used as Sim for the anchor variants")
    ap.add_argument("--anchor-file", default="anchors_both.csv",
                    help="matrix file name inside <anchor-dir>/<dataset>/<scenario>")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_name = args.out_name or f"fd_relevant_{args.variant}.txt"

    # gpdep is only consulted by the variants that weight by dependency
    # strength ('label', 'blind', 'anchorq'); plain 'anchor' judges pairing by
    # cross-instance evidence alone, so a scores file without per-FD
    # signatures is still usable there rather than being a hard error.
    scores = defaultdict(dict)
    for r in csv.DictReader(open(args.scores, newline="", encoding="utf-8")):
        sig = r.get("signature")
        if sig is None:
            continue
        try:
            scores[(r["dataset"], r["scenario"], r["side"])][sig] = float(r["ngpdep"])
        except (ValueError, TypeError):
            pass
    if not scores and args.variant in ("label", "blind", "anchorq"):
        print(f"variant={args.variant} needs gpdep scores, but {args.scores} "
              f"carries none (no 'signature' column)", file=sys.stderr)
        return 1

    # one first-line matrix per scenario supplies Sim and the column order
    sample = {}
    for csv_path in sorted(Path(args.first_line_dir).rglob("*.csv")):
        if "MatchingStepLine1" not in str(csv_path) and "Line1" not in str(csv_path):
            continue
        parts = csv_path.relative_to(args.first_line_dir).parts
        if len(parts) >= 2:
            if args.dataset and parts[0] != args.dataset:
                continue
            sample.setdefault((parts[0], parts[1]), csv_path)

    if not sample:
        print("no first-line matrices found", file=sys.stderr)
        return 1

    written = 0
    skipped = []
    for (dataset, scenario), path in sorted(sample.items()):
        matrix, row_labels, col_labels = read_matrix(path)
        if not row_labels or not col_labels:
            continue
        if args.variant in ("anchor", "anchorq"):
            anchor_path = Path(args.anchor_dir) / dataset / scenario / args.anchor_file
            if not anchor_path.is_file():
                skipped.append(str(anchor_path))
                continue
            matrix, _, _ = read_matrix(anchor_path)
        sim = build_similarity(matrix, row_labels, col_labels)

        src = scenario_fds(data_dir, dataset, scenario, "source")
        tgt = scenario_fds(data_dir, dataset, scenario, "target")
        s_scores = scores.get((dataset, scenario, "source"), {})
        t_scores = scores.get((dataset, scenario, "target"), {})

        for side, own, other, own_scores, other_scores, order in (
            ("source", src, tgt, s_scores, t_scores, "st"),
            ("target", tgt, src, t_scores, s_scores, "ts"),
        ):
            r = relevance(own, other, own_scores, other_scores, sim, order, args.variant)

            # Size-matched to the oracle list, so ranking quality is compared
            # rather than how many dependencies each rule happens to keep.
            out_dir = data_dir / dataset / scenario / "metadata" / side
            oracle = out_dir / "fd_matched.txt"
            budget = (len([l for l in oracle.read_text(encoding="utf-8").splitlines() if l.strip()])
                      if oracle.is_file() else len(r))
            keep = sorted(r, key=lambda s: -r[s])[:budget]
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / out_name).write_text("\n".join(sorted(keep)) + "\n", encoding="utf-8")
            written += 1

    print(f"wrote {written} '{out_name}' allow-lists (variant={args.variant})")
    # Writing nothing used to look like success, and the caller then ran on
    # whatever lists were already on disk. An anchor variant with no anchor
    # matrices is a configuration error, not an empty result.
    if written == 0:
        print(f"no allow-lists written; {len(skipped)} scenarios had no anchor "
              f"matrix" + (f", e.g. {skipped[0]}" if skipped else ""),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
