"""Continuous pairwise order relations, replacing exhaustive OD discovery.

FINDINGS 14 measured a hard width wall for discovered order dependencies: 13
columns profile in about a second, 14 take forty-eight, and 16 do not finish.
ODs range over column *permutations*, so the search is combinatorial in schema
width, and the 23-column benchmark schemas are out of reach.

The same problem arose for functional dependencies and was dissolved rather
than solved: dense gpdep gave a continuous relation per column pair, computable
directly, and 5aa showed it carried the same information as the discovered set.
This is the order-dependency analogue.

An order dependency A |-> B asserts that sorting by A also sorts by B. Its
continuous relaxation is the probability that it *nearly* does:

    mono(A -> B) = P( B_i <= B_j  |  A_i < A_j )

which is 1 exactly when the OD holds, degrades smoothly as it is violated, and
is directional -- mono(A -> B) and mono(B -> A) differ, as they must, since an
OD is not symmetric. Computing it needs one pass over sampled row pairs per
column pair, so cost is quadratic in width rather than combinatorial, and the
wall disappears by construction.

Orderability is decided per column and reported: a column parses as numeric or
it does not, and non-numeric columns fall back to lexicographic order, which is
meaningful for dates and identifiers and arbitrary for free text. The
`--numeric-only` switch restricts the relation to columns where the order is
unambiguous, at the cost of covering fewer columns.

Usage:
    python order_relation.py <data-dir> --dataset ChEMBL-Unionable
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

MAX_ROWS = 20000


def read_columns(path: Path, max_rows: int = MAX_ROWS):
    """Column values as lists of strings, in file order."""
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        cols = [[] for _ in header]
        for n, row in enumerate(reader):
            if n >= max_rows:
                break
            for i in range(len(header)):
                cols[i].append(row[i] if i < len(row) else "")
    return header, cols


def to_ordered(values, numeric_only: bool):
    """Numeric codes for a column, or None when it carries no usable order.

    Numeric parsing is attempted first. Otherwise the values are ranked
    lexicographically, which is the right order for dates and padded
    identifiers and an arbitrary one for free text; `numeric_only` excludes
    that case rather than pretending it is meaningful.
    """
    cleaned = [v.strip() for v in values]
    try:
        arr = np.array([float(v) if v not in ("", "null", "NULL") else np.nan
                        for v in cleaned], dtype=float)
        if np.isfinite(arr).sum() >= max(2, 0.5 * len(arr)):
            return arr, True
    except (ValueError, TypeError):
        pass
    if numeric_only:
        return None, False
    distinct = sorted({v for v in cleaned if v})
    if len(distinct) < 2:
        return None, False
    rank = {v: i for i, v in enumerate(distinct)}
    arr = np.array([rank.get(v, np.nan) if v else np.nan for v in cleaned],
                   dtype=float)
    return arr, False


def order_matrix(columns, pairs_i, pairs_j):
    """mono(a -> b) for every ordered column pair, over sampled row pairs."""
    n = len(columns)
    out = np.full((n, n), np.nan, dtype=float)
    valid = [c is not None for c in columns]
    sampled = [c[pairs_i] if v else None for c, v in zip(columns, valid)]
    sampled_j = [c[pairs_j] if v else None for c, v in zip(columns, valid)]
    for a in range(n):
        if not valid[a]:
            continue
        ai, aj = sampled[a], sampled_j[a]
        strictly_less = (ai < aj) & np.isfinite(ai) & np.isfinite(aj)
        support = int(strictly_less.sum())
        if support == 0:
            continue
        for b in range(n):
            if not valid[b] or a == b:
                continue
            bi, bj = sampled[b][strictly_less], sampled_j[b][strictly_less]
            ok = np.isfinite(bi) & np.isfinite(bj)
            if not ok.any():
                continue
            # Non-discordant: B does not decrease where A increases.
            out[a, b] = float((bi[ok] <= bj[ok]).mean())
    return out


def write_matrix(path: Path, matrix, labels, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Source\\Target"] + [f"{table}.{c}" for c in labels])
        for i, c in enumerate(labels):
            row = ["" if not np.isfinite(v) else f"{v:.6f}" for v in matrix[i]]
            w.writerow([f"{table}.{c}"] + row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="order_out")
    ap.add_argument("--samples", type=int, default=60000,
                    help="row pairs sampled per table; the estimate's only "
                         "source of noise, and fixed by --seed")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--numeric-only", action="store_true")
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    root = data_dir / args.dataset
    if not root.is_dir():
        print(f"no such dataset: {root}", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    written = 0
    numeric_share = []
    covered = []
    import time
    t0 = time.time()

    for scenario in sorted(p for p in root.iterdir() if p.is_dir()):
        for side in ("source", "target"):
            tables = sorted((scenario / side).glob("*.csv"))
            if not tables:
                continue
            table = tables[0]
            header, cols = read_columns(table, args.max_rows)
            if not header:
                continue
            n_rows = len(cols[0]) if cols else 0
            if n_rows < 2:
                continue
            prepared, is_num = [], []
            for values in cols:
                arr, numeric = to_ordered(values, args.numeric_only)
                prepared.append(arr)
                is_num.append(numeric)
            usable = sum(1 for c in prepared if c is not None)
            covered.append(usable / max(1, len(header)))
            numeric_share.append(sum(is_num) / max(1, len(header)))

            # One shared sample of row pairs per table, so every column pair is
            # estimated on the same rows and differences between them are not
            # sampling noise.
            k = min(args.samples, max(1000, n_rows * 20))
            pi = rng.integers(0, n_rows, size=k)
            pj = rng.integers(0, n_rows, size=k)
            keep = pi != pj
            pi, pj = pi[keep], pj[keep]

            m = order_matrix(prepared, pi, pj)
            out = (Path(args.out) / args.dataset / scenario.name
                   / f"order_{side}.csv")
            write_matrix(out, m, header, table.stem)
            written += 1

    print(f"{args.dataset}: wrote {written} order-relation matrices in "
          f"{time.time() - t0:.1f}s")
    if covered:
        print(f"  columns with usable order: {mean(covered):.1%}"
              f"   numeric: {mean(numeric_share):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
