"""Inter-schema pIND anchors: correspondence evidence taken from values.

Every selection signal tried so far reads one schema at a time, or reaches the
other schema through column labels. Both failed, and the label route failed
*because* it is circular: judging which dependencies transfer needs the
correspondence that matching is trying to produce.

A partial inclusion dependency across the two instances does not have that
problem. Whether the values of a source column appear among the values of a
target column is observable directly, without knowing any correspondence and
without reading a single column name.

For a source column A and target column B, over their distinct non-null values:

    cov(A,B) = |V(A) & V(B)| / |V(A)|            directional inclusion
    both     = min(cov(A,B), cov(B,A))           bidirectional, i.e. equivalence
    b        = |V(B)| / |V(A) | V(B)|            overlap expected by chance
    gcov     = (cov - b) / (1 - b)               chance-corrected, as in the
                                                 multi-model heterogeneity work

Chance correction matters: 90% overlap between two boolean columns is
meaningless, the same overlap between two identifier columns is not.

Emits a score matrix in the same layout the harness already reads, so anchors
can be scored on their own, and combined with a matcher's scores.

Usage:
    python anchors.py <first-line-run-dir> <data-dir> <out-dir> [--variant gcov_both]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smx.adapters import read_matrix  # noqa: E402

NULLS = {"", "null", "none", "nan", "na", "n/a", "-"}


def column_values(table_path: Path):
    """Distinct non-null values per column, keyed by 'Table.column'."""
    name = table_path.stem
    with table_path.open(newline="", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return {}
        sets = [set() for _ in header]
        for row in reader:
            for i, value in enumerate(row[:len(header)]):
                v = value.strip()
                if v.lower() not in NULLS:
                    sets[i].add(v)
    return {f"{name}.{col.strip()}": sets[i] for i, col in enumerate(header)}


def side_values(data_dir: Path, dataset: str, scenario: str, side: str):
    out = {}
    side_dir = data_dir / dataset / scenario / side
    if side_dir.is_dir():
        for table in sorted(side_dir.glob("*.csv")):
            out.update(column_values(table))
    return out


def anchor_score(va: set, vb: set, variant: str, background: int = 0) -> float:
    """Inclusion evidence between two value sets.

    The chance baseline is the share of the *scenario's whole value pool* that
    the referenced column covers, not the share of the pair's own union. Using
    the pair's union is wrong in the most important case: two identical columns
    then have a baseline of 1 and score zero, exactly inverting the signal.
    """
    if not va or not vb:
        return 0.0
    shared = len(va & vb)
    if shared == 0:
        return 0.0
    cov_ab = shared / len(va)
    cov_ba = shared / len(vb)

    if variant == "cov":
        return cov_ab
    if variant == "both":
        return min(cov_ab, cov_ba)
    if variant == "both_card":
        # Overlapping on many distinct values is harder to achieve by accident
        # than overlapping on two, so damp small-cardinality agreement.
        return min(cov_ab, cov_ba) * (1.0 - 1.0 / (1.0 + shared))

    n = background if background > 0 else len(va | vb)
    b_ab = min(len(vb) / n, 1.0)
    b_ba = min(len(va) / n, 1.0)
    g_ab = (cov_ab - b_ab) / (1 - b_ab) if b_ab < 1 else 0.0
    g_ba = (cov_ba - b_ba) / (1 - b_ba) if b_ba < 1 else 0.0

    if variant == "gcov":
        return max(0.0, g_ab)
    if variant == "gcov_both":
        return max(0.0, min(g_ab, g_ba))
    raise ValueError(variant)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("first_line_dir")
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--dataset", default=None,
                    help="restrict to one dataset")
    ap.add_argument("--variant", default="gcov_both",
                    choices=["cov", "both", "both_card", "gcov", "gcov_both"])
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    sample = {}
    for path in sorted(Path(args.first_line_dir).rglob("*.csv")):
        if "MatchingStepLine1" not in str(path):
            continue
        parts = path.relative_to(args.first_line_dir).parts
        if len(parts) >= 2 and (args.dataset is None or parts[0] == args.dataset):
            sample.setdefault((parts[0], parts[1]), path)

    written = 0
    for (dataset, scenario), path in sorted(sample.items()):
        _, row_labels, col_labels = read_matrix(path)
        if not row_labels or not col_labels:
            continue

        source = side_values(data_dir, dataset, scenario, "source")
        target = side_values(data_dir, dataset, scenario, "target")

        # background pool for the chance baseline: every distinct value the
        # scenario contains, on either side
        background = len(set().union(*source.values(), *target.values())) if (source and target) else 0

        rows = []
        for a in row_labels:
            va = source.get(a, set())
            row = []
            for b in col_labels:
                vb = target.get(b, set())
                row.append(anchor_score(va, vb, args.variant, background))
            rows.append(row)

        out_path = out_dir / dataset / scenario / f"anchors_{args.variant}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            fh.write("Source\\Target," + ",".join(col_labels) + "\n")
            for label, row in zip(row_labels, rows):
                fh.write(label + "," + ",".join(f"{v:.6f}" for v in row) + "\n")
        written += 1

    print(f"wrote {written} anchor matrices (variant={args.variant}) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
