"""The continuous order relation as a column role, in the section 11 protocol.

Section 11 substituted four within-table relations -- gcov, containment,
Jaccard and the uncertainty coefficient -- into one fixed pipeline using an
identical outgoing/incoming role descriptor, and found them nearly
interchangeable. Every one of them is built from value equality.

This adds the continuous order relation from `order_relation.py` as a fifth
substitution, under the same descriptor and the same fusion weight. Unlike the
discovered-OD descriptor in FINDINGS 14, it is defined on non-constant columns
and measures how far one column's ordering follows another's, so it carries
ordering information rather than column degeneracy.

Usage:
    python order_role.py <data-dir> --dataset ChEMBL-Unionable --first-line <run>
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from od_role import cosine_similarity, columns_of, scale_align  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def read_order_matrix(path: Path):
    if not path.is_file():
        return None, []
    rows = list(csv.reader(path.open(encoding="utf-8")))
    if len(rows) < 2:
        return None, []
    labels = [c.split(".")[-1] for c in rows[0][1:]]
    m = np.full((len(rows) - 1, len(labels)), np.nan)
    for r, line in enumerate(rows[1:]):
        for c, v in enumerate(line[1:]):
            if v:
                m[r, c] = float(v)
    return m, labels


def role_profile(matrix, n_cols):
    """Outgoing and incoming quantiles of the relation, per column.

    Same shape as the descriptor section 11 used for the equality-based
    relations, so the substitution is controlled. A column with no usable
    order contributes an all-zero profile rather than being dropped, which
    keeps the matrix aligned with the schema.
    """
    prof = np.zeros((n_cols, 2 * len(QUANTILES) + 1))
    if matrix is None:
        return prof
    for i in range(min(n_cols, matrix.shape[0])):
        out = matrix[i][np.isfinite(matrix[i])]
        inc = matrix[:, i][np.isfinite(matrix[:, i])]
        if out.size:
            prof[i, :len(QUANTILES)] = np.quantile(out, QUANTILES)
        if inc.size:
            prof[i, len(QUANTILES):2 * len(QUANTILES)] = np.quantile(inc, QUANTILES)
        prof[i, -1] = 1.0 if out.size or inc.size else 0.0
    return prof


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--first-line", required=True)
    ap.add_argument("--order-dir", default="order_out")
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("-o", "--out", default="reports/order_role.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    root = data_dir / args.dataset

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
        gt_files = sorted((scenario / "ground_truth").glob("*.csv"))
        src, tgt = sorted((scenario / "source").glob("*.csv")), \
            sorted((scenario / "target").glob("*.csv"))
        if not gt_files or not src or not tgt:
            continue
        gt, _, _ = read_matrix(gt_files[0])
        base, _, _ = read_matrix(base_of[name])
        if base.shape != gt.shape:
            continue
        s_cols, t_cols = columns_of(src[0]), columns_of(tgt[0])
        if len(s_cols) != gt.shape[0] or len(t_cols) != gt.shape[1]:
            continue

        odir = Path(args.order_dir) / args.dataset / name
        s_m, _ = read_order_matrix(odir / "order_source.csv")
        t_m, _ = read_order_matrix(odir / "order_target.csv")
        if s_m is None or t_m is None:
            continue

        role = cosine_similarity(role_profile(s_m, len(s_cols)),
                                 role_profile(t_m, len(t_cols)))
        methods = {
            "base": base,
            "base + order role": scale_align(base, role, args.weight),
            "order role alone": role,
        }
        for method, m in methods.items():
            for r in evaluate(ScoreMatrix(scores=m, ground_truth=gt > 0.5),
                              policies=["hungarian", "mutual_top1"]):
                rows.append({"scenario": name, "method": method,
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
    print(f"{args.dataset}: {n} scenarios\n")
    head = f"{'method':<20} {'extractor':<13} {'F1':>7} {'R@GT':>7} {'MRR':>7}"
    print(head)
    print("-" * len(head))
    for method in ("base", "base + order role", "order role alone"):
        for ex in ("hungarian", "mutual_top1"):
            sel = [r for r in rows if r["method"] == method
                   and r["extractor"] == ex]
            if sel:
                print(f"{method:<20} {ex:<13} {mean(r['f1'] for r in sel):>7.3f} "
                      f"{mean(r['recall_at_gt'] for r in sel):>7.3f} "
                      f"{mean(r['mrr'] for r in sel):>7.3f}")

    from scipy.stats import binomtest
    print()
    for ex in ("hungarian", "mutual_top1"):
        by = defaultdict(dict)
        for r in rows:
            if r["extractor"] == ex:
                by[r["scenario"]][r["method"]] = r["f1"]
        pairs = [(v["base + order role"], v["base"]) for v in by.values()
                 if "base + order role" in v and "base" in v]
        if pairs:
            W = sum(1 for a, b in pairs if a > b)
            L = sum(1 for a, b in pairs if a < b)
            p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
            print(f"order role vs base ({ex}): {mean(a - b for a, b in pairs):+.3f} "
                  f"{W}W {L}L {len(pairs) - W - L}T  p={p:.4f}")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
