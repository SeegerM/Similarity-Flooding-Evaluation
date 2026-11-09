"""Does dependency transferability generalise beyond functional dependencies?

Section 5b established the strongest dependency result in this record, for FDs
alone: local quality does not predict usefulness, but having a counterpart in
the other schema does. Keeping only transferable FDs was worth +0.061 F1 over
keeping all of them, and a size-matched random subset was *worse* than keeping
everything, so the gain is the selection rather than the smaller graph.

That was one dependency family. This asks whether it is a property of
dependencies or a property of FDs, by running the identical four-arm design for
functional, order and denial dependencies:

    no D   |   all D   |   GT-transferable D   |   random same-size D

All three classes reach the matcher through the *same* column-role descriptor
and the same fusion weight, so the comparison isolates the dependency class.

**FD is the positive control, and it is the point of including it.** If
FD-transferable clearly beats FD-all through this pipeline, the role descriptor
preserves transferability information and any OD/DC null is a statement about
those classes. If FD-transferable does *not* beat FD-all here -- while 5b says
it does through Similarity Flooding's graph -- then the descriptor is
destroying the very information the experiment is trying to measure, which
locates the problem in the representation rather than in the dependency class.

Ground truth is used only to define the oracle arm. Nothing else sees it.

Usage:
    python transferability_classes.py <data-dir> --dataset ChEMBL-Unionable_partial050
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from od_role import (  # noqa: E402
    DC_PRED_RE, cosine_similarity, columns_of, dc_profile, od_profile,
    parse_dcs, parse_ods, scale_align,
)
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402

FD_RE = re.compile(r"^\[(.*)\]\s*-->\s*(.+)$")
# Only the exact trailing "(pdep x, y)" annotation. A general trailing
# parenthetical would eat legitimate column names such as
# "Bilateral aid (international marker)" -- see the parser defect
# recorded against 29% of the benchmark in TODO item 41.
ANNOT_RE = re.compile(r"\s*\(pdep\s+[0-9.eEnN+-]+\s*,\s*[0-9.eEnN+-]+\s*\)\s*$")


def parse_fds(path: Path):
    """(determinant, dependant) pairs, bare column names."""
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = FD_RE.match(line.strip())
        if not m:
            continue
        # Schematch's FD files may carry a trailing "(pdep ...)" annotation on
        # the dependant. Left in place it becomes part of the column name, every
        # image lookup misses, and the transferable arm silently collapses onto
        # the base -- which is exactly how this was first mis-measured.
        dep = ANNOT_RE.sub("", m.group(2)).strip().split(".")[-1]
        det_raw = ANNOT_RE.sub("", m.group(1)).strip()
        det = [c.strip().split(".")[-1] for c in det_raw.split(",") if c.strip()]
        if dep:
            out.append((det, dep))
    return out


def fd_profile(fds, columns):
    """Same shape as the OD descriptor: outgoing, incoming, mean arity."""
    return od_profile([(det, [dep]) for det, dep in fds if det], columns)


def correspondence(gt: np.ndarray, s_cols, t_cols):
    """{source column -> target column} from the ground-truth matrix."""
    out = {}
    for i, s in enumerate(s_cols):
        js = np.flatnonzero(gt[i] > 0.5)
        if js.size:
            out[s] = t_cols[js[0]]
    return out


def transferable_fds(src, tgt, mapping):
    """Source FDs whose image under the correspondence is a target FD."""
    target = {(",".join(sorted(det)), dep) for det, dep in tgt}
    keep = []
    for det, dep in src:
        if dep not in mapping or not all(c in mapping for c in det):
            continue
        image = (",".join(sorted(mapping[c] for c in det)), mapping[dep])
        if image in target:
            keep.append((det, dep))
    return keep


def transferable_ods(src, tgt, mapping):
    """Order dependencies, image respecting operand order on both sides.

    Unlike an FD determinant, an OD's operand lists are ordered: swapping two
    left-hand columns is a different dependency. The image therefore preserves
    position rather than being compared as a set.
    """
    target = {(tuple(lhs), tuple(rhs)) for lhs, rhs in tgt}
    keep = []
    for lhs, rhs in src:
        if not all(c in mapping for c in lhs + rhs):
            continue
        image = (tuple(mapping[c] for c in lhs), tuple(mapping[c] for c in rhs))
        if image in target:
            keep.append((lhs, rhs))
    return keep


def dc_signature(names, ops):
    """Canonical form: the multiset of (column, operator) predicates."""
    return tuple(sorted(zip(names, ops)))


def parse_dcs_with_ops(path: Path):
    """As parse_dcs, but keeping the operators so images can respect them."""
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        preds = DC_PRED_RE.findall(line)
        if preds:
            out.append(([c for c, _ in preds], [o for _, o in preds]))
    return out


def transferable_dcs(src, tgt, mapping):
    """Denial constraints whose image preserves both columns and operators.

    A DC forbids a predicate combination, so the image must map every column
    *and* keep each predicate's operator; a constraint that becomes a different
    comparison is not the same constraint.
    """
    target = {dc_signature(names, ops) for names, ops in tgt}
    keep = []
    for names, ops in src:
        if not all(c in mapping for c in names):
            continue
        image = dc_signature([mapping[c] for c in names], ops)
        if image in target:
            keep.append((names, ops))
    return keep


def dc_to_profile_form(dcs):
    """(columns, size, equality count), the shape dc_profile expects."""
    return [(names, len(ops), sum(1 for o in ops if o == "==")) for names, ops in dcs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--dataset", default="ChEMBL-Unionable_partial050")
    ap.add_argument("--first-line", required=True)
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("-o", "--out", default="reports/transferability_classes.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    root = data_dir / args.dataset

    base_of = {}
    for p in sorted(Path(args.first_line).rglob("*.csv")):
        if "MatchingStepLine1" in str(p):
            parts = p.relative_to(args.first_line).parts
            if len(parts) >= 2 and parts[0] == args.dataset:
                base_of.setdefault(parts[1], p)

    rows = []
    shares = defaultdict(list)
    for scenario in sorted(p for p in root.iterdir() if p.is_dir()):
        name = scenario.name
        if name not in base_of:
            continue
        gt_files = sorted((scenario / "ground_truth").glob("*.csv"))
        src_t = sorted((scenario / "source").glob("*.csv"))
        tgt_t = sorted((scenario / "target").glob("*.csv"))
        if not gt_files or not src_t or not tgt_t:
            continue
        gt, _, _ = read_matrix(gt_files[0])
        base, _, _ = read_matrix(base_of[name])
        s_cols, t_cols = columns_of(src_t[0]), columns_of(tgt_t[0])
        if base.shape != gt.shape or len(s_cols) != gt.shape[0]:
            continue
        mapping = correspondence(gt, s_cols, t_cols)

        s_meta = scenario / "metadata" / "source" / src_t[0].stem
        t_meta = scenario / "metadata" / "target" / tgt_t[0].stem

        classes = {}
        s_fd, t_fd = parse_fds(s_meta / "FD_results.txt"), parse_fds(t_meta / "FD_results.txt")
        classes["FD"] = (s_fd, t_fd, transferable_fds(s_fd, t_fd, mapping),
                         fd_profile, lambda x: x)
        s_od, t_od = parse_ods(s_meta / "OD_results.txt"), parse_ods(t_meta / "OD_results.txt")
        classes["OD"] = (s_od, t_od, transferable_ods(s_od, t_od, mapping),
                         od_profile, lambda x: x)
        s_dc = parse_dcs_with_ops(s_meta / "DC_results.txt")
        t_dc = parse_dcs_with_ops(t_meta / "DC_results.txt")
        classes["DC"] = (s_dc, t_dc, transferable_dcs(s_dc, t_dc, mapping),
                         dc_profile, dc_to_profile_form)

        for cls, (s_all, t_all, s_keep, profile, adapt) in classes.items():
            if not s_all:
                continue
            shares[cls].append(len(s_keep) / len(s_all))
            # Random control drawn to the same size, so the arms differ in which
            # dependencies are kept and not in how many.
            s_rand = rng.sample(s_all, min(len(s_keep), len(s_all)))
            arms = {
                "all": (s_all, t_all),
                "transferable": (s_keep, t_all),
                "random": (s_rand, t_all),
            }
            for arm, (sub, tgt_set) in arms.items():
                role = cosine_similarity(profile(adapt(sub), s_cols),
                                         profile(adapt(tgt_set), t_cols))
                fused = scale_align(base, role, args.weight)
                for r in evaluate(ScoreMatrix(scores=fused, ground_truth=gt > 0.5),
                                  policies=["hungarian", "mutual_top1"]):
                    rows.append({"scenario": name, "cls": cls, "arm": arm,
                                 "extractor": r["extractor"], "f1": r["f1"],
                                 "recall_at_gt": r.get("recall_at_gt_pessimistic",
                                                       float("nan")),
                                 "mrr": r.get("mrr_pessimistic", float("nan"))})
        for r in evaluate(ScoreMatrix(scores=base, ground_truth=gt > 0.5),
                          policies=["hungarian", "mutual_top1"]):
            rows.append({"scenario": name, "cls": "-", "arm": "none",
                         "extractor": r["extractor"], "f1": r["f1"],
                         "recall_at_gt": r.get("recall_at_gt_pessimistic", float("nan")),
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

    print("Transferable share of source dependencies, by class\n")
    for cls in ("FD", "OD", "DC"):
        if shares[cls]:
            print(f"  {cls}: {mean(shares[cls]):.3f}")

    from scipy.stats import binomtest
    for ex in ("hungarian", "mutual_top1"):
        print(f"\n{ex}\n")
        head = (f"{'class':<6} {'arm':<14} {'F1':>7} {'R@GT':>7} {'MRR':>7}")
        print(head)
        print("-" * len(head))
        none = [r for r in rows if r["arm"] == "none" and r["extractor"] == ex]
        print(f"{'-':<6} {'no dependency':<14} {mean(r['f1'] for r in none):>7.3f} "
              f"{mean(r['recall_at_gt'] for r in none):>7.3f} "
              f"{mean(r['mrr'] for r in none):>7.3f}")
        for cls in ("FD", "OD", "DC"):
            for arm in ("all", "transferable", "random"):
                sel = [r for r in rows if r["cls"] == cls and r["arm"] == arm
                       and r["extractor"] == ex]
                if sel:
                    print(f"{cls:<6} {arm:<14} {mean(r['f1'] for r in sel):>7.3f} "
                          f"{mean(r['recall_at_gt'] for r in sel):>7.3f} "
                          f"{mean(r['mrr'] for r in sel):>7.3f}")

        print()
        for cls in ("FD", "OD", "DC"):
            by = defaultdict(dict)
            for r in rows:
                if r["cls"] == cls and r["extractor"] == ex:
                    by[r["scenario"]][r["arm"]] = r["f1"]
            for a, b in (("transferable", "all"), ("transferable", "random")):
                pairs = [(v[a], v[b]) for v in by.values() if a in v and b in v]
                if not pairs:
                    continue
                W = sum(1 for x, y in pairs if x > y)
                L = sum(1 for x, y in pairs if x < y)
                p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
                print(f"  {cls} {a} vs {b:<12} {mean(x - y for x, y in pairs):>+7.3f} "
                      f"{W:>3}W {L:>3}L {len(pairs) - W - L:>3}T  p={p:.3f}")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
