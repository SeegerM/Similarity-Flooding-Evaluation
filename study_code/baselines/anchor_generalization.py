"""Does the frozen anchor-guided FD selection generalise beyond the small cohort?

The strongest constructive dependency result in this record -- anchor-guided
selection recovering most of the oracle's benefit without ground truth -- was
measured on a few dozen ChEMBL/DeNorm scenarios. The 382-case Valentine
expansion that followed did not carry the anchor-selected architecture at all,
so the algorithmic contribution rests on the smallest cohort in the project.
This closes that asymmetry.

Nothing about the method is re-tuned. The selector is `relevance.py`'s `anchor`
variant, imported rather than reimplemented: dependencies are paired by
cross-instance value evidence (chance-corrected bidirectional inclusion,
`anchors.py` `gcov_both`), determinant arity must agree, and no column label and
no ground truth enters the ranking. The base is the frozen 65% Levenshtein plus
35% PMF seed cached per case by the 382 run, so the seed is byte-identical to
the one already reported.

Five arms per scenario and side:

    none            no dependency evidence
    all             every discovered FD
    anchor          the frozen unsupervised ranking, oracle-sized budget
    anchor-fixed    the same ranking at a budget estimated only from the 44
                    exploration cases, so no ground truth enters at all
    oracle          FDs whose image under the ground-truth correspondence is an
                    FD of the other side -- the ceiling
    random          a size-matched random subset, the control that separates
                    *which* dependencies from *how many*

The budget question matters and is easy to miss: the frozen rule sizes the
anchor list to the oracle list, which makes it a fair test of *ranking* and not
a deployable method. `anchor-fixed` is reported beside it for that reason.

Results are broken down by scenario type -- Unionable, Joinable, View-Unionable,
Semantically-Joinable -- because the original result comes from unionable splits,
where FINDINGS 28 showed direct value evidence is privileged by construction.

Usage:
    python anchor_generalization.py [--cases reports/valentine_full_382_20260903]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_paths import resolve  # noqa: E402

from anchors import anchor_score, column_values  # noqa: E402
from od_role import cosine_similarity, scale_align  # noqa: E402
from relevance import build_similarity, relevance, signature  # noqa: E402
from transferability_classes import fd_profile  # noqa: E402
from fd_pairing import parse_fds  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402

HERE = Path(__file__).resolve().parent


def side_tables(scenario: Path, side: str):
    return sorted((scenario / side).glob("*.csv"))


def bare(name: str) -> str:
    """Column name without its table qualifier.

    `column_values` and `parse_fds` both key by `<table>.<column>`, and in the
    prepared layout the table stem is literally `source` or `target`, while the
    cached seed matrices carry bare column names. Left unreconciled, every
    correspondence lookup misses, the oracle arm comes back empty and all five
    arms collapse onto the baseline -- a silent null rather than an error, and
    the same failure mode as the FD annotation defect in FINDINGS 15.
    """
    return name.split(".")[-1]


def load_side(scenario: Path, side: str):
    """Column values and FDs for one side, keyed by bare column name."""
    values = {}
    for table in side_tables(scenario, side):
        for key, vals in column_values(table).items():
            values.setdefault(bare(key), set()).update(vals)
    fds = []
    base = scenario / "metadata" / side
    if base.is_dir():
        for f in sorted(base.rglob("FD_results.txt")):
            fds.extend((frozenset(bare(c) for c in det), bare(dep))
                       for det, dep in parse_fds(f))
    return values, fds


def anchor_matrix(s_values, t_values, s_cols, t_cols, variant="gcov_both"):
    """Cross-instance inclusion evidence; no labels, no ground truth."""
    pool = len(set().union(*s_values.values(), *t_values.values())) \
        if s_values and t_values else 0
    m = np.zeros((len(s_cols), len(t_cols)))
    for i, a in enumerate(s_cols):
        va = s_values.get(a, set())
        for j, b in enumerate(t_cols):
            m[i, j] = anchor_score(va, t_values.get(b, set()), variant, pool)
    return m


def correspondence(gt, s_cols, t_cols):
    mapping = defaultdict(list)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i, j] > 0.5:
                mapping[s_cols[i]].append(t_cols[j])
    return mapping


def oracle_signatures(own, other, mapping):
    """FDs of one side whose image under the correspondence exists on the other."""
    have = {signature(list(x), a) for x, a in other}
    keep = set()
    for determinant, dependent in own:
        if dependent not in mapping or any(c not in mapping for c in determinant):
            continue
        # A 1:n correspondence gives several images; any of them counts.
        combos = [[]]
        for c in determinant:
            combos = [p + [img] for p in combos for img in mapping[c]]
        for dep_img in mapping[dependent]:
            for combo in combos:
                if len(set(combo)) == len(combo) and \
                        signature(combo, dep_img) in have:
                    keep.add(signature(list(determinant), dependent))
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="reports/valentine_full_382_20260903")
    ap.add_argument("--data", default="generated/benchmark-full")
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-o", "--out", default="reports/anchor_generalization.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cases_dir = Path(args.cases)
    data = Path(args.data)
    index = list(csv.DictReader(open(cases_dir / "cases.csv", encoding="utf-8")))
    if args.limit:
        index = index[:args.limit]

    # The exploration cohort fixes the budget fraction for `anchor-fixed`, so
    # the confirmation cases never contribute to a choice made about them.
    shares_explored = []
    rows, skipped = [], defaultdict(int)

    def process(rec, collect_share_only):
        parts = rec["id"].split("/")
        # FINDINGS 67: the legacy rule drops the Magellan and Wikidata
        # families. Default is unchanged; SMX_RESOLVER=index corrects it.
        scenario = resolve(data, rec["id"])
        digest = hashlib.sha256(rec["id"].encode()).hexdigest()[:16]
        seed_path = cases_dir / "cases" / digest / "seed.csv"
        gt_files = sorted((scenario / "ground_truth").glob("*.csv"))
        if not scenario.is_dir() or not seed_path.is_file() or not gt_files:
            skipped["missing artefacts"] += 1
            return
        base, row_labels, col_labels = read_matrix(seed_path)
        gt, _, _ = read_matrix(gt_files[0])
        if base.shape != gt.shape:
            skipped["shape mismatch"] += 1
            return
        s_values, s_fds = load_side(scenario, "source")
        t_values, t_fds = load_side(scenario, "target")
        if not s_fds or not t_fds:
            skipped["no FDs"] += 1
            return
        s_cols = [c.split(".")[-1] for c in row_labels]
        t_cols = [c.split(".")[-1] for c in col_labels]

        mapping_st = correspondence(gt, s_cols, t_cols)
        mapping_ts = correspondence(gt.T, t_cols, s_cols)
        s_oracle = oracle_signatures(s_fds, t_fds, mapping_st)
        t_oracle = oracle_signatures(t_fds, s_fds, mapping_ts)
        if collect_share_only:
            if s_fds:
                shares_explored.append(len(s_oracle) / len(s_fds))
            return

        anchors = anchor_matrix(s_values, t_values, s_cols, t_cols)
        sim = build_similarity(anchors, s_cols, t_cols)
        # variant "anchor": pairing judged by cross-instance evidence alone, so
        # the empty gpdep tables below are the documented no-op for it.
        r_s = relevance(s_fds, t_fds, {}, {}, sim, "st", "anchor")
        r_t = relevance(t_fds, s_fds, {}, {}, sim, "ts", "anchor")

        def pick(fds, ranking, budget):
            order = sorted(ranking, key=lambda s: -ranking[s])[:budget]
            keep = set(order)
            return [(x, a) for x, a in fds if signature(list(x), a) in keep]

        def by_sig(fds, sigs):
            return [(x, a) for x, a in fds if signature(list(x), a) in sigs]

        fixed_share = mean(shares_explored) if shares_explored else 0.2
        arms = {
            "all": (s_fds, t_fds),
            "oracle": (by_sig(s_fds, s_oracle), by_sig(t_fds, t_oracle)),
            "anchor": (pick(s_fds, r_s, len(s_oracle)),
                       pick(t_fds, r_t, len(t_oracle))),
            "anchor-fixed": (
                pick(s_fds, r_s, max(1, round(fixed_share * len(s_fds)))),
                pick(t_fds, r_t, max(1, round(fixed_share * len(t_fds))))),
            "random": (rng.sample(s_fds, min(len(s_oracle), len(s_fds))),
                       rng.sample(t_fds, min(len(t_oracle), len(t_fds)))),
        }

        truth = gt > 0.5
        out = [("none", base)]
        for arm, (sf, tf) in arms.items():
            role = cosine_similarity(fd_profile(sf, s_cols), fd_profile(tf, t_cols))
            out.append((arm, scale_align(base, role, args.weight)))
        for arm, m in out:
            for res in evaluate(ScoreMatrix(scores=m, ground_truth=truth),
                                policies=["hungarian", "mutual_top1"]):
                rows.append({"id": rec["id"], "cohort": rec["cohort"],
                             "family": rec["family"],
                             "scenario_type": rec["scenario_type"], "arm": arm,
                             "extractor": res["extractor"], "f1": res["f1"],
                             "recall_at_gt": res.get("recall_at_gt_pessimistic",
                                                     float("nan")),
                             "mrr": res.get("mrr_pessimistic", float("nan")),
                             "n_fds": len(s_fds),
                             "oracle_share": len(s_oracle) / max(1, len(s_fds))})

    explored = [r for r in index if r["cohort"] != "remaining338"]
    for rec in explored:
        process(rec, collect_share_only=True)
    print(f"budget fraction from {len(shares_explored)} exploration cases: "
          f"{mean(shares_explored) if shares_explored else float('nan'):.3f}")

    for rec in index:
        process(rec, collect_share_only=False)

    if not rows:
        print("nothing scored", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len({r["id"] for r in rows})
    print(f"scored {n} scenarios; skipped {dict(skipped)}\n")

    order = ["none", "all", "random", "anchor-fixed", "anchor", "oracle"]

    def table(subset, label):
        sel = [r for r in rows if subset(r)]
        if not sel:
            return
        k = len({r["id"] for r in sel})
        print(f"--- {label}  ({k} scenarios)")
        head = (f"  {'arm':<14}{'F1 full':>9}{'F1 mut':>9}{'R@GT':>8}{'MRR':>8}"
                f"{'gap closed':>12}")
        print(head)
        base_rg = mean(float(r["recall_at_gt"]) for r in sel
                       if r["arm"] == "none" and r["extractor"] == "hungarian")
        orc_rg = mean(float(r["recall_at_gt"]) for r in sel
                      if r["arm"] == "oracle" and r["extractor"] == "hungarian")
        for arm in order:
            h = [r for r in sel if r["arm"] == arm and r["extractor"] == "hungarian"]
            m = [r for r in sel if r["arm"] == arm and r["extractor"] == "mutual_top1"]
            if not h:
                continue
            rg = mean(float(r["recall_at_gt"]) for r in h)
            span = orc_rg - base_rg
            closed = (rg - base_rg) / span if abs(span) > 1e-9 else float("nan")
            print(f"  {arm:<14}{mean(float(r['f1']) for r in h):>9.3f}"
                  f"{mean(float(r['f1']) for r in m):>9.3f}{rg:>8.3f}"
                  f"{mean(float(r['mrr']) for r in h):>8.3f}"
                  f"{closed:>11.0%}")
        print()

    table(lambda r: True, "all scenarios")
    table(lambda r: r["cohort"] == "remaining338", "confirmation cohort only")
    for st in sorted({r["scenario_type"] for r in rows}):
        table(lambda r, s=st: r["scenario_type"] == s, st)

    print("paired comparisons, confirmation cohort, hungarian F1")
    by = defaultdict(dict)
    for r in rows:
        if r["extractor"] == "hungarian" and r["cohort"] == "remaining338":
            by[r["id"]][r["arm"]] = float(r["f1"])
    for a, b in (("anchor", "all"), ("anchor", "random"), ("anchor", "none"),
                 ("anchor-fixed", "all"), ("oracle", "all"), ("anchor", "oracle")):
        pr = [(v[a], v[b]) for v in by.values() if a in v and b in v]
        if not pr:
            continue
        W = sum(1 for x, y in pr if x > y)
        L = sum(1 for x, y in pr if x < y)
        p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
        print(f"  {a:<14} vs {b:<8}{mean(x - y for x, y in pr):>+8.3f}"
              f"  {W}W {L}L {len(pr) - W - L}T  p={p:.4f}")

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
