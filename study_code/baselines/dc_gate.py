"""The denial-constraint transferability gate: does the FD/order result hold for DCs?

FINDINGS 30 established transferability for functional dependencies on 335
unseen scenarios, and 17 established it for order. Both are equality- or
ordering-based. Denial constraints are the third semantics available here --
predicate sets over tuple pairs, carrying operators -- and they decide whether
transferability is a property of dependencies in general or stops at those two
families.

This is the gate, not another encoding study. One representation, the DC role
descriptor of FINDINGS 14, and the same arms used for FDs:

    none      no dependency evidence
    all       every discovered DC
    anchor    the frozen unsupervised pairing rule, applied to DCs
    oracle    DCs whose image under the ground-truth correspondence, operators
              preserved, is a DC of the other side
    random    a size-matched random subset

A DC's image must preserve operators as well as columns: `!(c = c)` says a
column is a key and `!(c != c)` says it is constant, and mapping one onto the
other would be counting opposite assertions as the same dependency.

The anchor rule is the DC analogue of `relevance.py`'s `anchor` variant -- the
pairing of two constraints is the best assignment between their predicates under
cross-instance value evidence, with operators required to agree, and no column
label or ground truth entering the ranking.

Whatever the outcome, the catalogue stops here. A negative result closes the
question of whether transferability is broad across constraint semantics, which
is worth as much as a positive one.

Usage:
    python dc_gate.py [--cases reports/valentine_full_382_20260903]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_paths import resolve  # noqa: E402

from anchor_generalization import anchor_matrix, bare, correspondence  # noqa: E402
from anchors import column_values  # noqa: E402
from od_role import cosine_similarity, dc_profile, scale_align  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402

MAX_PREDICATES = 4  # exact assignment below; LIMA returns at most two here


def load_dcs(scenario: Path, side: str):
    """LIMA constraints as tuples of (bare column, operator), plus the values."""
    values = {}
    for table in sorted((scenario / side).glob("*.csv")):
        for key, vals in column_values(table).items():
            values.setdefault(bare(key), set()).update(vals)
    dcs = []
    base = scenario / "metadata" / side
    for path in sorted(base.rglob("dc.scored.json")) if base.is_dir() else []:
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        labels = {c["id"]: bare(c["label"]) for c in artifact["spec"]["columns"]}
        for record in artifact.get("records", []):
            preds = tuple(sorted((labels[p["column"]], p["operator"])
                                 for p in record["predicates"]
                                 if p["column"] in labels))
            if preds:
                dcs.append(preds)
    return values, dcs


def degenerate(dc):
    """One predicate forbidding a column from ever differing: it is constant."""
    return len(dc) == 1 and dc[0][1] in ("!=", "<", ">")


def dc_images(dc, mapping):
    """All images of a DC under a possibly 1:n correspondence, operators kept."""
    combos = [[]]
    for col, op in dc:
        if col not in mapping:
            return []
        combos = [c + [(img, op)] for c in combos for img in mapping[col]]
    return [tuple(sorted(c)) for c in combos]


def oracle_subset(own, other, mapping):
    have = set(other)
    return [dc for dc in own if any(img in have for img in dc_images(dc, mapping))]


def pair_score(ds, dt, sim, order):
    """Best operator-respecting assignment between two predicate sets."""
    if not ds or not dt or len(ds) > MAX_PREDICATES or len(dt) > MAX_PREDICATES:
        return 0.0
    small, large = (ds, dt) if len(ds) <= len(dt) else (dt, ds)
    flip = len(ds) > len(dt)
    best = 0.0
    for combo in permutations(range(len(large)), len(small)):
        total = 0.0
        for (c1, op1), pick in zip(small, combo):
            c2, op2 = large[pick]
            if op1 != op2:
                total = -1.0
                break
            total += sim(c2, c1, order) if flip else sim(c1, c2, order)
        if total >= 0:
            best = max(best, total / len(small))
    return best * (len(small) / len(large))


def anchor_ranking(own, other, sim, order):
    return {dc: max((pair_score(dc, dt, sim, order) for dt in other), default=0.0)
            for dc in own}


def build_similarity(matrix, rows, cols):
    ri = {c: i for i, c in enumerate(rows)}
    ci = {c: j for j, c in enumerate(cols)}

    def sim(a, b, order):
        i = ri.get(a) if order == "st" else ci.get(a)
        j = ci.get(b) if order == "st" else ri.get(b)
        if i is None or j is None:
            return 0.0
        return float(matrix[i, j] if order == "st" else matrix[j, i])
    return sim


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="reports/valentine_full_382_20260903")
    ap.add_argument("--data", default="generated/benchmark-full")
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--drop-degenerate", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-o", "--out", default="reports/dc_gate.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cases_dir, data = Path(args.cases), Path(args.data)
    index = list(csv.DictReader(open(cases_dir / "cases.csv", encoding="utf-8")))
    if args.limit:
        index = index[:args.limit]

    rows, skipped, stats = [], defaultdict(int), defaultdict(list)
    shares_explored = []

    def process(rec, share_only):
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
        s_values, s_dcs = load_dcs(scenario, "source")
        t_values, t_dcs = load_dcs(scenario, "target")
        if args.drop_degenerate:
            s_dcs = [d for d in s_dcs if not degenerate(d)]
            t_dcs = [d for d in t_dcs if not degenerate(d)]
        if not s_dcs or not t_dcs:
            skipped["no DCs"] += 1
            return
        s_cols = [bare(c) for c in row_labels]
        t_cols = [bare(c) for c in col_labels]

        s_oracle = oracle_subset(s_dcs, t_dcs, correspondence(gt, s_cols, t_cols))
        t_oracle = oracle_subset(t_dcs, s_dcs, correspondence(gt.T, t_cols, s_cols))
        if share_only:
            shares_explored.append(len(s_oracle) / len(s_dcs))
            return

        stats["dcs"].append(len(s_dcs))
        stats["arity2"].append(sum(1 for d in s_dcs if len(d) > 1))
        stats["degenerate"].append(sum(1 for d in s_dcs if degenerate(d)) / len(s_dcs))
        stats["oracle_share"].append(len(s_oracle) / len(s_dcs))

        anchors = anchor_matrix(s_values, t_values, s_cols, t_cols)
        sim = build_similarity(anchors, s_cols, t_cols)
        r_s = anchor_ranking(s_dcs, t_dcs, sim, "st")
        r_t = anchor_ranking(t_dcs, s_dcs, sim, "ts")
        stats["anchor_nonzero"].append(
            sum(1 for v in r_s.values() if v > 0) / max(1, len(r_s)))

        def top(ranking, budget):
            return sorted(ranking, key=lambda d: -ranking[d])[:budget]

        fixed = mean(shares_explored) if shares_explored else 0.2
        arms = {
            "all": (s_dcs, t_dcs),
            "oracle": (s_oracle, t_oracle),
            "anchor": (top(r_s, len(s_oracle)), top(r_t, len(t_oracle))),
            "anchor-fixed": (top(r_s, max(1, round(fixed * len(s_dcs)))),
                             top(r_t, max(1, round(fixed * len(t_dcs))))),
            "random": (rng.sample(s_dcs, min(len(s_oracle), len(s_dcs))),
                       rng.sample(t_dcs, min(len(t_oracle), len(t_dcs)))),
        }

        truth = gt > 0.5
        evaluated = [("none", base)]
        for arm, (sd, td) in arms.items():
            # dc_profile expects (names, size, equalities) triples.
            def shape(dcs):
                return [([c for c, _ in d], len(d),
                         sum(1 for _, op in d if op == "=")) for d in dcs]
            role = cosine_similarity(dc_profile(shape(sd), s_cols),
                                     dc_profile(shape(td), t_cols))
            evaluated.append((arm, scale_align(base, role, args.weight)))
        for arm, m in evaluated:
            for res in evaluate(ScoreMatrix(scores=m, ground_truth=truth),
                                policies=["hungarian", "mutual_top1"]):
                rows.append({"id": rec["id"], "cohort": rec["cohort"],
                             "family": rec["family"],
                             "scenario_type": rec["scenario_type"], "arm": arm,
                             "extractor": res["extractor"], "f1": res["f1"],
                             "recall_at_gt": res.get("recall_at_gt_pessimistic",
                                                     float("nan")),
                             "mrr": res.get("mrr_pessimistic", float("nan"))})

    for rec in [r for r in index if r["cohort"] != "remaining338"]:
        process(rec, share_only=True)
    print(f"budget fraction from {len(shares_explored)} exploration cases: "
          f"{mean(shares_explored) if shares_explored else float('nan'):.3f}")
    for rec in index:
        process(rec, share_only=False)

    if not rows:
        print("nothing scored", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len({r["id"] for r in rows})
    print(f"scored {n} scenarios; skipped {dict(skipped)}")
    print(f"  DCs per source table {mean(stats['dcs']):.1f}, "
          f"{mean(stats['arity2']):.1f} of arity>1, "
          f"{mean(stats['degenerate']):.1%} constant-column")
    print(f"  oracle-transferable share {mean(stats['oracle_share']):.3f}; "
          f"anchor ranking nonzero for {mean(stats['anchor_nonzero']):.1%} of DCs\n")

    order = ["none", "all", "random", "anchor-fixed", "anchor", "oracle"]

    def table(pred, label):
        sel = [r for r in rows if pred(r)]
        if not sel:
            return
        print(f"--- {label}  ({len({r['id'] for r in sel})} scenarios)")
        print(f"  {'arm':<14}{'F1 full':>9}{'F1 mut':>9}{'R@GT':>8}{'MRR':>8}")
        for arm in order:
            h = [r for r in sel if r["arm"] == arm and r["extractor"] == "hungarian"]
            m = [r for r in sel if r["arm"] == arm and r["extractor"] == "mutual_top1"]
            if not h:
                continue
            print(f"  {arm:<14}{mean(float(r['f1']) for r in h):>9.3f}"
                  f"{mean(float(r['f1']) for r in m):>9.3f}"
                  f"{mean(float(r['recall_at_gt']) for r in h):>8.3f}"
                  f"{mean(float(r['mrr']) for r in h):>8.3f}")
        print()

    table(lambda r: True, "all scenarios")
    table(lambda r: r["cohort"] == "remaining338", "confirmation cohort only")
    for st in sorted({r["scenario_type"] for r in rows}):
        table(lambda r, s=st: r["scenario_type"] == s, st)

    print("paired, confirmation cohort")
    for ex in ("hungarian", "mutual_top1"):
        by = defaultdict(dict)
        for r in rows:
            if r["extractor"] == ex and r["cohort"] == "remaining338":
                by[r["id"]][r["arm"]] = r
        print(f"  {ex}")
        for a, b in (("oracle", "none"), ("oracle", "all"), ("oracle", "random"),
                     ("anchor", "all"), ("anchor", "random"), ("anchor", "none")):
            pr = [(v[a], v[b]) for v in by.values() if a in v and b in v]
            if not pr:
                continue
            W = sum(1 for x, y in pr if float(x["f1"]) > float(y["f1"]))
            L = sum(1 for x, y in pr if float(x["f1"]) < float(y["f1"]))
            p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
            print(f"    {a:<13} vs {b:<8}"
                  f"{mean(float(x['f1']) - float(y['f1']) for x, y in pr):>+8.3f}"
                  f"  {W}W {L}L {len(pr) - W - L}T  p={p:.4f}"
                  f"  dMRR{mean(float(x['mrr']) - float(y['mrr']) for x, y in pr):>+7.3f}")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
