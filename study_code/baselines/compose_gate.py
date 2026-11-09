"""Do transferability selection and Kang-Naughton graph matching compose?

FINDINGS 33 left one question open that matters for the paper's fourth
contribution. The anchor-guided dependency route and the classical
mutual-information graph match both win on Joinable scenarios, and they win for
different reasons: the anchor route keeps dependencies whose image exists on the
other side, while Kang-Naughton compares the shape of the two dependency graphs
without reference to any correspondence. If those are complementary, relational
evidence is filling a gap the classical structural matcher leaves; if they are
substitutes, the anchor route is an expensive way to recover what MI already
carries.

The arms hold everything else fixed -- the frozen 65% Levenshtein plus 35% PMF
seed read from the cached per-case matrices, the same 0.35 fusion weight, the
same extractors.

The budget convention matters and follows the one already used for the
gcov/uncertainty combination: when two evidence matrices are present they are
independently normalised and **split the same 0.35 weight equally**, so a
combined arm cannot win merely by receiving twice as much evidence weight.

Usage:
    python compose_gate.py [--cases reports/valentine_full_382_20260903]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_paths import resolve  # noqa: E402

from anchor_generalization import (anchor_matrix, bare, correspondence,  # noqa: E402
                                   load_side, oracle_signatures)
from class_comparison_conditioned import read_columns  # noqa: E402
from dc_gate import anchor_ranking, build_similarity, load_dcs, oracle_subset  # noqa: E402
from kang_naughton import kn_graph_scores, mi_matrix  # noqa: E402
from od_role import cosine_similarity, dc_profile, scale_align  # noqa: E402
from relevance import build_similarity as fd_similarity  # noqa: E402
from relevance import relevance, signature  # noqa: E402
from transferability_classes import fd_profile  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402


def unit(m):
    """Independent min-max normalisation, so two channels are comparable."""
    finite = m[np.isfinite(m)]
    if finite.size == 0:
        return np.zeros_like(m)
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(m)
    return np.clip((np.nan_to_num(m, nan=lo) - lo) / (hi - lo), 0.0, 1.0)


def fuse(base, channels, weight):
    """One fusion at the fixed weight; several channels share it equally."""
    if not channels:
        return base
    blended = np.mean([unit(c) for c in channels], axis=0)
    return scale_align(base, blended, weight)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="reports/valentine_full_382_20260903")
    ap.add_argument("--data", default="generated/benchmark-full")
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-o", "--out", default="reports/compose_gate.csv")
    args = ap.parse_args()

    cases_dir, data = Path(args.cases), Path(args.data)
    index = list(csv.DictReader(open(cases_dir / "cases.csv", encoding="utf-8")))
    if args.limit:
        index = index[:args.limit]

    rows, skipped = [], defaultdict(int)
    for rec in index:
        parts = rec["id"].split("/")
        # FINDINGS 67: the legacy rule drops the Magellan and Wikidata
        # families. Default is unchanged; SMX_RESOLVER=index corrects it.
        scenario = resolve(data, rec["id"])
        digest = hashlib.sha256(rec["id"].encode()).hexdigest()[:16]
        seed_path = cases_dir / "cases" / digest / "seed.csv"
        gt_files = sorted((scenario / "ground_truth").glob("*.csv"))
        if not scenario.is_dir() or not seed_path.is_file() or not gt_files:
            skipped["missing artefacts"] += 1
            continue
        base, row_labels, col_labels = read_matrix(seed_path)
        gt, _, _ = read_matrix(gt_files[0])
        if base.shape != gt.shape:
            skipped["shape mismatch"] += 1
            continue
        src = sorted((scenario / "source").glob("*.csv"))
        tgt = sorted((scenario / "target").glob("*.csv"))
        if not src or not tgt:
            skipped["no tables"] += 1
            continue
        s_names, s_raw = read_columns(src[0])
        t_names, t_raw = read_columns(tgt[0])
        if len(s_names) != gt.shape[0] or len(t_names) != gt.shape[1]:
            skipped["width mismatch"] += 1
            continue
        s_cols = [bare(c) for c in row_labels]
        t_cols = [bare(c) for c in col_labels]

        kn = kn_graph_scores(mi_matrix(s_raw), mi_matrix(t_raw))

        s_values, s_fds = load_side(scenario, "source")
        t_values, t_fds = load_side(scenario, "target")
        _, s_dcs = load_dcs(scenario, "source")
        _, t_dcs = load_dcs(scenario, "target")
        if not s_fds or not t_fds:
            skipped["no FDs"] += 1
            continue

        anchors = anchor_matrix(s_values, t_values, s_cols, t_cols)
        map_st = correspondence(gt, s_cols, t_cols)
        map_ts = correspondence(gt.T, t_cols, s_cols)

        # FD arms
        s_orc = oracle_signatures(s_fds, t_fds, map_st)
        t_orc = oracle_signatures(t_fds, s_fds, map_ts)
        fd_sim = fd_similarity(anchors, s_cols, t_cols)
        r_s = relevance(s_fds, t_fds, {}, {}, fd_sim, "st", "anchor")
        r_t = relevance(t_fds, s_fds, {}, {}, fd_sim, "ts", "anchor")

        def pick(fds, rank, budget):
            keep = set(sorted(rank, key=lambda x: -rank[x])[:budget])
            return [(x, a) for x, a in fds if signature(list(x), a) in keep]

        def by_sig(fds, sigs):
            return [(x, a) for x, a in fds if signature(list(x), a) in sigs]

        fd_anchor = cosine_similarity(
            fd_profile(pick(s_fds, r_s, len(s_orc)), s_cols),
            fd_profile(pick(t_fds, r_t, len(t_orc)), t_cols))
        fd_oracle = cosine_similarity(fd_profile(by_sig(s_fds, s_orc), s_cols),
                                      fd_profile(by_sig(t_fds, t_orc), t_cols))

        # DC arms, when LIMA profiled this scenario
        dc_anchor = None
        if s_dcs and t_dcs:
            dc_sim = build_similarity(anchors, s_cols, t_cols)
            d_s = anchor_ranking(s_dcs, t_dcs, dc_sim, "st")
            d_t = anchor_ranking(t_dcs, s_dcs, dc_sim, "ts")
            s_dorc = oracle_subset(s_dcs, t_dcs, map_st)
            t_dorc = oracle_subset(t_dcs, s_dcs, map_ts)

            def shape(dcs):
                return [([c for c, _ in d], len(d),
                         sum(1 for _, op in d if op == "=")) for d in dcs]

            def dtop(rank, budget):
                return sorted(rank, key=lambda d: -rank[d])[:budget]
            dc_anchor = cosine_similarity(
                dc_profile(shape(dtop(d_s, len(s_dorc))), s_cols),
                dc_profile(shape(dtop(d_t, len(t_dorc))), t_cols))

        truth = gt > 0.5
        arms = {
            "seed": base,
            "kn": fuse(base, [kn], args.weight),
            "fd-anchor": fuse(base, [fd_anchor], args.weight),
            "kn + fd-anchor": fuse(base, [kn, fd_anchor], args.weight),
            "fd-oracle": fuse(base, [fd_oracle], args.weight),
            "kn + fd-oracle": fuse(base, [kn, fd_oracle], args.weight),
        }
        if dc_anchor is not None:
            arms["dc-anchor"] = fuse(base, [dc_anchor], args.weight)
            arms["kn + dc-anchor"] = fuse(base, [kn, dc_anchor], args.weight)
            arms["fd-anchor + dc-anchor"] = fuse(base, [fd_anchor, dc_anchor],
                                                 args.weight)

        for arm, m in arms.items():
            for res in evaluate(ScoreMatrix(scores=m, ground_truth=truth),
                                policies=["hungarian", "mutual_top1"]):
                rows.append({"id": rec["id"], "cohort": rec["cohort"],
                             "family": rec["family"],
                             "scenario_type": rec["scenario_type"], "arm": arm,
                             "extractor": res["extractor"], "f1": res["f1"],
                             "recall_at_gt": res.get("recall_at_gt_pessimistic",
                                                     float("nan")),
                             "mrr": res.get("mrr_pessimistic", float("nan"))})

    if not rows:
        print("nothing scored", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    order = ["seed", "kn", "fd-anchor", "dc-anchor", "kn + fd-anchor",
             "kn + dc-anchor", "fd-anchor + dc-anchor", "fd-oracle",
             "kn + fd-oracle"]
    print(f"scored {len({r['id'] for r in rows})} scenarios; "
          f"skipped {dict(skipped)}\n")

    def table(pred, label):
        sel = [r for r in rows if pred(r)]
        if not sel:
            return
        print(f"--- {label}  ({len({r['id'] for r in sel})} scenarios)")
        print(f"  {'arm':<24}{'F1 full':>9}{'F1 mut':>9}{'R@GT':>8}{'MRR':>8}")
        for arm in order:
            h = [r for r in sel if r["arm"] == arm and r["extractor"] == "hungarian"]
            m = [r for r in sel if r["arm"] == arm and r["extractor"] == "mutual_top1"]
            if not h:
                continue
            print(f"  {arm:<24}{mean(float(r['f1']) for r in h):>9.3f}"
                  f"{mean(float(r['f1']) for r in m):>9.3f}"
                  f"{mean(float(r['recall_at_gt']) for r in h):>8.3f}"
                  f"{mean(float(r['mrr']) for r in h):>8.3f}")
        print()

    table(lambda r: r["cohort"] == "remaining338", "confirmation cohort")
    for st in sorted({r["scenario_type"] for r in rows}):
        table(lambda r, s=st: r["scenario_type"] == s
              and r["cohort"] == "remaining338", st)

    pairs = [("kn + fd-anchor", "kn"), ("kn + fd-anchor", "fd-anchor"),
             ("kn + dc-anchor", "kn"), ("kn + dc-anchor", "dc-anchor"),
             ("kn + fd-oracle", "fd-oracle"), ("kn + fd-oracle", "kn"),
             ("fd-anchor + dc-anchor", "fd-anchor")]
    for ex in ("hungarian", "mutual_top1"):
        print(f"paired, confirmation cohort, {ex}")
        for st in ["ALL"] + sorted({r["scenario_type"] for r in rows}):
            by = defaultdict(dict)
            for r in rows:
                if r["extractor"] == ex and r["cohort"] == "remaining338" \
                        and (st == "ALL" or r["scenario_type"] == st):
                    by[r["id"]][r["arm"]] = r
            for a, b in pairs:
                pr = [(v[a], v[b]) for v in by.values() if a in v and b in v]
                if not pr:
                    continue
                W = sum(1 for x, y in pr if float(x["f1"]) > float(y["f1"]))
                L = sum(1 for x, y in pr if float(x["f1"]) < float(y["f1"]))
                p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
                print(f"  {st:<22}{a:<22} vs {b:<16}"
                      f"{mean(float(x['f1']) - float(y['f1']) for x, y in pr):>+8.3f}"
                      f"  {W}W {L}L  p={p:.4f}"
                      f"  dMRR{mean(float(x['mrr']) - float(y['mrr']) for x, y in pr):>+7.3f}")
            print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
