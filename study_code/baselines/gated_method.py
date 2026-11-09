"""A deployable precondition-gated matcher, fitted on exploration cases only.

FINDINGS 30, 31 and 36 leave the algorithmic story in a precise state. The
transferability oracle generalises decisively, the anchor estimator ranks better
than random everywhere, and it converts into a win over using no dependencies on
Joinable scenarios only -- because anchors discriminate when cross-instance
value inclusion identifies corresponding columns, which a join key supplies and
a horizontal split destroys. FINDINGS 34 then showed the anchor channels compose
with Kang-Naughton rather than duplicating it.

That suggests a method rather than another experiment:

    score = seed + KN + 1[anchors are discriminative] * (FD-anchor + DC-anchor)

with every free choice fixed on the exploration cohort and the confirmation
cohort scored exactly once. Two things are fitted, both on the 44 exploration
cases and never on the 335:

  budgets   how many dependencies to keep, as a fraction of the discovered set.
            The frozen anchor rule of `relevance.py` sizes its list to the
            *oracle's* cardinality, which is not available at deployment; this
            replaces that with a constant.
  gate      a single threshold on an observable contrast statistic of the
            anchor matrix. No ground truth enters the statistic. Ground truth
            enters only in choosing the threshold, on the exploration cases,
            which is what a development set is for.

The contrast statistic is the mean relative margin between a source column's
best and second-best anchor score. It is high when each column has one clear
counterpart and low when many columns share a vocabulary, which is exactly the
distinction between the regimes where the estimator works and where it does not.

Usage:
    python gated_method.py [--cases reports/valentine_full_382_20260903]
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
from compose_gate import fuse  # noqa: E402
from dc_gate import anchor_ranking, build_similarity, load_dcs, oracle_subset  # noqa: E402
from kang_naughton import kn_graph_scores, mi_matrix  # noqa: E402
from od_role import cosine_similarity, dc_profile  # noqa: E402
from relevance import build_similarity as fd_similarity  # noqa: E402
from relevance import relevance, signature  # noqa: E402
from transferability_classes import fd_profile  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402


def anchor_contrast(anchors: np.ndarray) -> float:
    """How sharply the anchor matrix singles out one counterpart per column.

    Mean over source columns of (best - second) / best, on columns that have any
    anchor evidence at all. Scale-free, uses no labels and no ground truth.
    """
    rows = []
    for i in range(anchors.shape[0]):
        row = np.sort(anchors[i])[::-1]
        if row.size < 2 or row[0] <= 1e-12:
            continue
        rows.append(float((row[0] - row[1]) / row[0]))
    return mean(rows) if rows else 0.0


def scenario_evidence(scenario: Path, seed_path: Path, gt_path: Path | None, seed: int,
                      use_ground_truth: bool = True):
    """Everything a scenario contributes, computed once.

    ``use_ground_truth=False`` is the deployment path: oracle diagnostics and
    truth are omitted so producing the matcher matrix cannot even read GT.
    """
    base, row_labels, col_labels = read_matrix(seed_path)
    gt = None
    if use_ground_truth:
        if gt_path is None:
            raise ValueError("Ground truth path required for oracle diagnostics")
        gt, _, _ = read_matrix(gt_path)
        if base.shape != gt.shape:
            return None
    src = sorted((scenario / "source").glob("*.csv"))
    tgt = sorted((scenario / "target").glob("*.csv"))
    if not src or not tgt:
        return None
    s_names, s_raw = read_columns(src[0])
    t_names, t_raw = read_columns(tgt[0])
    if len(s_names) != base.shape[0] or len(t_names) != base.shape[1]:
        return None
    s_cols = [bare(c) for c in row_labels]
    t_cols = [bare(c) for c in col_labels]

    s_values, s_fds = load_side(scenario, "source")
    t_values, t_fds = load_side(scenario, "target")
    _, s_dcs = load_dcs(scenario, "source")
    _, t_dcs = load_dcs(scenario, "target")
    if not s_fds or not t_fds:
        return None

    anchors = anchor_matrix(s_values, t_values, s_cols, t_cols)
    kn = kn_graph_scores(mi_matrix(s_raw), mi_matrix(t_raw))
    map_st = correspondence(gt, s_cols, t_cols) if gt is not None else {}
    map_ts = correspondence(gt.T, t_cols, s_cols) if gt is not None else {}

    fd_sim = fd_similarity(anchors, s_cols, t_cols)
    r_s = relevance(s_fds, t_fds, {}, {}, fd_sim, "st", "anchor")
    r_t = relevance(t_fds, s_fds, {}, {}, fd_sim, "ts", "anchor")
    s_orc = oracle_signatures(s_fds, t_fds, map_st) if gt is not None else set()
    t_orc = oracle_signatures(t_fds, s_fds, map_ts) if gt is not None else set()

    dc_pack = None
    if s_dcs and t_dcs:
        dc_sim = build_similarity(anchors, s_cols, t_cols)
        dc_pack = (anchor_ranking(s_dcs, t_dcs, dc_sim, "st"),
                   anchor_ranking(t_dcs, s_dcs, dc_sim, "ts"),
                   s_dcs, t_dcs,
                   oracle_subset(s_dcs, t_dcs, map_st) if gt is not None else [],
                   oracle_subset(t_dcs, s_dcs, map_ts) if gt is not None else [])

    return {"base": base, "truth": gt > 0.5 if gt is not None else None,
            "s_cols": s_cols, "t_cols": t_cols,
            "kn": kn, "contrast": anchor_contrast(anchors),
            "fd": (s_fds, t_fds, r_s, r_t, len(s_orc) / max(1, len(s_fds)),
                   len(t_orc) / max(1, len(t_fds))),
            "dc": dc_pack}


def fd_channel(ev, share):
    s_fds, t_fds, r_s, r_t, _, _ = ev["fd"]

    def pick(fds, rank):
        budget = max(1, round(share * len(fds)))
        keep = set(sorted(rank, key=lambda x: -rank[x])[:budget])
        return [(x, a) for x, a in fds if signature(list(x), a) in keep]
    return cosine_similarity(fd_profile(pick(s_fds, r_s), ev["s_cols"]),
                             fd_profile(pick(t_fds, r_t), ev["t_cols"]))


def dc_channel(ev, share):
    if ev["dc"] is None:
        return None
    r_s, r_t, s_dcs, t_dcs, _, _ = ev["dc"]

    def shape(dcs):
        return [([c for c, _ in d], len(d), sum(1 for _, op in d if op == "="))
                for d in dcs]

    def top(rank, dcs):
        budget = max(1, round(share * len(dcs)))
        return sorted(rank, key=lambda d: -rank[d])[:budget]
    return cosine_similarity(dc_profile(shape(top(r_s, s_dcs)), ev["s_cols"]),
                             dc_profile(shape(top(r_t, t_dcs)), ev["t_cols"]))


def score_f1(matrix, truth, extractor="hungarian"):
    for r in evaluate(ScoreMatrix(scores=matrix, ground_truth=truth),
                      policies=[extractor]):
        return r["f1"]
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="reports/valentine_full_382_20260903")
    ap.add_argument("--data", default="generated/benchmark-full")
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-o", "--out", default="reports/gated_method.csv")
    args = ap.parse_args()

    cases_dir, data = Path(args.cases), Path(args.data)
    index = list(csv.DictReader(open(cases_dir / "cases.csv", encoding="utf-8")))
    if args.limit:
        index = index[:args.limit]

    def load(rec):
        parts = rec["id"].split("/")
        # FINDINGS 67: the legacy rule drops the Magellan and Wikidata
        # families. Default is unchanged; SMX_RESOLVER=index corrects it.
        scenario = resolve(data, rec["id"])
        digest = hashlib.sha256(rec["id"].encode()).hexdigest()[:16]
        seed_path = cases_dir / "cases" / digest / "seed.csv"
        gt = sorted((scenario / "ground_truth").glob("*.csv"))
        if not scenario.is_dir() or not seed_path.is_file() or not gt:
            return None
        return scenario_evidence(scenario, seed_path, gt[0], args.seed)

    # ---------------------------------------------------------------- fitting
    explore = [r for r in index if r["cohort"] != "remaining338"]
    fd_shares, dc_shares, points = [], [], []
    for rec in explore:
        ev = load(rec)
        if ev is None:
            continue
        fd_shares.append(ev["fd"][4])
        if ev["dc"] is not None:
            s_dcs, s_orc = ev["dc"][2], ev["dc"][4]
            dc_shares.append(len(s_orc) / max(1, len(s_dcs)))
        points.append((rec, ev))

    fd_share = mean(fd_shares) if fd_shares else 0.2
    dc_share = mean(dc_shares) if dc_shares else 0.5
    print(f"fitted on {len(points)} exploration cases")
    print(f"  FD budget share {fd_share:.3f}   DC budget share {dc_share:.3f}")

    # Ground truth is consulted here, on the exploration cases only, to learn
    # whether the dependency channels helped. That is development-set use.
    fitted = []
    for rec, ev in points:
        kn_only = fuse(ev["base"], [ev["kn"]], args.weight)
        chans = [ev["kn"], fd_channel(ev, fd_share)]
        dc = dc_channel(ev, dc_share)
        if dc is not None:
            chans.append(dc)
        with_dep = fuse(ev["base"], chans, args.weight)
        gain = score_f1(with_dep, ev["truth"]) - score_f1(kn_only, ev["truth"])
        fitted.append((ev["contrast"], gain))

    best_tau, best_score = 0.0, -1e9
    grid = sorted({round(c, 3) for c, _ in fitted})
    for tau in grid:
        on = [g for c, g in fitted if c >= tau]
        off = [g for c, g in fitted if c < tau]
        if not on:
            continue
        # Total realised gain if the gate fires above tau and abstains below.
        realised = sum(on) + sum(0.0 for _ in off)
        if realised > best_score:
            best_score, best_tau = realised, tau
    fires = sum(1 for c, _ in fitted if c >= best_tau)
    print(f"  gate threshold tau = {best_tau:.3f} on anchor contrast "
          f"(fires on {fires}/{len(fitted)} exploration cases)")
    print("  frozen. the confirmation cohort is scored once below.\n")

    # ------------------------------------------------------------- evaluation
    rows, gate_stats = [], defaultdict(lambda: [0, 0])
    for rec in [r for r in index if r["cohort"] == "remaining338"]:
        ev = load(rec)
        if ev is None:
            continue
        gate = ev["contrast"] >= best_tau
        st = rec["scenario_type"]
        gate_stats[st][0] += int(gate)
        gate_stats[st][1] += 1

        kn_only = fuse(ev["base"], [ev["kn"]], args.weight)
        deps = [fd_channel(ev, fd_share)]
        dc = dc_channel(ev, dc_share)
        if dc is not None:
            deps.append(dc)
        always = fuse(ev["base"], [ev["kn"], *deps], args.weight)
        gated = always if gate else kn_only

        arms = {"seed": ev["base"], "kn only": kn_only,
                "kn + deps always": always, "GATED METHOD": gated}
        for arm, m in arms.items():
            for res in evaluate(ScoreMatrix(scores=m, ground_truth=ev["truth"]),
                                policies=["hungarian", "mutual_top1"]):
                rows.append({"id": rec["id"], "scenario_type": st,
                             "gate": int(gate), "contrast": f"{ev['contrast']:.4f}",
                             "arm": arm, "extractor": res["extractor"],
                             "f1": res["f1"],
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

    print("gate firing rate on the confirmation cohort")
    for st in sorted(gate_stats):
        on, tot = gate_stats[st]
        print(f"  {st:<24}{on:>4}/{tot:<4}  {on/max(1,tot):>5.0%}")

    order = ["seed", "kn only", "kn + deps always", "GATED METHOD"]
    print(f"\nconfirmation cohort ({len({r['id'] for r in rows})} scenarios)")
    print(f"  {'arm':<20}{'F1 full':>9}{'F1 mut':>9}{'R@GT':>8}{'MRR':>8}")
    for arm in order:
        h = [r for r in rows if r["arm"] == arm and r["extractor"] == "hungarian"]
        m = [r for r in rows if r["arm"] == arm and r["extractor"] == "mutual_top1"]
        if not h:
            continue
        print(f"  {arm:<20}{mean(float(r['f1']) for r in h):>9.3f}"
              f"{mean(float(r['f1']) for r in m):>9.3f}"
              f"{mean(float(r['recall_at_gt']) for r in h):>8.3f}"
              f"{mean(float(r['mrr']) for r in h):>8.3f}")

    for ex in ("hungarian", "mutual_top1"):
        print(f"\npaired, {ex}")
        for st in ["ALL"] + sorted(gate_stats):
            by = defaultdict(dict)
            for r in rows:
                if r["extractor"] == ex and (st == "ALL" or r["scenario_type"] == st):
                    by[r["id"]][r["arm"]] = r
            for a, b in (("GATED METHOD", "kn only"),
                         ("GATED METHOD", "kn + deps always"),
                         ("GATED METHOD", "seed"),
                         ("kn + deps always", "kn only")):
                pr = [(v[a], v[b]) for v in by.values() if a in v and b in v]
                if not pr:
                    continue
                W = sum(1 for x, y in pr if float(x["f1"]) > float(y["f1"]))
                L = sum(1 for x, y in pr if float(x["f1"]) < float(y["f1"]))
                p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
                print(f"  {st:<22}{a:<18} vs {b:<18}"
                      f"{mean(float(x['f1']) - float(y['f1']) for x, y in pr):>+8.3f}"
                      f"  {W}W {L}L  p={p:.4f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
