"""Re-score the tuning screen on every metric, with the tie bias of Hungarian F1 removed.

The pre-registered objective J = (dRecall@GT_pess + dF1_H) / 2 mixes two tie-sensitive metrics.
Hungarian F1 is biased by index order: scipy's linear_sum_assignment breaks ties by row and
column index, so on Valentine's near-diagonal ground truth a tied matrix lands on the diagonal
and scores well. This script scores every (initial matcher, configuration, scenario) of a stage
on:

    F1_H        Hungarian, ties broken by index (as in score_tuning.py)
    F1_Hr       Hungarian, ties broken at random: the mean over K random row and column
                permutations (the same permutations for the initial matrix and SF's output, so
                the difference stays paired; seeded by the scenario id, so it is deterministic)
    F1_M        mutual top-1, ties included (tie inflation costs precision; no order bias)
    R_pess      Recall@GT, ties broken against the ground truth (as in score_tuning.py)
    R_exp       Recall@GT, expected value under random tie-breaking (exact)
    MRR_pess    MRR, ties broken against the ground truth
    MRR_exp     MRR, expected value under random tie-breaking (exact)

and reports, per metric, what the screen would conclude: the best configurations over the panel,
the factor main effects, alpha* per initial matcher, and the keep set that the pre-registered
pruning rule (top 10 per fold and initial matcher on the training families) would produce,
compared with the keep set of J. It does not touch score_tuning.py's outputs or keep.json.

    ../../harness/.venv-bdi/Scripts/python.exe score_screen_metrics.py --stage screen --regime plain
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(HERE))

from scipy.optimize import linear_sum_assignment  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.core import DEFAULT_EPS, extract, match_metrics, recall_at_ground_truth  # noqa: E402

import space  # noqa: E402

PROTOCOL = ROOT / "harness" / "reports" / "audited_md5_fused_20260904" / "protocol.json"
K_PERM = 50
TOP_K = 10
METRICS = ["F1_H", "F1_Hr", "F1_M", "R_pess", "R_exp", "MRR_pess", "MRR_exp"]
# Objectives compared: the pre-registered J, the same J with the tie-free F1, and each metric alone.
OBJECTIVES = {"J (pre-registered)": ("R_pess", "F1_H"), "J with F1_Hr": ("R_pess", "F1_Hr"),
              "J tie-free (R_exp, F1_Hr)": ("R_exp", "F1_Hr"),
              **{m: (m,) for m in METRICS}}


def one(folder: Path) -> Path | None:
    files = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    return files[0] if len(files) == 1 else None


def f1(M: np.ndarray, gt: np.ndarray) -> float:
    return float(match_metrics(M, gt)["f1"])


def hungarian_random(S: np.ndarray, gt: np.ndarray, perms: list[tuple[np.ndarray, np.ndarray]],
                     eps: float = DEFAULT_EPS) -> float:
    """Mean Hungarian F1 over random row/column orders: ties no longer follow the index."""
    vals = []
    for pr, pc in perms:
        rows, cols = linear_sum_assignment(S[np.ix_(pr, pc)], maximize=True)
        M = np.zeros(S.shape, dtype=bool)
        for r, c in zip(rows, cols):
            if S[pr[r], pc[c]] > eps:   # same drop_nonpositive rule as core.hungarian
                M[pr[r], pc[c]] = True
        vals.append(f1(M, gt))
    return float(np.mean(vals))


def recall_expected(S: np.ndarray, gt: np.ndarray, eps: float = DEFAULT_EPS) -> float:
    """Recall@GT with ties at the cut-off filled at random: expected hits are hypergeometric."""
    flat, truth = S.ravel(), gt.ravel()
    k = min(int(truth.sum()), flat.size)
    if k == 0:
        return float("nan")
    cutoff = np.partition(flat, flat.size - k)[flat.size - k]
    above = flat > cutoff + eps
    tied = np.abs(flat - cutoff) <= eps
    slots = k - int(above.sum())
    hits = int((above & truth).sum())
    if slots <= 0 or not tied.any():
        return hits / k
    return (hits + slots * int((tied & truth).sum()) / int(tied.sum())) / k


def mrr_both(S: np.ndarray, gt: np.ndarray, eps: float = DEFAULT_EPS) -> tuple[float, float]:
    """MRR with ties against the ground truth, and its exact expectation under random ties."""
    pess, exp = [], []
    for i in range(S.shape[0]):
        pos = np.flatnonzero(gt[i])
        if pos.size == 0:
            continue
        best = S[i, pos].max()
        neg = ~gt[i]
        n_above = int((neg & (S[i] > best + eps)).sum())
        t_neg = int((neg & (np.abs(S[i] - best) <= eps)).sum())
        t_pos = int((gt[i] & (np.abs(S[i] - best) <= eps)).sum())
        pess.append(1.0 / (1 + n_above + t_neg))
        # position j of the first relevant item in a random order of the tied group
        t, e, p_none_yet = t_pos + t_neg, 0.0, 1.0
        for j in range(1, t_neg + 2):
            p_here = p_none_yet * t_pos / (t - (j - 1))
            e += p_here / (n_above + j)
            p_none_yet *= (t_neg - (j - 1)) / (t - (j - 1)) if j <= t_neg else 0.0
        exp.append(e)
    return (float(np.mean(pess)), float(np.mean(exp))) if pess else (float("nan"), float("nan"))


def all_metrics(S: np.ndarray, gt: np.ndarray, perms) -> dict[str, float]:
    mp, me = mrr_both(S, gt)
    return {"F1_H": f1(extract(S, "hungarian"), gt),
            "F1_Hr": hungarian_random(S, gt, perms),
            "F1_M": f1(extract(S, "mutual_top1"), gt),
            "R_pess": float(recall_at_ground_truth(S, gt)["recall_at_gt_pessimistic"]),
            "R_exp": recall_expected(S, gt),
            "MRR_pess": mp, "MRR_exp": me}


def permutations(shape: tuple[int, int], key: str) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    return [(rng.permutation(shape[0]), rng.permutation(shape[1])) for _ in range(K_PERM)]


def score_run(job: tuple[str, str, str, str]) -> list[dict]:
    run_root, cfg_dir, corpus, family_json = Path(job[0]), Path(job[1]), Path(job[2]), job[3]
    family = {tuple(k.split("|")): v for k, v in json.loads(family_json).items()}
    meta = json.loads((cfg_dir / "meta.json").read_text(encoding="utf-8"))
    run = sorted(p for p in run_root.iterdir() if p.is_dir())[-1]
    out = []
    for scen in sorted(run.glob("*/*")):
        ds, sc = scen.parent.name, scen.name
        gt_file = one(corpus / ds / sc / "ground_truth")
        sf_file = one(scen / "_outputs" / "MatchingStepLine2")
        if gt_file is None or sf_file is None:
            continue
        gt = read_matrix(gt_file)[0] > 0.5
        q = read_matrix(Path(meta["seed_dir"]) / ds / sc / "seed.csv")[0]
        S = read_matrix(sf_file)[0]
        if q.shape != gt.shape or S.shape != gt.shape or not np.isfinite(S).all():
            continue
        perms = permutations(gt.shape, f"{ds}/{sc}")
        mq, ms = all_metrics(q, gt, perms), all_metrics(S, gt, perms)
        out.append({"seed": meta["seed"], "config": meta["config"], "operator": meta["operator"],
                    "value": meta["value"], "coeff": meta["coeff"], "attach": meta["attach"],
                    "fd": meta["fd"], "dataset": ds, "scenario": sc, "family": family[(ds, sc)],
                    **{f"seed_{m}": mq[m] for m in METRICS}, **{f"sf_{m}": ms[m] for m in METRICS},
                    **{f"d_{m}": ms[m] - mq[m] for m in METRICS}})
    return out


def macro(rows: list[dict], fams: set[str], key: str) -> float:
    per = defaultdict(list)
    for r in rows:
        if r["family"] in fams and not math.isnan(r[key]):
            per[r["family"]].append(r[key])
    vals = [mean(v) for v in per.values() if v]
    return mean(vals) if vals else float("nan")


def objective(rows: list[dict], fams: set[str], parts: tuple[str, ...]) -> float:
    return mean(macro(rows, fams, f"d_{m}") for m in parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("screen", "dev"), default="screen")
    ap.add_argument("--regime", choices=("plain", "md5"), default="plain")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    suffix = f"{args.stage}_{args.regime}"
    family = {f"{c['dataset']}|{c['scenario']}": c["family"]
              for c in json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]}
    family_json = json.dumps(family)
    corpus = HERE / "corpus" / args.regime / args.stage
    runs_dir = HERE / "results" / suffix
    out_dir = HERE / "results" / f"{suffix}_metrics"
    out_dir.mkdir(exist_ok=True)
    jobs = [(str(p), str(HERE / "configs" / suffix / p.name), str(corpus), family_json)
            for p in sorted(runs_dir.iterdir()) if p.is_dir() and (p / "DONE").is_file()]
    with Pool(args.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_run, jobs, chunksize=4) for r in part]
    rows.sort(key=lambda r: (r["seed"], r["config"], r["dataset"], r["scenario"]))
    with (out_dir / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[(r["seed"], r["config"])].append(r)
    fams = sorted({r["family"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    cfgs = sorted({c for _, c in by})
    prod, nop = space.name(space.PRODUCTION), space.name(space.NO_PROPAGATION)
    allf = set(fams)
    rep = [f"# Screen re-scored on every metric: {suffix}", "",
           f"{len(rows)} (initial matcher, configuration, scenario) triples; {len(jobs)} runs; "
           f"families {fams}. F1_Hr uses {K_PERM} random row/column permutations per scenario.", ""]

    # 1. tie bias at the input: how far the index-order Hungarian F1 is from the random-tie one
    rep += ["## Tie bias of the initial matchers (mean over scenarios)", "",
            "| initial matcher | F1_H | F1_Hr | F1_M | R_pess | R_exp | MRR_pess | MRR_exp |", "|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        rs = by[(s, nop)]
        rep.append(f"| {s} | " + " | ".join(f"{macro(rs, allf, 'seed_' + m):.3f}" for m in METRICS) + " |")

    # 2. per objective: best configurations over the panel, production, control
    summary = []
    for obj, parts in OBJECTIVES.items():
        panel = {c: mean(objective(by[(s, c)], allf, parts) for s in seeds) for c in cfgs
                 if all((s, c) in by for s in seeds)}
        for c, v in panel.items():
            summary.append({"objective": obj, "config": c, "panel_mean": v})
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["objective", "config", "panel_mean"])
        w.writeheader()
        w.writerows(summary)
    rep += ["", "## Best configurations over the panel (mean over initial matchers of the macro change)", "",
            "| objective | best | 2nd | 3rd | production | no propagation | rank of production |", "|---|---|---|---|---|---|---|"]
    rankings = {}
    for obj in OBJECTIVES:
        vals = {r["config"]: r["panel_mean"] for r in summary if r["objective"] == obj}
        order = sorted(vals, key=vals.get, reverse=True)
        rankings[obj] = order
        top = [f"{c} ({vals[c]:+.4f})" for c in order[:3]]
        rep.append(f"| {obj} | {' | '.join(top)} | {vals.get(prod, float('nan')):+.4f} | "
                   f"{vals.get(nop, float('nan')):+.4f} | {order.index(prod) + 1}/{len(order)} |")

    # 3. rank agreement between objectives (over configurations, panel means)
    from scipy.stats import spearmanr
    ref = {c: i for i, c in enumerate(rankings["J (pre-registered)"])}
    rep += ["", "## Agreement with the pre-registered J (Spearman over configuration ranks)", "",
            "| objective | rho |", "|---|---|"]
    for obj, order in rankings.items():
        rep.append(f"| {obj} | {spearmanr([ref[c] for c in order], list(range(len(order)))).statistic:+.3f} |")

    # 4. main effects per objective
    factors = {"operator": lambda r: f"{r['operator']}={r['value']}", "coeff": lambda r: r["coeff"],
               "attach": lambda r: r["attach"], "fd": lambda r: r["fd"]}
    rep += ["", "## Main effects (mean over initial matchers and the other factors)", ""]
    for obj, parts in OBJECTIVES.items():
        cells = defaultdict(list)
        for (s, c), rs in by.items():
            if c == nop:
                continue
            v = objective(rs, allf, parts)
            for f, get in factors.items():
                cells[(f, get(rs[0]))].append(v)
        rep.append(f"**{obj}:** " + "; ".join(f"{f} {lvl} {mean(v):+.3f}" for (f, lvl), v in sorted(cells.items())))
        rep.append("")

    # 5. alpha* per initial matcher (restart curve at production coefficient and dependency strength, ID)
    alphas = [c for c in cfgs if c.startswith("r") and c.endswith(f"__{space.PRODUCTION['coeff']}__ID__fdfull")]
    alphas = sorted(set(alphas) | {nop}, key=lambda c: float(c.split("__")[0][1:]))
    rep += ["## alpha* per initial matcher (restart, production coefficient and dependencies, ID attachment)", "",
            "| initial matcher | " + " | ".join(OBJECTIVES) + " |", "|---|" + "---|" * len(OBJECTIVES)]
    for s in seeds:
        cells = []
        for obj, parts in OBJECTIVES.items():
            curve = {c: objective(by[(s, c)], allf, parts) for c in alphas if (s, c) in by}
            best = max(curve, key=curve.get)
            cells.append(f"{best.split('__')[0][1:]} ({curve[best]:+.3f})")
        rep.append(f"| {s} | " + " | ".join(cells) + " |")

    # 6. keep sets under the pre-registered pruning rule, per objective
    def keep_set(parts):
        keep = defaultdict(set)
        for s in seeds:
            for held in fams:
                train = allf - {held}
                cands = {c: objective(rs, train, parts) for (x, c), rs in by.items() if x == s and c != nop}
                cands = {c: v for c, v in cands.items() if not math.isnan(v)}
                keep[s].update(sorted(cands, key=cands.get, reverse=True)[:TOP_K])
        return {(s, c) for s, cs in keep.items() for c in cs}
    ref_keep = keep_set(OBJECTIVES["J (pre-registered)"])
    rep += ["", "## Keep sets under the pre-registered pruning rule", "",
            "| objective | pairs | in J's keep set | Jaccard with J | its per-matcher best in J's keep set |", "|---|---|---|---|---|"]
    for obj, parts in OBJECTIVES.items():
        ks = keep_set(parts)
        best_in = 0
        for s in seeds:
            cands = {c: objective(rs, allf, parts) for (x, c), rs in by.items() if x == s and c != nop}
            best_in += (s, max(cands, key=cands.get)) in ref_keep
        rep.append(f"| {obj} | {len(ks)} | {len(ks & ref_keep)} | {len(ks & ref_keep) / len(ks | ref_keep):.2f} | "
                   f"{best_in}/{len(seeds)} |")

    (out_dir / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
