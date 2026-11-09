"""Score a tuning stage and apply the pre-registered decision rules (PREREGISTRATION.md).

Per (seed, configuration, scenario), with Δ = SF(seed) - seed and macro means over families:

    J     = (ΔRecall@GT_pessimistic + ΔF1_hungarian) / 2                 (pre-registered)
    J_tf  = (ΔRecall@GT_expected   + ΔF1_hungarian_random_ties) / 2      (Amendment 2, tie-free)

The screen selects by J, as registered, so keep.json stays what it was. Dev and lockbox select
by J_tf (Amendment 2) and report J next to it. The tie-free metrics are those of
score_screen_metrics.py: expected Recall@GT under random tie-breaking, and Hungarian F1 averaged
over random row and column orders (the same orders for seed and SF output, seeded by scenario).

Outputs under tuning/results/<stage>_<regime>/:
    scores.csv     one row per seed x configuration x scenario
    summary.csv    macro J per seed x configuration (+ per family)
    lofo.csv       leave-one-family-out: chosen configuration per fold, held-out J vs the
                   production configuration and vs no propagation (alpha = 0)
    alpha.csv      restart alpha curve per seed (other factors at production values)
    report.md      all of the above in prose-free tables, plus main effects and the adaptive-alpha test
and, by stage:
    screen  -> tuning/keep.json    union over folds and seeds of each fold's top 10 (training families)
    dev     -> tuning/chosen.json  best configuration per seed and one for all seeds, on all of dev

Usage (needs numpy/scipy; the harness's .venv-bdi has them):
    ../../harness/.venv-bdi/Scripts/python.exe score_tuning.py --stage screen --regime plain
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(HERE))

from smx.adapters import read_matrix  # noqa: E402
from smx.core import extract, match_metrics, recall_at_ground_truth  # noqa: E402

import space  # noqa: E402
from score_screen_metrics import hungarian_random, permutations, recall_expected  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from explore_posthoc import tiebreak  # noqa: E402  (Amendment 3; the same construction as the exploration)
from explore_penalty import pa  # noqa: E402  (report only: relative re-scaling, EXPLORATORY.md X6)

PROTOCOL = ROOT / "harness" / "reports" / "audited_md5_fused_20260904" / "protocol.json"
TIE = 0.002
TOP_K = 10
# The objective that drives selection: J at the screen (as registered), J_tf at dev and lockbox
# (Amendment 2). Set in main().
OBJ = "J"


def one(folder: Path) -> Path | None:
    files = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    return files[0] if len(files) == 1 else None


def metrics(S: np.ndarray, gt: np.ndarray, perms) -> tuple[float, float, float, float]:
    """Registered pair (pessimistic Recall@GT, index-order Hungarian F1) and tie-free pair."""
    r = recall_at_ground_truth(S, gt)["recall_at_gt_pessimistic"]
    f = match_metrics(extract(S, "hungarian"), gt)["f1"]
    return float(r), float(f), recall_expected(S, gt), hungarian_random(S, gt, perms)


def predictors(S: np.ndarray) -> dict[str, float]:
    """Ground-truth-free seed-matrix quality predictors, as CompositeMatcher.predict."""
    n, m = S.shape
    row_max = S.max(axis=1, keepdims=True)
    col_max = S.max(axis=0, keepdims=True)
    dominant = (S > 0) & (S == row_max) & (S == col_max)
    return {"DOMINANTS": float(dominant.any(axis=1).sum()) / min(n, m),
            "STDEV": float(S.std()),
            "MAXMEAN": float(row_max.mean())}


def macro(rows: list[dict], families: set[str], key: str | None = None) -> float:
    key = key or OBJ
    per = defaultdict(list)
    for r in rows:
        if r["family"] in families:
            per[r["family"]].append(r[key])
    vals = [mean(v) for v in per.values() if v]
    return mean(vals) if vals else float("nan")


def simplicity(cfg_name: str) -> tuple:
    """Tie-break: no restart first, then fewest factors away from production."""
    c = configs_by_name()[cfg_name]
    away = sum(c[k] != v for k, v in space.PRODUCTION.items() if k not in ("operator", "value"))
    return (c["operator"] == "restart", c["operator"] != "fixpoint" or c["value"] != "C", away)


_BY_NAME = None


def configs_by_name() -> dict[str, dict]:
    global _BY_NAME
    if _BY_NAME is None:
        _BY_NAME = {space.name(c): c for c in space.configurations()}
    return _BY_NAME


def choose(cands: dict[str, float]) -> str:
    """Best by score; within TIE of the best, the simplest."""
    best = max(cands.values())
    near = [c for c, v in cands.items() if v >= best - TIE]
    return min(near, key=simplicity)


def spearman(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("screen", "dev", "lockbox"), required=True)
    ap.add_argument("--regime", choices=("plain", "md5"), default="plain")
    args = ap.parse_args()
    global OBJ
    OBJ = "J" if args.stage == "screen" else "J_tf"

    suffix = f"{args.stage}_{args.regime}"
    family = {(c["dataset"], c["scenario"]): c["family"]
              for c in json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]}
    corpus = HERE / "corpus" / args.regime / args.stage
    out_dir = HERE / "results" / suffix
    rows, problems = [], []
    seed_cache: dict[tuple, tuple] = {}
    q_cache: dict[tuple, np.ndarray] = {}
    tb_rows: list[dict] = []   # Amendment 3 tie-breaker scores (dev and lockbox only)
    pa_rows: list[dict] = []   # relative re-scaling of SF's output (report only; dev and lockbox only)
    for run_root in sorted(p for p in out_dir.iterdir() if p.is_dir() and (p / "DONE").is_file()):
        meta = json.loads((HERE / "configs" / suffix / run_root.name / "meta.json").read_text(encoding="utf-8"))
        run = sorted(p for p in run_root.iterdir() if p.is_dir())[-1]
        for scen in sorted(run.glob("*/*")):
            ds, sc = scen.parent.name, scen.name
            gt_file = one(corpus / ds / sc / "ground_truth")
            sf_file = one(scen / "_outputs" / "MatchingStepLine2")
            if gt_file is None or sf_file is None:
                problems.append((run_root.name, sc, "missing ground truth or SF output"))
                continue
            gt = read_matrix(gt_file)[0] > 0.5
            perms = permutations(gt.shape, f"{ds}/{sc}")
            key = (meta["seed"], ds, sc)
            if key not in seed_cache:
                q = read_matrix(Path(meta["seed_dir"]) / ds / sc / "seed.csv")[0]
                seed_cache[key] = (*metrics(q, gt, perms), predictors(q)) if q.shape == gt.shape else None
                q_cache[key] = q
            S = read_matrix(sf_file)[0]
            if seed_cache[key] is None or S.shape != gt.shape or not np.isfinite(S).all():
                problems.append((run_root.name, sc, "shape or non-finite"))
                continue
            seed_r, seed_f, seed_re, seed_fr, pred = seed_cache[key]
            sf_r, sf_f, sf_re, sf_fr = metrics(S, gt, perms)
            if args.stage != "screen":
                # Amendment 3: SF as tie-breaker, report only (never used for selection)
                tb, _ = tiebreak(np.clip(q_cache[key], 0.0, None), S)
                _, _, tb_re, tb_fr = metrics(tb, gt, perms)
                tb_rows.append({"seed": meta["seed"], "config": meta["config"], "dataset": ds, "scenario": sc,
                                "initial_R_exp": seed_re, "initial_F1_Hr": seed_fr,
                                "sf_R_exp": sf_re, "sf_F1_Hr": sf_fr,
                                "tiebreak_R_exp": tb_re, "tiebreak_F1_Hr": tb_fr})
                # report only, never used for selection: SF's output x pa = s / max(row max, column max)
                Sc = np.clip(S, 0.0, None)
                _, _, pa_re, pa_fr = metrics(Sc * pa(Sc), gt, perms)
                pa_rows.append({"seed": meta["seed"], "config": meta["config"], "dataset": ds, "scenario": sc,
                                "family": family[(ds, sc)], "initial_R_exp": seed_re, "initial_F1_Hr": seed_fr,
                                "sf_R_exp": sf_re, "sf_F1_Hr": sf_fr, "sf_pa_R_exp": pa_re, "sf_pa_F1_Hr": pa_fr,
                                "J_tf": ((sf_re - seed_re) + (sf_fr - seed_fr)) / 2,
                                "J_tf_pa": ((pa_re - seed_re) + (pa_fr - seed_fr)) / 2})
            rows.append({"seed": meta["seed"], "config": meta["config"], "operator": meta["operator"],
                         "value": meta["value"], "coeff": meta["coeff"], "attach": meta["attach"],
                         "fd": meta["fd"], "dataset": ds, "scenario": sc, "family": family[(ds, sc)],
                         "seed_R": seed_r, "seed_F": seed_f, "sf_R": sf_r, "sf_F": sf_f,
                         "seed_Rexp": seed_re, "seed_FHr": seed_fr, "sf_Rexp": sf_re, "sf_FHr": sf_fr,
                         "dR": sf_r - seed_r, "dF": sf_f - seed_f, "dRexp": sf_re - seed_re, "dFHr": sf_fr - seed_fr,
                         "J": ((sf_r - seed_r) + (sf_f - seed_f)) / 2,
                         "J_tf": ((sf_re - seed_re) + (sf_fr - seed_fr)) / 2,
                         **{f"pred_{k}": v for k, v in pred.items()}})
    if not rows:
        raise SystemExit(f"nothing scored under {out_dir}")
    with (out_dir / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if tb_rows:
        # input of confirm_tiebreak.py (Amendment 3); one row per (initial matcher, configuration, scenario)
        with (out_dir / "tiebreak_scores.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(tb_rows[0]))
            w.writeheader()
            w.writerows(tb_rows)
    if pa_rows:
        # report only: SF's output re-scaled by its row and column maxima (EXPLORATORY.md X6)
        with (out_dir / "pa_scores.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pa_rows[0]))
            w.writeheader()
            w.writerows(pa_rows)

    by = defaultdict(list)
    for r in rows:
        by[(r["seed"], r["config"])].append(r)
    families = sorted({r["family"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    report = [f"# Tuning report: {suffix}", "",
              f"{len(rows)} scored (seed, configuration, scenario) triples; families {families}; "
              f"{len(problems)} problems.", "",
              f"Selection objective: **{OBJ}**"
              + (" (pre-registered)." if OBJ == "J" else " (Amendment 2, tie-free); the registered J is reported next to it."),
              ""]

    # --- summary ---------------------------------------------------------------------------
    summary = []
    for (seed, cfg), rs in sorted(by.items()):
        row = {"seed": seed, "config": cfg, "n": len(rs),
               **{f"macro_{k}": macro(rs, set(families), k) for k in ("J", "J_tf", "dR", "dF", "dRexp", "dFHr")}}
        for f in families:
            row[f"{OBJ}_{f}"] = macro(rs, {f})
        summary.append(row)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    prod, nop = space.name(space.PRODUCTION), space.name(space.NO_PROPAGATION)

    # --- leave-one-family-out ----------------------------------------------------------------
    lofo, keep = [], defaultdict(set)
    report += ["## Leave-one-family-out", "",
               f"| seed | held-out family | chosen (training families) | held-out {OBJ} | production | no propagation "
               f"| held-out {'J_tf' if OBJ == 'J' else 'J'} of the chosen |",
               "|---|---|---|---|---|---|---|"]
    other = "J_tf" if OBJ == "J" else "J"
    for seed in seeds + ["ALL_SEEDS"]:
        for held in families:
            train = set(families) - {held}
            if seed == "ALL_SEEDS":
                common = set.intersection(*[{c for (s, c) in by if s == x} for x in seeds])
                cands = {c: mean(macro(by[(x, c)], train) for x in seeds) for c in common}
                held_val = lambda c, k=None: mean(macro(by[(x, c)], {held}, k) for x in seeds)  # noqa: E731
            else:
                cands = {c: macro(rs, train) for (s, c), rs in by.items() if s == seed}
                held_val = lambda c, k=None: macro(by[(seed, c)], {held}, k)  # noqa: E731
            cands = {c: v for c, v in cands.items() if not math.isnan(v) and c != nop}
            if not cands:
                continue
            chosen = choose(cands)
            if seed != "ALL_SEEDS":
                keep[seed].update(sorted(cands, key=cands.get, reverse=True)[:TOP_K])
            # The objective's columns are named *_obj: under J_tf (dev, lockbox) a key "held_out_J"
            # would collide with the other objective's column f"held_out_{other}" (fixed 2026-10-02;
            # the dev report of 22:25 showed J under the J_tf heading; chosen.json was not affected).
            rec = {"seed": seed, "held_out": held, "chosen": chosen, "held_out_obj": held_val(chosen),
                   "production_obj": held_val(prod) if prod in cands or seed == "ALL_SEEDS" else float("nan"),
                   "no_propagation_obj": held_val(nop) if (seed, nop) in by or seed == "ALL_SEEDS" else float("nan"),
                   "objective": OBJ, f"held_out_{other}": held_val(chosen, other)}
            lofo.append(rec)
            report.append(f"| {seed} | {held} | {chosen} | {rec['held_out_obj']:+.4f} | "
                          f"{rec['production_obj']:+.4f} | {rec['no_propagation_obj']:+.4f} | "
                          f"{rec[f'held_out_{other}']:+.4f} |")
    with (out_dir / "lofo.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lofo[0]))
        w.writeheader()
        w.writerows(lofo)

    # --- alpha curve and adaptive-alpha test ---------------------------------------------------
    alphas = [0.0, 0.15, 0.30, 0.50, 0.70, 0.85]
    def alpha_cfg(a):  # restart at production coefficient and dependency strength, ID attachment
        return space.name({**space.PRODUCTION, "operator": "restart", "value": a,
                           "attach": space.ALPHA_CURVE_ATTACH})
    alpha_rows = []
    report += ["", "## Restart alpha (production coefficient and dependency strength, ID attachment)", "",
               "| seed | seed R@GT | " + " | ".join(f"α={a:.2f}" for a in alphas) + " | α* |",
               "|---|---|" + "---|" * len(alphas) + "---|"]
    astar, quality = [], []
    scen_best = defaultdict(list)   # predictor-vs-best-alpha points, with family for the fold check
    for seed in seeds:
        curve = {a: macro(by.get((seed, alpha_cfg(a)), []), set(families)) for a in alphas}
        if all(math.isnan(v) for v in curve.values()):
            continue
        best_a = max((a for a in alphas if not math.isnan(curve[a])), key=lambda a: curve[a])
        q = mean(r["seed_R"] for r in by[(seed, alpha_cfg(best_a))])
        astar.append(best_a)
        quality.append(q)
        alpha_rows.append({"seed": seed, "seed_R": q, **{f"a{a:.2f}": curve[a] for a in alphas}, "alpha_star": best_a})
        report.append(f"| {seed} | {q:.3f} | " + " | ".join(f"{curve[a]:+.4f}" for a in alphas) + f" | {best_a:.2f} |")
        per_scen = defaultdict(dict)
        for a in alphas:
            for r in by.get((seed, alpha_cfg(a)), []):
                per_scen[(r["dataset"], r["scenario"])][a] = r
        for scen, d in per_scen.items():
            if len(d) == len(alphas):
                any_r = next(iter(d.values()))
                b = max(alphas, key=lambda a: d[a][OBJ])
                scen_best[seed].append((b, any_r, any_r["family"]))
    if alpha_rows:
        with (out_dir / "alpha.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(alpha_rows[0]))
            w.writeheader()
            w.writerows(alpha_rows)
    rho_seed = spearman(astar, quality)
    report += ["", f"Spearman(α*, seed quality) across seeds: {rho_seed:+.3f}", "",
               "| predictor | ρ(best α, predictor), pooled scenarios | ρ per held-out family | same sign in every fold |",
               "|---|---|---|---|"]
    points = [p for v in scen_best.values() for p in v]
    decision = []
    for pred in ("DOMINANTS", "STDEV", "MAXMEAN"):
        rho = spearman([p[0] for p in points], [p[1][f"pred_{pred}"] for p in points])
        per_fold = {}
        for held in families:
            sub = [p for p in points if p[2] != held]
            per_fold[held] = spearman([p[0] for p in sub], [p[1][f"pred_{pred}"] for p in sub])
        signs = {math.copysign(1, v) for v in per_fold.values() if not math.isnan(v)}
        consistent = len(signs) == 1
        decision.append(abs(rho) >= 0.5 and consistent)
        report.append(f"| {pred} | {rho:+.3f} | " + ", ".join(f"{k} {v:+.2f}" for k, v in per_fold.items())
                      + f" | {'yes' if consistent else 'no'} |")
    report += ["", "**Adaptive-α rule (pre-registered: |ρ| ≥ 0.5, same sign in every fold):** "
               + ("met; continue to an adaptive-α design." if any(decision)
                  else "not met; α* is not predictable from the seed alone on this stage.")]

    # --- main effects ------------------------------------------------------------------------
    report += ["", f"## Main effects (macro {OBJ}, mean over seeds)", "", f"| factor | level | mean macro {OBJ} |", "|---|---|---|"]
    for factor in ("operator_value", "coeff", "attach", "fd"):
        levels = defaultdict(list)
        for (seed, cfg), rs in by.items():
            if cfg == nop:
                continue
            c = configs_by_name()[cfg]
            lvl = f"{c['operator']}={c['value']}" if factor == "operator_value" else c[factor]
            levels[lvl].append(macro(rs, set(families)))
        for lvl, vals in sorted(levels.items()):
            report.append(f"| {factor} | {lvl} | {mean(vals):+.4f} |")

    # --- stage outputs -----------------------------------------------------------------------
    if args.stage == "screen":
        (HERE / "keep.json").write_text(json.dumps({
            "rule": f"union over folds of the top {TOP_K} by macro J on each fold's training families",
            "source": suffix, "configurations": {s: sorted(v) for s, v in keep.items()}}, indent=1),
            encoding="utf-8")
        report += ["", f"keep.json: {sum(len(v) for v in keep.values())} (seed, configuration) pairs go to dev."]
    if args.stage == "dev":
        chosen = {}
        common = set.intersection(*[{c for (s, c) in by if s == x} for x in seeds]) - {nop}
        universal = choose({c: mean(macro(by[(x, c)], set(families)) for x in seeds) for c in common})
        for seed in seeds:
            cands = {c: macro(rs, set(families)) for (s, c), rs in by.items() if s == seed and c != nop}
            chosen[seed] = sorted({choose(cands), universal})
        (HERE / "chosen.json").write_text(json.dumps({
            "rule": f"best macro {OBJ} on all dev families, per seed, plus the best single configuration for all seeds"
                    + (" (Amendment 2: tie-free J)" if OBJ == "J_tf" else ""),
            "objective": OBJ,
            "universal": universal, "source": suffix, "configurations": chosen}, indent=1), encoding="utf-8")
        report += ["", f"chosen.json: universal configuration {universal}; per seed: "
                   + "; ".join(f"{s}: {', '.join(v)}" for s, v in chosen.items())]
    # --- report only: relative re-scaling (pa) -------------------------------------------------
    if pa_rows and (HERE / "chosen.json").is_file():
        cfgs = json.loads((HERE / "chosen.json").read_text(encoding="utf-8"))["configurations"]
        pa_by = defaultdict(list)
        for r in pa_rows:
            pa_by[(r["seed"], r["config"])].append(r)
        report += ["", "## Report only: SF's output re-scaled by its row and column maxima (pa)", "",
                   "Never used for selection (EXPLORATORY.md X6). Macro J_tf over all families, for the "
                   "chosen configurations.", "",
                   "| seed | configuration | J_tf SF | J_tf SF × pa | difference |", "|---|---|---|---|---|"]
        diffs = []
        for seed in sorted(cfgs):
            for c in cfgs[seed]:
                rs = pa_by.get((seed, c))
                if not rs:
                    continue
                j, jp = macro(rs, set(families), "J_tf"), macro(rs, set(families), "J_tf_pa")
                diffs.append(jp - j)
                report.append(f"| {seed} | {c} | {j:+.4f} | {jp:+.4f} | {jp - j:+.4f} |")
        if diffs:
            report.append(f"\nMean difference {mean(diffs):+.4f}; pa better in {sum(d > 1e-4 for d in diffs)} "
                          f"of {len(diffs)} (seed, configuration) pairs.")
    if problems:
        report += ["", f"## Problems ({len(problems)})", ""] + [f"- {p}" for p in problems[:50]]
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[:12]))
    print(f"... full report: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
