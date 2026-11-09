"""EXPLORATORY scorer for parallel (the parallel-edge rule for FD edges; layer3b/make_parallel_configs.py).

Checks first (refuse otherwise):
- gate: levenshtein__fd_droptwin (rule DROP set explicitly) = layer 1 levenshtein__fd, cell for cell;
- the rule changes only the propagation graph: for every arm and scenario, sf_stats.tsv's schema graph equals layer 1's
  fd (vertices and edges), wherever both exist.
Reported: levenshtein/pmf__melnik_split against layer 1 melnik (max |difference|; no FD edges, so about 0 expected).
Contrasts (J; statistic as Amendment 8: score_r7.Panel, family macro, 10,000 stratified resamples, seed 20261002;
no verdicts): fd_split / fd_pavg / fd_pmax − fd, fd_split − melnik, fd − melnik; and per family (point and interval).

    python layer3b/score_parallel.py --regime plain [--workers 10]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
from score_r7 import Panel, is_control, metrics, run_dir  # noqa: E402
from score_classfull import graph_sizes  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from score_seeds import CORPUS, cases_for, ground_truth_file, one  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402

RES = STUDY / "tuning" / "results"
ARMS = ("split", "pavg", "pmax")
F = ["J", "R_exp", "F1_Hr"]


def out_file(d, ds, sc):
    return one(d / ds / sc / "_outputs" / "MatchingStepLine2") if d else None


def score_seed(job):
    regime, seed, cases, pr, l1, seeds_root = job
    dirs = {"fd": run_dir(l1 / f"{seed}__fd"), "melnik": run_dir(l1 / f"{seed}__melnik"),
            **{f"fd_{a}": run_dir(pr / f"{seed}__fd_{a}") for a in ARMS}}
    out, skipped = [], 0
    for ds, sc, fam in cases:
        f_q, g = seeds_root / seed / ds / sc / "seed.csv", ground_truth_file(CORPUS[regime], ds, sc)
        if not f_q.is_file() or g is None:
            continue
        gt = read_matrix(g)[0] > 0.5
        q = read_matrix(f_q)[0]
        if q.shape != gt.shape or not np.isfinite(q).all():
            continue
        perms = permutations(gt.shape, f"{ds}/{sc}")
        for arm, d in dirs.items():
            f = out_file(d, ds, sc)
            if f is None:
                continue
            M = read_matrix(f)[0]
            if M.shape != q.shape or not np.isfinite(M).all():
                skipped += 1
                continue
            out.append({"seed": seed, "dataset": ds, "scenario": sc, "family": fam, "arm": arm,
                        **{f"sf_{k}": v for k, v in metrics(M, gt, perms).items() if k in F}})
    return out, skipped


def max_diff(a, b, cases):
    worst, n = 0.0, 0
    for ds, sc, _ in cases:
        fa, fb = out_file(a, ds, sc), out_file(b, ds, sc)
        if fa is None or fb is None:
            continue
        A_, B_ = read_matrix(fa)[0], read_matrix(fb)[0]
        worst = max(worst, float(np.abs(A_ - B_).max()) if A_.shape == B_.shape else float("inf"))
        n += 1
    return worst, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5"), required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--pr-root", default="")
    ap.add_argument("--l1-root", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    r = a.regime
    if os.environ.get("SMX_VALENTINE", "382") == "549":
        return main549(r)
    pr, l1 = Path(a.pr_root or RES / f"parallel_{r}"), Path(a.l1_root or RES / f"layer1_{r}")
    out = Path(a.out or RES / f"parallel_{r}_scores")
    out.mkdir(parents=True, exist_ok=True)
    cases = cases_for(r)
    fam_of = {(d, s): f for d, s, f in cases}

    x, y = run_dir(pr / "levenshtein__fd_droptwin"), run_dir(l1 / "levenshtein__fd")
    if x is None or y is None:
        raise SystemExit("gate impossible: levenshtein__fd_droptwin or layer 1 levenshtein__fd missing")
    w, n = max_diff(x, y, cases)
    checks = [f"gate levenshtein__fd_droptwin = layer 1 levenshtein__fd: max |difference| {w} on {n} scenarios"]
    print(checks[-1])
    if w != 0.0 or n != len(cases):
        raise SystemExit("gate FAILED: nothing is scored")

    seeds = sorted({p.name.split("__", 1)[0] for p in pr.iterdir() if "__fd_" in p.name and "droptwin" not in p.name})
    checked = bad = 0
    for s in seeds:
        base = graph_sizes(l1 / f"{s}__fd")
        for arm in ARMS:
            sizes = graph_sizes(pr / f"{s}__fd_{arm}")
            for k, v in sizes.items():
                if k in base:
                    checked += 1
                    bad += v != base[k]
    checks.append(f"schema graph unchanged by the rule (sf_stats, arm vs layer 1 fd): {checked} checked, {bad} different")
    print(checks[-1])
    if bad or checked == 0:
        raise SystemExit("schema-graph check FAILED: nothing is scored")
    for s in ("levenshtein", "pmf"):
        w, n = max_diff(run_dir(pr / f"{s}__melnik_split"), run_dir(l1 / f"{s}__melnik"), cases)
        checks.append(f"report: {s}__melnik_split vs layer 1 {s}__melnik (no FD edges): max |difference| {w} on {n} scenarios")

    with Pool(a.workers) as pool:
        parts = list(pool.imap_unordered(score_seed, [(r, s, cases, pr, l1, STUDY / "reference" / "seeds" / r)
                                                      for s in seeds]))
    rows = [x for p, _ in parts for x in p]
    checks.append(f"skipped (not finite or misshapen; reported only): {sum(k for _, k in parts)} (matcher, scenario, arm)")
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    write_report(r, rows, fam_of, checks, out)
    return 0


def write_report(r: str, rows: list[dict], fam_of: dict, checks: list[str], out: Path) -> None:
    """main()'s contrasts and report, unchanged (moved into a function 2026-11-01 for the 549 run)."""
    C = [("fd_split", "fd"), ("fd_pavg", "fd"), ("fd_pmax", "fd"), ("fd_split", "melnik"), ("fd", "melnik")]
    inc = sorted(s for s in {x["seed"] for x in rows} if not is_control(s))
    report = [f"# parallel (EXPLORATORY): the parallel-edge rule for FD edges, regime {r}", "",
              *[f"- {c}" for c in checks], "",
              "J = (R_exp + F1_Hr)/2; per matcher the macro over families with ≥ 5 scenarios; mean over included "
              "non-control matchers; 95% bootstrap stratified by family, 10,000 resamples, seed 20261002. No verdicts.", ""]
    store = []
    sets = [("all", list(fam_of))] + [(f, [k for k, v in fam_of.items() if v == f]) for f in sorted(set(fam_of.values()))]
    for name, keys in sets:
        if len(keys) < 5:
            continue
        P = Panel(rows, keys, fam_of)
        if not P.fams:
            continue
        report += [f"## {name} ({len(keys)} scenarios)", "",
                   "| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 |", "|---|---|---|---|---|---|"]
        for plus, minus in C:
            res = {}
            for f in F:
                E, kept = P.matrix(inc, plus, minus, f"sf_{f}")
                if not kept:
                    break
                res[f] = (*P.stat(E, np.nanmean), kept, E)
            if "J" not in res:
                report.append(f"| {plus} − {minus} | 0 | – | – | – | – |")
                continue
            est, lo, hi, kept, E = res["J"]
            pos = int((P.macro(E) > 0).sum())
            report.append(f"| {plus} − {minus} | {len(kept)} | {est:+.4f} [{lo:+.4f}, {hi:+.4f}] | {res['R_exp'][0]:+.4f} | "
                          f"{res['F1_Hr'][0]:+.4f} | {pos} of {len(kept)} |")
            store.append({"set": name, "contrast": f"{plus} − {minus}", "matchers": len(kept), "estimate": est, "lo": lo,
                          "hi": hi, "R_exp": res["R_exp"][0], "F1_Hr": res["F1_Hr"][0], "matchers_positive": pos})
        report.append("")
    with (out / "contrasts.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(store[0]))
        wr.writeheader()
        wr.writerows(store)
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


def main549(r: str) -> int:
    """SMX_VALENTINE=549: contrasts from the assembled rows (tuning/results/parallel_<r>549_scores: fd_split from
    parallelfix_<fix> + parallel_<new>, reference/assemble_549.py) with layer 1's fd and melnik from layer1_<r>549_scores,
    over the 549 scenarios; fd_pavg / fd_pmax are not rerun on 549 (their contrasts show 0 matchers). The gate and the
    schema-graph check need the run folders and are not repeated here. Written next to the assembled rows."""
    sys.path.insert(0, str(STUDY / "reference"))
    import valentine549 as v
    folder = f"parallel_{v.vreg(r)}_scores"
    par, l1 = v.assembled(folder), v.assembled(f"layer1_{v.vreg(r)}_scores")
    if par is None or l1 is None or v.arm_status(f"layer1_{r}", "fd") != "assembled":
        print(f"{folder}: pending (parallel or layer-1 fd not assembled); nothing written")
        return 0
    rows = [{"seed": x["seed"], "dataset": x["dataset"], "scenario": x["scenario"], "family": x["family"],
             "arm": x["arm"], "sf_J": (float(x["sf_R_exp"]) + float(x["sf_F1_Hr"])) / 2,
             "sf_R_exp": float(x["sf_R_exp"]), "sf_F1_Hr": float(x["sf_F1_Hr"])}
            for x in par + [y for y in l1 if y["arm"] in ("fd", "melnik")]]
    fam_of = {(d, s): f for d, s, f in v.cases549()}
    write_report(r, rows, fam_of, ["549 run (SMX_VALENTINE=549): assembled rows; the gate and the schema-graph check "
                                   "were not repeated (they need the run folders)"], v.RES / folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
