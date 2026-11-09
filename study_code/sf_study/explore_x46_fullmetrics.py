"""EXPLORATORY X46 rescoring (2026-10-08): the dependency role term on the full metric set.

X46 (explore_deproles.py, unchanged) scored A_d = q+ + 5D and A_role = A_d + gamma R by J_tf only. The results overview
figure ("SF's measured effect under different evaluation choices") shows the mean change over the initial matcher in five
metrics for SF as published and SF in its standard configuration. This script adds the role term as a third series.

Panel: the RQ1 panel of gen_figures.py (layer 1 arm melnik: initial matchers with all 382 scenarios, in META, not invalid,
not controls, without semstruct_lm_only; the data-type matcher from its NaN-free rerun), per regime.
Matrices: q (the initial matcher), A_d, A_role with gamma as chosen leave-one-family-out by X46 (read from
results/<r>/deproles_summary.md, line "- role:"), R exactly as in explore_deproles.py (its functions are imported).
Metrics: score_screen_metrics.all_metrics (R_exp, R_pess, F1_Hr, F1_H, F1_M, MRR_exp, MRR_pess), the same tie orders.
SF as published: the sf_* columns of layer 1 melnik (per scenario).
Statistic per contrast and metric: the mean over matchers of the family macro (families with >= 5 scenarios), which equals
the family macro of the per-scenario mean over matchers; 95% bootstrap of scenarios within family (10,000, seed 20261011,
a fresh generator per contrast and metric); Up = matchers whose macro difference is > 0.
Checks: q's metrics against layer 1's seed_* columns; A_d and A_role's J_tf against deproles_scores.csv where both exist.

    ../harness/.venv-bdi/Scripts/python.exe explore_x46_fullmetrics.py --regime plain|md5 [--workers 10]
Output: results/<r>/x46_fullmetrics_{scores.csv,summary.csv,checks.md}.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE / "tuning"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "latex" / "sf-evaluation"))

import explore_deproles as xd  # noqa: E402
V = xd.V   # reference/valentine549.py (rerun 549): SMX_VALENTINE=549 reads the assembled 549 inputs, else unused
from score_screen_metrics import all_metrics, permutations  # noqa: E402

METRICS = ("R_exp", "R_pess", "F1_Hr", "F1_H", "F1_M", "MRR_exp", "MRR_pess")
B, SEED = 10_000, 20261011


def role_gammas(regime: str) -> dict[str, float]:
    for line in (HERE / "results" / V.vreg(regime) / "deproles_summary.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("- role:"):
            return {f: float(g) for f, g in re.findall(r"([\w-]+) \(([\d.]+), [\d.]+\)", line)}
    raise SystemExit("no role line in deproles_summary.md")


def panel(regime: str) -> tuple[list[str], dict]:
    import gen_figures as gf
    rows = gf.layer1_rows(regime, "melnik")
    pts, _ = gf.load_rows(rows, gf.BASE_N if V.V549 else 382)
    seeds = sorted(p["seed"] for p in pts if p["group"] != "ctrl")
    sf = {(r["seed"], r["dataset"], r["scenario"]): r for r in rows if r["seed"] in seeds}
    return seeds, sf


def score(job):
    regime, m, cs, gam = job
    seeds = HERE / "reference" / "seeds" / regime
    src = "datatype_nanfix" if m == "datatype" else m
    out = []
    for ds, sc, fam in cs:
        f_q, f_d = seeds / src / ds / sc / "seed.csv", seeds / "datatype_nanfix" / ds / sc / "seed.csv"
        g = xd.ground_truth_file(xd.CORPUS[regime], ds, sc)
        if V.V549:   # 549: initial matrices per valentine549.seed_file; truth from the 382 tree or the new168 tree
            f_q, f_d = V.seed_file(regime, src, ds, sc), V.seed_file(regime, "datatype_nanfix", ds, sc)
            g = xd.ground_truth_file(V.NEW_TREE[regime] if V.is_new(ds, sc) else xd.CORPUS[regime], ds, sc)
        if not (f_q.is_file() and f_d.is_file() and g is not None):
            continue
        q, rows, cols = xd.read_matrix(f_q)
        D = xd.read_matrix(f_d)[0]
        gt = xd.read_matrix(g)[0] > 0.5
        if not (q.shape == D.shape == gt.shape):
            continue
        q = np.nan_to_num(q, nan=0.0) if m == "datatype" else q
        if not (np.isfinite(q).all() and np.isfinite(D).all()):
            continue
        qp = np.clip(q, 0, None)
        Fs, _ = xd.roles(list(rows), *xd.profiles(regime, ds, sc, "source"))
        Ft, _ = xd.roles(list(cols), *xd.profiles(regime, ds, sc, "target"))
        R = 1.0 - np.abs(Fs[:, None, :] - Ft[None, :, :]).mean(axis=2)
        perms = permutations(gt.shape, f"{ds}/{sc}")
        Ad = qp + 5.0 * D
        gm = gam.get(fam, gam["_other"])
        row = {"seed": m, "dataset": ds, "scenario": sc, "family": fam, "gamma": gm}
        for name, S in (("q", q), ("A_d", Ad), ("A_role", Ad + gm * R)):
            r = all_metrics(S, gt, perms)
            row.update({f"{name}_{k}": r[k] for k in METRICS})
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5"), required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    cs = V.cases549() if V.V549 else xd.cases_for(a.regime)
    if V.V549 and not (HERE / "results" / V.vreg(a.regime) / "deproles_summary.md").is_file():
        print(f"x46 fullmetrics {a.regime} 549: pending (no 549 deproles summary); nothing written")
        return 0
    gam = role_gammas(a.regime)
    seeds, sf = panel(a.regime)
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(score, [(a.regime, s, cs, gam) for s in seeds]) for r in part]
    for r in rows:
        x = sf.get((r["seed"], r["dataset"], r["scenario"]))
        for k in METRICS:
            r[f"SF_{k}"] = float(x[f"sf_{k}"]) if x else float("nan")
            r[f"L1q_{k}"] = float(x[f"seed_{k}"]) if x else float("nan")
    cover = defaultdict(int)
    for r in rows:
        if not np.isnan(r["SF_R_exp"]):
            cover[r["seed"]] += 1
    dropped = [s for s in seeds if cover[s] != len(cs)]
    seeds = [s for s in seeds if cover[s] == len(cs)]
    rows = sorted((r for r in rows if r["seed"] in seeds), key=lambda r: (r["seed"], r["dataset"], r["scenario"]))
    out = HERE / "results" / V.vreg(a.regime)
    with (out / "x46_fullmetrics_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = {(r["seed"], r["dataset"], r["scenario"]): r for r in rows}
    fams = defaultdict(list)
    for d, c, f in cs:
        fams[f].append((d, c))
    big = {f: v for f, v in fams.items() if len(v) >= 5}
    contrasts = (("A_role", "q"), ("A_d", "q"), ("A_role", "SF"), ("A_d", "SF"), ("A_role", "A_d"), ("SF", "q"))
    summ = []
    for x, y in contrasts:
        for k in METRICS:
            diff = lambda s, d, c: by[(s, d, c)][f"{x}_{k}"] - by[(s, d, c)][f"{y}_{k}"]  # noqa: E731
            per = {f: np.array([np.mean([diff(s, d, c) for s in seeds]) for d, c in v]) for f, v in big.items()}
            est = float(np.mean([v.mean() for v in per.values()]))
            rng = np.random.default_rng(SEED)
            boots = np.mean([[rng.choice(v, size=v.size).mean() for _ in range(B)] for v in per.values()], axis=0)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            mac = [np.mean([np.mean([diff(s, d, c) for d, c in v]) for v in big.values()]) for s in seeds]
            summ.append({"contrast": f"{x} - {y}", "metric": k, "estimate": f"{est:.5f}", "lo": f"{lo:.5f}",
                         "hi": f"{hi:.5f}", "up": sum(v > 0 for v in mac), "matchers": len(seeds),
                         "scenarios": sum(len(v) for v in big.values())})
            print(summ[-1], flush=True)
    with (out / "x46_fullmetrics_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)

    # checks
    qdev = max(abs(r[f"q_{k}"] - r[f"L1q_{k}"]) for r in rows for k in METRICS)
    old = {}
    f = out / "deproles_scores.csv"
    if f.is_file():
        for r in csv.DictReader(open(f, encoding="utf-8")):
            old[(r["seed"], r["dataset"], r["scenario"])] = r
    jd, n = [], 0
    for r in rows:
        o = old.get((r["seed"], r["dataset"], r["scenario"]))
        if o:
            n += 1
            jd.append(abs((r["A_d_R_exp"] + r["A_d_F1_Hr"]) / 2 - float(o["A_0.0_0.0"])))
            jd.append(abs((r["A_role_R_exp"] + r["A_role_F1_Hr"]) / 2 - float(o[f"A_{r['gamma']}_0.0"])))
    lines = [f"# X46 full-metric rescoring ({a.regime}): checks", "",
             f"- panel: {len(seeds)} initial matchers (RQ1 panel), {len(cs)} scenarios; dropped for missing cells: {dropped}",
             f"- gamma (X46, role only): {gam}",
             f"- q's metrics vs layer 1 seed_* columns: max |difference| {qdev:.2e}",
             f"- A_d and A_role J_tf vs deproles_scores.csv: {n} matcher-scenario cells in both; max |difference| "
             f"{max(jd) if jd else float('nan'):.2e}; cells with |difference| > 1e-9: {sum(v > 1e-9 for v in jd)}"]
    (out / "x46_fullmetrics_checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
