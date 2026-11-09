"""EXPLORATORY X82 (2026-10-07): (f) RQ1 with the wide matcher panel and (b) the iteration trajectory on the new
corpora. Predictions in EXPLORATORY.md X82 (2026-10-06 22:27), written before any X82 run.

(f) Panel = wide_<c>/<m>__melnik plus the base stage's <m>__melnik (x81_<c>, or <c> itself for final_*); controls and
    oracles excluded (score_r7.is_control). A matcher is included on a corpus if q and SF(melnik) are finite on every scenario.
    Per matcher: macro over groups of the per-scenario R_exp of q and of the gain R_exp(SF) − R_exp(q).
    ρ = Spearman over matchers; 95% interval by resampling matchers (2,000, as score_final.f3).
    Pooled: per matcher, the mean over the six single-table corpora of its per-corpus values (matchers included on all six).
(b) iter_<stage>/<m>__{melnik,fd}_t{1,2,3,5,10}; converged = the base stage's <m>__<arm>. Included: non-control matchers with
    every t and the converged run on every scenario. Per scenario, the mean over included matchers of gain(t) = R_exp(SF_t) − R_exp(q)
    (and J_tf); macro over groups. Share(t) = gain(t) / gain(converged), with a 95% interval from bootstrapping scenarios within groups.
Only collected runs are read: a label folder must contain merged/.
Groups: final_controlled by database, split into _plain/_opaque corpora; madi by domain (products separate); final_indep by
domain (bib, music); others one group.

    ../harness/.venv-bdi/Scripts/python.exe explore_x82_bf.py [--corpora ...] [--workers 10]
Output: results/x82/{rq1_rows.csv, iter_rows.csv, report.md}
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "harness", HERE / "tuning", HERE, HERE / "reference", HERE / "layer3b"):
    sys.path.insert(0, str(p))
from score_final import spearman  # noqa: E402

GEN = HERE.parent / "harness" / "generated"
RES = HERE / "tuning" / "results"
ROOTS = {"final_indep": HERE / "data" / "final_indep", "final_controlled": GEN / "final-controlled",
         "hdxsm": GEN / "hdxsm-usable", "gdcsm": GEN / "gdcsm-imported", "madi": GEN / "madi-imported",
         "goby": GEN / "goby-imported", "mimic": HERE / "data" / "mimic_omop"}
SINGLE = ["final_controlled_plain", "final_controlled_opaque", "hdxsm", "gdcsm", "goby", "madi_indep"]
TS = (1, 2, 3, 5, 10)
OUT = HERE / "results" / "x82"
sys.path.insert(0, str(HERE / "reference"))
import valentine549 as V  # noqa: E402  (HyFD fix: used only under SMX_VALENTINE=549)
# SMX_VALENTINE=549 (fixed HyFD profiles): FD arms (fd, fd_t*) only from the *fix reruns (x81fix_<c>, iterx81fix_<c>; none
# is named for final_*), never from the old stages; the melnik arms (part (f) and the melnik trajectory) are unchanged.
# Output results/x82fix/.
if V.HYFDFIX:
    OUT = HERE / "results" / "x82fix"
SEED = 20261082


def base_stage(c: str) -> str:
    return c if c.startswith("final_") else f"x81_{c}"


def label_of(c: str, ds: str, sc: str) -> tuple[str, str]:
    """(reported corpus, group)."""
    if c == "final_controlled":
        return f"final_controlled_{'plain' if sc.endswith('_plain') else 'opaque'}", ds.split("-")[0]
    if c == "madi":
        d = sc.split("__")[0]
        return ("madi_products" if d == "products" else "madi_indep"), d
    if c == "final_indep":
        return c, ds.split("_")[0]
    return c, c


def collected(d: Path) -> Path | None:
    return d / "merged" if (d / "merged").is_dir() else None


def runs(c: str) -> dict[tuple[str, str], Path]:
    """(matcher, arm) -> merged folder, for melnik (wide + base), fd (base) and the iteration arms."""
    out = {}
    for stage, arms in ((f"wide_{c}", ("melnik",)), (base_stage(c), ("melnik", "fd")),
                        (f"iter_{base_stage(c)}", tuple(f"{a}_t{t}" for a in ("melnik", "fd") for t in TS))):
        if V.HYFDFIX:   # FD arms from the fixed-FD rerun of the stage (if collected), non-FD arms from the stage
            groups = [(stage, tuple(x for x in arms if not V.is_fd_arm(x)))]
            if V.fix_ready(stage):
                groups.append((V.fix_stage(stage), tuple(x for x in arms if V.is_fd_arm(x))))
        else:
            groups = [(stage, arms)]
        for stg, arms_ in groups:
            root = RES / stg
            if not root.is_dir():
                continue
            for d in root.iterdir():
                if d.is_dir() and "__" in d.name:
                    m, arm = d.name.split("__", 1)
                    r = collected(d)
                    if arm in arms_ and r is not None and (m, arm) not in out:
                        out[(m, arm)] = r
    return out


def job(args):
    c, ds, sc, rd = args
    import score_seeds
    from explore_composite import load, sf_out
    from score_r7 import is_control
    from score_screen_metrics import all_metrics, permutations
    from smx.adapters import read_matrix
    g = score_seeds.ground_truth_file(ROOTS[c], ds, sc)
    f_d = HERE / "reference" / "seeds" / c / "datatype_nanfix" / ds / sc / "seed.csv"
    if g is None or not f_d.is_file():
        return []
    _, rows, cols = read_matrix(f_d)
    rows, cols = list(rows), list(cols)
    gt = read_matrix(g)[0] > 0.5
    if gt.shape != (len(rows), len(cols)) or not gt.any():
        return []
    perms = permutations(gt.shape, f"{ds}/{sc}")
    rep, grp = label_of(c, ds, sc)
    out = []
    for m in sorted({m for m, _ in rd}):
        if is_control(m):
            continue
        q = load(HERE / "reference" / "seeds" / c / m / ds / sc / "seed.csv", rows, cols)
        if q is None:
            continue
        r = all_metrics(np.clip(q, 0, None), gt, perms)
        row = {"corpus": rep, "group": grp, "dataset": ds, "scenario": sc, "matcher": m,
               "q_R": r["R_exp"], "q_J": (r["R_exp"] + r["F1_Hr"]) / 2}
        for (mm, arm), path in rd.items():
            if mm != m:
                continue
            S = load(sf_out(path, ds, sc), rows, cols)
            if S is None:
                continue
            r = all_metrics(np.clip(S, 0, None), gt, perms)
            row[f"{arm}_R"], row[f"{arm}_J"] = r["R_exp"], (r["R_exp"] + r["F1_Hr"]) / 2
        out.append(row)
    return out


def per_matcher(rows: list[dict]) -> dict[str, tuple[float, float]]:
    """matcher -> (macro q R_exp, macro melnik gain R_exp); only matchers with both on every scenario."""
    scen = {(r["dataset"], r["scenario"]) for r in rows}
    by = defaultdict(list)
    for r in rows:
        if "melnik_R" in r:
            by[r["matcher"]].append(r)
    out = {}
    for m, rs in by.items():
        if {(r["dataset"], r["scenario"]) for r in rs} != scen:
            continue
        gq, gg = defaultdict(list), defaultdict(list)
        for r in rs:
            gq[r["group"]].append(r["q_R"])
            gg[r["group"]].append(r["melnik_R"] - r["q_R"])
        out[m] = (float(np.mean([np.mean(v) for v in gq.values()])), float(np.mean([np.mean(v) for v in gg.values()])))
    return out


def rho_ci(pm: dict, rng) -> tuple[float, float, float, int]:
    ms = sorted(pm)
    x, y = np.array([pm[m][0] for m in ms]), np.array([pm[m][1] for m in ms])
    rho = spearman(x, y)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(x), len(x))
        if len(set(i)) > 2:
            bs.append(spearman(x[i], y[i]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return rho, float(lo), float(hi), len(ms)


def trajectory(rows: list[dict], arm: str, rng):
    """Per t: macro gain (R_exp, J) over q, and the share of the converged gain (R_exp) with a bootstrap interval."""
    keys = [f"{arm}_t{t}" for t in TS] + [arm]
    # the scenarios the arm ran on (HDXSM's FD-edge arms: the 193 pairs below the FD cap)
    scen = {(r["dataset"], r["scenario"]) for r in rows if f"{arm}_R" in r}
    rows = [r for r in rows if (r["dataset"], r["scenario"]) in scen]
    by = defaultdict(list)
    for r in rows:
        by[r["matcher"]].append(r)
    inc = [m for m, rs in by.items() if {(r["dataset"], r["scenario"]) for r in rs if all(f"{k}_R" in r for k in keys)} == scen]
    if not inc:
        return None, 0
    cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["matcher"] in inc:
            for k in keys:
                cell[(r["group"], r["dataset"], r["scenario"])][k].append((r[f"{k}_R"] - r["q_R"], r[f"{k}_J"] - r["q_J"]))
    groups = defaultdict(list)
    for (g, _, _), v in cell.items():
        groups[g].append({k: np.mean(np.array(x), axis=0) for k, x in v.items()})

    def macro(gs, k, i):
        return float(np.mean([np.mean([s[k][i] for s in v]) for v in gs.values()]))

    est = {k: (macro(groups, k, 0), macro(groups, k, 1)) for k in keys}
    shares = {}
    for t in TS:
        k = f"{arm}_t{t}"
        bs = []
        for _ in range(2000):
            res = {g: [v[j] for j in rng.integers(0, len(v), len(v))] for g, v in groups.items()}
            den = macro(res, arm, 0)
            if abs(den) > 1e-9:
                bs.append(macro(res, k, 0) / den)
        den = est[arm][0]
        shares[t] = (est[k][0] / den if abs(den) > 1e-9 else float("nan"), *np.nanpercentile(bs, [2.5, 97.5])) if bs else None
    return (est, shares), len(inc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="final_controlled,gdcsm,goby,madi,mimic,final_indep,hdxsm")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for c in a.corpora.split(","):
        rd = runs(c)
        if not rd:
            print(f"skip {c}: no collected runs")
            continue
        jobs = [(c, d.name, s.name, rd) for d in sorted(p for p in ROOTS[c].iterdir() if p.is_dir())
                for s in sorted(p for p in d.iterdir() if p.is_dir())]
        with Pool(a.workers) as pool:
            for rr in pool.imap_unordered(job, jobs, chunksize=2):
                rows += rr
        print(f"{c}: {len(jobs)} scenarios, {len({m for m, _ in rd})} matchers with runs")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    corpora = sorted({r["corpus"] for r in rows})
    lines = ["# X82 (B): RQ1 with the wide panel (f) and the iteration trajectory (b) on the new corpora (EXPLORATORY)", "",
             "Predictions: EXPLORATORY.md X82, B 2026-10-06 22:27. R_exp = expected Recall@GT; gain = SF − q.", "",
             "## (f) Spearman ρ over matchers of (R_exp of q, R_exp gain of SF melnik)", "",
             "| corpus | matchers | ρ [95%] | mean gain R_exp |", "|---|---|---|---|"]
    pms = {}
    for c in corpora:
        pm = per_matcher([r for r in rows if r["corpus"] == c])
        pms[c] = pm
        if len(pm) < 5:
            lines.append(f"| {c} | {len(pm)} | – | – |")
            continue
        rho, lo, hi, n = rho_ci(pm, np.random.default_rng(SEED))   # per corpus: independent of corpus order
        lines.append(f"| {c} | {n} | {rho:+.2f} [{lo:+.2f}, {hi:+.2f}] | {np.mean([v[1] for v in pm.values()]):+.4f} |")
    have = [c for c in SINGLE if c in pms and pms[c]]
    common = set.intersection(*(set(pms[c]) for c in have)) if have else set()
    if len(common) >= 5:
        pooled = {m: (float(np.mean([pms[c][m][0] for c in have])), float(np.mean([pms[c][m][1] for c in have]))) for m in common}
        rho, lo, hi, n = rho_ci(pooled, np.random.default_rng(SEED))
        neg = sum(1 for c in have if len(pms[c]) >= 5 and rho_ci(pms[c], np.random.default_rng(SEED))[0] < 0)
        lines += ["", f"Pooled over {len(have)} single-table corpora ({', '.join(have)}): {n} matchers, "
                      f"ρ = {rho:+.2f} [{lo:+.2f}, {hi:+.2f}]; negative sign in {neg} of {len(have)} corpora."]
    lines += ["", "## (b) Trajectory: gain over q (R_exp / J_tf) after t updates and at convergence; share = gain(t)/gain(conv), R_exp", ""]
    for arm in ("melnik", "fd"):
        lines += [f"### {arm}", "", "| corpus | matchers | " + " | ".join(f"t{t}" for t in TS) + " | conv | share t2 [95%] | share t1 |",
                  "|" + "---|" * (len(TS) + 5)]
        for c in corpora:
            res, n = trajectory([r for r in rows if r["corpus"] == c], arm, np.random.default_rng(SEED))
            if res is None:
                lines.append(f"| {c} | 0 |" + " – |" * (len(TS) + 3))
                continue
            est, sh = res
            cells = " | ".join(f"{est[f'{arm}_t{t}'][0]:+.3f}/{est[f'{arm}_t{t}'][1]:+.3f}" for t in TS)
            s2, s1 = sh.get(2), sh.get(1)
            fmt = lambda s: (f"{s[0]:.2f} [{s[1]:.2f}, {s[2]:.2f}]" if s else "–")  # noqa: E731
            lines.append(f"| {c} | {n} | {cells} | {est[arm][0]:+.3f}/{est[arm][1]:+.3f} | {fmt(s2)} | "
                         f"{(f'{s1[0]:.2f}' if s1 else '–')} |")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
