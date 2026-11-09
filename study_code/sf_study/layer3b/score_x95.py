"""X95 scorer (EXPLORATORY; pre-registration layer3b/X95_PREREGISTRATION.md, frozen; addendum in EXPLORATORY.md X95). Written
2026-10-08 before any stage-2 output.

Per scenario and base: q (base matrix, clipped at 0), L (raw literal term P^2 sigma0_lit, graph-only export), D (datatype_nanfix),
R13 (x95_roles.r13), V (x95_roles.v_term, char_distribution; single-table corpora only).
Arms: base = q; +L = q + 0.3 L; +D = +L + w_D D; +R = +D + w_R R13; full = +R + w_V V (w_V only for name-based bases).
Comparators: SF converged and SF(t2) on the same base; for base (iii) F_std = explore_composite.fuse(SF_std(coma), [char3_soft,
cupid, coma_instances], 0.35); X52's frozen plain composite = unit-mean(qgram2_dice, composite_dominants, composite_average)
+ 1 D + 0.1 eq_shape + 0.2 R6 (f7 value shapes; explore_deproles roles).

    python layer3b/score_x95.py --validate   # Valentine only: F_std and X52's composite against results/plain/composite_scores.csv
    python layer3b/score_x95.py --fit        # fitting set only (Valentine screen + T2D) -> tuning/results/x95/weights.json
    python layer3b/score_x95.py --score      # everything, with the frozen weights -> tuning/results/x95/{rows.csv, contrasts.csv, report.md}
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
for p in (STUDY.parent / "harness", STUDY / "tuning", STUDY, STUDY / "reference", HERE):
    sys.path.insert(0, str(p))
import score_x93 as s93  # noqa: E402
import x95_roles as xr  # noqa: E402

RES = STUDY / "tuning" / "results"
OUT = RES / "x95"
BASES = STUDY / "data" / "x95_bases"
GRID = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
W_L = 0.3
SEED, B, MIN_CL = 20261009, 10_000, 5
PANEL = ("coma", "containment", "cupid", "distinct_count", "jaro_winkler", "jsd_values", "kang_naughton", "ks", "levenshtein",
         "pmf", "prefix_suffix", "qgram3_jaccard", "role_gcov", "sbert_mpnet", "unary_statistics_invariant")
NAME_BASED = {"coma", "cupid", "jaro_winkler", "levenshtein", "prefix_suffix", "qgram3_jaccard", "sbert_mpnet", "ii", "iii"}
AVG = {"ii": ("qgram2_dice", "composite_dominants", "composite_average"), "iii": ("coma", "char3_soft", "cupid")}
HDXSM_CAP = {f"pair_{i}" for i in (43, 47, 50, 52, 53, 55)}
# corpus: (converged stage of base (i), t2 stage of base (i), SF_std(coma) folder for F_std, single-table)
STAGES = {
    "valentine_plain": ("layer1_plain", "iter_plain", STUDY / "results" / "plain" / "coma", True),
    "valentine_md5": ("layer1_md5", "iter_md5", None, True),
    "t2d": ("e2_t2d", "iter_e2_t2d", STUDY / "results" / "t2d" / "coma", True),
    "ctl": ("e2_ctl", "iter_e2_ctl", STUDY / "results" / "ctl" / "coma", True),
    "final_controlled": ("final_controlled", "iter_final_controlled", RES / "x95sf_final_controlled" / "coma__std", True),
    "goby": ("x81_goby", "iter_x81_goby", RES / "x95sf_goby" / "coma__std", True),
    "hdxsm": ("x81_hdxsm", "iter_x81_hdxsm", RES / "x95sf_hdxsm" / "coma__std", True),
    "gdcsm": ("x81_gdcsm", "iter_x81_gdcsm", RES / "x95sf_gdcsm" / "coma__std", True),
    "madi": ("x81_madi", "iter_x81_madi", RES / "x95sf_madi" / "coma__std", True),
    "final_indep": ("final_indep", "iter_final_indep", None, False),
    "mimic": ("x81_mimic", "iter_x81_mimic", None, False),
}
FIT = ("valentine_plain", "t2d")
# rerun 549 (2026-11-02): under SMX_VALENTINE=549 readable and recoded Valentine are scored on the 549 scenarios: the 381
# under their corpus keys with the role terms (R13, X52's roles) on the fixed FD profiles (harness/generated_hyfdfix) and
# the standard-configuration comparator from std_p382fix; the 168 under valentine_pnew / valentine_mnew (new trees, pnew /
# mnew seeds, layer1_<new> / iter_<new>, x95sf_valentine_pnew bases, the seed study's pnew coma run as SF std),
# relabelled valentine_plain / valentine_md5. Every other corpus keeps its stored rows and contrasts. Output x95_549/.
import os  # noqa: E402
V549 = os.environ.get("SMX_VALENTINE", "382") == "549"
OUT549 = RES / "x95_549"
if V549:
    _GEN = STUDY.parent / "harness" / "generated"
    _FIX = STUDY.parent / "harness" / "generated_hyfdfix"
    xr.ROOTS["valentine_plain"] = (_FIX / "valentine_plain", "plain")
    xr.ROOTS["valentine_md5"] = (_FIX / "valentine_md5", "md5")
    xr.ROOTS["valentine_pnew"] = (_GEN / "valentine-new168", "pnew")
    xr.ROOTS["valentine_mnew"] = (_GEN / "valentine-new168-md5", "mnew")
    STAGES["valentine_plain"] = ("layer1_plain", "iter_plain", RES / "std_p382fix" / "coma__std", True)
    STAGES["valentine_pnew"] = ("layer1_pnew", "iter_pnew", STUDY / "results" / "pnew" / "coma", True)
    STAGES["valentine_mnew"] = ("layer1_mnew", "iter_mnew", None, True)
    s93.CORP.setdefault("valentine_pnew", ("pnew", "pnew", "layer1_pnew", "iter_pnew", "valentine"))
    s93.CORP.setdefault("valentine_mnew", ("mnew", "mnew", "layer1_mnew", "iter_mnew", "valentine"))
PRIMARY = ("goby", "hdxsm", "gdcsm")


def truth_root(corpus):
    import score_seeds
    r = s93.CORP.get(corpus, (None, None))[1] if corpus in s93.CORP else xr.ROOTS[corpus][0]
    return score_seeds.CORPUS[r] if isinstance(r, str) else r


def scenarios(corpus):
    if corpus in s93.CORP:
        return s93.scenarios(corpus)
    return xr.scenarios(xr.ROOTS[corpus][0])


def load_cell(corpus, ds, sc):
    """Everything a scenario needs, or None. Matrices aligned to the datatype matrix's labels."""
    import score_seeds
    from explore_composite import load
    from score_screen_metrics import permutations
    from smx.adapters import read_matrix
    root, seed = xr.ROOTS[corpus]
    seeds = STUDY / "reference" / "seeds" / seed
    g = score_seeds.ground_truth_file(truth_root(corpus), ds, sc)
    fD = seeds / "datatype_nanfix" / ds / sc / "seed.csv"
    if g is None or not fD.is_file():
        return None
    D, rows, cols = read_matrix(fD)
    rows, cols = list(rows), list(cols)
    gt = read_matrix(g)[0] > 0.5
    if D.shape != gt.shape or not np.isfinite(D).all():
        return None
    L = s93.l_matrix(corpus, ds, sc, gt.shape)              # L~ (scaled); the raw L below
    dexp = s93.EXPORT / corpus / ds / sc
    if L is None:
        return None
    Lraw, nodes = s93.literal_term(dexp)
    Lr = np.zeros(gt.shape)
    Lr[nodes.row.to_numpy(), nodes.col.to_numpy()] = Lraw
    cell = {"corpus": corpus, "dataset": ds, "scenario": sc, "gt": gt, "perms": permutations(gt.shape, f"{ds}/{sc}"), "D": D,
            "L": Lr, "R": xr.r13(root, ds, sc, rows, cols), "rows": rows, "cols": cols, "q": {}}
    single = STAGES[corpus][3]
    cell["V"] = xr.v_term(STUDY / "reference" / "seeds" / f"self_{seed}", "char_distribution", ds, sc, rows, cols) if single else None
    for m in PANEL + tuple(c for v in AVG.values() for c in v) + ("coma_instances",):
        M = load(seeds / m / ds / sc / "seed.csv", rows, cols)
        if M is not None and M.shape == gt.shape:
            cell["q"][m] = M
    for b in AVG:
        M = load(BASES / b / corpus / ds / sc / "seed.csv", rows, cols)
        if M is not None:
            cell["q"][b] = M
    return cell


def arms(cell, base, w):
    q = np.clip(cell["q"][base], 0, None)
    out = {"base": q, "+L": q + W_L * cell["L"]}
    out["+D"] = out["+L"] + w["D"] * cell["D"]
    out["+R"] = out["+D"] + w["R"] * cell["R"]
    out["full"] = out["+R"] + (w["V"] * cell["V"] if base in NAME_BASED and cell["V"] is not None else 0.0)
    return out


def sf(stage_dir: Path, label: str, cell):
    from explore_composite import load, sf_out
    from score_r7 import run_dir
    d = stage_dir / label if label else stage_dir
    rd = run_dir(d) if d.is_dir() else None
    return load(sf_out(rd, cell["dataset"], cell["scenario"]), cell["rows"], cell["cols"]) if rd else None


def x52_composite(cell):
    import f7
    from explore_composite import unit
    from explore_deproles import roles
    if not all(m in cell["q"] for m in AVG["ii"]):
        return None
    root = xr.ROOTS[cell["corpus"]][0]
    cl = {}
    for side in ("source", "target"):
        for t in sorted((root / cell["dataset"] / cell["scenario"] / side).glob("*.csv")):
            h, k = f7.column_classes(t)
            cl.update({f"{t.stem}.{c}": x for c, x in zip(h, k)})
    rc, cc = [cl.get(r) for r in cell["rows"]], [cl.get(c) for c in cell["cols"]]
    eq = np.array([[1.0 if a is not None and a == b else 0.0 for b in cc] for a in rc])
    Fs, _ = roles(cell["rows"], *xr.profiles_at(root, cell["dataset"], cell["scenario"], "source"))
    Ft, _ = roles(cell["cols"], *xr.profiles_at(root, cell["dataset"], cell["scenario"], "target"))
    R6 = 1.0 - np.abs(Fs[:, None, :] - Ft[None, :, :]).mean(axis=2)
    base = np.mean([unit(cell["q"][m]) for m in AVG["ii"]], axis=0)
    return base + 1.0 * cell["D"] + 0.1 * eq + 0.2 * R6


def f_std(cell):
    from explore_composite import fuse
    sd = STAGES[cell["corpus"]][2]
    if sd is None or not all(m in cell["q"] for m in ("char3_soft", "cupid", "coma_instances")):
        return None
    S = sf(sd, "", cell)
    return fuse(S, [cell["q"][m] for m in ("char3_soft", "cupid", "coma_instances")]) if S is not None else None


def J(M, cell):
    from score_r7 import metrics
    return metrics(M, cell["gt"], cell["perms"])["J"]


def score_job(args):
    corpus, ds, sc, w = args
    cell = load_cell(corpus, ds, sc)
    if cell is None:
        return []
    conv, t2, _, _ = STAGES[corpus]
    out = []
    for base in PANEL + tuple(AVG):
        if base not in cell["q"]:
            continue
        wb = w["i"] if base in PANEL else w[base]
        row = {"corpus": corpus, "dataset": ds, "scenario": sc, "matcher": base, "family": "i" if base in PANEL else base}
        for k, M in arms(cell, base, wb).items():
            row[k] = J(M, cell)
        if base in PANEL:
            S1, S2 = sf(RES / conv, f"{base}__melnik", cell), sf(RES / t2, f"{base}__melnik_t2", cell)
        else:
            S1, S2 = sf(RES / f"x95sf_{corpus}", f"{base}__melnik", cell), sf(RES / f"x95sf_{corpus}", f"{base}__melnik_t2", cell)
        row["SF_conv"] = J(S1, cell) if S1 is not None else np.nan
        row["SF_t2"] = J(S2, cell) if S2 is not None else np.nan
        if base == "iii":
            F = f_std(cell)
            row["F_std"] = J(F, cell) if F is not None else np.nan
        if base == "ii":
            C = x52_composite(cell)
            row["X52"] = J(C, cell) if C is not None else np.nan
        out.append(row)
    return out


# ---------------------------------------------------------------------------------------------------------------- validate
def validate() -> int:
    """F_std and X52's frozen composite on Valentine plain against X52's stored per-scenario scores."""
    sys.stdout.reconfigure(encoding="utf-8")
    ref = pd.read_csv(STUDY / "results" / "plain" / "composite_scores.csv").set_index(["dataset", "scenario"])
    col = "A|composite_dominants+qgram2_dice+composite_average|1.0|shape|0.1|0.2"
    dif_f, dif_c, n = [], [], 0
    for ds, sc in scenarios("valentine_plain")[::8]:
        cell = load_cell("valentine_plain", ds, sc)
        if cell is None or (ds, sc) not in ref.index:
            continue
        F, C = f_std(cell), x52_composite(cell)
        if F is not None:
            dif_f.append(abs(J(F, cell) - ref.loc[(ds, sc), "F_std"]))
        if C is not None and col in ref.columns:
            dif_c.append(abs(J(C, cell) - ref.loc[(ds, sc), col]))
        n += 1
    lines = ["# X95 scorer validation (Valentine plain, every 8th scenario)", "",
             f"- scenarios: {n}",
             f"- F_std vs X52's stored F_std: max |ΔJ| {max(dif_f) if dif_f else float('nan'):.2e} ({len(dif_f)} compared)",
             f"- X52 frozen composite vs X52's stored column: max |ΔJ| {max(dif_c) if dif_c else float('nan'):.2e} ({len(dif_c)} compared)"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# --------------------------------------------------------------------------------------------------------------------- fit
def _fit_cells(args):
    corpus, ds, sc = args
    cell = load_cell(corpus, ds, sc)
    if cell is None:
        return None
    for k in ("rows", "cols"):
        cell.pop(k)
    return cell


def _eval(args):
    cell, base, w = args
    return J(arms(cell, base, w)["full"], cell)


def fit() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    scr = s93.screen_cases()
    jobs = [("valentine_plain", ds, sc) for ds, sc in scenarios("valentine_plain") if (ds, sc) in scr]
    jobs += [("t2d", ds, sc) for ds, sc in scenarios("t2d")]
    with Pool(10) as pool:
        cells = [c for c in pool.map(_fit_cells, jobs, chunksize=4) if c is not None]
        weights = {}
        for fam, members in (("i", PANEL), ("ii", ("ii",)), ("iii", ("iii",))):
            w = {"D": 0.0, "R": 0.0, "V": 0.0}

            def objective(wt):
                tasks = [(c, b, wt) for c in cells for b in members if b in c["q"]]
                vals = pool.map(_eval, tasks, chunksize=16)
                df = pd.DataFrame([(t[0]["corpus"], t[0]["dataset"], t[0]["scenario"], v) for t, v in zip(tasks, vals)],
                                  columns=["corpus", "dataset", "scenario", "J"])
                per = df.groupby(["corpus", "dataset", "scenario"]).J.mean().groupby(level="corpus").mean()
                return float(per.mean())                      # Valentine screen and T2D weighted equally
            curve = {}
            for _ in range(2):
                for k in ("D", "R", "V"):
                    vals = []
                    for g in GRID:
                        wt = dict(w, **{k: g})
                        vals.append(objective(wt))
                    best = max(vals)
                    w[k] = GRID[vals.index(best)]           # ties -> the smaller weight (first in the grid)
                    curve[f"{k}"] = dict(zip(map(str, GRID), vals))
            weights[fam] = {**w, "objective": objective(w), "last_pass_curves": curve}
    weights.update({"w_L": W_L, "fitting": "60 Valentine readable screen scenarios + T2D (175), equally weighted",
                    "cells": len(cells)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weights.json").write_text(json.dumps(weights, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "last_pass_curves"})
                      for k, v in weights.items()}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------------------- score
def score() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    import x86_cluster_ci as x86
    wj = json.loads((OUT / "weights.json").read_text(encoding="utf-8"))
    w = {f: {k: wj[f][k] for k in ("D", "R", "V")} for f in ("i", "ii", "iii")}
    scr = s93.screen_cases()
    jobs = []
    for c in STAGES:
        for ds, sc in scenarios(c):
            if c == "valentine_plain" and (ds, sc) in scr:
                continue
            if c == "hdxsm" and sc in HDXSM_CAP:
                continue
            jobs.append((c, ds, sc, w))
    with Pool(10) as pool:
        df = pd.DataFrame([r for rr in pool.imap_unordered(score_job, jobs, chunksize=2) for r in rr])
    m = df.corpus == "madi"
    df.loc[m, "corpus"] = np.where(df.loc[m, "scenario"].str.startswith("products__"), "madi_products", "madi_indep")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "rows.csv", index=False)
    props = pd.read_csv(STUDY / "results" / "benchmark_properties.csv")
    src = props[props.corpus == "valentine_plain"].set_index(["dataset", "scenario"]).source_id.to_dict()
    rng = np.random.default_rng(SEED)
    contrasts = [("full", "SF_conv"), ("full", "SF_t2"), ("full", "+L"), ("+D", "+L"), ("+R", "+D"), ("full", "+R"),
                 ("+L", "SF_t2"), ("full", "F_std"), ("full", "X52"), ("full", "base"), ("SF_conv", "base")]
    res = []
    for (corpus, fam), g in df.groupby(["corpus", "family"]):
        for a, b in contrasts:
            if a not in g or b not in g:
                continue
            v = (g[a] - g[b])
            ok = v.notna()
            if not ok.any():
                continue
            gg = g[ok]
            # base family (i): only matchers complete on every scenario of the contrast
            if fam == "i":
                cnt = gg.groupby("matcher").size()
                gg = gg[gg.matcher.isin(cnt[cnt == cnt.max()].index)]
            per = (gg[a] - gg[b]).groupby([gg.dataset, gg.scenario]).mean()
            keys = list(per.index)
            if corpus.startswith("valentine"):
                cl = np.array([src.get(k, k[0]) for k in keys])
            else:
                kk = [k[1] if corpus.startswith("madi") else f"{k[0]}/{k[1]}" for k in keys]
                cmap, _ = x86.cluster_map(corpus if corpus in ("goby", "hdxsm", "gdcsm", "madi_indep", "madi_products",
                                                               "final_indep", "final_controlled", "mimic") else "other", kk)
                cl = np.array([cmap[k] for k in kk])
            ids = np.unique(cl)
            x = per.to_numpy()
            lo = hi = np.nan
            if len(ids) >= MIN_CL:
                sums = np.array([x[cl == i].sum() for i in ids]); cnt_ = np.array([(cl == i).sum() for i in ids])
                idx = rng.integers(0, len(ids), size=(B, len(ids)))
                lo, hi = np.percentile(sums[idx].sum(axis=1) / cnt_[idx].sum(axis=1), [2.5, 97.5])
            res.append({"corpus": corpus, "family": fam, "contrast": f"{a} − {b}", "estimate": float(x.mean()), "lo": lo,
                        "hi": hi, "scenarios": len(x), "clusters": len(ids), "bases": int(gg.matcher.nunique())})
    r = pd.DataFrame(res)
    r.to_csv(OUT / "contrasts.csv", index=False)
    write_report(r, w, OUT)
    return 0


def contrasts_of(df, src, rng) -> "pd.DataFrame":
    """score()'s contrast loop, unchanged (moved into a function 2026-11-02 for the 549 run)."""
    import x86_cluster_ci as x86
    contrasts = [("full", "SF_conv"), ("full", "SF_t2"), ("full", "+L"), ("+D", "+L"), ("+R", "+D"), ("full", "+R"),
                 ("+L", "SF_t2"), ("full", "F_std"), ("full", "X52"), ("full", "base"), ("SF_conv", "base")]
    res = []
    for (corpus, fam), g in df.groupby(["corpus", "family"]):
        for a, b in contrasts:
            if a not in g or b not in g:
                continue
            v = (g[a] - g[b])
            ok = v.notna()
            if not ok.any():
                continue
            gg = g[ok]
            if fam == "i":
                cnt = gg.groupby("matcher").size()
                gg = gg[gg.matcher.isin(cnt[cnt == cnt.max()].index)]
            per = (gg[a] - gg[b]).groupby([gg.dataset, gg.scenario]).mean()
            keys = list(per.index)
            if corpus.startswith("valentine"):
                cl = np.array([src.get(k, k[0]) for k in keys])
            else:
                kk = [k[1] if corpus.startswith("madi") else f"{k[0]}/{k[1]}" for k in keys]
                cmap, _ = x86.cluster_map(corpus if corpus in ("goby", "hdxsm", "gdcsm", "madi_indep", "madi_products",
                                                               "final_indep", "final_controlled", "mimic") else "other", kk)
                cl = np.array([cmap[k] for k in kk])
            ids = np.unique(cl)
            x = per.to_numpy()
            lo = hi = np.nan
            if len(ids) >= MIN_CL:
                sums = np.array([x[cl == i].sum() for i in ids]); cnt_ = np.array([(cl == i).sum() for i in ids])
                idx = rng.integers(0, len(ids), size=(B, len(ids)))
                lo, hi = np.percentile(sums[idx].sum(axis=1) / cnt_[idx].sum(axis=1), [2.5, 97.5])
            res.append({"corpus": corpus, "family": fam, "contrast": f"{a} − {b}", "estimate": float(x.mean()), "lo": lo,
                        "hi": hi, "scenarios": len(x), "clusters": len(ids), "bases": int(gg.matcher.nunique())})
    return pd.DataFrame(res)


def write_report(r, w, OUT: Path) -> None:
    """score()'s report, unchanged (moved into a function 2026-11-02 for the 549 run)."""
    lines = ["# X95: a fixed-weight one-step matcher against SF (EXPLORATORY, pre-registered)", "",
             f"- frozen weights: {json.dumps(w)}; w_L = {W_L}", "",
             "| corpus | base family | contrast | estimate [cluster 95% CI] | scenarios / clusters | bases |", "|---|---|---|---|---|---|"]
    for _, x in r.iterrows():
        ci = f"[{x.lo:+.4f}, {x.hi:+.4f}]" if np.isfinite(x.lo) else "–"
        lines.append(f"| {x.corpus} | {x.family} | {x.contrast} | {x.estimate:+.4f} {ci} | {x.scenarios} / {x.clusters} | {x.bases} |")
    prim = r[(r.family == "i") & (r.contrast == "full − SF_conv") & r.corpus.isin(PRIMARY)]
    ni = {c: bool(np.isfinite(lo) and lo > -0.01) for c, lo in zip(prim.corpus, prim.lo)}
    sup = {c: bool(np.isfinite(lo) and lo > 0) for c, lo in zip(prim.corpus, prim.lo)}
    lines += ["", "## Primary (base family (i), full − SF converged)", "",
              f"- non-inferior (lower bound > −0.01): {ni}", f"- superior (lower bound > 0): {sup}",
              f"- **decision: {'non-inferior on all three' if len(ni) == 3 and all(ni.values()) else 'not non-inferior everywhere'}**"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def score549() -> int:
    """SMX_VALENTINE=549: see the note at STAGES. Valentine rows recomputed on the 549 (frozen weights.json, not refit; the
    60 screen scenarios stay out of valentine_plain); the other corpora's rows and contrasts are the stored ones."""
    sys.stdout.reconfigure(encoding="utf-8")
    import valentine549 as v
    from explore_benchmark_properties import source_id
    if not (RES / "x95sf_valentine_pnew" / "COLLECTED").is_file():
        print("x95 549: pending (x95sf_valentine_pnew not collected)")
        return 0
    wj = json.loads((OUT / "weights.json").read_text(encoding="utf-8"))
    w = {f: {k: wj[f][k] for k in ("D", "R", "V")} for f in ("i", "ii", "iii")}
    scr = s93.screen_cases()
    jobs = []
    for ds, sc, _ in v.cases549():
        new = v.is_new(ds, sc)
        if not new and (ds, sc) in scr:
            jobs.append(("valentine_md5", ds, sc, w))
            continue
        jobs += [("valentine_pnew" if new else "valentine_plain", ds, sc, w), ("valentine_mnew" if new else "valentine_md5", ds, sc, w)]
    with Pool(8) as pool:
        val = pd.DataFrame([r for rr in pool.imap_unordered(score_job, jobs, chunksize=2) for r in rr])
    val["corpus"] = val.corpus.replace({"valentine_pnew": "valentine_plain", "valentine_mnew": "valentine_md5"})
    old = pd.read_csv(OUT / "rows.csv")
    df = pd.concat([old[~old.corpus.str.startswith("valentine")], val], ignore_index=True)
    OUT549.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT549 / "rows.csv", index=False)
    props = pd.read_csv(STUDY / "results" / "benchmark_properties.csv")
    src = props[props.corpus == "valentine_plain"].set_index(["dataset", "scenario"]).source_id.to_dict()
    src.update({(ds, sc): source_id("valentine_plain", ds, sc) for ds, sc, _ in v.cases549() if v.is_new(ds, sc)})
    new_c = contrasts_of(val, src, np.random.default_rng(SEED))
    old_c = pd.read_csv(OUT / "contrasts.csv")
    r = pd.concat([old_c[~old_c.corpus.str.startswith("valentine")], new_c], ignore_index=True)
    r.to_csv(OUT549 / "contrasts.csv", index=False)
    write_report(r, w, OUT549)
    with (OUT549 / "report.md").open("a", encoding="utf-8") as fh:
        fh.write("\n549 run: Valentine recomputed on the 549 scenarios (role terms on the fixed FD profiles; valentine_md5 "
                 "of the 168 without the value-shape term V, whose within-table matrices (self_mnew) do not exist); the "
                 "other corpora's rows and contrasts are the stored x95 ones (their +R / full arms use the old FD "
                 "profiles; only the FD-free contrasts of those corpora feed macros).\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true")
    g.add_argument("--fit", action="store_true")
    g.add_argument("--score", action="store_true")
    a = ap.parse_args()
    raise SystemExit(validate() if a.validate else fit() if a.fit else (score549() if V549 else score()))
