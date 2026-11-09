"""Review check (EXPLORATORY, 2026-10-29): two sensitivity analyses of the compressed ICDE draft, from cached scores only
(no matcher or SF runs).

B1  The RQ1 panel (Sec. IV) without the SemStruct reimplementation. The panel is gen_figures.py's: layer 1, arm melnik
    (SF as published), readable Valentine, the data-type matcher from its NaN-free rerun, non-control matchers on all
    382 scenarios. SemStruct-derived members: `semstruct` (FILE seed imported from the harness's
    semstruct_20260908 checkpoint by import_file_seeds.py). Its LM-only control `semstruct_lm_only` is never in the panel
    (CONTROL_ARMS, and load_rows drops it as a duplicate of `sbert_e5`). `sbert_e5` is an independent run of the same
    frozen encoder (PythonServerMatcher, intfloat/e5-small) whose scores coincide with the LM-only control on every
    scenario; it is not SemStruct-derived, but a stricter variant drops it too.
    The full panel is first reproduced against figures/rq1_numbers.tex (gen_figures.rq_numbers, imported, not modified)
    and figures/review_numbers.tex (rq1_artefact.py's split-half, headroom and ceiling-null computation, replicated here
    with the panel as a parameter; same seed and draw order, so the full panel reproduces it exactly). The script stops
    on any mismatch.

B2  Leave-one-source-out on readable Valentine. Source = X77's source_id (explore_benchmark_properties.py):
    <family prefix>/<first token of the scenario name>; 10 sources (ChEMBL/assays, OpenData/miller2, TPC/prospect,
    Wikidata/musicians, and six Magellan source tables; dblp_acm and dblp_scholar share Magellan/dblp).
    Contrasts, each the paper's statistic (mean over matchers of the macro over families with >= 5 scenarios):
      (i)   SF as published - initial matcher, R_exp, F1_Hr, J_tf = their mean (RQ1 panel, 76 matchers);
      (ii)  datatype term A_d = q+ + 5D - SF as published, J_tf (X46 panel, 75 matchers; the info-ladder row
            "Valentine readable" = -\drPlRefVsAdEst; per-scenario values: results/plain/deproles_scores.csv column
            A_0.0_0.0 and tuning/results/layer1_plain_scores/scores.csv arm melnik, as explore_deproles.py);
      (iii) Spearman rho over matchers between the initial matcher's R_exp and SF's change (R_exp, F1_Hr).
    Per source: the plain mean over its scenarios (one family per source). Leave one source out: the macro statistic
    on the remaining scenarios; families are re-checked for >= 5 scenarios after the drop (dropping TPC/prospect,
    ChEMBL/assays or OpenData/miller2 removes a whole family; dropping Wikidata/musicians changes nothing, since
    Wikidata has 2 scenarios and is never in the macro). Descriptive, not inference.

Outputs (tuning/results/review_checks/): source_sensitivity.md, source_sensitivity.json, valentine_source_map.csv,
source_sensitivity_per_source.csv.

    python review_checks/source_sensitivity.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY.parent / "latex" / "sf-evaluation"))
import gen_figures as gf  # noqa: E402

OUT = STUDY / "tuning" / "results" / "review_checks"
FIG_SHARED = STUDY.parent / "latex" / "sf-evaluation" / "figures"
FIG_DRAFT = STUDY.parent / "latex" / "sf-evaluation-compressed-draft" / "figures"
# rerun 549 (2026-11-02): under SMX_VALENTINE=549 gen_figures reads the 549 inputs, the reproduction checks use the 549
# macros, the cases are the 549 (source ids by the same name rule), A_d and SF melnik come from results/plain549 and
# layer1_plain549, and the output goes next to the 549 figures.
sys.path.insert(0, str(STUDY / "reference"))
import valentine549 as V  # noqa: E402
if gf.V549:
    OUT = STUDY.parent / "latex" / "sf-evaluation-549" / "review_checks"
    FIG_SHARED = FIG_DRAFT = STUDY.parent / "latex" / "sf-evaluation-549" / "figures"
R, SEED = 1000, 20261026          # as rq1_artefact.py
SEMSTRUCT = {"semstruct"}
SEMSTRUCT_E5 = {"semstruct", "sbert_e5"}
METS = ("R_exp", "F1_Hr")


def source_of(dataset: str, scenario: str) -> str:
    """X77's source_id for Valentine (explore_benchmark_properties.py)."""
    return f"{dataset.split('-')[0]}/{scenario.split('_')[0]}"


def macros(path: Path) -> dict[str, str]:
    return dict(re.findall(r"\\newcommand{\\([A-Za-z]+)}{([^}]*)}", path.read_text(encoding="utf-8")))


def rho(x, y) -> float:
    return gf.spearman(list(map(float, x)), list(map(float, y)))


# ---------------------------------------------------------------------------------------------------- B1: artefact
def artefact(rows: list[dict], panel: list[str]) -> dict[str, dict[str, float]]:
    """rq1_artefact.py's main() with the panel as a parameter (same arrays, rng seed and draw order)."""
    panel = sorted(panel)
    cell = defaultdict(dict)
    for r in rows:
        if r["seed"] in panel:
            fam = r.get("family") or gf.family_of(r["dataset"], r["scenario"])
            cell[(fam, r["dataset"], r["scenario"])][r["seed"]] = r
    fam_n = defaultdict(int)
    for k in cell:
        fam_n[k[0]] += 1
    keys = sorted(k for k in cell if fam_n[k[0]] >= gf.MIN_FAMILY and all(m in cell[k] for m in panel))
    fams = sorted({k[0] for k in keys})
    idx = {f: np.array([i for i, k in enumerate(keys) if k[0] == f]) for f in fams}
    Q = np.array([[float(cell[k][m]["seed_R_exp"]) for k in keys] for m in panel])
    G = {met: np.array([[float(cell[k][m][f"sf_{met}"]) - float(cell[k][m][f"seed_{met}"]) for k in keys] for m in panel])
         for met in METS}
    QB = {met: np.array([[float(cell[k][m][f"seed_{met}"]) for k in keys] for m in panel]) for met in METS}

    def macro(E, sel=None):
        return np.mean([E[:, (sel[f] if sel is not None else ix)].mean(axis=1) for f, ix in idx.items()], axis=0)

    q_all = macro(Q)
    full = {met: rho(q_all, macro(G[met])) for met in G}
    rng = np.random.default_rng(SEED)
    res = {}
    for met in G:
        sh = []
        for _ in range(R):
            A, Bh = {}, {}
            for f, ix in idx.items():
                p = rng.permutation(ix)
                A[f], Bh[f] = p[: len(p) // 2], p[len(p) // 2:]
            sh.append(rho(macro(Q, A), macro(G[met], Bh)))
            sh.append(rho(macro(Q, Bh), macro(G[met], A)))
        sh = np.array(sh)
        base = macro(QB[met])
        head = macro(G[met]) / np.clip(1 - base, 1e-9, None)
        nul, nul_rel = [], []
        g, qb = G[met], QB[met]
        for _ in range(R):
            perm = np.array([rng.permutation(len(panel)) for _ in range(g.shape[1])]).T
            gp = np.take_along_axis(g, perm, axis=0)
            gp = np.clip(gp, -qb, 1 - qb)
            nul.append(rho(q_all, macro(gp)))
            nul_rel.append(rho(q_all, macro(gp) / np.clip(1 - base, 1e-9, None)))
        nul, nul_rel = np.array(nul), np.array(nul_rel)
        head_obs = rho(q_all, head)
        res[met] = dict(full=full[met], sh_med=float(np.median(sh)), sh_lo=float(np.percentile(sh, 2.5)),
                        sh_hi=float(np.percentile(sh, 97.5)), head=head_obs,
                        nul_med=float(np.median(nul)), nul_lo=float(np.percentile(nul, 2.5)),
                        nul_hi=float(np.percentile(nul, 97.5)), p_nul=float((np.sum(nul <= full[met]) + 1) / (R + 1)),
                        rel_med=float(np.median(nul_rel)), rel_lo=float(np.percentile(nul_rel, 2.5)),
                        rel_hi=float(np.percentile(nul_rel, 97.5)),
                        p_rel=float((np.sum(nul_rel <= head_obs) + 1) / (R + 1)))
    return res


def artefact_macros(a: dict) -> dict[str, str]:
    """The keys and formats of make_review_numbers.py (rv...)."""
    out = {}
    for met, key in (("R_exp", "Rexp"), ("F1_Hr", "FHr")):
        x = a[met]
        f2 = lambda v: f"{float(f'{v:+.2f}'):+.2f}"  # noqa: E731  (the md table rounds first, then the macro reformats)
        out[f"Split{key}"], out[f"Split{key}Lo"], out[f"Split{key}Hi"] = f2(x["sh_med"]), f2(x["sh_lo"]), f2(x["sh_hi"])
        out[f"Head{key}"] = f2(x["head"])
        out[f"Null{key}"], out[f"Null{key}Lo"], out[f"Null{key}Hi"] = f2(x["nul_med"]), f2(x["nul_lo"]), f2(x["nul_hi"])
        p_nul, p_rel = float(f"{x['p_nul']:.3f}"), float(f"{x['p_rel']:.4f}")
        out[f"NullP{key}"] = f"{p_nul:.2f}"
        out[f"RelNull{key}"], out[f"RelNull{key}Lo"], out[f"RelNull{key}Hi"] = (f2(x["rel_med"]), f2(x["rel_lo"]),
                                                                                f2(x["rel_hi"]))
        out[f"RelNullP{key}"] = f"{p_rel:.3f}"
    return out


# ------------------------------------------------------------------------------------------------------- B2: LOSO
def gain_stats(rows: list[dict], panel: list[str]) -> dict[str, float]:
    """Panel mean of SF's macro change (R_exp, F1_Hr, J_tf) and rho(quality, change), on the given rows."""
    cols = [f"{p}_{m}" for p in ("seed", "sf") for m in METS]
    agg = gf.macro_by_seed([r for r in rows if r["seed"] in panel], cols)
    pts = [agg[s] for s in sorted(panel)]
    out = {}
    for met in METS:
        d = [p[f"sf_{met}"] - p[f"seed_{met}"] for p in pts]
        out[f"d_{met}"] = float(np.mean(d))
        out[f"rho_{met}"] = rho([p["seed_R_exp"] for p in pts], d)
    out["d_J"] = (out["d_R_exp"] + out["d_F1_Hr"]) / 2
    return out


def dt_rows() -> tuple[list[dict], list[str]]:
    """Per (matcher, scenario): A_d (q+ + 5D) and SF melnik in R_exp, F1_Hr, J_tf, as explore_deproles.py reads them."""
    dep = list(csv.DictReader(open(STUDY / "results" / V.vreg("plain") / "deproles_scores.csv", encoding="utf-8")))
    seeds = sorted({r["seed"] for r in dep})
    sf = {}
    for r in csv.DictReader(open(STUDY / "tuning" / "results" / f"layer1_{V.vreg('plain')}_scores" / "scores.csv", encoding="utf-8")):
        if r["arm"] == "melnik":
            sf[(r["seed"], r["dataset"], r["scenario"])] = r
    out = []
    for r in dep:
        s = sf[(r["seed"], r["dataset"], r["scenario"])]
        out.append({"seed": r["seed"], "dataset": r["dataset"], "scenario": r["scenario"], "family": r["family"],
                    "ad_R_exp": float(r["A_d_R"]), "ad_F1_Hr": float(r["A_d_F"]), "ad_J": float(r["A_0.0_0.0"]),
                    "sf_R_exp": float(s["sf_R_exp"]), "sf_F1_Hr": float(s["sf_F1_Hr"]),
                    "sf_J": (float(s["sf_R_exp"]) + float(s["sf_F1_Hr"])) / 2})
    return out, seeds


def dt_stats(rows: list[dict], panel: list[str]) -> dict[str, float]:
    """Mean over matchers of the family macro of A_d - SF (equal to explore_deproles.contrast's estimate)."""
    agg = gf.macro_by_seed([r for r in rows if r["seed"] in panel],
                           ["ad_R_exp", "ad_F1_Hr", "ad_J", "sf_R_exp", "sf_F1_Hr", "sf_J"])
    return {f"dt_{m}": float(np.mean([agg[s][f"ad_{m}"] - agg[s][f"sf_{m}"] for s in panel])) for m in ("R_exp", "F1_Hr", "J")}


def per_source(rows: list[dict], panel: list[str], pairs: list[tuple[str, str]]) -> dict[str, float]:
    """Plain mean over the source's scenarios and the panel of x - y for each (key, (x, y))."""
    acc = defaultdict(list)
    for r in rows:
        if r["seed"] in panel:
            for key, (x, y) in pairs:
                acc[key].append(float(r[x]) - float(r[y]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def per_source_rho(rows: list[dict], panel: list[str]) -> dict[str, float]:
    """rho over matchers within one source (plain scenario means); nan if a side is constant."""
    by = defaultdict(list)
    for r in rows:
        if r["seed"] in panel:
            by[r["seed"]].append(r)
    out = {}
    for met in METS:
        x = [np.mean([float(r["seed_R_exp"]) for r in by[s]]) for s in sorted(panel)]
        y = [np.mean([float(r[f"sf_{met}"]) - float(r[f"seed_{met}"]) for r in by[s]]) for s in sorted(panel)]
        out[f"rho_{met}"] = rho(x, y) if len(set(x)) > 1 and len(set(y)) > 1 else float("nan")
    return out


def opposite(v: float, ref: float) -> bool:
    """Strictly opposite sign to the reference estimate."""
    return v * ref < 0


def near_zero(v: float, digits: int = 3) -> bool:
    """Rounds to zero at the reported precision."""
    return abs(v) < 0.5 * 10 ** -digits


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = gf.layer1_rows("plain", "melnik")
    pts, n_full = gf.load_rows(rows)
    panel = sorted(p["seed"] for p in pts if p["group"] != "ctrl")
    ss_in_panel = sorted(s for s in panel if s.startswith("semstruct") or gf.META[s][1].lower().startswith("semstruct"))
    assert ss_in_panel == ["semstruct"], ss_in_panel
    lm_only_in_panel = "semstruct_lm_only" in panel
    # sbert_e5 vs the LM-only control, per scenario
    by = defaultdict(dict)
    for r in rows:
        if r["seed"] in ("sbert_e5", "semstruct_lm_only"):
            by[r["seed"]][(r["dataset"], r["scenario"])] = r
    e5_same = sum(all(abs(float(by["sbert_e5"][k][c]) - float(by["semstruct_lm_only"][k][c])) < 1e-12
                      for c in ("seed_R_exp", "seed_F1_Hr", "sf_R_exp", "sf_F1_Hr")) for k in by["sbert_e5"])

    # ---- B1: reproduce, then reduce
    pub = macros(FIG_SHARED / "rq1_numbers.tex")
    pub_draft = macros(FIG_DRAFT / "rq1_numbers.tex")
    rv = macros(FIG_DRAFT / "review_numbers.tex")
    variants = {"full": panel, "noss": [s for s in panel if s not in SEMSTRUCT],
                "noss_e5": [s for s in panel if s not in SEMSTRUCT_E5]}
    b1, mismatches = {}, []
    for name, pan in variants.items():
        p_sub = [p for p in pts if p["group"] == "ctrl" or p["seed"] in pan]
        nums = gf.rq_numbers("plain", p_sub, rows)
        art = artefact(rows, pan)
        b1[name] = {"panel": pan, "rq": nums, "art": art, "rv": artefact_macros(art)}
        print(f"B1 {name}: N={nums['N']} rhoR={nums['RhoRexp']} rhoF={nums['RhoFHr']}", flush=True)
    for k in ("N", "RhoRexp", "UpRexp", "MeanRexp", "RhoFHr", "UpFHr", "MeanFHr", "RhoFamRexp", "RhoFamFHr", "NFam"):
        for src, ref in (("shared rq1_numbers.tex", pub), ("draft rq1_numbers.tex", pub_draft)):
            if b1["full"]["rq"][k] != ref[f"rqOne{k}"]:
                mismatches.append(f"{src} rqOne{k}: published {ref[f'rqOne{k}']}, reproduced {b1['full']['rq'][k]}")
    for k, v in b1["full"]["rv"].items():
        if rv.get(f"rv{k}") != v:
            mismatches.append(f"review_numbers.tex rv{k}: published {rv.get(f'rv{k}')}, reproduced {v}")
    if mismatches:
        print("REPRODUCTION MISMATCH:\n" + "\n".join(mismatches))
        (OUT / "source_sensitivity_MISMATCH.txt").write_text("\n".join(mismatches) + "\n", encoding="utf-8")
        return 1

    # ---- B2: sources
    cases = json.loads(gf.PROTOCOL.read_text(encoding="utf-8"))["cases"]
    if gf.V549:   # the 549 scenarios; scenario type = the dataset name after the family prefix
        cases = [{"dataset": d, "scenario": s, "family": f, "scenario_type": d[len(f) + 1:]} for d, s, f in V.cases549()]
    smap = {(c["dataset"], c["scenario"]): source_of(c["dataset"], c["scenario"]) for c in cases}
    fam_of = {(c["dataset"], c["scenario"]): c["family"] for c in cases}
    with (OUT / "valentine_source_map.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "scenario", "family", "scenario_type", "source"])
        for c in sorted(cases, key=lambda c: (smap[(c["dataset"], c["scenario"])], c["dataset"], c["scenario"])):
            w.writerow([c["dataset"], c["scenario"], c["family"], c["scenario_type"], smap[(c["dataset"], c["scenario"])]])
    sources = sorted(set(smap.values()), key=lambda s: (-sum(v == s for v in smap.values()), s))
    src_info = {s: {"n": sum(v == s for v in smap.values()),
                    "family": sorted({fam_of[k] for k, v in smap.items() if v == s}),
                    "datasets": sorted({k[0] for k, v in smap.items() if v == s})} for s in sources}
    for r in rows:
        r["_src"] = smap[(r["dataset"], r["scenario"])]
    drows, dpanel = dt_rows()
    for r in drows:
        r["_src"] = smap[(r["dataset"], r["scenario"])]
    dpanel_noss = [s for s in dpanel if s not in SEMSTRUCT]

    full_g = gain_stats(rows, panel)
    full_d = dt_stats(drows, dpanel)
    full_d_noss = dt_stats(drows, dpanel_noss)
    # checks against the published macros
    chk = []
    if f"{full_g['d_R_exp']:+.3f}" != pub["rqOneMeanRexp"] or f"{full_g['d_F1_Hr']:+.3f}" != pub["rqOneMeanFHr"]:
        chk.append("gain mean")
    if f"{full_g['rho_R_exp']:.2f}" != pub["rqOneRhoRexp"] or f"{full_g['rho_F1_Hr']:.2f}" != pub["rqOneRhoFHr"]:
        chk.append("gain rho")
    dr = macros(FIG_DRAFT / "dr_numbers.tex")
    if f"{-full_d['dt_J']:+.3f}" != dr["drPlRefVsAdEst"] or str(len(dpanel)) != dr["drPlRefVsAdN"]:
        chk.append(f"dt: reproduced SF - A_d {-full_d['dt_J']:+.4f}, N {len(dpanel)}; published {dr['drPlRefVsAdEst']}, "
                   f"N {dr['drPlRefVsAdN']}")
    if chk:
        print("B2 REPRODUCTION MISMATCH: " + "; ".join(chk))
        (OUT / "source_sensitivity_MISMATCH.txt").write_text("\n".join(chk) + "\n", encoding="utf-8")
        return 1

    per, loso = {}, {}
    for s in sources:
        sub = [r for r in rows if r["_src"] == s]
        dsub = [r for r in drows if r["_src"] == s]
        e = per_source(sub, panel, [("d_R_exp", ("sf_R_exp", "seed_R_exp")), ("d_F1_Hr", ("sf_F1_Hr", "seed_F1_Hr"))])
        e["d_J"] = (e["d_R_exp"] + e["d_F1_Hr"]) / 2
        e.update(per_source_rho(sub, panel))
        e.update({f"dt_{m}": v for m, v in per_source(dsub, dpanel, [(m, (f"ad_{m}", f"sf_{m}"))
                                                                      for m in ("R_exp", "F1_Hr", "J")]).items()})
        per[s] = e
        keep = [r for r in rows if r["_src"] != s]
        dkeep = [r for r in drows if r["_src"] != s]
        lo = gain_stats(keep, panel)
        lo.update(dt_stats(dkeep, dpanel))
        lo["dt_noss_J"] = dt_stats(dkeep, dpanel_noss)["dt_J"]
        fam_n = defaultdict(int)
        for (d, c), f in fam_of.items():
            if smap[(d, c)] != s:
                fam_n[f] += 1
        lo["families"] = sorted(f for f, k in fam_n.items() if k >= gf.MIN_FAMILY)
        loso[s] = lo
        print(f"LOSO {s}: dJ={lo['d_J']:+.4f} dt={lo['dt_J']:+.4f} rhoR={lo['rho_R_exp']:+.2f}", flush=True)
    full = {**full_g, **full_d, "dt_noss_J": full_d_noss["dt_J"]}

    keys = ["d_R_exp", "d_F1_Hr", "d_J", "rho_R_exp", "rho_F1_Hr", "dt_R_exp", "dt_F1_Hr", "dt_J", "dt_noss_J"]
    summ = {}
    for k in keys:
        vals = [loso[s][k] for s in sources]
        summ[k] = {"full": full[k], "min": min(vals), "max": max(vals),
                   "argmin": sources[int(np.argmin(vals))], "argmax": sources[int(np.argmax(vals))],
                   "flips": sum(opposite(v, full[k]) for v in vals),
                   "near_zero": sum(near_zero(v, 2 if k.startswith("rho") else 3) for v in vals)}
        pk = k.replace("dt_noss_J", "")
        if pk and pk in per[sources[0]]:
            pv = [per[s][pk] for s in sources if not math.isnan(per[s][pk])]
            summ[k].update({"src_min": min(pv), "src_max": max(pv),
                            "src_opposite": sum(opposite(v, full[k]) for v in pv), "src_n": len(pv)})

    with (OUT / "source_sensitivity_per_source.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "n_scenarios", "family"] + [f"per_{k}" for k in keys[:-1]] + [f"loso_{k}" for k in keys])
        for s in sources:
            w.writerow([s, src_info[s]["n"], "+".join(src_info[s]["family"])] + [per[s][k] for k in keys[:-1]]
                       + [loso[s][k] for k in keys])

    result = {"b1": {"semstruct_members": ss_in_panel, "lm_only_in_panel": lm_only_in_panel,
                     "e5_equal_lm_only_scenarios": e5_same, "n_scenarios": n_full,
                     **{v: {"n": len(b1[v]["panel"]), "panel": b1[v]["panel"], "dropped": sorted(set(panel) - set(b1[v]["panel"])),
                            "rq": b1[v]["rq"], "art": b1[v]["art"], "rv": b1[v]["rv"]} for v in variants}},
              "b2": {"sources": src_info, "full": full, "per_source": per, "loso": loso, "summary": summ,
                     "n_panel_gain": len(panel), "n_panel_dt": len(dpanel), "n_panel_dt_noss": len(dpanel_noss)}}
    (OUT / "source_sensitivity.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (OUT / "source_sensitivity.md").write_text(report(result), encoding="utf-8")
    print("wrote", OUT / "source_sensitivity.md")
    return 0


# ---------------------------------------------------------------------------------------------------------- report
def report(res: dict) -> str:
    b1, b2 = res["b1"], res["b2"]
    f2, f3 = (lambda v: f"{v:+.2f}"), (lambda v: f"{v:+.3f}")
    L = ["# Source sensitivity of the compressed draft's Sec. IV and Sec. V-A numbers (EXPLORATORY review check)", "",
         "Generated by `sf-matcher-study/review_checks/source_sensitivity.py` from cached scores only (no matcher or SF "
         "runs). Macros: `latex/sf-evaluation-compressed-draft/make_src_numbers.py` -> `figures/src_numbers.tex`.", "",
         "## B1. The RQ1 panel without the SemStruct reimplementation", "",
         "**SemStruct-derived panel members.** Exactly one: `semstruct` (SemStruct, KDD 2026, as reimplemented in the "
         "harness; the 2026-09-08 checkpoint `harness/reports/semstruct_20260908/matrices/plain/semstruct/`, imported as "
         "a FILE seed by `sf-matcher-study/import_file_seeds.py`).",
         f"- `semstruct_lm_only` (the LM-only control arm) is not in the panel: it is in gen_figures.py's CONTROL_ARMS, "
         f"and `load_rows` drops it as a duplicate of `sbert_e5` (in panel: {b1['lm_only_in_panel']}).",
         f"- `sbert_e5` (E5-small, run independently through the Python matcher server) has the same per-scenario "
         f"scores as the LM-only control on {b1['e5_equal_lm_only_scenarios']} of {b1['n_scenarios']} scenarios (same frozen "
         "encoder). It is not SemStruct-derived; the stricter variant below drops it as well.",
         "- No other member uses SemStruct as an input: the composites combine Levenshtein, prefix/suffix, WordNet, "
         "containment and distribution (make_configs.COMPOSITE); `learned_lofo` combines levenshtein, prefix_suffix, "
         "wordnet_wup, containment, pmf and ks (learned_combiner.py).", "",
         "**Reproduction.** With the full panel, gen_figures.rq_numbers (imported unmodified) reproduces every checked "
         "`\\rqOne...` macro (N, Up, Mean, Rho, RhoFam for R_exp and F1_Hr; shared and draft copies) and the replicated "
         "rq1_artefact computation reproduces all `\\rv{Split,Head,Null,RelNull}...` macros of review_numbers.tex exactly. "
         "The script stops on any mismatch.", "",
         "| statistic | full panel | without `semstruct` | without `semstruct`, `sbert_e5` |", "|---|---|---|---|"]
    V = ("full", "noss", "noss_e5")
    rowsdef = [("N (matchers)", lambda v: b1[v]["rq"]["N"]),
               ("SF improves R_exp (matchers)", lambda v: b1[v]["rq"]["UpRexp"]),
               ("SF improves F1_Hr (matchers)", lambda v: b1[v]["rq"]["UpFHr"]),
               ("mean change R_exp", lambda v: b1[v]["rq"]["MeanRexp"]),
               ("mean change F1_Hr", lambda v: b1[v]["rq"]["MeanFHr"]),
               ("rho(quality, change R_exp)", lambda v: b1[v]["rq"]["RhoRexp"]),
               ("rho(quality, change F1_Hr)", lambda v: b1[v]["rq"]["RhoFHr"]),
               ("rho, one point per family, R_exp", lambda v: f"{b1[v]['rq']['RhoFamRexp']} ({b1[v]['rq']['NFam']} fam.)"),
               ("split-half rho R_exp: median [2.5, 97.5]",
                lambda v: f"{b1[v]['rv']['SplitRexp']} [{b1[v]['rv']['SplitRexpLo']}, {b1[v]['rv']['SplitRexpHi']}]"),
               ("ceiling-null rho R_exp: median [2.5, 97.5]",
                lambda v: f"{b1[v]['rv']['NullRexp']} [{b1[v]['rv']['NullRexpLo']}, {b1[v]['rv']['NullRexpHi']}]"),
               ("ceiling-null rho F1_Hr: median [2.5, 97.5]",
                lambda v: f"{b1[v]['rv']['NullFHr']} [{b1[v]['rv']['NullFHrLo']}, {b1[v]['rv']['NullFHrHi']}]"),
               ("relative-gain rho R_exp (gain / headroom)", lambda v: b1[v]["rv"]["HeadRexp"]),
               ("its ceiling null: median [2.5, 97.5]; P",
                lambda v: f"{b1[v]['rv']['RelNullRexp']} [{b1[v]['rv']['RelNullRexpLo']}, {b1[v]['rv']['RelNullRexpHi']}]; "
                          f"{b1[v]['rv']['RelNullPRexp']}"),
               ("relative-gain rho F1_Hr", lambda v: b1[v]["rv"]["HeadFHr"]),
               ("its ceiling null: median [2.5, 97.5]; P",
                lambda v: f"{b1[v]['rv']['RelNullFHr']} [{b1[v]['rv']['RelNullFHrLo']}, {b1[v]['rv']['RelNullFHrHi']}]; "
                          f"{b1[v]['rv']['RelNullPFHr']}")]
    for name, fn in rowsdef:
        L.append(f"| {name} | " + " | ".join(str(fn(v)) for v in V) + " |")
    fam_full = {gf.META[s][0] for s in b1["full"]["panel"]}
    fam_noss = {gf.META[s][0] for s in b1["noss"]["panel"]}
    if fam_full - fam_noss:
        L += ["", f"The per-family rho loses the family {', '.join(sorted(fam_full - fam_noss))}: `semstruct` is its only "
              "panel member (REMA and SiMa lack full coverage, EmbDI was not run)."]
    L += ["", "Ceiling null and split-half: 1,000 draws each, seed 20261026, as rq1_artefact.py; the reduced panels draw "
          "with the same seed but over fewer matchers, so their null bands are new draws, not subsets.", "",
          "## B2. Leave one source out on readable Valentine", "",
          "**Source mapping.** X77's `source_id` (`explore_benchmark_properties.py`): `<family prefix>/<first token of the "
          "scenario name>`; the scenario folders in `protocol.json` (`id` = valentine/<family>/<type>/<scenario>) agree. "
          "Valentine's split-built families are each cut from one table; Magellan's seven pairs come from six left tables "
          "(dblp_acm and dblp_scholar share DBLP). Per scenario: `valentine_source_map.csv`.", "",
          "| source | family | scenarios | datasets |", "|---|---|---|---|"]
    for s, i in b2["sources"].items():
        L.append(f"| {s} | {', '.join(i['family'])} | {i['n']} | {', '.join(i['datasets'])} |")
    fam_note = ("Families in the macro need >= 5 scenarios (gen_figures.MIN_FAMILY), so the full macro averages ChEMBL, "
                "Magellan, OpenData and TPC-DI. Dropping one of the three large sources drops its whole family; dropping one "
                "Magellan source leaves 5 or 6 Magellan scenarios (still in the macro); dropping Wikidata/musicians changes "
                "nothing.")
    L += ["", fam_note, "",
          f"Contrasts: (i) SF as published minus the initial matcher, RQ1 panel ({b2['n_panel_gain']} matchers); (ii) the "
          f"datatype term A_d = q+ + 5D minus SF as published, X46 panel ({b2['n_panel_dt']} matchers: the RQ1 panel without "
          "the data-type matcher), the info-ladder row 'Valentine readable' (= -\\drPlRefVsAdEst); per-scenario values from "
          "`sf-matcher-study/results/plain/deproles_scores.csv` (A_d) and `tuning/results/layer1_plain_scores/scores.csv` "
          "(arm melnik); (iii) Spearman rho over matchers of quality (R_exp alone) and the change in (i). Both full-corpus "
          "estimates reproduce the published macros (\\rqOneMean..., \\rqOneRho..., \\drPlRefVsAdEst).", "",
          "### Per source (plain mean over the source's scenarios and the panel)", "",
          "| source | n | (i) dR_exp | (i) dF1_Hr | (i) dJ_tf | (iii) rho R_exp | (iii) rho F1_Hr | (ii) A_d - SF, J_tf |",
          "|---|---|---|---|---|---|---|---|"]
    fmt_r = lambda v: "n/a" if math.isnan(v) else f2(v)  # noqa: E731
    for s, e in b2["per_source"].items():
        L.append(f"| {s} | {b2['sources'][s]['n']} | {f3(e['d_R_exp'])} | {f3(e['d_F1_Hr'])} | {f3(e['d_J'])} | "
                 f"{fmt_r(e['rho_R_exp'])} | {fmt_r(e['rho_F1_Hr'])} | {f3(e['dt_J'])} |")
    L += ["", "rho within a source of 1-2 scenarios is over a single pair or two and only descriptive.", "",
          "### Leave one source out (the macro statistic without the source)", "",
          "| dropped source | (i) dR_exp | (i) dF1_Hr | (i) dJ_tf | (iii) rho R_exp | (iii) rho F1_Hr | (ii) A_d - SF, J_tf | "
          "(ii) without semstruct |", "|---|---|---|---|---|---|---|---|"]
    fu = b2["full"]
    L.append(f"| none (full) | {f3(fu['d_R_exp'])} | {f3(fu['d_F1_Hr'])} | {f3(fu['d_J'])} | {f2(fu['rho_R_exp'])} | "
             f"{f2(fu['rho_F1_Hr'])} | {f3(fu['dt_J'])} | {f3(fu['dt_noss_J'])} |")
    for s, e in b2["loso"].items():
        L.append(f"| {s} | {f3(e['d_R_exp'])} | {f3(e['d_F1_Hr'])} | {f3(e['d_J'])} | {f2(e['rho_R_exp'])} | "
                 f"{f2(e['rho_F1_Hr'])} | {f3(e['dt_J'])} | {f3(e['dt_noss_J'])} |")
    L += ["", "### Summary", "",
          "| statistic | full | LOSO min (dropped) | LOSO max (dropped) | LOSO sign flips | per-source range | "
          "sources of opposite sign |", "|---|---|---|---|---|---|---|"]
    names = {"d_R_exp": "(i) dR_exp", "d_F1_Hr": "(i) dF1_Hr", "d_J": "(i) dJ_tf", "rho_R_exp": "(iii) rho R_exp",
             "rho_F1_Hr": "(iii) rho F1_Hr", "dt_R_exp": "(ii) A_d - SF, R_exp", "dt_F1_Hr": "(ii) A_d - SF, F1_Hr",
             "dt_J": "(ii) A_d - SF, J_tf", "dt_noss_J": "(ii) J_tf, panel without semstruct"}
    for k, x in b2["summary"].items():
        fm = f2 if k.startswith("rho") else f3
        src = (f"{fm(x['src_min'])} to {fm(x['src_max'])}" if "src_min" in x else "--")
        opp = f"{x['src_opposite']} of {x['src_n']}" if "src_n" in x else "--"
        nz = f" ({x['near_zero']} round to 0)" if x["near_zero"] else ""
        L.append(f"| {names[k]} | {fm(x['full'])} | {fm(x['min'])} ({x['argmin']}) | {fm(x['max'])} ({x['argmax']}) | "
                 f"{x['flips']} of {len(b2['loso'])}{nz} | {src} | {opp} |")
    L += ["", "A sign flip counts a leave-one-out (or per-source) estimate strictly of the opposite sign to the full estimate; "
          "'round to 0' counts leave-one-out estimates that are 0 at the reported precision (3 decimals, rho 2). With 10 sources, three of which carry 373 of 382 scenarios, this is descriptive, not "
          "inference.", ""]
    L += ["## Macros (`figures/src_numbers.tex`, written by the paper folder's `make_src_numbers.py` from "
          "`source_sensitivity.json`)", "",
          "| macro | meaning |", "|---|---|",
          "| `\\srcNoSsN`, `\\srcNoSsDropped` | RQ1 panel size without `semstruct`; members dropped (1) |",
          "| `\\srcNoSs{Up,Mean,Rho}{Rexp,FHr}` | as `\\rqOne{Up,Mean,Rho}...`, panel without `semstruct` |",
          "| `\\srcNoSsRhoFam{Rexp,FHr}`, `\\srcNoSsNFam` | as `\\rqOneRhoFam...`, `\\rqOneNFam` (one point per matcher family) |",
          "| `\\srcNoSs{Split,Null,RelNull}{Rexp,FHr}[Lo,Hi]`, `\\srcNoSsHead{Rexp,FHr}`, `\\srcNoSsRelNullP{Rexp,FHr}` | "
          "as `\\rv...` in review_numbers.tex: split-half rho, ceiling-null rho, relative-gain rho (gain / headroom), "
          "its ceiling null (median, 2.5/97.5 %) and P(null <= observed) |",
          "| `\\srcNoSsEfive...` | the same set, panel also without `sbert_e5` |",
          "| `\\srcNSources`, `\\srcNScen` | Valentine source tables (10) and scenarios (382) |",
          "| `\\srcNSourcesBig`, `\\srcNScenBig` | sources with >= 5 scenarios (3) and their scenarios (373) |",
          "| `\\srcNMagellanSources` | Magellan source tables (6) |",
          "| `\\srcNPanelGain`, `\\srcNPanelDt` | matchers behind (i) (76) and (ii) (75) |",
          "| `\\srcFull<X>` | full-corpus estimate of statistic X |",
          "| `\\srcLoso<X>Min`, `\\srcLoso<X>Max` | range of the macro estimate of X over the 10 leave-one-source-out drops |",
          "| `\\srcLoso<X>MinSrc`, `\\srcLoso<X>MaxSrc` | the dropped source at the minimum / maximum (family name for the "
          "three large sources, 'Magellan (table)' otherwise) |",
          "| `\\srcLoso<X>Flips` | drops whose estimate has the opposite sign to the full estimate |",
          "| `\\srcLoso<X>NearZero` | drops whose estimate is 0 at the reported precision |",
          "| `\\srcSrc<X>Min`, `\\srcSrc<X>Max`, `\\srcSrc<X>Opp` | range of the per-source estimates; sources of the "
          "opposite sign to the full estimate |",
          "| X = `Gain`, `GainRexp`, `GainFHr` | (i) SF as published - initial matcher in J_tf, R_exp, F1_Hr |",
          "| X = `RhoRexp`, `RhoFHr` | (iii) rho(quality, SF's change in R_exp / F1_Hr) |",
          "| X = `Dt`, `DtRexp`, `DtFHr` | (ii) datatype term - SF as published in J_tf, R_exp, F1_Hr |",
          "| X = `DtNoSs` | (ii) in J_tf on the X46 panel without `semstruct` (no per-source macros) |",
          "| `\\srcSrc{Tpc,Opendata,Chembl}{N,Gain,GainRexp,GainFHr,RhoRexp,RhoFHr,Dt}` | per-source estimates of the three "
          "large sources (each one Valentine family) |", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
