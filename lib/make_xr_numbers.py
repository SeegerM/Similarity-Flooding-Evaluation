"""Write output/xr_numbers.tex: macros for the external review of 2026-11-03 (prefix xr). Numbers are parsed, never typed:

  results/sf_study/tuning/results/review_checks/xr_rq1_null_variants.json      ceiling-null variants of Fig. 3
  results/sf_study/tuning/results/review_checks/xr_literal_decomposition.json  literal term vs SF after two updates
  results/sf_study/tuning/results/review_checks/xr_gamma_lambda_sweep.json     role / table-context weights
  results/sf_study/tuning/results/review_checks/xr_semstruct_seeds.json        SemStruct per training seed
      (written by experiments/sf_study/review_checks/xr_*.py)
  definitions of the terms and graphs (xrDef*), from the code and reports that produced the results:
      experiments/sf_study/explore_gv2.py (k of the key + FD graph), explore_deproles.py (gamma grid),
      results/sf_study/results/{plain549,md549}/deproles_summary.md (gamma chosen per held-out family),
      experiments/sf_study/reference/score_final.py (frozen gamma), reference/score_tablectx.py (lambda grid, class
      weights); Schematch's Java sources (datatype matcher, FD/UCC size limits, key score, TOPK_HYBRID weights) are not
      part of this repository: their values are cached with pattern, file hash and Schematch commit in
      results/sf_study/tuning/results/review_checks/xr_definitions.json (experiments/sf_study/review_checks/xr_definitions.py)
  Synthea->OMOP construction (xrSyn*): experiments/sf_study/data/convert_synthea.py and
      experiments/sf_study/data/synthea_plain/conversion_report.json; the scenario counts of the converted corpus
      (not redistributed) from xr_definitions.json
  matcher panel per corpus (xrPanel*, Table I): output/abs_numbers_report.md (make_abs_numbers.py, run before)

    python make_xr_numbers.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from paths import OUT, STUDY, STUDY_CODE
RC = STUDY / "tuning" / "results" / "review_checks"
M: dict[str, str] = {}


def s3(x: float) -> str:
    return f"{x:+.3f}"


def r2(x: float) -> str:
    return f"{x:+.2f}"


def grab(path: Path, pattern: str, group: int = 1) -> str:
    m = re.search(pattern, path.read_text(encoding="utf-8"), re.S)
    assert m, (path.name, pattern)
    return m.group(group)


WORD = {"0.0": "Zero", "0.1": "PtOne", "0.2": "PtTwo", "0.5": "Half", "1.0": "One", "2.0": "Two"}


def main() -> int:
    # ---------------- ceiling-null variants (Fig. 3)
    nv = json.loads((RC / "xr_rq1_null_variants.json").read_text(encoding="utf-8"))
    V = {"published": "Pub", "V1_matcher_level": "Mat", "V2_floor_scaled": "Flo", "V3_nonnegative": "Pos"}
    for met, mk in (("R_exp", "Rank"), ("F1_Hr", "Asg")):
        for v, vk in V.items():
            for row, rk in (("raw", "Raw"), ("rel", "Rel")):
                x = nv["metrics"][met][v][row]
                b = f"xrNul{vk}{rk}{mk}"
                M[b + "Obs"], M[b + "Lo"], M[b + "Hi"] = r2(x["obs"]), r2(x["lo"]), r2(x["hi"])
                M[b + "Below"] = "yes" if x["below"] else "no"
        M[f"xrNulPosN{mk}"] = str(nv["metrics"][met]["V3_nonnegative"]["matchers"])
    M["xrNulDraws"] = str(nv["draws"])
    # independent check: the published variant equals the stored ceiling-null check behind review_numbers.tex
    rv = (RC / "rq1_artefact_plain.md").read_text(encoding="utf-8")
    for met, label in (("R_exp", "expected Recall@GT"), ("F1_Hr", "tie-free F1")):
        row = next(l for l in rv.splitlines() if l.startswith(f"| {label} |"))
        null = [float(x) for x in re.findall(r"[+-]\d+\.\d+", row.split("|")[6])]
        pub = nv["metrics"][met]["published"]["raw"]
        assert r2(pub["lo"]) == r2(null[1]) and r2(pub["hi"]) == r2(null[2]), (met, pub, null)
    # ---------------- literal decomposition
    ld = json.loads((RC / "xr_literal_decomposition.json").read_text(encoding="utf-8"))
    CK = {"hdxsm": "Hdx", "goby": "Goby", "gdcsm": "Gdc", "madi_indep": "Madi"}
    CT = {"A_L − SF_t2": "ALTwo", "A_L − SF_conv": "ALConv", "A_dL − SF_t2": "DLTwo", "A_dL − A_L": "DLAL",
          "+L − SF_t2": "PLTwo", "+L − SF_conv": "PLConv", "+L − A_L": "PLAL"}
    for c, ck in CK.items():
        for k, kk in CT.items():
            r = ld[c][k]
            b = f"xrLit{ck}{kk}"
            M[b] = s3(r["est"])
            if r["lo"] == r["lo"]:   # not NaN (>= 5 clusters)
                M[b + "Lo"], M[b + "Hi"] = s3(r["lo"]), s3(r["hi"])
    # ---------------- gamma / lambda sweep
    sw = json.loads((RC / "xr_gamma_lambda_sweep.json").read_text(encoding="utf-8"))
    GK = {"gdcsm": "Gdc", "hdxsm": "Hdx", "goby": "Goby", "madi_indep": "Madi", "madi_products": "MadiP"}
    for c, ck in GK.items():
        by = sw["gamma"][c]["by"]
        for g, gw in WORD.items():
            M[f"xrGam{ck}{gw}"] = s3(by[g]["est"])
            if by[g]["lo"] is not None:
                M[f"xrGam{ck}{gw}Lo"], M[f"xrGam{ck}{gw}Hi"] = s3(by[g]["lo"]), s3(by[g]["hi"])
        ests = [by[g]["est"] for g in WORD]
        M[f"xrGam{ck}Min"], M[f"xrGam{ck}Max"] = s3(min(ests)), s3(max(ests))
    lam = sw["lambda"]["final_indep"]
    for g, gw in WORD.items():
        x = lam["by"][g]
        M[f"xrLamIndep{gw}"], M[f"xrLamIndep{gw}Lo"], M[f"xrLamIndep{gw}Hi"] = s3(x["est"]), s3(x["lo"]), s3(x["hi"])
        M[f"xrLamIndep{gw}Pos"] = str(x["pos"])
    his = [lam["by"][g]["hi"] for g in WORD]
    M["xrLamIndepMaxHi"] = s3(max(his))
    M["xrSweepGridMax"] = f"{max(float(g) for g in WORD):g}"
    # ---------------- SemStruct per seed
    ss = json.loads((RC / "xr_semstruct_seeds.json").read_text(encoding="utf-8"))
    for s, sk in (("0", "SOne"), ("1", "STwo"), ("2", "SThree")):
        M[f"xrSsFull{sk}"] = f"{ss['arms']['full'][s]:.3f}"
        M[f"xrSsVal{sk}"] = f"{ss['arms']['valavg'][s]:.3f}"
        M[f"xrSsValFull{sk}"] = s3(ss["per_seed"][s]["valavg_minus_full"])
        M[f"xrSsGate{sk}"] = s3(ss["per_seed"][s]["full_minus_lm_only"])
    M["xrSsOneFullSOne"] = s3(ss["onehop_minus_full_seed0"])
    M["xrSsFullMin"], M["xrSsFullMax"] = (f"{v:.3f}" for v in ss["spread"]["full"])
    M["xrSsValMin"], M["xrSsValMax"] = (f"{v:.3f}" for v in ss["spread"]["valavg"])
    # ---------------- definitions: Schematch (cached, see the docstring), then this repository's code and reports
    defs = json.loads((RC / "xr_definitions.json").read_text(encoding="utf-8"))
    for k in ("xrDefIntFloat", "xrDefStrText", "xrDefTypeShare", "xrDefMaxLhs", "xrDefMaxUcc", "xrDefKeyLen",
              "xrDefTopkHigh", "xrDefTopkLow"):
        M[k] = defs["facts"][k]["value"]
    M["xrDefTopk"] = grab(STUDY_CODE / "explore_gv2.py", r'"ucc1_fd_topk": \(.*?"smx\.sf\.topK": "(\d+)"')
    grid = grab(STUDY_CODE / "explore_deproles.py", r"\nG = \(([^)]*)\)")
    M["xrDefGammaGridMax"] = f"{max(float(x) for x in grid.split(',')):g}"
    for reg, rk in (("plain549", "Pl"), ("md549", "Md")):
        line = grab(STUDY / "results" / reg / "deproles_summary.md", r"\n- role: ([^\n]*)")
        gs = sorted({float(m) for m in re.findall(r"\(([0-9.]+), 0\.0\)", line)})
        assert gs, line
        M[f"xrDefRoleGamma{rk}Lo"], M[f"xrDefRoleGamma{rk}Hi"] = f"{gs[0]:g}", f"{gs[-1]:g}"
    fz = grab(STUDY_CODE / "reference" / "score_final.py", r'GAMMA = \{"plain": ([0-9.]+), "opaque": ([0-9.]+)\}', 0)
    a, b = re.findall(r"([0-9.]+)", fz.split("{", 1)[1])[:2]
    M["xrDefRoleFrozenPl"], M["xrDefRoleFrozenMd"] = f"{float(a):g}", f"{float(b):.1f}"
    tc = STUDY_CODE / "reference" / "score_tablectx.py"
    lg = grab(tc, r"LAMBDAS = \(([^)]*)\)")
    M["xrDefLambdaGridMax"] = f"{max(float(x) for x in lg.split(',')):g}"
    M["xrDefClassShape"] = grab(tc, r"A_pt = q\+ \+ 5 D \+ ([0-9.]+) eq_p")
    M["xrDefClassSem"] = grab(tc, r"A_st = q\+ \+ 5 D \+ ([0-9.]+) eq_s")
    # ---------------- Synthea -> OMOP construction (convert_synthea.py and its conversion report)
    syn = STUDY_CODE / "data" / "synthea_plain"
    rep_s = json.loads((syn / "conversion_report.json").read_text(encoding="utf-8"))
    M["xrSynPosBench"], M["xrSynPosKept"] = str(rep_s["positives_benchmark"]), str(rep_s["positives_kept"])
    conv = STUDY_CODE / "data" / "convert_synthea.py"
    cap = int(grab(conv, r"\nCAP = (\d+)"))
    M["xrSynCap"] = f"{cap:,}".replace(",", "{,}")
    seeds = [x.strip() for x in grab(conv, r"\nSEEDS = \(([^)]*)\)").split(",") if x.strip()]
    assert defs["synthea"]["full_schemas"] == len(seeds), (defs["synthea"], seeds)   # one whole schema per seed
    M["xrSynSeeds"], M["xrSynPairs"] = str(len(seeds)), str(defs["synthea"]["pairs"])
    # ---------------- matcher panel per corpus (Table I): make_abs_numbers.py's report
    rep = (OUT / "abs_numbers_report.md").read_text(encoding="utf-8")
    for key in ("Val", "Ind", "LmPl", "LmM", "CtlPl", "MadiP", "Ttd", "Syn", "Indep", "Gdc", "Hdx", "Goby", "Madi", "Mimic"):
        m = re.search(rf"^\| {key} \| [^|]+ \| [0-9.]+ \| [0-9.]+ \| [+-][0-9.]+ \| (\d+) \|", rep, re.M)
        assert m, key
        M[f"xrPanel{key}"] = m.group(1)
    for k in M:
        assert re.fullmatch(r"xr[A-Za-z]+", k), k
    head = ("% Generated by make_xr_numbers.py from results/sf_study/tuning/results/review_checks/xr_*.json "
            "(experiments/sf_study/review_checks/xr_*.py), the\n"
            "% term and graph definitions in the code and reports that produced the results, and abs_numbers_report.md;\n"
            "% EXPLORATORY (external review 2026-11-03); do not edit.\n")
    (OUT / "xr_numbers.tex").write_text(head + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()),
                                        encoding="utf-8")
    print(f"wrote {OUT / 'xr_numbers.tex'} ({len(M)} macros)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
