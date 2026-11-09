"""Write output/abs_numbers.tex: absolute per-corpus levels for Table I (readable regime only), and
figures/abs_numbers_report.md (sources, panels, checks).

Per corpus: \\absQ<Key> = mean J_tf of the corpus's panel of initial matchers alone; \\absSf<Key> = mean J_tf of SF as
published (arm melnik) run after those matchers. J_tf = (expected Recall@GT + tie-free Hungarian F1) / 2. Aggregation as
the paper: per matcher, mean over scenarios within a family / database (Valentine: families with >= 5 scenarios), mean
over families; then mean over the panel's non-control matchers. Panels and score files are those the paper's existing
macros come from (make_study_numbers.py and the scorers it parses):

  Val     tuning/results/layer1_plain549_scores/scores.csv, arm melnik, via make_study_numbers.layer1_rows/load_rows
          (rqOne panel, the 549 Valentine scenarios)
  Ind     tuning/results/tablectx_indep_scores/scores.csv (score_tablectx.py; q|J, sf|J; tc macros)
  LmPl    tuning/results/e2_ctl_scores/scores.csv, arm melnik, plain scenarios (score_e2.py panel; etwo macros)
  LmM     results/multitable_fix/indep2/multitable_scores.csv, cells Lahman and MusicBrainz, macro (explore_multitable_ext.py; mt macros)
  Ttd     tuning/results/t2d_exploratory_scores/scores.csv, arm melnik (score_e2.py panel; ttd macros)
  Syn     results/multitable_fix/synthea/multitable_scores.csv, cell SyntheaOMOP-Pair (mt macros, mtSynPair*)
  CtlPl, MadiP, Indep, Gdc, Hdx, Goby, Madi, Mimic
          results/x82/rows.csv, wide panel (explore_x82_bf.py (f); wp macros), q_J and melnik_J

Read-only on every input; nothing is rerun. Run:
    python make_abs_numbers.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

from paths import OUT  # noqa: E402
import make_study_numbers as gf  # noqa: E402

STUDY = gf.STUDY
csv.field_size_limit(1 << 30)

# score_e2.CONTROLS (reference/score_e2.py), copied: importing score_e2 pulls in the harness and a multiprocessing pool
E2_CONTROLS = {"constant", "random", "oracle", "oracle_noise025", "oracle_noise050", "oracle_erase050",
               "oracle_decoy100", "embedding_random", "rema_shuffled", "semstruct_lm_only", "datatype_nanfix"}
X82_KEY = {"final_controlled_plain": "CtlPl", "madi_products": "MadiP", "final_indep": "Indep", "gdcsm": "Gdc",
           "hdxsm": "Hdx", "goby": "Goby", "madi_indep": "Madi", "mimic": "Mimic"}
ORDER = ["Val", "Ind", "LmPl", "LmM", "CtlPl", "MadiP", "Ttd", "Syn", "Indep", "Gdc", "Hdx", "Goby", "Madi", "Mimic"]
NAME = {"Val": "Valentine (5 families)", "Ind": "3-DB multi-table", "LmPl": "Lahman/MusicBrainz",
        "LmM": "Lahman/MB multi-table", "CtlPl": "New databases (4)", "MadiP": "MaDI products",
        "Ttd": "T2D web tables", "Syn": "Synthea->OMOP (table pairs)", "Indep": "Indep. schema pairs",
        "Gdc": "GDC-SM", "Hdx": "HDXSM", "Goby": "GOBY", "Madi": "MaDI use cases", "Mimic": "MIMIC->OMOP"}


def fnum(x) -> float | None:
    if x is None or x == "":
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def mean(v):
    v = list(v)
    return sum(v) / len(v)


def panel_mean(cells: dict[str, dict[str, list[tuple[float, float]]]]) -> tuple[float, float, int]:
    """cells: matcher -> group -> [(q_J, sf_J) per scenario]. Mean over groups of the group mean, then over matchers."""
    q, s = [], []
    for m, groups in cells.items():
        q.append(mean(mean(a for a, _ in v) for v in groups.values()))
        s.append(mean(mean(b for _, b in v) for v in groups.values()))
    return mean(q), mean(s), len(cells)


RESULTS: dict[str, dict] = {}
CHECKS: list[str] = []


def put(key: str, q: float, s: float, n: int, scen: int, src: str, panel: str):
    RESULTS[key] = {"q": q, "sf": s, "n": n, "scen": scen, "src": src, "panel": panel}


# ---------------------------------------------------------------- Valentine (layer 1, arm melnik; rqOne's panel)
rows = gf.layer1_rows("plain", "melnik")
pts, n_scen = gf.load_rows(rows)
mat = [p for p in pts if p["group"] != "ctrl"]
jq = [(p["base_R_exp"] + p["base_F1_Hr"]) / 2 for p in mat]
js = [(p["base_R_exp"] + p["d_R_exp"] + p["base_F1_Hr"] + p["d_F1_Hr"]) / 2 for p in mat]
put("Val", mean(jq), mean(js), len(mat), n_scen, "tuning/results/layer1_plain549_scores/scores.csv (arm melnik)",
    f"rqOne panel: non-control matchers scored on all {gf.BASE_N} scenarios; macro over families with >= 5 scenarios")
VAL_DR = mean(p["d_R_exp"] for p in mat)
VAL_DF = mean(p["d_F1_Hr"] for p in mat)
del rows


# ---------------------------------------------------------------- 3-DB multi-table (tablectx; every seed of the file)
f = STUDY / "tuning" / "results" / "tablectx_indep_scores" / "scores.csv"
cells = defaultdict(lambda: defaultdict(list))
scen = set()
for r in csv.DictReader(open(f, encoding="utf-8")):
    cells[r["seed"]][r["dataset"]].append((float(r["q|J"]), float(r["sf|J"])))
    scen.add((r["dataset"], r["scenario"]))
q, s, n = panel_mean(cells)
put("Ind", q, s, n, len(scen), "tuning/results/tablectx_indep_scores/scores.csv (q|J, sf|J; sf = layer1_indep melnik)",
    "Amendment 5's frozen list (every seed in the file); macro over the 3 databases")


# ---------------------------------------------------------------- E2 scorer: Lahman/MusicBrainz (ctl, plain) and T2D
def e2(f: Path, plain_only: bool):
    data = defaultdict(dict)   # (seed, arm) -> (ds, sc) -> row
    cases = {}
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if plain_only and not r["scenario"].endswith("_plain"):
            continue
        data[(r["seed"], r["arm"])][(r["dataset"], r["scenario"])] = r
        cases[(r["dataset"], r["scenario"])] = r["family"]
    seeds = sorted({s for s, a in data if a == "melnik" and s not in E2_CONTROLS
                    and set(data[(s, "melnik")]) == set(cases)})

    def J(r, p):
        return (float(r[f"{p}_R_exp"]) + float(r[f"{p}_F1_Hr"])) / 2
    cells = defaultdict(lambda: defaultdict(list))
    for s in seeds:
        for k, r in data[(s, "melnik")].items():
            cells[s][cases[k]].append((J(r, "seed"), J(r, "sf")))
    q, sf, n = panel_mean(cells)
    # check: P3 additive_d - melnik (J), with score_e2's panel for that contrast (both arms on every scenario)
    s3 = [s for s in seeds if set(data.get((s, "additive_d"), {})) == set(cases)]
    fam = defaultdict(list)
    for k, famname in cases.items():
        fam[famname].append(mean(J(data[(s, "additive_d")][k], "sf") - J(data[(s, "melnik")][k], "sf") for s in s3))
    p3 = mean(mean(v) for v in fam.values())
    return q, sf, n, len(cases), p3, len(s3)


q, s, n, ns, p3, n3 = e2(STUDY / "tuning" / "results" / "e2_ctl_scores" / "scores.csv", True)
put("LmPl", q, s, n, ns, "tuning/results/e2_ctl_scores/scores.csv (arm melnik, *_plain scenarios)",
    "score_e2.py: non-controls with melnik on every plain scenario; mean within family (Lahman, MusicBrainz), mean over families")
P3_LM = (p3, n3)
q, s, n, ns, p3, n3 = e2(STUDY / "tuning" / "results" / "t2d_exploratory_scores" / "scores.csv", False)
put("Ttd", q, s, n, ns, "tuning/results/t2d_exploratory_scores/scores.csv (arm melnik)",
    "score_e2.py (explore_e2_t2d.py): non-controls with melnik on every scenario; one family")
P3_TTD = (p3, n3)


# ---------------------------------------------------------------- multi-table X54/X55: Synthea pairs, Lahman/MB
NEED = ("q", "A_d", "A_dt", "A_role", "A_rt", "melnik", "fd", "ucc_top1")


def multitable(f: Path) -> dict[str, tuple[float, float, int, int]]:
    by = defaultdict(dict)        # cell -> (seed, ds, sc) -> row
    sc_of = defaultdict(set)
    for r in csv.DictReader(open(f, encoding="utf-8")):
        by[r["cell"]][(r["seed"], r["dataset"], r["scenario"])] = r
        sc_of[r["cell"]].add((r["dataset"], r["scenario"]))
    out = {}
    for cell, rs in by.items():
        seeds = sorted({k[0] for k in rs})
        inc = [s for s in seeds if all((s, d, c) in rs and all(fnum(rs[(s, d, c)][x]) is not None for x in NEED)
                                       for d, c in sc_of[cell])]
        if not inc:
            continue
        cl = {s: {cell: [(float(rs[(s, d, c)]["q"]), float(rs[(s, d, c)]["melnik"])) for d, c in sc_of[cell]]} for s in inc}
        q, sf, n = panel_mean(cl)
        out[cell] = (q, sf, n, len(sc_of[cell]))
    return out


syn = multitable(STUDY / "results" / "multitable_fix" / "synthea" / "multitable_scores.csv")
q, s, n, ns = syn["SyntheaOMOP-Pair"]
put("Syn", q, s, n, ns, "results/multitable_fix/synthea/multitable_scores.csv (cell SyntheaOMOP-Pair; q, melnik)",
    "explore_multitable_ext.py: non-controls with q, the terms and the three SF arms on every scenario of the cell")
SYN_FULL = syn.get("SyntheaOMOP-Full")
lm = multitable(STUDY / "results" / "multitable_fix" / "indep2" / "multitable_scores.csv")
LM_CELLS = {c: lm[c] for c in ("Lahman", "MusicBrainz")}
put("LmM", mean(v[0] for v in LM_CELLS.values()), mean(v[1] for v in LM_CELLS.values()),
    min(v[2] for v in LM_CELLS.values()), sum(v[3] for v in LM_CELLS.values()),
    "results/multitable_fix/indep2/multitable_scores.csv (cells Lahman, MusicBrainz; q, melnik)",
    "explore_multitable_ext.py panel per cell (" + "/".join(str(v[2]) for v in LM_CELLS.values())
    + " matchers); macro over the two databases")


# ---------------------------------------------------------------- X82 wide panel (and the 15-matcher trajectory subset)
TS = (1, 2, 3, 5, 10)
x82 = defaultdict(list)
for r in csv.DictReader(open(STUDY / "results" / "x82" / "rows.csv", encoding="utf-8")):
    if r["corpus"] in X82_KEY:
        x82[r["corpus"]].append(r)
X82_CHECK = {}
for corpus, rs in x82.items():
    key = X82_KEY[corpus]
    scen = {(r["dataset"], r["scenario"]) for r in rs}
    by = defaultdict(list)
    for r in rs:
        if fnum(r.get("melnik_R")) is not None:
            by[r["matcher"]].append(r)
    cells, gainR = {}, []
    for m, mr in by.items():   # explore_x82_bf.per_matcher: melnik on every scenario of the corpus
        if {(r["dataset"], r["scenario"]) for r in mr} != scen:
            continue
        g = defaultdict(list)
        gr = defaultdict(list)
        for r in mr:
            g[r["group"]].append((float(r["q_J"]), float(r["melnik_J"])))
            gr[r["group"]].append(float(r["melnik_R"]) - float(r["q_R"]))
        cells[m] = g
        gainR.append(mean(mean(v) for v in gr.values()))
    q, s, n = panel_mean(cells)
    put(key, q, s, n, len(scen), f"results/x82/rows.csv (corpus {corpus}; q_J, melnik_J)",
        "explore_x82_bf.py (f) wide panel: non-controls with q and SF(melnik) on every scenario; macro over groups")
    # trajectory subset (explore_x82_bf.trajectory, arm melnik): every t and the converged run on every scenario
    keys = [f"melnik_t{t}" for t in TS] + ["melnik"]
    scen_t = {(r["dataset"], r["scenario"]) for r in rs if fnum(r.get("melnik_R")) is not None}
    byall = defaultdict(list)
    for r in rs:
        byall[r["matcher"]].append(r)
    inc = [m for m, mr in byall.items()
           if {(r["dataset"], r["scenario"]) for r in mr if all(fnum(r.get(f"{k}_R")) is not None for k in keys)} == scen_t]
    cell = defaultdict(list)
    for r in rs:
        if r["matcher"] in inc and (r["dataset"], r["scenario"]) in scen_t:
            cell[(r["group"], r["dataset"], r["scenario"])].append((float(r["q_J"]), float(r["melnik_J"])))
    groups = defaultdict(list)
    for (g, _, _), v in cell.items():
        groups[g].append((mean(a for a, _ in v), mean(b for _, b in v)))
    tq = mean(mean(a for a, _ in v) for v in groups.values())
    ts = mean(mean(b for _, b in v) for v in groups.values())
    X82_CHECK[key] = {"gainR": mean(gainR), "n15": len(inc), "q15": tq, "sf15": ts}


# ---------------------------------------------------------------- checks against the paper's macros
macros = {}
for fm in OUT.glob("*_numbers.tex"):
    for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", fm.read_text(encoding="utf-8")):
        macros[name] = val


def chk(label: str, mine: float, macro: str, tol: float | None = None):
    v = float(macros[macro])
    dec = len(macros[macro].split(".")[1]) if "." in macros[macro] else 0
    tol = tol if tol is not None else 0.5 * 10 ** -dec + 1e-9
    ok = abs(mine - v) <= tol
    CHECKS.append(f"| {label} | {mine:+.4f} | \\{macro} = {macros[macro]} | {'ok' if ok else 'MISMATCH'} |")
    return ok


ok = True
v = RESULTS["Val"]
ok &= chk("Val: mean d R_exp (layer 1, melnik)", VAL_DR, "rqOneMeanRexp")
ok &= chk("Val: mean d F1_Hr (layer 1, melnik)", VAL_DF, "rqOneMeanFHr")
exp_j = (float(macros["rqOneMeanRexp"]) + float(macros["rqOneMeanFHr"])) / 2
good = abs((v["sf"] - v["q"]) - exp_j) <= 0.0005 + 1e-9   # each macro is rounded to 0.001
ok &= good
CHECKS.append(f"| Val: absSf - absQ (J_tf) | {v['sf'] - v['q']:+.4f} | (\\rqOneMeanRexp + \\rqOneMeanFHr)/2 = {exp_j:+.4f} | "
              f"{'ok' if good else 'MISMATCH'} |")
ok &= chk("Val: drm SF-q R_exp", VAL_DR, "drmPlSfQRexpEst")
ok &= chk("Ind: absSf - absQ", RESULTS["Ind"]["sf"] - RESULTS["Ind"]["q"], "tcGainSfJ")
ok &= chk("Syn (pairs): absQ", RESULTS["Syn"]["q"], "mtSynPairQEst")
ok &= chk("Syn (pairs): absSf", RESULTS["Syn"]["sf"], "mtSynPairSfEst")
ok &= chk("Syn (pairs): absSf - absQ", RESULTS["Syn"]["sf"] - RESULTS["Syn"]["q"], "mtSynPairSfQEst")
ok &= chk("LmM: Lahman q", LM_CELLS["Lahman"][0], "mtLahmanQEst")
ok &= chk("LmM: MusicBrainz SF", LM_CELLS["MusicBrainz"][1], "mtMbSfEst")
ok &= chk("LmM: absSf - absQ (macro)", RESULTS["LmM"]["sf"] - RESULTS["LmM"]["q"], "mtMacroSfQEst")
ok &= chk(f"LmPl: P3 additive_d - melnik J ({P3_LM[1]} matchers; same rows, aggregation check)", P3_LM[0], "etwoPThreeEst")
ok &= chk(f"Ttd: P3 additive_d - melnik J ({P3_TTD[1]} matchers; aggregation check)", P3_TTD[0], "ttdPThreeEst")
for k, c in X82_CHECK.items():
    if f"wp{k}TrJConv" in macros:
        ok &= chk(f"{k}: SF - q in J_tf, {c['n15']}-matcher trajectory subset", c["sf15"] - c["q15"], f"wp{k}TrJConv")
    if f"wp{k}N" in macros:
        CHECKS.append(f"| {k}: wide-panel size | {RESULTS[k]['n']} | \\wp{k}N = {macros[f'wp{k}N']} | "
                      f"{'ok' if str(RESULTS[k]['n']) == macros[f'wp{k}N'] else 'MISMATCH'} |")
        ok &= str(RESULTS[k]["n"]) == macros[f"wp{k}N"]
if not ok:
    print("\n".join(CHECKS))
    raise SystemExit("a check failed; nothing written")

# x82 report's mean R_exp gain (wide panel), not a macro: parsed from results/x82/report.md
rep = (STUDY / "results" / "x82" / "report.md").read_text(encoding="utf-8")
inv = {v: k for k, v in X82_KEY.items()}
for k, c in X82_CHECK.items():
    m = re.search(r"\| " + re.escape(inv[k]) + r" \| \d+ \| [^|]+ \| ([+-]\d\.\d+) \|", rep)
    if m:
        good = abs(c["gainR"] - float(m.group(1))) < 5e-5
        ok &= good
        CHECKS.append(f"| {k}: mean R_exp gain, wide panel | {c['gainR']:+.4f} | x82 report.md (f) = {m.group(1)} | "
                      f"{'ok' if good else 'MISMATCH'} |")
if not ok:
    print("\n".join(CHECKS))
    raise SystemExit("a check failed; nothing written")

# ---------------------------------------------------------------- write
srcs = sorted({r["src"].split(" (")[0] for r in RESULTS.values()})
tex = ["% Generated by make_abs_numbers.py from " + ", ".join(srcs)
       + " (results/sf_study/); EXPLORATORY where the sources are; do not edit.",
       "% \\absQ<Key>: mean J_tf of the corpus's initial-matcher panel alone; \\absSf<Key>: of SF as published after it.",
       "% Readable regime; macro over families/databases, then mean over non-control matchers (see abs_numbers_report.md)."]
for k in ORDER:
    if k in RESULTS:
        tex.append(f"\\newcommand{{\\absQ{k}}}{{{RESULTS[k]['q']:.2f}}}")
        tex.append(f"\\newcommand{{\\absSf{k}}}{{{RESULTS[k]['sf']:.2f}}}")
(OUT / "abs_numbers.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

md = ["# Absolute J_tf levels per corpus (make_abs_numbers.py)", "",
      "Readable regime. absQ = mean J_tf of the initial matchers alone; absSf = SF as published (arm melnik) after them. "
      "Per matcher: mean within family/database, mean over families/databases; then mean over the panel's non-control "
      "matchers. Generated; do not edit.", "",
      "| key | corpus | absQ | absSf | SF - q | matchers | scenarios | source | panel |", "|---|---|---|---|---|---|---|---|---|"]
for k in ORDER:
    if k in RESULTS:
        r = RESULTS[k]
        md.append(f"| {k} | {NAME[k]} | {r['q']:.4f} | {r['sf']:.4f} | {r['sf'] - r['q']:+.4f} | {r['n']} | {r['scen']} | "
                  f"{r['src']} | {r['panel']} |")
md += ["", "Not emitted: none of the 14 corpora lacks stored per-scenario absolute scores for q and SF as published.", "",
       "## Alternative panels (not emitted)", "",
       "| corpus | panel | absQ | absSf |", "|---|---|---|---|"]
for k, c in X82_CHECK.items():
    md.append(f"| {k} | {c['n15']}-matcher X82(b) trajectory panel (= X81's panel on the external corpora; wp Tr* macros) | {c['q15']:.4f} | {c['sf15']:.4f} |")
if SYN_FULL:
    md.append(f"| Syn | SyntheaOMOP-Full (whole schemas, {SYN_FULL[3]} scenarios, {SYN_FULL[2]} matchers) | "
              f"{SYN_FULL[0]:.4f} | {SYN_FULL[1]:.4f} |")
for c, v in LM_CELLS.items():
    md.append(f"| LmM | cell {c} ({v[3]} scenarios, {v[2]} matchers) | {v[0]:.4f} | {v[1]:.4f} |")
md += ["", "## Checks against the paper's macros (output/*_numbers.tex)", "",
       "| check | this script | paper | result |", "|---|---|---|---|", *CHECKS, "",
       "Notes: Valentine has no J_tf macro for SF's mean change; its R_exp and F1_Hr components are checked against "
       "\\rqOneMeanRexp and \\rqOneMeanFHr, and absSf - absQ equals their mean by construction. The P3 checks validate "
       "the E2 rows, panel and aggregation (same file and scorer rules), not q itself; no macro states SF - q on "
       "Lahman/MusicBrainz single-table or T2D. On the X82 corpora the wide panel is the one RQ1's rho uses (\\wp<C>N); "
       "the 15-matcher subset reproduces \\wp<C>TrJConv. On HDXSM the wide panel covers all scenarios SF as published ran "
       "on."]
(OUT / "abs_numbers_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(tex))
print("\n".join(CHECKS))
