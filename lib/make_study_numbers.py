"""Generate the paper's RQ1 figure and numbers from the study's results, so the draft cannot drift.

    figures/rq1_gain_data.csv   one row per initial matcher: its quality and SF's changes, plot group
    figures/rq1_gain.tex        Fig. 2, two pgfplots panels: SF's change in expected Recall@GT (a) and
                                in tie-free Hungarian F1 (b), against the expected Recall@GT of the
                                initial matcher alone
    figures/rq1_numbers.tex     the numbers the introduction and the results quote (rqOne...)
    (Fig. 1, the four-axes diagram, is hand-written in figures/teaser_axes.tex.)

Reads results/sf_study/results/<regime>549/scores_tiefree.csv (the seed study on the 549 Valentine scenarios,
assembled by experiments/sf_study/reference/assemble_549.py from score_seeds_tiefree.py's output), and the family of
each initial matcher from make_corpus_tables.META. Only initial matchers scored on all 549 scenarios enter, so every
point covers the same scenarios.
Controls (oracle variants, constant, random and the control arms of the learned matchers) are
drawn as reference points, not as matchers, and are left out of every correlation.

The tie-free metrics are those of Amendment 2 (preregistration/PREREGISTRATION.md):
expected Recall@GT and MRR under random tie-breaking, and Hungarian F1 with ties broken at random.

    python make_study_numbers.py [--regime plain]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict

from make_corpus_tables import META, STUDY
from paths import BASELINES, OUT


# Plot groups over the matcher families of Table 1. Colours are the first four slots of the
# validated categorical palette (dataviz skill); marker shapes repeat the grouping for
# greyscale print and colour-vision deficiency.
# Slots 5-8 of the palette (violet, red, green, magenta), NOT slots 1-4: those mark the four axes
# in Fig. 1, and a matcher group must not borrow an axis colour. Validated with the dataviz
# validator: passes within the figure (worst adjacent pair red/green, legal with the marker shapes).
GROUPS = [
    ("label", "Labels", ("str", "lex", "emb"), "4a3aa7", "*"),
    ("inst", "Instances", ("meta", "ovl", "dist", "fmt"), "e34948", "square*"),
    ("dep", "Dependencies, graphs", ("dep", "graph"), "008300", "triangle*"),
    ("comb", "Composite, learned", ("comp", "learn", "types", "llm", "unc"), "e87ba4", "diamond*"),
]
CONTROL_ARMS = {"semstruct_lm_only", "rema_shuffled", "embedding_random"}
# Results known to be invalid, left out of every figure and number until they are rerun.
# (magneto was excluded until its patched rerun; the plain rerun was imported 2026-10-02 20:40.)
INVALID = {"random"}
FAMILIES = (("ChEMBL", "ChEMBL"), ("OpenData", "OpenData"), ("TPC", "TPCDI"))

# The Valentine base: the 549 valid scenarios of the official release (381 distinct scenarios of the earlier 382-case
# copy + 168 new; results/baselines/valentine549/cohort-549.json), FD profiles of the fixed HyFD. The seed study and
# every Valentine score file built from it are assembled per regime into <regime>549 folders
# (results/sf_study/results/valentine549/ASSEMBLY.md); registered inputs on the 382 keep their folders.
BASE_N = 549
_V549_NAME = {"plain": "plain549", "md5": "md549"}


def vreg(regime: str) -> str:
    """The folder name of a Valentine regime's 549 inputs (plain549, md549); other corpora keep their name."""
    return _V549_NAME.get(regime, regime)


def nfull(regime: str) -> int:
    """Scenarios a configuration must cover to enter a Valentine panel of this regime."""
    return BASE_N if regime in _V549_NAME else 382


def group_of(seed: str) -> str:
    fam = META[seed][0]
    if fam == "ctrl" or seed in CONTROL_ARMS:
        return "ctrl"
    return next(g for g, _, fams, _, _ in GROUPS if fam in fams)


def load(regime: str) -> tuple[list[dict], int]:
    """The seed study (standard configuration), per-scenario rows of scores_tiefree.csv, macro over
    families (see macro_by_seed). sbert_e5 and semstruct_lm_only are the same frozen encoder; one point."""
    rows = csv.DictReader(open(STUDY / "results" / vreg(regime) / "scores_tiefree.csv", encoding="utf-8"))
    return load_rows(rows, nfull(regime))


METRICS7 = ("R_exp", "R_pess", "F1_Hr", "F1_H", "F1_M", "MRR_exp", "MRR_pess")
# Aggregation as stated in the paper (setup): macro mean, first over the scenarios of each data family,
# then over the families; families with fewer than MIN_FAMILY scenarios (Wikidata, 2) are left out of the
# average, as in layer-1 scorer. Changed 2026-10-03 after an external review: until then the
# seed-study summaries were plain scenario means.
MIN_FAMILY = 5
PROTOCOL = BASELINES / "reports" / "audited_md5_fused_20260904" / "protocol.json"
_FAMILY: dict[tuple[str, str], str] = {}


def family_of(dataset: str, scenario: str) -> str:
    if not _FAMILY:
        for c in json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]:
            _FAMILY[(c["dataset"], c["scenario"])] = c["family"]
        # the 168 new scenarios: family from their manifests (assemble_549.py's case list)
        for c in csv.DictReader(open(STUDY / "results" / "valentine549" / "cases.csv", encoding="utf-8")):
            _FAMILY[(c["dataset"], c["scenario"])] = c["family"]
    return _FAMILY.get((dataset, scenario), dataset.split("-")[0])


def macro_by_seed(rows, cols: list[str]) -> dict[str, dict[str, float]]:
    """Per initial matcher: macro mean over families (>= MIN_FAMILY scenarios) of each column, plus
    'n', the number of scenarios. Rows need seed, dataset, scenario (or family)."""
    acc: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    count: dict[str, int] = defaultdict(int)
    for r in rows:
        fam = r.get("family") or family_of(r["dataset"], r["scenario"])
        count[r["seed"]] += 1
        for c in cols:
            acc[r["seed"]][fam][c].append(float(r[c]))
    out = {}
    for s, fams in acc.items():
        big = [f for f, d in fams.items() if len(next(iter(d.values()))) >= MIN_FAMILY]
        out[s] = {c: sum(sum(fams[f][c]) / len(fams[f][c]) for f in big) / len(big) for c in cols}
        out[s]["n"] = count[s]
    return out


def layer1_rows(regime: str, arm: str) -> list[dict]:
    """Per-scenario rows of one layer-1 arm (score_layer1.py), or [] if not scored."""
    f = STUDY / "tuning" / "results" / f"layer1_{vreg(regime)}_scores" / "scores.csv"
    if not f.is_file():
        return []
    rows = [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["arm"] == arm]
    # Disclosed replacement (X30): the data-type matcher's rows come from its rerun on the NaN-free seeds
    # (layer1nan_<regime>, seed datatype_nanfix: NaN cells set to 0, every finite cell unchanged).
    fn = STUDY / "tuning" / "results" / f"layer1nan_{vreg(regime)}_scores" / "scores.csv"
    if fn.is_file():
        nan = [{**r, "seed": "datatype"} for r in csv.DictReader(open(fn, encoding="utf-8"))
               if r["arm"] == arm and r["seed"] == "datatype_nanfix"]
        if nan:
            rows = [r for r in rows if r["seed"] != "datatype"] + nan
    return rows


def load_rows(rows, full: int = BASE_N) -> tuple[list[dict], int]:
    """Points as in load(), from per-scenario rows (macro mean over families per initial matcher).
    full: scenarios a matcher must cover (the Valentine base)."""
    cols = [f"{p}_{m}" for p in ("seed", "sf") for m in METRICS7]
    agg = macro_by_seed(rows, cols)
    out = []
    for s, a in agg.items():
        if a["n"] != full or s not in META or s in INVALID:
            continue
        out.append({"seed": s, "group": group_of(s), "x": a["seed_R_exp"], "x_pess": a["seed_R_pess"],
                    **{f"base_{m}": a[f"seed_{m}"] for m in METRICS7},
                    **{f"d_{m}": a[f"sf_{m}"] - a[f"seed_{m}"] for m in METRICS7}})
    return [o for o in out if o["seed"] != "semstruct_lm_only"], full


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def per_family(regime: str, matchers: set[str], rows: list[dict] | None = None) -> dict[str, float]:
    """rho(expected Recall@GT alone, SF's change) within each large data family, per metric.
    rows: per-scenario rows to use (layer 1); default the seed study's scores_tiefree.csv."""
    acc: dict[tuple, list[dict]] = {}
    src = rows if rows is not None else csv.DictReader(open(STUDY / "results" / vreg(regime) / "scores_tiefree.csv", encoding="utf-8"))
    for r in src:
        if r["seed"] in matchers:
            acc.setdefault((r["seed"], r["dataset"].split("-")[0]), []).append(r)
    out = {}
    for fam, key in FAMILIES:
        for metric, mkey in (("R_exp", "Rexp"), ("F1_Hr", "FHr")):
            xs, ys = [], []
            for s in sorted(matchers):
                rs = acc.get((s, fam), [])
                if rs:
                    x = sum(float(r["seed_R_exp"]) for r in rs) / len(rs)
                    d = sum(float(r[f"sf_{metric}"]) - float(r[f"seed_{metric}"]) for r in rs) / len(rs)
                    xs.append(x)
                    ys.append(d)
            out[f"Rho{mkey}{key}"] = spearman(xs, ys)
    return out


def robust(regime: str) -> dict[str, str]:
    """EXPLORATORY R1: Melnik et al.'s robustness test (multiplicative noise of +-p on the initial
    similarities, or flattening), with the perturbed input q' scored next to SF(q').
    Reads r1_robust_scores.csv and r1_margins.csv (explore_robust.py)."""
    rows = list(csv.DictReader(open(STUDY / "results" / regime / "r1_robust_scores.csv", encoding="utf-8")))
    runs = defaultdict(list)
    for r in rows:
        runs[(r["seed"], r["config"], r["variant"])].append(r)

    def jt(rs: list[dict], n: str) -> float:
        # macro over families (screen scenarios), as everywhere in the paper
        a = macro_by_seed([{**r, "seed": "x"} for r in rs], [f"{n}_R_exp", f"{n}_F1_Hr"])["x"]
        return (a[f"{n}_R_exp"] + a[f"{n}_F1_Hr"]) / 2

    lvl = defaultdict(list)
    for (s, c, v), rs in runs.items():
        lvl[(s, c, v.split("_")[0])].append((jt(rs, "q"), jt(rs, "sf")))
    agg = {k: (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v)) for k, v in lvl.items()}
    noisy = [f"p{p}" for p in (25, 50, 100, 200, 400)]
    seeds = ("prefix_suffix", "levenshtein", "pmf")

    def gap(c: str) -> list[float]:
        return [agg[(s, c, l)][1] - agg[(s, c, l)][0] for s in seeds for l in noisy]

    def loss(s: str, l: str) -> int:
        return round(100 * (1 - agg[(s, "ref", l)][0] / agg[(s, "ref", "p0")][0]))

    flat_q = [loss(s, "flat") for s in seeds]
    rec = {c: [agg[(s, c, "flat")][1] - agg[(s, c, "flat")][0] for s in seeds] for c in ("ref", "std")}
    mg = {r["seed"]: r for r in csv.DictReader(open(STUDY / "results" / regime / "r1_margins.csv", encoding="utf-8"))}
    return {"RefMax": f"{max(abs(x) for x in gap('ref')):.3f}",
            "StdLo": f"{min(gap('std')):+.3f}", "StdHi": f"{max(gap('std')):+.3f}",
            "LossPS": loss("prefix_suffix", "p100"), "LossLev": loss("levenshtein", "p100"),
            "LossPmf": loss("pmf", "p25"),
            "FlatLossLo": min(flat_q), "FlatLossHi": max(flat_q),
            "FlatRefLo": f"{min(rec['ref']):+.2f}", "FlatRefHi": f"{max(rec['ref']):+.2f}",
            "FlatStdLo": f"{min(rec['std']):+.2f}", "FlatStdHi": f"{max(rec['std']):+.2f}",
            "MarginPmf": f"{float(mg['pmf']['median_rel_margin']):.2f}",
            "MarginPS": f"{float(mg['prefix_suffix']['median_rel_margin']):.2f}",
            "MarginLev": f"{float(mg['levenshtein']['median_rel_margin']):.2f}",
            "SmallPmf": round(100 * float(mg["pmf"]["share_margin_0_to_10pct"])),
            "N": len({(r["dataset"], r["scenario"]) for r in rows})}


def noise_ref() -> dict[str, str]:
    """EXPLORATORY X84 (layer3b/score_x84.py): Melnik et al.'s noise test with SF's full reference configuration
    (literal similarities included), 20 scenarios each of Valentine, HDXSM and GOBY, 3 matchers.
    Reads tuning/results/x84_noise_scores/summary.csv. Merged into robust_numbers.tex with prefix rob (keys Lit*)."""
    f = STUDY / "tuning" / "results" / "x84_noise_scores" / "summary.csv"
    if not f.is_file():
        return {}
    v = {(r["corpus"], r["arm"], r["level"], r["stat"]): float(r["estimate"])
         for r in csv.DictReader(open(f, encoding="utf-8"))}
    cs = sorted({c for c, a, _, _ in v if a == "melnik"})
    gain = [v[(c, "melnik", "p0", "delta_J")] for c in cs]
    hi = [v[(c, "melnik", l, "delta_J")] for c in cs for l in ("p100", "p200")]
    ab = [v[(c, "melnik", "p100", "absorb_J")] for c in cs]
    fl = [v[(c, "melnik", "flat", "delta_J")] for c in cs]
    n = {r["scenarios"] for r in csv.DictReader(open(f, encoding="utf-8"))}
    return {"LitGainLo": f"{min(gain):+.3f}", "LitGainHi": f"{max(gain):+.3f}",
            "LitNoisyLo": f"{min(hi):+.3f}", "LitNoisyHi": f"{max(hi):+.3f}",
            "LitAbsorbLo": f"{min(ab):+.3f}", "LitAbsorbHi": f"{max(ab):+.3f}",
            "LitFlatLo": f"{min(fl):+.2f}", "LitFlatHi": f"{max(fl):+.2f}",
            "LitN": "/".join(sorted(n)), "LitCorpora": str(len(cs))}


XC = {"gdcsm": "Gdc", "hdxsm": "Hdx", "goby": "Goby", "madi_indep": "Madi", "madi_products": "MadiP", "mimic": "Mimic",
      "final_indep": "Indep", "final_controlled": "Ctl", "final_controlled_plain": "CtlPl",
      "final_controlled_opaque": "CtlOp", "valentine_plain": "Val", "t2d": "Ttd", "ctl_plain": "LmPl",
      "synthea": "Syn"}
INDEP4 = ("Gdc", "Hdx", "Goby", "Madi")   # the independently produced single-table corpora


def _s3(v: float) -> str:
    return f"{v:+.3f}"


def external_x81() -> dict[str, str] | None:
    """EXPLORATORY X81 (explore_x81.py): Amendment 12's frozen terms on the external corpora. Parses
    results/sf_study/results/x81/report.md. Prefix xt. Per corpus C (Gdc, Hdx, Goby, Madi, MadiP):
      C{Ad,RoleKey,RoleFd,Fd,Tc}{Est,Lo,Hi}: A_d − SF(melnik), A_role − SF(ucc_top1), A_role − SF(ucc1_fd_topk),
      R_exp SF(fd) − SF(melnik), A_dt − SF(melnik); CScen; Matchers.
      Mimic: MimicTcEst, MimicTcPos, MimicTcN (descriptive, one scenario).
      Ranges over the four independent corpora: IndAdMin/Max, IndRoleMin/Max (E2a and E2b over Gdc, Goby, Madi)."""
    f = STUDY / "results" / "x81fix" / "report.md"   # the FD arms rerun on the fixed FD profiles
    if not f.is_file():
        return None
    key = {"E1": "Ad", "E2a": "RoleKey", "E2b": "RoleFd", "E4": "Fd", "E5'": "Tc"}
    out, raw, cur, cl = {}, {}, None, False
    for line in f.read_text(encoding="utf-8").splitlines():
        m = re.match(r"## (\S+)", line)
        if m:
            cur, cl = XC.get(m.group(1)), False
            continue
        if cur is None:
            continue
        m = re.match(r"Cluster-aware intervals \((\d+) clusters", line)
        if m:   # X86 clustering (tables shared across scenarios): {C}{K}Cl{Lo,Hi}, {C}Clusters; no interval below 5 clusters
            cl, out[f"{cur}Clusters"] = True, m.group(1)
            continue
        if cl:
            m = re.match(r"\| (E\S+) .*? \| ([+-]\d\.\d+) \[([+-]\d\.\d+), ([+-]\d\.\d+)\] \|", line)
            if m and m.group(1) in key:
                k = key[m.group(1)]
                out[f"{cur}{k}ClLo"], out[f"{cur}{k}ClHi"] = _s3(float(m.group(3))), _s3(float(m.group(4)))
            continue
        m = re.match(r"- scenarios (\d+);.*matchers included (\d+)", line)
        if m:
            out[f"{cur}Scen"], out["Matchers"] = m.group(1), m.group(2)
        m = re.match(r"\| (E\S+) .*? \| ([+-]\d\.\d+) \[([+-]\d\.\d+), ([+-]\d\.\d+)\] \|", line)
        if m and m.group(1) in key:
            k = key[m.group(1)]
            raw[(cur, k)] = float(m.group(2))
            out[f"{cur}{k}Est"], out[f"{cur}{k}Lo"], out[f"{cur}{k}Hi"] = (_s3(float(m.group(i))) for i in (2, 3, 4))
        m = re.match(r"- E5 \(descriptive, n = 1\): .* ([+-]\d\.\d+); positive for (\d+) of (\d+)", line)
        if m:
            out["MimicTcEst"], out["MimicTcPos"], out["MimicTcN"] = _s3(float(m.group(1))), m.group(2), m.group(3)
    cap = STUDY / "results" / "hdxsm_fd_cap.json"
    if cap.is_file():   # a request: the FD-edge cap on HDXSM
        j = json.loads(cap.read_text(encoding="utf-8"))
        out["HdxFdScen"], out["HdxFdExcl"] = str(j["pairs_fd_arms"]), str(j["pairs_excluded"])
        out["HdxFdUnfinishedMin"] = str(round(j["unfinished_observation"]["seconds_without_finishing"] / 60))
        out["HdxFdCapT"] = f"{j['T_product']:,}".replace(",", "{,}")
        out["HdxFdCapSec"], out["HdxFdProbeMaxSec"] = str(j["T_pair_seconds"]), str(j["probe_max_seconds"])
    ad = [raw[(c, "Ad")] for c in INDEP4 if (c, "Ad") in raw]
    ro = [raw[(c, k)] for c in ("Gdc", "Goby", "Madi") for k in ("RoleKey", "RoleFd") if (c, k) in raw]
    if ad:
        out["IndAdMin"], out["IndAdMax"] = _s3(min(ad)), _s3(max(ad))
    # counts over the four independent corpora: A_d's cluster interval below 0; the role term ahead of the key graph
    out["IndN"] = str(sum((c, "Ad") in raw for c in INDEP4))
    out["IndAdClearN"] = str(sum(f"{c}AdClHi" in out and float(out[f"{c}AdClHi"]) < 0 for c in INDEP4))
    out["IndRoleAheadN"] = str(sum(raw.get((c, "RoleKey"), 0) > 0 for c in INDEP4))
    if ro:
        out["IndRoleMin"], out["IndRoleMax"] = _s3(min(ro)), _s3(max(ro))
    return out


def wide_x82() -> dict[str, str] | None:
    """EXPLORATORY X82 (f), (b) (explore_x82_bf.py): RQ1 with the wide panel, and the iteration trajectory, on the new
    corpora. Parses results/sf_study/results/x82/report.md. Prefix wp.
      C{Rho,RhoLo,RhoHi,N} per corpus; Pooled{Est,Lo,Hi,N,Corpora,Neg};
      trajectory (melnik, R_exp gain over q): C{TrOne,TrTwo,TrThree,TrConv}; ShareTwoMin/Max over the single-table corpora
      whose converged gain is at least 0.005."""
    f = STUDY / "results" / "x82" / "report.md"
    if not f.is_file():
        return None
    out, sec, shares = {}, None, []
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith("## (f)"):
            sec = "f"
        elif line.startswith("### melnik"):
            sec = "bm"
        elif line.startswith("### fd"):
            sec = "bf"
        m = re.match(r"Pooled over (\d+) single-table corpora .*: (\d+) matchers, ρ = ([+-]\d\.\d+) \[([+-]\d\.\d+), ([+-]\d\.\d+)\]; "
                     r"negative sign in (\d+) of", line)
        if m:
            out["PooledCorpora"], out["PooledN"] = m.group(1), m.group(2)
            out["PooledEst"], out["PooledLo"], out["PooledHi"] = m.group(3), m.group(4), m.group(5)
            out["PooledNeg"] = m.group(6)
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if sec == "f" and len(c) == 4 and c[0] in XC:
            m = re.match(r"([+-]\d\.\d+) \[([+-]\d\.\d+), ([+-]\d\.\d+)\]", c[2])
            if m:
                k = XC[c[0]]
                out[f"{k}Rho"], out[f"{k}RhoLo"], out[f"{k}RhoHi"], out[f"{k}N"] = m.group(1), m.group(2), m.group(3), c[1]
        if sec == "bm" and len(c) >= 10 and c[0] in XC:
            k = XC[c[0]]
            r = [float(x.split("/")[0]) for x in c[2:8]]
            out[f"{k}TrOne"], out[f"{k}TrTwo"], out[f"{k}TrThree"], out[f"{k}TrConv"] = (_s3(r[i]) for i in (0, 1, 2, 5))
            out[f"{k}TrFive"], out[f"{k}TrTen"] = _s3(r[3]), _s3(r[4])
            j = [float(x.split("/")[1]) for x in c[2:8]]   # J_tf gains at t = 1, 2, 3, 5, 10 and converged
            for name, v in zip(("One", "Two", "Three", "Five", "Ten", "Conv"), j):
                out[f"{k}TrJ{name}"] = _s3(v)
            if k not in ("Indep", "Mimic") and abs(r[5]) >= 0.005:
                shares.append(float(c[8].split()[0]))
    if shares:
        out["ShareTwoMin"], out["ShareTwoMax"] = f"{min(shares):.2f}", f"{max(shares):.2f}"
    return out


def characterise_x85() -> dict[str, str] | None:
    """EXPLORATORY X85 (explore_x85_characterise.py, explore_x85_leak*.py). Prefix xc.
      Const{H,R}C: constant matcher Hungarian F1 with index-order / random ties per corpus (Val, Ttd, Gdc, Hdx, Goby, Madi, MadiP);
      HdxSrc, HdxTgt, HdxPerTable: distinct HDXSM tables by content; LeakCases, LeakAll: selecting-arm outputs that change
      under a column permutation; LeakValMax: largest |ΔJ_tf| on Valentine; LeakFloatMax: largest output change of the
      position-free arms."""
    f = STUDY / "results" / "x85" / "report.md"
    if not f.is_file():
        return None
    out, sec = {}, None
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            sec = line[3:5]
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if sec == "1." and c[0] == "hdxsm" and len(c) >= 4:
            a, b = c[2].split(" / ")
            out["HdxSrc"], out["HdxTgt"], out["HdxPerTable"] = a, b, c[3]
        if sec == "2." and len(c) == 6 and c[0] in XC:
            k = XC[c[0]]
            out[f"ConstH{k}"], out[f"ConstR{k}"] = f"{float(c[3]):.2f}", f"{float(c[4]):.2f}"
    # the other corpora's constant matcher (results/x85/corpus_table.csv, same computation; means over scenarios)
    ct = STUDY / "results" / "x85" / "corpus_table.csv"
    if ct.is_file():
        for r in csv.DictReader(open(ct, encoding="utf-8")):
            k = XC.get(r["corpus"])
            if k and f"ConstH{k}" not in out:
                out[f"ConstH{k}"], out[f"ConstR{k}"] = f"{float(r['const_F1_H']):.2f}", f"{float(r['const_F1_Hr']):.2f}"
    # Valentine's constant matcher on the 549: the seed study's constant rows (same metrics and tie orders as X85's
    # rescoring; mean over scenarios, rounded to 3 decimals as X85's report before the 2-decimal macro)
    cr = [r for r in csv.DictReader(open(STUDY / "results" / "plain549" / "scores_tiefree.csv", encoding="utf-8"))
          if r["seed"] == "constant"]
    if len(cr) == BASE_N:
        for m, kk in (("F1_H", "ConstHVal"), ("F1_Hr", "ConstRVal")):
            v = sum(float(r["seed_" + m]) for r in cr) / len(cr)
            out[kk] = "{:.2f}".format(float("{:.3f}".format(v)))
    else:
        out.pop("ConstHVal", None), out.pop("ConstRVal", None)
    # X87 (C): permuted columns on HDXSM and MaDI; the single-key graph's change in J_tf (permuted − original), and the
    # share of cells that change: OrdKey{C}, OrdKeyShare{C}
    s87 = STUDY / "tuning" / "results" / "x87_scores" / "summary.csv"
    if s87.is_file():
        for r in csv.DictReader(open(s87, encoding="utf-8")):
            if r["arm"] == "ucc_top1" and r["corpus"] in XC:
                k = XC[r["corpus"]]
                out[f"OrdKey{k}"] = _s3(float(r["mean_dJ"]))
                out[f"OrdKeyShare{k}"] = str(round(100 * float(r["share_dJ"])))
    # X87 (C): approximate FD discovery under permuted columns (x87_fd_diff.py), and SF with FD edges on the re-profiled
    # tables (x87rp): ProfShare{C} % of FDs that differ; ProfSides{C}/ProfSidesAll{C}; ProfFdCells{C} % of cells of SF with
    # FD edges that change; ProfFdMean{C} mean change in J_tf
    fdd = STUDY / "tuning" / "results" / "x87_scores" / "fd_diff_summary.csv"
    if fdd.is_file():
        for r in csv.DictReader(open(fdd, encoding="utf-8")):
            k = {"hdxsm": "Hdx", "madi": "Madi"}.get(r["corpus"])
            if k:
                out[f"ProfShare{k}"] = f"{100 * float(r['share_differing']):.1f}"
                out[f"ProfSides{k}"], out[f"ProfSidesAll{k}"] = r["sides_differing"], r["sides"]
    rp = STUDY / "tuning" / "results" / "x87rp_scores" / "summary.csv"
    if rp.is_file():
        for r in csv.DictReader(open(rp, encoding="utf-8")):
            if r["arm"] == "fd" and r["corpus"] == "hdxsm":
                out["ProfFdCellsHdx"] = str(round(100 * float(r["share_dJ"])))
                v = abs(float(r["mean_dJ"]))   # an upper bound on |mean change|, at least 0.001
                out["ProfFdMeanBoundHdx"] = f"{max(math.ceil(v * 1000) / 1000, 0.001):.3f}"
    lk = STUDY / "results" / "x85" / "leak.md"
    le = STUDY / "results" / "x85" / "leak_effect.md"
    if lk.is_file():
        rows = [[x.strip() for x in l.strip().strip("|").split("|")] for l in lk.read_text(encoding="utf-8").splitlines()]
        arms = {r[0]: r for r in rows if len(r) == 5 and r[0] in ("melnik", "fd", "ucc_top1", "ucc1_fd_topk")}
        out["LeakCases"], out["LeakAll"] = arms["ucc_top1"][3], arms["ucc_top1"][1]
        e = max(float(arms[a][2]) for a in ("melnik", "fd"))
        out["LeakFloatExp"] = str(math.ceil(math.log10(e)))
    if le.is_file():
        vals = [float(r[4]) for r in ([x.strip() for x in l.strip().strip("|").split("|")]
                                      for l in le.read_text(encoding="utf-8").splitlines())
                if len(r) == 7 and r[0] == "valentine"]
        out["LeakValMax"] = f"{max(vals):.4f}"
    return out


def properties_x81() -> dict[str, str] | None:
    """EXPLORATORY X77/X81 properties (explore_x81_props.py): medians over scenarios of value overlap and name similarity of
    true pairs (and non-pairs), and the numeric-column share. Parses results/sf_study/results/x81/benchmark_profile_x81.md.
    Prefix pr: {Ovl,Name,NameNon,Num}C."""
    f = STUDY / "results" / "x81_549" / "benchmark_profile_x81.md"
    if not f.is_file():
        return None
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 13 and c[0] in XC:
            k = XC[c[0]]
            o, n = c[5].split(" / "), c[6].split(" / ")
            out[f"Ovl{k}"], out[f"Name{k}"], out[f"NameNon{k}"], out[f"Num{k}"] = o[0], n[0], n[1], c[8]
    return out


def knobs_x79() -> dict[str, str] | None:
    """EXPLORATORY X79 (b) (layer3b/x79_knobs.py): controlled knobs. Parses tuning/results/x79_knobs_scores/report.md.
    Prefix kn, per base corpus V (Valentine) / C (final_controlled):
      {V,C}FdBase: fd − melnik in R_exp at the base level; {V,C}{Rename,Recode,Thin}: its change at full renaming (K3p100),
      full value recoding (K2p100) and FDs thinned to 25% (K4f25); {V,C}RenameShare: % of the base cost recovered by renaming;
      RecodeAdMin/Max: the change of A_d − SF under recoding (K2, all levels, both corpora)."""
    f = STUDY / "tuning" / "results" / "x79_knobs_scores" / "report.md"
    if not f.is_file():
        return None
    out, cur, ad = {}, None, []
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith("## valentine"):
            cur = "V"
        elif line.startswith("## final_controlled"):
            cur = "C"
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if cur and len(c) == 6:
            if c[0] == "base level":
                base = float(c[3])
                out[f"{cur}FdBase"] = _s3(base)
            name = {"K3p100": "Rename", "K2p100": "Recode", "K4f25": "Thin"}.get(c[0])
            if name:
                v = float(c[3].split()[0])
                out[f"{cur}{name}"] = _s3(v)
                if name == "Rename":
                    out[f"{cur}RenameShare"] = str(round(100 * v / abs(base)))
            if c[0].startswith("K2p"):
                ad.append(float(c[1].split()[0]))
    if ad:
        out["RecodeAdMin"], out["RecodeAdMax"] = _s3(min(ad)), _s3(max(ad))
    return out


def c_x82_x83() -> dict[str, str] | None:
    """EXPLORATORY X82 (classes, parallel edges, E2x/R8) and X83 (additive class terms) on the new corpora.
    Parses the contrasts.csv of tuning/results/{x82_classfull_scores_dev1_cap, x82_parallel_scores_dev1,
    x82_e2x_scores_dev1_cap, x83_class_additive}. Prefix cx; per corpus C, J_tf unless noted:
      ShapeC, SemC: class nodes − SF as published; ShapeAddC, SemAddC: additive class term − class nodes (X83);
      SplitC: parallel FD edges kept (SPLIT) − default; TopC: top-1 scope − SF as published;
      ShapeConfirm/SemConfirm: independent corpora where class nodes help (interval above 0), of IndN."""
    base = STUDY / "tuning" / "results"
    # arms without FD edges use all 199 HDXSM pairs (the uncapped files); the FD arms (parallel) the 193 below the cap
    spec = {"x82_classfull_scores_dev1": {"melnik_shape − melnik": "Shape", "melnik_semtype − melnik": "Sem"},
            "x83_class_additive": {"A_p − melnik_shape": "ShapeAdd", "A_s − melnik_semtype": "SemAdd"},
            "x82_parallel_scores_dev1": {"fd_split − fd": "Split"},
            "x82_e2x_scores_dev1": {"melnik_top1 − melnik": "Top"}}
    out, conf = {}, defaultdict(int)
    indep = ("hdxsm", "gdcsm", "madi_indep", "goby", "final_indep")
    for d, names in spec.items():
        f = base / d / "contrasts.csv"
        if not f.is_file():
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["metric"] != "J" or r["contrast"] not in names or r["corpus"] not in XC:
                continue
            k = names[r["contrast"]]
            out[f"{k}{XC[r['corpus']]}"] = _s3(float(r["estimate"]))
            if r["corpus"] in indep and k in ("Shape", "Sem") and r["lo"] and float(r["lo"]) > 0:
                conf[k] += 1
            if r["corpus"] in indep:   # signs over the independent corpora (MaDI has too few clusters for an interval)
                conf[f"{k}Pos"] += float(r["estimate"]) > 0
                conf[f"{k}Neg"] += float(r["estimate"]) < 0
    out["ShapeConfirm"], out["SemConfirm"], out["IndN"] = str(conf["Shape"]), str(conf["Sem"]), str(len(indep))
    for k in ("ShapePos", "SemPos", "ShapeAddNeg", "SemAddNeg"):
        out[k] = str(conf[k])
    # X86 (C): cluster-aware intervals; ShapeAdd{C}Cl{Lo,Hi}; ShapeAddClNeg = independent corpora whose interval lies below 0
    f86 = base / "x86_cluster_ci" / "contrasts.csv"
    if f86.is_file():
        neg = 0
        for r in csv.DictReader(open(f86, encoding="utf-8")):
            if r["contrast"] == "A_p − melnik_shape" and r["metric"] == "J" and r["corpus"] in XC and r["lo"]:
                k = XC[r["corpus"]]
                out[f"ShapeAdd{k}ClLo"], out[f"ShapeAdd{k}ClHi"] = _s3(float(r["lo"])), _s3(float(r["hi"]))
                neg += r["corpus"] in indep and float(r["hi"]) < 0
        out["ShapeAddClNeg"] = str(neg)
        # class nodes − SF as published, and the top-1 scope − SF as published, with cluster intervals:
        # {Shape,Sem,Top}{C}Cl{Lo,Hi}; empty for corpora with fewer than 5 clusters (descriptive)
        want = {("X82 classfull", "melnik_shape − melnik"): "Shape", ("X82 classfull", "melnik_semtype − melnik"): "Sem",
                ("X82 e2x", "melnik_top1 − melnik"): "Top"}
        for r in csv.DictReader(open(f86, encoding="utf-8")):
            k = want.get((r["family"], r["contrast"]))
            if k and r["metric"] == "J" and r["corpus"] in XC:
                c = XC[r["corpus"]]
                lo, hi = r["lo"], r["hi"]
                out[f"{k}{c}ClLo"] = _s3(float(lo)) if lo not in ("", "nan") else ""
                out[f"{k}{c}ClHi"] = _s3(float(hi)) if hi not in ("", "nan") else ""
                out[f"{k}{c}Clusters"] = r["clusters"]
    return out


def rho_matcher_ci(points: list[dict], draws: int = 2000, seed: int = 20261082) -> tuple[float, float, float, int]:
    """Spearman over matchers of (R_exp of q, R_exp gain of SF); 95% interval resampling matchers, as
    experiments/sf_study/explore_x82_bf.py (X82 (f)), so that Valentine and the new corpora are comparable."""
    import random
    pts = [q for q in points if q["group"] != "ctrl"]
    xs, ys = [q["base_R_exp"] for q in pts], [q["d_R_exp"] for q in pts]
    rho = spearman(xs, ys)
    rng, boots = random.Random(seed), []
    for _ in range(draws):
        idx = [rng.randrange(len(xs)) for _ in xs]
        if len(set(idx)) > 2:
            boots.append(spearman([xs[i] for i in idx], [ys[i] for i in idx]))
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    return rho, lo, hi, len(pts)


def iter_rexp() -> dict[str, str] | None:
    """EXPLORATORY X42 (explore_iterations.py): expected Recall@GT gain over q of SF as published after t updates, from
    tuning/results/iter_{plain,md5,indep}_scores/report.md (section "## melnik", column R_exp). Prefix itr:
    {Pl,Md,Ind}{One,Two,Three,Five,Ten,Conv}."""
    out = {}
    names = {"1": "One", "2": "Two", "3": "Three", "5": "Five", "10": "Ten", "conv": "Conv"}
    for reg, key in (("plain", "Pl"), ("md5", "Md"), ("indep", "Ind")):
        f = STUDY / "tuning" / "results" / f"iter_{vreg(reg)}_scores" / "report.md"
        if not f.is_file():
            continue
        sec = False
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                sec = line.startswith("## melnik (")
                continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if sec and len(c) > 3 and c[0] in names:
                out[f"{key}{names[c[0]]}"] = _s3(float(c[2]))
    return out or None


def stop_two() -> dict[str, str] | None:
    """EXPLORATORY X88 held-out test (explore_x88_validate.py; pre-specified rule "stop after two updates"):
    results/x88/validate.md. Prefix sv: {Ttd,LmS,Syn,LmM}{Est,Lo,Hi,Verdict,Iter,Share}; NonInf, Sup, N."""
    f = STUDY / "results" / "x88" / "validate.md"
    if not f.is_file():
        return None
    key = {"T2D web tables": "Ttd", "Lahman/MusicBrainz single-table": "LmS", "Synthea-OMOP": "Syn",
           "Lahman/MusicBrainz multi-table": "LmM"}
    out, ni, su, n = {}, 0, 0, 0
    for line in f.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 10 and c[0] in key:
            k = key[c[0]]
            m = re.match(r"([+-]\d\.\d+) \[([+-]\d\.\d+), ([+-]\d\.\d+)\]", c[3])
            out[f"{k}Est"], out[f"{k}Lo"], out[f"{k}Hi"] = (_s3(float(m.group(i))) for i in (1, 2, 3))
            out[f"{k}Verdict"], out[f"{k}Iter"], out[f"{k}Share"] = c[4], c[8], c[7].rstrip("%")
            n += 1
            ni += c[4] != "NOT non-inferior"
            su += c[4] == "superior"
    out["NonInf"], out["Sup"], out["N"] = str(ni), str(su), str(n)
    return out


def literal_offsets() -> dict[str, str] | None:
    """EXPLORATORY X90 (layer3b/score_x90.py): SF as published vs without literal similarities, stopped after t updates.
    tuning/results/x90_scores/report.md. Prefix lt, per corpus {Val,Goby,Hdx}: S2Lit, S2None (median offset strength at
    t = 2), OConvLit, OConvNone (truth-directed offset, converged), PLit, PNone (J(t2) − J(conv)); ValConvNoneJ."""
    f = STUDY / "tuning" / "results" / "x90_549_scores" / "report.md"
    if not f.is_file():
        return None
    key = {"valentine_plain": "Val", "goby": "Goby", "hdxsm": "Hdx"}
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 9 and c[0] in key:
            k, arm, tt = key[c[0]], ("Lit" if c[1] == "melnik" else "None"), c[2]
            if tt == "t2":
                out[f"{k}STwo{arm}"] = f"{float(c[5]):.2f}"
            if tt == "conv":
                out[f"{k}OConv{arm}"] = f"{float(c[6]):.2f}"
                out[f"{k}ConvJ{arm}"] = _s3(float(c[8]))
        if len(c) == 4 and c[0] in key:
            k = key[c[0]]
            m = re.match(r"([+-]\d\.\d+)", c[2])
            if c[1].startswith("P melnik"):
                out[f"{k}PLit"] = _s3(float(m.group(1)))
            elif c[1].startswith("P lit_none"):
                out[f"{k}PNone"] = _s3(float(m.group(1)))
    return out


def rq1_checks() -> dict[str, str] | None:
    """EXPLORATORY X89 (explore_x89_rq1.py): coupling checks and range restriction per corpus.
    results/x89/report.md. Prefix cp: Old{C}, Fam{C}, Rd{C}; Ceil{C}, Floor{C} (%); Rng{C} rho of corpus C restricted to
    the generated readable databases' quality range; RngLo, RngHi (that range)."""
    f = STUDY / "results" / "x89" / "report.md"
    if not f.is_file():
        return None
    key = {"valentine_plain": "Val", "valentine_md5": "VMd", "goby": "Goby", "hdxsm": "Hdx", "madi_indep": "Madi",
           "final_controlled_plain": "CtlPl", "final_controlled_opaque": "CtlOp", "gdcsm": "Gdc", "final_indep": "Indep"}
    out, sec = {}, 0
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            sec += 1
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if sec == 1 and len(c) == 8 and c[0] in key:
            k = key[c[0]]
            out[f"Old{k}"], out[f"Fam{k}"], out[f"Rd{k}"] = c[3], c[4].split(")")[1].strip(), c[6]
        if sec == 2 and len(c) == 7 and c[0] in key:
            k = key[c[0]]
            out[f"Ceil{k}"], out[f"Floor{k}"] = c[3].rstrip("%"), c[4].rstrip("%")
        if sec == 3 and len(c) == 5 and c[1].startswith("final_controlled_plain") and c[0] in key:
            out[f"Rng{key[c[0]]}"] = c[3]
            m = re.search(r"\[([0-9.]+), ([0-9.]+)\]", c[1])
            out["RngLo"], out["RngHi"] = m.group(1), m.group(2)
    return out


def export_check() -> dict[str, str] | None:
    """EXPLORATORY X91 (export checker on SF graph export): results/sf_export_check/{report.md, components.csv}.
    Prefix gx: Runs; ResMant/ResExp (largest relative residual of Proposition 1 on the graph's own blocks); Viol, PairsM
    (order violations / strict within-block pairs, millions); TiesBroken, TiesM; RecoMant/RecoExp (reconstruction of the
    decomposition); reference configuration (NAME, literals) offsets by origin: TwoLitSd/O, TwoNameSd (t = 2, paths of
    length 2), ThreeLitSd/O, ThreeNameSd/O (t = 3; literal paths of length 2, the initial similarities' paths of length 3);
    NRef (exports of the reference configuration)."""
    import math as _m
    f = STUDY / "results" / "sf_export_check" / "report.md"
    fc = STUDY / "results" / "sf_export_check" / "components.csv"
    if not f.is_file():
        return None
    out, res, viol, pairs, tb, ties = {}, 0.0, 0, 0, 0, 0
    for line in f.read_text(encoding="utf-8").splitlines():
        mm = re.match(r"- runs: (\d+)", line)
        if mm:
            out["Runs"] = mm.group(1)
        mm = re.match(r"- decomposition .* reconstructs x_t to ([0-9.e+-]+)", line)
        if mm:
            v = float(mm.group(1))
            out["RecoExp"] = str(_m.floor(_m.log10(v)))
            out["RecoMant"] = f"{v / 10 ** _m.floor(_m.log10(v)):.1f}"
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 7 and c[0] in ("ID", "NAME", "SPLIT") and "/" in c[5]:
            res = max(res, float(c[4]))
            a, b = c[5].split(" / "); viol += int(a); pairs += int(b)
            a, b = c[6].split(" / "); tb += int(a); ties += int(b)
    e = _m.floor(_m.log10(res))
    out["ResMant"], out["ResExp"] = f"{res / 10 ** e:.1f}", str(e)
    out["Viol"], out["PairsM"], out["TiesBroken"], out["TiesM"] = str(viol), f"{pairs / 1e6:.1f}", str(tb), f"{ties / 1e6:.1f}"
    if fc.is_file():
        rows = [r for r in csv.DictReader(open(fc, encoding="utf-8")) if r["attach"] == "NAME" and r["lit"] == "PREFIX_SUFFIX"]
        def agg(t_, origin, path):
            v = [r for r in rows if r["t"] == t_ and r["origin"] == origin and r["path"] == path]
            sd = [float(r["offset_sd"]) for r in v]
            med = statistics.median(sd) if sd else 0.0   # the plain median (report), averaging the middle pair
            return med, (sum(float(r["o"]) for r in v) / len(v) if v else 0.0), len(v)
        s2, o2, n = agg("2", "literal", "2")
        out["TwoLitSd"], out["TwoLitO"], out["NRef"] = f"{s2:.3f}", _s3(o2), str(n)
        out["TwoNameSd"] = f"{max(agg('2', 'name', p)[0] for p in ('0', '1', '2')):.3f}"
        s3, o3, _ = agg("3", "literal", "2")
        out["ThreeLitSd"], out["ThreeLitO"] = f"{s3:.3f}", _s3(o3)
        s3n, o3n, _ = agg("3", "name", "3")
        out["ThreeNameSd"], out["ThreeNameO"] = f"{s3n:.3f}", _s3(o3n)
        s3l, o3l, _ = agg("3", "literal", "3")   # the literal three-edge paths (check): median small, direction negative
        out["ThreeLitThreeO"] = _s3(o3l)
    return out


def corpus_effects() -> dict[str, str] | None:
    """EXPLORATORY X78 (layer3b/x78_analysis.py): pooled within-corpus Spearman rho between an effect and a corpus
    property, 95% interval over source clusters. results/sf_study/results/benchmark_effects_report.md. Prefix ce:
    {AdTables, RoleFd, RoleKey, FdNameGap, GainEntropy, GainNameGap, TabTables, TabKeys}{,Lo,Hi}."""
    f = STUDY / "results" / "x78_549" / "benchmark_effects_report.md"
    if not f.is_file():
        return None
    want = {("E1 datatype − SF", "src_tables"): "AdTables", ("E2 role − key", "fd_per_col"): "RoleFd",
            ("E2 role − key", "key_share"): "RoleKey", ("E3 fd − melnik (R_exp)", "name_gap"): "FdNameGap",
            ("E4 SF − q", "dtype_entropy"): "GainEntropy", ("E4 SF − q", "name_gap"): "GainNameGap",
            ("E5 table − datatype", "src_tables"): "TabTables", ("E5 table − datatype", "key_share"): "TabKeys"}
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 5 and (c[0], c[1]) in want:
            k = want[(c[0], c[1])]
            m = re.match(r"([+-]\d\.\d+) \[([+-]\d\.\d+), ([+-]\d\.\d+)\]", c[2])
            if m:
                out[k], out[f"{k}Lo"], out[f"{k}Hi"] = m.group(1), m.group(2), m.group(3)
    return out


def rule_checks() -> dict[str, str] | None:
    """EXPLORATORY X92 (explore_x92_rules.py): averaging updates 2 and 3, and the additive datatype term against SF after
    two updates. results/x92/report.md. Prefix rl, per corpus C: AvgConv{C}, AvgTwo{C}, AdTwo{C} (estimates); counts
    AvgTwoNeg (avg23 < t2), AvgConvNear (|avg23 − conv| <= 0.005), N; AdTwoSplitMax (largest |A_d − t2| on Valentine readable
    and the generated databases); AdTwoIndMin/Max (A_d − t2 on the independently produced single-table corpora)."""
    f = STUDY / "results" / "x92" / "report.md"
    if not f.is_file():
        return None
    key = {"Valentine readable": "Val", "Valentine recoded": "VMd", "Indep. multi-table (3 DBs)": "Ind",
           "New databases readable": "CtlPl", "New databases recoded": "CtlOp", "Independent schema pairs": "Indep",
           "GDC-SM": "Gdc", "HDXSM": "Hdx", "GOBY": "Goby", "MaDI use cases": "Madi", "MaDI products": "MadiP",
           "T2D web tables": "Ttd", "Lahman/MB single-table": "LmS", "Synthea-OMOP": "Syn", "Lahman/MB multi-table": "LmM"}
    out, vals = {}, {}
    for line in f.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 8 and c[0] in key:
            k = key[c[0]]
            e = [float(x.split()[0]) for x in c[3:7]]
            vals[k] = e
            out[f"AvgConv{k}"], out[f"AvgTwo{k}"], out[f"AdTwo{k}"] = _s3(e[0]), _s3(e[1]), _s3(e[2])
            out[f"TwoConv{k}"] = _s3(e[3])   # SF after two updates − converged
    if vals:
        out["N"] = str(len(vals))
        out["AvgTwoNeg"] = str(sum(v[1] < 0 for v in vals.values()))
        out["AvgConvNear"] = str(sum(abs(v[0]) <= 0.005 for v in vals.values()))
        split = [abs(vals[k][2]) for k in ("Val", "CtlPl", "CtlOp") if k in vals]
        ind = [vals[k][2] for k in ("Ttd", "Indep", "Hdx", "Madi", "Goby") if k in vals]
        out["AdTwoSplitMax"] = f"{max(split):.3f}"
        out["AdTwoIndMin"], out["AdTwoIndMax"] = _s3(min(ind)), _s3(max(ind))
    return out


def literal_term() -> dict[str, str] | None:
    """EXPLORATORY X93: an information-matched one-step term carrying SF's literal two-edge
    evidence L (A_L = q+ + lambda_L L~, A_dL = q+ + 5D + lambda_dL L~; lambda frozen on the Valentine screen) against SF.
    Sources: tuning/results/x93_scores/{contrasts.csv, mechanism.md, lambda.json}. Prefix ld:
      {DLConv, DLTwo, LConv, DLAd, DConv, TwoConv}{J,R}{C}{Est,Lo,Hi,Clusters}: A_dL − SF converged, A_dL − SF after two
        updates, A_L − SF converged, A_dL − A_d, A_d − SF converged, SF after two − converged; bounds empty where the corpus
        has fewer than 5 clusters (descriptive);
      LamL, LamDL: the frozen weights; Rho{C}, RhoShare{C}: P1's median Spearman rho between L~ and SF(t2)'s block offsets,
        and the share of scenarios with rho >= 0.9."""
    base = STUDY / "tuning" / "results" / "x93_549_scores"
    f = base / "contrasts.csv"
    if not f.is_file():
        return None
    ck = {"A_dL − SF_conv": "DLConv", "A_dL − SF_t2": "DLTwo", "A_L − SF_conv": "LConv", "A_dL − A_d": "DLAd",
          "A_d − SF_conv": "DConv", "SF_t2 − SF_conv": "TwoConv"}
    corp = {**XC, "valentine_md5": "VMd", "madi_indep": "Madi", "madi_products": "MadiP", "madi": "MadiAll"}
    out = {}
    for r in csv.DictReader(open(f, encoding="utf-8")):
        k, c = ck.get(r["contrast"]), corp.get(r["corpus"])
        if not k or not c:
            continue
        m = "J" if r["metric"] == "J" else "R"
        key = f"{k}{m}{c}"
        out[f"{key}Est"] = _s3(float(r["estimate"]))
        out[f"{key}Lo"] = _s3(float(r["lo"])) if r["lo"] not in ("", "nan") else ""
        out[f"{key}Hi"] = _s3(float(r["hi"])) if r["hi"] not in ("", "nan") else ""
        out[f"{key}Clusters"] = r["clusters"]
    lam = base / "lambda.json"
    if lam.is_file():
        j = json.loads(lam.read_text(encoding="utf-8"))
        out["LamL"], out["LamDL"] = str(j["A_L"]), str(j["A_dL"])
    mech = base / "mechanism.md"
    if mech.is_file():
        for line in mech.read_text(encoding="utf-8").splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) == 5 and c[0] in corp:
                out[f"Rho{corp[c[0]]}"] = f"{float(c[2]):.3f}"
                out[f"RhoShare{corp[c[0]]}"] = c[3].rstrip("%")
    return out


_EGR = {("plain", "levenshtein"): "PlLev", ("plain", "pmf"): "PlPmf", ("md5", "pmf"): "MdPmf",
        ("md5", "unary_statistics_invariant"): "MdUsi"}
_EGF = {"ovl": "Ovl", "dep": "Dep", "meta": "Meta", "fmt": "Fmt", "str": "Str", "dist": "Dist", "unc": "Unc",
        "llm": "Llm", "emb": "Emb", "comp": "Comp", "lex": "Lex"}
# readable names for the measures a macro may name (no identifiers in prose); others: underscores to spaces
_EGNAME = {"valentine_distribution": "Valentine's distribution matcher", "char_distribution": "character distribution",
           "semint": "SemInt", "containment": "value containment", "jsd_values": "Jensen--Shannon divergence of values",
           "distribution_shape": "distribution shape", "unionability_pair": "pairwise unionability",
           "unary_statistics": "unary statistics", "levenshtein": "Levenshtein", "pmf": "value distribution (PMF)"}
_CI = re.compile(r"([+-]?\d+\.\d+) \[([+-]?\d+\.\d+), ([+-]?\d+\.\d+)\]")


def _s3z(v: float) -> str:
    s = f"{v:+.3f}"
    return "0.000" if s in ("+0.000", "-0.000") else s


def _egname(m: str) -> str:
    return _EGNAME.get(m, m.replace("_", " "))


def matcher_roles_x73() -> dict[str, str] | None:
    """EXPLORATORY X73 (B): each matcher's within-table similarity as a one-step role term (ROLE, added to A_d with weight
    chosen leave-one-family-out), against the same measure as K = 2 edges in SF (EDGE) and fused after propagation (POST).
    J_tf, family macro, 95% CI. Sources: results/sf_study/results/{plain,md5}/matcherroles_summary.md. Prefix mr. RI as for
    eg (PlLev, PlPmf, MdPmf, MdUsi):
      {Pl,Md}N: measures; RI{RefSf,RefAd,RefSfd}: SF, A_d, SF + 5D;
      RI{Six,G}{Est,Lo,Hi}: the dependency roles dep_R6 and dep_Rg − A_d;
      RI{RoleAd,EdgeSf,RoleEdge,PostRole,PostSfd}{Pos,Neg}: measures whose interval is above / below 0;
      RI{RoleAd,EdgeSf,RoleEdge,PostRole,PostSfd}Mean: mean of the estimates over measures;
      RIBest{Name,Est,Lo,Hi}: the best non-dependency role (ROLE − A_d), readable name;
      RIBestMinus{Six,G}{Est,Lo,Hi}: best − dep_R6, best − dep_Rg;
      RIFam{Ovl,...}{RoleAd,EdgeSf,RoleEdge,PostRole}: family means; RITopFam: the family with the highest ROLE − A_d mean
      (family code); RITopFamEst."""
    cols = ["RoleAd", "EdgeSf", "RoleEdge", "PostRole", "PostSfd"]
    cnt = {"ROLE − A_d": "RoleAd", "EDGE − SF": "EdgeSf", "ROLE − EDGE": "RoleEdge", "POST − ROLE": "PostRole",
           "POST − SFD": "PostSfd"}
    out = {}
    for regime, rc in (("plain", "Pl"), ("md5", "Md")):
        f = STUDY / "results" / regime / "matcherroles_summary.md"
        if not f.is_file():
            return None
        init, rows, ri = None, {}, None
        for line in f.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("## initial matcher "):
                init = s.split("## initial matcher ", 1)[1].strip()
                ri = _EGR[(regime, init)]
                rows[ri] = []
            elif ri is None and re.match(r"^\d+ measures;", s):
                out[f"{rc}N"] = s.split()[0]
            elif ri and s.startswith("References:"):
                v = re.findall(r"(\d+\.\d+)", s)
                out[f"{ri}RefSf"], out[f"{ri}RefAd"], out[f"{ri}RefSfd"] = v[0], v[1], v[2]
            elif ri and s.startswith("- dep_R"):
                k = "Six" if s.startswith("- dep_R6") else "G"
                mm = _CI.search(s)
                out.update({f"{ri}{k}Est": _s3z(float(mm.group(1))), f"{ri}{k}Lo": _s3z(float(mm.group(2))),
                            f"{ri}{k}Hi": _s3z(float(mm.group(3)))})
            elif ri and s.startswith("Counts over"):
                for name, pos, neg in re.findall(r"([A-Z]+ − [A-Za-z_]+) (\d+)/(\d+)", s):
                    if name in cnt:
                        out[f"{ri}{cnt[name]}Pos"], out[f"{ri}{cnt[name]}Neg"] = pos, neg
            elif ri and re.match(r"^- \w+ \(\d+\): ", s):
                fam = s[2:].split(" ", 1)[0]
                v = re.findall(r"([+-]\d+\.\d+)", s)
                for c, x in zip(cols[:4], v):
                    out[f"{ri}Fam{_EGF[fam]}{c}"] = _s3z(float(x))
            elif ri and s.startswith("Best non-dependency role:"):
                name = re.match(r"Best non-dependency role: (\w+)", s).group(1)
                out[f"{ri}BestName"] = _egname(name)
                cis = _CI.findall(s)
                for k, (e, lo, hi) in zip(("Six", "G"), cis):
                    out.update({f"{ri}BestMinus{k}Est": _s3z(float(e)), f"{ri}BestMinus{k}Lo": _s3z(float(lo)),
                                f"{ri}BestMinus{k}Hi": _s3z(float(hi))})
                hit = [r for r in rows[ri] if r[0] == name]
                if hit:
                    e, lo, hi = hit[0][1]
                    out.update({f"{ri}BestEst": _s3z(e), f"{ri}BestLo": _s3z(lo), f"{ri}BestHi": _s3z(hi)})
            elif ri and s.startswith("| ") and not s.startswith("| measure"):
                c = [x.strip() for x in s.strip("|").split("|")]
                cis = [_CI.match(x) for x in c[2:7]]
                if len(c) >= 7 and all(cis):
                    rows[ri].append((c[0],) + tuple((float(m.group(1)), float(m.group(2)), float(m.group(3)))
                                                    for m in cis))
        for ri, rs in rows.items():
            for j, c in enumerate(cols):
                out[f"{ri}{c}Mean"] = _s3z(sum(r[j + 1][0] for r in rs) / len(rs))
            fams = {k[len(ri) + 3:-6]: float(v) for k, v in out.items()
                    if k.startswith(f"{ri}Fam") and k.endswith("RoleAd")}
            top = max(fams, key=fams.get)
            out[f"{ri}TopFam"], out[f"{ri}TopFamEst"] = top, _s3z(fams[top])
    return out


def role_fullmetrics() -> dict[str, str] | None:
    """EXPLORATORY X46 rescoring (explore_x46_fullmetrics.py): the dependency role term A_role
    (X46's leave-one-family-out weights) and A_d on the full metric set, on the RQ1 panel, aggregated as rqOneMean*
    (mean over initial matchers of the family macro). Sources: results/{plain,md5}/x46_fullmetrics_summary.csv. Prefix drm,
    then Pl / Md, then the contrast RoleQ (A_role − initial matcher), AdQ, RoleSf (− SF as published), AdSf, RoleAd, SfQ
    (SF as published − initial matcher; equals rqOneMean* on plain), then the metric Rexp, Rpess, FHr, FH, FM, MRRexp,
    MRRpess, then Est / Lo / Hi (95% bootstrap of scenarios within family) / Up (matchers with a macro difference > 0);
    {Pl,Md}N: initial matchers; {Pl,Md}Scen: scenarios in the families averaged."""
    cc = {"A_role - q": "RoleQ", "A_d - q": "AdQ", "A_role - SF": "RoleSf", "A_d - SF": "AdSf",
          "A_role - A_d": "RoleAd", "SF - q": "SfQ"}
    mc = {"R_exp": "Rexp", "R_pess": "Rpess", "F1_Hr": "FHr", "F1_H": "FH", "F1_M": "FM", "MRR_exp": "MRRexp",
          "MRR_pess": "MRRpess"}
    out = {}
    for regime, rc in (("plain", "Pl"), ("md5", "Md")):
        f = STUDY / "results" / vreg(regime) / "x46_fullmetrics_summary.csv"
        if not f.is_file():
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            k = f"{rc}{cc[r['contrast']]}{mc[r['metric']]}"
            out.update({f"{k}Est": _s3z(float(r["estimate"])), f"{k}Lo": _s3z(float(r["lo"])),
                        f"{k}Hi": _s3z(float(r["hi"])), f"{k}Up": r["up"]})
            out[f"{rc}N"], out[f"{rc}Scen"] = r["matchers"], r["scenarios"]
    return out or None


_FWC = {"valentine_plain": "Val", "valentine_md5": "VMd", "goby": "Goby", "hdxsm": "Hdx", "gdcsm": "Gdc",
        "madi_indep": "Madi", "madi_products": "MadiP", "t2d": "Ttd", "ctl": "LmCtl", "final_controlled": "FinCtl",
        "final_indep": "FinInd", "mimic": "Mimic"}
_FWN = {"valentine_plain": "Valentine", "valentine_md5": "Valentine recoded", "goby": "GOBY", "hdxsm": "HDXSM",
        "gdcsm": "GDC-SM", "madi_indep": "MaDI", "madi_products": "MaDI products", "t2d": "T2D",
        "ctl": "the controlled corpora", "final_controlled": "the controlled multi-table corpus",
        "final_indep": "the independent multi-table corpus", "mimic": "MIMIC"}
_FWB = {"i": "single matchers", "ii": "averaged base", "iii": "COMA, char3 and Cupid"}


def _s4(v: float) -> str:
    s = f"{v:+.4f}"
    return "0.0000" if s in ("+0.0000", "-0.0000") else s


def followup_x94_x95() -> dict[str, str] | None:
    """EXPLORATORY X94 (C: per-scenario weights for the datatype and literal terms, stage 1) and X95 (the fixed-weight
    one-step matcher; pre-registered in the follow-up, layer3b/X95_PREREGISTRATION.md; not X97). Prefix fw.
    Sources: results/x94/stage1.md, tuning/results/x94/cells.csv.gz, tuning/results/x95/contrasts.csv,
    tuning/results/x95/posthoc_matched.csv.
    X94 (3 decimals):
      Ca{Med,Qlo,Qhi,Min,Max,N}: SF's two-update weight c/a over the cells (matcher x scenario); RuleaMaxDiff: max |J_a − J_sf_t2|;
      <C>{HeadMs,HeadS}: headroom, per-cell and per-scenario oracle − best global weights (C in Val, Goby, Hdx, Gdc, Madi, MadiP);
      <C>Rec: share of the headroom recovered by learned weights (features); RecMaxInd (+ RecMaxIndName): max over GOBY,
      HDXSM, GDC-SM; RecMaxAll: max over all held-out corpora;
      <C>RuleA{Est,Lo,Hi}, <C>RuleB{Est,Lo,Hi}: rule (a) = SF(t2) weights and rule (b) − frozen_V (cluster CI; empty if < 5 clusters).
    X95 (4 decimals):
      LTwo{MaxAbs,MaxAbsWhere}: max |(+L) − SF_t2| over corpus x base, and where; LTwoMaxAbsInd: the same over GOBY, HDXSM, GDC-SM;
      LTwoWithinN / LTwoCells: cells with |Δ| <= 0.0006, of all; LTwoExceptions: the other cells, readable;
      Prim<C>{Est,Lo,Hi}: full − SF_conv, base family (i) (CI only where >= 5 clusters);
      Ph<C>{Kf,Key,RKf,RKey}{Est,Lo,Hi}: post hoc, full / +R − SF key graph + FD edges (Kf) / top-1 key graph (Key)."""
    f1 = STUDY / "results" / "x94_549" / "stage1.md"
    f2 = STUDY / "tuning" / "results" / "x95_549" / "contrasts.csv"
    if not (f1.is_file() and f2.is_file()):
        return None
    out = {}
    import gzip
    fc = STUDY / "tuning" / "results" / "x94_549" / "cells.csv.gz"
    if fc.is_file():
        with gzip.open(fc, "rt", encoding="utf-8") as fh:
            ca = sorted(float(r["ca"]) for r in csv.DictReader(fh) if r.get("ca") not in (None, "", "nan"))
        q = lambda p_: ca[min(len(ca) - 1, int(round(p_ * (len(ca) - 1))))]  # noqa: E731
        out.update({"CaMed": f"{statistics.median(ca):.3f}", "CaQlo": f"{q(0.25):.3f}", "CaQhi": f"{q(0.75):.3f}",
                    "CaMin": f"{ca[0]:.3f}", "CaMax": f"{ca[-1]:.3f}", "CaN": str(len(ca))})
    rec = {}
    for line in f1.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        m = re.match(r"- rule \(a\) check: max \|J_a − J_sf_t2\| = ([\d.e+-]+)", s)
        if m:
            out["RuleaMaxDiff"] = f"{float(m.group(1)):.2f}"
        c = [x.strip() for x in s.strip("|").split("|")] if s.startswith("|") else []
        if len(c) == 9 and c[0] in _FWC:
            hm, hs_ = c[8].split("/")
            out[f"{_FWC[c[0]]}HeadMs"], out[f"{_FWC[c[0]]}HeadS"] = _s3z(float(hm)), _s3z(float(hs_))
        if len(c) == 7 and c[0] in _FWC and c[1] == "features":
            rec[c[0]] = float(c[5])
            out[f"{_FWC[c[0]]}Rec"] = f"{float(c[5]):.2f}"
        m = re.match(r"- (\w+): rule \((a|b)\) − frozen_V ([+-]?\d+\.\d+)(?: \[([+-]?\d+\.\d+), ([+-]?\d+\.\d+)\])?", s)
        if m and m.group(1) in _FWC:
            k = f"{_FWC[m.group(1)]}Rule{m.group(2).upper()}"
            out[f"{k}Est"] = _s3z(float(m.group(3)))
            out[f"{k}Lo"] = _s3z(float(m.group(4))) if m.group(4) else ""
            out[f"{k}Hi"] = _s3z(float(m.group(5))) if m.group(5) else ""
    ind = {k: v for k, v in rec.items() if k in ("goby", "hdxsm", "gdcsm")}
    if ind:
        best = max(ind, key=ind.get)
        out["RecMaxInd"], out["RecMaxIndName"] = f"{ind[best]:.2f}", _FWN[best]
        out["RecMaxAll"] = f"{max(rec.values()):.2f}"
    rows = list(csv.DictReader(open(f2, encoding="utf-8")))
    lt = [(r["corpus"], r["family"], float(r["estimate"])) for r in rows if r["contrast"] == "+L − SF_t2"]
    if lt:
        mx = max(lt, key=lambda x: abs(x[2]))
        out["LTwoMaxAbs"] = f"{abs(mx[2]):.4f}"
        out["LTwoMaxAbsWhere"] = f"{_FWN[mx[0]]} ({_FWB[mx[1]]})"
        out["LTwoMaxAbsInd"] = f"{max(abs(x[2]) for x in lt if x[0] in ('goby', 'hdxsm', 'gdcsm')):.4f}"
        out["LTwoWithinN"] = str(sum(abs(x[2]) <= 0.0006 + 1e-12 for x in lt))
        out["LTwoCells"] = str(len(lt))
        exc = sorted((x for x in lt if abs(x[2]) > 0.0006 + 1e-12), key=lambda x: -abs(x[2]))
        out["LTwoExceptions"] = "; ".join(f"{_FWN[c_]} ({_FWB[b_]}) ${_s4(v)}$" for c_, b_, v in exc)
    for r in rows:
        if r["contrast"] == "full − SF_conv" and r["family"] == "i" and r["corpus"] in _FWC:
            k = f"Prim{_FWC[r['corpus']]}"
            out[f"{k}Est"] = _s4(float(r["estimate"]))
            out[f"{k}Lo"] = _s4(float(r["lo"])) if r["lo"] else ""
            out[f"{k}Hi"] = _s4(float(r["hi"])) if r["hi"] else ""
    f3 = STUDY / "tuning" / "results" / "x95" / "posthoc_matched.csv"
    if f3.is_file():
        cmap = {"full − SF_ucc1_fd_topk": "Kf", "full − SF_ucc_top1": "Key", "+R − SF_ucc1_fd_topk": "RKf",
                "+R − SF_ucc_top1": "RKey"}
        for r in csv.DictReader(open(f3, encoding="utf-8")):
            if r["contrast"] in cmap and r["corpus"] in _FWC:
                k = f"Ph{_FWC[r['corpus']]}{cmap[r['contrast']]}"
                out[f"{k}Est"] = _s4(float(r["estimate"]))
                out[f"{k}Lo"] = _s4(float(r["lo"])) if r["lo"] else ""
                out[f"{k}Hi"] = _s4(float(r["hi"])) if r["hi"] else ""
    return out


LAYER_KEYS = {
    1: {"melnik": "Ref", "coef_prod": "Prod", "fd": "Fd", "coef_prod_fd": "ProdFd", "attach_id": "Id",
        "lit_none": "NoLit"},
    3: {"melnik": "Ref", "ucc": "Ucc", "ind": "Ind", "fd_ucc_ind": "FdUccInd", "fd_reified": "Reif",
        "fd_damped": "Damped", "col_fd": "ColFd", "col_fd_gcov": "ColFdGcov", "col_fd_gcov_u": "ColFdGcovU",
        "col_fd_jaccard": "ColFdJacc", "col_gpdep": "ColGpdep", "col_gpdep_gcov": "ColGpdepGcov"},
}


def layer1(regime: str, layer: int = 1) -> dict[str, str] | None:
    """Layer 1 or 3 (Melnik reference and one-module arms), as scored by scorer: the arm table of
    tuning/results/layer<layer>_<regime>_scores/report.md (means over the non-control initial
    matchers, macro over families). Returns None while the regime is not scored in full."""
    f = STUDY / "tuning" / "results" / f"layer{layer}_{vreg(regime)}_scores" / "report.md"
    if not f.is_file():
        return None
    text = f.read_text(encoding="utf-8")
    import re
    m = re.search(r"Means over (\d+) non-control initial matchers with every arm on all (\d+) scenarios", text)
    if m is None or int(m.group(2)) < (81 if regime == "indep" else nfull(regime)):   # indep: 81 multi-table scenarios
        return None
    key = LAYER_KEYS[layer]
    arms = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 11 and cells[0] in key:
            arms[cells[0]] = cells
    out = {"N": m.group(1)}
    for arm, cells in arms.items():
        k = key[arm]
        out.update({f"{k}FM": cells[1], f"{k}FHr": cells[2], f"{k}Rexp": cells[3], f"{k}MRR": cells[4],
                    f"{k}Iter": cells[8]})
        if arm != "melnik":
            out.update({f"{k}VsFHr": cells[6], f"{k}VsRexp": cells[7]})
    return out


def rq4(plain: list[dict], md5: list[dict], n_md5: int) -> tuple[str, dict[str, str]]:
    """RQ4: the same initial matchers on the plain and the value-recoded (MD5) regime, SF in its
    standard configuration. Figure: per matcher a hollow mark (plain) and a filled mark (MD5),
    joined by a thin line; numbers for the text."""
    P = {p["seed"]: p for p in plain if p["group"] != "ctrl"}
    M = {p["seed"]: p for p in md5 if p["group"] != "ctrl"}
    both = sorted(set(P) & set(M))

    def avg(d: dict, k: str) -> float:
        return sum(d[s][k] for s in both) / len(both)

    groups = defaultdict(list)
    for s in both:
        groups[P[s]["group"]].append(s)
    lab = groups["label"]
    nums = {"N": len(both), "NScen": n_md5,
            "QPlain": f"{avg(P, 'x'):.3f}", "QMd": f"{avg(M, 'x'):.3f}",
            "RPlain": f"{avg(P, 'd_R_exp'):+.3f}", "RMd": f"{avg(M, 'd_R_exp'):+.3f}",
            "FPlain": f"{avg(P, 'd_F1_Hr'):+.3f}", "FMd": f"{avg(M, 'd_F1_Hr'):+.3f}",
            "UpRPlain": sum(P[s]["d_R_exp"] > 0 for s in both), "UpRMd": sum(M[s]["d_R_exp"] > 0 for s in both),
            "UpFPlain": sum(P[s]["d_F1_Hr"] > 0 for s in both), "UpFMd": sum(M[s]["d_F1_Hr"] > 0 for s in both),
            "LabN": len(lab),
            "LabQPlain": f"{sum(P[s]['x'] for s in lab) / len(lab):.2f}",
            "LabQMd": f"{sum(M[s]['x'] for s in lab) / len(lab):.2f}",
            "LabRPlain": f"{sum(P[s]['d_R_exp'] for s in lab) / len(lab):+.3f}",
            "LabRMd": f"{sum(M[s]['d_R_exp'] for s in lab) / len(lab):+.3f}",
            "RhoShiftR": f"{spearman([M[s]['x'] - P[s]['x'] for s in both], [M[s]['d_R_exp'] - P[s]['d_R_exp'] for s in both]):.2f}",
            "RhoShiftF": f"{spearman([M[s]['x'] - P[s]['x'] for s in both], [M[s]['d_F1_Hr'] - P[s]['d_F1_Hr'] for s in both]):.2f}",
            "RhoMdR": f"{spearman([M[s]['x'] for s in both], [M[s]['d_R_exp'] for s in both]):.2f}",
            "RhoMdF": f"{spearman([M[s]['x'] for s in both], [M[s]['d_F1_Hr'] for s in both]):.2f}"}

    def one_panel(metric: str, ylabel: str, yr: tuple[float, float], legend: bool) -> list[str]:
        lines = [r"\begin{tikzpicture}", r"\begin{axis}[width=\linewidth, height=5.4cm,",
                 f"  xmin=-0.03, xmax=1.03, ymin={yr[0]}, ymax={yr[1]},",
                 r"  xlabel={Expected Recall@GT of the initial matcher alone},", f"  ylabel={{{ylabel}}},",
                 r"  label style={font=\footnotesize}, tick label style={font=\scriptsize},",
                 r"  ylabel shift=-3pt, yticklabel style={/pgf/number format/fixed},",
                 r"  axis lines=left, axis line style={gray!70}, tick style={gray!70},"]
        if legend:
            lines += [r"  legend to name=rqfourlegend, legend columns=6,",
                      r"  legend style={font=\scriptsize, draw=none, column sep=6pt},"]
        lines += [r"  legend cell align=left]",
                  r"\addplot[gray!60, thin, forget plot] coordinates {(-0.03,0) (1.03,0)};"]
        for s in both:
            lines.append(f"\\addplot[gray!45, very thin, forget plot] coordinates {{({P[s]['x']:.4f},{P[s][f'd_{metric}']:.4f}) "
                         f"({M[s]['x']:.4f},{M[s][f'd_{metric}']:.4f})}};")
        for g, title, _, _, mark in GROUPS:
            ss = groups.get(g, [])
            if not ss:
                continue
            hollow = " ".join(f"({P[s]['x']:.4f},{P[s][f'd_{metric}']:.4f})" for s in ss)
            filled = " ".join(f"({M[s]['x']:.4f},{M[s][f'd_{metric}']:.4f})" for s in ss)
            lines.append(f"\\addplot[only marks, mark={mark}, mark size=1.5pt, color=c{g}, forget plot, "
                         f"mark options={{draw=c{g}, fill=white, line width=0.4pt}}] coordinates {{{hollow}}};")
            lines.append(f"\\addplot[only marks, mark={mark}, mark size=1.7pt, color=c{g}, "
                         f"mark options={{draw=black!55, line width=0.2pt, fill=c{g}}}] coordinates {{{filled}}};")
            if legend:
                lines.append(f"\\addlegendentry{{{title}}}")
        if legend:
            lines += [r"\addlegendimage{only marks, mark=o, mark size=1.7pt, black!60}",
                      r"\addlegendentry{hollow: plain, filled: MD5}"]
        lines += [r"\end{axis}", r"\end{tikzpicture}"]
        return lines

    lines = ["% Generated by make_study_numbers.py from summary_tiefree.csv (plain and md5); do not edit.",
             r"\begin{figure*}[t]", r"\centering"]
    lines += [f"\\definecolor{{c{g}}}{{HTML}}{{{color.upper()}}}" for g, _, _, color, _ in GROUPS]
    ylo_r = min(min(P[s]["d_R_exp"], M[s]["d_R_exp"]) for s in both) - 0.02
    yhi_r = max(max(P[s]["d_R_exp"], M[s]["d_R_exp"]) for s in both) + 0.02
    ylo_f = min(min(P[s]["d_F1_Hr"], M[s]["d_F1_Hr"]) for s in both) - 0.02
    yhi_f = max(max(P[s]["d_F1_Hr"], M[s]["d_F1_Hr"]) for s in both) + 0.02
    for i, (metric, ylabel, yr) in enumerate((("R_exp", "SF's change in expected Recall@GT", (round(ylo_r, 2), round(yhi_r, 2))),
                                              ("F1_Hr", "SF's change in tie-free Hungarian F1", (round(ylo_f, 2), round(yhi_f, 2))))):
        lines.append(r"\begin{minipage}[t]{0.485\textwidth}\centering")
        lines += one_panel(metric, ylabel, yr, legend=(i == 0))
        lines.append(f"\\\\[-2pt]{{\\footnotesize ({'ab'[i]})}}")
        lines.append(r"\end{minipage}\hfill")
    lines[-1] = r"\end{minipage}"
    lines += [r"\\[2pt]\ref{rqfourlegend}",
              f"\\caption{{Recoding the values moves the initial matchers to the left and SF's effect up. "
              f"Each line joins one of {len(both)} initial matchers on the plain regime (hollow) and on the "
              f"value-recoded regime (MD5, filled), SF in its standard configuration, {n_md5} scenarios: "
              f"(a) the change in the corpus-level ranking, (b) the change in the one-to-one assignment, "
              f"ties broken at random. Included are the initial matchers with a valid matrix on every scenario of both regimes.}}",
              r"\label{fig:rq4-recode}", r"\end{figure*}"]
    return "\n".join(lines) + "\n", nums


def dtype_baseline() -> dict[str, str] | None:
    """EXPLORATORY (explore_dtype_baseline.py, the exploratory log, X19; review ER6 and ER3): the additive
    datatype baseline q + beta D against the reference SF, and the residual of SF = a q + b_block.
    Parses results/plain/dtype_baseline_summary.md. Prefix dt."""
    f = STUDY / "results" / "plain" / "dtype_baseline_summary.md"
    if not f.is_file():
        return None
    t = f.read_text(encoding="utf-8")
    out = {}
    m = re.search(r"(\d+) initial matchers; .*? evaluated on (\d+) dev scenarios", t)
    out["NSeeds"], out["NScen"] = m.group(1), m.group(2)
    out["UniBeta"] = re.search(r"Universal beta: (\d+(?:\.\d+)?)", t).group(1)
    for label, k in (("SF reference", "Sf"), ("q + beta D, per-matcher", "Own"), ("q + beta D, universal", "Uni")):
        line = next(l for l in t.splitlines() if l.startswith(f"| {label}"))
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        out.update({f"{k}J": f"{float(c[1]):+.3f}", f"{k}R": f"{float(c[2]):+.3f}", f"{k}F": f"{float(c[3]):+.3f}",
                    f"{k}Up": c[4].split("/")[0].strip()})
    m = re.search(r"universal beta mean ([+-][\d.]+), SF ahead (\d+), baseline ahead (\d+)", t)
    out["UniDiff"], out["UniSfAhead"], out["UniBaseAhead"] = f"{float(m.group(1)):+.3f}", m.group(2), m.group(3)
    m = re.search(r"constant matcher \(Amendment 1\): (\d+) fits; .*?max ([\d.e+-]+);", t)
    out["ResN"] = f"{int(m.group(1)):,}".replace(",", "{,}")
    mant, exp = f"{float(m.group(2)):.1e}".split("e")
    out["ResMax"] = f"{mant}\\cdot10^{{{int(exp)}}}"
    m = re.search(r"datatype matcher: .*?share below 1e-4: (\d+\.\d+)", t)
    out["DtShare"] = f"{100 * float(m.group(1)):.0f}"
    m = re.search(r"median (\d+), max (\d+); blocks per cell median ([\d.]+)", t)
    out["BlocksMed"], out["BlocksPerCell"] = m.group(1), m.group(3)
    return out


def multitable_ext() -> dict[str, str] | None:
    """EXPLORATORY X54/X55 (explore_multitable_ext.py): Synthea -> OMOP and Lahman/MusicBrainz multi-table.
    Parses results/{synthea,indep2}/multitable_summary.md. Prefix mt, then the cell (SynFull, SynPair, Lahman,
    Mb, Macro), then the quantity (Q, Sf, SfQ, FdQ, KeyQ, AdSf, AdtSf, RoleSf, RoleKey, RtSf, AdtAd), then Est, Lo, Hi;
    and <cell>N (scenarios), <cell>M (matchers)."""
    cells = {"SyntheaOMOP-Full": "SynFull", "SyntheaOMOP-Pair": "SynPair", "Lahman": "Lahman", "MusicBrainz": "Mb",
             "macro over Lahman and MusicBrainz": "Macro"}
    qty = {"q": "Q", "SF as published": "Sf", "SF as published − q": "SfQ", "SF with FD edges − q": "FdQ",
           "SF top-1 key − q": "KeyQ", "A_d − SF as published": "AdSf", "A_dt − SF as published": "AdtSf",
           "A_role − SF as published": "RoleSf", "A_role − SF top-1 key": "RoleKey", "A_rt − SF as published": "RtSf",
           "A_dt − A_d": "AdtAd"}
    out = {}
    for corpus in ("synthea", "indep2"):
        f = STUDY / "results" / "multitable_fix" / corpus / "multitable_summary.md"   # FD arms on the fixed profiles
        if not f.is_file():
            continue
        cell = None
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                head = line[3:]
                m = re.match(r"(.+?) \((\d+) scenarios, (\d+) matchers\)", head)
                name = m.group(1) if m else head
                cell = cells.get(name)
                if cell and m:
                    out[f"{cell}N"], out[f"{cell}M"] = m.group(2), m.group(3)
                continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if cell and len(c) == 2 and c[0] in qty:
                v = re.findall(r"([+-]?\d+\.\d+)", c[1])
                if len(v) == 3:
                    signed = qty[c[0]] not in ("Q", "Sf")
                    fmt = (lambda x: f"{x:+.3f}") if signed else (lambda x: f"{x:.3f}")
                    for s, x in zip(("Est", "Lo", "Hi"), v):
                        out[f"{cell}{qty[c[0]]}{s}"] = fmt(float(x))
    return out or None


def family_strength() -> dict[str, str] | None:
    """EXPLORATORY X58 (layer3b/x58_family.py): does the matcher's family change SF's gain beyond its strength?
    Parses tuning/results/x58_<regime>/report.md. Prefix fam, then the regime (Pl, Md), then the arm (Mel, Fd), then
    QualRtwo (R² of quality alone), FamRtwo (R² with family), FamInc (increment), FamP (permutation p),
    FamBeyondTiesInc / FamBeyondTiesP; and <regime>MelOvlResid (value-overlap family's mean residual)."""
    out = {}
    for regime, tag in (("plain", "Pl"), ("md5", "Md")):
        f = STUDY / "tuning" / "results" / f"x58_{vreg(regime)}" / "report.md"
        if not f.is_file():
            continue
        arm = None
        for line in f.read_text(encoding="utf-8").splitlines():
            m = re.match(r"## Arm `(\w+)`", line)
            if m:
                arm = {"melnik": "Mel", "fd": "Fd"}.get(m.group(1))
                continue
            if arm and line.startswith("- R²:"):
                m = re.search(r"quality \(Q \+ Q²\) ([\d.]+); \+ family ([\d.]+) \(Δ \+([\d.]+), permutation p ([\d.]+)\)", line)
                if m:
                    q, fr, inc, p = m.groups()
                    out.update({f"{tag}{arm}QualRtwo": f"{float(q):.2f}", f"{tag}{arm}FamRtwo": f"{float(fr):.2f}",
                                f"{tag}{arm}FamInc": f"{float(inc):.3f}", f"{tag}{arm}FamP": f"{float(p):.4g}"})
                m = re.search(r"family beyond Q \+ ties Δ \+([\d.]+), p ([\d.]+)\)", line)
                if m:
                    out[f"{tag}{arm}FamBeyondTiesInc"], out[f"{tag}{arm}FamBeyondTiesP"] = \
                        f"{float(m.group(1)):.3f}", f"{float(m.group(2)):.2g}"
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if arm == "Mel" and len(c) == 6 and c[0] == "ovl":
                out[f"{tag}MelOvlResid"] = f"{float(c[4]):+.3f}"
    return out or None


def parallel_edges() -> dict[str, str] | None:
    """EXPLORATORY (layer3b/score_parallel.py): the parallel-edge rule for FD edges. Parses
    tuning/results/parallel_<regime>_scores/report.md. Prefix pe, then the regime (Pl, Md), then the contrast
    (SplitFd = keep both messages − default, AvgFd, MaxFd, SplitMel = SPLIT − melnik, FdMel = default − melnik),
    then Est, Lo, Hi, R (R_exp part), F (F1_Hr part), Pos, N; and <regime>TpcSplitFdEst (TPC-DI)."""
    names = {"fd_split − fd": "SplitFd", "fd_pavg − fd": "AvgFd", "fd_pmax − fd": "MaxFd",
             "fd_split − melnik": "SplitMel", "fd − melnik": "FdMel"}
    out = {}
    for regime, tag in (("plain", "Pl"), ("md5", "Md")):
        f = STUDY / "tuning" / "results" / f"parallel_{vreg(regime)}_scores" / "report.md"
        if not f.is_file():
            continue
        section = None
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                section = line[3:].split(" (")[0]
                continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) != 6 or c[0] not in names:
                continue
            v = re.findall(r"([+-]?\d+\.\d+)", c[2])
            if len(v) != 3:
                continue
            k = names[c[0]]
            if section == "all":
                out.update({f"{tag}{k}Est": f"{float(v[0]):+.3f}", f"{tag}{k}Lo": f"{float(v[1]):+.3f}",
                            f"{tag}{k}Hi": f"{float(v[2]):+.3f}", f"{tag}{k}R": f"{float(c[3]):+.3f}",
                            f"{tag}{k}F": f"{float(c[4]):+.3f}"})
                m = re.match(r"(\d+) of (\d+)", c[5])
                if m:
                    out[f"{tag}{k}Pos"], out[f"{tag}{k}N"] = m.groups()
            elif section == "TPC-DI" and k == "SplitFd":
                out[f"{tag}TpcSplitFdEst"] = f"{float(v[0]):+.3f}"
    return out or None


def scaling_sweep() -> dict[str, str] | None:
    """EXPLORATORY X51 (controlled scaling and memory sweep; predictions before any run). Parses
    tuning/results/x51_scaling_report.md. Prefix sw: Scen; per arm (Mel, Fd, Reif) VertK, EdgesK, BuildK, IterK,
    Iters, EdgesSmall, EdgesLarge, BuildSmall, BuildLarge (s); FdPairsK, FdProdK, FdProdRtwo; FdMaxSide; RowsBuildLo,
    RowsBuildHi (s, 500 and 8000 rows); DtSmall, DtLarge (s); RatioSmall, RatioLarge (SF build / datatype task);
    RssMin, RssMax (GB, one scenario per task)."""
    f = STUDY / "tuning" / "results" / "x51_scaling_report.md"
    if not f.is_file():
        return None
    t = f.read_text(encoding="utf-8")
    out = {}
    m = re.search(r"^(\d+) synthetic scenarios", t, re.M)
    if m:
        out["Scen"] = m.group(1)
    tags = {"melnik": "Mel", "fd": "Fd", "fd_reified": "Reif"}
    fds, rss = [], []
    for line in t.splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 8 and c[0] in tags and "(" in c[1]:
            k = tags[c[0]]
            for key, cell in zip(("Vert", "Edges", "Build", "Iter"), c[1:5]):
                out[f"{k}{key}K"] = f"{float(cell.split()[0]):.2f}"
            out[f"{k}Iters"] = c[5]
            es, el = (int(x) for x in c[6].split("/"))
            bs, bl = (int(x) for x in c[7].split("/"))
            out.update({f"{k}EdgesSmall": f"{es:,}", f"{k}EdgesLarge": f"{el:,}", f"{k}BuildSmall": f"{bs / 1000:.1f}",
                        f"{k}BuildLarge": f"{bl / 1000:.1f}"})
        if c and re.fullmatch(r"\d+", c[0]) and len(c) == 6 and "/" in c[1]:
            fds += [int(x) for cell in c[1:] for x in cell.split("/")]
        if c and c[0] in ("500", "8000") and len(c) == 7:
            out["RowsBuildLo" if c[0] == "500" else "RowsBuildHi"] = f"{int(c[4]) / 1000:.1f}"
        if c and c[0] in ("16×16", "256×256") and len(c) == 8:
            tag = "Small" if c[0] == "16×16" else "Large"
            out[f"Dt{tag}"], out[f"Ratio{tag}"] = c[1], c[6]
        if c and re.fullmatch(r"\d+×\d+", c[0]) and len(c) == 5:
            rss += [int(x) for x in c[1:4]]
    m = re.search(r"against n·m k = ([\d.]+).*?against \|FD_s\|·\|FD_t\| k = ([\d.]+) \(R² ([\d.]+)\)", t)
    if m:
        out["FdPairsK"], out["FdProdK"], out["FdProdRtwo"] = m.groups()
    if fds:
        out["FdMaxSide"] = str(max(fds))
    if rss:
        out["RssMin"], out["RssMax"] = f"{min(rss) / 1000:.2f}", f"{max(rss) / 1000:.2f}"
    return out or None


def extraction() -> dict[str, str] | None:
    """EXPLORATORY X50 (explore_extraction.py): extraction with abstention and many-to-many matches, the policy
    chosen leave-one-family-out per matrix type. Parses results/<regime>/extraction_summary.md. Prefix xe, then the
    regime (Pl, Md, In), then the row: Q, Ad, Sf (Forced, Val; each Est, Lo, Hi), SfQ, AdSf (Forced, Val; Est, Lo, Hi),
    ValGainQ, ValGainSf (Est, Lo, Hi); and <regime>Matchers."""
    rows = {"initial matcher q": "Q", "additive datatype term": "Ad", "SF as published": "Sf",
            "SF − q": "SfQ", "A_d − SF": "AdSf", "validated − forced, q": "ValGainQ",
            "validated − forced, SF": "ValGainSf"}
    num = r"([+-]?\d+\.\d+)"
    out = {}
    for regime, tag in (("plain", "Pl"), ("md5", "Md"), ("indep", "In")):
        f = STUDY / "results" / vreg(regime) / "extraction_summary.md"
        if not f.is_file():
            continue
        t = f.read_text(encoding="utf-8")
        m = re.search(r"^(\d+) initial matchers", t, re.M)
        if m:
            out[f"{tag}Matchers"] = m.group(1)
        for line in t.splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if not c or c[0] not in rows:
                continue
            key = rows[c[0]]
            signed = key in ("SfQ", "AdSf", "ValGainQ", "ValGainSf")
            fmt = (lambda x: f"{x:+.3f}") if signed else (lambda x: f"{x:.3f}")
            cols = [("Forced", c[1]), ("Val", c[2])] if key not in ("ValGainQ", "ValGainSf") else [("", c[1])]
            for lab, cell in cols:
                v = re.findall(num, cell)
                if len(v) == 3:
                    for s, x in zip(("Est", "Lo", "Hi"), v):
                        out[f"{tag}{key}{lab}{s}"] = fmt(float(x))
    return out or None


def final_validation() -> dict[str, str] | None:
    """REGISTERED Amendment 12 (reference/score_final.py,): frozen recommendations on untouched data.
    Parses tuning/results/final_{A,B}_scores/report.md. Prefix fin.
      Part B: Pl / Op (regime), then FOne, FTwoA, FTwoB, FThree, FFour with Est, Lo, Hi, Verdict; Pl/OpScen, Pl/OpMatchers.
      Part A: A (primary truth) / AC (certain only): FFivePos, FFiveN, FFiveEst, FFiveLo, FFiveHi, FFiveVerdict; ANMatchers."""
    num = r"([+-]\d+\.\d+)"
    out = {}
    hk = {"F1": "FOne", "F2a": "FTwoA", "F2b": "FTwoB", "F3": "FThree", "F4": "FFour"}
    fb = STUDY / "tuning" / "results" / "final_B_scores" / "report.md"
    if fb.is_file():
        reg = None
        for line in fb.read_text(encoding="utf-8").splitlines():
            m = re.match(r"## Part B, (plain|opaque): (\d+) scenarios, (\d+) domains, (\d+) matchers", line)
            if m:
                reg = "Pl" if m.group(1) == "plain" else "Op"
                out[f"{reg}Scen"], out[f"{reg}Matchers"] = m.group(2), m.group(4)
                continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if reg and len(c) == 4 and c[0] in hk:
                v = re.findall(num, c[2])
                k = f"{reg}{hk[c[0]]}"
                fmt = "{:+.2f}" if c[0] == "F3" else "{:+.3f}"
                out.update({f"{k}Est": fmt.format(float(v[0])), f"{k}Lo": fmt.format(float(v[1])),
                            f"{k}Hi": fmt.format(float(v[2])), f"{k}Verdict": c[3].lower()})
    fa = STUDY / "tuning" / "results" / "final_A_scores" / "report.md"
    if fa.is_file():
        tag = None
        for line in fa.read_text(encoding="utf-8").splitlines():
            m = re.match(r"## Part A, (primary|sensitivity).*: (\d+) pairs, (\d+) matchers", line)
            if m:
                tag = "A" if m.group(1) == "primary" else "AC"
                out["ANMatchers"] = m.group(3)
                continue
            m = re.match(r"F5: (\d+) of (\d+) pairs positive; mean " + num + r" \[" + num + ", " + num + r"\] -> (.*)$", line)
            if m and tag:
                out.update({f"{tag}FFivePos": m.group(1), f"{tag}FFiveN": m.group(2),
                            f"{tag}FFiveEst": f"{float(m.group(3)):+.3f}", f"{tag}FFiveLo": f"{float(m.group(4)):+.3f}",
                            f"{tag}FFiveHi": f"{float(m.group(5)):+.3f}", f"{tag}FFiveVerdict": m.group(6)})
    return out or None


def t2d_exploratory() -> dict[str, str] | None:
    """EXPLORATORY T2D replication (reference/explore_e2_t2d.py: the frozen score_e2.py on t2d-usable). Parses
    tuning/results/t2d_exploratory_scores/report.md. Prefix ttd: POne, PTwo, PThree, PFour, CoefProdFd, Ucc with Est,
    Lo, Hi, Consistent (true/false); N (matchers)."""
    f = STUDY / "tuning" / "results" / "t2d_exploratory_scores" / "report.md"
    if not f.is_file():
        return None
    num = r"([+-]\d+\.\d+)"
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 6:
            continue
        lab = c[0].replace("**", "")
        key = ("POne" if lab.startswith("P1") else "PTwo" if lab.startswith("P2") else "PThree" if lab.startswith("P3")
               else "PFour" if lab.startswith("P4") else "CoefProdFd" if lab.startswith("coef_prod_fd")
               else "Ucc" if lab.startswith("ucc") else None)
        if key is None:
            continue
        est = re.findall(num, c[2])
        ci = re.findall(num, c[3])
        if not est or len(ci) != 2:
            continue
        out.update({f"{key}Est": f"{float(est[0]):+.3f}", f"{key}Lo": f"{float(ci[0]):+.3f}", f"{key}Hi": f"{float(ci[1]):+.3f}",
                    f"{key}Consistent": "true" if c[-1].startswith("consistent") else "false"})
        out["N"] = c[1]
    return out or None


def e2_registered() -> dict[str, str] | None:
    """REGISTERED E2 (Amendment 7; score_e2.py on ctl). Parses tuning/results/e2_ctl_scores/report.md.
    Read under the rule logged before any E2 output: P1, P3 decided on ctl-plain, P4 on ctl (both regimes),
    P2 as registered; the P1 opaque line is descriptive. Prefix etwo: P1Pl, P1Md, P2Pl, P2Md, P3, P4Pl, P4Md, then Est,
    Lo, Hi, Verdict (lowercase; P1Md "descriptive"); N (matchers)."""
    f = STUDY / "tuning" / "results" / "e2_ctl_scores" / "report.md"
    if not f.is_file():
        return None
    num = r"([+-]\d+\.\d+)"
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 6:
            continue
        lab = c[0].replace("**", "")
        reg = "Pl" if "(plain" in lab else "Md" if "(opaque" in lab else None
        key = None
        if lab.startswith("P1"):
            key = f"POne{reg}"
        elif lab.startswith("P2"):
            key = f"PTwo{reg}"
        elif lab.startswith("P3"):
            key = "PThree"
        elif lab.startswith("P4"):
            key = f"PFour{reg}"
        if key is None or reg is None:
            continue
        est = re.findall(num, c[2]) or re.findall(num, c[3])
        ci = re.findall(num, c[3]) if len(re.findall(num, c[3])) == 2 else re.findall(num, c[4])
        out.update({f"{key}Est": f"{float(est[0]):+.3f}", f"{key}Lo": f"{float(ci[0]):+.3f}", f"{key}Hi": f"{float(ci[1]):+.3f}"})
        verdict = c[-1].lower().replace(" (decided on t2d)", "")
        out[f"{key}Verdict"] = "descriptive" if key == "POneMd" else verdict
        out["N"] = c[1]
    return out or None


def deproles() -> dict[str, str] | None:
    """EXPLORATORY X46 (explore_deproles.py): an additive term with dependency roles against SF with dependency
    edges. Parses results/<regime>/deproles_summary.md. Prefix dr, then Pl / Md, then <plus>Vs<minus> (Role, Nb, Both,
    Ad, and SF arms Ref, Fd, Ucc, FdUccInd, UccTop, UccFdTopk), then Est, Lo, Hi, Pos, N; and <regime>RoleGamma (the
    role weight if one value was chosen in every fold) and <regime>N (matchers)."""
    arm = {"A_role": "Role", "A_nb": "Nb", "A_both": "Both", "A_d": "Ad", "SF melnik": "Ref", "SF fd": "Fd",
           "SF ucc": "Ucc", "SF fd_ucc_ind": "FdUccInd", "SF ucc_top1": "UccTop", "SF ucc1_fd_topk": "UccFdTopk"}
    num = r"([+-]\d+\.\d+)"
    out = {}
    for reg, pre in (("plain", "Pl"), ("md5", "Md")):
        f = STUDY / "results" / vreg(reg) / "deproles_summary.md"
        if not f.is_file():
            continue
        t = f.read_text(encoding="utf-8")
        m = re.search(r"(\d+) initial matchers", t)
        if m:
            out[f"{pre}N"] = m.group(1)
        m = re.search(r"- role: (.*)", t)
        if m:
            gammas = set(re.findall(r"\(([\d.]+), [\d.]+\)", m.group(1)))
            if len(gammas) == 1:
                out[f"{pre}RoleGamma"] = gammas.pop()
        for line in t.splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) != 3 or " − " not in c[0]:
                continue
            x, y = c[0].split(" − ")
            if x not in arm or y not in arm:
                continue
            k = f"{pre}{arm[x]}Vs{arm[y]}"
            v = re.findall(num, c[1])
            p = re.search(r"(\d+) of (\d+)", c[2])
            out.update({f"{k}Est": f"{float(v[0]):+.3f}", f"{k}Lo": f"{float(v[1]):+.3f}",
                        f"{k}Hi": f"{float(v[2]):+.3f}", f"{k}Pos": p.group(1), f"{k}N": p.group(2)})
    return out or None


def timing_pass() -> dict[str, str] | None:
    """The dedicated RQ5 timing pass (descriptive): 40 configurations on the 382 plain scenarios, at most 4 timing
    tasks per host. Parses tuning/results/timing_plain_report.md. Prefix tpass, then the configuration (Ref = levenshtein
    melnik, Fd, Reif, Std = levenshtein seedstudy, Ind, Ucc, All = fd_ucc_ind, SbertStd, PmfStd), then WallScen and
    CpuScen (s per scenario, wall and CPU summed over the dataset tasks), RssGB (peak RSS), BuildS (median build, s),
    IterMs (median iterate, ms); OtherRssLo/OtherRssHi (GB, range over the configurations without reified FDs),
    Configs, Scenarios."""
    f = STUDY / "tuning" / "results" / "timing_plain_report.md"
    if not f.is_file():
        return None
    names = {"levenshtein__melnik__L1": "Ref", "levenshtein__fd__L1": "Fd", "levenshtein__fd_reified__L3": "Reif",
             "levenshtein__seedstudy": "Std", "levenshtein__ind__L3": "Ind", "levenshtein__ucc__L3": "Ucc",
             "levenshtein__fd_ucc_ind__L3": "All", "sbert_mpnet__seedstudy": "SbertStd", "pmf__seedstudy": "PmfStd",
             "kang_naughton__seedstudy": "KnStd"}
    out, other_rss, n_cfg, n_scen = {}, [], 0, 0
    for line in f.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) != 11 or not re.fullmatch(r"\d+", c[1]):
            continue
        n_cfg += 1
        wall, cpu, rss, scen = float(c[2]), float(c[3]), float(c[4]) / 1024, int(c[5])
        n_scen = max(n_scen, scen)
        if "reified" not in c[0]:
            other_rss.append(rss)
        if c[0] in names and scen:
            k = names[c[0]]
            out.update({f"{k}WallScen": f"{wall / scen:.1f}", f"{k}CpuScen": f"{cpu / scen:.1f}",
                        f"{k}RssGB": f"{rss:.1f}", f"{k}BuildS": f"{float(c[9]) / 1000:.1f}",
                        f"{k}IterMs": f"{float(c[10]):,.0f}"})
    if other_rss:
        out["OtherRssLo"], out["OtherRssHi"] = f"{min(other_rss):.1f}", f"{max(other_rss):.1f}"
    out["Configs"], out["Scenarios"] = str(n_cfg), str(n_scen)
    return out if n_cfg else None


def iterations() -> dict[str, str] | None:
    """EXPLORATORY X42 (explore_iterations.py, written before outputs): SF's iteration trajectory. Parses
    tuning/results/iter_<regime>_scores/report.md. Prefix it, then Ind / Pl / Md, then the arm (Ref = melnik,
    Shape = melnik_shape, Fd = fd), then the step (One, Two, Three, Five, Ten, Conv), then Gain, GainLo, GainHi, R, F,
    Diff (minus converged), Build, Iter (median ms); and <regime><arm>N (matchers), <regime>InvMax (largest number of
    within-block inversions over t; Valentine only)."""
    regs = {"indep": "Ind", "plain": "Pl", "md5": "Md"}
    arms = {"melnik": "Ref", "melnik_shape": "Shape", "fd": "Fd"}
    steps = {"1": "One", "2": "Two", "3": "Three", "5": "Five", "10": "Ten", "conv": "Conv"}
    num = r"([+-]\d+\.\d+)"
    out = {}
    for reg, pre in regs.items():
        f = STUDY / "tuning" / "results" / f"iter_{vreg(reg)}_scores" / "report.md"
        if not f.is_file():
            continue
        arm, inv = None, []
        for line in f.read_text(encoding="utf-8").splitlines():
            m = re.match(r"## (\w+) \((\d+) matchers\)", line)
            if m:
                arm = arms.get(m.group(1))
                if arm:
                    out[f"{pre}{arm}N"] = m.group(2)
                continue
            if line.startswith("## "):
                arm = "inv" if line.startswith("## Proposition 1") else None
                continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if arm == "inv" and len(c) == 3 and c[0].isdigit():
                inv.append(int(c[1]))
            elif arm and len(c) == 8 and c[0] in steps:
                k = f"{pre}{arm}{steps[c[0]]}"
                g = re.findall(num, c[1])
                out.update({f"{k}Gain": f"{float(g[0]):+.3f}", f"{k}GainLo": f"{float(g[1]):+.3f}",
                            f"{k}GainHi": f"{float(g[2]):+.3f}", f"{k}R": f"{float(c[2]):+.3f}",
                            f"{k}F": f"{float(c[3]):+.3f}", f"{k}Build": f"{float(c[6]):,.0f}",
                            f"{k}Iter": f"{float(c[7]):.0f}"})
                d = re.findall(num, c[4])
                if d:
                    out[f"{k}Diff"] = f"{float(d[0]):+.3f}"
        if inv:
            out[f"{pre}InvMax"] = str(max(inv))
    return out or None


def tablectx() -> dict[str, str] | None:
    """REGISTERED Amendment 9 (reference/score_tablectx.py,): a non-iterative table-aware additive
    baseline against SF on the independent corpus. Parses tuning/results/tablectx_indep_scores/report.md. Prefix tc:
    HC0/HC1/HC2/HC3 as HZero/HOne/HTwo/HThree with Est, Lo, Hi, Pos, N, Decision and per database (AW, Ch, TPCH);
    gains over q per matrix (Sf, SfP, SfS, Ad, Ap, As, Adt, Apt, Ast) with J, R, F; Lambda (if one value everywhere);
    and LamAdt<k> for the λ curve of A_dt (k = position in the grid: One .. Seven)."""
    f = STUDY / "tuning" / "results" / "tablectx_indep_scores" / "report.md"
    if not f.is_file():
        return None
    t = f.read_text(encoding="utf-8")
    num = r"([+-]\d+\.\d+)"
    hk = {"HC0": "HZero", "HC1 (primary)": "HOne", "HC2": "HTwo", "HC3": "HThree"}
    mk = {"sf": "Sf", "sf_p": "SfP", "sf_s": "SfS", "A_d": "Ad", "A_p": "Ap", "A_s": "As", "A_dt": "Adt",
          "A_pt": "Apt", "A_st": "Ast"}
    db = {"Indep-AdventureWorks": "AW", "Indep-Chinook": "Ch", "Indep-TPCH": "TPCH"}
    pos_words = ("One", "Two", "Three", "Four", "Five", "Six", "Seven")
    out, header = {}, []
    for line in t.splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if c and c[0] == "hypothesis":
            header = c
            continue
        if c and c[0] in hk and header:
            k = hk[c[0]]
            ci = re.findall(num, c[3])
            out.update({f"{k}Est": f"{float(c[2]):+.3f}", f"{k}Lo": f"{float(ci[0]):+.3f}", f"{k}Hi": f"{float(ci[1]):+.3f}"})
            for i, name in enumerate(header):
                if name in db:
                    out[f"{k}{db[name]}"] = f"{float(c[i]):+.3f}"
            m = re.search(r"(\d+) of (\d+)", c[-2])
            out[f"{k}Pos"], out[f"{k}N"], out[f"{k}Decision"] = m.group(1), m.group(2), c[-1].lower()
        elif len(c) == 4 and c[0] in mk and re.fullmatch(num, c[1] or ""):
            k = mk[c[0]]
            out.update({f"Gain{k}J": f"{float(c[1]):+.3f}", f"Gain{k}R": f"{float(c[2]):+.3f}",
                        f"Gain{k}F": f"{float(c[3]):+.3f}"})
        elif len(c) == 8 and c[0] == "A_dt" and re.fullmatch(num, c[1] or ""):
            for i, v in enumerate(c[1:]):
                out[f"LamAdt{pos_words[i]}"] = f"{float(v):+.3f}"
    lams = set(re.findall(r"Indep-\w+ (\d+\.\d+)", t.split("\n\n")[1] if "\n\n" in t else t))
    if len(lams) == 1:
        out["Lambda"] = lams.pop()
    return out or None


def classfull() -> dict[str, str] | None:
    """REGISTERED Amendment 8 (+ erratum 1; score_classfull.py, run once per regime): class nodes on the full
    panel. Parses tuning/results/classfull_<regime>_scores/report.md, primary case set (non-screen). Prefix cf, then
    Pl / Md, then <plus>Vs<minus> (Shape, ShapeShuf, Sem, SemShuf, Ref, Lp, Ls, Bp, Bs, Ap, As), then Est, Lo, Hi, R,
    F, Pos, N, Verdict; and Pl/Md + InvShape/InvSem (inversions) and PairsShape/PairsSem (millions of ordered pairs)."""
    arm = {"shape": "Shape", "shape_shuffled": "ShapeShuf", "semtype": "Sem", "semtype_shuffled": "SemShuf",
           "melnik": "Ref", "L_p": "Lp", "L_s": "Ls", "B_p": "Bp", "B_s": "Bs", "A_p": "Ap", "A_s": "As"}
    out = {}
    num = r"([+-]\d+\.\d+)"
    for reg, pre in (("plain", "Pl"), ("md5", "Md")):
        f = STUDY / "tuning" / "results" / f"classfull_{reg}_scores" / "report.md"
        if not f.is_file():
            continue
        t = f.read_text(encoding="utf-8")
        on = False
        for line in t.splitlines():
            if line.startswith("## "):
                on = line.startswith("## primary")
                continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if not on or len(c) != 7 or " − " not in c[0]:
                continue
            plus, minus = c[0].split(": ", 1)[1].split(" − ")
            k = f"{pre}{arm[plus]}Vs{arm[minus]}"
            v = re.findall(num, c[2])
            m = re.search(r"(\d+) of (\d+)", c[5])
            out.update({f"{k}Est": f"{float(v[0]):+.3f}", f"{k}Lo": f"{float(v[1]):+.3f}", f"{k}Hi": f"{float(v[2]):+.3f}",
                        f"{k}R": f"{float(c[3]):+.3f}", f"{k}F": f"{float(c[4]):+.3f}", f"{k}Pos": m.group(1),
                        f"{k}N": m.group(2)})
            if c[6]:
                out[f"{k}Verdict"] = c[6].lower()
        for v, key in (("shape", "Shape"), ("semtype", "Sem")):
            m = re.search(rf"H8a \({v}\): (\d+) within-block inversions of (\d+) ordered pairs", t)
            if m:
                out[f"{pre}Inv{key}"], out[f"{pre}Pairs{key}"] = m.group(1), f"{int(m.group(2)) / 1e6:,.0f}"
    return out or None


def lockbox() -> dict[str, str] | None:
    """REGISTERED lockbox (tuning Amendment 4; score_lockbox_hl.py and confirm_tiebreak.py --role own). Parses
    tuning/results/lockbox_plain/{hl_report.md, tiebreak_primary.txt}. Prefix lbx."""
    d = STUDY / "tuning" / "results" / "lockbox_plain"
    f = d / "hl_report.md"
    if not f.is_file():
        return None
    t = f.read_text(encoding="utf-8")
    num = r"([+-]\d+\.\d+)"
    out = {}
    m = re.search(r"(\d+) lockbox scenarios .*?; (\d+) non-control initial matchers", t)
    out["Scen"], out["N"] = m.group(1), m.group(2)
    line = next(l for l in t.splitlines() if l.startswith("| mean J_tf"))
    v = re.findall(num, line)
    out.update({"Own": f"{float(v[0]):+.3f}", "Uni": f"{float(v[1]):+.3f}", "Prod": f"{float(v[2]):+.3f}",
                "AlphaZero": f"{float(v[3]):+.3f}".replace("-0.000", "0.000"), "Ref": f"{float(v[4]):+.3f}"})
    for h, k in (("HL1", "HOne"), ("HL2", "HTwo"), ("HL3", "HThree"), ("HL4", "HFour")):
        line = next(l for l in t.splitlines() if l.startswith(f"| {h} |"))
        v = re.findall(num, line)
        mm = re.search(r"(\d+) of (\d+)", line)
        out.update({f"{k}": f"{float(v[0]):+.3f}", f"{k}Lo": f"{float(v[1]):+.3f}", f"{k}Hi": f"{float(v[2]):+.3f}",
                    f"{k}CLo": f"{float(v[3]):+.3f}", f"{k}CHi": f"{float(v[4]):+.3f}", f"{k}Pos": mm.group(1)})
    # per matcher: where the report-only reference beats the matcher's own tuned configuration
    ahead = n = 0
    for line in t.split("## Per initial matcher")[-1].splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 7 and "control" not in c[0] and re.fullmatch(r"[+-]\d+\.\d+", c[2]):
            n += 1
            ahead += float(c[6]) > float(c[2]) + 1e-4
    out["RefAhead"], out["RefN"] = str(ahead), str(n)
    tb = d / "tiebreak_primary.txt"
    if tb.is_file():
        tt = tb.read_text(encoding="utf-8")
        for h, k in (("H1", "TbOne"), ("H2", "TbTwo"), ("H3", "TbThree")):
            mm = re.search(rf"^{h} .*?: {num}\s+95% CI \[{num}, {num}\]", tt, re.M)
            out.update({k: f"{float(mm.group(1)):+.3f}", f"{k}Lo": f"{float(mm.group(2)):+.3f}",
                        f"{k}Hi": f"{float(mm.group(3)):+.3f}"})
        mm = re.search(rf"F1_Hr\(tiebreak\) - F1_Hr\(initial\): {num}", tt)
        out["TbF"] = f"{float(mm.group(1)):+.3f}"
    return out


def er10() -> dict[str, str] | None:
    """EXPLORATORY ER10 (reference/explore_er10.py, written before outputs): SF with the same class nodes as
    Amendment 5's additive terms, on the independent corpus. Parses tuning/results/er10_indep_scores/report.md.
    Prefix erten: method gains over q (SfJ, SfPJ, SfSJ, ApJ, AsJ) and contrasts (Est, Lo, Hi, Pos, N)."""
    f = STUDY / "tuning" / "results" / "er10_indep_scores" / "report.md"
    if not f.is_file():
        return None
    out = {}
    num = r"([+-]\d+\.\d+)"
    meth = {"SF": "Sf", "SF+P": "SfP", "SF+S": "SfS", "A_p": "Ap", "A_s": "As", "SF+P~": "SfPShuf", "SF+S~": "SfSShuf"}
    con = {"A_p − SF+P": "ApVsSfP", "A_s − SF+S": "AsVsSfS", "SF+P − SF": "SfPVsSf", "SF+S − SF": "SfSVsSf",
           "SF+P − SF+P~": "SfPVsShuf", "SF+S − SF+S~": "SfSVsShuf"}
    for line in f.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 5 and c[0] in meth:
            out[f"{meth[c[0]]}J"] = f"{float(c[1]):+.3f}"
        if len(c) == 7 and c[0] in con:
            k = con[c[0]]
            ci = re.findall(num, c[2])
            m = re.search(r"(\d+) of (\d+)", c[6])
            out.update({f"{k}Est": f"{float(c[1]):+.3f}", f"{k}Lo": f"{float(ci[0]):+.3f}", f"{k}Hi": f"{float(ci[1]):+.3f}",
                        f"{k}Pos": m.group(1), f"{k}N": m.group(2)})
    return out or None


def amendment5() -> dict[str, str] | None:
    """REGISTERED (Amendment 5 of PREREGISTRATION_LAYERS.md): additive datatype-and-class terms against SF as
    published on the independent corpus. Parses tuning/results/layer1_indep_scores/additive_report.md (
    reference/score_additive_indep.py, hash-pinned). Prefix amf; HOne/HTwo/HThree = A_p/A_s/A_d − SF."""
    f = STUDY / "tuning" / "results" / "layer1_indep_scores" / "additive_report.md"
    if not f.is_file():
        return None
    t = f.read_text(encoding="utf-8")
    out = {}
    m = re.search(r"(\d+) initial matchers with full coverage, (\d+) scenarios", t)
    out["N"], out["Scen"] = m.group(1), m.group(2)
    num = r"([+-]\d+\.\d+)"
    for h, k in (("HB1", "HOne"), ("HB2", "HTwo"), ("HB3", "HThree")):
        line = next(l for l in t.splitlines() if l.startswith(f"| {h} "))
        v = re.findall(num, line)
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        out.update({f"{k}": f"{float(v[0]):+.3f}", f"{k}Lo": f"{float(v[1]):+.3f}", f"{k}Hi": f"{float(v[2]):+.3f}",
                    f"{k}Aw": f"{float(v[3]):+.3f}", f"{k}Ch": f"{float(v[4]):+.3f}", f"{k}Tp": f"{float(v[5]):+.3f}",
                    f"{k}Pos": c[-2].split()[0], f"{k}Verdict": c[-1]})
    for name, k in (("sf", "Sf"), ("A_d", "Ad"), ("A_p", "Ap"), ("A_s", "As")):
        line = next(l for l in t.splitlines() if l.startswith(f"| {name} |"))
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        out.update({f"{k}J": f"{float(c[1]):+.3f}", f"{k}R": f"{float(c[2]):+.3f}", f"{k}F": f"{float(c[3]):+.3f}"})
    return out


SHORT = {"semstruct": "SemStruct", "learned_lofo": "Learned", "jsd_values": "Jsd", "kang_naughton": "Kn",
         "null_percentage": "Null", "uniqueness": "Uniq", "prefix_suffix": "Ps", "pmf": "Pmf",
         "role_gcov": "Gcov", "constant": "Const", "oracle": "Oracle", "containment": "Contain",
         "unionability": "Union", "levenshtein": "Lev", "unary_statistics": "Unary",
         "char_distribution": "Chardist", "sbert_e5": "Efive", "valentine_jaccard": "Vjacc",
         "composite_max": "Compmax"}
MSHORT = {"R_exp": "R", "R_pess": "Rp", "F1_Hr": "F", "MRR_exp": "M", "F1_M": "Fm"}


def rq_numbers(regime: str, points: list[dict], rows: list[dict] | None) -> dict[str, str]:
    matchers = [p for p in points if p["group"] != "ctrl"]
    x = [p["x"] for p in matchers]
    nums: dict[str, str] = {"N": str(len(matchers))}
    for metric, key in (("R_exp", "Rexp"), ("F1_Hr", "FHr"), ("MRR_exp", "MRRexp"), ("F1_M", "FM"),
                        ("R_pess", "Rpess"), ("MRR_pess", "MRRpess"), ("F1_H", "FH")):
        ys = [p[f"d_{metric}"] for p in matchers]
        nums[f"Rho{key}"] = f"{spearman(x, ys):.2f}"
        nums[f"Up{key}"] = str(sum(y > 0 for y in ys))
        nums[f"Mean{key}"] = f"{sum(ys) / len(ys):+.3f}"
        nums[f"Min{key}"] = f"{min(ys):+.2f}"
        nums[f"Max{key}"] = f"{max(ys):+.2f}"
    nums.update({k: f"{v:.2f}" for k, v in per_family(regime, {p['seed'] for p in matchers}, rows).items()})
    # registered H2 (PREREGISTRATION_LAYERS.md): as published, |macro dR_exp| < 0.03 for every matcher
    nums["BigR"] = str(sum(abs(p["d_R_exp"]) >= 0.03 for p in matchers))
    # Mathematical coupling (external review): x and SF's change d share the baseline, so a negative
    # rho can arise from regression to the mean or a ceiling. Oldham's method correlates d with the
    # mean of before and after, (x + (x + d)) / 2 = x + d/2, which removes the shared term.
    for metric, key in (("R_exp", "Rexp"), ("F1_Hr", "FHr")):
        ds = [p[f"d_{metric}"] for p in matchers]
        own = [p[f"base_{metric}"] for p in matchers]   # the metric's own baseline
        nums[f"RhoOld{key}"] = f"{spearman([o + d / 2 for o, d in zip(own, ds)], ds):.2f}"
        # one point per matcher family (configurations of one family are not independent)
        fam = defaultdict(list)
        for p in matchers:
            fam[META[p["seed"]][0]].append(p)
        fx = [sum(q["x"] for q in v) / len(v) for v in fam.values()]
        fy = [sum(q[f"d_{metric}"] for q in v) / len(v) for v in fam.values()]
        nums[f"RhoFam{key}"] = f"{spearman(fx, fy):.2f}"
        nums["NFam"] = str(len(fam))
    by = {p["seed"]: p for p in points}
    for s, short in SHORT.items():
        if s in by:
            for m, ms in MSHORT.items():
                nums[f"D{short}{ms}"] = f"{by[s][f'd_{m}']:+.3f}"
    return nums


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="plain")
    a = ap.parse_args()
    # Standard configuration (the seed study): RQ4's figure, the exploratory section, the contrast in RQ1.
    points, n = load(a.regime)
    matchers = [p for p in points if p["group"] != "ctrl"]
    out = OUT
    out.mkdir(exist_ok=True)
    # Reference configuration (layer 1, arm melnik): Fig. 3 and RQ1, once layer 1 covers every matcher.
    ref_rows = layer1_rows(a.regime, "melnik")
    ref_points, n_ref = load_rows(ref_rows) if ref_rows else ([], 0)
    use_ref = ({p["seed"] for p in ref_points if p["group"] != "ctrl"} >= {p["seed"] for p in matchers})
    fig_points = ref_points if use_ref else points
    for prefix, pts, rows in (("rqOne", fig_points, ref_rows if use_ref else None), ("rqStd", points, None)):
        nums = rq_numbers(a.regime, pts, rows)
        (out / f"{'rq1' if prefix == 'rqOne' else 'rq1_std'}_numbers.tex").write_text(
            f"% Generated by make_study_numbers.py ({'layer 1, arm melnik' if prefix == 'rqOne' and use_ref else 'seed study, standard configuration'}); do not edit.\n"
            "% Rho<metric>: Spearman between the expected Recall@GT of the initial matcher alone and SF's\n"
            "% change in <metric>, over the matchers (controls excluded). Up<metric>: matchers SF improves.\n"
            "% D<matcher><metric>: SF's change for one matcher (R expected Recall@GT, Rp pessimistic, F tie-free F1,\n"
            "% M expected MRR, Fm F1 after mutual top-1).\n"
            + "".join(f"\\newcommand{{\\{prefix}{k}}}{{{v}}}\n" for k, v in nums.items()), encoding="utf-8")
    (out / "robust_numbers.tex").write_text(
        "% Generated by make_study_numbers.py from r1_robust_scores.csv and r1_margins.csv (explore_robust.py) and\n"
        "% tuning/results/x84_noise_scores/summary.csv (X84, keys Lit*);\n"
        "% EXPLORATORY; do not edit.\n"
        + "".join(f"\\newcommand{{\\rob{k}}}{{{v}}}\n" for k, v in {**robust(a.regime), **noise_ref()}.items()),
        encoding="utf-8")
    if a.regime == "plain" and (STUDY / "results" / vreg("md5") / "summary_tiefree.csv").is_file():
        md5_points, n_md5 = load("md5")
        fig4, nums4 = rq4(points, md5_points, n_md5)
        (out / "rq4_numbers.tex").write_text(
            "% Generated by make_study_numbers.py from results/{plain549,md549}/summary_tiefree.csv; do not edit.\n"
            + "".join(f"\\newcommand{{\\rqFour{k}}}{{{v}}}\n" for k, v in nums4.items()), encoding="utf-8")
    for fn, name, prefix, src in ((dtype_baseline, "dt_numbers.tex", "dt", "results/plain/dtype_baseline_summary.md"),
                                  (multitable_ext, "mt_numbers.tex", "mt",
                                   "results/multitable_fix/{synthea,indep2}/multitable_summary.md (explore_multitable_ext.py, X54/X55)"),
                                  (family_strength, "fam_numbers.tex", "fam",
                                   "tuning/results/x58_{plain549,md549}/report.md (x58_family.py, X58)"),
                                  (parallel_edges, "pe_numbers.tex", "pe",
                                   "tuning/results/parallel_{plain549,md549}_scores/report.md (score_parallel.py, EXPLORATORY)"),
                                  (scaling_sweep, "sw_numbers.tex", "sw",
                                   "tuning/results/x51_scaling_report.md (X51 sweep; predictions before any run)"),
                                  (extraction, "xe_numbers.tex", "xe",
                                   "results/{plain549,md549}/extraction_summary.md (explore_extraction.py, X50)"),
                                  (final_validation, "fin_numbers.tex", "fin",
                                   "tuning/results/final_{A,B}_scores/report.md (score_final.py, registered, Amendment 12)"),
                                  (external_x81, "xt_numbers.tex", "xt",
                                   "results/x81fix/report.md and results/hdxsm_fd_cap.json (explore_x81.py; the cap from the run log; X81, EXPLORATORY)"),
                                  (followup_x94_x95, "fw_numbers.tex", "fw",
                                   "results/x94_549/stage1.md, tuning/results/x94_549/cells.csv.gz, tuning/results/x95_549/contrasts.csv, tuning/results/x95/posthoc_matched.csv (X94 and X95, EXPLORATORY; X95 fixed in advance for the follow-up paper)"),
                                  (role_fullmetrics, "drm_numbers.tex", "drm",
                                   "results/plain549/x46_fullmetrics_summary.csv, results/md549/x46_fullmetrics_summary.csv (explore_x46_fullmetrics.py, X46 rescoring, EXPLORATORY)"),
                                  (matcher_roles_x73, "mr_numbers.tex", "mr",
                                   "results/plain/matcherroles_summary.md, results/md5/matcherroles_summary.md (X73, EXPLORATORY)"),
                                  (literal_term, "ld_numbers.tex", "ld",
                                   "tuning/results/x93_549_scores/contrasts.csv, tuning/results/x93_549_scores/mechanism.md, tuning/results/x93_549_scores/lambda.json (X93, EXPLORATORY)"),
                                  (corpus_effects, "ce_numbers.tex", "ce",
                                   "results/x78_549/benchmark_effects_report.md (x78_analysis.py, X78, EXPLORATORY)"),
                                  (rule_checks, "rlx_numbers.tex", "rlx",
                                   "results/x92/report.md (explore_x92_rules.py, X92, EXPLORATORY)"),
                                  (export_check, "gx_numbers.tex", "gx",
                                   "results/sf_export_check/report.md and results/sf_export_check/components.csv (X91 export checker, EXPLORATORY)"),
                                  (stop_two, "sv_numbers.tex", "sv",
                                   "results/x88/validate.md (explore_x88_validate.py, X88 held-out test, EXPLORATORY)"),
                                  (literal_offsets, "lt_numbers.tex", "lt",
                                   "tuning/results/x90_549_scores/report.md (score_x90.py, X90, EXPLORATORY)"),
                                  (rq1_checks, "cp_numbers.tex", "cp",
                                   "results/x89/report.md (explore_x89_rq1.py, X89, EXPLORATORY)"),
                                  (iter_rexp, "itr_numbers.tex", "itr",
                                   "tuning/results/iter_{plain549,md549,indep}_scores/report.md (explore_iterations.py, X42, EXPLORATORY)"),
                                  (wide_x82, "wp_numbers.tex", "wp",
                                   "results/x82/report.md (explore_x82_bf.py, X82 (f), (b), EXPLORATORY)"),
                                  (characterise_x85, "xc_numbers.tex", "xc",
                                   "results/x85/{report,leak,leak_effect}.md, results/x85/corpus_table.csv (explore_x85_*.py, X85) and tuning/results/x87_scores/summary.csv, tuning/results/x87_scores/fd_diff_summary.csv, tuning/results/x87rp_scores/summary.csv (X87); EXPLORATORY"),
                                  (properties_x81, "pr_numbers.tex", "pr",
                                   "results/x81_549/benchmark_profile_x81.md (explore_x81_props.py, X77/X81, EXPLORATORY)"),
                                  (knobs_x79, "kn_numbers.tex", "kn",
                                   "tuning/results/x79_knobs_scores/report.md (x79_knobs.py, X79 (b), EXPLORATORY)"),
                                  (c_x82_x83, "cx_numbers.tex", "cx",
                                   "tuning/results/{x82_classfull_scores_dev1,x82_parallel_scores_dev1,x82_e2x_scores_dev1,x83_class_additive,x86_cluster_ci}/contrasts.csv (X82, X83, X86, EXPLORATORY)"),
                                  (t2d_exploratory, "ttd_numbers.tex", "ttd",
                                   "tuning/results/t2d_exploratory_scores/report.md (explore_e2_t2d.py, EXPLORATORY T2D)"),
                                  (e2_registered, "etwo_numbers.tex", "etwo",
                                   "tuning/results/e2_ctl_scores/report.md (score_e2.py, registered, Amendment 7)"),
                                  (deproles, "dr_numbers.tex", "dr",
                                   "results/{plain549,md549}/deproles_summary.md (explore_deproles.py, X46)"),
                                  (timing_pass, "tpass_numbers.tex", "tpass",
                                   "tuning/results/timing_plain_report.md (dedicated timing pass; descriptive)"),
                                  (iterations, "it_numbers.tex", "it",
                                   "tuning/results/iter_{plain549,md549}_scores/report.md (explore_iterations.py, X42)"),
                                  (tablectx, "tc_numbers.tex", "tc",
                                   "tuning/results/tablectx_indep_scores/report.md (score_tablectx.py, registered, "
                                   "Amendment 9)"),
                                  (classfull, "cf_numbers.tex", "cf",
                                   "tuning/results/classfull_<regime>_scores/report.md (score_classfull.py, registered, Amendment 8)"),
                                  (lockbox, "lbx_numbers.tex", "lbx",
                                   "tuning/results/lockbox_plain/hl_report.md (score_lockbox_hl.py, registered)"),
                                  (er10, "erten_numbers.tex", "erten",
                                   "tuning/results/er10_indep_scores/report.md (explore_er10.py, ER10)"),
                                  (amendment5, "amf_numbers.tex", "amf",
                                   "tuning/results/layer1_indep_scores/additive_report.md (Amendment 5, registered)")):
        vals = fn()
        if vals is not None:
            (out / name).write_text(
                f"% Generated by make_study_numbers.py from {src}; "
                f"{'REGISTERED' if 'registered' in src else 'EXPLORATORY'}; do not edit.\n"
                + "".join(f"\\newcommand{{\\{prefix}{k}}}{{{v}}}\n" for k, v in vals.items()), encoding="utf-8")
    for reg, prefix in (("plain", "lp"), ("md5", "lm")):
        nums = layer1(reg)
        if nums is not None:
            (out / f"layer1_{reg}_numbers.tex").write_text(
                f"% Generated by make_study_numbers.py from tuning/results/layer1_{vreg(reg)}_scores/report.md; do not edit.\n"
                + "".join(f"\\newcommand{{\\{prefix}{k}}}{{{v}}}\n" for k, v in nums.items()), encoding="utf-8")
    # RQ1 per corpus (rqc): Valentine readable (layer 1, as Fig. 3) and recoded, with matcher-resampled intervals
    rqc = {}
    for reg, key, pts in (("plain", "VPl", fig_points), ("md5", "VMd", None)):
        if pts is None:
            rows_md = layer1_rows(reg, "melnik")
            pts = load_rows(rows_md)[0] if rows_md else []
        if pts:
            rho, lo, hi, nm = rho_matcher_ci(pts)
            rqc.update({f"{key}Rho": f"{rho:+.2f}", f"{key}RhoLo": f"{lo:+.2f}", f"{key}RhoHi": f"{hi:+.2f}", f"{key}N": str(nm)})
    (out / "rqc_numbers.tex").write_text(
        "% Generated by make_study_numbers.py: from tuning/results/layer1_{plain549,md549}_scores/scores.csv (layer 1, arm melnik): RQ1's rho on Valentine, 95% interval\n"
        "% resampling matchers as explore_x82_bf.py does for the new corpora; EXPLORATORY; do not edit.\n"
        + "".join(f"\\newcommand{{\\rqc{k}}}{{{v}}}\n" for k, v in rqc.items()), encoding="utf-8")
    print(f"{len(points)} points ({n} scenarios each); figures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
