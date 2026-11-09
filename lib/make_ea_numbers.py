"""Write output/ea_numbers.tex (+ output/ea_numbers_report.md): two results first reported in an earlier
manuscript. EXPLORATORY: both are post-hoc follow-ups, not registered tests.

Result 1, positional subsidy on Valentine (FD-free), on the 549 scenarios. Sources:
  results/sf_study/results/valentine549/positional_survey.json   ground-truth diagonal survey of the 549
                                    (experiments/sf_study/reference/positional_survey_549.py);
  results/baselines/reports/positional_subsidy_549.json   Hungarian F1 with ties broken by index order (hung) minus the
                                    same after permuting the target columns, mean of 20 permutations (hung_perm), per
                                    arm, regime, kind and cohort (positional_subsidy.py --mode extractor on the 549);
  results/baselines/reports/positional_subsidy_rerun_coma.json (382) + positional_subsidy_549new168_rerun_coma.json
                                    COMA re-run on physically permuted target columns (--mode rerun), the earlier copy's
                                    rows without the duplicate's Joinable copy plus the 168 new;
  results/baselines/reports/valentine549_run/protocol.json   the old80 (PRISMA benchmark) flags of the 549.
  SiMa has no matrices for the 168 new scenarios, so its audit stays on the 382 scenarios of the earlier copy
  (positional_subsidy.json, positional_alignment.json), as does Cupid's rerun (positional_subsidy_rerun_cupid.json).
  The numbers printed in the earlier manuscript are asserted where they did not change.

Result 2, independent sampling on TPC-H, AdventureWorks and Chinook (FD-free base matchers). Sources:
  results/baselines/reports/independent_sampling_20260929/  metrics.csv (F1_M as committed) and
      cases/<DB>_<seed>/<condition>/<pair>_<regime>.npz (gt, pmf, levenshtein, seed = LexPMF, char3);
  results/baselines/reports/independent_sampling_magneto/   metrics.csv and summary.json (Magneto zero-shot), with its matrices in
      results/baselines/generated/independent_sampling_magneto_stage/<DB>_<seed>/<condition>/<pair>_<regime>/magneto.npz.
  The ICDE paper's tie-free metrics are not reimplemented: score_r7.metrics (experiments/sf_study/layer3b), i.e.
  score_screen_metrics.all_metrics with J = (R_exp + F1_Hr) / 2, where R_exp is the expected Recall@GT (ties at the
  cut-off filled at random, exact) and F1_Hr the Hungarian F1 averaged over 50 fixed random row and column orders.
  The orders are seeded by '<DB>_<seed>/<pair>', so identical and independent samples, both regimes and all
  matchers share them (paired). Means: over the three schema pairs within a sampling seed, then over the three seeds,
  as in the earlier manuscript. Every J_tf is re-derived by an independent implementation below; F1_M and
  index-order F1_H recomputed from the matrices must equal the committed metrics.csv rows.

Fails loudly (AssertionError) on any mismatch. Run:

    python make_ea_numbers.py
"""

from __future__ import annotations

import csv
import json
import sys
from statistics import mean

import numpy as np
from scipy.optimize import linear_sum_assignment

from paths import BASELINES, STUDY, STUDY_CODE
from paths import OUT as OUT_DIR
REP = BASELINES / "reports"
SAMP = REP / "independent_sampling_20260929"
MAG = REP / "independent_sampling_magneto"
MAG_STAGE = BASELINES / "generated" / "independent_sampling_magneto_stage"
OUT = OUT_DIR / "ea_numbers.tex"
REPORT = OUT_DIR / "ea_numbers_report.md"

sys.dont_write_bytecode = True  # imports below live in other folders; leave no __pycache__ there
sys.path.insert(0, str(STUDY_CODE / "layer3b"))
from score_r7 import metrics as tf_metrics  # noqa: E402  (J = (R_exp + F1_Hr) / 2, the ICDE paper's J_tf)
from score_screen_metrics import K_PERM, permutations  # noqa: E402
from smx.core import DEFAULT_EPS  # noqa: E402

MACROS: list[tuple[str, str, str, str]] = []  # name, value, source, meaning
NOTES: list[str] = []


def add(name: str, value: str, source: str, meaning: str) -> None:
    assert name.startswith("ea") and name.isalpha(), name
    assert name not in {m[0] for m in MACROS}, name
    MACROS.append((name, value, source, meaning))


def s3(x: float) -> str:
    return f"{x:+.3f}"


def u3(x: float) -> str:
    return f"{x:.3f}"


def n(x: int) -> str:
    return f"{x:,}".replace(",", "{,}")


def same(a: float, b: float, what: str, tol: float = 1e-12) -> None:
    assert abs(a - b) <= tol, f"MISMATCH {what}: {a!r} vs {b!r}"


def printed(x: float, p: float, digits: int, what: str) -> None:
    """The earlier manuscript's printed number, with its printed rounding."""
    assert round(x, digits) == round(p, digits) or abs(x - p) <= 0.5 * 10 ** -digits + 1e-12, \
        f"MISMATCH with earlier manuscript, {what}: {x!r} vs printed {p}"


# ====================================================================================== result 1
BASE_N = 549   # the Valentine base
DUP = "valentine/TPC-DI/Joinable/prospect_horizontal_50_ec_av/prospect_horizontal_50_ec_av"   # dropped copy


def survey549() -> None:
    """The ground-truth diagonal survey of the 549 scenarios; its 381 old scenarios must equal the 382 survey."""
    sv = json.loads((STUDY / "results" / "valentine549" / "positional_survey.json").read_text(encoding="utf-8"))
    t, sc = sv["totals"], sv["scenarios"]
    pairs, off = sum(s["n_ground_truth"] for s in sc), sum(s["n_off_diagonal"] for s in sc)
    assert (pairs, off, len(sc)) == (t["pairs"], t["off_diagonal"], t["scenarios"]) and len(sc) == BASE_N, (pairs, off, t)
    # independent check against the earlier copy's survey (positional_alignment.py): same counts per shared scenario
    old = {}
    for s in json.loads((REP / "positional_alignment.json").read_text(encoding="utf-8"))["scenarios"]:
        if s["id"] != DUP:
            key = (f"{s['family']}-{s['scenario_type']}", s["id"].split("/")[-1])
            old[key] = (s["n_ground_truth"], s["n_off_diagonal"])
    shared = [(old[(s["dataset"], s["scenario"])], (s["n_ground_truth"], s["n_off_diagonal"])) for s in sc
              if (s["dataset"], s["scenario"]) in old]
    assert len(shared) == len(old) == BASE_N - 168 and all(a == b for a, b in shared), \
        "the 549 survey differs from the 382 survey on the scenarios they share"
    NOTES.append(f"Positional survey: {len(shared)} scenarios shared with the earlier copy's survey "
                 "(positional_alignment.json) have identical ground-truth and off-diagonal counts.")
    src = "valentine549/positional_survey.json"
    add("eaPosScenarios", str(t["scenarios"]), src, "Valentine scenarios in the positional survey (549)")
    add("eaPosPairs", n(pairs), src, "ground-truth pairs over the 549 scenarios")
    add("eaPosDiagPairs", n(pairs - off), src, "ground-truth pairs on the diagonal")
    add("eaPosOffDiagPairs", n(off), src, "ground-truth pairs off the diagonal")


def positional549() -> None:
    """The extractor-mode survey over the 549 scenarios (positional_subsidy.py on the 549, with the 168's COMA and Cupid
    matrices) and COMA's rerun check (the earlier copy's rerun without the duplicate's Joinable copy + the 168's rerun)."""
    sub = json.loads((REP / "positional_subsidy_549.json").read_text(encoding="utf-8"))
    rows = sub["rows"]
    src = "positional_subsidy_549.json"

    def row(arm: str, cohort: str = "all") -> dict:
        r = [x for x in rows if x["arm"] == arm and x["regime"] in ("plain", "--") and x["kind"] in ("ranked", "synthetic")
             and x["cohort"] == cohort]
        assert len(r) == 1, (arm, cohort, len(r))
        return r[0]
    proto = json.loads((REP / "valentine549_run" / "protocol.json").read_text(encoding="utf-8"))
    assert len(proto["cases"]) == BASE_N
    old = [c for c in proto["cases"] if c.get("old80")]
    assert proto["old80_expected_cases"] == 80 and proto["old80_cases"] == len(old), proto["old80_cases"]
    orow = row("Constant matrix", "old80")
    # constant matrix: Hungarian on an all-ties matrix assigns the identity, and PRISMA's scenarios have the ground
    # truth on the diagonal, so F1 = 1 on every resolvable one
    assert orow["n"] == len(old) and abs(orow["hung"] - 1) < 1e-12, (orow["n"], len(old), orow["hung"])
    add("eaPosOldTotal", str(proto["old80_expected_cases"]), "valentine549_run/protocol.json",
        "scenarios in PRISMA's 80-scenario benchmark")
    add("eaPosOldResolvable", str(len(old)), "valentine549_run/protocol.json",
        "of them present in the 549 corpus (the release adds 3 of the 4 missing; Wikidata musicians_unionable invalid)")
    add("eaPosOldConstHung", u3(orow["hung"]), src, "constant-matrix Hungarian F1 on the resolvable PRISMA scenarios")
    for name, arm in (("Coma", "COMA (schema)"), ("Cupid", "Cupid")):
        r = row(arm)
        assert r["n"] == BASE_N, (arm, r["n"])
        same(r["subsidy"], r["hung"] - r["hung_perm"], f"subsidy arithmetic, {arm} (549)")
        add(f"eaPos{name}Subsidy", s3(r["subsidy"]), src, f"positional subsidy of {arm} on the 549 scenarios")
        add(f"eaPos{name}Hung", u3(r["hung"]), src, f"{arm} Hungarian F1 (index order), readable, 549")
    native = [x for x in rows if x["kind"] == "native"]
    assert all(abs(x["subsidy"]) < 1e-12 for x in native) and all(abs(x.get("subsidy_mutual", 0)) < 1e-12 for x in rows)
    NOTES.append("Every native-output arm and every mutual-top-1 score has subsidy 0 (|.| < 1e-12), as equivariance "
                 "implies; checked, not emitted.")
    old_rr = [x for x in json.loads((REP / "positional_subsidy_rerun_coma.json").read_text(encoding="utf-8"))["rows"]
              if x["id"] != DUP]
    rr = old_rr + json.loads((REP / "positional_subsidy_549new168_rerun_coma.json").read_text(encoding="utf-8"))["rows"]
    assert len(rr) == BASE_N and len({x["id"] for x in rr}) == BASE_N
    # independent check of 'hung': the rerun scores COMA afresh, unpermuted, on the same 549 scenarios
    same(mean(x["f1_hung"] for x in rr), row("COMA (schema)")["hung"], "COMA: rerun F1_H vs. extractor-mode F1_H", 1e-12)
    ident = sum(1 for x in rr if x["max_abs_delta"] == 0.0 and x["native_identical"])
    assert ident == len(rr), f"COMA: {len(rr) - ident} scenarios not bit-identical under permutation"
    add("eaPosPermComaIdentical", str(ident), "positional_subsidy_rerun_coma.json + positional_subsidy_549new168_rerun_coma.json",
        "scenarios on which COMA (schema), re-run on column-permuted targets, returns exactly S[:, pi] (549)")
    add("eaPosPermRepeats", str(sub["repeats"]), src, "random target-column permutations averaged per scenario")


def sima382() -> None:
    """SiMa has no matrices for the 168 new scenarios (its held-out model was not rerun), so its positional audit stays
    on the 382 scenarios of the earlier copy (positional_subsidy.json); Cupid's rerun on permuted columns exists for the
    382 only (the 549 run reran COMA alone on the 168)."""
    rows = json.loads((REP / "positional_subsidy.json").read_text(encoding="utf-8"))["rows"]
    src = "positional_subsidy.json (382, SiMa only)"

    def row(arm: str, cohort: str = "all") -> dict:
        r = [x for x in rows if x["arm"] == arm and x["regime"] == "plain" and x["kind"] == "ranked" and x["cohort"] == cohort]
        assert len(r) == 1, (arm, cohort, len(r))
        same(r[0]["subsidy"], r[0]["hung"] - r[0]["hung_perm"], f"subsidy arithmetic, {arm} {cohort} (382)")
        return r[0]
    # SiMa: largest subsidy over the family cohorts (the earlier manuscript's 'one source table')
    sima = [x for x in rows if x["arm"].startswith("SiMa") and x["regime"] == "plain"]
    fam = [x for x in sima if x["cohort"].startswith("family:")]
    best = max(fam, key=lambda x: x["subsidy"])
    same(best["subsidy"], max(x["subsidy"] for x in sima), "SiMa max over families = max over all cohorts")
    best = row(best["arm"], best["cohort"])
    printed(best["subsidy"], 0.116, 3, "largest SiMa subsidy")
    add("eaPosSimaMaxSubsidy", s3(best["subsidy"]), src, "largest SiMa (same-family) subsidy over source-table families "
        "(382 scenarios of the earlier copy)")
    add("eaPosSimaMaxFamily", best["cohort"].split(":", 1)[1], src, "the family where it occurs (382)")
    add("eaPosSimaMaxFamilyN", str(best["n"]), src, "SiMa scenarios in that family cohort (382)")
    add("eaPosSimaSubsidy", s3(row(best["arm"])["subsidy"]), src,
        f"SiMa subsidy over all its {row(best['arm'])['n']} scenarios (382)")
    base = len(json.loads((REP / "positional_alignment.json").read_text(encoding="utf-8"))["scenarios"])
    assert base == 382 and row(best["arm"])["n"] <= base, (base, row(best["arm"])["n"])
    add("eaPosSimaBaseN", str(base), "positional_alignment.json", "Valentine scenarios of the earlier copy (the base "
        "of the SiMa audit)")
    rr = json.loads((REP / "positional_subsidy_rerun_cupid.json").read_text(encoding="utf-8"))["rows"]
    assert len(rr) == base
    ident = sum(1 for x in rr if x["max_abs_delta"] == 0.0 and x["native_identical"])
    assert ident == len(rr), f"Cupid: {len(rr) - ident} scenarios not bit-identical under permutation"
    add("eaPosPermCupidIdenticalPrior", str(ident), "positional_subsidy_rerun_cupid.json (382)",
        "scenarios of the earlier copy on which Cupid, re-run on column-permuted targets, returns exactly S[:, pi]")
    NOTES.append("SiMa (eaPosSima*): on the 382 scenarios of the earlier copy (positional_subsidy.json), by decision; "
                 "SiMa has no matrices for the 168.")


# ====================================================================================== result 2
DBS = (("TPCH", "Tpch", "TPC-H"), ("AdventureWorks", "Aw", "AdventureWorks"), ("Chinook", "Chin", "Chinook"))
SEEDS = ("20260929", "20260930", "20260931")
PAIRS = ("N_D1", "N_D2", "D1_D2")
MATCHERS = (("pmf", "Pmf", "PMF"), ("levenshtein", "Lev", "Levenshtein"), ("seed", "LexPmf", "LexPMF"),
            ("char3", "CharTri", "char3"), ("magneto", "Mag", "Magneto zero-shot"))


def own_tf(S: np.ndarray, gt: np.ndarray, perms, eps: float = DEFAULT_EPS) -> float:
    """Independent re-derivation of J_tf for verification (not used for the macros).

    R_exp by counting cells strictly above the |GT|-th largest score and sharing the remaining slots pro rata over the
    tied group; F1_Hr by assigning on the permuted matrix and mapping back, with F1 from counts.
    """
    k = int(gt.sum())
    v = np.sort(S.ravel())[::-1]
    c = v[k - 1]
    hi, tie = S > c + eps, np.abs(S - c) <= eps
    r = (int((hi & gt).sum()) + (k - int(hi.sum())) * int((tie & gt).sum()) / int(tie.sum())) / k
    fs = []
    for pr, pc in perms:
        a, b = linear_sum_assignment(-S[pr][:, pc])
        ri, ci = pr[a], pc[b]
        keep = S[ri, ci] > eps
        tp, pred = int(gt[ri[keep], ci[keep]].sum()), int(keep.sum())
        fs.append(0.0 if tp == 0 else 2 * tp / (pred + k))
    return (r + float(np.mean(fs))) / 2


def sampling() -> dict:
    committed = {}
    for r in csv.DictReader((SAMP / "metrics.csv").open(encoding="utf-8")):
        if r["arm"] == "base":
            committed[(r["database"], r["seed"], r["condition"], r["pair"], r["regime"], r["matcher"])] = r
    for r in csv.DictReader((MAG / "metrics.csv").open(encoding="utf-8")):
        if r["arm"] == "base":
            committed[(r["database"], r["seed"], r["condition"], r["pair"], r["regime"], r["matcher"])] = r
    scen = {k[:5] for k in committed}
    assert len(scen) == 216, len(scen)
    mag_scen = {k[:5] for k in committed if k[5] == "magneto"}
    assert len(mag_scen) == 216, len(mag_scen)

    val: dict[tuple, dict] = {}
    worst_own = worst_f1 = 0.0
    for db, _, _ in DBS:
        for sd in SEEDS:
            for cond in ("identical", "independent"):
                for pr in PAIRS:
                    for reg in ("plain", "opaque"):
                        z = np.load(SAMP / "cases" / f"{db}_{sd}" / cond / f"{pr}_{reg}.npz")
                        gt = z["gt"].astype(bool)
                        mz = MAG_STAGE / f"{db}_{sd}" / cond / f"{pr}_{reg}"
                        assert np.array_equal(np.load(mz / "inputs.npz")["gt"], z["gt"]), mz
                        perms = permutations(gt.shape, f"{db}_{sd}/{pr}")
                        assert len(perms) == K_PERM == 50
                        mats = {m: z[m].astype(float) for m, _, _ in MATCHERS if m != "magneto"}
                        mats["magneto"] = np.load(mz / "magneto.npz")["S"].astype(float)
                        for m, S in mats.items():
                            assert S.shape == gt.shape and np.isfinite(S).all(), (db, sd, cond, pr, reg, m)
                            x = tf_metrics(S, gt, perms)
                            c = committed[(db, sd, cond, pr, reg, m)]
                            for mine, theirs in (("F1_M", "f1_mutual"), ("F1_H", "f1_hungarian")):
                                d = abs(x[mine] - float(c[theirs]))
                                worst_f1 = max(worst_f1, d)
                                assert d < 1e-9, ("committed", db, sd, cond, pr, reg, m, mine, x[mine], c[theirs])
                            o = own_tf(S, gt, perms)
                            worst_own = max(worst_own, abs(o - x["J"]))
                            assert abs(o - x["J"]) < 1e-9, ("own J_tf", db, sd, cond, pr, reg, m, o, x["J"])
                            val[(db, sd, cond, pr, reg, m)] = x
    NOTES.append(f"Per-scenario checks over the 108 identical- and independent-sample scenarios x 5 matchers: F1_M and index-order F1_H recomputed from the "
                 f"matrices equal metrics.csv (max |diff| {worst_f1:.1e}); J_tf from score_r7.metrics equals the "
                 f"independent re-derivation (max |diff| {worst_own:.1e}).")
    return val


def agg(val: dict, db: str, cond: str, reg: str, m: str, key: str, seeds=SEEDS) -> float:
    return mean(mean(val[(db, sd, cond, pr, reg, m)][key] for pr in PAIRS) for sd in seeds)


def sampling_macros(val: dict) -> list[str]:
    src = "independent_sampling_20260929/cases/*.npz (+ magneto stage); score_r7.metrics"
    allscen = {tuple(r[k] for k in ("database", "seed", "condition", "pair", "regime"))
               for r in csv.DictReader((SAMP / "metrics.csv").open(encoding="utf-8"))}
    assert len(allscen) == 3 * 3 * 4 * 3 * 2 == 216, len(allscen)
    used = {k[:5] for k in val}
    assert used <= allscen and len(used) == 108, len(used)
    printed(len(allscen), 216, 0, "sampling scenarios")
    add("eaSmpScenarios", str(len(allscen)), "independent_sampling_20260929/metrics.csv",
        "independent-sampling scenarios: 3 DBs x 3 seeds x 4 conditions x 3 pairs x 2 regimes")
    add("eaSmpScenariosUsed", str(len(used)), "independent_sampling_20260929/cases",
        "of them in the identical and independent conditions, the two compared here (both regimes)")
    add("eaSmpPermOrders", str(K_PERM), "score_screen_metrics.K_PERM", "random row/column orders behind tie-free Hungarian F1")
    table = ["| DB | matcher | regime | J_tf ident. | J_tf indep. | change | change per seed | R_exp ident.->indep. | "
             "F1_Hr ident.->indep. | F1_M ident.->indep. |", "|---|---|---|---:|---:|---:|---|---|---|---|"]
    for reg in ("plain", "opaque"):
        for db, dk, dl in DBS:
            for m, mk, ml in MATCHERS:
                a = {c: agg(val, db, c, reg, m, "J") for c in ("identical", "independent")}
                ch = a["independent"] - a["identical"]
                per = [agg(val, db, "independent", reg, m, "J", (s,)) - agg(val, db, "identical", reg, m, "J", (s,))
                       for s in SEEDS]
                rx = [agg(val, db, c, reg, m, "R_exp") for c in ("identical", "independent")]
                fh = [agg(val, db, c, reg, m, "F1_Hr") for c in ("identical", "independent")]
                fm = [agg(val, db, c, reg, m, "F1_M") for c in ("identical", "independent")]
                same(a["identical"], (rx[0] + fh[0]) / 2, "J_tf = mean of components")
                table.append(f"| {dl} | {ml} | {reg} | {a['identical']:.3f} | {a['independent']:.3f} | {ch:+.3f} | "
                             f"{', '.join(f'{p:+.3f}' for p in per)} | {rx[0]:.3f} -> {rx[1]:.3f} | "
                             f"{fh[0]:.3f} -> {fh[1]:.3f} | {fm[0]:.3f} -> {fm[1]:.3f} |")
                if reg == "plain":
                    add(f"eaSmp{dk}{mk}Ident", u3(a["identical"]), src, f"{dl}, {ml}, readable: J_tf, identical samples")
                    add(f"eaSmp{dk}{mk}Indep", u3(a["independent"]), src, f"{dl}, {ml}, readable: J_tf, independent samples")
                    add(f"eaSmp{dk}{mk}Change", s3(ch), src, f"{dl}, {ml}, readable: J_tf change, independent minus identical")
                elif m != "pmf":
                    add(f"eaSmpOpq{dk}{mk}Change", s3(ch), src,
                        f"{dl}, {ml}, opaque: J_tf change, independent minus identical")
                if m == "pmf" and reg == "opaque":
                    pl = agg(val, db, "independent", "plain", m, "J") - agg(val, db, "identical", "plain", m, "J")
                    same(ch, pl, f"PMF identical in both regimes, {db}", 1e-12)
    # F1_M as the earlier manuscript printed it
    f1m = {"TPCH": (0.604, 0.384), "AdventureWorks": (0.485, 0.442), "Chinook": (0.541, 0.427)}
    magsum = json.loads((MAG / "summary.json").read_text(encoding="utf-8"))
    losses = []
    for db, dk, dl in DBS:
        for m, mk, ml in (("pmf", "Pmf", "PMF"), ("magneto", "Mag", "Magneto")):
            a = {c: agg(val, db, c, "plain", m, "F1_M") for c in ("identical", "independent")}
            if m == "pmf":
                printed(a["identical"], f1m[db][0], 3, f"{dl} PMF F1_M identical")
                printed(a["independent"], f1m[db][1], 3, f"{dl} PMF F1_M independent")
            else:
                same(a["identical"], magsum[f"plain/{db}"]["identical"], f"Magneto F1_M identical {db}", 1e-9)
                same(a["independent"], magsum[f"plain/{db}"]["independent"], f"Magneto F1_M independent {db}", 1e-9)
                losses.append(a["identical"] - a["independent"])
            src = ("independent_sampling_20260929/metrics.csv" if m == "pmf" else "independent_sampling_magneto/summary.json") \
                + " (recomputed from the matrices)"
            add(f"eaSmpFm{dk}{mk}Ident", u3(a["identical"]), src, f"{dl}, {ml}, readable: F1_M (metric of the earlier manuscript), identical")
            add(f"eaSmpFm{dk}{mk}Indep", u3(a["independent"]), src, f"{dl}, {ml}, readable: F1_M (metric of the earlier manuscript), independent")
            add(f"eaSmpFm{dk}{mk}Change", s3(a["independent"] - a["identical"]), src,
                f"{dl}, {ml}, readable: F1_M change, independent minus identical")
    printed(min(losses), 0.005, 3, "Magneto smallest F1_M loss")
    printed(max(losses), 0.039, 3, "Magneto largest F1_M loss")
    printed(agg(val, "TPCH", "identical", "plain", "pmf", "F1_M") - agg(val, "TPCH", "independent", "plain", "pmf", "F1_M"),
            0.22, 2, "TPC-H PMF loss")
    return table


def main() -> int:
    survey549()
    positional549()
    sima382()
    val = sampling()
    table = sampling_macros(val)
    lines = ["% Generated by make_ea_numbers.py from results/baselines/reports/positional_*.json and "
             "results/baselines/reports/independent_sampling_{20260929,magneto}/ (tie-free metrics: experiments/sf_study/layer3b/score_r7.py); "
             "EXPLORATORY; do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v, _, _ in MACROS]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rep = ["# ea_numbers.tex: report", "",
           "Generated by `make_ea_numbers.py`; EXPLORATORY; do not edit. Two results first reported in an earlier "
           "manuscript. Every macro is asserted against an independent recomputation and, where the "
           "earlier manuscript printed it, against that number; the script fails on any mismatch.", "",
           "J_tf = (expected Recall@GT + tie-free Hungarian F1) / 2, from experiments/sf_study/layer3b/score_r7.py "
           f"({K_PERM} random row/column orders seeded by '<DB>_<seed>/<pair>', shared by conditions, regimes and "
           "matchers). Means over the three schema pairs within a sampling seed, then over seeds 20260929-31. "
           "'LexPMF' is the harness matcher `seed` (scale-aligned fusion of Levenshtein and PMF, weight 0.35 on PMF).", "",
           "## Macros", "", "| macro | value | source | meaning |", "|---|---|---|---|"]
    rep += [f"| `\\{k}` | {v.replace('{,}', ',')} | {s} | {m} |" for k, v, s, m in MACROS]
    rep += ["", "## Independent sampling, all matchers and both regimes", "", *table, "",
            "## Checks and notes", "", *[f"- {x}" for x in NOTES],
            "- Not computable: nothing requested is missing. The earlier 'bit-identical' claim is machine-checked from the "
            "rerun files (readable regime only; no audited-MD5 rerun exists).",
            "- J_tf macros compare only the identical and independent conditions (half-shared and disjoint roots are "
            "in the table sources but not emitted). Opaque macros are emitted as changes only, and not for PMF, whose "
            "matrices and scores are identical in both regimes (asserted).",
            "- Expected Recall@GT ranks all cells of the matrix jointly, so it penalises matchers whose scores are not "
            "comparable across columns: readable TPC-H LexPMF has R_exp 0.118 against F1_Hr 0.702.",
            "- A change printed as -0.000 is negative but below 0.0005 in magnitude; +0.000 for Levenshtein is exactly "
            "zero (labels do not depend on the sample).", ""]
    REPORT.write_text("\n".join(rep), encoding="utf-8")
    print(f"wrote {OUT} ({len(MACROS)} macros) and {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
