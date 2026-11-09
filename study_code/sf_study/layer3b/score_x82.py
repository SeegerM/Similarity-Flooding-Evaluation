"""X82 (EXPLORATORY, replications on the new corpora; design and predictions in EXPLORATORY.md X82, written
2026-10-06 before any X82 output existed): contrasts and statistic for P10 (parallel-edge rule), P11 (class nodes) and P12
(remaining E2x arms, R8 primary scope).

Input: one long CSV per family, tuning/results/x82_metrics/<family>.csv, with the columns
    corpus, dataset, scenario, matcher, arm, J, R_exp, F1_Hr
from the X81 metric step (J = J_tf = (R_exp + F1_Hr)/2, 50 seeded tie permutations; the same code as X81), one row per
(matcher, arm, scenario). This script reads nothing else.

Statistic, per corpus (madi is split into madi_indep and madi_products by the scenario prefix "products__"; products is
never pooled):
- matchers: non-control (score_r7.is_control, plus constant/random/datatype), complete in both arms of the contrast on
  every scenario of the corpus that has any row of the base arm;
- per matcher: the mean over scenarios, or for final_controlled the macro over databases (dataset);
- the contrast: the mean over those matchers of (arm − base);
- 95% percentile bootstrap over scenarios, stratified by dataset, 10,000 draws, seed 20261006; with fewer than 5
  scenarios (the study's rule; mimic has 1) no interval, and the verdict goes by the point estimate.
Verdict per corpus against the Valentine sign: "confirms" (interval excludes 0 on that side), "contradicts" (excludes 0
on the other side), else "agrees" / "opposite" by the point estimate. The summary counts the five independent corpora
(hdxsm, gdcsm, madi_indep, goby, final_indep); final_controlled, madi_products and mimic are reported beside them.

    python layer3b/score_x82.py --family parallel|classfull|e2x [--contrast arm:base:sign ...]
Writes tuning/results/x82_<family>_scores/{contrasts.csv, report.md}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
from score_r7 import is_control  # noqa: E402

RES = STUDY / "tuning" / "results"
B, SEED, MIN_CI = 10_000, 20261006, 5
CONTROL_EXTRA = {"constant", "random", "random_fresh", "datatype", "datatype_nanfix"}
INDEPENDENT = ["hdxsm", "gdcsm", "madi_indep", "goby", "final_indep"]
BESIDE = ["final_controlled", "madi_products", "mimic"]
# (arm, base, Valentine sign on plain, metric); the Valentine values are in EXPLORATORY X82.
CONTRASTS = {
    "parallel": [("fd_split", "fd", -1, "J"), ("fd_split", "fd", -1, "R_exp")],
    "classfull": [("melnik_shape", "melnik_shape_shuffled", +1, "J"), ("melnik_semtype", "melnik_semtype_shuffled", +1, "J"),
                  ("melnik_shape", "melnik", +1, "J"), ("melnik_semtype", "melnik", +1, "J")],
    "e2x": [],      # given with --contrast, listed in EXPLORATORY X82 before any output
}


def split_corpus(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df.corpus == "madi"
    df.loc[m, "corpus"] = np.where(df.loc[m, "scenario"].str.startswith("products__"), "madi_products", "madi_indep")
    return df


def controls(m: str) -> bool:
    return is_control(m) or m.split("__")[0] in CONTROL_EXTRA or m in CONTROL_EXTRA


def contrast(d: pd.DataFrame, arm: str, base: str, metric: str, rng) -> dict | None:
    piv = d[d.arm.isin([arm, base])].pivot_table(index=["dataset", "scenario"], columns=["arm", "matcher"],
                                                    values=metric, aggfunc="first")
    if arm not in piv.columns.get_level_values(0) or base not in piv.columns.get_level_values(0):
        return None
    scen = piv[base].dropna(how="all").index
    a, b = piv[arm].reindex(scen), piv[base].reindex(scen)
    ms = [m for m in b.columns if not controls(m) and m in a.columns and a[m].notna().all() and b[m].notna().all()]
    if not ms:
        return None
    diff = (a[ms] - b[ms])                              # scenarios x matchers
    ds = np.array(scen.get_level_values(0))
    by_db = d.corpus.iloc[0] == "final_controlled"

    def stat(rows):
        x = diff.iloc[rows]
        if by_db:
            x = x.groupby(ds[rows]).mean()
        return float(x.mean(axis=0).mean())
    obs = stat(np.arange(len(scen)))
    lo = hi = np.nan
    if len(scen) >= MIN_CI:
        strata = [np.where(ds == g)[0] for g in np.unique(ds)]
        bs = [stat(np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata])) for _ in range(B)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
    better = int((diff.mean(axis=0) > 0).sum())
    return {"estimate": obs, "lo": lo, "hi": hi, "scenarios": len(scen), "matchers": len(ms), "matchers_positive": better}


def verdict(r: dict, sign: int) -> str:
    if np.isfinite(r["lo"]) and (r["lo"] > 0 if sign > 0 else r["hi"] < 0):
        return "confirms"
    if np.isfinite(r["lo"]) and (r["hi"] < 0 if sign > 0 else r["lo"] > 0):
        return "contradicts"
    return "agrees" if np.sign(r["estimate"]) == sign else "opposite"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(CONTRASTS))
    ap.add_argument("--contrast", action="append", default=[], help="arm:base:sign (sign +1/-1), metric J")
    ap.add_argument("--metrics", default=None, help="override the input CSV (tests only)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cons = CONTRASTS[a.family] + [(c.split(":")[0], c.split(":")[1], int(c.split(":")[2]), "J") for c in a.contrast]
    if not cons:
        raise SystemExit("no contrasts given")
    df = split_corpus(pd.read_csv(a.metrics or RES / "x82_metrics" / f"{a.family}.csv"))
    need = {"corpus", "dataset", "scenario", "matcher", "arm", "J", "R_exp", "F1_Hr"}
    if need - set(df.columns):
        raise SystemExit(f"missing columns {need - set(df.columns)}")
    bad = (df.J - (df.R_exp + df.F1_Hr) / 2).abs().max()
    if bad > 1e-9:
        raise SystemExit(f"J is not (R_exp + F1_Hr)/2 (max |diff| {bad})")
    rng = np.random.default_rng(SEED)
    rows = []
    for arm, base, sign, metric in cons:
        for corpus in INDEPENDENT + BESIDE + sorted(set(df.corpus) - set(INDEPENDENT) - set(BESIDE)):
            d = df[df.corpus == corpus]
            if d.empty:
                continue
            r = contrast(d, arm, base, metric, rng)
            if r is None:
                continue
            rows.append({"contrast": f"{arm} − {base}", "metric": metric, "valentine_sign": sign, "corpus": corpus, **r,
                         "verdict": verdict(r, sign), "independent": corpus in INDEPENDENT})
    out = Path(a.out) if a.out else RES / f"x82_{a.family}_scores"
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "contrasts.csv", index=False)
    lines = [f"# X82 {a.family} (EXPLORATORY replication on the new corpora)", "",
             "| contrast | metric | corpus | estimate [95% CI] | scenarios | matchers (> 0) | verdict |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = f"[{r['lo']:+.4f}, {r['hi']:+.4f}]" if np.isfinite(r["lo"]) else f"(n = {r['scenarios']}, no CI)"
        lines.append(f"| {r['contrast']} | {r['metric']} | {r['corpus']} | {r['estimate']:+.4f} {ci} | {r['scenarios']} | "
                     f"{r['matchers']} ({r['matchers_positive']}) | {r['verdict']} |")
    lines += ["", "Summary over the five independent corpora (confirms / agrees / opposite / contradicts):", ""]
    for (c, m), g in res[res.independent].groupby(["contrast", "metric"], sort=False):
        v = g.verdict.value_counts()
        lines.append(f"- {c} ({m}): {v.get('confirms', 0)} / {v.get('agrees', 0)} / {v.get('opposite', 0)} / "
                     f"{v.get('contradicts', 0)} of {len(g)}")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
