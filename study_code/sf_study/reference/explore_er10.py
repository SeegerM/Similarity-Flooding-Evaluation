"""EXPLORATORY (external review ER10): Amendment 5's additive baselines against SF given the same class
information, on the independent corpus. Written 2026-10-03 ~22:00, before any er10_indep output existed.

Per scenario and initial matcher, J_tf = (ΔR_exp + ΔF1_Hr)/2 against the initial matrix q, for
    SF          layer1_indep `melnik` (from additive_scores.csv, as in Amendment 5)
    SF+P, SF+P~ `melnik` with pattern-class nodes, and with the classes shuffled within each table (er10_indep)
    SF+S, SF+S~ `melnik` with semantic-type nodes, and shuffled
    A_p, A_s    Amendment 5's additive baselines (additive_scores.csv)
Macro over the three databases, mean over the matchers that have every method on all 81 scenarios.
Contrasts: A_p − (SF+P), A_s − (SF+S) (same information, additive vs propagated); (SF+P) − SF, (SF+S) − SF
(what the class nodes add); (SF+P) − (SF+P~), (SF+S) − (SF+S~) (real vs shuffled classes). 95% percentile bootstrap
over scenarios stratified by database, 10,000 draws, seed 20261004; conditional on this corpus.

    ../harness/.venv-bdi/Scripts/python.exe reference/explore_er10.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STUDY))
import score_layer1 as sl  # noqa: E402
from score_seeds import cases_for  # noqa: E402

RES = STUDY / "tuning" / "results"
ARMS = {"melnik_shape": "SF+P", "melnik_shape_shuffled": "SF+P~", "melnik_semtype": "SF+S",
        "melnik_semtype_shuffled": "SF+S~"}
CONTRASTS = [("A_p", "SF+P"), ("A_s", "SF+S"), ("SF+P", "SF"), ("SF+S", "SF"), ("SF+P", "SF+P~"), ("SF+S", "SF+S~"),
             ("A_p", "SF"), ("A_s", "SF")]
B, SEED = 10_000, 20261004


def jtf(r: dict, p: str) -> float:
    return ((float(r[f"{p}_R_exp"]) - float(r["q_R_exp"])) + (float(r[f"{p}_F1_Hr"]) - float(r["q_F1_Hr"]))) / 2


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cases = cases_for("indep")
    fam = {(d, s): f for d, s, f in cases}
    val = defaultdict(dict)   # (matcher, dataset, scenario) -> method -> J_tf
    for r in csv.DictReader((RES / "layer1_indep_scores" / "additive_scores.csv").open(encoding="utf-8")):
        k = (r["seed"], r["dataset"], r["scenario"])
        for p, name in (("sf", "SF"), ("A_p", "A_p"), ("A_s", "A_s")):
            val[k][name] = jtf(r, p)
    root = RES / "er10_indep"
    by_seed = defaultdict(list)
    for d in sorted(p for p in root.iterdir() if (p / "DONE").is_file() and p.name.split("__", 1)[1] in ARMS):
        by_seed[d.name.split("__", 1)[0]].append(str(d))
    with Pool(8) as pool:
        parts = pool.map(sl.score_seed, [("indep", s, dirs, cases) for s, dirs in sorted(by_seed.items())])
    for rows in parts:
        for r in rows:
            k = (r["seed"], r["dataset"], r["scenario"])
            val[k][ARMS[r["arm"]]] = ((r["sf_R_exp"] - r["seed_R_exp"]) + (r["sf_F1_Hr"] - r["seed_F1_Hr"])) / 2
    methods = ["SF", "SF+P", "SF+P~", "SF+S", "SF+S~", "A_p", "A_s"]
    scen = sorted({(d, s) for (_, d, s) in val})
    matchers = sorted({m for (m, _, _) in val
                       if all(all(x in val.get((m, d, s), {}) for x in methods) for d, s in scen)})
    fams = sorted({fam[x] for x in scen})
    idx = {f: np.array([i for i, x in enumerate(scen) if fam[x] == f]) for f in fams}
    E = {x: np.array([[val[(m, d, s)][x] for d, s in scen] for m in matchers]) for x in methods}

    def macro(M):  # (matchers, scenarios) -> per-matcher family macro
        return np.mean([M[:, ix].mean(axis=1) for ix in idx.values()], axis=0)

    rng = np.random.default_rng(SEED)
    draws = [{f: ix[rng.integers(0, len(ix), size=len(ix))] for f, ix in idx.items()} for _ in range(B)]
    lines = ["# ER10 (EXPLORATORY): additive baselines against SF with the same class information, independent corpus", "",
             f"{len(matchers)} initial matchers with every method on all {len(scen)} scenarios; J_tf against the initial "
             f"matrix, macro over {', '.join(fams)}. Bootstrap: {B} scenario draws stratified by database, seed {SEED}.", "",
             "| method | J_tf | " + " | ".join(fams) + " |", "|---|---|" + "---|" * len(fams)]
    for x in methods:
        per = [E[x][:, idx[f]].mean() for f in fams]
        lines.append(f"| {x} | {macro(E[x]).mean():+.4f} | " + " | ".join(f"{v:+.4f}" for v in per) + " |")
    lines += ["", "| contrast | mean | 95% CI | " + " | ".join(fams) + " | matchers > 0 |", "|---|---|---|" + "---|" * len(fams) + "---|"]
    for a, b in CONTRASTS:
        D = E[a] - E[b]
        per_m = macro(D)
        boot = np.array([np.mean([D[:, dr[f]].mean(axis=1) for f in fams], axis=0).mean() for dr in draws])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        lines.append(f"| {a} − {b} | {per_m.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     + " | ".join(f"{D[:, idx[f]].mean():+.4f}" for f in fams) + f" | {int((per_m > 0).sum())} of {len(matchers)} |")
    out = RES / "er10_indep_scores"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
