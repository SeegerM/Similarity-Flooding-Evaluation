"""Registered lockbox test (tuning PREREGISTRATION.md Amendment 4, draft E1): HL1–HL4. Written 2026-10-04 ~00:30,
before any lockbox configuration was run.

Input: tuning/results/lockbox_plain/scores.csv (score_tuning.py --stage lockbox; one row per initial matcher x
configuration x scenario, column J_tf as Amendment 2 defines it). Arms per initial matcher:
    own         its chosen.json entry that is not chosen.json's universal one, or that one if they coincide
                (the rule of confirm_tiebreak.py --role own)
    universal   results/supp_selection_plain/universal_choice.json "universal" (ER1; fpC__INV_PROD__ID__fdoff)
    production  space.PRODUCTION
    alpha0      space.NO_PROPAGATION (restart alpha = 0, ID attachment)
    melnik      report only: layer 1's `melnik` on the same scenarios (layer1_plain_scores/scores.csv)
Contrasts (J_tf): HL1 own − production, HL2 own − alpha0, HL3 own − universal, HL4 universal − production.
Statistic as registered: per initial matcher the mean per family, then the mean over families with >= 5 lockbox
scenarios, then the mean over the non-control matchers (constant and oracle_noise050 reported separately).
95% percentile bootstrap over scenarios stratified by family, 10,000 resamples, seed 20261002; confirmed if the
lower bound is > 0. Sensitivity: column-split cluster bootstrap (explore_cluster_bootstrap.cluster_of), same seed.

    ../../harness/.venv-bdi/Scripts/python.exe score_lockbox_hl.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STUDY))
import space  # noqa: E402
from explore_cluster_bootstrap import cluster_of  # noqa: E402

CONTROLS = ("constant", "oracle_noise050")
MIN_FAMILY, B, SEED = 5, 10_000, 20261002
HYP = [("HL1", "own", "production"), ("HL2", "own", "alpha0"), ("HL3", "own", "universal"),
       ("HL4", "universal", "production")]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(HERE / "results" / "lockbox_plain" / "scores.csv"),
                    help="testing only; the registered input is the default")
    ap.add_argument("--out", default=str(HERE / "results" / "lockbox_plain" / "hl_report.md"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    chosen = json.loads((HERE / "chosen.json").read_text(encoding="utf-8"))
    uni_old = chosen["universal"]
    universal = json.loads((HERE / "results" / "supp_selection_plain" / "universal_choice.json")
                           .read_text(encoding="utf-8"))["universal"]
    arms = {}
    for seed, cs in chosen["configurations"].items():
        own = next((c for c in cs if c != uni_old), uni_old)
        arms[seed] = {"own": own, "universal": universal, "production": space.name(space.PRODUCTION),
                      "alpha0": space.name(space.NO_PROPAGATION)}
    val = defaultdict(dict)   # (seed, dataset, scenario) -> arm -> J_tf
    fam = {}
    for r in csv.DictReader(Path(args.scores).open(encoding="utf-8")):
        k = (r["seed"], r["dataset"], r["scenario"])
        fam[(r["dataset"], r["scenario"])] = r["family"]
        for arm, cfg in arms.get(r["seed"], {}).items():
            if r["config"] == cfg:
                val[k][arm] = float(r["J_tf"])
    lock = set(fam)
    for r in csv.DictReader((STUDY / "tuning" / "results" / "layer1_plain_scores" / "scores.csv").open(encoding="utf-8")):
        if r["arm"] == "melnik" and (r["dataset"], r["scenario"]) in lock and r["seed"] in arms:
            val[(r["seed"], r["dataset"], r["scenario"])]["melnik"] = (
                (float(r["sf_R_exp"]) - float(r["seed_R_exp"])) + (float(r["sf_F1_Hr"]) - float(r["seed_F1_Hr"]))) / 2
    n_f = defaultdict(int)
    for x in lock:
        n_f[fam[x]] += 1
    scen = sorted(x for x in lock if n_f[fam[x]] >= MIN_FAMILY)
    fams = sorted({fam[x] for x in scen})
    idx = {f: np.array([i for i, x in enumerate(scen) if fam[x] == f]) for f in fams}
    clus = {f: [np.array(v) for v in _groups(scen, idx[f], f)] for f in fams}
    names = ["own", "universal", "production", "alpha0", "melnik"]
    complete = {s for s in arms if all(all(a in val.get((s, d, c), {}) for a in names[:4]) for d, c in scen)}
    panel = sorted(s for s in complete if s not in CONTROLS)
    rng = np.random.default_rng(SEED)
    strat = [{f: ix[rng.integers(0, len(ix), size=len(ix))] for f, ix in idx.items()} for _ in range(B)]
    rng = np.random.default_rng(SEED)
    clus_draws = []
    for _ in range(B):
        d = {}
        for f, groups in clus.items():
            pick = rng.integers(0, len(groups), size=len(groups))
            d[f] = np.concatenate([groups[p] for p in pick])
        clus_draws.append(d)

    def M(arm, seeds):
        return np.array([[val[(s, d, c)].get(arm, np.nan) for d, c in scen] for s in seeds])

    def macro(E, draw=None):
        return np.mean([np.nanmean(E[:, (draw[f] if draw else ix)], axis=1) for f, ix in idx.items()], axis=0)

    lines = ["# Lockbox, registered test HL1–HL4 (tuning Amendment 4)", "",
             f"{len(scen)} lockbox scenarios in families with >= {MIN_FAMILY} ({', '.join(f'{f} {len(idx[f])}' for f in fams)}); "
             f"{len(panel)} non-control initial matchers with every arm on all of them. J_tf (Amendment 2). Universal: "
             f"`{universal}`. Stratified bootstrap {B}, seed {SEED}; cluster bootstrap over column splits as sensitivity.", "",
             "| | own | universal | production | α = 0 | melnik (report only) |", "|---|---|---|---|---|---|"]
    lines.append("| mean J_tf | " + " | ".join(f"{np.nanmean(macro(M(a, panel))):+.4f}" for a in names) + " |")
    lines += ["", "| hypothesis | contrast | mean | 95% CI (stratified) | 95% CI (clusters) | matchers > 0 | verdict |",
              "|---|---|---|---|---|---|---|"]
    for h, a, b in HYP:
        D = M(a, panel) - M(b, panel)
        per = macro(D)
        bs = np.array([macro(D, d).mean() for d in strat])
        bc = np.array([macro(D, d).mean() for d in clus_draws])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        clo, chi = np.percentile(bc, [2.5, 97.5])
        lines.append(f"| {h} | {a} − {b} | {per.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | [{clo:+.4f}, {chi:+.4f}] | "
                     f"{int((per > 0).sum())} of {len(panel)} | **{'confirmed' if lo > 0 else 'not confirmed'}** |")
    lines += ["", "## Per initial matcher (mean J_tf)", "", "| matcher | own config | " + " | ".join(names) + " |",
              "|---|---|" + "---|" * len(names)]
    for s in sorted(complete):
        lines.append(f"| {s}{' (control)' if s in CONTROLS else ''} | {arms[s]['own']} | "
                     + " | ".join(f"{macro(M(a, [s]))[0]:+.4f}" for a in names) + " |")
    missing = sorted(set(arms) - complete)
    if missing:
        lines.append(f"\nIncomplete (not in the panel): {missing}")
    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def _groups(scen, ix, f):
    g = defaultdict(list)
    for i in ix:
        g[cluster_of(f, scen[i][1])].append(i)
    return list(g.values())


if __name__ == "__main__":
    raise SystemExit(main())
