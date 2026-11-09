"""Pre-specified confirmation of "SF as tie-breaker" (tuning/PREREGISTRATION.md, Amendment 3).

Written on 2026-10-02, before any MD5 or lockbox data was scored for this question; its SHA-256
is recorded in Amendment 3. Do not change the analysis below; a change is a new amendment.

Input: a per-scenario score file with, for every (initial matcher, scenario), the tie-free metrics
of the initial similarities ("initial"), of SF's output ("sf") and of the tie-breaker
("tiebreak"), as written by explore_posthoc.py (columns <variant>_<metric>), or, for the lockbox,
by score_tuning.py's tie-breaker report (same column names).

Hypotheses, each on J_tf = (R_exp + F1_Hr) / 2 and its parts:
    H1  tiebreak - initial > 0       (the tie-breaker improves on the initial matcher)
    H2  tiebreak - sf      > 0       (the tie-breaker improves on SF as run)
    H3  R_exp(tiebreak) - R_exp(initial) > -0.01   (non-inferiority: no damage to the ranking)

Statistic: per scenario, the mean difference over the included initial matchers; then the mean
over scenarios within each data family; then the mean over families (macro). Confidence
interval: 95% percentile bootstrap, resampling scenarios within each family (stratified),
10,000 resamples, seed 20261002. H1 and H2 are confirmed if the lower bound is above 0; H3 if the
lower bound is above -0.01.

Included initial matchers: every initial matcher with a full score set on the scenarios analysed,
except the controls (constant, random, the oracle variants, embedding_random, rema_shuffled,
semstruct_lm_only), and except Magneto in the plain regime of the exploratory analysis (absent
there). Families with fewer than 5 scenarios in the analysed set (Wikidata) are left out of the
macro mean.

    ../harness/.venv-bdi/Scripts/python.exe confirm_tiebreak.py --scores results/md5/posthoc_scores.csv
    ../harness/.venv-bdi/Scripts/python.exe confirm_tiebreak.py --scores tuning/results/lockbox_plain/tiebreak_scores.csv         --scenarios lockbox --chosen tuning/chosen.json --role own        # primary lockbox analysis
    (and --role universal for the configuration chosen for all initial matchers, secondary)

On the lockbox, the score file holds one row per (initial matcher, configuration, scenario); --chosen
keeps, per initial matcher, its own chosen configuration (role "own": the entry of chosen.json that
is not the universal one, or the universal one if they coincide) or the universal configuration
(role "universal").
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / "harness" / "reports" / "audited_md5_fused_20260904" / "protocol.json"
SPLITS = HERE / "tuning" / "splits.json"
CONTROLS = {"constant", "random", "oracle", "oracle_noise025", "oracle_noise050", "oracle_erase050",
            "oracle_decoy100", "embedding_random", "rema_shuffled", "semstruct_lm_only"}
MIN_FAMILY = 5
B, SEED = 10_000, 20261002


def jtf(r: dict, v: str) -> float:
    return (float(r[f"{v}_R_exp"]) + float(r[f"{v}_F1_Hr"])) / 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--scenarios", choices=("all", "lockbox", "dev"), default="all")
    ap.add_argument("--chosen", help="chosen.json; keeps one configuration per initial matcher")
    ap.add_argument("--role", choices=("own", "universal"), default="own")
    args = ap.parse_args()

    cases = json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]
    family = {(c["dataset"], c["scenario"]): c["family"] for c in cases}
    case_key = {(c["dataset"], c["scenario"]): c["key"] for c in cases}   # splits.json uses these keys
    keep = None
    if args.scenarios != "all":
        splits = json.loads(SPLITS.read_text(encoding="utf-8"))
        keep = set(splits[args.scenarios])
    rows = [r for r in csv.DictReader(open(args.scores, encoding="utf-8")) if r["seed"] not in CONTROLS]
    if args.chosen:
        ch = json.loads(Path(args.chosen).read_text(encoding="utf-8"))
        uni = ch["universal"]
        pick = {s: (uni if args.role == "universal" else next((c for c in cs if c != uni), uni))
                for s, cs in ch["configurations"].items()}
        rows = [r for r in rows if pick.get(r["seed"]) == r["config"]]
    if keep is not None:
        rows = [r for r in rows if case_key.get((r["dataset"], r["scenario"])) in keep]
    scen = sorted({(r["dataset"], r["scenario"]) for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    by = {(r["seed"], r["dataset"], r["scenario"]): r for r in rows}
    full = [s for s in seeds if all((s, d, c) in by for d, c in scen)]

    contrasts = {
        "H1 J_tf(tiebreak) - J_tf(initial)": lambda r: jtf(r, "tiebreak") - jtf(r, "initial"),
        "H2 J_tf(tiebreak) - J_tf(sf)": lambda r: jtf(r, "tiebreak") - jtf(r, "sf"),
        "H3 R_exp(tiebreak) - R_exp(initial)": lambda r: float(r["tiebreak_R_exp"]) - float(r["initial_R_exp"]),
        "   F1_Hr(tiebreak) - F1_Hr(initial)": lambda r: float(r["tiebreak_F1_Hr"]) - float(r["initial_F1_Hr"]),
    }
    fams = defaultdict(list)
    for d, c in scen:
        fams[family[(d, c)]].append((d, c))
    fams = {f: v for f, v in fams.items() if len(v) >= MIN_FAMILY}
    rng = np.random.default_rng(SEED)
    print(f"{len(full)} initial matchers, {sum(len(v) for v in fams.values())} scenarios in "
          f"{sorted(fams)}; scenarios: {args.scenarios}")
    for name, fn in contrasts.items():
        per = {f: np.array([np.mean([fn(by[(s, d, c)]) for s in full]) for d, c in v]) for f, v in fams.items()}
        est = float(np.mean([x.mean() for x in per.values()]))
        boots = np.mean([[rng.choice(x, size=x.size, replace=True).mean() for _ in range(B)]
                         for x in per.values()], axis=0)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        bound = -0.01 if name.startswith("H3") else 0.0
        verdict = "" if name.startswith("   ") else ("CONFIRMED" if lo > bound else "not confirmed")
        print(f"{name}: {est:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
