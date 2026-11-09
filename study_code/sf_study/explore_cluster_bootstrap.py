"""EXPLORATORY (EXPLORATORY.md X18; external review ER4): how much of the Amendment 3 confidence
intervals rests on treating scenarios as independent.

The registered test (confirm_tiebreak.py, hash-pinned, not edited and only imported here) resamples
scenarios within each family. Scenarios are not independent:
  - each of the three large families comes from one source table (ChEMBL: assays, OpenData: miller2,
    TPC-DI: prospect); Magellan's 7 scenarios are 7 different table pairs. With Wikidata left out,
    the macro rests on 10 independent sources;
  - within a family, the scenarios of one column split (same table, split type and overlaps) differ
    only in the noise on column names (ac1..ac5, ec) and values (av, ev), and they share columns and
    ground truth.

Three views of the same statistic (per scenario: mean over matchers; then the mean within each family;
then the mean over families):
  1. registered: resample scenarios within family (reproduces confirm_tiebreak.py's numbers);
  2. split clusters: resample whole column splits within family (all their scenarios together, all
     matcher contrasts paired); Magellan's scenarios are their own clusters;
  3. between sources: the estimate per family and leaving one family out. With three single-source
     families there is nothing to resample between tables within them; this view shows whether the
     sign holds on every source family.

    ../harness/.venv-bdi/Scripts/python.exe explore_cluster_bootstrap.py --regime md5
    ../harness/.venv-bdi/Scripts/python.exe explore_cluster_bootstrap.py --regime plain
Output: results/<regime>/cluster_bootstrap.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict

import numpy as np

import confirm_tiebreak as ct   # constants and J_tf only; the registered analysis is not changed

HERE = ct.HERE
SEED, B = 20261003, 10_000


def cluster_of(family: str, scenario: str) -> str:
    """Column split of a scenario: its name without the name/value noise suffix."""
    m = re.match(r"(.*?)_(ac\d|ec)_(av|ev)$", scenario)
    return f"{family}/{m.group(1) if m else scenario}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("md5", "plain"), default="md5")
    args = ap.parse_args()
    cases = json.loads(ct.PROTOCOL.read_text(encoding="utf-8"))["cases"]
    family = {(c["dataset"], c["scenario"]): c["family"] for c in cases}
    rows = [r for r in csv.DictReader(open(HERE / "results" / args.regime / "posthoc_scores.csv", encoding="utf-8"))
            if r["seed"] not in ct.CONTROLS and not (args.regime == "plain" and r["seed"] == "magneto")]
    scen = sorted({(r["dataset"], r["scenario"]) for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    by = {(r["seed"], r["dataset"], r["scenario"]): r for r in rows}
    full = [s for s in seeds if all((s, d, c) in by for d, c in scen)]
    fams = defaultdict(list)
    for d, c in scen:
        fams[family[(d, c)]].append((d, c))
    fams = {f: v for f, v in fams.items() if len(v) >= ct.MIN_FAMILY}
    contrasts = {
        "H1 J_tf(tiebreak) - J_tf(initial)": lambda r: ct.jtf(r, "tiebreak") - ct.jtf(r, "initial"),
        "H2 J_tf(tiebreak) - J_tf(sf)": lambda r: ct.jtf(r, "tiebreak") - ct.jtf(r, "sf"),
        "H3 R_exp(tiebreak) - R_exp(initial)": lambda r: float(r["tiebreak_R_exp"]) - float(r["initial_R_exp"]),
        "F1_Hr(tiebreak) - F1_Hr(initial)": lambda r: float(r["tiebreak_F1_Hr"]) - float(r["initial_F1_Hr"]),
    }
    nclus = {f: len({cluster_of(f, c) for _, c in v}) for f, v in fams.items()}
    lines = [f"# Source-aware uncertainty for the Amendment 3 contrasts ({args.regime}; exploratory, X18)", "",
             f"{len(full)} initial matchers, {sum(len(v) for v in fams.values())} scenarios; families "
             + ", ".join(f"{f} ({len(v)} scenarios, {nclus[f]} column splits)" for f, v in sorted(fams.items()))
             + ". Independent sources: one table each for ChEMBL, OpenData and TPC-DI; 7 table pairs for "
               "Magellan.", "",
             "| contrast | estimate | CI, scenarios (registered) | CI, column splits | "
             + " | ".join(sorted(fams)) + " | min leave-one-family-out |",
             "|---|---|---|---|" + "---|" * len(fams) + "---|"]
    rng_reg = np.random.default_rng(ct.SEED)
    for name, fn in contrasts.items():
        per = {f: np.array([np.mean([fn(by[(s, d, c)]) for s in full]) for d, c in v]) for f, v in fams.items()}
        est = float(np.mean([x.mean() for x in per.values()]))
        # 1. registered: scenarios within family; one stream over the contrasts, as in confirm_tiebreak.py
        boots = np.mean([[rng_reg.choice(x, size=x.size, replace=True).mean() for _ in range(ct.B)]
                         for x in per.values()], axis=0)
        reg = np.percentile(boots, [2.5, 97.5])
        # 2. column splits within family: resample clusters, keep each cluster's scenarios together
        rng = np.random.default_rng(SEED)
        fam_boot = []
        for f, v in fams.items():
            groups = defaultdict(list)
            for i, (_, c) in enumerate(v):
                groups[cluster_of(f, c)].append(per[f][i])
            sums = np.array([sum(g) for g in groups.values()])
            ns = np.array([len(g) for g in groups.values()])
            idx = rng.integers(0, len(sums), size=(B, len(sums)))
            fam_boot.append(sums[idx].sum(axis=1) / ns[idx].sum(axis=1))
        clu = np.percentile(np.mean(fam_boot, axis=0), [2.5, 97.5])
        # 3. between sources: per family, and the macro without each family
        fm = {f: float(x.mean()) for f, x in per.items()}
        lofo = min(float(np.mean([fm[g] for g in fm if g != f])) for f in fm)
        lines.append(f"| {name} | {est:+.4f} | [{reg[0]:+.4f}, {reg[1]:+.4f}] | [{clu[0]:+.4f}, {clu[1]:+.4f}] | "
                     + " | ".join(f"{fm[f]:+.4f}" for f in sorted(fams)) + f" | {lofo:+.4f} |")
    # ER5: the same estimate and split-cluster CI without the (matcher, scenario) pairs where the
    # tie-breaker's assignment is not optimal for q (check_tiebreak_guarantee.py, violation = 1)
    viol = {(r["seed"], r["dataset"], r["scenario"]) for r in csv.DictReader(
        open(HERE / "results" / args.regime / "tiebreak_guarantee.csv", encoding="utf-8")) if r["violation"] == "1"}
    nv = sum((s, d, c) in viol for s in full for v in fams.values() for d, c in v)
    lines += ["", f"Without the {nv} (matcher, scenario) pairs where the tie-breaker's assignment is not optimal "
              "for the initial similarities (per scenario, the mean over the remaining matchers):", "",
              "| contrast | estimate | CI, column splits |", "|---|---|---|"]
    for name, fn in contrasts.items():
        per = {f: np.array([np.mean([fn(by[(s, d, c)]) for s in full if (s, d, c) not in viol]) for d, c in v])
               for f, v in fams.items()}
        est = float(np.mean([x.mean() for x in per.values()]))
        rng = np.random.default_rng(SEED)
        fam_boot = []
        for f, v in fams.items():
            groups = defaultdict(list)
            for i, (_, c) in enumerate(v):
                groups[cluster_of(f, c)].append(per[f][i])
            sums = np.array([sum(g) for g in groups.values()])
            ns = np.array([len(g) for g in groups.values()])
            idx = rng.integers(0, len(sums), size=(B, len(sums)))
            fam_boot.append(sums[idx].sum(axis=1) / ns[idx].sum(axis=1))
        clu = np.percentile(np.mean(fam_boot, axis=0), [2.5, 97.5])
        lines.append(f"| {name} | {est:+.4f} | [{clu[0]:+.4f}, {clu[1]:+.4f}] |")
    out = HERE / "results" / args.regime / "cluster_bootstrap.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
