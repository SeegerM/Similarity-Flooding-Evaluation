"""Score every seed and its SF output with the harness's evaluation (harness/smx/core.py).

For each seed run under results/<suffix>/<seed>/ (latest timestamp), every scenario contributes
two matrices: MatchingStepLine1 (the seed) and MatchingStepLine2 (SF flooded from that seed).
Both are scored against the scenario's ground truth with the same extractors, so the SF uplift
is a paired difference on identical scenarios:

    delta = metric(SF(seed)) - metric(seed)

Writes results/<suffix>/scores.csv (one row per seed x scenario x stage x extractor) and
results/<suffix>/summary.csv (scenario means per seed), and prints the summary.

Usage (needs numpy/scipy; the harness's .venv-bdi has them):
    ../harness/.venv-bdi/Scripts/python.exe score_seeds.py --suffix plain_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))

from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402

CORPUS = {
    "plain": HERE.parent / "harness" / "generated" / "benchmark-full",
    "md5": HERE.parent / "harness" / "generated" / "benchmark-full-md5",
}
PROTOCOL = HERE.parent / "harness" / "reports" / "audited_md5_fused_20260904" / "protocol.json"
CORPUS["indep"] = HERE / "data" / "indep_plain"            # data/convert_indep.py, multi-table
CORPUS["indepsmoke"] = HERE / "data" / "indep_test" / "corpus"
EXTERNAL = {"indep", "indepsmoke"}
# the 168 Valentine scenarios of the official release that the 382-case corpus lacked (2026-10-30; make_configs.py)
CORPUS["pnew"] = HERE.parent / "harness" / "generated" / "valentine-new168"
CORPUS["mnew"] = HERE.parent / "harness" / "generated" / "valentine-new168-md5"
VALENTINE_NEW = {"pnew", "mnew"}


def cases_for(regime: str) -> list[tuple[str, str, str]]:
    """(dataset, scenario, family) of a regime: the 382-case protocol for plain and md5, every
    scenario of the corpus for the external ones (family = the database), and for the new Valentine
    scenarios every scenario with the Valentine family from its manifest."""
    if regime in VALENTINE_NEW:
        root = CORPUS[regime]
        return [(d.name, s.name, json.loads((s / "manifest.json").read_text(encoding="utf-8"))["family"])
                for d in sorted(root.iterdir()) if d.is_dir() for s in sorted(d.iterdir()) if s.is_dir()]
    if regime in EXTERNAL:
        root = CORPUS[regime]
        return [(d.name, s.name, d.name.replace("Indep-", "")) for d in sorted(root.iterdir()) if d.is_dir()
                for s in sorted(d.iterdir()) if s.is_dir()]
    return [(c["dataset"], c["scenario"], c["family"])
            for c in json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]]


def ground_truth_file(corpus: Path, dataset: str, scenario: str) -> Path | None:
    """The scenario's full ground-truth matrix: the single file in ground_truth/ (Valentine), or
    ground_truth_full.csv for multi-table scenarios, whose ground_truth/ holds one file per table pair."""
    full = corpus / dataset / scenario / "ground_truth_full.csv"
    return full if full.is_file() else one(corpus / dataset / scenario / "ground_truth")
POLICIES = ("hungarian", "mutual_top1")
METRICS = ("f1", "recall_at_gt_pessimistic", "mrr_pessimistic")
CONTROLS = {"constant", "random", "embedding_random"}


def latest_run(seed_dir: Path) -> Path | None:
    runs = sorted(p for p in seed_dir.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def one(folder: Path) -> Path | None:
    files = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    return files[0] if len(files) == 1 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--all-scenarios", action="store_true",
                    help="also score scenarios outside the harness's 382-case audited protocol")
    args = ap.parse_args()

    regime = args.suffix.split("_")[0]
    corpus = CORPUS[regime]
    root = HERE / "results" / args.suffix
    allowed = {(d, s) for d, s, _ in cases_for(regime)}
    outside = set()
    rows = []
    problems = []
    for seed_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        run = latest_run(seed_dir)
        if run is None or not (seed_dir / "DONE").is_file():
            continue
        for scenario_dir in sorted(run.glob("*/*")):
            dataset, scenario = scenario_dir.parent.name, scenario_dir.name
            if (dataset, scenario) not in allowed and not args.all_scenarios:
                outside.add((dataset, scenario))
                continue
            gt_file = ground_truth_file(corpus, dataset, scenario)
            if gt_file is None:
                problems.append((seed_dir.name, scenario, "no ground truth"))
                continue
            gt = read_matrix(gt_file)[0] > 0.5
            marker = HERE / "configs" / args.suffix / seed_dir.name / "seed_file_dir.txt"
            file_seed = Path(marker.read_text(encoding="utf-8").strip()) if marker.is_file() else None
            for stage, step in (("seed", "MatchingStepLine1"), ("sf", "MatchingStepLine2")):
                if stage == "seed" and file_seed is not None:
                    # FILE seed: line 1 is a placeholder; the seed is the file SF read
                    f = file_seed / dataset / scenario / "seed.csv"
                    f = f if f.is_file() else None
                else:
                    f = one(scenario_dir / "_outputs" / step)
                if f is None:
                    problems.append((seed_dir.name, scenario, f"missing {step}"))
                    continue
                S = read_matrix(f)[0]
                if S.shape != gt.shape:
                    problems.append((seed_dir.name, scenario, f"{stage} shape {S.shape} != {gt.shape}"))
                    continue
                if not np.isfinite(S).all():
                    problems.append((seed_dir.name, scenario, f"{stage} has non-finite scores"))
                    continue
                if S.min() < -1e-6 or S.max() > 1 + 1e-6:
                    # scored anyway (extractors only need an order), but a seed outside [0, 1]
                    # changes SF's normalisation and is reported
                    problems.append((seed_dir.name, scenario,
                                     f"{stage} outside [0,1]: [{S.min():.3g}, {S.max():.3g}]"))
                meta = {"seed": seed_dir.name, "dataset": dataset, "scenario": scenario, "stage": stage}
                for r in evaluate(ScoreMatrix(S, gt, meta=meta), policies=POLICIES):
                    rows.append({k: r[k] for k in ("seed", "dataset", "scenario", "stage", "extractor",
                                                    *METRICS, "frac_zero_scores")})

    if not rows:
        raise SystemExit(f"nothing to score under {root}")
    with (root / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # scenario means, paired: only scenarios where both stages were scored
    by = defaultdict(dict)
    for r in rows:
        by[(r["seed"], r["extractor"], r["dataset"], r["scenario"])][r["stage"]] = r
    agg = defaultdict(lambda: defaultdict(list))
    for (seed, extractor, _, _), stages in by.items():
        if "seed" in stages and "sf" in stages:
            for m in METRICS:
                agg[(seed, extractor)][f"seed_{m}"].append(stages["seed"][m])
                agg[(seed, extractor)][f"sf_{m}"].append(stages["sf"][m])
            agg[(seed, extractor)]["zero"].append(stages["seed"]["frac_zero_scores"])

    summary = []
    for (seed, extractor), cols in sorted(agg.items()):
        row = {"seed": seed, "extractor": extractor, "n": len(cols["seed_f1"]),
               "seed_zero_frac": mean(cols["zero"])}
        for m in METRICS:
            a = [x for x in cols[f"seed_{m}"] if not math.isnan(x)]
            b = [x for x in cols[f"sf_{m}"] if not math.isnan(x)]
            row[f"seed_{m}"] = mean(a) if a else float("nan")
            row[f"sf_{m}"] = mean(b) if b else float("nan")
            row[f"delta_{m}"] = row[f"sf_{m}"] - row[f"seed_{m}"]
        summary.append(row)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    print(f"{'seed':28s} {'n':>3s} {'zero':>5s}  {'seed F1':>7s} {'SF F1':>7s} {'dF1':>7s}  "
          f"{'seed R@GT':>9s} {'SF R@GT':>8s}")
    for r in sorted((r for r in summary if r["extractor"] == "hungarian"),
                    key=lambda r: -r["seed_f1"]):
        tag = " (control)" if r["seed"] in CONTROLS or r["seed"].startswith("oracle") else ""
        print(f"{r['seed']:28s} {r['n']:3d} {r['seed_zero_frac']:5.2f}  {r['seed_f1']:7.3f} "
              f"{r['sf_f1']:7.3f} {r['delta_f1']:+7.3f}  {r['seed_recall_at_gt_pessimistic']:9.3f} "
              f"{r['sf_recall_at_gt_pessimistic']:8.3f}{tag}")
    if outside:
        print(f"\n{len(outside)} scenarios outside the audited protocol were not scored "
              f"(--all-scenarios to include them)")
    if problems:
        print(f"\n{len(problems)} problems:", *problems, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
