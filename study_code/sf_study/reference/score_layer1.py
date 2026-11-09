"""Score layer 1 (reference/make_layer1_configs.py) on the tie-free metrics, with SF statistics.

For every collected configuration tuning/results/layer1_<regime>/<seed>__<arm>/ (DONE, merged
dataset shards) and every protocol scenario, scores
  - the seed (reference/seeds/<regime>/<seed>/<dataset>/<scenario>/seed.csv), once per seed, and
  - SF's output under the arm (MatchingStepLine2)
with tuning/score_screen_metrics.py's all_metrics (F1_H, F1_Hr, F1_M, R_pess, R_exp, MRR_pess,
MRR_exp; the same per-scenario random orders as score_seeds_tiefree.py, so the numbers are
comparable). F1_M is mutual top-1, which equals Melnik et al.'s SelectThreshold at t_rel = 1.0.
The SFSTATS line of each scenario (iterations, converged, graph sizes, build and iteration ms) is
attached from the configuration's sf_stats.tsv.

Writes tuning/results/layer1_<regime>_scores/
  scores.csv    one row per seed x arm x scenario: seed_<m>, sf_<m>, and the SF statistics
  summary.csv   per seed x arm: macro mean over families (>= 5 scenarios) of seed_<m>, sf_<m>,
                delta_<m> = sf - seed, and vs_melnik_<m> = sf(arm) - sf(melnik), paired
  report.md     the arm means over the non-control seeds with full coverage

    ../harness/.venv-bdi/Scripts/python.exe reference/score_layer1.py --regime plain
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "harness"))
sys.path.insert(0, str(STUDY / "tuning"))
sys.path.insert(0, str(STUDY))

from smx.adapters import read_matrix  # noqa: E402
from score_screen_metrics import METRICS, all_metrics, permutations  # noqa: E402
from score_seeds import CONTROLS, CORPUS, cases_for, ground_truth_file, one  # noqa: E402

STATS = ["iterations", "converged", "schema_v", "schema_e", "pcg_v", "pcg_e", "build_ms", "iterate_ms"]
ORACLES = ("oracle",)
MIN_FAMILY = 5


def run_dir(label_dir: Path) -> Path | None:
    if (label_dir / "merged").is_dir():
        return label_dir / "merged"
    runs = sorted(p for p in label_dir.iterdir() if p.is_dir() and p.name != "_shards")
    return runs[-1] if runs else None


def score_seed(job: tuple) -> list[dict]:
    regime, seed, label_dirs, cases = job
    corpus = CORPUS[regime]
    seeds = HERE / "seeds" / regime / seed
    arms = {}
    for d in label_dirs:
        d = Path(d)
        r = run_dir(d)
        if r is None:
            continue
        stats = {}
        if (d / "sf_stats.tsv").is_file():
            for line in (d / "sf_stats.tsv").read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) == 2 + len(STATS):
                    stats[(parts[0], parts[1])] = dict(zip(STATS, parts[2:]))
        arms[d.name.split("__", 1)[1]] = (r, stats)
    out = []
    for ds, sc, fam in cases:
        f_q = seeds / ds / sc / "seed.csv"
        g = ground_truth_file(corpus, ds, sc)
        if not f_q.is_file() or g is None:
            continue
        gt = read_matrix(g)[0] > 0.5
        q = read_matrix(f_q)[0]
        if q.shape != gt.shape or not np.isfinite(q).all():
            continue
        perms = permutations(gt.shape, f"{ds}/{sc}")
        mq = all_metrics(q, gt, perms)
        for arm, (r, stats) in sorted(arms.items()):
            f_s = one(r / ds / sc / "_outputs" / "MatchingStepLine2")
            if f_s is None:
                continue
            S = read_matrix(f_s)[0]
            if S.shape != gt.shape or not np.isfinite(S).all():
                continue
            ms = all_metrics(S, gt, perms)
            out.append({"seed": seed, "arm": arm, "dataset": ds, "scenario": sc, "family": fam,
                        **{f"seed_{m}": mq[m] for m in METRICS}, **{f"sf_{m}": ms[m] for m in METRICS},
                        **{k: stats.get((ds, sc), {}).get(k, "") for k in STATS}})
    return out


def macro(rows: list[dict], key: str) -> float:
    by = defaultdict(list)
    for r in rows:
        by[r["family"]].append(r[key])
    fams = [f for f, v in by.items() if len(v) >= MIN_FAMILY]
    return mean(mean(by[f]) for f in fams) if fams else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=sorted(CORPUS), required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--layer", type=int, choices=(1, 3), default=1,
                    help="3: layer-3 arms, compared with layer 1's melnik arm of the same seed")
    ap.add_argument("--variant", default="",
                    help="e.g. 'nan': score layer<L>nan_<regime> (X30 reruns) into its own _scores folder; the "
                         "layer-3 reference is then layer1nan's melnik. Default: the registered stages.")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    root = STUDY / "tuning" / "results" / f"layer{a.layer}{a.variant}_{a.regime}"
    out_dir = STUDY / "tuning" / "results" / f"layer{a.layer}{a.variant}_{a.regime}_scores"
    ref_root = STUDY / "tuning" / "results" / f"layer1{a.variant}_{a.regime}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = cases_for(a.regime)
    by_seed = defaultdict(list)
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "DONE").is_file() and "__" in p.name):
        by_seed[d.name.split("__", 1)[0]].append(str(d))
    if a.layer == 3:  # the reference arm lives in layer 1
        for s, dirs in by_seed.items():
            ref = ref_root / f"{s}__melnik"
            if (ref / "DONE").is_file():
                dirs.append(str(ref))
    jobs = [(a.regime, s, dirs, cases) for s, dirs in sorted(by_seed.items())]
    rows = []
    with Pool(a.workers) as pool:
        for part in pool.imap_unordered(score_seed, jobs):
            rows.extend(part)
    if not rows:
        raise SystemExit(f"nothing collected under {root}")
    rows.sort(key=lambda r: (r["seed"], r["arm"], r["dataset"], r["scenario"]))
    with (out_dir / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # paired summaries
    cell = defaultdict(dict)  # (seed, arm) -> {(ds, sc): row}
    for r in rows:
        cell[(r["seed"], r["arm"])][(r["dataset"], r["scenario"])] = r
    summary = []
    for (seed, arm), scen in sorted(cell.items()):
        ref = cell.get((seed, "melnik"), {})
        rs = list(scen.values())
        s = {"seed": seed, "arm": arm, "scenarios": len(rs)}
        for m in METRICS:
            for r in rs:
                r[f"delta_{m}"] = r[f"sf_{m}"] - r[f"seed_{m}"]
            s[f"seed_{m}"] = macro(rs, f"seed_{m}")
            s[f"sf_{m}"] = macro(rs, f"sf_{m}")
            s[f"delta_{m}"] = macro(rs, f"delta_{m}")
            paired = [dict(r, vs=r[f"sf_{m}"] - ref[k][f"sf_{m}"]) for k, r in scen.items() if k in ref]
            s[f"vs_melnik_{m}"] = macro(paired, "vs") if paired and arm != "melnik" else float("nan")
        its = [int(r["iterations"]) for r in rs if str(r["iterations"]).lstrip("-").isdigit()]
        s["mean_iterations"] = mean(its) if its else float("nan")
        s["converged_share"] = mean(r["converged"] == "1" for r in rs if r["converged"] != "") if any(r["converged"] != "" for r in rs) else float("nan")
        ms = [int(r["build_ms"]) + int(r["iterate_ms"]) for r in rs if str(r["build_ms"]).isdigit()]
        s["mean_sf_ms"] = mean(ms) if ms else float("nan")
        summary.append(s)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    # report: arm means over non-control, non-oracle seeds that have every arm and full coverage
    n_full = max(s["scenarios"] for s in summary)
    arms = sorted({s["arm"] for s in summary}, key=lambda x: (x != "melnik", x))
    seeds = sorted({s["seed"] for s in summary if s["seed"] not in CONTROLS and not s["seed"].startswith(ORACLES)})
    full = [x for x in seeds if all(any(s["seed"] == x and s["arm"] == arm and s["scenarios"] == n_full for s in summary)
                                    for arm in arms)]
    lines = [f"# Layer {a.layer} ({a.regime}): {len(rows)} scored (seed, arm, scenario) triples", "",
             f"Means over {len(full)} non-control initial matchers with every arm on all {n_full} scenarios; "
             f"each value is the macro mean over families with >= {MIN_FAMILY} scenarios.", "",
             "| arm | " + " | ".join(f"Δ{m} (SF−seed)" for m in ("F1_M", "F1_Hr", "R_exp", "MRR_exp")) +
             " | " + " | ".join(f"vs melnik {m}" for m in ("F1_M", "F1_Hr", "R_exp")) + " | iterations | converged | SF ms |",
             "|---|" + "---|" * 10]
    for arm in arms:
        ss = [s for s in summary if s["arm"] == arm and s["seed"] in full]
        if not ss:
            continue
        f = lambda k: mean(s[k] for s in ss)
        lines.append(f"| {arm} | " + " | ".join(f"{f('delta_' + m):+.3f}" for m in ("F1_M", "F1_Hr", "R_exp", "MRR_exp")) +
                     " | " + " | ".join("—" if arm == "melnik" else f"{f('vs_melnik_' + m):+.3f}" for m in ("F1_M", "F1_Hr", "R_exp")) +
                     f" | {f('mean_iterations'):.1f} | {f('converged_share'):.0%} | {f('mean_sf_ms'):.0f} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
