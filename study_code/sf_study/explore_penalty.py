"""EXPLORATORY R5 (ideas from the related-work review): Lily's post-propagation penalties.

Wang & Xu (ASWC 2009, Sec. 4, "Penalty for Propagation") multiply each propagated score by two
factors before extraction:

    pa(i, j) = s(i, j) / max(rowmax_i, colmax_j)                       relative similarity (Eq. 5)
    pb(i, j) = 1 / (1 + exp(-alpha * t)),
               t = ((N + 1) / (n_ij + 1)) / log(N + 1),  alpha = 3        density penalty (Eq. 6)
    S'(i, j) = S(i, j) * pa * pb                                         (Eq. 7)

with N = rows + columns and n_ij the number of cells with a value > 0 in row i and column j (the
cell itself counted once). pa is Melnik's relative similarity (SelectThreshold) used as a score;
pb lowers cells whose row and column hold many positive candidates.

For every initial matcher and scenario of the plain seed study, this script scores, on the tie-free
metrics, the initial matrix q and SF's output S, each as is, x pa, x pb and x pa*pb. The penalties
on q are the control: a gain on S that q shows too is not about propagation. No SF runs. Plain
only (the MD5 tie-breaker stays blind until Amendment 3's primary test).

Writes results/plain/penalty_scores.csv; summarise with --summary.

    ../harness/.venv-bdi/Scripts/python.exe explore_penalty.py --suffix plain
    ../harness/.venv-bdi/Scripts/python.exe explore_penalty.py --suffix plain --summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path
from statistics import mean, median

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE / "tuning"))

from smx.adapters import read_matrix  # noqa: E402
from smx.core import DEFAULT_EPS  # noqa: E402
from score_screen_metrics import all_metrics, permutations  # noqa: E402
from score_seeds import CORPUS, cases_for, ground_truth_file, latest_run, one  # noqa: E402

ALPHA = 3.0
METRICS = ("F1_Hr", "R_exp", "MRR_exp", "F1_M")
INVALID = {"random"}       # Schematch's fixed-seed random matcher is not a valid control (Q8)


def pa(S: np.ndarray) -> np.ndarray:
    den = np.maximum(S.max(axis=1, keepdims=True), S.max(axis=0, keepdims=True))
    return np.divide(S, den, out=np.zeros_like(S), where=den > DEFAULT_EPS)


def pb(S: np.ndarray) -> np.ndarray:
    pos = S > DEFAULT_EPS
    n = pos.sum(axis=1, keepdims=True) + pos.sum(axis=0, keepdims=True) - pos   # union of row i and column j
    N = S.shape[0] + S.shape[1]
    t = ((N + 1) / (n + 1)) / math.log(N + 1)
    return 1.0 / (1.0 + np.exp(-ALPHA * t))


def variants(M: np.ndarray) -> dict[str, np.ndarray]:
    a, b = pa(M), pb(M)
    return {"": M, "_pa": M * a, "_pb": M * b, "_papb": M * a * b}


def score_seed(job) -> list[dict]:
    seed_dir, corpus, marker, allowed_json = Path(job[0]), Path(job[1]), Path(job[2]), job[3]
    allowed = {tuple(k.split("|")) for k in json.loads(allowed_json)}
    run = latest_run(seed_dir)
    if run is None or not (seed_dir / "DONE").is_file():
        return []
    file_seed = Path(marker.read_text(encoding="utf-8").strip()) if marker.is_file() else None
    out = []
    for scen in sorted(run.glob("*/*")):
        ds, sc = scen.parent.name, scen.name
        if (ds, sc) not in allowed:
            continue
        gt_file = ground_truth_file(corpus, ds, sc)
        f_q = (file_seed / ds / sc / "seed.csv") if file_seed else one(scen / "_outputs" / "MatchingStepLine1")
        f_s = one(scen / "_outputs" / "MatchingStepLine2")
        if gt_file is None or f_q is None or not Path(f_q).is_file() or f_s is None:
            continue
        gt = read_matrix(gt_file)[0] > 0.5
        q, S = read_matrix(f_q)[0], read_matrix(f_s)[0]
        if q.shape != gt.shape or S.shape != gt.shape or not (np.isfinite(q).all() and np.isfinite(S).all()):
            continue
        q, S = np.clip(q, 0.0, None), np.clip(S, 0.0, None)
        perms = permutations(gt.shape, f"{ds}/{sc}")
        row = {"seed": seed_dir.name, "dataset": ds, "scenario": sc,
               "q_pos_share": float((q > DEFAULT_EPS).mean()), "sf_pos_share": float((S > DEFAULT_EPS).mean())}
        for base, M in (("q", q), ("sf", S)):
            for suf, V in variants(M).items():
                mm = all_metrics(V, gt, perms)
                for k in METRICS:
                    row[f"{base}{suf}_{k}"] = mm[k]
        out.append(row)
    return out


def jtf(r: dict, v: str) -> float:
    return (r[f"{v}_R_exp"] + r[f"{v}_F1_Hr"]) / 2


def summary(root: Path) -> None:
    rows = list(csv.DictReader(open(root / "penalty_scores.csv", encoding="utf-8")))
    for r in rows:
        for k, v in r.items():
            if k not in ("seed", "dataset", "scenario"):
                r[k] = float(v)
    names = [f"{b}{s}" for b in ("q", "sf") for s in ("", "_pa", "_pb", "_papb")]
    seeds = sorted({r["seed"] for r in rows})
    # scenario means per matcher, then mean and median over matchers (every matcher weighs the same)
    per = {s: {v: {k: mean(r[f"{v}_{k}"] for r in rows if r["seed"] == s) for k in METRICS + ("Jtf",)
                   if k != "Jtf"} for v in names} for s in seeds}
    for s in seeds:
        for v in names:
            per[s][v]["Jtf"] = mean(jtf(r, v) for r in rows if r["seed"] == s)
    lines = [f"# R5: Lily's penalties after SF (exploratory, {root.name})", "",
             f"{len(rows)} (initial matcher, scenario) pairs, {len(seeds)} initial matchers. "
             f"Positive cells: q median {median(r['q_pos_share'] for r in rows):.2f}, "
             f"SF output median {median(r['sf_pos_share'] for r in rows):.2f}.", "",
             "Mean over initial matchers of the scenario means (difference to the unpenalised matrix in brackets):", "",
             "| Matrix | " + " | ".join(METRICS) + " | J_tf |", "|---|" + "---|" * (len(METRICS) + 1)]
    for v in names:
        base = v.split("_")[0]
        cells = []
        for k in METRICS + ("Jtf",):
            val = mean(per[s][v][k] for s in seeds)
            ref = mean(per[s][base][k] for s in seeds)
            cells.append(f"{val:.4f}" + ("" if v == base else f" ({val - ref:+.4f})"))
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines += ["", "Initial matchers where the penalty helps SF (J_tf, SF x penalty minus SF), and the same for q:", "",
              "| Penalty | SF: up / down / median Δ | q: up / down / median Δ |", "|---|---|---|"]
    for suf in ("_pa", "_pb", "_papb"):
        parts = []
        for base in ("sf", "q"):
            d = [per[s][base + suf]["Jtf"] - per[s][base]["Jtf"] for s in seeds]
            parts.append(f"{sum(x > 1e-4 for x in d)} / {sum(x < -1e-4 for x in d)} / {median(d):+.4f}")
        lines.append(f"| {suf[1:]} | " + " | ".join(parts) + " |")
    lines += ["", "Per initial matcher, J_tf of SF and of SF x pa (sorted by the change):", "",
              "| Initial matcher | J_tf q | J_tf SF | J_tf SF x pa | Δ | J_tf q x pa |", "|---|---|---|---|---|---|"]
    for s in sorted(seeds, key=lambda s: per[s]["sf_pa"]["Jtf"] - per[s]["sf"]["Jtf"]):
        d = per[s]["sf_pa"]["Jtf"] - per[s]["sf"]["Jtf"]
        lines.append(f"| {s} | {per[s]['q']['Jtf']:.3f} | {per[s]['sf']['Jtf']:.3f} | {per[s]['sf_pa']['Jtf']:.3f} | "
                     f"{d:+.3f} | {per[s]['q_pa']['Jtf']:.3f} |")
    (root / "penalty_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:22]).encode("ascii", "replace").decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="plain")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="only the first N seed folders (timing)")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    # MD5 allowed since 2026-10-03 09:40 (Amendment 3's primary test has run).
    assert args.suffix in ("plain", "md5")
    root = HERE / "results" / args.suffix
    if args.summary:
        summary(root)
        return 0
    corpus = CORPUS[args.suffix]
    allowed = json.dumps([f"{d}|{s}" for d, s, _ in cases_for(args.suffix)])
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name not in INVALID]
    if args.limit:
        dirs = dirs[:args.limit]
    jobs = [(str(p), str(corpus), str(HERE / "configs" / args.suffix / p.name / "seed_file_dir.txt"), allowed)
            for p in dirs]
    with Pool(args.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_seed, jobs) for r in part]
    rows.sort(key=lambda r: (r["seed"], r["dataset"], r["scenario"]))
    name = "penalty_scores.csv" if not args.limit else "penalty_scores_sample.csv"
    with (root / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {root / name}")
    if not args.limit:
        summary(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
