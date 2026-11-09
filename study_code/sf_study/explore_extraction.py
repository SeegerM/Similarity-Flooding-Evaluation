"""EXPLORATORY X50 (2026-10-05; the external reviewer's optional point (1), approved): do the
conclusions survive a deployable extraction policy that may leave columns unmatched, or match a column more than once?

Matrices per (initial matcher, scenario): q (the initial matcher), A_d = q+ + 5 D (Amendment 5's term), and SF as
published (layer 1's `melnik`). Policies, applied identically to all three (scores clipped at 0):
  forced          the one-to-one assignment (Hungarian), ties at random over 50 seeded orders (tie-free F1, as in the
                  paper)
  hung@τ          the same assignment, keeping only pairs with s >= τ · max(S): abstention, still one-to-one
  rel@t           Melnik et al.'s relative filter: keep (i, j) if s_ij >= t · max(row i) and s_ij >= t · max(col j),
                  s_ij > 0; t = 1 is mutual best (ties kept); t < 1 allows a column several matches
  glob@τ          keep every pair with s >= τ · max(S) (many-to-many)
Selection: per matrix type and regime, the policy and parameter with the highest mean over matchers of the family-
macro F1 on the training families, leave one family out (Valentine: families with >= 5 scenarios; independent corpus:
databases). Each held-out family is scored with its fold's choice. Ties: forced first, then fewer kept pairs.
Reported per regime (plain, md5 on Valentine with one-to-one, partial-overlap truth; indep with many-to-many truth):
held-out F1 of every matrix type under forced and under the validated policy, and the contrasts SF − q and A_d − SF
under both. Statistic: per scenario the mean over included matchers (non-controls with q, D and SF on every scenario),
mean within family, mean over families; 95% bootstrap within family, 10,000 resamples, seed 20261015.

    ../harness/.venv-bdi/Scripts/python.exe explore_extraction.py --regime plain|md5|indep
Output: results/<regime>/extraction_{scores.csv,summary.md}
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE / "tuning"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "reference"))

import score_layer1  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402
from score_seeds import CORPUS, cases_for, ground_truth_file, one  # noqa: E402
sys.path.insert(0, str(HERE / "reference"))
import valentine549 as V  # noqa: E402  (rerun 549: SMX_VALENTINE=549 reads the 549 inputs; unused otherwise)
from smx.adapters import read_matrix  # noqa: E402

RES = HERE / "tuning" / "results"
CONTROLS = {"constant", "random", "oracle", "oracle_noise025", "oracle_noise050", "oracle_erase050",
            "oracle_decoy100", "embedding_random", "rema_shuffled", "semstruct_lm_only", "datatype", "datatype_nanfix"}
TAUS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
RELS = (1.0, 0.95, 0.9, 0.8, 0.7)
GLOBS = (0.5, 0.6, 0.7, 0.8, 0.9)
POLICIES = ["forced"] + [f"hung@{t}" for t in TAUS] + [f"rel@{t}" for t in RELS] + [f"glob@{t}" for t in GLOBS]
B, SEED = 10_000, 20261015


def f1(M: np.ndarray, gt: np.ndarray) -> float:
    tp = int((M & gt).sum())
    p, r = (tp / M.sum() if M.sum() else 0.0), (tp / gt.sum() if gt.sum() else 0.0)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def policy_f1(S: np.ndarray, gt: np.ndarray, perms) -> dict[str, float]:
    S = np.clip(S, 0, None)
    top = S.max()
    out = {}
    hung = defaultdict(list)
    for pr, pc in perms:
        rows, cols = linear_sum_assignment(S[np.ix_(pr, pc)], maximize=True)
        rr, cc = pr[rows], pc[cols]
        vals = S[rr, cc]
        for name, thr in [("forced", 1e-12)] + [(f"hung@{t}", t * top) for t in TAUS]:
            M = np.zeros(S.shape, dtype=bool)
            keep = vals >= max(thr, 1e-12)
            M[rr[keep], cc[keep]] = True
            hung[name].append(f1(M, gt))
    out.update({k: float(np.mean(v)) for k, v in hung.items()})
    rmax, cmax = S.max(axis=1, keepdims=True), S.max(axis=0, keepdims=True)
    for t in RELS:
        out[f"rel@{t}"] = f1((S > 0) & (S >= t * rmax - 1e-12) & (S >= t * cmax - 1e-12), gt)
    for t in GLOBS:
        out[f"glob@{t}"] = f1((S > 0) & (S >= t * top - 1e-12), gt)
    return out


def score_matcher(job):
    regime, m, cases = job
    seeds = HERE / "reference" / "seeds" / regime
    sf_root = RES / f"layer1_{regime}" / f"{m}__melnik"
    sf_dir = score_layer1.run_dir(sf_root) if sf_root.is_dir() else None
    out = []
    for ds, sc, fam in cases:
        f_q, f_d = seeds / m / ds / sc / "seed.csv", seeds / "datatype_nanfix" / ds / sc / "seed.csv"
        g = ground_truth_file(CORPUS[regime], ds, sc)
        f_s = one(sf_dir / ds / sc / "_outputs" / "MatchingStepLine2") if sf_dir else None
        if V.V549 and regime in V.NAME:   # 549: matrices by the assembly rules (valentine549.seed_file / sf_run)
            f_q, f_d = V.seed_file(regime, m, ds, sc), V.seed_file(regime, "datatype_nanfix", ds, sc)
            g = ground_truth_file(V.NEW_TREE[regime] if V.is_new(ds, sc) else CORPUS[regime], ds, sc)
            rd = V.sf_run(regime, m, "melnik", ds, sc)
            f_s = one(rd / ds / sc / "_outputs" / "MatchingStepLine2") if rd else None
        if not (f_q.is_file() and f_d.is_file() and g is not None and f_s is not None):
            continue
        q, D, S = read_matrix(f_q)[0], np.nan_to_num(read_matrix(f_d)[0]), read_matrix(f_s)[0]
        gt = read_matrix(g)[0] > 0.5
        if not (q.shape == D.shape == S.shape == gt.shape) or not (np.isfinite(q).all() and np.isfinite(S).all()):
            continue
        perms = permutations(gt.shape, f"{ds}/{sc}")
        row = {"seed": m, "dataset": ds, "scenario": sc, "family": fam}
        for name, M in (("q", q), ("A_d", np.clip(q, 0, None) + 5.0 * D), ("sf", S)):
            for pol, v in policy_f1(M, gt, perms).items():
                row[f"{name}|{pol}"] = v
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5", "indep"), required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    cases = V.cases549() if V.V549 and a.regime in V.NAME else cases_for(a.regime)
    seeds_root = HERE / "reference" / "seeds" / a.regime
    matchers = sorted(p.name for p in seeds_root.iterdir() if p.is_dir() and p.name not in CONTROLS)
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_matcher, [(a.regime, m, cases) for m in matchers]) for r in part]
    cover = defaultdict(int)
    for r in rows:
        cover[r["seed"]] += 1
    used = sorted(m for m, n in cover.items() if n == len(cases))
    rows = [r for r in rows if r["seed"] in used]
    out = HERE / "results" / (V.vreg(a.regime) if a.regime in V.NAME else a.regime)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "extraction_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = {(r["seed"], r["dataset"], r["scenario"]): r for r in rows}
    fams = defaultdict(list)
    for d, c, f in cases:
        fams[f].append((d, c))
    big = {f: v for f, v in fams.items() if len(v) >= 5} if a.regime != "indep" else dict(fams)
    fam_of = {(d, c): f for d, c, f in cases}

    def macro(key, famset):
        return float(np.mean([np.mean([np.mean([by[(m, d, c)][key] for d, c in fams[f]]) for f in famset]) for m in used]))

    choice = {}
    for name in ("q", "A_d", "sf"):
        choice[name] = {}
        for held in list(big) + ["_other"]:
            train = [f for f in big if f != held]
            scores = {p: macro(f"{name}|{p}", train) for p in POLICIES}
            best = max(scores.values())
            choice[name][held] = next(p for p in POLICIES if scores[p] >= best - 1e-12)
    val = lambda name: (lambda m, d, c: by[(m, d, c)][f"{name}|{choice[name].get(fam_of[(d, c)], choice[name]['_other'])}"])  # noqa: E731
    forced = lambda name: (lambda m, d, c: by[(m, d, c)][f"{name}|forced"])  # noqa: E731
    rng = np.random.default_rng(SEED)

    def stat(fx, fy=None) -> str:
        per = {f: np.array([np.mean([fx(m, d, c) - (fy(m, d, c) if fy else 0.0) for m in used]) for d, c in v])
               for f, v in big.items()}
        est = float(np.mean([v.mean() for v in per.values()]))
        boots = np.mean([[rng.choice(v, size=v.size).mean() for _ in range(B)] for v in per.values()], axis=0)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return f"{est:+.4f} [{lo:+.4f}, {hi:+.4f}]" if fy else f"{est:.4f} [{lo:.4f}, {hi:.4f}]"

    lines = [f"# X50 ({a.regime}): extraction with abstention and many-to-many matches (EXPLORATORY)", "",
             f"{len(used)} initial matchers, {len(cases)} scenarios; families: {', '.join(big)}. Policies chosen "
             "leave-one-family-out per matrix type:", ""]
    for name in ("q", "A_d", "sf"):
        lines.append(f"- {name}: " + "; ".join(f"{f} {p}" for f, p in choice[name].items()))
    lines += ["", "| | forced (one-to-one) F1 | validated policy F1 |", "|---|---|---|"]
    for name, lab in (("q", "initial matcher q"), ("A_d", "additive datatype term"), ("sf", "SF as published")):
        lines.append(f"| {lab} | {stat(forced(name))} | {stat(val(name))} |")
    lines += ["", "| contrast | forced | validated policy |", "|---|---|---|",
              f"| SF − q | {stat(forced('sf'), forced('q'))} | {stat(val('sf'), val('q'))} |",
              f"| A_d − SF | {stat(forced('A_d'), forced('sf'))} | {stat(val('A_d'), val('sf'))} |",
              f"| validated − forced, q | {stat(val('q'), forced('q'))} | |",
              f"| validated − forced, SF | {stat(val('sf'), forced('sf'))} | |"]
    (out / "extraction_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
