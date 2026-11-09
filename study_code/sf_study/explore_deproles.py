"""EXPLORATORY X46 (2026-10-05): can an additive term with dependency information match SF with
dependency edges? Written before any of its numbers were computed.

Motivation: every additive baseline so far used datatypes, classes or table context, never FDs, UCCs or INDs, while
SF's largest gains come from dependency edges on recoded data (FD+UCC+IND about +0.13, top-1 key + weighted FDs
+0.142 over melnik). X32 showed FD edges carry each column's role (its FD degrees), not the pairs.

Per initial matcher and scenario (Valentine, all 382 scenarios; plain or md5), with q+ = max(q, 0) and D the NaN-free
datatype matcher (Amendment 5's A_d = q+ + 5D):
  role vector r(c) per column, from its side's profiles (metadata/<side>/<table>/{FD,UCC}_results.txt, <side>/inds.txt),
  each feature divided by its maximum over the side's columns (0 if that is 0):
      FD left-hand-side count, FD right-hand-side count, UCCs containing c, key (c is in a UCC of the smallest arity
      on its side), IND out-degree, IND in-degree
  R_ij = 1 − mean_k |r_k(i) − r_k(j)|                      (role similarity, in [0, 1])
  N_ij = one-step FD-neighbour evidence: with nb(c) = the columns sharing an FD with c (LHS member ↔ RHS),
      1/2 (mean over k in nb(i) of max over l in nb(j) of q+_kl + the same from j's side), 0 if either has no
      neighbour, divided by its maximum over the scenario
  A(γ, λ) = q+ + 5D + γ R + λ N,  γ, λ in G = (0, 0.1, 0.2, 0.5, 1)
Weights are chosen by leave-one-family-out (families with >= 5 scenarios; Wikidata uses the choice on all of them):
the (γ, λ) with the highest mean over included matchers of the family-macro J_tf on the training families (ties: the
smaller sum). Three choices are reported: role only (λ = 0), neighbours only (γ = 0), and both.

Compared with SF from the registered and exploratory runs (same initial matrices, metrics and tie orders;
score_layer1 rows): layer 1 melnik, fd; layer 3 ucc, fd_ucc_ind; X33 ucc_top1, ucc1_fd_topk.
Statistic: per scenario the mean over the included matchers (non-controls with q, D and every SF arm on every
scenario), mean within family (>= 5 scenarios), mean over families; 95% bootstrap within family, 10,000 resamples,
seed 20261011.

    ../harness/.venv-bdi/Scripts/python.exe explore_deproles.py --regime plain|md5 [--workers 10]
Output: results/<regime>/deproles_{scores.csv,summary.md}.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE / "tuning"))
sys.path.insert(0, str(HERE))

from score_screen_metrics import all_metrics, permutations  # noqa: E402
from score_seeds import CORPUS, cases_for, ground_truth_file  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

RES = HERE / "tuning" / "results"
sys.path.insert(0, str(HERE / "reference"))
import valentine549 as V  # noqa: E402  (rerun 549: used only under SMX_VALENTINE=549)
G = (0.0, 0.1, 0.2, 0.5, 1.0)
GRID = list(itertools.product(G, G))
SF_ARMS = {"melnik": "layer1", "fd": "layer1", "ucc": "layer3", "fd_ucc_ind": "layer3",
           "ucc_top1": "gvfull", "ucc1_fd_topk": "gvfull"}
CONTROLS = {"constant", "random", "oracle", "oracle_noise025", "oracle_noise050", "oracle_erase050",
            "oracle_decoy100", "embedding_random", "rema_shuffled", "semstruct_lm_only", "datatype", "datatype_nanfix"}
B, SEED = 10_000, 20261011
# Corrected 2026-10-06: the right-hand side runs up to " (pdep", so column names with spaces are kept.
# The original pattern stopped at the first space (OpenData plain: 68% of FDs, T2D: 18%). Setting
# DEPROLES_FROZEN_FD_PARSE=1 reproduces the original parse (for the registered Amendment 11 report, frozen vs corrected).
FD_RE_FROZEN = re.compile(r"^\[(?P<lhs>[^\]]*)\]\s*-->\s*(?P<rhs>\S+)")
FD_RE_FIXED = re.compile(r"^\[(?P<lhs>[^\]]*)\]\s*-->\s*(?P<rhs>.+?)(?:\s+\(pdep\b.*)?$")
FD_RE = FD_RE_FROZEN if os.environ.get("DEPROLES_FROZEN_FD_PARSE") == "1" else FD_RE_FIXED
BR_RE = re.compile(r"^\[(?P<cols>[^\]]*)\]")
IND_RE = re.compile(r"^\[(?P<lhs>[^\]]+)\]\s*\[=\s*\[(?P<rhs>[^\]]+)\]")


def col(qualified: str) -> str:
    """'source.csv.assay_id' -> 'source.assay_id' (the matrices' label form)."""
    t, c = qualified.strip().split(".csv.", 1)
    return f"{t}.{c}"


def profiles(regime: str, ds: str, sc: str, side: str):
    """FDs (lhs tuple, rhs), UCCs (tuples) and unary INDs (dep, ref) of one side, in matrix-label form."""
    meta = (V.corpus_dir(regime, ds, sc) if V.V549 and regime in V.NAME else CORPUS[regime] / ds / sc) / "metadata" / side
    fds, uccs, inds = [], [], []
    for tdir in sorted(p for p in meta.iterdir() if p.is_dir()) if meta.is_dir() else []:
        f = tdir / "FD_results.txt"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                m = FD_RE.match(line.strip())
                if m:
                    lhs = tuple(col(x) for x in m.group("lhs").split(",") if x.strip())
                    fds.append((lhs, col(m.group("rhs"))))
        u = tdir / "UCC_results.txt"
        if u.is_file():
            for line in u.read_text(encoding="utf-8").splitlines():
                m = BR_RE.match(line.strip())
                if m and m.group("cols").strip():
                    uccs.append(tuple(col(x) for x in m.group("cols").split(",")))
    fi = meta / "inds.txt"
    if fi.is_file():
        for line in fi.read_text(encoding="utf-8").splitlines():
            m = IND_RE.match(line.strip())
            if m and "," not in m.group("lhs") and "," not in m.group("rhs"):
                inds.append((col(m.group("lhs")), col(m.group("rhs"))))
    return fds, uccs, inds


def roles(labels: list[str], fds, uccs, inds) -> tuple[np.ndarray, dict[str, set[str]]]:
    idx = {c: i for i, c in enumerate(labels)}
    F = np.zeros((len(labels), 6))
    nb = defaultdict(set)
    for lhs, rhs in fds:
        for x in lhs:
            if x in idx:
                F[idx[x], 0] += 1
            if x in idx and rhs in idx:
                nb[x].add(rhs)
                nb[rhs].add(x)
        if rhs in idx:
            F[idx[rhs], 1] += 1
    min_arity = min((len(u) for u in uccs), default=0)
    for u in uccs:
        for x in u:
            if x in idx:
                F[idx[x], 2] += 1
                if len(u) == min_arity:
                    F[idx[x], 3] = 1
    for d, r in inds:
        if d in idx:
            F[idx[d], 4] += 1
        if r in idx:
            F[idx[r], 5] += 1
    mx = F.max(axis=0)
    F = np.divide(F, mx, out=np.zeros_like(F), where=mx > 0)
    return F, nb


def neighbour_term(qp: np.ndarray, rows, cols, nb_s, nb_t) -> np.ndarray:
    ri, ci = {c: i for i, c in enumerate(rows)}, {c: j for j, c in enumerate(cols)}
    N = np.zeros_like(qp)
    for i, a in enumerate(rows):
        na = [ri[x] for x in nb_s.get(a, ()) if x in ri]
        if not na:
            continue
        for j, b in enumerate(cols):
            nbb = [ci[x] for x in nb_t.get(b, ()) if x in ci]
            if not nbb:
                continue
            block = qp[np.ix_(na, nbb)]
            N[i, j] = 0.5 * (block.max(axis=1).mean() + block.max(axis=0).mean())
    top = N.max()
    return N / top if top > 0 else N


def score_matcher(job):
    regime, m, cs = job
    seeds = HERE / "reference" / "seeds" / regime
    out = []
    for ds, sc, fam in cs:
        f_q, f_d = seeds / m / ds / sc / "seed.csv", seeds / "datatype_nanfix" / ds / sc / "seed.csv"
        g = ground_truth_file(CORPUS[regime], ds, sc)
        if V.V549:   # 549: initial matrices per valentine549.seed_file; truth from the 382 tree or the new168 tree
            f_q, f_d = V.seed_file(regime, m, ds, sc), V.seed_file(regime, "datatype_nanfix", ds, sc)
            g = ground_truth_file((V.NEW_TREE[regime] if V.is_new(ds, sc) else CORPUS[regime]), ds, sc)
        if not (f_q.is_file() and f_d.is_file() and g is not None):
            continue
        q, rows, cols = read_matrix(f_q)
        D = read_matrix(f_d)[0]
        gt = read_matrix(g)[0] > 0.5
        if not (q.shape == D.shape == gt.shape) or not (np.isfinite(q).all() and np.isfinite(D).all()):
            continue
        qp = np.clip(q, 0, None)
        Fs, nbs = roles(list(rows), *profiles(regime, ds, sc, "source"))
        Ft, nbt = roles(list(cols), *profiles(regime, ds, sc, "target"))
        R = 1.0 - np.abs(Fs[:, None, :] - Ft[None, :, :]).mean(axis=2)
        N = neighbour_term(qp, list(rows), list(cols), nbs, nbt)
        perms = permutations(gt.shape, f"{ds}/{sc}")
        row = {"seed": m, "dataset": ds, "scenario": sc, "family": fam}
        base = qp + 5.0 * D
        for gm, lm in GRID:
            r = all_metrics(base + gm * R + lm * N, gt, perms)
            row[f"A_{gm}_{lm}"] = (r["R_exp"] + r["F1_Hr"]) / 2
            if (gm, lm) == (0.0, 0.0):
                row["A_d_R"], row["A_d_F"] = r["R_exp"], r["F1_Hr"]
        out.append(row)
    return out


def sf_rows(regime: str) -> dict[tuple[str, str, str, str], float]:
    if V.V549:   # the assembled 549 rows; layer 3 is not rerun on 549
        out = {}
        for folder, arms in ((f"layer1_{V.vreg(regime)}_scores", ("melnik", "fd")),
                             (f"gvfull_{V.vreg(regime)}_scores", ("ucc_top1", "ucc1_fd_topk"))):
            for r in V.assembled(folder) or []:
                if r["arm"] in arms and r["arm"] in SF_ARMS:
                    out[(r["seed"], r["arm"], r["dataset"], r["scenario"])] = (float(r["sf_R_exp"]) + float(r["sf_F1_Hr"])) / 2
        return out
    files = {"layer1": RES / f"layer1_{regime}_scores" / "scores.csv", "layer3": RES / f"layer3_{regime}_scores" / "scores.csv",
             "gvfull": RES / f"gvfull_{regime}_scores" / "scores.csv"}
    want = defaultdict(set)
    for arm, src in SF_ARMS.items():
        want[src].add(arm)
    out = {}
    for src, f in files.items():
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["arm"] in want[src]:
                out[(r["seed"], r["arm"], r["dataset"], r["scenario"])] = (float(r["sf_R_exp"]) + float(r["sf_F1_Hr"])) / 2
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5"), required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    cs = cases_for(a.regime)
    out = HERE / "results" / a.regime
    if V.V549:
        # SF arms on 549: layer 1 melnik, fd and X33 ucc_top1 (+ ucc1_fd_topk on plain); layer 3 (ucc, fd_ucc_ind) and
        # the recoded ucc1_fd_topk are not rerun, so they leave the panel definition and the contrasts
        for arm in ("ucc", "fd_ucc_ind") + (("ucc1_fd_topk",) if a.regime == "md5" else ()):
            SF_ARMS.pop(arm)
        need = [("layer1", "melnik"), ("layer1", "fd"), ("gvfull", "ucc_top1"), ("layer1nan", "melnik")] + \
            ([("gvfull", "ucc1_fd_topk")] if a.regime == "plain" else [])
        miss = [f"{f}_{a.regime}/{x}" for f, x in need if V.arm_status(f"{f}_{a.regime}", x) != "assembled"]
        if miss:
            print(f"deproles {a.regime} 549: pending ({', '.join(miss)} not assembled); nothing written")
            return 0
        cs, out = V.cases549(), HERE / "results" / V.vreg(a.regime)
    sf = sf_rows(a.regime)
    seeds = sorted({k[0] for k in sf} - CONTROLS)
    seeds = [s for s in seeds if all((s, arm, d, c) in sf for arm in SF_ARMS for d, c, _ in cs)]
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_matcher, [(a.regime, s, cs) for s in seeds]) for r in part]
    cover = defaultdict(int)
    for r in rows:
        cover[r["seed"]] += 1
    seeds = [s for s in seeds if cover[s] == len(cs)]
    rows = [r for r in rows if r["seed"] in seeds]
    with (out / "deproles_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = {(r["seed"], r["dataset"], r["scenario"]): r for r in rows}
    fams = defaultdict(list)
    for d, c, f in cs:
        fams[f].append((d, c))
    big = {f: v for f, v in fams.items() if len(v) >= 5}

    def macro(val, famset):     # mean over matchers of the family macro
        return np.mean([np.mean([np.mean([val(s, d, c) for d, c in fams[f]]) for f in famset]) for s in seeds])

    def choose(cands, train):
        scores = {gl: macro(lambda s, d, c, gl=gl: by[(s, d, c)][f"A_{gl[0]}_{gl[1]}"], train) for gl in cands}
        best = max(scores.values())
        return min((gl for gl, v in scores.items() if v >= best - 1e-12), key=lambda gl: (gl[0] + gl[1], gl))

    choices = {}
    for name, cands in (("role", [(g, 0.0) for g in G]), ("nb", [(0.0, l) for l in G]), ("both", GRID)):
        choices[name] = {held: choose(cands, [f for f in big if f != held]) for held in big}
        choices[name]["_other"] = choose(cands, list(big))
    pick = lambda name, fam: choices[name].get(fam, choices[name]["_other"])  # noqa: E731
    A = lambda name: (lambda s, d, c, fam: by[(s, d, c)]["A_{}_{}".format(*pick(name, fam))])  # noqa: E731
    fam_of = {(d, c): f for d, c, f in cs}
    rng = np.random.default_rng(SEED)

    def contrast(fx, fy):
        per = {f: np.array([np.mean([fx(s, d, c) - fy(s, d, c) for s in seeds]) for d, c in v]) for f, v in big.items()}
        est = float(np.mean([v.mean() for v in per.values()]))
        boots = np.mean([[rng.choice(v, size=v.size).mean() for _ in range(B)] for v in per.values()], axis=0)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pos = sum(np.mean([np.mean([fx(s, d, c) - fy(s, d, c) for d, c in v]) for v in big.values()]) > 1e-4 for s in seeds)
        return f"{est:+.4f} [{lo:+.4f}, {hi:+.4f}] | {pos} of {len(seeds)}"

    add = {name: (lambda s, d, c, name=name: A(name)(s, d, c, fam_of[(d, c)])) for name in ("role", "nb", "both")}
    A_d = lambda s, d, c: by[(s, d, c)]["A_0.0_0.0"]  # noqa: E731
    S = {arm: (lambda s, d, c, arm=arm: sf[(s, arm, d, c)]) for arm in SF_ARMS}
    lines = [f"# X46 ({a.regime}): an additive term with dependency information against SF with dependency edges "
             "(EXPLORATORY)", "", f"{len(seeds)} initial matchers, {len(cs)} scenarios. Weights chosen leave-one-family-out "
             "from G = (0, 0.1, 0.2, 0.5, 1); chosen (γ, λ) per held-out family:", ""]
    for name, ch in choices.items():
        lines.append(f"- {name}: " + "; ".join(f"{f} {v}" for f, v in ch.items()))
    lines += ["", "| contrast | J_tf [95% CI] | matchers > 0 |", "|---|---|---|"]
    for name in ("role", "nb", "both"):
        lines.append(f"| A_{name} − A_d | {contrast(add[name], A_d)} |")
    for arm in SF_ARMS:
        lines.append(f"| SF {arm} − A_d | {contrast(S[arm], A_d)} |")
    for name in ("role", "both"):
        for arm in [x for x in ("melnik", "fd", "ucc", "fd_ucc_ind", "ucc_top1", "ucc1_fd_topk") if x in SF_ARMS]:
            lines.append(f"| A_{name} − SF {arm} | {contrast(add[name], S[arm])} |")
    (out / "deproles_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
