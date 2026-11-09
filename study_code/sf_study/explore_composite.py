"""EXPLORATORY X52 (2026-10-05): a composite additive matcher built from the study's findings,
against the old harness's best fusion arm. Written before any of its numbers were computed.

Question: if the study's lessons are combined into one matcher -- a strong base, the datatype term, a column-class
term and the dependency-role term, no propagation -- how good is it, and how does it compare with SF as published and
with the old harness's leader, SF seeded by COMA fused with value 3-grams, Cupid and COMA (schema + instances)?

Stage 1 (this script): Valentine, all 382 scenarios, plain or md5. Everything is chosen leave-one-family-out (families
with >= 5 scenarios; the statistic uses only those families).

Candidates for the base: every non-control initial matcher with a layer-1 row on every scenario of the regime, except
the datatype matchers (they are a term here) and learned_lofo (trained on Valentine itself, and not available on the
held-out corpora). unit(.) = min-max to [0, 1] per scenario (compose_gate.unit).
  base B_K  = mean over the top-K candidates of unit(q_m), the candidates ranked by their own family-macro J_tf on the
              training families (layer-1 seed scores); K in (1, 3, 5)
  composite A = B_K + beta D + g_c eq_c + g_r R
      D    = datatype_nanfix (Amendment 5's term)
      eq_c = 1 where source and target column have the same class (layer3b/f7 value shapes or layer3b/f13 rule-based
             semantic types, from data/classfull_<regime>; MIXED = MIXED counts as equal), c in (shape, semtype)
      R    = role similarity from X46 (explore_deproles.roles: FD/UCC/IND degrees and the key flag)
      beta in (0, 1, 2, 5), g_c in (0, 0.1, 0.2, 0.5), g_r in (0, 0.2, 0.5, 1)
  The (K, beta, c, g_c, g_r) with the highest mean over training families of the family-mean J_tf is chosen per held-out
  family (ties: the earlier in grid order, which lists smaller values first).

Fixed arms (no choice):
  F_std   = fuse(SF_std(coma), [char3_soft, cupid, coma_instances], 0.35)    the old harness's leader (N2), with SF
            in the study's standard configuration (QUICK + FD edges, as the old harness), seeded by COMA (schema)
  F_pub   = the same with SF as published (layer 1 melnik, seeded by COMA)
  F_noSF  = fuse(coma, [char3_soft, cupid, coma_instances], 0.35)            the same fusion without SF (control)
  SF_std(coma), SF_pub(coma), coma                                          the ingredients alone
  fuse(base, ch, w) = (1 - w) base + w (lo + mean_k unit(ch_k) (hi - lo)), lo/hi = base's range (harness compose_gate.fuse
  with od_role.scale_align, reimplemented identically)
Reported against the LOFO-chosen best single matcher (K = 1, no terms, chosen like the others), and SF as published and
SF's best single-table graph (most key-like UCC, gvfull ucc_top1) on that same matcher.

Statistic: J_tf = (R_exp + F1_Hr)/2 (score_screen_metrics; 50 seeded tie orders), per scenario; mean within family,
mean over families; 95% bootstrap within family, 10,000 resamples, seed 20261020.

Stage 2 (after A computes the needed matchers on the held-out corpora): the recipe chosen on all Valentine families
(plain -> ctl plain and T2D; md5 -> ctl opaque) is frozen and scored there, with the fusion arms, unchanged. Classes
there come from layer3b/f7 and f13 on the tables, roles from the same profiles as Amendment 11.

    ../harness/.venv-bdi/Scripts/python.exe explore_composite.py --regime plain|md5 [--workers 10]
Output: results/<regime>/composite_{scores.csv,summary.md,recipe.json}
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE / "tuning"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "reference"))

import score_layer1  # noqa: E402
from explore_deproles import profiles, roles  # noqa: E402
from score_screen_metrics import hungarian_random, permutations, recall_expected  # noqa: E402
from score_seeds import CORPUS, cases_for, ground_truth_file, one  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

RES = HERE / "tuning" / "results"
CONTROLS = {"constant", "random", "random_fresh", "oracle", "oracle_noise025", "oracle_noise050", "oracle_erase050",
            "oracle_decoy100", "embedding_random", "rema_shuffled", "semstruct_lm_only", "datatype", "datatype_nanfix",
            "learned_lofo"}
KS = (1, 3, 5)
BETAS = (0.0, 1.0, 2.0, 5.0)
GCS = (0.0, 0.1, 0.2, 0.5)
GRS = (0.0, 0.2, 0.5, 1.0)
CLASSES = ("shape", "semtype")
TERMS = [(b, c, gc, gr) for b in BETAS for c in CLASSES for gc in GCS for gr in GRS if not (c == "semtype" and gc == 0)]
CHANNELS = ("char3_soft", "cupid", "coma_instances")
W = 0.35
B, SEED = 10_000, 20261020


def unit(m: np.ndarray) -> np.ndarray:
    finite = m[np.isfinite(m)]
    if finite.size == 0:
        return np.zeros_like(m)
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(m)
    return np.clip((np.nan_to_num(m, nan=lo) - lo) / (hi - lo), 0.0, 1.0)


def fuse(base: np.ndarray, channels: list[np.ndarray], weight: float = W) -> np.ndarray:
    blended = np.mean([unit(c) for c in channels], axis=0)
    lo, hi = float(base.min()), float(base.max())
    if hi <= lo:
        return base
    e_lo, e_hi = float(blended.min()), float(blended.max())
    scaled = np.full_like(blended, 0.5) if e_hi <= e_lo else (blended - e_lo) / (e_hi - e_lo)
    return (1.0 - weight) * base + weight * (lo + scaled * (hi - lo))


def J(S: np.ndarray, gt: np.ndarray, perms) -> float:
    return (recall_expected(S, gt) + hungarian_random(S, gt, perms)) / 2


def load(f: Path | None, rows=None, cols=None) -> np.ndarray | None:
    """A matrix, reordered to (rows, cols) when given; None if absent, non-finite or not alignable."""
    if f is None or not f.is_file():
        return None
    M, r, c = read_matrix(f)
    if rows is not None and (list(r) != list(rows) or list(c) != list(cols)):
        ri, ci = {x: i for i, x in enumerate(r)}, {x: j for j, x in enumerate(c)}
        if not (set(rows) <= set(ri) and set(cols) <= set(ci)):
            return None
        M = M[np.ix_([ri[x] for x in rows], [ci[x] for x in cols])]
    return M if np.isfinite(M).all() else None


def classes(regime: str, variant: str, ds: str, sc: str) -> dict[str, str]:
    out = {}
    for side in ("source", "target"):
        f = HERE / "data" / f"classfull_{regime}" / variant / ds / sc / f"{side}.tsv"
        for line in f.read_text(encoding="utf-8").splitlines():
            _, table, colname, cls, _ = line.split("\t")
            out[f"{table}.{colname}"] = cls
    return out


def sf_out(root: Path | None, ds: str, sc: str) -> Path | None:
    return one(root / ds / sc / "_outputs" / "MatchingStepLine2") if root else None


def score_case(job):
    regime, ds, sc, fam, bases = job
    seeds = HERE / "reference" / "seeds" / regime
    g = ground_truth_file(CORPUS[regime], ds, sc)
    f_d = seeds / "datatype_nanfix" / ds / sc / "seed.csv"
    if g is None or not f_d.is_file():
        return None
    D, rows, cols = read_matrix(f_d)
    rows, cols, gt = list(rows), list(cols), read_matrix(g)[0] > 0.5
    if D.shape != gt.shape or not np.isfinite(D).all():
        return None
    perms = permutations(gt.shape, f"{ds}/{sc}")
    q = {}
    for m in sorted({m for b in bases for m in b} | {"coma", *CHANNELS}):
        M = load(seeds / m / ds / sc / "seed.csv", rows, cols)
        if M is not None:
            q[m] = M
    eq = {}
    for v in CLASSES:
        cl = classes(regime, v, ds, sc)
        rc, cc = [cl.get(r) for r in rows], [cl.get(c) for c in cols]
        eq[v] = np.array([[1.0 if a is not None and a == b else 0.0 for b in cc] for a in rc])
    Fs, _ = roles(rows, *profiles(regime, ds, sc, "source"))
    Ft, _ = roles(cols, *profiles(regime, ds, sc, "target"))
    R = 1.0 - np.abs(Fs[:, None, :] - Ft[None, :, :]).mean(axis=2)
    row = {"dataset": ds, "scenario": sc, "family": fam}
    for b in bases:
        if not all(m in q for m in b):
            continue
        base = np.mean([unit(q[m]) for m in b], axis=0)
        key = "+".join(b)
        for beta, c, gc, gr in TERMS:
            row[f"A|{key}|{beta}|{c}|{gc}|{gr}"] = J(base + beta * D + gc * eq[c] + gr * R, gt, perms)
    std_root = score_layer1.run_dir(HERE / "results" / regime / "coma") if (HERE / "results" / regime / "coma").is_dir() else None
    pub_root = HERE / "tuning" / "results" / f"layer1_{regime}" / "coma__melnik"
    pub_root = score_layer1.run_dir(pub_root) if pub_root.is_dir() else None
    sf_std = load(sf_out(std_root, ds, sc), rows, cols)
    sf_pub = load(sf_out(pub_root, ds, sc), rows, cols)
    ch = [q[m] for m in CHANNELS if m in q]
    if len(ch) == len(CHANNELS):
        if sf_std is not None:
            row["F_std"], row["SF_std_coma"] = J(fuse(sf_std, ch), gt, perms), J(sf_std, gt, perms)
        if sf_pub is not None:
            row["F_pub"], row["SF_pub_coma"] = J(fuse(sf_pub, ch), gt, perms), J(sf_pub, gt, perms)
        if "coma" in q:
            row["F_noSF"], row["coma"] = J(fuse(q["coma"], ch), gt, perms), J(q["coma"], gt, perms)
    return row


def layer1_scores(regime: str, src: str = "layer1", arm: str = "melnik") -> dict[tuple[str, str, str], tuple[float, float]]:
    """(matcher, dataset, scenario) -> (J of the initial matcher, J of SF in that arm)."""
    out = {}
    f = RES / f"{src}_{regime}_scores" / "scores.csv"
    if not f.is_file():
        return out
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if r["arm"] == arm:
            out[(r["seed"], r["dataset"], r["scenario"])] = (
                (float(r["seed_R_exp"]) + float(r["seed_F1_Hr"])) / 2, (float(r["sf_R_exp"]) + float(r["sf_F1_Hr"])) / 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5"), required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    cases = cases_for(a.regime)
    fams = defaultdict(list)
    for d, c, f in cases:
        fams[f].append((d, c))
    big = [f for f, v in fams.items() if len(v) >= 5]

    l1 = layer1_scores(a.regime)
    cover = defaultdict(int)
    for (m, _, _) in l1:
        cover[m] += 1
    cands = sorted(m for m, n in cover.items() if n == len(cases) and m not in CONTROLS)
    fam_of = {(d, c): f for d, c, f in cases}

    def own(m: str, famset) -> float:
        return float(np.mean([np.mean([l1[(m, d, c)][0] for d, c in fams[f]]) for f in famset]))

    folds = {h: [f for f in big if f != h] for h in big}
    folds["_all"] = list(big)
    top = {h: sorted(cands, key=lambda m: (-own(m, tr), m)) for h, tr in folds.items()}
    base_of = {(h, k): tuple(top[h][:k]) for h in folds for k in KS}
    bases = sorted(set(base_of.values()))
    print(f"{len(cands)} candidates; {len(bases)} distinct bases", flush=True)

    with Pool(a.workers) as pool:
        rows = [r for r in pool.imap_unordered(score_case, [(a.regime, d, c, f, bases) for d, c, f in cases]) if r]
    by = {(r["dataset"], r["scenario"]): r for r in rows}
    missing = [k for k in fams_cases(big, fams) if k not in by]
    out = HERE / "results" / a.regime
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with (out / "composite_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def fmean(fn, famset) -> float:
        return float(np.mean([np.mean([fn(d, c) for d, c in fams[f]]) for f in famset]))

    def comp_key(h, k, t) -> str:
        beta, c, gc, gr = t
        return f"A|{'+'.join(base_of[(h, k)])}|{beta}|{c}|{gc}|{gr}"

    choice = {}
    for h, tr in folds.items():
        best, arg = -1.0, None
        for k in KS:
            for t in TERMS:
                key = comp_key(h, k, t)
                if not all(key in by[(d, c)] for f in tr for d, c in fams[f]):
                    continue
                v = fmean(lambda d, c: by[(d, c)][key], tr)
                if v > best + 1e-12:
                    best, arg = v, (k, t)
        choice[h] = arg
    single = {h: top[h][0] for h in folds}

    def held(fn_of_fold):
        return lambda d, c: fn_of_fold(fam_of[(d, c)])(d, c)

    comp = held(lambda h: (lambda d, c: by[(d, c)][comp_key(h, *choice[h])]))
    base_only = held(lambda h: (lambda d, c: by[(d, c)][comp_key(h, choice[h][0], (0.0, "shape", 0.0, 0.0))]))
    best1 = held(lambda h: (lambda d, c: l1[(single[h], d, c)][0]))
    sf_on_best1 = held(lambda h: (lambda d, c: l1[(single[h], d, c)][1]))
    gv = layer1_scores(a.regime, "gvfull", "ucc_top1")
    ucc_on_best1 = held(lambda h: (lambda d, c: gv[(single[h], d, c)][1]))
    fixed = {n: (lambda n: lambda d, c: by[(d, c)][n])(n) for n in ("F_std", "F_pub", "F_noSF", "SF_std_coma", "SF_pub_coma", "coma")}
    rng = np.random.default_rng(SEED)

    def ok(fn) -> bool:
        try:
            [fn(d, c) for f in big for d, c in fams[f]]
            return True
        except KeyError:
            return False

    def stat(fx, fy=None) -> str:
        if not ok(fx) or (fy and not ok(fy)):
            return "n/a (missing scenarios)"
        per = [np.array([fx(d, c) - (fy(d, c) if fy else 0.0) for d, c in fams[f]]) for f in big]
        est = float(np.mean([v.mean() for v in per]))
        boots = np.mean([[rng.choice(v, size=v.size).mean() for _ in range(B)] for v in per], axis=0)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return f"{est:+.4f} [{lo:+.4f}, {hi:+.4f}]" if fy else f"{est:.4f} [{lo:.4f}, {hi:.4f}]"

    arms = [("composite A (chosen LOFO)", comp), ("its base B_K alone", base_only),
            ("best single initial matcher (chosen LOFO)", best1), ("SF as published on that matcher", sf_on_best1),
            ("SF, most key-like UCC graph, on that matcher", ucc_on_best1),
            ("fusion F_std: SF_std(COMA) + 3-gram + Cupid + COMA-inst. (old leader)", fixed["F_std"]),
            ("fusion F_pub: SF as published (COMA) + the same", fixed["F_pub"]),
            ("fusion without SF: COMA + the same", fixed["F_noSF"]),
            ("SF_std seeded by COMA", fixed["SF_std_coma"]), ("SF as published seeded by COMA", fixed["SF_pub_coma"]),
            ("COMA alone", fixed["coma"])]
    contrasts = [("composite − best single matcher", comp, best1), ("composite − its base alone", comp, base_only),
                 ("composite − SF as published on the best matcher", comp, sf_on_best1),
                 ("composite − SF ucc_top1 on the best matcher", comp, ucc_on_best1),
                 ("composite − F_std (old leader)", comp, fixed["F_std"]), ("composite − F_pub", comp, fixed["F_pub"]),
                 ("F_std − fusion without SF", fixed["F_std"], fixed["F_noSF"]),
                 ("F_pub − fusion without SF", fixed["F_pub"], fixed["F_noSF"]),
                 ("fusion without SF − COMA alone", fixed["F_noSF"], fixed["coma"])]
    recipe = {"regime": a.regime, "K": choice["_all"][0], "base": list(base_of[("_all", choice["_all"][0])]),
              "beta": choice["_all"][1][0], "class": choice["_all"][1][1], "g_c": choice["_all"][1][2],
              "g_r": choice["_all"][1][3], "candidates": len(cands)}
    (out / "composite_recipe.json").write_text(json.dumps(recipe, indent=1), encoding="utf-8")
    lines = [f"# X52 ({a.regime}): a composite additive matcher against the old fusion leader (EXPLORATORY)", "",
             f"{len(cands)} base candidates; {len(rows)} scenarios scored ({len(missing)} of the families' scenarios "
             f"missing); families: {', '.join(big)}. J_tf, family macro, 95% CI.", "",
             "Choices per held-out family (K, beta, class, g_c, g_r; base; best single matcher):", ""]
    for h in big:
        k, (beta, c, gc, gr) = choice[h]
        lines.append(f"- {h}: K {k}, beta {beta:g}, {c} {gc:g}, roles {gr:g}; base {'+'.join(base_of[(h, k)])}; "
                     f"single {single[h]}")
    lines += ["", f"Frozen recipe (all families): {json.dumps(recipe)}", "", "| arm | J_tf |", "|---|---|"]
    lines += [f"| {n} | {stat(fn)} |" for n, fn in arms]
    lines += ["", "| contrast | J_tf |", "|---|---|"] + [f"| {n} | {stat(x, y)} |" for n, x, y in contrasts]
    (out / "composite_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


def fams_cases(big, fams):
    return [k for f in big for k in fams[f]]


if __name__ == "__main__":
    raise SystemExit(main())
