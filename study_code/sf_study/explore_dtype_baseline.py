"""EXPLORATORY (EXPLORATORY.md X19; external review ER6, with the residual check ER3 asks for).

Proposition 1: on a single table without dependency edges, the reference configuration's output is
a * q + b(datatype pair). If that holds, flooding adds only an offset per datatype pair. Is that
offset worth propagating for, or does a cheap additive datatype term do as well?

Baseline: S_beta = q + beta * D, with D the `datatype` initial matcher's matrix (Schematch's
datatype compatibility: 1 same type, 0.8 compatible, 0 otherwise; its NaN cells, a known bug of
that matcher, are set to 0 and counted). beta is chosen on the screen scenarios only, either per
initial matcher or one value for all; everything is evaluated on the dev scenarios outside the
screen. The lockbox scenarios are not read.

SF: layer 1's `melnik` arm (the reference configuration of RQ1), on the same scenarios.

Residual check (ER3): per (matcher, scenario), least squares of SF's output on a*q + b_block, with
blocks (a) from the constant initial matcher's SF output (Amendment 1's partition) and (b) from D's
distinct rows and columns. Reports the largest residual relative to SF's range.

    ../harness/.venv-bdi/Scripts/python.exe explore_dtype_baseline.py            # score
    ../harness/.venv-bdi/Scripts/python.exe explore_dtype_baseline.py --summary  # summarise
Output: results/plain/dtype_baseline_{scores.csv,summary.md}.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE / "tuning"))
sys.path.insert(0, str(HERE / "reference"))

from smx.adapters import read_matrix  # noqa: E402
from score_screen_metrics import all_metrics, permutations  # noqa: E402
from score_seeds import CORPUS, ground_truth_file, one  # noqa: E402
from check_prop1 import classes  # noqa: E402

PROTOCOL = HERE.parent / "harness" / "reports" / "audited_md5_fused_20260904" / "protocol.json"
SPLITS = HERE / "tuning" / "splits.json"
# The regime comes from the environment so that the pool's worker processes see it too (X26 adds md5).
REGIME = os.environ.get("SMX_DTB_REGIME", "plain")
SEEDS = HERE / "reference" / "seeds" / REGIME
LAYER1 = HERE / "tuning" / "results" / f"layer1_{REGIME}"
OUT = HERE / "results" / REGIME
# X26: class terms (pattern classes, X22, and rule-based semantic types, X23) on the dev scenarios,
# plain only: q + beta D + gamma [same class]. Classes are computed once per scenario (class_cache).
CLASS_BETAS, CLASS_GAMMAS = (0.2, 1.0, 5.0), (0.1, 0.2, 0.5)
CLASS_CACHE = OUT / "dtype_baseline_classes.json"
CONTROLS = {"constant", "random", "oracle", "oracle_noise025", "oracle_noise050", "oracle_erase050",
            "oracle_decoy100", "embedding_random", "rema_shuffled", "semstruct_lm_only"}
BETAS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 1000.0)   # 1000: datatype first, then q
MIN_FAMILY = 5


def sf_out(seed: str, ds: str, sc: str) -> Path | None:
    d = LAYER1 / f"{seed}__melnik" / "merged" / ds / sc / "_outputs"
    return one(d / "MatchingStepLine2") if d.is_dir() else None


def residual(S: np.ndarray, q: np.ndarray, blocks: np.ndarray) -> float:
    """Largest |S - (a q + b_block)| over the cells, relative to S's range; least squares fit."""
    _, b = np.unique(blocks, return_inverse=True)
    X = np.zeros((S.size, b.max() + 2))
    X[:, 0] = q.ravel()
    X[np.arange(S.size), 1 + b] = 1.0
    coef, *_ = np.linalg.lstsq(X, S.ravel(), rcond=None)
    rng = float(S.max() - S.min())
    return float(np.abs(X @ coef - S.ravel()).max() / rng) if rng > 0 else 0.0


def class_job(case) -> tuple[str, dict]:
    sys.path.insert(0, str(HERE / "layer3b"))
    from f7 import column_classes
    from f13 import column_types
    ds, sc = case
    out = {}
    for side in ("source", "target"):
        p = CORPUS[REGIME] / ds / sc / side / f"{side}.csv"
        h, shp = column_classes(p)
        _, sem = column_types(p)
        out[side] = {"f7": dict(zip(h, shp)), "f13": dict(zip(h, sem))}
    return f"{ds}/{sc}", out


def class_eq(cls: dict, labels, fam: str) -> np.ndarray:
    """eq[i, j] = 1 if source column i and target column j have the same class; labels are the matrix's
    row and column labels ("source.<column>", "target.<column>")."""
    src = [cls["source"][fam].get(l.split(".", 1)[1]) for l in labels[0]]
    tgt = [cls["target"][fam].get(l.split(".", 1)[1]) for l in labels[1]]
    if None in src or None in tgt:
        raise KeyError("class missing for a column")
    return (np.array(src)[:, None] == np.array(tgt)[None, :]).astype(float)


def score_seed(job) -> list[dict]:
    seed, cases = job
    class_cache = json.loads(CLASS_CACHE.read_text(encoding="utf-8")) if CLASS_CACHE.is_file() else {}
    out = []
    for ds, sc, split in cases:
        f_q, f_d, f_s = SEEDS / seed / ds / sc / "seed.csv", SEEDS / "datatype" / ds / sc / "seed.csv", sf_out(seed, ds, sc)
        f_c = sf_out("constant", ds, sc)
        gt_file = ground_truth_file(CORPUS[REGIME], ds, sc)
        if not (f_q.is_file() and f_d.is_file() and f_s and gt_file):
            continue
        gt = read_matrix(gt_file)[0] > 0.5
        (q, *labels), D, S = read_matrix(f_q), read_matrix(f_d)[0], read_matrix(f_s)[0]
        if not (q.shape == D.shape == S.shape == gt.shape) or not (np.isfinite(q).all() and np.isfinite(S).all()):
            continue
        nan = int((~np.isfinite(D)).sum())
        D = np.nan_to_num(D, nan=0.0)
        q_raw = q                     # SF received q as is; the residual fit uses it
        q, S = np.clip(q, 0.0, None), np.clip(S, 0.0, None)
        perms = permutations(gt.shape, f"{ds}/{sc}")
        row = {"seed": seed, "dataset": ds, "scenario": sc, "split": split, "d_nan_cells": nan}
        mats = {"q": q, "sf": S, **{f"b{b:g}": q + b * D for b in BETAS}}
        cls = class_cache.get(f"{ds}/{sc}")
        if cls is not None:
            for fam in ("f7", "f13"):
                E = class_eq(cls, labels, fam)
                for b in CLASS_BETAS:
                    for g in CLASS_GAMMAS:
                        mats[f"c{fam}_{b:g}_{g:g}"] = q + b * D + g * E
        for name, M in mats.items():
            mm = all_metrics(M, gt, perms)
            row[f"{name}_R_exp"], row[f"{name}_F1_Hr"] = mm["R_exp"], mm["F1_Hr"]
        if f_c is not None:
            C = read_matrix(f_c)[0]
            ok = C.shape == S.shape and np.isfinite(C).all()
            row["res_const"] = residual(S, q_raw, classes(C)) if ok else ""
            row["blocks_const"] = len(np.unique(classes(C))) if ok else ""
            row["cells"] = S.size
        row["res_dtype"] = residual(S, q_raw, classes(np.where(np.isfinite(read_matrix(f_d)[0]), D, -1.0)))
        out.append(row)
    return out


def score(workers: int) -> None:
    cases = json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]
    sp = json.loads(SPLITS.read_text(encoding="utf-8"))
    screen, dev = set(sp["screen"]), set(sp["dev"])
    todo = [(c["dataset"], c["scenario"], "screen" if c["key"] in screen else "dev")
            for c in cases if c["key"] in dev]          # dev includes the screen; lockbox left out
    seeds = sorted(p.name.split("__")[0] for p in LAYER1.glob("*__melnik") if p.name.split("__")[0] not in CONTROLS)
    if REGIME == "plain" and not CLASS_CACHE.is_file():   # recoded values carry no shapes or types
        with Pool(workers) as pool:
            CLASS_CACHE.write_text(json.dumps(dict(pool.map(class_job, [(d, s) for d, s, _ in todo]))), encoding="utf-8")
        print(f"classes for {len(todo)} scenarios -> {CLASS_CACHE}", flush=True)
    with Pool(workers) as pool:
        rows = [r for part in pool.imap_unordered(score_seed, [(s, todo) for s in seeds]) for r in part]
    rows.sort(key=lambda r: (r["seed"], r["dataset"], r["scenario"]))
    keys = list(rows[0])
    for r in rows:
        keys += [k for k in r if k not in keys]
    with (OUT / "dtype_baseline_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows, {len(seeds)} matchers")


def paper_matchers() -> set[str]:
    """RQ1's matcher set: gen_figures.py's points (in Table 1, not invalid, not a control group, and
    semstruct_lm_only folded into sbert_e5)."""
    sys.path.insert(0, str(HERE.parent / "latex" / "sf-evaluation"))
    import gen_figures as g
    return {s for s in g.META if s not in g.INVALID and g.group_of(s) != "ctrl" and s != "semstruct_lm_only"}


def summary() -> None:
    fam = {(c["dataset"], c["scenario"]): c["family"] for c in json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]}
    rows = list(csv.DictReader(open(OUT / "dtype_baseline_scores.csv", encoding="utf-8")))
    # RQ1's matchers, complete on every screen and dev scenario (as RQ1 requires all 382)
    n = defaultdict(int)
    for r in rows:
        n[r["seed"]] += 1
    full = {s for s in paper_matchers() if n[s] == max(n.values())}
    rows = [r for r in rows if r["seed"] in full]
    by = defaultdict(list)
    for r in rows:
        by[(r["seed"], r["split"])].append(r)
    seeds = sorted(full)

    def macro(rs, f):
        acc = defaultdict(list)
        for r in rs:
            acc[fam[(r["dataset"], r["scenario"])]].append(f(r))
        big = [v for v in acc.values() if len(v) >= MIN_FAMILY]
        return sum(sum(v) / len(v) for v in big) / len(big)

    def J(name):
        return lambda r: (float(r[f"{name}_R_exp"]) + float(r[f"{name}_F1_Hr"])) / 2

    def gain(seed, split, name, metric="J"):
        f = J(name) if metric == "J" else (lambda r: float(r[f"{name}_{metric}"]))
        g = J("q") if metric == "J" else (lambda r: float(r[f"q_{metric}"]))
        return macro(by[(seed, split)], f) - macro(by[(seed, split)], g)

    names = [f"b{b:g}" for b in BETAS]
    # selection on the screen
    own = {s: max(names, key=lambda n: gain(s, "screen", n)) for s in seeds}
    uni = max(names, key=lambda n: np.mean([gain(s, "screen", n) for s in seeds]))
    res = defaultdict(list)
    for s in seeds:
        res["sf"].append([gain(s, "dev", "sf", m) for m in ("J", "R_exp", "F1_Hr")])
        res["own"].append([gain(s, "dev", own[s], m) for m in ("J", "R_exp", "F1_Hr")])
        res["uni"].append([gain(s, "dev", uni, m) for m in ("J", "R_exp", "F1_Hr")])
    sf, ow, un = (np.array(res[k]) for k in ("sf", "own", "uni"))
    nd = len({(r["dataset"], r["scenario"]) for r in rows if r["split"] == "dev"})
    lines = [f"# Additive datatype baseline against the reference SF ({REGIME}; exploratory, X19, X26)", "",
             f"{len(seeds)} initial matchers; beta chosen on the screen scenarios, evaluated on {nd} dev scenarios outside "
             f"the screen (lockbox not read). Macro over families (>= 5 scenarios), then the mean over matchers. "
             f"D's NaN cells set to 0 in {sum(int(r['d_nan_cells']) > 0 for r in rows if r['seed'] == seeds[0])} "
             "scenarios.", "",
             f"Universal beta: {uni[1:]}. Per-matcher beta: " + ", ".join(
                 f"{n[1:]} ({sum(own[s] == n for s in seeds)})" for n in names if any(own[s] == n for s in seeds)), "",
             "| method (gain over q) | J_tf | R_exp | F1_Hr | matchers with J gain > 0 |", "|---|---|---|---|---|"]
    for label, A in (("SF reference (melnik)", sf), ("q + beta D, per-matcher beta", ow), ("q + beta D, universal beta", un)):
        lines.append(f"| {label} | {A[:, 0].mean():+.4f} | {A[:, 1].mean():+.4f} | {A[:, 2].mean():+.4f} | "
                     f"{int((A[:, 0] > 1e-4).sum())} / {len(seeds)} |")
    class_rows = []
    for cfam, label in (("f7", "pattern classes"), ("f13", "semantic types")):   # X26, plain only
        cn = [k[:-6] for k in rows[0] if k.startswith(f"c{cfam}_") and k.endswith("_R_exp")]
        if not cn:
            continue
        best = max(cn, key=lambda n: np.mean([gain(s, "screen", n) for s in seeds]))
        A = np.array([[gain(s, "dev", best, m) for m in ("J", "R_exp", "F1_Hr")] for s in seeds])
        ahead = int((A[:, 0] - sf[:, 0] > 1e-4).sum())
        lines.append(f"| q + beta D + gamma [same class], {label} (beta, gamma = {', '.join(best.split('_')[1:])}) | "
                     f"{A[:, 0].mean():+.4f} | {A[:, 1].mean():+.4f} | {A[:, 2].mean():+.4f} | "
                     f"{int((A[:, 0] > 1e-4).sum())} / {len(seeds)} |")
        class_rows.append(f"With {label}: ahead of SF for {ahead} of {len(seeds)} matchers; minus q + beta D (universal) "
                          f"{(A[:, 0] - un[:, 0]).mean():+.4f}.")
    lines += [""] + class_rows if class_rows else []
    d_own, d_uni = sf[:, 0] - ow[:, 0], sf[:, 0] - un[:, 0]
    lines += ["", f"SF minus baseline (J_tf): per-matcher beta mean {d_own.mean():+.4f}, SF ahead for "
              f"{int((d_own > 1e-4).sum())}, baseline ahead for {int((d_own < -1e-4).sum())}; universal beta mean "
              f"{d_uni.mean():+.4f}, SF ahead {int((d_uni > 1e-4).sum())}, baseline ahead {int((d_uni < -1e-4).sum())}.", ""]
    for col, label in (("res_const", "blocks from SF on the constant matcher (Amendment 1)"),
                       ("res_dtype", "blocks from the datatype matcher")):
        v = np.array([float(r[col]) for r in rows if r.get(col) not in (None, "")])
        if v.size:
            lines.append(f"Residual of SF = a q + b_block, {label}: {v.size} fits; max relative residual median "
                         f"{np.median(v):.2e}, 95th pct {np.percentile(v, 95):.2e}, max {v.max():.2e}; "
                         f"share below 1e-4: {(v < 1e-4).mean():.3f}.")
    bl = np.array([float(r["blocks_const"]) / float(r["cells"]) for r in rows if r.get("blocks_const") not in (None, "")])
    nb = np.array([float(r["blocks_const"]) for r in rows if r.get("blocks_const") not in (None, "")])
    if nb.size:
        lines.append(f"Blocks per fit (constant partition): median {np.median(nb):.0f}, max {nb.max():.0f}; blocks per cell "
                     f"median {np.median(bl):.3f}, max {bl.max():.3f} (a fit has blocks + 1 parameters).")
    (OUT / "dtype_baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if not a.summary:
        score(a.workers)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
