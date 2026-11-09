"""X58 (EXPLORATORY, offline, design and predictions before computing): does the initial
matcher's family change SF's effect beyond the matcher's strength?

Per non-control matcher with all 382 scenarios (layer 1): Δ = J_tf(arm) − J_tf(q), family macro, for arm melnik (as
published) and fd; Q = J_tf(q); family and inputs from latex/sf-evaluation/gen_tables.py (read only); tie share = mean
over scenarios of the share of q's cells whose value repeats in the same row (reference/seeds/<regime>).
Models: Δ ~ Q + Q², + family, + input flags, + tie share, + tie share + family; R², increments, permutation p-values
(10,000 label permutations), per-family mean residuals of Δ ~ Q + Q².

    python layer3b/x58_family.py --regime plain|md5
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STUDY))
from score_r7 import is_control  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

RES = STUDY / "tuning" / "results"
sys.path.insert(0, str(STUDY / "reference"))
import valentine549 as V  # noqa: E402  (rerun 549: SMX_VALENTINE=549 reads the assembled layer-1 rows; unused otherwise)
PAPER = STUDY.parent / "latex" / "sf-evaluation"
NPERM, SEED = 10_000, 20261005


def tie_share(job):
    regime, seed, cases = job
    vals = []
    for ds, sc in cases:
        f = V.seed_file(regime, seed, ds, sc) if V.V549 else STUDY / "reference" / "seeds" / regime / seed / ds / sc / "seed.csv"
        if not f.is_file():
            continue
        q = read_matrix(f)[0]
        if not np.isfinite(q).all():
            continue
        tied = 0
        for row in np.round(q, 9):
            _, inv, cnt = np.unique(row, return_inverse=True, return_counts=True)
            tied += int((cnt[inv] > 1).sum())
        vals.append(tied / q.size)
    return seed, float(np.mean(vals)) if vals else float("nan")


def design(cols):
    return np.column_stack([np.ones(len(cols[0]))] + list(cols))


def r2(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum(), res


def dummies(labels):
    lv = sorted(set(labels))
    return [np.array([l == v for l in labels], dtype=float) for v in lv[1:]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5"), required=True)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if V.V549 and any(V.arm_status(f"layer1_{a.regime}", x) != "assembled" for x in ("melnik", "fd")):
        print(f"x58 {a.regime} 549: pending (layer-1 melnik/fd not assembled); nothing written")
        return 0
    d = pd.read_csv(RES / f"layer1_{V.vreg(a.regime)}_scores" / "scores.csv",
                    usecols=["seed", "arm", "dataset", "scenario", "family", "seed_R_exp", "seed_F1_Hr", "sf_R_exp",
                             "sf_F1_Hr"])
    d = d[~d.seed.map(is_control) & d.arm.isin(["melnik", "fd"])].copy()
    d["J"], d["q"] = (d.sf_R_exp + d.sf_F1_Hr) / 2, (d.seed_R_exp + d.seed_F1_Hr) / 2
    n_scen = d.drop_duplicates(["dataset", "scenario"]).shape[0]
    full = d.groupby(["seed", "arm"]).scenario.count().unstack()
    keep = full[(full == n_scen).all(axis=1)].index
    d = d[d.seed.isin(keep)]

    big_fams = d.drop_duplicates(["dataset", "scenario"]).family.value_counts()
    big_fams = set(big_fams[big_fams >= 5].index)       # the study's macro rule (fixed 2026-10-06)

    def mac(col, arm):
        x = d[(d.arm == arm) & d.family.isin(big_fams)].groupby(["seed", "family"])[col].mean().groupby("seed").mean()
        return x
    Q = mac("q", "melnik")
    D = {arm: mac("J", arm) - Q for arm in ("melnik", "fd")}
    sys.path.insert(0, str(PAPER))
    import gen_tables
    meta = gen_tables.META
    seeds = sorted(Q.index)
    fam = [meta[s][0] if s in meta else "other" for s in seeds]
    big = {f for f in set(fam) if fam.count(f) >= 3}
    fam = [f if f in big else "other" for f in fam]
    inputs = [meta[s][2] if s in meta else "" for s in seeds]
    flags = [np.array([("L" in x.split(",")) for x in inputs], float), np.array([("I" in x.split(",")) for x in inputs], float),
             np.array([("D" in x.split(",")) for x in inputs], float), np.array([("X" in x.split(",")) for x in inputs], float)]
    cases = sorted(set(zip(d.dataset, d.scenario)))
    with Pool(10) as pool:
        ties = dict(pool.map(tie_share, [(a.regime, s, cases) for s in seeds]))
    T = np.array([ties[s] for s in seeds])
    q = Q.loc[seeds].to_numpy()
    qc = q - q.mean()
    base_cols = [qc, qc ** 2]
    rng = np.random.default_rng(SEED)
    out = [f"# X58 (EXPLORATORY): matcher family and SF's gain, regime {a.regime}", "",
           f"- {len(seeds)} non-control matchers with all {n_scen} scenarios; families (≥ 3 matchers, others pooled): "
           + ", ".join(f"{f} {fam.count(f)}" for f in sorted(set(fam))) + ".",
           f"- Tie share of q (mean share of cells whose value repeats in its row): median {np.median(T):.2f}, "
           f"range {np.min(T):.2f}–{np.max(T):.2f}.", ""]
    for arm in ("melnik", "fd"):
        y = D[arm].loc[seeds].to_numpy()
        r_q, res_q = r2(y, design(base_cols))
        r_f, _ = r2(y, design(base_cols + dummies(fam)))
        r_i, _ = r2(y, design(base_cols + flags))
        r_t, _ = r2(y, design(base_cols + [T]))
        r_tf, _ = r2(y, design(base_cols + [T] + dummies(fam)))

        def perm(extra_fn, r_base, r_obs, with_t=False):
            hits = 0
            for _ in range(NPERM):
                p = rng.permutation(len(seeds))
                cols = base_cols + ([T] if with_t else []) + extra_fn(p)
                hits += r2(y, design(cols))[0] - r_base >= r_obs - r_base - 1e-12
            return (hits + 1) / (NPERM + 1)
        p_f = perm(lambda p: dummies([fam[i] for i in p]), r_q, r_f)
        p_i = perm(lambda p: [f[p] for f in flags], r_q, r_i)
        p_tf = perm(lambda p: dummies([fam[i] for i in p]), r_t, r_tf, with_t=True)
        rho_t = float(pd.Series(res_q).corr(pd.Series(T), method="spearman"))
        out += [f"## Arm `{arm}`: Δ = J_tf({arm}) − J_tf(q)", "",
                f"- Mean Δ {y.mean():+.4f}; Spearman ρ(Δ, Q) {pd.Series(y).corr(pd.Series(q), method='spearman'):+.2f}.",
                f"- R²: quality (Q + Q²) {r_q:.3f}; + family {r_f:.3f} (Δ {r_f - r_q:+.3f}, permutation p {p_f:.4f}); "
                f"+ input flags {r_i:.3f} (Δ {r_i - r_q:+.3f}, p {p_i:.4f}); + tie share {r_t:.3f} (Δ {r_t - r_q:+.3f}); "
                f"+ tie share + family {r_tf:.3f} (family beyond Q + ties Δ {r_tf - r_t:+.3f}, p {p_tf:.4f}).",
                f"- Spearman ρ(residual of Δ ~ Q, tie share) {rho_t:+.2f}.", "",
                "| family | matchers | mean Q | mean Δ | mean residual (Δ − quality fit) | SE |", "|---|---|---|---|---|---|"]
        for f in sorted(set(fam), key=lambda f: -np.mean([res_q[i] for i in range(len(seeds)) if fam[i] == f])):
            idx = [i for i in range(len(seeds)) if fam[i] == f]
            rr = res_q[idx]
            se = rr.std(ddof=1) / np.sqrt(len(rr)) if len(rr) > 1 else float("nan")
            out.append(f"| {f} | {len(idx)} | {q[idx].mean():.3f} | {y[idx].mean():+.4f} | {rr.mean():+.4f} | {se:.4f} |")
        out.append("")
    rep = RES / f"x58_{V.vreg(a.regime)}"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
