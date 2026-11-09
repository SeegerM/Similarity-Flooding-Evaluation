"""Review check 4 (EXPLORATORY, after the ICDE review of the compressed draft, 2026-10-26): statistics for the
corpus-property associations of the paper's "What the corpus properties explain" paragraph.

(a) Multiplicity. The X78 analysis (layer3b/x78_analysis.py, not modified) tests every effect x property pair with a
    pooled within-corpus Spearman rho (ranks centred within corpus x regime) and a bootstrap over source_id clusters, and
    lists those with |rho| >= 0.15 whose interval excludes 0. Here the same statistic and bootstrap (same seed, 2,000
    resamples) give a two-sided bootstrap p-value per pair, p = 2 min(P*(rho* <= 0), P*(rho* >= 0)), and a
    Benjamini-Hochberg correction over all tested pairs (q = 0.05, 0.10).
(b) The construction claim between corpora: Spearman, over corpus x regime groups, between the group's median value
    overlap of true pairs and its mean datatype-term lead over SF (E1), with a Monte Carlo permutation p-value
    (100,000 permutations of the group labels). Few groups, so this is the weakest form of evidence; reported as such.
Output: tuning/results/review_checks/corpus_assoc.md

    python review_checks/corpus_assoc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "layer3b"))
import x78_analysis as xa  # noqa: E402

RES = STUDY / "results"
OUT = STUDY / "tuning" / "results" / "review_checks"
# rerun 549 (2026-11-02): SMX_VALENTINE=549 reads x78_analysis's 549 inputs and writes next to the 549 figures
if xa.IO != RES:
    OUT = STUDY.parent / "latex" / "sf-evaluation-549" / "review_checks"


def pooled_with_bs(df, e, p, rng):
    """xa.pooled's statistic and bootstrap, also returning the bootstrap draws."""
    d = df[[e, p, "grp", "source_id"]].dropna().reset_index(drop=True)
    if len(d) < 20:
        return None
    ev, pv = d[e].to_numpy(float), d[p].to_numpy(float)
    gidx = [np.where(d.grp.to_numpy() == g)[0] for g in d.grp.unique()]
    src = [[ii[d.source_id.to_numpy()[ii] == s_] for s_ in np.unique(d.source_id.to_numpy()[ii])] for ii in gidx]

    def corr(idx_by_group):
        flat = np.concatenate(idx_by_group)
        pos, start = [], 0
        for ii in idx_by_group:
            pos.append(np.arange(start, start + len(ii)))
            start += len(ii)
        a = xa._centred_ranks(ev[flat], pos)
        b = xa._centred_ranks(pv[flat], pos)
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    obs = corr(gidx)
    bs = np.array([corr([np.concatenate([cl[k] for k in rng.integers(0, len(cl), len(cl))]) for cl in src])
                   for _ in range(xa.B)])
    bs = bs[np.isfinite(bs)]
    if not np.isfinite(obs) or len(bs) < xa.B // 2:
        return None
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return obs, np.percentile(bs, 2.5), np.percentile(bs, 97.5), p, len(d), d.grp.nunique()


def bh(p: np.ndarray, q: float) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    ok = p[order] <= thresh
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    out = np.zeros(n, bool)
    out[order[:k]] = True
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    eff = pd.read_csv(xa.IO / "benchmark_effects.csv")
    eff["corpus"] = np.where(eff.corpus == "valentine", "valentine_" + eff.regime, eff.corpus)
    prop = pd.read_csv(xa.IO / "benchmark_properties.csv")
    df = eff.merge(prop, on=["corpus", "regime", "dataset", "scenario"], how="inner")
    df["value_gap"] = df.ovl_true - df.ovl_non
    df["dist_gap"] = df.dist_true - df.dist_non
    df["name_gap"] = df.name_true - df.name_non
    df["log_cols"] = np.log(df.src_cols + df.tgt_cols)
    df["grp"] = df.corpus + "/" + df.regime
    big = [g for g, n in df.groupby("grp").size().items() if n >= 15]
    dfb = df[df.grp.isin(big)]
    rng = np.random.default_rng(xa.SEED)
    rows = []
    for e, en in xa.EFFECTS.items():
        if e not in df:
            continue
        for p in xa.PROPS:
            r = pooled_with_bs(dfb, e, p, rng)
            if r is None:
                continue
            rho, lo, hi, pv, n, k = r
            rows.append({"effect": en, "property": p, "rho": rho, "lo": lo, "hi": hi, "p": pv, "n": n, "groups": k,
                         "listed": abs(rho) >= 0.15 and (lo > 0 or hi < 0)})
    t = pd.DataFrame(rows)
    t["bh05"] = bh(t.p.to_numpy(), 0.05)
    t["bh10"] = bh(t.p.to_numpy(), 0.10)
    t.to_csv(OUT / "corpus_assoc_grid.csv", index=False)
    lines = ["# Corpus-property associations: multiplicity and a between-corpus test (EXPLORATORY review check)", "",
             f"## (a) {len(t)} effect x property pairs tested; {int(t.listed.sum())} listed in the X78 report "
             f"(|rho| >= 0.15, interval excludes 0)", "",
             f"Survive Benjamini-Hochberg at q = 0.05: {int(t.bh05.sum())}; at q = 0.10: {int(t.bh10.sum())}. "
             f"Bootstrap p-values (two-sided, cluster bootstrap, {xa.B} resamples) are coarse: the smallest is "
             f"{1 / xa.B * 2:.4f}.", "",
             "| effect | property | rho [95% CI] | p | listed | BH 0.05 | BH 0.10 |", "|---|---|---|---|---|---|---|"]
    for _, r in t[t.listed | t.bh10].sort_values(["effect", "p"]).iterrows():
        lines.append(f"| {r.effect} | {r.property} | {r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] | {r.p:.4f} | "
                     f"{'yes' if r.listed else ''} | {'yes' if r.bh05 else ''} | {'yes' if r.bh10 else ''} |")
    # (b) between corpora: E1 lead vs value overlap of true pairs
    g = df.groupby("grp").agg(e1=("E1_datatype_vs_SF", "mean"), ovl=("ovl_true", "median"),
                              n=("E1_datatype_vs_SF", "count")).dropna()
    g = g[g.n > 0]
    x, y = g.ovl.rank().to_numpy(), g.e1.rank().to_numpy()
    rho = float(np.corrcoef(x, y)[0, 1])
    prng = np.random.default_rng(xa.SEED)
    perm = np.array([np.corrcoef(x, prng.permutation(y))[0, 1] for _ in range(100_000)])
    p = (np.sum(np.abs(perm) >= abs(rho)) + 1) / (len(perm) + 1)
    lines += ["", f"## (b) Between corpora: datatype-term lead over SF (E1, mean) vs value overlap of true pairs (median)", "",
              f"{len(g)} corpus x regime groups: Spearman rho = {rho:+.2f}, two-sided permutation p = {p:.4f} "
              f"(100,000 permutations).", "", "| group | median value overlap of true pairs | mean E1 (datatype - SF, J) | scenarios |",
              "|---|---|---|---|"]
    for k, r in g.sort_values("ovl").iterrows():
        lines.append(f"| {k} | {r.ovl:.2f} | {r.e1:+.4f} | {int(r.n)} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "corpus_assoc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
