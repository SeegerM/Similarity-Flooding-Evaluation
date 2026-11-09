"""X78 part (b), step 2 (EXPLORATORY; design and predictions in EXPLORATORY.md X78, written before this ran): which
effects depend on which benchmark properties?

Inputs: results/benchmark_effects.csv (x78_effects.py) and results/benchmark_properties.csv (X77), joined on
(corpus, regime, dataset, scenario); Valentine's effects carry corpus "valentine" and are mapped to valentine_<regime>.
(1) within each corpus × regime with >= 15 scenarios: Spearman ρ, 95% CI by a bootstrap over source_id clusters;
(2) pooled within-corpus ρ: ranks centred within corpus × regime, one correlation, same cluster bootstrap (stratified);
(3) between corpora: Spearman of corpus × regime effect means against property medians (descriptive).
2,000 resamples, seed 20261006.

    python layer3b/x78_analysis.py   -> results/benchmark_effects_report.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent.parent
RES = STUDY / "results"
B, SEED = 2000, 20261006
# rerun 549 (2026-11-02): SMX_VALENTINE=549 reads and writes results/x78_549/ (x78_effects.py and
# reference/benchmark_properties_549.py under the switch)
import os  # noqa: E402
IO = RES / "x78_549" if os.environ.get("SMX_VALENTINE", "382") == "549" else RES
EFFECTS = {"E1_datatype_vs_SF": "E1 datatype − SF", "E2_role_vs_key": "E2 role − key",
           "E3_fd_vs_melnik_R": "E3 fd − melnik (R_exp)", "E3_fd_vs_melnik_J": "E3 fd − melnik (J)",
           "E4_SF_vs_q": "E4 SF − q", "E5_table_vs_datatype": "E5 table − datatype"}
PROPS = ["diag_share", "true_share", "size_ratio", "log_cols", "num_share", "dtype_entropy", "value_gap", "ovl_true",
         "dist_gap", "name_gap", "label_hit", "fd_per_col", "key_share", "ind_per_col", "src_tables"]


def rank_centre(x, g):
    r = x.groupby(g).rank(pct=True)
    return r - r.groupby(g).transform("mean")


def _centred_ranks(v, groups_idx):
    out = np.empty(len(v))
    for ii in groups_idx:
        r = pd.Series(v[ii]).rank(pct=True).to_numpy()
        out[ii] = r - r.mean()
    return out


def pooled(df, e, p, rng):
    """Pooled within-group Spearman (ranks centred within corpus × regime), with a bootstrap over source_id
    clusters stratified by group. Vectorised (2026-10-06): the same statistic as the first draft, faster."""
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
            pos.append(np.arange(start, start + len(ii))); start += len(ii)
        a = _centred_ranks(ev[flat], pos); b = _centred_ranks(pv[flat], pos)
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    obs = corr(gidx)
    bs = []
    for _ in range(B):
        bs.append(corr([np.concatenate([cl[k] for k in rng.integers(0, len(cl), len(cl))]) for cl in src]))
    bs = np.array([b for b in bs if np.isfinite(b)])
    if not np.isfinite(obs) or len(bs) < B // 2:      # a property constant within the groups: no correlation defined
        return None
    return obs, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d), d.grp.nunique()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    eff = pd.read_csv(IO / "benchmark_effects.csv")
    eff["corpus"] = np.where(eff.corpus == "valentine", "valentine_" + eff.regime, eff.corpus)
    prop = pd.read_csv(IO / "benchmark_properties.csv")
    df = eff.merge(prop, on=["corpus", "regime", "dataset", "scenario"], how="inner")
    unmatched = eff.merge(prop, on=["corpus", "regime", "dataset", "scenario"], how="left", indicator=True)
    miss = unmatched[unmatched._merge == "left_only"].groupby(["corpus", "regime"]).size()
    df["value_gap"] = df.ovl_true - df.ovl_non
    df["dist_gap"] = df.dist_true - df.dist_non
    df["name_gap"] = df.name_true - df.name_non
    df["log_cols"] = np.log(df.src_cols + df.tgt_cols)
    df["grp"] = df.corpus + "/" + df.regime
    rng = np.random.default_rng(SEED)
    out = ["# X78 part (b): which effects depend on benchmark properties? (EXPLORATORY)", "",
           f"- Joined {len(df)} scenarios; unmatched effect rows per corpus: {miss.to_dict() if len(miss) else 'none'}.",
           f"- Scenarios per corpus × regime: {df.groupby('grp').size().to_dict()}.",
           "- Within-corpus groups for (1)/(2): ≥ 15 scenarios. CIs: bootstrap over source_id clusters within each group.",
           ""]
    big = [g for g, n in df.groupby("grp").size().items() if n >= 15]
    dfb = df[df.grp.isin(big)]
    # (2) pooled within-corpus
    out += ["## (2) Pooled within-corpus Spearman ρ (ranks centred within corpus × regime)", "",
            "Only |ρ| ≥ 0.15 with the interval excluding 0 are listed; the full grid is in benchmark_effects_within.csv.", "",
            "| effect | property | ρ [95% CI] | scenarios | groups |", "|---|---|---|---|---|"]
    rows = []
    for e, en in EFFECTS.items():
        if e not in df:
            continue
        for p in PROPS:
            r = pooled(dfb, e, p, rng)
            if r is None:
                continue
            rho, lo, hi, n, k = r
            rows.append({"effect": en, "property": p, "rho": rho, "lo": lo, "hi": hi, "n": n, "groups": k})
            if abs(rho) >= 0.15 and (lo > 0 or hi < 0):
                out.append(f"| {en} | {p} | {rho:+.2f} [{lo:+.2f}, {hi:+.2f}] | {n} | {k} |")
    pd.DataFrame(rows).to_csv(RES / "benchmark_effects_within.csv", index=False)
    # (1) per group, for the strongest pooled associations
    out += ["", "## (1) Per corpus × regime Spearman ρ for the pooled associations above (descriptive signs)", ""]
    sel = [r for r in rows if abs(r["rho"]) >= 0.15 and (r["lo"] > 0 or r["hi"] < 0)]
    inv = {v: k for k, v in EFFECTS.items()}
    for r in sel:
        e, p = inv[r["effect"]], r["property"]
        per = []
        for g, v in dfb.groupby("grp"):
            v = v[[e, p]].dropna()
            if len(v) >= 15 and v[e].std() > 0 and v[p].std() > 0:
                per.append(f"{g} {v[e].rank().corr(v[p].rank()):+.2f}")
        out.append(f"- {r['effect']} ~ {p}: " + "; ".join(per))
    # (3) between corpora
    gm = df.groupby("grp")
    means = gm[[e for e in EFFECTS if e in df]].mean()
    meds = gm[PROPS].median()
    out += ["", "## (3) Between corpora: corpus × regime effect means and property medians (descriptive, n = groups)", "",
            "| group | n | " + " | ".join(EFFECTS[e] for e in means.columns) + " | value_gap | fd_per_col | diag_share | "
            "dtype_entropy | name_gap |", "|---|---|" + "---|" * (len(means.columns) + 5)]
    for g in means.index:
        out.append(f"| {g} | {gm.size()[g]} | " + " | ".join(f"{means.loc[g, e]:+.3f}" if pd.notna(means.loc[g, e]) else "–"
                                                       for e in means.columns)
                   + " | " + " | ".join(f"{meds.loc[g, p]:.2f}" if pd.notna(meds.loc[g, p]) else "–"
                                       for p in ("value_gap", "fd_per_col", "diag_share", "dtype_entropy", "name_gap")) + " |")
    out += ["", "Between-corpus Spearman (effect mean ~ property median; descriptive):", ""]
    for e in means.columns:
        cs = []
        for p in PROPS:
            x = pd.concat([means[e], meds[p]], axis=1).dropna()
            if len(x) >= 5 and x.iloc[:, 1].std() > 0:
                cs.append((p, x.iloc[:, 0].rank().corr(x.iloc[:, 1].rank()), len(x)))
        cs.sort(key=lambda t: -abs(t[1]))
        out.append(f"- {EFFECTS[e]}: " + "; ".join(f"{p} {c:+.2f} (n {n})" for p, c, n in cs[:4]))
    (IO / "benchmark_effects_report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
