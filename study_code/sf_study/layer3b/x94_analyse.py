"""X94 stage 1, analysis (EXPLORATORY; EXPLORATORY.md X94, design and predictions written 2026-10-08 before computing).
Reads tuning/results/x94/cells.csv.gz from x94_stage1.py --compute.

Q1 headroom: oracle per (matcher, scenario), oracle per scenario (shared by its matchers), best global combination per corpus
(chosen on the corpus), frozen_V (best on the 60 Valentine screen scenarios); SF(t2) = rule (a); SF converged.
Q3: rule (a) (SF-implied c/a, exact replay of two updates) and rule (b) (lambda = kappa SD(q)/SD(L~), kappa and w_D frozen on
the screen), against frozen_V with X86 cluster intervals.
Q2: depth-3 regression trees on ground-truth-free features predict the oracle grid indices; leave one corpus out over
{Valentine, GOBY, HDXSM, GDC-SM}; recovered share = (pred − frozen(train)) / (oracle − frozen(train)).

    python layer3b/x94_analyse.py   -> results/x94/stage1.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
for p in (STUDY.parent / "harness", STUDY / "tuning", STUDY, HERE):
    sys.path.insert(0, str(p))
import x94_stage1 as s1  # noqa: E402

FEATS = ["n_src", "n_tgt", "n_blocks", "dtype_entropy", "sd_L", "name_levenshtein_mean", "name_levenshtein_sd",
         "name_prefix_suffix_mean", "name_prefix_suffix_sd", "name_based", "tie_all", "tie_row", "q_entropy", "q_gap", "q_sd",
         "q_mean", "sd_ratio"]
SCORED = ["valentine_plain", "goby", "hdxsm", "gdcsm", "madi_indep", "madi_products"]
INDEP = ["goby", "hdxsm", "gdcsm"]
GROUPS = ["valentine_plain", "goby", "hdxsm", "gdcsm"]


def macro(x: pd.Series, df: pd.DataFrame) -> float:
    """mean over matchers per scenario, then over scenarios"""
    return float(x.groupby([df.dataset, df.scenario]).mean().mean())


def cluster_ci(v: pd.Series, df: pd.DataFrame, corpus: str, rng, B=10_000):
    import x86_cluster_ci as x86
    per = v.groupby([df.dataset, df.scenario]).mean()
    keys = list(per.index)
    if corpus == "valentine_plain":
        props = pd.read_csv(STUDY / "results" / "benchmark_properties.csv")
        src = props[props.corpus == "valentine_plain"].set_index(["dataset", "scenario"]).source_id.to_dict()
        cl = np.array([src.get(k, SRC_EXTRA.get(k, k[0])) for k in keys])
    else:
        kk = [k[1] if corpus.startswith("madi") else f"{k[0]}/{k[1]}" for k in keys]
        cmap, _ = x86.cluster_map(corpus, kk)
        cl = np.array([cmap[k] for k in kk])
    ids = np.unique(cl)
    if len(ids) < 5:
        return np.nan, np.nan, len(ids)
    x = per.to_numpy()
    sums = np.array([x[cl == i].sum() for i in ids])
    cnt = np.array([(cl == i).sum() for i in ids])
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    lo, hi = np.percentile(sums[idx].sum(axis=1) / cnt[idx].sum(axis=1), [2.5, 97.5])
    return float(lo), float(hi), len(ids)


SRC_EXTRA: dict = {}   # 549 run: source ids of the 168 new Valentine scenarios (the benchmark_properties rule)


def main(cells: Path | None = None, rep: Path | None = None) -> int:
    from sklearn.tree import DecisionTreeRegressor
    sys.stdout.reconfigure(encoding="utf-8")
    G, K = s1.GRID, s1.KAPPA
    df = pd.read_csv(cells or s1.CELLS / "cells.csv.gz")
    jcols = [f"J_{i}_{j}" for i in range(len(G)) for j in range(len(G))]
    J = df[jcols].to_numpy()
    df["oracle_ms"] = J.max(axis=1)
    best = J.argmax(axis=1)                                  # ties -> the first, i.e. the smaller weights
    df["i_star"], df["j_star"] = best // len(G), best % len(G)
    rng = np.random.default_rng(20261008)
    scr = df[df.screen]
    frozen = jcols[int(np.argmax([macro(scr[c], scr) for c in jcols]))]
    fi, fj = map(int, frozen.split("_")[1:])
    bcols = [f"Jb_{fi}_{k}" for k in range(len(K))]
    kb = int(np.argmax([macro(scr[c], scr) for c in bcols]))
    df["J_b"] = df[f"Jb_{fi}_{kb}"]
    lines = ["# X94 stage 1: headroom of per-scenario weights for D and L (EXPLORATORY)", "",
             f"- frozen_V (best on the 60 Valentine screen scenarios): w_D = {G[fi]}, w_L = {G[fj]}; rule (b): kappa = {K[kb]} "
             f"with that w_D.",
             f"- rule (a) check: max |J_a − J_sf_t2| = {(df.J_a - df.J_sf_t2).abs().max():.2e} over {int(df.J_sf_t2.notna().sum())} "
             f"cells; c/a median {df.ca.median():.3f} (IQR {df.ca.quantile(.25):.3f}–{df.ca.quantile(.75):.3f}).", "",
             "## Q1 headroom and Q3 rules (J_tf; mean over matchers per scenario, then scenarios; Valentine without the screen)", "",
             "| corpus | oracle_ms | oracle_s | best global (on corpus) | frozen_V | rule (a) = SF(t2) | rule (b) | SF conv | "
             "headroom ms / s |", "|---|---|---|---|---|---|---|---|---|"]
    summ = {}
    for c in SCORED:
        g = df[(df.corpus == c) & ~df.screen]
        if g.empty:
            continue
        per_s = g.groupby(["dataset", "scenario"])[jcols].mean()
        r = {"oracle_ms": macro(g.oracle_ms, g), "oracle_s": float(per_s.max(axis=1).mean()),
             "best_global": max(macro(g[col], g) for col in jcols), "frozen": macro(g[frozen], g),
             "a": macro(g.J_a, g), "b": macro(g.J_b, g), "conv": macro(g.J_sf_conv, g)}
        summ[c] = r
        lines.append(f"| {c} | {r['oracle_ms']:.4f} | {r['oracle_s']:.4f} | {r['best_global']:.4f} | {r['frozen']:.4f} | "
                     f"{r['a']:.4f} | {r['b']:.4f} | {r['conv']:.4f} | {r['oracle_ms'] - r['best_global']:+.4f} / "
                     f"{r['oracle_s'] - r['best_global']:+.4f} |")
    lines += ["", "Contrasts against frozen_V (cluster 95% CI where ≥ 5 clusters):", ""]
    for c in SCORED:
        g = df[(df.corpus == c) & ~df.screen]
        if g.empty:
            continue
        for name, col in (("rule (a) − frozen_V", "J_a"), ("rule (b) − frozen_V", "J_b")):
            v = g[col] - g[frozen]
            lo, hi, k = cluster_ci(v, g, c, rng)
            ci = f"[{lo:+.4f}, {hi:+.4f}]" if np.isfinite(lo) else "–"
            lines.append(f"- {c}: {name} {macro(v, g):+.4f} {ci} ({k} clusters)")
        ga, gb = summ[c]["a"] - summ[c]["frozen"], summ[c]["b"] - summ[c]["frozen"]
        lines.append(f"  - rule (b)'s share of rule (a)'s gain: {gb / ga:.2f}" if abs(ga) > 1e-9 else "  - rule (a) gain ≈ 0")
    lines += ["", "Oracle w_D (share of cells with w_D ≤ 0.5): " +
              "; ".join(f"{c} {(df[df.corpus == c].i_star <= G.index(0.5)).mean():.0%}" for c in SCORED)]
    lines += ["", "## Q2 predictability: depth-3 trees on ground-truth-free features, leave one corpus out", "",
              "| held-out | variant | J_pred | J_frozen(train) | oracle_ms | recovered share | share vs best global (on corpus) |",
              "|---|---|---|---|---|---|---|"]
    rec = {}
    for variant, feats in (("features", FEATS), ("features + c/a", FEATS + ["ca"])):
        for hold in GROUPS + ["madi_indep"]:
            tr = df[df.corpus.isin([x for x in GROUPS if x != hold])].dropna(subset=feats)
            te = df[(df.corpus == hold) & ~df.screen].dropna(subset=feats)
            if te.empty:
                continue
            pi = DecisionTreeRegressor(max_depth=3, random_state=0).fit(tr[feats], tr.i_star).predict(te[feats])
            pj = DecisionTreeRegressor(max_depth=3, random_state=0).fit(tr[feats], tr.j_star).predict(te[feats])
            pi = np.clip(np.rint(pi), 0, len(G) - 1).astype(int)
            pj = np.clip(np.rint(pj), 0, len(G) - 1).astype(int)
            jp = pd.Series(J[te.index.to_numpy(), pi * len(G) + pj], index=te.index)
            ftr = jcols[int(np.argmax([float(tr[c].mean()) for c in jcols]))]
            jf, jo, jpm = macro(te[ftr], te), macro(te.oracle_ms, te), macro(jp, te)
            bg = summ.get(hold, {}).get("best_global", np.nan)
            share = (jpm - jf) / (jo - jf) if jo - jf > 1e-9 else np.nan
            share_bg = (jpm - bg) / (jo - bg) if np.isfinite(bg) and jo - bg > 1e-9 else np.nan
            rec[(variant, hold)] = share
            lines.append(f"| {hold} | {variant} | {jpm:.4f} | {jf:.4f} | {jo:.4f} | {share:.2f} | {share_bg:.2f} |")
    head = {c: summ[c]["oracle_ms"] - summ[c]["best_global"] for c in INDEP if c in summ}
    stop1 = all(h < 0.01 for h in head.values())
    rec_ok = sum(1 for c in INDEP if (rec.get(("features", c)) or 0) >= 1 / 3)
    rules_beat = any(summ[c]["a"] > summ[c]["frozen"] or summ[c]["b"] > summ[c]["frozen"] for c in INDEP if c in summ)
    stop2 = rec_ok == 0 and not rules_beat
    lines += ["", "## Stop rules (D's)", "",
              "- headroom oracle_ms − best global on the independent corpora: " + ", ".join(f"{c} {h:+.4f}" for c, h in head.items())
              + f" → stop rule 1 {'MET (stop)' if stop1 else 'not met'}",
              f"- learned weights (without c/a) recover ≥ 1/3 on {rec_ok} of 3; an analytic rule beats frozen_V on some independent "
              f"corpus: {rules_beat} → stop rule 2 {'MET (stop)' if stop2 else 'not met'}",
              f"- **verdict: {'STOP' if stop1 or stop2 else 'go to stage 2 (after a pre-registration)'}**"]
    rep = rep or s1.REP
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "stage1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main549() -> int:
    """SMX_VALENTINE=549: the same analysis on x94_stage1.py's 549 cells; output results/x94_549/stage1.md."""
    sys.path.insert(0, str(STUDY / "reference"))
    import valentine549 as v
    from explore_benchmark_properties import source_id
    if not (s1.CELLS549 / "cells.csv.gz").is_file():
        print("x94 549: pending (x94_stage1.py --compute under SMX_VALENTINE=549 first)")
        return 0
    SRC_EXTRA.update({(ds, sc): source_id("valentine_plain", ds, sc) for ds, sc, _ in v.cases549() if v.is_new(ds, sc)})
    return main(s1.CELLS549 / "cells.csv.gz", s1.REP549)


if __name__ == "__main__":
    raise SystemExit(main549() if s1.V549 else main())
