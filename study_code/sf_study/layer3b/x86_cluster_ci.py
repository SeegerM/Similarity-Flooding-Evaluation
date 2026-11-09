"""X86 (EXPLORATORY; EXPLORATORY.md X86, design and predictions written 2026-10-07 before computing): source-aware intervals for
X82 (classfull, parallel, e2x), X83 and X84 contrasts. Clusters are table-sharing components (scenarios linked by an
identical source or target table, SHA-256 from the import manifests); final_indep by schema domain, final_controlled by
database, X84's Valentine cases by source_id (results/benchmark_properties.csv). Statistic as score_x82.contrast (same matcher
inclusion, per-matcher mean over scenarios, mean over matchers); the bootstrap resamples whole clusters (10,000, seed 20261007),
only with >= 5 clusters. GDC-SM: source-only clusters, "conditional on the GDC target".

    python layer3b/x86_cluster_ci.py   -> tuning/results/x86_cluster_ci/{contrasts.csv, report.md}
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STUDY))
import score_x82 as sx  # noqa: E402

RES = STUDY / "tuning" / "results"
MET = RES / "x82_metrics"
GEN = STUDY.parent / "harness" / "generated"
OUT = RES / "x86_cluster_ci"
B, SEED, MIN_CL = 10_000, 20261007, 5


def components(edges: dict[str, list[str]]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for sc, ts in edges.items():
        for t in ts:
            ra, rb = find("T:" + t), find("S:" + sc)
            if ra != rb:
                parent[ra] = rb
    return {sc: find("S:" + sc) for sc in edges}


def raw_tables(c: str) -> dict[str, tuple[str, str]]:
    m = json.loads((GEN / f"{c}-raw" / "convert_manifest.json").read_text(encoding="utf-8"))["scenarios"]
    return {sc: (v["sha256"][f"{sc}_source.csv"], v["sha256"][f"{sc}_target.csv"]) for sc, v in m.items()}


def cluster_map(corpus: str, scenarios: list[str]) -> tuple[dict[str, str], str]:
    """scenario key (as in the metric files) -> cluster id, and a label of the clustering."""
    name = {k: k.split("/")[-1] for k in scenarios}
    if corpus in ("hdxsm", "goby", "madi_indep", "madi_products", "gdcsm"):
        tabs = raw_tables("madi" if corpus.startswith("madi") else corpus)
        if corpus == "gdcsm":
            comp = components({s: [tabs[s][0]] for s in set(name.values())})
            return {k: comp[name[k]] for k in scenarios}, "source studies (conditional on the shared GDC target)"
        comp = components({s: list(tabs[s]) for s in set(name.values())})
        return {k: comp[name[k]] for k in scenarios}, "table-sharing components"
    if corpus == "final_indep":
        return {k: k.split("_")[0] for k in scenarios}, "schema domains"
    if corpus == "final_controlled":
        return {k: k.split("-")[0] for k in scenarios}, "databases"
    return {k: "all" for k in scenarios}, "one scenario"


def contrast_cl(d: pd.DataFrame, arm: str, base: str, metric: str, rng) -> dict | None:
    piv = d[d.arm.isin([arm, base])].pivot_table(index=["dataset", "scenario"], columns=["arm", "matcher"],
                                                    values=metric, aggfunc="first")
    if arm not in piv.columns.get_level_values(0) or base not in piv.columns.get_level_values(0):
        return None
    scen = piv[base].dropna(how="all").index
    a, b = piv[arm].reindex(scen), piv[base].reindex(scen)
    ms = [m for m in b.columns if not sx.controls(m) and m in a.columns and a[m].notna().all() and b[m].notna().all()]
    if not ms:
        return None
    diff = (a[ms] - b[ms]).mean(axis=1).to_numpy()           # mean over matchers per scenario
    keys = list(scen.get_level_values(1))
    cmap, how = cluster_map(d.corpus.iloc[0], keys)
    cl = np.array([cmap[k] for k in keys])
    ids = np.unique(cl)
    obs = float(diff.mean())
    sums = np.array([diff[cl == i].sum() for i in ids])
    cnts = np.array([(cl == i).sum() for i in ids])
    lo = hi = np.nan
    if len(ids) >= MIN_CL:
        idx = rng.integers(0, len(ids), size=(B, len(ids)))
        bs = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"estimate": obs, "lo": lo, "hi": hi, "scenarios": len(keys), "clusters": len(ids), "matchers": len(ms),
            "clustering": how}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    rows = []
    fams = {
        "X82 classfull": ("classfull_dev1.csv", [(a, b, s, "J") for a, b, s, _ in sx.CONTRASTS["classfull"]]),
        "X82 parallel": ("parallel_dev1.csv", sx.CONTRASTS["parallel"]),
        "X82 e2x (FD arms, HDXSM 193)": ("e2x_dev1_cap.csv", [("coef_prod_fd", "ucc_top1", -1, "J"), ("fd_top1", "ucc_top1", -1, "J")]),
        "X82 e2x": ("e2x_dev1.csv", [("ucc", "ucc_top1", -1, "J"), ("melnik_top1", "melnik", 1, "J")]),
    }
    frames = {f: sx.split_corpus(pd.read_csv(MET / f)) for f, _ in fams.values()}
    x83 = sx.split_corpus(pd.concat([frames["classfull_dev1.csv"], pd.read_csv(RES / "x83_class_additive" / "rows.csv")],
                                    ignore_index=True))
    sys.path.insert(0, str(HERE))
    import x83_class_additive as x83m  # noqa: E402
    fams["X83"] = (None, [(a, b, s, "J") for a, b, s in x83m.CONTRASTS])
    for fam, (f, cons) in fams.items():
        df = frames[f] if f else x83
        for arm, base, sign, metric in cons:
            for corpus in sx.INDEPENDENT + sx.BESIDE:
                d = df[df.corpus == corpus]
                if d.empty:
                    continue
                r = contrast_cl(d, arm, base, metric, rng)
                if r is None:
                    continue
                v = sx.verdict(r, sign) if sign else "report"
                if not np.isfinite(r["lo"]):
                    v = f"no interval ({r['clusters']} clusters)"
                rows.append({"family": fam, "contrast": f"{arm} − {base}", "metric": metric, "corpus": corpus, **r,
                             "verdict": v})
    # X84
    x84 = pd.read_csv(RES / "x84_noise_scores" / "rows.csv")
    props = pd.read_csv(STUDY / "results" / "benchmark_properties.csv")
    src = props[props.corpus == "valentine_plain"].set_index(["dataset", "scenario"]).source_id.to_dict()
    per = x84.groupby(["corpus", "arm", "level", "matcher", "dataset", "scenario"])[["q_J", "sf_J"]].mean().reset_index()
    per["delta"] = per.sf_J - per.q_J
    sc = per.groupby(["corpus", "arm", "level", "dataset", "scenario"]).delta.mean().reset_index()
    for c in ("valentine", "hdxsm", "goby"):
        s = sc[sc.corpus == c]
        keys = sorted(set(zip(s.dataset, s.scenario)))
        if c == "valentine":
            cmap = {k: src[k] for k in keys}
        else:
            tabs = raw_tables(c)
            comp = components({k[1]: list(tabs[k[1]]) for k in keys})
            cmap = {k: comp[k[1]] for k in keys}
        w = s.pivot_table(index=["dataset", "scenario", "level"], columns="arm", values="delta").reset_index()
        w["melnik−lit_none"] = w["melnik"] - w["lit_none"]
        for lv in ("p0", "p50", "p100", "p200", "flat"):
            g = w[w.level == lv]
            cl = np.array([cmap[(a, b)] for a, b in zip(g.dataset, g.scenario)])
            ids = np.unique(cl)
            for col in ("melnik", "lit_none", "melnik−lit_none"):
                x = g[col].to_numpy()
                lo = hi = np.nan
                if len(ids) >= MIN_CL:
                    sums = np.array([x[cl == i].sum() for i in ids]); cnts = np.array([(cl == i).sum() for i in ids])
                    idx = rng.integers(0, len(ids), size=(B, len(ids)))
                    lo, hi = np.percentile(sums[idx].sum(axis=1) / cnts[idx].sum(axis=1), [2.5, 97.5])
                rows.append({"family": "X84", "contrast": f"delta {col} {lv}", "metric": "J", "corpus": c,
                             "estimate": float(x.mean()), "lo": lo, "hi": hi, "scenarios": len(x), "clusters": len(ids),
                             "matchers": 3, "clustering": "source_id" if c == "valentine" else "table-sharing components",
                             "verdict": ("excludes 0" if np.isfinite(lo) and (lo > 0 or hi < 0) else
                                         "includes 0" if np.isfinite(lo) else f"no interval ({len(ids)} clusters)")})
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "contrasts.csv", index=False)
    lines = ["# X86: cluster-aware intervals for C's X82–X84 contrasts (EXPLORATORY)", "",
             "| family | contrast | corpus | estimate [95% cluster CI] | scenarios / clusters | clustering | verdict |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = f"[{r['lo']:+.4f}, {r['hi']:+.4f}]" if np.isfinite(r["lo"]) else "–"
        lines.append(f"| {r['family']} | {r['contrast']} ({r['metric']}) | {r['corpus']} | {r['estimate']:+.4f} {ci} | "
                     f"{r['scenarios']} / {r['clusters']} | {r['clustering']} | {r['verdict']} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
