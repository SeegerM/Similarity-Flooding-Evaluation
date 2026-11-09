"""External-review check (EXPLORATORY, 2026-11-03): why does the combined one-step term (datatype + rescaled literal term,
X93's A_dL = q+ + 5D + lam_dL L/max L) fall short of SF after two updates on HDXSM and GOBY?

Decomposition from existing scored rows only (no matcher or SF run):
  X93 rows (tuning/results/x93_549_scores/rows.csv; J_tf per matcher and scenario):
      A_L  = q+ + lam_L L/max L   (rescaled literal term, no datatype term)
      A_dL = q+ + 5D + lam_dL L/max L
      SF_t2, SF_conv              (SF as published after two updates / converged)
  X95 rows (tuning/results/x95_549/rows.csv, family i = each panel matcher as its own base):
      +L   = q+ + 0.3 L           (the unrescaled one-step proxy, weight 0.3 specified in advance)
Contrasts (term minus SF): A_L − SF_t2, A_L − SF_conv, A_dL − SF_t2 (reproduces X93), A_dL − A_L (adding 5D to the
rescaled term), +L − SF_t2, +L − SF_conv, and +L − A_L (unrescaled 0.3 L instead of the rescaled term; joined on matcher
and scenario). Aggregation and intervals as X93's summarise(): mean over the matchers complete on every scenario of the
contrast, mean over scenarios, 95% bootstrap over the corpus's clusters (x86 cluster map; >= 5 clusters), seed 20261103.
Output (JSON + MD) in SMX_REVIEW_DIR, default latex/sf-evaluation-presentation-549/review_checks/.

    ../harness/.venv-bdi/Scripts/python.exe -B review_checks/xr_literal_decomposition.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "layer3b"))
import x86_cluster_ci as x86  # noqa: E402

RES = STUDY / "tuning" / "results"
OUT = Path(os.environ.get("SMX_REVIEW_DIR") or STUDY.parent / "latex" / "sf-evaluation-presentation-549" / "review_checks")
CORPORA = ["hdxsm", "goby", "gdcsm", "madi_indep"]
SEED, B, MIN_CL = 20261103, 10_000, 5


def contrast(w: pd.DataFrame, a: str, b: str, corpus: str, rng) -> dict | None:
    if a not in w or b not in w:
        return None
    v = (w[a] - w[b]).dropna()
    okm = v.groupby(level="matcher").size()
    full = okm[okm == okm.max()].index
    v = v[v.index.get_level_values("matcher").isin(full)].groupby(level=["dataset", "scenario"]).mean()
    keys = list(v.index)
    kk = [k[1] if corpus.startswith("madi") else f"{k[0]}/{k[1]}" for k in keys]
    cmap, _ = x86.cluster_map(corpus, kk)
    cl = np.array([cmap[k] for k in kk])
    ids = np.unique(cl)
    x = v.to_numpy()
    lo = hi = float("nan")
    if len(ids) >= MIN_CL:
        sums = np.array([x[cl == i].sum() for i in ids])
        cnts = np.array([(cl == i).sum() for i in ids])
        idx = rng.integers(0, len(ids), size=(B, len(ids)))
        lo, hi = (float(t) for t in np.percentile(sums[idx].sum(axis=1) / cnts[idx].sum(axis=1), [2.5, 97.5]))
    return {"est": float(x.mean()), "lo": lo, "hi": hi, "scenarios": len(x), "clusters": len(ids), "matchers": len(full)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    x93 = pd.read_csv(RES / "x93_549_scores" / "rows.csv")
    x95 = pd.read_csv(RES / "x95_549" / "rows.csv")
    x95 = x95[x95.family == "i"]
    stored = pd.read_csv(RES / "x93_549_scores" / "contrasts.csv")
    rng = np.random.default_rng(SEED)
    out = {}
    for c in CORPORA:
        w = x93[x93.corpus == c].pivot_table(index=["dataset", "scenario", "matcher"], columns="arm", values="J")
        p = x95[x95.corpus == c].set_index(["dataset", "scenario", "matcher"])[["+L", "SF_conv", "SF_t2"]]
        p = p.rename(columns={"SF_conv": "SF_conv95", "SF_t2": "SF_t295"})
        wj = w.join(p, how="inner")
        res = {}
        for a, b, src in (("A_L", "SF_t2", w), ("A_L", "SF_conv", w), ("A_dL", "SF_t2", w), ("A_dL", "A_L", w),
                          ("+L", "SF_t295", p), ("+L", "SF_conv95", p), ("+L", "A_L", wj)):
            r = contrast(src, a, b, c, rng)
            if r is not None:
                res[f"{a} − {b.replace('95', '')}"] = r
        s = stored[(stored.corpus == c) & (stored.metric == "J") & (stored.contrast == "A_dL − SF_t2")]
        if len(s) and "A_dL − SF_t2" in res:
            assert abs(float(s.estimate.iloc[0]) - res["A_dL − SF_t2"]["est"]) < 1e-9, (c, s.estimate.iloc[0])
        sf_same = (wj["SF_t2"] - wj["SF_t295"]).abs().max() if len(wj) else float("nan")
        res["_check"] = {"x93_x95_joined_rows": int(len(wj)), "max_abs_SF_t2_x93_vs_x95": float(sf_same)}
        out[c] = res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "xr_literal_decomposition.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# Literal-term decomposition (EXPLORATORY, external review 2026-11-03)", "",
          "J_tf, term minus SF; cluster bootstrap 95% (– : fewer than 5 clusters).", "",
          "| corpus | contrast | estimate [95%] | scenarios / clusters | matchers |", "|---|---|---|---|---|"]
    for c, res in out.items():
        for k, r in res.items():
            if k.startswith("_"):
                continue
            ci = f"[{r['lo']:+.4f}, {r['hi']:+.4f}]" if np.isfinite(r["lo"]) else "–"
            md.append(f"| {c} | {k} | {r['est']:+.4f} {ci} | {r['scenarios']} / {r['clusters']} | {r['matchers']} |")
        md.append(f"| {c} | check | {res['_check']} | | |")
    (OUT / "xr_literal_decomposition.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
