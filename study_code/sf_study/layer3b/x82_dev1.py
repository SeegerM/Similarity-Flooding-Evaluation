"""X82's recorded deviations (EXPLORATORY.md X82, 2026-10-07 00:20), applied to a metric file before score_x82.py:
1. final_indep: `dataset` = the domain (bib, music), since every pair is its own dataset (degenerate stratified bootstrap);
2. madi: `scenario` without the "<dataset>/" prefix, so that score_x82's "products__" split works.

The agreed HDXSM FD cap (results/hdxsm_fd_cap.json): with --cap, the 6 excluded pairs are also removed
from every HDXSM row, for the contrasts that involve an FD arm (their base, e.g. ucc_top1, has all 199).

    python layer3b/x82_dev1.py <family> [--cap]   -> tuning/results/x82_metrics/<family>_dev1.csv (or _dev1_cap.csv)
"""
import json
import sys
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "tuning" / "results" / "x82_metrics"
fam, cap = sys.argv[1], "--cap" in sys.argv[2:]
d = pd.read_csv(RES / f"{fam}.csv")
m = d.corpus == "final_indep"
d.loc[m, "dataset"] = d.loc[m, "dataset"].str.split("_").str[0]
m = d.corpus == "madi"
d.loc[m, "scenario"] = d.loc[m, "scenario"].str.split("/", n=1).str[-1]
if cap:
    excl = set(json.loads((RES.parents[2] / "results" / "hdxsm_fd_cap.json").read_text(encoding="utf-8"))["excluded"])
    d = d[~((d.corpus == "hdxsm") & d.scenario.str.split("/").str[-1].isin(excl))]
d.to_csv(RES / (f"{fam}_dev1_cap.csv" if cap else f"{fam}_dev1.csv"), index=False)
print(fam, len(d), "rows;", d[d.corpus == "final_indep"].dataset.unique().tolist(),
      d[d.corpus == "madi"].scenario.str.startswith("products__").sum(), "madi products rows")
