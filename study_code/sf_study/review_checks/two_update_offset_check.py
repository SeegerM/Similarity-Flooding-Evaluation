"""Is the direct-literal offset of SF's two-update closed form common to all scored pairs? (Cached exports only.)

An external review (2026-10-29) derived, for SF as published with name attachment, the column part of two updates as
    x2 = a * (q + B + c * L),  B = (P ell)_scored,  L = (P^2 ell)_scored,  c = 2 / (M1 + 4),  a = (M1 + 4) / (M1 M2),
and showed on one GOBY case that B is the same number on every scored pair (so x2 is a positive-affine transform of
q + c L: same ranking, same assignment). This script checks B on every exported SF graph: per scenario the number of
scored pairs, min/max/spread of B, and whether the scenario is single-table (one table per side). ell = the literal
seeds of init.tsv (origin "literal_seed:*"); P from edges.tsv (dst <- src with the propagation coefficient), exactly as
the review's replay, which matched the stored SF matrix to 1e-8.

    python review_checks/two_update_offset_check.py  ->  tuning/results/review_checks/two_update_offset.{csv,md}
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
EXPORT = STUDY / "results" / "sf_export_graph"
OUT = STUDY / "tuning" / "results" / "review_checks"
# rerun 549 (2026-11-01): the exports of the 168 new Valentine scenarios (valentine_pnew, valentine_mnew) are part of the
# 549 run only. Without the switch they are skipped (the 382 check as stored); with SMX_VALENTINE=549 they are included,
# the Joinable copy of the duplicate TPC-DI scenario is dropped, and the output goes next to the 549 figures.
V549 = os.environ.get("SMX_VALENTINE", "382") == "549"
NEW168 = ("valentine_pnew", "valentine_mnew")
DUP = "TPC-DI-Joinable/prospect_horizontal_50_ec_av"
if V549:
    OUT = STUDY.parent / "latex" / "sf-evaluation-549" / "review_checks"
if os.environ.get("SMX_REVIEW_DIR"):
    OUT = Path(os.environ["SMX_REVIEW_DIR"])


def read_tsv(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> int:
    rows = []
    for nodes_path in sorted(EXPORT.rglob("nodes.tsv")):
        d = nodes_path.parent
        corpus = d.relative_to(EXPORT).parts[0]
        if not V549 and corpus in NEW168:
            continue
        if V549 and corpus in ("valentine_plain", "valentine_md5") and "/".join(d.relative_to(EXPORT).parts[1:]) == DUP:
            continue
        nodes = read_tsv(nodes_path)
        init = {int(r["idx"]): r for r in read_tsv(d / "init.tsv")}
        n = len(nodes)
        lit = np.zeros(n)
        for i, r in init.items():
            if r["origin"].startswith("literal_seed"):
                lit[i] = float(r["sigma0"])
        b = np.zeros(n)
        with (d / "edges.tsv").open(encoding="utf-8", newline="") as fh:
            for e in csv.DictReader(fh, delimiter="\t"):
                b[int(e["dst"])] += float(e["coefficient"]) * lit[int(e["src"])]
        scored = [int(r["idx"]) for r in nodes if r["scored"].lower() == "true"]
        tables_l = {r["table_l"] for r in nodes if r["scored"].lower() == "true"}
        tables_r = {r["table_r"] for r in nodes if r["scored"].lower() == "true"}
        bs = b[scored] if scored else np.zeros(0)
        rows.append({"corpus": corpus, "scenario": "/".join(d.relative_to(EXPORT).parts[1:]),
                     "n_scored": len(scored), "single_table": int(len(tables_l) == 1 and len(tables_r) == 1),
                     "B_min": float(bs.min()) if len(bs) else float("nan"),
                     "B_max": float(bs.max()) if len(bs) else float("nan"),
                     "B_spread": float(bs.max() - bs.min()) if len(bs) else float("nan")})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "two_update_offset.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["# Direct-literal offset B of SF's two-update closed form: common to all scored pairs?", "",
             "Cached SF graph exports (results/sf_export_graph); B = (P ell) on scored column pairs. A spread of 0 "
             "means x2 = a (q + c L) + a B with one constant B, i.e. the same ranking and assignment as q + c L.", "",
             "| corpus | scenarios | single-table | max spread of B (single-table) | scenarios with spread > 1e-9 "
             "(single / multi) |", "|---|---|---|---|---|"]
    for corpus in sorted({r["corpus"] for r in rows}):
        sel = [r for r in rows if r["corpus"] == corpus]
        single = [r for r in sel if r["single_table"]]
        multi = [r for r in sel if not r["single_table"]]
        maxs = max((r["B_spread"] for r in single), default=float("nan"))
        lines.append(f"| {corpus} | {len(sel)} | {len(single)} | {maxs:.2e} | "
                     f"{sum(r['B_spread'] > 1e-9 for r in single)} / {sum(r['B_spread'] > 1e-9 for r in multi)} |")
    single_all = [r for r in rows if r["single_table"]]
    summary = {"scenarios": len(rows), "single_table": len(single_all),
               "single_table_nonconstant_B": sum(r["B_spread"] > 1e-9 for r in single_all),
               "max_single_table_spread": max(r["B_spread"] for r in single_all),
               "multi_table_nonconstant_B": sum(r["B_spread"] > 1e-9 for r in rows if not r["single_table"])}
    lines += ["", "Summary: " + json.dumps(summary)]
    (OUT / "two_update_offset.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
