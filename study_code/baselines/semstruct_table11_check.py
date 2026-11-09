"""Diagnostic (EXPLORATORY): do classic matchers on OUR readable Valentine reproduce SemStruct's Table 11 (Recall@GT by
match type, median (IQR))? If they do, corpus and evaluation agree and the SemStruct gap is specific to the method;
if they do not, the corpus or the evaluation differs.

Matrices: sf-matcher-study/reference/seeds/plain/<matcher>/<dataset>/<scenario>/seed.csv (the paper panel's initial
matchers). Recall@GT two ways, ties by index: Valentine's whole-matrix definition and the per-source-column one of
SemStruct's Sec. 2.5. Fabricated families only (ChEMBL, OpenData, TPC-DI), grouped by Valentine scenario type.

    harness/.venv-embed/Scripts/python.exe -I harness/semstruct_table11_check.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code" / "semstruct"))
import semstruct_run as sr  # noqa: E402
from semstruct_recall_check import recalls  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

SEEDS = HERE.parent / "sf-matcher-study" / "reference" / "seeds" / "plain"
MATCHERS = {"COMA (schema)": "coma", "COMA (schema+instances)": "coma_instances", "Cupid": "cupid",
            "Distribution-based (Valentine)": "valentine_distribution", "Jaccard (Valentine)": "valentine_jaccard"}
TYPES = ["Joinable", "Semantically-Joinable", "Unionable", "View-Unionable"]
PUBLISHED = {"COMA": "0.986 / 0.167 / 0.189 / 0.200", "Cupid": "0.167 / 0.553 / 0.784 / 0.667",
             "Distribution-based": "0.815 / 0.625 / 0.909 / 0.724"}


def main() -> int:
    res = {m: defaultdict(list) for m in MATCHERS}
    for case in sr.load_cases():
        if case["family"] not in ("ChEMBL", "OpenData", "TPC-DI"):
            continue
        paths = sr.case_paths(case, "plain")
        gt = sr.read_ground_truth(paths[2])
        for label, m in MATCHERS.items():
            f = SEEDS / m / case["dataset"] / case["scenario"] / "seed.csv"
            if not f.is_file():
                continue
            S = read_matrix(f)[0]
            if S.shape != gt.shape or not np.isfinite(S).all():
                continue
            whole, _, _, per_col = recalls(S, gt)
            res[label][case["scenario_type"]].append((whole, per_col))
    lines = ["# SemStruct Table 11 check: classic matchers on our readable Valentine (fabricated families)", "",
             "Median (IQR) Recall@GT by match type, ties by index. Published (SemStruct Table 11, first row per matcher): "
             + "; ".join(f"{k} {v}" for k, v in PUBLISHED.items()) + " (Joinable / Sem.-Joinable / Unionable / "
             "View-Unionable).", ""]
    for j, kind in ((0, "whole matrix (Valentine)"), (1, "per source column (SemStruct Sec. 2.5)")):
        lines += [f"## {kind}", "", "| matcher | " + " | ".join(TYPES) + " | n |", "|---" * (len(TYPES) + 2) + "|"]
        for label in MATCHERS:
            cells, n = [], 0
            for t in TYPES:
                v = np.array([r[j] for r in res[label].get(t, [])])
                n += len(v)
                if len(v) == 0:
                    cells.append("–")
                    continue
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                cells.append(f"{med:.3f} ({q3 - q1:.3f})")
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {n} |")
        lines.append("")
    out = HERE / "reports" / "semstruct_ablation_20261027" / "table11_check.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
