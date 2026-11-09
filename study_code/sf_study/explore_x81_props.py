"""EXPLORATORY X81 (2026-10-07): the X77 benchmark properties of the external corpora. This wrapper runs
explore_benchmark_properties.score_case unchanged on the new corpora. Rows of the corpora already characterised are read from
results/benchmark_properties.csv for comparison and are not recomputed; that file is not overwritten.

    ../harness/.venv-bdi/Scripts/python.exe explore_x81_props.py [--workers 10]
Output: results/x81/benchmark_properties_x81.csv, results/x81/benchmark_profile_x81.md
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import explore_benchmark_properties as ebp

HERE = Path(__file__).resolve().parent
GEN = HERE.parent / "harness" / "generated"
NEW = {"hdxsm": (GEN / "hdxsm-usable", "hdxsm", False), "gdcsm": (GEN / "gdcsm-imported", "gdcsm", False),
       "madi": (GEN / "madi-imported", "madi", False), "goby": (GEN / "goby-imported", "goby", False),
       "mimic": (HERE / "data" / "mimic_omop", "mimic", True)}
ebp.CORPORA.update(NEW)          # module level, so spawned workers see it too
_src = ebp.source_id
ebp.source_id = lambda corpus, ds, sc: (sc if corpus in NEW else _src(corpus, ds, sc))
OLD = ("valentine_plain", "t2d", "final_controlled", "final_indep", "synthea")
COLS = ("diag_share", "true_share", "ovl_true", "ovl_non", "name_true", "name_non", "dist_true", "dist_non",
        "num_share", "dtype_entropy", "fd_per_col", "key_share", "ind_per_col", "src_cols", "tgt_cols")


def label(r: dict) -> str:
    if r["corpus"] == "madi":
        return "madi_products" if r["scenario"].startswith("products__") else "madi_indep"
    if r["corpus"] == "final_controlled":
        return f"final_controlled_{r['regime']}"
    return r["corpus"]


def job549(args):
    """explore_benchmark_properties.score_case for one of the 168 new readable Valentine scenarios, under the corpus key
    valentine_plain (so the seeded non-pair sample follows the same rule as for the 382), root and seeds of the new tree."""
    ds, sc = args
    ebp.CORPORA["valentine_plain"] = (HERE.parent / "harness" / "generated" / "valentine-new168", "pnew", False)
    return ebp.score_case(("valentine_plain", ds, sc))


def main549() -> int:
    """SMX_VALENTINE=549: the profile table with Valentine readable on the 549 scenarios: the stored rows of the 381 (the
    duplicate's Joinable copy dropped; value overlap, names and numeric share do not depend on the FD profiles) and
    score_case on the 168; every other corpus from the stored files (no rescoring). Output results/x81_549/ (the FD/col
    column of the Valentine row mixes old profiles for the 381 with new ones for the 168 and is not used by any macro)."""
    root = HERE.parent / "harness" / "generated" / "valentine-new168"
    jobs = [(d.name, s.name) for d in sorted(p for p in root.iterdir() if p.is_dir()) for s in sorted(p for p in d.iterdir() if p.is_dir())]
    with Pool(10) as pool:
        new = [r for r in pool.imap_unordered(job549, jobs, chunksize=2) if r]
    out = HERE / "results" / "x81_549"
    out.mkdir(parents=True, exist_ok=True)
    new.sort(key=lambda r: (r["dataset"], r["scenario"]))
    with (out / "benchmark_properties_valentine_new168.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(new[0]))
        w.writeheader()
        w.writerows(new)
    with (HERE / "results" / "benchmark_properties.csv").open(encoding="utf-8") as fh:
        old = [r for r in csv.DictReader(fh) if r["corpus"] in OLD and not (
            r["corpus"] == "valentine_plain" and (r["dataset"], r["scenario"]) == ("TPC-DI-Joinable", "prospect_horizontal_50_ec_av"))]
    with (HERE / "results" / "x81" / "benchmark_properties_x81.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    profile(old + new, rows, out, f"549 run: valentine_plain = {sum(r['corpus'] == 'valentine_plain' for r in old)} stored + "
                                  f"{len(new)} new scenarios.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    jobs = [(c, d.name, s.name) for c, (root, _, _) in NEW.items() if root.is_dir()
            for d in sorted(p for p in root.iterdir() if p.is_dir()) for s in sorted(p for p in d.iterdir() if p.is_dir())]
    with Pool(a.workers) as pool:
        rows = [r for r in pool.imap_unordered(ebp.score_case, jobs, chunksize=2) if r]
    rows.sort(key=lambda r: (r["corpus"], r["dataset"], r["scenario"]))
    out = HERE / "results" / "x81"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "benchmark_properties_x81.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (HERE / "results" / "benchmark_properties.csv").open(encoding="utf-8") as fh:
        old = [r for r in csv.DictReader(fh) if r["corpus"] in OLD]
    profile(old, rows, out)
    return 0


def profile(old, rows, out: Path, note: str = "") -> None:
    """main()'s profile table, unchanged (moved into a function 2026-11-02 for the 549 run)."""
    by = defaultdict(list)
    for r in old + rows:
        by[label(r)].append(r)
    med = lambda rs, k: float(np.nanmedian([float(r[k]) for r in rs]))  # noqa: E731
    lines = ["# X81: X77 properties of the external corpora (EXPLORATORY)", "",
             "Medians over scenarios. ovl = value-set Jaccard, name = name similarity, dist = profile similarity; true pairs vs "
             "sampled non-pairs. num share = columns ≥ 95% numeric; dtype H = entropy of (num, text, empty) in bits.", "",
             "| corpus | scen. | cols (src×tgt) | true share | diag | ovl true / non | name true / non | dist true / non | "
             "num share | dtype H | FD/col | key share | IND/col |", "|" + "---|" * 13]
    order = [*OLD[:2], "final_controlled_plain", "final_controlled_opaque", "final_indep", "synthea",
             "gdcsm", "hdxsm", "goby", "madi_indep", "madi_products", "mimic"]
    for c in order:
        rs = by.get(c)
        if not rs:
            continue
        lines.append(f"| {c} | {len(rs)} | {med(rs, 'src_cols'):.0f}×{med(rs, 'tgt_cols'):.0f} | {med(rs, 'true_share'):.3f} | "
                     f"{med(rs, 'diag_share'):.2f} | {med(rs, 'ovl_true'):.2f} / {med(rs, 'ovl_non'):.2f} | "
                     f"{med(rs, 'name_true'):.2f} / {med(rs, 'name_non'):.2f} | {med(rs, 'dist_true'):.2f} / {med(rs, 'dist_non'):.2f} | "
                     f"{med(rs, 'num_share'):.2f} | {med(rs, 'dtype_entropy'):.2f} | {med(rs, 'fd_per_col'):.1f} | "
                     f"{med(rs, 'key_share'):.2f} | {med(rs, 'ind_per_col'):.2f} |")
    if note:
        lines += ["", note]
    (out / "benchmark_profile_x81.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


if __name__ == "__main__":
    import os
    raise SystemExit(main549() if os.environ.get("SMX_VALENTINE", "382") == "549" else main())
