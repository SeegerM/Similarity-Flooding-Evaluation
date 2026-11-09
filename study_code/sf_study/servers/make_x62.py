"""X62 (EXPLORATORY, 2026-10-05): step 2, edges and SF configurations from the stage-1 self matrices.
Design and predictions: EXPLORATORY.md X62. Run after A has exported reference/seeds/self_<r>/<m>/.

Edges: for every matcher m, scenario and side, the self matrix (rows: the side's columns, cols: the same columns of
the copy) gives each column's similarity to the other columns of its table. Each column gets an edge to its K = 2 most
similar OTHER columns. Ties are broken at random
(seeded per regime, matcher, scenario, side and column): the ground truth is 99.8% diagonal, so a tie-break by
column order would make tied edges correspond across the two sides exactly where the ground truth does (C). A column whose off-diagonal row is constant or not finite gets no edge
(a constant row carries no ranking; it would only add arbitrary edges). Written in the extra-edges format
"label TAB table TAB fromColumn TAB toColumn TAB weight" to data/x62_<r>/<m>/<ds>/<sc>/<side>.tsv, label "sim",
weight 1.0 (as the f6 distribution edges).

Configs, tuning/configs/x62_<r>/:
  <init>__e_<m>    clone of layer1_<r>/<init>__melnik + smx.sf.extraEdgesDir = data/x62_<r>/<m>
  <init>__e_none   the same without edges (the gate: must equal layer 1 <init>__melnik cell for cell)
with init in INITS[r]; every config runs with smx.cp.overlay=overlay_merged.

    ../harness/.venv-bdi/Scripts/python.exe servers/make_x62.py --regime plain|md5 [--workers 10]
"""

from __future__ import annotations

import argparse
import zlib
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
for p in (STUDY.parent / "harness", STUDY, STUDY / "servers"):
    sys.path.insert(0, str(p))
from make_x61 import clone  # noqa: E402
from score_seeds import cases_for  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

INITS = {"plain": ("levenshtein", "pmf"), "md5": ("pmf", "unary_statistics_invariant"),
         "pnew": ("levenshtein",)}   # pnew: the 168 new, levenshtein only (2026-11-02: the manuscript's mr macros)
K = 2
SKIP = {"datatype", "datatype_nanfix", "constant", "random", "random_fresh", "oracle", "oracle_noise025",
        "oracle_noise050", "oracle_erase050", "oracle_decoy100", "embedding_random"}
CONFIGS = STUDY / "tuning" / "configs"
RES = STUDY / "tuning" / "results"


def edges_for(job):
    regime, m = job
    src, dst = STUDY / "reference" / "seeds" / f"self_{regime}" / m, STUDY / "data" / f"x62_{regime}" / m
    if dst.exists():
        shutil.rmtree(dst)
    n_files = n_edges = n_missing = 0
    for ds, sc, _ in cases_for(regime):
        for side in ("source", "target"):
            f = src / ds / f"{sc}__{side}" / "seed.csv"
            if not f.is_file():
                n_missing += 1
                continue
            M, rows, cols = read_matrix(f)
            names = [r.split(".", 1)[1] if "." in r else r for r in rows]
            lines = []
            for i in range(M.shape[0]):
                v = np.array(M[i], dtype=float)
                v[i] = np.nan
                other = np.delete(v, i)
                if not np.isfinite(other).all() or np.nanmax(other) <= np.nanmin(other):
                    continue
                rng = np.random.default_rng(zlib.crc32(f"x62|{regime}|{m}|{ds}|{sc}|{side}|{i}".encode()))
                key = np.nan_to_num(v, nan=-np.inf)
                order = [j for j in np.lexsort((rng.random(len(key)), -key)) if j != i][:K]
                lines += [f"sim\t{side}\t{names[i]}\t{names[j]}\t1.0" for j in order]
            out = dst / ds / sc / f"{side}.tsv"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_files += 1
            n_edges += len(lines)
    return m, n_files, n_edges, n_missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5", "pnew"), required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    seeds = STUDY / "reference" / "seeds" / f"self_{a.regime}"
    matchers = sorted(p.name for p in seeds.iterdir() if p.is_dir() and p.name not in SKIP)
    with Pool(a.workers) as pool:
        stats = list(pool.imap_unordered(edges_for, [(a.regime, m) for m in matchers]))
    complete = []
    for m, nf, ne, miss in sorted(stats):
        print(f"{m}: {nf} edge files, {ne} edges, {miss} self matrices missing")
        if miss == 0 and ne > 0:
            complete.append(m)
    root = CONFIGS / f"x62_{a.regime}"
    if root.exists():
        shutil.rmtree(root)
    for init in INITS[a.regime]:
        base = CONFIGS / f"layer1_{a.regime}" / f"{init}__melnik"
        clone(base, root / f"{init}__e_none", RES / f"x62_{a.regime}" / f"{init}__e_none", {})
        for m in complete:
            clone(base, root / f"{init}__e_{m}", RES / f"x62_{a.regime}" / f"{init}__e_{m}",
                  {"smx.sf.extraEdgesDir": (STUDY / "data" / f"x62_{a.regime}" / m).as_posix()})
    print(f"{a.regime}: {len(complete)} edge sets with every self matrix; "
          f"{len(list(root.iterdir()))} configurations in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
