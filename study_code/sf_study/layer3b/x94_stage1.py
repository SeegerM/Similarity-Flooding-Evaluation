"""X94 stage 1 (EXPLORATORY; EXPLORATORY.md X94, design and predictions written 2026-10-08 before computing): headroom of
scenario-adaptive weights for the one-step terms D and L, their predictability, and two analytic rules for L's weight.

    python layer3b/x94_stage1.py --compute     -> tuning/results/x94/cells.parquet (one row per matcher x scenario)
    python layer3b/x94_stage1.py --analyse     -> results/x94/{headroom,rules,predict}.md
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
for p in (STUDY.parent / "harness", STUDY / "tuning", STUDY, HERE):
    sys.path.insert(0, str(p))
import score_x93 as s93  # noqa: E402

CELLS = STUDY / "tuning" / "results" / "x94"
REP = STUDY / "results" / "x94"
# rerun 549 (2026-11-02): SMX_VALENTINE=549 computes the cells of the 168 new readable Valentine scenarios (corpus key
# valentine_pnew: seeds pnew, exports valentine_pnew, SF from layer1_pnew / iter_pnew) and joins them, labelled
# valentine_plain, to the stored cells (whose inputs, q, D, L and the melnik runs, have no FD input; the duplicate's
# Joinable copy dropped): tuning/results/x94_549/cells.csv.gz, results/x94_549/stage1.md.
import os  # noqa: E402
V549 = os.environ.get("SMX_VALENTINE", "382") == "549"
CELLS549, REP549 = STUDY / "tuning" / "results" / "x94_549", STUDY / "results" / "x94_549"
if V549:
    s93.CORP.setdefault("valentine_pnew", ("pnew", "pnew", "layer1_pnew", "iter_pnew", "valentine"))
GRID = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
KAPPA = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
PANEL = ("coma", "containment", "cupid", "distinct_count", "jaro_winkler", "jsd_values", "kang_naughton", "ks", "levenshtein",
         "pmf", "prefix_suffix", "qgram3_jaccard", "role_gcov", "sbert_mpnet", "unary_statistics_invariant")
NAME_BASED = {"coma", "cupid", "jaro_winkler", "levenshtein", "prefix_suffix", "qgram3_jaccard", "sbert_mpnet"}
CORPORA = ("valentine_plain", "goby", "hdxsm", "gdcsm", "madi")


def graph(d: Path):
    nodes = pd.read_csv(d / "nodes.tsv", sep="\t")
    e = pd.read_csv(d / "edges.tsv", sep="\t")
    init = pd.read_csv(d / "init.tsv", sep="\t").set_index("idx").reindex(nodes.idx)
    n = len(nodes)
    P = csr_matrix((e.coefficient.to_numpy(float), (e.dst.to_numpy(), e.src.to_numpy())), shape=(n, n))
    sc = nodes[nodes.scored]
    key = dict(zip(zip(sc.name_id_l, sc.name_id_r), zip(sc.row, sc.col)))
    origin = init.origin.fillna("").to_numpy()
    nm = nodes[np.char.startswith(origin.astype(str), "name_pair")]
    name_cells = np.array([key.get((a, b), (-1, -1)) for a, b in zip(nm.name_id_l, nm.name_id_r)])
    return P, init.sigma0.to_numpy(float), origin, nm.idx.to_numpy(), name_cells, sc


def tie_stats(q):
    flat = q.ravel()
    _, cnt = np.unique(np.round(flat, 9), return_counts=True)
    tie_all = float((cnt[cnt > 1]).sum() / len(flat))
    rows = []
    gaps = []
    for r in q:
        _, c = np.unique(np.round(r, 9), return_counts=True)
        rows.append((c[c > 1]).sum() / len(r))
        srt = np.sort(r)[::-1]
        gaps.append(srt[0] - srt[1] if len(srt) > 1 else 0.0)
    p = flat - flat.min()
    p = p / p.sum() if p.sum() > 0 else np.full(len(flat), 1 / len(flat))
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(flat))) if len(flat) > 1 else 0.0
    return tie_all, float(np.mean(rows)), ent, float(np.mean(gaps))


def job(args):
    corpus, ds, sc = args
    import score_seeds
    from score_r7 import metrics, run_dir
    from score_screen_metrics import permutations
    from score_seeds import one
    from smx.adapters import read_matrix
    seedname, root, conv, t2, _ = s93.CORP[corpus]
    root = score_seeds.CORPUS[root] if isinstance(root, str) else root
    g = score_seeds.ground_truth_file(root, ds, sc)
    seeds = STUDY / "reference" / "seeds" / seedname
    fD = seeds / "datatype_nanfix" / ds / sc / "seed.csv"
    dexp = s93.EXPORT / corpus / ds / sc
    if g is None or not fD.is_file() or not (dexp / "nodes.tsv").is_file():
        return []
    gt = read_matrix(g)[0] > 0.5
    D = read_matrix(fD)[0]
    if D.shape != gt.shape or not np.isfinite(D).all():
        return []
    P, s0, origin, name_idx, name_cells, scn = graph(dexp)
    L = np.zeros(gt.shape)
    lit = np.where(np.char.startswith(origin.astype(str), "literal_seed"), s0, 0.0)
    Lsc = P @ (P @ lit)
    L[scn.row.to_numpy(), scn.col.to_numpy()] = Lsc[scn.idx.to_numpy()]
    Lt = L / L.max() if L.max() > 0 else L
    perms = permutations(gt.shape, f"{ds}/{sc}")
    blocks = scn.dtype_pair_id.fillna("")
    bp = blocks.value_counts(normalize=True).to_numpy()
    feat_s = {"n_src": gt.shape[0], "n_tgt": gt.shape[1], "n_blocks": int(blocks.nunique()),
              "dtype_entropy": float(-(bp * np.log(bp)).sum()), "sd_L": float(Lt.std())}
    for nm_ in ("levenshtein", "prefix_suffix"):
        f = seeds / nm_ / ds / sc / "seed.csv"
        a = read_matrix(f)[0] if f.is_file() else None
        feat_s[f"name_{nm_}_mean"] = float(a.mean()) if a is not None and a.shape == gt.shape else np.nan
        feat_s[f"name_{nm_}_sd"] = float(a.std()) if a is not None and a.shape == gt.shape else np.nan
    out = []
    for m in PANEL:
        f = seeds / m / ds / sc / "seed.csv"
        if not f.is_file():
            continue
        q = read_matrix(f)[0]
        if q.shape != gt.shape or not np.isfinite(q).all():
            continue
        qp = np.clip(q, 0, None)
        row = {"corpus": corpus, "dataset": ds, "scenario": sc, "matcher": m, "name_based": int(m in NAME_BASED), **feat_s}
        row["tie_all"], row["tie_row"], row["q_entropy"], row["q_gap"] = tie_stats(q)
        row["q_sd"], row["q_mean"] = float(q.std()), float(q.mean())
        row["sd_ratio"] = float(q.std() / Lt.std()) if Lt.std() > 0 else np.nan
        for i, wd in enumerate(GRID):
            base = qp + wd * D
            for j, wl in enumerate(GRID):
                row[f"J_{i}_{j}"] = metrics(base + wl * Lt, gt, perms)["J"]
        # rule (a): SF(t2) from the graph with this matcher's q on the name pairs (float32 like Schematch)
        s = s0.copy()
        ok = name_cells[:, 0] >= 0
        s[name_idx[ok]] = q.astype(np.float32).astype(float)[name_cells[ok, 0], name_cells[ok, 1]]
        u1 = 2 * s + P @ (2 * s)
        M1 = float(u1.max())
        x1 = u1 / M1
        v = s + x1
        u2 = v + P @ v
        x2 = u2 / float(u2.max())
        Sa = np.zeros(gt.shape)
        Sa[scn.row.to_numpy(), scn.col.to_numpy()] = x2[scn.idx.to_numpy()]
        row["M1"], row["ca"] = M1, 2.0 / (M1 + 4.0)
        row["J_a"] = metrics(Sa.astype(np.float32).astype(float), gt, perms)["J"]
        row["J_a_closed"] = metrics(qp + row["ca"] * L, gt, perms)["J"]        # q + (c/a)·L with the unscaled L
        rd = run_dir(s93.RES / t2 / f"{m}__melnik_t2") if (s93.RES / t2 / f"{m}__melnik_t2").is_dir() else None
        o = one(rd / ds / sc / "_outputs" / "MatchingStepLine2") if rd else None
        if o is not None:
            St2 = read_matrix(o)[0]
            row["J_sf_t2"] = metrics(St2, gt, perms)["J"] if St2.shape == gt.shape and np.isfinite(St2).all() else np.nan
        rc = run_dir(s93.RES / conv / f"{m}__melnik") if (s93.RES / conv / f"{m}__melnik").is_dir() else None
        oc = one(rc / ds / sc / "_outputs" / "MatchingStepLine2") if rc else None
        if oc is not None:
            Sc = read_matrix(oc)[0]
            row["J_sf_conv"] = metrics(Sc, gt, perms)["J"] if Sc.shape == gt.shape and np.isfinite(Sc).all() else np.nan
        # rule (b): lambda = kappa · SD(q) / SD(L~), for every D weight of the grid
        if Lt.std() > 0:
            for i, wd in enumerate(GRID):
                for k, kap in enumerate(KAPPA):
                    row[f"Jb_{i}_{k}"] = metrics(qp + wd * D + kap * q.std() / Lt.std() * Lt, gt, perms)["J"]
        out.append(row)
    return out


def compute() -> int:
    jobs = [(c, ds, sc) for c in CORPORA for ds, sc in s93.scenarios(c)]
    with Pool(10) as pool:
        rows = [r for rr in pool.imap_unordered(job, jobs, chunksize=2) for r in rr]
    df = pd.DataFrame(rows)
    scr = s93.screen_cases()
    df["screen"] = [(c == "valentine_plain") and ((d, s) in scr) for c, d, s in zip(df.corpus, df.dataset, df.scenario)]
    m = df.corpus == "madi"
    df.loc[m, "corpus"] = np.where(df.loc[m, "scenario"].str.startswith("products__"), "madi_products", "madi_indep")
    CELLS.mkdir(parents=True, exist_ok=True)
    df.to_csv(CELLS / "cells.csv.gz", index=False, compression="gzip")
    print(len(df), "cells;", df.groupby("corpus").size().to_dict())
    return 0


def compute549() -> int:
    sys.path.insert(0, str(STUDY / "reference"))
    import valentine549 as v
    jobs = [("valentine_pnew", ds, sc) for ds, sc, _ in v.cases549() if v.is_new(ds, sc)]
    with Pool(8) as pool:
        rows = [r for rr in pool.imap_unordered(job, jobs, chunksize=2) for r in rr]
    new = pd.DataFrame(rows)
    new["corpus"], new["screen"] = "valentine_plain", False
    old = pd.read_csv(CELLS / "cells.csv.gz")
    dup = (old.corpus == "valentine_plain") & (old.dataset == "TPC-DI-Joinable") & (old.scenario == "prospect_horizontal_50_ec_av")
    df = pd.concat([old[~dup], new], ignore_index=True)[list(old.columns)]
    CELLS549.mkdir(parents=True, exist_ok=True)
    df.to_csv(CELLS549 / "cells.csv.gz", index=False, compression="gzip")
    print(len(df), "cells;", df.groupby("corpus").size().to_dict())
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", action="store_true")
    a = ap.parse_args()
    raise SystemExit((compute549() if V549 else compute()) if a.compute else 0)
