"""X61 (EXPLORATORY, 2026-10-05): build step. Design and predictions: EXPLORATORY.md X61.

1. Chooses the term weights on the 60 screen scenarios (tuning/corpus/<r>/screen) by the panel-mean J_tf:
     ADD  = q+ + 5D + g R + l N_fd + m N_dist          (no SF; PRE uses the same weights)
     POST = SF+(X) + 5D + g' R + l' N_fd + m' N_dist   (SF as published, layer 1 melnik)
   R and N_fd as explore_deproles (roles, neighbour_term); N_dist the same neighbour term along the f6 knn2
   distribution edges (data/f6_<r>/all/knn2/<ds>/<sc>/{source,target}.tsv). Written to tuning/x61_weights.json.
2. Writes the PRE seeds: reference/seeds/<r>/x61pre_<m>/<ds>/<sc>/seed.csv = ADD with the chosen weights, all 382
   scenarios, panel matchers.
3. Writes tuning/configs/x61_<r>/:
     graph_<m>          clone of gvfull_<r>/<m>__ucc1_fd_topk + smx.sf.extraEdgesDir = data/f6_<r>/all/knn2
     pre_<m>            clone of layer1_<r>/<m>__melnik with initialMatrixDir = x61pre_<m>
     gate_coma__ucc1_fd_topk   gvfull's coma config under overlay_merged, no extra edges: must equal gvfull's run
   Every config runs with smx.cp.overlay=overlay_merged (gvb's key options and er10's extra-edges channel).

    ../harness/.venv-bdi/Scripts/python.exe servers/make_x61.py [--workers 10]
"""

from __future__ import annotations

import argparse
import os
import itertools
import json
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
for p in (STUDY.parent / "harness", STUDY / "tuning", STUDY, STUDY / "reference", STUDY / "servers"):
    sys.path.insert(0, str(p))

import score_layer1  # noqa: E402
from explore_composite import J, load, sf_out  # noqa: E402
from explore_deproles import neighbour_term, profiles, roles  # noqa: E402
from make_x60_configs import write_seed  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402
from score_seeds import CORPUS, cases_for, ground_truth_file  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

PANEL = ["coma", "coma_instances", "kang_naughton", "kang_naughton_profile", "distribution_shape",
         "distribution_shape_z", "bert_base", "sbert_e5", "char_distribution", "pattern_charclass", "semstruct",
         "thesaurus_generic", "wordnet_jcn", "magneto", "distinct_count", "null_percentage", "char3_soft",
         "containment", "cosine", "double_metaphone", "probabilistic_levenshtein", "topk_uniform_levenshtein"]
GAMMAS, LAMBDAS, MUS = (0.0, 0.1, 0.2, 0.5, 1.0), (0.0, 0.1, 0.2), (0.0, 0.1, 0.2, 0.5)
GRID = list(itertools.product(GAMMAS, LAMBDAS, MUS))
SEEDS = STUDY / "reference" / "seeds"
CONFIGS = STUDY / "tuning" / "configs"
RES = STUDY / "tuning" / "results"
OVERLAY = "overlay_merged"


def screen_cases(regime: str) -> set[tuple[str, str]]:
    root = STUDY / "tuning" / "corpus" / regime / "screen"
    return {(d.name, s.name) for d in root.iterdir() if d.is_dir() for s in d.iterdir()}


def dist_neighbours(regime: str, ds: str, sc: str) -> tuple[dict, dict]:
    out = []
    for side in ("source", "target"):
        nb = defaultdict(set)
        f = STUDY / "data" / f"f6_{regime}" / "all" / os.environ.get("X61_KNN", "knn2") / ds / sc / f"{side}.tsv"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) >= 4:
                    a, b = f"{parts[1]}.{parts[2]}", f"{parts[1]}.{parts[3]}"
                    nb[a].add(b)
                    nb[b].add(a)
        out.append(nb)
    return out[0], out[1]


def terms(regime: str, m: str, ds: str, sc: str):
    """(q+, D, R, N_fd, N_dist, SF+ or None, rows, cols, gt) for one matcher and scenario, or None."""
    f_q = SEEDS / regime / m / ds / sc / "seed.csv"
    f_d = SEEDS / regime / "datatype_nanfix" / ds / sc / "seed.csv"
    g = ground_truth_file(CORPUS[regime], ds, sc)
    if not (f_q.is_file() and f_d.is_file() and g is not None):
        return None
    q, rows, cols = read_matrix(f_q)
    rows, cols = list(rows), list(cols)
    D = load(f_d, rows, cols)
    gt = read_matrix(g)[0] > 0.5
    if D is None or q.shape != gt.shape or not np.isfinite(q).all():
        return None
    D = np.nan_to_num(D)
    qp = np.clip(q, 0, None)
    Fs, nbs = roles(rows, *profiles(regime, ds, sc, "source"))
    Ft, nbt = roles(cols, *profiles(regime, ds, sc, "target"))
    R = 1.0 - np.abs(Fs[:, None, :] - Ft[None, :, :]).mean(axis=2)
    Nfd = neighbour_term(qp, rows, cols, nbs, nbt)
    ds_s, ds_t = dist_neighbours(regime, ds, sc)
    Nd = neighbour_term(qp, rows, cols, ds_s, ds_t)
    rd = score_layer1.run_dir(RES / f"layer1_{regime}" / f"{m}__melnik")
    S = load(sf_out(rd, ds, sc), rows, cols) if rd else None
    return qp, D, R, Nfd, Nd, (np.clip(S, 0, None) if S is not None else None), rows, cols, gt


def grid_job(job):
    regime, m, ds, sc = job
    t = terms(regime, m, ds, sc)
    if t is None:
        return None
    qp, D, R, Nfd, Nd, S, _, _, gt = t
    perms = permutations(gt.shape, f"{ds}/{sc}")
    out = {}
    for gm, lm, mu in GRID:
        extra = gm * R + lm * Nfd + mu * Nd
        out[("ADD", gm, lm, mu)] = J(qp + 5.0 * D + extra, gt, perms)
        if S is not None:
            out[("POST", gm, lm, mu)] = J(S + 5.0 * D + extra, gt, perms)
    return m, ds, sc, out


def seed_job(job):
    regime, m, ds, sc, w = job
    t = terms(regime, m, ds, sc)
    if t is None:
        return f"missing {m} {ds}/{sc}"
    qp, D, R, Nfd, Nd, _, rows, cols, _ = t
    write_seed(SEEDS / regime / f"x61pre_{m}" / ds / sc / "seed.csv",
               qp + 5.0 * D + w[0] * R + w[1] * Nfd + w[2] * Nd, rows, cols)
    return None


def clone(src: Path, dst: Path, results: Path, jvm_add: dict, seed_dir: Path | None = None) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    gen = [f'resultsDir: "{results.as_posix()}"' if g.startswith("resultsDir:") else g
           for g in (dst / "general.yaml").read_text(encoding="utf-8").splitlines()]
    (dst / "general.yaml").write_text("\n".join(gen) + "\n", encoding="utf-8")
    if seed_dir is not None:
        sl = [f'  initialMatrixDir: ["{seed_dir.as_posix()}"]' if s.strip().startswith("initialMatrixDir:") else s
              for s in (dst / "second_line_matchers.yaml").read_text(encoding="utf-8").splitlines()]
        (dst / "second_line_matchers.yaml").write_text("\n".join(sl) + "\n", encoding="utf-8")
    jvm = [l.strip() for l in (dst / "jvm.properties").read_text(encoding="utf-8").splitlines()
           if l.strip() and not l.startswith("smx.cp.overlay")]
    jvm += [f"{k}={v}" for k, v in jvm_add.items()] + [f"smx.cp.overlay={OVERLAY}"]
    (dst / "jvm.properties").write_text("\n".join(jvm) + "\n", encoding="utf-8")
    meta = json.loads((dst / "meta.json").read_text(encoding="utf-8")) if (dst / "meta.json").is_file() else {}
    meta.update({"stage": "x61", "label": dst.name, "overlay": OVERLAY,
                 "jvm": {l.split("=", 1)[0]: l.split("=", 1)[1] for l in jvm}})
    if seed_dir is not None:
        meta["seed_dir"] = seed_dir.as_posix()
        meta.setdefault("sf", {})["initialMatrixDir"] = seed_dir.as_posix()
    (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--regime", choices=("plain", "md5"), action="append",
                    help="rebuild only these regimes (default both); the other regimes' weights are kept")
    a = ap.parse_args()
    wfile = STUDY / "tuning" / "x61_weights.json"
    weights = json.loads(wfile.read_text(encoding="utf-8")) if a.regime and wfile.is_file() else {}
    for regime in (a.regime or ["plain", "md5"]):
        screen = screen_cases(regime)
        jobs = [(regime, m, ds, sc) for m in PANEL for ds, sc in sorted(screen)]
        with Pool(a.workers) as pool:
            res = [r for r in pool.imap_unordered(grid_job, jobs) if r]
        acc = defaultdict(list)
        for m, ds, sc, out in res:
            for k, v in out.items():
                acc[k].append(v)
        n_full = len(PANEL) * len(screen)
        choice = {}
        for form in ("ADD", "POST"):
            cand = {k: float(np.mean(v)) for k, v in acc.items() if k[0] == form and len(v) == max(
                len(x) for kk, x in acc.items() if kk[0] == form)}
            best = max(cand.values())
            k = next(k for k in sorted(cand, key=lambda k: (k[1] + k[2] + k[3], k)) if cand[k] >= best - 1e-12)
            choice[form] = {"gamma": k[1], "lambda": k[2], "mu": k[3], "screen_mean_J": round(best, 5),
                            "cells": len(acc[k]), "of": n_full}
        weights[regime] = {**choice, "knn_edges": os.environ.get("X61_KNN", "knn2"), "fd_parse": "fixed 2026-10-06"}
        print(regime, json.dumps(choice), flush=True)
        w = (choice["ADD"]["gamma"], choice["ADD"]["lambda"], choice["ADD"]["mu"])
        for m in PANEL:
            d = SEEDS / regime / f"x61pre_{m}"
            if d.exists():
                shutil.rmtree(d)
        with Pool(a.workers) as pool:
            miss = [r for r in pool.imap_unordered(seed_job, [(regime, m, ds, sc, w) for m in PANEL
                                                               for ds, sc, _ in cases_for(regime)]) if r]
        print(regime, "PRE seeds written;", len(miss), "missing", miss[:5], flush=True)
        root = CONFIGS / f"x61_{regime}"
        if root.exists():
            shutil.rmtree(root)
        knn2 = STUDY / "data" / f"f6_{regime}" / "all" / "knn2"
        for m in PANEL:
            clone(CONFIGS / f"gvfull_{regime}" / f"{m}__ucc1_fd_topk", root / f"graph_{m}",
                  RES / f"x61_{regime}" / f"graph_{m}", {"smx.sf.extraEdgesDir": knn2.as_posix()})
            clone(CONFIGS / f"layer1_{regime}" / f"{m}__melnik", root / f"pre_{m}",
                  RES / f"x61_{regime}" / f"pre_{m}", {}, seed_dir=SEEDS / regime / f"x61pre_{m}")
        clone(CONFIGS / f"gvfull_{regime}" / "coma__ucc1_fd_topk", root / "gate_coma__ucc1_fd_topk",
              RES / f"x61_{regime}" / "gate_coma__ucc1_fd_topk", {})
        print(regime, len(list(root.iterdir())), "configurations", flush=True)
    wfile.write_text(json.dumps(weights, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
