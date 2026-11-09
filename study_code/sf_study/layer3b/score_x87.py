"""X87 scorer (EXPLORATORY; EXPLORATORY.md X87, written 2026-10-07 before any X87 run): graph-construction order bias.

Runs (A): tuning/results/<prefix>_<c>/<m>__<arm>, prefix x87 = the original FD/UCC/IND profiles carried over (primary;
isolates SF's construction order), x87rp = re-profiled permuted tables (FD arms only; profiler + SF together), on the
column-permuted corpora (layer3b/x87_perm_prepare.py) with the permuted
initial matrices; arms melnik, fd, ucc_top1, ucc1_fd_topk (originals in x81_<c>), fd_top1 (e2rest_<c>), melnik_top1 (r8_<c>).
Each permuted SF output is aligned back to the original row and column labels and compared with the original run's output:
    matrix: max |S_perm − S_orig| over cells;
    metric: J_tf (and R_exp, F1_Hr) of both against the ORIGINAL truth with the ORIGINAL tie orders (score_r7.metrics,
            permutations(shape, "<DS>-Unionable/<sc>")), so an order-invariant construction gives ΔJ = 0 exactly.
Report per corpus (madi split into madi_indep / madi_products) and arm: cells, share with |ΔJ| > 1e-9, share with a matrix change
> 1e-6, max |ΔJ|, mean ΔJ; and the change of the contrasts fd_top1 − ucc_top1, ucc1_fd_topk − ucc_top1, melnik_top1 − melnik,
fd − melnik (mean over the matchers complete in both arms, then scenarios). Intervals for mean ΔJ: X86's cluster bootstrap
(10,000, seed 20261007), only with >= 5 clusters.

    python layer3b/score_x87.py [--runs-root DIR]      -> tuning/results/x87_scores/{rows.csv, report.md}
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY.parent / "harness"))
import score_x82 as sx  # noqa: E402
import x86_cluster_ci as x86  # noqa: E402
from score_r7 import metrics, run_dir  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402
from score_seeds import ground_truth_file, one  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

RES = STUDY / "tuning" / "results"
GEN = STUDY.parent / "harness" / "generated"
PERM = STUDY / "data" / "x87_perm"
ROOT = {"madi": GEN / "madi-imported", "hdxsm": GEN / "hdxsm-usable"}
DS = {"madi": "MADI-Unionable", "hdxsm": "HDXSM-Unionable"}
ORIG = {"melnik": "x81_{c}", "fd": "x81_{c}", "ucc_top1": "x81_{c}", "ucc1_fd_topk": "x81_{c}", "fd_top1": "e2rest_{c}",
        "melnik_top1": "r8_{c}"}
CONTRASTS = [("fd_top1", "ucc_top1"), ("ucc1_fd_topk", "ucc_top1"), ("melnik_top1", "melnik"), ("fd", "melnik")]
F = ("J", "R_exp", "F1_Hr")


def job(args):
    c, sc, runs_root, matchers, prefix = args
    ds = DS[c]
    gt = read_matrix(ground_truth_file(ROOT[c], ds, sc))[0] > 0.5
    perms = permutations(gt.shape, f"{ds}/{sc}")
    out = []
    for m in matchers:
        for arm, stage in ORIG.items():
            ro = run_dir(RES / stage.format(c=c) / f"{m}__{arm}")
            rp = run_dir(runs_root / f"{prefix}_{c}" / f"{m}__{arm}")
            fo = one(ro / ds / sc / "_outputs" / "MatchingStepLine2") if ro else None
            fp = one(rp / ds / sc / "_outputs" / "MatchingStepLine2") if rp else None
            if fo is None or fp is None:
                continue
            So, rlo, clo = read_matrix(fo)
            Sp, rlp, clp = read_matrix(fp)
            rlp, clp = list(rlp), list(clp)
            if sorted(rlp) != sorted(rlo) or sorted(clp) != sorted(clo):
                out.append({"corpus": c, "scenario": sc, "matcher": m, "arm": arm, "error": "labels differ"})
                continue
            Sa = Sp[np.ix_([rlp.index(x) for x in rlo], [clp.index(x) for x in clo])]
            if Sa.shape != gt.shape or not (np.isfinite(Sa).all() and np.isfinite(So).all()):
                continue
            mo, mp = metrics(So, gt, perms), metrics(Sa, gt, perms)
            out.append({"corpus": c, "scenario": sc, "matcher": m, "arm": arm, "matrix_maxdiff": float(np.abs(Sa - So).max()),
                        **{f"orig_{k}": mo[k] for k in F}, **{f"perm_{k}": mp[k] for k in F}})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=str(RES))
    ap.add_argument("--prefix", default="x87", help="x87 (original profiles carried over, primary) or x87rp (re-profiled)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cases = json.loads((PERM / "cases.json").read_text(encoding="utf-8"))
    a.out = a.out or str(RES / f"{a.prefix}_scores")
    jobs = []
    for c, scs in cases.items():
        matchers = sorted(p.name for p in (PERM / "seeds" / c).iterdir() if p.is_dir() and p.name != "datatype_nanfix")
        jobs += [(c, sc, Path(a.runs_root), matchers, a.prefix) for sc in scs]
    rows = []
    with Pool(10) as pool:
        for r in pool.imap_unordered(job, jobs):
            rows += r
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rows.csv", index=False)
    err = int(df["error"].notna().sum()) if "error" in df else 0
    df = df[df["error"].isna()] if "error" in df else df
    df["corpus"] = np.where((df.corpus == "madi") & df.scenario.str.startswith("products__"), "madi_products",
                            np.where(df.corpus == "madi", "madi_indep", df.corpus))
    df["dJ"] = df.perm_J - df.orig_J
    df = df[~df.matcher.map(sx.controls)]
    rng = np.random.default_rng(20261007)
    lines = [f"# X87 ({a.prefix}): graph-construction order bias (EXPLORATORY)", "", f"- rows with differing labels (refused): {err}", "",
             "| corpus | arm | cells | ΔJ ≠ 0 | matrix changed | max |ΔJ| | mean ΔJ [cluster CI] | clusters |",
             "|---|---|---|---|---|---|---|---|"]

    def cl_ci(x: pd.Series, corpus: str):
        keys = list(x.index)
        cmap, _ = x86.cluster_map(corpus, keys)
        cl = np.array([cmap[k] for k in keys]); ids = np.unique(cl); v = x.to_numpy()
        if len(ids) < x86.MIN_CL:
            return np.nan, np.nan, len(ids)
        sums = np.array([v[cl == i].sum() for i in ids]); cnts = np.array([(cl == i).sum() for i in ids])
        idx = rng.integers(0, len(ids), size=(x86.B, len(ids)))
        lo, hi = np.percentile(sums[idx].sum(axis=1) / cnts[idx].sum(axis=1), [2.5, 97.5])
        return lo, hi, len(ids)
    summ = []
    for (c, arm), g in df.groupby(["corpus", "arm"]):
        per = g.groupby("scenario").dJ.mean()
        lo, hi, k = cl_ci(per, c)
        ci = f"[{lo:+.4f}, {hi:+.4f}]" if np.isfinite(lo) else "–"
        summ.append({"corpus": c, "arm": arm, "cells": len(g), "share_dJ": float((g.dJ.abs() > 1e-9).mean()),
                     "share_matrix": float((g.matrix_maxdiff > 1e-6).mean()), "max_abs_dJ": float(g.dJ.abs().max()),
                     "mean_dJ": float(per.mean()), "lo": lo, "hi": hi, "clusters": k})
        lines.append(f"| {c} | {arm} | {len(g)} | {summ[-1]['share_dJ']:.1%} | {summ[-1]['share_matrix']:.1%} | "
                     f"{summ[-1]['max_abs_dJ']:.4f} | {per.mean():+.4f} {ci} | {k} |")
    lines += ["", "## Contrasts: original vs permuted (J), and the change", "",
              "| corpus | contrast | original | permuted | change [cluster CI] |", "|---|---|---|---|---|"]
    for c in sorted(df.corpus.unique()):
        g = df[df.corpus == c]
        for arm, base in CONTRASTS:
            w = g[g.arm.isin([arm, base])].pivot_table(index=["scenario", "matcher"], columns="arm",
                                                       values=["orig_J", "perm_J"]).dropna()
            if w.empty or ("orig_J", arm) not in w or ("orig_J", base) not in w:
                continue
            co = (w[("orig_J", arm)] - w[("orig_J", base)]).groupby(level="scenario").mean()
            cp = (w[("perm_J", arm)] - w[("perm_J", base)]).groupby(level="scenario").mean()
            lo, hi, k = cl_ci(cp - co, c)
            ci = f"[{lo:+.4f}, {hi:+.4f}]" if np.isfinite(lo) else f"– ({k} clusters)"
            lines.append(f"| {c} | {arm} − {base} | {co.mean():+.4f} | {cp.mean():+.4f} | {(cp - co).mean():+.4f} {ci} |")
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
