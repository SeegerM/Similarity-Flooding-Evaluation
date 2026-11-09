"""X95 POST-HOC secondary analysis (not pre-registered; requested on 2026-10-08, after the primary result was
known): information-matched comparators.

The full one-step matcher uses dependency information (R13). SF as published uses none. The matched comparators are SF with the key
graph + FD edges (ucc1_fd_topk) and SF with the key graph (ucc_top1), from X81 (x81_<c>/<m>__<arm>), for base family (i) on GOBY,
HDXSM (193) and GDC-SM. Contrasts: full − each, and +R − each. Same J_tf, the same completeness rule and cluster bootstrap as
score_x95.py (10,000 draws, seed 20261009).

    python layer3b/x95_posthoc_matched.py   -> tuning/results/x95/posthoc_matched.{csv,md}
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
for p in (STUDY.parent / "harness", STUDY / "tuning", STUDY, STUDY / "reference", HERE):
    sys.path.insert(0, str(p))
import score_x95 as s95  # noqa: E402

CORPORA = ("goby", "hdxsm", "gdcsm")
ARMS = ("ucc1_fd_topk", "ucc_top1")


def job(args):
    corpus, ds, sc = args
    cell = s95.load_cell(corpus, ds, sc)
    if cell is None:
        return []
    out = []
    for m in s95.PANEL:
        if m not in cell["q"]:
            continue
        row = {"corpus": corpus, "dataset": ds, "scenario": sc, "matcher": m}
        for arm in ARMS:
            S = s95.sf(s95.RES / f"x81_{corpus}", f"{m}__{arm}", cell)
            row[f"SF_{arm}"] = s95.J(S, cell) if S is not None else np.nan
        out.append(row)
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    import x86_cluster_ci as x86
    jobs = [(c, ds, sc) for c in CORPORA for ds, sc in s95.scenarios(c) if not (c == "hdxsm" and sc in s95.HDXSM_CAP)]
    with Pool(10) as pool:
        sf = pd.DataFrame([r for rr in pool.imap_unordered(job, jobs, chunksize=2) for r in rr])
    rows = pd.read_csv(s95.OUT / "rows.csv")
    rows = rows[(rows.family == "i") & rows.corpus.isin(CORPORA)]
    df = rows.merge(sf, on=["corpus", "dataset", "scenario", "matcher"], how="left")
    rng = np.random.default_rng(s95.SEED)
    res = []
    for corpus, g in df.groupby("corpus"):
        for a in ("full", "+R", "SF_conv"):
            for b in ("SF_ucc1_fd_topk", "SF_ucc_top1"):
                gg = g[g[a].notna() & g[b].notna()]
                cnt = gg.groupby("matcher").size()
                gg = gg[gg.matcher.isin(cnt[cnt == cnt.max()].index)]
                per = (gg[a] - gg[b]).groupby([gg.dataset, gg.scenario]).mean()
                kk = [f"{d}/{s}" for d, s in per.index]
                cmap, _ = x86.cluster_map(corpus, kk)
                cl = np.array([cmap[k] for k in kk])
                ids = np.unique(cl)
                x = per.to_numpy()
                lo = hi = np.nan
                if len(ids) >= s95.MIN_CL:
                    sums = np.array([x[cl == i].sum() for i in ids]); cn = np.array([(cl == i).sum() for i in ids])
                    idx = rng.integers(0, len(ids), size=(s95.B, len(ids)))
                    lo, hi = np.percentile(sums[idx].sum(axis=1) / cn[idx].sum(axis=1), [2.5, 97.5])
                res.append({"corpus": corpus, "contrast": f"{a} − {b}", "estimate": float(x.mean()), "lo": lo, "hi": hi,
                            "scenarios": len(x), "clusters": len(ids), "matchers": int(gg.matcher.nunique())})
    r = pd.DataFrame(res)
    r.to_csv(s95.OUT / "posthoc_matched.csv", index=False)
    lines = ["# X95 POST-HOC (not pre-registered): information-matched comparators, base family (i)", "",
             "| corpus | contrast | estimate [cluster 95% CI] | scenarios / clusters | matchers |", "|---|---|---|---|---|"]
    for _, x in r.iterrows():
        ci = f"[{x.lo:+.4f}, {x.hi:+.4f}]" if np.isfinite(x.lo) else "–"
        lines.append(f"| {x.corpus} | {x.contrast} | {x.estimate:+.4f} {ci} | {x.scenarios} / {x.clusters} | {x.matchers} |")
    (s95.OUT / "posthoc_matched.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
