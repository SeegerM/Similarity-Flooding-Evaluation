"""EXPLORATORY X88, mechanism part (2026-10-07): why SF's trajectory overshoots at two updates. Design and predictions
(5, 6) in EXPLORATORY.md X88, written before computing.

By Proposition 1, on one table per side without dependency edges sigma^t(p_ij) = a_t q_ij + b_t(dt(i), dt(j)). For every
scenario, matcher and t (1, 2, 3, 5, 10, converged) this fits a_t (one pooled within-block slope) and b_t (one offset per
datatype-pair block) to SF's stopped output, by least squares, and reports:
  resid  RMS residual / SD of the output (Proposition 1 predicts about 0)
  s_t    offset strength: SD over cells of b_t(block of the cell) / (a_t * SD over cells of q)
  gain   R_exp(SF_t) - R_exp(q)
Blocks: the unique rows and columns of the datatype matrix (datatype_nanfix). Runs: SF as published (melnik), stopped
after t updates (X42 iter_plain; X82 iter_x81_goby, iter_x81_hdxsm) and converged (layer1_plain, x81_*).

    ../harness/.venv-bdi/Scripts/python.exe explore_x88_mechanism.py
Output: results/x88/{mechanism.csv, mechanism.md}
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "harness", HERE / "tuning", HERE, HERE / "reference", HERE / "layer3b"):
    sys.path.insert(0, str(p))

RES = HERE / "tuning" / "results"
GEN = HERE.parent / "harness" / "generated"
TS = ("t1", "t2", "t3", "t5", "t10", "conv")
MATCHERS = ("levenshtein", "prefix_suffix", "pmf", "jaro_winkler")
CORPORA = {  # corpus: (seed folder, corpus root, iteration stage, converged stage)
    "valentine_plain": ("plain", None, "iter_plain", "layer1_plain"),
    "goby": ("goby", GEN / "goby-imported", "iter_x81_goby", "x81_goby"),
    "hdxsm": ("hdxsm", GEN / "hdxsm-usable", "iter_x81_hdxsm", "x81_hdxsm"),
}


def decompose(S: np.ndarray, q: np.ndarray, blk: np.ndarray, gt: np.ndarray | None = None):
    """(a, s, resid, o) of S = a q + b[blk] by pooled within-block least squares; o = truth-directed offset: mean offset of
    the true pairs minus the mean offset of all cells, in units of a * SD(q)."""
    s, qq, b = S.ravel(), q.ravel(), blk.ravel()
    sx = sy = 0.0
    for k in np.unique(b):
        m = b == k
        dq, ds = qq[m] - qq[m].mean(), s[m] - s[m].mean()
        sx += float(dq @ ds)
        sy += float(dq @ dq)
    a = sx / sy if sy > 0 else 0.0
    off = np.zeros_like(s)
    for k in np.unique(b):
        m = b == k
        off[m] = (s[m] - a * qq[m]).mean()
    abs_resid = float(np.sqrt(np.mean((s - a * qq - off) ** 2)))
    resid = abs_resid / (s.std() or 1.0)
    unit = abs(a) * qq.std()
    strength = float(off.std() / unit) if unit > 0 else float("nan")
    o = float((off[gt.ravel()].mean() - off.mean()) / unit) if gt is not None and gt.any() and unit > 0 else float("nan")
    # flags (the reviewer's point): no within-block variation of q -> slope unidentifiable; near-constant output
    flag = "slope_unidentifiable" if sy <= 1e-12 else ("constant_output" if s.std() <= 1e-9 else "")
    return a, strength, resid, o, abs_resid, float(s.std()), flag


def job(args):
    corpus, ds, sc = args
    import score_layer1
    import score_seeds
    from explore_composite import load, sf_out
    from r7_score import blocks
    from score_screen_metrics import recall_expected
    from smx.adapters import read_matrix
    seedname, root, it_stage, conv_stage = CORPORA[corpus]
    root = root or score_seeds.CORPUS["plain"]
    f_d = HERE / "reference" / "seeds" / seedname / "datatype_nanfix" / ds / sc / "seed.csv"
    g = score_seeds.ground_truth_file(root, ds, sc)
    if g is None or not f_d.is_file():
        return []
    D, rows, cols = read_matrix(f_d)
    rows, cols = list(rows), list(cols)
    gt = read_matrix(g)[0] > 0.5
    if gt.shape != D.shape:
        return []
    blk = blocks(np.nan_to_num(D)).reshape(D.shape)
    out = []
    for m in MATCHERS:
        q = load(HERE / "reference" / "seeds" / seedname / m / ds / sc / "seed.csv", rows, cols)
        if q is None:
            continue
        rq = recall_expected(q, gt)
        for t in TS:
            d = RES / (conv_stage if t == "conv" else it_stage) / (f"{m}__melnik" if t == "conv" else f"{m}__melnik_{t}")
            rd = score_layer1.run_dir(d) if d.is_dir() else None
            S = load(sf_out(rd, ds, sc), rows, cols) if rd else None
            if S is None:
                continue
            a, s, res, o, ares, sds, flag = decompose(S, q, blk, gt)
            out.append({"corpus": corpus, "dataset": ds, "scenario": sc, "matcher": m, "t": t, "a": a, "s": s, "o": o,
                        "abs_resid": ares, "sd_out": sds, "flag": flag,
                        "resid": res, "gain": recall_expected(S, gt) - rq, "blocks": int(len(np.unique(blk)))})
    return out


def cases(corpus):
    seedname, root, _, _ = CORPORA[corpus]
    if root is None:
        import score_seeds
        return [(corpus, ds, sc) for ds, sc, _ in score_seeds.cases_for("plain")]
    return [(corpus, d.name, s.name) for d in sorted(p for p in root.iterdir() if p.is_dir())
            for s in sorted(p for p in d.iterdir() if p.is_dir())]


def main() -> int:
    jobs = [j for c in CORPORA for j in cases(c)]
    with Pool(10) as pool:
        rows = [r for rr in pool.imap_unordered(job, jobs, chunksize=4) for r in rr]
    out = HERE / "results" / "x88"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "mechanism.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["# X88 mechanism: SF's stopped outputs decomposed as a_t q + b_t(block) (EXPLORATORY)", "",
             f"Matchers {', '.join(MATCHERS)}; SF as published. resid = RMS residual / SD(output); s = offset strength; "
             "gain = R_exp gain over q. Medians over scenario × matcher, gain as mean.", "",
             "| corpus | t | cases | median resid | median s | mean o | mean gain | Spearman(s, gain) | Spearman(o, gain) |", "|---|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[(r["corpus"], r["t"])].append(r)
    for c in CORPORA:
        for t in TS:
            rs = by.get((c, t), [])
            if not rs:
                continue
            s = np.array([r["s"] for r in rs]); g = np.array([r["gain"] for r in rs])
            ok = np.isfinite(s)
            rk = lambda v: pd.Series(v).rank(method="average").to_numpy()  # noqa: E731
            rho = float(np.corrcoef(rk(s[ok]), rk(g[ok]))[0, 1]) if ok.sum() > 3 else float("nan")
            o = np.array([r["o"] for r in rs]); ko = np.isfinite(o)
            rho_o = float(np.corrcoef(rk(o[ko]), rk(g[ko]))[0, 1]) if ko.sum() > 3 else float("nan")
            lines.append(f"| {c} | {t} | {len(rs)} | {np.median([r['resid'] for r in rs]):.4f} | {np.nanmedian(s):.3f} | "
                         f"{np.nanmean(o):+.3f} | {g.mean():+.4f} | {rho:+.2f} | {rho_o:+.2f} |")
    lines += ["", "## Residual tails (the reviewer's point 3): relative residual = RMS residual / SD(output)", "",
              "| corpus | t | cases | flagged (slope unidentifiable / constant output) | median rel. | 99th pct rel. | max rel. | "
              "share rel. > 1e-3 | share rel. > 0.1 | max abs. residual |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in CORPORA:
        for t in TS:
            rs = by.get((c, t), [])
            if not rs:
                continue
            ok = [r for r in rs if not r["flag"]]
            fl = Counter(r["flag"] for r in rs if r["flag"])
            rel = np.array([r["resid"] for r in ok]); ab = np.array([r["abs_resid"] for r in ok])
            lines.append(f"| {c} | {t} | {len(rs)} | {fl.get('slope_unidentifiable', 0)} / {fl.get('constant_output', 0)} | "
                         f"{np.median(rel):.1e} | {np.percentile(rel, 99):.1e} | {rel.max():.1e} | {np.mean(rel > 1e-3):.1%} | "
                         f"{np.mean(rel > 0.1):.1%} | {ab.max():.1e} |")
    (out / "mechanism.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
