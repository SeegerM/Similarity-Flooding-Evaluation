"""X91 checker (EXPLORATORY; EXPLORATORY.md X91, design and predictions written 2026-10-07 before computing) for SF export
(results/sf_export/<m>__<attach>_<policy>_<lit>/<ds>/<sc>/{nodes,edges,init,sigma_t,trace}.tsv, meta.json).

Per run:
  1. replay formula C from edges + init + trace (v = σ0 + x, u = v + P v, x' = u / M_t) and compare with sigma_t and M_t;
  2. Proposition 1 on the graph's own blocks: on the scored nodes, x_t = a_t q + b_t(dtype_pair_id) by pooled within-block least
     squares (q = the panel's initial matrix at (row, col)); absolute / relative residual, flags; within-block strict order kept
     and ties kept (tolerance 1e-9 · max x) at every t;
  3. decomposition x_t = Σ_j c_{t,j} P^j σ0 split by origin of σ0 (name / literal / other) and path length j, with
     c_{t,j} = Σ_k w_{t,k} C(k, j), w from the normalisers; each component is fitted like x_t (pooled within-block least squares,
     linear in the data, so the components' slopes and offsets sum exactly to a_t and b_t); reported: its slope share, the SD of
     its offsets in units of a_t·SD q, and its truth-directed part.

    python layer3b/check_sf_export.py [--limit N]   -> results/sf_export_check/{runs.csv, components.csv, report.md}
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY.parent / "harness"))

EXP = STUDY / "results" / "sf_export"
WORK = STUDY / "results" / "sf_export_work"
OUT = STUDY / "results" / "sf_export_check"
GEN = STUDY.parent / "harness" / "generated"
TOL = 1e-9


def origin_class(o: str) -> str:
    if o.startswith("name_pair"):
        return "name"
    if o.startswith("literal_seed"):
        return "literal"
    return "zero" if o == "zero" else "other"


def truth(ds: str, sc: str):
    from score_seeds import CORPUS, ground_truth_file
    from smx.adapters import read_matrix
    root = GEN / "goby-imported" if ds.startswith("GOBY") else GEN / "hdxsm-usable" if ds.startswith("HDXSM") else CORPUS["plain"]
    g = ground_truth_file(root, ds, sc)
    return read_matrix(g)[0] > 0.5 if g is not None else None


def fit(x, q, blk):
    """pooled within-block slope a, offsets b (per cell), residual."""
    sx = sy = 0.0
    for k in np.unique(blk):
        m = blk == k
        dq = q[m] - q[m].mean()
        sx += float(dq @ (x[m] - x[m].mean()))
        sy += float(dq @ dq)
    a = sx / sy if sy > 1e-12 else 0.0
    off = np.zeros_like(x)
    for k in np.unique(blk):
        m = blk == k
        off[m] = (x[m] - a * q[m]).mean()
    return a, off, x - a * q - off, sy <= 1e-12


def order_check(x, q, blk, scale):
    viol = ties_broken = pairs = ties = 0
    for k in np.unique(blk):
        m = np.where(blk == k)[0]
        if len(m) < 2:
            continue
        dq = q[m][:, None] - q[m][None, :]
        dx = x[m][:, None] - x[m][None, :]
        strict = dq > 1e-12
        pairs += int(strict.sum())
        viol += int((strict & (dx <= TOL * scale)).sum())
        tie = (np.abs(dq) <= 1e-12) & ~np.eye(len(m), dtype=bool)
        ties += int(tie.sum())
        ties_broken += int((tie & (np.abs(dx) > TOL * scale)).sum())
    return viol, pairs, ties_broken, ties


def run(path: Path):
    from smx.adapters import read_matrix
    label = path.parents[1].name
    m, setting = label.split("__", 1)
    parts = setting.split("_")
    attach, policy, lit = parts[0], "_".join(parts[1:3]), "_".join(parts[3:])
    ds, sc = path.parent.name, path.name
    nodes = pd.read_csv(path / "nodes.tsv", sep="\t")
    edges = pd.read_csv(path / "edges.tsv", sep="\t")
    init = pd.read_csv(path / "init.tsv", sep="\t").set_index("idx").loc[nodes.idx]
    sig = pd.read_csv(path / "sigma_t.tsv", sep="\t")
    trace = pd.read_csv(path / "trace.tsv", sep="\t")
    n = len(nodes)
    P = csr_matrix((edges.coefficient.to_numpy(float), (edges.dst.to_numpy(), edges.src.to_numpy())), shape=(n, n))
    s0 = init.sigma0.to_numpy(float)
    T = int(trace.t.max())
    X = {t: g.set_index("idx").value.reindex(range(n)).to_numpy(float) for t, g in sig.groupby("t")}
    M = dict(zip(trace.t, trace.M_t))
    # 1. replay
    x, rep, mdiff = s0.copy(), 0.0, 0.0
    if not np.allclose(X[0], s0, atol=0, rtol=0):
        rep = float(np.abs(X[0] - s0).max())
    for t in range(1, T + 1):
        v = s0 + x
        u = v + P @ v
        mt = float(u.max())
        mdiff = max(mdiff, abs(mt - M[t]) / max(abs(M[t]), 1e-300))
        x = u / mt if mt != 0 else u
        rep = max(rep, float(np.abs(x - X[t]).max() / max(np.abs(X[t]).max(), 1e-300)))
    # 2. Proposition 1 on the scored nodes
    sc_nodes = nodes[nodes.scored]
    q_full = read_matrix(WORK / "seeds" / m / ds / sc / "seed.csv")[0]
    # Schematch parses the initial matrix as float32; q must carry the same rounding (else residuals of about 5e-8)
    q = q_full[sc_nodes.row.to_numpy(), sc_nodes.col.to_numpy()].astype(np.float32).astype(float)
    blk = pd.factorize(sc_nodes.dtype_pair_id.fillna(""))[0]
    gt = truth(ds, sc)
    gtv = gt[sc_nodes.row.to_numpy(), sc_nodes.col.to_numpy()] if gt is not None else None
    idx = sc_nodes.idx.to_numpy()
    rows = []
    checkpoints = sorted({1, 2, 3, T} & set(range(1, T + 1)))
    for t in range(1, T + 1):
        xs = X[t][idx]
        a, off, res, unid = fit(xs, q, blk)
        sd = float(xs.std())
        viol, pairs, tb, ties = order_check(xs, q, blk, float(np.abs(xs).max()) or 1.0)
        unit = abs(a) * q.std()
        rows.append({"label": label, "matcher": m, "attach": attach, "policy": policy, "lit": lit, "dataset": ds, "scenario": sc,
                     "t": t, "T": T, "replay_rel": rep, "M_rel": mdiff, "a": a, "abs_resid": float(np.sqrt(np.mean(res ** 2))),
                     "rel_resid": float(np.sqrt(np.mean(res ** 2)) / sd) if sd > 0 else 0.0,
                     "flag": "slope_unidentifiable" if unid else ("constant_output" if sd <= 1e-12 else ""),
                     "order_viol": viol, "strict_pairs": pairs, "ties_broken": tb, "ties": ties,
                     "s": float(off.std() / unit) if unit > 0 else np.nan,
                     "o": float((off[gtv].mean() - off.mean()) / unit) if gtv is not None and gtv.any() and unit > 0 else np.nan,
                     "blocks": int(len(np.unique(blk)))})
    # 3. decomposition by origin and path length
    w = {0: {0: 1.0}}
    for t in range(1, T + 1):
        prev = dict(w[t - 1]); prev[0] = prev.get(0, 0.0) + 1.0
        w[t] = {k + 1: c / M[t] for k, c in prev.items()}
    orig = init.origin.map(origin_class).to_numpy()
    comps, total_err = [], 0.0
    base = {o: np.where(orig == o, s0, 0.0) for o in ("name", "literal", "other") if (orig == o).any()}
    powers = {o: [vec] for o, vec in base.items()}
    for o in powers:
        for _ in range(T):
            powers[o].append(P @ powers[o][-1])
    for t in checkpoints:
        c = {j: sum(wk * comb(k, j) for k, wk in w[t].items() if k >= j) for j in range(t + 1)}
        recon = sum(c[j] * powers[o][j] for o in powers for j in c)
        total_err = max(total_err, float(np.abs(recon - X[t]).max() / max(np.abs(X[t]).max(), 1e-300)))
        xs = X[t][idx]
        a, off, _, _ = fit(xs, q, blk)
        unit = abs(a) * q.std()
        for o in powers:
            for j in c:
                comp = (c[j] * powers[o][j])[idx]
                # the pooled within-block fit is linear in x, so each component's own (slope, offsets) sum to x_t's exactly
                ac, coff, _, _ = fit(comp, q, blk)
                coff = coff - coff.mean()
                comps.append({"label": label, "attach": attach, "policy": policy, "lit": lit, "matcher": m, "dataset": ds,
                              "scenario": sc, "t": t, "T": T, "origin": o, "path": j,
                              "slope_share": ac / a if a else np.nan,
                              "offset_sd": float(coff.std() / unit) if unit > 0 else np.nan,
                              "o": float(coff[gtv].mean() / unit) if gtv is not None and gtv.any() and unit > 0 else np.nan})
    for r in rows:
        r["decomp_rel_err"] = total_err
    return rows, comps


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    runs = sorted(p.parent for p in EXP.glob("*/*/*/meta.json"))
    if a.limit:
        runs = runs[: a.limit]
    R, C = [], []
    with Pool(10) as pool:
        for r, c in pool.imap_unordered(run, runs):
            R += r
            C += c
    OUT.mkdir(parents=True, exist_ok=True)
    df, cf = pd.DataFrame(R), pd.DataFrame(C)
    df.to_csv(OUT / "runs.csv", index=False)
    cf.to_csv(OUT / "components.csv", index=False)
    lines = ["# X91: SF export check (EXPLORATORY)", "", f"- runs: {df.label.str.cat(df.dataset + '/' + df.scenario).nunique()}",
             f"- replay: max relative deviation of x_t {df.replay_rel.max():.2e}; of M_t {df.M_rel.max():.2e}",
             f"- decomposition (origin × path length) reconstructs x_t to {df.decomp_rel_err.max():.2e} (max relative)", "",
             "## Proposition 1 on the graph's own blocks (all t)", "",
             "| attach | lit | runs × t | flagged | max rel. resid (unflagged) | order violations / strict pairs | ties broken / ties |",
             "|---|---|---|---|---|---|---|"]
    for (at, lt), g in df.groupby(["attach", "lit"]):
        u = g[g.flag == ""]
        lines.append(f"| {at} | {lt} | {len(g)} | {(g.flag != '').sum()} | {u.rel_resid.max():.2e} | "
                     f"{int(g.order_viol.sum())} / {int(g.strict_pairs.sum())} | {int(g.ties_broken.sum())} / {int(g.ties.sum())} |")
    lines += ["", "## Offset strength s_t on the real blocks (median)", "", "| attach | lit | t1 | t2 | t3 | final |", "|---|---|---|---|---|---|"]
    for (at, lt), g in df.groupby(["attach", "lit"]):
        med = lambda tt: g[g.t == tt].s.median() if tt != "final" else g[g.t == g["T"]].s.median()  # noqa: E731
        lines.append(f"| {at} | {lt} | {med(1):.3f} | {med(2):.3f} | {med(3):.3f} | {med('final'):.3f} |")
    lines += ["", "## Block-offset contributions at t = 2 by origin and path length (median offset SD in units of a·SD q; mean o)", "",
              "| attach | lit | origin | path | median offset SD | mean o |", "|---|---|---|---|---|---|"]
    g2 = cf[cf.t == 2]
    for (at, lt, o, j), g in g2.groupby(["attach", "lit", "origin", "path"]):
        lines.append(f"| {at} | {lt} | {o} | {j} | {g.offset_sd.median():.3f} | {g.o.mean():+.3f} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
