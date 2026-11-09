"""X93 P1 (preregistered in EXPLORATORY.md X93; omitted from score_x93.py by mistake and added 2026-10-08 after the main scoring):
within each single-table scenario, Spearman across the real datatype blocks (the export's dtype_pair_id) between L~'s block value
and SF(t2)'s block offset b_2 / a_2 (pooled within-block fit of the stopped output on q), per matcher; median per corpus over
scenarios with >= 3 blocks. Also, descriptively (not preregistered): A_L - SF_t2 from score_x93's rows.csv.

    python layer3b/x93_mechanism.py   -> tuning/results/x93_scores/mechanism.csv, mechanism.md
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
for p in (STUDY.parent / "harness", STUDY / "tuning", STUDY, HERE):
    sys.path.insert(0, str(p))
import score_x93 as s93  # noqa: E402

SINGLE = ["goby", "hdxsm", "gdcsm", "madi", "valentine_plain", "final_controlled"]


def job(args):
    corpus, ds, sc, ms = args
    import check_sf_export as cx
    from score_r7 import run_dir
    from score_seeds import one
    from smx.adapters import read_matrix
    d = s93.EXPORT / corpus / ds / sc
    if not (d / "nodes.tsv").is_file():
        return []
    L, nodes = s93.literal_term(d)
    blk = pd.factorize(nodes.dtype_pair_id.fillna(""))[0]
    if len(np.unique(blk)) < 3:
        return []
    r, c = nodes.row.to_numpy(), nodes.col.to_numpy()
    seedname, _, _, t2, _ = s93.CORP[corpus]
    out = []
    for m in ms:
        f = STUDY / "reference" / "seeds" / seedname / m / ds / sc / "seed.csv"
        rd = run_dir(s93.RES / t2 / f"{m}__melnik_t2") if (s93.RES / t2 / f"{m}__melnik_t2").is_dir() else None
        o = one(rd / ds / sc / "_outputs" / "MatchingStepLine2") if rd else None
        if not f.is_file() or o is None:
            continue
        q = read_matrix(f)[0][r, c].astype(np.float32).astype(float)
        S = read_matrix(o)[0][r, c].astype(float)
        a, off, res, unid = cx.fit(S, q, blk)
        if unid or a <= 0:
            continue
        ids = np.unique(blk)
        bv = np.array([off[blk == k][0] / a for k in ids])
        lv = np.array([L[blk == k].mean() for k in ids])
        if bv.std() == 0 or lv.std() == 0:
            continue
        rho = pd.Series(bv).rank().corr(pd.Series(lv).rank())
        out.append({"corpus": corpus, "dataset": ds, "scenario": sc, "matcher": m, "blocks": len(ids), "rho": rho,
                    "rel_resid": float(np.sqrt(np.mean(res ** 2)) / (S.std() or 1))})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    jobs = []
    for c in SINGLE:
        ms = s93.matchers(c)[:15]
        jobs += [(c, ds, sc, ms) for ds, sc in s93.scenarios(c)]
    with Pool(10) as pool:
        df = pd.DataFrame([x for xs in pool.imap_unordered(job, jobs, chunksize=4) for x in xs])
    df.to_csv(s93.OUT / "mechanism.csv", index=False)
    report(df, pd.read_csv(s93.OUT / "rows.csv"), s93.OUT)
    return 0


def report(df, rows, out: Path) -> None:
    """main()'s mechanism.md, unchanged (moved into a function 2026-11-01 for the 549 run)."""
    g = df.groupby(["corpus", "scenario"]).rho.median().groupby(level="corpus")
    lines = ["# X93 P1: L~ vs SF(t2)'s block offsets on the real blocks (Spearman across blocks, scenarios with >= 3 blocks)", "",
             "| corpus | scenarios | median ρ | share ρ ≥ 0.9 | max rel. resid of the fit |", "|---|---|---|---|---|"]
    for c, v in g:
        lines.append(f"| {c} | {len(v)} | {v.median():+.3f} | {(v >= 0.9).mean():.0%} | {df[df.corpus == c].rel_resid.max():.1e} |")
    w = rows.pivot_table(index=["corpus", "dataset", "scenario", "matcher"], columns="arm", values="J")
    v = (w["A_L"] - w["SF_t2"]).dropna().groupby(level=["corpus", "dataset", "scenario"]).mean().groupby(level="corpus").mean()
    lines += ["", "Descriptive (not preregistered): mean A_L − SF_t2 in J: " + "; ".join(f"{k} {x:+.4f}" for k, x in v.items())]
    (out / "mechanism.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def job549(args):
    """job() for one Valentine readable scenario of the 549: L from the graph export (valentine_plain for the 381,
    valentine_pnew for the 168), q and SF(t2) by the assembly rules (valentine549.seed_file / sf_run)."""
    ds, sc, ms = args
    import check_sf_export as cx
    import valentine549 as v
    from score_seeds import one
    from smx.adapters import read_matrix
    d = s93.EXPORT / ("valentine_pnew" if v.is_new(ds, sc) else "valentine_plain") / ds / sc
    if not (d / "nodes.tsv").is_file():
        return []
    L, nodes = s93.literal_term(d)
    blk = pd.factorize(nodes.dtype_pair_id.fillna(""))[0]
    if len(np.unique(blk)) < 3:
        return []
    r, c = nodes.row.to_numpy(), nodes.col.to_numpy()
    out = []
    for m in ms:
        f = v.seed_file("plain", m, ds, sc)
        rd = v.sf_run("plain", m, "melnik_t2", ds, sc)
        o = one(rd / ds / sc / "_outputs" / "MatchingStepLine2") if rd else None
        if not f.is_file() or o is None:
            continue
        q = read_matrix(f)[0][r, c].astype(np.float32).astype(float)
        S = read_matrix(o)[0][r, c].astype(float)
        a, off, res, unid = cx.fit(S, q, blk)
        if unid or a <= 0:
            continue
        ids = np.unique(blk)
        bv = np.array([off[blk == k][0] / a for k in ids])
        lv = np.array([L[blk == k].mean() for k in ids])
        if bv.std() == 0 or lv.std() == 0:
            continue
        rho = pd.Series(bv).rank().corr(pd.Series(lv).rank())
        out.append({"corpus": "valentine_plain", "dataset": ds, "scenario": sc, "matcher": m, "blocks": len(ids),
                    "rho": rho, "rel_resid": float(np.sqrt(np.mean(res ** 2)) / (S.std() or 1))})
    return out


def main549() -> int:
    """SMX_VALENTINE=549: P1 for Valentine readable on the 549 scenarios (same 15 matchers as the 382 run), the other
    corpora's rows from x93_scores/mechanism.csv unchanged; output next to score_x93's 549 rows (x93_549_scores)."""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(STUDY / "reference"))
    import valentine549 as v
    out = s93.OUT549
    if not (out / "rows.csv").is_file():
        print("x93 mechanism 549: pending (score_x93.py --score under SMX_VALENTINE=549 first)")
        return 0
    ms = s93.matchers("valentine_plain")[:15]
    with Pool(10) as pool:
        new = pd.DataFrame([x for xs in pool.imap_unordered(job549, [(ds, sc, ms) for ds, sc, _ in v.cases549()],
                                                            chunksize=4) for x in xs])
    old = pd.read_csv(s93.OUT / "mechanism.csv")
    df = pd.concat([old[old.corpus != "valentine_plain"], new], ignore_index=True)
    df.to_csv(out / "mechanism.csv", index=False)
    report(df, pd.read_csv(out / "rows.csv"), out)
    return 0


if __name__ == "__main__":
    import os
    raise SystemExit(main549() if os.environ.get("SMX_VALENTINE", "382") == "549" else main())
