"""X90 scorer (EXPLORATORY; EXPLORATORY.md X90, written 2026-10-07 before any lit_none output): melnik vs lit_none trajectories.

Per corpus (valentine readable, goby, hdxsm), matcher (levenshtein, prefix_suffix, pmf, jaro_winkler), scenario, arm and t
(t1, t2, t3, t5, t10, conv): s_t, o_t, resid, abs_resid and flag from explore_x88_mechanism.decompose (fa991e6a, unchanged; blocks from datatype_nanfix),
the R_exp gain over q (recall_expected, as X88) and the J_tf gain over q (score_r7.metrics, permutations of "<ds>/<sc>").
Runs: melnik as X88 (iter_plain / iter_x81_<c>, converged layer1_plain / x81_<c>); lit_none in x90_<c> (<m>__lit_none_t<k>,
converged <m>__lit_none). Contrasts lit_none − melnik: Δs_2, ΔJ gain at t2, ΔP with P = J(t2) − J(conv); mean over matchers per
scenario, mean over scenarios; 95% cluster bootstrap (X86 clusters; Valentine by source_id), 10,000, seed 20261007, >= 5 clusters.

    python layer3b/score_x90.py [--arms melnik,lit_none]   -> tuning/results/x90_scores/{rows.csv, report.md}
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
for p in (STUDY.parent / "harness", STUDY / "tuning", STUDY, STUDY / "reference", HERE):
    sys.path.insert(0, str(p))
import explore_x88_mechanism as x88  # noqa: E402

RES = STUDY / "tuning" / "results"
TS = x88.TS
CORP = {"valentine_plain": "valentine", "goby": "goby", "hdxsm": "hdxsm"}
B, SEED, MIN_CL = 10_000, 20261007, 5


def run_path(corpus: str, arm: str, m: str, t: str) -> Path:
    seedname, root, it_stage, conv_stage = x88.CORPORA[corpus]
    if arm == "melnik":
        return RES / (conv_stage if t == "conv" else it_stage) / (f"{m}__melnik" if t == "conv" else f"{m}__melnik_{t}")
    st = RES / f"x90_{CORP[corpus]}"
    return st / (f"{m}__lit_none" if t == "conv" else f"{m}__lit_none_{t}")


def job(args):
    corpus, ds, sc, arms = args
    import score_layer1
    import score_seeds
    from explore_composite import load, sf_out
    from r7_score import blocks
    from score_r7 import metrics
    from score_screen_metrics import permutations, recall_expected
    from smx.adapters import read_matrix
    seedname, root, _, _ = x88.CORPORA[corpus]
    root = root or score_seeds.CORPUS["plain"]
    f_d = STUDY / "reference" / "seeds" / seedname / "datatype_nanfix" / ds / sc / "seed.csv"
    g = score_seeds.ground_truth_file(root, ds, sc)
    if g is None or not f_d.is_file():
        return []
    D, rows, cols = read_matrix(f_d)
    rows, cols = list(rows), list(cols)
    gt = read_matrix(g)[0] > 0.5
    if gt.shape != D.shape:
        return []
    blk = blocks(np.nan_to_num(D)).reshape(D.shape)
    perms = permutations(gt.shape, f"{ds}/{sc}")
    out = []
    for m in x88.MATCHERS:
        q = load(STUDY / "reference" / "seeds" / seedname / m / ds / sc / "seed.csv", rows, cols)
        if q is None:
            continue
        rq, jq = recall_expected(q, gt), metrics(q, gt, perms)["J"]
        for arm in arms:
            for t in TS:
                d = run_path(corpus, arm, m, t)
                rd = score_layer1.run_dir(d) if d.is_dir() else None
                S = load(sf_out(rd, ds, sc), rows, cols) if rd else None
                if S is None:
                    continue
                a, s, res, o, abs_res, sd, flag = x88.decompose(S, q, blk, gt)
                out.append({"corpus": corpus, "dataset": ds, "scenario": sc, "matcher": m, "arm": arm, "t": t, "a": a, "s": s,
                            "o": o, "resid": res, "abs_resid": abs_res, "sd_out": sd, "flag": flag,
                            "gain": recall_expected(S, gt) - rq, "gainJ": metrics(S, gt, perms)["J"] - jq})
    return out


def clusters(corpus: str, keys: list[tuple[str, str]]) -> np.ndarray:
    if corpus == "valentine_plain":
        props = pd.read_csv(STUDY / "results" / "benchmark_properties.csv")
        src = props[props.corpus == "valentine_plain"].set_index(["dataset", "scenario"]).source_id.to_dict()
        return np.array([src[k] for k in keys])
    import x86_cluster_ci as x86
    cmap, _ = x86.cluster_map(corpus, [f"{a}/{b}" for a, b in keys])
    return np.array([cmap[f"{a}/{b}"] for a, b in keys])


def boot(x: np.ndarray, cl: np.ndarray, rng) -> tuple[float, float, int]:
    ids = np.unique(cl)
    if len(ids) < MIN_CL:
        return np.nan, np.nan, len(ids)
    sums = np.array([x[cl == i].sum() for i in ids]); cnts = np.array([(cl == i).sum() for i in ids])
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    lo, hi = np.percentile(sums[idx].sum(axis=1) / cnts[idx].sum(axis=1), [2.5, 97.5])
    return float(lo), float(hi), len(ids)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="melnik,lit_none")
    ap.add_argument("--out", default=str(RES / "x90_scores"))
    a = ap.parse_args()
    arms = a.arms.split(",")
    jobs = [(c, ds, sc, arms) for c in x88.CORPORA for (_, ds, sc) in x88.cases(c)]
    with Pool(10) as pool:
        rows = [r for rr in pool.imap_unordered(job, jobs, chunksize=4) for r in rr]
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rows.csv", index=False)
    report(df, arms, out)
    return 0


def report(df, arms, out: Path) -> None:
    """main()'s report, unchanged (moved into a function 2026-11-02 for the 549 run)."""
    rng = np.random.default_rng(SEED)
    lines = ["# X90: melnik vs lit_none trajectories (EXPLORATORY)", "",
             "| corpus | arm | t | cases | median resid | median s | mean o | mean R_exp gain | mean J gain |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in x88.CORPORA:
        for arm in arms:
            for t in TS:
                g = df[(df.corpus == c) & (df.arm == arm) & (df.t == t)]
                if g.empty:
                    continue
                lines.append(f"| {c} | {arm} | {t} | {len(g)} | {g.resid.median():.2e} | {g.s.median():.3f} | {g.o.mean():.3f} | "
                             f"{g.gain.mean():+.4f} | {g.gainJ.mean():+.4f} |")
    if set(arms) >= {"melnik", "lit_none"}:
        lines += ["", "## lit_none − melnik (mean over matchers per scenario, then scenarios; cluster 95% CI)", "",
                  "| corpus | contrast | estimate [CI] | scenarios / clusters |", "|---|---|---|---|"]
        for c in x88.CORPORA:
            g = df[df.corpus == c]
            w = g.pivot_table(index=["dataset", "scenario", "matcher"], columns=["arm", "t"], values=["s", "gainJ"])
            stats = {}
            try:
                stats["Δ s_2"] = w[("s", "lit_none", "t2")] - w[("s", "melnik", "t2")]
                stats["Δ J gain at t2"] = w[("gainJ", "lit_none", "t2")] - w[("gainJ", "melnik", "t2")]
                pm = w[("gainJ", "melnik", "t2")] - w[("gainJ", "melnik", "conv")]
                pl = w[("gainJ", "lit_none", "t2")] - w[("gainJ", "lit_none", "conv")]
                stats["P melnik = J(t2) − J(conv)"] = pm
                stats["P lit_none"] = pl
                stats["ΔP = P(lit_none) − P(melnik)"] = pl - pm
            except KeyError:
                continue
            for name, v in stats.items():
                v = v.dropna().groupby(level=["dataset", "scenario"]).mean()
                keys = list(v.index)
                lo, hi, k = boot(v.to_numpy(), clusters(c, keys), rng)
                ci = f"[{lo:+.4f}, {hi:+.4f}]" if np.isfinite(lo) else "–"
                lines.append(f"| {c} | {name} | {v.mean():+.4f} {ci} | {len(v)} / {k} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def job549(args):
    """job() for one Valentine readable scenario of the 549: q by valentine549.seed_file, truth from the 382 or the new168
    tree, melnik from iter_<r>/layer1_<r> (new: iter_pnew/layer1_pnew) and lit_none from x90_valentine (new:
    x90_valentine_pnew); the corpus label stays valentine_plain."""
    ds, sc, arms = args
    import score_layer1
    import score_seeds
    import valentine549 as v
    from explore_composite import load, sf_out
    from r7_score import blocks
    from score_r7 import metrics
    from score_screen_metrics import permutations, recall_expected
    from smx.adapters import read_matrix
    new = v.is_new(ds, sc)
    f_d = v.seed_file("plain", "datatype_nanfix", ds, sc)
    g = score_seeds.ground_truth_file(v.NEW_TREE["plain"] if new else score_seeds.CORPUS["plain"], ds, sc)
    if g is None or not f_d.is_file():
        return []
    D, rows, cols = read_matrix(f_d)
    rows, cols = list(rows), list(cols)
    gt = read_matrix(g)[0] > 0.5
    if gt.shape != D.shape:
        return []
    blk = blocks(np.nan_to_num(D)).reshape(D.shape)
    perms = permutations(gt.shape, f"{ds}/{sc}")
    it, conv, lit = ("iter_pnew", "layer1_pnew", "x90_valentine_pnew") if new else ("iter_plain", "layer1_plain", "x90_valentine")
    out = []
    for m in x88.MATCHERS:
        q = load(v.seed_file("plain", m, ds, sc), rows, cols)
        if q is None:
            continue
        rq, jq = recall_expected(q, gt), metrics(q, gt, perms)["J"]
        for arm in arms:
            for t in TS:
                if arm == "melnik":
                    d = RES / (conv if t == "conv" else it) / (f"{m}__melnik" if t == "conv" else f"{m}__melnik_{t}")
                else:
                    d = RES / lit / (f"{m}__lit_none" if t == "conv" else f"{m}__lit_none_{t}")
                rd = score_layer1.run_dir(d) if d.is_dir() else None
                S = load(sf_out(rd, ds, sc), rows, cols) if rd else None
                if S is None:
                    continue
                a, s, res, o, abs_res, sd, flag = x88.decompose(S, q, blk, gt)
                out.append({"corpus": "valentine_plain", "dataset": ds, "scenario": sc, "matcher": m, "arm": arm, "t": t,
                            "a": a, "s": s, "o": o, "resid": res, "abs_resid": abs_res, "sd_out": sd, "flag": flag,
                            "gain": recall_expected(S, gt) - rq, "gainJ": metrics(S, gt, perms)["J"] - jq})
    return out


def main549() -> int:
    """SMX_VALENTINE=549: X90 for readable Valentine on the 549 scenarios (lit_none of the 168 from x90_valentine_pnew),
    GOBY / HDXSM rows from the stored x90_scores/rows.csv (melnik and lit_none, no FD edges); output x90_549_scores/.
    Valentine's cluster units for the 168: source_id by the rule behind benchmark_properties.csv."""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(STUDY / "reference"))
    import valentine549 as v
    if not (RES / "x90_valentine_pnew" / "COLLECTED").is_file():
        print("x90 549: pending (x90_valentine_pnew not collected)")
        return 0
    arms = ["melnik", "lit_none"]
    with Pool(8) as pool:
        rows = [r for rr in pool.imap_unordered(job549, [(ds, sc, arms) for ds, sc, _ in v.cases549()], chunksize=4)
                for r in rr]
    old = pd.read_csv(RES / "x90_scores" / "rows.csv")
    df = pd.concat([pd.DataFrame(rows), old[old.corpus != "valentine_plain"]], ignore_index=True)
    out = RES / "x90_549_scores"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rows.csv", index=False)
    global clusters
    _clusters = clusters

    def clusters(corpus, keys):   # noqa: F811  (the 168's source ids by the same rule)
        if corpus == "valentine_plain":
            from explore_benchmark_properties import source_id
            props = pd.read_csv(STUDY / "results" / "benchmark_properties.csv")
            src = props[props.corpus == "valentine_plain"].set_index(["dataset", "scenario"]).source_id.to_dict()
            return np.array([src.get(k) or source_id("valentine_plain", *k) for k in keys])
        return _clusters(corpus, keys)
    report(df, arms, out)
    return 0


if __name__ == "__main__":
    import os
    raise SystemExit(main549() if os.environ.get("SMX_VALENTINE", "382") == "549" else main())
