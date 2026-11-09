"""X93 scorer (EXPLORATORY; EXPLORATORY.md X93, design and predictions written 2026-10-08 before any output): an
information-matched one-step term against SF.

L = P^2 sigma0_lit on the scored column pairs, from a graph-only SF export (nodes, edges, init; melnik configuration):
sigma0_lit keeps the initial vector's literal-seed entries (origin literal_seed:*), P is the exported propagation matrix
(incoming CSR, u_dst += c * v_src). L~ = L / max(L) per scenario. Arms per matcher: A_d = q+ + 5D, A_L = q+ + lam_L L~,
A_dL = q+ + 5D + lam_dL L~; comparators SF converged (melnik) and SF after two updates (melnik_t2).

    python layer3b/score_x93.py --validate          # on 504-run panel (results/sf_export): L against X91's components
    python layer3b/score_x93.py --fit               # lam_L, lam_dL on the 60 Valentine readable screen scenarios only
    python layer3b/score_x93.py --score             # all corpora with the frozen lambdas
Writes tuning/results/x93_scores/{validate.md, lambda.json, rows.csv, contrasts.csv, mechanism.csv, report.md}.
"""

from __future__ import annotations

import argparse
import json
import os
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

RES = STUDY / "tuning" / "results"
OUT = RES / "x93_scores"
GEN = STUDY.parent / "harness" / "generated"
EXPORT = STUDY / "results" / "sf_export_graph"
GRID = (0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10)
SEED, B, MIN_CL = 20261008, 10_000, 5
# corpus: (seed folder, truth root (None = score_seeds.CORPUS[regime]), converged stage, t2 stage, x86 cluster corpus)
CORP = {
    "valentine_plain": ("plain", "plain", "layer1_plain", "iter_plain", "valentine"),
    "valentine_md5": ("md5", "md5", "layer1_md5", "iter_md5", "valentine"),
    "final_controlled": ("final_controlled", GEN / "final-controlled", "final_controlled", "iter_final_controlled", "final_controlled"),
    "goby": ("goby", GEN / "goby-imported", "x81_goby", "iter_x81_goby", "goby"),
    "hdxsm": ("hdxsm", GEN / "hdxsm-usable", "x81_hdxsm", "iter_x81_hdxsm", "hdxsm"),
    "gdcsm": ("gdcsm", GEN / "gdcsm-imported", "x81_gdcsm", "iter_x81_gdcsm", "gdcsm"),
    "madi": ("madi", GEN / "madi-imported", "x81_madi", "iter_x81_madi", "madi"),
    "final_indep": ("final_indep", STUDY / "data" / "final_indep", "final_indep", "iter_final_indep", "final_indep"),
    "mimic": ("mimic", STUDY / "data" / "mimic_omop", "x81_mimic", "iter_x81_mimic", "mimic"),
}
INDEPENDENT = ["goby", "hdxsm", "gdcsm", "madi_indep"]
CONTROLS = ["valentine_plain", "valentine_md5", "final_controlled", "madi_products"]
DESCRIPTIVE = ["final_indep", "mimic"]


def literal_term(d: Path):
    """(L on the scored nodes, scored nodes frame). L = P^2 sigma0_lit."""
    nodes = pd.read_csv(d / "nodes.tsv", sep="\t")
    e = pd.read_csv(d / "edges.tsv", sep="\t")
    init = pd.read_csv(d / "init.tsv", sep="\t").set_index("idx").reindex(nodes.idx)
    n = len(nodes)
    P = csr_matrix((e.coefficient.to_numpy(float), (e.dst.to_numpy(), e.src.to_numpy())), shape=(n, n))
    lit = np.where(init.origin.fillna("").str.startswith("literal_seed"), init.sigma0.to_numpy(float), 0.0)
    sc = nodes[nodes.scored]
    return (P @ (P @ lit))[sc.idx.to_numpy()], sc


def l_matrix(corpus: str, ds: str, sc: str, shape) -> np.ndarray | None:
    d = EXPORT / corpus / ds / sc
    if not (d / "nodes.tsv").is_file():
        return None
    L, nodes = literal_term(d)
    M = np.zeros(shape)
    M[nodes.row.to_numpy(), nodes.col.to_numpy()] = L
    mx = M.max()
    return M / mx if mx > 0 else M


# ----------------------------------------------------------------------------------------------------------------- validate
def validate() -> int:
    """On 504-run panel: (1) L is constant within the real datatype blocks (single tables); (2) c22 * L reproduces X91's
    literal path-2 offset strength at t = 2."""
    sys.path.insert(0, str(HERE))
    import check_sf_export as cx
    comp = pd.read_csv(STUDY / "results" / "sf_export_check" / "components.csv")
    comp = comp[(comp.t == 2) & (comp.origin == "literal") & (comp.path == 2)].set_index(["label", "dataset", "scenario"])
    rows = []
    for meta in sorted((STUDY / "results" / "sf_export").glob("*_PREFIX_SUFFIX/*/*/meta.json")):
        d = meta.parent
        label = d.parents[1].name
        L, sc = literal_term(d)
        blk = pd.factorize(sc.dtype_pair_id.fillna(""))[0]
        within = max((float(L[blk == k].std()) for k in np.unique(blk)), default=0.0) / (float(np.abs(L).max()) or 1.0)
        tr = pd.read_csv(d / "trace.tsv", sep="\t")
        M = dict(zip(tr.t, tr.M_t))
        sig = pd.read_csv(d / "sigma_t.tsv", sep="\t")
        x2 = sig[sig.t == 2].set_index("idx").value.reindex(sc.idx).to_numpy(float) if 2 in M else None
        if x2 is None:
            continue
        from smx.adapters import read_matrix
        m = label.split("__")[0]
        q = read_matrix(cx.WORK / "seeds" / m / d.parent.name / d.name / "seed.csv")[0][sc.row, sc.col].astype(np.float32).astype(float)
        a, _, _, _ = cx.fit(x2, q, blk)
        c22 = 2.0 / (M[1] * M[2])
        _, off, _, _ = cx.fit(c22 * L, q, blk)
        unit = abs(a) * q.std()
        mine = float((off - off.mean()).std() / unit) if unit > 0 else np.nan
        key = (label, d.parent.name, d.name)
        ref = float(comp.loc[key, "offset_sd"]) if key in comp.index else np.nan
        rows.append({"label": label, "dataset": d.parent.name, "scenario": d.name, "attach": label.split("__")[1].split("_")[0],
                     "within_block_rel_sd": within, "offset_sd_from_L": mine, "offset_sd_x91": ref,
                     "abs_diff": abs(mine - ref) if np.isfinite(ref) else np.nan})
    v = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    v.to_csv(OUT / "validate.csv", index=False)
    lines = ["# X93 validation on A's export panel", "",
             f"- runs (literal seeds on): {len(v)}",
             f"- L within-block relative SD, max: {v.within_block_rel_sd.max():.2e} (NAME {v[v.attach == 'NAME'].within_block_rel_sd.max():.2e})",
             f"- c22·L offset strength vs X91's literal path-2 component at t = 2: max |diff| {v.abs_diff.max():.2e}"]
    (OUT / "validate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# --------------------------------------------------------------------------------------------------------------- scoring
def scenarios(corpus: str):
    import score_seeds
    seedname, root, *_ = CORP[corpus]
    if isinstance(root, str):
        return [(ds, sc) for ds, sc, _ in score_seeds.cases_for(root)]
    return [(d.name, s.name) for d in sorted(p for p in root.iterdir() if p.is_dir())
            for s in sorted(p for p in d.iterdir() if p.is_dir())]


def matchers(corpus: str) -> list[str]:
    from score_r7 import is_control
    conv = RES / CORP[corpus][2]
    return sorted({p.name.split("__")[0] for p in conv.iterdir() if p.name.endswith("__melnik")
                   and not is_control(p.name.split("__")[0]) and not p.name.startswith(("datatype", "constant", "random"))})


def job(args):
    corpus, ds, sc, lams, ms = args
    import score_seeds
    from score_r7 import metrics, run_dir
    from score_screen_metrics import permutations
    from smx.adapters import read_matrix
    seedname, root, conv, t2, _ = CORP[corpus]
    root = score_seeds.CORPUS[root] if isinstance(root, str) else root
    g = score_seeds.ground_truth_file(root, ds, sc)
    seeds = STUDY / "reference" / "seeds" / seedname
    fD = seeds / "datatype_nanfix" / ds / sc / "seed.csv"
    if g is None or not fD.is_file():
        return []
    gt = read_matrix(g)[0] > 0.5
    D = read_matrix(fD)[0]
    Lt = l_matrix(corpus, ds, sc, gt.shape)
    if Lt is None or D.shape != gt.shape:
        return []
    perms = permutations(gt.shape, f"{ds}/{sc}")
    out = []
    for m in ms:
        f = seeds / m / ds / sc / "seed.csv"
        if not f.is_file():
            continue
        q = read_matrix(f)[0]
        if q.shape != gt.shape or not np.isfinite(q).all() or not np.isfinite(D).all():
            continue
        qp = np.clip(q, 0, None)
        mats = {"A_d": qp + 5.0 * D}
        for lam_key, base in (("A_L", qp), ("A_dL", qp + 5.0 * D)):
            for lam in (lams[lam_key] if isinstance(lams[lam_key], (list, tuple)) else [lams[lam_key]]):
                mats[f"{lam_key}@{lam}"] = base + lam * Lt
        for arm, stage, lab in (("SF_conv", conv, f"{m}__melnik"), ("SF_t2", t2, f"{m}__melnik_t2")):
            rd = run_dir(RES / stage / lab) if (RES / stage / lab).is_dir() else None
            from score_seeds import one
            o = one(rd / ds / sc / "_outputs" / "MatchingStepLine2") if rd else None
            if o is not None:
                S = read_matrix(o)[0]
                if S.shape == gt.shape and np.isfinite(S).all():
                    mats[arm] = S
        for arm, M in mats.items():
            r = metrics(M, gt, perms)
            out.append({"corpus": corpus, "dataset": ds, "scenario": sc, "matcher": m, "arm": arm, "J": r["J"],
                        "R_exp": r["R_exp"]})
    return out


def screen_cases() -> set:
    proto = json.loads((STUDY.parent / "harness" / "reports" / "audited_md5_fused_20260904" / "protocol.json")
                       .read_text(encoding="utf-8"))["cases"]
    keys = set(json.loads((STUDY / "tuning" / "splits.json").read_text(encoding="utf-8"))["screen"])
    return {(c["dataset"], c["scenario"]) for c in proto if c["key"] in keys}


def run_jobs(jobs):
    with Pool(10) as pool:
        return pd.DataFrame([r for rr in pool.imap_unordered(job, jobs, chunksize=4) for r in rr])


def fit() -> int:
    scr = screen_cases()
    ms = matchers("valentine_plain")
    jobs = [("valentine_plain", ds, sc, {"A_L": list(GRID), "A_dL": list(GRID)}, ms)
            for ds, sc in scenarios("valentine_plain") if (ds, sc) in scr]
    df = run_jobs(jobs)
    best = {}
    for key in ("A_L", "A_dL"):
        g = df[df.arm.str.startswith(key + "@")].copy()
        g["lam"] = g.arm.str.split("@").str[1].astype(float)
        mean = g.groupby("lam").J.mean()
        best[key] = float(mean[mean == mean.max()].index.min())       # ties -> smaller lambda
        best[key + "_curve"] = {str(k): float(v) for k, v in mean.items()}
    best.update({"screen_scenarios": len(jobs), "matchers": len(ms), "objective": "mean J_tf over matchers x screen scenarios"})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lambda.json").write_text(json.dumps(best, indent=1), encoding="utf-8")
    print(json.dumps(best, indent=1))
    return 0


def score() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    import x86_cluster_ci as x86
    lam = json.loads((OUT / "lambda.json").read_text(encoding="utf-8"))
    lams = {"A_L": lam["A_L"], "A_dL": lam["A_dL"]}
    scr = screen_cases()
    jobs = []
    for c in CORP:
        ms = matchers(c)
        for ds, sc in scenarios(c):
            if c == "valentine_plain" and (ds, sc) in scr:
                continue                                              # the tuning set is not scored
            jobs.append((c, ds, sc, lams, ms))
    df = run_jobs(jobs)
    df["arm"] = df.arm.str.replace(r"@.*$", "", regex=True)
    m = df.corpus == "madi"
    df.loc[m, "corpus"] = np.where(df.loc[m, "scenario"].str.startswith("products__"), "madi_products", "madi_indep")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "rows.csv", index=False)
    summarise(df, lams, OUT)
    return 0


def summarise(df, lams, out: Path, src_extra: dict | None = None) -> None:
    """score()'s contrasts and report, unchanged (moved into a function 2026-11-01 for the 549 run). src_extra: source
    ids of Valentine scenarios outside benchmark_properties.csv (the 168 new ones)."""
    import x86_cluster_ci as x86
    OUT = out
    props = pd.read_csv(STUDY / "results" / "benchmark_properties.csv")
    src = props[props.corpus == "valentine_plain"].set_index(["dataset", "scenario"]).source_id.to_dict()
    if src_extra:
        src = {**src, **src_extra}
    rng = np.random.default_rng(SEED)
    contrasts = [("A_dL", "SF_conv"), ("A_dL", "SF_t2"), ("A_L", "SF_conv"), ("A_dL", "A_d"), ("A_d", "SF_conv"), ("SF_t2", "SF_conv")]
    res = []
    for corpus in INDEPENDENT + CONTROLS + DESCRIPTIVE:
        g = df[df.corpus == corpus]
        if g.empty:
            continue
        for metric in ("J", "R_exp"):
            w = g.pivot_table(index=["dataset", "scenario", "matcher"], columns="arm", values=metric)
            for a, b in contrasts:
                if a not in w or b not in w:
                    continue
                v = (w[a] - w[b]).dropna()
                okm = v.groupby(level="matcher").size()
                full = okm[okm == okm.max()].index           # matchers complete on every scenario of the contrast
                v = v[v.index.get_level_values("matcher").isin(full)].groupby(level=["dataset", "scenario"]).mean()
                keys = list(v.index)
                if corpus.startswith("valentine"):
                    cl = np.array([src.get(k, k[0]) for k in keys])
                else:
                    ck = corpus if corpus in ("goby", "hdxsm", "gdcsm", "madi_indep", "madi_products", "final_indep",
                                              "final_controlled", "mimic") else corpus
                    cmap, _ = x86.cluster_map(ck, [k[1] if corpus.startswith("madi") else f"{k[0]}/{k[1]}" for k in keys])
                    cl = np.array([cmap[k[1] if corpus.startswith("madi") else f"{k[0]}/{k[1]}"] for k in keys])
                ids = np.unique(cl)
                x = v.to_numpy()
                lo = hi = np.nan
                if len(ids) >= MIN_CL:
                    sums = np.array([x[cl == i].sum() for i in ids]); cnts = np.array([(cl == i).sum() for i in ids])
                    idx = rng.integers(0, len(ids), size=(B, len(ids)))
                    lo, hi = np.percentile(sums[idx].sum(axis=1) / cnts[idx].sum(axis=1), [2.5, 97.5])
                res.append({"corpus": corpus, "metric": metric, "contrast": f"{a} − {b}", "estimate": float(x.mean()),
                            "lo": lo, "hi": hi, "scenarios": len(x), "clusters": len(ids), "matchers": len(full)})
    r = pd.DataFrame(res)
    r.to_csv(OUT / "contrasts.csv", index=False)
    lines = ["# X93: information-matched one-step term vs SF (EXPLORATORY)", "",
             f"- frozen on the Valentine screen: λ_L = {lams['A_L']}, λ_dL = {lams['A_dL']}", "",
             "| corpus | metric | contrast | estimate [cluster 95% CI] | scenarios / clusters | matchers |", "|---|---|---|---|---|---|"]
    for _, x in r.iterrows():
        ci = f"[{x.lo:+.4f}, {x.hi:+.4f}]" if np.isfinite(x.lo) else "–"
        lines.append(f"| {x.corpus} | {x.metric} | {x.contrast} | {x.estimate:+.4f} {ci} | {x.scenarios} / {x.clusters} | {x.matchers} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ------------------------------------------------------------------------------------------------- rerun 549 (2026-11-01)
OUT549 = RES / "x93_549_scores"
if str(STUDY / "reference") not in sys.path:
    sys.path.insert(0, str(STUDY / "reference"))   # valentine549 (rerun 549 helpers)
VAL549 = {"valentine_plain": ("plain", "valentine_pnew"), "valentine_md5": ("md5", "valentine_mnew")}


def job549(args):
    """job() for one Valentine scenario of the 549: A_d, A_L, A_dL from the initial matrices (valentine549.seed_file) and
    L from the graph export (valentine_plain/md5 for the 381, valentine_pnew/mnew for the 168); SF_conv and SF_t2 are
    taken from the assembled per-scenario scores (layer-1 melnik, iter melnik_t2: the same metrics and tie orders)."""
    corpus, ds, sc, lams, ms, sfv = args
    import score_seeds
    import valentine549 as v
    from score_r7 import metrics
    from score_screen_metrics import permutations
    from smx.adapters import read_matrix
    regime, newkey = VAL549[corpus]
    new = v.is_new(ds, sc)
    g = score_seeds.ground_truth_file(v.NEW_TREE[regime] if new else score_seeds.CORPUS[regime], ds, sc)
    fD = v.seed_file(regime, "datatype_nanfix", ds, sc)
    if g is None or not fD.is_file():
        return []
    gt = read_matrix(g)[0] > 0.5
    D = read_matrix(fD)[0]
    Lt = l_matrix(newkey if new else corpus, ds, sc, gt.shape)
    if Lt is None or D.shape != gt.shape:
        return []
    perms = permutations(gt.shape, f"{ds}/{sc}")
    out = []
    for m in ms:
        f = v.seed_file(regime, m, ds, sc)
        if not f.is_file():
            continue
        q = read_matrix(f)[0]
        if q.shape != gt.shape or not np.isfinite(q).all() or not np.isfinite(D).all():
            continue
        qp = np.clip(q, 0, None)
        mats = {"A_d": qp + 5.0 * D, "A_L": qp + lams["A_L"] * Lt, "A_dL": qp + 5.0 * D + lams["A_dL"] * Lt}
        for arm, M in mats.items():
            r = metrics(M, gt, perms)
            out.append({"corpus": corpus, "dataset": ds, "scenario": sc, "matcher": m, "arm": arm, "J": r["J"],
                        "R_exp": r["R_exp"]})
        for arm, val in sfv.get(m, {}).items():
            out.append({"corpus": corpus, "dataset": ds, "scenario": sc, "matcher": m, "arm": arm, "J": val[0],
                        "R_exp": val[1]})
    return out


def score549() -> int:
    """SMX_VALENTINE=549: X93's Valentine rows on the 549 scenarios (frozen lambdas from x93_scores/lambda.json, not refit;
    the 60 screen scenarios stay out of valentine_plain), the other corpora's rows from x93_scores/rows.csv unchanged;
    contrasts and report by summarise() (same order and random stream as score()), cluster units = source_id
    (explore_benchmark_properties.source_id, the rule behind benchmark_properties.csv) for the 168 new scenarios.
    Output tuning/results/x93_549_scores/ (rows.csv: all rows; contrasts.csv; report.md; lambda.json copied;
    mechanism.md without the Valentine rows, whose mechanism check is not rerun)."""
    import csv as _csv
    import valentine549 as v
    from explore_benchmark_properties import source_id
    from score_r7 import is_control
    sys.stdout.reconfigure(encoding="utf-8")
    lam = json.loads((OUT / "lambda.json").read_text(encoding="utf-8"))
    lams = {"A_L": lam["A_L"], "A_dL": lam["A_dL"]}
    scr = screen_cases()
    jobs = []
    for corpus, (regime, _) in VAL549.items():
        l1, it = v.assembled(f"layer1_{v.vreg(regime)}_scores"), v.assembled(f"iter_{v.vreg(regime)}_scores")
        if l1 is None or it is None:
            print(f"x93 549: {corpus} pending (layer 1 or iter not assembled)")
            return 0
        sfv = {}
        for rows, arm, lab in ((l1, "melnik", "SF_conv"), (it, "melnik_t2", "SF_t2")):
            for r in rows:
                if r["arm"] == arm:
                    sfv.setdefault((r["dataset"], r["scenario"]), {}).setdefault(r["seed"], {})[lab] = (
                        (float(r["sf_R_exp"]) + float(r["sf_F1_Hr"])) / 2, float(r["sf_R_exp"]))
        ms = sorted({r["seed"] for r in l1 if r["arm"] == "melnik"} - {s for s in {r["seed"] for r in l1}
                    if is_control(s) or s.startswith(("datatype", "constant", "random"))})
        for ds, sc, _ in v.cases549():
            if corpus == "valentine_plain" and (ds, sc) in scr:
                continue
            jobs.append((corpus, ds, sc, lams, ms, sfv.get((ds, sc), {})))
    with Pool(10) as pool:
        val = pd.DataFrame([r for rr in pool.imap_unordered(job549, jobs, chunksize=4) for r in rr])
    old = pd.read_csv(OUT / "rows.csv")
    df = pd.concat([old[~old.corpus.isin(list(VAL549))], val], ignore_index=True)
    OUT549.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT549 / "rows.csv", index=False)
    (OUT549 / "lambda.json").write_bytes((OUT / "lambda.json").read_bytes())
    m549 = OUT549 / "mechanism.md"
    if not (m549.is_file() and "| valentine_plain |" in m549.read_text(encoding="utf-8")):   # x93_mechanism.py's 549 run
        mech = (OUT / "mechanism.md").read_text(encoding="utf-8").splitlines()   # not there yet: drop the 382 Valentine rows
        m549.write_text("\n".join(x for x in mech if "valentine" not in x) + "\n", encoding="utf-8")
    extra = {(ds, sc): source_id("valentine_plain", ds, sc) for ds, sc, _ in v.cases549() if v.is_new(ds, sc)}
    summarise(df, lams, OUT549, src_extra=extra)
    # The cluster intervals depend on the process's string-hash seed (x86 cluster labels), so the other corpora's rows
    # recomputed here need not match the stored ones bit for bit although their inputs are unchanged: keep the stored rows
    # for every non-Valentine corpus (same inputs, same estimates) and the recomputed rows only for Valentine.
    new = pd.read_csv(OUT549 / "contrasts.csv")
    old_c = pd.read_csv(OUT / "contrasts.csv")
    merged = pd.concat([old_c[~old_c.corpus.isin(list(VAL549))], new[new.corpus.isin(list(VAL549))]], ignore_index=True)
    order = {c: i for i, c in enumerate(dict.fromkeys(old_c.corpus))}
    merged = merged.sort_values("corpus", key=lambda s: s.map(order), kind="stable")
    merged.to_csv(OUT549 / "contrasts.csv", index=False)
    with (OUT549 / "report.md").open("a", encoding="utf-8") as fh:
        fh.write("\n549 run: contrasts.csv keeps the stored x93_scores rows for every non-Valentine corpus (their inputs are "
                 "unchanged; the cluster intervals depend on the string-hash seed); the Valentine rows are recomputed on "
                 "the 549 scenarios (PYTHONHASHSEED as set by reference/run_549.py).\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true")
    g.add_argument("--fit", action="store_true")
    g.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.score and os.environ.get("SMX_VALENTINE", "382") == "549":
        return score549()
    return validate() if a.validate else fit() if a.fit else score()


if __name__ == "__main__":
    raise SystemExit(main())
