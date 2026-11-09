"""Review check 2 (EXPLORATORY, after the ICDE review of the compressed draft, 2026-10-26): weight sensitivity of the
datatype term. The paper freezes beta = 5 (chosen on Valentine's screen) in A_d = max(q,0) + beta * D and compares it with SF
as published; does the sign of (A_d - SF) depend on that value?

Per corpus, matcher (non-control) and scenario: q = reference/seeds/<c>/<m>, D = reference/seeds/<c>/datatype_nanfix,
SF = SF as published on q (the base stage's <m>__melnik run, as x82_metrics / the layer-1 runs), all scored with
score_r7.metrics (same seeded tie permutations as every other contrast). Contrast A_d(beta) - SF in J (= J_tf difference),
R_exp and F1_Hr with score_x82.contrast (95% bootstrap, scenarios stratified by dataset; by database on the new databases).
Corpora: Valentine readable and recoded (the external panel's matchers, so the run stays small), the four single-table
external corpora (GDC-SM, HDXSM, GOBY, MaDI use cases), the MaDI products and the new databases (final_controlled). The
multi-table corpora are left out: there the datatype term is deliberately not the information-matched comparison.
Output: tuning/results/review_checks/beta_sweep.{csv,md}

    python review_checks/beta_sweep.py
"""

from __future__ import annotations

import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent.parent
L3 = STUDY / "layer3b"
sys.path.insert(0, str(L3))
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY.parent / "harness"))
import make_configs  # noqa: E402
import score_x82 as sx  # noqa: E402
import x82_metrics as xm  # noqa: E402
from score_r7 import metrics, run_dir  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402
from score_seeds import ground_truth_file, one  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

RES = STUDY / "tuning" / "results"
OUT = RES / "review_checks"
BETAS = [0, 1, 2, 3, 5, 8, 12, 20]
EXTERNAL = ["gdcsm", "hdxsm", "goby", "madi", "final_controlled"]
VALENTINE = {"valentine_plain": ("plain", "layer1_plain"), "valentine_md5": ("md5", "layer1_md5")}


def job(args):
    c, root, seed_corpus, ds, sc, sf_runs = args
    g = ground_truth_file(root, ds, sc)
    seeds = STUDY / "reference" / "seeds" / seed_corpus
    fd = seeds / "datatype_nanfix" / ds / sc / "seed.csv"
    if g is None or not fd.is_file():
        return [], Counter({"no_truth_or_D": 1})
    gt = read_matrix(g)[0] > 0.5
    D = read_matrix(fd)[0]
    perms = permutations(gt.shape, f"{ds}/{sc}")
    dataset = ds.split("-")[0] if c == "final_controlled" else ds
    scen = sc if c == "madi" else f"{ds}/{sc}"
    rows, skip = [], Counter()
    for m, r in sf_runs.items():
        f = seeds / m / ds / sc / "seed.csv"
        o = one(r / ds / sc / "_outputs" / "MatchingStepLine2") if r is not None else None
        if not f.is_file() or o is None:
            skip["missing"] += 1
            continue
        q = read_matrix(f)[0]
        S = read_matrix(o)[0]
        if q.shape != gt.shape or S.shape != q.shape or D.shape != q.shape or not (
                np.isfinite(q).all() and np.isfinite(S).all() and np.isfinite(D).all()):
            skip["shape_or_nonfinite"] += 1
            continue
        mats = {"SF": S, **{f"A_{b}": np.clip(q, 0, None) + b * D for b in BETAS}}
        for arm, M in mats.items():
            x = metrics(M, gt, perms)
            rows.append({"corpus": c, "dataset": dataset, "scenario": scen, "matcher": m, "arm": arm,
                         "J": x["J"], "R_exp": x["R_exp"], "F1_Hr": x["F1_Hr"]})
    return rows, skip


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    jobs, notes, panel = [], [], set()
    for c in EXTERNAL:
        runs = xm.arm_dirs("classfull", c)
        sf_runs = {m: r for (m, arm), (_, r) in runs.items() if arm == "melnik" and not sx.controls(m)}
        panel |= set(sf_runs)
        root = xm.ROOTS[c]
        scen = [(d.name, s.name) for d in sorted(root.iterdir()) if d.is_dir() for s in sorted(d.iterdir()) if s.is_dir()]
        jobs += [(c, root, c, ds, sc, sf_runs) for ds, sc in scen]
        notes.append(f"{c}: {len(scen)} scenarios, {len(sf_runs)} matchers")
    for c, (regime, stage) in VALENTINE.items():
        root = Path(make_configs.CORPUS[regime])
        sf_runs = {}
        for m in sorted(panel):
            r = run_dir(RES / stage / f"{m}__melnik")
            if r is not None:
                sf_runs[m] = r
        scen = [(d.name, s.name) for d in sorted(root.iterdir()) if d.is_dir() for s in sorted(d.iterdir()) if s.is_dir()]
        jobs += [(c, root, regime, ds, sc, sf_runs) for ds, sc in scen]
        notes.append(f"{c}: {len(scen)} scenarios, {len(sf_runs)} matchers (the external panel's)")
    rows, skip = [], Counter()
    with Pool(10) as pool:
        for r, s in pool.imap_unordered(job, jobs, chunksize=4):
            rows += r
            skip += s
    OUT.mkdir(parents=True, exist_ok=True)
    df = sx.split_corpus(pd.DataFrame(rows))
    df.to_csv(OUT / "beta_sweep_rows.csv", index=False)
    summarise(df, notes, skip, OUT)
    return 0


def summarise(df, notes, skip, OUT: Path) -> None:
    """main()'s contrasts and report, unchanged (moved into a function 2026-11-01 for the 549 run)."""
    rng = np.random.default_rng(sx.SEED)
    res = []
    order = ["valentine_plain", "valentine_md5", "final_controlled", "madi_products", "gdcsm", "hdxsm", "goby", "madi_indep"]
    for corpus in order:
        d = df[df.corpus == corpus]
        if d.empty:
            continue
        for b in BETAS:
            for metric in ("J", "R_exp", "F1_Hr"):
                r = sx.contrast(d, f"A_{b}", "SF", metric, rng)
                if r is not None:
                    res.append({"corpus": corpus, "beta": b, "metric": metric, **r})
    out = pd.DataFrame(res)
    out.to_csv(OUT / "beta_sweep.csv", index=False)
    lines = ["# Datatype-term weight sweep: A_d(beta) - SF as published (EXPLORATORY review check)", "",
             *[f"- {n}" for n in notes], f"- skipped: {dict(skip)}", "",
             "J_tf difference (positive: the datatype term is better); 95% bootstrap over scenarios stratified by dataset "
             "(by database on the new databases); * = interval excludes 0. beta = 5 is the frozen value.", "",
             "| corpus | " + " | ".join(f"beta={b}" for b in BETAS) + " |", "|---|" + "---|" * len(BETAS)]
    for corpus in order:
        cells = []
        for b in BETAS:
            x = out[(out.corpus == corpus) & (out.beta == b) & (out.metric == "J")]
            if x.empty:
                cells.append("--")
                continue
            x = x.iloc[0]
            star = "*" if np.isfinite(x["lo"]) and (x["lo"] > 0 or x["hi"] < 0) else ""
            cells.append(f"{x['estimate']:+.3f}{star}")
        if any(c != "--" for c in cells):
            lines.append(f"| {corpus} | " + " | ".join(cells) + " |")
    lines += ["", "Sign of A_d(beta) - SF per corpus over beta in [1, 20]: "
              + "; ".join(f"{c}: " + "".join("+" if out[(out.corpus == c) & (out.beta == b) & (out.metric == 'J')]
                                               .estimate.iloc[0] > 0 else "-" for b in BETAS if b > 0)
                          for c in order if not out[out.corpus == c].empty)]
    (OUT / "beta_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def job549(args):
    """job() for one Valentine scenario of the 549: q, D and SF as published by the assembly rules (valentine549)."""
    c, regime, ds, sc, panel = args
    import valentine549 as v
    g = ground_truth_file(v.NEW_TREE[regime] if v.is_new(ds, sc) else Path(make_configs.CORPUS[regime]), ds, sc)
    fd = v.seed_file(regime, "datatype_nanfix", ds, sc)
    if g is None or not fd.is_file():
        return [], Counter({"no_truth_or_D": 1})
    gt = read_matrix(g)[0] > 0.5
    D = read_matrix(fd)[0]
    perms = permutations(gt.shape, f"{ds}/{sc}")
    rows, skip = [], Counter()
    for m in panel:
        f = v.seed_file(regime, m, ds, sc)
        r = v.sf_run(regime, m, "melnik", ds, sc)
        o = one(r / ds / sc / "_outputs" / "MatchingStepLine2") if r is not None else None
        if not f.is_file() or o is None:
            skip["missing"] += 1
            continue
        q, S = read_matrix(f)[0], read_matrix(o)[0]
        if q.shape != gt.shape or S.shape != q.shape or D.shape != q.shape or not (
                np.isfinite(q).all() and np.isfinite(S).all() and np.isfinite(D).all()):
            skip["shape_or_nonfinite"] += 1
            continue
        for arm, M in {"SF": S, **{f"A_{b}": np.clip(q, 0, None) + b * D for b in BETAS}}.items():
            x = metrics(M, gt, perms)
            rows.append({"corpus": c, "dataset": ds, "scenario": f"{ds}/{sc}", "matcher": m, "arm": arm,
                         "J": x["J"], "R_exp": x["R_exp"], "F1_Hr": x["F1_Hr"]})
    return rows, skip


def main549() -> int:
    """SMX_VALENTINE=549: Valentine rows recomputed on the 549 scenarios (the external panel's matchers, as main()),
    the external corpora's rows from the stored beta_sweep_rows.csv (inputs unchanged); output next to the 549 figures
    (latex/sf-evaluation-549/review_checks/). Only the estimates enter the review macros."""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(STUDY / "reference"))
    import valentine549 as v
    old = pd.read_csv(OUT / "beta_sweep_rows.csv")
    ext = old[~old.corpus.str.startswith("valentine")]
    panel = sorted(set(ext.matcher))
    jobs = [(c, regime, ds, sc, panel) for c, (regime, _) in VALENTINE.items() for ds, sc, _ in v.cases549()]
    rows, skip = [], Counter()
    with Pool(10) as pool:
        for r, s in pool.imap_unordered(job549, jobs, chunksize=4):
            rows += r
            skip += s
    out = STUDY.parent / "latex" / "sf-evaluation-549" / "review_checks"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.DataFrame(rows), ext], ignore_index=True)
    df.to_csv(out / "beta_sweep_rows.csv", index=False)
    notes = [f"549 run: Valentine on the 549 scenarios ({len(panel)} matchers, the external panel's); external rows "
             "from the stored beta_sweep_rows.csv"]
    summarise(df, notes, skip, out)
    return 0


if __name__ == "__main__":
    import os
    raise SystemExit(main549() if os.environ.get("SMX_VALENTINE", "382") == "549" else main())
