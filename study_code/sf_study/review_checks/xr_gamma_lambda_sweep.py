"""External-review check (EXPLORATORY, 2026-11-03): weight sensitivity of the role and table-context terms on the
independently produced corpora, as review_checks/beta_sweep.py does for the datatype weight beta.

gamma (role term A_role = q+ + 5D + gamma R against SF with the top-1 key, the ucc_top1 arm): the four single-table
external corpora (GDC-SM, HDXSM, GOBY, MaDI use cases; MaDI products reported separately), on the fixed FD profiles, with
explore_x81.py's inputs, panel and inclusion rule (SMX_VALENTINE=549 -> its fixed-profile mode); gamma = 0.2 (the frozen
value) must reproduce the stored contrast (results/x81fix/report.md, E2a).
lambda (table-context term A_dt = q+ + 5D + lambda C against SF as published): the independent schema pairs (final test,
part A, primary truth), with reference/score_final.py's inputs and inclusion rule; lambda = 0.5 (frozen) must reproduce
the registered estimate (tuning/results/final_A_scores/report.md, F5). (On single-table corpora C is constant over the
column pairs, so lambda cannot change the ranking there.)
Grids: gamma, lambda in (0, 0.1, 0.2, 0.5, 1, 2). Estimates as the stored reports compute them (x81: macro over groups of
scenario means over the included matchers, cluster-bootstrap interval where a corpus has >= 5 clusters; part A: mean over
pairs, bootstrap over pairs). Output (JSON + MD) in SMX_REVIEW_DIR, default latex/sf-evaluation-presentation-549/review_checks/.

    SMX_VALENTINE=549 PYTHONHASHSEED=0 ../harness/.venv-bdi/Scripts/python.exe -B review_checks/xr_gamma_lambda_sweep.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent.parent
for p in (STUDY.parent / "harness", STUDY / "tuning", STUDY, STUDY / "reference", STUDY / "layer3b"):
    sys.path.insert(0, str(p))
OUT = Path(os.environ.get("SMX_REVIEW_DIR") or STUDY.parent / "latex" / "sf-evaluation-presentation-549" / "review_checks")
GRID = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0)
SEED = 20261103


def x81_case(job):
    import explore_x81 as x
    import score_layer1
    import score_seeds
    from explore_composite import load, sf_out
    from explore_deproles import profiles, roles
    from score_screen_metrics import all_metrics, permutations
    from smx.adapters import read_matrix
    c, ds, sc, ms = job
    V = x.V
    score_seeds.CORPUS[c] = V.FIXED_MIRROR / c if V.HYFDFIX else x.ROOTS[c]
    seeds = x.HERE / "reference" / "seeds" / c
    f_d = seeds / "datatype_nanfix" / ds / sc / "seed.csv"
    g = score_seeds.ground_truth_file(x.ROOTS[c], ds, sc)
    if g is None or not f_d.is_file():
        return []
    D, rows, cols = read_matrix(f_d)
    rows, cols, D = list(rows), list(cols), np.nan_to_num(D)
    gt = read_matrix(g)[0] > 0.5
    if gt.shape != D.shape or not gt.any():
        return []
    Fs, _ = roles(rows, *profiles(c, ds, sc, "source"))
    Ft, _ = roles(cols, *profiles(c, ds, sc, "target"))
    R = 1.0 - np.abs(Fs[:, None, :] - Ft[None, :, :]).mean(axis=2)
    perms = permutations(gt.shape, f"{ds}/{sc}")
    rep, grp = x.group_of(c, sc)
    out = []
    for m in ms:
        q = load(seeds / m / ds / sc / "seed.csv", rows, cols)
        if q is None:
            continue
        A_d = np.clip(q, 0, None) + 5.0 * D
        mats = {f"g{gm}": A_d + gm * R for gm in GRID}
        rd = x.HERE / "tuning" / "results" / V.run_stage(f"x81_{c}", "ucc_top1") / f"{m}__ucc_top1"
        S = load(sf_out(score_layer1.run_dir(rd) if rd.is_dir() else None, ds, sc), rows, cols)
        if S is None or not np.isfinite(S).all():
            continue
        mats["ucc_top1"] = np.clip(S, 0, None)
        row = {"seed": m, "corpus": rep, "group": grp, "dataset": ds, "scenario": sc}
        for name, M in mats.items():
            r = all_metrics(M, gt, perms)
            row[name] = (r["R_exp"] + r["F1_Hr"]) / 2
        out.append(row)
    return out


def final_case(job):
    import score_final as f
    import score_layer1
    import score_seeds
    from explore_composite import load, sf_out
    from score_screen_metrics import all_metrics, permutations
    from score_tablectx import table_context
    from smx.adapters import read_matrix
    ds, sc, ms = job
    corpus = f.CORPUS_NAME["A"]
    score_seeds.CORPUS[corpus] = f.ROOTS["A"]
    seeds = f.STUDY / "reference" / "seeds" / corpus
    D, rows, cols = read_matrix(seeds / "datatype_nanfix" / ds / sc / "seed.csv")
    rows, cols, D = list(rows), list(cols), np.nan_to_num(D)
    gt = f.truth_from_csv(ds, rows, cols, certain_only=False)
    perms = permutations(gt.shape, f"{ds}/{sc}")
    out = []
    for m in ms:
        q = load(seeds / m / ds / sc / "seed.csv", rows, cols)
        if q is None:
            continue
        A_d = np.clip(q, 0, None) + 5.0 * D
        C = table_context(q, rows, cols)
        mats = {f"l{lm}": A_d + lm * C for lm in GRID}
        present = True
        for arm in f.ARMS:   # the registered inclusion rule: every SF arm on every pair
            rd = f.STUDY / "tuning" / "results" / corpus / f"{m}__{arm}"
            S = load(sf_out(score_layer1.run_dir(rd) if rd.is_dir() else None, ds, sc), rows, cols)
            if S is None:
                present = False
                break
            if arm == "melnik":
                mats["melnik"] = np.clip(S, 0, None)
        if not present:
            continue
        row = {"seed": m, "group": "pairs", "dataset": ds, "scenario": sc}
        for name, M in mats.items():
            r = all_metrics(M, gt, perms)
            row[name] = (r["R_exp"] + r["F1_Hr"]) / 2
        out.append(row)
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    import explore_x81 as x
    import score_final as f
    from score_final import cell_values, macro_contrast
    res = {"gamma": {}, "lambda": {}, "grid": list(GRID)}
    # ---------------- gamma
    jobs = [(c, ds, sc, x.matchers(c)) for c in ("gdcsm", "hdxsm", "goby", "madi") for ds, sc in x.cases(c)]
    with Pool(12) as pool:
        rows = [r for rr in pool.imap_unordered(x81_case, jobs, chunksize=2) for r in rr]
    stored = (STUDY / "results" / ("x81fix" if x.V.HYFDFIX else "x81") / "report.md").read_text(encoding="utf-8")
    for rep in ("gdcsm", "hdxsm", "goby", "madi_indep", "madi_products"):
        rr = [r for r in rows if r["corpus"] == rep]
        need = [f"g{g}" for g in GRID] + ["ucc_top1"]
        inc, means, _ = cell_values(rr, need)
        cl = x.clusters(rep, sorted({k[2] for k in means}))
        ncl = len(set(cl.values()))
        out = {}
        for g in GRID:
            est = macro_contrast(means, f"g{g}", "ucc_top1", np.random.default_rng(SEED))
            t = x.cluster_ci(means, f"g{g}", "ucc_top1", cl, np.random.default_rng(SEED)) if ncl >= 5 else (est[0], None, None)
            out[str(g)] = {"est": est[0], "lo": t[1], "hi": t[2]}
        sec = stored.split(f"\n## {rep}\n", 1)[1].split("\n## ", 1)[0]
        m = re.search(r"\| E2a A_role − SF\(ucc_top1\) \| ([+-][0-9.]+)", sec)
        assert m and abs(float(m.group(1)) - out["0.2"]["est"]) < 6e-5, (rep, m and m.group(1), out["0.2"]["est"])
        res["gamma"][rep] = {"matchers": len(inc), "scenarios": len(means), "clusters": ncl, "by": out}
    # ---------------- lambda (independent schema pairs)
    cases = [(ds, sc) for _, ds, sc, _ in f.cases("A")]
    ms = sorted({d.name.split("__")[0] for d in (f.STUDY / "tuning" / "results" / "final_indep").iterdir() if "__" in d.name})
    with Pool(9) as pool:
        rows = [r for rr in pool.imap_unordered(final_case, [(ds, sc, ms) for ds, sc in cases]) for r in rr]
    need = [f"l{lm}" for lm in GRID] + ["melnik"]
    inc, means, _ = cell_values(rows, need)
    pairs = sorted(means)
    out = {}
    for lm in GRID:
        d = np.array([means[k][f"l{lm}"] - means[k]["melnik"] for k in pairs])
        rng = np.random.default_rng(SEED)
        boots = [rng.choice(d, size=d.size).mean() for _ in range(10_000)]
        out[str(lm)] = {"est": float(d.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
                        "pos": int((d > 0).sum())}
    rep_a = (STUDY / "tuning" / "results" / "final_A_scores" / "report.md").read_text(encoding="utf-8")
    m = re.search(r"Part A, primary truth.*?F5: (\d+) of (\d+) pairs positive; mean ([+-][0-9.]+)", rep_a, re.S)
    assert m and abs(float(m.group(3)) - out["0.5"]["est"]) < 6e-5 and int(m.group(1)) == out["0.5"]["pos"], \
        (m and m.groups(), out["0.5"])
    res["lambda"]["final_indep"] = {"matchers": len(inc), "pairs": len(pairs), "by": out}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "xr_gamma_lambda_sweep.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = ["# Role and table-context weights on the independent corpora (EXPLORATORY, external review 2026-11-03)", "",
          "J_tf, term minus SF; gamma: role term minus SF with the top-1 key; lambda: table-context term minus SF as "
          "published. Frozen values: gamma 0.2 (readable), lambda 0.5 (both reproduce the stored estimates).", "",
          "| corpus | weight | " + " | ".join(str(g) for g in GRID) + " |", "|---|---|" + "---|" * len(GRID)]
    for rep, v in res["gamma"].items():
        md.append(f"| {rep} ({v['scenarios']} sc., {v['clusters']} cl.) | gamma | " + " | ".join(
            f"{v['by'][str(g)]['est']:+.4f}" + (f" [{v['by'][str(g)]['lo']:+.4f}, {v['by'][str(g)]['hi']:+.4f}]"
                                                 if v['by'][str(g)]['lo'] is not None else "") for g in GRID) + " |")
    v = res["lambda"]["final_indep"]
    md.append(f"| final_indep ({v['pairs']} pairs) | lambda | " + " | ".join(
        f"{v['by'][str(g)]['est']:+.4f} [{v['by'][str(g)]['lo']:+.4f}, {v['by'][str(g)]['hi']:+.4f}] ({v['by'][str(g)]['pos']} pos.)"
        for g in GRID) + " |")
    (OUT / "xr_gamma_lambda_sweep.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
