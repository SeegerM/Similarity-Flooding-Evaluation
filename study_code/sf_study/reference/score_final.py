"""Scorer for Amendment 12 (draft: reference/AMENDMENT_FINAL_DRAFT.md): frozen recommendations on untouched data.
Written on 2026-10-06 before any run on these data. Frozen with the amendment.

Corpora (built; layouts as the harness's Valentine format):
  part A  data/final_indep            one dataset per pair, named as the truth file stem (bib_s1a_s2b, music_mb1_dis2, ...),
                                       one scenario each; multi-table sides; matrix labels "table.column"
  part B  harness/generated/final-controlled   smx.controlled scenarios; dataset "<Domain>-<Type>", scenario suffix
                                       _plain / _opaque (as controlled-imported-v3)
Initial matrices: reference/seeds/<corpus>/<m>/<ds>/<sc>/seed.csv; D = datatype_nanfix.
SF arms: tuning/results/<corpus>/<m>__<arm> (<corpus> = final_indep or final_controlled), arm in melnik, fd, ucc_top1, ucc1_fd_topk.
Terms (frozen): A_d = q+ + 5D; A_role = A_d + γR (γ 0.2 plain, 1.0 opaque; R = explore_deproles roles, corrected parse);
A_dt = A_d + 0.5C (C = score_tablectx.table_context).
Truth: part B from the corpus; part A from the frozen CSVs (data/final_gt/<ds>.csv): primary C + L, sensitivity C only.
The importer's matrix must equal the CSV's C + L truth (checked; a mismatch stops the scorer).

    ../harness/.venv-bdi/Scripts/python.exe reference/score_final.py --selftest          # synthetic only
    ../harness/.venv-bdi/Scripts/python.exe reference/score_final.py --part A|B           # after the runs
Output: tuning/results/final_<part>_scores/{scores.csv, report.md}
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
for p in (STUDY.parent / "harness", STUDY / "tuning", STUDY, HERE):
    sys.path.insert(0, str(p))

ROOTS = {"A": STUDY / "data" / "final_indep", "B": STUDY.parent / "harness" / "generated" / "final-controlled"}
CORPUS_NAME = {"A": "final_indep", "B": "final_controlled"}
ARMS = ("melnik", "fd", "ucc_top1", "ucc1_fd_topk")
GAMMA = {"plain": 0.2, "opaque": 1.0}
TERMS = ("q", "A_d", "A_role", "A_dt")
B, SEED = 10_000, 20261076
MARGIN = -0.01
GT_DIR = STUDY / "data" / "final_gt"


# ---------------------------------------------------------------- truth (part A)
def truth_from_csv(ds: str, rows: list[str], cols: list[str], certain_only: bool) -> np.ndarray:
    """Boolean truth matrix over (rows, cols) labels "table.column" from the frozen CSV of pair `ds`."""
    ri, ci = {r: i for i, r in enumerate(rows)}, {c: j for j, c in enumerate(cols)}
    gt = np.zeros((len(rows), len(cols)), dtype=bool)
    missing = []
    with (GT_DIR / f"{ds}.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if certain_only and r["confidence"] != "C":
                continue
            a, b = f"{r['source_table']}.{r['source_column']}", f"{r['target_table']}.{r['target_column']}"
            if a in ri and b in ci:
                gt[ri[a], ci[b]] = True
            else:
                missing.append((a, b))
    if missing:
        raise SystemExit(f"{ds}: {len(missing)} truth correspondences name columns absent from the matrices, e.g. {missing[:3]}")
    return gt


# ---------------------------------------------------------------- statistics and decisions (tested by --selftest)
def cell_values(rows: list[dict], arms_needed: list[str]):
    """Included matchers (every arm on every scenario) and per-scenario means over them: {(group, ds, sc): {arm: v}}."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["group"], r["dataset"], r["scenario"])][r["seed"]] = r
    matchers = sorted({r["seed"] for r in rows})
    inc = [m for m in matchers if all(m in v and all(a in v[m] for a in arms_needed) for v in by.values())]
    means = {k: {a: float(np.mean([v[m][a] for m in inc])) for a in arms_needed} for k, v in by.items()} if inc else {}
    return inc, means, by


def macro_contrast(means: dict, x: str, y: str, rng) -> tuple[float, float, float]:
    """Mean over groups of the within-group mean of x − y; 95% percentile bootstrap over scenarios within group."""
    per = defaultdict(list)
    for (g, _, _), v in sorted(means.items()):
        per[g].append(v[x] - v[y])
    arrs = [np.array(v) for v in per.values()]
    est = float(np.mean([a.mean() for a in arrs]))
    boots = np.mean([[rng.choice(a, size=a.size).mean() for _ in range(B)] for a in arrs], axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)


def spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def f3(by: dict, inc: list[str], rng) -> tuple[float, float, float]:
    """ρ over matchers of (group-macro R_exp of q, group-macro R_exp gain of SF melnik); bootstrap over matchers."""
    groups = sorted({k[0] for k in by})
    xs, ys = [], []
    for m in inc:
        qv = np.mean([np.mean([v[m]["q_R"] for k, v in by.items() if k[0] == g]) for g in groups])
        gv = np.mean([np.mean([v[m]["melnik_R"] - v[m]["q_R"] for k, v in by.items() if k[0] == g]) for g in groups])
        xs.append(qv)
        ys.append(gv)
    xs, ys = np.array(xs), np.array(ys)
    rho = spearman(xs, ys)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(xs), len(xs))
        if len(set(idx)) > 2:
            boots.append(spearman(xs[idx], ys[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, float(lo), float(hi)


def decide_B(est: dict) -> dict:
    return {"F1": est["F1"][1] > 0, "F2a": est["F2a"][1] > MARGIN, "F2b": est["F2b"][1] > MARGIN,
            "F3": est["F3"][2] < 0, "F4": est["F4"][2] < 0}


def decide_A(per_pair: list[float], mean_ci: tuple[float, float, float]) -> bool:
    return sum(d > 0 for d in per_pair) >= 7 and mean_ci[1] > 0


# ---------------------------------------------------------------- data (real runs)
def cases(part: str) -> list[tuple[str, str, str, str]]:
    """(group, dataset, scenario, regime)."""
    out = []
    root = ROOTS[part]
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for s in sorted(p for p in d.iterdir() if p.is_dir()):
            if part == "A":
                out.append(("pairs", d.name, s.name, "plain"))
            else:
                out.append((d.name.split("-")[0], d.name, s.name, "plain" if s.name.endswith("_plain") else "opaque"))
    return out


def score_case(job):
    part, group, ds, sc, regime, matchers = job
    import score_layer1
    import score_seeds
    from explore_composite import load, sf_out
    from explore_deproles import profiles, roles
    from score_screen_metrics import all_metrics, permutations
    from score_tablectx import table_context
    from smx.adapters import read_matrix
    corpus = CORPUS_NAME[part]
    score_seeds.CORPUS[corpus] = ROOTS[part]
    seeds = STUDY / "reference" / "seeds" / corpus
    f_d = seeds / "datatype_nanfix" / ds / sc / "seed.csv"
    g = score_seeds.ground_truth_file(ROOTS[part], ds, sc)
    if g is None or not f_d.is_file():
        return []
    D, rows, cols = read_matrix(f_d)
    rows, cols, D = list(rows), list(cols), np.nan_to_num(D)
    gt_imp = read_matrix(g)[0] > 0.5
    truths = {"": gt_imp}
    if part == "A":
        prim = truth_from_csv(ds, rows, cols, certain_only=False)
        if not np.array_equal(prim, gt_imp):
            raise SystemExit(f"{ds}: the imported truth differs from the frozen CSV (C + L)")
        truths["_C"] = truth_from_csv(ds, rows, cols, certain_only=True)
    Fs, _ = roles(rows, *profiles(corpus, ds, sc, "source"))
    Ft, _ = roles(cols, *profiles(corpus, ds, sc, "target"))
    R = 1.0 - np.abs(Fs[:, None, :] - Ft[None, :, :]).mean(axis=2)
    out = []
    for m in matchers:
        q = load(seeds / m / ds / sc / "seed.csv", rows, cols)
        if q is None:
            continue
        qp = np.clip(q, 0, None)
        A_d = qp + 5.0 * D
        mats = {"q": qp, "A_d": A_d, "A_role": A_d + GAMMA[regime] * R, "A_dt": A_d + 0.5 * table_context(q, rows, cols)}
        for arm in ARMS:
            rd = STUDY / "tuning" / "results" / corpus / f"{m}__{arm}"
            S = load(sf_out(score_layer1.run_dir(rd) if rd.is_dir() else None, ds, sc), rows, cols)
            if S is not None:
                mats[arm] = np.clip(S, 0, None)
        for suffix, gt in truths.items():
            perms = permutations(gt.shape, f"{ds}/{sc}")
            row = {"seed": m, "group": group, "dataset": ds, "scenario": sc, "regime": regime, "truth": suffix or "primary"}
            for name, M in mats.items():
                r = all_metrics(M, gt, perms)
                row[name], row[f"{name}_R"] = (r["R_exp"] + r["F1_Hr"]) / 2, r["R_exp"]
            out.append(row)
    return out


def report_B(rows: list[dict], rng) -> list[str]:
    lines = []
    for regime in ("plain", "opaque"):
        rr = [r for r in rows if r["regime"] == regime and r["truth"] == "primary"]
        if not rr:
            continue
        need = [*TERMS, *ARMS, "q_R", "melnik_R", "fd_R"]
        inc, means, by = cell_values(rr, need)
        est = {"F1": macro_contrast(means, "A_d", "melnik", rng), "F2a": macro_contrast(means, "A_role", "ucc_top1", rng),
               "F2b": macro_contrast(means, "A_role", "ucc1_fd_topk", rng), "F4": macro_contrast(means, "fd_R", "melnik_R", rng),
               "F3": f3(by, inc, rng)}
        dec = decide_B(est)
        lines += [f"## Part B, {regime}: {len(means)} scenarios, {len({k[0] for k in means})} domains, {len(inc)} matchers", "",
                  "| hypothesis | contrast | estimate [95% CI] | verdict |", "|---|---|---|---|"]
        labels = {"F1": "A_d − SF(melnik), J_tf", "F2a": "A_role − SF(ucc_top1), J_tf (margin −0.01)",
                  "F2b": "A_role − SF(ucc1_fd_topk), J_tf (margin −0.01)", "F3": "Spearman ρ(q quality, SF gain), R_exp",
                  "F4": "SF(fd) − SF(melnik), R_exp"}
        for h in ("F1", "F2a", "F2b", "F3", "F4"):
            if h == "F4" and regime != "plain":
                verdict = "reported only"
            else:
                verdict = "CONFIRMED" if dec[h] else "not confirmed"
            e = est[h]
            lines.append(f"| {h} | {labels[h]} | {e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}] | {verdict} |")
        lines.append("")
    return lines


def report_A(rows: list[dict], rng) -> list[str]:
    lines = []
    for truth in ("primary", "_C"):
        rr = [r for r in rows if r["truth"] == truth]
        if not rr:
            continue
        need = [*TERMS, *ARMS]
        inc, means, _ = cell_values(rr, need)
        pairs = sorted(means)
        d = [means[k]["A_dt"] - means[k]["melnik"] for k in pairs]
        arr = np.array(d)
        boots = [rng.choice(arr, size=arr.size).mean() for _ in range(B)]
        ci = (float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        verdict = "SUPPORTED" if decide_A(d, ci) else "not supported"
        tag = "primary truth (C + L)" if truth == "primary" else "sensitivity: certain correspondences only"
        lines += [f"## Part A, {tag}: {len(pairs)} pairs, {len(inc)} matchers", "",
                  "| pair | A_d | A_dt | A_role | SF melnik | SF fd | A_dt − SF melnik |", "|---|---|---|---|---|---|---|"]
        for k, x in zip(pairs, d):
            v = means[k]
            lines.append(f"| {k[1]} | {v['A_d']:.4f} | {v['A_dt']:.4f} | {v['A_role']:.4f} | {v['melnik']:.4f} | "
                         f"{v['fd']:.4f} | {x:+.4f} |")
        lines += ["", f"F5: {sum(x > 0 for x in d)} of {len(d)} pairs positive; mean {ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}] "
                  f"-> {verdict}", ""]
    return lines


# ---------------------------------------------------------------- self-test (synthetic only)
def selftest() -> int:
    rng = np.random.default_rng(1)
    # part B: two domains x 6 scenarios x 3 matchers, built so that F1, F2a, F2b, F3, F4 hold
    rows = []
    for dom in ("D1", "D2"):
        for s in range(6):
            for k, m in enumerate(("m_weak", "m_mid", "m_strong")):
                q = 0.2 + 0.3 * k
                rows.append({"seed": m, "group": dom, "dataset": f"{dom}-X", "scenario": f"s{s}_plain", "regime": "plain",
                             "truth": "primary", "q": q, "A_d": q + 0.05, "A_role": q + 0.06, "A_dt": q + 0.05,
                             "melnik": q + 0.01, "fd": q - 0.02, "ucc_top1": q + 0.04, "ucc1_fd_topk": q + 0.05,
                             "q_R": q, "melnik_R": q + 0.15 - 0.1 * k + 0.001 * s, "fd_R": q + 0.05 - 0.1 * k})
    inc, means, by = cell_values(rows, [*TERMS, *ARMS, "q_R", "melnik_R", "fd_R"])
    assert inc == ["m_mid", "m_strong", "m_weak"], inc
    e1 = macro_contrast(means, "A_d", "melnik", rng)
    assert abs(e1[0] - 0.04) < 1e-9 and e1[1] > 0, e1
    est = {"F1": e1, "F2a": macro_contrast(means, "A_role", "ucc_top1", rng),
           "F2b": macro_contrast(means, "A_role", "ucc1_fd_topk", rng), "F4": macro_contrast(means, "fd_R", "melnik_R", rng),
           "F3": f3(by, inc, rng)}
    assert abs(est["F2b"][0] - 0.01) < 1e-9 and est["F3"][0] < 0, est
    dec = decide_B(est)
    assert all(dec.values()), dec
    # a matcher missing one arm on one scenario is excluded
    rows2 = [dict(r) for r in rows]
    del rows2[0]["fd"]
    inc2, _, _ = cell_values(rows2, [*TERMS, *ARMS])
    assert rows2[0]["seed"] not in inc2
    # F2 fails below the margin
    est2 = dict(est, F2a=(-0.02, -0.03, -0.011))
    assert not decide_B(est2)["F2a"]
    # part A decision rule: 7 of 9 positive and lower bound > 0
    assert decide_A([0.01] * 7 + [-0.01] * 2, (0.006, 0.001, 0.01))
    assert not decide_A([0.01] * 6 + [-0.01] * 3, (0.003, 0.001, 0.01))
    assert not decide_A([0.01] * 9, (0.01, -0.001, 0.02))
    # truth mapping from a frozen CSV (bib_s3a_s4b): every correspondence lands, certain-only is a subset
    src = ["article.articleid", "article.title", "article.year", "article.note", "article.abstract", "articleauthor.articleid",
           "articleauthor.authorid", "author.authorid", "author.name", "unpublished.unpubid", "unpublished.title",
           "unpublished.note", "unpubauthor.unpubid", "unpubauthor.authorid", "article.pages"]
    tgt = ["publication.pid", "publication.title", "written.pubyear", "record.notes", "publication.abstract", "written.pid",
           "written.aid", "author.aid", "author.name", "publication.journal"]
    prim = truth_from_csv("bib_s3a_s4b", src, tgt, certain_only=False)
    cert = truth_from_csv("bib_s3a_s4b", src, tgt, certain_only=True)
    assert prim.sum() == 14 and cert.sum() == 12 and not (cert & ~prim).any(), (prim.sum(), cert.sum())
    assert prim[src.index("article.note"), tgt.index("record.notes")] and not cert[src.index("article.note"), tgt.index("record.notes")]
    try:
        truth_from_csv("bib_s3a_s4b", src[:-6], tgt, certain_only=False)
        raise AssertionError("missing columns were not detected")
    except SystemExit:
        pass
    print("selftest passed: aggregation, inclusion rule, decisions F1-F5, truth mapping and its consistency check")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--part", choices=("A", "B"))
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    sys.stdout.reconfigure(encoding="utf-8")
    cs = cases(a.part)
    corpus = CORPUS_NAME[a.part]
    seeds = STUDY / "reference" / "seeds" / corpus
    controls = {"constant", "random", "random_fresh", "oracle", "oracle_noise025", "oracle_noise050", "oracle_erase050",
                "oracle_decoy100", "embedding_random", "rema_shuffled", "semstruct_lm_only", "datatype", "datatype_nanfix"}
    matchers = sorted(p.name for p in seeds.iterdir() if p.is_dir() and p.name not in controls)
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_case, [(a.part, *c, matchers) for c in cs]) for r in part]
    out = STUDY / "tuning" / "results" / f"final_{a.part}_scores"
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    rng = np.random.default_rng(SEED)
    lines = [f"# Amendment 12, part {a.part} (registered): frozen recommendations on untouched data", ""]
    lines += report_B(rows, rng) if a.part == "B" else report_A(rows, rng)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
