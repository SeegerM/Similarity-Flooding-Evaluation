"""Registered scorer for Amendment 8 (layer3b/AMENDMENT_CLASSFULL_DRAFT.md): class nodes on the full panel.

Written 2026-10-04 before any classfull output was read; tested only on local screen runs (f7_screen, f13_screen).
Implements the draft's §3–§5:
- arms (tuning/results/classfull_<regime>/<m>__melnik_{shape,shape_shuffled,semtype,semtype_shuffled}) against layer 1's
  `<m>__melnik`; late fusion L = q + 0.2·eq and B = SF + 0.2·max(SF)·eq with eq from the edge files
  (data/classfull_<regime>/{shape,semtype}); reported: A_p = q⁺ + 5 D + 0.2 eq_p, A_s = q⁺ + 5 D + 0.1 eq_s;
- statistic: score_r7.Panel (J_tf; per matcher the macro over families with >= 5 scenarios; mean over included
  non-control matchers with full coverage in every arm of the contrast; 95% bootstrap stratified by family, 10,000,
  seed 20261002); confirmed if the lower bound > 0; H8f equivalence: interval inside (−0.01, +0.01);
- checks first (refuse otherwise): the devnone gate, and for every class configuration and scenario the schema graph of
  sf_stats.tsv larger than layer 1's melnik by exactly Δv = Σ_side(2k+1), Δe = Σ_side(columns + 2k);
- H8a: 0 within-block inversions in the true refined blocks (blocks of layer 1's constant__melnik × class pair).
Case sets: primary = the 322 non-screen scenarios; secondary = all 382.

Writes tuning/results/classfull_<regime>_scores/{scores.csv, contrasts.csv, report.md}.

    python layer3b/score_classfull.py --regime plain [--workers 10]
Test (local screen runs only): layer3b/make_classfull_scorer_test.py, then the --*-root options and --cases screen.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
from score_r7 import STATS, Panel, blocks, is_control, metrics, run_dir  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from score_seeds import CORPUS, PROTOCOL, cases_for, ground_truth_file, one  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402

RES = STUDY / "tuning" / "results"
VARIANTS = ("shape", "shape_shuffled", "semtype", "semtype_shuffled")
F = ["J", "R_exp", "F1_Hr", "MRR_exp", "F1_M"]
LAM = 0.2


def edge_classes(edges: Path, variant: str, ds: str, sc: str) -> dict[str, str]:
    out = {}
    for side in ("source", "target"):
        for line in (edges / variant / ds / sc / f"{side}.tsv").read_text(encoding="utf-8").splitlines():
            label, table, col, cls, _ = line.split("\t")
            out[f"{table}.{col}"] = cls
    return out


def expected_growth(edges: Path, variant: str, ds: str, sc: str) -> tuple[int, int]:
    dv = de = 0
    for side in ("source", "target"):
        lines = (edges / variant / ds / sc / f"{side}.tsv").read_text(encoding="utf-8").splitlines()
        k = len({l.split("\t")[3] for l in lines})
        dv += 2 * k + 1
        de += len(lines) + 2 * k
    return dv, de


def graph_sizes(d: Path) -> dict:
    f, log = d / "sf_stats.tsv", d / "run.log"
    lines = (f.read_text(encoding="utf-8").splitlines() if f.is_file() else
             [l[len("SFSTATS\t"):] for l in log.read_text(encoding="utf-8", errors="replace").splitlines()
              if l.startswith("SFSTATS\t")] if log.is_file() else [])
    out = {}
    for line in lines:
        p = line.split("\t")
        if len(p) == 2 + len(STATS):
            s = dict(zip(STATS, p[2:]))
            out[(p[0], p[1])] = (int(s["schema_v"]), int(s["schema_e"]))
    return out


def score_seed(job):
    regime, seed, cases, cf, l1, edges, seeds_root = job
    q_dir, d_dir = seeds_root / seed, seeds_root / "datatype_nanfix"
    sf = run_dir(l1 / f"{seed}__melnik")
    const = run_dir(l1 / "constant__melnik")
    arms = {v: run_dir(cf / f"{seed}__melnik_{v}") for v in VARIANTS}
    out, inv = [], {v: [0, 0] for v in ("shape", "semtype")}
    # skipped (matcher, scenario[, arm]) counts per reason (erratum 1; review): layer-1 melnik output not finite or
    # misshapen; class-arm output not finite or misshapen; constant__melnik (H8a blocks) not finite or misshapen;
    # datatype matrix D missing or not finite (A_p / A_s, report only)
    for k in ("nonfinite_sf", "nonfinite_arm", "nonfinite_const", "missing_D"):
        inv[k] = [0, 0]
    for ds, sc, fam, screen in cases:
        f_q, f_d = q_dir / ds / sc / "seed.csv", d_dir / ds / sc / "seed.csv"
        g = ground_truth_file(CORPUS[regime], ds, sc)
        if not f_q.is_file() or g is None or sf is None:
            continue
        gt = read_matrix(g)[0] > 0.5
        q, rows, cols = read_matrix(f_q)
        if q.shape != gt.shape or not np.isfinite(q).all():
            continue
        perms = permutations(gt.shape, f"{ds}/{sc}")
        mq = metrics(q, gt, perms)
        base = {"seed": seed, "dataset": ds, "scenario": sc, "family": fam, "screen": int(screen),
                **{f"q_{k}": mq[k] for k in F}}
        f_s = one(sf / ds / sc / "_outputs" / "MatchingStepLine2")
        if f_s is None:
            continue
        S = read_matrix(f_s)[0]
        if S.shape != q.shape or not np.isfinite(S).all():
            # erratum 1 (crash report, 2026-10-04): a non-finite layer-1 output cannot be scored; as for every
            # other arm (§3 inclusion), the scenario drops out for this matcher, so the matcher leaves every contrast
            inv["nonfinite_sf"][0] += 1
            continue
        mats = {"melnik": S}
        for v, d in arms.items():
            f = one(d / ds / sc / "_outputs" / "MatchingStepLine2") if d else None
            if f is not None:
                M = read_matrix(f)[0]
                if M.shape == q.shape and np.isfinite(M).all():
                    mats[v] = M
                else:
                    inv["nonfinite_arm"][0] += 1
        D = read_matrix(f_d)[0] if f_d.is_file() else None
        if D is None or D.shape != q.shape or not np.isfinite(D).all():
            inv["missing_D"][0] += 1
            D = None
        C = None
        if const is not None:
            fc = one(const / ds / sc / "_outputs" / "MatchingStepLine2")
            if fc is not None:
                C = read_matrix(fc)[0]
                if C.shape != q.shape or not np.isfinite(C).all():
                    inv["nonfinite_const"][0] += 1
                    C = None
        for fam_name, w in (("shape", 0.2), ("semtype", 0.1)):
            cls = edge_classes(edges, fam_name, ds, sc)
            eq = (np.array([cls[r] for r in rows])[:, None] == np.array([cls[c] for c in cols])[None, :]).astype(float)
            tag = "p" if fam_name == "shape" else "s"
            mats[f"L_{tag}"] = q + LAM * eq
            mats[f"B_{tag}"] = S + LAM * S.max() * eq
            if D is not None:
                mats[f"A_{tag}"] = np.clip(q, 0, None) + 5.0 * D + w * eq
            if fam_name in mats and C is not None:
                bdt = blocks(C)
                n, m = q.shape
                keys = [f"{bdt[i * m + j]}|{cls[rows[i]]}|{cls[cols[j]]}" for i in range(n) for j in range(m)]
                blk = np.unique(np.array(keys), return_inverse=True)[1]
                s, qq = mats[fam_name].ravel(), q.ravel()
                for b in np.unique(blk):
                    idx = np.where(blk == b)[0]
                    if len(idx) < 2:
                        continue
                    dq = qq[idx][:, None] - qq[idx][None, :]
                    dS = s[idx][:, None] - s[idx][None, :]
                    inv[fam_name][0] += int(((dq > 1e-9) & (dS < -1e-6)).sum())
                    inv[fam_name][1] += int((dq > 1e-9).sum())
        for arm, M in mats.items():
            out.append({**base, "arm": arm, **{f"sf_{k}": v for k, v in metrics(M, gt, perms).items() if k in F}})
    return out, inv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5"), required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cf-root", default="")
    ap.add_argument("--l1-root", default="")
    ap.add_argument("--edges-root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--cases", choices=("protocol", "screen"), default="protocol")
    ap.add_argument("--matchers", default="", help="testing only")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cf = Path(a.cf_root or RES / f"classfull_{a.regime}")
    l1 = Path(a.l1_root or RES / f"layer1_{a.regime}")
    edges = Path(a.edges_root or STUDY / "data" / f"classfull_{a.regime}")
    out = Path(a.out or RES / f"classfull_{a.regime}_scores")
    seeds_root = STUDY / "reference" / "seeds" / a.regime
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]
    screen_keys = set(json.loads((STUDY / "tuning" / "splits.json").read_text(encoding="utf-8"))["screen"])
    screen = {(c["dataset"], c["scenario"]) for c in protocol if c["key"] in screen_keys}
    cases = [(d, s, f, (d, s) in screen) for d, s, f in cases_for(a.regime)]
    if a.cases == "screen":
        cases = [c for c in cases if c[3]]
    fam_of = {(d, s): f for d, s, f, _ in cases}

    # check 1: gate
    x, y = run_dir(cf / "levenshtein__melnik_devnone"), run_dir(l1 / "levenshtein__melnik")
    if x is None or y is None:
        raise SystemExit("gate impossible: levenshtein__melnik_devnone or layer 1 levenshtein__melnik missing")
    worst = 0.0
    for d, s, _, _ in cases:
        fa, fb = one(x / d / s / "_outputs" / "MatchingStepLine2"), one(y / d / s / "_outputs" / "MatchingStepLine2")
        if fa is None or fb is None:
            raise SystemExit(f"gate: output missing for {d}/{s}")
        A_, B_ = read_matrix(fa)[0], read_matrix(fb)[0]
        worst = max(worst, float(np.abs(A_ - B_).max()) if A_.shape == B_.shape else float("inf"))
    checks = [f"gate levenshtein__melnik_devnone = layer 1 levenshtein__melnik: max |difference| {worst}"]
    print(checks[-1])
    if worst != 0.0:
        raise SystemExit("gate FAILED: nothing is scored")

    # check 2: the class edges reached SF (graph growth exactly as the edge files imply). Checked wherever layer 1's
    # <m>__melnik has an output (partial-coverage matchers have fewer scenarios; review, 2026-10-04). Refused: a
    # class arm with an output there but no statistics row, or a wrong growth. Counted and reported only: scenarios
    # without a class-arm output (Panel drops such a matcher from every contrast) and class outputs without a layer-1
    # output (they cannot be checked and cannot enter a contrast against melnik).
    seeds = sorted({p.name.split("__", 1)[0] for p in cf.iterdir() if "__" in p.name and p.name.endswith(VARIANTS)})
    if a.matchers:
        seeds = [s for s in seeds if s in a.matchers.split(",")]

    def has_output(d, ds_, sc_):
        return d is not None and one(d / ds_ / sc_ / "_outputs" / "MatchingStepLine2") is not None

    exp_cache, checked, bad, no_stats, no_arm, unverifiable = {}, 0, 0, 0, 0, 0
    for s in seeds:
        base_dir = l1 / f"{s}__melnik"
        base_run = run_dir(base_dir)
        base_sizes = graph_sizes(base_dir)
        for v in VARIANTS:
            arm_dir = cf / f"{s}__melnik_{v}"
            arm_run = run_dir(arm_dir)
            sizes = graph_sizes(arm_dir)
            for d, sc, _, _ in cases:
                base_has, arm_has = has_output(base_run, d, sc), has_output(arm_run, d, sc)
                if not arm_has:
                    no_arm += 1
                    continue
                if not base_has:
                    unverifiable += 1
                    continue
                if (d, sc) not in sizes or (d, sc) not in base_sizes:
                    no_stats += 1
                    continue
                key = (v, d, sc)
                if key not in exp_cache:
                    exp_cache[key] = expected_growth(edges, v, d, sc)
                got = (sizes[(d, sc)][0] - base_sizes[(d, sc)][0], sizes[(d, sc)][1] - base_sizes[(d, sc)][1])
                checked += 1
                bad += got != exp_cache[key]
    checks.append(f"class edges in the schema graph (sf_stats growth = edge files, where layer 1 has an output): "
                  f"{checked} checked, {bad} wrong, {no_stats} outputs without statistics; reported only: "
                  f"{no_arm} (matcher, arm, scenario) without a class-arm output, {unverifiable} class outputs without "
                  f"a layer-1 output")
    print(checks[-1])
    if bad or no_stats or checked == 0:
        raise SystemExit("edge check FAILED (wrong growth or missing statistics): nothing is scored")

    jobs = [(a.regime, s, cases, cf, l1, edges, seeds_root) for s in seeds]
    with Pool(a.workers) as pool:
        parts = list(pool.imap_unordered(score_seed, jobs))
    rows = [r for p, _ in parts for r in p]
    inv = {v: [sum(i[v][0] for _, i in parts), sum(i[v][1] for _, i in parts)] for v in ("shape", "semtype")}
    skipped = {k: sum(i[k][0] for _, i in parts) for k in ("nonfinite_sf", "nonfinite_arm", "nonfinite_const", "missing_D")}
    checks.append("skipped cells per reason (reported only): "
                  f"layer-1 melnik output not finite or misshapen {skipped['nonfinite_sf']} (matcher, scenario; the scenario "
                  f"drops out for that matcher); class-arm output not finite or misshapen {skipped['nonfinite_arm']} "
                  f"(matcher, scenario, arm); constant__melnik not finite or misshapen {skipped['nonfinite_const']} "
                  f"(matcher, scenario; no H8a blocks there); datatype matrix D missing or not finite "
                  f"{skipped['missing_D']} (matcher, scenario; no A_p / A_s there)")
    print(checks[-1])
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)

    H = [("H8b (shape)", "shape", "shape_shuffled"), ("H8b (semtype)", "semtype", "semtype_shuffled"),
         ("H8c (shape)", "shape", "melnik"), ("H8c (semtype)", "semtype", "melnik"),
         ("H8d (shape)", "L_p", "shape"), ("H8d (semtype)", "L_s", "semtype"),
         ("H8e (shape)", "B_p", "shape"), ("H8e (semtype)", "B_s", "semtype"),
         ("report", "A_p", "shape"), ("report", "A_s", "semtype"),
         ("report", "shape_shuffled", "melnik"), ("report", "semtype_shuffled", "melnik")]
    if a.regime == "md5":
        H = [("H8f (semtype, equivalence ±0.01)" if lab == "H8c (semtype)" else
              ("report" if lab.startswith("H8") else lab), x, y) for lab, x, y in H]
    report = [f"# Amendment 8, class nodes on the full panel: regime {a.regime}", "", *[f"- {c} — passed" for c in checks], "",
              "J = (R_exp + F1_Hr)/2; per matcher the macro over families with ≥ 5 scenarios; mean over the included "
              "matchers; 95% bootstrap stratified by family, 10,000 resamples, seed 20261002. Confirmed: lower bound > 0 "
              "(H8f: interval inside (−0.01, +0.01)). Columns R_exp and F1_Hr: estimates of the parts.", ""]
    store = []
    sets = (("primary: non-screen", [k for k in fam_of if k not in screen]), ("secondary: all", list(fam_of)))
    for name, keys in ((f"{n} ({len(k)} scenarios)", k) for n, k in sets):
        if not keys:
            continue
        P = Panel(rows, keys, fam_of)
        if not P.fams:
            continue
        inc = sorted(s for s in {r["seed"] for r in rows} if not is_control(s))
        report += [f"## {name}", "", "| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 | verdict |",
                   "|---|---|---|---|---|---|---|"]
        for label, plus, minus in H:
            res = {}
            for f in ("J", "R_exp", "F1_Hr"):
                E, kept = P.matrix(inc, plus, minus, f"sf_{f}")
                if not kept:
                    break
                est, lo, hi = P.stat(E, np.nanmean)
                res[f] = (est, lo, hi, kept, E)
            if "J" not in res:
                report.append(f"| {label}: {plus} − {minus} | 0 | – | – | – | – | – |")
                continue
            est, lo, hi, kept, E = res["J"]
            pos = int((P.macro(E) > 0).sum())
            if label.startswith("H8f"):
                verdict = "CONFIRMED" if lo > -0.01 and hi < 0.01 else "not confirmed"
            elif label.startswith("H8"):
                verdict = "CONFIRMED" if lo > 0 else "not confirmed"
            else:
                verdict = ""
            report.append(f"| {label}: {plus} − {minus} | {len(kept)} | {est:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                          f"{res['R_exp'][0]:+.4f} | {res['F1_Hr'][0]:+.4f} | {pos} of {len(kept)} | {verdict} |")
            store.append({"set": name, "hypothesis": label, "contrast": f"{plus} − {minus}", "matchers": len(kept),
                          "estimate": est, "lo": lo, "hi": hi, "R_exp": res["R_exp"][0], "F1_Hr": res["F1_Hr"][0],
                          "matchers_positive": pos, "verdict": verdict})
        report.append("")
    for v in ("shape", "semtype"):
        verdict = "CONFIRMED" if inv[v][0] == 0 and inv[v][1] > 0 else "not confirmed"
        report.append(f"- H8a ({v}): {inv[v][0]} within-block inversions of {inv[v][1]} ordered pairs inside the true "
                      f"refined blocks (all matchers, all scenarios) — {verdict}.")
        store.append({"set": "all", "hypothesis": f"H8a ({v})", "contrast": "inversions", "estimate": inv[v][0], "verdict": verdict})
    with (out / "contrasts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for s in store for k in s)))
        w.writeheader()
        w.writerows(store)
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
