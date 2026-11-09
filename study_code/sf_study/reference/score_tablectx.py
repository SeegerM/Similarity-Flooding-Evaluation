"""Scorer for the table-context control (draft amendment: reference/AMENDMENT_TABLECTX_DRAFT.md; the external
reviewer's point 2, 2026-10-04). Written before any table-context metric was computed. Frozen with the draft;
a change after appending is an erratum.

Question: T1 showed that SF beats the column-level additive baselines on the independent corpus only with the correct
tables. Those baselines see no table context. Does a non-iterative baseline that does see it explain SF's advantage?

Per initial matcher m and scenario (81 scenarios of the independent corpus; plain), with q+ = max(q, 0):
    T(a, b)   table-pair score for source table a and target table b:
              1/2 (mean over columns i of a of max over columns j of b of q+_ij
                   + mean over columns j of b of max over columns i of a of q+_ij)
    That      T / max over table pairs of T (0 if that max is 0)
    C_ij      That(table(i), table(j)); table(x) is the label before the first '.'
    A_dt = q+ + 5 D + λ C                 (against SF as published, layer1_indep/<m>__melnik)
    A_pt = q+ + 5 D + 0.2 eq_p + λ C      (against SF + pattern-class nodes, er10_indep/<m>__melnik_shape)
    A_st = q+ + 5 D + 0.1 eq_s + λ C      (against SF + semantic-type nodes, er10_indep/<m>__melnik_semtype)
    D, eq_p, eq_s, the weights 5 / 0.2 / 0.1 and the matcher list are Amendment 5's (score_additive_indep.py,
    imported unchanged; reference/additive_indep_matchers.json).
λ is chosen per term from LAMBDAS by leave-one-database-out: for each held-out database, the λ with the highest mean
over the included matchers of the database-macro J_tf on the two other databases (ties: the smaller λ); the held-out
database's scenarios are scored with that λ. Nothing is chosen on a database's own scenarios.

Statistic (Amendment 5): J_tf = (R_exp + F1_Hr)/2; per scenario the mean over the included matchers; mean within
database; mean over the three databases; 95% percentile bootstrap over scenarios within database, 10,000 resamples,
seed 20261007. Included matchers: Amendment 5's frozen list with q, D, and all three SF arms finite on all 81 scenarios.

Decisions (draft §5): HC0 A_dt − A_d > 0 (confirmed: lower bound > 0 and positive in every database). HC1 (primary),
HC2, HC3 on SF − A_xt with a three-way rule: "propagation adds beyond table context" if the lower bound > 0 and the
estimate is positive in every database; "table context suffices" if the upper bound < 0 and the estimate is negative
in every database; otherwise "inconclusive", with "practically equivalent" added when the interval lies inside
(−0.01, +0.01).

    ../harness/.venv-bdi/Scripts/python.exe reference/score_tablectx.py --check-inputs   # no metrics
    ../harness/.venv-bdi/Scripts/python.exe reference/score_tablectx.py --selftest       # synthetic data only
    ../harness/.venv-bdi/Scripts/python.exe reference/score_tablectx.py                  # after the amendment
Output: tuning/results/tablectx_indep_scores/{scores.csv, lambdas.json, report.md}.
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
sys.path.insert(0, str(HERE))
from score_additive_indep import (CORPUS, STUDY, all_metrics, cases, classes, eq, ground_truth_file,  # noqa: E402
                                  load, matchers, one, permutations, read_matrix)

PREREG = HERE / "PREREGISTRATION_LAYERS.md"
OUT = STUDY / "tuning" / "results" / "tablectx_indep_scores"
SF_ARMS = {"sf": STUDY / "tuning" / "results" / "layer1_indep" / "{m}__melnik",
           "sf_p": STUDY / "tuning" / "results" / "er10_indep" / "{m}__melnik_shape",
           "sf_s": STUDY / "tuning" / "results" / "er10_indep" / "{m}__melnik_semtype"}
TERMS = {"A_dt": (5.0, 0.0, None), "A_pt": (5.0, 0.2, "f7"), "A_st": (5.0, 0.1, "f13")}
BASE_OF = {"A_dt": "A_d", "A_pt": "A_p", "A_st": "A_s"}
SF_OF = {"A_dt": "sf", "A_pt": "sf_p", "A_st": "sf_s"}
LAMBDAS = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
B, SEED = 10_000, 20261007
EQUIV = 0.01


def table_of(label: str) -> str:
    return label.split(".", 1)[0]


def table_context(q: np.ndarray, rows: list[str], cols: list[str]) -> np.ndarray:
    """C_ij = normalised table-pair score of (table(i), table(j)) from q+ (no iteration, no ground truth)."""
    qp = np.clip(q, 0, None)
    rt, ct = [table_of(r) for r in rows], [table_of(c) for c in cols]
    rows_of = defaultdict(list)
    cols_of = defaultdict(list)
    for i, t in enumerate(rt):
        rows_of[t].append(i)
    for j, t in enumerate(ct):
        cols_of[t].append(j)
    T = {}
    for a, ri in rows_of.items():
        for b, cj in cols_of.items():
            block = qp[np.ix_(ri, cj)]
            T[(a, b)] = 0.5 * (block.max(axis=1).mean() + block.max(axis=0).mean())
    top = max(T.values()) if T else 0.0
    C = np.zeros_like(qp)
    if top > 0:
        for (a, b), v in T.items():
            C[np.ix_(rows_of[a], cols_of[b])] = v / top
    return C


def sf_dir(arm: str, m: str) -> Path | None:
    d = Path(str(SF_ARMS[arm]).format(m=m))
    if not d.is_dir():
        return None
    if (d / "merged").is_dir():
        return d / "merged"
    runs = sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("_"))
    return runs[-1] if runs else None


def sf_matrix(arm: str, m: str, ds: str, sc: str, shape) -> np.ndarray | None:
    d = sf_dir(arm, m)
    f = one(d / ds / sc / "_outputs" / "MatchingStepLine2") if d else None
    if f is None:
        return None
    S = read_matrix(f)[0]
    return S if S.shape == shape and np.isfinite(S).all() else None


def matrices(q, D, rows, cols, cls, sfs: dict) -> dict[str, np.ndarray]:
    """Every matrix scored for one (matcher, scenario): the SF arms, A_d/A_p/A_s, and A_xt for every λ."""
    qp = np.clip(q, 0, None)
    C = table_context(q, rows, cols)
    out = {k: np.clip(v, 0, None) for k, v in sfs.items()}
    out["q"] = qp
    for name, (b, g, fam) in TERMS.items():
        base = qp + b * D + (g * eq(cls, rows, cols, fam) if fam else 0.0)
        out[BASE_OF[name]] = base
        for lam in LAMBDAS:
            out[f"{name}@{lam}"] = base + lam * C
    return out


def score_matcher(job):
    m, cs = job
    out = []
    for ds, sc in cs:
        x = load(m, ds, sc, need_sf=False)
        if x is None:
            return []
        q, D, _, rows, cols = x
        sfs = {arm: sf_matrix(arm, m, ds, sc, q.shape) for arm in SF_ARMS}
        if any(v is None for v in sfs.values()):
            return []                       # incomplete: the matcher is not used
        gt = read_matrix(ground_truth_file(CORPUS["indep"], ds, sc))[0] > 0.5
        perms = permutations(gt.shape, f"{ds}/{sc}")
        row = {"seed": m, "dataset": ds, "scenario": sc}
        for name, M in matrices(q, D, rows, cols, classes(ds, sc), sfs).items():
            r = all_metrics(M, gt, perms)
            row[f"{name}|J"] = (r["R_exp"] + r["F1_Hr"]) / 2
            row[f"{name}|R_exp"], row[f"{name}|F1_Hr"] = r["R_exp"], r["F1_Hr"]
        out.append(row)
    return out


def choose_lambdas(rows: list[dict], dbs: list[str]) -> dict[str, dict[str, float]]:
    """term -> held-out database -> λ chosen on the two other databases (mean over matchers of the database macro)."""
    used = sorted({r["seed"] for r in rows})
    by = defaultdict(list)
    for r in rows:
        by[(r["seed"], r["dataset"])].append(r)
    out = {}
    for name in TERMS:
        out[name] = {}
        for held in dbs:
            train = [d for d in dbs if d != held]

            def fit(lam):
                return np.mean([np.mean([np.mean([r[f"{name}@{lam}|J"] for r in by[(m, d)]]) for d in train]) for m in used])
            vals = {lam: fit(lam) for lam in LAMBDAS}
            best = max(vals.values())
            out[name][held] = min(lam for lam, v in vals.items() if v >= best - 1e-12)
    return out


def contrast(rows, dbs, f, rng):
    """Amendment 5's statistic for a per-row difference f(row)."""
    used = sorted({r["seed"] for r in rows})
    by = {(r["seed"], r["dataset"], r["scenario"]): r for r in rows}
    scen = sorted({(r["dataset"], r["scenario"]) for r in rows})
    per = {d: np.array([np.mean([f(by[(m, d, s)]) for m in used]) for dd, s in scen if dd == d]) for d in dbs}
    est = float(np.mean([v.mean() for v in per.values()]))
    boots = np.mean([[rng.choice(v, size=v.size, replace=True).mean() for _ in range(B)] for v in per.values()], axis=0)
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    fam = {d: float(v.mean()) for d, v in per.items()}
    per_m = [np.mean([np.mean([f(by[(m, d, s)]) for dd, s in scen if dd == d]) for d in dbs]) for m in used]
    return est, lo, hi, fam, sum(x > 1e-4 for x in per_m), len(used)


def decide(est, lo, hi, fam) -> str:
    if lo > 0 and all(v > 0 for v in fam.values()):
        return "propagation adds beyond table context"
    if hi < 0 and all(v < 0 for v in fam.values()):
        return "table context suffices"
    return "inconclusive" + ("; practically equivalent" if -EQUIV < lo and hi < EQUIV else "")


def report(rows: list[dict]) -> tuple[list[str], dict]:
    dbs = sorted({r["dataset"] for r in rows})
    lam = choose_lambdas(rows, dbs)
    for r in rows:                      # the held-out choice for the scenario's own database
        for name in TERMS:
            for part in ("J", "R_exp", "F1_Hr"):
                r[f"{name}|{part}"] = r[f"{name}@{lam[name][r['dataset']]}|{part}"]
    rng = np.random.default_rng(SEED)
    used = sorted({r["seed"] for r in rows})
    lines = ["# Table-context control: a non-iterative table-aware baseline against SF (independent corpus, plain)", "",
             f"{len(used)} initial matchers, {len({(r['dataset'], r['scenario']) for r in rows})} scenarios in "
             f"{', '.join(dbs)}. λ chosen leave-one-database-out per term: "
             + "; ".join(f"{n}: " + ", ".join(f"{d} {lam[n][d]}" for d in dbs) for n in TERMS) + ".", "",
             "| hypothesis | contrast | estimate | 95% CI | " + " | ".join(dbs) + " | matchers > 0 | decision |",
             "|---|---|---|---|" + "---|" * len(dbs) + "---|---|"]
    est, lo, hi, fam, pos, n = contrast(rows, dbs, lambda r: r["A_dt|J"] - r["A_d|J"], rng)
    ok = lo > 0 and all(v > 0 for v in fam.values())
    lines.append(f"| HC0 | A_dt − A_d | {est:+.4f} | [{lo:+.4f}, {hi:+.4f}] | " + " | ".join(f"{fam[d]:+.4f}" for d in dbs)
                 + f" | {pos} of {n} | {'CONFIRMED' if ok else 'not confirmed'} |")
    for h, name in (("HC1 (primary)", "A_dt"), ("HC2", "A_pt"), ("HC3", "A_st")):
        sf = SF_OF[name]
        est, lo, hi, fam, pos, n = contrast(rows, dbs, lambda r, s=sf, x=name: r[f"{s}|J"] - r[f"{x}|J"], rng)
        lines.append(f"| {h} | {sf} − {name} | {est:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     + " | ".join(f"{fam[d]:+.4f}" for d in dbs) + f" | {pos} of {n} | {decide(est, lo, hi, fam)} |")
    lines += ["", "Reported only: gains over q (macro over databases, mean over matchers), with the held-out λ:", "",
              "| matrix | J_tf | R_exp | F1_Hr |", "|---|---|---|---|"]
    for name in ("sf", "sf_p", "sf_s", "A_d", "A_p", "A_s", "A_dt", "A_pt", "A_st"):
        vals = [contrast(rows, dbs, lambda r, x=name, p=part: r[f"{x}|{p}"] - r[f"q|{p}"], rng)[0]
                for part in ("J", "R_exp", "F1_Hr")]
        lines.append(f"| {name} | " + " | ".join(f"{v:+.4f}" for v in vals) + " |")
    lines += ["", "Reported only: every λ on all databases (J_tf gain over q):", "",
              "| term | " + " | ".join(str(x) for x in LAMBDAS) + " |", "|---|" + "---|" * len(LAMBDAS)]
    for name in TERMS:
        vals = [contrast(rows, dbs, lambda r, x=f"{name}@{lv}": r[f"{x}|J"] - r["q|J"], rng)[0] for lv in LAMBDAS]
        lines.append(f"| {name} | " + " | ".join(f"{v:+.4f}" for v in vals) + " |")
    return lines, lam


def selftest() -> int:
    """Synthetic data only: table_context, the λ selection and the decision rule; no corpus file is read."""
    rows_l = ["a.x", "a.y", "b.z"]
    cols_l = ["c.u", "c.v", "d.w"]
    q = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 0.5]])
    C = table_context(q, rows_l, cols_l)
    assert np.isclose(C[0, 0], 1.0) and np.isclose(C[2, 2], 0.5 / 0.85) and C[0, 2] == 0.0, C
    rng = np.random.default_rng(0)
    rows = []
    for m in ("m1", "m2"):
        for d in ("X", "Y", "Z"):
            for s in range(6):
                r = {"seed": m, "dataset": d, "scenario": f"s{s}"}
                for part in ("J", "R_exp", "F1_Hr"):
                    r[f"q|{part}"] = 0.3
                    for k in ("sf", "sf_p", "sf_s", "A_d", "A_p", "A_s"):
                        r[f"{k}|{part}"] = 0.40 + 0.01 * rng.standard_normal()
                    for name in TERMS:
                        for lam in LAMBDAS:   # synthetic optimum at λ = 0.5
                            r[f"{name}@{lam}|{part}"] = 0.42 - 0.02 * abs(lam - 0.5) + 0.001 * rng.standard_normal()
                rows.append(r)
    lines, lam = report(rows)
    assert all(v == 0.5 for t in lam.values() for v in t.values()), lam
    assert decide(0.02, 0.01, 0.03, {"X": 0.02, "Y": 0.01}) == "propagation adds beyond table context"
    assert decide(-0.02, -0.03, -0.01, {"X": -0.02, "Y": -0.01}) == "table context suffices"
    assert decide(0.0, -0.005, 0.005, {"X": 0.0, "Y": 0.001}) == "inconclusive; practically equivalent"
    print("\n".join(lines).encode("ascii", "replace").decode())
    print("selftest passed")
    return 0


def check_inputs() -> int:
    cs = cases()
    ok, bad_labels = defaultdict(int), 0
    for m in matchers():
        for ds, sc in cs:
            x = load(m, ds, sc, need_sf=False)
            if x is None:
                continue
            _, _, _, rows, cols = x
            bad_labels += sum("." not in l for l in rows + cols)
            if all(sf_dir(arm, m) is not None and one(sf_dir(arm, m) / ds / sc / "_outputs" / "MatchingStepLine2")
                   is not None for arm in SF_ARMS):
                ok[m] += 1
    full = [m for m in matchers() if ok[m] == len(cs)]
    print(f"{len(cs)} scenarios; {len(matchers())} frozen matchers; {len(full)} with q, D and all three SF outputs on "
          f"every scenario (outputs located, not read); labels without a table prefix: {bad_labels}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-inputs", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.check_inputs:
        return check_inputs()
    if "table-context control" not in PREREG.read_text(encoding="utf-8"):
        raise SystemExit("The table-context amendment is not in PREREGISTRATION_LAYERS.md; the test does not run before it.")
    cs = cases()
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_matcher, [(m, cs) for m in matchers()]) for r in part]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines, lam = report(rows)
    (OUT / "lambdas.json").write_text(json.dumps(lam, indent=1), encoding="utf-8")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
