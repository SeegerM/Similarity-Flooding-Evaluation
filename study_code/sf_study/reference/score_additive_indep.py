"""Registered scorer for Amendment 5 of PREREGISTRATION_LAYERS.md (draft: reference/AMENDMENT_ADDITIVE_DRAFT.md):
does an additive datatype-and-class term beat SF as published on the independent multi-table corpus?

Written on 2026-10-03 before any layer-1 output of the independent corpus existed, and before any metric
of the additive terms on that corpus was computed. Its SHA-256 is recorded in the amendment. A change is a new
amendment.

Per initial matcher m and scenario (81 scenarios: AdventureWorks, Chinook, TPC-H, 27 each; plain):
    q      the initial matrix, reference/seeds/indep/<m>
    D      the datatype matcher without NaN cells, reference/seeds/indep/datatype_nanfix (X30)
    eq_p   1 where the source and target column have the same pattern class (layer3b/f7.py column_classes)
    eq_s   1 where they have the same rule-based semantic type (layer3b/f13.py column_types)
    SF     layer 1's `melnik` arm, tuning/results/layer1_indep/<m>__melnik
Additive terms, with the weights fixed from the dev selection of X26 (one value for all matchers):
    A_p = q + 5 D + 0.2 eq_p      (primary)
    A_s = q + 5 D + 0.1 eq_s
    A_d = q + 5 D
Metric: J_tf = (R_exp + F1_Hr) / 2 (tuning/score_screen_metrics.py; 50 seeded tie orders).

Hypotheses, each on the contrast A − SF in J_tf:
    HB1 (primary)  A_p − SF > 0
    HB2            A_s − SF > 0
    HB3            A_d − SF > 0
Statistic: per scenario, the mean difference over the included initial matchers; the mean within each
database; the mean over the three databases. 95% percentile bootstrap, resampling scenarios within each
database, 10,000 resamples, seed 20261003. A hypothesis is confirmed if the lower bound is above 0 AND the
estimate is positive in each of the three databases (the corpus has three independent sources, so the
scenario bootstrap alone overstates the evidence). Also reported: the number of matchers for which the
contrast is positive (macro over databases), and every part (R_exp, F1_Hr) and every term against q.

Included initial matchers: RQ1's non-control set (latex/sf-evaluation/gen_figures.py: in Table 1, not invalid,
not a control group, semstruct_lm_only folded into sbert_e5) with a finite q and D on all 81 scenarios, frozen
at registration in reference/additive_indep_matchers.json; of these, those with an SF output on all 81
scenarios are used, the others are listed and not used.

    ../harness/.venv-bdi/Scripts/python.exe reference/score_additive_indep.py --check-inputs   # no metrics, no SF
    ../harness/.venv-bdi/Scripts/python.exe reference/score_additive_indep.py                  # the registered test
Output: tuning/results/layer1_indep_scores/additive_{scores.csv,report.md}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "harness"))
sys.path.insert(0, str(STUDY / "tuning"))
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY / "layer3b"))

from smx.adapters import read_matrix  # noqa: E402
from score_screen_metrics import all_metrics, permutations  # noqa: E402
from score_seeds import CORPUS, ground_truth_file, one  # noqa: E402

PREREG = HERE / "PREREGISTRATION_LAYERS.md"
SEEDS = HERE / "seeds" / "indep"
DTYPE = SEEDS / "datatype_nanfix"
LAYER1 = STUDY / "tuning" / "results" / "layer1_indep"
OUT = STUDY / "tuning" / "results" / "layer1_indep_scores"
TERMS = {"A_p": (5.0, 0.2, "f7"), "A_s": (5.0, 0.1, "f13"), "A_d": (5.0, 0.0, None)}
B, SEED = 10_000, 20261003


def cases() -> list[tuple[str, str]]:
    root = CORPUS["indep"]
    return [(d.name, s.name) for d in sorted(root.iterdir()) if d.is_dir() for s in sorted(d.iterdir()) if s.is_dir()]


FROZEN = HERE / "additive_indep_matchers.json"   # frozen at registration (hash in the amendment)


def matchers() -> list[str]:
    if FROZEN.is_file():
        return json.loads(FROZEN.read_text(encoding="utf-8"))
    sys.path.insert(0, str(STUDY.parent / "latex" / "sf-evaluation"))
    import gen_figures as g
    return sorted(s for s in g.META if s not in g.INVALID and g.group_of(s) != "ctrl" and s != "semstruct_lm_only")


@lru_cache(maxsize=None)
def classes(ds: str, sc: str) -> dict[str, dict[str, str]]:
    """'table.column' -> class, per family, for one side's tables (one CSV per table)."""
    from f7 import column_classes
    from f13 import column_types
    out = {}
    for side in ("source", "target"):
        f7, f13 = {}, {}
        for t in sorted((CORPUS["indep"] / ds / sc / side).glob("*.csv")):
            h, shp = column_classes(t)
            _, sem = column_types(t)
            f7.update({f"{t.stem}.{c}": v for c, v in zip(h, shp)})
            f13.update({f"{t.stem}.{c}": v for c, v in zip(h, sem)})
        out[side] = {"f7": f7, "f13": f13}
    return out


def eq(cls: dict, rows: list[str], cols: list[str], fam: str) -> np.ndarray:
    s = [cls["source"][fam][r] for r in rows]
    t = [cls["target"][fam][c] for c in cols]
    return (np.array(s)[:, None] == np.array(t)[None, :]).astype(float)


def sf_dir(m: str) -> Path | None:
    d = LAYER1 / f"{m}__melnik"
    if not d.is_dir():
        return None
    if (d / "merged").is_dir():
        return d / "merged"
    runs = sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("_"))
    return runs[-1] if runs else None


def load(m: str, ds: str, sc: str, need_sf: bool):
    f_q, f_d = SEEDS / m / ds / sc / "seed.csv", DTYPE / ds / sc / "seed.csv"
    if not (f_q.is_file() and f_d.is_file()):
        return None
    q, rows, cols = read_matrix(f_q)
    D, rd, cd = read_matrix(f_d)
    if q.shape != D.shape or rows != rd or cols != cd or not (np.isfinite(q).all() and np.isfinite(D).all()):
        return None
    S = None
    if need_sf:
        d = sf_dir(m)
        f_s = one(d / ds / sc / "_outputs" / "MatchingStepLine2") if d else None
        if f_s is None:
            return None
        S = read_matrix(f_s)[0]
        if S.shape != q.shape or not np.isfinite(S).all():
            return None
    return q, D, S, [r for r in rows], [c for c in cols]


def check_inputs() -> int:
    cs = cases()
    cls = {f"{d}/{s}": classes(d, s) for d, s in cs}
    ok = defaultdict(int)
    missing_cls = 0
    for m in matchers():
        for ds, sc in cs:
            x = load(m, ds, sc, need_sf=False)
            if x is None:
                continue
            ok[m] += 1
            c = cls[f"{ds}/{sc}"]
            for fam in ("f7", "f13"):
                try:
                    eq(c, x[3], x[4], fam)
                except KeyError:
                    missing_cls += 1
    full = sorted(m for m in ok if ok[m] == len(cs))
    if not FROZEN.is_file():
        FROZEN.write_text(json.dumps(full, indent=0), encoding="utf-8")
        print(f"froze {len(full)} matchers -> {FROZEN}")
    print(f"{len(cs)} scenarios; {len(full)} matchers with q and D on every scenario; "
          f"{len([m for m in matchers() if m not in full])} without; class lookups failed: {missing_cls}")
    print("SF outputs present:", sum(sf_dir(m) is not None for m in full), "(not read)")
    return 0


def score_matcher(job):
    m, cs = job
    out = []
    for ds, sc in cs:
        x = load(m, ds, sc, need_sf=True)
        if x is None:
            return []                       # incomplete: the matcher is not used
        q, D, S, rows, cols = x
        c = classes(ds, sc)
        gt = read_matrix(ground_truth_file(CORPUS["indep"], ds, sc))[0] > 0.5
        perms = permutations(gt.shape, f"{ds}/{sc}")
        mats = {"q": np.clip(q, 0, None), "sf": np.clip(S, 0, None)}
        for name, (b, g, fam) in TERMS.items():
            mats[name] = np.clip(q, 0, None) + b * D + (g * eq(c, rows, cols, fam) if fam else 0.0)
        row = {"seed": m, "dataset": ds, "scenario": sc}
        for name, M in mats.items():
            r = all_metrics(M, gt, perms)
            row[f"{name}_R_exp"], row[f"{name}_F1_Hr"] = r["R_exp"], r["F1_Hr"]
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-inputs", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.check_inputs:
        return check_inputs()
    if "## Amendment 5" not in PREREG.read_text(encoding="utf-8"):
        raise SystemExit("Amendment 5 is not in PREREGISTRATION_LAYERS.md; the registered test does not run before it.")
    cs = cases()
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_matcher, [(m, cs) for m in matchers()]) for r in part]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "additive_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    used = sorted({r["seed"] for r in rows})
    dbs = sorted({r["dataset"] for r in rows})
    by = {(r["seed"], r["dataset"], r["scenario"]): r for r in rows}
    scen = sorted({(r["dataset"], r["scenario"]) for r in rows})
    J = lambda r, v: (r[f"{v}_R_exp"] + r[f"{v}_F1_Hr"]) / 2  # noqa: E731
    rng = np.random.default_rng(SEED)
    lines = ["# Amendment 5: additive datatype and class terms against SF as published (independent corpus, plain)", "",
             f"{len(used)} initial matchers with full coverage, {len(scen)} scenarios in {', '.join(dbs)}. "
             f"Not used (incomplete): {', '.join(sorted(set(matchers()) - set(used))) or 'none'}.", "",
             "| contrast | estimate | 95% CI | " + " | ".join(dbs) + " | matchers > 0 | verdict |",
             "|---|---|---|" + "---|" * len(dbs) + "---|---|"]
    for h, name in (("HB1", "A_p"), ("HB2", "A_s"), ("HB3", "A_d")):
        per = {d: np.array([np.mean([J(by[(m, d, s)], name) - J(by[(m, d, s)], "sf") for m in used])
                            for dd, s in scen if dd == d]) for d in dbs}
        est = float(np.mean([v.mean() for v in per.values()]))
        boots = np.mean([[rng.choice(v, size=v.size, replace=True).mean() for _ in range(B)] for v in per.values()], axis=0)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        fam_est = {d: float(v.mean()) for d, v in per.items()}
        per_m = [np.mean([np.mean([J(by[(m, d, s)], name) - J(by[(m, d, s)], "sf") for dd, s in scen if dd == d])
                          for d in dbs]) for m in used]
        ok = lo > 0 and all(v > 0 for v in fam_est.values())
        lines.append(f"| {h} {name} − SF | {est:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     + " | ".join(f"{fam_est[d]:+.4f}" for d in dbs)
                     + f" | {sum(x > 1e-4 for x in per_m)} of {len(used)} | {'CONFIRMED' if ok else 'not confirmed'} |")
    lines += ["", "Descriptive (macro over databases, mean over matchers), gains over q:", "",
              "| matrix | J_tf | R_exp | F1_Hr |", "|---|---|---|---|"]
    for name in ("sf", "A_d", "A_p", "A_s"):
        vals = []
        for part in ("J", "R_exp", "F1_Hr"):
            f = (lambda r, v=name: J(r, v) - J(r, "q")) if part == "J" else (lambda r, v=name, p=part: r[f"{v}_{p}"] - r[f"q_{p}"])
            vals.append(np.mean([np.mean([np.mean([f(by[(m, d, s)]) for dd, s in scen if dd == d]) for d in dbs]) for m in used]))
        lines.append(f"| {name} | " + " | ".join(f"{v:+.4f}" for v in vals) + " |")
    (OUT / "additive_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
