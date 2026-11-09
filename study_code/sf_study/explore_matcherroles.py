"""EXPLORATORY X73 (2026-10-06): role terms from every matcher, one measure in three uses. Written before
any role feature was computed; design and predictions in EXPLORATORY.md X73.

For initial matcher X (make_x62.INITS) and measure m (X62's edge sets minus extrema), per scenario:
  EDGE      SF(X) + m's within-table edges                     tuning/results/x62_<r>/<X>__e_<m>  (existing)
  ROLE(γ)   q_X+ + 5D + γ R_m                                    (no SF)
  POST(γ)   SF(X)+ + 5D + γ R_m                                  (fusion after propagation; SF as published)
  R_m(i, j) = 1 − mean |f(i) − f(j)|, f = (twin, typ, share) of the column's off-diagonal row in m's within-table matrix
            (reference/seeds/self_<r>/<m>/<ds>/<sc>__<side>/seed.csv), values clipped to [0, 1], share = fraction >= 0.5
References: SF(X); A_d = ROLE(0); SFD = POST(0); dependency role terms R6 and Rg (explore_genroles) as ROLE.
γ in GAMMA, chosen leave-one-family-out per (X, m, form). J_tf as explore_composite.J; family macro; 95% bootstrap within
family, 10,000 resamples, seed 20261073.

    ../harness/.venv-bdi/Scripts/python.exe explore_matcherroles.py --regime plain|md5 [--workers 10]
Output: results/<regime>/matcherroles_{scores.csv,summary.md}
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "harness", HERE / "tuning", HERE, HERE / "reference", HERE / "servers"):
    sys.path.insert(0, str(p))
import score_layer1  # noqa: E402
from explore_composite import J, load, sf_out  # noqa: E402
from explore_genroles import side_features, sim  # noqa: E402
from make_x62 import INITS  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402
from score_seeds import CORPUS, cases_for, ground_truth_file  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

RES = HERE / "tuning" / "results"
GAMMA = (0.1, 0.2, 0.5, 1.0, 2.0)
EXCLUDE = {"extrema"}
TAU = 0.5
B, SEED = 10_000, 20261073


def measures(regime: str) -> list[str]:
    d = HERE / "data" / f"x62_{regime}"
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name not in EXCLUDE) if d.is_dir() else []


def family_of() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("gt", HERE.parent / "latex" / "sf-evaluation" / "gen_tables.py")
    gt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gt)
    return {m: v[0] for m, v in gt.META.items()}


def role_features(regime: str, m: str, ds: str, sc: str, side: str) -> dict[str, np.ndarray]:
    """bare column name -> (twin, typ, share) from m's within-table matrix of one side."""
    f = HERE / "reference" / "seeds" / f"self_{regime}" / m / ds / f"{sc}__{side}" / "seed.csv"
    if not f.is_file():
        return {}
    M, rows, _ = read_matrix(f)
    M = np.clip(np.nan_to_num(np.asarray(M, dtype=float), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    out = {}
    n = M.shape[0]
    for i, r in enumerate(rows):
        other = np.delete(M[i], i) if n > 1 else np.zeros(0)
        out[r.split(".", 1)[1]] = (np.array([other.max(), other.mean(), (other >= TAU).mean()])
                                   if other.size else np.zeros(3))
    return out


def rdir(p: Path):
    return score_layer1.run_dir(p) if p.is_dir() else None


def score_case(job):
    regime, ds, sc, fam, ms = job
    seeds = HERE / "reference" / "seeds" / regime
    g_f = ground_truth_file(CORPUS[regime], ds, sc)
    f_d = seeds / "datatype_nanfix" / ds / sc / "seed.csv"
    if g_f is None or not f_d.is_file():
        return []
    D, rows, cols = read_matrix(f_d)
    rows, cols = list(rows), list(cols)
    gt = read_matrix(g_f)[0] > 0.5
    if D.shape != gt.shape or not np.isfinite(D).all():
        return []
    perms = permutations(gt.shape, f"{ds}/{sc}")
    rn, cn = [r.split(".", 1)[1] for r in rows], [c.split(".", 1)[1] for c in cols]
    Fs, _ = side_features(regime, ds, sc, "source", rows)
    Ft, _ = side_features(regime, ds, sc, "target", cols)
    Rdep = {"dep_R6": sim(Fs["6"], Ft["6"]), "dep_Rg": sim(Fs["g"], Ft["g"])}
    Rm = {}
    for m in ms:
        fs, ft = role_features(regime, m, ds, sc, "source"), role_features(regime, m, ds, sc, "target")
        if not fs or not ft:
            continue
        A = np.array([fs.get(x, np.zeros(3)) for x in rn])
        Bm = np.array([ft.get(x, np.zeros(3)) for x in cn])
        Rm[m] = sim(A, Bm)
    out = []
    for X in INITS[regime]:
        q = load(seeds / X / ds / sc / "seed.csv", rows, cols)
        S = load(sf_out(rdir(RES / f"layer1_{regime}" / f"{X}__melnik"), ds, sc), rows, cols)
        if q is None or S is None:
            continue
        qb, Sb = np.clip(q, 0, None) + 5.0 * D, np.clip(S, 0, None) + 5.0 * D
        base = {"init": X, "dataset": ds, "scenario": sc, "family": fam}
        ref = {**base, "measure": "_ref", "SF": J(np.clip(S, 0, None), gt, perms), "A_d": J(qb, gt, perms),
               "SFD": J(Sb, gt, perms)}
        for name, R in Rdep.items():
            for gm in GAMMA:
                ref[f"{name}|{gm}"] = J(qb + gm * R, gt, perms)
        out.append(ref)
        for m, R in Rm.items():
            row = {**base, "measure": m}
            E = load(sf_out(rdir(RES / f"x62_{regime}" / f"{X}__e_{m}"), ds, sc), rows, cols)
            if E is not None:
                row["EDGE"] = J(np.clip(E, 0, None), gt, perms)
            for gm in GAMMA:
                row[f"ROLE|{gm}"] = J(qb + gm * R, gt, perms)
                row[f"POST|{gm}"] = J(Sb + gm * R, gt, perms)
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5"), required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    import os
    if os.environ.get("SMX_VALENTINE", "382") == "549":
        return main549(a.regime, a.workers)
    ms = measures(a.regime)
    cs = cases_for(a.regime)
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_case, [(a.regime, *c, ms) for c in cs]) for r in part]
    out = HERE / "results" / a.regime
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "matcherroles_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    summarise(a.regime, rows, cs, ms, out, INITS[a.regime])
    return 0


def summarise(regime, rows, cs, ms, out: Path, inits) -> None:
    """main()'s summary, unchanged (moved into a function 2026-11-02 for the 549 run; inits = INITS[regime] there)."""
    class _A:
        pass
    a = _A()
    a.regime = regime
    INITS_ = {regime: tuple(inits)}
    fams = defaultdict(list)
    for d, c, f in cs:
        fams[f].append((d, c))
    big = {f: v for f, v in fams.items() if len(v) >= 5}
    fam_of = {(d, c): f for d, c, f in cs}
    by = {(r["init"], r["measure"], r["dataset"], r["scenario"]): r for r in rows}
    rng = np.random.default_rng(SEED)
    idx = {f: rng.integers(0, len(v), size=(B, len(v))) for f, v in big.items()}
    mfam = family_of()

    def macro(get, famset):
        return float(np.mean([np.mean([get(d, c) for d, c in fams[f]]) for f in famset]))

    def chosen(X, m, form):
        """per-scenario value with γ chosen leave-one-family-out; None if incomplete."""
        try:
            sc_ = {}
            for held in list(big) + ["_other"]:
                train = [f for f in big if f != held]
                vals = {gm: macro(lambda d, c, gm=gm: by[(X, m, d, c)][f"{form}|{gm}"], train) for gm in GAMMA}
                best = max(vals.values())
                sc_[held] = min(gm for gm, v in vals.items() if v >= best - 1e-12)
            return (lambda d, c: by[(X, m, d, c)][f"{form}|{sc_.get(fam_of[(d, c)], sc_['_other'])}"]), sc_
        except KeyError:
            return None, None

    def stat(fx, fy):
        per = {f: np.array([fx(d, c) - fy(d, c) for d, c in v]) for f, v in big.items()}
        est = float(np.mean([v.mean() for v in per.values()]))
        boots = np.mean([per[f][idx[f]].mean(axis=1) for f in big], axis=0)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return est, float(lo), float(hi)

    fmt = lambda e: f"{e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}]"  # noqa: E731
    sig = lambda e: "+" if e[1] > 0 else "−" if e[2] < 0 else "0"  # noqa: E731
    lines = [f"# X73 ({a.regime}): role terms from every matcher; role vs edges vs fusion after propagation (EXPLORATORY)", "",
             f"{len(ms)} measures; initial matchers {', '.join(INITS_[a.regime])}; {len(cs)} scenarios. J_tf, family macro, 95% CI; "
             "γ chosen leave-one-family-out from " + str(GAMMA) + ". Signs: + / − = interval excludes 0.", ""]
    for X in INITS_[a.regime]:
        ref = lambda k: (lambda d, c: by[(X, "_ref", d, c)][k])  # noqa: E731
        dep = {}   # dependency role terms, stored as '<name>|γ' on the reference row
        lines += [f"## initial matcher {X}", "",
                  f"References: SF {macro(ref('SF'), big):.4f}; A_d {macro(ref('A_d'), big):.4f}; SF + 5D {macro(ref('SFD'), big):.4f}.", ""]
        for name in ("dep_R6", "dep_Rg"):
            vals = {}
            for held in list(big) + ["_other"]:
                train = [f for f in big if f != held]
                sc_ = {gm: macro(lambda d, c, gm=gm: by[(X, "_ref", d, c)][f"{name}|{gm}"], train) for gm in GAMMA}
                best = max(sc_.values())
                vals[held] = min(gm for gm, v in sc_.items() if v >= best - 1e-12)
            dep[name] = lambda d, c, name=name, vals=vals: by[(X, "_ref", d, c)][f"{name}|{vals.get(fam_of[(d, c)], vals['_other'])}"]
            lines.append(f"- {name} (dependency role) − A_d: {fmt(stat(dep[name], ref('A_d')))}; chosen γ {vals}")
        lines += ["", "| measure | family | ROLE − A_d | EDGE − SF | ROLE − EDGE | POST − ROLE | POST − SFD | γ ROLE / POST (all) |",
                  "|---|---|---|---|---|---|---|---|"]
        res = []
        for m in ms:
            fr, chr_ = chosen(X, m, "ROLE")
            fp, chp = chosen(X, m, "POST")
            if fr is None or fp is None or not all("EDGE" in by.get((X, m, d, c), {}) for f in big for d, c in fams[f]):
                continue
            fe = lambda d, c, m=m: by[(X, m, d, c)]["EDGE"]  # noqa: E731
            e1, e2, e3 = stat(fr, ref("A_d")), stat(fe, ref("SF")), stat(fr, fe)
            e4, e5 = stat(fp, fr), stat(fp, ref("SFD"))
            res.append((m, mfam.get(m, "?"), e1, e2, e3, e4, e5))
            lines.append(f"| {m} | {mfam.get(m, '?')} | {fmt(e1)} | {fmt(e2)} | {fmt(e3)} | {fmt(e4)} | {fmt(e5)} | "
                         f"{chr_['_other']} / {chp['_other']} |")
        n = len(res)
        cnt = lambda i, s: sum(sig(r[i]) == s for r in res)  # noqa: E731
        lines += ["", f"Counts over {n} measures (significant +/−): ROLE − A_d {cnt(2, '+')}/{cnt(2, '−')}; EDGE − SF "
                  f"{cnt(3, '+')}/{cnt(3, '−')}; ROLE − EDGE {cnt(4, '+')}/{cnt(4, '−')} (point estimate > 0: "
                  f"{sum(r[4][0] > 0 for r in res)}); POST − ROLE {cnt(5, '+')}/{cnt(5, '−')}; POST − SFD {cnt(6, '+')}/{cnt(6, '−')}.",
                  "", "Family means (ROLE − A_d | EDGE − SF | ROLE − EDGE | POST − ROLE):", ""]
        byf = defaultdict(list)
        for r in res:
            byf[r[1]].append(r)
        for f, rs in sorted(byf.items(), key=lambda kv: -np.mean([r[2][0] for r in kv[1]])):
            lines.append(f"- {f} ({len(rs)}): " + " | ".join(f"{np.mean([r[i][0] for r in rs]):+.4f}" for i in (2, 3, 4, 5)))
        if res:
            best = max(res, key=lambda r: r[2][0])
            fb, _ = chosen(X, best[0], "ROLE")
            lines += ["", f"Best non-dependency role: {best[0]} (ROLE − A_d {best[2][0]:+.4f}); best − dep_R6: "
                      f"{fmt(stat(fb, dep['dep_R6']))}; best − dep_Rg: {fmt(stat(fb, dep['dep_Rg']))}", ""]
    (out / "matcherroles_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


def score_case549(job):
    """score_case() for one Valentine readable scenario of the 549, initial matcher levenshtein only: q, D, SF as
    published by the assembly rules (valentine549), within-table matrices from reference/seeds/self_plain (381) or
    self_pnew (168), EDGE runs from x62_plain (381) or x62_pnew (168), dependency roles on the fixed FD profiles."""
    import valentine549 as v
    regime, ds, sc, fam, ms = job
    new = v.is_new(ds, sc)
    g_f = ground_truth_file(v.NEW_TREE[regime] if new else CORPUS[regime], ds, sc)
    f_d = v.seed_file(regime, "datatype_nanfix", ds, sc)
    if g_f is None or not f_d.is_file():
        return []
    D, rows, cols = read_matrix(f_d)
    rows, cols = list(rows), list(cols)
    gt = read_matrix(g_f)[0] > 0.5
    if D.shape != gt.shape or not np.isfinite(D).all():
        return []
    perms = permutations(gt.shape, f"{ds}/{sc}")
    rn, cn = [r.split(".", 1)[1] for r in rows], [c.split(".", 1)[1] for c in cols]
    Fs, _ = side_features(regime, ds, sc, "source", rows)
    Ft, _ = side_features(regime, ds, sc, "target", cols)
    Rdep = {"dep_R6": sim(Fs["6"], Ft["6"]), "dep_Rg": sim(Fs["g"], Ft["g"])}
    selfreg = "pnew" if new else regime
    Rm = {}
    for m in ms:
        fs, ft = role_features(selfreg, m, ds, sc, "source"), role_features(selfreg, m, ds, sc, "target")
        if not fs or not ft:
            continue
        A = np.array([fs.get(x, np.zeros(3)) for x in rn])
        Bm = np.array([ft.get(x, np.zeros(3)) for x in cn])
        Rm[m] = sim(A, Bm)
    out = []
    for X in ("levenshtein",):
        q = load(v.seed_file(regime, X, ds, sc), rows, cols)
        S = load(sf_out(v.sf_run(regime, X, "melnik", ds, sc), ds, sc), rows, cols)
        if q is None or S is None:
            continue
        qb, Sb = np.clip(q, 0, None) + 5.0 * D, np.clip(S, 0, None) + 5.0 * D
        base = {"init": X, "dataset": ds, "scenario": sc, "family": fam}
        ref = {**base, "measure": "_ref", "SF": J(np.clip(S, 0, None), gt, perms), "A_d": J(qb, gt, perms),
               "SFD": J(Sb, gt, perms)}
        for name, R in Rdep.items():
            for gm in GAMMA:
                ref[f"{name}|{gm}"] = J(qb + gm * R, gt, perms)
        out.append(ref)
        for m, R in Rm.items():
            row = {**base, "measure": m}
            E = load(sf_out(rdir(RES / ("x62_pnew" if new else f"x62_{regime}") / f"{X}__e_{m}"), ds, sc), rows, cols)
            if E is not None:
                row["EDGE"] = J(np.clip(E, 0, None), gt, perms)
            for gm in GAMMA:
                row[f"ROLE|{gm}"] = J(qb + gm * R, gt, perms)
                row[f"POST|{gm}"] = J(Sb + gm * R, gt, perms)
            out.append(row)
    return out


def main549(regime: str, workers: int) -> int:
    """SMX_VALENTINE=549 (readable only): X73 for levenshtein on the 549 scenarios; pending until the within-table
    matrices of the 168 (reference/seeds/self_pnew) and their EDGE runs (tuning/results/x62_pnew) exist."""
    import valentine549 as v
    if regime != "plain":
        print("X73 549: only the readable regime is rerun (levenshtein); nothing written")
        return 0
    if not (HERE / "reference" / "seeds" / "self_pnew").is_dir() or not (RES / "x62_pnew" / "COLLECTED").is_file():
        print("X73 549: pending (reference/seeds/self_pnew or tuning/results/x62_pnew not there yet); nothing written")
        return 0
    ms = measures(regime)
    cs = v.cases549()
    with Pool(workers) as pool:
        rows = [r for part in pool.imap_unordered(score_case549, [(regime, *c, ms) for c in cs]) for r in part]
    out = HERE / "results" / v.vreg(regime)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "matcherroles_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    summarise(regime, rows, cs, ms, out, ("levenshtein",))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
