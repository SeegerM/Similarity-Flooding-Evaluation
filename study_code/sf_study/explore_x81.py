"""EXPLORATORY X81 (2026-10-07): Amendment 12's tests (E1-E5 = F1-F5) on the external corpora, with the frozen terms.
Plan and predictions in EXPLORATORY.md X81, written before any X81 output was read.

The term construction copies reference/score_final.py's score_case (frozen; not edited, not importable for these paths
because it ties the seed and run folders to one corpus name). macro_contrast, f3, TERMS, ARMS and B are imported from it.
    A_d = q+ + 5D;  A_role = A_d + 0.2R (plain gamma; all new corpora have readable names);  A_dt = A_d + 0.5C
SF arms: tuning/results/x81_<c>/<m>__{melnik, fd, ucc_top1, ucc1_fd_topk}. Initial matrices: reference/seeds/<c>.
Groups: gdcsm one; goby one; madi_indep the 4 independent domains; madi_products separately; mimic one scenario.
A matcher is included if it has every arm on every scenario of the corpus (score_final.cell_values).

    ../harness/.venv-bdi/Scripts/python.exe explore_x81.py [--corpora gdcsm,madi,goby,mimic] [--exclude-fd hdxsm:pair_53,...]
Output: results/x81/{scores_<c>.csv, report.md}
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
for p in (HERE.parent / "harness", HERE / "tuning", HERE, HERE / "reference"):
    sys.path.insert(0, str(p))
from score_final import ARMS, SEED, TERMS, cell_values, f3, macro_contrast  # noqa: E402
import valentine549 as V  # noqa: E402  (rerun 549 / HyFD fix: used only under SMX_VALENTINE=549)

GEN = HERE.parent / "harness" / "generated"
ROOTS = {"hdxsm": GEN / "hdxsm-usable", "gdcsm": GEN / "gdcsm-imported", "madi": GEN / "madi-imported",
         "goby": GEN / "goby-imported", "mimic": HERE / "data" / "mimic_omop"}
GAMMA = 0.2
OUT = HERE / "results" / "x81"
# SMX_VALENTINE=549 (fixed HyFD profiles): FD arms (fd, ucc1_fd_topk) from x81fix_<c>, the other arms from x81_<c>, the
# role vectors of A_role from the fixed-FD mirror harness/generated_hyfdfix/<c>; a corpus whose x81fix_<c> is not
# collected is reported as pending. Output results/x81fix/ (the 382-era results/x81/ is not touched).
if V.HYFDFIX:
    OUT = HERE / "results" / "x81fix"
FD_ARMS = ("fd", "ucc1_fd_topk")


def group_of(c: str, sc: str) -> tuple[str, str]:
    """(reported corpus, macro group)."""
    if c == "madi":
        dom = sc.split("__")[0]
        return ("madi_products" if dom == "products" else "madi_indep"), dom
    return c, c


def cases(c: str):
    root = ROOTS[c]
    return [(d.name, s.name) for d in sorted(p for p in root.iterdir() if p.is_dir())
            for s in sorted(p for p in d.iterdir() if p.is_dir())]


def matchers(c: str) -> list[str]:
    ms = {d.name.split("__")[0] for d in (HERE / "tuning" / "results" / f"x81_{c}").iterdir() if "__" in d.name}
    return sorted(ms)


def score_case(job):
    c, ds, sc, ms = job
    import score_layer1
    import score_seeds
    from explore_composite import load, sf_out
    from explore_deproles import profiles, roles
    from score_screen_metrics import all_metrics, permutations
    from score_tablectx import table_context
    from smx.adapters import read_matrix
    score_seeds.CORPUS[c] = V.FIXED_MIRROR / c if V.HYFDFIX else ROOTS[c]   # profiles (role vectors) only
    seeds = HERE / "reference" / "seeds" / c
    f_d = seeds / "datatype_nanfix" / ds / sc / "seed.csv"
    g = score_seeds.ground_truth_file(ROOTS[c], ds, sc)
    if g is None or not f_d.is_file():
        return [], f"{ds}/{sc}: no truth or no datatype matrix"
    D, rows, cols = read_matrix(f_d)
    rows, cols, D = list(rows), list(cols), np.nan_to_num(D)
    gt = read_matrix(g)[0] > 0.5
    if gt.shape != D.shape or not gt.any():
        return [], f"{ds}/{sc}: truth shape {gt.shape} vs {D.shape} or empty"
    Fs, _ = roles(rows, *profiles(c, ds, sc, "source"))
    Ft, _ = roles(cols, *profiles(c, ds, sc, "target"))
    R = 1.0 - np.abs(Fs[:, None, :] - Ft[None, :, :]).mean(axis=2)
    perms = permutations(gt.shape, f"{ds}/{sc}")
    rep, grp = group_of(c, sc)
    out = []
    for m in ms:
        q = load(seeds / m / ds / sc / "seed.csv", rows, cols)
        if q is None:
            continue
        qp = np.clip(q, 0, None)
        A_d = qp + 5.0 * D
        mats = {"q": qp, "A_d": A_d, "A_role": A_d + GAMMA * R, "A_dt": A_d + 0.5 * table_context(q, rows, cols)}
        for arm in ARMS:
            rd = HERE / "tuning" / "results" / V.run_stage(f"x81_{c}", arm) / f"{m}__{arm}"
            S = load(sf_out(score_layer1.run_dir(rd) if rd.is_dir() else None, ds, sc), rows, cols)
            if S is not None and np.isfinite(S).all():
                mats[arm] = np.clip(S, 0, None)
        row = {"seed": m, "corpus": rep, "group": grp, "dataset": ds, "scenario": sc}
        for name, M in mats.items():
            r = all_metrics(M, gt, perms)
            row[name], row[f"{name}_R"] = (r["R_exp"] + r["F1_Hr"]) / 2, r["R_exp"]
        out.append(row)
    return out, None


def ci(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def report(rep: str, rows: list[dict], excl: set, rng) -> list[str]:
    lines = [f"## {rep}", ""]
    full = [*TERMS, *ARMS, "q_R", "melnik_R", "fd_R"]
    nofd = [x for x in full if x not in (*FD_ARMS, "fd_R")]
    inc, means, by = cell_values([r for r in rows if (r["dataset"], r["scenario"]) not in excl] if excl else rows, full)
    inc_all, means_all, by_all = cell_values(rows, nofd)
    nsc = len({(r["dataset"], r["scenario"]) for r in rows})
    lines.append(f"- scenarios {nsc}; groups {sorted({r['group'] for r in rows})}; matchers included {len(inc)} "
                 f"(FD arms), {len(inc_all)} (other arms); FD-arm exclusions {len(excl)}")
    if not inc_all:
        return lines + ["- no matcher has every arm on every scenario", ""]
    avg = {k: np.mean([v[k] for v in means_all.values()]) for k in nofd if k in TERMS or k in ARMS}
    lines.append("- mean J_tf: " + ", ".join(f"{k} {v:.4f}" for k, v in avg.items()))
    if inc:
        lines.append(f"- mean J_tf, FD arms: fd {np.mean([v['fd'] for v in means.values()]):.4f}, ucc1_fd_topk "
                     f"{np.mean([v['ucc1_fd_topk'] for v in means.values()]):.4f}")
    if nsc == 1:   # mimic: descriptive per matcher
        (k, v), = by_all.items()
        d = [v[m]["A_dt"] - v[m]["melnik"] for m in inc_all]
        lines += [f"- E5 (descriptive, n = 1): A_dt − SF(melnik) mean over {len(d)} matchers {np.mean(d):+.4f}; "
                  f"positive for {sum(x > 0 for x in d)} of {len(d)}",
                  f"- E1 A_d − SF(melnik) {np.mean([v[m]['A_d'] - v[m]['melnik'] for m in inc_all]):+.4f}; "
                  f"E2 A_role − SF(ucc_top1) {np.mean([v[m]['A_role'] - v[m]['ucc_top1'] for m in inc_all]):+.4f}", ""]
        return lines
    est = {"E1 A_d − SF(melnik)": macro_contrast(means_all, "A_d", "melnik", rng),
           "E2a A_role − SF(ucc_top1)": macro_contrast(means_all, "A_role", "ucc_top1", rng),
           "E5' A_dt − SF(melnik)": macro_contrast(means_all, "A_dt", "melnik", rng),
           "E3 ρ(q R_exp, melnik gain R_exp)": f3(by_all, inc_all, rng)}
    if inc:
        est["E2b A_role − SF(ucc1_fd_topk)"] = macro_contrast(means, "A_role", "ucc1_fd_topk", rng)
        est["E4 R_exp SF(fd) − SF(melnik)"] = macro_contrast(means, "fd_R", "melnik_R", rng)
    rule = {"E1": lambda t: t[1] > 0, "E2a": lambda t: t[1] > -0.01, "E2b": lambda t: t[1] > -0.01,
            "E3": lambda t: t[2] < 0, "E4": lambda t: t[2] < 0, "E5'": lambda t: t[1] > 0}
    lines += ["", "| test | estimate [95%] | rule met |", "|---|---|---|"]
    for k, t in est.items():
        lines.append(f"| {k} | {ci(t)} | {'yes' if rule[k.split()[0]](t) else 'no'} |")
    # cluster-aware intervals (X86 clustering: scenarios sharing a table form one cluster; >= 5 clusters required)
    cl = clusters(rep, sorted({k[2] for k in means_all}))
    ncl = len(set(cl.values()))
    pairs = {"E1 A_d − SF(melnik)": (means_all, "A_d", "melnik"), "E2a A_role − SF(ucc_top1)": (means_all, "A_role", "ucc_top1"),
             "E5' A_dt − SF(melnik)": (means_all, "A_dt", "melnik")}
    if inc:
        pairs["E2b A_role − SF(ucc1_fd_topk)"] = (means, "A_role", "ucc1_fd_topk")
        pairs["E4 R_exp SF(fd) − SF(melnik)"] = (means, "fd_R", "melnik_R")
    lines += ["", f"Cluster-aware intervals ({ncl} clusters; resampling whole clusters, 10,000 draws; X86 clustering):", "",
              "| test | estimate [95% cluster] | rule met (cluster) |", "|---|---|---|"]
    for k, (mm, x, y) in pairs.items():
        if ncl < 5:
            lines.append(f"| {k} | {est[k][0]:+.4f} (only {ncl} clusters; no interval) | – |")
            continue
        t = cluster_ci(mm, x, y, cl, np.random.default_rng(SEED))
        lines.append(f"| {k} | {ci(t)} | {'yes' if rule[k.split()[0]](t) else 'no'} |")
    return lines + [""]


def clusters(rep: str, scenarios: list[str]) -> dict[str, str]:
    """scenario -> cluster, as layer3b/x86_cluster_ci.py (X86): table-sharing components; GDC-SM by source study."""
    sys.path.insert(0, str(HERE / "layer3b"))
    from x86_cluster_ci import cluster_map
    m, _ = cluster_map(rep, scenarios)
    return m


def cluster_ci(means: dict, x: str, y: str, cl: dict, rng) -> tuple[float, float, float]:
    """Mean over scenarios of x − y; 95% percentile bootstrap resampling whole clusters (ratio estimator)."""
    sums, cnt = defaultdict(float), defaultdict(int)
    for (_, _, sc), v in means.items():
        sums[cl[sc]] += v[x] - v[y]
        cnt[cl[sc]] += 1
    ks = sorted(sums)
    s, n = np.array([sums[k] for k in ks]), np.array([cnt[k] for k in ks])
    est = float(s.sum() / n.sum())
    idx = rng.integers(0, len(ks), size=(10_000, len(ks)))
    boots = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="gdcsm,madi,goby,mimic")
    ap.add_argument("--exclude-fd", default="", help="c:scenario,... excluded from the FD-arm contrasts only")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    excl = defaultdict(set)
    for x in filter(None, a.exclude_fd.split(",")):
        c, sc = x.split(":")
        excl[c].add(sc)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    lines = ["# X81: Amendment 12's tests on the external corpora (EXPLORATORY)", "",
             "Frozen terms (A_d, A_role γ 0.2, A_dt λ 0.5); 15-matcher panel; SF arms of x81_<c>. Rules as Amendment 12: E1 lower > 0; "
             "E2 lower > −0.01; E3 upper < 0; E4 upper < 0; E5' (single-table A_dt) lower > 0. Predictions: EXPLORATORY.md X81.", ""]
    if V.HYFDFIX:
        lines.insert(2, "Fixed HyFD profiles: FD arms from x81fix_<c>, role vectors from harness/generated_hyfdfix/<c> "
                        "(SMX_VALENTINE=549).")
    for c in a.corpora.split(","):
        if V.HYFDFIX and not V.fix_ready(f"x81_{c}"):
            lines += [f"## {c}: pending (x81fix_{c} not collected)", ""]
            continue
        ms = matchers(c)
        jobs = [(c, ds, sc, ms) for ds, sc in cases(c)]
        with Pool(a.workers) as pool:
            res = pool.map(score_case, jobs)
        rows = [r for rr, _ in res for r in rr]
        skips = [s for _, s in res if s]
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with (OUT / f"scores_{c}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, keys)
                w.writeheader()
                w.writerows(rows)
        for rep in sorted({r["corpus"] for r in rows}):
            rr = [r for r in rows if r["corpus"] == rep]
            ex = {(r["dataset"], r["scenario"]) for r in rr if r["scenario"] in excl.get(c, set())}
            lines += report(rep, rr, ex, np.random.default_rng(SEED))   # per corpus: independent of corpus order
        if skips:
            lines += [f"- {c} skipped: {len(skips)}: {skips[:5]}", ""]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
