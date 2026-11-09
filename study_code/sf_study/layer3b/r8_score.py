"""Score the exploratory R8 grid (layer3b/r8_smoke.py; plain, 60 screen scenarios).

Per (initial matcher, arm in {melnik, fd}, scope) and scenario, with keep = the scope's kept cells of q
(r8_smoke.keep_rule, the mirror of the Java rule) and q_s = q with the pruned cells set to 0:
  - SF_s        SF's output under the scope (pruned cells are 0);
  - q_s         blocking alone, no propagation: the control for "the scope is just a threshold on q";
  - SF          SF without scope (the R7 smoke test's melnik / fd runs);
  - tb(q, SF_s) Amendment 3's tie-breaker from SF_s (plain; exploratory).
Contrasts (J_tf = (R_exp + F1_Hr)/2 and its parts; macro over families with >= 5 scenarios, mean over
the matchers): SF_s − q_s (what propagation adds on the candidate set), SF_s − SF (what the scope does
to SF), q_s − q (what blocking alone does), tb − q. Within-block inversions among the kept cells (blocks
of SF(constant) under melnik): zero for melnik by Proposition 1. Runtime: configuration wall time,
median pairwise-graph size and iteration time (SFSTATS), kept share (PAIRSCOPE).

Writes layer3b/results/r8_screen_scores/{scores.csv, report.md}.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from statistics import mean, median

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "harness"))
sys.path.insert(0, str(STUDY / "tuning"))
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(HERE))

from smx.adapters import read_matrix  # noqa: E402
from score_screen_metrics import all_metrics, permutations  # noqa: E402
from score_seeds import CORPUS, cases_for, ground_truth_file, one  # noqa: E402
from explore_posthoc import tiebreak  # noqa: E402
from r8_smoke import keep_rule  # noqa: E402

R7 = HERE / "results" / "r7_screen"
R8 = HERE / "results" / "r8_screen"
OUT = HERE / "results" / "r8_screen_scores"
SEEDS = STUDY / "reference" / "seeds" / "plain"
SCREEN = STUDY / "tuning" / "corpus" / "plain" / "screen"
K = ["R_exp", "F1_Hr", "MRR_exp", "F1_M"]
MIN_FAMILY = 5


def latest(d: Path) -> Path | None:
    runs = sorted(p for p in d.iterdir() if p.is_dir()) if d.is_dir() else []
    return runs[-1] if runs else None


def met(S, gt, perms) -> dict:
    m = all_metrics(S, gt, perms)
    return {"J": (m["R_exp"] + m["F1_Hr"]) / 2, **{k: m[k] for k in K}}


def blocks(D):
    _, ci = np.unique(D.round(7), axis=0, return_inverse=True)
    _, cj = np.unique(D.T.round(7), axis=0, return_inverse=True)
    return (ci.ravel()[:, None] * (cj.max() + 1) + cj.ravel()[None, :]).ravel()


def score(job) -> list[dict]:
    seed, arm, scopes, cases = job
    const = latest(R7 / "constant__melnik")
    base = latest(R7 / f"{seed}__{arm}")
    runs = {sc: latest(R8 / f"{seed}__{arm}__{sc}") for sc in scopes}
    out = []
    for ds, sn, fam in cases:
        g = ground_truth_file(CORPUS["plain"], ds, sn)
        q = read_matrix(SEEDS / seed / ds / sn / "seed.csv")[0]
        gt = read_matrix(g)[0] > 0.5
        perms = permutations(gt.shape, f"{ds}/{sn}")
        mq = met(q, gt, perms)
        f_base = one(base / ds / sn / "_outputs" / "MatchingStepLine2")
        msf = met(read_matrix(f_base)[0], gt, perms) if f_base else None
        f_d = one(const / ds / sn / "_outputs" / "MatchingStepLine2") if const else None
        block = blocks(read_matrix(f_d)[0]) if f_d else None
        for sc, r in runs.items():
            if r is None:
                continue
            f = one(r / ds / sn / "_outputs" / "MatchingStepLine2")
            if f is None:
                continue
            S = read_matrix(f)[0]
            kind = "QUANTILE" if sc.startswith("quantile") else "TOPK"
            keep = keep_rule(q, kind, float(sc[len(kind):]))
            qs = np.where(keep, q, 0.0)
            T, _ = tiebreak(q, S)
            row = {"seed": seed, "arm": arm, "scope": sc, "dataset": ds, "scenario": sn, "family": fam,
                   "kept_share": float(keep.mean()), "pruned_nonzero": int((S[~keep] != 0).any())}
            for name, m in (("q", mq), ("sf", msf), ("sfs", met(S, gt, perms)), ("qs", met(qs, gt, perms)),
                            ("tb", met(T, gt, perms))):
                if m:
                    row.update({f"{name}_{k}": v for k, v in m.items()})
            if block is not None:
                kk, ss, qq = keep.ravel(), S.ravel(), q.ravel()
                inv = pairs = 0
                for b in np.unique(block[kk]):
                    idx = np.where((block == b) & kk)[0]
                    dq = qq[idx][:, None] - qq[idx][None, :]
                    dS = ss[idx][:, None] - ss[idx][None, :]
                    inv += int(((dq > 1e-9) & (dS < -1e-6)).sum())
                    pairs += int((dq > 1e-9).sum())
                row["inv"], row["inv_pairs"] = inv, pairs
            out.append(row)
    return out


def macro(rows, key):
    by = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            by[r["family"]].append(r[key])
    fams = [f for f, v in by.items() if len(v) >= MIN_FAMILY]
    return mean(mean(by[f]) for f in fams) if fams else float("nan")


def stats(label: str) -> dict:
    log = R8 / label / "run.log" if (R8 / label).is_dir() else R7 / label.rsplit("__", 1)[0] / "run.log"
    d = R8 / label if (R8 / label).is_dir() else R7 / label.rsplit("__", 1)[0]
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.is_file() else []
    st = [l.split("\t") for l in lines if l.startswith("SFSTATS\t")]
    wall = float((d / "DONE").read_text().strip().rstrip("s")) if (d / "DONE").is_file() else float("nan")
    return {"wall": wall, "pcg_v": median(int(s[7]) for s in st) if st else float("nan"),
            "iter_ms": median(int(s[10]) for s in st) if st else float("nan")}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    screen = {(d.name, s.name) for d in SCREEN.iterdir() if d.is_dir() for s in d.iterdir()}
    cases = [c for c in cases_for("plain") if (c[0], c[1]) in screen]
    labels = sorted(p.name for p in R8.iterdir() if (p / "DONE").is_file())
    grid = defaultdict(set)
    for lab in labels:
        seed, arm, sc = lab.split("__")
        grid[(seed, arm)].add(sc)
    with Pool(6) as pool:
        rows = [r for part in pool.imap_unordered(score, [(s, a, sorted(v), cases) for (s, a), v in sorted(grid.items())])
                for r in part]
    rows.sort(key=lambda r: (r["seed"], r["arm"], r["scope"], r["dataset"], r["scenario"]))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        for k in ["J"] + K:
            r[f"d_prop_{k}"] = r[f"sfs_{k}"] - r[f"qs_{k}"]
            r[f"d_scope_{k}"] = r[f"sfs_{k}"] - r[f"sf_{k}"] if f"sf_{k}" in r else None
            r[f"d_block_{k}"] = r[f"qs_{k}"] - r[f"q_{k}"]
            r[f"d_sf_{k}"] = r[f"sf_{k}"] - r[f"q_{k}"] if f"sf_{k}" in r else None
        r["d_tb_J"] = r["tb_J"] - r["q_J"]
    seeds = sorted({r["seed"] for r in rows})
    order = ["quantile0.5", "quantile0.75", "quantile0.9", "topk1", "topk3"]
    lines = ["# R8 exploratory grid (plain, 60 screen scenarios)", "",
             f"Matchers: {', '.join(seeds)}. Macro over families with ≥ {MIN_FAMILY} scenarios, then the mean over matchers.",
             "`SF_s − q_s`: what propagation adds on the kept candidates; `SF_s − SF`: the scope against unrestricted SF; "
             "`q_s − q`: blocking alone; `SF − q`: unrestricted SF (reference row).", ""]
    for arm in ("melnik", "fd"):
        lines += [f"## Arm `{arm}`", "",
                  "| scope | kept | SF_s − q_s J | R_exp | F1_Hr | SF_s − SF J | R_exp | F1_Hr | q_s − q J | tb − q J | inversions (survivors) | wall s | pair-graph vertices | iterate ms |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        base_rows = [r for r in rows if r["arm"] == arm and r["scope"] == order[0]]
        sf_j = mean(macro([r for r in base_rows if r["seed"] == s], "d_sf_J") for s in seeds)
        sf_r = mean(macro([r for r in base_rows if r["seed"] == s], "d_sf_R_exp") for s in seeds)
        sf_f = mean(macro([r for r in base_rows if r["seed"] == s], "d_sf_F1_Hr") for s in seeds)
        st0 = [stats(f"{s}__{arm}__none") for s in seeds]
        lines.append(f"| none (SF − q) | 1.00 | {sf_j:+.4f} | {sf_r:+.4f} | {sf_f:+.4f} | 0 | 0 | 0 | 0 | | | "
                     f"{mean(x['wall'] for x in st0):.0f} | {mean(x['pcg_v'] for x in st0):.0f} | {mean(x['iter_ms'] for x in st0):.0f} |")
        for sc in order:
            rs = [r for r in rows if r["arm"] == arm and r["scope"] == sc]
            if not rs:
                continue
            f = lambda k: mean(macro([r for r in rs if r["seed"] == s], k) for s in seeds)
            inv = sum(r.get("inv", 0) for r in rs)
            pairs = sum(r.get("inv_pairs", 0) for r in rs)
            st = [stats(f"{s}__{arm}__{sc}") for s in seeds]
            lines.append(f"| {sc} | {mean(r['kept_share'] for r in rs):.2f} | {f('d_prop_J'):+.4f} | {f('d_prop_R_exp'):+.4f} | "
                         f"{f('d_prop_F1_Hr'):+.4f} | {f('d_scope_J'):+.4f} | {f('d_scope_R_exp'):+.4f} | {f('d_scope_F1_Hr'):+.4f} | "
                         f"{f('d_block_J'):+.4f} | {f('d_tb_J'):+.4f} | {inv / pairs if pairs else float('nan'):.4f} | "
                         f"{mean(x['wall'] for x in st):.0f} | {mean(x['pcg_v'] for x in st):.0f} | {mean(x['iter_ms'] for x in st):.0f} |")
        lines.append("")
    bad = sum(r["pruned_nonzero"] for r in rows)
    lines.append(f"Pruned cells with a non-zero SF output: {bad} scenario runs (must be 0).")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
