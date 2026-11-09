"""Registered scorer for the external replication (backlog E2; draft reference/AMENDMENT_E2_DRAFT.md). Written on
2026-10-04 before any initial-matcher or SF output on these corpora existed. Its SHA-256 goes into the amendment.

Corpora (harness/generated/):
  ctl  controlled-imported-v3: Lahman and MusicBrainz, 4 Valentine relationship types each, in two evidence regimes
       (scenario name ending in `_plain`, or an opaque recoding: `_monotone-numeric`, `_independent-recode`); 16
       single-table scenarios with FD/UCC/IND profiles.
  t2d  t2d-sm-wh: WDC T2D web tables against the knowledge base, single-table; used only if profiled before any SF
       run (stage 2).
Inputs (produced's runs; layout as layer 1):
  initial matrices  reference/seeds/<corpus>/<matcher>/<dataset>/<scenario>/seed.csv   (incl. `datatype_nanfix`)
  SF outputs        tuning/results/e2_<corpus>/<matcher>__<arm>/{merged|<run>}/…
  arms              melnik, fd, coef_prod_fd (standard), fd_top1 (layer 1's fd with R8's TOPK 1 scope), ucc (layer 3)
Metrics per (matcher, scenario): reference/score_layer1.py's score_seed (imported unchanged).

Predictions (frozen from Valentine; no tuning on these corpora). Statistic: per scenario the mean over the included
matchers (non-controls with every arm of a contrast on every scenario of the cell), mean within family (database),
mean over families; 95% percentile bootstrap over scenarios within family, 10,000 resamples, seed 20261005.
Families are Lahman, MusicBrainz (ctl) and T2D (t2d); the >= 5-scenario rule does not apply (4 per family and regime).
  P1  the better the initial matcher, the less SF adds: Spearman over matchers of (initial R_exp, SF(melnik) − initial
      on R_exp) < 0; decided on t2d (upper bound < 0), reported on ctl.
  P2  FD edges cost the ranking on readable data and help on recoded data: fd − melnik on R_exp < 0 on ctl-plain and
      t2d (upper bound < 0); fd − melnik on J_tf > 0 on ctl-opaque (lower bound > 0).
  P3  on single tables an additive datatype term does at least as well as SF: (q + 5 D) − SF(melnik) on J_tf > 0 on
      t2d and ctl-plain (lower bound > 0); D = the NaN-free datatype matcher.
  P4  restricting propagation to each row's and column's best candidate repairs FD edges' ranking loss:
      fd_top1 − fd on R_exp > 0 on ctl (both regimes) and t2d (lower bound > 0).
With 8 scenarios per regime on ctl the intervals there are wide; each prediction is also reported per family.

    ../harness/.venv-bdi/Scripts/python.exe reference/score_e2.py --corpus ctl|t2d
Output: tuning/results/e2_<corpus>_scores/{scores.csv,report.md}.
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
sys.path.insert(0, str(STUDY.parent / "harness"))
sys.path.insert(0, str(STUDY / "tuning"))
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(HERE))

import score_layer1  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from score_screen_metrics import all_metrics, permutations  # noqa: E402
from score_seeds import ground_truth_file, one  # noqa: E402

CORPORA = {"ctl": STUDY.parent / "harness" / "generated" / "controlled-imported-v3",
           "t2d": STUDY.parent / "harness" / "generated" / "t2d-sm-wh"}
CONTROLS = {"constant", "random", "oracle", "oracle_noise025", "oracle_noise050", "oracle_erase050",
            "oracle_decoy100", "embedding_random", "rema_shuffled", "semstruct_lm_only", "datatype_nanfix"}
ARMS = ("melnik", "fd", "coef_prod_fd", "fd_top1", "ucc")
score_layer1.CORPUS.update(CORPORA)   # at import, so the pool's worker processes see it too (score_seed reads CORPUS)
B, SEED = 10_000, 20261005


def cases(corpus: str) -> list[tuple[str, str, str]]:
    root = CORPORA[corpus]
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for s in sorted(p for p in d.iterdir() if p.is_dir()):
            family = d.name.split("-")[0] if corpus == "ctl" else "T2D"
            out.append((d.name, s.name, family))
    return out


def evidence(corpus: str, scenario: str) -> str:
    return "plain" if corpus == "t2d" or scenario.endswith("_plain") else "opaque"


def additive_rows(job) -> list[dict]:
    corpus, seed, cs = job
    seeds = HERE / "seeds" / corpus
    sf_root = STUDY / "tuning" / "results" / f"e2_{corpus}" / f"{seed}__melnik"
    sf_dir = score_layer1.run_dir(sf_root) if sf_root.is_dir() else None
    out = []
    for ds, sc, fam in cs:
        f_q, f_d = seeds / seed / ds / sc / "seed.csv", seeds / "datatype_nanfix" / ds / sc / "seed.csv"
        f_s = one(sf_dir / ds / sc / "_outputs" / "MatchingStepLine2") if sf_dir else None
        g = ground_truth_file(CORPORA[corpus], ds, sc)
        if not (f_q.is_file() and f_d.is_file() and f_s and g):
            continue
        gt = read_matrix(g)[0] > 0.5
        q, D = np.clip(read_matrix(f_q)[0], 0, None), np.nan_to_num(read_matrix(f_d)[0])
        if q.shape != gt.shape or D.shape != gt.shape:
            continue
        m = all_metrics(q + 5.0 * D, gt, permutations(gt.shape, f"{ds}/{sc}"))
        out.append({"seed": seed, "arm": "additive_d", "dataset": ds, "scenario": sc, "family": fam,
                    "sf_R_exp": m["R_exp"], "sf_F1_Hr": m["F1_Hr"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=tuple(CORPORA), required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    cs = cases(a.corpus)
    root = STUDY / "tuning" / "results" / f"e2_{a.corpus}"
    by_seed = defaultdict(list)
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "DONE").is_file() and "__" in p.name):
        by_seed[d.name.split("__", 1)[0]].append(str(d))
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_layer1.score_seed, [(a.corpus, s, ds, cs) for s, ds in by_seed.items()])
                for r in part]
        rows += [r for part in pool.imap_unordered(additive_rows, [(a.corpus, s, cs) for s in by_seed]) for r in part]
    out = STUDY / "tuning" / "results" / f"e2_{a.corpus}_scores"
    out.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    val = {(r["seed"], r["arm"], r["dataset"], r["scenario"]): r for r in rows}
    base = {(r["seed"], r["dataset"], r["scenario"]): r for r in rows if r["arm"] == "melnik"}
    rng = np.random.default_rng(SEED)

    def metric(s, arm, d, c, m):
        r = val[(s, arm, d, c)]
        if m == "J":
            return (float(r["sf_R_exp"]) + float(r["sf_F1_Hr"])) / 2
        return float(r[f"sf_{m}"])

    def contrast(x, y, m, ev, sign):
        cell = [(d, c, f) for d, c, f in cs if evidence(a.corpus, c) == ev]
        seeds = sorted(s for s in {k[0] for k in val} if s not in CONTROLS
                       and all((s, x, d, c) in val and (s, y, d, c) in val for d, c, _ in cell))
        if not seeds or not cell:
            return f"| {x} − {y} ({ev}, {m}) | 0 | | | | not evaluable |"
        fams = defaultdict(list)
        for d, c, f in cell:
            fams[f].append(np.mean([metric(s, x, d, c, m) - metric(s, y, d, c, m) for s in seeds]))
        fams = {f: np.array(v) for f, v in fams.items()}
        est = float(np.mean([v.mean() for v in fams.values()]))
        boots = np.mean([[rng.choice(v, size=v.size).mean() for _ in range(B)] for v in fams.values()], axis=0)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ok = (lo > 0) if sign > 0 else (hi < 0)
        per = ", ".join(f"{f} {v.mean():+.4f}" for f, v in sorted(fams.items()))
        return (f"| {x} − {y} ({ev}, {m}) | {len(seeds)} | {est:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {per} | "
                f"{'CONFIRMED' if ok else 'not confirmed'} |")

    def rho_line(ev):
        cell = [(d, c, f) for d, c, f in cs if evidence(a.corpus, c) == ev]
        seeds = sorted(s for s in {k[0] for k in base} if s not in CONTROLS and all((s, d, c) in base for d, c, _ in cell))
        if len(seeds) < 4:
            return f"| P1 ρ ({ev}) | {len(seeds)} | | | | not evaluable |"
        fams = defaultdict(list)
        for i, (d, c, f) in enumerate(cell):
            fams[f].append(i)

        def rho(idx_by_fam):
            x = np.array([np.mean([np.mean([float(base[(s, cell[i][0], cell[i][1])]["seed_R_exp"]) for i in ix])
                                   for ix in idx_by_fam]) for s in seeds])
            y = np.array([np.mean([np.mean([float(base[(s, cell[i][0], cell[i][1])]["sf_R_exp"])
                                            - float(base[(s, cell[i][0], cell[i][1])]["seed_R_exp"]) for i in ix])
                                   for ix in idx_by_fam]) for s in seeds])
            rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
            return float(np.corrcoef(rx, ry)[0, 1])
        est = rho(list(fams.values()))
        boots = [rho([rng.choice(ix, size=len(ix)) for ix in fams.values()]) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return (f"| P1 ρ(initial R_exp, ΔR_exp melnik) ({ev}) | {len(seeds)} | {est:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                f"| {'CONFIRMED' if hi < 0 else 'not confirmed'} (decided on t2d) |")

    evs = ("plain",) if a.corpus == "t2d" else ("plain", "opaque")
    lines = [f"# External replication (E2), corpus {a.corpus}: registered analysis", "",
             "| prediction | matchers | estimate | 95% CI | per family | verdict |", "|---|---|---|---|---|---|"]
    for ev in evs:
        lines.append(rho_line(ev))
    for ev in evs:
        if ev == "plain":
            lines.append("| **P2** " + contrast("fd", "melnik", "R_exp", ev, -1)[2:])
        else:
            lines.append("| **P2** " + contrast("fd", "melnik", "J", ev, +1)[2:])
    if "plain" in evs:
        lines.append("| **P3** " + contrast("additive_d", "melnik", "J", "plain", +1)[2:])
    for ev in evs:
        lines.append("| **P4** " + contrast("fd_top1", "fd", "R_exp", ev, +1)[2:])
    lines += ["", "Reported, not tested: coef_prod_fd − melnik and ucc − melnik (J_tf), per regime."]
    for ev in evs:
        lines.append(contrast("coef_prod_fd", "melnik", "J", ev, +1).replace("CONFIRMED", "(descriptive)")
                     .replace("not confirmed", "(descriptive)"))
        lines.append(contrast("ucc", "melnik", "J", ev, +1).replace("CONFIRMED", "(descriptive)")
                     .replace("not confirmed", "(descriptive)"))
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
