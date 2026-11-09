"""Scorer for the registered replication of the dependency-role term (draft: reference/AMENDMENT_ROLES_DRAFT.md).
Written on 2026-10-05 before any role-term metric on these corpora was computed. Frozen with the draft.

Question: X46 (exploratory, Valentine) found that an additive term on each column's dependency roles, added to the
datatype term, is level with SF's best dependency graphs and beats the others. Does this replicate, with the weights
fixed from Valentine, on data it was not developed on?

Corpora: t2d = harness/generated/t2d-usable (175 single-table WDC T2D scenarios with profiles; primary) and
ctl = harness/generated/controlled-imported-v3 (Lahman, MusicBrainz; plain and opaque regimes; secondary).
Per initial matcher and scenario, with q+ = max(q, 0) and D the NaN-free datatype matcher:
    A_d    = q+ + 5 D                                (Amendment 5's term)
    A_role = q+ + 5 D + γ R,  R = role similarity exactly as in explore_deproles.py (imported unchanged):
             1 − mean |r(i) − r(j)| over six features normalised per side (FD LHS count, FD RHS count, UCC count, key
             flag, IND out-degree, IND in-degree)
    γ fixed from X46's choice on all Valentine families: 0.2 on readable data (t2d, ctl-plain), 1.0 on recoded data
    (ctl-opaque). No weight is chosen on these corpora. The neighbour term is not used (it added nothing in X46).
SF outputs (same initial matrices, metrics and tie orders; score_layer1.score_seed):
    melnik, fd                  tuning/results/e2_<corpus>/<m>__<arm>   (E2's arms, and exploratory t2d run)
    ucc_top1, ucc1_fd_topk      tuning/results/e2x_<corpus>/<m>__<arm>  (built for this amendment, as X33's arms)

Hypotheses (decided on t2d; ctl reported per regime with the same rules, as secondary):
    HR1 (primary)  A_role − A_d > 0                   (lower bound > 0)
    HR2            A_role − SF(melnik) > 0
    HR3            A_role − SF(fd) > 0
    HR4a           A_role − SF(ucc_top1) > −0.01       (non-inferiority: lower bound > −0.01)
    HR4b           A_role − SF(ucc1_fd_topk) > −0.01
Statistic: J_tf = (R_exp + F1_Hr)/2; per scenario the mean over included matchers (non-controls with q, D and every SF
arm on every scenario of the cell); mean within family (t2d: one family; ctl: Lahman, MusicBrainz); mean over families;
95% percentile bootstrap over scenarios within family, 10,000 resamples, seed 20261012.

    ../harness/.venv-bdi/Scripts/python.exe reference/score_roles_reg.py --check-inputs   # no metrics
    ../harness/.venv-bdi/Scripts/python.exe reference/score_roles_reg.py --selftest       # synthetic only
    ../harness/.venv-bdi/Scripts/python.exe reference/score_roles_reg.py --corpus t2d|ctl # after the amendment
Output: tuning/results/rolesreg_<corpus>_scores/{scores.csv, report.md}.
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

import explore_deproles as xd  # noqa: E402
import score_layer1  # noqa: E402
from score_screen_metrics import all_metrics, permutations  # noqa: E402
from score_seeds import CORPUS, ground_truth_file, one  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

PREREG = HERE / "PREREGISTRATION_LAYERS.md"
CORPORA = {"t2d": STUDY.parent / "harness" / "generated" / "t2d-usable",
           "ctl": STUDY.parent / "harness" / "generated" / "controlled-imported-v3"}
CORPUS.update(CORPORA)                       # module level: pool workers re-import this module
score_layer1.CORPUS.update(CORPORA)
GAMMA = {"plain": 0.2, "opaque": 1.0}
ARMS = {"melnik": "e2", "fd": "e2", "ucc_top1": "e2x", "ucc1_fd_topk": "e2x"}
CONTROLS = {"constant", "random", "oracle", "oracle_noise025", "oracle_noise050", "oracle_erase050",
            "oracle_decoy100", "embedding_random", "rema_shuffled", "semstruct_lm_only", "datatype_nanfix"}
B, SEED = 10_000, 20261012
MARGIN = -0.01


def cases(corpus: str) -> list[tuple[str, str, str, str]]:
    """(dataset, scenario, family, regime)."""
    out = []
    for d in sorted(p for p in CORPORA[corpus].iterdir() if p.is_dir()):
        for s in sorted(p for p in d.iterdir() if p.is_dir()):
            fam = "T2D" if corpus == "t2d" else d.name.split("-")[0]
            reg = "plain" if corpus == "t2d" or s.name.endswith("_plain") else "opaque"
            out.append((d.name, s.name, fam, reg))
    return out


def role_matrix(corpus: str, ds: str, sc: str, rows, cols) -> np.ndarray:
    Fs, _ = xd.roles(list(rows), *xd.profiles(corpus, ds, sc, "source"))
    Ft, _ = xd.roles(list(cols), *xd.profiles(corpus, ds, sc, "target"))
    return 1.0 - np.abs(Fs[:, None, :] - Ft[None, :, :]).mean(axis=2)


def additive_rows(job) -> list[dict]:
    corpus, m, cs = job
    seeds = HERE / "seeds" / corpus
    out = []
    for ds, sc, fam, reg in cs:
        f_q, f_d = seeds / m / ds / sc / "seed.csv", seeds / "datatype_nanfix" / ds / sc / "seed.csv"
        g = ground_truth_file(CORPORA[corpus], ds, sc)
        if not (f_q.is_file() and f_d.is_file() and g is not None):
            continue
        q, rows, cols = read_matrix(f_q)
        D = np.nan_to_num(read_matrix(f_d)[0])
        gt = read_matrix(g)[0] > 0.5
        if not (q.shape == D.shape == gt.shape) or not np.isfinite(q).all():
            continue
        qp = np.clip(q, 0, None)
        perms = permutations(gt.shape, f"{ds}/{sc}")
        R = role_matrix(corpus, ds, sc, rows, cols)
        for name, M in (("A_d", qp + 5.0 * D), ("A_role", qp + 5.0 * D + GAMMA[reg] * R)):
            r = all_metrics(M, gt, perms)
            out.append({"seed": m, "arm": name, "dataset": ds, "scenario": sc, "family": fam, "regime": reg,
                        "J": (r["R_exp"] + r["F1_Hr"]) / 2})
    return out


def sf_rows(corpus: str, cs) -> list[dict]:
    dirs = defaultdict(list)
    for arm, stage in ARMS.items():
        root = STUDY / "tuning" / "results" / f"{stage}_{corpus}"
        if root.is_dir():
            for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith(f"__{arm}")):
                dirs[d.name.split("__", 1)[0]].append(str(d))
    plain_cases = [(d, s, f) for d, s, f, _ in cs]
    reg = {(d, s): r for d, s, _, r in cs}
    with Pool(8) as pool:
        rows = [r for part in pool.imap_unordered(score_layer1.score_seed, [(corpus, s, ds, plain_cases)
                                                                           for s, ds in dirs.items()]) for r in part]
    return [{"seed": r["seed"], "arm": r["arm"], "dataset": r["dataset"], "scenario": r["scenario"],
             "family": r["family"], "regime": reg[(r["dataset"], r["scenario"])],
             "J": (float(r["sf_R_exp"]) + float(r["sf_F1_Hr"])) / 2} for r in rows]


def decide(rows: list[dict], cell: list[tuple], rng) -> list[str]:
    val = {(r["seed"], r["arm"], r["dataset"], r["scenario"]): r["J"] for r in rows}
    arms_needed = ["A_d", "A_role", *ARMS]
    seeds = sorted(s for s in {r["seed"] for r in rows} if s not in CONTROLS
                   and all((s, a, d, c) in val for a in arms_needed for d, c, *_ in cell))
    fams = defaultdict(list)
    for d, c, f, *_ in cell:
        fams[f].append((d, c))
    out = [f"included initial matchers: {len(seeds)} ({', '.join(seeds)})", "",
           "| hypothesis | contrast | estimate [95% CI] | " + " | ".join(sorted(fams)) + " | matchers > 0 | verdict |",
           "|---|---|---|" + "---|" * len(fams) + "---|---|"]
    if not seeds:
        return out + ["| – | no matcher with every arm on every scenario | | | | not evaluable |"]
    for h, x, y, bound in (("HR1 (primary)", "A_role", "A_d", 0.0), ("HR2", "A_role", "melnik", 0.0),
                           ("HR3", "A_role", "fd", 0.0), ("HR4a", "A_role", "ucc_top1", MARGIN),
                           ("HR4b", "A_role", "ucc1_fd_topk", MARGIN)):
        per = {f: np.array([np.mean([val[(s, x, d, c)] - val[(s, y, d, c)] for s in seeds]) for d, c in v])
               for f, v in fams.items()}
        est = float(np.mean([v.mean() for v in per.values()]))
        boots = np.mean([[rng.choice(v, size=v.size).mean() for _ in range(B)] for v in per.values()], axis=0)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pos = sum(np.mean([np.mean([val[(s, x, d, c)] - val[(s, y, d, c)] for d, c in v]) for v in fams.values()]) > 1e-4
                  for s in seeds)
        verdict = "CONFIRMED" if lo > bound else "not confirmed"
        out.append(f"| {h} | {x} − {y}{' (margin −0.01)' if bound else ''} | {est:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                   + " | ".join(f"{per[f].mean():+.4f}" for f in sorted(fams)) + f" | {pos} of {len(seeds)} | {verdict} |")
    return out


def selftest() -> int:
    rng = np.random.default_rng(0)
    rows, cell = [], [(f"X", f"s{i}", "F1", "plain") for i in range(12)]
    for s in ("m1", "m2", "m3"):
        for d, c, f, reg in cell:
            base = 0.4 + 0.01 * rng.standard_normal()
            for arm, delta in (("A_d", 0.0), ("A_role", 0.05), ("melnik", 0.0), ("fd", -0.1), ("ucc_top1", 0.045),
                               ("ucc1_fd_topk", 0.08)):
                rows.append({"seed": s, "arm": arm, "dataset": d, "scenario": c, "family": f, "regime": reg,
                             "J": base + delta + 0.002 * rng.standard_normal()})
    lines = decide(rows, cell, np.random.default_rng(1))
    print("\n".join(lines).encode("ascii", "replace").decode())
    v = {l.split("|")[1].strip(): l.split("|")[-2].strip() for l in lines if l.startswith("| HR")}
    assert v["HR1 (primary)"] == "CONFIRMED" and v["HR3"] == "CONFIRMED" and v["HR4a"] == "CONFIRMED", v
    assert v["HR4b"] == "not confirmed", v               # synthetic: role 0.03 below the graph, beyond the margin
    print("selftest passed")
    return 0


def check_inputs(corpus: str) -> int:
    cs = cases(corpus)
    prof = sum(1 for d, s, *_ in cs if xd.profiles(corpus, d, s, "source")[0])
    seeds = HERE / "seeds" / corpus
    have = sorted(p.name for p in seeds.iterdir() if p.is_dir()) if seeds.is_dir() else []
    sf = {arm: sum(1 for _ in (STUDY / "tuning" / "results" / f"{st}_{corpus}").glob(f"*__{arm}"))
          if (STUDY / "tuning" / "results" / f"{st}_{corpus}").is_dir() else 0 for arm, st in ARMS.items()}
    print(f"{corpus}: {len(cs)} scenarios ({sum(r == 'plain' for *_, r in cs)} plain); with source FDs: {prof}; "
          f"seed directories: {len(have)}; SF run directories per arm (not read): {sf}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=tuple(CORPORA))
    ap.add_argument("--check-inputs", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.check_inputs:
        for c in ([a.corpus] if a.corpus else CORPORA):
            check_inputs(c)
        return 0
    if not a.corpus:
        raise SystemExit("--corpus is required")
    if "registered replication of the dependency-role term" not in PREREG.read_text(encoding="utf-8"):
        raise SystemExit("The amendment is not in PREREGISTRATION_LAYERS.md; the test does not run before it.")
    cs = cases(a.corpus)
    seeds_dir = HERE / "seeds" / a.corpus
    matchers = sorted(p.name for p in seeds_dir.iterdir() if p.is_dir() and p.name not in CONTROLS)
    with Pool(8) as pool:
        rows = [r for part in pool.imap_unordered(additive_rows, [(a.corpus, m, cs) for m in matchers]) for r in part]
    rows += sf_rows(a.corpus, cs)
    out = STUDY / "tuning" / "results" / f"rolesreg_{a.corpus}_scores"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rng = np.random.default_rng(SEED)
    lines = [f"# Registered replication of the dependency-role term: corpus {a.corpus}", "",
             "J_tf; per scenario the mean over included matchers, mean within family, mean over families; 95% bootstrap "
             "within family, 10,000 resamples, seed 20261012. γ fixed: 0.2 readable, 1.0 recoded. "
             + ("Decided on t2d." if a.corpus == "t2d" else "Secondary corpus: reported per regime with the same rules."), ""]
    for reg in sorted({r for *_, r in cs}):
        cell = [c for c in cs if c[3] == reg]
        lines += [f"## {a.corpus} {reg} ({len(cell)} scenarios)", "", *decide(rows, cell, rng), ""]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
