"""EXPLORATORY X42 scorer (written 2026-10-04 before any iter_* output existed): SF's iteration trajectory.

Runs: servers/make_iter_configs.py (tuning/results/iter_<regime>/<m>__<arm>_t<t>, t updates, t in 1, 2, 3, 5, 10).
The converged reference is the registered base arm itself (layer1_<regime>/<m>__<arm>, er10_indep/<m>__melnik_shape),
rescored here with the same code. t = 0 is the initial matrix q.

Reported per base arm (melnik, fd on Valentine plain/MD5; melnik, melnik_shape on the independent corpus):
  - gain over q at each t and at convergence: J_tf, R_exp, F1_Hr, with a 95% interval for J_tf;
  - the difference to the converged run at each t (J_tf with interval);
  - the share of runs that reached the cap (iterations = t − 1 in sf_stats) rather than converging earlier;
  - median build and iteration time per t (sf_stats);
  - Valentine `melnik_t*` only: within-block inversions against q (Proposition 1 holds at every t >= 1), blocks from
    layer 1's constant__melnik.
Statistic: per scenario the mean over the included matchers (non-controls with every t and the converged run on every
scenario), the mean within family (Valentine: families with >= 5 scenarios; independent corpus: the database), the
mean over families; 95% percentile bootstrap over scenarios within family, 10,000 resamples, seed 20261006.
Gate (checked first; nothing is scored otherwise): levenshtein__melnik_gate = layer 1's levenshtein__melnik cell
for cell.

    ../harness/.venv-bdi/Scripts/python.exe explore_iterations.py --regime plain|md5|indep
Output: tuning/results/iter_<regime>_scores/{scores.csv, inversions.csv, report.md}.
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
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE / "tuning"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "reference"))
sys.path.insert(0, str(HERE / "layer3b"))

from score_layer1 import run_dir as _run_dir, score_seed  # noqa: E402
from score_seeds import CORPUS, cases_for, one  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
import valentine549  # noqa: E402  (rerun 549: used only under SMX_VALENTINE=549)

RES = HERE / "tuning" / "results"
T = (1, 2, 3, 5, 10)
BASES = {"plain": {"melnik": "layer1_plain", "fd": "layer1_plain"},
         "md5": {"melnik": "layer1_md5", "fd": "layer1_md5"},
         "indep": {"melnik": "layer1_indep", "melnik_shape": "er10_indep"}}
CONTROLS = {"constant", "random", "oracle", "oracle_noise025", "oracle_noise050", "oracle_erase050",
            "oracle_decoy100", "embedding_random", "rema_shuffled", "semstruct_lm_only"}
B, SEED = 10_000, 20261006
ITER = {}  # regime -> results root of the iter runs (tests may override with --iter-root)


def run_dir(p: Path):
    return _run_dir(p) if p.is_dir() else None


def iter_root(regime: str) -> Path:
    return ITER.get(regime, RES / f"iter_{regime}")


def gate(regime: str) -> float:
    x, y = run_dir(iter_root(regime) / "levenshtein__melnik_gate"), run_dir(RES / f"layer1_{regime}" / "levenshtein__melnik")
    if x is None or y is None:
        raise SystemExit("gate impossible: levenshtein__melnik_gate or layer 1 levenshtein__melnik missing")
    worst = 0.0
    for ds, sc, _ in cases_for(regime):
        fa, fb = one(x / ds / sc / "_outputs" / "MatchingStepLine2"), one(y / ds / sc / "_outputs" / "MatchingStepLine2")
        if fb is None:
            continue
        if fa is None:
            return float("inf")
        A, Bm = read_matrix(fa)[0], read_matrix(fb)[0]
        worst = max(worst, float(np.abs(A - Bm).max()) if A.shape == Bm.shape else float("inf"))
    return worst


def inversions_job(job):
    """Within-block inversions of melnik_t<t> against q (Valentine only)."""
    regime, seed, cases, root = job
    ITER[regime] = Path(root)
    from score_r7 import blocks, inversions
    q_root = HERE / "reference" / "seeds" / regime / seed
    const = run_dir(RES / f"layer1_{regime}" / "constant__melnik")
    out = []
    for t in T:
        r = run_dir(iter_root(regime) / f"{seed}__melnik_t{t}")
        if r is None or const is None:
            continue
        inv = pairs = 0
        for ds, sc, _ in cases:
            fq = q_root / ds / sc / "seed.csv"
            fs, fc = (one(x / ds / sc / "_outputs" / "MatchingStepLine2") for x in (r, const))
            if not fq.is_file() or fs is None or fc is None:
                continue
            q, S, C = read_matrix(fq)[0], read_matrix(fs)[0], read_matrix(fc)[0]
            if not (q.shape == S.shape == C.shape) or not (np.isfinite(q).all() and np.isfinite(S).all() and np.isfinite(C).all()):
                continue
            i, p = inversions(q, S, blocks(C))
            inv, pairs = inv + i, pairs + p
        out.append({"seed": seed, "t": t, "inversions": inv, "pairs": pairs})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=tuple(BASES), required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--iter-root", default="", help="testing only")
    ap.add_argument("--out", default="", help="testing only")
    a = ap.parse_args()
    if valentine549.V549 and a.regime in ("plain", "md5"):
        return main549(a.regime)
    if a.iter_root:
        ITER[a.regime] = Path(a.iter_root)
    root = iter_root(a.regime)
    worst = gate(a.regime)
    print(f"gate levenshtein__melnik_gate = layer 1 levenshtein__melnik: max |difference| {worst}")
    if worst != 0.0:
        raise SystemExit("gate FAILED: nothing is scored")
    cases = cases_for(a.regime)

    by_seed = defaultdict(list)
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "DONE").is_file() and "__" in p.name):
        seed, arm = d.name.split("__", 1)
        if arm.endswith("_gate"):
            continue
        by_seed[seed].append(str(d))
    for seed in list(by_seed):
        for arm, stage in BASES[a.regime].items():
            base = RES / stage / f"{seed}__{arm}"
            if run_dir(base) is not None:
                by_seed[seed].append(str(base))
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_seed, [(a.regime, s, ds, cases) for s, ds in by_seed.items()])
                for r in part]
    out = Path(a.out) if a.out else RES / f"iter_{a.regime}_scores"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    inv = None
    if a.regime != "indep":
        seeds = sorted(s for s in by_seed if s not in CONTROLS)
        with Pool(a.workers) as pool:
            inv = [r for part in pool.imap_unordered(inversions_job, [(a.regime, s, cases, str(root)) for s in seeds]) for r in part]
        with (out / "inversions.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["seed", "t", "inversions", "pairs"])
            w.writeheader()
            w.writerows(inv)
    report(a.regime, rows, cases, set(by_seed), worst, inv, out)
    return 0


def main549(regime: str) -> int:
    """SMX_VALENTINE=549: the report from the assembled rows (tuning/results/iter_<r>549_scores/scores.csv, written by
    reference/assemble_549.py: melnik_t* from iter_<r> + iter_<new>, the converged melnik = layer-1 melnik, identical rows
    on the 382) over the 549 scenarios; report.md goes next to them. fd and fd_t* are not rerun on 549, so the fd section
    finds no matcher; the inversion check needs the matrices of the 168 and is not computed here."""
    v = valentine549
    folder = f"iter_{v.vreg(regime)}_scores"
    arms = ["melnik"] + [f"melnik_t{t}" for t in T]
    st = {x: v.arm_status(f"iter_{regime}", x) for x in arms}
    rows = v.assembled(folder)
    if rows is None or any(s != "assembled" for s in st.values()):
        print(f"{folder}: pending (arms {st}); nothing written")
        return 0
    report(regime, rows, v.cases549(), {r["seed"] for r in rows}, None, None, RES / folder,
           gate_line="- 549 run (SMX_VALENTINE=549): rows assembled by reference/assemble_549.py from iter_<r> (381) and "
                     "iter_<new> (168), converged melnik = layer-1 melnik; the gate was checked when the 382 batch was "
                     "scored",
           inv_note="## Proposition 1 at every t: not computed on 549 (needs the melnik_t* matrices of the 168 new "
                    "scenarios; pending)", label=f"{regime}, 549 scenarios")
    return 0


def report(regime: str, rows: list[dict], cases, seed_set: set, worst, inv, out: Path,
           gate_line: str | None = None, inv_note: str | None = None, label: str | None = None) -> None:
    """The report of main() (unchanged; moved into a function 2026-11-01 so that the 549 run can build it from the
    assembled rows). inv: the inversion rows, or None (inv_note then explains why the section is missing)."""
    val = {(r["seed"], r["arm"], r["dataset"], r["scenario"]): r for r in rows}
    by_seed = seed_set

    class _A:
        pass
    a = _A()
    a.regime = regime

    fams = defaultdict(list)
    for d, c, f in cases:
        fams[f].append((d, c))
    if a.regime != "indep":
        fams = {f: v for f, v in fams.items() if len(v) >= 5}
    rng = np.random.default_rng(SEED)

    def J(r):
        return (float(r["sf_R_exp"]) + float(r["sf_F1_Hr"])) / 2

    def Jq(r):
        return (float(r["seed_R_exp"]) + float(r["seed_F1_Hr"])) / 2

    def stat(seeds, f):
        per = {fam: np.array([np.mean([f(s, d, c) for s in seeds]) for d, c in v]) for fam, v in fams.items()}
        est = float(np.mean([v.mean() for v in per.values()]))
        boots = np.mean([[rng.choice(v, size=v.size).mean() for _ in range(B)] for v in per.values()], axis=0)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return est, float(lo), float(hi)

    lines = [f"# X42 ({label or a.regime}): SF's iteration trajectory (EXPLORATORY)", "",
             gate_line or f"- gate levenshtein__melnik_gate = layer 1 levenshtein__melnik: max |difference| {worst} — passed", "",
             "Gain over the initial matrix q (t = number of updates; conv = the registered converged run). J_tf = "
             "(R_exp + F1_Hr)/2; per scenario the mean over included matchers, mean within family "
             + ("(database)" if a.regime == "indep" else "(families with >= 5 scenarios)")
             + ", mean over families; 95% bootstrap within family, 10,000 resamples, seed 20261006.", ""]
    for arm in BASES[a.regime]:
        arms = [f"{arm}_t{t}" for t in T] + [arm]
        seeds = sorted(s for s in by_seed if s not in CONTROLS
                       and all((s, x, d, c) in val for x in arms for d, c, _ in cases))
        if not seeds:
            lines += [f"## {arm}: no matcher with every arm on every scenario", ""]
            continue
        lines += [f"## {arm} ({len(seeds)} matchers)", "",
                  "| t | J_tf gain over q [95% CI] | R_exp | F1_Hr | minus converged J_tf [95% CI] | runs at the cap | "
                  "median build ms | median iterate ms |", "|---|---|---|---|---|---|---|---|"]
        for x, t in [(f"{arm}_t{t}", t) for t in T] + [(arm, "conv")]:
            g = stat(seeds, lambda s, d, c: J(val[(s, x, d, c)]) - Jq(val[(s, x, d, c)]))
            gr = stat(seeds, lambda s, d, c: float(val[(s, x, d, c)]["sf_R_exp"]) - float(val[(s, x, d, c)]["seed_R_exp"]))[0]
            gf = stat(seeds, lambda s, d, c: float(val[(s, x, d, c)]["sf_F1_Hr"]) - float(val[(s, x, d, c)]["seed_F1_Hr"]))[0]
            if t == "conv":
                dc, cap = "—", "—"
            else:
                e = stat(seeds, lambda s, d, c: J(val[(s, x, d, c)]) - J(val[(s, arm, d, c)]))
                dc = f"{e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}]"
                its = [val[(s, x, d, c)]["iterations"] for s in seeds for d, c, _ in cases]
                cap = f"{np.mean([str(i) == str(t - 1) for i in its]):.2f}"
            bm = np.median([float(val[(s, x, d, c)]["build_ms"] or "nan") for s in seeds for d, c, _ in cases])
            im = np.median([float(val[(s, x, d, c)]["iterate_ms"] or "nan") for s in seeds for d, c, _ in cases])
            lines.append(f"| {t} | {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}] | {gr:+.4f} | {gf:+.4f} | {dc} | {cap} | "
                         f"{bm:.0f} | {im:.1f} |")
        lines.append("")

    if inv is None and inv_note:
        lines += [inv_note, ""]
    if inv is not None:
        lines += ["## Proposition 1 at every t (melnik_t<t>, within-block inversions against q)", "",
                  "| t | inversions | ordered pairs |", "|---|---|---|"]
        for t in T:
            rs = [r for r in inv if r["t"] == t]
            lines.append(f"| {t} | {sum(r['inversions'] for r in rs)} | {sum(r['pairs'] for r in rs)} |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
