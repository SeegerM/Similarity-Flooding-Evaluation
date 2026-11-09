"""Score the R7 smoke test (layer3b/r7_configs.py) on the tie-free metrics.

Per seed x arm x screen scenario:
  - the tie-free metrics of SF's output (tuning/score_screen_metrics.all_metrics, the same seeded
    random orders as every other scorer) and of the seed q;
  - the same for the tie-breaker of Amendment 3 (explore_posthoc.tiebreak, imported, not edited):
    q's order kept, SF orders only q's ties. Plain only, exploratory;
  - within-block inversions against q: blocks are the classes of SF's output on the constant seed
    under `melnik` (check_prop1.py --classes constant). Proposition 1: zero without dependency edges;
    the inversion rate measures how much an FD set reorders q inside the blocks SF cannot otherwise
    separate;
  - SFSTATS (iterations, graph sizes, ms) from run.log, and the configuration's wall time.
The `fd_same` control is compared cell by cell with `fd` (max |difference| per scenario).

Writes layer3b/results/r7_screen_scores/{scores.csv, report.md}.

    python layer3b/r7_score.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "harness"))
sys.path.insert(0, str(STUDY / "tuning"))
sys.path.insert(0, str(STUDY))

from smx.adapters import read_matrix  # noqa: E402
from score_screen_metrics import all_metrics, permutations  # noqa: E402
from score_seeds import CORPUS, cases_for, ground_truth_file, one  # noqa: E402
from explore_posthoc import tiebreak  # noqa: E402

SUFFIX = "r7_screen"
ROOT = HERE / "results" / SUFFIX
OUT = HERE / "results" / f"{SUFFIX}_scores"
SCREEN = STUDY / "tuning" / "corpus" / "plain" / "screen"
SEEDS = STUDY / "reference" / "seeds" / "plain"
M = ["F1_Hr", "R_exp", "MRR_exp", "F1_M"]
MIN_FAMILY = 5
SFSTATS = re.compile(r"SFSTATS\s+(.*)")


def run_dir(d: Path) -> Path | None:
    runs = sorted(p for p in d.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def jtf(m: dict) -> float:
    return (m["R_exp"] + m["F1_Hr"]) / 2


def blocks(D: np.ndarray) -> np.ndarray:
    _, ci = np.unique(D.round(7), axis=0, return_inverse=True)
    _, cj = np.unique(D.T.round(7), axis=0, return_inverse=True)
    return (ci.ravel()[:, None] * (cj.max() + 1) + cj.ravel()[None, :]).ravel()


def inversions(q: np.ndarray, S: np.ndarray, block: np.ndarray) -> tuple[int, int]:
    inv = pairs = 0
    s, qq = S.ravel(), q.ravel()
    for b in np.unique(block):
        idx = np.where(block == b)[0]
        if len(idx) < 2:
            continue
        dq = qq[idx][:, None] - qq[idx][None, :]
        dS = s[idx][:, None] - s[idx][None, :]
        inv += int(((dq > 1e-9) & (dS < -1e-6)).sum())
        pairs += int((dq > 1e-9).sum())
    return inv, pairs


def sfstats(d: Path) -> list[str]:
    log = d / "run.log"
    if not log.is_file():
        return []
    return [m.group(1) for m in SFSTATS.finditer(log.read_text(encoding="utf-8", errors="replace"))]


def score_seed(job) -> list[dict]:
    seed, arms, cases = job
    const = run_dir(ROOT / "constant__melnik") if (ROOT / "constant__melnik" / "DONE").is_file() else None
    out = []
    for ds, sc, fam in cases:
        f_q = SEEDS / seed / ds / sc / "seed.csv"
        g = ground_truth_file(CORPUS["plain"], ds, sc)
        if not f_q.is_file() or g is None:
            continue
        gt = read_matrix(g)[0] > 0.5
        q = read_matrix(f_q)[0]
        if q.shape != gt.shape or not np.isfinite(q).all():
            continue
        perms = permutations(gt.shape, f"{ds}/{sc}")
        mq = all_metrics(q, gt, perms)
        block = None
        if const is not None:
            f_d = one(const / ds / sc / "_outputs" / "MatchingStepLine2")
            if f_d is not None:
                D = read_matrix(f_d)[0]
                block = blocks(D) if D.shape == q.shape else None
        outs = {}
        for arm, r in arms.items():
            f_s = one(r / ds / sc / "_outputs" / "MatchingStepLine2")
            if f_s is None:
                continue
            S = read_matrix(f_s)[0]
            if S.shape == gt.shape and np.isfinite(S).all():
                outs[arm] = S
        for arm, S in sorted(outs.items()):
            ms = all_metrics(S, gt, perms)
            T, _ = tiebreak(q, S)
            mt = all_metrics(T, gt, perms)
            row = {"seed": seed, "arm": arm, "dataset": ds, "scenario": sc, "family": fam,
                   "seed_J": jtf(mq), "sf_J": jtf(ms), "tb_J": jtf(mt),
                   **{f"seed_{m}": mq[m] for m in M}, **{f"sf_{m}": ms[m] for m in M},
                   **{f"tb_{m}": mt[m] for m in M}}
            if block is not None:
                row["inv"], row["inv_pairs"] = inversions(q, S, block)
            if arm == "fd_same" and "fd" in outs:
                row["same_maxdiff"] = float(np.abs(S - outs["fd"]).max())
            out.append(row)
    return out


def macro(rows, key):
    by = defaultdict(list)
    for r in rows:
        if r.get(key) not in (None, ""):
            by[r["family"]].append(float(r[key]))
    fams = [f for f, v in by.items() if len(v) >= MIN_FAMILY]
    return mean(mean(by[f]) for f in fams) if fams else float("nan")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    screen = {(d.name, s.name) for d in SCREEN.iterdir() if d.is_dir() for s in d.iterdir()}
    cases = [c for c in cases_for("plain") if (c[0], c[1]) in screen]
    by_seed = defaultdict(dict)
    wall, stats = {}, {}
    for d in sorted(p for p in ROOT.iterdir() if (p / "DONE").is_file()):
        seed, arm = d.name.split("__", 1)
        if seed == "constant":
            continue
        by_seed[seed][arm] = run_dir(d)
        wall[(seed, arm)] = float((d / "DONE").read_text().strip().rstrip("s"))
        stats[(seed, arm)] = sfstats(d)
    with Pool(6) as pool:
        rows = [r for part in pool.imap_unordered(score_seed, [(s, a, cases) for s, a in sorted(by_seed.items())])
                for r in part]
    rows.sort(key=lambda r: (r["seed"], r["arm"], r["dataset"], r["scenario"]))
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with (OUT / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # paired contrasts per (seed, scenario)
    idx = {(r["seed"], r["arm"], r["dataset"], r["scenario"]): r for r in rows}
    arms = sorted({r["arm"] for r in rows}, key=lambda a: (("reified" in a), a))
    seeds = sorted(by_seed)
    base = {a: ("fd_reified" if "reified" in a else "fd") for a in arms}
    lines = ["# R7 smoke test (exploratory, plain, 60 screen scenarios)", "",
             f"Seeds: {', '.join(seeds)}. Macro mean over families with >= {MIN_FAMILY} scenarios, then "
             "mean over seeds. J = (R_exp + F1_Hr) / 2. `vs base`: paired against `fd` (QUICK arms) or "
             "`fd_reified` (COMPLETE arms). `tb`: Amendment 3's tie-breaker built from this arm's output.", ""]
    hdr = ["arm", "J−seed", "R_exp−seed", "F1_Hr−seed", "MRR−seed", "J vs melnik", "J vs base", "R_exp vs base", "F1_Hr vs base",
           "tb J−seed", "tb J vs base", "inversions", "wall s", "seeds up/down vs base"]
    lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for a in arms:
        vals = defaultdict(list)
        for s in seeds:
            rs = [r for r in rows if r["seed"] == s and r["arm"] == a]
            if not rs:
                continue
            for k in ("J", "R_exp", "F1_Hr", "MRR_exp"):
                for r in rs:
                    r[f"d_{k}"] = r[f"sf_{k}"] - r[f"seed_{k}"]
                vals[k].append(macro(rs, f"d_{k}"))
            for r in rs:
                r["d_tbJ"] = r["tb_J"] - r["seed_J"]
                m = idx.get((s, "melnik", r["dataset"], r["scenario"]))
                r["v_mel"] = r["sf_J"] - m["sf_J"] if m else ""
                b = idx.get((s, base[a], r["dataset"], r["scenario"]))
                r["v_base"] = r["sf_J"] - b["sf_J"] if b else ""
                r["v_baseR"] = r["sf_R_exp"] - b["sf_R_exp"] if b else ""
                r["v_baseF"] = r["sf_F1_Hr"] - b["sf_F1_Hr"] if b else ""
                r["v_tbbase"] = r["tb_J"] - b["tb_J"] if b else ""
            vals["tbJ"].append(macro(rs, "d_tbJ"))
            vals["mel"].append(macro(rs, "v_mel"))
            vb = macro(rs, "v_base")
            vals["base"].append(vb)
            vals["tbbase"].append(macro(rs, "v_tbbase"))
            vals["baseR"].append(macro(rs, "v_baseR"))
            vals["baseF"].append(macro(rs, "v_baseF"))
            inv = sum(r.get("inv", 0) or 0 for r in rs)
            pairs = sum(r.get("inv_pairs", 0) or 0 for r in rs)
            vals["inv"].append(inv / pairs if pairs else float("nan"))
            vals["wall"].append(wall.get((s, a), float("nan")))
        f = lambda k, d=3: (f"{mean(vals[k]):+.{d}f}" if vals[k] and all(v == v for v in vals[k]) else "–")
        ud = "" if a == base[a] else f"{sum(1 for v in vals['base'] if v > 0.0005)}/{sum(1 for v in vals['base'] if v < -0.0005)}"
        lines.append(f"| {a} | {f('J')} | {f('R_exp')} | {f('F1_Hr')} | {f('MRR_exp')} | {f('mel')} | "
                     f"{f('base') if a != base[a] else '0'} | {f('baseR') if a != base[a] else '0'} | "
                     f"{f('baseF') if a != base[a] else '0'} | {f('tbJ')} | "
                     f"{f('tbbase') if a != base[a] else '0'} | "
                     f"{(mean(vals['inv']) if vals['inv'] else float('nan')):.4f} | "
                     f"{mean(vals['wall']):.0f} | {ud} |")
    same = [r["same_maxdiff"] for r in rows if "same_maxdiff" in r]
    lines += ["", f"**Control `fd_same` vs `fd`:** {len(same)} (seed, scenario) pairs, max |difference| "
              f"{max(same) if same else float('nan'):.3g}" + (" — identical." if same and max(same) == 0 else ""), ""]
    lines += ["Per seed, J vs base:", "", "| seed | " + " | ".join(a for a in arms if a != base[a]) + " |",
              "|---|" + "---|" * sum(1 for a in arms if a != base[a])]
    for s in seeds:
        cells = []
        for a in arms:
            if a == base[a]:
                continue
            rs = [r for r in rows if r["seed"] == s and r["arm"] == a]
            cells.append(f"{macro(rs, 'v_base'):+.4f}" if rs else "–")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    st = [(k, v) for k, v in stats.items() if v]
    if st:
        lines += ["", f"SFSTATS lines found for {len(st)} configurations (first: `{st[0][1][0][:160]}`)."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
