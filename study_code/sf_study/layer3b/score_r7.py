"""Registered scorer for R7 (Amendment 2 of reference/PREREGISTRATION_LAYERS.md; layer 3b).

Written 2026-10-03, before any registered R7 result existed. It implements the amendment's §3, §4, §7
and §8, with the metrics, contrasts, aggregation and bootstrap of PREREGISTRATION_LAYERS.md §3.

Inputs (per regime):
  tuning/results/r7_<regime>/<seed>__<arm>/        the R7 arms (dispatch.py collect; merged/)
  tuning/results/layer1_<regime>/<seed>__fd         base of the QUICK arms (and __melnik, constant__melnik)
  tuning/results/layer3_<regime>/<seed>__fd_reified base of the reified arms
  reference/seeds/<regime>/<seed>/...               the initial matrices q

Procedure:
 0. **Pipeline check.** r7 levenshtein__fd_same must equal layer 1's levenshtein__fd cell for cell on every
    scenario of the analysed case set. Otherwise nothing is scored (§8).
 1. Per (initial matcher, arm, scenario): the tie-free metrics (score_screen_metrics.all_metrics; the
    same seeded orders as every scorer) of SF's output; on **plain only**, also of Amendment 3's
    tie-breaker built from it (explore_posthoc.tiebreak, imported). Within-block inversions against q,
    with the blocks of SF(constant) under `melnik` (check_prop1.py --classes constant).
 2. Contrasts, paired per scenario: arm effect = arm − base (`fd` for the QUICK arms, `fd_reified` for
    the reified ones). The random null `thin` is the mean of the three draws per scenario (§7); its
    contrast cover − thin is added there.
 3. Aggregation: per matcher, the mean over each family's scenarios, then over the families with at
    least 5 scenarios in the case set (ChEMBL, Magellan, OpenData, TPC-DI on the full set); then the
    mean over the **included** matchers: non-controls (not constant, random, embedding_random, oracle*)
    with an output for every scenario of the case set in every arm of the contrast. Controls and matchers
    with partial coverage are reported per matcher only.
 4. Uncertainty: 95% percentile bootstrap over scenarios, stratified by family, 10,000 resamples,
    seed 20261002. One set of resamples per case set, shared by all contrasts.
 5. Hypotheses (amendment §4):
    - H-R7a: mean over matchers of |macro arm effect of fd_closure on J_tf| < 0.005: confirmed if the
      bootstrap upper bound is below 0.005.
    - H-R7b, per encoding: mean |macro effect on R_exp| of the cover minus that of the closure > 0:
      confirmed if the lower bound is above 0.
    - H-R7c: inversion share (inversions / ordered within-block pairs of q, per scenario) under fd_cover
      minus under fd, macro, mean over matchers > 0: confirmed if the lower bound is above 0. Family form:
      the per-family increase is largest on TPC-DI (point estimate decides; the share of bootstrap
      resamples in which TPC-DI is largest is reported). The reified pair is reported as secondary.
    - No direction for the cover's effect on J_tf; cover − thin likewise: estimates with intervals.
 6. Case sets: **primary = the 322 scenarios outside the screen** (§7); the full 382 and the 60 screen
    scenarios are reported as well.
 7. RQ5 (descriptive): medians of sf_stats.tsv per arm (iterations, pairwise-graph size, ms).

Writes tuning/results/r7_<regime>_scores/: scores.csv (per matcher, arm, scenario), per_matcher.csv,
contrasts.csv, hypotheses.csv, rq5.csv, report.md.

    python layer3b/score_r7.py --regime plain [--workers 10]
Test on the smoke test (layer3b/make_r7_scorer_test.py builds the roots): --r7-root/--l1-root/--l3-root
and --cases screen.
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
sys.path.insert(0, str(STUDY.parent / "harness"))
sys.path.insert(0, str(STUDY / "tuning"))
sys.path.insert(0, str(STUDY))

from smx.adapters import read_matrix  # noqa: E402
from score_screen_metrics import all_metrics, permutations  # noqa: E402
from score_seeds import CONTROLS, CORPUS, PROTOCOL, cases_for, ground_truth_file, one  # noqa: E402
from explore_posthoc import tiebreak  # noqa: E402

DRAWS = ("thin1", "thin2", "thin3")
QUICK = ["fd_closure", "fd_cover"] + [f"fd_{d}" for d in DRAWS]
REIFIED = ["fd_reified_closure", "fd_reified_cover"] + [f"fd_reified_{d}" for d in DRAWS]
BASES = ("fd", "fd_reified", "melnik")
M = ["J", "R_exp", "F1_Hr", "MRR_exp", "F1_M", "R_pess", "F1_H"]
MIN_FAMILY = 5
N_BOOT, BOOT_SEED = 10_000, 20261002
STATS = ["iterations", "converged", "schema_v", "schema_e", "pcg_v", "pcg_e", "build_ms", "iterate_ms"]


# ------------------------------------------------------------------------------------- inputs

def run_dir(label_dir: Path) -> Path | None:
    if not label_dir.is_dir():
        return None
    if (label_dir / "merged").is_dir():
        return label_dir / "merged"
    runs = sorted(p for p in label_dir.iterdir() if p.is_dir() and p.name != "_shards")
    return runs[-1] if runs else None


def is_control(seed: str) -> bool:
    return seed in CONTROLS or seed.startswith("oracle")


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


def metrics(S, gt, perms) -> dict:
    m = all_metrics(S, gt, perms)
    m["J"] = (m["R_exp"] + m["F1_Hr"]) / 2
    return m


def score_seed(job) -> list[dict]:
    regime, seed, dirs, cases, const_dir, with_tb = job
    seeds = STUDY / "reference" / "seeds" / regime / seed
    runs = {arm: run_dir(Path(d)) for arm, d in dirs.items()}
    runs = {a: r for a, r in runs.items() if r is not None}
    const = run_dir(Path(const_dir)) if const_dir else None
    out = []
    for ds, sc, fam, screen in cases:
        f_q = seeds / ds / sc / "seed.csv"
        g = ground_truth_file(CORPUS[regime], ds, sc)
        if not f_q.is_file() or g is None:
            continue
        gt = read_matrix(g)[0] > 0.5
        q = read_matrix(f_q)[0]
        if q.shape != gt.shape or not np.isfinite(q).all():
            continue
        perms = permutations(gt.shape, f"{ds}/{sc}")
        block = None
        if const is not None:
            f_d = one(const / ds / sc / "_outputs" / "MatchingStepLine2")
            if f_d is not None:
                D = read_matrix(f_d)[0]
                block = blocks(D) if D.shape == q.shape else None
        mq = metrics(q, gt, perms)
        for arm, r in sorted(runs.items()):
            f_s = one(r / ds / sc / "_outputs" / "MatchingStepLine2")
            if f_s is None:
                continue
            S = read_matrix(f_s)[0]
            if S.shape != gt.shape or not np.isfinite(S).all():
                continue
            row = {"seed": seed, "arm": arm, "dataset": ds, "scenario": sc, "family": fam, "screen": int(screen),
                   **{f"q_{k}": mq[k] for k in M}, **{f"sf_{k}": v for k, v in metrics(S, gt, perms).items() if k in M}}
            if with_tb:
                T, _ = tiebreak(q, S)
                row.update({f"tb_{k}": v for k, v in metrics(T, gt, perms).items() if k in M})
            if block is not None:
                row["inv"], row["inv_pairs"] = inversions(q, S, block)
            out.append(row)
    return out


# ----------------------------------------------------------------------------------- analysis

class Panel:
    """Matcher x scenario matrices of one case set, with shared stratified bootstrap resamples."""

    def __init__(self, rows: list[dict], case_keys: list[tuple[str, str]], fam_of: dict, seed=BOOT_SEED):
        self.keys = case_keys
        self.col = {k: i for i, k in enumerate(case_keys)}
        fams = defaultdict(list)
        for i, k in enumerate(case_keys):
            fams[fam_of[k]].append(i)
        self.fams = {f: np.array(ix) for f, ix in sorted(fams.items()) if len(ix) >= MIN_FAMILY}
        self.val = {}  # (seed, arm, field) -> vector over scenarios (NaN where missing)
        for r in rows:
            j = self.col.get((r["dataset"], r["scenario"]))
            if j is None:
                continue
            for field, v in r.items():
                if field.startswith(("sf_", "tb_", "q_")) or field in ("inv", "inv_pairs"):
                    key = (r["seed"], r["arm"], field)
                    if key not in self.val:
                        self.val[key] = np.full(len(case_keys), np.nan)
                    self.val[key][j] = float(v)
        rng = np.random.default_rng(seed)
        self.boot = [{f: rng.choice(ix, size=len(ix), replace=True) for f, ix in self.fams.items()}
                     for _ in range(N_BOOT)]

    def vec(self, seed: str, arm: str, field: str) -> np.ndarray | None:
        if arm == "fd_thin" or arm == "fd_reified_thin":
            parts = [self.val.get((seed, f"{arm}{d[-1]}", field)) for d in DRAWS]
            return None if any(p is None for p in parts) else np.mean(parts, axis=0)
        if field == "inv_share":
            i, p = self.val.get((seed, arm, "inv")), self.val.get((seed, arm, "inv_pairs"))
            if i is None or p is None:
                return None
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(p > 0, i / p, np.nan)
        return self.val.get((seed, arm, field))

    def matrix(self, seeds, arm, base, field) -> tuple[np.ndarray, list[str]]:
        """Effect matrix (included matchers x scenarios): arm − base (base None: the arm itself)."""
        rows, kept = [], []
        for s in seeds:
            a = self.vec(s, arm, field)
            b = self.vec(s, base, field) if base else np.zeros_like(a) if a is not None else None
            if a is None or b is None:
                continue
            e = a - b
            if field == "inv_share":  # undefined where a scenario has no within-block pairs
                rows.append(e)
                kept.append(s)
            elif np.isfinite(e).all():
                rows.append(e)
                kept.append(s)
        return (np.array(rows) if rows else np.empty((0, len(self.keys)))), kept

    def macro(self, E: np.ndarray, idx: dict | None = None) -> np.ndarray:
        """Per-matcher macro mean over families (idx: resampled indices per family)."""
        idx = idx or self.fams
        return np.nanmean(np.stack([np.nanmean(E[:, idx[f]], axis=1) for f in self.fams], axis=1), axis=1)

    def stat(self, E, fn) -> tuple[float, float, float]:
        est = fn(self.macro(E))
        bs = np.array([fn(self.macro(E, b)) for b in self.boot])
        return float(est), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))

    def family_means(self, E) -> dict[str, float]:
        return {f: float(np.nanmean(np.nanmean(E[:, ix], axis=1))) for f, ix in self.fams.items()}


def mean_(x):
    return np.nanmean(x)


def mean_abs(x):
    return np.nanmean(np.abs(x))


# ------------------------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5"), required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--r7-root", default="")
    ap.add_argument("--l1-root", default="")
    ap.add_argument("--l3-root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--cases", choices=("protocol", "screen"), default="protocol",
                    help="screen: only the 60 screen scenarios (for the test on the smoke test)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    res = STUDY / "tuning" / "results"
    r7 = Path(a.r7_root or res / f"r7_{a.regime}")
    l1 = Path(a.l1_root or res / f"layer1_{a.regime}")
    l3 = Path(a.l3_root or res / f"layer3_{a.regime}")
    out = Path(a.out or res / f"r7_{a.regime}_scores")
    out.mkdir(parents=True, exist_ok=True)
    with_tb = a.regime == "plain"  # Amendment 3 blinding: no tie-breaker on MD5

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]
    screen_keys = set(json.loads((STUDY / "tuning" / "splits.json").read_text(encoding="utf-8"))["screen"])
    screen = {(c["dataset"], c["scenario"]) for c in protocol if c["key"] in screen_keys}
    cases = [(d, s, f, (d, s) in screen) for d, s, f in cases_for(a.regime)]
    if a.cases == "screen":
        cases = [c for c in cases if c[3]]
    fam_of = {(d, s): f for d, s, f, _ in cases}

    # 0. pipeline check
    a_dir, b_dir = run_dir(r7 / "levenshtein__fd_same"), run_dir(l1 / "levenshtein__fd")
    if a_dir is None or b_dir is None:
        raise SystemExit("pipeline check impossible: levenshtein__fd_same or layer 1's levenshtein__fd missing")
    worst, n = 0.0, 0
    for d, s, _, _ in cases:
        fa = one(a_dir / d / s / "_outputs" / "MatchingStepLine2")
        fb = one(b_dir / d / s / "_outputs" / "MatchingStepLine2")
        if fa is None or fb is None:
            raise SystemExit(f"pipeline check: output missing for {d}/{s}")
        A, B = read_matrix(fa)[0], read_matrix(fb)[0]
        worst = max(worst, float(np.abs(A - B).max()) if A.shape == B.shape else float("inf"))
        n += 1
    check = f"pipeline check: fd_same vs layer 1 fd on {n} scenarios, max |difference| {worst}"
    print(check)
    if worst != 0.0:
        raise SystemExit("pipeline check FAILED: nothing is scored (Amendment 2, §8)")

    # 1. per-scenario scores
    seeds = sorted({p.name.split("__", 1)[0] for p in r7.iterdir() if "__" in p.name and (p / "DONE").is_file()})
    jobs = []
    for seed in seeds:
        dirs = {arm: str(r7 / f"{seed}__{arm}") for arm in QUICK + REIFIED if (r7 / f"{seed}__{arm}" / "DONE").is_file()}
        for arm, root in (("fd", l1), ("melnik", l1), ("fd_reified", l3)):
            if (root / f"{seed}__{arm}" / "DONE").is_file():
                dirs[arm] = str(root / f"{seed}__{arm}")
        jobs.append((a.regime, seed, dirs, cases, str(l1 / "constant__melnik"), with_tb))
    rows = []
    with Pool(a.workers) as pool:
        for part in pool.imap_unordered(score_seed, jobs):
            rows.extend(part)
    rows.sort(key=lambda r: (r["seed"], r["arm"], r["dataset"], r["scenario"]))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # 2.-5. contrasts and hypotheses per case set
    sets = {"primary (non-screen)": [k for k in fam_of if k not in screen],
            "all": list(fam_of), "screen": [k for k in fam_of if k in screen]}
    contrasts, hyps, per_matcher = [], [], []
    report = [f"# R7 (Amendment 2), regime {a.regime}: registered analysis", "",
              f"{check} — passed.", "",
              "J = (R_exp + F1_Hr) / 2. Effects are paired per scenario, macro over families with ≥ "
              f"{MIN_FAMILY} scenarios, then the mean over included matchers (non-controls with every scenario "
              "in every arm of the contrast). 95% percentile bootstrap over scenarios, stratified by family, "
              f"{N_BOOT:,} resamples, seed {BOOT_SEED}. `thin` = mean of the three random draws."
              + ("" if with_tb else " No tie-breaker on MD5 (Amendment 3 blinding)."), ""]
    for name, keys in sets.items():
        if not keys:
            continue
        P = Panel(rows, keys, fam_of)
        if not P.fams:
            continue
        all_seeds = sorted({r["seed"] for r in rows})
        included = [s for s in all_seeds if not is_control(s)]
        report += [f"## Case set: {name} ({len(keys)} scenarios; families {', '.join(f'{f} {len(i)}' for f, i in P.fams.items())})", ""]

        def contrast(arm, base, field, label=None, fn=mean_):
            E, kept = P.matrix(included, arm, base, field)
            if not kept:
                return None
            est, lo, hi = P.stat(E, fn)
            rec = {"set": name, "contrast": label or f"{arm} − {base or '0'}", "field": field,
                   "statistic": fn.__name__.strip("_"), "matchers": len(kept), "estimate": est, "lo": lo, "hi": hi}
            contrasts.append(rec)
            return rec, E, kept

        # effects of every arm against its base, and cover − thin
        table = ["| Contrast | matchers | " + " | ".join(M[:5]) + " |", "|---|---|" + "---|" * 5]
        pairs = [(arm, "fd") for arm in ("fd_closure", "fd_cover", "fd_thin")] + \
                [(arm, "fd_reified") for arm in ("fd_reified_closure", "fd_reified_cover", "fd_reified_thin")] + \
                [("fd_cover", "fd_thin"), ("fd_reified_cover", "fd_reified_thin"), ("fd", "melnik"), ("fd_reified", "melnik")]
        for arm, base in pairs:
            cells, nm = [], 0
            for k in M[:5]:
                r = contrast(arm, base, f"sf_{k}")
                if r is None:
                    cells.append("–")
                    continue
                rec, _, kept = r
                nm = len(kept)
                cells.append(f"{rec['estimate']:+.4f} [{rec['lo']:+.4f}, {rec['hi']:+.4f}]")
            table.append(f"| {arm} − {base} | {nm} | " + " | ".join(cells) + " |")
        report += ["Arm effects on SF's output (estimate [95% CI]):", ""] + table + [""]
        if with_tb:
            tb = ["| Contrast (tie-breaker built from each arm) | J |", "|---|---|"]
            for arm, base in pairs[:8]:
                r = contrast(arm, base, "tb_J", label=f"tb: {arm} − {base}")
                if r:
                    tb.append(f"| {arm} − {base} | {r[0]['estimate']:+.4f} [{r[0]['lo']:+.4f}, {r[0]['hi']:+.4f}] |")
            report += ["Tie-breaker (Amendment 3's construction, plain only):", ""] + tb + [""]

        # hypotheses
        h = []
        r = contrast("fd_closure", "fd", "sf_J", "H-R7a: mean |fd_closure − fd| on J", fn=mean_abs)
        if r:
            rec = r[0]
            h.append(("H-R7a", "mean |effect of fd_closure on J_tf| < 0.005 (upper bound)", rec,
                      "confirmed" if rec["hi"] < 0.005 else "not confirmed"))
        for enc, cover, closure, base in (("QUICK", "fd_cover", "fd_closure", "fd"),
                                          ("reified", "fd_reified_cover", "fd_reified_closure", "fd_reified")):
            Ec, kc = P.matrix(included, cover, base, "sf_R_exp")
            El, kl = P.matrix(included, closure, base, "sf_R_exp")
            both = sorted(set(kc) & set(kl))
            if both:
                Ec = Ec[[kc.index(s) for s in both]]
                El = El[[kl.index(s) for s in both]]
                est = mean_abs(P.macro(Ec)) - mean_abs(P.macro(El))
                bs = np.array([mean_abs(P.macro(Ec, b)) - mean_abs(P.macro(El, b)) for b in P.boot])
                rec = {"set": name, "contrast": f"H-R7b {enc}: mean|R_exp effect| {cover} − {closure}",
                       "field": "sf_R_exp", "statistic": "diff of mean abs", "matchers": len(both),
                       "estimate": float(est), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
                contrasts.append(rec)
                h.append((f"H-R7b ({enc})", "mean |R_exp effect| cover > closure (lower bound > 0)", rec,
                          "confirmed" if rec["lo"] > 0 else "not confirmed"))
        for enc, cover, base, primary in (("QUICK", "fd_cover", "fd", True), ("reified", "fd_reified_cover", "fd_reified", False)):
            r = contrast(cover, base, "inv_share", f"H-R7c {enc}: inversion share {cover} − {base}")
            if r:
                rec, E, kept = r
                fam = P.family_means(E)
                top = max(fam, key=fam.get)
                wins = 0
                for b in P.boot:
                    fm = {f: float(np.nanmean(np.nanmean(E[:, b[f]], axis=1))) for f in P.fams}
                    wins += max(fm, key=fm.get) == "TPC-DI"
                label = "H-R7c" if primary else "H-R7c (reified, secondary)"
                h.append((label, "inversion share cover > fd (lower bound > 0)", rec,
                          "confirmed" if rec["lo"] > 0 else "not confirmed"))
                h.append((label + ", family form", "increase largest on TPC-DI (point estimate)",
                          {"estimate": fam.get("TPC-DI", float("nan")), "lo": float("nan"), "hi": float("nan"),
                           "matchers": len(kept)},
                          f"{'confirmed' if top == 'TPC-DI' else 'not confirmed'} (largest: {top}; "
                          f"TPC-DI largest in {wins / len(P.boot):.1%} of resamples; "
                          + ", ".join(f"{f} {v:+.4f}" for f, v in fam.items()) + ")"))
        report += ["Hypotheses (Amendment 2, §4):", "", "| Hypothesis | Test | matchers | estimate [95% CI] | verdict |",
                   "|---|---|---|---|---|"]
        for label, test, rec, verdict in h:
            ci = "" if rec["lo"] != rec["lo"] else f" [{rec['lo']:+.4f}, {rec['hi']:+.4f}]"
            report.append(f"| {label} | {test} | {rec['matchers']} | {rec['estimate']:+.4f}{ci} | {verdict} |")
            hyps.append({"set": name, "hypothesis": label, "test": test, "matchers": rec["matchers"],
                         "estimate": rec["estimate"], "lo": rec["lo"], "hi": rec["hi"], "verdict": verdict})
        report.append("")

        # per matcher (all matchers, controls and partial coverage included, flagged)
        for s in all_seeds:
            for arm, base in pairs:
                for k in ("J", "R_exp", "F1_Hr"):
                    av, bv = P.vec(s, arm, f"sf_{k}"), P.vec(s, base, f"sf_{k}")
                    if av is None or bv is None:
                        continue
                    e = av - bv
                    cover = int(np.isfinite(e).sum())
                    per_matcher.append({"set": name, "seed": s, "control": int(is_control(s)), "arm": arm, "base": base,
                                        "metric": k, "scenarios": cover, "full": int(cover == len(keys)),
                                        "macro_effect": float(P.macro(e[None, :])[0])})
        excluded = sorted({p["seed"] for p in per_matcher if p["set"] == name and (p["control"] or not p["full"])})
        report += [f"Reported per matcher only (controls or partial coverage): {', '.join(excluded) or 'none'}.", ""]

    # 7. RQ5
    rq5 = []
    for arm, root in [(x, r7) for x in QUICK + REIFIED] + [("fd", l1), ("melnik", l1), ("fd_reified", l3)]:
        vals = defaultdict(list)
        for seed in seeds:
            f = root / f"{seed}__{arm}" / "sf_stats.tsv"
            log = root / f"{seed}__{arm}" / "run.log"  # local runs: the SFSTATS lines themselves
            lines = (f.read_text(encoding="utf-8").splitlines() if f.is_file() else
                     [l[len("SFSTATS\t"):] for l in log.read_text(encoding="utf-8", errors="replace").splitlines()
                      if l.startswith("SFSTATS\t")] if log.is_file() else [])
            if lines:
                for line in lines:
                    parts = line.split("\t")
                    if len(parts) == 2 + len(STATS) and (parts[0], parts[1]) in fam_of:
                        for k, v in zip(STATS, parts[2:]):
                            vals[k].append(float(v))
        if vals:
            rq5.append({"arm": arm, "runs": len(vals["iterations"]), **{f"median_{k}": float(np.median(v)) for k, v in vals.items()}})
    if rq5:
        report += ["## RQ5: medians per scenario run (sf_stats.tsv)", "",
                   "| arm | runs | iterations | pairwise-graph vertices | edges | build ms | iterate ms |", "|---|---|---|---|---|---|---|"]
        for r in rq5:
            report.append(f"| {r['arm']} | {r['runs']} | {r.get('median_iterations', float('nan')):.0f} | "
                          f"{r.get('median_pcg_v', float('nan')):.0f} | {r.get('median_pcg_e', float('nan')):.0f} | "
                          f"{r.get('median_build_ms', float('nan')):.0f} | {r.get('median_iterate_ms', float('nan')):.0f} |")
    for fname, data in (("contrasts.csv", contrasts), ("hypotheses.csv", hyps),
                        ("per_matcher.csv", per_matcher), ("rq5.csv", rq5)):
        if data:
            with (out / fname).open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for d in data for k in d)))
                w.writeheader()
                w.writerows(data)
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
