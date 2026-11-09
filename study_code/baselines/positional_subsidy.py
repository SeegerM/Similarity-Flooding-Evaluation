"""How much of each matcher's Hungarian F1 is this corpus's column order?

FINDINGS 90 established that the corpus ground truth is the diagonal (6,422 of
6,432 pairs) and that a constant matrix therefore scores 0.7229 Hungarian F1.
It measured that for an informationless reference and asserted, without
measuring, that "no matcher here reads column position". This quantifies the
first claim per arm and tests the second directly.

The two questions are different and need different experiments.

`--mode extractor` -- the EXTRACTOR-side subsidy. Permute the target columns of
    a cached score matrix and of the ground truth by the same pi, then
    re-extract. The matcher is untouched, so any change is the extraction
    policy alone: `linear_sum_assignment` resolves an indifference toward the
    identity, and in this corpus the identity is correct.

    Read the scope of this carefully. For any permutation-EQUIVARIANT extractor
    -- E(S[:, pi]) == E(S)[:, pi] -- the F1 is unchanged identically, because
    relabelling a prediction and its ground truth together cannot move a score.
    `mutual_top1` is equivariant (a predicate on row and column maxima), and so
    is a cached native decision (a fixed boolean map). Both are therefore zero
    by construction and are reported only as a check on this harness. Hungarian
    is the sole non-equivariant extractor here, and only through tie-breaking.

    So this mode CANNOT detect a matcher that reads column order internally.
    It measures what the tie policy collects, nothing else.

`--mode rerun` -- the MATCHER-side test, and the one that answers "does this
    system read column position". Physically permute the target table's columns,
    re-run the matcher on the permuted input, and compare the matrix it returns
    against the matrix it returned before, permuted: S_permuted == S[:, pi]?
    An order-blind matcher reproduces it to the bit. Anything else is reading
    presentation order, and the elementwise check says so without a metric.

Neither existing null covers either question: SemStruct's `permuted` arm
permutes cells within columns, and REMA's `shuffled` arm recodes values per
side. Both hold column order fixed.

Usage:
    .venv-bdi/Scripts/python.exe positional_subsidy.py
    .venv-bdi/Scripts/python.exe positional_subsidy.py --cohorts
    .venv-bdi/Scripts/python.exe positional_subsidy.py --json reports/positional_subsidy.json
    .venv-bdi/Scripts/python.exe positional_subsidy.py --mode rerun --matcher coma
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
RUN = HERE / "reports" / "audited_md5_fused_20260904"
# rerun 549 (2026-11-02): SMX_POSITIONAL_RUN names another folder holding a protocol.json (the 549 protocol or the
# 168 new cases); unset = the frozen 382-case protocol
import os as _os  # noqa: E402
if _os.environ.get("SMX_POSITIONAL_RUN"):
    RUN = Path(_os.environ["SMX_POSITIONAL_RUN"])
REPEATS = 20
SEED = 20260909


# --------------------------------------------------------------------------
# scoring -- copied from valentine_matchers_corpus so the unpermuted column
# reproduces the published run rather than re-deciding the zero-cell policy.
# That policy matters: COMA's matrix is sparse, and admitting its zeros would
# let Hungarian assign on indifference and inflate the very effect measured.
# --------------------------------------------------------------------------


def read_gt(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as fh:
        for row in csv.reader(fh):
            if row and any(c.strip() for c in row):
                rows.append([float(c) if c.strip() else 0.0 for c in row])
    return np.array(rows) > 0.5


def f1(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = int((pred & gt).sum())
    if not pred.sum() or not gt.sum():
        return 0.0
    p, r = tp / pred.sum(), tp / gt.sum()
    return 2 * p * r / (p + r) if p + r else 0.0


def hungarian(S: np.ndarray) -> np.ndarray:
    M = np.zeros(S.shape, dtype=bool)
    if S.size == 0:
        return M
    r, c = linear_sum_assignment(S, maximize=True)
    for i, j in zip(r, c):
        if S[i, j] > 1e-9:
            M[i, j] = True
    return M


def mutual_top1(S: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if S.size == 0:
        return np.zeros(S.shape, dtype=bool)
    rmax = S.max(axis=1, keepdims=True)
    cmax = S.max(axis=0, keepdims=True)
    return (np.abs(S - rmax) <= eps) & (np.abs(S - cmax) <= eps) & (S > eps)


def tie_fraction(S: np.ndarray, eps: float = 1e-9) -> float:
    """Fraction of source rows whose maximum is not unique -- the material
    Hungarian has to resolve, and the only route by which this corpus's column
    order can reach a score matrix that does not itself encode position."""
    if S.size == 0:
        return 0.0
    rmax = S.max(axis=1, keepdims=True)
    return float(((S >= rmax - eps).sum(axis=1) > 1).mean())


def spread(S: np.ndarray) -> float:
    """Relative dynamic range: how far from constant the matrix actually is."""
    if S.size == 0:
        return 0.0
    scale = max(abs(float(np.nanmax(np.abs(S)))), 1e-12)
    return float((np.nanmax(S) - np.nanmin(S)) / scale)


# --------------------------------------------------------------------------
# cached-matrix sources
# --------------------------------------------------------------------------

VM = HERE / "reports" / "valentine_matchers_20260909" / "matrices"
SS = HERE / "reports" / "semstruct_20260908" / "matrices"
RM = HERE / "reports" / "rema_20260908" / "cases"
SIMA = HERE / "reports" / "sima_valentine_3fam_10silo" / "matrices"


def read_dense(path: Path) -> np.ndarray | None:
    """Tolerant reader: these caches carry trailing commas and label headers."""
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            cells = [c for c in row if c.strip() != ""]
            if not cells:
                continue
            try:
                rows.append([float(c) for c in cells])
            except ValueError:
                continue
    if not rows or len({len(r) for r in rows}) != 1:
        return None
    return np.array(rows, dtype=float)


def build_arms(cases: dict) -> dict:
    """name -> (kind, loader). `kind` is 'ranked' or 'native'."""
    arms: dict[str, tuple] = {}

    for label, sub in (("COMA (schema)", "coma"),
                       ("COMA (schema+inst)", "coma_instances"),
                       ("Cupid", "cupid")):
        for regime, tag in (("plain", "plain"), ("audited_md5", "md5")):
            d = VM / f"{sub}_{tag}"
            if not d.is_dir():
                continue
            for kind, field in (("ranked", "S"), ("native", "P")):
                def load(key, _d=d, _f=field):
                    p = _d / f"{key}.npz"
                    if not p.is_file() and _os.environ.get("SMX_POSITIONAL_VM_EXTRA"):   # 549: the 168's matrices
                        p = Path(_os.environ["SMX_POSITIONAL_VM_EXTRA"]) / _d.name / f"{key}.npz"
                    if not p.is_file():
                        return None
                    z = np.load(p)
                    return np.asarray(z[_f], dtype=float) if _f in z else None
                suffix = "" if kind == "ranked" else " [native]"
                arms[(f"{label}{suffix}", regime)] = (kind, load)

    for arm in ("semstruct", "lm_only"):
        for regime in ("plain", "audited_md5"):
            d = SS / regime / arm
            if not d.is_dir():
                continue

            def load(key, _d=d):
                p = _d / f"{key}.csv"
                return read_dense(p) if p.is_file() else None
            arms[(f"SemStruct [{arm}]", regime)] = ("ranked", load)

    for arm in ("rema", "shuffled"):
        for regime in ("plain", "audited_md5"):
            if not RM.is_dir():
                continue

            def load(key, _a=arm, _r=regime):
                p = RM / key / _r / _a / "frozen" / "scores.csv"
                return read_dense(p) if p.is_file() else None
            arms[(f"REMA [{arm}]", regime)] = ("ranked", load)

    # SiMa is keyed by scenario name under family-type directories, not by the
    # protocol's case key, so it needs a name index.
    if SIMA.is_dir():
        index = {f.stem: f for fam in SIMA.iterdir() if fam.is_dir()
                 for f in fam.glob("*.csv")}

        def load(key, _index=index, _cases=cases):
            name = _cases[key][1]["id"].rstrip("/").split("/")[-1]
            p = _index.get(name)
            return read_dense(p) if p else None
        arms[("SiMa (same-family)", "plain")] = ("ranked", load)

    return arms


# --------------------------------------------------------------------------
# mode: extractor
# --------------------------------------------------------------------------


def load_cases() -> dict:
    protocol = json.loads((RUN / "protocol.json").read_text(encoding="utf-8"))
    cases = {}
    for case in protocol["cases"]:
        files = sorted((Path(case["plain"]) / "ground_truth").glob("*.csv"))
        if files:
            gt = read_gt(files[0])
            if gt.sum():
                cases[case["key"]] = (gt, case)
    return cases


def cohorts_of(case: dict) -> list[str]:
    return ["all", "old80" if case.get("old80") else "remaining",
            f"family:{case['family']}", f"type:{case['scenario_type']}"]


def mode_extractor(args) -> list[dict]:
    cases = load_cases()
    rng = np.random.default_rng(SEED)
    # One permutation set per scenario, shared by every arm, so arms are
    # compared under identical relabellings rather than different draws.
    perms = {k: [rng.permutation(gt.shape[1]) for _ in range(args.repeats)]
             for k, (gt, _) in cases.items()}

    arms = build_arms(cases)
    arms[("Constant matrix", "--")] = ("synthetic", lambda sh: np.ones(sh))
    arms[("Random matrix", "--")] = ("synthetic", rng.random)

    rows = []
    for (name, regime), (kind, load) in arms.items():
        acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        skipped = 0
        for key, (gt, case) in cases.items():
            S = load(gt.shape) if kind == "synthetic" else load(key)
            if S is None:
                continue
            if S.shape != gt.shape or not np.isfinite(S).all():
                skipped += 1
                continue
            # A cached native decision is its own selection; do not re-extract.
            if kind == "native":
                base_h = f1(S > 0.5, gt)
                perm_h = [f1(S[:, pi] > 0.5, gt[:, pi]) for pi in perms[key]]
                base_m = perm_m = None
            else:
                base_h = f1(hungarian(S), gt)
                perm_h = [f1(hungarian(S[:, pi]), gt[:, pi]) for pi in perms[key]]
                base_m = f1(mutual_top1(S), gt)
                perm_m = [f1(mutual_top1(S[:, pi]), gt[:, pi]) for pi in perms[key]]
            for c in cohorts_of(case):
                a = acc[c]
                a["hung"].append(base_h)
                a["hung_perm"].append(float(np.mean(perm_h)))
                a["tie"].append(tie_fraction(S))
                a["spread"].append(spread(S))
                if base_m is not None:
                    a["mut"].append(base_m)
                    a["mut_perm"].append(float(np.mean(perm_m)))
        for cohort, a in acc.items():
            if not a["hung"]:
                continue
            m = {k: float(np.mean(v)) for k, v in a.items() if v}
            m.update(arm=name, regime=regime, kind=kind, cohort=cohort,
                     n=len(a["hung"]), skipped=skipped)
            m["subsidy"] = m["hung"] - m["hung_perm"]
            if "mut" in m:
                m["subsidy_mutual"] = m["mut"] - m["mut_perm"]
            rows.append(m)
    return rows


def report_extractor(rows: list[dict], cohorts: bool) -> None:
    wanted = ["all"]
    if cohorts:
        wanted += ["old80", "remaining"]
        wanted += sorted({r["cohort"] for r in rows
                          if r["cohort"].startswith(("family:", "type:"))})
    for cohort in wanted:
        sel = [r for r in rows if r["cohort"] == cohort]
        if not sel:
            continue
        print(f"\n{cohort}")
        hdr = (f"  {'arm':<26} {'regime':<12} {'n':>4} {'F1 hung':>8} "
               f"{'perm':>8} {'subsidy':>9} {'F1 mut':>8} {'tie%':>6} "
               f"{'spread':>9}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in sorted(sel, key=lambda r: -r["subsidy"]):
            mut = f"{r['mut']:>8.4f}" if "mut" in r else f"{'--':>8}"
            print(f"  {r['arm']:<26} {r['regime']:<12} {r['n']:>4} "
                  f"{r['hung']:>8.4f} {r['hung_perm']:>8.4f} "
                  f"{r['subsidy']:>+9.4f} {mut} {100*r['tie']:>6.1f} "
                  f"{r['spread']:>9.2e}")

    equivariant = [r for r in rows if r["cohort"] == "all"
                   and (r["kind"] == "native" or "subsidy_mutual" in r)]
    worst_native = max((abs(r["subsidy"]) for r in equivariant
                        if r["kind"] == "native"), default=0.0)
    worst_mutual = max((abs(r["subsidy_mutual"]) for r in equivariant
                        if "subsidy_mutual" in r), default=0.0)
    print(f"\nequivariance check (both must be 0, by construction): "
          f"native max |subsidy| {worst_native:.2e}, "
          f"mutual_top1 max |subsidy| {worst_mutual:.2e}")


# --------------------------------------------------------------------------
# mode: rerun -- the matcher-side test
# --------------------------------------------------------------------------


def mode_rerun(args) -> list[dict]:
    """Permute the target table's columns, re-run, and demand S' == S[:, pi].

    This is the only test here that can see a matcher reading presentation
    order, and it is elementwise rather than metric-based: an order-blind
    matcher reproduces its own matrix exactly under relabelling.
    """
    import pandas as pd

    sys.path.insert(0, str(HERE))
    import valentine_matchers_corpus as V

    V.load_valentine()
    from valentine.data_sources import DataframeTable

    kind, options = V.METHODS[args.matcher]
    runner = V.RUNNERS[kind]

    protocol = json.loads((RUN / "protocol.json").read_text(encoding="utf-8"))
    cases = protocol["cases"]
    if args.limit:
        cases = cases[: args.limit]

    rng = np.random.default_rng(SEED)
    rows, mismatched, failures = [], [], []
    for case in cases:
        folder = Path(case[args.regime])
        gts = sorted((Path(case["plain"]) / "ground_truth").glob("*.csv"))
        srcs = sorted((folder / "source").glob("*.csv"))
        tgts = sorted((folder / "target").glob("*.csv"))
        if not (gts and srcs and tgts):
            continue
        try:
            gt = read_gt(gts[0])
            source = pd.read_csv(srcs[0], nrows=V.MAX_ROWS, low_memory=False)
            target = pd.read_csv(tgts[0], nrows=V.MAX_ROWS, low_memory=False)
            if gt.shape != (source.shape[1], target.shape[1]):
                continue
            if len(set(source.columns)) != source.shape[1] or \
               len(set(target.columns)) != target.shape[1]:
                continue

            def run(t: "pd.DataFrame", t_cols: list[str]):
                s_table = DataframeTable(source, name="source",
                                         instance_sample_size=None)
                t_table = DataframeTable(t, name="target",
                                         instance_sample_size=None)
                return runner(s_table, t_table, list(source.columns), t_cols,
                              options, False)

            S, P = run(target, list(target.columns))
            pi = rng.permutation(target.shape[1])
            t_perm = target.iloc[:, pi]
            S2, P2 = run(t_perm, list(t_perm.columns))
        except Exception as exc:
            failures.append((case["id"], f"{type(exc).__name__}: {exc}"))
            continue

        # S2 is indexed by the permuted target order; S[:, pi] is the same
        # pairing read from the original run. They must agree elementwise.
        delta = float(np.abs(S2 - S[:, pi]).max()) if S.size else 0.0
        native_same = bool(np.array_equal(P2, P[:, pi]))
        rows.append({"id": case["id"], "family": case["family"],
                     "max_abs_delta": delta, "native_identical": native_same,
                     "f1_hung": f1(hungarian(S), gt),
                     "f1_hung_permuted": f1(hungarian(S2), gt[:, pi]),
                     "f1_native": f1(P > 0.5, gt),
                     "f1_native_permuted": f1(P2 > 0.5, gt[:, pi])})
        if delta > 1e-12 or not native_same:
            mismatched.append(rows[-1])

    if not rows:
        print("no scenarios ran")
        return rows

    worst = max(r["max_abs_delta"] for r in rows)
    print(f"{args.matcher}, regime {args.regime}: {len(rows)} scenarios, "
          f"{len(failures)} failures")
    print(f"  max |S_permuted - S[:, pi]| over all scenarios : {worst:.3e}")
    # Two thresholds, because FINDINGS 70 showed a 1-ULP difference is enough
    # to flip a Hungarian tie-break: a count at 1e-9 alone hides deviations
    # that are numerically trivial and metrically not.
    for t in (1e-12, 1e-9):
        print(f"  scenarios deviating above {t:.0e}              : "
              f"{sum(1 for r in rows if r['max_abs_delta'] > t)}")
    print(f"  scenarios whose native decision differs        : "
          f"{sum(1 for r in rows if not r['native_identical'])}")
    print(f"  scenarios whose Hungarian F1 moved             : "
          f"{sum(1 for r in rows if abs(r['f1_hung'] - r['f1_hung_permuted']) > 1e-12)}")
    for tag, a, b in (("hungarian", "f1_hung", "f1_hung_permuted"),
                      ("native", "f1_native", "f1_native_permuted")):
        base = float(np.mean([r[a] for r in rows]))
        perm = float(np.mean([r[b] for r in rows]))
        print(f"  mean F1 {tag:<10} {base:.4f} -> {perm:.4f} "
              f"({base - perm:+.4f})")
    if mismatched:
        print(f"\n  first mismatches:")
        for r in mismatched[:5]:
            print(f"    {r['id']}  delta={r['max_abs_delta']:.3e} "
                  f"native_same={r['native_identical']}")
    for cid, err in failures[:5]:
        print(f"  FAILED {cid}: {err}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("extractor", "rerun"),
                    default="extractor")
    ap.add_argument("--repeats", type=int, default=REPEATS,
                    help="permutations per scenario, extractor mode")
    ap.add_argument("--cohorts", action="store_true",
                    help="also break the extractor table down by cohort")
    ap.add_argument("--matcher", default="coma",
                    help="rerun mode: coma, coma_instances or cupid")
    ap.add_argument("--regime", choices=("plain", "audited_md5"),
                    default="plain")
    ap.add_argument("--limit", type=int, default=0,
                    help="rerun mode: first N cases of the protocol order")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if args.mode == "extractor":
        rows = mode_extractor(args)
        report_extractor(rows, args.cohorts)
    else:
        rows = mode_rerun(args)

    if args.json:
        dest = Path(args.json)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(
            {"mode": args.mode, "seed": SEED, "repeats": args.repeats,
             "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
