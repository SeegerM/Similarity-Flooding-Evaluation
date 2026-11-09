"""COMA and Cupid on the Valentine corpus, scored under this project's framework.

Two of the original Valentine baselines, reached through the Valentine package
itself (`code/valentine`, v1.0.0). Both are *classical* matchers and both are
older than everything else in this study, which is the point: they are the
reference points the schema-matching literature has used for two decades, and
this project has so far compared only structural systems (SF, PDED, PRISMA), a
contemporary LM matcher (Magneto) and two reconstructions (REMA, SemStruct).

    COMA    Do & Rahm, VLDB 2002 (papers/S17P03.pdf). A combination of
            matchers, filtered by COMA's own bidirectional DIR_BOTH /
            SEL_MULTIPLE selection.
    Cupid   Madhavan et al., VLDB 2001 (papers/P049.pdf). Linguistic name
            similarity (WordNet + tokenisation) weighted against structural
            similarity propagated over the schema tree.

What COMA actually reduces to here is worth stating plainly, because the class
docstring promises more than `build_matchers` delivers. On flat DataFrame input
it instantiates NAME_CM and, with `use_instances`, InstancesCM -- and nothing
else. Upstream drops the paper's Path/Leaves/Parents matchers deliberately
("redundant or constant in two-level schemata"), which is right for relational
tables but means:

    coma            trigram Dice over column names, plus DIR_BOTH selection.
                    A *name* matcher, and the honest comparison for it is our
                    Levenshtein baseline, not a structural system.
    coma_instances  that, plus TF-IDF cosine over cell values.

Cupid keeps its structural half, but over a two-level tree (root -> table ->
columns) there is one inner node per side, so its propagation has far less to
work with than the nested XML schemas the paper targets. Neither reduction is
a defect of this run -- it is what these algorithms are on relational input --
but a claim about "COMA" here is a claim about that reduction.

Valentine 1.0.0 reimplemented both in pure Python; neither needs a JVM. The
0.5.0 copy installed in `.venv-bdi` is the *old* one, whose COMA shells out to
`coma.jar` -- so the checkout is forced to the front of `sys.path` and the
import is asserted, rather than trusting whichever copy wins.

Both arms come from ONE computation per scenario, because this project scores
matchers by their score matrix and not by their published decision:

    native   what the matcher's own selection emits at its published defaults:
             COMA's DIR_BOTH at delta=0.15, Cupid's th_accept=0.7. A decision,
             scored directly as a match set.
    ranked   the full (source column x target column) score matrix that
             selection was applied to, passed through the same common
             extractors as every other matcher here.

The dense matrix is *not* obtained by relaxing the matcher's parameters, which
would change the algorithm -- Cupid's th_accept also drives its structural
propagation, so a run at th_accept=0 is a different matcher. It is obtained by
reading the scores before the final cut, and `--verify` asserts on every
scenario that re-applying the published cut to the dense matrix reproduces the
matcher's own `get_matches` output exactly.

Runs in the same virtual environment as Magneto and REMA:

    .venv-bdi/Scripts/python.exe valentine_matchers_corpus.py --matcher cupid
    .venv-bdi/Scripts/python.exe valentine_matchers_corpus.py --matcher coma --pilot 12
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
RUN = HERE / "reports" / "audited_md5_fused_20260904"
# rerun 549 (2026-11-02): SMX_POSITIONAL_RUN names another folder holding a protocol.json (the 549 protocol or the
# 168 new cases); unset = the frozen 382-case protocol
import os as _os  # noqa: E402
if _os.environ.get("SMX_POSITIONAL_RUN"):
    RUN = Path(_os.environ["SMX_POSITIONAL_RUN"])
CHECKOUT = HERE.parent / "code" / "valentine"

# Rows read per table. Declared, not tuned. Valentine's own `valentine_match`
# samples 1000 non-empty rows for instance matchers by default; this caps the
# read at the same number so COMA's TF-IDF corpus sees the same order of
# magnitude and the 2.2 GB corpus is not parsed in full. Cupid reads no
# instances at all, so this only bounds its I/O and dtype inference.
MAX_ROWS = 1000

METHODS = {
    # name -> (use_schema, use_instances) for COMA; Cupid takes no such knobs.
    "coma": ("coma", dict(use_schema=True, use_instances=False)),
    "coma_instances": ("coma", dict(use_schema=True, use_instances=True)),
    "cupid": ("cupid", {}),
}

# COMA's published selection defaults, applied to the dense matrix for the
# native arm. These are Coma.__init__'s defaults; kept here explicitly so a
# change upstream shows up as a --verify failure rather than silently.
COMA_NATIVE = dict(max_n=0, delta=0.15, threshold=0.0)
CUPID_NATIVE_TH_ACCEPT = 0.7


def load_valentine():
    """Import the v1.0.0 checkout, never the 0.5.0 copy in site-packages."""
    sys.path.insert(0, str(CHECKOUT))
    import valentine

    got = Path(valentine.__file__).resolve()
    want = (CHECKOUT / "valentine" / "__init__.py").resolve()
    if got != want:
        raise SystemExit(
            f"imported valentine from {got}, expected the checkout at {want}. "
            "The 0.5.0 package installed in .venv-bdi shells out to coma.jar "
            "and is not the implementation this run reports."
        )
    return valentine


# --------------------------------------------------------------------------
# metrics -- identical definitions to magneto_corpus.py, so the two reports
# are commensurable without a translation step
# --------------------------------------------------------------------------


def read_gt(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as fh:
        for row in csv.reader(fh):
            if row and any(c.strip() for c in row):
                rows.append([float(c) if c.strip() else 0.0 for c in row])
    return np.array(rows) > 0.5


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = int((pred & gt).sum())
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / gt.sum() if gt.sum() else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "n_pred": int(pred.sum())}


def hungarian(S: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

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


def recall_at_gt(S: np.ndarray, gt: np.ndarray, eps: float = 1e-9) -> float:
    k = int(gt.sum())
    if k == 0 or S.size == 0:
        return float("nan")
    flat = S.ravel()
    order = np.argsort(-flat)
    cutoff = flat[order[min(k, flat.size) - 1]]
    above = flat > cutoff + eps
    tie = np.abs(flat - cutoff) <= eps
    g = gt.ravel()
    hits = int((above & g).sum())
    q = k - int(above.sum())
    tie_wrong = int((tie & ~g).sum())
    return (hits + max(0, q - tie_wrong)) / k


def mrr_pessimistic(S: np.ndarray, gt: np.ndarray, eps: float = 1e-9) -> float:
    ranks = []
    for i in range(S.shape[0]):
        pos = np.flatnonzero(gt[i])
        if pos.size == 0:
            continue
        best = S[i, pos].max()
        ranks.append(1 + int(((~gt[i]) & (S[i] >= best - eps)).sum()))
    return float(np.mean([1.0 / r for r in ranks])) if ranks else float("nan")


# --------------------------------------------------------------------------
# the two matchers, each returning (dense S, native decision P)
# --------------------------------------------------------------------------


def run_coma(source_table, target_table, s_cols, t_cols, options, verify: bool):
    """COMA at delta=0 gives every positively-scored pair, i.e. the matrix.

    `select_both_multiple` is a pure function of that matrix, so re-applying it
    at COMA's published delta reproduces `Coma(delta=0.15).get_matches` exactly
    -- which is what --verify checks, at the cost of a second matcher run.
    """
    from valentine.algorithms import Coma
    from valentine.algorithms.coma.selection import select_both_multiple

    dense = Coma(max_n=0, delta=0.0, threshold=0.0, **options).get_matches(
        source_table, target_table
    )
    sim = {(p.source_column, p.target_column): v for p, v in dense.items()}

    s_index = {c: i for i, c in enumerate(s_cols)}
    t_index = {c: j for j, c in enumerate(t_cols)}
    S = np.zeros((len(s_cols), len(t_cols)))
    for (a, b), v in sim.items():
        S[s_index[a], t_index[b]] = float(v)

    selected = select_both_multiple(sim, list(s_cols), list(t_cols), **COMA_NATIVE)
    P = np.zeros(S.shape, dtype=bool)
    for a, b in selected:
        P[s_index[a], t_index[b]] = True

    if verify:
        native = Coma(**COMA_NATIVE, **options).get_matches(source_table, target_table)
        want = {(p.source_column, p.target_column) for p in native}
        if want != set(selected):
            raise AssertionError(
                f"COMA native arm does not reproduce get_matches: "
                f"{len(want)} vs {len(selected)} pairs"
            )
    return S, P


def run_cupid(source_table, target_table, s_cols, t_cols, options, verify: bool):
    """Cupid's leaf-pair wsim matrix, read before `mapping_generation_leaves`.

    Cupid has no public accessor for it, and relaxing th_accept is not an
    option -- `tree_match` passes th_accept into `compute_ssim`, so it changes
    the propagation and not just the cut. So the matcher is driven through its
    own private helpers at its published parameters, and the wsim map that
    `get_matches` would threshold is read out directly.
    """
    from valentine.algorithms import Cupid
    from valentine.algorithms.cupid.tree_match import recompute_wsim, tree_match

    cu = Cupid(**options)
    cu._Cupid__add_data("DB__" + source_table.name, source_table)
    cu._Cupid__add_data("DB__" + target_table.name, target_table)
    s_tree = cu._Cupid__get_schema_by_name("DB__" + source_table.name)
    t_tree = cu._Cupid__get_schema_by_name("DB__" + target_table.name)
    sims = tree_match(
        s_tree, t_tree, cu._Cupid__categories,
        cu._Cupid__leaf_w_struct, cu._Cupid__w_struct, cu._Cupid__th_accept,
        cu._Cupid__th_high, cu._Cupid__th_low, cu._Cupid__c_inc,
        cu._Cupid__c_dec, cu._Cupid__th_ns, cu._Cupid__process_num,
    )
    sims = recompute_wsim(s_tree, t_tree, sims)

    leaves_s, leaves_t = set(s_tree.get_leaf_names()), set(t_tree.get_leaf_names())
    s_index = {c: i for i, c in enumerate(s_cols)}
    t_index = {c: j for j, c in enumerate(t_cols)}
    S = np.zeros((len(s_cols), len(t_cols)))
    pairs = {}
    for (a, b), v in sims.items():
        if a in leaves_s and b in leaves_t:
            # long_name is (table, table_guid, column, column_guid)
            i, j = s_index.get(a[2]), t_index.get(b[2])
            if i is not None and j is not None:
                S[i, j] = float(v["wsim"])
                pairs[(a[2], b[2])] = float(v["wsim"])

    P = S >= CUPID_NATIVE_TH_ACCEPT

    if verify:
        native = Cupid(**options).get_matches(source_table, target_table)
        want = {(p.source_column, p.target_column) for p in native}
        got = {k for k, v in pairs.items() if v >= CUPID_NATIVE_TH_ACCEPT}
        if want != got:
            raise AssertionError(
                f"Cupid native arm does not reproduce get_matches: "
                f"{len(want)} vs {len(got)} pairs"
            )
    return S, P


RUNNERS = {"coma": run_coma, "cupid": run_cupid}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matcher", required=True, choices=sorted(METHODS),
                    help="coma (schema only), coma_instances, or cupid")
    ap.add_argument("--regime", choices=("plain", "audited_md5"), default="plain")
    ap.add_argument("--corpus", choices=("valentine", "external"),
                    default="valentine",
                    help="'valentine' is the frozen 382-case protocol, where a "
                         "case carries both regimes as parallel copies. "
                         "'external' is the controlled MusicBrainz/Lahman v3 "
                         "corpus, where each evidence regime is its own case "
                         "with its own id -- so --regime does not apply and "
                         "every case is run exactly once.")
    ap.add_argument("--cases",
                    default="reports/controlled_matchers_v3_20260905/cases.csv",
                    help="external corpus case index")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N cases of the frozen protocol order")
    ap.add_argument("--pilot", type=int, default=0,
                    help="size-stratified sample for a runtime estimate; "
                         "prints a projection to the full corpus and writes "
                         "nothing unless -o is given")
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS)
    ap.add_argument("--verify", action="store_true",
                    help="re-run each matcher at its published parameters and "
                         "assert the native arm reproduces it exactly")
    ap.add_argument("--matrices", default="",
                    help="directory to cache dense matrices in, for resume")
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()

    load_valentine()
    from valentine.data_sources import DataframeTable

    kind, options = METHODS[args.matcher]
    runner = RUNNERS[kind]

    if args.corpus == "external":
        # Normalise the v3 index into the same shape the protocol cases have,
        # so nothing downstream needs to know which corpus it is reading. Both
        # regime keys point at the one directory because the regime is already
        # baked into the case.
        cases = []
        with Path(args.cases).open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                cases.append({"id": r["id"], "key": r["key"],
                              "family": r["dataset"],
                              "cohort": "external",
                              "scenario_type": r["scenario_type"],
                              "scenario": r["scenario"],
                              "dataset": r["group"],
                              "old80": False,
                              "evidence_regime": r["evidence_regime"],
                              "plain": r["path"], "audited_md5": r["path"]})
        protocol = {"cases": cases}
    else:
        protocol = json.loads((RUN / "protocol.json").read_text(encoding="utf-8"))
    cases = protocol["cases"]

    if args.pilot:
        cases = stratified_pilot(cases, args.pilot)
        print(f"pilot: {len(cases)} of {len(protocol['cases'])} cases, "
              f"stratified by column-pair count")
    elif args.limit:
        cases = cases[: args.limit]

    cache = Path(args.matrices) if args.matrices else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    rows, failures, sizes = [], [], []
    t_start = time.perf_counter()
    for index, case in enumerate(cases, 1):
        folder = Path(case[args.regime])
        gts = sorted((Path(case["plain"]) / "ground_truth").glob("*.csv"))
        srcs = sorted((folder / "source").glob("*.csv"))
        tgts = sorted((folder / "target").glob("*.csv"))
        if not gts or not srcs or not tgts:
            failures.append((case["id"], "missing source/target/ground truth"))
            continue
        try:
            gt = read_gt(gts[0])
            source = pd.read_csv(srcs[0], nrows=args.max_rows, low_memory=False)
            target = pd.read_csv(tgts[0], nrows=args.max_rows, low_memory=False)
            if gt.shape != (source.shape[1], target.shape[1]):
                failures.append((case["id"], "shape mismatch"))
                continue
            # Position is the identity everywhere else in this harness, but
            # both matchers key on labels; a duplicated label would silently
            # merge two columns' scores.
            if len(set(source.columns)) != source.shape[1] or \
               len(set(target.columns)) != target.shape[1]:
                failures.append((case["id"], "duplicate column labels"))
                continue

            key = None if cache is None else cache / (case["key"] + ".npz")
            t0 = time.perf_counter()
            if key is not None and key.is_file():
                z = np.load(key)
                S, P, elapsed = z["S"], z["P"].astype(bool), float(z["seconds"])
            else:
                s_table = DataframeTable(source, name="source",
                                         instance_sample_size=None)
                t_table = DataframeTable(target, name="target",
                                         instance_sample_size=None)
                S, P = runner(s_table, t_table, list(source.columns),
                              list(target.columns), options, args.verify)
                elapsed = time.perf_counter() - t0
                if key is not None:
                    np.savez_compressed(key, S=S, P=P, seconds=elapsed)
        except Exception as exc:  # recorded, never silently skipped
            failures.append((case["id"], f"{type(exc).__name__}: {exc}"))
            continue

        sizes.append((source.shape[1] * target.shape[1], elapsed))
        base = {"id": case["id"], "family": case["family"],
                "cohort": case["cohort"], "seconds": elapsed,
                "scenario_type": case["scenario_type"],
                "old80": str(case["old80"]),
                # For the external corpus the regime is a property of the case,
                # not of the run, so record the case's own value rather than
                # the --regime flag, which does not apply there.
                "regime": case.get("evidence_regime", args.regime)}
        rgt = recall_at_gt(S, gt)
        rows.append({**base, "arm": "native", **metrics(P, gt),
                     "mrr": float("nan"), "recall_at_gt": float("nan")})
        rows.append({**base, "arm": "ranked_hungarian",
                     **metrics(hungarian(S), gt),
                     "mrr": mrr_pessimistic(S, gt), "recall_at_gt": rgt})
        rows.append({**base, "arm": "ranked_mutual_top1",
                     **metrics(mutual_top1(S), gt),
                     "mrr": mrr_pessimistic(S, gt), "recall_at_gt": rgt})
        if index % 10 == 0 or index == len(cases):
            done = time.perf_counter() - t_start
            print(f"  {index}/{len(cases)}  {len(rows)//3} scored, "
                  f"{len(failures)} failed, {done/60:.1f} min elapsed",
                  flush=True)

    if not rows:
        print("no scenarios scored")
        for cid, why in failures[:20]:
            print(f"   {cid.split('/')[-1]}: {why[:100]}")
        return 1

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {out}")

    print(f"scored {len(rows)//3} scenarios, {len(failures)} failures")
    for cid, why in failures[:10]:
        print(f"   {cid.split('/')[-1]}: {why[:100]}")

    print(f"\n{'arm':<22}{'F1':>8}{'precision':>11}{'recall':>9}"
          f"{'|M|':>7}{'MRR':>8}{'R@GT':>8}")
    for arm in ("native", "ranked_hungarian", "ranked_mutual_top1"):
        sel = [r for r in rows if r["arm"] == arm]
        if not sel:
            continue
        mrrs = [r["mrr"] for r in sel if r["mrr"] == r["mrr"]]
        rgts = [r["recall_at_gt"] for r in sel if r["recall_at_gt"] == r["recall_at_gt"]]
        print(f"  {arm:<20}{mean(r['f1'] for r in sel):>8.4f}"
              f"{mean(r['precision'] for r in sel):>11.4f}"
              f"{mean(r['recall'] for r in sel):>9.4f}"
              f"{mean(r['n_pred'] for r in sel):>7.1f}"
              f"{(mean(mrrs) if mrrs else float('nan')):>8.4f}"
              f"{(mean(rgts) if rgts else float('nan')):>8.4f}")

    report_timing(sizes, protocol["cases"], args)
    return 0


def stratified_pilot(cases: list[dict], n: int) -> list[dict]:
    """Sample across the column-pair size range, which is what cost tracks.

    Deterministic: cases are ordered by n_src * n_tgt and sampled at evenly
    spaced quantiles, so the pilot spans the cheapest and the most expensive
    scenario rather than whatever the protocol happens to list first.
    """
    sized = []
    for c in cases:
        f = Path(c["plain"])
        srcs = sorted((f / "source").glob("*.csv"))
        tgts = sorted((f / "target").glob("*.csv"))
        if not srcs or not tgts:
            continue
        sized.append((_ncols(srcs[0]) * _ncols(tgts[0]), c))
    sized.sort(key=lambda x: (x[0], x[1]["id"]))
    if n >= len(sized):
        return [c for _, c in sized]
    step = (len(sized) - 1) / (n - 1) if n > 1 else 0
    return [sized[round(i * step)][1] for i in range(n)]


def _ncols(path: Path) -> int:
    with path.open(newline="", encoding="utf-8", errors="ignore") as fh:
        try:
            return len(next(csv.reader(fh)))
        except StopIteration:
            return 0


def report_timing(sizes, all_cases, args) -> None:
    """Project the corpus cost from the measured per-scenario cost.

    Two projections, because neither is safe alone: the mean is what a uniform
    sample supports, and the size-fitted one is what a stratified pilot
    supports. When they disagree the cost is not linear in column pairs and
    the fitted number is the one to trust for a stratified sample.
    """
    if not sizes:
        return
    secs = [s for _, s in sizes]
    print(f"\nper scenario: median {median(secs):.2f}s, mean {mean(secs):.2f}s, "
          f"max {max(secs):.2f}s over {len(secs)} scenarios")

    pairs = np.array([p for p, _ in sizes], dtype=float)
    times = np.array(secs, dtype=float)
    total_pairs = 0
    for c in all_cases:
        f = Path(c["plain"])
        srcs = sorted((f / "source").glob("*.csv"))
        tgts = sorted((f / "target").glob("*.csv"))
        if srcs and tgts:
            total_pairs += _ncols(srcs[0]) * _ncols(tgts[0])

    flat = mean(secs) * len(all_cases)
    print(f"corpus projection ({len(all_cases)} scenarios, "
          f"{total_pairs:,} column pairs):")
    print(f"  from the mean          {flat/60:>8.1f} min")
    if pairs.sum() > 0 and len(sizes) > 2:
        # cost ~ a + b * pairs, least squares through the pilot
        A = np.vstack([np.ones_like(pairs), pairs]).T
        a, b = np.linalg.lstsq(A, times, rcond=None)[0]
        fitted = a * len(all_cases) + b * total_pairs
        print(f"  fitted a + b*pairs     {fitted/60:>8.1f} min   "
              f"(a={a:.3f}s/scenario, b={b*1000:.3f}s per 1000 pairs)")
    print(f"  both regimes           {'~2x the above':>8}")


if __name__ == "__main__":
    raise SystemExit(main())
