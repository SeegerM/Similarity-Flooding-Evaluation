"""TODO item B: what does the dependency layer add over a *strong* direct baseline?

Every dependency result in this project is measured against the frozen 65/35
seed. FINDINGS 73 showed why that is no longer good enough: on plaintext SF,
char3 alone reaches 0.6585 while char3 plus the three dependency channels
reaches 0.6281. Dependency evidence is valuable against a weak base and largely
*substitutive* once a strong direct value signal is present. A reviewer is
entitled to ask whether the dependency contribution is an artifact of comparing
against a deliberately weak seed.

So this builds a deliberately boring ladder, each rung adding one kind of
information, all under **identical information access** -- same scenarios, same
extractor, same fusion weight, no fitted parameter anywhere:

    1. lexical          the production Levenshtein label matcher alone
    2. seed             the frozen 0.65 Levenshtein + 0.35 PMF composite
    3. direct           seed + char3 + exact value overlap + unary cardinality
                        statistics -- everything obtainable from labels and
                        values without any dependency profiling
    4. direct + deps    rung 3 plus the FD, DC and OD anchor channels at their
                        frozen budgets

The quantity the paper needs is `rung 4 - rung 3`: what the dependency layer
adds once the direct signal is already strong. `rung 3 - rung 2` is the other
half of the story -- how much was being left on the table by the seed.

No router, no feature sweep, no weighting. The channels within a rung are fused
by the same `fuse` used everywhere else, which unit-normalises and averages
them at weight 0.35.

The cardinality rung deserves one caution. Distinctness and entropy are
*unary* statistics: they describe a column without reference to the other
schema, so a pair of columns matches on them by having similar shape, which
many unrelated pairs do. It is included because it is cheap and standard, not
because it is expected to help.

Usage:
    python baseline_ladder.py [--limit N] [-o reports/baseline_ladder.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from anchor_generalization import anchor_matrix, load_side  # noqa: E402
from compose_gate import fuse  # noqa: E402
from controlled_matcher_experiment import od_channel  # noqa: E402
from encrypted_feature_ablation import read_matrix as read_truth  # noqa: E402
from encrypted_feature_ablation import unary_statistics  # noqa: E402
from gated_method import dc_channel, fd_channel, scenario_evidence  # noqa: E402
from matrix_health import health  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402
from smx.soft_overlap import (exact_matrix, fuzzy_matrix, normalise,  # noqa: E402
                              sample_values)
from valentine_sweep import strict_columns  # noqa: E402

FULL = HERE / "reports" / "valentine_full_382_20260903"
SEEDS = HERE / "reports" / "hellinger_seed_20260905"
RUN = HERE / "reports" / "audited_md5_fused_20260904"
WEIGHT = 0.35
CAP, SEED = 128, 20260903
FD_BUDGET, DC_BUDGET, OD_BUDGET = 0.282, 0.576, 0.423
POLICIES = ("hungarian", "mutual_top1")

RUNGS = ["1. lexical", "2. seed (0.65 Lev + 0.35 PMF)",
         "3. direct (seed + char3 + overlap + cardinality)",
         "4. direct + FD + DC + OD"]


def bare_headers(folder: Path, side: str):
    f = sorted((folder / side).glob("*.csv"))[0]
    with f.open(newline="", encoding="utf-8", errors="ignore") as fh:
        return [h.strip().split(".")[-1] for h in next(csv.reader(fh), [])]


def columns_of(folder: Path):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return [strict_columns(folder / manifest["tables"][side]["path"],
                           manifest["tables"][side])
            for side in ("source", "target")]


def cardinality_channel(raw):
    """Cosine over the unary statistics already defined for the ablation."""
    stats = [unary_statistics(side) for side in raw]
    names = sorted(stats[0])
    left = np.stack([stats[0][n] for n in names], axis=1)
    right = np.stack([stats[1][n] for n in names], axis=1)
    ln = np.linalg.norm(left, axis=1, keepdims=True)
    rn = np.linalg.norm(right, axis=1, keepdims=True)
    left = left / np.where(ln == 0, 1, ln)
    right = right / np.where(rn == 0, 1, rn)
    return np.clip(left @ right.T, 0.0, 1.0)


def build(case):
    """Every rung's matrix for one scenario, or None."""
    folder = Path(case["plain"])
    gts = sorted((folder / "ground_truth").glob("*.csv"))
    seed_path = SEEDS / "cases" / case["key"] / "seed.csv"
    lex_path = FULL / "cases" / case["key"] / "levenshtein.csv"
    if not gts or not seed_path.is_file() or not lex_path.is_file():
        return None, None
    gt = read_truth(gts[0])[0] > 0.5
    lex = read_matrix(lex_path)[0]
    seed = read_matrix(seed_path)[0]
    if lex.shape != gt.shape or seed.shape != gt.shape:
        return None, None
    # FINDINGS 76: two ChEMBL scenarios carry all-zero matrices for every
    # matcher, recorded as successes. Scoring them yields a clean 0.000 that
    # is not a result. Dropped here rather than rewritten upstream, because
    # the old artifact stays immutable.
    if health(lex)["degenerate"] or health(seed)["degenerate"]:
        return None, "degenerate"

    raw = columns_of(folder)
    s_cols, t_cols = bare_headers(folder, "source"), bare_headers(folder, "target")

    non_null = [[[v for v in col if v.strip()] for col in side] for side in raw]
    normalised = [[[v for v in sorted({normalise(x) for x in col}) if v]
                   for col in side] for side in non_null]
    samples = [[sample_values(col, CAP, SEED) for col in side] for side in normalised]
    direct = [fuzzy_matrix(*samples, kind="char3"),
              exact_matrix(*non_null, variant="jaccard"),
              cardinality_channel(raw)]
    if any(m.shape != gt.shape for m in direct):
        return None, None

    ev = scenario_evidence(folder, seed_path, None, SEED, use_ground_truth=False)
    deps = []
    if ev is not None:
        try:
            deps.append(fd_channel(ev, FD_BUDGET))
        except ValueError:
            deps = []
        if deps:
            dc = dc_channel(ev, DC_BUDGET)
            if dc is not None:
                deps.append(dc)
            s_values, _ = load_side(folder, "source")
            t_values, _ = load_side(folder, "target")
            anchors = anchor_matrix(s_values, t_values, s_cols, t_cols)
            od, _ = od_channel(raw[0], raw[1], s_cols, t_cols, anchors)
            if od is not None:
                deps.append(od)
    deps = [d for d in deps if d is not None and d.shape == gt.shape]

    rung3 = fuse(seed, direct, WEIGHT)
    matrices = {
        RUNGS[0]: lex,
        RUNGS[1]: seed,
        RUNGS[2]: rung3,
        RUNGS[3]: fuse(rung3, deps, WEIGHT) if deps else None,
    }
    return gt, matrices


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="reports/baseline_ladder.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    protocol = json.loads((RUN / "protocol.json").read_text(encoding="utf-8"))
    cases = protocol["cases"][: args.limit] if args.limit else protocol["cases"]

    rows, skipped = [], defaultdict(int)
    for index, case in enumerate(cases, 1):
        gt, matrices = build(case)
        if gt is None:
            skipped["degenerate" if matrices == "degenerate" else "unbuildable"] += 1
            continue
        for rung, matrix in matrices.items():
            if matrix is None:
                skipped[f"no deps: {rung}"] += 1
                continue
            for res in evaluate(ScoreMatrix(scores=matrix, ground_truth=gt),
                                policies=list(POLICIES)):
                rows.append({"id": case["id"], "family": case["family"],
                             "cohort": case["cohort"], "rung": rung,
                             "extractor": res["extractor"], "f1": res["f1"],
                             "recall_at_gt": res["recall_at_gt_pessimistic"],
                             "mrr": res["mrr_pessimistic"]})
        if index % 50 == 0:
            print(f"  {index}/{len(cases)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")
    if skipped:
        print("skipped:", dict(skipped))

    # Report on one cohort: the scenarios every rung scored.
    by = defaultdict(dict)
    for r in rows:
        if r["extractor"] == "hungarian":
            by[r["rung"]][r["id"]] = float(r["f1"])
    for cohort_name, keep in (("confirmation", "remaining338"), ("all", None)):
        ids = {r["id"] for r in rows if keep is None or r["cohort"] == keep}
        common = set.intersection(*(set(v) for v in by.values())) & ids
        if not common:
            continue
        print(f"\n=== {cohort_name}: {len(common)} scenarios scored by every rung ===")
        prev = None
        for rung in RUNGS:
            vals = by.get(rung)
            if not vals:
                continue
            m = mean(vals[i] for i in common)
            step = ""
            if prev is not None:
                d = [vals[i] - prev[i] for i in common]
                w = sum(1 for x in d if x > 1e-12)
                l = sum(1 for x in d if x < -1e-12)
                step = f"   step {mean(d):+.4f}  ({w}W {l}L)"
            print(f"  {rung:<50} F1 {m:.4f}{step}")
            prev = vals
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
