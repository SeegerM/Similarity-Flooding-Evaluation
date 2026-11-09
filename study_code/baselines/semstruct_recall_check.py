"""Does Recall@GT, as SemStruct reports it (median per family, Valentine's implementation), reproduce their numbers?

Valentine's RecallAtSizeofGroundTruth sorts all pairs by score (descending) and takes the first |GT|; ties keep the
order in which pairs are enumerated (row-major: source column, then target column), i.e. ties are broken by index.
Here per scenario: Recall@GT with ties broken (a) by index (Valentine), (b) in expectation over random orders (the SF
paper's expected Recall@GT), (c) pessimistically (true pairs last among ties). Reported: median and IQR per family, and
the mean, for each arm. Arms: the ablation's full / onehop / lm_only (readable Valentine, merged values) and the
collapsed 2026-09-08 checkpoint. EXPLORATORY diagnostic, no registration.

    harness/.venv-embed/Scripts/python.exe -I harness/semstruct_recall_check.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code" / "semstruct"))
import semstruct_run as sr  # noqa: E402

A = HERE / "reports" / "semstruct_ablation_20261027"
ARMS = {"SemStruct (collapsed, 2026-09-08)": HERE / "reports" / "semstruct_20260908" / "matrices" / "plain" / "semstruct",
        "full (repaired)": A / "full" / "matrices" / "plain" / "semstruct",
        "onehop": A / "onehop" / "matrices" / "plain" / "semstruct",
        "LM only": A / "full" / "matrices" / "plain" / "lm_only",
        # the paper's own scoring (softmax, then bipartite reranking; Section 4.2.1), added 2026-10-27
        "SemStruct (collapsed), reranked": HERE / "reports" / "semstruct_20260908" / "matrices" / "plain" / "semstruct_native",
        "full (repaired), reranked": A / "full" / "matrices" / "plain" / "semstruct_native",
        "onehop, reranked": A / "onehop" / "matrices" / "plain" / "semstruct_native",
        "LM only (header), reranked": A / "full" / "matrices" / "plain" / "lm_only_native"}


# Amendment 2 (`--a2`): the revised-setup run (reports/semstruct_a2_20261029, plain, row cap 2000, first seed); the
# paper's reranking is applied here to the saved cosine matrices (semstruct.native_matrix), the same function `score
# --native` would have written.
# Amendment 2b (549): SEMSTRUCT_A2_ROOT=reports/semstruct_a2_549 with SEMSTRUCT_RUN=reports/valentine549_protocol
A2 = Path(os.environ.get("SEMSTRUCT_A2_ROOT", HERE / "reports" / "semstruct_a2_20261029"))
if not A2.is_absolute():
    A2 = HERE / A2
A2_ARMS = {}
for _label, _run, _arm in (("full (revised setup)", "a2_full_V41", "semstruct"),
                           ("ValAvg (revised setup)", "a2_valavg_V41", "semstruct"),
                           ("onehop (revised setup)", "a2_onehop_V41", "semstruct"),
                           ("LM only (e5-small-v2, header)", "a2_full_V41", "lm_only")):
    _d = A2 / _run / "matrices" / "plain" / f"{_arm}__eval2000"
    A2_ARMS[_label] = _d
    A2_ARMS[_label + ", reranked"] = (_d, "rerank")


def recalls(S: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    n = int(gt.sum())
    flat, truth = S.ravel(), gt.ravel()
    order = np.argsort(-flat, kind="stable")                  # Valentine: descending, ties in enumeration order
    index = truth[order[:n]].sum() / n
    cut = flat[order[n - 1]]
    above = flat > cut
    tied = flat == cut
    k = n - int(above.sum())                                   # slots left for the tied block
    t_above, t_tied, m = int(truth[above].sum()), int(truth[tied].sum()), int(tied.sum())
    expected = (t_above + k * t_tied / m) / n
    pessimistic = (t_above + max(0, k - (m - t_tied))) / n
    # (d) per source column, as the paper's text describes it: for each source column with a true match, the share of
    # its true targets among its top-|GT_row| targets (ties by index), averaged over those columns
    rows = [i for i in range(gt.shape[0]) if gt[i].any()]
    per_row = []
    for i in rows:
        k_i = int(gt[i].sum())
        top = np.argsort(-S[i], kind="stable")[:k_i]
        per_row.append(gt[i, top].sum() / k_i)
    row_wise = float(np.mean(per_row)) if per_row else float("nan")
    return index, expected, pessimistic, row_wise


def main() -> int:
    global ARMS, A
    if "--a2" in sys.argv:
        ARMS, A = A2_ARMS, A2
    cases = sr.load_cases()
    res = {arm: defaultdict(list) for arm in ARMS}
    for case in cases:
        paths = sr.case_paths(case, "plain")
        if not paths:
            continue
        gt = sr.read_ground_truth(paths[2])
        for arm, d in ARMS.items():
            rerank = isinstance(d, tuple)
            f = (d[0] if rerank else d) / f"{case['key']}.csv"
            if not f.is_file():
                continue
            S = sr.read_written_matrix(f)
            if rerank:
                S = sr.native_matrix(S)
            if S.shape != gt.shape:
                continue
            r = recalls(S, gt)
            res[arm][case["family"]].append(r)
            res[arm]["all"].append(r)
    lines = ["# Recall@GT as SemStruct reports it: median (IQR) per family, and mean", "",
             "Ties broken by index (Valentine's implementation) / in expectation / pessimistically. Readable Valentine, "
             f"all {len(cases)} scenarios. EXPLORATORY diagnostic."
             + (" Amendment 2: the revised setup, row cap 2000, first seed." if "--a2" in sys.argv else ""), ""]
    fams = ["ChEMBL", "OpenData", "TPC-DI", "Magellan", "Wikidata", "all"]
    for kind, j in (("index order (Valentine)", 0), ("expected", 1), ("pessimistic", 2), ("per source column, index order (paper text)", 3)):
        lines += [f"## Ties: {kind}", "", "| arm | " + " | ".join(fams) + " | mean (all) |",
                  "|---" * (len(fams) + 2) + "|"]
        for arm in ARMS:
            cells = []
            for f in fams:
                v = np.array([r[j] for r in res[arm].get(f, [])])
                if len(v) == 0:
                    cells.append("–")
                    continue
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                cells.append(f"{med:.3f} ({q3 - q1:.3f})")
            allv = np.array([r[j] for r in res[arm]["all"]])
            lines.append(f"| {arm} | " + " | ".join(cells) + f" | {allv.mean():.3f} |")
        lines.append("")
    out = A / "recall_check.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
