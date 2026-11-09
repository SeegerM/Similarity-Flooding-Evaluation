"""How much of this corpus is solved by column position alone?

Found while building a chance reference for the REMA comparison: an all-ones
matrix, which carries no information whatsoever, scores 0.72 Hungarian F1 over
all 382 scenarios. That is not a bug in the metric. `linear_sum_assignment`
returns the identity permutation for a constant matrix, and in this corpus the
ground truth *is* almost entirely the identity: source column `i` corresponds
to target column `i`.

FINDINGS 89 saw the tip of this. It reported 68 scenarios with exactly one true
pair, all at (0,0), and measured an all-ones matrix at 0.1369 on that subset.
This is the same phenomenon at corpus scale: 312 of the 314 *multi*-pair
scenarios are fully diagonal too, and those carry 6,364 of the 6,432 pairs. The
single-pair set FINDINGS 89 examined is 1% of the ground truth.

What this does and does not mean:

  * It does **not** mean any matcher here cheats. None of them reads column
    position. The regularity is in the corpus, and it is a consequence of how
    Valentine's splits are generated: they preserve schema order.
  * It **does** mean a position-aware baseline is very strong on this corpus,
    and that any *absolute* F1 level reported on it is not comparable to a
    level from a corpus without the regularity.
  * It **does** mean matchers are not equally exposed. Extraction resolves
    ties, `linear_sum_assignment` resolves them toward the diagonal, and the
    diagonal is correct here -- so a matcher that ties a lot collects free
    correctness. FINDINGS 78 already established that thin matchers tie most.

Paired contrasts on a fixed cohort are the reason this does not overturn the
project's conclusions: a corpus-level regularity both arms see cancels in the
difference, exactly as FINDINGS 89 argued. But the argument has to be made at
this scale rather than for 18% of the corpus.

Two independent verification paths, because the claim is a large one. The
manifest's recorded pair list and the ground-truth CSV are parsed separately
and required to agree, and the identity baseline is scored both analytically
and through the harness's own extractors.

Usage:
    python positional_alignment.py
    python positional_alignment.py --json reports/positional_alignment.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from smx.core import ScoreMatrix, evaluate  # noqa: E402

RUN = HERE / "reports" / "audited_md5_fused_20260904"
OFF_DIAGONAL_TOLERANCE = 0.0


def load_protocol() -> dict:
    return json.loads((RUN / "protocol.json").read_text(encoding="utf-8"))


def pairs_from_csv(path: Path) -> list[tuple[int, int]]:
    """Parse the ground-truth grid directly, not through the harness reader.

    A bug in the shared reader must not be able to manufacture this finding.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        grid = [[cell.strip() for cell in row]
                for row in csv.reader(handle) if row and any(row)]
    return [(i, j) for i, row in enumerate(grid)
            for j, value in enumerate(row) if value not in ("", "0", "0.0")]


def survey(protocol: dict) -> tuple[list[dict], list[str]]:
    rows, disagreements = [], []
    for case in protocol["cases"]:
        folder = Path(case["plain"])
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        from_csv = pairs_from_csv(folder / manifest["ground_truth"]["path"])
        recorded = [tuple(p) for p in manifest["ground_truth"]["pairs"]]
        if sorted(from_csv) != sorted(recorded):
            disagreements.append(case["id"])
        off = [(i, j) for i, j in from_csv if i != j]
        rows.append({
            "id": case["id"], "family": case["family"],
            "scenario_type": case["scenario_type"], "cohort": case["cohort"],
            "n_source": len(manifest["tables"]["source"]["columns"]),
            "n_target": len(manifest["tables"]["target"]["columns"]),
            "n_ground_truth": len(from_csv), "n_off_diagonal": len(off),
            "fully_diagonal": not off,
            "single_pair": len(from_csv) == 1,
        })
    return rows, disagreements


def identity_scores(rows: list[dict], protocol: dict) -> dict:
    """Score the informationless baseline through the harness's own extractors.

    A constant matrix, so the extractors see no signal at all; whatever they
    return is the tie-resolution policy meeting the corpus's column order.
    """
    by_key = {c["id"]: c for c in protocol["cases"]}
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        case = by_key[row["id"]]
        folder = Path(case["plain"])
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        shape = (len(manifest["tables"]["source"]["columns"]),
                 len(manifest["tables"]["target"]["columns"]))
        gt = np.zeros(shape, dtype=bool)
        for i, j in pairs_from_csv(folder / manifest["ground_truth"]["path"]):
            gt[i, j] = True
        constant = np.ones(shape)
        for res in evaluate(ScoreMatrix(scores=constant, ground_truth=gt),
                            policies=["hungarian", "mutual_top1",
                                      "top1_per_source"]):
            out[res["extractor"]]["f1"].append(res["f1"])
            out[res["extractor"]]["precision"].append(res["precision"])
            out[res["extractor"]]["recall"].append(res["recall"])
    return {k: {m: mean(v) for m, v in d.items()} for k, d in out.items()}


def project_methods() -> dict[str, float]:
    """The project's own frozen plaintext Hungarian F1, for the contrast."""
    path = RUN / "scenarios.csv"
    if not path.is_file():
        return {}
    per: dict[str, list[float]] = defaultdict(list)
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if row["regime"] == "plain" and row["extractor"] == "hungarian":
            per[row["method"]].append(float(row["f1"]))
    return {k: mean(v) for k, v in per.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    protocol = load_protocol()
    rows, disagreements = survey(protocol)
    if disagreements:
        print(f"ground truth disagrees with its own manifest in "
              f"{len(disagreements)} scenarios: {disagreements[:5]}")

    total_pairs = sum(r["n_ground_truth"] for r in rows)
    off = sum(r["n_off_diagonal"] for r in rows)
    single = [r for r in rows if r["single_pair"]]
    multi = [r for r in rows if not r["single_pair"]]

    print("=" * 72)
    print("Is the ground truth on the diagonal?")
    print("=" * 72)
    print(f"  scenarios                                {len(rows):>6}")
    print(f"  ground-truth pairs                       {total_pairs:>6}")
    print(f"  pairs off the diagonal                   {off:>6}"
          f"  ({100 * off / total_pairs:.2f}%)")
    print(f"  scenarios with any off-diagonal pair     "
          f"{sum(1 for r in rows if not r['fully_diagonal']):>6}")
    print()
    print(f"  single-pair scenarios (FINDINGS 89)      {len(single):>6}"
          f"  fully diagonal {sum(1 for r in single if r['fully_diagonal']):>4}"
          f"  pairs {sum(r['n_ground_truth'] for r in single):>5}")
    print(f"  multi-pair scenarios                     {len(multi):>6}"
          f"  fully diagonal {sum(1 for r in multi if r['fully_diagonal']):>4}"
          f"  pairs {sum(r['n_ground_truth'] for r in multi):>5}")
    print(f"  multi-pair |GT|: median "
          f"{median(r['n_ground_truth'] for r in multi):.0f}, "
          f"max {max(r['n_ground_truth'] for r in multi)}")

    print()
    print("=" * 72)
    print("What an informationless constant matrix scores, per extractor")
    print("=" * 72)
    scores = identity_scores(rows, protocol)
    for extractor, metrics in scores.items():
        print(f"  {extractor:<18} F1 {metrics['f1']:.4f}"
              f"  P {metrics['precision']:.4f}  R {metrics['recall']:.4f}")
    print("  (Hungarian assigns the identity on a constant matrix, so its")
    print("   recall is the share of true pairs that lie on the diagonal.)")

    print()
    print("=" * 72)
    print("Beside the project's own frozen methods, plaintext Hungarian F1")
    print("=" * 72)
    methods = project_methods()
    constant_f1 = scores.get("hungarian", {}).get("f1", float("nan"))
    for name, value in sorted(methods.items(), key=lambda kv: -kv[1]):
        flag = "  <-- below the constant matrix" if value < constant_f1 else ""
        print(f"  {name:<34} {value:.4f}{flag}")
    print(f"  {'constant matrix (no information)':<34} {constant_f1:.4f}")

    print()
    print("=" * 72)
    print("Per family and scenario type")
    print("=" * 72)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[f"{row['family']}/{row['scenario_type']}"].append(row)
    print(f"  {'family/type':<38}{'n':>4}{'diagonal':>10}{'pairs':>8}")
    for key in sorted(groups):
        group = groups[key]
        print(f"  {key:<38}{len(group):>4}"
              f"{sum(1 for r in group if r['fully_diagonal']):>10}"
              f"{sum(r['n_ground_truth'] for r in group):>8}")

    if args.json:
        payload = {"scenarios": rows, "constant_matrix": scores,
                   "project_methods": methods,
                   "totals": {"pairs": total_pairs, "off_diagonal": off,
                              "single_pair": len(single),
                              "multi_pair": len(multi)},
                   "manifest_disagreements": disagreements}
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
