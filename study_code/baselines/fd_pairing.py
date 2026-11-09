"""Do source FDs have counterparts in the target schema?

The budget experiments failed to predict a useful k from any single-schema
statistic. The standing hypothesis is that whether an FD helps depends on
whether the *other* schema carries the same dependency, which no per-schema
summary can see.

Before building a cross-schema filter, this checks whether the hypothesis can
discriminate at all. Using the ground-truth column correspondence as an oracle,
each source FD X -> A is mapped into target column names and we ask whether the
target schema actually contains that dependency.

Three outcomes matter:
  - nearly every FD is matched  -> the feature cannot discriminate, idea dead
  - almost none are matched     -> the signal is too sparse to filter on
  - a middling, varying fraction-> worth building

Usage:
    python fd_pairing.py <results-run-dir> <schematch-data-dir>
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
import re
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smx.adapters import read_matrix  # noqa: E402


PDEP_RE = re.compile(r"\s*\(pdep\s+[0-9.eEnN+-]+\s*,\s*[0-9.eEnN+-]+\s*\)\s*$")


def normalise(qualified: str) -> str:
    """'Books.csv.BookID' and 'Books.BookID' both become 'Books.BookID'."""
    name = qualified.strip()
    # No " (" truncation here: the "(pdep x, y)" annotation is stripped in
    # parse_fds by an exact pattern, and cutting here as well would shorten
    # every legitimate column name containing a parenthesis.
    parts = [p for p in name.split(".") if p.lower() != "csv"]
    return ".".join(parts)


def parse_fds(path: Path):
    """Read one FD_results.txt into (frozenset(determinant), dependent)."""
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or "-->" not in line:
            continue
        left, right = line.split("-->", 1)
        left = left.strip()
        if left.startswith("["):
            left = left[1:]
        if left.endswith("]"):
            left = left[:-1]
        determinant = frozenset(normalise(p) for p in left.split(",") if p.strip())
        # Strip only the trailing "(pdep x, y)" annotation. Cutting at the
        # first " (" instead truncates any column name that contains one --
        # every OpenData table has such names, e.g. "Climate change
        # adaptation (marker)" -- which silently shortens the column and makes
        # every correspondence lookup for it miss.
        dependent = normalise(PDEP_RE.sub("", right).strip())
        if determinant and dependent:
            out.append((determinant, dependent))
    return out


def scenario_fds(data_dir: Path, dataset: str, scenario: str, side: str):
    base = data_dir / dataset / scenario / "metadata" / side
    fds = []
    if base.is_dir():
        for fd_file in sorted(base.rglob("FD_results.txt")):
            fds.extend(parse_fds(fd_file))
    return fds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("data_dir")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)

    # one verbose sim matrix per scenario gives the column order the ground
    # truth matrix is indexed by
    sample = {}
    for csv_path in sorted(results_dir.rglob("*.csv")):
        if "MatchingStepLine2" not in str(csv_path):
            continue
        parts = csv_path.relative_to(results_dir).parts
        if len(parts) >= 2:
            sample.setdefault((parts[0], parts[1]), csv_path)

    print(f"{'dataset':<18} {'scenario':<26} {'srcFD':>6} {'tgtFD':>6} {'matched':>8} {'frac':>6}")
    fractions, per_dataset = [], defaultdict(list)
    total_src = total_matched = 0

    for (dataset, scenario), path in sorted(sample.items()):
        _, row_labels, col_labels = read_matrix(path)
        gt_dir = data_dir / dataset / scenario / "ground_truth"
        gt_files = sorted(gt_dir.glob("*.csv")) if gt_dir.is_dir() else []
        if not gt_files or not row_labels or not col_labels:
            continue
        gt, _, _ = read_matrix(gt_files[0])
        if gt.shape != (len(row_labels), len(col_labels)):
            continue

        # source column -> its ground-truth counterparts in the target
        image = defaultdict(set)
        for i, s in enumerate(row_labels):
            for j, t in enumerate(col_labels):
                if gt[i][j] > 0.5:
                    image[s].add(t)

        src = scenario_fds(data_dir, dataset, scenario, "source")
        tgt = set(scenario_fds(data_dir, dataset, scenario, "target"))

        matched = 0
        for determinant, dependent in src:
            if dependent not in image or any(c not in image for c in determinant):
                continue
            # every combination of counterparts, so a 1:n correspondence still counts
            options = [image[c] for c in determinant]
            candidates = [frozenset()]
            for opt in options:
                candidates = [c | {o} for c in candidates for o in opt]
            if any((cand, dep) in tgt for cand in candidates for dep in image[dependent]):
                matched += 1

        frac = matched / len(src) if src else float("nan")
        if src:
            fractions.append(frac)
            per_dataset[dataset].append(frac)
            total_src += len(src)
            total_matched += matched
        if len(sample) <= 12 or dataset == "DeNorm":
            print(f"{dataset:<18} {scenario[:26]:<26} {len(src):>6} {len(tgt):>6} {matched:>8} {frac:>6.3f}")

    print()
    for dataset in sorted(per_dataset):
        vals = per_dataset[dataset]
        print(f"{dataset:<18} scenarios={len(vals):<4} matched fraction: "
              f"mean {mean(vals):.3f}, min {min(vals):.3f}, max {max(vals):.3f}")
    if fractions:
        print(f"\noverall: {total_matched}/{total_src} source FDs have a target counterpart "
              f"({total_matched / total_src:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
