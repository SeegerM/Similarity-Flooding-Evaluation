"""Loaders that turn each matcher's native output into a ScoreMatrix.

The canonical interchange artifact is a dense numeric matrix over
(source columns x target columns), comma separated, one row per source
column.  schematch already emits exactly this from OutputWriter.writeSimMatrix,
in two variants:

  plain    - numbers only (what gets cached)
  verbose  - a "Source\\Target" header row plus a row-label first column;
             cells may be either a soft score or ``binary / soft-score``

Ground truth uses the same shape with 0/1 entries and may have column labels.
Both variants are auto-detected, so a caller does not need to know which
schematch configuration produced a file.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import ScoreMatrix


def _parse_numeric(token: str) -> float:
    """Parse a plain number or schematch's ``binary / similarity`` cell."""
    value = token.strip()
    if "/" in value:
        parts = value.split("/")
        if len(parts) != 2:
            raise ValueError(f"not a numeric matrix cell: {token!r}")
        # Validate the selected binary match as well, but return the underlying
        # soft score.  Selection is deliberately redone by this harness.
        float(parts[0].strip())
        value = parts[1].strip()
    return float(value)


def _looks_numeric(token: str) -> bool:
    try:
        _parse_numeric(token)
    except ValueError:
        return False
    return True


def read_matrix(
    path: Path, delimiter: str = ","
) -> Tuple[np.ndarray, Optional[List[str]], Optional[List[str]]]:
    """Read a dense matrix, auto-detecting schematch's verbose header/labels.

    Returns (matrix, row_labels, col_labels); labels are None for the plain
    variant.  Ragged rows are rejected loudly rather than silently padded,
    because a truncated matrix would quietly corrupt every downstream metric.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh, delimiter=delimiter) if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path} is empty")

    col_labels: Optional[List[str]] = None
    # A header row is present iff its first cell is not a number.
    if not _looks_numeric(rows[0][0].strip()):
        col_labels = [c.strip() for c in rows[0][1:]]
        rows = rows[1:]
        if not rows:
            raise ValueError(f"{path} has a header but no data rows")

    row_labels: Optional[List[str]] = None
    # A label column is present iff the first cell of a data row is not a number.
    if not _looks_numeric(rows[0][0].strip()):
        row_labels = [r[0].strip() for r in rows]
        rows = [r[1:] for r in rows]

    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{path} has ragged rows (widths {sorted(widths)})")

    matrix = np.array([[_parse_numeric(c) for c in r] for r in rows], dtype=float)
    return matrix, row_labels, col_labels


def load_pair(
    scores_path: Path,
    ground_truth_path: Path,
    meta: Optional[Dict[str, str]] = None,
    delimiter: str = ",",
) -> ScoreMatrix:
    """Load a score matrix together with its ground truth."""
    scores, row_labels, col_labels = read_matrix(scores_path, delimiter)
    gt, gt_rows, gt_cols = read_matrix(ground_truth_path, delimiter)
    if scores.shape != gt.shape:
        raise ValueError(
            f"shape mismatch: scores {scores.shape} from {scores_path.name} vs "
            f"ground truth {gt.shape} from {ground_truth_path.name}"
        )
    if not np.isfinite(scores).all() or not np.isfinite(gt).all():
        raise ValueError("Score/ground-truth matrix contains non-finite values")
    if not np.isin(gt, [0, 1]).all():
        raise ValueError("Ground truth must be binary")
    for labels, size in ((row_labels,scores.shape[0]),(col_labels,scores.shape[1]),
                         (gt_rows,gt.shape[0]),(gt_cols,gt.shape[1])):
        if labels is not None and (len(labels) != size or len(set(labels)) != len(labels)):
            raise ValueError("Missing, duplicate, or mis-sized matrix labels")
    def order(wanted, available):
        if wanted is None or available is None:
            return None  # Unlabeled files carry only a positional-order contract.
        if set(wanted) == set(available):
            lookup = {v:i for i,v in enumerate(available)}
            return [lookup[v] for v in wanted]
        # Only strip an unambiguous table suffix, never '.csv.' inside a column name.
        normal = lambda value: re.sub(r"^([^.]+)\.csv\.", r"\1.", value)
        wanted, available = list(map(normal,wanted)), list(map(normal,available))
        if len(set(wanted)) != len(wanted) or len(set(available)) != len(available) or set(wanted) != set(available):
            raise ValueError("Score and ground-truth labels do not identify the same columns")
        lookup = {v:i for i,v in enumerate(available)}
        return [lookup[v] for v in wanted]
    row_order, col_order = order(row_labels,gt_rows), order(col_labels,gt_cols)
    if row_order is not None:
        gt = gt[row_order,:]
    if col_order is not None:
        gt = gt[:,col_order]
    return ScoreMatrix(
        scores=scores,
        ground_truth=gt > 0.5,
        source_labels=row_labels,
        target_labels=col_labels,
        meta=dict(meta or {}),
    )


# --------------------------------------------------------------------------
# schematch
# --------------------------------------------------------------------------


def find_schematch_ground_truth(scenario_dir: Path) -> Optional[Path]:
    """Prefer full scenario truth; never mistake one table-pair block for a multi-table matrix."""
    full = scenario_dir / "ground_truth_full.csv"
    if full.is_file():
        return full
    gt_dir = scenario_dir / "ground_truth"
    if not gt_dir.is_dir():
        return None
    candidates = sorted(gt_dir.glob("*.csv"))
    if len(candidates) > 1:
        raise ValueError(f"Multi-table scenario has no ground_truth_full.csv: {scenario_dir}; assemble blocks through Schematch first")
    return candidates[0] if candidates else None


def discover_schematch_runs(
    output_dir: Path,
    data_dir: Path,
    dataset: Optional[str] = None,
    system: str = "schematch",
) -> List[Dict[str, object]]:
    """Pair every schematch sim-matrix CSV with the ground truth of its scenario.

    schematch writes results under a path that ends with
    <dataset>/<scenario>/.../<matcherInfo>.csv, while ground truth lives at
    <data_dir>/<dataset>/<scenario>/ground_truth/*.csv.  We recover the
    (dataset, scenario) key by looking for the deepest path prefix that
    exists under data_dir, which keeps this robust to the intermediate
    directories schematch inserts for line/matcher/config.
    """
    runs: List[Dict[str, object]] = []
    if not output_dir.is_dir():
        return runs

    for csv_path in sorted(output_dir.rglob("*.csv")):
        parts = csv_path.relative_to(output_dir).parts
        # Similarity matrices are written below _outputs.  The sibling
        # _performances tree also contains CSVs, but those are scalar reports
        # and per-pair diagnostics rather than matcher score matrices.
        if "_outputs" not in parts:
            continue
        scenario_dir = None
        key: Tuple[str, str] = ("", "")
        # walk prefixes to find <dataset>/<scenario> that exists under data_dir
        for i in range(len(parts) - 1):
            candidate = data_dir / parts[i] / parts[i + 1]
            if (candidate / "ground_truth").is_dir() or (candidate / "ground_truth_full.csv").is_file():
                scenario_dir = candidate
                key = (parts[i], parts[i + 1])
                break
        if scenario_dir is None:
            continue
        if dataset and key[0] != dataset:
            continue
        gt_path = find_schematch_ground_truth(scenario_dir)
        if gt_path is None:
            continue
        runs.append(
            {
                "scores_path": csv_path,
                "ground_truth_path": gt_path,
                "meta": {
                    "system": system,
                    "dataset": key[0],
                    "scenario": key[1],
                    "config": csv_path.stem,
                },
            }
        )
    return runs


# --------------------------------------------------------------------------
# PDED
# --------------------------------------------------------------------------


def load_pded_matrix(
    scores_path: Path,
    ground_truth_path: Path,
    meta: Optional[Dict[str, str]] = None,
) -> ScoreMatrix:
    """Load a PDED soft-assignment matrix M dumped before discretisation.

    ``PdedSchematchRunner`` writes the canonical layout and restores the
    original source-by-target orientation if the matcher swapped its graphs.
    This named entry point remains the place for any future PDED quirks.
    """
    base = {"system": "pded"}
    base.update(meta or {})
    return load_pair(scores_path, ground_truth_path, meta=base)


def discover_pded_runs(
    output_dir: Path,
    data_dir: Path,
    dataset: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Pair PDED runner matrices with schematch ground truth by scenario."""
    runs: List[Dict[str, object]] = []
    if not output_dir.is_dir():
        return runs

    for csv_path in sorted(output_dir.rglob("pded.csv")):
        parts = csv_path.relative_to(output_dir).parts
        for i in range(len(parts) - 1):
            candidate = data_dir / parts[i] / parts[i + 1]
            if not (candidate / "ground_truth").is_dir() and not (candidate / "ground_truth_full.csv").is_file():
                continue
            if dataset and parts[i] != dataset:
                break
            gt_path = find_schematch_ground_truth(candidate)
            if gt_path is not None:
                runs.append(
                    {
                        "scores_path": csv_path,
                        "ground_truth_path": gt_path,
                        "meta": {
                            "system": "pded",
                            "dataset": parts[i],
                            "scenario": parts[i + 1],
                            "config": csv_path.stem,
                        },
                    }
                )
            break
    return runs


def discover_sima_runs(
    matrices_dir: Path,
    data_dir: Path,
    dataset: Optional[str] = None,
    config: str = "sima",
) -> List[Dict[str, object]]:
    """Pair SiMa's exported per-scenario matrices with the benchmark ground truth.

    ``sima_run.py --export-matrices`` writes ``<out>/matrices/<family>/<scenario>.csv``
    over the source and target columns of that scenario, in manifest order, so a
    matrix pairs with ``<data_dir>/<family>/<scenario>/ground_truth/*.csv``.
    """
    runs: List[Dict[str, object]] = []
    if not matrices_dir.is_dir():
        return runs

    for csv_path in sorted(matrices_dir.glob("*/*.csv")):
        family = csv_path.parent.name
        scenario = csv_path.stem
        if dataset and family != dataset:
            continue
        gt_path = find_schematch_ground_truth(data_dir / family / scenario)
        if gt_path is None:
            continue
        runs.append(
            {
                "scores_path": csv_path,
                "ground_truth_path": gt_path,
                "meta": {
                    "system": "sima",
                    "dataset": family,
                    "scenario": scenario,
                    "config": config,
                },
            }
        )
    return runs


def load_runs(specs: Sequence[Dict[str, object]]) -> List[ScoreMatrix]:
    """Materialise a list of run specs (as produced by the discover_* helpers)."""
    out: List[ScoreMatrix] = []
    for spec in specs:
        out.append(
            load_pair(
                Path(spec["scores_path"]),
                Path(spec["ground_truth_path"]),
                meta=dict(spec.get("meta") or {}),
            )
        )
    return out
