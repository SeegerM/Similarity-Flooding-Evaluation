"""TODO item D, second half: fail-closed checks around matcher output.

FINDINGS 71 is the worked example. A crashed PRISMA service produced a complete
set of all-zero score matrices. Nothing raised. The files were well-formed CSV
of the right shape, the reader accepted them, the scorer ranked them, and every
cell scored a clean `0.000` that looked like a real negative result. The defect
was found only because a table cell was implausible, not because anything
checked.

That is the silent-null defect family: a fault that produces plausible numbers
with no error. The rule that catches it -- refuse to score a matrix carrying no
information -- already exists in `regime_tables.py`. It belongs in the readers,
because every consumer needs it, not only the one that happened to be written
after the bug.

Two entry points, deliberately separated:

  `health(matrix)`         reports, never raises. Used to audit artifacts that
                           already exist, which must not be modified or
                           re-judged retroactively.
  `checked_matrix(path)`   raises `DegenerateMatrix`. Used by new code at the
                           point of reading, so a crashed matcher stops the run
                           instead of publishing a number.

What counts as degenerate is deliberately narrow -- only conditions under which
no ranking exists at all, never a quality judgement:

  all_zero        no cell carries any score. A matcher that ran cannot produce
                  this; a matcher that crashed routinely does.
  constant        every cell identical and non-zero. The ranking is entirely
                  ties, so any extractor's output is an artifact of tie order.
  non_finite      any NaN or infinity. These propagate silently through argmax
                  and through Hungarian assignment.
  shape_mismatch  only when an expected shape is supplied.

Deliberately *not* degenerate, because these are real matcher behaviour and
rejecting them would discard valid negative results:

  empty rows or columns (a source column with no candidate)
  very low scores, or a low-rank matrix
  a matrix whose argmax is wrong everywhere

Usage:
    python matrix_health.py --audit reports          # scan existing artifacts
    python matrix_health.py --audit reports --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from smx.adapters import read_matrix  # noqa: E402

# Score matrices live at `<run>/cases/<key>/<matcher>.csv`. Everything else in
# `reports/` is a results table -- long-format CSV with string columns -- and
# judging those as matrices produced ~18,900 spurious parse failures on the
# first pass, which is its own kind of silent-null defect: a check so noisy
# nobody reads it is a check that is not running.
SKIP_PARTS = ("/ground_truth/", "/source/", "/target/")
SKIP_NAMES = {"cases.csv", "metrics.csv", "summary.csv", "manifest.json",
              "protocol.json"}


class DegenerateMatrix(ValueError):
    """A matrix that carries no ranking information."""


def health(matrix: np.ndarray, expect_shape: tuple | None = None) -> dict:
    """Describe a matrix. Never raises; used to audit existing artifacts."""
    finite = np.isfinite(matrix)
    report = {
        "shape": list(matrix.shape),
        "non_finite": int((~finite).sum()),
        "all_zero": bool(finite.all() and not matrix.any()),
        "constant": False,
        "shape_mismatch": bool(expect_shape is not None
                               and tuple(matrix.shape) != tuple(expect_shape)),
        "empty_rows": 0,
        "empty_cols": 0,
        "distinct_values": 0,
    }
    if matrix.size and finite.any():
        values = matrix[finite]
        report["distinct_values"] = int(np.unique(values).size)
        report["constant"] = bool(report["distinct_values"] == 1
                                  and not report["all_zero"])
        report["min"] = float(values.min())
        report["max"] = float(values.max())
        if matrix.ndim == 2:
            report["empty_rows"] = int((~matrix.any(axis=1)).sum())
            report["empty_cols"] = int((~matrix.any(axis=0)).sum())
    report["degenerate"] = bool(report["all_zero"] or report["constant"]
                                or report["non_finite"]
                                or report["shape_mismatch"])
    return report


def reasons(report: dict) -> list[str]:
    out = []
    if report["all_zero"]:
        out.append("all cells are zero (a crashed matcher produces exactly this)")
    if report["constant"]:
        out.append(f"every cell is {report.get('max')} -- the ranking is all ties")
    if report["non_finite"]:
        out.append(f"{report['non_finite']} non-finite cells")
    if report["shape_mismatch"]:
        out.append(f"shape {report['shape']} does not match the ground truth")
    return out


def checked_matrix(path: Path, expect_shape: tuple | None = None) -> np.ndarray:
    """Read a score matrix, refusing to return one that cannot be ranked.

    Fail-closed: new code should read matrices through this, so that a matcher
    which crashed stops the run rather than publishing a clean-looking 0.000.
    """
    matrix = read_matrix(Path(path))[0]
    report = health(matrix, expect_shape)
    if report["degenerate"]:
        raise DegenerateMatrix(f"{path}: " + "; ".join(reasons(report)))
    return matrix


def is_candidate(path: Path) -> bool:
    """A score matrix, not a results table.

    Positive identification by location rather than exclusion by name: the
    matrices this project scores are all written under a `cases/<key>/`
    directory. An exclusion list can only ever be as complete as the last time
    someone added a report, which is not a property to rely on.
    """
    if path.suffix.lower() != ".csv" or path.name in SKIP_NAMES:
        return False
    text = str(path).replace("\\", "/")
    if any(part in text for part in SKIP_PARTS):
        return False
    return "/cases/" in text


def audit(root: Path) -> tuple[list[dict], Counter]:
    findings, tally = [], Counter()
    for path in sorted(root.rglob("*.csv")):
        if not is_candidate(path):
            tally["skipped"] += 1
            continue
        try:
            matrix = read_matrix(path)[0]
        except (ValueError, OSError, UnicodeDecodeError) as exc:
            tally["unreadable"] += 1
            findings.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        tally["read"] += 1
        report = health(matrix)
        if report["degenerate"]:
            tally["degenerate"] += 1
            findings.append({"path": str(path), **report,
                             "reasons": reasons(report)})
    return findings, tally


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="reports", help="directory tree to scan")
    ap.add_argument("--json", default=None, help="write the findings here")
    args = ap.parse_args()

    root = (HERE / args.audit) if not Path(args.audit).is_absolute() else Path(args.audit)
    if not root.is_dir():
        print(f"no such directory: {root}")
        return 1

    findings, tally = audit(root)
    print(f"scanned {root}")
    print(f"  matrices read   {tally['read']}")
    print(f"  skipped         {tally['skipped']} (not score matrices)")
    print(f"  unreadable      {tally['unreadable']}")
    print(f"  DEGENERATE      {tally['degenerate']}")
    for f in findings[:40]:
        rel = Path(f["path"]).relative_to(root)
        why = "; ".join(f.get("reasons", [])) or f.get("error", "")
        print(f"    {rel}\n      {why}")
    if len(findings) > 40:
        print(f"    ... and {len(findings) - 40} more")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(findings, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    # Auditing existing artifacts reports; it does not fail the build, because
    # old artifacts are immutable and a historical degenerate matrix is a
    # finding to record, not a file to change.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
