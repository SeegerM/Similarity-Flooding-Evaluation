"""Rerun 549 (2026-11-02): the ground-truth part of the positional survey (harness/reports/positional_alignment.json,
"totals" and per-scenario n_ground_truth / n_off_diagonal) on the 549 scenarios. A true pair is on the diagonal when
source column i is matched to target column i (matrix index order, as the survey counts it).

First recomputes the survey on the 382 cases and requires every scenario's counts to equal the stored survey; then writes
results/valentine549/positional_survey.json (549 scenarios: the 381 old ones, the Joinable copy of the duplicate dropped,
and the 168 new ones). The constant-matrix and matcher-subsidy parts need matcher reruns and are not computed here.

    ../harness/.venv-bdi/Scripts/python.exe -B reference/positional_survey_549.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent
for p in (ROOT / "harness", STUDY, HERE):
    sys.path.insert(0, str(p))
import valentine549 as v  # noqa: E402
from score_seeds import CORPUS, ground_truth_file  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

PROTOCOL = ROOT / "harness" / "reports" / "audited_md5_fused_20260904" / "protocol.json"
SURVEY = ROOT / "harness" / "reports" / "positional_alignment.json"


def counts(root: Path, ds: str, sc: str) -> tuple[int, int]:
    gt = read_matrix(ground_truth_file(root, ds, sc))[0] > 0.5
    i, j = np.nonzero(gt)
    return int(gt.sum()), int((i != j).sum())


def main() -> int:
    stored = {s["id"]: s for s in json.loads(SURVEY.read_text(encoding="utf-8"))["scenarios"]}
    bad = 0
    for c in json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]:
        n, off = counts(CORPUS["plain"], c["dataset"], c["scenario"])
        s = stored[c["id"]]
        bad += (n, off) != (s["n_ground_truth"], s["n_off_diagonal"])
    print(f"382 survey recomputed: {len(stored)} scenarios, {bad} differ from positional_alignment.json")
    if bad:
        raise SystemExit("method does not reproduce the stored survey; nothing written")
    rows = []
    for ds, sc, fam in v.cases549():
        n, off = counts(v.NEW_TREE["plain"] if v.is_new(ds, sc) else CORPUS["plain"], ds, sc)
        rows.append({"dataset": ds, "scenario": sc, "family": fam, "n_ground_truth": n, "n_off_diagonal": off})
    tot = {"scenarios": len(rows), "pairs": sum(r["n_ground_truth"] for r in rows),
           "off_diagonal": sum(r["n_off_diagonal"] for r in rows)}
    out = STUDY / "results" / "valentine549" / "positional_survey.json"
    out.write_text(json.dumps({"totals": tot, "scenarios": rows, "check_382": "reproduced exactly"}, indent=1),
                   encoding="utf-8")
    print(tot, "->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
