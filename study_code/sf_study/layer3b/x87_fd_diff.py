"""X87 profiler finding (EXPLORATORY; EXPLORATORY.md X87): is approximate FD discovery invariant under column order?
Compares, per scenario side, the FD set of the original import (madi-imported, hdxsm-usable) with the FD set HyFD found on the
column-permuted copy (harness/generated/x87-<c>-perm, profiled as the originals: threshold 0.95, max arity 2). FDs are
compared by name as (determinant set, dependent); pdep is the first value in "(pdep a, b)".

    python layer3b/x87_fd_diff.py   -> tuning/results/x87_scores/{fd_diff_sides.csv, fd_diff_summary.csv}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
GEN = STUDY.parent / "harness" / "generated"
OUT = STUDY / "tuning" / "results" / "x87_scores"
ORIG = {"madi": GEN / "madi-imported", "hdxsm": GEN / "hdxsm-usable"}
DS = {"madi": "MADI-Unionable", "hdxsm": "HDXSM-Unionable"}


def parse(p: Path) -> dict:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if " --> " not in line:
            continue
        lhs, rest = line.split(" --> ", 1)
        rhs, _, pd_ = rest.rpartition(" (pdep ")
        key = (frozenset(x.strip() for x in lhs.strip("[]").split(", ") if x.strip()), rhs.strip())
        out[key] = float(pd_.split(",")[0])
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cases = json.loads((STUDY / "data" / "x87_perm" / "cases.json").read_text(encoding="utf-8"))
    rows, pds = [], {c: [] for c in cases}
    for c, scs in cases.items():
        for sc in scs:
            for side in ("source", "target"):
                fo = ORIG[c] / DS[c] / sc / "metadata" / side / side / "FD_results.txt"
                fp = GEN / f"x87-{c}-perm" / DS[c] / sc / "metadata" / side / side / "FD_results.txt"
                if not (fo.is_file() and fp.is_file()):
                    rows.append({"corpus": c, "scenario": sc, "side": side, "missing": True})
                    continue
                a, b = parse(fo), parse(fp)
                oo, op = set(a) - set(b), set(b) - set(a)
                shared = set(a) & set(b)
                pds[c] += [a[k] for k in oo] + [b[k] for k in op]
                rows.append({"corpus": c, "scenario": sc, "side": side, "missing": False, "fds_original": len(a),
                             "fds_permuted": len(b), "only_original": len(oo), "only_permuted": len(op),
                             "max_pdep_change_shared": max((abs(a[k] - b[k]) for k in shared), default=0.0)})
    sides = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    sides.to_csv(OUT / "fd_diff_sides.csv", index=False)
    summ = []
    for c, g in sides[~sides.missing].groupby("corpus"):
        sym = int((g.only_original + g.only_permuted).sum())
        x = np.array(pds[c])
        summ.append({"corpus": c, "sides": len(g), "sides_differing": int(((g.only_original + g.only_permuted) > 0).sum()),
                     "fds_original": int(g.fds_original.sum()), "fds_differing": sym,
                     "share_differing": sym / int(g.fds_original.sum()),
                     "max_pdep_change_shared": float(g.max_pdep_change_shared.max()),
                     "pdep_differing_median": float(np.median(x)) if len(x) else np.nan,
                     "pdep_differing_share_below_096": float((x < 0.96).mean()) if len(x) else np.nan,
                     "pdep_differing_share_exact": float((x >= 1 - 1e-12).mean()) if len(x) else np.nan})
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "fd_diff_summary.csv", index=False)
    print(s.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
