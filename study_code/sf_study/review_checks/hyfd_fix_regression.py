"""HyFD partial-fix regression on real data (2026-10-26; the user's go to update code/HyFD to the `partial` branch, 27425fc,
and rerun). X87 found that approximate FD discovery (threshold 0.95, arity 2) was not invariant under column order: FD sets
differed between the original and the column-permuted copy on 16 of 80 HDXSM and 8 of 36 MaDI sides
(tuning/results/x87_scores/fd_diff_sides.csv). With the fixed build, original and permuted FD sets should be equal.

Copies the affected scenarios (original: harness/generated/{hdxsm-usable,madi-imported}; permuted: x87-<c>-perm) to a scratch
folder, re-profiles FDs there with the CURRENT code/HyFD build through harness/smx/profiling.py (threshold 0.95, max arity 2,
--materialize; the corpus folders themselves are not touched), and compares per side with layer3b/x87_fd_diff.parse
(FDs by name). It also compares the new FD sets with the OLD ones (old-build originals) to size the change.
Output: tuning/results/review_checks/hyfd_fix_regression.{csv,md}

    python review_checks/hyfd_fix_regression.py <scratch dir>
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

STUDY = Path(__file__).resolve().parent.parent
REPO = STUDY.parent
sys.path.insert(0, str(STUDY / "layer3b"))
from x87_fd_diff import parse  # noqa: E402

GEN = REPO / "harness" / "generated"
OUT = STUDY / "tuning" / "results" / "review_checks"
ORIG = {"madi": GEN / "madi-imported", "hdxsm": GEN / "hdxsm-usable"}
PERM = {"madi": GEN / "x87-madi-perm", "hdxsm": GEN / "x87-hdxsm-perm"}
DS = {"madi": "MADI-Unionable", "hdxsm": "HDXSM-Unionable"}
PY = REPO / "harness" / ".venv-isresmat" / "Scripts" / "python.exe"


def fd_file(scen: Path, side: str) -> Path | None:
    hits = sorted((scen / "metadata" / side).rglob("FD_results.txt"))
    return hits[0] if hits else None


def main() -> int:
    scratch = Path(sys.argv[1])
    sides = pd.read_csv(STUDY / "tuning" / "results" / "x87_scores" / "fd_diff_sides.csv")
    bad = sides[(sides.only_original > 0) | (sides.only_permuted > 0)]
    rows = []
    for c in ("madi", "hdxsm"):
        scen_names = sorted(set(bad[bad.corpus == c].scenario))
        for kind, src in (("orig", ORIG[c]), ("perm", PERM[c])):
            dst_root = scratch / f"{c}_{kind}"
            for sc in scen_names:
                s, d = src / DS[c] / sc, dst_root / DS[c] / sc
                if not d.exists():
                    shutil.copytree(s, d)
                old = {side: fd_file(s, side) for side in ("source", "target")}
                for side, f in old.items():
                    shutil.copyfile(f, scratch / f"old_{c}_{kind}_{sc}_{side}.txt") if f else None
            log = scratch / f"profile_{c}_{kind}.log"
            with log.open("w", encoding="utf-8") as fh:
                code = subprocess.run([str(PY), "-m", "smx.profiling", "--input", str(dst_root), "--cache",
                                       str(REPO / "harness" / "profile-cache"), "--kinds", "fd", "--threshold", "0.95",
                                       "--max-arity", "2", "--timeout", "7200", "--heap", "8g", "--materialize"],
                                      cwd=REPO / "harness", stdout=fh, stderr=subprocess.STDOUT).returncode
            print(f"{c} {kind}: profiler exit {code} ({log})", flush=True)
        for sc in scen_names:
            for side in ("source", "target"):
                fo = fd_file(scratch / f"{c}_orig" / DS[c] / sc, side)
                fp = fd_file(scratch / f"{c}_perm" / DS[c] / sc, side)
                old_o = scratch / f"old_{c}_orig_{sc}_{side}.txt"
                if fo is None or fp is None:
                    rows.append({"corpus": c, "scenario": sc, "side": side, "status": "missing"})
                    continue
                a, b = parse(fo), parse(fp)
                o = parse(old_o) if old_o.is_file() else {}
                rows.append({"corpus": c, "scenario": sc, "side": side, "status": "ok",
                             "new_orig": len(a), "new_perm": len(b), "new_only_orig": len(set(a) - set(b)),
                             "new_only_perm": len(set(b) - set(a)), "old_orig": len(o),
                             "old_vs_new_only_old": len(set(o) - set(a)), "old_vs_new_only_new": len(set(a) - set(o))})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "hyfd_fix_regression.csv", index=False)
    ok = df[df.status == "ok"]
    inv = ok[(ok.new_only_orig == 0) & (ok.new_only_perm == 0)]
    lines = ["# HyFD partial fix: column-order regression on the X87 sides (fixed build 27425fc)", "",
             f"- sides compared: {len(ok)} (missing: {int((df.status != 'ok').sum())})",
             f"- original and permuted FD sets equal on {len(inv)} of {len(ok)} sides (old build: equal on none of these)",
             f"- old vs new FD sets on the original tables: {int((ok.old_vs_new_only_old + ok.old_vs_new_only_new > 0).sum())} "
             f"of {len(ok)} sides change; FDs only in the old set {int(ok.old_vs_new_only_old.sum())}, only in the new set "
             f"{int(ok.old_vs_new_only_new.sum())} (of {int(ok.old_orig.sum())} old FDs)", "",
             df.to_markdown(index=False)]
    (OUT / "hyfd_fix_regression.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
