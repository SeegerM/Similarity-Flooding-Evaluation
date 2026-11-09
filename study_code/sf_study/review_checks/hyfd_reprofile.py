"""Stage 1 of the HyFD-fix rerun (2026-10-26; the user's go: "update our version and then rerun"): re-profile the FDs of every
corpus with the fixed HyFD (code/HyFD at the `partial` branch, 27425fc) at the paper's settings (threshold 0.95, max arity 2)
and measure how much the FD sets change. The live corpora are NOT modified.

Each corpus is mirrored under harness/generated_hyfdfix/<name>/: table CSVs are HARD LINKS (only ever read), everything the
profiler writes (manifest.json, metadata/, ground_truth/) is a real copy. Single-table imported corpora are profiled with
harness/smx/profiling.py (--kinds fd --materialize); multi-table corpora with data/profile_multitable.py (--workers 1; UCC and
IND come from the cache, unchanged builds). Then old vs new FD files are compared per side (FDs by name).

    python review_checks/hyfd_reprofile.py mirror  <name>
    python review_checks/hyfd_reprofile.py profile <name>
    python review_checks/hyfd_reprofile.py compare [<name> ...]   -> tuning/results/review_checks/hyfd_reprofile_<name>.csv, hyfd_reprofile.md
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

STUDY = Path(__file__).resolve().parent.parent
REPO = STUDY.parent
GEN = REPO / "harness" / "generated"
MIRROR = REPO / "harness" / "generated_hyfdfix"
OUT = STUDY / "tuning" / "results" / "review_checks"
PY = REPO / "harness" / ".venv-isresmat" / "Scripts" / "python.exe"
SINGLE = {"valentine_plain": GEN / "benchmark-full", "valentine_md5": GEN / "benchmark-full-md5",
          "ctl": GEN / "controlled-imported-v3", "final_controlled": GEN / "final-controlled", "goby": GEN / "goby-imported",
          "hdxsm": GEN / "hdxsm-usable", "gdcsm": GEN / "gdcsm-imported", "madi": GEN / "madi-imported",
          "t2d": GEN / "t2d-usable"}
MULTI = {"synthea": STUDY / "data" / "synthea_plain", "indep2": STUDY / "data" / "indep2_plain",
         "final_indep": STUDY / "data" / "final_indep", "mimic": STUDY / "data" / "mimic_omop",
         "indep_prof": STUDY / "data" / "indep_prof", "test": STUDY / "data" / "test_plain"}
TABLE_DIRS = {"source", "target"}


def src_of(name: str) -> Path:
    return SINGLE.get(name) or MULTI[name]


def mirror(name: str) -> None:
    src, dst = src_of(name), MIRROR / name
    if dst.exists():
        raise SystemExit(f"{dst} exists; remove it first")
    n_link = n_copy = 0
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(src)
        q = dst / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        if rel.parts and any(part in TABLE_DIRS for part in rel.parts[:-1]) and "metadata" not in rel.parts \
                and p.suffix.lower() in (".csv", ".tsv"):
            os.link(p, q)
            n_link += 1
        else:
            shutil.copy2(p, q)
            n_copy += 1
    print(f"{name}: mirrored to {dst} ({n_link} tables hard-linked, {n_copy} files copied)")


def profile(name: str) -> int:
    dst = MIRROR / name
    log = OUT / f"hyfd_reprofile_{name}.log"
    OUT.mkdir(parents=True, exist_ok=True)
    if name == "valentine_md5":
        # The recoded tree's metadata is by design the plaintext metadata with labels renamed (make_md5_audited.py);
        # its UCC/IND/DC entries point at plaintext profiles, so smx.profiling --materialize refuses it ("Profile does not
        # belong to this exported table"). Relabel the new plaintext FDs instead, as the tree was built. Profiling the
        # recoded tables directly gives the same FD count on 773 of 780 sides (7 differ by <=0.07%; 2026-10-26 log).
        # Run after valentine_plain. Writes only metadata/ and manifest.json (real copies, not hard links).
        cmd = [str(PY), "make_md5_audited.py", "--repair-metadata", "--data", str(MIRROR / "valentine_plain"),
               "--out", str(dst)]
        cwd = REPO / "harness"
    elif name in SINGLE:
        cmd = [str(PY), "-m", "smx.profiling", "--input", str(dst), "--cache", str(REPO / "harness" / "profile-cache"),
               "--kinds", "fd", "--threshold", "0.95", "--max-arity", "2", "--timeout", "7200", "--heap", "8g", "--materialize"]
        cwd = REPO / "harness"
    else:
        cmd = [str(PY), str(STUDY / "data" / "profile_multitable.py"), str(dst), "--workers", "1"]
        cwd = STUDY
    with log.open("w", encoding="utf-8") as fh:
        code = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
    print(f"{name}: profiling exit {code} ({log})")
    return code


def fds(p: Path) -> set:
    out = set()
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if " --> " in line:
            lhs, rhs = line.split(" --> ", 1)
            rhs = rhs.split(" (pdep")[0].strip()
            out.add((frozenset(x.strip() for x in lhs.strip().strip("[]").split(",") if x.strip()), rhs))
    return out


def compare(names: list[str]) -> int:
    summary = []
    for name in names:
        src, dst = src_of(name), MIRROR / name
        rows = []
        for new in sorted(dst.rglob("FD_results.txt")):
            rel = new.relative_to(dst)
            old = src / rel
            a, b = fds(old), fds(new)
            rows.append({"file": str(rel), "old": len(a), "new": len(b), "only_old": len(a - b), "only_new": len(b - a)})
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"hyfd_reprofile_{name}.csv", index=False)
        if df.empty:
            summary.append(f"| {name} | 0 | | | | | |")
            continue
        ch = df[(df.only_old + df.only_new) > 0]
        summary.append(f"| {name} | {len(df)} | {len(ch)} ({len(ch) / len(df):.0%}) | {int(df.old.sum())} | {int(df.new.sum())} | "
                       f"{int(df.only_old.sum())} | {int(df.only_new.sum())} |")
    lines = ["# HyFD fix: old vs new FD sets per corpus (threshold 0.95, max arity 2)", "",
             "| corpus | FD files | files that change | FDs old | FDs new | only old | only new |", "|---|---|---|---|---|---|---|",
             *summary]
    path = OUT / "hyfd_reprofile.md"
    prev = path.read_text(encoding="utf-8").splitlines()[4:] if path.is_file() else []
    keep = [l for l in prev if l.startswith("| ") and l.split(" | ")[0][2:] not in names]
    path.write_text("\n".join(lines[:4] + keep + summary) + "\n", encoding="utf-8")
    print("\n".join(lines[:4] + keep + summary))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "compare":
        raise SystemExit(compare(sys.argv[2:] or list(SINGLE) + list(MULTI)))
    {"mirror": mirror, "profile": profile}[cmd](sys.argv[2])
