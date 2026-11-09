"""Dependency profiles for multi-table scenarios (2026-10-05; for X54 and the multi-table corpora).

The harness profiler (harness/smx/profiling.py) profiles one table per side. This wrapper profiles every table of a
multi-table scenario on its own, with the settings used for T2D and the controlled corpus (threshold 0.95, maximum
arity 2, heap 2g): FDs with HyFD and UCCs with HyUCC through `Profiler.profile` on a one-table manifest, exported with
the profiler's own `legacy_lines`, so the files read exactly like the single-table ones. Unary INDs are computed
here, exactly, across all tables of a side: A ⊆ B if the distinct non-null values of A are a subset of B's (A ≠ B,
A non-empty), written in the legacy grammar "[referenced] [= [dependent]".

Writes, per scenario and side: metadata/<side>/<table>/FD_results.txt, UCC_results.txt; metadata/<side>/inds.txt;
metadata/multitable_profile.json (settings, counts, the profile cache keys). Labels are "<table>.csv.<column>", which
explore_deproles.col maps to the matrices' "<table>.<column>".

    ../harness/.venv-bdi/Scripts/python.exe data/profile_multitable.py <corpus root> [--select GLOB] [--workers 4]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent.parent / "harness"
sys.path.insert(0, str(HARNESS))
from smx.profiling import Profiler, legacy_lines  # noqa: E402

THRESHOLD, ARITY, HEAP, TIMEOUT = 0.95, 2, "2g", 600
CSV_POLICY = {"encoding": "utf-8", "delimiter": ",", "quotechar": "\"", "escapechar": None, "null_token": "",
              "type_inference": False, "trim_header_whitespace": False}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p: Path) -> tuple[list[str], list[list[str]]]:
    with p.open(encoding="utf-8", newline="") as fh:
        r = csv.reader(fh)
        hdr = next(r)
        return hdr, [row for row in r]


def profile_table(table: Path, work: Path, cache: Path) -> dict:
    """FD and UCC lines for one table, labels qualified with the table name."""
    hdr, rows = read(table)
    d = work / table.stem
    (d / "t").mkdir(parents=True, exist_ok=True)
    copy = d / "t" / "table.csv"
    shutil.copyfile(table, copy)
    cols = [{"id": f"c{i:04d}", "index": i, "label": h, "original_label": h} for i, h in enumerate(hdr)]
    entry = {"path": "t/table.csv", "sha256": sha(copy), "semantics": "observations", "rows": len(rows),
             "columns": cols}
    manifest = d / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "csv_policy": CSV_POLICY,
                                    "tables": {"source": entry, "target": entry}}), encoding="utf-8")
    prof = Profiler(cache, repo=HARNESS.parent, timeout=TIMEOUT, heap=HEAP)
    out = {}
    for kind in ("fd", "ucc"):
        res = prof.profile(manifest, "source", kind, threshold=THRESHOLD, max_arity=ARITY)
        if res["status"] != "complete":
            out[kind] = {"status": res["status"], "lines": [], "reason": res.get("reason") or res.get("error")}
            continue
        art = json.loads(Path(res["path"]).read_text(encoding="utf-8"))
        lines, omitted = legacy_lines(art["records"], kind, table.stem, cols)
        out[kind] = {"status": "complete", "lines": lines, "omitted": omitted, "key": res["key"]}
    return out


def inds(side_dir: Path) -> list[str]:
    vals = {}
    for t in sorted(side_dir.glob("*.csv")):
        hdr, rows = read(t)
        for i, h in enumerate(hdr):
            vals[f"{t.stem}.csv.{h}"] = {r[i] for r in rows if i < len(r) and r[i] != ""}
    lines = []
    for dep, dv in vals.items():
        if not dv:
            continue
        for ref, rv in vals.items():
            if ref != dep and dv <= rv:
                lines.append(f"[{ref}] [= [{dep}]")
    return sorted(lines)


def profile_scenario(job) -> str:
    scen, cache = Path(job[0]), Path(job[1])
    meta = scen / "metadata"
    summary = {"settings": {"threshold": THRESHOLD, "max_arity": ARITY, "heap": HEAP, "ind": "exact unary, distinct"},
               "tables": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for side in ("source", "target"):
            for t in sorted((scen / side).glob("*.csv")):
                res = profile_table(t, Path(tmp) / side, cache)
                tdir = meta / side / t.stem
                tdir.mkdir(parents=True, exist_ok=True)
                for kind, fname in (("fd", "FD_results.txt"), ("ucc", "UCC_results.txt")):
                    lines = res[kind]["lines"]
                    (tdir / fname).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                summary["tables"][f"{side}/{t.stem}"] = {k: {kk: (len(vv) if kk == "lines" else vv)
                                                            for kk, vv in v.items()} for k, v in res.items()}
            il = inds(scen / side)
            (meta / side).mkdir(parents=True, exist_ok=True)
            (meta / side / "inds.txt").write_text("\n".join(il) + ("\n" if il else ""), encoding="utf-8")
            summary[f"{side}_inds"] = len(il)
    (meta / "multitable_profile.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    fds = sum(v["fd"]["lines"] for v in summary["tables"].values())
    return f"{scen.parent.name}/{scen.name}: {len(summary['tables'])} tables, {fds} FDs, " \
           f"{summary['source_inds']}+{summary['target_inds']} INDs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--select", default="*/*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", type=Path, default=HARNESS / "profile-cache")
    a = ap.parse_args()
    scens = sorted(p for p in a.root.glob(a.select) if (p / "source").is_dir() and (p / "target").is_dir())
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(profile_scenario, [(str(s), str(a.cache)) for s in scens]):
            print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
