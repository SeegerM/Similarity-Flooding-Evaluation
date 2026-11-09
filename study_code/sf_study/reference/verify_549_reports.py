"""Rerun 549 (2026-11-01): the report scripts refactored for the 549 run still write their 382 reports byte for byte.
Rebuilds each report from the stored 382 rows with the refactored report function (run WITHOUT SMX_VALENTINE) into a
temporary folder and compares it with the stored report:
  explore_iterations.report        iter_{plain,md5,indep}_scores/report.md (stored scores.csv + inversions.csv)
  explore_cluster_batches.report   {gvfull,fdnull}_{plain,md5}_scores/report.md
  layer3b/score_parallel.write_report  parallel_{plain,md5}_scores/{report.md, contrasts.csv}
explore_deproles.py's 382 path is unchanged apart from guarded 549 branches (checked by diff, not rerun: ~1 h).
Prints one line per report and exits 1 on any difference.

    ../harness/.venv-bdi/Scripts/python.exe -B reference/verify_549_reports.py
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import sys
import tempfile
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
if os.environ.get("SMX_VALENTINE", "382") != "382":
    raise SystemExit("run without SMX_VALENTINE")
for p in (STUDY / "layer3b", STUDY):
    sys.path.insert(0, str(p))
import explore_cluster_batches as cb  # noqa: E402
import explore_iterations as ei  # noqa: E402
import score_parallel as sp  # noqa: E402
from score_seeds import cases_for  # noqa: E402

RES = STUDY / "tuning" / "results"


def rows(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p, encoding="utf-8")))


def main() -> int:
    bad = 0
    quiet = contextlib.redirect_stdout(io.StringIO())
    for reg in ("plain", "md5", "indep"):
        d = RES / f"iter_{reg}_scores"
        inv = None
        if reg != "indep":
            inv = [{**r, "t": int(r["t"]), "inversions": int(r["inversions"]), "pairs": int(r["pairs"])}
                   for r in rows(d / "inversions.csv")]
        out = Path(tempfile.mkdtemp())
        rs = rows(d / "scores.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            ei.report(reg, rs, cases_for(reg), {r["seed"] for r in rs}, 0.0, inv, out)
        ok = (out / "report.md").read_bytes() == (d / "report.md").read_bytes()
        bad += not ok
        print(f"{'ok' if ok else 'DIFFERENT'}: explore_iterations report, {reg}")
    for batch in ("gvfull", "fdnull"):
        for reg in ("plain", "md5"):
            d = RES / f"{batch}_{reg}_scores"
            if not (d / "report.md").is_file():
                continue
            rs = rows(d / "scores.csv")
            out = Path(tempfile.mkdtemp())
            with contextlib.redirect_stdout(io.StringIO()):
                cb.report(batch, reg, rs, rs + cb.reference_rows(reg), cases_for(reg), out)
            ok = (out / "report.md").read_bytes() == (d / "report.md").read_bytes()
            bad += not ok
            print(f"{'ok' if ok else 'DIFFERENT'}: explore_cluster_batches report, {batch} {reg}")
    for reg in ("plain", "md5"):
        d = RES / f"parallel_{reg}_scores"
        rs = [{**r, **{k: float(r[k]) for k in ("sf_F1_Hr", "sf_R_exp", "sf_J")}} for r in rows(d / "scores.csv")]
        checks = [x[2:] for x in (d / "report.md").read_text(encoding="utf-8").splitlines()[2:] if x.startswith("- ")]
        out = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            sp.write_report(reg, rs, {(a, b): f for a, b, f in cases_for(reg)}, checks, out)
        ok = all((out / f).read_bytes() == (d / f).read_bytes() for f in ("report.md", "contrasts.csv"))
        bad += not ok
        print(f"{'ok' if ok else 'DIFFERENT'}: score_parallel report + contrasts, {reg}")
    import pandas as pd
    import score_x93 as s93
    import x93_mechanism as xm
    out = Path(tempfile.mkdtemp())
    with contextlib.redirect_stdout(io.StringIO()):
        xm.report(pd.read_csv(s93.OUT / "mechanism.csv"), pd.read_csv(s93.OUT / "rows.csv"), out)
    ok = (out / "mechanism.md").read_bytes() == (s93.OUT / "mechanism.md").read_bytes()
    bad += not ok
    print(f"{'ok' if ok else 'DIFFERENT'}: x93_mechanism report")
    # X95's report, X81's property profile and X73's summaries (refactored 2026-11-02)
    import json as _json
    import explore_matcherroles as mr
    import explore_x81_props as xp
    import score_x95 as x95
    wj = _json.loads((x95.OUT / "weights.json").read_text(encoding="utf-8"))
    w = {f: {k: wj[f][k] for k in ("D", "R", "V")} for f in ("i", "ii", "iii")}
    checks = []
    out = Path(tempfile.mkdtemp())
    with contextlib.redirect_stdout(io.StringIO()):
        x95.write_report(pd.read_csv(x95.OUT / "contrasts.csv"), w, out)
    checks.append(("score_x95 report", out / "report.md", x95.OUT / "report.md"))
    out = Path(tempfile.mkdtemp())
    old = [r for r in rows(STUDY / "results" / "benchmark_properties.csv") if r["corpus"] in xp.OLD]
    with contextlib.redirect_stdout(io.StringIO()):
        xp.profile(old, rows(STUDY / "results" / "x81" / "benchmark_properties_x81.csv"), out)
    checks.append(("explore_x81_props profile", out / "benchmark_profile_x81.md", STUDY / "results" / "x81" / "benchmark_profile_x81.md"))
    for reg in ("plain", "md5"):
        rs = rows(STUDY / "results" / reg / "matcherroles_scores.csv")
        for r in rs:
            for k in [k for k in r if k not in ("init", "dataset", "scenario", "family", "measure")]:
                if r[k] == "":
                    del r[k]
                else:
                    r[k] = float(r[k])
        out = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            mr.summarise(reg, rs, cases_for(reg), mr.measures(reg), out, mr.INITS[reg])
        checks.append((f"explore_matcherroles summary, {reg}", out / "matcherroles_summary.md",
                       STUDY / "results" / reg / "matcherroles_summary.md"))
    for label, a, b in checks:
        ok = a.read_bytes() == b.read_bytes()
        bad += not ok
        print(f"{'ok' if ok else 'DIFFERENT'}: {label}")
    del quiet
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
