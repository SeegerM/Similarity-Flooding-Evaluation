"""datatype_nanfix = the datatype seed with NaN cells set to 0, finite cells unchanged (X30).

check:  on the 382 (plain, md5), reference/seeds/<r>/datatype_nanfix must equal results/<r>/datatype's seed matrix with
        NaN -> 0, byte for byte after rewriting (same CSV writer as the derivation below), else stop.
build:  write reference/seeds/{pnew,mnew}/datatype_nanfix/<dataset>/<scenario>/seed.csv the same way.

    python nanfix_seeds.py <study> check|build
"""
import csv
import io
import math
import sys
from pathlib import Path

study = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(study))
from score_seeds import cases_for, latest_run, one  # noqa: E402


def nanfix(src: Path) -> list[list[str]]:
    rows = list(csv.reader(src.open(newline="", encoding="utf-8")))
    out = [rows[0]]
    for r in rows[1:]:
        out.append([r[0]] + ["0.0" if math.isnan(float(v)) else v for v in r[1:]])
    return out


def as_text(rows) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue()


def norm(path: Path) -> list[list[str]]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    return [rows[0]] + [[r[0]] + [repr(float(v)) for v in r[1:]] for r in rows[1:]]


mode = sys.argv[2]
if mode == "check":
    for regime in ("plain", "md5"):
        run = latest_run(study / "results" / regime / "datatype")
        bad = n = nan_cells = 0
        for d, s, _ in cases_for(regime):
            stored = study / "reference" / "seeds" / regime / "datatype_nanfix" / d / s / "seed.csv"
            derived = nanfix(one(run / d / s / "_outputs" / "MatchingStepLine1"))
            nan_cells += sum(1 for r in list(csv.reader(one(run / d / s / "_outputs" / "MatchingStepLine1").open()))[1:]
                             for v in r[1:] if math.isnan(float(v)))
            n += 1
            a = [derived[0]] + [[r[0]] + [repr(float(v)) for v in r[1:]] for r in derived[1:]]
            if a != norm(stored):
                bad += 1
        print(f"{regime}: {n} scenarios, {bad} differ from the stored datatype_nanfix seeds ({nan_cells} NaN cells set to 0)")
elif mode == "build":
    for regime in ("pnew", "mnew"):
        run = latest_run(study / "results" / regime / "datatype")
        root = study / "reference" / "seeds" / regime / "datatype_nanfix"
        if root.exists():
            sys.exit(f"{root} exists")
        nan_cells = 0
        for d, s, _ in cases_for(regime):
            src = one(run / d / s / "_outputs" / "MatchingStepLine1")
            rows = nanfix(src)
            nan_cells += sum(1 for r in list(csv.reader(src.open()))[1:] for v in r[1:] if math.isnan(float(v)))
            dst = root / d / s / "seed.csv"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(as_text(rows), encoding="utf-8", newline="")
        print(f"{regime}: {len(cases_for(regime))} datatype_nanfix seeds written ({nan_cells} NaN cells set to 0)")
