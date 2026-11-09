"""T2D-SM-WH (harness/generated/t2d-sm-wh, a custom port without manifests; harness/SF_T2D.md) back into the raw
Valentine layout, so that the harness pipeline (smx.datasets import, then smx.profiling with the controlled corpus's
parameters) can import and profile it. For the EXPLORATORY T2D replication.

    <out>/Valentine-datasets/T2D/<scenario>/<scenario>_{source,target}.csv   (tables copied byte for byte)
    <out>/Valentine-datasets/T2D/<scenario>/<scenario>_mapping.json          ({"matches": [...]} from the ground-truth
                                                                              matrix: row i = source column i, column j
                                                                              = target column j, cell 1 = a match)

    python data/convert_t2d_raw.py <out root>
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "harness" / "generated" / "t2d-sm-wh" / "T2D-SM-WH"


def header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def main() -> int:
    out = Path(sys.argv[1]) / "Valentine-datasets" / "T2D"
    n = m = 0
    for scen in sorted(p for p in SRC.iterdir() if p.is_dir()):
        s_tab, t_tab = scen / "source" / "source.csv", scen / "target" / "target.csv"
        gt = scen / "ground_truth" / "source___target.csv"
        sh, th = header(s_tab), header(t_tab)
        with gt.open(encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.reader(fh) if r]
        assert len(rows) == len(sh) and all(len(r) == len(th) for r in rows), scen.name
        stem = scen.name
        d = out / stem
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s_tab, d / f"{stem}_source.csv")
        shutil.copyfile(t_tab, d / f"{stem}_target.csv")
        matches = [{"source_table": f"{stem}_source", "source_column": sh[i],
                    "target_table": f"{stem}_target", "target_column": th[j]}
                   for i, r in enumerate(rows) for j, v in enumerate(r) if float(v) > 0.5]
        (d / f"{stem}_mapping.json").write_text(json.dumps({"matches": matches}, indent=2), encoding="utf-8")
        n += 1
        m += len(matches)
    print(f"{n} scenarios, {m} correspondences -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
