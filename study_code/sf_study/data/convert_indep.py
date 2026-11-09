"""Convert the independent-sampling corpus (harness/INDEPENDENT_SAMPLING_20260929.md) into
Schematch multi-table scenarios, the study's external test set.

Source: harness/generated/independent_sampling_20260929/ (tables written by
harness/independent_sampling.py) and the ground truth it stored in
harness/reports/independent_sampling_20260929/cases/<db>_<seed>/<condition>/<a>_<b>_<regime>.npz.
The stored matrices index each side's columns in the driver's flatten order: tables sorted by
name, columns in file order; this script checks that the shapes match the files.

For each database (TPC-H, AdventureWorks, Chinook), sampling seed (3), overlap condition
(identical, half, disjoint) and schema pairing (N-D1, N-D2, D1-D2) it writes

    <out>/Indep-<db>/s<seed>_<condition>_<a>-<b>/
        source/<table>.csv       variant a's tables (the source sample)
        target/<table>.csv       variant b's tables (the target sample of the condition)
        ground_truth/<s>___<t>.csv   0/1 matrix for every table pair (rows: s's columns)
        ground_truth_full.csv    the whole 0/1 matrix in Schematch's column order (tables by name)
        provenance.json          the stored flatten order, for checks and scoring

The ground truth is schema provenance: n:m (denormalised keys appear in several tables), so 1:1
extraction cannot reach full recall. Only the plain regime is written; the opaque regime is the
driver's in-memory recoding and would need its recode() and label hashing.

    python data/convert_indep.py --out data/indep_plain
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent.parent / "harness"
TABLES = HARNESS / "generated" / "independent_sampling_20260929"
CASES = HARNESS / "reports" / "independent_sampling_20260929" / "cases"
CONDITIONS = ("identical", "half", "disjoint")


def columns(folder: Path) -> list[tuple[str, str]]:
    out = []
    for f in sorted(folder.glob("*.csv"), key=lambda p: p.stem):
        with f.open(encoding="utf-8", newline="") as fh:
            out += [(f.stem, c) for c in next(csv.reader(fh))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "indep_plain"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    n, stats = 0, []
    for case_dir in sorted(p for p in CASES.iterdir() if p.is_dir()):
        db, seed = case_dir.name.rsplit("_", 1)
        for cond in CONDITIONS:
            for npz in sorted((case_dir / cond).glob("*_plain.npz")):
                sv, tv = npz.stem.split("_")[:2]
                src = TABLES / case_dir.name / "source" / sv / "source"
                tgt = TABLES / case_dir.name / cond / tv / "target"
                s_cols, t_cols = columns(src), columns(tgt)
                gt = np.load(npz)["gt"]
                if gt.shape != (len(s_cols), len(t_cols)):
                    raise SystemExit(f"{npz}: ground truth {gt.shape} != columns {(len(s_cols), len(t_cols))}")
                scen = out / f"Indep-{db}" / f"s{seed}_{cond}_{sv}-{tv}"
                for side, folder in (("source", src), ("target", tgt)):
                    (scen / side).mkdir(parents=True)
                    for f in folder.glob("*.csv"):
                        shutil.copyfile(f, scen / side / f.name)
                (scen / "ground_truth").mkdir()
                s_tables = sorted({t for t, _ in s_cols})
                t_tables = sorted({t for t, _ in t_cols})
                for st in s_tables:
                    rows = [i for i, (t, _) in enumerate(s_cols) if t == st]
                    for tt in t_tables:
                        cols = [j for j, (t, _) in enumerate(t_cols) if t == tt]
                        block = gt[np.ix_(rows, cols)].astype(int)
                        with (scen / "ground_truth" / f"{st}___{tt}.csv").open("w", newline="", encoding="utf-8") as fh:
                            csv.writer(fh).writerows(block.tolist())
                with (scen / "ground_truth_full.csv").open("w", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerows(gt.astype(int).tolist())
                (scen / "provenance.json").write_text(json.dumps(
                    {"database": db, "seed": seed, "condition": cond, "source_variant": sv, "target_variant": tv,
                     "source_columns": s_cols, "target_columns": t_cols, "gt_pairs": int(gt.sum())}, indent=1),
                    encoding="utf-8")
                diag = int(sum(gt[i, i] for i in range(min(gt.shape))))
                stats.append((db, cond, f"{sv}-{tv}", gt.shape, int(gt.sum()), diag, len(s_tables), len(t_tables)))
                n += 1
    print(f"wrote {n} scenarios under {out}")
    tot = sum(s[4] for s in stats)
    print(f"ground-truth pairs {tot}, on the flattened index diagonal {sum(s[5] for s in stats)} "
          f"({sum(s[5] for s in stats) / tot:.1%}); tables per side {min(s[6] for s in stats)}-{max(s[6] for s in stats)} "
          f"/ {min(s[7] for s in stats)}-{max(s[7] for s in stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
