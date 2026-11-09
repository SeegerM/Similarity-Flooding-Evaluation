"""Turn cached harness matrices of learned systems into SF FILE seeds.

The harness has already run SemStruct, ReMa and SiMa over the corpus, and their score matrices are
cached. They are written in the tables' own column order without labels; SF's FILE seed reader
realigns by label, so each matrix is written here as Schematch's verbose seed.csv with the
regime's own header labels:

    seeds_imported/<regime>/<name>/<dataset>/<scenario>/seed.csv

Sources (all from harness/reports):
    semstruct            semstruct_20260908/matrices/<plain|audited_md5>/semstruct/<key>.csv   382 cases
    semstruct_lm_only    .../lm_only/<key>.csv   control: the LM encoder without structure   382 cases
    rema                 rema_20260908/cases/<key>/<regime>/rema/frozen/scores.csv          297 cases
    rema_shuffled        .../shuffled/frozen/scores.csv   control: rows shuffled              297 cases
    sima_tpcdi           sima_holdout_tpcdi/matrices/TPC-DI-Unionable/<scenario>.csv     plain only;
                         the only leakage-free family of the held-out run (SIMA_INTEGRATION.md)

Leakage: SemStruct trains on distinct source tables only and never sees a target (its paper's
rule, kept in the reimplementation); ReMa is unsupervised per pair; SiMa is supervised, hence
only its held-out family is imported. Every matrix's shape is checked against the ground truth
and a mismatch is skipped and reported, never padded.

Usage:
    python import_file_seeds.py            # both regimes, all sources
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / "harness"
sys.path.insert(0, str(HARNESS))

import numpy as np  # noqa: E402

from smx.adapters import read_matrix  # noqa: E402

REPORTS = HARNESS / "reports"
PROTOCOL = REPORTS / "audited_md5_fused_20260904" / "protocol.json"
REGIMES = {"plain": "plain", "md5": "audited_md5"}   # study regime -> harness regime key


def headers(table: Path) -> list[str]:
    with table.open(newline="", encoding="utf-8", errors="ignore") as fh:
        return [h.strip() for h in next(csv.reader(fh))]


def one(folder: Path) -> Path | None:
    files = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    return files[0] if len(files) == 1 else None


def write_seed(path: Path, S: np.ndarray, rows: list[str], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Source\\Target"] + [f"target.{c}" for c in cols])
        for label, row in zip(rows, S):
            w.writerow([f"source.{label}"] + [repr(float(v)) for v in row])


def sources(case: dict, harness_regime: str) -> dict[str, Path]:
    key = case["key"]
    out = {
        "semstruct": REPORTS / "semstruct_20260908" / "matrices" / harness_regime / "semstruct" / f"{key}.csv",
        "semstruct_lm_only": REPORTS / "semstruct_20260908" / "matrices" / harness_regime / "lm_only" / f"{key}.csv",
        "rema": REPORTS / "rema_20260908" / "cases" / key / harness_regime / "rema" / "frozen" / "scores.csv",
        "rema_shuffled": REPORTS / "rema_20260908" / "cases" / key / harness_regime / "shuffled" / "frozen" / "scores.csv",
    }
    if harness_regime == "plain" and case["dataset"] == "TPC-DI-Unionable":
        out["sima_tpcdi"] = REPORTS / "sima_holdout_tpcdi" / "matrices" / "TPC-DI-Unionable" / f"{case['scenario']}.csv"
    return out


def main() -> int:
    cases = json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]
    summary = {}
    for regime, harness_regime in REGIMES.items():
        for case in cases:
            folder = Path(case[harness_regime])
            src, tgt = one(folder / "source"), one(folder / "target")
            gt_file = one(Path(case["plain"]) / "ground_truth")
            if src is None or tgt is None or gt_file is None:
                continue
            s_labels, t_labels = headers(src), headers(tgt)
            shape = read_matrix(gt_file)[0].shape
            for name, path in sources(case, harness_regime).items():
                stats = summary.setdefault(f"{regime}/{name}", {"written": 0, "missing": 0, "bad_shape": 0})
                if not path.is_file():
                    stats["missing"] += 1
                    continue
                S = read_matrix(path)[0]
                if S.shape != shape or S.shape != (len(s_labels), len(t_labels)) or not np.isfinite(S).all():
                    stats["bad_shape"] += 1
                    continue
                write_seed(HERE / "seeds_imported" / regime / name / case["dataset"] / case["scenario"] / "seed.csv",
                           S, s_labels, t_labels)
                stats["written"] += 1
    for k, v in sorted(summary.items()):
        print(f"{k:28s} {v}")
    (HERE / "seeds_imported").mkdir(exist_ok=True)
    (HERE / "seeds_imported" / "provenance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
