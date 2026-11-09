"""Freeze the seed panel as FILE seeds, from the full seed-study run.

For every panel seed the first-line matrix (MatchingStepLine1) of the latest finished run under
results/<regime>/<seed>/ is copied, unchanged, to

    tuning/seeds/<regime>/<seed>/<dataset>/<scenario>/seed.csv

so every tuning configuration floods exactly the matrix the seed study scored. Seeds that are
already FILE seeds in the study (semstruct) are linked to their directory instead. A seed whose
run is not finished is reported and skipped; rerun this after the full run.

Usage:
    python export_seeds.py --regime plain
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))

from space import FILE_SEED_DIRS, SEED_PANEL  # noqa: E402


def one(folder: Path) -> Path | None:
    files = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    return files[0] if len(files) == 1 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5"), default="plain")
    args = ap.parse_args()
    for seed in SEED_PANEL:
        dest_root = HERE / "seeds" / args.regime / seed
        if seed in FILE_SEED_DIRS:
            src = STUDY / FILE_SEED_DIRS[seed].format(regime=args.regime)
            print(f"{seed:18s} FILE seed, used in place: {src}" if src.is_dir()
                  else f"{seed:18s} MISSING {src}")
            continue
        seed_dir = STUDY / "results" / args.regime / seed
        if not (seed_dir / "DONE").is_file():
            print(f"{seed:18s} not finished in the full run yet; skipped")
            continue
        run = sorted(p for p in seed_dir.iterdir() if p.is_dir())[-1]
        n = 0
        for scenario in sorted(run.glob("*/*")):
            f = one(scenario / "_outputs" / "MatchingStepLine1")
            if f is None:
                continue
            dest = dest_root / scenario.parent.name / scenario.name / "seed.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)
            n += 1
        print(f"{seed:18s} {n} seed matrices from {run.name}")
    return 0


def seed_dir(seed: str, regime: str) -> Path:
    if seed in FILE_SEED_DIRS:
        return (STUDY / FILE_SEED_DIRS[seed].format(regime=regime)).resolve()
    return HERE / "seeds" / regime / seed


if __name__ == "__main__":
    raise SystemExit(main())
