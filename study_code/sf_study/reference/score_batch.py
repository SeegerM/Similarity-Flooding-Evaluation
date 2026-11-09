"""Score any collected batch tuning/results/<batch>/<seed>__<arm>/ on the tie-free metrics (rerun 549, 2026-10-31).

The batches of the Valentine 549 rerun keep the layout of the batch they replace but carry new names (std_p382fix,
l1fd_p382fix, layer1_pnew, ...), so reference/score_layer1.py, which derives its folders from layer<L>_<regime>, cannot
find them. This script applies score_layer1.score_seed (seed from reference/seeds/<regime>/, SF from
MatchingStepLine2, the same per-scenario random orders as score_seeds_tiefree.py) unchanged to every configuration of
the named batch that has DONE, and writes tuning/results/<batch>_scores/scores.csv with score_layer1's columns.
Only batches marked COLLECTED by the collecting session are scored unless --partial is given.

    ../harness/.venv-bdi/Scripts/python.exe -B reference/score_batch.py --batch std_p382fix --regime plain
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))

import score_layer1  # noqa: E402
from score_seeds import CORPUS, cases_for  # noqa: E402

# seed folders that hold initial matrices for the 382-case corpora under another name (PRISMA on the fixed FD
# profiles, the rema fill): reference/seeds/<alias>/<seed>/, scored against the corpus of the base regime. Registered at
# import so that the pool's spawned workers see them too (score_seed looks up CORPUS[regime]).
SEED_ALIAS = {"plain_fdfix": "plain", "md5_fdfix": "md5", "plain_fill": "plain", "md5_fill": "md5"}
for _alias, _base in SEED_ALIAS.items():
    CORPUS.setdefault(_alias, CORPUS[_base])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--regime", choices=sorted(CORPUS), required=True,
                    help="corpus and seed folder of the batch (plain for *_p382fix, md5 for *_m382fix, pnew, mnew)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--partial", action="store_true", help="score a batch that is not marked COLLECTED")
    ap.add_argument("--seed-regime", choices=sorted(SEED_ALIAS),
                    help="read the initial matrices from reference/seeds/<this>/ instead of reference/seeds/<regime>/")
    a = ap.parse_args()
    if a.seed_regime and SEED_ALIAS[a.seed_regime] != a.regime:
        ap.error(f"--seed-regime {a.seed_regime} belongs to regime {SEED_ALIAS[a.seed_regime]}")
    sys.stdout.reconfigure(encoding="utf-8")
    root = STUDY / "tuning" / "results" / a.batch
    if not (root / "COLLECTED").is_file() and not a.partial:
        print(f"{a.batch} is not marked COLLECTED; use --partial to score it anyway")
        return 1
    cases = cases_for(a.regime)
    by_seed = defaultdict(list)
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "DONE").is_file() and "__" in p.name):
        by_seed[d.name.split("__", 1)[0]].append(str(d))
    jobs = [(a.seed_regime or a.regime, s, dirs, cases) for s, dirs in sorted(by_seed.items())]
    rows = []
    with Pool(a.workers) as pool:
        for part in pool.imap_unordered(score_layer1.score_seed, jobs):
            rows += part
    rows.sort(key=lambda r: (r["seed"], r["arm"], r["dataset"], r["scenario"]))
    out_dir = STUDY / "tuning" / "results" / f"{a.batch}_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with (out_dir / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    per = defaultdict(int)
    for r in rows:
        per[(r["seed"], r["arm"])] += 1
    print(f"{a.batch}: {len(rows)} rows, {len(per)} configurations, {len(cases)} protocol cases, "
          f"scenarios per configuration {min(per.values()) if per else 0}-{max(per.values()) if per else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
