"""Rerun 549 (2026-11-02): the X77 benchmark properties (explore_benchmark_properties.score_case, unchanged) for the corpora
whose effects are recomputed on 549 / the fixed HyFD profiles, as input to X78 (layer3b/x78_analysis.py) and the corpus
associations (review_checks/corpus_assoc.py):

  valentine_plain, valentine_md5   the 549 scenarios: the 381 on the fixed-FD mirror (harness/generated_hyfdfix/valentine_<r>;
                                   tables identical, FD files fixed), the 168 on the new trees (seeds pnew / mnew)
  indep2, synthea                  the fixed-FD mirrors harness/generated_hyfdfix/<corpus>

Every other corpus keeps its stored row: their effects in X78 are registered results on the old profiles (final test,
Amendments 5, 7, 9, 11), so their properties stay paired with them. Output results/x78_549/benchmark_properties.csv.

    SMX_VALENTINE=549 ../harness/.venv-bdi/Scripts/python.exe -B reference/benchmark_properties_549.py
"""

from __future__ import annotations

import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent
for p in (ROOT / "harness", STUDY / "tuning", STUDY, HERE):
    sys.path.insert(0, str(p))
import valentine549 as v  # noqa: E402

FIX = ROOT / "harness" / "generated_hyfdfix"
RECOMPUTED = ("valentine_plain", "valentine_md5", "indep2", "synthea")
OUT = STUDY / "results" / "x78_549"


def job(args):
    corpus, root, seed, multi, ds, sc = args
    import explore_benchmark_properties as ebp
    ebp.CORPORA[corpus] = (Path(root), seed, multi)
    return ebp.score_case((corpus, ds, sc))


def main() -> int:
    if not v.V549:
        raise SystemExit("run with SMX_VALENTINE=549")
    jobs = []
    for corpus, r in (("valentine_plain", "plain"), ("valentine_md5", "md5")):
        for ds, sc, _ in v.cases549():
            if v.is_new(ds, sc):
                jobs.append((corpus, str(v.NEW_TREE[r]), v.NEW[r], False, ds, sc))
            else:
                jobs.append((corpus, str(FIX / f"valentine_{r}"), r, False, ds, sc))
    for corpus in ("indep2", "synthea"):
        root = FIX / corpus
        jobs += [(corpus, str(root), corpus, True, d.name, s.name) for d in sorted(p for p in root.iterdir() if p.is_dir())
                 for s in sorted(p for p in d.iterdir() if p.is_dir())]
    with Pool(8) as pool:
        new = [r for r in pool.imap_unordered(job, jobs, chunksize=4) if r]
    with open(STUDY / "results" / "benchmark_properties.csv", encoding="utf-8") as fh:
        old = [r for r in csv.DictReader(fh)]
    keep = [r for r in old if r["corpus"] not in RECOMPUTED]
    rows = keep + new
    rows.sort(key=lambda r: (r["corpus"], r["dataset"], r["scenario"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "benchmark_properties.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(old[0].keys()))
        w.writeheader()
        w.writerows(rows)
    counts = {c: sum(r["corpus"] == c for r in new) for c in RECOMPUTED}
    print(f"{len(rows)} rows ({len(keep)} stored, recomputed {counts}) -> {OUT / 'benchmark_properties.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
