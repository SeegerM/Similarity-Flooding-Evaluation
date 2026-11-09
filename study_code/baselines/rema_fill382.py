"""Complete REMA-ARC (rema, rema_shuffled) on the old 382-case protocol: the cases the 2026-09-08 run never reached.

The 382 run (rema_reproduction.py, reports/rema_20260908) stopped part-way in protocol order, so TPC-DI-SJ is
partial and TPC-DI-Unionable, TPC-DI-View-Unionable and Wikidata are missing; the reference seed copy keeps only
fully covered datasets (reference/seeds/{plain,md5}/rema*: 253). This script fills the rest with the identical
frozen configuration (rema_reproduction.scenario_matrix; nothing re-chosen), caching the new matrices in
reports/rema_fill382_20261031 (the pinned 382 folder is not written), and exports every case of the datasets the
reference copy lacks -- cached 382 matrices where they exist, new ones otherwise -- to

    sf-matcher-study/reference/seeds/{plain,md5}_fill/{rema,rema_shuffled}/<dataset>/<scenario>/seed.csv

so reference/seeds/<r>/<seed> and <r>_fill/<seed> together cover all 382 cases, each dataset completely in one of them.

    python rema_fill382.py compute --shards 8 --shard k
    python rema_fill382.py verify --n 3          # recompute cached TPC-DI-SJ cases, compare bytes
    python rema_fill382.py export
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "sf-matcher-study"))

import rema_reproduction as rr  # noqa: E402
from matrix_health import health  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

OLD = rr.OUT / "cases"
FILL = HERE / "reports" / "rema_fill382_20261031"
SEEDS = HERE.parent / "sf-matcher-study" / "reference" / "seeds"
OLD_REGIME = {"plain": "plain", "audited_md5": "md5"}
SEED_NAME = {"rema": "rema", "shuffled": "rema_shuffled"}


def cached(case, regime, arm, root=OLD) -> Path:
    return root / case["key"] / regime / arm / "frozen" / "scores.csv"


def todo() -> list[tuple[dict, str, str]]:
    out = []
    for c in rr.load_protocol()["cases"]:
        for regime in rr.REGIMES:
            for arm in rr.ARMS:
                if not cached(c, regime, arm).is_file():
                    out.append((c, regime, arm))
    return out


def command_compute(args) -> int:
    jobs = [j for i, j in enumerate(todo()) if i % args.shards == args.shard]
    t0 = time.perf_counter()
    for i, (c, regime, arm) in enumerate(jobs, 1):
        try:
            S, info = rr.scenario_matrix(c, regime, arm, FILL)
            print(f"[{args.shard}] {i}/{len(jobs)} {c['dataset']}/{c['scenario']} {regime} {arm} "
                  f"{info.get('seconds', 0)}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{args.shard}] ! {c['id']} {regime} {arm}: {type(exc).__name__}: {exc}", flush=True)
    print(f"[{args.shard}] done {len(jobs)} matrices in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


def command_verify(args) -> int:
    scratch = Path(args.scratch)
    picks = [c for c in rr.load_protocol()["cases"]
             if c["dataset"] == "TPC-DI-Semantically-Joinable"
             and all(cached(c, r, a).is_file() for r in rr.REGIMES for a in rr.ARMS)][: args.n]
    ok = True
    for c in picks:
        for regime in rr.REGIMES:
            for arm in rr.ARMS:
                rr.scenario_matrix(c, regime, arm, scratch)
                same = cached(c, regime, arm, scratch / "cases").read_bytes() == cached(c, regime, arm).read_bytes()
                ok &= same
                print(f"{c['scenario']} {regime} {arm}: bytes identical={same}", flush=True)
    print("VERIFY", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


def command_export(args) -> int:
    import import_file_seeds as ifs
    stats = {}
    for c in rr.load_protocol()["cases"]:
        shape = read_matrix(ifs.one(Path(c["plain"]) / "ground_truth"))[0].shape
        for regime in rr.REGIMES:
            folder = Path(c[regime])
            s_labels = ifs.headers(ifs.one(folder / "source"))
            t_labels = ifs.headers(ifs.one(folder / "target"))
            for arm in rr.ARMS:
                ref = SEEDS / OLD_REGIME[regime] / SEED_NAME[arm]
                if (ref / c["dataset"]).is_dir():
                    continue                                  # dataset already in the reference copy
                name = f"{OLD_REGIME[regime]}_fill/{SEED_NAME[arm]}"
                st = stats.setdefault(name, {"from_382_cache": 0, "new": 0, "missing": 0, "bad_shape": 0,
                                             "degenerate": 0})
                p_old, p_new = cached(c, regime, arm), cached(c, regime, arm, FILL / "cases")
                p = p_old if p_old.is_file() else p_new
                if not p.is_file():
                    st["missing"] += 1
                    continue
                S = read_matrix(p)[0]
                if S.shape != shape or S.shape != (len(s_labels), len(t_labels)) or not np.isfinite(S).all():
                    st["bad_shape"] += 1
                    continue
                st["degenerate"] += int(health(S)["degenerate"])
                ifs.write_seed(SEEDS / name / c["dataset"] / c["scenario"] / "seed.csv", S, s_labels, t_labels)
                st["from_382_cache" if p is p_old else "new"] += 1
    for k, v in sorted(stats.items()):
        print(f"{k:24s} {v}")
    rr.json_write(FILL / "export.json", stats)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--shards", type=int, default=1)
    c.add_argument("--shard", type=int, default=0)
    v = sub.add_parser("verify")
    v.add_argument("--n", type=int, default=3)
    v.add_argument("--scratch", required=True)
    sub.add_parser("export")
    args = ap.parse_args()
    return {"compute": command_compute, "verify": command_verify, "export": command_export}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
