"""REMA-ARC (frozen configuration) on the 168 new Valentine scenarios of the 549 rerun.

The algorithm and every parameter are rema_reproduction.py's frozen ones (smx/rema.py, row cap 1000,
1 epoch, seed 20260908, shuffle salt "rema-null-control-20260908"); nothing is re-chosen. Only the
cohort differs: the new scenarios (harness/generated/valentine-new168 and -md5), keyed exactly as the
protocol keys its cases (sha256(id)[:16]), so the shuffled arm's per-case salt follows the same rule.
Matrices are cached under reports/rema_new168_20261031/cases/<key>/<regime>/<arm>/frozen/ (separate from
the 382 run's pinned directory) and exported as SF FILE seeds in import_file_seeds.py's format:

    sf-matcher-study/reference/seeds/{pnew,mnew}/{rema,rema_shuffled}/<dataset>/<scenario>/seed.csv

Commands (gensim lives in .venv-bdi):
    python rema_new168.py verify --n 6                       # recompute old 382 matrices, compare bitwise
    python rema_new168.py compute --regime plain --shards 12 --shard k
    python rema_new168.py export
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "sf-matcher-study"))

import rema_reproduction as rr  # noqa: E402
from matrix_health import health, reasons  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

NEW = {"plain": HERE / "generated" / "valentine-new168",
       "audited_md5": HERE / "generated" / "valentine-new168-md5"}
OUT = HERE / "reports" / "rema_new168_20261031"
STUDY_SEEDS = HERE.parent / "sf-matcher-study" / "reference" / "seeds"
STUDY_REGIME = {"plain": "pnew", "audited_md5": "mnew"}
OLD_STUDY_REGIME = {"plain": "plain", "audited_md5": "md5"}
SEED_NAME = {"rema": "rema", "shuffled": "rema_shuffled"}


def new_cases() -> list[dict]:
    cases = []
    for folder in sorted(p for p in NEW["plain"].glob("*/*") if (p / "manifest.json").is_file()):
        man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        md5 = NEW["audited_md5"] / folder.parent.name / folder.name
        cases.append({"id": man["id"], "key": hashlib.sha256(man["id"].encode()).hexdigest()[:16],
                      "cohort": "new168", "family": man["family"],
                      "scenario_type": man["scenario_type"], "dataset": folder.parent.name,
                      "scenario": folder.name, "plain": str(folder), "audited_md5": str(md5)})
    assert len({c["key"] for c in cases}) == len(cases)
    return cases


def import_module():
    import import_file_seeds as ifs  # sf-matcher-study
    return ifs


def seed_csv(case: dict, regime: str, S: np.ndarray, dest: Path) -> None:
    ifs = import_module()
    folder = Path(case[regime])
    src, tgt = ifs.one(folder / "source"), ifs.one(folder / "target")
    ifs.write_seed(dest, S, ifs.headers(src), ifs.headers(tgt))


def command_verify(args) -> int:
    """Recompute cached 382 matrices from scratch and compare with the cache and the reference seeds."""
    protocol = rr.load_protocol()
    cached = rr.OUT / "cases"
    # spread over datasets and both regimes; only cases that the 382 run finished
    picks, seen = [], set()
    for c in protocol["cases"]:
        if c["dataset"] in seen:
            continue
        if all((cached / c["key"] / r / a / "frozen" / "scores.csv").is_file()
               for r in rr.REGIMES for a in rr.ARMS):
            picks.append(c)
            seen.add(c["dataset"])
    picks = picks[args.start: args.start + args.n]
    scratch = Path(args.scratch)
    ok = True
    for c in picks:
        for regime in rr.REGIMES:
            for arm in rr.ARMS:
                t = time.perf_counter()
                S, info = rr.scenario_matrix(c, regime, arm, scratch)
                secs = time.perf_counter() - t
                new_csv = scratch / "cases" / c["key"] / regime / arm / "frozen" / "scores.csv"
                old_csv = cached / c["key"] / regime / arm / "frozen" / "scores.csv"
                same_bytes = new_csv.read_bytes() == old_csv.read_bytes()
                ref = STUDY_SEEDS / OLD_STUDY_REGIME[regime] / SEED_NAME[arm] / c["dataset"] / c["scenario"] / "seed.csv"
                tmp = scratch / "seedcheck" / c["key"] / regime / arm / "seed.csv"
                seed_csv(c, regime, S, tmp)
                ref_same = ref.read_bytes() == tmp.read_bytes() if ref.is_file() else None
                maxdiff = float(np.abs(S - read_matrix(old_csv)[0]).max())
                ok &= same_bytes and ref_same is not False
                print(f"{c['dataset']}/{c['scenario']} {regime} {arm}: cache bytes identical={same_bytes} "
                      f"max|diff|={maxdiff:.3g} reference seed.csv identical={ref_same} ({secs:.1f}s)", flush=True)
    print("VERIFY", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


def command_compute(args) -> int:
    cases = [c for i, c in enumerate(new_cases()) if i % args.shards == args.shard]
    regimes = rr.REGIMES if args.regime == "both" else (args.regime,)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for i, c in enumerate(cases, 1):
        for regime in regimes:
            for arm in rr.ARMS:
                try:
                    S, info = rr.scenario_matrix(c, regime, arm, OUT)
                    print(f"[{args.shard}] {i}/{len(cases)} {c['dataset']}/{c['scenario']} {regime} {arm} "
                          f"{info.get('seconds', 0)}s", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[{args.shard}] ! {c['id']} {regime} {arm}: {type(exc).__name__}: {exc}", flush=True)
    print(f"[{args.shard}] done {len(cases)} cases in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


def command_export(args) -> int:
    ifs = import_module()
    stats, diag = {}, []
    for c in new_cases():
        gt_file = ifs.one(Path(c["plain"]) / "ground_truth")
        shape = read_matrix(gt_file)[0].shape
        for regime in rr.REGIMES:
            folder = Path(c[regime])
            s_labels = ifs.headers(ifs.one(folder / "source"))
            t_labels = ifs.headers(ifs.one(folder / "target"))
            for arm in rr.ARMS:
                name = f"{STUDY_REGIME[regime]}/{SEED_NAME[arm]}"
                st = stats.setdefault(name, {"written": 0, "missing": 0, "bad_shape": 0, "degenerate": 0})
                p = OUT / "cases" / c["key"] / regime / arm / "frozen" / "scores.csv"
                if not p.is_file():
                    st["missing"] += 1
                    continue
                S = read_matrix(p)[0]
                if S.shape != shape or S.shape != (len(s_labels), len(t_labels)) or not np.isfinite(S).all():
                    st["bad_shape"] += 1
                    continue
                h = health(S)
                if h["degenerate"]:
                    st["degenerate"] += 1   # kept, as import_file_seeds kept them; recorded only
                    diag.append({"id": c["id"], "regime": regime, "arm": arm, "reasons": ";".join(reasons(h))})
                ifs.write_seed(STUDY_SEEDS / STUDY_REGIME[regime] / SEED_NAME[arm] / c["dataset"] / c["scenario"]
                               / "seed.csv", S, s_labels, t_labels)
                st["written"] += 1
    for k, v in sorted(stats.items()):
        print(f"{k:24s} {v}")
    rr.json_write(OUT / "export.json", {"stats": stats, "degenerate": diag})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--n", type=int, default=6)
    v.add_argument("--start", type=int, default=0)
    v.add_argument("--scratch", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--regime", default="both", choices=("both",) + rr.REGIMES)
    c.add_argument("--shards", type=int, default=1)
    c.add_argument("--shard", type=int, default=0)
    sub.add_parser("export")
    args = ap.parse_args()
    return {"verify": command_verify, "compute": command_compute, "export": command_export}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
