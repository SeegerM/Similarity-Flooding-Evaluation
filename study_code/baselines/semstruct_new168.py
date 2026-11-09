"""SemStruct-reimpl seeds (semstruct, semstruct_lm_only) for the 168 new Valentine scenarios of the 549 rerun.

No training. The checkpoint is the one behind the existing seed matrices
(sf-matcher-study/reference/seeds/{plain,md5}/semstruct*, via import_file_seeds.py):
reports/semstruct_20260908/checkpoints/full.pt (config fingerprint adfce6f86b200e60, trained on plaintext source
tables of the 382, hardest-negative mining; NOT the 2026-10-27 semi-hard ablation checkpoints). It is copied
byte-identically (sha256 checked) into the new output folder, and semstruct_run.py's own `command_score` is called
with its defaults (tag full, merge raw, training row cap 400); only the case list (RUN), the output folder (OUT) and,
for the verification, the cache folder (CACHE, cold) are redirected.

    .venv-embed/Scripts/python.exe semstruct_new168.py verify --scratch <dir> [--per-dataset 1] [--regimes plain]
    .venv-embed/Scripts/python.exe semstruct_new168.py score --regime plain|audited_md5 --shards n --shard k
    .venv-embed/Scripts/python.exe semstruct_new168.py export
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # no downloads: e5-small must be in the local HF cache
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "sf-matcher-study"))

import numpy as np  # noqa: E402

import semstruct_run as sr  # noqa: E402

OLD = HERE / "reports" / "semstruct_20260908"
NEWOUT = HERE / "reports" / "semstruct_new168_20261031"
STUDY_SEEDS = HERE.parent / "sf-matcher-study" / "reference" / "seeds"
STUDY_REGIME = {"plain": "pnew", "audited_md5": "mnew"}
SEED_NAME = {"semstruct": "semstruct", "lm_only": "semstruct_lm_only"}
ARMS = ["semstruct", "lm_only"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def new_cases() -> list[dict]:
    """Same case list and keys as rema_new168.new_cases (key = sha256(id)[:16], as the 382 protocol)."""
    cases = []
    root = HERE / "generated" / "valentine-new168"
    for folder in sorted(p for p in root.glob("*/*") if (p / "manifest.json").is_file()):
        man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        cases.append({"id": man["id"], "key": hashlib.sha256(man["id"].encode()).hexdigest()[:16],
                      "cohort": "new168", "family": man["family"], "scenario_type": man["scenario_type"],
                      "dataset": folder.parent.name, "scenario": folder.name, "plain": str(folder),
                      "audited_md5": str(HERE / "generated" / "valentine-new168-md5" / folder.parent.name
                                         / folder.name)})
    return cases


def prepare_out(out: Path) -> None:
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    for name in ("full.pt", "full.json"):
        dst = out / "checkpoints" / name
        if not dst.is_file():
            shutil.copyfile(OLD / "checkpoints" / name, dst)
        assert sha(dst) == sha(OLD / "checkpoints" / name), dst


def score_args(regime: str, threads: int | None) -> argparse.Namespace:
    return argparse.Namespace(regime=regime, arms=list(ARMS), row_cap=0, epochs=0, limit=0, tag="full",
                              merge="raw", threads=threads, native=False, force=False, eval_row_cap=0)


def run_score(cases: list[dict], regime: str, out: Path, run_dir: Path, threads: int | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "protocol.json").write_text(json.dumps({"cases": cases}, indent=1), encoding="utf-8")
    sr.RUN = run_dir
    sr.OUT = out
    sr.command_score(score_args(regime, threads))


def command_verify(args) -> int:
    protocol = json.loads((sr.RUN / "protocol.json").read_text(encoding="utf-8"))
    picks, seen = [], {}
    for c in protocol["cases"]:
        if seen.get(c["dataset"], 0) < args.per_dataset:
            picks.append(c)
            seen[c["dataset"]] = seen.get(c["dataset"], 0) + 1
    scratch = Path(args.scratch)
    out = scratch / "out"
    prepare_out(out)
    sr.CACHE = scratch / "cache"                      # cold cache: the encoder and graph builder really run
    ok = True
    for regime in args.regimes:
        run_score(picks, regime, out, scratch / f"run_{regime}", args.threads)
        for c in picks:
            for arm in ARMS:
                new = out / "matrices" / regime / arm / f"{c['key']}.csv"
                old = OLD / "matrices" / regime / arm / f"{c['key']}.csv"
                same = new.is_file() and new.read_bytes() == old.read_bytes()
                diff = (float(np.abs(sr.read_written_matrix(new) - sr.read_written_matrix(old)).max())
                        if new.is_file() else float("nan"))
                ok &= same
                print(f"{c['dataset']}/{c['scenario']} {regime} {arm}: bytes identical={same} max|diff|={diff:.3g}",
                      flush=True)
    print("VERIFY", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


def command_score(args) -> int:
    prepare_out(NEWOUT)
    cases = [c for i, c in enumerate(new_cases()) if i % args.shards == args.shard]
    run_score(cases, args.regime, NEWOUT, NEWOUT / "runs" / f"{args.regime}_{args.shard}of{args.shards}",
              args.threads)
    return 0


def command_export(args) -> int:
    import import_file_seeds as ifs
    from smx.adapters import read_matrix
    stats = {}
    for c in new_cases():
        shape = read_matrix(ifs.one(Path(c["plain"]) / "ground_truth"))[0].shape
        for regime in sr.REGIMES:
            folder = Path(c[regime])
            s_labels = ifs.headers(ifs.one(folder / "source"))
            t_labels = ifs.headers(ifs.one(folder / "target"))
            for arm in ARMS:
                name = f"{STUDY_REGIME[regime]}/{SEED_NAME[arm]}"
                st = stats.setdefault(name, {"written": 0, "missing": 0, "bad_shape": 0})
                p = NEWOUT / "matrices" / regime / arm / f"{c['key']}.csv"
                if not p.is_file():
                    st["missing"] += 1
                    continue
                S = read_matrix(p)[0]
                if S.shape != shape or S.shape != (len(s_labels), len(t_labels)) or not np.isfinite(S).all():
                    st["bad_shape"] += 1
                    continue
                ifs.write_seed(STUDY_SEEDS / STUDY_REGIME[regime] / SEED_NAME[arm] / c["dataset"] / c["scenario"]
                               / "seed.csv", S, s_labels, t_labels)
                st["written"] += 1
    for k, v in sorted(stats.items()):
        print(f"{k:28s} {v}")
    sr.json_write(NEWOUT / "export.json", stats)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--scratch", required=True)
    v.add_argument("--per-dataset", type=int, default=1)
    v.add_argument("--regimes", nargs="+", default=list(sr.REGIMES))
    v.add_argument("--threads", type=int, default=4)
    s = sub.add_parser("score")
    s.add_argument("--regime", choices=sr.REGIMES, required=True)
    s.add_argument("--shards", type=int, default=1)
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--threads", type=int, default=2)
    sub.add_parser("export")
    args = ap.parse_args()
    return {"verify": command_verify, "score": command_score, "export": command_export}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
