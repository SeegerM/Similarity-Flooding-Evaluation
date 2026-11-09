"""Export seed-study configurations (configs/<regime>/<seed>/) for the cluster dispatcher.

dispatch.py works on tuning/configs/<suffix>/<label>/ with a meta.json; this writes the Java seeds
of the seed study in that layout under the suffix seedstudy_<regime>:

  - defaultDatasetBasePath points to the hosts' corpus ($SMX_REMOTE_ROOT/smx/corpus/<regime>, same scenarios as
    harness/generated/benchmark-full-<regime> without GDC); resultsDir to tuning/results/<suffix>/<seed>;
  - meta.json names the datasets (from datasets.yaml) and their scenario counts, so
    `dispatch.py start --split-datasets` can schedule the largest (seed, dataset) tasks first.
    Its seed_dir is resources/ (the thesaurus), which `push` copies along.

--python exports the PythonServerMatcher seeds instead, under seedpy_<regime>: remote_runner.py
gives each of their tasks a private seed server from $SMX_REMOTE_ROOT/smx/pyroot (built on <gpu-host>, see
CPU_SERVERS.md); local model paths under code/ are rewritten to the bundled copies. ISResMat stays
on the GPU server. --seeds restricts the export to the named seeds. Seeds already DONE in
results/<regime>/ are skipped unless named with --include (e.g. one seed as a determinism check).
After `dispatch.py collect`, import_cluster_results() moves each merged run to results/<regime>/<seed>/,
where score_seeds.py reads it like a local run.

Usage:
    python export_seed_study.py --regime md5 --include levenshtein
    python export_seed_study.py --regime md5 --import-results
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))
import make_configs  # noqa: E402

REMOTE_CORPUS = "$SMX_REMOTE_ROOT/smx/corpus"
REMOTE_PYROOT = "$SMX_REMOTE_ROOT/smx/pyroot"
GPU_ONLY = {"isresmat", "isresmat_hybrid"}
# 2026-10-30, the authors' decision: valentine_distribution is no longer run (it never finished on some recoded
# datasets and takes hours on others; it was already dropped from the md5 panel on 2026-10-06)
EXCLUDED = {"valentine_distribution"}
LOCAL_CORPUS = STUDY.parent / "harness" / "generated"


def java_seeds() -> list[str]:
    return [n for n, (cls, _, _) in make_configs.SEEDS.items() if cls != "PythonServerMatcher"]


def python_seeds() -> list[str]:
    return [n for n, (cls, _, _) in make_configs.SEEDS.items()
            if cls == "PythonServerMatcher" and n not in GPU_ONLY and n not in EXCLUDED]


def suffix_of(regime: str, python: bool) -> str:
    return f"{'seedpy' if python else 'seedstudy'}_{regime}"


def export(regime: str, include: set[str], python: bool = False, only: set[str] | None = None) -> int:
    suffix = suffix_of(regime, python)
    src_root = STUDY / "configs" / regime
    dst_root = STUDY / "tuning" / "configs" / suffix
    done = {p.parent.name for p in (STUDY / "results" / regime).glob("*/DONE")}
    pool = python_seeds() if python else java_seeds()
    seeds = [s for s in pool if (s not in done or s in include) and (not only or s in only)]
    missing = [s for s in seeds if not (src_root / s).is_dir()]
    if missing:
        raise SystemExit(f"no configs for {missing}; run make_configs.py --regime {regime} first")
    corpus = make_configs.CORPUS[regime]  # local copy of what the hosts hold at $SMX_REMOTE_ROOT/smx/corpus/<regime>
    if dst_root.exists():
        shutil.rmtree(dst_root)
    for seed in seeds:
        dst = dst_root / seed
        dst.mkdir(parents=True)
        for f in (src_root / seed).iterdir():
            text = f.read_text(encoding="utf-8")
            if f.name == "general.yaml":
                text = re.sub(r'(?m)^defaultDatasetBasePath: ".*"$',
                              f'defaultDatasetBasePath: "{REMOTE_CORPUS}/{regime}"', text)
                text = re.sub(r'(?m)^resultsDir: ".*"$',
                              f'resultsDir: "{(STUDY / "tuning" / "results" / suffix / seed).as_posix()}"', text)
            text = text.replace((STUDY.parent / "code").as_posix(), f"{REMOTE_PYROOT}/code")  # e.g. bert-base
            (dst / f.name).write_text(text, encoding="utf-8")
        datasets = re.findall(r'(?m)^path: "(.*)"$', (src_root / seed / "datasets.yaml").read_text(encoding="utf-8"))
        meta = {"seed": seed, "regime": regime, "seed_dir": (STUDY / "resources").as_posix(),
                "datasets": datasets,
                "dataset_sizes": {d: sum(1 for p in (corpus / d).iterdir() if p.is_dir()) for d in datasets}}
        (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{suffix}: {len(seeds)} seeds x {len(datasets)} datasets -> {dst_root}")
    print(f"  skipped (DONE locally): {sorted(done & set(pool) - include)}")
    return 0


def import_cluster_results(regime: str, python: bool = False) -> int:
    """Move collected, merged cluster runs into results/<regime>/<seed>/ (merged/ acts as the run folder)."""
    src_root = STUDY / "tuning" / "results" / suffix_of(regime, python)
    moved, waiting = [], []
    for d in sorted(p for p in src_root.iterdir() if p.is_dir()):
        if not ((d / "DONE").is_file() and (d / "merged").is_dir()):
            waiting.append(d.name)
            continue
        dst = STUDY / "results" / regime / d.name
        if (dst / "DONE").is_file():
            continue  # a local run exists (e.g. the determinism check); keep it
        if dst.exists():
            shutil.rmtree(dst)  # a partial local run
        shutil.move(str(d), str(dst))
        moved.append(d.name)
    print(f"moved {len(moved)} seeds into results/{regime}/: {moved}")
    if waiting:
        print(f"not complete yet: {waiting}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--include", default="", help="comma-separated seeds to export even if DONE locally")
    ap.add_argument("--python", action="store_true", help="the PythonServerMatcher seeds (seedpy_<regime>)")
    ap.add_argument("--seeds", default="", help="comma-separated: export only these")
    ap.add_argument("--import-results", action="store_true")
    a = ap.parse_args()
    if a.import_results:
        return import_cluster_results(a.regime, a.python)
    return export(a.regime, {s for s in a.include.split(",") if s}, a.python,
                  {s for s in a.seeds.split(",") if s} or None)


if __name__ == "__main__":
    raise SystemExit(main())
