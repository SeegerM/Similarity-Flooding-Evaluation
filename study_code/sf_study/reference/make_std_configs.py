"""The seed study's own SF configuration ("std": Table II) on the 382 audited Valentine scenarios, re-run from the
stored seed matrices on a corpus whose FD profiles come from the fixed HyFD (stage 1, harness/generated_hyfdfix).

SF exactly as the seed study ran it (make_configs.SF: inverse-product coefficients, fixpoint formula C, QUICK
encoding, every FD edge, no constraint weighting) with initialMatrix FILE and no JVM properties, i.e. the defaults
(no literal seeds, epsilon 1e-4); C2 in sanity_checks.py shows that a FILE seed reproduces the RAW_FIRST_LINE run.
Seeds are the seed study's first-line matrices, as layer 1 uses them (reference/seeds/<regime>/).

Writes tuning/configs/std_<corpus>/<seed>__std/ for servers/dispatch.py (--split-datasets). The hosts' corpus is
$SMX_REMOTE_ROOT/smx/corpus/<corpus>; with --corpus plain the run is a twin of the seed study (same FDs) and must reproduce it.

    python reference/make_std_configs.py --corpus p382fix
    python reference/make_std_configs.py --corpus plain --seeds levenshtein --datasets Magellan-Unionable   # twin
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
sys.path.insert(0, str(HERE))
from make_layer1_configs import ARMS, MELNIK_PROPS, MELNIK_SF, REMOTE_CORPUS, TEMPLATE, seed_source  # noqa: E402
from score_seeds import cases_for  # noqa: E402

STD_SF = {**MELNIK_SF, "propCoeffPolicy": "INV_PROD", "fdFilterThreshold": "0.0"}

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True, help="hosts' corpus name under $SMX_REMOTE_ROOT/smx/corpus (p382fix, or plain for the twin)")
ap.add_argument("--regime", default="plain", help="regime of the stored seeds and the 382 cases")
ap.add_argument("--arm", default="std", help="std (the seed study's SF), or a layer-1 arm (make_layer1_configs.ARMS), "
                                             "written exactly as layer 1 writes it; suffix l1<arm>_<corpus>")
ap.add_argument("--seeds", default="")
ap.add_argument("--datasets", default="")
ap.add_argument("--file-seeds", default="", help="comma-separated seeds whose matrices are already in "
                                                  "reference/seeds/<regime>/<seed>/ (e.g. datatype_nanfix); only these")
ap.add_argument("--suffix", default="", help="stage name instead of <prefix>_<corpus>")
ap.add_argument("--append", action="store_true", help="add to an existing stage")
a = ap.parse_args()
if a.arm != "std" and a.arm not in ARMS:
    sys.exit(f"unknown arm {a.arm}")
prefix = "std" if a.arm == "std" else f"l1{a.arm.replace('_', '')}"

by_dataset: dict[str, list[str]] = {}
for d, s, _ in cases_for(a.regime):
    by_dataset.setdefault(d, []).append(s)
if a.datasets:
    by_dataset = {d: v for d, v in by_dataset.items() if d in a.datasets.split(",")}
EXCLUDED = {"valentine_distribution"}   # 2026-10-30, the authors' decision: no longer run (too slow)
seeds = [s for s in a.seeds.split(",") if s] or sorted(p.name for p in (STUDY / "results" / a.regime).iterdir()
                                                      if p.is_dir() and (p / "DONE").is_file()
                                                      and p.name not in EXCLUDED)
suffix = a.suffix or (f"{prefix}_{a.corpus}" if not a.datasets else f"{prefix}twin_{a.corpus}")
cfg_root = STUDY / "tuning" / "configs" / suffix
if cfg_root.exists() and not a.append:
    sys.exit(f"{cfg_root} exists; remove it first")
seed_root = HERE / "seeds" / a.regime
if a.file_seeds:
    seeds = a.file_seeds.split(",")
written, skipped, partial = 0, [], {}
for seed in seeds:
    if a.file_seeds:   # matrices already in place: reference/seeds/<regime>/<seed>/<dataset>/<scenario>/seed.csv
        root = seed_root / seed
        src = lambda d, s, root=root: (root / d / s / "seed.csv") if (root / d / s / "seed.csv").is_file() else None  # noqa: E731
    else:
        src = seed_source(a.regime, seed)
    if src is None:
        skipped.append(seed)
        continue
    datasets = []
    for d, scens in sorted(by_dataset.items()):
        files = {s: src(d, s) for s in scens}
        if all(files.values()):
            datasets.append(d)
            for s, f in files.items():
                dst = seed_root / seed / d / s / "seed.csv"
                if not dst.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(f, dst)
        elif any(files.values()):
            partial.setdefault(seed, []).append(d)
    if not datasets:
        skipped.append(seed)
        continue
    if a.arm == "std":
        sf, props = {**STD_SF, "initialMatrixDir": (seed_root / seed).as_posix()}, {}   # production defaults
    else:
        sf_change, prop_change = ARMS[a.arm]
        sf = {**MELNIK_SF, **sf_change, "initialMatrixDir": (seed_root / seed).as_posix()}
        props = {**MELNIK_PROPS, **prop_change}
    label = f"{seed}__{a.arm}"
    dst = cfg_root / label
    dst.mkdir(parents=True)
    for f in ("first_line_matchers.yaml", "general.yaml", "second_line_matchers.yaml"):
        text = (TEMPLATE / f).read_text(encoding="utf-8")
        if f == "general.yaml":
            text = re.sub(r'(?m)^resultsDir: ".*"$',
                          f'resultsDir: "{(STUDY / "tuning" / "results" / suffix / label).as_posix()}"', text)
            text = re.sub(r'(?m)^defaultDatasetBasePath: ".*"$', f'defaultDatasetBasePath: "{REMOTE_CORPUS}/{a.corpus}"', text)
        if f == "second_line_matchers.yaml":
            for k, v in sf.items():
                text, n = re.subn(rf'(?m)^(  {k}: )\[.*\]$', rf'\g<1>["{v}"]', text)
                if n != 1:
                    sys.exit(f"template has no single '{k}' parameter")
        (dst / f).write_text(text, encoding="utf-8")
    (dst / "datasets.yaml").write_text("".join(f'---\nname: "{d}"\npath: "{d}"\n' for d in datasets), encoding="utf-8")
    (dst / "jvm.properties").write_text("".join(f"{k}={v}\n" for k, v in props.items()), encoding="utf-8")
    meta = {"seed": seed, "arm": a.arm, "regime": a.regime, "corpus": a.corpus, "stage": prefix, "sf": sf, "jvm": props,
            "seed_dir": (seed_root / seed).as_posix(), "datasets": datasets,
            "dataset_sizes": {d: len(by_dataset[d]) for d in datasets}}
    (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    written += 1
print(f"{suffix}: {written} configurations -> {cfg_root}")
print(f"  not finished, skipped: {skipped}")
if partial:
    print(f"  datasets left out because some scenarios have no seed: {partial}")
