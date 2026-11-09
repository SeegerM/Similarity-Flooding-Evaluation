"""Copy existing SF configurations onto a new FILE-seed folder (2026-10-31: PRISMA recomputed on the fixed-HyFD
profiles, the rema fill for the scenarios the original rema run never reached).

Each source configuration <stage>/<seed>__<arm> becomes <suffix>/<new seed>__<arm>; only initialMatrixDir, resultsDir,
defaultDatasetBasePath (the hosts' corpus) and datasets.yaml change, every SF field and JVM property stays as in the
source. datasets.yaml lists the datasets whose every case (score_seeds.cases_for(--regime)) has a seed.csv in --seed-dir,
optionally restricted to --datasets. Any other absolute path in a source file stops the script.

    python servers/reseed_stage.py --suffix prismafix_p382fix --seed prisma --regime plain --corpus p382fix \
        --seed-dir reference/seeds/plain_fdfix/prisma --src std_p382fix/rema__std layer1_plain/rema__melnik ...
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY))
from score_seeds import cases_for  # noqa: E402

CFG, RES = STUDY / "tuning" / "configs", STUDY / "tuning" / "results"
ap = argparse.ArgumentParser()
ap.add_argument("--suffix", required=True)
ap.add_argument("--seed", required=True, help="label of the new seed (the <seed> in <seed>__<arm>)")
ap.add_argument("--seed-dir", required=True)
ap.add_argument("--regime", required=True, help="case list for the dataset check: plain, md5, pnew, mnew")
ap.add_argument("--corpus", required=True, help="hosts' corpus under $SMX_REMOTE_ROOT/smx/corpus")
ap.add_argument("--datasets", default="")
ap.add_argument("--src", nargs="+", required=True, help="source configurations <stage>/<label>")
a = ap.parse_args()

seed_dir = (STUDY / a.seed_dir).resolve() if not Path(a.seed_dir).is_absolute() else Path(a.seed_dir)
if not (seed_dir / "COMPLETE").is_file() and not any(seed_dir.glob("COMPLETE*")):
    sys.exit(f"{seed_dir} has no COMPLETE marker")
dst_root = CFG / a.suffix
if dst_root.exists():
    sys.exit(f"{dst_root} exists")
by_dataset: dict[str, list[str]] = {}
for d, s, _ in cases_for(a.regime):
    by_dataset.setdefault(d, []).append(s)
want = set(a.datasets.split(",")) if a.datasets else set(by_dataset)
if want - set(by_dataset):
    sys.exit(f"unknown datasets {want - set(by_dataset)}")
datasets = sorted(d for d in want if all((seed_dir / d / s / "seed.csv").is_file() for s in by_dataset[d]))
missing = sorted(want - set(datasets))
if a.datasets and missing:
    sys.exit(f"datasets asked for but incomplete in {seed_dir}: {missing}")
seed_posix = seed_dir.as_posix()
for src in a.src:
    sdir = CFG / src
    arm = sdir.name.split("__", 1)[1]
    label = f"{a.seed}__{arm}"
    dst = dst_root / label
    dst.mkdir(parents=True)
    for f in ("first_line_matchers.yaml", "general.yaml", "second_line_matchers.yaml", "jvm.properties"):
        text = (sdir / f).read_text(encoding="utf-8")
        text, n1 = re.subn(r'(?m)^(  initialMatrixDir: )\[".*"\]$', rf'\g<1>["{seed_posix}"]', text)
        text, n2 = re.subn(r'(?m)^resultsDir: ".*"$', f'resultsDir: "{(RES / a.suffix / label).as_posix()}"', text)
        text, n3 = re.subn(r'(?m)^(defaultDatasetBasePath: ".*/smx/corpus/)[^"/]+"$', rf'\g<1>{a.corpus}"', text)
        if f == "second_line_matchers.yaml" and n1 != 1 or f == "general.yaml" and (n2, n3) != (1, 1):
            sys.exit(f"{sdir / f}: expected fields not found")
        rest = [ln for ln in text.splitlines() if re.search(r'"(F:/|/app/)', ln) and not re.match(
            r'\s*(initialMatrixDir|resultsDir|defaultDatasetBasePath):', ln)]
        if rest:
            sys.exit(f"{sdir / f}: other absolute paths {rest}")
        (dst / f).write_text(text, encoding="utf-8")
    (dst / "datasets.yaml").write_text("".join(f'---\nname: "{d}"\npath: "{d}"\n' for d in datasets), encoding="utf-8")
    meta = json.loads((sdir / "meta.json").read_text(encoding="utf-8"))
    meta.update({"seed": a.seed, "regime": a.regime, "corpus": a.corpus, "seed_dir": seed_posix, "datasets": datasets,
                 "dataset_sizes": {d: len(by_dataset[d]) for d in datasets}, "reseeded_from": src})
    meta.setdefault("sf", {})["initialMatrixDir"] = seed_posix
    (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print(f"{a.suffix}: {len(a.src)} configurations, {len(datasets)} datasets "
      f"({sum(len(by_dataset[d]) for d in datasets)} cases); not complete in the seed folder: {missing}")
