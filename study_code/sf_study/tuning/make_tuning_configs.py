"""Write one Schematch configuration per (seed, SF configuration) for a tuning stage.

Stages (PREREGISTRATION.md):
    screen    all 109 configurations x panel seeds, on the screen split
    dev       the configurations in keep.json (from score_tuning.py --stage screen), on dev
    lockbox   the configurations in chosen.json (from score_tuning.py --stage dev), on lockbox

Each directory tuning/configs/<stage>_<regime>/<seed>__<config>/ holds general.yaml,
datasets.yaml, a placeholder first-line matcher (the seed is a FILE seed), SF with the
configuration's matcher parameters, jvm.properties with its smx.sf.* switches, and meta.json.

Usage:
    python make_tuning_configs.py --stage screen --regime plain
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import space  # noqa: E402
from export_seeds import seed_dir  # noqa: E402
from make_configs import datasets_doc, matcher_doc  # noqa: E402

GENERAL = """applicationName: "schematch"
defaultSeparator: ","
defaultTablePairSeparator: "___"
defaultSourceDatabaseDir: "source"
defaultTargetDatabaseDir: "target"
defaultGroundTruthDir: "ground_truth"
cacheDir: "cache"
resultsDir: "{results}"
performanceDir: "_performances"
performanceSummaryFileName: "performance_summary.csv"
performanceOverviewFilePrefix: "performance_overview_"
outputDir: "_outputs"
defaultDatasetBasePath: "{corpus}"
evaluateAttributes: false
saveOutputPerTablePair: false
saveOutputVerbose: true
saveOutputTablePairGeneration: false
evaluateTablePairGeneration: false
# the seed is a FILE seed; the placeholder first-line matrix is not needed
saveOutputFirstLineMatchers: false
evaluateFirstLineMatchers: false
readCacheFirstLineMatchers: false
writeCacheFirstLineMatchers: false
# dependency loading is coupled to this flag in this Schematch fork (see make_configs.py)
runSimMatrixBoostingOnFirstLineMatchers: true
saveOutputSimMatrixBoostingOnFirstLineMatchers: false
evaluateSimMatrixBoostingOnFirstLineMatchers: false
readCacheSimMatrixBoostingOnFirstLineMatchers: false
writeCacheSimMatrixBoostingOnFirstLineMatchers: false
runSecondLineMatchers: true
saveOutputSecondLineMatchers: true
evaluateSecondLineMatchers: false
readCacheSecondLineMatchers: false
writeCacheSecondLineMatchers: false
runSimMatrixBoostingOnSecondLineMatchers: false
saveOutputSimMatrixBoostingOnSecondLineMatchers: false
evaluateSimMatrixBoostingOnSecondLineMatchers: false
readCacheSimMatrixBoostingOnSecondLineMatchers: false
writeCacheSimMatrixBoostingOnSecondLineMatchers: false
"""


def stage_configs(stage: str) -> dict[str, list[dict]]:
    """seed -> configurations to run in this stage."""
    if stage == "screen":
        return {seed: space.configurations() for seed in space.SEED_PANEL}
    source = HERE / ("keep.json" if stage == "dev" else "chosen.json")
    if not source.is_file():
        raise SystemExit(f"{source.name} not found; run score_tuning.py on the previous stage first")
    wanted = json.loads(source.read_text(encoding="utf-8"))["configurations"]
    by_name = {space.name(c): c for c in space.configurations()}
    out = {}
    for seed, names in wanted.items():
        # the production configuration and the no-propagation control always come along
        keep = set(names) | {space.name(space.PRODUCTION), space.name(space.NO_PROPAGATION)}
        out[seed] = [by_name[n] for n in sorted(keep)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("screen", "dev", "lockbox"), required=True)
    ap.add_argument("--regime", choices=("plain", "md5"), default="plain")
    args = ap.parse_args()

    prereg = hashlib.sha256((HERE / "PREREGISTRATION.md").read_bytes()).hexdigest()
    corpus = HERE / "corpus" / args.regime / args.stage
    if not corpus.is_dir():
        raise SystemExit(f"{corpus} missing; run make_splits.py first")
    datasets = sorted(p.name for p in corpus.iterdir() if p.is_dir())
    suffix = f"{args.stage}_{args.regime}"
    written, missing = 0, set()
    for seed, configs in stage_configs(args.stage).items():
        sdir = seed_dir(seed, args.regime)
        if not sdir.is_dir():
            missing.add(seed)
            continue
        # a FILE seed must cover every scenario of the datasets it runs on
        names = [d for d in datasets
                 if all((sdir / d / s.name / "seed.csv").is_file() for s in (corpus / d).iterdir())]
        if not names:
            missing.add(seed)
            continue
        for cfg in configs:
            label = f"{seed}__{space.name(cfg)}"
            out = HERE / "configs" / suffix / label
            out.mkdir(parents=True, exist_ok=True)
            results = (HERE / "results" / suffix / label).as_posix()
            (out / "general.yaml").write_text(GENERAL.format(results=results, corpus=corpus.as_posix()),
                                              encoding="utf-8")
            (out / "datasets.yaml").write_text(datasets_doc(names), encoding="utf-8")
            (out / "first_line_matchers.yaml").write_text(
                matcher_doc("ConstantMatcher", "control", {"value": 0.0}), encoding="utf-8")
            params = space.sf_params(cfg)
            params["initialMatrixDir"] = sdir.as_posix()
            (out / "second_line_matchers.yaml").write_text(
                matcher_doc("SimilarityFlooding", "similarityFlooding", params), encoding="utf-8")
            (out / "jvm.properties").write_text(
                "".join(f"{k}={v}\n" for k, v in space.system_properties(cfg).items()), encoding="utf-8")
            (out / "meta.json").write_text(json.dumps({
                "seed": seed, "config": space.name(cfg), **cfg, "stage": args.stage,
                "regime": args.regime, "seed_dir": sdir.as_posix(), "datasets": names,
                "preregistration_sha256": prereg}, indent=1), encoding="utf-8")
            written += 1
    print(f"wrote {written} configurations under {HERE / 'configs' / suffix}")
    if missing:
        print(f"seeds without a complete seed directory (run export_seeds.py after the full run): "
              f"{sorted(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
