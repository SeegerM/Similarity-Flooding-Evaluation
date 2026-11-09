"""Layer 1: every finished seed of the seed study, flooded under the Melnik et al. (2002) reference
configuration and under one-factor deviations from it, on all 382 audited scenarios.

The reference (arm "melnik") follows the ICDE 2002 paper as closely as Schematch allows:
  - graph: the OIM-style schema graph (Table, Column, ColumnType nodes; type, name, column and
    datatype edges), no dependency edges. The paper's graph also has declared keys, which the
    Valentine CSV tables do not have.
  - fixpoint formula C, inverse-average propagation coefficients (best in the paper's user study),
    normalisation by the maximum, stop when the Euclidean residual is below 0.05.
  - initial similarities: the seed on column-name pairs (NAME attachment) and, as the paper's
    StringMatch does, the prefix/suffix string similarity on every other pair of non-ID nodes
    (smx.sf.literalSeed=PREFIX_SUFFIX).
  - extraction is done when scoring: the paper's SelectThreshold with t_rel = 1.0 equals mutual
    top-1; Hungarian and the ranking metrics are reported alongside.
Each other arm changes exactly one module (ARMS). The seed study itself is the thesis's production
configuration (inverse product, all FDs, no literal seeds, epsilon 1e-4).

Seeds are the seed study's own first-line matrices, copied unchanged to reference/seeds/<regime>/
(or, for FILE seeds, the files SF read); C2 in sanity_checks.py shows SF reproduces the study
exactly from them. A dataset is included for a seed only if every one of its scenarios has a seed.

Writes tuning/configs/layer1_<regime>/<seed>__<arm>/ for servers/dispatch.py (--split-datasets).

    python reference/make_layer1_configs.py --regime plain
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
from score_seeds import CORPUS, EXTERNAL, cases_for, latest_run, one  # noqa: E402

REMOTE_CORPUS = "$SMX_REMOTE_ROOT/smx/corpus"
UNPROFILED = {"indep", "indepsmoke"}  # corpora without FD/UCC/IND profiles (data/convert_indep.py)
TEMPLATE = STUDY / "tuning" / "configs" / "screen_plain" / "levenshtein__fpC__INV_AVG__NAME__fdoff"

MELNIK_SF = {"graphEncoding": "QUICK", "propCoeffPolicy": "INV_AVG", "fixpoint": "C",
             "fdFilterThreshold": "", "uccFilterThreshold": "", "indFilterThreshold": "",
             "constraintWeighting": "NONE", "dependencyTopK": "", "fdAllowListFile": "",
             "initialMatrix": "FILE"}
MELNIK_PROPS = {"smx.sf.seedAttachment": "NAME", "smx.sf.literalSeed": "PREFIX_SUFFIX", "smx.sf.epsilon": "0.05"}

# arm -> (SF parameter changes, JVM property changes), each relative to the Melnik reference
ARMS: dict[str, tuple[dict, dict]] = {
    "melnik": ({}, {}),
    "coef_prod": ({"propCoeffPolicy": "INV_PROD"}, {}),                     # coefficient policy
    "fd": ({"fdFilterThreshold": "0.0"}, {}),                                # + all FD edges
    "coef_prod_fd": ({"propCoeffPolicy": "INV_PROD", "fdFilterThreshold": "0.0"}, {}),  # 2x2 interaction
    "attach_id": ({}, {"smx.sf.seedAttachment": "ID"}),                      # column seed on ID pairs
    "lit_none": ({}, {"smx.sf.literalSeed": "NONE"}),                        # no StringMatch on other literals
}


# Layer 3 (RQ2, graph construction): one information family or representation at a time, relative to
# the Melnik reference. "fd" is already layer 1's arm and is not repeated. Two representations:
#   schema graph (Melnik's OIM-style graph, QUICK = projected dependency edges, COMPLETE = reified)
#   column-identity graph (ColumnRelationFlooding: column nodes only, no schema/type hubs; channels
#   from the data: gpdep, within-table value inclusion gcov, Jaccard overlap, uncertainty U)
# The column-identity encodings use their own convergence test (1e-4) and ignore seed attachment and
# literal seeds, which only exist in the schema graph; recorded in PREREGISTRATION_LAYERS.md.
ARMS_L3: dict[str, tuple[dict, dict]] = {
    "ucc": ({"uccFilterThreshold": "0.0"}, {}),
    "ind": ({"indFilterThreshold": "0.0"}, {}),
    "fd_ucc_ind": ({"fdFilterThreshold": "0.0", "uccFilterThreshold": "0.0", "indFilterThreshold": "0.0"}, {}),
    "fd_reified": ({"fdFilterThreshold": "0.0", "graphEncoding": "COMPLETE"}, {}),
    "fd_damped": ({"fdFilterThreshold": "0.0", "constraintWeighting": "CONSTANT"}, {"smx.sf.constantWeight": "0.35"}),
    "col_fd": ({"fdFilterThreshold": "0.0", "graphEncoding": "COLUMN_ONLY"}, {}),
    "col_gpdep": ({"fdFilterThreshold": "0.0", "graphEncoding": "RELATIONAL"}, {}),
    "col_gpdep_gcov": ({"fdFilterThreshold": "0.0", "graphEncoding": "RELATIONAL_IND"}, {}),
    "col_fd_gcov": ({"fdFilterThreshold": "0.0", "graphEncoding": "RELATIONAL_FD_IND"}, {}),
    "col_fd_gcov_u": ({"fdFilterThreshold": "0.0", "graphEncoding": "RELATIONAL_FD_IND_U"}, {}),
    "col_fd_jaccard": ({"fdFilterThreshold": "0.0", "graphEncoding": "RELATIONAL_FD_JACCARD"}, {}),
}


def seed_source(regime: str, seed: str):
    """Function (dataset, scenario) -> seed matrix file, or None if the seed is not finished."""
    seed_dir = STUDY / "results" / regime / seed
    if not (seed_dir / "DONE").is_file():
        return None
    marker = STUDY / "configs" / regime / seed / "seed_file_dir.txt"
    if marker.is_file():
        root = Path(marker.read_text(encoding="utf-8").strip())
        return lambda d, s: (root / d / s / "seed.csv") if (root / d / s / "seed.csv").is_file() else None
    run = latest_run(seed_dir)
    if run is None:
        return None
    return lambda d, s: one(run / d / s / "_outputs" / "MatchingStepLine1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=sorted(CORPUS), required=True)
    ap.add_argument("--seeds", default="", help="comma-separated subset (default: every finished seed)")
    ap.add_argument("--layer", type=int, choices=(1, 3), default=1)
    ap.add_argument("--arms", default="", help="comma-separated; default: every arm of the layer")
    a = ap.parse_args()

    by_dataset: dict[str, list[str]] = {}
    for d, s, _ in cases_for(a.regime):
        by_dataset.setdefault(d, []).append(s)
    seeds = [s for s in a.seeds.split(",") if s] or sorted(p.name for p in (STUDY / "results" / a.regime).iterdir()
                                                          if p.is_dir() and (p / "DONE").is_file())
    table = ARMS if a.layer == 1 else ARMS_L3
    arms = [x for x in a.arms.split(",") if x] or list(table)
    if a.regime in UNPROFILED and any(table[x][0].get(k) for x in arms
                                      for k in ("fdFilterThreshold", "uccFilterThreshold", "indFilterThreshold")):
        # without FD profiles Schematch builds no FD edges and does not complain: an "fd" arm would
        # silently equal its FD-free twin
        raise SystemExit(f"{a.regime} has no FD profiles yet; leave out the FD arms "
                         f"(layer 1: --arms {','.join(x for x in ARMS if not ARMS[x][0].get('fdFilterThreshold'))})")
    suffix = f"layer{a.layer}_{a.regime}"
    cfg_root = STUDY / "tuning" / "configs" / suffix
    if cfg_root.exists():
        shutil.rmtree(cfg_root)
    seed_root = HERE / "seeds" / a.regime
    written, skipped, partial = 0, [], {}
    for seed in seeds:
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
        for arm in arms:
            sf_change, prop_change = table[arm]
            sf = {**MELNIK_SF, **sf_change, "initialMatrixDir": (seed_root / seed).as_posix()}
            props = {**MELNIK_PROPS, **prop_change}
            label = f"{seed}__{arm}"
            dst = cfg_root / label
            dst.mkdir(parents=True)
            for f in ("first_line_matchers.yaml", "general.yaml", "second_line_matchers.yaml"):
                text = (TEMPLATE / f).read_text(encoding="utf-8")
                if f == "general.yaml":
                    text = re.sub(r'(?m)^resultsDir: ".*"$',
                                  f'resultsDir: "{(STUDY / "tuning" / "results" / suffix / label).as_posix()}"', text)
                    text = re.sub(r'(?m)^defaultDatasetBasePath: ".*"$',
                                  f'defaultDatasetBasePath: "{REMOTE_CORPUS}/{a.regime}"', text)
                if f == "second_line_matchers.yaml":
                    for k, v in sf.items():
                        text, n = re.subn(rf'(?m)^(  {k}: )\[.*\]$', rf'\g<1>["{v}"]', text)
                        if n != 1:
                            raise SystemExit(f"template has no single '{k}' parameter")
                (dst / f).write_text(text, encoding="utf-8")
            (dst / "datasets.yaml").write_text("".join(f'---\nname: "{d}"\npath: "{d}"\n' for d in datasets),
                                               encoding="utf-8")
            (dst / "jvm.properties").write_text("".join(f"{k}={v}\n" for k, v in props.items()), encoding="utf-8")
            meta = {"seed": seed, "arm": arm, "regime": a.regime, "stage": f"layer{a.layer}", "sf": sf, "jvm": props,
                    "seed_dir": (seed_root / seed).as_posix(), "datasets": datasets,
                    "dataset_sizes": {d: len(by_dataset[d]) for d in datasets}}
            (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            written += 1
    n_seeds = len({p.name.split("__")[0] for p in cfg_root.iterdir()}) if cfg_root.exists() else 0
    print(f"{suffix}: {written} configurations ({n_seeds} seeds x {len(arms)} arms) -> {cfg_root}")
    print(f"  not finished, skipped: {skipped}")
    if partial:
        print(f"  datasets left out because some scenarios have no seed: {partial}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
