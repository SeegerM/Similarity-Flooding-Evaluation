"""EXPLORATORY X42 (2026-10-04; the external reviewer's point 1): SF's iteration trajectory.

Does useful evidence accumulate through repeated propagation, or does an early update give the benefit? Each arm is
a copy of a registered configuration with only the iteration cap changed (`smx.sf.maxIterations`). In
SimilarityFlooding.java the loop runs while iterationCount <= MAX_ITERATIONS from 0, so a cap of k performs k + 1
updates: t updates need maxIterations = t − 1. With Melnik et al.'s tolerance (0.05) a run may converge before
the cap; the scorer reads the actual count from sf_stats.tsv.

Bases (configurations copied byte for byte except as stated):
    plain, md5  layer1_<regime>/<m>__melnik   -> <m>__melnik_t<t>
                layer1_<regime>/<m>__fd       -> <m>__fd_t<t>
    indep       layer1_indep/<m>__melnik      -> <m>__melnik_t<t>
                er10_indep/<m>__melnik_shape  -> <m>__melnik_shape_t<t>
with t in T = (1, 2, 3, 5, 10). The converged runs are the bases themselves (already scored).
Gate per regime: levenshtein__melnik_gate = layer 1's levenshtein__melnik with maxIterations=200 stated explicitly
(the default); it must equal layer 1 cell for cell.

In general.yaml only resultsDir changes; jvm.properties gains one line; the other files are copied and asserted.
Writes tuning/configs/iter_<regime>/ (suffix iter_<regime>, as dispatch.py expects).

    python servers/make_iter_configs.py --regime plain|md5|indep
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
CONFIGS = STUDY / "tuning" / "configs"
RESULTS = STUDY / "tuning" / "results"
T = (1, 2, 3, 5, 10)
BASES = {"plain": [("layer1_plain", "melnik"), ("layer1_plain", "fd")],
         "md5": [("layer1_md5", "melnik"), ("layer1_md5", "fd")],
         "indep": [("layer1_indep", "melnik"), ("er10_indep", "melnik_shape")]}
GATE_SEED = "levenshtein"


def clone(base: Path, dst: Path, suffix: str, label: str, cap: int, meta_extra: dict) -> None:
    dst.mkdir(parents=True)
    for f in base.iterdir():
        if f.name == "general.yaml":
            text = f.read_text(encoding="utf-8")
            old = f'resultsDir: "{(RESULTS / base.parent.name / base.name).as_posix()}"'
            new = f'resultsDir: "{(RESULTS / suffix / label).as_posix()}"'
            if text.count(old) != 1:
                raise SystemExit(f"{f}: expected exactly one '{old}'")
            (dst / f.name).write_text(text.replace(old, new), encoding="utf-8")
        elif f.name == "jvm.properties":
            text = f.read_text(encoding="utf-8")
            if "smx.sf.maxIterations" in text:
                raise SystemExit(f"{f}: already sets maxIterations")
            if text and not text.endswith("\n"):
                text += "\n"
            (dst / f.name).write_text(text + f"smx.sf.maxIterations={cap}\n", encoding="utf-8")
        elif f.name == "meta.json":
            meta = json.loads(f.read_text(encoding="utf-8"))
            meta.update({"stage": "iter", "base_config": f"{base.parent.name}/{base.name}", "maxIterations": cap,
                         **meta_extra, "purpose": "X42 EXPLORATORY: SF's iteration trajectory"})
            (dst / f.name).write_text(json.dumps(meta, indent=1), encoding="utf-8")
        elif f.is_file():
            shutil.copyfile(f, dst / f.name)
    for name in ("second_line_matchers.yaml", "first_line_matchers.yaml", "datasets.yaml"):
        if (base / name).is_file() and (base / name).read_bytes() != (dst / name).read_bytes():
            raise SystemExit(f"{dst / name} differs from its base")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=tuple(BASES), required=True)
    a = ap.parse_args()
    suffix = f"iter_{a.regime}"
    root = CONFIGS / suffix
    if root.exists():
        shutil.rmtree(root)
    n, missing = 0, []
    for stage, arm in BASES[a.regime]:
        bases = sorted(p for p in (CONFIGS / stage).iterdir() if p.is_dir() and p.name.endswith(f"__{arm}"))
        for base in bases:
            seed = base.name[: -len(f"__{arm}")]
            for t in T:
                label = f"{seed}__{arm}_t{t}"
                clone(base, root / label, suffix, label, t - 1, {"arm": f"{arm}_t{t}", "updates": t, "base_arm": arm})
                n += 1
        if not bases:
            missing.append(f"{stage}/*__{arm}")
    gate_base = CONFIGS / f"layer1_{a.regime}" / f"{GATE_SEED}__melnik"
    if gate_base.is_dir():
        clone(gate_base, root / f"{GATE_SEED}__melnik_gate", suffix, f"{GATE_SEED}__melnik_gate", 200,
              {"arm": "melnik_gate", "updates": "default cap 200 stated explicitly", "base_arm": "melnik"})
        n += 1
    else:
        missing.append(str(gate_base))
    print(f"{suffix}: {n} configurations under {root}" + (f"; missing bases: {missing}" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
