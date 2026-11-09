"""X82 (EXPLORATORY to replicate existing experiments on the new corpora, 2026-10-06 ~22:25):
X42's iteration trajectory on the new corpora. servers/make_iter_configs.py is used UNCHANGED (its clone() and T); this
wrapper only adds the base table for each new corpus and the gate, whose layer-1 base does not exist there:

    corpus            bases (converged runs)                    suffix
    final_indep       final_indep/<m>__melnik, <m>__fd          iter_final_indep
    final_controlled  final_controlled/<m>__melnik, <m>__fd     iter_final_controlled
    x81_<c>           x81_<c>/<m>__melnik, <m>__fd              iter_x81_<c>   (c = hdxsm, gdcsm, madi, goby, mimic)

<m>__<arm>_t<t> for t in (1, 2, 3, 5, 10); gate levenshtein__melnik_gate = <corpus>/levenshtein__melnik with the default
cap 200 stated (must equal it cell for cell). The *_twin configurations are not bases.

    python servers/make_x82_iter.py final_indep|final_controlled|x81_hdxsm|...
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_iter_configs as mic  # noqa: E402


def main(stage: str) -> int:
    suffix = f"iter_{stage}"
    root = mic.CONFIGS / suffix
    if root.exists():
        shutil.rmtree(root)
    n = 0
    for arm in ("melnik", "fd"):
        for base in sorted(p for p in (mic.CONFIGS / stage).iterdir() if p.is_dir() and p.name.endswith(f"__{arm}")):
            seed = base.name[: -len(f"__{arm}")]
            for t in mic.T:
                label = f"{seed}__{arm}_t{t}"
                mic.clone(base, root / label, suffix, label, t - 1,
                          {"arm": f"{arm}_t{t}", "updates": t, "base_arm": arm, "corpus_stage": stage})
                n += 1
    gate = mic.CONFIGS / stage / f"{mic.GATE_SEED}__melnik"
    mic.clone(gate, root / f"{mic.GATE_SEED}__melnik_gate", suffix, f"{mic.GATE_SEED}__melnik_gate", 200,
              {"arm": "melnik_gate", "updates": "default cap 200 stated explicitly", "base_arm": "melnik"})
    print(f"{suffix}: {n + 1} configurations under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
