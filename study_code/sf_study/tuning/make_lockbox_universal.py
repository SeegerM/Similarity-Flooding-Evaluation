"""Tuning Amendment 4 (draft E1): add the clean universal configuration (results/supp_selection_plain/universal_choice.json)
for every initial matcher of chosen.json to configs/lockbox_plain, in exactly the format make_tuning_configs.py writes.
Run after `make_tuning_configs.py --stage lockbox --regime plain` (which writes chosen.json's configurations, production
and alpha = 0 and is not changed). chosen.json itself is not rewritten.

    python make_lockbox_universal.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import make_tuning_configs as mtc  # noqa: E402
import space  # noqa: E402
from export_seeds import seed_dir  # noqa: E402
from make_configs import datasets_doc, matcher_doc  # noqa: E402


def main() -> int:
    uni = json.loads((HERE / "results" / "supp_selection_plain" / "universal_choice.json").read_text(encoding="utf-8"))["universal"]
    cfg = {space.name(c): c for c in space.configurations()}[uni]
    chosen = json.loads((HERE / "chosen.json").read_text(encoding="utf-8"))["configurations"]
    corpus = HERE / "corpus" / "plain" / "lockbox"
    root = HERE / "configs" / "lockbox_plain"
    if not root.is_dir():
        raise SystemExit("run make_tuning_configs.py --stage lockbox --regime plain first")
    prereg = hashlib.sha256((HERE / "PREREGISTRATION.md").read_bytes()).hexdigest()
    n = 0
    for seed in sorted(chosen):
        ref = json.loads((root / f"{seed}__{space.name(space.PRODUCTION)}" / "meta.json").read_text(encoding="utf-8"))
        sdir, ds = seed_dir(seed, "plain"), ref["datasets"]
        label = f"{seed}__{uni}"
        out = root / label
        if out.exists():
            continue                      # already among the matcher's own configurations
        out.mkdir()
        (out / "general.yaml").write_text(mtc.GENERAL.format(results=(HERE / "results" / "lockbox_plain" / label).as_posix(),
                                                             corpus=corpus.as_posix()), encoding="utf-8")
        (out / "datasets.yaml").write_text(datasets_doc(ds), encoding="utf-8")
        (out / "first_line_matchers.yaml").write_text(matcher_doc("ConstantMatcher", "control", {"value": 0.0}), encoding="utf-8")
        params = space.sf_params(cfg)
        params["initialMatrixDir"] = sdir.as_posix()
        (out / "second_line_matchers.yaml").write_text(matcher_doc("SimilarityFlooding", "similarityFlooding", params),
                                                       encoding="utf-8")
        (out / "jvm.properties").write_text("".join(f"{k}={v}\n" for k, v in space.system_properties(cfg).items()),
                                            encoding="utf-8")
        (out / "meta.json").write_text(json.dumps({
            "seed": seed, "config": uni, **cfg, "stage": "lockbox", "regime": "plain", "seed_dir": sdir.as_posix(),
            "datasets": ds, "preregistration_sha256": prereg, "purpose": "tuning Amendment 4: clean universal (ER1)"},
            indent=1), encoding="utf-8")
        n += 1
    print(f"lockbox_plain: {n} universal configurations ({uni}) added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
