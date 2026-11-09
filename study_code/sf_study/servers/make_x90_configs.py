"""X90 (EXPLORATORY; after the external reviewer's X88 theory review): melnik vs lit_none trajectories.
Stage x90_<c>, c in valentine, goby, hdxsm; matchers levenshtein, prefix_suffix, pmf, jaro_winkler; all scenarios.
    <m>__lit_none            converged base. valentine: layer1_plain/<m>__lit_none (= melnik + smx.sf.literalSeed=NONE);
                             goby, hdxsm: x81_<c>/<m>__melnik with its one literalSeed line set to NONE (asserted)
                             -- only resultsDir changes otherwise
    <m>__lit_none_t<t>       make_iter_configs.clone() UNCHANGED on that base, t in make_iter_configs.T (1, 2, 3, 5, 10)
Gate (C): x90_valentine/<m>__lit_none = layer1_plain/<m>__lit_none cell for cell.

    python servers/make_x90_configs.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_iter_configs as mic  # noqa: E402

MATCHERS = ["levenshtein", "prefix_suffix", "pmf", "jaro_winkler"]
SOURCES = {"valentine": ("layer1_plain", "lit_none"), "goby": ("x81_goby", "melnik"), "hdxsm": ("x81_hdxsm", "melnik")}


def main() -> int:
    for c, (stage, arm) in SOURCES.items():
        suffix = f"x90_{c}"
        root = mic.CONFIGS / suffix
        if root.exists():
            shutil.rmtree(root)
        for m in MATCHERS:
            src, base = mic.CONFIGS / stage / f"{m}__{arm}", root / f"{m}__lit_none"
            shutil.copytree(src, base)
            g = (base / "general.yaml").read_text(encoding="utf-8")
            old = f'resultsDir: "{(mic.RESULTS / stage / src.name).as_posix()}"'
            assert g.count(old) == 1, src
            (base / "general.yaml").write_text(g.replace(old, f'resultsDir: "{(mic.RESULTS / suffix / base.name).as_posix()}"'),
                                               encoding="utf-8")
            jvm = (base / "jvm.properties").read_text(encoding="utf-8")
            if arm == "melnik":
                assert jvm.count("smx.sf.literalSeed=PREFIX_SUFFIX") == 1, src
                (base / "jvm.properties").write_text(jvm.replace("smx.sf.literalSeed=PREFIX_SUFFIX", "smx.sf.literalSeed=NONE"),
                                                     encoding="utf-8")
            else:
                assert "smx.sf.literalSeed=NONE" in jvm, src
            meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
            meta.update({"stage": "x90", "arm": "lit_none", "source": f"{stage}/{src.name}",
                         "note": "X90 EXPLORATORY: converged lit_none base (C)"})
            (base / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            for t in mic.T:
                label = f"{m}__lit_none_t{t}"
                mic.clone(base, root / label, suffix, label, t - 1,
                          {"arm": f"lit_none_t{t}", "updates": t, "base_arm": "lit_none", "corpus": c})
        print(f"{suffix}: {len(list(root.iterdir()))} configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
