"""X30 follow-up: layer-1 and layer-3 configurations for the NaN-free datatype seed (B, 17:10/17:37).

Clones the registered `datatype__<arm>` configurations of tuning/configs/layer{1,3}_<regime> into
tuning/configs/layer{1,3}nan_<regime>/datatype_nanfix__<arm>. Only the seed directory
(reference/seeds/<regime>/datatype -> datatype_nanfix), the label and the results directory change
(asserted). The registered directories are read, never written; make_layer1_configs.py is not run because
it rewrites them.

    python servers/make_nanfix_configs.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
CFG = STUDY / "tuning" / "configs"


def main() -> int:
    total = 0
    for layer in (1, 3):
        for regime in ("plain", "md5"):
            src_root, dst_root = CFG / f"layer{layer}_{regime}", CFG / f"layer{layer}nan_{regime}"
            if dst_root.exists():
                shutil.rmtree(dst_root)
            seed_old = f"reference/seeds/{regime}/datatype"
            seed_new = f"reference/seeds/{regime}/datatype_nanfix"
            if not (STUDY / seed_new).is_dir():
                raise SystemExit(f"{seed_new} missing")
            for src in sorted(src_root.glob("datatype__*")):
                arm = src.name.split("__", 1)[1]
                label = f"datatype_nanfix__{arm}"
                dst = dst_root / label
                dst.mkdir(parents=True)
                for f in src.iterdir():
                    old = f.read_text(encoding="utf-8")
                    new = (old.replace(f"/{seed_old}\"", f"/{seed_new}\"")
                              .replace(f"/results/layer{layer}_{regime}/datatype__{arm}\"",
                                       f"/results/layer{layer}nan_{regime}/{label}\""))
                    if f.name == "meta.json":
                        m = json.loads(new)
                        m["seed"], m["stage"] = "datatype_nanfix", f"layer{layer}nan"
                        m["note"] = "X30: datatype seed with NaN cells set to 0 (finite cells unchanged)"
                        new = json.dumps(m, indent=1)
                    # only the seed path, the results path and the meta fields may differ
                    rest_old = old.replace(f"/{seed_old}\"", "").replace(f"/results/layer{layer}_{regime}/datatype__{arm}\"", "")
                    rest_new = new.replace(f"/{seed_new}\"", "").replace(f"/results/layer{layer}nan_{regime}/{label}\"", "")
                    if f.name != "meta.json" and rest_old != rest_new:
                        raise SystemExit(f"{src.name}/{f.name}: unexpected difference")
                    if f.name in ("second_line_matchers.yaml",) and seed_new not in new:
                        raise SystemExit(f"{label}: seed path not replaced")
                    if f.name == "general.yaml" and f"layer{layer}nan_{regime}" not in new:
                        raise SystemExit(f"{label}: results path not replaced")
                    (dst / f.name).write_text(new, encoding="utf-8")
                total += 1
            print(f"layer{layer}nan_{regime}: {len(list(dst_root.iterdir()))} configurations")
    print(f"{total} in all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
